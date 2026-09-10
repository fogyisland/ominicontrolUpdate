import os
import gc
import torch
from typing import Optional
from PIL import Image
from diffusers import FluxPipeline, AutoencoderKL
from ominicontrol_lowvram.src.generate import generate, seed_everything
from ominicontrol_lowvram.src.condition import Condition

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
)

g_width = 512
g_height = 512


def release_gpu():
    """清理 Python 与 CUDA 缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _has_bitsandbytes():
    """检测 bitsandbytes 是否可用。"""
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _has_torchao():
    """检测 torchao 是否可用（FP8 路径）。"""
    try:
        import torchao  # noqa: F401
        from torchao.quantization import quantize_, float8_weight_only  # noqa: F401
        return True
    except Exception:
        return False


def quantize_transformer_inplace(transformer, method: str = "nf4"):
    """
    将 FLUX Transformer 在加载后就地量化为低精度。
    - 'nf4'：使用 bitsandbytes 的 Linear4bit（NF4）就地替换 Linear 层。
             优先使用 bnb.nn.Linear4bit.from_linear（bnb >= 0.41 推荐）。
             失败时回退到手动构造 Int4Params。
    - 'fp8'：使用 torchao float8 weight only（需要 torch >= 2.4）。
    - 'none' / '' / 'bf16' / 'fp16'：不量化（按原始精度运行）。
    返回 transformer（已就地修改）。
    """
    if method in (None, "none", "", "bf16", "fp16"):
        return transformer

    if method == "nf4":
        return _bnb_nf4_quantize(transformer)

    if method == "fp8":
        return _torchao_fp8_quantize(transformer)

    print(f"[ominicontrol_lowvram] 未知量化方法 {method!r}，回退到不量化。")
    return transformer


def _bnb_nf4_quantize(transformer):
    """使用 bitsandbytes NF4 就地量化 Linear 层。"""
    if not _has_bitsandbytes():
        print("[ominicontrol_lowvram] bitsandbytes 不可用，回退到不量化。请 pip install bitsandbytes 或选择 quantization='none'。")
        return transformer

    import bitsandbytes as bnb

    # 目标：FLUX 中所有 nn.Linear（attention 投影、MLP、AdaLayerNorm 等）
    target_keywords = (
        "to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj",
        "add_out", "proj_mlp", "proj_out", "ff.net", "norm.linear",
    )

    converted = 0
    skipped = 0
    failed = 0
    # 用 list(...) 避免就地修改 named_modules() 迭代器
    for name, module in list(transformer.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(k in name for k in target_keywords):
            continue
        # 已经量化过则跳过
        if isinstance(module, bnb.nn.Linear4bit):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer

        new_layer = None
        # 路径 A（推荐）：bnb >= 0.41 提供 from_linear
        from_linear = getattr(bnb.nn.Linear4bit, "from_linear", None)
        if callable(from_linear):
            try:
                new_layer = from_linear(
                    module,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                )
            except Exception as e:
                print(f"[ominicontrol_lowvram] from_linear 失败 ({name}): {e}")
                new_layer = None

        # 路径 B（兜底）：手动构造 Linear4bit + Int4Params
        if new_layer is None:
            try:
                new_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                )
                # 关键：必须给 weight 赋 Int4Params(data, ...)，bnb 会就地量化 data
                weight_data = module.weight.data.detach().clone().to(torch.bfloat16)
                new_layer.weight = bnb.nn.Int4Params(weight_data, requires_grad=False)
                if module.bias is not None:
                    new_layer.bias = torch.nn.Parameter(module.bias.data.detach().clone().to(torch.bfloat16))
            except Exception as e:
                print(f"[ominicontrol_lowvram] 构造 Linear4bit 失败 ({name}): {e}")
                failed += 1
                continue

        try:
            setattr(parent, attr, new_layer)
            converted += 1
        except Exception as e:
            print(f"[ominicontrol_lowvram] 替换模块失败 ({name}): {e}")
            failed += 1

    print(f"[ominicontrol_lowvram] NF4 量化：成功 {converted} 层，跳过 {skipped} 层，失败 {failed} 层")
    return transformer


def _torchao_fp8_quantize(transformer):
    """使用 torchao float8 weight only 量化整个 transformer。"""
    if not _has_torchao():
        print("[ominicontrol_lowvram] torchao 不可用，回退到不量化。")
        return transformer
    try:
        from torchao.quantization import quantize_, float8_weight_only
        quantize_(transformer, float8_weight_only())
        print("[ominicontrol_lowvram] FP8 (torchao float8 weight only) 量化完成")
        return transformer
    except Exception as e:
        print(f"[ominicontrol_lowvram] FP8 量化失败 ({e})，回退到不量化。")
        return transformer


# ---------------------------------------------------------------------------
# 给 BNB 节点用的入口函数：加载 + 量化 + 可选搬设备
# ---------------------------------------------------------------------------

def load_and_quantize_transformer(
    flux_dir: str,
    method: str = "nf4",
    skip_keywords: str = "",
    keep_on_cpu: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    从 flux_dir 加载 transformer，按 method 进行量化，可选择保留在 CPU。

    Args:
        flux_dir: FLUX 模型目录（含 transformer/ 子目录）。
        method: 'nf4' / 'fp8' / 'none'。
        skip_keywords: 逗号分隔的子串，模块名包含任一子串则跳过量化（如 "norm,time_text_embed"）。
        keep_on_cpu: True 时量化后保留在 CPU；False 时自动 .to('cuda')。
        torch_dtype: 加载基础权重所用的 dtype（默认 bf16）。

    Returns:
        FluxTransformer2DModel 实例，已就地量化（如适用）。
    """
    from diffusers import FluxTransformer2DModel

    print(f"[ominicontrol_lowvram] 加载 transformer: {flux_dir}, method={method}, keep_on_cpu={keep_on_cpu}")
    transformer = FluxTransformer2DModel.from_pretrained(
        flux_dir, subfolder="transformer", torch_dtype=torch_dtype
    )

    if method != "none":
        skip_set = tuple(s.strip() for s in skip_keywords.split(",") if s.strip())
        if skip_set:
            transformer = _bnb_nf4_quantize_with_skip(transformer, skip_set)
        else:
            transformer = quantize_transformer_inplace(transformer, method=method)
    else:
        print("[ominicontrol_lowvram] quantization='none'，跳过量化")

    if not keep_on_cpu and torch.cuda.is_available():
        try:
            transformer.to("cuda")
            print("[ominicontrol_lowvram] transformer 已搬到 cuda")
        except Exception as e:
            print(f"[ominicontrol_lowvram] transformer.to('cuda') 失败: {e}")
            for name, module in transformer.named_modules():
                try:
                    module.to("cuda")
                except Exception:
                    pass
    else:
        print("[ominicontrol_lowvram] 保持 transformer 在 CPU")

    return transformer


def _bnb_nf4_quantize_with_skip(transformer, skip_keywords):
    """带跳过规则的 NF4 量化。skip_keywords 是子串元组。"""
    if not _has_bitsandbytes():
        print("[ominicontrol_lowvram] bitsandbytes 不可用，跳过量化。")
        return transformer
    import bitsandbytes as bnb

    target_keywords = (
        "to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj",
        "add_out", "proj_mlp", "proj_out", "ff.net", "norm.linear",
    )

    converted = 0
    skipped = 0
    failed = 0
    for name, module in list(transformer.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(k in name for k in target_keywords):
            continue
        if any(s in name for s in skip_keywords):
            skipped += 1
            continue
        if isinstance(module, bnb.nn.Linear4bit):
            continue

        parent_name, _, attr = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer

        new_layer = None
        from_linear = getattr(bnb.nn.Linear4bit, "from_linear", None)
        if callable(from_linear):
            try:
                new_layer = from_linear(module, compute_dtype=torch.bfloat16, quant_type="nf4")
            except Exception as e:
                print(f"[ominicontrol_lowvram] from_linear 失败 ({name}): {e}")

        if new_layer is None:
            try:
                new_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                )
                weight_data = module.weight.data.detach().clone().to(torch.bfloat16)
                new_layer.weight = bnb.nn.Int4Params(weight_data, requires_grad=False)
                if module.bias is not None:
                    new_layer.bias = torch.nn.Parameter(module.bias.data.detach().clone().to(torch.bfloat16))
            except Exception as e:
                print(f"[ominicontrol_lowvram] 构造 Linear4bit 失败 ({name}): {e}")
                failed += 1
                continue

        try:
            setattr(parent, attr, new_layer)
            converted += 1
        except Exception as e:
            print(f"[ominicontrol_lowvram] 替换模块失败 ({name}): {e}")
            failed += 1

    print(f"[ominicontrol_lowvram] NF4 量化（含跳过）：成功 {converted} 层，跳过 {skipped} 层，失败 {failed} 层")
    return transformer


# ---------------------------------------------------------------------------
# 8G 显存友好的子函数：只装必要组件
# ---------------------------------------------------------------------------

def _load_only_vae(flux_dir, dtype=torch.bfloat16):
    """仅加载 VAE，避免把整个 pipeline 拉上卡。"""
    vae = AutoencoderKL.from_pretrained(flux_dir, subfolder="vae", torch_dtype=dtype)
    return vae


def _load_only_text_encoders(flux_dir, dtype=torch.bfloat16):
    """仅加载文本编码器 + tokenizer。"""
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    text_encoder = CLIPTextModel.from_pretrained(flux_dir, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(flux_dir, subfolder="text_encoder_2", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(flux_dir, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(flux_dir, subfolder="tokenizer_2")
    return text_encoder, text_encoder_2, tokenizer, tokenizer_2


def _make_pipeline_with_components(
    flux_dir,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    vae=None,
    transformer=None,
    dtype=torch.bfloat16,
):
    """构建一个只含必要组件的 FluxPipeline，避免加载多余的子模块占显存。"""
    kwargs = dict(
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
        torch_dtype=dtype,
    )
    # from_pretrained 会按需加载未提供的组件；为节省内存显式传 None 不会跳过，
    # 所以采用先建一个空壳，再覆盖。
    pipeline = FluxPipeline.from_pretrained(flux_dir, **kwargs)
    return pipeline


def encode_condition(flux_dir, image, condition_type='subject'):
    """
    编码条件图：仅加载 VAE，避免 transformer/文本编码器占用显存。
    """
    # 1) 准备 VAE
    vae = _load_only_vae(flux_dir)
    # 2) 用一个轻量 pipeline 壳调用 image_processor
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=None,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    condition = Condition(condition_type, image)
    try:
        # Condition.encode 内部会使用 pipeline.image_processor 与 pipeline.vae
        tokens, ids, type_id = condition.encode(pipeline)
    finally:
        del condition
        del pipeline
        release_gpu()

    return (tokens, ids, type_id)


def encode_prompt_only(flux_dir, prompt: str, max_sequence_length: int = 256):
    """
    仅加载文本编码器，编码完成后立即释放。
    返回 (prompt_embeds, pooled_prompt_embeds, text_ids)。
    """
    text_encoder, text_encoder_2, tokenizer, tokenizer_2 = _load_only_text_encoders(flux_dir)
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=None,
        vae=None,
    )
    # 不把 pipeline 整体 to('cuda')，只把文本编码器送到 GPU 上
    if torch.cuda.is_available():
        pipeline.text_encoder.to("cuda")
        pipeline.text_encoder_2.to("cuda")

    try:
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=prompt, prompt_2=None, max_sequence_length=max_sequence_length,
            )
    finally:
        # 立即卸载文本编码器
        try:
            pipeline.text_encoder.to("cpu")
            pipeline.text_encoder_2.to("cpu")
        except Exception:
            pass
        del text_encoder
        del text_encoder_2
        del tokenizer
        del tokenizer_2
        del pipeline
        release_gpu()

    return prompt_embeds, pooled_prompt_embeds, text_ids


def decode_latents(flux_dir, latents):
    """
    仅加载 VAE 解码 latent，避免 transformer / 文本编码器占用显存。
    """
    vae = _load_only_vae(flux_dir)
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=None,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    try:
        latents = pipeline._unpack_latents(latents, g_height, g_width, pipeline.vae_scale_factor)
        latents = (
            latents / pipeline.vae.config.scaling_factor
        ) + pipeline.vae.config.shift_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type="pil")
    finally:
        del vae
        del pipeline
        release_gpu()
    return FluxPipelineOutput(images=image)


# ---------------------------------------------------------------------------
# 兼容旧接口（保留，便于平滑迁移）
# ---------------------------------------------------------------------------
def encode_condition_legacy(flux_dir, image, condition_type='subject'):
    return encode_condition(flux_dir, image, condition_type)
