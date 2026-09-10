import torch
from PIL import Image
import numpy as np
from diffusers import FluxPipeline, FluxTransformer2DModel
from ominicontrol_lowvram.src.generate import generate, seed_everything
from ominicontrol_lowvram.src.condition import Condition
import folder_paths
import os
from ominicontrol_lowvram.oc_utils import (
    release_gpu,
    encode_condition,
    encode_prompt_only,
    decode_latents,
    quantize_transformer_inplace,
    g_width,
    g_height,
)


def run(t_img, prompt, seed, low_vram: bool = True, quantization: str = "nf4", transformer=None):
    """
    8G 显存适配版 Subject 节点。
    - low_vram: True 时启用逐阶段显存清理 + sequential cpu offload + attention slicing。
    - quantization: 'nf4' | 'fp8' | 'none'。仅在 low_vram=True 且未传入外部 transformer 时生效。
    - transformer (可选): 已量化的 FluxTransformer2DModel，传入则复用，跳过自加载/量化。
    """
    assert t_img.shape[0] == 1

    i = 255. * t_img[0].numpy()
    image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB").resize((g_width, g_height))

    release_gpu()

    flux_dir = os.path.join(folder_paths.models_dir, 'flux', 'FLUX.1-schnell')
    lora_model = os.path.join(folder_paths.models_dir, 'flux', 'OminiControl', 'omini', 'subject_512.safetensors')

    # 1) 编码条件图（只装 VAE）
    encoded_condition = encode_condition(flux_dir, image)
    release_gpu()

    # 2) 文本编码（只装文本编码器）
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_only(flux_dir, prompt)
    release_gpu()

    # 3) 加载 Transformer，按需量化（若已由外部节点提供则复用）
    transformer_owned_locally = (transformer is None)
    if transformer is None:
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_dir, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        if low_vram:
            transformer = quantize_transformer_inplace(transformer, method=quantization)
            # 量化后整体 to('cuda')：bnb 的 Linear4bit 会在 to 时自动把 4bit 权重搬到目标设备。
            try:
                transformer.to("cuda")
            except Exception as e:
                print(f"[ominicontrol_lowvram.oc_omini_subject] transformer.to('cuda') 失败: {e}")
                # 兜底：手动把所有非量化 Linear 搬到 cuda，量化层留给 bnb lazy load
                for name, module in transformer.named_modules():
                    try:
                        module.to("cuda")
                    except Exception:
                        pass
        else:
            transformer.to("cuda")
    else:
        print("[ominicontrol_lowvram.oc_omini_subject] 复用外部 transformer 引用")
        # 确保外部 transformer 在 cuda 上
        if torch.cuda.is_available():
            try:
                transformer.to("cuda")
            except Exception:
                pass

    # 4) 装一个轻量 pipeline 壳用于 generate
    pipeline = FluxPipeline.from_pretrained(
        flux_dir,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        vae=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    # 防御性：attention slicing
    if low_vram:
        try:
            pipeline.enable_attention_slicing("max")
        except Exception:
            pass
        try:
            # accelerate 可选；没有也无所谓
            pipeline.enable_sequential_cpu_offload()
        except Exception:
            try:
                pipeline.enable_model_cpu_offload()
            except Exception:
                pass

    pipeline.load_lora_weights(lora_model, adapter_name="subject")

    condition = Condition("subject", image)

    # 修正种子 bug：原代码用 '^'，应改为取模
    seed_everything(int(seed) % 65536)

    try:
        result_latents = generate(
            pipeline,
            encoded_condition=encoded_condition,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            conditions=[condition],
            output_type="latent",
            return_dict=False,
            num_inference_steps=8,
            height=g_height,
            width=g_width,
        )
    finally:
        # 5) 推理结束立即释放 transformer 与 pipeline
        try:
            pipeline.unload_lora_weights()
        except Exception:
            pass
        del pipeline
        if transformer_owned_locally:
            del transformer
        release_gpu()

    # 6) 解码 latent（只装 VAE）
    result_img = decode_latents(flux_dir, result_latents[0]).images[0]

    return torch.from_numpy(np.array(result_img).astype(np.float32) / 255.0).unsqueeze(0)
