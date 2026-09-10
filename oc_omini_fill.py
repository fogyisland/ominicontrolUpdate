import torch
from PIL import Image, ImageOps
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


def run(t_img, t_mask, prompt, seed, low_vram: bool = True, quantization: str = "nf4", transformer=None):
    """
    8G 显存适配版 Fill 节点。
    - transformer (可选): 已量化的 FluxTransformer2DModel，传入则复用，跳过自加载/量化。
    """
    assert t_img.shape[0] == 1
    assert t_mask.shape[0] == 1

    i = 255. * t_img[0].numpy()
    ori_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB").resize((g_width, g_height))

    mi = 255. * t_mask[0].numpy()
    mask = Image.fromarray(np.clip(mi, 0, 255).astype(np.uint8)).convert("L").resize((g_width, g_height))
    mask = ImageOps.invert(mask)

    image = Image.new('RGB', ori_image.size, (0, 0, 0))
    image.paste(ori_image, (0, 0), mask)

    release_gpu()

    flux_dir = os.path.join(folder_paths.models_dir, 'flux', 'FLUX.1-schnell')
    lora_model = os.path.join(folder_paths.models_dir, 'flux', 'OminiControl', 'experimental', 'fill.safetensors')

    encoded_condition = encode_condition(flux_dir, image, 'fill')
    release_gpu()

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_only(flux_dir, prompt)
    release_gpu()

    transformer_owned_locally = (transformer is None)
    if transformer is None:
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_dir, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        if low_vram:
            transformer = quantize_transformer_inplace(transformer, method=quantization)
            try:
                transformer.to("cuda")
            except Exception as e:
                print(f"[ominicontrol_lowvram.oc_omini_fill] transformer.to('cuda') 失败: {e}")
                for name, module in transformer.named_modules():
                    try:
                        module.to("cuda")
                    except Exception:
                        pass
        else:
            transformer.to("cuda")
    else:
        print("[ominicontrol_lowvram.oc_omini_fill] 复用外部 transformer 引用")
        if torch.cuda.is_available():
            try:
                transformer.to("cuda")
            except Exception:
                pass

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
    if low_vram:
        try:
            pipeline.enable_attention_slicing("max")
        except Exception:
            pass
        try:
            pipeline.enable_sequential_cpu_offload()
        except Exception:
            try:
                pipeline.enable_model_cpu_offload()
            except Exception:
                pass

    pipeline.load_lora_weights(lora_model, adapter_name='fill')

    condition = Condition('fill', image)

    # 修正种子 bug
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
        try:
            pipeline.unload_lora_weights()
        except Exception:
            pass
        del pipeline
        if transformer_owned_locally:
            del transformer
        release_gpu()

    result_img = decode_latents(flux_dir, result_latents[0]).images[0]

    return torch.from_numpy(np.array(result_img).astype(np.float32) / 255.0).unsqueeze(0)
