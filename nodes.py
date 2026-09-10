import sys
import os
import importlib
import folder_paths

# ---------------------------------------------------------------------------
# 通用工具：让节点可按需重新加载内部模块，便于在 ComfyUI 不重启的情况下
# 反复调整低显存逻辑。
# ---------------------------------------------------------------------------
def _reload_rh_module(short_name: str):
    full_name = f"ominicontrol_lowvram.oc_omini_{short_name}"
    if full_name in sys.modules:
        importlib.reload(sys.modules[full_name])
    return importlib.import_module(full_name)


def _resolve_flux_dir() -> str:
    """解析 FLUX-schnell 模型目录。"""
    return os.path.join(folder_paths.models_dir, 'flux', 'FLUX.1-schnell')


# ---------------------------------------------------------------------------
# BNB 量化加载节点
# ---------------------------------------------------------------------------
class Kiki_BNB_QuantizeTransformer:
    """
    独立加载 FLUX Transformer 并就地量化为 NF4（bitsandbytes）/ FP8（torchao）/ 不量化。

    输出 TRANSFORMER 给下游 Omini 节点使用（拖一根线连到 Subject/Spatial/Fill 的 transformer 即可）。
    若不连接，Omini 节点会按各自参数自行加载 transformer。

    输入：
        - flux_dir（默认自动定位到 models/flux/FLUX.1-schnell）
        - quantization: 'nf4' / 'fp8' / 'none'
        - skip_keywords: 逗号分隔子串，模块名含其中任一则跳过量化
                        （例 "norm,time_text_embed" — 保留归一化层为 bf16）
        - keep_on_cpu: True 时留在 CPU，下游使用时再 .to('cuda')（多卡/offload 友好）
        - torch_dtype: 加载基础权重的 dtype
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "quantization": (["nf4", "fp8", "none"], {"default": "nf4"}),
                "skip_keywords": ("STRING", {"default": "",
                                              "multiline": False,
                                              "tooltip": "逗号分隔子串；模块名含其中任一则跳过量化（例: norm,time_text_embed）"}),
                "keep_on_cpu": ("BOOLEAN", {"default": False,
                                            "tooltip": "True 时保留 transformer 在 CPU；False 时自动 .to('cuda')"}),
                "torch_dtype": (["bf16", "fp16"], {"default": "bf16"}),
            },
            "optional": {
                "flux_dir": ("STRING", {"default": "",
                                         "tooltip": "留空则自动使用 models/flux/FLUX.1-schnell"}),
            }
        }

    RETURN_TYPES = ("TRANSFORMER",)
    RETURN_NAMES = ("transformer",)
    FUNCTION = "run"
    TITLE = "BNB Quantize Transformer"
    CATEGORY = "OminiControl/Loader"
    DESCRIPTION = "独立加载并量化 FLUX Transformer（NF4/FP8/none），供下游 OminiControl 节点复用"

    def run(self, quantization, skip_keywords, keep_on_cpu, torch_dtype, flux_dir=""):
        # 决定 flux_dir
        target_dir = flux_dir.strip() or _resolve_flux_dir()
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"FLUX 模型目录不存在: {target_dir}")

        # 决定 dtype
        dtype_map = {"bf16": "bfloat16", "fp16": "float16"}
        dtype_str = dtype_map.get(torch_dtype, "bfloat16")
        import torch as _torch
        dtype = getattr(_torch, dtype_str)

        from ominicontrol_lowvram.oc_utils import load_and_quantize_transformer
        transformer = load_and_quantize_transformer(
            flux_dir=target_dir,
            method=str(quantization),
            skip_keywords=str(skip_keywords or ""),
            keep_on_cpu=bool(keep_on_cpu),
            torch_dtype=dtype,
        )
        return (transformer,)


# ---------------------------------------------------------------------------
# Omini 三件套（接受可选的 transformer 输入）
# ---------------------------------------------------------------------------
class Kiki_Omini_Subject:
    """
    OminiControl Subject 节点（8G 显存适配版）。

    新增输入：
        - low_vram:  True（默认）启用逐阶段显存清理 + CPU offload + attention slicing。
        - quantization: 'nf4' / 'fp8' / 'none'。仅在 low_vram=True 时生效。
                       'nf4' 把 23GB 的 transformer 压到约 5.8GB；'none' 走原始 bf16（24G+ 卡）。
        - transformer (可选): 来自 Kiki_BNB_QuantizeTransformer 的输出，复用已量化的 transformer。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subject_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": ''}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "low_vram": ("BOOLEAN", {"default": True,
                                         "tooltip": "启用低显存模式"}),
                "quantization": (["nf4", "fp8", "none"], {"default": "nf4",
                                                          "tooltip": "当未连接外部 transformer 时生效"}),
            },
            "optional": {
                "transformer": ("TRANSFORMER", {"tooltip": "可选：来自 BNB 节点的量化 transformer"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Subject (Low VRAM)'

    CATEGORY = "OminiControl"
    DESCRIPTION = "OminiControl subject node (8G VRAM compatible)"

    def run(self, subject_image, prompt, seed, low_vram, quantization, transformer=None):
        mod = _reload_rh_module("subject")
        img = mod.run(subject_image, prompt, seed,
                      low_vram=bool(low_vram),
                      quantization=str(quantization),
                      transformer=transformer)
        return (img,)


class Kiki_Omini_Spatial:
    """
    OminiControl Spatial 节点（8G 显存适配版）。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": ''}),
                "condition_type": (["canny", "depth", "coloring", "deblurring"], ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "low_vram": ("BOOLEAN", {"default": True,
                                         "tooltip": "启用低显存模式"}),
                "quantization": (["nf4", "fp8", "none"], {"default": "nf4"}),
            },
            "optional": {
                "transformer": ("TRANSFORMER",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Spatial (Low VRAM)'

    CATEGORY = "OminiControl"
    DESCRIPTION = "OminiControl spatial node (8G VRAM compatible)"

    def run(self, ref_image, prompt, condition_type, seed, low_vram, quantization, transformer=None):
        mod = _reload_rh_module("spatial")
        img = mod.run(ref_image, prompt, condition_type, seed,
                      low_vram=bool(low_vram),
                      quantization=str(quantization),
                      transformer=transformer)
        return (img,)


class Kiki_Omini_Fill:
    """
    OminiControl Fill 节点（8G 显存适配版）。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ori_image": ("IMAGE",),
                "mask": ("MASK", ),
                "prompt": ("STRING", {"multiline": True,
                                      "default": ''}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "low_vram": ("BOOLEAN", {"default": True,
                                         "tooltip": "启用低显存模式"}),
                "quantization": (["nf4", "fp8", "none"], {"default": "nf4"}),
            },
            "optional": {
                "transformer": ("TRANSFORMER",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Fill (Low VRAM)'

    CATEGORY = "OminiControl"
    DESCRIPTION = "OminiControl fill node (8G VRAM compatible)"

    def run(self, ori_image, mask, prompt, seed, low_vram, quantization, transformer=None):
        mod = _reload_rh_module("fill")
        img = mod.run(ori_image, mask, prompt, seed,
                      low_vram=bool(low_vram),
                      quantization=str(quantization),
                      transformer=transformer)
        return (img,)


NODE_CLASS_MAPPINGS = {
    "OminiControl_BNB_QuantizeTransformer": Kiki_BNB_QuantizeTransformer,
    "OminiControl_Subject": Kiki_Omini_Subject,
    "OminiControl_Spatial": Kiki_Omini_Spatial,
    "OminiControl_Fill": Kiki_Omini_Fill,
}
