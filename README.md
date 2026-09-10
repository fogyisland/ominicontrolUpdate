# OminiControl (Low VRAM)

> Repository: [fogyisland/ominicontrol](https://github.com/fogyisland/ominicontrol)
>
> Based on [Yuanshi9815/OminiControl](https://github.com/Yuanshi9815/OminiControl) and the earlier ComfyUI plugin from HM-RunningHub.

A ComfyUI plugin that wraps **OminiControl** (FLUX.1-schnell based subject / spatial / fill generation) and adds **low-VRAM support** so the pipeline can run on 8 GB GPUs (e.g. RTX 3060 / 4060). The plugin splits the pipeline into discrete load steps and quantizes the FLUX Transformer with bitsandbytes (NF4) or torchao (FP8). A dedicated BNB loader node is provided so the quantized Transformer can be reused across multiple generations.

## Features

- **8 GB VRAM friendly** — runs on GPUs where the original 23 GB FLUX Transformer cannot fit.
- **Pipeline splitting** — VAE, text encoders, and Transformer are loaded only when needed and freed in between.
- **NF4 / FP8 quantization** — Transformer weights compressed to ~5.8 GB (NF4) or ~11 GB (FP8).
- **CPU offload + attention slicing** — defensive fallbacks when VRAM is tight.
- **Shared Transformer** — the new `OminiControl_BNB_QuantizeTransformer` node loads and quantizes the Transformer once, then feeds it to downstream OminiControl nodes.
- **Schnell-based generation** — Spatial and Fill use the schnell model with reduced sampling steps for fast inference.
- **Custom model paths** — easy model management and updates.

## Nodes

| Node | Type | Purpose |
|---|---|---|
| `OminiControl_BNB_QuantizeTransformer` | Loader | Load and quantize the FLUX Transformer once (NF4 / FP8 / none). |
| `OminiControl_Subject` | Generator | Subject-driven generation (keeps the subject identity). |
| `OminiControl_Spatial` | Generator | Spatial-conditioned generation (canny / depth / coloring / deblurring). |
| `OminiControl_Fill` | Generator | Inpainting — repaints only the masked region. |

All generator nodes expose the following inputs:

| Input | Default | Meaning |
|---|---|---|
| `low_vram` | `True` | Enable the four-stage VRAM-friendly pipeline + CPU offload + attention slicing. |
| `quantization` | `nf4` | Quantization applied when no external Transformer is provided. `nf4` / `fp8` / `none`. |
| `transformer` (optional) | — | Plug in the output of `OminiControl_BNB_QuantizeTransformer` to share one quantized Transformer. |

## Installation

### Prerequisites

- **ComfyUI** — installed and configured: [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- **Python dependencies** — most are bundled with ComfyUI. We recommend:
  ```
  pip install diffusers==0.31.0
  pip install bitsandbytes     # required for NF4 quantization
  # Optional: pip install torchao   # required for FP8 quantization (torch >= 2.4)
  ```

### Clone

```
git clone https://github.com/fogyisland/ominicontrol.git
```

## Model Directory Structure

```
/models/flux
tree
.
├── FLUX.1-schnell
│   ├── ae.safetensors
│   ├── model_index.json
│   ├── README.md
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── schnell_grid.jpeg
│   ├── text_encoder
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── text_encoder_2
│   │   ├── config.json
│   │   ├── model-00001-of-00002.safetensors
│   │   ├── model-00002-of-00002.safetensors
│   │   └── model.safetensors.index.json
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── tokenizer_2
│   │   ├── special_tokens_map.json
│   │   ├── spiece.model
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── transformer
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model-00001-of-00003.safetensors
│   │   ├── diffusion_pytorch_model-00002-of-00003.safetensors
│   │   ├── diffusion_pytorch_model-00003-of-00003.safetensors
│   │   └── diffusion_pytorch_model.safetensors.index.json
│   └── vae
│       ├── config.json
│       └── diffusion_pytorch_model.safetensors
└── OminiControl
    ├── depth-anything-small-hf
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── preprocessor_config.json
    │   └── README.md
    ├── experimental
    │   ├── canny.safetensors
    │   ├── coloring.safetensors
    │   ├── deblurring.safetensors
    │   ├── depth.safetensors
    │   ├── fill.safetensors
    │   └── subject.safetensors
    ├── omini
    │   ├── subject_1024_beta.safetensors
    │   └── subject_512.safetensors
    └── README.md
```

### Download the following models into the layout above:

```
FLUX.1-schnell in diffusers format:  https://huggingface.co/black-forest-labs/FLUX.1-schnell
depth-anything-small-hf/ (depth):     https://huggingface.co/LiheYoung/depth-anything-small-hf/tree/main
experimental/ (LoRA):                 https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental
omini/ (subject LoRA):                https://huggingface.co/Yuanshi/OminiControl/tree/main/omini
```

## VRAM Usage

| Stage | Original (bf16) | This fork (NF4, low_vram=True) |
|---|---|---|
| T5-XXL text encoder | ~5.2 GB | ~0 GB (freed before Transformer loads) |
| FLUX Transformer | ~23 GB | **~5.8 GB (NF4)** |
| Attention scratch | ~3 GB | ~1.2 GB (slicing) |
| VAE encode / decode | ~0.5 GB | ~0.5 GB |
| **Peak** | **~31 GB** | **~7.5 GB** ✓ |

On 24 GB+ cards (e.g. RTX 4090), set `low_vram=False` and `quantization=none` to skip quantization and recover the original performance.

## Example Workflows

The `examples/` folder contains three reference workflows (Subject, Spatial, Fill) using the `OminiControl_*` node names.

![image](https://github.com/user-attachments/assets/cc60cbc0-3c44-4da0-8e96-c2f5f89122be)

## Acknowledgments

- Yuanshi9815 and the [OminiControl](https://github.com/Yuanshi9815/OminiControl) project — base implementation.
- HM-RunningHub — original ComfyUI plugin from which this fork inherits the structure.
- bitsandbytes / torchao — quantization backends.
