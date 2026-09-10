# OminiControl (Low VRAM)

> 仓库地址：[fogyisland/ominicontrol](https://github.com/fogyisland/ominicontrol)
>
> 基于 [Yuanshi9815/OminiControl](https://github.com/Yuanshi9815/OminiControl)，并参考了 HM-RunningHub 的早期 ComfyUI 插件实现。

一个 ComfyUI 插件，封装了 **OminiControl**（基于 FLUX.1-schnell 的主体 / 空间 / 填充生成），并增加 **低显存支持**，让 pipeline 能在 8GB 显存的显卡（如 RTX 3060 / 4060）上跑起来。核心思路是把 pipeline 拆成阶段式加载，并用 bitsandbytes（NF4）或 torchao（FP8）把 FLUX Transformer 量化压缩。同时新增了一个独立的 BNB 加载节点，量化后的 Transformer 可以被多个 OminiControl 节点复用。

## 特性

- **8GB 显存可用** — 原版 23GB 的 FLUX Transformer 装不下的卡也能跑。
- **分阶段加载** — VAE、文本编码器、Transformer 各阶段独立加载、立即释放。
- **NF4 / FP8 量化** — Transformer 权重压到 ~5.8GB（NF4）或 ~11GB（FP8）。
- **CPU offload + attention slicing** — 显存紧张时的兜底机制。
- **共享 Transformer** — 新增的 `OminiControl_BNB_QuantizeTransformer` 节点一次加载量化，下游多个 OminiControl 节点共用。
- **schnell 加速** — Spatial 与 Fill 使用 schnell 模型，减少采样步数。
- **模型路径可自定义** — 便于管理和更新。

## 节点

| 节点 | 类型 | 功能 |
|---|---|---|
| `OminiControl_BNB_QuantizeTransformer` | 加载器 | 加载并量化 FLUX Transformer（NF4 / FP8 / none）。 |
| `OminiControl_Subject` | 生成器 | 主体一致性生成（保留主体身份）。 |
| `OminiControl_Spatial` | 生成器 | 空间条件生成（canny / depth / coloring / deblurring）。 |
| `OminiControl_Fill` | 生成器 | 局部重绘（只改 mask 区域）。 |

所有生成节点的通用参数：

| 输入 | 默认 | 含义 |
|---|---|---|
| `low_vram` | `True` | 启用四阶段流水线 + CPU offload + attention slicing。 |
| `quantization` | `nf4` | 未连接外部 Transformer 时使用的量化方式。`nf4` / `fp8` / `none`。 |
| `transformer` (可选) | — | 接入 `OminiControl_BNB_QuantizeTransformer` 的输出，可共享同一份量化权重。 |

## 安装指南

### 前置条件

- **ComfyUI**：已安装并配置好 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。
- **Python 依赖**：通常无需额外安装，但建议：
  ```
  pip install diffusers==0.31.0
  pip install bitsandbytes     # NF4 量化必需
  # 可选：pip install torchao   # FP8 量化需要（torch >= 2.4）
  ```

### 克隆

```
git clone https://github.com/fogyisland/ominicontrol.git
```

## 模型目录结构

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

### 根据上面的目录结构下载并放置以下模型：

```
FLUX.1-schnell（diffusers 格式）：   https://huggingface.co/black-forest-labs/FLUX.1-schnell
depth-anything-small-hf/（depth）： https://huggingface.co/LiheYoung/depth-anything-small-hf/tree/main
experimental/（LoRA）：             https://huggingface.co/Yuanshi/OminiControl/tree/main/experimental
omini/（subject LoRA）：           https://huggingface.co/Yuanshi/OminiControl/tree/main/omini
```

## 显存占用

| 阶段 | 原版（bf16） | 本 fork（NF4, low_vram=True） |
|---|---|---|
| T5-XXL 文本编码器 | ~5.2 GB | ~0 GB（Transformer 加载前已释放） |
| FLUX Transformer | ~23 GB | **~5.8 GB（NF4）** |
| Attention 中间张量 | ~3 GB | ~1.2 GB（slicing） |
| VAE 编码 / 解码 | ~0.5 GB | ~0.5 GB |
| **峰值** | **~31 GB** | **~7.5 GB** ✓ |

24GB+ 显卡（如 RTX 4090）可把 `low_vram=False`、`quantization=none`，回到原版高性能模式。

## 运行示例

`examples/` 目录下有三个参考工作流（Subject / Spatial / Fill），节点名均为 `OminiControl_*`。

![image](https://github.com/user-attachments/assets/2db219bb-957f-4563-9285-d1a62deb77d1)

### 致谢

- Yuanshi9815 与 [OminiControl](https://github.com/Yuanshi9815/OminiControl) 项目 — 底层实现。
- HM-RunningHub — 早期 ComfyUI 插件实现，本 fork 在结构上有所继承。
- bitsandbytes / torchao — 量化后端。
