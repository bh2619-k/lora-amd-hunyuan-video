# HunyuanVideo 1.5 I2V LoRA Training on AMD GPUs (ROCm)

A complete, battle-tested guide for training HunyuanVideo 1.5 Image-to-Video LoRAs on AMD Radeon GPUs using ROCm. Born from real production training runs — not theory.

> **Why this exists:** Almost every ML training guide assumes NVIDIA. AMD GPU users face undocumented compatibility issues, broken fp8 paths, and cryptic crashes. This repo documents what actually works, what doesn't, and why.

## Overview

HunyuanVideo 1.5 is Tencent's latest video generation model with Image-to-Video (I2V) capability. Training LoRAs lets you teach it new characters, styles, or concepts. This guide covers the full pipeline on AMD GPUs with ROCm.

**Tested on:**
- AMD Radeon RX 9700 XT (32GB VRAM)
- ROCm 7.2 (official AMD wheels)
- Ubuntu 24.04 / Linux 6.17
- musubi-tuner v0.2.15

**Results achieved:**
- 6,400 training steps in ~9 hours
- Final loss: sub-0.001
- Stable training with no crashes after proper setup

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [ROCm Setup (Critical)](#rocm-setup-critical)
3. [Model Downloads](#model-downloads)
4. [Dataset Preparation](#dataset-preparation)
5. [Caching Pipeline](#caching-pipeline)
6. [Training](#training)
7. [Inference](#inference)
8. [Troubleshooting](#troubleshooting)
9. [Performance Notes](#performance-notes)

## Prerequisites

- AMD GPU with 24GB+ VRAM (32GB recommended)
- ROCm 7.2+ installed and working
- Python 3.10+
- ~60GB free disk space (models) + space for cache files
- HuggingFace account (for gated model access)

## ROCm Setup (Critical)

### ⚠️ The #1 Cause of Training Failures on AMD

**Use official AMD ROCm wheels. NOT PyTorch nightlies.**

PyTorch nightly ROCm builds (especially 7.0) have broken `block_swap` functionality that causes `hipErrorIllegalAddress` crashes at random steps (typically 33-77). This is not a config issue — it's a driver/kernel bug in the nightly builds.

```bash
# Create a fresh venv
python3 -m venv ~/hv-venv
source ~/hv-venv/bin/activate

# Install official AMD wheels (ROCm 7.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### fp8 Status on ROCm

| Format | ROCm 7.0 | ROCm 7.2 | Notes |
|--------|-----------|-----------|-------|
| `e4m3fn` (NVIDIA format) | ❌ Broken | ✅ Works | Used by FLUX, HV1.5 fp8 checkpoints |
| `e4m3fnuz` (AMD native) | ❌ Broken | ❌ Broken | hipBLAS unsupported for matmul |
| `e5m2` | Untested | Untested | Rarely used |

**For training:** Use `bf16`. fp8 training is untested on ROCm 7.2 and bf16 is proven stable.

**For inference:** fp8 `e4m3fn` works on ROCm 7.2, halving VRAM usage (33GB → ~17GB for the DiT).

### Validate fp8 Support

```bash
python scripts/check-rocm-fp8.py
```

## Model Downloads

HunyuanVideo 1.5 I2V requires ~53GB of models:

| Model | Size | Purpose |
|-------|------|---------|
| DiT I2V (bf16) | 33.3GB | Main diffusion transformer |
| VAE | 4.7GB | Video autoencoder |
| Qwen2.5-VL-7B | ~16GB | Text encoder |
| BYT5-small | 419MB | Byte-level text encoder |
| SigLIP clip_vision | 817MB | Image encoder (for I2V) |

```bash
# Download from HuggingFace (requires accepted license)
# You need a HuggingFace token with access to gated models
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
# Download the Tencent HunyuanVideo models
snapshot_download('tencent/HunyuanVideo-I2V', local_dir='./models/hunyuan1.5/')
"
```

### Directory Structure

```
models/hunyuan1.5/
├── dit/
│   └── hunyuanvideo_i2v_720_bf16.safetensors  (33.3GB)
├── vae/
│   └── hunyuanvideo_vae_fp32.safetensors       (4.7GB)
├── text_encoder/
│   └── qwen2.5-vl-7b/                          (~16GB)
├── text_encoder_2/
│   └── google_byt5-small_glyphxl/               (419MB)
└── clip_vision/
    └── sigclip_vision_patch14_384.safetensors    (817MB)
```

## Dataset Preparation

### Image Requirements

- **Minimum:** 30 images (50-120 recommended for best quality)
- **Resolution:** Training will bucket automatically, but source images should be at least 960x544
- **Format:** PNG or JPG
- **Variety:** Include different angles, expressions, lighting conditions
- **Face crops:** Include 10-15% face closeups to help lock identity

### Captioning

Each image needs a matching `.txt` caption file. Use a vision model for initial captions, then review manually.

```bash
# Example: Generate captions with a local vision model
# Adjust the endpoint and model to your setup
python scripts/caption-dataset.py --input ./dataset/ --model qwen2.5-vl-7b
```

**Caption tips:**
- Be specific and descriptive
- Include consistent character identifiers (trigger words)
- Describe pose, clothing, setting, lighting
- Fix any hallucinated details (wrong hair color, etc.)

### Folder Structure

```
training/
├── dataset.toml           # Dataset config
├── images/
│   ├── image_001.png
│   ├── image_001.txt      # Caption for image_001
│   ├── image_002.png
│   ├── image_002.txt
│   └── ...
└── cache/                 # Generated cache files go here
```

### Dataset Config (dataset.toml)

```toml
[general]
resolution = [960, 544]    # Width, Height
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_dir = "./images"
num_repeats = 8            # More repeats = more training steps per epoch
```

## Caching Pipeline

HunyuanVideo 1.5 requires pre-caching both latents and text encodings. This is done in two steps.

### Step 1: Cache Latents

This encodes images through the VAE and SigLIP vision encoder.

```bash
python hv_1_5_cache_latents.py \
    --dataset_config ./dataset.toml \
    --vae ./models/hunyuan1.5/vae/hunyuanvideo_vae_fp32.safetensors \
    --vae_chunk_size 32 \
    --vae_spatial_tile_sample_min_size 128 \
    --clip_vision_model ./models/hunyuan1.5/clip_vision/sigclip_vision_patch14_384.safetensors \
    --i2v \
    --batch_size 1 \
    --cache_dir ./cache/
```

### Step 2: Cache Text Encoders

⚠️ **AMD-specific:** Use `--device cpu` for text encoder caching. The Qwen2.5-VL model crashes on AMD GPUs even in bf16 during encoding. CPU caching takes ~17 seconds per item but is rock-solid stable.

```bash
python hv_1_5_cache_text_encoder_outputs.py \
    --dataset_config ./dataset.toml \
    --text_encoder1 ./models/hunyuan1.5/text_encoder/qwen2.5-vl-7b/ \
    --text_encoder2 ./models/hunyuan1.5/text_encoder_2/google_byt5-small_glyphxl/ \
    --batch_size 1 \
    --device cpu \
    --cache_dir ./cache/
```

## Training

### The Training Script

⚠️ **Critical:** Use `hv_1_5_train_network.py` (NOT `hv_train_network.py` — that's for HV 1.0).

```bash
#!/bin/bash
# HunyuanVideo 1.5 I2V LoRA Training - AMD ROCm

# Required environment variables for AMD stability
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=1

python hv_1_5_train_network.py \
    --dit ./models/hunyuan1.5/dit/hunyuanvideo_i2v_720_bf16.safetensors \
    --dataset_config ./dataset.toml \
    --sdpa \
    --mixed_precision bf16 \
    --optimizer_type adafactor \
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
    --learning_rate 2e-5 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora_hv_1_5 \
    --network_dim 128 \
    --network_alpha 64 \
    --timestep_sampling sigmoid \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 32 \
    --save_every_n_epochs 8 \
    --seed 42 \
    --blocks_to_swap 20 \
    --max_grad_norm 1.0 \
    --lr_warmup_steps 50 \
    --cache_latents \
    --cache_text_encoder_outputs \
    --latents_cache_dir ./cache/ \
    --text_encoder_cache_dir ./cache/ \
    --output_dir ./output/ \
    --output_name my-lora-v1 \
    --logging_dir ./logs/
```

### Key Parameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `--network_module` | `networks.lora_hv_1_5` | HV 1.5 specific (NOT `lora_hv`) |
| `--network_dim` | 128 | Higher = more capacity. 64 is standard, 128 for max quality on 32GB+ |
| `--network_alpha` | 64 | Half of dim is the standard ratio |
| `--mixed_precision` | bf16 | Only stable option on AMD ROCm (fp8 training untested) |
| `--optimizer_type` | adafactor | More stable than adamw8bit for LoRA training |
| `--learning_rate` | 2e-5 | Conservative but stable. Range: 1e-5 to 5e-5 |
| `--blocks_to_swap` | 20 | Offloads transformer blocks to CPU. Required for 32GB GPUs |
| `--max_grad_norm` | 1.0 | **CRITICAL on AMD.** Without this, bf16 training goes NaN at step 2 |
| `--lr_warmup_steps` | 50 | **CRITICAL on AMD.** Prevents gradient explosion at training start |
| `--max_train_epochs` | 32 | With 8 repeats × ~90 images = ~6400 steps total |

### ⚠️ bf16 Gradient Explosion (AMD-specific)

Without `--max_grad_norm 1.0` AND `--lr_warmup_steps 50`, training produces NaN loss at step 2. This appears to be an AMD-specific precision issue with bf16 gradients. Both flags are required — using only one is not sufficient.

### Environment Variables

```bash
# Required for AMD GPU stability
export HSA_ENABLE_SDMA=0        # Disable SDMA engine (prevents memory corruption)
export GPU_MAX_HW_QUEUES=1      # Single hardware queue (prevents race conditions)
```

These significantly reduce throughput but prevent random `hipErrorIllegalAddress` crashes.

### Expected Training Performance

| GPU | Speed | 6400 Steps |
|-----|-------|------------|
| RX 9700 XT (32GB) | ~5.2 s/step | ~9 hours |
| RX 7900 XTX (24GB) | ~6-8 s/step (est.) | ~11-14 hours |

## Inference

### ComfyUI (Recommended)

ComfyUI natively supports HunyuanVideo 1.5 I2V. Place your LoRA in the `models/loras/` directory.

**Key inference settings:**
- CFG: 6.0 (non-distilled) or 1.0 (CFG-distilled checkpoint)
- Shift: 7.0
- Steps: 20-30
- Sampler: euler
- Resolution: 544×960 (or 720p for higher quality)

### Known Inference Issues

- Inference is significantly slower than training (~100s/step vs 5.2s/step at 544×960)
- High resolutions (960×960+) may OOM on 32GB
- Start with low resolution and short frame counts to test your LoRA before full generation

## Troubleshooting

### Common Issues & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| `hipErrorIllegalAddress` at random steps | PyTorch nightly ROCm bug | Use official AMD ROCm 7.2 wheels |
| NaN loss at step 2 | bf16 gradient explosion | Add `--max_grad_norm 1.0` AND `--lr_warmup_steps 50` |
| Text encoder crashes on GPU | Qwen2.5-VL memory issue on AMD | Use `--device cpu` for text encoder caching |
| `clip_vision.py` shape error | Expects H,W,C but gets C,H,W | Patch musubi-tuner's `clip_vision.py` with shape transpose |
| Import errors with `lora_hv_1_5` | Wrong musubi-tuner version | Ensure v0.2.15+ (`pip install -e .` from latest git) |
| Training crash at steps 33-77 | ROCm 7.0 block_swap bug | Upgrade to ROCm 7.2 |
| OOM during caching | VAE on GPU with large images | Reduce `--vae_chunk_size` or lower source image resolution |

### The musubi-tuner CLIP Vision Bug

musubi-tuner's `clip_vision.py` may expect image tensors in H,W,C format but receives C,H,W from the data loader. Fix:

```python
# In clip_vision.py, in the encoding function, add:
if image.shape[0] == 3:  # C,H,W format
    image = image.permute(1, 2, 0)  # Convert to H,W,C
```

### Verifying Your ROCm Installation

```bash
python scripts/check-rocm-fp8.py
```

This script checks:
- PyTorch ROCm availability
- GPU detection and VRAM
- bf16 support
- fp8 `_scaled_mm` compatibility
- Basic matmul performance

## Performance Notes

### VRAM Usage (32GB GPU, blocks_to_swap=20)

| Phase | VRAM Usage |
|-------|------------|
| Latent caching | ~8-12GB |
| Text encoder caching (CPU) | ~2GB GPU |
| Training (bf16, dim128) | ~28-30GB |
| Inference (bf16) | ~30GB |
| Inference (fp8) | ~17GB |

### Optimizing Training Speed

1. **`blocks_to_swap`**: Lower = faster but more VRAM. On 32GB, 20 is the sweet spot.
2. **`max_data_loader_n_workers`**: 2-4 depending on CPU cores.
3. **SSD storage**: Cache files are read repeatedly. NVMe recommended.
4. **`HSA_ENABLE_SDMA=0`**: Required for stability but costs ~10-15% speed.

## ☕ Buy Me a Coffee

If this guide saved you from the ROCm 7.0 crash loop or helped you train your first HunyuanVideo LoRA:

<div align="center">

[![Buy Me a Coffee](https://img.shields.io/badge/Buy_Me_a_Coffee-PayPal-ff6b6b.svg)](https://www.paypal.com/ncp/payment/7ABKEV8WHA3KL)

**[☕ Buy me a coffee](https://www.paypal.com/ncp/payment/7ABKEV8WHA3KL)**

</div>

Keeps the GPUs warm and the repos coming.

## Contributing

Found a fix for another AMD GPU model? Training tips for different ROCm versions? PRs welcome.

## License

MIT

## Acknowledgments

- [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) by kohya-ss — The training framework
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) by Tencent — The base model
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — Inference UI
