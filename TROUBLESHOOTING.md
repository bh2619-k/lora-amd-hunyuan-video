# Troubleshooting Guide

Comprehensive solutions for issues encountered during HunyuanVideo 1.5 LoRA training on AMD GPUs.

## Critical Issues (Training Won't Start)

### NaN Loss at Step 2

**Symptoms:** Loss shows `NaN` or `inf` immediately at step 1-2, training continues but produces garbage.

**Cause:** bf16 gradient explosion on AMD GPUs. The initial gradients are too large and overflow bf16 range.

**Fix:** Add BOTH flags to your training command:
```bash
--max_grad_norm 1.0 \
--lr_warmup_steps 50
```

Both are required. Using only one is insufficient — `max_grad_norm` clips the gradients, and `lr_warmup_steps` ramps the learning rate slowly to prevent the initial spike.

### hipErrorIllegalAddress (Random Crashes)

**Symptoms:** Training crashes at seemingly random steps (often 33-77) with `hipErrorIllegalAddress` or memory access violations.

**Cause:** PyTorch nightly ROCm 7.0 has a bug in the `block_swap` mechanism (offloading transformer blocks between GPU and CPU memory).

**Fix:** 
1. Uninstall current PyTorch
2. Install official AMD ROCm 7.2 wheels:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

**Verification:** If you ran 7+ attempts with crashes, a single successful run past step 100 confirms the fix.

### Import Error: `networks.lora_hv_1_5`

**Cause:** Using an older version of musubi-tuner that doesn't have HV 1.5 support.

**Fix:**
```bash
cd musubi-tuner
git pull origin main
pip install -e .
```

Ensure you see `networks/lora_hv_1_5.py` in the repository.

## Caching Issues

### Text Encoder Crashes on GPU

**Symptoms:** OOM or `hipErrorIllegalAddress` during text encoder caching, even with plenty of VRAM.

**Cause:** The Qwen2.5-VL-7B text encoder has issues running on AMD GPUs during the caching phase, even in bf16.

**Fix:** Force CPU mode for text encoder caching:
```bash
python hv_1_5_cache_text_encoder_outputs.py \
    --device cpu \
    ... (other args)
```

This is slower (~17 seconds per caption vs <1 second on GPU) but completely stable. For a 90-image dataset, expect ~25 minutes. This only needs to run once — cache files are reusable.

### CLIP Vision Shape Mismatch

**Symptoms:** Error in `clip_vision.py` about tensor shape, expecting 3 dimensions but getting different order.

**Cause:** musubi-tuner's CLIP vision processing expects H,W,C format but the data pipeline provides C,H,W.

**Fix:** Edit `clip_vision.py` in musubi-tuner:
```python
# Find the image processing section and add:
if image.dim() == 3 and image.shape[0] == 3:  # C,H,W format
    image = image.permute(1, 2, 0)  # Convert to H,W,C
```

### Latent Caching OOM

**Symptoms:** Out of memory during VAE encoding of large images.

**Fix options:**
1. Reduce `--vae_chunk_size` from 32 to 16 or 8
2. Reduce source image resolution
3. Increase `--vae_spatial_tile_sample_min_size` (trades speed for VRAM)

## Training Issues

### Training Extremely Slow

**Expected speeds:**
| GPU (32GB) | blocks_to_swap=20 | Speed |
|------------|-------------------|-------|
| RX 9700 XT | 20 | ~5.2 s/step |
| RX 7900 XTX (24GB) | 24-28 | ~6-8 s/step (est.) |

If you're seeing 50+ s/step:
1. Check `blocks_to_swap` — higher values = more CPU offloading = slower
2. Verify you're using the GPU: `rocm-smi` during training should show GPU utilization
3. Check for CPU bottleneck: `htop` during training
4. Ensure `HSA_ENABLE_SDMA=0` and `GPU_MAX_HW_QUEUES=1` are set (stability trade-off)

### Loss Not Decreasing

For character LoRAs with 50-100 images:
- Epochs 1-8: Loss should drop significantly (0.1 → 0.01 range)
- Epochs 8-16: Continued decrease, slower
- Epochs 16-32: Fine-tuning, sub-0.01, eventually sub-0.001

If loss plateaus early:
1. Check learning rate (try 5e-5 for faster convergence, but risk overfitting)
2. Check `num_repeats` in dataset.toml (too low = underfitting)
3. Verify captions are accurate and consistent
4. Check that images are diverse enough

### Overfitting

**Symptoms:** Loss drops to near 0 very quickly, generated videos look like exact copies of training images.

**Fix:**
1. Reduce epochs (try 16 instead of 32)
2. Lower learning rate (1e-5)
3. Reduce `network_dim` (64 instead of 128)
4. Add more diverse training images
5. Use the intermediate checkpoints (saved every N epochs) — often the epoch 16 checkpoint is best

## Inference Issues

### Inference Much Slower Than Training

**This is normal.** Training at ~5 s/step vs inference at ~100 s/step is expected because:
- Training processes single images
- Inference generates multi-frame video sequences
- Denoising runs the full DiT for each step × each frame

**Tips:**
- Start with low resolution (544×960) and few frames (33-65)
- Use fp8 checkpoint if ROCm 7.2 supports it (halves VRAM)
- CFG-distilled model needs fewer steps (20 vs 50)

### OOM During Inference

The full bf16 model needs ~30GB for inference at 544×960. Options:
1. Use fp8 checkpoint (~17GB) if on ROCm 7.2
2. Lower resolution
3. Reduce frame count
4. Use ComfyUI's built-in memory management

## Environment Variables Reference

```bash
# Required for AMD GPU stability
export HSA_ENABLE_SDMA=0         # Disables System DMA engine
                                  # Prevents memory corruption on some AMD GPUs
                                  # Cost: ~10-15% speed reduction

export GPU_MAX_HW_QUEUES=1       # Limits hardware queue count
                                  # Prevents race conditions in ROCm runtime
                                  # Cost: reduces GPU parallelism slightly

# Optional
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
                                  # May help with memory fragmentation
                                  # Try this if you get OOM with plenty of free VRAM
```

## Getting Help

1. Check musubi-tuner issues: https://github.com/kohya-ss/musubi-tuner/issues
2. ROCm issues: https://github.com/ROCm/ROCm/issues
3. PyTorch ROCm: https://github.com/pytorch/pytorch/issues (tag: module: rocm)
