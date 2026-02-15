#!/bin/bash
# train-hv15-amd.sh — HunyuanVideo 1.5 I2V LoRA Training (AMD ROCm)
#
# Usage:
#   ./train-hv15-amd.sh <config_file>
#   ./train-hv15-amd.sh   (uses default paths)
#
# This is a template — adjust paths to match your setup.
# See README.md for detailed parameter explanations.

set -euo pipefail

# ============================================================
# CONFIGURATION — Edit these paths for your setup
# ============================================================

# Model paths
DIT_MODEL="${DIT_MODEL:-./models/hunyuan1.5/dit/hunyuanvideo_i2v_720_bf16.safetensors}"
VAE_MODEL="${VAE_MODEL:-./models/hunyuan1.5/vae/hunyuanvideo_vae_fp32.safetensors}"
TEXT_ENCODER1="${TEXT_ENCODER1:-./models/hunyuan1.5/text_encoder/qwen2.5-vl-7b/}"
TEXT_ENCODER2="${TEXT_ENCODER2:-./models/hunyuan1.5/text_encoder_2/google_byt5-small_glyphxl/}"
CLIP_VISION="${CLIP_VISION:-./models/hunyuan1.5/clip_vision/sigclip_vision_patch14_384.safetensors}"

# Dataset paths
DATASET_CONFIG="${1:-./dataset.toml}"
CACHE_DIR="${CACHE_DIR:-./cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
LOG_DIR="${LOG_DIR:-./logs}"

# Training parameters
OUTPUT_NAME="${OUTPUT_NAME:-my-hv15-lora-v1}"
NETWORK_DIM="${NETWORK_DIM:-128}"
NETWORK_ALPHA="${NETWORK_ALPHA:-64}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
SAVE_EVERY="${SAVE_EVERY:-8}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-20}"
SEED="${SEED:-42}"

# ============================================================
# AMD ROCm Environment (DO NOT REMOVE)
# ============================================================
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=1

# ============================================================
# Pre-flight checks
# ============================================================

echo "🔬 HunyuanVideo 1.5 I2V LoRA Training (AMD ROCm)"
echo "=================================================="

# Check musubi-tuner
if [ ! -f "hv_1_5_train_network.py" ]; then
    echo "❌ hv_1_5_train_network.py not found!"
    echo "   Run this script from the musubi-tuner directory."
    echo "   git clone https://github.com/kohya-ss/musubi-tuner && cd musubi-tuner"
    exit 1
fi

# Check model files
for f in "$DIT_MODEL" "$VAE_MODEL" "$CLIP_VISION"; do
    if [ ! -f "$f" ]; then
        echo "❌ Model not found: $f"
        exit 1
    fi
done

for d in "$TEXT_ENCODER1" "$TEXT_ENCODER2"; do
    if [ ! -d "$d" ]; then
        echo "❌ Directory not found: $d"
        exit 1
    fi
done

if [ ! -f "$DATASET_CONFIG" ]; then
    echo "❌ Dataset config not found: $DATASET_CONFIG"
    exit 1
fi

# Check ROCm
python3 -c "import torch; assert torch.cuda.is_available(), 'No GPU'" 2>/dev/null || {
    echo "❌ PyTorch ROCm not available!"
    echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2"
    exit 1
}

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "🖥️  GPU: $GPU_NAME"
echo "📁 Dataset: $DATASET_CONFIG"
echo "📁 Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "🔧 dim=$NETWORK_DIM alpha=$NETWORK_ALPHA lr=$LEARNING_RATE epochs=$MAX_EPOCHS"
echo ""

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# ============================================================
# Step 1: Cache Latents
# ============================================================

echo "📦 Step 1/3: Caching latents..."
python hv_1_5_cache_latents.py \
    --dataset_config "$DATASET_CONFIG" \
    --vae "$VAE_MODEL" \
    --vae_chunk_size 32 \
    --vae_spatial_tile_sample_min_size 128 \
    --clip_vision_model "$CLIP_VISION" \
    --i2v \
    --batch_size 1 \
    --cache_dir "$CACHE_DIR"

echo "✅ Latents cached."

# ============================================================
# Step 2: Cache Text Encoders (CPU — required for AMD)
# ============================================================

echo "📦 Step 2/3: Caching text encoders (CPU mode — this takes a while)..."
python hv_1_5_cache_text_encoder_outputs.py \
    --dataset_config "$DATASET_CONFIG" \
    --text_encoder1 "$TEXT_ENCODER1" \
    --text_encoder2 "$TEXT_ENCODER2" \
    --batch_size 1 \
    --device cpu \
    --cache_dir "$CACHE_DIR"

echo "✅ Text encoders cached."

# ============================================================
# Step 3: Train
# ============================================================

echo "🚀 Step 3/3: Training LoRA..."
echo "   This will take several hours. Monitor loss in $LOG_DIR/"
echo ""

python hv_1_5_train_network.py \
    --dit "$DIT_MODEL" \
    --dataset_config "$DATASET_CONFIG" \
    --sdpa \
    --mixed_precision bf16 \
    --optimizer_type adafactor \
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora_hv_1_5 \
    --network_dim "$NETWORK_DIM" \
    --network_alpha "$NETWORK_ALPHA" \
    --timestep_sampling sigmoid \
    --discrete_flow_shift 1.0 \
    --max_train_epochs "$MAX_EPOCHS" \
    --save_every_n_epochs "$SAVE_EVERY" \
    --seed "$SEED" \
    --blocks_to_swap "$BLOCKS_TO_SWAP" \
    --max_grad_norm 1.0 \
    --lr_warmup_steps 50 \
    --cache_latents \
    --cache_text_encoder_outputs \
    --latents_cache_dir "$CACHE_DIR" \
    --text_encoder_cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --logging_dir "$LOG_DIR"

echo ""
echo "🎉 Training complete!"
echo "   LoRA saved to: $OUTPUT_DIR/$OUTPUT_NAME.safetensors"
echo "   Copy to ComfyUI/models/loras/ for inference."
