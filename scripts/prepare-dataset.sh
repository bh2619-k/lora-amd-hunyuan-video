#!/bin/bash
# prepare-dataset.sh — Prepare image dataset for HunyuanVideo 1.5 LoRA training
#
# Usage:
#   ./prepare-dataset.sh <input_dir> <output_dir> [trigger_word] [num_repeats]
#
# This script:
#   1. Copies images to output directory with sequential naming
#   2. Resizes images that are too small (below 960x544)
#   3. Creates placeholder caption files
#   4. Generates dataset.toml config
#   5. Prints dataset statistics

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [trigger_word] [num_repeats]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [trigger_word] [num_repeats]}"
TRIGGER="${3:-character}"
REPEATS="${4:-8}"

MIN_WIDTH=960
MIN_HEIGHT=544

# Check dependencies
for cmd in convert identify; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "❌ ImageMagick required. Install: sudo apt install imagemagick"
        exit 1
    fi
done

echo "📁 Input:  $INPUT_DIR"
echo "📁 Output: $OUTPUT_DIR"
echo "🏷️  Trigger: $TRIGGER"
echo "🔄 Repeats: $REPEATS"
echo ""

mkdir -p "$OUTPUT_DIR/images"

# Count source images
SOURCES=()
for ext in jpg jpeg png webp; do
    while IFS= read -r -d '' f; do
        SOURCES+=("$f")
    done < <(find "$INPUT_DIR" -maxdepth 1 -iname "*.$ext" -print0 2>/dev/null)
done

if [ ${#SOURCES[@]} -eq 0 ]; then
    echo "❌ No images found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#SOURCES[@]} source images"
echo ""

# Process images
COUNT=0
RESIZED=0
SKIPPED=0

for src in "${SOURCES[@]}"; do
    COUNT=$((COUNT + 1))
    NUM=$(printf "%03d" $COUNT)
    BASENAME="${TRIGGER}_${NUM}"

    # Get dimensions
    DIMS=$(identify -format "%w %h" "$src" 2>/dev/null) || {
        echo "  ⚠️  Skipping (can't read): $(basename "$src")"
        SKIPPED=$((SKIPPED + 1))
        COUNT=$((COUNT - 1))
        continue
    }

    W=$(echo "$DIMS" | cut -d' ' -f1)
    H=$(echo "$DIMS" | cut -d' ' -f2)

    # Check if resize needed
    if [ "$W" -lt "$MIN_WIDTH" ] || [ "$H" -lt "$MIN_HEIGHT" ]; then
        echo "  📐 Resizing $(basename "$src") (${W}x${H} → ≥${MIN_WIDTH}x${MIN_HEIGHT})"
        convert "$src" -resize "${MIN_WIDTH}x${MIN_HEIGHT}^" -quality 95 "$OUTPUT_DIR/images/${BASENAME}.png"
        RESIZED=$((RESIZED + 1))
    else
        # Copy as-is, convert to PNG
        convert "$src" -quality 95 "$OUTPUT_DIR/images/${BASENAME}.png"
    fi

    # Create placeholder caption
    echo "$TRIGGER, [describe this image: pose, clothing, setting, expression, lighting]" > "$OUTPUT_DIR/images/${BASENAME}.txt"

    echo "  ✅ ${BASENAME}.png (${W}x${H})"
done

# Generate dataset.toml
cat > "$OUTPUT_DIR/dataset.toml" << EOF
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_dir = "./images"
num_repeats = $REPEATS
EOF

echo ""
echo "=" * 50
echo "✅ Dataset prepared!"
echo "   Images: $COUNT"
echo "   Resized: $RESIZED"
echo "   Skipped: $SKIPPED"
echo "   Config: $OUTPUT_DIR/dataset.toml"
echo ""
echo "⚠️  NEXT STEPS:"
echo "   1. Edit caption files in $OUTPUT_DIR/images/*.txt"
echo "      (Replace placeholder descriptions with accurate captions)"
echo "   2. Review images for quality and consistency"
echo "   3. Run caching pipeline (see README.md)"
echo ""
echo "📊 Training estimate:"
STEPS=$((COUNT * REPEATS * 32))
HOURS=$(echo "scale=1; $STEPS * 5.2 / 3600" | bc)
echo "   Steps per epoch: $((COUNT * REPEATS))"
echo "   Total steps (32 epochs): $STEPS"
echo "   Estimated time (AMD 32GB): ~${HOURS} hours"
