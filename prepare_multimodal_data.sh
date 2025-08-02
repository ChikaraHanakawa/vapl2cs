#!/bin/bash

# Script to prepare and process data for Multimodal VAP model
# This script handles the creation of datasets with image paths

# Set default paths
AUDIO_DIR="${1:-/path/to/audio/files}"
VAD_DIR="${2:-/path/to/vad/files}"
IMAGE_ROOT="${3:-/autofs/diamond3/share/users/hanakawa/datasets}"
OUTPUT_DIR="${4:-data}"

# Print usage information
if [[ "$1" == "-h" || "$1" == "--help" || "$AUDIO_DIR" == "/path/to/audio/files" ]]; then
    echo "Usage: $0 <audio_dir> <vad_dir> [image_root] [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  audio_dir   Directory containing audio files"
    echo "  vad_dir     Directory containing VAD JSON files"
    echo "  image_root  Root directory for image files (default: /autofs/diamond3/share/users/hanakawa/datasets)"
    echo "  output_dir  Directory for output files (default: data)"
    exit 0
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/splits"
mkdir -p "$OUTPUT_DIR/splits_with_images"

echo "======================================================="
echo "Creating Multimodal VAP Datasets"
echo "======================================================="
echo "Audio Directory:     $AUDIO_DIR"
echo "VAD Directory:       $VAD_DIR"
echo "Image Root:          $IMAGE_ROOT"
echo "Output Directory:    $OUTPUT_DIR"
echo "======================================================="

# Step 1: Create initial audio-VAD CSV and sliding window dataset
echo -e "\nStep 1: Creating audio-VAD dataset..."
python create_audio_csv.py \
    --audio_dir "$AUDIO_DIR" \
    --vad_dir "$VAD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --duration 20 \
    --overlap 5 \
    --horizon 2

# Step 2: Add image paths to the datasets
echo -e "\nStep 2: Adding image paths to datasets..."
python add_image_paths.py \
    --train_csv "$OUTPUT_DIR/splits/sliding_window_dset_train.csv" \
    --val_csv "$OUTPUT_DIR/splits/sliding_window_dset_val.csv" \
    --test_csv "$OUTPUT_DIR/splits/sliding_window_dset_test.csv" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR/splits_with_images"

# Step 3: Verify the datasets
echo -e "\nStep 3: Checking datasets..."
TRAIN_CSV="$OUTPUT_DIR/splits_with_images/sliding_window_dset_train_with_images.csv"
VAL_CSV="$OUTPUT_DIR/splits_with_images/sliding_window_dset_val_with_images.csv"
TEST_CSV="$OUTPUT_DIR/splits_with_images/sliding_window_dset_test_with_images.csv"

# Print dataset statistics
echo "Training set:   $(cat "$TRAIN_CSV" | wc -l) samples"
echo "Validation set: $(cat "$VAL_CSV" | wc -l) samples"
echo "Test set:       $(cat "$TEST_CSV" | wc -l) samples"

# Check image availability
echo -e "\nImage availability statistics:"
for CSV in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
    TOTAL=$(cat "$CSV" | wc -l)
    WITH_IMAGES=$(grep -i "true" "$CSV" | wc -l)
    PERCENTAGE=$((WITH_IMAGES * 100 / TOTAL))
    echo "$(basename "$CSV"): $WITH_IMAGES/$TOTAL samples have images ($PERCENTAGE%)"
done

echo -e "\nDataset preparation complete!"
echo "You can now run training with: python multimodal_main.py"
