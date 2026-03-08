#!/bin/bash
# Download script for Cropper datasets: GAICD, FCDB, SACD

set -e

# Create data directories
DATA_DIR="$(dirname "$0")"
mkdir -p "$DATA_DIR/GAICD"
mkdir -p "$DATA_DIR/FCDB"
mkdir -p "$DATA_DIR/SACD"

echo "=========================================="
echo "Cropper Dataset Download Script"
echo "=========================================="

# Function to check if directory has content
check_dir() {
    if [ "$(ls -A $1 2>/dev/null)" ]; then
        echo "[INFO] $1 already contains files. Skipping download."
        return 0
    fi
    return 1
}

# ==========================================
# GAICD Dataset (Free-form cropping)
# ==========================================
echo ""
echo "[1/3] GAICD Dataset"
echo "-------------------------------------------"
echo "Source: https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch"
echo ""

if ! check_dir "$DATA_DIR/GAICD"; then
    echo "To download GAICD dataset:"
    echo ""
    echo "Option 1: Clone the repository and download from Google Drive"
    echo "  git clone https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch.git"
    echo "  # Download images from the Google Drive link in the repository README"
    echo ""
    echo "Option 2: Direct download (if available)"
    echo "  # The dataset contains 3,336 images with ~90 annotated crops each"
    echo "  # - Training: 2,636 images"
    echo "  # - Validation: 200 images"
    echo "  # - Test: 500 images"
    echo ""
    echo "Expected structure:"
    echo "  GAICD/"
    echo "  ├── images/"
    echo "  │   ├── image_0001.jpg"
    echo "  │   └── ..."
    echo "  ├── annotations/"
    echo "  │   ├── image_0001.txt  (contains crop coordinates and MOS scores)"
    echo "  │   └── ..."
    echo "  └── splits/"
    echo "      ├── train.txt"
    echo "      ├── val.txt"
    echo "      └── test.txt"
    echo ""

    # Try to clone the repo for reference
    if command -v git &> /dev/null; then
        echo "Cloning GAICD repository for reference..."
        git clone --depth 1 https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch.git "$DATA_DIR/GAICD_repo" 2>/dev/null || echo "Could not clone repo, please download manually"
    fi
fi

# ==========================================
# FCDB Dataset (Free-form + Aspect-ratio cropping)
# ==========================================
echo ""
echo "[2/3] FCDB Dataset"
echo "-------------------------------------------"
echo "Source: https://github.com/yiling-chen/flickr-cropping-dataset"
echo ""

if ! check_dir "$DATA_DIR/FCDB"; then
    echo "To download FCDB dataset:"
    echo ""
    echo "Option 1: Clone the repository"
    echo "  git clone https://github.com/yiling-chen/flickr-cropping-dataset.git"
    echo ""
    echo "Option 2: Direct download"
    echo "  # The dataset contains 348 test images"
    echo "  # Each image has a single user-annotated crop box"
    echo ""
    echo "Expected structure:"
    echo "  FCDB/"
    echo "  ├── images/"
    echo "  │   ├── 1.jpg"
    echo "  │   └── ..."
    echo "  └── cropping_testing_set.json"
    echo ""

    # Try to clone the repo
    if command -v git &> /dev/null; then
        echo "Cloning FCDB repository..."
        git clone --depth 1 https://github.com/yiling-chen/flickr-cropping-dataset.git "$DATA_DIR/FCDB_repo" 2>/dev/null || echo "Could not clone repo, please download manually"
    fi
fi

# ==========================================
# SACD Dataset (Subject-aware cropping)
# ==========================================
echo ""
echo "[3/3] SACD Dataset"
echo "-------------------------------------------"
echo "Source: https://github.com/bcmi/Human-Centric-Image-Cropping"
echo ""

if ! check_dir "$DATA_DIR/SACD"; then
    echo "To download SACD dataset:"
    echo ""
    echo "Option 1: Clone the repository and download from provided links"
    echo "  git clone https://github.com/bcmi/Human-Centric-Image-Cropping.git"
    echo "  # Follow the download instructions in the README"
    echo ""
    echo "Option 2: Direct download"
    echo "  # The dataset contains 2,906 images"
    echo "  # - Training: 2,326 images"
    echo "  # - Validation: 290 images"
    echo "  # - Test: 290 images"
    echo "  # Each image has multiple subject masks and corresponding ground-truth crops"
    echo ""
    echo "Expected structure:"
    echo "  SACD/"
    echo "  ├── images/"
    echo "  │   ├── image_001.jpg"
    echo "  │   └── ..."
    echo "  ├── masks/"
    echo "  │   ├── image_001_mask_0.png"
    echo "  │   └── ..."
    echo "  ├── annotations/"
    echo "  │   ├── image_001.json"
    echo "  │   └── ..."
    echo "  └── splits/"
    echo "      ├── train.txt"
    echo "      ├── val.txt"
    echo "      └── test.txt"
    echo ""

    # Try to clone the repo
    if command -v git &> /dev/null; then
        echo "Cloning SACD repository..."
        git clone --depth 1 https://github.com/bcmi/Human-Centric-Image-Cropping.git "$DATA_DIR/SACD_repo" 2>/dev/null || echo "Could not clone repo, please download manually"
    fi
fi

echo ""
echo "=========================================="
echo "Download Script Complete"
echo "=========================================="
echo ""
echo "Please ensure all datasets are placed in the correct directories:"
echo "  - GAICD: $DATA_DIR/GAICD/"
echo "  - FCDB:  $DATA_DIR/FCDB/"
echo "  - SACD:  $DATA_DIR/SACD/"
echo ""
echo "After downloading, verify the data with:"
echo "  python -c \"from datasets import GAICDDataset; d = GAICDDataset('$DATA_DIR/GAICD', 'test'); print(f'GAICD test: {len(d)} images')\""
echo ""
