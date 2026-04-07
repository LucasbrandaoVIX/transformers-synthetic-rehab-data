#!/bin/bash
# Setup script for MotionGPT3 Rehabilitation Data Generation
# This script sets up the conda environment and downloads all required dependencies.

set -e

echo "========================================="
echo "  MotionGPT3 Rehab Data - Setup Script"
echo "========================================="

# 1. Create conda environment
echo "[1/6] Creating conda environment..."
if conda info --envs | grep -q "mgpt"; then
    echo "  Environment 'mgpt' already exists. Activating..."
else
    conda create python=3.11 --name mgpt -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate mgpt

# 2. Install requirements
echo "[2/6] Installing Python dependencies..."
cd MotionGPT3
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Download SMPL models
echo "[3/6] Downloading SMPL models..."
if [ -d "deps/smpl_models" ]; then
    echo "  SMPL models already downloaded. Skipping..."
else
    bash prepare/download_smpl_model.sh
fi

# 4. Prepare GPT2
echo "[4/6] Preparing GPT2 models..."
if [ -d "deps/gpt2" ]; then
    echo "  GPT2 already downloaded. Skipping..."
else
    bash prepare/prepare_gpt2.sh
fi

# 5. Download pre-trained models
echo "[5/6] Downloading pre-trained models..."
if [ -f "checkpoints/1222_mld_humanml3d_FID041.ckpt" ]; then
    echo "  MLD VAE checkpoint already exists. Skipping..."
else
    bash prepare/download_mld_pretrained_models.sh
fi

if [ -f "checkpoints/motiongpt3.ckpt" ]; then
    echo "  MotionGPT3 checkpoint already exists. Skipping..."
else
    bash prepare/download_pretrained_motiongpt3_model.sh
fi

# 6. Download T2M evaluators
echo "[6/6] Downloading T2M evaluators..."
if [ -d "deps/t2m" ]; then
    echo "  T2M evaluators already downloaded. Skipping..."
else
    bash prepare/download_t2m_evaluators.sh
fi

# Process checkpoints
echo "Processing checkpoints..."
python -m scripts.gen_mot_gpt

# Create data directories for rehabilitation data
echo "Creating rehabilitation data directories..."
cd ..
mkdir -p data/rehab/new_joint_vecs
mkdir -p data/rehab/texts
mkdir -p data/rehab/joints
mkdir -p data/rehab/tmp
mkdir -p data/kimore/raw

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Download KIMORE dataset to data/kimore/raw/"
echo "  2. Run: python scripts/convert_kimore.py"
echo "  3. Run: python scripts/generate_rehab_texts.py"
echo "  4. Fine-tune: python -m train --cfg configs/rehab_stage1.yaml --nodebug"
echo ""
