#!/bin/bash

# OmniWhisper Setup Script - VERSION 1.0 (Whisper-Large-v3-Turbo)
set -e

echo "================================================"
echo "   OmniWhisper Setup VERSION 1.0"
echo "================================================"

# 1. System Dependencies
echo "[1/4] Step 1: System Dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update -y
    sudo apt-get install -y ffmpeg libsndfile1-dev python3-dev build-essential
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "MacOS detected. Ensuring ffmpeg is installed..."
    brew install ffmpeg || echo "Please install ffmpeg manually via brew."
fi

# 2. Python Selection
echo "[2/4] Step 2: Python Environment..."
PYTHON_BIN=$(which python3)
# Handle Lightning AI specific path
if [ -f "/home/zeus/miniconda3/envs/cloudspace/bin/python3" ]; then
    PYTHON_BIN="/home/zeus/miniconda3/envs/cloudspace/bin/python3"
fi
echo "Selected Python: $PYTHON_BIN"

# --- Milestone A: Torch Ecosystem ---
echo "--- MILESTONE A: Installing Torch ---"
$PYTHON_BIN -m pip install --upgrade pip setuptools wheel
$PYTHON_BIN -m pip install "torch>=2.4.0" "torchaudio>=2.4.0" "torchvision>=0.19.0"

# --- Milestone B: Whisper & App Dependencies ---
echo "--- MILESTONE B: Installing Dependencies ---"
if [ -f "requirements.txt" ]; then
    $PYTHON_BIN -m pip install -r requirements.txt
fi
$PYTHON_BIN -m pip install --upgrade "transformers>=4.48.0,<5.0.0" accelerate
$PYTHON_BIN -m pip install "numpy<2" "gradio<6" huggingface_hub

# 3. Verification
echo "[4/4] Step 4: Verification..."
$PYTHON_BIN -c "import torch; import transformers; print('✅ Environment check: OK')"

echo ""
echo "✅ OMNIWHISPER SETUP COMPLETE!"
echo "------------------------------------------------"
echo "To start: bash boot_lightning.sh"
echo "------------------------------------------------"
