#!/bin/bash

# Omni_CTC Setup Script - VERSION 1.0 (CTC-ONNX)
set -e

echo "================================================"
echo "   Omni_CTC Setup VERSION 1.0 (CTC-ONNX)"
echo "================================================"

# 1. System Dependencies
echo "[1/4] Step 1: System Dependencies..."
sudo apt-get update -y
sudo apt-get install -y ffmpeg libsndfile1-dev python3-dev build-essential

# 2. Python Selection
echo "[2/4] Step 2: Python Environment..."
PYTHON_BIN=$(which python3)
# Handle Lightning AI specific path
if [ -f "/home/zeus/miniconda3/envs/cloudspace/bin/python3" ]; then
    PYTHON_BIN="/home/zeus/miniconda3/envs/cloudspace/bin/python3"
fi
echo "Selected Python: $PYTHON_BIN"

# --- Milestone A: Torch Ecosystem ---
# Stable Torch version for Lightning AI
echo "--- MILESTONE A: Installing Stable Torch ---"
$PYTHON_BIN -m pip install --upgrade pip setuptools wheel
$PYTHON_BIN -m pip install torch torchvision torchaudio

# --- Milestone B: CTC-Forced-Aligner ---
echo "--- MILESTONE B: Installing Alignment Packages ---"
$PYTHON_BIN -m pip install ctc-forced-aligner unidecode transformers pypinyin tiktoken sentencepiece

# --- Milestone C: App Dependencies ---
echo "--- MILESTONE C: Installing App Dependencies ---"
if [ -f "requirements.txt" ]; then
    $PYTHON_BIN -m pip install -r requirements.txt
fi
$PYTHON_BIN -m pip install "numpy<2" "gradio<6"

# 3. Verification
echo "[4/4] Step 4: Verification..."
$PYTHON_BIN -c "import ctc_forced_aligner; print('✅ ctc-forced-aligner found')"

echo ""
echo "✅ OMNI_CTC SETUP COMPLETE!"
echo "------------------------------------------------"
echo "To start: bash boot.sh"
echo "------------------------------------------------"
