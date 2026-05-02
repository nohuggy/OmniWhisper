#!/bin/bash

# Omni_CTC Sync and Setup Script
# This script handles GitHub synchronization and large model management for Lightning.ai

REPO_URL="https://github.com/nohuggy/Omni_CTC"
MODEL_DIR="omnivoice/ctc_forced_aligner"
ONNX_MODEL="$MODEL_DIR/model.onnx"

echo "🔄 Initializing Omni_CTC Repository..."

# 1. Git Initialization
if [ ! -d ".git" ]; then
    git init
    git remote add origin $REPO_URL
    echo "✅ Git initialized and remote added."
else
    echo "ℹ️ Git already initialized."
fi

# 2. Large Model Management (ONNX)
# Shifting to ctc_forced_aligner/model.onnx
mkdir -p $MODEL_DIR

if [ ! -f "$ONNX_MODEL" ]; then
    echo "📥 Downloading essential ONNX model for CTC alignment..."
    # Placeholder for actual download link if known, or using a utility
    # For now, we suggest using a python snippet to download it if not present
    python3 -c "
import os
from huggingface_hub import hf_hub_download
try:
    print('Downloading model.onnx from MahmoudAshraf/mms-300m-1130-forced-aligner...')
    path = hf_hub_download(repo_id='MahmoudAshraf/mms-300m-1130-forced-aligner', filename='model.onnx', local_dir='$MODEL_DIR')
    print(f'✅ Model saved to {path}')
except Exception as e:
    print(f'❌ Download failed: {e}')
"
fi

# 3. Large Model Management (TTS)
# Ensure models/ directory is prepared for Lightning.ai
if [ -d "models/zh_alignment" ]; then
    echo "✅ TTS Models found in models/zh_alignment"
fi

# 4. GitHub Sync
echo "🚀 Syncing with GitHub..."
git add .
# Avoid committing huge binary models directly to git if possible, use .gitignore or LFS
if [ ! -f ".gitignore" ]; then
    cat <<EOF > .gitignore
__pycache__/
*.wav
*.srt
*.zip
temp/
$ONNX_MODEL
models/zh_alignment/*.bin
EOF
fi

git commit -m "Update Omni_CTC with premium UI and CTC-ONNX alignment"
# git push -u origin main  # Uncomment this to push automatically

echo "💡 NOTE: Large models are excluded from Git to keep the repo fast."
echo "💡 To save large models to lightning.ai, ensure they are in the persistent storage path."
echo "✅ Sync preparation complete."
