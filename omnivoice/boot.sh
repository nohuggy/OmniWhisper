#!/bin/bash

# OmniVoice + CTC-ONNX Master Boot Script
# This script ensures the environment is perfect before launching.

echo "🚀 Starting OmniVoice CTC Engine..."

# 1. Clear Port 7860 (in case a previous session crashed)
echo "🧹 Cleaning up port 7860..."
if command -v lsof >/dev/null 2>&1; then
    PID=$(lsof -t -i:7860)
    if [ ! -z "$PID" ]; then
        echo "Killing process $PID on port 7860"
        kill -9 $PID
    fi
else
    # Fallback to pkill if lsof is missing
    echo "ℹ️ lsof not found, using pkill fallback..."
    pkill -9 -f "omnivoice/cli/demo.py" || true
    pkill -9 -f "gradio" || true
fi

# 2. Enter project directory
# We assume the script is run from the project root or we can use dirname
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 3. Ensure local models are used
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 4. Pre-cache Whisper model locally (only runs once; skipped if already present)
WHISPER_DIR="$DIR/../whisper-large-v3-turbo"
if [ ! -d "$WHISPER_DIR" ]; then
    echo "📥 Whisper model not found locally. Downloading once to $WHISPER_DIR..."
    unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
    python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='openai/whisper-large-v3-turbo',
    local_dir='$WHISPER_DIR',
    local_dir_use_symlinks=False
)
print('Whisper download complete.')
"
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    echo "✅ Whisper cached at $WHISPER_DIR"
else
    echo "✅ Whisper model found at $WHISPER_DIR"
fi

# 5. Launch the WebUI with Public Sharing
echo "🌐 Launching WebUI..."
# Ensure project root is in PYTHONPATH for module resolution
export PYTHONPATH=$PYTHONPATH:..
# We point to the local folder for the model
python3 app.py --model "./resources" --whisper "$WHISPER_DIR" --share
