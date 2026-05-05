#!/bin/bash

# OmniWhisper Kaggle Boot Script
# Specialized for Kaggle environment and T4 GPU stability.

echo "🚀 Starting OmniWhisper Engine (Kaggle Edition)..."

# 1. Enter project directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 2. Clear Port 7860 (Safety check)
if command -v lsof >/dev/null 2>&1; then
    PID=$(lsof -t -i:7860)
    if [ ! -z "$PID" ]; then
        kill -9 $PID
    fi
fi

# 3. Environment Variables for Kaggle
# Force unbuffered output so logs show up immediately in Kaggle console
export PYTHONUNBUFFERED=1
# Ensure current directory is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# 4. Model Check
# app_kaggle.py handles the complex path resolution for /kaggle/input, 
# so we just need to ensure the script is launched correctly.

# 5. Launch the WebUI
echo "🌐 Launching WebUI..."
# Use the -u flag for absolute log transparency
python3 -u omnivoice/app_kaggle.py --share
