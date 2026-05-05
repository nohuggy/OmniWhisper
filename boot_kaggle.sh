set -x # Enable shell debug mode

echo "🚀 Starting OmniWhisper Engine (Kaggle Edition)..."

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
