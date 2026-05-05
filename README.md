# 🎙️ OmniWhisper

OmniWhisper is a production-grade **Text-to-Speech (TTS)** and **SRT Generation** pipeline. It integrates the high-fidelity voice cloning of OmniVoice with the robust, millisecond-accurate alignment of Whisper-Large-v3-Turbo.

## 🚀 Installation & Launch

### 1. Google Colab
Run it in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nohuggy/OmniWhisper/blob/master/colab.ipynb)

### 2. Lightning.ai
If this is your first time, clone the repository and run the setup script:
```bash
git clone https://github.com/nohuggy/OmniWhisper.git
cd OmniWhisper && bash setup_lightning.sh
```

To launch the server:
```bash
cd OmniWhisper && git pull && bash boot.sh
```

### 3. Kaggle

#### Fresh Initialisation

```bash
%%bash
# 1. Go to the working directory and remove the old project
cd /kaggle/working
rm -rf OmniWhisper

# 2. Clone the updated repo
git clone https://github.com/nohuggy/OmniWhisper.git
cd OmniWhisper

# 3. Re-install the new Kaggle requirements
pip install -r requirements_kaggle.txt

# 4. Create the parent folders for the links
mkdir -p omnivoice
mkdir -p whisper-large-v3-turbo

# 5. THE SMART LINKER (Connects your 100GB Datasets to the code)
OMNI_SRC=$(find /kaggle/input -name "OmniVoice" -type d -print -quit)
WHISPER_SRC=$(find /kaggle/input -name "whisper-large-v3-turbo" -type d -print -quit)

if [ -n "$OMNI_SRC" ]; then
    ln -s "$OMNI_SRC" ./omnivoice/weights
    echo "✅ Linked OmniVoice"
fi

if [ -n "$WHISPER_SRC" ]; then
    ln -s "$WHISPER_SRC" ./whisper-large-v3-turbo/weights
    echo "✅ Linked Whisper Turbo"
fi

echo "----------------------------------------"
echo "📂 Verifying paths exist before launching:"
ls -d ./omnivoice/weights
ls -d ./whisper-large-v3-turbo/weights
```

#### Pull and Boot

```bash
%%bash
# 1. Move to project directory
cd /kaggle/working/OmniWhisper

# 2. Sync latest code from GitHub
echo "🔄 Pulling latest GitHub updates..."
git pull origin main

# 3. Smart Linker (Finds the 6GB models in /kaggle/input)
echo "🔗 Refreshing model links..."
# Remove old links/folders to prevent "broken link" errors
rm -rf ./omnivoice/weights ./whisper-large-v3-turbo/weights
mkdir -p ./omnivoice ./whisper-large-v3-turbo

# Automatically find the real paths in your Kaggle Input
OMNI_SRC=$(find /kaggle/input -name "OmniVoice" -type d -print -quit)
WHISPER_SRC=$(find /kaggle/input -name "whisper-large-v3-turbo" -type d -print -quit)

if [ -n "$OMNI_SRC" ] && [ -n "$WHISPER_SRC" ]; then
    ln -s "$OMNI_SRC" ./omnivoice/weights
    ln -s "$WHISPER_SRC" ./whisper-large-v3-turbo/weights
    echo "✅ Models linked successfully."
else
    echo "❌ ERROR: Datasets not found in /kaggle/input!"
fi

# 4. Launch the App
echo "🚀 Launching OmniWhisper..."
# We set OFFLINE=1 to guarantee it NEVER downloads 5GB to your 20GB space
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python omnivoice/app_kaggle.py
```

## 🔧 Technical Notes
- **Bracket-Aware Splitting**: The SRT alignment engine (`whisper_engine.py`) handles Chinese/English punctuation and brackets correctly to prevent orphaned marks at the start of lines.
- **Gradio 5 UI Optimization**: The audio player uses a dual-format approach (MP3 for web streaming, WAV for processing) to ensure zero-latency loading. The UI generation is optimized to prevent redundant component reloads by maintaining a stable path yield.
- **Auto-Revert Lyric Mode**: The web interface automatically switches between dynamic lyric mode and static SRT text mode when the audio player is paused or stopped using a motion-aware JS scraper.

## 🛠️ Credits
Powered by **OmniVoice** and **OpenAI Whisper**.
Custom calibration logic by **nohuggy**.
