# 🎙️ OmniWhisper: Precision TTS & Whisper-Large-v3-Turbo Alignment

OmniWhisper is a production-grade **Text-to-Speech (TTS)** and **SRT Generation** pipeline. It merges the high-fidelity voice cloning of **OmniVoice** with the robust, millisecond-accurate alignment of **Whisper-Large-v3-Turbo**.

## ✨ Key Features

- **🗣️ Zero-Shot Voice Cloning**: Clone any voice with high fidelity.
- **⏱️ Whisper-Large-v3-Turbo SRT**: Millisecond-accurate subtitle generation using Whisper's word-level timestamps instead of CTC.
- **🎵 Lyrics Scrolling**: Real-time subtitle preview in the WebUI.
- **📦 Zip Download**: One-click download of synchronized WAV and SRT pairs.
- **⚡ Lightning.ai Optimized**: Designed for high-performance cloud environments.

## 📂 Project Structure

```text
OmniWhisper/
├── app.py                  # Main WebUI entry point
├── tts_engine.py           # TTS Generation Logic
├── whisper_engine.py       # Whisper SRT Alignment Logic
├── whisper-large-v3-turbo/ # Local Whisper weights
├── omnivoice/              # Core OmniVoice library
└── README.md
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Launch WebUI
```bash
python3 app.py --model . --whisper ./whisper-large-v3-turbo --share
```
> [!NOTE]
> The whisper model is excluded from the repository. On the first run, the system will automatically download `whisper-large-v3-turbo` (approx 1.6GB) from HuggingFace to the `./whisper-large-v3-turbo` folder.

## 🛠️ Components

- **TTS Engine**: Handles voice cloning and speech synthesis.
- **SRT Engine**: Uses `whisper-large-v3-turbo` for robust non-overlapping alignment, especially effective for mixed-language content.
- **WebUI**: Enhanced with real-time lyrics scrolling and automatic ZIP packaging.

## 📜 Credits
Powered by **OmniVoice** and **OpenAI Whisper**.
Mapping & Calibration logic optimized for production-grade subtitle synchronization.
