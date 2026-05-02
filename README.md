# 🎙️ OmniWhisper (Optimized for Lightning.ai)

OmniWhisper is a production-grade **Text-to-Speech (TTS)** and **SRT Generation** pipeline specifically optimized for deployment on **Lightning.ai**. It integrates the high-fidelity voice cloning of OmniVoice with the robust, millisecond-accurate alignment of Whisper-Large-v3-Turbo.

## 🚀 Installation & Launch (Lightning.ai)

### 1. New Installation
If this is your first time, clone the repository and run the setup script:
```bash
git clone https://github.com/nohuggy/OmniWhisper.git
cd OmniWhisper && bash setup_lightning.sh
```

### 2. Launch WebUI
After setup, use the master boot script to start the server:
```bash
cd OmniWhisper && bash boot.sh
```

> [!NOTE]
> This setup is optimized for **Lightning.ai**. It automatically handles model pre-caching and ensures the environment is cleared of hung processes before launch.

## ✨ Key Features
- **🗣️ Zero-Shot Voice Cloning**: High-fidelity cloning via OmniVoice.
- **⏱️ Whisper SRT Alignment**: Millisecond-accurate word mapping using Whisper-Large-v3-Turbo.
- **🎵 Lyrics Scrolling**: Real-time subtitle synchronization in the WebUI.
- **📦 ZIP Packaging**: Integrated packaging of WAV and SRT outputs.

## 🛠️ Credits
Powered by **OmniVoice** and **OpenAI Whisper**.
Custom calibration logic by **nohuggy**.
