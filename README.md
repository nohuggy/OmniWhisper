# 🎙️ OmniWhisper (Optimized for Lightning.ai)

OmniWhisper is a production-grade **Text-to-Speech (TTS)** and **SRT Generation** pipeline specifically optimized for deployment on **Lightning.ai**. It integrates the high-fidelity voice cloning of OmniVoice with the robust, millisecond-accurate alignment of Whisper-Large-v3-Turbo.

## 📂 Project Structure

```text
OmniWhisper/
├── README.md               # Project documentation
├── requirements.txt        # Shared dependencies
├── setup_lightning.sh      # Master Setup Script (Root)
├── boot.sh                 # Master Boot Script (Root)
├── omnivoice/              # Core Library & Application
│   ├── app.py              # Master WebUI
│   ├── omni_engine.py      # TTS Engine Interface
│   └── resources/          # OmniVoice weights & config
└── whisper-large-v3-turbo/ # Whisper Engine & Calibration
    ├── whisper_engine.py   # Whisper Alignment Interface
    └── ...                 
```

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
bash boot.sh
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
