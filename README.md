# 🎙️ OmniWhisper (Optimized for Lightning.ai)

OmniWhisper is a production-grade **Text-to-Speech (TTS)** and **SRT Generation** pipeline specifically optimized for deployment on **Lightning.ai**. It integrates the high-fidelity voice cloning of OmniVoice with the robust, millisecond-accurate alignment of Whisper-Large-v3-Turbo.

## 📂 Project Structure

The repository is organized for a clean root environment:

```text
OmniWhisper/
├── README.md               # Project documentation
├── requirements.txt        # Shared dependencies
├── omnivoice/              # Core Library & Application
│   ├── app.py              # Master WebUI (Lightning.ai Entry Point)
│   ├── omni_engine.py      # TTS Engine Interface
│   ├── resources/          # OmniVoice weights & config
│   ├── boot.sh / setup.sh  # Deployment scripts
│   └── ...                 
└── whisper-large-v3-turbo/ # Whisper Engine & Calibration
    ├── whisper_engine.py   # Whisper Alignment Interface
    ├── Whisper ASR Mapping and Calibration.md # User Calibration Notes
    └── ...                 
```

## 🚀 Quick Start (Lightning.ai)

### 1. Environment Setup
Run the setup script inside the `omnivoice` folder to install dependencies:
```bash
cd omnivoice && bash setup.sh
```

### 2. Launch WebUI
Use the master boot script to start the server:
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
