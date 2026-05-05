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

## 🔧 Technical Notes
- **Bracket-Aware Splitting**: The SRT alignment engine (`whisper_engine.py`) handles Chinese/English punctuation and brackets correctly to prevent orphaned marks at the start of lines.
- **Gradio 5 UI Optimization**: The audio player uses a dual-format approach (MP3 for web streaming, WAV for processing) to ensure zero-latency loading. The UI generation is optimized to prevent redundant component reloads by maintaining a stable path yield.
- **Auto-Revert Lyric Mode**: The web interface automatically switches between dynamic lyric mode and static SRT text mode when the audio player is paused or stopped using a motion-aware JS scraper.

## 🛠️ Credits
Powered by **OmniVoice** and **OpenAI Whisper**.
Custom calibration logic by **nohuggy**.
