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
cd OmniWhisper && git pull && bash boot_lightning.sh
```

### 3. Kaggle

#### Fresh Initialisation

```python
# 1. Go to the working directory and remove the old project
!cd /kaggle/working && rm -rf OmniWhisper

# 2. Clone the updated repo
!git clone https://github.com/nohuggy/OmniWhisper.git

# 3. Re-install the new Kaggle requirements
!cd /kaggle/working/OmniWhisper && pip install -r requirements_kaggle.txt

# 4. Create the parent folders and Link (Using Direct Paths)
!mkdir -p /kaggle/working/OmniWhisper/omnivoice /kaggle/working/OmniWhisper/whisper-large-v3-turbo
!ln -sfn /kaggle/input/datasets/etallion/omniaudio/OmniVoice /kaggle/working/OmniWhisper/omnivoice/weights
!ln -sfn /kaggle/input/datasets/etallion/whisper-turbo/whisper-large-v3-turbo /kaggle/working/OmniWhisper/whisper-large-v3-turbo/weights

# 5. Launch the App
!cd /kaggle/working/OmniWhisper && bash boot_kaggle.sh
```

#### Pull and Boot (Unbuffered Logs)

```python
# 1. Sync latest code from GitHub
!cd /kaggle/working/OmniWhisper && git pull origin main

# 2. Direct Linker (Ensures links are fresh and correct)
!mkdir -p /kaggle/working/OmniWhisper/omnivoice /kaggle/working/OmniWhisper/whisper-large-v3-turbo
!ln -sfn /kaggle/input/datasets/etallion/omniaudio/OmniVoice /kaggle/working/OmniWhisper/omnivoice/weights
!ln -sfn /kaggle/input/datasets/etallion/whisper-turbo/whisper-large-v3-turbo /kaggle/working/OmniWhisper/whisper-large-v3-turbo/weights

# 3. Launch the App
!cd /kaggle/working/OmniWhisper && bash boot_kaggle.sh
```

#### Preparing Kaggle Datasets (One-time Setup)

> [!TIP]
> **Why do this?** Kaggle provides over 240GB of permanent storage for Datasets, but only 20GB of temporary space in a Notebook's working directory. By moving these massive models (totaling ~6.6GB) into Datasets, you not only avoid hitting the 20GB disk limit but also significantly speed up the boot time since the models are pre-cached and mounted instantly.

This is even easier because Whisper Large-v3-Turbo is a standard Transformers model. We will follow the same "Worker Notebook" pattern as before to ensure you get the directory structure exactly right.

**Step 1: Create a New "Worker" Notebook**
- Click **+ Create** -> **New Notebook**.
- **Settings (Right Sidebar)**: Toggle **Internet on**.

**Step 2: Run the Download Code**
Paste and run this code. This will download the Turbo model weights from OpenAI's Hugging Face repo.

```python
import os
from huggingface_hub import snapshot_download

# 1. Define the model ID and destination
repo_id = "openai/whisper-large-v3-turbo"
local_dir = "/kaggle/working/whisper-large-v3-turbo"

# 2. Download the repository
print("Starting download of Whisper Turbo...")
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

print(f"Download complete! Files are in: {local_dir}")

# 3. Verify files (You should see model.safetensors, config.json, etc.)
!ls -R {local_dir}
```

**Step 3: Create the Permanent Dataset**
Now, use the same trick as before to save this into your account:
1. Click **"Save Version"** in the top right.
2. Select **"Quick Save"**.
3. Click **"Advanced Settings"** and make sure **"Always save output"** is checked.
4. Click **Save**.
5. Once the bar at the bottom says it is finished, click the **Version Number** (the "1") to view the output.
6. Under the **Output** section, click **"Create Dataset"**.
7. Name it something like `whisper-turbo-weights` and hit **Create**.

**Step 4: Add Both to your Main Notebook**
Now you are ready to build! Go to your Main OmniWhisper Notebook and attach both datasets:
1. Click **+ Add Data**.
2. Add your `omniaudio` dataset.
3. Click **+ Add Data** again.
4. Add your `whisper-turbo-weights` dataset.

## 🚀 Recent Stabilization Fixes (Post-Mortem)

The following critical updates were implemented to ensure stability in production environments (specifically Kaggle):

### 1. "Pro-Subtitle" Splitting (11±4 Rule)
Implemented a **5-Token Guard** in the SRT alignment engine.
- **Orphan Prevention:** If a split would leave fewer than 5 tokens for the next line, the engine automatically merges them into the current line.
- **Conservative Planning:** Switched from `round()` to `int()` for segment budgeting to ensure balanced line distribution across long paragraphs.

### 2. Kaggle Discovery & "Direct Linker"
Resolved the **"X-Ray Path"** issue where models were hidden deep within Kaggle's nested mount system.
- **Hyphen Resolution:** Corrected pathing from `whisper_turbo` to `whisper-turbo` as discovered by recursive scan logs.
- **Symlink Enforcement:** Switched to `ln -sfn` to ensure links are atomic and valid across session restarts.

### 3. Log Buffering Fix
- **The Problem:** `%%bash` in Kaggle cells buffers all output until the process ends, hiding logs from web servers.
- **The Fix:** Replaced with `!` commands and added `sys.stdout.flush()` to all Python logs. You now see the `⚡️ SCRIPT STARTING` heartbeat instantly.

### 4. Scope & Import Integrity
- **NameError Prevention:** All heavy dependencies (Transformers, OmniEngine) are now localized within their specific execution functions. This allows for a fast 1-second startup while ensuring that tools like `unify_punctuation` and `get_slug` are available at runtime.
- **`re` Restoration:** Re-implemented missing regex imports that were causing SRT processing crashes.

## 🔧 Technical Notes
- **Bracket-Aware Splitting**: The SRT alignment engine (`whisper_engine.py`) handles Chinese/English punctuation and brackets correctly to prevent orphaned marks at the start of lines.
- **Gradio 5 UI Optimization**: The audio player uses a dual-format approach (MP3 for web streaming, WAV for processing) to ensure zero-latency loading. The UI generation is optimized to prevent redundant component reloads by maintaining a stable path yield.
- **Auto-Revert Lyric Mode**: The web interface automatically switches between dynamic lyric mode and static SRT text mode when the audio player is paused or stopped using a motion-aware JS scraper.
- **Platform-Specific CPU Scaling**: To prevent "Thread Thrashing" (10x slowdowns on cloud CPUs), the engine now enforces core-specific threading limits:
  - **Lightning.ai & Kaggle**: Locked to 4 cores to match vCPU allocation.
  - **Google Colab**: Locked to 2 cores to match free/pro tier constraints.

## 🛠️ Credits
Powered by **OmniVoice** and **OpenAI Whisper**.
Custom calibration logic by **nohuggy**.
