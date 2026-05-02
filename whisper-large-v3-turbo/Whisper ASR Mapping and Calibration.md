# Whisper ASR Mapping and Calibration

This document outlines the professional-grade alignment and formatting logic implemented for the OmniVoice SRT production pipeline.

## 💻 Installation & Environment (MacBook Pro Intel 2017)
The system is optimized for a 2017 Intel Core i7 MacBook Pro with 16GB RAM. To ensure stable execution without memory spikes:

1.  **Python Environment**: Use Python 3.9 (recommended for compatibility with `torch` and `transformers` on this hardware).
2.  **Core Dependencies**:
    ```bash
    pip install torch==2.2.2 torchaudio==2.2.2 transformers==4.45.0 numpy
    ```
3.  **Model Storage**: The model is loaded from the local path `./whisper-large-v3-turbo` to bypass internet checks and background verification.

---

## ⚠️ The Root Cause: "The Whisper 30s Wall"
During early validation, we identified a critical drift issue caused by Whisper's internal architecture:
*   **Segment Reset**: Whisper processes audio in 30-second sliding windows. By default, many implementations reset the timestamp to `0.0` at every 30s boundary.
*   **Linear Drift**: Naive "proportional mapping" assumes a constant speech rate. In reality, speakers pause or emphasize words, causing static mapping to drift by up to 2-5 seconds in long articles.
*   **Granularity Gap**: Whisper outputs coarse segments (5-10s). Mapping these directly to short SRT lines results in "frozen" timestamps where multiple lines have the exact same timing.

---

## 🛠 The Professional Fix: Character-Level Sequence Mapping
To resolve the 30s drift and ensure millisecond accuracy, the engine now uses a **Robust Sequence Aligner**:
1.  **Global Clock Reconstruction**: The script implements an offset-accumulator that detects 30s resets and restitches the timeline into a continuous stream.
2.  **Edit-Distance Mapping**: It uses a character-level sequence matching algorithm (`difflib.SequenceMatcher`) to find the exact alignment between your provided reference text and Whisper’s transcription.
3.  **Boundary Locking**: Every line's `Start Time` is mathematically locked to the previous line's `End Time`. This eliminates flickering and ensures a smooth visual flow.
4.  **Non-Linear Interpolation**: Line boundaries are calculated based on the precise character-weight within a vocal burst, rather than a simple duration split.

---

## 🌍 Scalability for 20k Articles (CN, EN, JP, KR)
The engine is architected to handle massive batch processing:
*   **CJK Tokenization**: It uses a dedicated regex engine `[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]` to treat Chinese, Japanese, and Korean characters as individual "units" while keeping English words intact.
*   **Language Agnostic**: The character-mapping logic works identically for all 4 languages, ensuring the same quality standards across your entire library.
*   **VAD Awareness**: The engine "listens" for vocal onsets, ensuring subtitles only appear when there is actual speech, ignoring long silences.

---

## 📏 Formatting & Line Rules
The engine enforces the following production standards:

### 1. Line Length Rule (11 ± 3)
*   The system calculates a **Global Balanced Average** based on the total text length.
*   It targets **11 units** per line (CJK characters or English words).
*   It allows a tolerance of **±3 units** to avoid breaking words or awkward punctuation.

### 2. Smart Punctuation Binding
*   **No Hanging Punctuation**: Line breaks will never occur immediately before a punctuation mark.
*   **Double Punctuation Binding**: Complex markers like `......`, `?!`, `!?`, or `)“` are treated as a single atomic unit and are always kept with their preceding word.
*   **Natural Breaks**: The segmenter heavily penalizes splits that occur mid-sentence. It prioritizes breaking at periods `。`, exclamation marks `！`, and commas `，`.

### 3. Whitespace & Flow
*   **Line Break Removal**: All original newlines in the raw text are merged into spaces before processing.
*   **Space Normalization**: Multiple spaces are collapsed to prevent erratic subtitle alignment.

---

**Engine Script**: `whisper_srt_engine.py` (located in this directory)
**Primary Model**: Whisper-large-v3-turbo

### 🚀 Production Usage (Batch Processing)
To process your 20,000 articles, run the engine from your terminal using the following command:

```bash
# Navigate to the model directory
cd "/Users/mbp/Downloads/OmniVoice/whisper-large-v3-turbo"

# Run the alignment engine
python3 whisper_srt_engine.py \
  --input "/path/to/your/raw/wav_and_txt_folder" \
  --output "/path/to/your/output_srt_folder" \
  --model "."
```

The engine will automatically:
1. Scan the input folder for all `.wav` files.
2. Find the matching `.txt` for each audio file.
3. Apply the Robust Alignment logic to generate high-precision SRTs.
4. Save the results to your specified output folder.
