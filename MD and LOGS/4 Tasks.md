# OmniVoice Pipeline Stabilization - Technical Report

This document details the solutions implemented for the four critical stabilization tasks required for the OmniVoice production pipeline.

## 🛠️ Files Edited
- `/Users/mbp/Downloads/OmniVoice/OmniWhisper/omnivoice/omni_engine.py` (TTS Generator refactor)
- `/Users/mbp/Downloads/OmniVoice/OmniWhisper/omnivoice/app.py` (UI Logic, ZIP generation, Cache busting)
- `/Users/mbp/Downloads/OmniVoice/OmniWhisper/whisper-large-v3-turbo/whisper_engine.py` (Punctuation preservation, Unification utility)

---

## 📝 Solution Notes

### 1. Real-time Progress Reporting (Chunk X/Y)
**Problem**: The UI remained static with a generic "Synthesizing..." message until the entire audio was finished, which is frustrating for long texts.

**Solution**:
- Refactored `TTSEngine.generate` in `omni_engine.py` to act as a **Python Generator**. Instead of returning a single waveform, it now `yield`s progress metadata `(current_chunk, total_chunks, partial_waveform)`.
- Updated `generate_core` in `app.py` to iterate over this generator and `yield` status updates directly to the Gradio frontend.

**Rationale**: Generators allow "streaming" of state without blocking the main event loop. This is the most efficient way to keep the UI responsive in Gradio.
**Risks**: Minimal overhead for yielding, but negligible compared to inference time.

---

### 2. UI Flow & 0-min Preview Stabilization
**Problem**: The "0-min" audio bug prevented web previewing even if the file was valid. ZIP files were often missing the SRT if downloaded too early.

**Solution**:
- **Cache Busting**: Appended a unique timestamp `_123456789` to every generated filename. This forces the browser to treat it as a new file and reload metadata correctly.
- **Synchronized ZIP**: ZIP creation now happens *after* both Audio and SRT are confirmed on disk. The UI only shows the Download button once the archive is ready.

**Rationale**: The 0-min bug is typically a browser-side caching issue where the player tries to use metadata from a previous file with the same name. Unique naming is the industry-standard fix.
**Risks**: Accumulation of files in `outputs/`. Recommendation: Implement a simple cleanup script or manual purge periodically.

---

### 3. Punctuation Preservation
**Problem**: Leading symbols (e.g., `「`, `“`, `【`) were being stripped by the text-splitting logic before synthesis.

**Solution**:
- Modified the regex in `smart_balanced_split` (`whisper_engine.py`). Changed the split pattern from consuming delimiters `[\s.!?。！？…]` to using **Positive Lookbehind** `(?<=[.!?。！？…])`.
- This ensures the punctuation remains attached to the end of the previous segment rather than being discarded or starting a new empty segment.

**Rationale**: Lookbehind allows the regex to match the "gap" after a mark without including the mark itself in the match (which would remove it).
**Risks**: Very complex nested punctuation might require further regex tuning, but current implementation covers all common CJK/English symbols.

---

### 4. Chinese Conversion & Punctuation Unification
**Problem**: Mixed punctuation styles (traditional vs simplified, full-width vs half-width) caused inconsistent subtitle rendering and vocal timing.

**Solution**:
- **Unification Utility**: Created `unify_punctuation` to map all variations of Ellipses, Title Marks, and Quotation marks to standard CJK forms.
- **OpenCC Integration**: Integrated the `opencc-python-reimplemented` library to handle S2T/T2S conversion toggles in the UI.
- **UI Toggle**: Added a "Convert Punctuation" checkbox (Default: ON) in the Advanced settings.

**Rationale**: Standardizing text *before* it hits the TTS engine ensures the AI models see consistent patterns, leading to more stable vocal prosody and cleaner SRT files.
**Risks**: Some users may prefer non-standard marks for artistic reasons. Mitigation: Provided a toggle to disable this behavior.

---

## 🚀 Final Summary
All tasks are verified and integrated. The pipeline now feels like a professional product with real-time feedback and robust file handling.
