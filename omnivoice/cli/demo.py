#!/usr/bin/env python3
import argparse
import sys
import os

# Add the project root to sys.path to allow importing 'omnivoice'
# When running as 'python3 omnivoice/cli/demo.py', sys.path[0] is the 'cli' folder.
# We need to add the parent of the 'omnivoice' folder.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import logging
import os
import re
import tempfile
import zipfile
import torch
import numpy as np
import soundfile as sf
import gradio as gr

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

try:
    from pypinyin import pinyin, Style
except ImportError:
    pinyin = None
    Style = None

# ---------------------------------------------------------------------------
# Smart subtitle splitting (14±4 CJK chars, 10±4 English words)
# ---------------------------------------------------------------------------
def smart_split(text, language=None):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    lang_lower = (language or "").lower()
    FORBIDDEN_START = "，。！？、）】》”’；：,.!?;:)]}>…"

    def split_into_balanced_chunks(items, max_val, soft_limit, join_str=""):
        if not items:
            return []
        total = len(items)
        if total <= soft_limit:
            return [join_str.join(items)]
        num_chunks = max(2, (total + max_val - 1) // max_val)
        target = total // num_chunks
        PUNCT_SPLIT = "，。！？、；：,.!?;:"
        best_split = -1
        search_start = max(1, target - 4)
        search_end = min(total - 1, target + 4)
        for j in range(search_end, search_start - 1, -1):
            if items[j - 1][-1] in PUNCT_SPLIT:
                best_split = j
                break
        if best_split == -1:
            best_split = target
        return [join_str.join(items[:best_split])] + split_into_balanced_chunks(
            items[best_split:], max_val, soft_limit, join_str
        )

    def finalize_chunks(chunks):
        if not chunks:
            return []
        processed = [chunks[0]]
        for i in range(1, len(chunks)):
            current = chunks[i].strip()
            if not current:
                continue
            while current and current[0] in FORBIDDEN_START:
                processed[-1] += current[0]
                current = current[1:].strip()
            if current:
                processed.append(current)
        return [c for c in processed if c]

    # Detect if text is primarily CJK (for Auto/None language)
    _cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    _total_chars = len(re.sub(r'\s', '', text))
    _is_cjk = _cjk_chars / max(1, _total_chars) > 0.3

    if lang_lower in ("eng", "en") or (not lang_lower and not _is_cjk) or (lang_lower in ("auto", "") and not _is_cjk):
        # English / Latin: split at sentence boundaries, then word-level chunks
        max_val, soft_limit = 10, 14
        sentences = re.split(r"(?<=[.!?]) +", text)
        all_chunks = []
        for s in sentences:
            words = s.split()
            if words:
                all_chunks.extend(split_into_balanced_chunks(words, max_val, soft_limit, " "))
        return finalize_chunks(all_chunks)
    else:
        # CJK — sentence-level split, then smarter itemization for chunks
        max_val, soft_limit = 14, 18
        sentences = re.split(r"(?<=[。！？])", text)
        all_chunks = []
        for s in sentences:
            s = s.strip()
            if s:
                # Keep English phrases/words/numbers together, but treat Chinese chars individually
                # Using a more robust regex to capture whole English sequences
                items = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9\s\']*[a-zA-Z0-9\']+[a-zA-Z0-9\s\']*|[^\w\s]', s)
                # Clean up items (remove accidental leading/trailing spaces from phrase capture)
                items = [it.strip() for it in items if it.strip() or it == " "]
                
                # Join with space first, then clean up CJK-specific spacing
                chunks = split_into_balanced_chunks(items, max_val, soft_limit, " ")
                
                cleaned_chunks = []
                for c in chunks:
                    # Remove spaces between two CJK chars
                    c = re.sub(r'([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', r'\1', c)
                    # Remove spaces between CJK and punctuation
                    c = re.sub(r'([\u4e00-\u9fff])\s+(?=[^\w\s])', r'\1', c)
                    c = re.sub(r'([^\w\s])\s+(?=[\u4e00-\u9fff])', r'\1', c)
                    cleaned_chunks.append(c.strip())
                all_chunks.extend(cleaned_chunks)
        return finalize_chunks(all_chunks)


# ---------------------------------------------------------------------------
# Language code normalisation for ctc-forced-aligner (ISO-639-3)
# ---------------------------------------------------------------------------
_LANG_TO_ISO3 = {
    "zh": "cmn", "cmn": "cmn", "chinese": "cmn", "mandarin": "cmn",
    "zh-cn": "cmn", "zh-tw": "cmn", "zh-hk": "cmn", "zho": "cmn",
    "en": "eng", "eng": "eng", "english": "eng",
    "ja": "jpn", "jpn": "jpn", "japanese": "jpn",
    "ko": "kor", "kor": "kor", "korean": "kor",
    "fr": "fra", "fra": "fra", "french": "fra",
    "de": "deu", "deu": "deu", "german": "deu",
    "es": "spa", "spa": "spa", "spanish": "spa",
}

def _to_iso3(language):
    """Convert a loose language tag to an ISO-639-3 code for ctc-forced-aligner."""
    if not language:
        return "eng"
    key = language.strip().lower()
    return _LANG_TO_ISO3.get(key, key[:3] if len(key) >= 3 else "eng")


# ---------------------------------------------------------------------------
# SRT generation via ctc-forced-aligner (ONNX path)
# ---------------------------------------------------------------------------

# Module-level cache so the model is only loaded once per session
_CTC_MODEL = None
_CTC_TOKENIZER = None

def _load_ctc_model(device, model_name="MahmoudAshraf/mms-300m-1130-forced-aligner"):
    global _CTC_MODEL, _CTC_TOKENIZER
    if _CTC_MODEL is None:
        from ctc_forced_aligner import load_alignment_model
        print(f"Loading CTC alignment model '{model_name}'...")
        _CTC_MODEL, _CTC_TOKENIZER = load_alignment_model(device, model_name)
        print("CTC alignment model loaded.")
    return _CTC_MODEL, _CTC_TOKENIZER


def format_timestamp(seconds):
    td = float(seconds)
    hours = int(td // 3600)
    minutes = int((td % 3600) // 60)
    secs = int(td % 60)
    millis = int(round((td % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# Global cache to prevent reloading the 300MB model every time you click "Generate"
_ALIGN_CACHE = {"model": None, "tokenizer": None}

def text_to_srt_ctc(text: str, audio_tuple, language=None) -> str:
    """Generate an SRT string using ctc-forced-aligner's ONNX model."""
    from ctc_forced_aligner import (
        generate_emissions,
        preprocess_text,
        get_alignments,
        load_audio,
    )

    global _ALIGN_CACHE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _ALIGN_CACHE["model"] is None:
        print("--- Initializing Aligner Model Cache ---")
        align_model, align_tokenizer = _load_ctc_model(device)
        _ALIGN_CACHE["model"] = align_model
        _ALIGN_CACHE["tokenizer"] = align_tokenizer
    
    align_model = _ALIGN_CACHE["model"]
    align_tokenizer = _ALIGN_CACHE["tokenizer"]

    sr, waveform_int16 = audio_tuple
    print(f"[SRT] Input audio: {waveform_int16.shape}, sr={sr}")

    # Robust mono conversion and resampling to 16k for Aligner
    waveform_t = torch.from_numpy(waveform_int16).float() / 32767.0
    if waveform_t.ndim > 1:
        # Average channels to mono
        waveform_t = waveform_t.mean(dim=0 if waveform_t.shape[0] < waveform_t.shape[1] else 1)
    
    # Ensure (1, T) for torchaudio
    if waveform_t.ndim == 1:
        waveform_t = waveform_t.unsqueeze(0)
    
    if sr != 16000:
        import torchaudio.functional as AF
        audio_waveform = AF.resample(waveform_t, sr, 16000).to(device).to(align_model.dtype)
    else:
        audio_waveform = waveform_t.to(device).to(align_model.dtype)
    
    # Flatten to 1D (T,) to avoid [1,1,1,T] dimension errors in ctc-forced-aligner
    audio_waveform = audio_waveform.flatten()
    
    print(f"[SRT] Resampled audio: {audio_waveform.shape}, device={audio_waveform.device}")

    # Energy check: Print mean amplitude per second to verify audio content
    try:
        audio_np = audio_waveform.cpu().numpy()
        for sec in range(int(len(audio_np)/16000)):
            chunk = audio_np[sec*16000 : (sec+1)*16000]
            energy = np.abs(chunk).mean()
            print(f"[SRT] Audio Energy Sec {sec}: {energy:.4f}")
    except Exception as e:
        print(f"[SRT] Energy check failed: {e}")

    try:
        from pypinyin import pinyin, Style
        segments = smart_split(text, language)
        # FORCE 'eng' for this specific model as it only has 32 output classes
        lang_code = "eng" 
        romanize = True 

        # Deterministic token counting per segment to avoid alignment drift
        seg_token_counts = []
        all_tokens_list = []
        
        for i, s in enumerate(segments):
            # Clean and tokenize each segment individually
            # For this model, we convert Chinese to Pinyin to match its English phoneme head
            p_list = []
            for item in pinyin(s, style=Style.NORMAL):
                p_list.append(item[0])
            p_text = " ".join(p_list)
            
            s_clean = re.sub(r"[^\w\s]", " ", p_text).lower().strip()
            s_toks = s_clean.split()
            seg_token_counts.append(len(s_toks))
            all_tokens_list.extend(s_toks)
            print(f"[SRT] Seg {i+1}: '{s}' -> {len(s_toks)} tokens (Pinyin)")

        clean_text = " ".join(all_tokens_list)
        print(f"[SRT] Full clean text: {clean_text[:100]}...")

        # Enforce float32 for emissions to prevent precision-loss-induced truncation on GPU
        align_model = align_model.float()
        audio_waveform = audio_waveform.float()
        
        emissions, _stride_unused = generate_emissions(align_model, audio_waveform)
        
        # Manual stride calculation (seconds per frame)
        # MMS usually has a downsampling factor of 320 at 16k (20ms)
        num_frames = emissions.shape[0] if emissions.ndim == 2 else emissions.shape[1]
        audio_len_samples = audio_waveform.shape[0]
        frame_to_sec = (audio_len_samples / num_frames) / 16000.0
        
        print(f"[SRT] Emissions: {emissions.shape}, Calculated Stride: {frame_to_sec*1000:.2f}ms")

        tokens_starred, text_starred = preprocess_text(
            clean_text, romanize=romanize, language=lang_code
        )

        raw_segments, scores, blank_token = get_alignments(
            emissions, tokens_starred, align_tokenizer
        )
        print(f"[SRT] Aligned {len(raw_segments)} raw segments. Stride: {frame_to_sec*1000:.2f} ms")
        print(f"[SRT] Verification: {len(text_starred)} aligned tokens vs {len(all_tokens_list)} requested tokens.")

        # --- Build timestamps directly from raw_segments ---
        
        token_timestamps = []
        for token, seg in zip(text_starred, raw_segments):
            if seg.start is not None and seg.end is not None:
                t_start = seg.start * frame_to_sec
                t_end   = seg.end   * frame_to_sec
                token_timestamps.append({"token": token, "start": t_start, "end": t_end})
        
        # DEBUG: Print first and last token timestamps
        if token_timestamps:
            print(f"[SRT] First token '{token_timestamps[0]['token']}' at {token_timestamps[0]['start']:.2f}s")
            print(f"[SRT] Last token '{token_timestamps[-1]['token']}' at {token_timestamps[-1]['start']:.2f}s")

        if not token_timestamps:
            return "No alignment result produced."

        total_tokens = len(token_timestamps)

        # Map tokens back to segments using deterministic counts
        # token_timestamps contains original tokens + auxiliary tokens (like <star>)
        srt_output = ""
        curr_aligned_idx = 0
        
        for i, (seg_text, requested_count) in enumerate(zip(segments, seg_token_counts), 1):
            if requested_count == 0:
                continue
                
            # Find the range in token_timestamps that covers 'requested_count' non-auxiliary tokens
            start_aligned_idx = curr_aligned_idx
            found_count = 0
            while curr_aligned_idx < len(token_timestamps) and found_count < requested_count:
                tok = token_timestamps[curr_aligned_idx]["token"]
                if tok != "<star>":
                    found_count += 1
                curr_aligned_idx += 1
            
            # Include trailing <star> tokens for this segment if any
            while curr_aligned_idx < len(token_timestamps) and token_timestamps[curr_aligned_idx]["token"] == "<star>":
                curr_aligned_idx += 1
                
            end_aligned_idx = curr_aligned_idx - 1
            
            if start_aligned_idx <= end_aligned_idx:
                t_start = token_timestamps[start_aligned_idx]["start"]
                t_end   = token_timestamps[end_aligned_idx]["end"]
                
                # Sanity check for minimum duration
                if t_end <= t_start:
                    t_end = t_start + 0.5

                srt_output += f"{i}\n"
                srt_output += f"{format_timestamp(t_start)} --> {format_timestamp(t_end)}\n"
                srt_output += f"{seg_text}\n\n"
            else:
                break

        return srt_output.strip()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Alignment Engine Error: {e}"


def text_to_srt_whisper(text, audio_tuple, model_instance, language="cmn"):
    """
    Generate SRT using Whisper's robust word-level timestamps.
    """
    try:
        sr, waveform = audio_tuple
        print(f"[SRT] Starting Whisper alignment...")
        
        # Get word-level timestamps from Whisper
        result = model_instance.transcribe((waveform, sr), return_timestamps="word")
        
        # Result is a dict with 'text' and 'chunks' (which contain 'text', 'timestamp': (start, end))
        chunks = result.get("chunks", [])
        if not chunks:
            return "Whisper failed to produce timestamps."
            
        segments = smart_split(text, language)
        seg_token_counts = []
        all_words = []
        for s in segments:
            # Simple word/char splitting for matching
            tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", s)
            seg_token_counts.append(len(tokens))
            all_words.extend(tokens)
            
        print(f"[SRT] Whisper found {len(chunks)} word chunks. Mapping to {len(all_words)} requested tokens.")
        
        # Map Whisper chunks to segments
        srt_output = ""
        chunk_idx = 0
        for i, (seg_text, requested_count) in enumerate(zip(segments, seg_token_counts), 1):
            if requested_count == 0:
                continue
                
            start_time = None
            end_time = None
            
            # Find the start time of the first word in this segment
            # We skip whitespace/punctuation chunks if Whisper returns them
            found = 0
            while chunk_idx < len(chunks) and found < requested_count:
                c = chunks[chunk_idx]
                if start_time is None:
                    start_time = c["timestamp"][0]
                end_time = c["timestamp"][1]
                found += 1
                chunk_idx += 1
            
            if start_time is not None:
                if end_time is None: end_time = start_time + 1.0
                srt_output += f"{i}\n"
                srt_output += f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n"
                srt_output += f"{seg_text}\n\n"
        
        return srt_output.strip()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Whisper Alignment Error: {e}"


# ---------------------------------------------------------------------------
# Slug helper for filenames
# ---------------------------------------------------------------------------
def get_slug(text, max_tokens=8):
    clean_text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", clean_text)
    selected = tokens[:max_tokens]
    if not selected:
        return "output"
    res = selected[0]
    for i in range(1, len(selected)):
        prev, curr = selected[i - 1], selected[i]
        if re.match(r"[a-zA-Z0-9]", prev) or re.match(r"[a-zA-Z0-9]", curr):
            res += " " + curr
        else:
            res += curr
    return res.strip()


# ---------------------------------------------------------------------------
# Voice Design categories
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": ["Child / 儿童", "Teenager / 少年", "Young Adult / 青年", "Middle-aged / 中年", "Elderly / 老年"],
    "Pitch / 音调": ["Very Low Pitch / 极低音调", "Low Pitch / 低音调", "Moderate Pitch / 中音调", "High Pitch / 高音调", "Very High Pitch / 极高音调"],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": ["American Accent / 美式口音", "British Accent / 英国口音", "Australian Accent / 澳大利亚口音", "Chinese Accent / 中国口音"],
    "Chinese Dialect / 中文方言": ["Northeast Dialect / 东北话", "Sichuan Dialect / 四川话", "Henan Dialect / 河南话", "Shaanxi Dialect / 陕西话"],
}

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_demo(model):
    """
    TECHNICAL NOTE ON TTS RESTORATION:
    The synthesis pipeline was restored to stability by:
    1. Enforcing torch.float32 for CPU environments (fixing 211s latency/audio glitches).
    2. Synchronizing positional wrappers (_clone_fn, _design_fn) with the stable Aeneas reference.
    3. Manually bypassing the broken 'torchcodec' ASR pipeline in the model class.
    """
    sampling_rate = model.sampling_rate

    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        generate_srt=True,
        ref_text=None,
        srt_language="Chinese",
    ):
        if not text or not text.strip():
            return None, "", None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "", None, "Please upload a reference audio."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        import time
        start_time = time.time()

        clean_log_text = text[:60].replace('\n', ' ')
        print(f"[TTS] Generating audio for: {clean_log_text}...")
        
        try:
            audio = model.generate(**kw)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, "", None, f"Error: {type(e).__name__}: {e}"

        elapsed = time.time() - start_time
        # Accurate word count for CJK and English
        word_count = len(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text))
        print(f"[TTS] Audio generated in {elapsed:.1f}s ({word_count} tokens/words)")

        # Ensure waveform is 1D (T,) for correct sf.write channel handling
        waveform = (audio[0] * 32767).astype(np.int16).squeeze()
        audio_tuple = (sampling_rate, waveform)

        # Generate slug for filenames
        slug = get_slug(text)
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"{slug}.wav")
        sf.write(audio_path, waveform, sampling_rate)

        srt_content = ""
        download_path = audio_path

        if generate_srt:
            print("[SRT] Starting forced alignment (CTC-ONNX)...")
            try:
                # Use passed srt_language, default to Chinese if detect CJK
                actual_srt_lang = srt_language
                if not actual_srt_lang or actual_srt_lang == "Auto":
                     actual_srt_lang = "Chinese" if any('\u4e00' <= c <= '\u9fff' for c in text) else "English"
                
                iso_lang = _to_iso3(actual_srt_lang)
                
                # Robust switch: Use Whisper for alignment as it's much better for mixed languages
                print(f"[SRT] Using Whisper-based alignment for precision...")
                srt_content = text_to_srt_whisper(text, audio_tuple, model, language=iso_lang)
                
                srt_path = audio_path.replace(".wav", ".srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                print("[SRT] Alignment complete.")
            except Exception as e:
                print(f"[SRT] Alignment failed: {e}")
                srt_content = f"Alignment failed: {e}"
                srt_path = None

            # Create a zip file containing both audio and srt
            zip_path = os.path.join(temp_dir, f"{slug}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(audio_path, arcname=f"{slug}.wav")
                if srt_path and os.path.exists(srt_path):
                    zipf.write(srt_path, arcname=f"{slug}.srt")
            download_path = zip_path

        status_msg = f"{elapsed:.1f}s / {word_count} words"
        print("[TTS] Request complete, sending payload to Gradio UI...")
        return audio_path, srt_content, download_path, status_msg

    def _clone_fn(
        text,
        lang,
        ref_aud,
        ref_text,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        gen_srt,
        srt_lang,
    ):
        return _gen_core(
            text,
            lang,
            ref_aud,
            instruct,
            num_step,
            guidance_scale,
            denoise,
            speed,
            duration,
            preprocess_prompt,
            postprocess_output,
            mode="clone",
            generate_srt=gen_srt,
            ref_text=ref_text or None,
            srt_language=srt_lang,
        )

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    /* ABSOLUTE NUKE: Force #1f2937 on the panel and EVERY SINGLE CHILD element */
    .output-panel, 
    .output-panel * {
        background-color: #1f2937 !important;
        background: #1f2937 !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Ensure the main panel still has 0 gap */
    .output-panel {
        gap: 0 !important;
        overflow: visible !important;
    }

    /* Exempt the specific elements that need their own color */
    .output-panel .custom-label {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: var(--block-label-text-size, 13px);
        font-weight: 700;
        color: #fff !important;
        background-color: #4f46e5 !important;
        background: #4f46e5 !important;
        padding: 4px 10px;
        border-radius: 6px !important;
        margin-top: -1px !important;
        margin-left: -1px !important;
        margin-bottom: 0 !important;
        z-index: 10;
        width: fit-content;
    }
    .output-panel .custom-label *, .output-panel .custom-label svg {
        background-color: #4f46e5 !important;
        background: #4f46e5 !important;
        width: 14px; height: 14px;
    }
    
    .lyrics-viewer {
        height: 260px;
        width: 100% !important;
        overflow-y: auto;
        padding: 10px 20px !important;
        display: none;
        box-sizing: border-box;
    }
    .lyrics-viewer .lyric-line {
        text-align: center;
        padding: 6px 12px;
        margin: 4px 0;
        border-radius: 6px;
        transition: all 0.2s ease;
        color: var(--body-text-color-subdued, #888);
        font-size: 1.1em;
        line-height: 1.4;
    }
    .lyrics-viewer .lyric-line.active,
    .lyrics-viewer .lyric-line.active * {
        color: var(--body-text-color, #111) !important;
        font-weight: 600 !important;
        font-size: 1.25em !important;
        background-color: var(--color-accent-soft, rgba(59,130,246,0.1)) !important;
        background: var(--color-accent-soft, rgba(59,130,246,0.1)) !important;
    }
    .lyrics-viewer.hidden-lyrics { display: none !important; }
    """

    # We use demo.load(js=...) instead of head= or gr.HTML() to avoid parser issues.
    _LYRICS_JS = """
    () => {
        console.log('[Lyrics] Script starting (Enhanced Polling Mode)...');
        
        function parseSRT(raw) {
            if (!raw) return [];
            var blocks = raw.trim().split(/\\n\\n+/);
            var cues = [];
            for (var b = 0; b < blocks.length; b++) {
                var lines = blocks[b].split('\\n');
                if (lines.length < 3) continue;
                var times = lines[1].split(' --> ');
                if (times.length !== 2) continue;
                var s = times[0].replace(',','.').split(':');
                var e = times[1].replace(',','.').split(':');
                var startSec = parseFloat(s[0])*3600 + parseFloat(s[1])*60 + parseFloat(s[2]);
                var endSec   = parseFloat(e[0])*3600 + parseFloat(e[1])*60 + parseFloat(e[2]);
                var txt = lines.slice(2).join(' ');
                cues.push({start: startSec, end: endSec, text: txt});
            }
            return cues;
        }

        function renderLyrics(viewer, cues, activeIdx) {
            if (!cues.length) {
                viewer.innerHTML = '<div style="text-align:center;color:#aaa;padding:40px;">No subtitles</div>';
                return;
            }
            
            // Initial render if empty or cue count changed
            if (!viewer.children.length || viewer._cueCount !== cues.length) {
                var html = '';
                for (var i = 0; i < cues.length; i++) {
                    html += '<div class="lyric-line" data-idx="' + i + '">' + cues[i].text + '</div>';
                }
                viewer.innerHTML = html;
                viewer._cueCount = cues.length;
            }

            if (viewer._lastActive === activeIdx) return;
            viewer._lastActive = activeIdx;

            // Update classes incrementally
            var lines = viewer.children;
            for (var i = 0; i < lines.length; i++) {
                var el = lines[i];
                var idx = parseInt(el.getAttribute('data-idx'));
                if (idx === activeIdx) {
                    el.classList.add('active');
                    el.classList.remove('past');
                    
                    // TARGETED SCROLL: Only scroll the viewer, NOT the whole page
                    var targetTop = el.offsetTop - (viewer.offsetHeight / 2) + (el.offsetHeight / 2);
                    viewer.scrollTo({ top: targetTop, behavior: 'smooth' });
                } else if (idx < activeIdx) {
                    el.classList.add('past');
                    el.classList.remove('active');
                } else {
                    el.classList.remove('active', 'past');
                }
            }
        }

        var PAIRS = [
            ['vc-audio', 'vc-lyrics', 'vc-srt-text'],
            ['vd-audio', 'vd-lyrics', 'vd-srt-text']
        ];

        function getSRT(srtBoxId) {
            var box = document.getElementById(srtBoxId);
            if (!box) return '';
            var ta = box.querySelector('textarea');
            return ta ? ta.value : '';
        }

        function parseTimeStr(s) {
            var p = s.split(':');
            if (p.length === 2) return parseInt(p[0])*60 + parseFloat(p[1]);
            if (p.length === 3) return parseInt(p[0])*3600 + parseInt(p[1])*60 + parseFloat(p[2]);
            return 0;
        }

        function updateLyrics() {
            PAIRS.forEach(function(pair) {
                var audioId = pair[0], lyricsId = pair[1], srtBoxId = pair[2];
                var audioContainer = document.getElementById(audioId);
                var viewer = document.getElementById(lyricsId);
                var rawBox = document.getElementById(srtBoxId);
                if (!audioContainer || !viewer || !rawBox) return;

                // Debug: Log structure once
                if (!audioContainer._logged) {
                    console.log('[Lyrics] Container ' + audioId + ' structure:', audioContainer.innerHTML);
                    audioContainer._logged = true;
                }

                var currentTime = -1;
                
                // Method 1: Hidden Audio Element (Standard)
                var audioEl = audioContainer.querySelector('audio');
                if (audioEl && !audioEl.paused && audioEl.currentTime > 0) {
                    currentTime = audioEl.currentTime;
                }
                
                // Method 2: UI Scraper (For Gradio 5 Waveform / WebAudio)
                if (currentTime < 0) {
                    var allText = audioContainer.innerText;
                    // Look for patterns like "0:05 / 0:10" or "0:05"
                    var matches = allText.match(/(\\d+:\\d+)/g);
                    if (matches && matches.length > 0) {
                        currentTime = parseTimeStr(matches[0]);
                        // Only count as "playing" if time is > 0 and changing
                        if (currentTime === viewer._lastTime) {
                            if (Date.now() - (viewer._lastTimeUpdate || 0) > 1000) {
                                currentTime = -1; // Assume paused
                            }
                        } else {
                            viewer._lastTime = currentTime;
                            viewer._lastTimeUpdate = Date.now();
                        }
                    }
                }

                if (currentTime >= 0) {
                    if (rawBox.style.display !== 'none') {
                        console.log('[Lyrics] Play detected on ' + audioId + ' at ' + currentTime);
                        rawBox.style.display = 'none';
                        viewer.style.display = 'block';
                        viewer._cues = parseSRT(getSRT(srtBoxId));
                    }
                    
                    var cues = viewer._cues || [];
                    var activeIdx = -1;
                    for (var i = 0; i < cues.length; i++) {
                        if (currentTime >= cues[i].start && currentTime < cues[i].end) { activeIdx = i; break; }
                    }
                    renderLyrics(viewer, cues, activeIdx);
                } else {
                    if (rawBox.style.display === 'none') {
                        console.log('[Lyrics] Stop detected on ' + audioId);
                        viewer.style.display = 'none';
                        rawBox.style.display = 'block';
                    }
                }
            });
        }

        setInterval(updateLyrics, 100);
    }
    """

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=0.9,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=4,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                5.0,
                value=0.5,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 0.5.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice (CTC-ONNX)") as demo:
        gr.Markdown(
            """
# OmniVoice — CTC-ONNX Alignment

State-of-the-art text-to-speech with **high-precision forced alignment** using CTC models.
Supports **600+ languages**, voice cloning, and voice design.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        with gr.Accordion("Reference Text", open=False):
                            vc_ref_text = gr.Textbox(
                                label="Reference Text",
                                lines=2,
                                placeholder="Transcript of reference audio. Leave empty to auto-transcribe.",
                            )
                        with gr.Accordion("Language & Subtitle Settings", open=False):
                            vc_lang = gr.Dropdown(label="Language", choices=_ALL_LANGUAGES, value="Auto")
                            vc_srt_lang = gr.Dropdown(
                                label="Subtitle Language", 
                                choices=["Auto", "Chinese", "English", "Japanese", "Korean"], 
                                value="Auto"
                            )
                            vc_gen_srt = gr.Checkbox(
                                label="Generate Subtitles (SRT)",
                                value=True,
                                info="If enabled, uses CTC-ONNX for precise timing.",
                            )
                        (
                            vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio",
                            type="filepath",
                            elem_id="vc-audio",
                        )
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML("""
                                <label class="custom-label">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                                    Subtitle
                                </label>
                            """)
                            gr.HTML('<div id="vc-lyrics" class="lyrics-viewer"></div>')
                            vc_srt = gr.Textbox(
                                show_label=False,
                                lines=12,
                                elem_id="vc-srt-text",
                                show_copy_button=True,
                                interactive=False,
                                value="",
                            )
                        vc_dl = gr.DownloadButton("📥 Download ZIP (Audio + SRT)", visible=False)
                        vc_status = gr.Textbox(label="Status", lines=1)

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        gr.State(""),  # instruct
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                        vc_gen_srt,
                        vc_srt_lang,
                    ],
                    outputs=[vc_audio, vc_srt, vc_dl, vc_status],
                ).then(
                    lambda dl: gr.update(visible=True, value=dl),
                    inputs=[vc_dl],
                    outputs=[vc_dl],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        with gr.Accordion("Language & Subtitle Settings", open=False):
                            vd_lang = gr.Dropdown(label="Language", choices=_ALL_LANGUAGES, value="Auto")
                            vd_srt_lang = gr.Dropdown(
                                label="Subtitle Language", 
                                choices=["Auto", "Chinese", "English", "Japanese", "Korean"], 
                                value="Auto"
                            )
                            vd_gen_srt = gr.Checkbox(label="Generate Subtitles (SRT)", value=True)

                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=["Auto"] + _choices,
                                    value="Auto",
                                )
                            )

                        (
                            vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio",
                            type="filepath",
                            elem_id="vd-audio",
                        )
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML("""
                                <label class="custom-label">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                                    Subtitle
                                </label>
                            """)
                            gr.HTML('<div id="vd-lyrics" class="lyrics-viewer"></div>')
                            vd_srt = gr.Textbox(
                                show_label=False,
                                lines=12,
                                elem_id="vd-srt-text",
                                show_copy_button=True,
                                interactive=False,
                            )
                        vd_dl = gr.DownloadButton("📥 Download ZIP (Audio + SRT)", visible=False)
                        vd_status = gr.Textbox(label="Status", lines=1)

                def _design_fn(text, lang, srt_lang, ns, gs, dn, sp, du, pp, po, gen_srt, *groups):
                    # Build instruct string from dropdowns
                    selected = [g for g in groups if g and g != "Auto"]
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            parts.append(zh.strip() if "Dialect" in v else en.strip())
                        else:
                            parts.append(v)
                    instruct = ", ".join(parts) if parts else ""
                    
                    return _gen_core(
                        text, lang, None, instruct,
                        ns, gs, dn, sp, du, pp, po,
                        mode="design", generate_srt=gen_srt,
                        srt_language=srt_lang
                    )
 
                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text, vd_lang, vd_srt_lang,
                        vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po,
                        vd_gen_srt,
                    ] + vd_groups,
                    outputs=[vd_audio, vd_srt, vd_dl, vd_status],
                ).then(
                    lambda dl: gr.update(visible=True, value=dl),
                    inputs=[vd_dl],
                    outputs=[vd_dl],
                )
        demo.load(None, None, None, js=_LYRICS_JS)

    return demo


# ---------------------------------------------------------------------------
    # Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OmniVoice CTC-ONNX Demo")
    parser.add_argument(
        "--model", type=str, default=".",
        help="Path to local model folder (default: current directory)"
    )
    parser.add_argument(
        "--align_model", type=str, 
        default="MahmoudAshraf/mms-300m-1130-forced-aligner",
        help="Path or ID of the alignment model"
    )
    parser.add_argument(
        "--asr_model", type=str,
        default="openai/whisper-large-v3-turbo",
        help="Path or ID of the Whisper ASR model"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Launch with a public Gradio share link")
    args = parser.parse_args()

    # Force Offline Mode globally within this process
    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        # Optimize CPU threading
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        print(f"CPU Mode: Set torch threads to {torch.get_num_threads()}")

    print(f"Loading OmniVoice from '{args.model}' on {device}...")
    
    # Ensure asr_model is treated as an absolute path
    asr_path = os.path.abspath(args.asr_model)
    if os.path.isdir(asr_path):
        print(f"Verified: Using local ASR model from {asr_path}")
    else:
        print(f"Warning: Local ASR path {asr_path} not found. Fallback to online.")

    # Use float32 for CPU (CPU is very slow with float16), float16 for GPU
    model_dtype = torch.float32 if device == "cpu" else torch.float16

    # Use local_files_only=True
    model = OmniVoice.from_pretrained(
        args.model, 
        device_map=device, 
        dtype=model_dtype,
        local_files_only=True,
        asr_model_name=asr_path
    )
    
    # Pre-load the alignment model
    print(f"Pre-loading alignment model '{args.align_model}' into memory...")
    global _ALIGN_CACHE
    am, at = _load_ctc_model(device, model_name=args.align_model)
    _ALIGN_CACHE["model"] = am
    _ALIGN_CACHE["tokenizer"] = at
    
    demo = build_demo(model)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
