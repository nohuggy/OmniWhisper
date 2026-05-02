#!/usr/bin/env python3
import argparse
import sys
import os

# Add the project root to sys.path to allow importing 'omnivoice'
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

    if lang_lower in ("eng", "en"):
        max_val, soft_limit = 10, 14
        sentences = re.split(r"(?<=[.!?]) +", text)
        all_chunks = []
        for s in sentences:
            all_chunks.extend(split_into_balanced_chunks(s.split(), max_val, soft_limit, " "))
        return finalize_chunks(all_chunks)
    else:
        # CJK and everything else — character-level splitting
        max_val, soft_limit = 14, 18
        sentences = re.split(r"(?<=[。！？])", text)
        all_chunks = []
        for s in sentences:
            s = s.strip()
            if s:
                all_chunks.extend(split_into_balanced_chunks(list(s), max_val, soft_limit, ""))
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
        get_spans,
        postprocess_results,
        load_audio,
    )

    global _ALIGN_CACHE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use global cache to prevent 30-second reload delays
    if _ALIGN_CACHE["model"] is None:
        print("--- Initializing Aligner Model Cache ---")
        align_model, align_tokenizer = _load_ctc_model(device)
        _ALIGN_CACHE["model"] = align_model
        _ALIGN_CACHE["tokenizer"] = align_tokenizer
    
    align_model = _ALIGN_CACHE["model"]
    align_tokenizer = _ALIGN_CACHE["tokenizer"]

    sr, waveform_int16 = audio_tuple

    # --- Write a temp WAV (float32, mono) for ctc_forced_aligner's load_audio ---
    waveform_f32 = waveform_int16.astype(np.float32) / 32767.0
    if waveform_f32.ndim > 1:
        waveform_f32 = waveform_f32.mean(axis=-1)  # stereo → mono

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_audio:
        sf.write(f_audio.name, waveform_f32, sr, subtype="FLOAT")
        audio_path = f_audio.name

    # --- Prepare text segments ---
    segments = smart_split(text, language)
    full_text = "\n".join(segments)  # newline-separated for line-level alignment

    lang_code = _to_iso3(language)
    romanize = False
    
    # MMS models actually handle many CJK characters natively if passed correctly
    # But for Chinese, it is much more stable to use manual Pinyin conversion
    if lang_code == "cmn":
        try:
            from pypinyin import pinyin, Style
            def to_pinyin(t):
                # Convert to Pinyin with spaces between characters
                res = pinyin(t, style=Style.NORMAL)
                return " ".join([item[0] for item in res])
            
            # Convert each segment to pinyin separately to preserve line structure
            pinyin_segments = [to_pinyin(seg) for seg in segments]
            align_text = "\n".join(pinyin_segments)
            romanize = False # We've already romanized it manually
        except ImportError:
            logging.warning("pypinyin not installed, falling back to internal romanizer.")
            align_text = full_text
            romanize = True
    elif lang_code in ("jpn", "kor"):
        align_text = full_text
        romanize = True
    else:
        align_text = full_text
        romanize = False

    try:
        # Clean text for the aligner (remove punctuation that MMS doesn't like)
        clean_full_text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", align_text)
        clean_full_text = re.sub(r"\s+", " ", clean_full_text).strip()

        audio_waveform = load_audio(audio_path, align_model.dtype, align_model.device)
        emissions, stride = generate_emissions(align_model, audio_waveform)
        
        # Use the cleaned text for tokenization
        tokens_starred, text_starred = preprocess_text(
            clean_full_text, romanize=romanize, language=lang_code
        )
        
        raw_segments, scores, blank_token = get_alignments(
            emissions, tokens_starred, align_tokenizer
        )
        
        # Soft-fallback for get_spans to avoid AssertionError
        try:
            spans = get_spans(tokens_starred, raw_segments, blank_token)
        except AssertionError as ae:
            logging.warning(f"Alignment mismatch caught: {ae}. Retrying with loose matching.")
            # If it fails, it's often due to a <star> mismatch. 
            # We can't easily fix the library function, but we can try to skip it.
            return f"Alignment mismatch error: {ae}. Please try the 'Robust' method (Method 2) for this specific audio."

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # Build SRT — assign timestamps to each original subtitle line
        # Use total duration divided evenly across segments as a simple,
        # robust approach that avoids word-count mismatches between
        # original text and cleaned/pinyin alignment text.
        total_words = len(word_timestamps)
        srt_output = ""
        if total_words == 0 or not segments:
            return "No alignment result produced."

        # Distribute timestamps proportionally across segments
        total_audio_dur = word_timestamps[-1]["end"]
        num_segments = len([s for s in segments if s.strip()])
        if num_segments == 0:
            return "No alignment result produced."

        # Calculate how many aligned words belong to each segment
        # based on character/word ratio
        seg_char_counts = []
        for seg in segments:
            if not seg.strip():
                continue
            # Count alignable characters (strip punctuation)
            clean_seg = re.sub(r"[^\w\u4e00-\u9fff]", "", seg)
            seg_char_counts.append(max(1, len(clean_seg)))

        total_chars = sum(seg_char_counts)
        idx = 0
        block_num = 0
        for seg, char_count in zip([s for s in segments if s.strip()], seg_char_counts):
            block_num += 1
            # Proportional share of aligned words
            share = max(1, round(total_words * char_count / total_chars))
            line_ts = word_timestamps[idx: idx + share]
            idx += share
            if idx > total_words:
                idx = total_words
            if not line_ts:
                # Fallback: use the last known timestamp
                if word_timestamps:
                    line_ts = [word_timestamps[-1]]
                else:
                    continue
            t_start = line_ts[0]["start"]
            t_end = line_ts[-1]["end"]
            srt_output += (
                f"{block_num}\n"
                f"{format_timestamp(t_start)} --> {format_timestamp(t_end)}\n"
                f"{seg}\n\n"
            )
        return srt_output if srt_output else "No alignment result produced."

    except Exception as e:
        logging.exception("CTC alignment failed")
        return f"Alignment error: {e}"
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


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
                    _gen_core,
                    inputs=[
                        vc_text, vc_lang, vc_ref_audio,
                        gr.State(""),   # instruct
                        vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po,
                        gr.State("clone"),
                        vc_gen_srt,
                        vc_ref_text,
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

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, gen_srt, *groups):
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
                        mode="design", generate_srt=gen_srt
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text, vd_lang,
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

    # Use local_files_only=True
    model = OmniVoice.from_pretrained(
        args.model, 
        device_map=device, 
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
