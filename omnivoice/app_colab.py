#!/usr/bin/env python3
import sys
import os
import re
import tempfile
import zipfile
import torch
import numpy as np
import soundfile as sf
import gradio as gr
import warnings
import time
import subprocess
import shutil
# Suppress annoying warnings for a cleaner "pro" boot
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Unset offline mode variables immediately if they exist to prevent blocking downloads
for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
    if var in os.environ:
        del os.environ[var]

# ---------------------------------------------------------------------------
# 🚀 COLAB STABILITY & DEBUG ANNOTATION (IMMUTABLE GUARDRAILS)
# ---------------------------------------------------------------------------
# This edition solves the "Broken Connection" issue specific to Google Colab.
# 
# ## 1. THE DEBUG JOURNEY (KEY MILESTONES)
# - VRAM STABILITY: "Broken Connection" was proven to be a CUDA OOM crash, 
#   not a timeout. Solved via "Radical Unloading" and batch_size=1.
# - UI STABILITY (MINUTE JUMP): Highlights used to jump back 60s at minute 
#   boundaries (e.g. 4:00 -> 3:00). Fixed via JS Hysteresis filter.
# - UI STABILITY (MODE FLICKER): UI used to flash back to raw text for 0.5s 
#   at certain time marks. Fixed via Mode Persistence filter (>600ms).
# - UI STABILITY (PLAYER REFRESH): Audio player used to reload/restart 10-20s 
#   into playback. Fixed via "Immutable Audio Path" locking in all yields.
# - THE SMOKING GUN: VRAM tracking captured: "SRT Error: CUDA out of memory. 
#   Tried to allocate 602.00 MiB. GPU 0 has a total capacity of 14.56 GiB..."
# 
# ## 2. WHY LIGHTNING.AI WORKS BUT COLAB FAILED
# - SYSTEM OVERHEAD: Colab's Jupyter backend reserves ~800MB of VRAM for its UI 
#   bridge. Lightning.ai (with VSCode) has near-zero system VRAM overhead.
# - MARGIN OF ERROR: That 800MB is the critical margin required for Whisper 
#   Large V3 "Word" alignment to run while OmniVoice is in memory.
# 
# ## 3. THE ACTUAL FIX (DO NOT ALTER)
# - RADICAL UNLOADING: Models MUST be deleted from GPU to provide 15GB free space.
# - VRAM GUARDRAILS: batch_size=1 and chunk_length_s=30 are MANDATORY.
# - PRECISION: return_timestamps="word" is required for accuracy, but ONLY 
#   safe when combined with the VRAM guardrails above.
# 
# ## 4. GENERAL TECHNICAL MAINTENANCE NOTES (FROM LIGHTNING AI)
# - ORCHESTRATION: generate_core() must follow: Text -> TTS -> MP3 -> ASR Align.
# - UI STABILITY: To prevent the Gradio 5 "Triple Reload" player glitch, 
#   generate_core MUST keep the audio_path consistent (pointing to MP3) 
#   across all yields. Switching paths (MP3->WAV) triggers reloads.
# - DOM IDs: The lyric viewer relies on IDs: 'vc-audio', 'vd-audio', etc.
# - LYRIC SCRAPER: JS uses a "Motion-Aware" scraper that only trusts timestamps 
#   if they are changing and checks primaryAudio.paused to prevent hanging.
# - MINUTE JUMP FIX: A Hysteresis filter is applied to ignore sudden 60s jumps 
#   caused by UI text flickers at minute boundaries (e.g. 3:59 -> 4:00).
# - MODE FLICKER FIX: A persistence filter ensures the UI doesn't switch back 
#   to raw text mode unless the time-scraper fails for >600ms.
# ---------------------------------------------------------------------------

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [project_root, os.path.dirname(os.path.abspath(__file__)), os.path.join(project_root, "whisper-large-v3-turbo")]:
    if p not in sys.path:
        sys.path.append(p)

from omnivoice.omni_engine_colab import TTSEngine, get_slug
from whisper_engine_colab import format_timestamp, unify_punctuation, smart_balanced_split, align_robust
from transformers import pipeline

# ---------------------------------------------------------------------------
# Global Engines
# ---------------------------------------------------------------------------
TTS_ENGINE = None
WHISPER_PIPE = None

def get_tts_engine(model_path=None):
    global TTS_ENGINE
    if TTS_ENGINE is None:
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "resources")
            
        # Ensure model directory exists and has weights
        has_weights = os.path.exists(model_path) and any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(model_path))
        if not has_weights:
            print(f"📥 Downloading OmniVoice Weights (~1.5GB) to ephemeral storage...")
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id="k2-fsa/OmniVoice", local_dir=model_path, local_dir_use_symlinks=False)
            except Exception as e:
                print(f"⚠️ Download failed: {e}. Trying git fallback...")
                os.system(f"git clone https://huggingface.co/k2-fsa/OmniVoice {model_path}_tmp && mv {model_path}_tmp/* {model_path}/ && rm -rf {model_path}_tmp")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        TTS_ENGINE = TTSEngine(model_path, device=device, dtype=dtype)
    return TTS_ENGINE

def get_whisper_pipe(whisper_path=None):
    global WHISPER_PIPE
    if WHISPER_PIPE is None:
        if whisper_path is None:
            whisper_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "whisper-large-v3-turbo")
            
        # Ensure whisper directory exists and has weights
        has_whisper = os.path.exists(whisper_path) and any(f.endswith(('.bin', '.safetensors', '.pt')) for f in os.listdir(whisper_path) if os.path.isfile(os.path.join(whisper_path, f)))
        if not has_whisper:
            print(f"📥 Downloading Whisper Turbo (~1.6GB) to ephemeral storage...")
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id='openai/whisper-large-v3-turbo', local_dir=whisper_path, local_dir_use_symlinks=False)
            except Exception as e:
                print(f"⚠️ Download failed: {e}. Trying git fallback...")
                os.system(f"git clone https://huggingface.co/openai/whisper-large-v3-turbo {whisper_path}_tmp && mv {whisper_path}_tmp/* {whisper_path}/ && rm -rf {whisper_path}_tmp")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        from transformers import pipeline
        WHISPER_PIPE = pipeline("automatic-speech-recognition", model=whisper_path, device=device, torch_dtype=dtype)
    return WHISPER_PIPE

def unload_tts():
    global TTS_ENGINE
    if TTS_ENGINE:
        print("🗑️ Unloading TTS Engine...")
        del TTS_ENGINE
        TTS_ENGINE = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def unload_whisper():
    global WHISPER_PIPE
    if WHISPER_PIPE:
        print("🗑️ Unloading Whisper Engine...")
        del WHISPER_PIPE
        WHISPER_PIPE = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_engines(model_path=None, whisper_path=None):
    # This is now just a pre-warm helper for the first load
    get_tts_engine(model_path)
    get_whisper_pipe(whisper_path)
    return TTS_ENGINE, WHISPER_PIPE

# ---------------------------------------------------------------------------
# UI Helpers & Constants
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very low pitch", "Low pitch", "Moderate pitch", "High pitch", "Very high pitch"],
    "Style": ["Whisper"],
    "Accent": ["American accent", "British accent", "Australian accent", "Chinese accent", "Canadian accent", "Indian accent", "Korean accent", "Portuguese accent", "Russian accent", "Japanese accent"],
    "Dialect": ["东北话", "四川话", "河南话", "陕西话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话"],
}

from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
_LANG_DISPLAY = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

CSS = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.compact-audio audio {height: 60px !important;}

/* Restore the output-panel aesthetics */
.output-panel, 
.output-panel * {
    background-color: #1f2937 !important;
    border: none !important;
    box-shadow: none !important;
}
.output-panel { gap: 0 !important; overflow: hidden !important; border: 1px solid #374151 !important; border-radius: 12px; }

.custom-label {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13px; font-weight: 700;
    color: #fff !important;
    background-color: #4f46e5 !important;
    padding: 4px 10px;
    border-radius: 6px !important;
    margin-top: -1px !important;
    margin-left: -1px !important;
    width: fit-content;
}
.custom-label svg { width: 14px; height: 14px; background: transparent !important;}

/* Make all buttons bold and consistent */
button.primary, button.secondary {
    font-weight: 700 !important;
    font-family: inherit !important;
}
"""

_SUBTITLE_CSS = """
#vc-lyrics, #vd-lyrics {
    background: #111;
    color: #fff;
    border-radius: 12px;
    height: 260px;
    width: 100% !important;
    overflow-y: auto !important;
    padding: 10px 20px !important;
    box-sizing: border-box;
    display: none;
}
#vc-srt-text, #vd-srt-text { 
    height: 260px !important; 
    overflow: visible !important; 
}
#vc-srt-text textarea, #vd-srt-text textarea { 
    height: 260px !important;
    overflow-y: auto !important; 
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.9em;
    background: transparent !important;
    padding-bottom: 20px !important;
}
.lyric-line {
    text-align: center;
    padding: 6px 12px;
    margin: 4px 0;
    border-radius: 6px;
    transition: all 0.2s ease;
    color: #888;
    font-size: 1.1em;
    line-height: 1.4;
}
.lyric-line.active {
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1.25em !important;
    background-color: rgba(79, 70, 229, 0.2) !important;
}
.lyric-line.past {
    color: rgba(255, 255, 255, 0.7) !important;
}
.hidden-lyrics { display: none !important }
"""

CSS += _SUBTITLE_CSS

_LYRICS_JS = r"""
() => {
    console.log('[Lyrics] Initializing Robust Polling Engine...');
    
    function parseTimestamp(s) {
        if (!s) return 0;
        var p = s.replace(',','.').split(':');
        if (p.length === 3) return parseInt(p[0])*3600 + parseInt(p[1])*60 + parseFloat(p[2]);
        return parseFloat(p[p.length-1]);
    }

    function parseSRT(data) {
        if (!data) return [];
        var res = [];
        var blocks = data.trim().split(/\n\s*\n/);
        blocks.forEach(function(block) {
            try {
                var lines = block.split('\n');
                if (lines.length >= 3) {
                    var timeMatch = lines[1].match(/(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)/);
                    if (timeMatch) {
                        res.push({
                            start: parseTimestamp(timeMatch[1]),
                            end: parseTimestamp(timeMatch[2]),
                            text: lines.slice(2).join(' ')
                        });
                    }
                }
            } catch(e) { console.error("SRT Parse Error", e); }
        });
        return res;
    }

    function renderLyrics(viewer, cues, activeIdx) {
        if (!cues.length) {
            viewer.innerHTML = '<div style="text-align:center;color:#aaa;padding:40px;">No subtitles loaded</div>';
            return;
        }
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
        var lines = viewer.children;
        for (var i = 0; i < lines.length; i++) {
            var el = lines[i];
            var idx = parseInt(el.getAttribute('data-idx'));
            if (idx === activeIdx) {
                el.classList.add('active');
                el.classList.remove('past');
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

    function parseTimeStr(s) {
        if (!s) return -1;
        var p = s.trim().split(':');
        if (p.length < 2) return -1;
        try {
            if (p.length === 2) return parseInt(p[0])*60 + parseFloat(p[1]);
            if (p.length === 3) return parseInt(p[0])*3600 + parseInt(p[1])*60 + parseFloat(p[2]);
        } catch(e) {}
        return -1;
    }

    function getSRT(srtBoxId) {
        var box = document.getElementById(srtBoxId);
        if (!box) return '';
        var ta = box.querySelector('textarea');
        if (ta) return ta.value;
        return box.innerText || '';
    }

    function updateLyrics() {
        try {
            PAIRS.forEach(function(pair) {
                var audioId = pair[0], lyricsId = pair[1], srtBoxId = pair[2];
                var audioContainer = document.getElementById(audioId);
                var viewer = document.getElementById(lyricsId);
                var rawBox = document.getElementById(srtBoxId);
                if (!audioContainer || !viewer || !rawBox) return;

                var currentTime = -1;

                // 1. Check for audio element in container
                var internalAudios = audioContainer.querySelectorAll('audio');
                if (internalAudios.length === 0 && audioContainer.shadowRoot) {
                    internalAudios = audioContainer.shadowRoot.querySelectorAll('audio');
                }
                
                var primaryAudio = internalAudios[0];
                if (primaryAudio && primaryAudio.paused) {
                    currentTime = -1; 
                } else {
                    for (var i = 0; i < internalAudios.length; i++) {
                        if (!internalAudios[i].paused && internalAudios[i].currentTime > 0) {
                            currentTime = internalAudios[i].currentTime;
                            break;
                        }
                    }
                    if (currentTime < 0) {
                        var allAudios = document.querySelectorAll('audio');
                        for (var j = 0; j < allAudios.length; j++) {
                            if (!allAudios[j].paused && allAudios[j].currentTime > 0) {
                                currentTime = allAudios[j].currentTime;
                                break;
                            }
                        }
                    }
                }

                // 2. UI Scraper Fallback
                if (currentTime < 0) {
                    var txt = audioContainer.innerText || "";
                    var matches = txt.match(/(\d+:\d+)/g);
                    if (matches) {
                        var p = parseTimeStr(matches[0]); 
                        if (p > 0.05) {
                            if (p !== viewer._lastP) { 
                                viewer._lastP = p; 
                                viewer._lastT = Date.now(); 
                            }
                            if (Date.now() - (viewer._lastT || 0) < 3000) { 
                                currentTime = p; 
                            }
                        }
                    }
                }

                // 3. Consistency Filter (Anti-Jump)
                // Prevents the "1-minute jump" glitch by ignoring sudden large jumps 
                // unless they persist, indicating a real user seek.
                var now = Date.now();
                var dt = (now - (viewer._lastTick || now)) / 1000;
                viewer._lastTick = now;

                if (currentTime >= 0 && viewer._lastSafeTime >= 0) {
                    var delta = Math.abs(currentTime - (viewer._lastSafeTime + dt));
                    // If jump is > 2s and we haven't confirmed it yet
                    if (delta > 2.0) {
                        if (viewer._jumpTime !== currentTime) {
                            viewer._jumpTime = currentTime;
                            viewer._jumpCount = 1;
                            // Temporarily ignore the jump and use projected time
                            currentTime = viewer._lastSafeTime + dt;
                        } else {
                            viewer._jumpCount++;
                            if (viewer._jumpCount < 4) { // Wait ~800ms to confirm seek
                                currentTime = viewer._lastSafeTime + dt;
                            } else {
                                // Confirmed seek
                                viewer._lastSafeTime = currentTime;
                            }
                        }
                    } else {
                        viewer._lastSafeTime = currentTime;
                        viewer._jumpCount = 0;
                    }
                } else if (currentTime >= 0) {
                    viewer._lastSafeTime = currentTime;
                }

                if (currentTime >= 0) {
                    viewer._modeFailCount = 0;
                    var srtVal = getSRT(srtBoxId);
                    if (viewer.style.display === 'none' || viewer._lastSRT !== srtVal) {
                        rawBox.style.display = 'none';
                        viewer.style.display = 'block';
                        viewer._cues = parseSRT(srtVal);
                        viewer._lastSRT = srtVal;
                        viewer._cueCount = -1;
                    }
                    var cues = viewer._cues || [];
                    var activeIdx = -1;
                    for (var k = 0; k < cues.length; k++) {
                        if (currentTime >= cues[k].start && currentTime < cues[k].end) { activeIdx = k; break; }
                    }
                    renderLyrics(viewer, cues, activeIdx);
                } else {
                    viewer._lastSafeTime = -1;
                    viewer._modeFailCount = (viewer._modeFailCount || 0) + 1;
                    if (viewer._modeFailCount >= 3 && viewer.style.display !== 'none') {
                        viewer.style.display = 'none';
                        rawBox.style.display = 'block';
                    }
                }
            });
        } catch(e) { console.warn("Polling Engine Recovered", e); }
    }

    setInterval(updateLyrics, 200);
}
"""

def optimize_audio_for_web(wav_path):
    """Convert WAV to a lightweight MP3 for fast web loading."""
    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        # -q:a 5 is approx 128kbps variable bitrate, balanced for speech
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "5", mp3_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(mp3_path):
            return mp3_path
    except Exception as e:
        print(f"Error optimizing audio: {e}")
    return wav_path

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def text_to_srt_whisper(text, audio_tuple, pipe, language="zh", target_words=12, max_words=17, progress=None):
    """Generate SRT using Whisper word-level timestamps via pipeline."""
    try:
        if progress: progress(0.1, desc="🔍 Aligning subtitles (Whisper)...")
        sr, waveform = audio_tuple
        # Normalize to float32 for pipeline
        waveform_f32 = waveform.astype(np.float32) / 32767.0
        
        # Whisper pipeline expects a dict or numpy array
        # 🚀 RADICAL FIX: For long audio, we MUST use chunk_length_s to prevent OOM and timeouts.
        # This forces Whisper to process in 30s windows internally.
        print(f"[SRT] Starting Chunked Whisper inference (Audio: {len(waveform_f32)/sr:.1f}s)...")
        if torch.cuda.is_available():
            print(f"[SRT] VRAM Before ASR: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            
        result = pipe(
            {"sampling_rate": sr, "raw": waveform_f32}, 
            chunk_length_s=30, 
            batch_size=1, 
            return_timestamps="word"
        )
        
        chunks = result.get("chunks", [])
        print(f"[SRT] Whisper inference complete. Got {len(chunks)} word chunks.")
        if torch.cuda.is_available():
            print(f"[SRT] VRAM After ASR: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        print(f"[SRT] Running smart_balanced_split on {len(text)} chars with max_words={max_words}...")
        segments = smart_balanced_split(text, target_words=target_words, max_words=max_words)
        print(f"[SRT] Split into {len(segments)} segments.")
        
        # 1. Global Clock Reconstruction (Fixing the Whisper 30s Wall)
        abs_chunks = []
        offset = 0.0
        last_chunk_e = 0.0
        for c in chunks:
            s, e = c["timestamp"]
            if s is None: s = last_chunk_e
            if e is None: e = s + 0.5
            
            # Reset Detection
            if (s + offset) < (last_chunk_e - 1.0) and s < 10.0:
                while (s + offset) < (last_chunk_e - 1.0):
                    offset += 30.0
            
            abs_s = s + offset
            abs_e = e + offset
            
            # Monotonicity Enforcement
            if abs_s < last_chunk_e: abs_s = last_chunk_e
            if abs_e < abs_s: abs_e = abs_s + 0.1
            
            c["timestamp"] = (abs_s, abs_e)
            abs_chunks.append(c)
            last_chunk_e = abs_e
            
        import difflib
        user_clean = [re.sub(r'[^\w\u4e00-\u9fff]', '', s).lower() for s in segments]
        whisper_full_text = "".join([c["text"] for c in abs_chunks])
        whisper_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', whisper_full_text).lower()
        
        char_times = []
        for c in abs_chunks:
            txt = c["text"]
            s, e = c["timestamp"]
            c_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', txt).lower()
            if not c_clean: continue
            duration = e - s
            for i in range(len(c_clean)):
                char_times.append((s + (i / len(c_clean)) * duration, s + ((i + 1) / len(c_clean)) * duration))
                
        user_full_clean = "".join(user_clean)
        matcher = difflib.SequenceMatcher(None, user_full_clean, whisper_clean)
        
        mapping = [None] * len(user_full_clean)
        last_w_end = 0
        for u_s, w_s, length in matcher.get_matching_blocks():
            if w_s < last_w_end: continue
            for i in range(length):
                if w_s + i < len(char_times):
                    mapping[u_s + i] = char_times[w_s + i]
            last_w_end = w_s + length
                    
        matched_indices = [i for i, x in enumerate(mapping) if x is not None]
        if not matched_indices:
            total_dur = char_times[-1][1] if char_times else 10.0
            for i in range(len(mapping)):
                mapping[i] = ((i / len(mapping)) * total_dur, ((i + 1) / len(mapping)) * total_dur)
        else:
            first_idx = matched_indices[0]
            first_s = mapping[first_idx][0]
            for i in range(first_idx):
                mapping[i] = ((i / first_idx) * first_s if first_idx > 0 else 0, ((i + 1) / first_idx) * first_s if first_idx > 0 else 0)
            for j in range(len(matched_indices) - 1):
                idx1, idx2 = matched_indices[j], matched_indices[j+1]
                t1, t2 = mapping[idx1][1], mapping[idx2][0]
                gap_len = idx2 - idx1 - 1
                if gap_len > 0:
                    for k in range(1, gap_len + 1):
                        s_interp = t1 + ((k-1) / gap_len) * (t2 - t1)
                        e_interp = t1 + (k / gap_len) * (t2 - t1)
                        mapping[idx1 + k] = (s_interp, e_interp)
            last_idx = matched_indices[-1]
            last_e = mapping[last_idx][1]
            total_end = char_times[-1][1] if char_times else last_e + 1.0
            rem_len = len(mapping) - 1 - last_idx
            if rem_len > 0:
                for k in range(1, rem_len + 1):
                    s_interp = last_e + ((k-1) / rem_len) * (total_end - last_e)
                    e_interp = last_e + (k / rem_len) * (total_end - last_e)
                    mapping[last_idx + k] = (s_interp, e_interp)
                    
        if progress: progress(0.9, desc="📝 Finalizing SRT formatting...")
        srt_output = ""
        curr = 0
        for i, (seg_text, s_clean) in enumerate(zip(segments, user_clean), 1):
            if not s_clean: continue
            start_time = mapping[curr][0]
            end_time = mapping[curr + len(s_clean) - 1][1]
            srt_output += f"{i}\n{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n{seg_text}\n\n"
            curr += len(s_clean)
        
        return srt_output.strip()
    except Exception as e:
        return f"SRT Error: {e}"

def generate_core(text, language, ref_audio, ref_text, instruct, num_step, guidance, denoise, speed, duration, pp, po, mode, gen_srt=True, convert_punc=True, srt_max_words=17, progress=gr.Progress()):
    """
    Central orchestration loop for OmniVoice synthesis.
    Flow: 
    1. Text Processing & Punctuation Unification.
    2. TTS Synthesis (yields chunk progress).
    3. ASR-based word alignment (Whisper).
    4. SRT & ZIP generation.
    5. Final synchronized yield of all result components.
    """
    if not text or not text.strip():
        yield None, "", None, "Text is required."
        return
    
    # 1. Punctuation Unification
    if convert_punc:
        text = unify_punctuation(text)
        if ref_text:
            ref_text = unify_punctuation(ref_text)
            
    start_time = time.time()
    lang_code = language if language != "Auto" else None
    
    try:
        # 🚀 RADICAL VRAM: Unload Whisper before TTS
        unload_whisper()
        
        engine = get_tts_engine()
        
        start_time = time.time()

        # 0. Immediate yield to remove "Loader GIF" and show activity
        if progress: progress(0.01, desc="🚀 Initializing Engines...")
        yield gr.update(), gr.update(), gr.update(), "🚀 Initializing synthesis..."
        
        # 2. TTS Generation (Streaming Progress)
        gen_iter = engine.generate(
            text=text.strip(),
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            language=lang_code,
            num_step=num_step,
            guidance_scale=guidance,
            denoise=denoise,
            speed=speed,
            duration=duration,
            preprocess_prompt=pp,
            postprocess_output=po
        )
        
        # 2. TTS Generation
        full_waveform = []
        sr = 16000
        
        last_yield_time = time.time()
        for curr, total, chunk_wave in gen_iter:
            if chunk_wave is None:
                # Update progress bar
                if progress:
                    progress(curr/total * 0.8, desc=f"⏳ TTS ({curr}/{total})")
                
                # 🚀 HEARTBEAT: Constant yield to keep connection alive
                # Yield every 1.5s to ensure Gradio/Colab proxy never times out
                if time.time() - last_yield_time > 1.5:
                    yield gr.update(), gr.update(), gr.update(), f"⏳ TTS Generating: Chunk {curr}/{total}"
                    last_yield_time = time.time()
            else:
                full_waveform = chunk_wave
                sr = engine.sampling_rate
        
        # 3. Save Audio Immediately to show player early
        slug = get_slug(text)
        unique_slug = f"{slug}_{int(time.time())}"
        os.makedirs("outputs", exist_ok=True)
        wav_path = f"outputs/{unique_slug}.wav"
        sf.write(wav_path, full_waveform, sr)
        
        # Optimize for web (mp3 is much smaller and loads faster in Gradio)
        audio_path = optimize_audio_for_web(wav_path)
        
        # 🚀 CRITICAL: Yield the audio_path NOW so the player displays it while ASR runs
        yield audio_path, gr.update(), gr.update(), "⏳ TTS Done. Starting ASR alignment..."
        
        # 4. SRT Generation
        srt_content = ""
        audio_tuple = (sr, full_waveform)
        duration_s = len(full_waveform) / sr
        
        if gen_srt:
            # 🚀 RADICAL VRAM: Unload TTS before ASR
            unload_tts()
            pipe = get_whisper_pipe()
            
            # 3. SRT Generation (🚀 RADICAL HEARTBEAT)
            # Run Whisper in a thread so we can keep yielding heartbeats to the browser
            import threading
            asr_res = {"content": None, "error": None}
            def run_asr():
                try:
                    asr_res["content"] = text_to_srt_whisper(text, audio_tuple, pipe, target_words=12, max_words=srt_max_words, progress=progress)
                except Exception as e:
                    asr_res["error"] = str(e)
            
            thread = threading.Thread(target=run_asr)
            thread.start()
            
            dot_count = 0
            while thread.is_alive():
                dot_count = (dot_count + 1) % 4
                dots = "." * dot_count
                if progress:
                    # Slow crawl from 0.8 to 0.95 during ASR
                    progress(0.8 + (dot_count * 0.03), desc=f"🔍 Aligning{dots}")
                
                elapsed = int(time.time() - start_time)
                # HEARTBEAT: Constant yield to keep Colab/Gradio connection active
                # Faster heartbeat (0.8s) for long-form stability
                # 🚀 CRITICAL: We yield audio_path explicitly to keep the player from refreshing
                yield audio_path, gr.update(), gr.update(), f"⏳ ASR Alignment in progress ({elapsed}s)... {dots}"
                time.sleep(0.8)
            
            thread.join()
            if asr_res["error"]:
                srt_content = f"SRT Error: {asr_res['error']}"
            else:
                srt_content = asr_res["content"]
            
            yield audio_path, srt_content, gr.update(), "⏳ ASR Complete. Packaging ZIP..."
        
        zip_path = None
        if gen_srt and srt_content and not srt_content.startswith("SRT Error"):
            elapsed = int(time.time() - start_time)
            # 🚀 CRITICAL: We yield audio_path explicitly to keep the player from refreshing
            yield audio_path, gr.update(), gr.update(), f"📦 Packaging results ({elapsed}s)..."
            
            srt_path = f"outputs/{unique_slug}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            
            zip_path = f"outputs/{unique_slug}.zip"
            with zipfile.ZipFile(zip_path, 'w') as z:
                z.write(wav_path, arcname=f"{slug}.wav")
                z.write(srt_path, arcname=f"{slug}.srt")
        
        elapsed = time.time() - start_time
        tokens = len(text.strip())
        status_msg = f"✅ Done in {elapsed:.1f}s | {duration_s:.1f}s Audio | {tokens} Chars"

        # FINAL YIELD: Maintain the SAME audio_path (MP3) to avoid the final UI reload glitch.
        # Gradio 5 reloads the component if the source path changes. 
        # By keeping it as the MP3, the user's preview remains uninterrupted.
        yield audio_path, srt_content, zip_path if zip_path else gr.update(), status_msg
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, "", None, f"Error: {e}"

# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------

def build_app(model_path=None, whisper_path=None):
    load_engines(model_path, whisper_path)
    
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="OmniWhisper") as demo:
        gr.Markdown("# OmniWhisper")
        
        with gr.Tabs():
            # VOICE CLONE TAB
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(label="Text to Synthesize", lines=5, placeholder="Enter text...")
                        vc_ref = gr.Audio(label="Reference Audio", type="filepath", elem_classes="compact-audio")
                        
                        vc_ref_text = gr.Textbox(
                            label="Reference Text", 
                            lines=2, 
                            placeholder="Transcript of reference audio. Click 'Transcribe' to process."
                        )
                        
                        with gr.Row():
                            vc_transcribe_btn = gr.Button("Trans Ref", variant="secondary")
                            vc_ref_zip_btn = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="secondary")
                            vc_ref_txt_btn = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="secondary")

                        with gr.Accordion("Advanced Settings", open=False):
                            vc_instruct = gr.Textbox(
                                label="Voice Instruction", 
                                placeholder="e.g. 'male, high pitch' or '男，河南话'",
                                lines=1
                            )

                            vc_lang = gr.Dropdown(label="Language", choices=_LANG_DISPLAY, value="Auto")
                            vc_speed = gr.Slider(0.5, 2.0, value=0.9, step=0.05, label="Speed")
                            vc_dur = gr.Number(label="Fixed Duration (sec)", value=0)
                            vc_steps = gr.Slider(4, 64, value=32, step=4, label="Inference Steps")
                            vc_gs = gr.Slider(0, 5, value=3.0, step=0.1, label="Guidance Scale")
                            vc_srt_max_words = gr.Slider(15, 22, value=17, step=1, label="Max Words Per Line")
                            vc_dn = gr.Checkbox(label="Denoise", value=True)
                            vc_pp = gr.Checkbox(label="Clean Ref Audio (Silence Removal)", value=True)
                            vc_po = gr.Checkbox(label="Trim Output Silence", value=True)
                            vc_gen_srt = gr.Checkbox(label="Generate Subtitles (SRT)", value=True)
                            vc_punc = gr.Checkbox(label="Convert Punctuation", value=True)
                            
                        vc_btn = gr.Button("Generate Voice", variant="primary")
                        
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(label="Result", type="filepath", elem_id="vc-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<label class="custom-label">Subtitle</label>')
                            gr.HTML('<div id="vc-lyrics" class="lyrics-viewer"></div>')
                            vc_srt = gr.Textbox(show_label=False, lines=10, elem_id="vc-srt-text", interactive=False)
                        
                        vc_dl = gr.DownloadButton("📥 Download ZIP (Audio + SRT)", visible=False)
                        vc_status = gr.Textbox(label="Status", interactive=False, lines=3)

            # VOICE DESIGN TAB
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(label="Text to Synthesize", lines=5, placeholder="Enter text...")
                        
                        vd_groups = []
                        with gr.Row():
                            for cat, choices in list(_CATEGORIES.items())[:2]:
                                vd_groups.append(gr.Dropdown(label=cat, choices=["Auto"] + choices, value="Auto"))
                        with gr.Row():
                            for cat, choices in list(_CATEGORIES.items())[2:4]:
                                vd_groups.append(gr.Dropdown(label=cat, choices=["Auto"] + choices, value="Auto"))
                        with gr.Row():
                            for cat, choices in list(_CATEGORIES.items())[4:]:
                                vd_groups.append(gr.Dropdown(label=cat, choices=["Auto"] + choices, value="Auto"))
                                
                        with gr.Accordion("Advanced Settings", open=False):
                            vd_lang = gr.Dropdown(label="Language", choices=_LANG_DISPLAY, value="Auto")
                            vd_speed = gr.Slider(0.5, 2.0, value=0.9, step=0.05, label="Speed")
                            vd_dur = gr.Number(label="Fixed Duration (sec)", value=0)
                            vd_steps = gr.Slider(4, 64, value=32, step=4, label="Inference Steps")
                            vd_gs = gr.Slider(0, 5, value=3.0, step=0.1, label="Guidance Scale")
                            vd_srt_max_words = gr.Slider(15, 22, value=17, step=1, label="Max Words Per Line")
                            vd_dn = gr.Checkbox(label="Denoise", value=True)
                            vd_po = gr.Checkbox(label="Trim Output Silence", value=True)
                            vd_gen_srt = gr.Checkbox(label="Generate Subtitles (SRT)", value=True)
                            vd_punc = gr.Checkbox(label="Convert Punctuation", value=True)

                        vd_btn = gr.Button("Create Voice", variant="primary")

                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(label="Result", type="filepath", elem_id="vd-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<label class="custom-label">Subtitle</label>')
                            gr.HTML('<div id="vd-lyrics" class="lyrics-viewer"></div>')
                            vd_srt = gr.Textbox(show_label=False, lines=10, elem_id="vd-srt-text", interactive=False)
                        
                        vd_dl = gr.DownloadButton("📥 Download ZIP (Audio + SRT)", visible=False)
                        vd_status = gr.Textbox(label="Status", interactive=False, lines=3)

        # Event Handlers
        def transcribe_ref(audio, progress=gr.Progress()):
            if not audio: 
                yield ""
                return
            try:
                # 🚀 Support On-Demand Loading
                pipe = get_whisper_pipe()
                
                import threading
                res = {"text": None, "error": None}
                def run():
                    try: 
                        # Use the pipeline directly on the audio path
                        # Whisper turbo is very fast for short ref audio
                        out = pipe(audio, chunk_length_s=30, batch_size=1)
                        res["text"] = out.get("text", "").strip()
                    except Exception as e: 
                        res["error"] = str(e)
                
                thread = threading.Thread(target=run)
                thread.start()
                
                dot_count = 0
                while thread.is_alive():
                    dot_count = (dot_count + 1) % 4
                    dots = "." * dot_count
                    if progress: progress(0.5, desc=f"🎙️ Transcribing{dots}")
                    yield f"🎙️ Transcribing{dots}"
                    time.sleep(1.0)
                
                thread.join()
                if res["error"]: yield f"Error: {res['error']}"
                else: yield res["text"]
            except Exception as e:
                yield f"Error: {e}"

        def vc_handler(
            text, lang, ref, ref_text, instruct, steps, gs, srt_max_words, dn, punc, speed, dur, pp, po, gen_srt,
            progress: gr.Progress = gr.Progress()
        ):
            if progress: progress(0, desc="🚀 Initializing...")
            for res in generate_core(
                text=text, language=lang, ref_audio=ref, ref_text=ref_text, 
                instruct=instruct, num_step=steps, guidance=gs, denoise=dn, 
                convert_punc=punc, speed=speed, duration=dur, 
                pp=pp, po=po, mode="clone", gen_srt=gen_srt,
                srt_max_words=srt_max_words,
                progress=progress
            ):
                yield res
            
        def process_ref_zip(zip_file):
            if not zip_file: return None, ""
            import zipfile, tempfile
            audio_path = None
            text_content = ""
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_file.name, 'r') as z:
                z.extractall(tmp)
                for f in z.namelist():
                    if f.endswith(('.wav', '.mp3', '.flac')) and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.'):
                        audio_path = os.path.join(tmp, f)
                    if f.endswith('.txt') and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.'):
                        txt_path = os.path.join(tmp, f)
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as tf:
                                text_content = tf.read()
                        except UnicodeDecodeError:
                            with open(txt_path, 'r', encoding='gbk') as tf:
                                text_content = tf.read()
            return audio_path, text_content
            
        def process_ref_txt(txt_file):
            if not txt_file: return ""
            try:
                with open(txt_file.name, 'r', encoding='utf-8') as tf:
                    return tf.read()
            except UnicodeDecodeError:
                with open(txt_file.name, 'r', encoding='gbk') as tf:
                    return tf.read()
            
        # Smart Reference Handlers
        vc_ref_zip_btn.upload(
            process_ref_zip,
            inputs=[vc_ref_zip_btn],
            outputs=[vc_ref, vc_ref_text]
        )
        
        vc_ref_txt_btn.upload(
            process_ref_txt,
            inputs=[vc_ref_txt_btn],
            outputs=[vc_ref_text]
        )
            
        # Transcription events (Manual only)
        vc_transcribe_btn.click(transcribe_ref, inputs=[vc_ref], outputs=[vc_ref_text])

        vc_btn.click(
            lambda: (gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), "⏳ Initializing..."),
            outputs=[vc_btn, vc_transcribe_btn, vc_dl, vc_status]
        ).then(
            vc_handler,
            inputs=[vc_text, vc_lang, vc_ref, vc_ref_text, vc_instruct, vc_steps, vc_gs, vc_srt_max_words, vc_dn, vc_punc, vc_speed, vc_dur, vc_pp, vc_po, vc_gen_srt],
            outputs=[vc_audio, vc_srt, vc_dl, vc_status]
        ).then(
            lambda dl: (gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=bool(dl))), 
            inputs=[vc_dl], 
            outputs=[vc_btn, vc_transcribe_btn, vc_dl]
        )


        def vd_handler(
            text, lang, speed, dur, steps, gs, srt_max_words, dn, punc, po, gen_srt, g1, g2, g3, g4, g5, g6,
            progress: gr.Progress = gr.Progress()
        ):
            if progress: progress(0, desc="🚀 Designing Voice...")
            instruct = ", ".join([g for g in [g1, g2, g3, g4, g5, g6] if g != "Auto"])
            for res in generate_core(text, lang, None, None, instruct, steps, gs, dn, speed, dur, False, po, "design", gen_srt, punc, srt_max_words=srt_max_words, progress=progress):
                yield res

        vd_btn.click(
            vd_handler,
            inputs=[vd_text, vd_lang, vd_speed, vd_dur, vd_steps, vd_gs, vd_srt_max_words, vd_dn, vd_punc, vd_po, vd_gen_srt] + vd_groups,
            outputs=[vd_audio, vd_srt, vd_dl, vd_status]
        ).then(lambda dl: gr.update(visible=bool(dl)), inputs=[vd_dl], outputs=[vd_dl])

        demo.load(None, None, None, js=_LYRICS_JS)
        
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--whisper", type=str, default=None)
    parser.add_argument("--share", action="store_true", help="Launch with a public Gradio share link")
    args = parser.parse_args()
    
    app = build_app(args.model, args.whisper)
    
    # Enable queueing for long-running tasks and progress bar visibility
    # This also handles the block_thread logic automatically
    app.queue().launch(server_name="0.0.0.0", server_port=7860, share=args.share)
