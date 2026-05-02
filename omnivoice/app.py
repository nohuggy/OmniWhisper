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

# Suppress annoying warnings for a cleaner "pro" boot
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [project_root, os.path.dirname(os.path.abspath(__file__)), os.path.join(project_root, "whisper-large-v3-turbo")]:
    if p not in sys.path:
        sys.path.append(p)

from omni_engine import TTSEngine, get_slug
from whisper_engine import format_timestamp, smart_balanced_split, align_robust
from transformers import pipeline

# ---------------------------------------------------------------------------
# Global Engines
# ---------------------------------------------------------------------------
TTS_ENGINE = None
WHISPER_PIPE = None

def load_engines(model_path=None, whisper_path=None):
    global TTS_ENGINE, WHISPER_PIPE
    
    # Default paths relative to this script
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "resources")
    if whisper_path is None:
        whisper_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "whisper-large-v3-turbo")

    # 1. Handle OmniVoice TTS Model
    if "HF_HUB_OFFLINE" in os.environ:
        del os.environ["HF_HUB_OFFLINE"]

    if not os.path.exists(model_path) or not any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))):
        print(f"📥 Downloading OmniVoice Weights to {model_path}...")
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(repo_id="k2-fsa/OmniVoice", local_dir=model_path, local_dir_use_symlinks=False, local_files_only=False)
            print("✅ OmniVoice Weights: Ready")
        except Exception:
            print("⚠️ Switching to Git fallback for OmniVoice...")
            os.system(f"git clone https://huggingface.co/k2-fsa/OmniVoice {model_path}_tmp && mv {model_path}_tmp/* {model_path}/ && rm -rf {model_path}_tmp")
            print("✅ OmniVoice Weights: Ready (Fallback)")

    if TTS_ENGINE is None:
        # Use float32 for CPU, float16 for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        TTS_ENGINE = TTSEngine(model_path, device=device, dtype=dtype)
    
    # 2. Handle Whisper Model
    if WHISPER_PIPE is None:
        if not os.path.exists(whisper_path) or not any(f.endswith(('.bin', '.safetensors', '.pt')) for f in os.listdir(whisper_path) if os.path.isfile(os.path.join(whisper_path, f))):
            print(f"📥 Downloading Whisper Turbo to {whisper_path}...")
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id="openai/whisper-large-v3-turbo", local_dir=whisper_path, local_dir_use_symlinks=False, local_files_only=False)
                print("✅ Whisper Turbo: Ready")
            except Exception:
                print("⚠️ Switching to Git fallback for Whisper...")
                os.system(f"git clone https://huggingface.co/openai/whisper-large-v3-turbo {whisper_path}_tmp && mv {whisper_path}_tmp/* {whisper_path}/ && rm -rf {whisper_path}_tmp")
                print("✅ Whisper Turbo: Ready (Fallback)")
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        WHISPER_PIPE = pipeline("automatic-speech-recognition", model=whisper_path, device=device)
        print(f"✅ Engines Initialized on {device.upper()}")
        
        # Share the same pipe with the TTS engine to save ~1.6GB VRAM/RAM
        if TTS_ENGINE and hasattr(TTS_ENGINE.model, "_asr_pipe"):
            print("🔄 Injecting shared Whisper pipe into TTS Engine...")
            TTS_ENGINE.model._asr_pipe = WHISPER_PIPE

    return TTS_ENGINE, WHISPER_PIPE

# ---------------------------------------------------------------------------
# UI Helpers & Constants
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": ["Child / 儿童", "Teenager / 少年", "Young Adult / 青年", "Middle-aged / 中年", "Elderly / 老年"],
    "Pitch / 音调": ["Very Low Pitch / 极低音调", "Low Pitch / 低音调", "Moderate Pitch / 中音调", "High Pitch / 高音调", "Very High Pitch / 极高音调"],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": ["American Accent / 美式口音", "British Accent / 英国口音", "Australian Accent / 澳大利亚口音", "Chinese Accent / 中国口音"],
    "Chinese Dialect / 中文方言": ["Northeast Dialect / 东北话", "Sichuan Dialect / 四川话", "Henan Dialect / 河南话", "Shaanxi Dialect / 陕西话"],
}

_LANG_DISPLAY = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish"]

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
.output-panel { gap: 0 !important; overflow: visible !important; }

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

.lyrics-viewer {
    height: 260px;
    width: 100% !important;
    overflow-y: auto;
    padding: 10px 20px !important;
    box-sizing: border-box;
    display: none;
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
"""

_LYRICS_JS = """
() => {
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
                var targetTop = el.offsetTop - (viewer.offsetHeight / 2) + (el.offsetHeight / 2);
                viewer.scrollTo({ top: targetTop, behavior: 'smooth' });
            } else {
                el.classList.remove('active');
            }
        }
    }

    var PAIRS = [['vc-audio', 'vc-lyrics', 'vc-srt-text'], ['vd-audio', 'vd-lyrics', 'vd-srt-text']];

    function updateLyrics() {
        PAIRS.forEach(function(pair) {
            var audioId = pair[0], lyricsId = pair[1], srtBoxId = pair[2];
            var audioContainer = document.getElementById(audioId);
            var viewer = document.getElementById(lyricsId);
            var rawBox = document.getElementById(srtBoxId);
            if (!audioContainer || !viewer || !rawBox) return;

            var currentTime = -1;
            var audioEl = audioContainer.querySelector('audio');
            if (audioEl && !audioEl.paused && audioEl.currentTime > 0) {
                currentTime = audioEl.currentTime;
            }

            if (currentTime >= 0) {
                if (rawBox.style.display !== 'none') {
                    rawBox.style.display = 'none';
                    viewer.style.display = 'block';
                    var ta = rawBox.querySelector('textarea');
                    viewer._cues = parseSRT(ta ? ta.value : '');
                }
                var cues = viewer._cues || [];
                var activeIdx = -1;
                for (var i = 0; i < cues.length; i++) {
                    if (currentTime >= cues[i].start && currentTime < cues[i].end) { activeIdx = i; break; }
                }
                renderLyrics(viewer, cues, activeIdx);
            } else {
                if (rawBox.style.display === 'none') {
                    viewer.style.display = 'none';
                    rawBox.style.display = 'block';
                }
            }
        });
    }
    setInterval(updateLyrics, 100);
}
"""

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def text_to_srt_whisper(text, audio_tuple, pipe, language="zh"):
    """Generate SRT using Whisper word-level timestamps via pipeline."""
    try:
        sr, waveform = audio_tuple
        # Normalize to float32 for pipeline
        waveform_f32 = waveform.astype(np.float32) / 32767.0
        
        # Whisper pipeline expects a dict or numpy array
        result = pipe({"sampling_rate": sr, "raw": waveform_f32}, return_timestamps="word")
        chunks = result.get("chunks", [])
        if not chunks:
            return "Whisper failed to produce timestamps."
            
        segments = smart_balanced_split(text)
        seg_token_counts = []
        all_words = []
        for s in segments:
            tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", s)
            seg_token_counts.append(len(tokens))
            all_words.extend(tokens)
            
        srt_output = ""
        chunk_idx = 0
        for i, (seg_text, requested_count) in enumerate(zip(segments, seg_token_counts), 1):
            if requested_count == 0: continue
            start_time, end_time = None, None
            found = 0
            while chunk_idx < len(chunks) and found < requested_count:
                c = chunks[chunk_idx]
                if start_time is None: start_time = c["timestamp"][0]
                end_time = c["timestamp"][1]
                found += 1
                chunk_idx += 1
            
            if start_time is not None:
                if end_time is None: end_time = start_time + 1.0
                srt_output += f"{i}\n{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n{seg_text}\n\n"
        
        return srt_output.strip()
    except Exception as e:
        return f"SRT Error: {e}"

def generate_core(text, language, ref_audio, instruct, num_step, guidance, denoise, speed, duration, pp, po, mode, gen_srt=True):
    if not text or not text.strip():
        return None, "", None, "Text is required."
    
    start_time = time.time()
    lang_code = language if language != "Auto" else None
    
    try:
        audio_tuple = TTS_ENGINE.generate(
            text=text.strip(),
            ref_audio=ref_audio,
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
        
        elapsed = time.time() - start_time
        word_count = len(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text))
        status_msg = f"Done in {elapsed:.1f}s | {word_count} tokens"
        
        # Audio path
        sampling_rate, waveform = audio_tuple
        slug = get_slug(text)
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"{slug}.wav")
        sf.write(audio_path, waveform, sampling_rate)
        
        # SRT
        srt_content = ""
        zip_path = None
        if gen_srt:
            srt_content = text_to_srt_whisper(text, audio_tuple, WHISPER_PIPE)
            srt_path = audio_path.replace(".wav", ".srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            
            # ZIP
            zip_path = os.path.join(temp_dir, f"{slug}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(audio_path, arcname=f"{slug}.wav")
                zipf.write(srt_path, arcname=f"{slug}.srt")
        else:
            # If no SRT, download is just the WAV
            zip_path = audio_path
            
        return audio_path, srt_content, zip_path, status_msg
    except Exception as e:
        return None, "", None, f"Error: {e}"

# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------

def build_app(model_path=None, whisper_path=None):
    load_engines(model_path, whisper_path)
    
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="OmniVoice Pro") as demo:
        gr.Markdown("# OmniVoice Pro — Advanced TTS & Subtitles")
        
        with gr.Tabs():
            # VOICE CLONE TAB
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(label="Text to Synthesize", lines=5, placeholder="Enter text...")
                        vc_ref = gr.Audio(label="Reference Audio (Voice to Clone)", type="filepath", elem_classes="compact-audio")
                        
                        vc_ref_text = gr.Textbox(
                            label="Reference Text (Transcribed)", 
                            lines=2, 
                            placeholder="Transcript of reference audio. Click 'Transcribe' or wait for auto-transcribe.",
                            info="AI uses this to understand the reference voice's style."
                        )
                        vc_transcribe_btn = gr.Button("🔍 Transcribe Reference", variant="secondary", size="sm")

                        with gr.Accordion("Advanced Settings", open=False):
                            vc_lang = gr.Dropdown(label="Language", choices=_LANG_DISPLAY, value="Auto")
                            vc_speed = gr.Slider(0.5, 2.0, value=0.9, step=0.05, label="Speed")
                            vc_dur = gr.Number(label="Fixed Duration (sec)", value=0)
                            vc_steps = gr.Slider(4, 64, value=32, step=4, label="Inference Steps")
                            vc_gs = gr.Slider(0, 5, value=0.5, step=0.1, label="Guidance Scale")
                            vc_dn = gr.Checkbox(label="Denoise", value=True)
                            vc_pp = gr.Checkbox(label="Clean Ref Audio (Silence Removal)", value=True)
                            vc_po = gr.Checkbox(label="Trim Output Silence", value=True)
                            vc_gen_srt = gr.Checkbox(label="Generate Subtitles (SRT)", value=True)
                            
                        vc_btn = gr.Button("Generate Voice", variant="primary")
                        
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(label="Result", type="filepath", elem_id="vc-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<label class="custom-label">Subtitle</label>')
                            gr.HTML('<div id="vc-lyrics" class="lyrics-viewer"></div>')
                            vc_srt = gr.Textbox(show_label=False, lines=10, elem_id="vc-srt-text", interactive=False)
                        
                        vc_dl = gr.DownloadButton("📥 Download ZIP (Audio + SRT)", visible=False)
                        vc_status = gr.Textbox(label="Status", interactive=False)

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
                            vd_gs = gr.Slider(0, 5, value=0.5, step=0.1, label="Guidance Scale")
                            vd_dn = gr.Checkbox(label="Denoise", value=True)
                            vd_po = gr.Checkbox(label="Trim Output Silence", value=True)
                            vd_gen_srt = gr.Checkbox(label="Generate Subtitles (SRT)", value=True)

                        vd_btn = gr.Button("Create Voice", variant="primary")

                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(label="Result", type="filepath", elem_id="vd-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<label class="custom-label">Subtitle</label>')
                            gr.HTML('<div id="vd-lyrics" class="lyrics-viewer"></div>')
                            vd_srt = gr.Textbox(show_label=False, lines=10, elem_id="vd-srt-text", interactive=False)
                        
                        vd_dl = gr.DownloadButton("📥 Download ZIP (Audio + SRT)", visible=False)
                        vd_status = gr.Textbox(label="Status", interactive=False)

        # Event Handlers
        def transcribe_ref(audio):
            if not audio: return ""
            print(f"🎙️ Auto-transcribing reference audio...")
            result = WHISPER_PIPE(audio)
            text = result.get("text", "").strip()
            print(f"📝 Transcription: {text}")
            return text

        def vc_handler(*args):
            # args: text, lang, ref_audio, ref_text, steps, gs, denoise, speed, dur, pp, po, gen_srt
            return generate_core(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], "clone", args[11])
            
        # Transcription events
        vc_ref.change(transcribe_ref, inputs=[vc_ref], outputs=[vc_ref_text])
        vc_transcribe_btn.click(transcribe_ref, inputs=[vc_ref], outputs=[vc_ref_text])

        vc_btn.click(
            vc_handler,
            inputs=[vc_text, vc_lang, vc_ref, vc_ref_text, vc_steps, vc_gs, vc_dn, vc_speed, vc_dur, vc_pp, vc_po, vc_gen_srt],
            outputs=[vc_audio, vc_srt, vc_dl, vc_status]
        ).then(lambda dl: gr.update(visible=True, value=dl), inputs=[vc_dl], outputs=[vc_dl])

        def vd_handler(text, lang, speed, dur, steps, gs, dn, po, gen_srt, *groups):
            instruct = ", ".join([g for g in groups if g != "Auto"])
            return generate_core(text, lang, None, instruct, steps, gs, dn, speed, dur, False, po, "design", gen_srt)

        vd_btn.click(
            vd_handler,
            inputs=[vd_text, vd_lang, vd_speed, vd_dur, vd_steps, vd_gs, vd_dn, vd_po, vd_gen_srt] + vd_groups,
            outputs=[vd_audio, vd_srt, vd_dl, vd_status]
        ).then(lambda dl: gr.update(visible=True, value=dl), inputs=[vd_dl], outputs=[vd_dl])

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
    app.launch(server_name="0.0.0.0", server_port=7860, share=args.share)
