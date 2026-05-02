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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add paths for engine imports
current_dir = os.path.dirname(os.path.abspath(__file__))
whisper_root = os.path.join(project_root, "whisper-large-v3-turbo")

for p in [current_dir, whisper_root]:
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
    if not os.path.exists(model_path) or not any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))):
        print(f"📥 OmniVoice model weights not found in {model_path}. Downloading from HuggingFace...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="k2-fsa/OmniVoice",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        print("✅ OmniVoice weights downloaded successfully.")

    if TTS_ENGINE is None:
        TTS_ENGINE = TTSEngine(model_path)
    
    # 2. Handle Whisper Model
    if WHISPER_PIPE is None:
        if not os.path.exists(whisper_path) or not any(f.endswith(('.bin', '.safetensors', '.pt')) for f in os.listdir(whisper_path) if os.path.isfile(os.path.join(whisper_path, f))):
            print(f"📥 Whisper model not found at {whisper_path}. Downloading from HuggingFace...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="openai/whisper-large-v3-turbo",
                local_dir=whisper_path,
                local_dir_use_symlinks=False
            )
            print("✅ Whisper model downloaded successfully.")
            
        print(f"Loading Whisper model from: {whisper_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        WHISPER_PIPE = pipeline("automatic-speech-recognition", model=whisper_path, device=device)
        
        # Share the same pipe with the TTS engine to save ~1.6GB VRAM/RAM
        if TTS_ENGINE and hasattr(TTS_ENGINE.model, "_asr_pipe"):
            print("🔄 Injecting shared Whisper pipe into TTS Engine...")
            TTS_ENGINE.model._asr_pipe = WHISPER_PIPE
    return TTS_ENGINE, WHISPER_PIPE

# ---------------------------------------------------------------------------
# SRT Logic using Whisper
# ---------------------------------------------------------------------------
def text_to_srt_whisper(text, audio_tuple, pipe):
    sampling_rate, waveform = audio_tuple
    # Convert to float32 if needed
    if waveform.dtype == np.int16:
        audio_np = waveform.astype(np.float32) / 32767.0
    else:
        audio_np = waveform

    # Ensure mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    print("Running Whisper Inference for SRT...")
    res = pipe(audio_np, return_timestamps=True, chunk_length_s=30, batch_size=1)
    whisper_chunks = res.get("chunks", [])
    
    # Clean timestamps
    last_s = 0.0
    for c in whisper_chunks:
        s, e = c["timestamp"]
        if s is None: c["timestamp"] = (last_s, last_s + 0.5)
        elif e is None: c["timestamp"] = (s, s + 0.5)
        last_s = c["timestamp"][1]

    user_segments = smart_balanced_split(text)
    aligned = align_robust(user_segments, whisper_chunks)
    
    srt_output = ""
    for i, ((start, end), seg_text) in enumerate(zip(aligned, user_segments)):
        srt_output += f"{i+1}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{seg_text}\n\n"
    
    return srt_output.strip()

# ---------------------------------------------------------------------------
# Gradio UI Components
# ---------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}

.output-panel, 
.output-panel * {
    background-color: #1f2937 !important;
    background: #1f2937 !important;
    border: none !important;
    box-shadow: none !important;
}

.output-panel {
    gap: 0 !important;
    overflow: visible !important;
}

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
    color: #888;
    font-size: 1.1em;
    line-height: 1.4;
}

.lyrics-viewer .lyric-line.active {
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1.25em !important;
    background-color: rgba(59,130,246,0.2) !important;
}
"""

LYRICS_JS = """
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

    function updateLyrics() {
        var audioId = 'vc-audio', lyricsId = 'vc-lyrics', srtBoxId = 'vc-srt-text';
        var audioContainer = document.getElementById(audioId);
        var viewer = document.getElementById(lyricsId);
        var rawBox = document.getElementById(srtBoxId);
        if (!audioContainer || !viewer || !rawBox) return;

        var audioEl = audioContainer.querySelector('audio');
        if (!audioEl) return;

        var srtText = '';
        var ta = rawBox.querySelector('textarea');
        if (ta) srtText = ta.value;
        
        if (!viewer._cues || viewer._lastSRT !== srtText) {
            viewer._cues = parseSRT(srtText);
            viewer._lastSRT = srtText;
            var html = '';
            for (var i = 0; i < viewer._cues.length; i++) {
                html += '<div class="lyric-line" data-idx="' + i + '">' + viewer._cues[i].text + '</div>';
            }
            viewer.innerHTML = html;
            viewer.style.display = srtText ? 'block' : 'none';
        }

        var currentTime = audioEl.currentTime;
        var activeIdx = -1;
        for (var i = 0; i < viewer._cues.length; i++) {
            if (currentTime >= viewer._cues[i].start && currentTime <= viewer._cues[i].end) {
                activeIdx = i;
                break;
            }
        }

        if (viewer._lastActive !== activeIdx) {
            viewer._lastActive = activeIdx;
            var lines = viewer.children;
            for (var i = 0; i < lines.length; i++) {
                if (i === activeIdx) {
                    lines[i].classList.add('active');
                    var targetTop = lines[i].offsetTop - (viewer.offsetHeight / 2) + (lines[i].offsetHeight / 2);
                    viewer.scrollTo({ top: targetTop, behavior: 'smooth' });
                } else {
                    lines[i].classList.remove('active');
                }
            }
        }
    }
    setInterval(updateLyrics, 100);
}
"""

def build_app(model_path=".", whisper_path="./whisper_model"):
    tts, whisper = load_engines(model_path, whisper_path)

    def process_tts(text, ref_audio, ref_text, speed, gen_srt):
        waveform, sr = tts.generate(
            text=text,
            voice_clone_audio=ref_audio,
            voice_clone_text=ref_text,
            speed=speed
        )
        
        slug = get_slug(text)
        temp_dir = tempfile.mkdtemp()
        wav_path = os.path.join(temp_dir, f"{slug}.wav")
        sf.write(wav_path, waveform, sr)
        
        srt_content = ""
        download_path = wav_path
        
        if gen_srt:
            srt_content = text_to_srt_whisper(text, (sr, waveform), whisper)
            srt_path = wav_path.replace(".wav", ".srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            
            zip_path = os.path.join(temp_dir, f"{slug}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(wav_path, arcname=f"{slug}.wav")
                zipf.write(srt_path, arcname=f"{slug}.srt")
            download_path = zip_path
            
        return wav_path, srt_content, download_path

    with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# 🎙️ OmniWhisper: Precision TTS & Whisper SRT")
        
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(label="Text to Synthesize", lines=5)
                ref_audio = gr.Audio(label="Reference Voice (Clone)", type="filepath")
                ref_text = gr.Textbox(label="Reference Text (Optional)")
                speed = gr.Slider(0.5, 2.0, value=1.0, label="Speed")
                gen_srt = gr.Checkbox(label="Generate SRT (Whisper)", value=True)
                btn = gr.Button("Generate", variant="primary")
            
            with gr.Column(scale=1, elem_classes="output-panel"):
                gr.HTML('<div class="custom-label">Output Audio & Lyrics</div>')
                out_audio = gr.Audio(label="Result", elem_id="vc-audio")
                out_lyrics = gr.HTML('<div id="vc-lyrics" class="lyrics-viewer"></div>')
                out_srt = gr.Textbox(label="SRT Content", visible=False, elem_id="vc-srt-text")
                out_download = gr.File(label="Download (WAV/ZIP)")

        btn.click(
            process_tts,
            inputs=[text_input, ref_audio, ref_text, speed, gen_srt],
            outputs=[out_audio, out_srt, out_download]
        )
        
        demo.load(js=LYRICS_JS)

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--whisper", default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    app = build_app(args.model, args.whisper)
    app.launch(share=args.share)
