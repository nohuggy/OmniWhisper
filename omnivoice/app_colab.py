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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [project_root, os.path.dirname(os.path.abspath(__file__)), os.path.join(project_root, "whisper-large-v3-turbo")]:
    if p not in sys.path:
        sys.path.append(p)

from omnivoice.omni_engine_colab import TTSEngine, get_slug
from whisper_engine import format_timestamp, unify_punctuation, smart_balanced_split, align_robust
from transformers import pipeline

# ---------------------------------------------------------------------------
# Global Engines
# ---------------------------------------------------------------------------
TTS_ENGINE = None
WHISPER_PIPE = None
def load_engines(model_path=None, whisper_path=None):
    global TTS_ENGINE, WHISPER_PIPE
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "resources")
    if whisper_path is None:
        whisper_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "whisper-large-v3-turbo")

    # 1. Handle OmniVoice TTS Model
    has_tts = os.path.exists(model_path) and any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(model_path))
    if not has_tts:
        print(f"📥 Downloading OmniVoice Weights...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="k2-fsa/OmniVoice", local_dir=model_path, local_dir_use_symlinks=False)

    if TTS_ENGINE is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        TTS_ENGINE = TTSEngine(model_path, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    
    # 2. Handle Whisper Model
    if WHISPER_PIPE is None:
        has_whisper = os.path.exists(whisper_path) and any(f.endswith(('.bin', '.safetensors', '.pt')) for f in os.listdir(whisper_path))
        if not has_whisper:
            print(f"📥 Downloading Whisper Turbo...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id='openai/whisper-large-v3-turbo', local_dir=whisper_path, local_dir_use_symlinks=False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # If local loading fails, attempt to load directly from Hub
        try:
            WHISPER_PIPE = pipeline("automatic-speech-recognition", model=whisper_path, device=device)
        except Exception as e:
            print(f"⚠️ Local Whisper load failed ({e}), trying Hub fallback...")
            WHISPER_PIPE = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=device)
        if TTS_ENGINE and hasattr(TTS_ENGINE.model, "_asr_pipe"):
            TTS_ENGINE.model._asr_pipe = WHISPER_PIPE
    return TTS_ENGINE, WHISPER_PIPE

# ---------------------------------------------------------------------------
# Logic Chaining (Radical Solution for Timeouts)
# ---------------------------------------------------------------------------

def optimize_audio_for_web(wav_path):
    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "5", mp3_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp3_path if os.path.exists(mp3_path) else wav_path
    except: return wav_path

def text_to_srt_whisper(text, audio_tuple, pipe, progress=None):
    try:
        if progress: progress(0.2, desc="🔍 Aligning subtitles...")
        sr, waveform = audio_tuple
        waveform_f32 = waveform.astype(np.float32) / 32767.0
        result = pipe({"sampling_rate": sr, "raw": waveform_f32}, return_timestamps="word")
        chunks = result.get("chunks", [])
        segments = smart_balanced_split(text)
        
        # Simple reconstruction for Colab stability
        abs_chunks = []
        offset = 0.0
        last_e = 0.0
        for c in chunks:
            s, e = c["timestamp"]
            if s is None: s = last_e
            if e is None: e = s + 0.5
            if (s + offset) < (last_e - 1.0) and s < 10.0: offset += 30.0
            c["timestamp"] = (s + offset, e + offset)
            abs_chunks.append(c)
            last_e = e + offset

        if progress: progress(0.7, desc="📝 Formatting SRT...")
        # (Simplified alignment for speed)
        srt_output = ""
        for i, seg in enumerate(segments, 1):
            s_time = (i-1) * (last_e / len(segments))
            e_time = i * (last_e / len(segments))
            srt_output += f"{i}\n{format_timestamp(s_time)} --> {format_timestamp(e_time)}\n{seg}\n\n"
        return srt_output.strip()
    except Exception as e: return f"SRT Error: {e}"

def tts_stage(text, language, ref_audio, ref_text, instruct, num_step, guidance, denoise, punc, speed, duration, pp, po, progress=gr.Progress()):
    if not text or not text.strip(): return None, None, "❌ Text is empty"
    if progress: progress(0, desc="🚀 Synthesizing Audio...")
    
    if punc: text = unify_punctuation(text)
    lang_code = language if language != "Auto" else None
    
    gen_iter = TTS_ENGINE.generate(
        text=text.strip(), ref_audio=ref_audio, ref_text=ref_text, instruct=instruct,
        language=lang_code, num_step=num_step, guidance_scale=guidance,
        denoise=denoise, speed=speed, duration=duration, preprocess_prompt=pp, postprocess_output=po
    )
    
    full_waveform = []
    sr = 16000
    for curr, total, chunk_wave in gen_iter:
        if chunk_wave is None:
            if progress: progress(curr/total, desc=f"⏳ TTS ({curr}/{total})")
        else:
            full_waveform = chunk_wave
            sr = TTS_ENGINE.sampling_rate
            
    os.makedirs("outputs", exist_ok=True)
    slug = get_slug(text)
    unique_slug = f"{slug}_{int(time.time())}"
    wav_path = f"outputs/{unique_slug}.wav"
    sf.write(wav_path, full_waveform, sr)
    audio_path = optimize_audio_for_web(wav_path)
    
    # 🏁 RADICAL FIX: ONLY pass the PATH in the state, NOT the huge raw array.
    # Passing large arrays in gr.State causes WebSocket overflow and disconnection.
    state = {
        "text": text,
        "wav_path": wav_path,
        "unique_slug": unique_slug,
        "slug": slug
    }
    
    return audio_path, state, f"✅ TTS Done. Starting ASR..."

def asr_stage(gen_srt, state, progress=gr.Progress()):
    if not gen_srt or not state: return gr.update(), gr.update(), "✅ Complete"
    
    text = state["text"]
    wav_path = state["wav_path"]
    unique_slug = state["unique_slug"]
    slug = state["slug"]
    
    if progress: progress(0, desc="🔍 Generating Subtitles...")
    
    # Load audio from file instead of state
    import soundfile as sf
    waveform, sr = sf.read(wav_path)
    audio_tuple = (sr, waveform)
    
    import torch
    torch.cuda.empty_cache()
    
    srt_content = text_to_srt_whisper(text, audio_tuple, WHISPER_PIPE, progress=progress)
    
    srt_path = f"outputs/{unique_slug}.srt"
    with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
    
    zip_path = f"outputs/{unique_slug}.zip"
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.write(wav_path, arcname=f"{slug}.wav")
        z.write(srt_path, arcname=f"{slug}.srt")
        
    return srt_content, zip_path, "✅ All Done"

# ---------------------------------------------------------------------------
# UI Construction
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
.output-panel { background: #1f2937; border-radius: 12px; padding: 12px; border: 1px solid #374151; }
#vc-lyrics, #vd-lyrics { background: #111; color: #fff; border-radius: 12px; height: 260px; overflow-y: auto; display: none; padding: 20px; }
.lyric-line { text-align: center; padding: 8px; color: #888; font-size: 1.1em; transition: all 0.2s; }
.lyric-line.active { color: #fff; font-weight: bold; background: rgba(79, 70, 229, 0.3); border-radius: 6px; }
"""

def build_app(model_path=None, whisper_path=None):
    load_engines(model_path, whisper_path)
    
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="OmniWhisper") as demo:
        gr.Markdown("# OmniWhisper Pro")
        
        with gr.Tabs():
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column():
                        vc_text = gr.Textbox(label="Text", lines=4)
                        vc_ref = gr.Audio(label="Reference", type="filepath")
                        vc_ref_text = gr.Textbox(label="Ref Transcript", lines=2)
                        with gr.Row():
                            vc_trans_btn = gr.Button("Trans Ref")
                            vc_gen_srt = gr.Checkbox(label="SRT", value=True)
                        with gr.Accordion("Settings", open=False):
                            vc_lang = gr.Dropdown(choices=_LANG_DISPLAY, value="Auto", label="Lang")
                            vc_speed = gr.Slider(0.5, 2.0, value=1.0, label="Speed")
                            vc_steps = gr.Slider(4, 64, value=32, step=4, label="Steps")
                            vc_punc = gr.Checkbox(label="Unify Punc", value=True)
                        vc_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column():
                        vc_audio = gr.Audio(label="Result", type="filepath", elem_id="vc-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div id="vc-lyrics"></div>')
                            vc_srt = gr.Textbox(show_label=False, lines=8, elem_id="vc-srt-text")
                        vc_dl = gr.DownloadButton("Download ZIP", visible=False)
                        vc_status = gr.Textbox(label="Status")
                        vc_state = gr.State()

            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column():
                        vd_text = gr.Textbox(label="Text", lines=4)
                        vd_gen_srt = gr.Checkbox(label="SRT", value=True)
                        vd_btn = gr.Button("Create", variant="primary")
                    with gr.Column():
                        vd_audio = gr.Audio(label="Result", type="filepath", elem_id="vd-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div id="vd-lyrics"></div>')
                            vd_srt = gr.Textbox(show_label=False, lines=8, elem_id="vd-srt-text")
                        vd_dl = gr.DownloadButton("Download ZIP", visible=False)
                        vd_status = gr.Textbox(label="Status")
                        vd_state = gr.State()

        # Logic Chaining
        vc_btn.click(
            tts_stage,
            inputs=[vc_text, vc_lang, vc_ref, vc_ref_text, gr.State(""), vc_steps, gr.State(3.0), gr.State(True), vc_punc, vc_speed, gr.State(0), gr.State(True), gr.State(True)],
            outputs=[vc_audio, vc_state, vc_status]
        ).then(
            asr_stage,
            inputs=[vc_gen_srt, vc_state],
            outputs=[vc_srt, vc_dl, vc_status]
        ).then(lambda dl: gr.update(visible=bool(dl)), inputs=[vc_dl], outputs=[vc_dl])

        vd_btn.click(
            tts_stage,
            inputs=[vd_text, gr.State("Auto"), gr.State(None), gr.State(None), gr.State("Male"), gr.State(32), gr.State(3.0), gr.State(True), gr.State(True), gr.State(1.0), gr.State(0), gr.State(False), gr.State(True)],
            outputs=[vd_audio, vd_state, vd_status]
        ).then(
            asr_stage,
            inputs=[vd_gen_srt, vd_state],
            outputs=[vd_srt, vd_dl, vd_status]
        ).then(lambda dl: gr.update(visible=bool(dl)), inputs=[vd_dl], outputs=[vd_dl])

        vc_trans_btn.click(lambda a: TTS_ENGINE.transcribe(a), inputs=[vc_ref], outputs=[vc_ref_text])
        demo.load(None, None, None, js=_LYRICS_JS)
        
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--whisper", type=str, default=None)
    args = parser.parse_args()
    app = build_app(model_path=args.model, whisper_path=args.whisper)
    app.queue().launch(server_name="0.0.0.0", server_port=7860, share=args.share)
