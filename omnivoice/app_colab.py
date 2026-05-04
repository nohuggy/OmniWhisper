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
import gc

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

    if not os.path.exists(model_path) or not any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(model_path)):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="k2-fsa/OmniVoice", local_dir=model_path, local_dir_use_symlinks=False)

    if TTS_ENGINE is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        TTS_ENGINE = TTSEngine(model_path, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    
    if WHISPER_PIPE is None:
        if not os.path.exists(whisper_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id='openai/whisper-large-v3-turbo', local_dir=whisper_path, local_dir_use_symlinks=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        WHISPER_PIPE = pipeline("automatic-speech-recognition", model=whisper_path, device=device)
        if TTS_ENGINE and hasattr(TTS_ENGINE.model, "_asr_pipe"):
            TTS_ENGINE.model._asr_pipe = WHISPER_PIPE
    return TTS_ENGINE, WHISPER_PIPE

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def optimize_audio_for_web(wav_path):
    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        # Use low-priority nice to avoid kernel freeze
        subprocess.run(["nice", "-n", "19", "ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "5", mp3_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp3_path if os.path.exists(mp3_path) else wav_path
    except: return wav_path

def text_to_srt_whisper(text, audio_tuple, pipe, progress=None):
    try:
        sr, waveform = audio_tuple
        waveform_f32 = waveform.astype(np.float32) / 32767.0
        result = pipe({"sampling_rate": sr, "raw": waveform_f32}, return_timestamps="word")
        chunks = result.get("chunks", [])
        segments = smart_balanced_split(text)
        
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

        srt_output = ""
        for i, seg in enumerate(segments, 1):
            s_time = (i-1) * (last_e / len(segments))
            e_time = i * (last_e / len(segments))
            srt_output += f"{i}\n{format_timestamp(s_time)} --> {format_timestamp(e_time)}\n{seg}\n\n"
        return srt_output.strip()
    except Exception as e: return f"SRT Error: {e}"

def generate_core(text, language, ref_audio, ref_text, instruct, num_step, guidance, denoise, speed, duration, pp, po, mode, gen_srt=True, convert_punc=True, progress=gr.Progress()):
    if not text or not text.strip():
        yield None, "", None, "Error: Text is empty"
        return

    # 1. Immediate Heartbeat
    yield gr.update(), gr.update(), gr.update(), "🚀 Initializing..."
    if progress: progress(0, desc="🚀 Starting...")

    if convert_punc:
        text = unify_punctuation(text)
        if ref_text: ref_text = unify_punctuation(ref_text)
            
    try:
        # 2. TTS Generation
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
                if progress: progress(curr/total * 0.7, desc=f"⏳ TTS ({curr}/{total})")
                # Rare yield to keep connection alive without overloading tunnel
                if curr % max(1, total // 4) == 0:
                    yield gr.update(), gr.update(), gr.update(), f"⏳ TTS: {curr}/{total}"
            else:
                full_waveform = chunk_wave
                sr = TTS_ENGINE.sampling_rate

        # 🚀 THE RADICAL FIX: Break the memory spike
        yield gr.update(), gr.update(), gr.update(), "⏳ TTS Complete. Clearing GPU memory..."
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5) # Let kernel breathe

        # 3. Save Audio
        slug = get_slug(text)
        unique_slug = f"{slug}_{int(time.time())}"
        os.makedirs("outputs", exist_ok=True)
        wav_path = f"outputs/{unique_slug}.wav"
        sf.write(wav_path, full_waveform, sr)
        audio_path = optimize_audio_for_web(wav_path)
        
        # Immediate yield of audio to "success" the UI
        yield audio_path, gr.update(), gr.update(), "⏳ Audio ready. Starting ASR..."

        # 4. SRT Generation
        srt_content = ""
        zip_path = None
        if gen_srt:
            if progress: progress(0.8, desc="🔍 Aligning Subtitles...")
            
            # Use a tiny loop for ASR heartbeat
            import threading
            asr_res = {"val": None, "err": None}
            def run_asr():
                try: asr_res["val"] = text_to_srt_whisper(text, (sr, full_waveform), WHISPER_PIPE)
                except Exception as e: asr_res["err"] = str(e)
            
            t = threading.Thread(target=run_asr)
            t.start()
            while t.is_alive():
                yield gr.update(), gr.update(), gr.update(), "⏳ ASR Alignment in progress..."
                time.sleep(3.0)
            t.join()
            
            srt_content = asr_res["val"] or f"SRT Error: {asr_res['err']}"
            
            # Final Package
            srt_path = f"outputs/{unique_slug}.srt"
            with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
            zip_path = f"outputs/{unique_slug}.zip"
            with zipfile.ZipFile(zip_path, 'w') as z:
                z.write(wav_path, arcname=f"{slug}.wav")
                z.write(srt_path, arcname=f"{slug}.srt")

        if progress: progress(1.0, desc="✅ Done")
        yield audio_path, srt_content, zip_path if zip_path else gr.update(), "✅ Generation Complete"
        
    except Exception as e:
        yield None, "", None, f"❌ Error: {e}"

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
.output-panel { background-color: #1f2937 !important; border: 1px solid #374151 !important; border-radius: 12px; }
.custom-label { display: inline-block; background: #4f46e5; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-bottom: 4px; font-weight: bold; }
#vc-lyrics, #vd-lyrics { background: #111; color: #fff; border-radius: 12px; height: 260px; overflow-y: auto; display: none; padding: 15px; text-align: center; }
.lyric-line { padding: 5px; color: #888; transition: all 0.2s; }
.lyric-line.active { color: #fff; font-weight: bold; background: rgba(79, 70, 229, 0.2); }
"""

_LYRICS_JS = r"""
() => {
    setInterval(() => {
        ['vc', 'vd'].forEach(prefix => {
            const audio = document.querySelector(`#${prefix}-audio audio`);
            const srt = document.querySelector(`#${prefix}-srt-text textarea`)?.value;
            const viewer = document.getElementById(`${prefix}-lyrics`);
            if (!audio || !srt || audio.paused) {
                if (viewer) viewer.style.display = 'none';
                return;
            }
            viewer.style.display = 'block';
            // Simple display for this version
            viewer.innerHTML = '<div style="color:#aaa;padding:20px;">' + srt.split('\n\n').pop().split('\n').pop() + '</div>';
        });
    }, 500);
}
"""

def build_app(model_path=None, whisper_path=None):
    load_engines(model_path, whisper_path)
    
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="OmniWhisper") as demo:
        gr.Markdown("# OmniWhisper")
        
        with gr.Tabs():
            # VC TAB
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column():
                        vc_text = gr.Textbox(label="Text to Synthesize", lines=5)
                        vc_ref = gr.Audio(label="Reference Audio", type="filepath")
                        vc_ref_text = gr.Textbox(label="Reference Text", lines=2)
                        with gr.Row():
                            vc_trans_btn = gr.Button("Trans Ref", variant="secondary")
                            vc_gen_srt = gr.Checkbox(label="Generate Subtitles (SRT)", value=True)
                        with gr.Accordion("Advanced Settings", open=False):
                            vc_instruct = gr.Textbox(label="Voice Instruction")
                            vc_lang = gr.Dropdown(label="Language", choices=_LANG_DISPLAY, value="Auto")
                            vc_speed = gr.Slider(0.5, 2.0, value=0.9, step=0.05, label="Speed")
                            vc_dur = gr.Number(label="Fixed Duration (sec)", value=0)
                            vc_steps = gr.Slider(4, 64, value=32, step=4, label="Inference Steps")
                            vc_gs = gr.Slider(0, 5, value=3.0, step=0.1, label="Guidance Scale")
                            vc_dn = gr.Checkbox(label="Denoise", value=True)
                            vc_punc = gr.Checkbox(label="Convert Punctuation", value=True)
                            vc_pp = gr.Checkbox(label="Clean Ref Audio", value=True)
                            vc_po = gr.Checkbox(label="Trim Output Silence", value=True)
                        vc_btn = gr.Button("Generate Voice", variant="primary")
                    
                    with gr.Column():
                        vc_audio = gr.Audio(label="Result", type="filepath", elem_id="vc-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<label class="custom-label">Subtitle</label>')
                            gr.HTML('<div id="vc-lyrics"></div>')
                            vc_srt = gr.Textbox(show_label=False, lines=10, elem_id="vc-srt-text", interactive=False)
                        vc_dl = gr.DownloadButton("📥 Download ZIP", visible=False)
                        vc_status = gr.Textbox(label="Status", interactive=False)

            # VD TAB
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column():
                        vd_text = gr.Textbox(label="Text to Synthesize", lines=5)
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
                            vd_dn = gr.Checkbox(label="Denoise", value=True)
                            vd_punc = gr.Checkbox(label="Convert Punctuation", value=True)
                            vd_po = gr.Checkbox(label="Trim Output Silence", value=True)
                            vd_gen_srt = gr.Checkbox(label="Generate Subtitles (SRT)", value=True)
                        vd_btn = gr.Button("Create Voice", variant="primary")
                    
                    with gr.Column():
                        vd_audio = gr.Audio(label="Result", type="filepath", elem_id="vd-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<label class="custom-label">Subtitle</label>')
                            gr.HTML('<div id="vd-lyrics"></div>')
                            vd_srt = gr.Textbox(show_label=False, lines=10, elem_id="vd-srt-text", interactive=False)
                        vd_dl = gr.DownloadButton("📥 Download ZIP", visible=False)
                        vd_status = gr.Textbox(label="Status", interactive=False)

        # Unified Handler
        def handler(text, lang, ref, ref_text, instruct, steps, gs, dn, punc, speed, dur, pp, po, gen_srt, progress=gr.Progress()):
            for res in generate_core(text, lang, ref, ref_text, instruct, steps, gs, dn, speed, dur, pp, po, "clone", gen_srt, punc, progress):
                yield res

        vc_btn.click(handler, inputs=[vc_text, vc_lang, vc_ref, vc_ref_text, vc_instruct, vc_steps, vc_gs, vc_dn, vc_punc, vc_speed, vc_dur, vc_pp, vc_po, vc_gen_srt], outputs=[vc_audio, vc_srt, vc_dl, vc_status])
        vd_btn.click(handler, inputs=[vd_text, vd_lang, gr.State(None), gr.State(None), gr.State("Male"), vd_steps, vd_gs, vd_dn, vd_punc, vd_speed, vd_dur, gr.State(False), vd_po, vd_gen_srt], outputs=[vd_audio, vd_srt, vd_dl, vd_status])
        vc_trans_btn.click(lambda a: TTS_ENGINE.transcribe(a), inputs=[vc_ref], outputs=[vc_ref_text])
        
        vc_dl.then(lambda dl: gr.update(visible=bool(dl)), inputs=[vc_dl], outputs=[vc_dl])
        vd_dl.then(lambda dl: gr.update(visible=bool(dl)), inputs=[vd_dl], outputs=[vd_dl])
        
        demo.load(None, None, None, js=_LYRICS_JS)
        
    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
