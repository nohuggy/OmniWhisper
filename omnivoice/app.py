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
from whisper_engine import format_timestamp, unify_punctuation, smart_balanced_split, align_robust
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
    background-color: rgba(79, 70, 229, 0.4) !important;
}
"""

_LYRICS_JS = """
function() {
    const setupSync = (prefix) => {
        const container = document.querySelector(`#${prefix}-audio`);
        if (!container) return;

        const sync = () => {
            const audio = container.querySelector('audio');
            const lyrics = document.getElementById(`${prefix}-lyrics`);
            const srtBox = document.querySelector(`#${prefix}-srt-text`);
            const srt = srtBox ? srtBox.querySelector('textarea') : null;
            if (!audio || !lyrics || !srt || !srt.value) return;

            const parseSRT = (raw) => {
                var blocks = raw.trim().split(/\\n\\n+/);
                var cues = [];
                for (var b = 0; b < blocks.length; b++) {
                    var lines = blocks[b].split('\\n');
                    if (lines.length < 3) continue;
                    var times = lines[1].split(' --> ');
                    if (times.length !== 2) continue;
                    const parseTime = (s) => {
                        const p = s.replace(',','.').split(':');
                        return parseFloat(p[0])*3600 + parseFloat(p[1])*60 + parseFloat(p[2]);
                    };
                    cues.push({start: parseTime(times[0]), end: parseTime(times[1]), text: lines.slice(2).join('<br>')});
                }
                return cues;
            };

            const cues = parseSRT(srt.value);
            if (cues.length > 0) {
                lyrics.style.display = 'block';
                if (srtBox) srtBox.style.display = 'none';
            }

            audio.ontimeupdate = () => {
                const now = audio.currentTime;
                let html = '';
                cues.forEach(item => {
                    const active = now >= item.start && now <= item.end ? 'active' : '';
                    html += `<div class="lyric-line ${active}">${item.text}</div>`;
                });
                lyrics.innerHTML = html;
                const activeNode = lyrics.querySelector('.active');
                if (activeNode) activeNode.scrollIntoView({ behavior: 'smooth', block: 'center' });
            };
        };

        const observer = new MutationObserver(sync);
        observer.observe(container, { childList: true, subtree: true });
        const srtDiv = document.querySelector(`#${prefix}-srt-text`);
        if (srtDiv) {
            const ta = srtDiv.querySelector('textarea');
            if (ta) {
                const srtObs = new MutationObserver(sync);
                srtObs.observe(ta, { attributes: true });
            }
        }
        sync();
    };

    const init = () => {
        if (document.querySelector('#vc-audio')) {
            setupSync('vc');
            setupSync('vd');
        } else {
            setTimeout(init, 1000);
        }
    };
    init();
}
"""

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def text_to_srt_whisper(text, audio_tuple, pipe):
    try:
        sr, waveform = audio_tuple
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        # Run Whisper with word-level timestamps
        result = pipe(waveform, return_timestamps="word", generate_kwargs={"task": "transcribe"})
        chunks = result.get("chunks", [])
        
        # Alignment logic
        segments = smart_balanced_split(text)
        aligned = align_robust(segments, chunks)
        
        srt_output = ""
        for i, ((start_time, end_time), seg_text) in enumerate(zip(aligned, segments), 1):
            srt_output += f"{i}\n{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n{seg_text}\n\n"
        
        return srt_output.strip()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"SRT Error: {e}"

def generate_core(text, language, ref_audio, ref_text, instruct, num_step, guidance, denoise, speed, duration, pp, po, mode, gen_srt=True, convert_punc=True):
    if not text or not text.strip():
        yield None, "", gr.update(visible=False), "Text is required."
        return
    
    # 1. Punctuation Unification
    if convert_punc:
        text = unify_punctuation(text)
        if ref_text:
            ref_text = unify_punctuation(ref_text)
            
    start_time = time.time()
    lang_code = language if language != "Auto" else None
    
    try:
        # 2. TTS Generation (Streaming Progress)
        full_waveform = []
        sr = 16000
        
        gen_iter = TTS_ENGINE.generate(
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
        
        for curr, total, chunk_wave in gen_iter:
            if chunk_wave is None:
                yield gr.update(), gr.update(), gr.update(), f"⏳ Generation in progress... Synthesizing TTS (Chunk {curr}/{total})"
            else:
                full_waveform = chunk_wave
                sr = TTS_ENGINE.sampling_rate
        
        audio_tuple = (sr, full_waveform)
        duration_s = len(full_waveform) / sr
        
        # 3. SRT Generation
        srt_content = ""
        if gen_srt:
            yield gr.update(), gr.update(), gr.update(), f"⏳ TTS Done ({duration_s:.1f}s). Running ASR for alignment..."
            srt_content = text_to_srt_whisper(text, audio_tuple, WHISPER_PIPE)
            yield gr.update(), srt_content, gr.update(), f"⏳ ASR Done. Finalizing files..."
        
        # 4. Final Result Preparation
        slug = get_slug(text)
        unique_slug = f"{slug}_{int(time.time())}"
        
        os.makedirs("outputs", exist_ok=True)
        audio_path = f"outputs/{unique_slug}.wav"
        sf.write(audio_path, full_waveform, sr)
        
        zip_path = None
        if srt_content:
            srt_path = audio_path.replace(".wav", ".srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            
            zip_path = audio_path.replace(".wav", ".zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(audio_path, arcname=f"{slug}.wav")
                zipf.write(srt_path, arcname=f"{slug}.srt")
        else:
            zip_path = audio_path

        elapsed = time.time() - start_time
        tokens = len(text.strip())
        status_msg = f"✅ Done in {elapsed:.1f}s | {duration_s:.1f}s Audio | {tokens} Chars"
        
        # Final yield: Show everything at once to prevent double-loading flicker
        yield audio_path, srt_content, gr.update(value=zip_path, visible=True), status_msg
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield gr.update(), "", gr.update(visible=False), f"Error: {e}"

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
                            placeholder="Transcript of reference audio. Click 'Trans Ref' to process."
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
                            vc_gs = gr.Slider(0, 5, value=2.0, step=0.1, label="Guidance Scale")
                            vc_dn = gr.Checkbox(label="Denoise", value=True)
                            vc_pp = gr.Checkbox(label="Clean Ref Audio", value=True)
                            vc_po = gr.Checkbox(label="Trim Output Silence", value=True)
                            vc_gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                            vc_punc = gr.Checkbox(label="Convert Punctuation", value=True)
                            
                        vc_btn = gr.Button("Generate Voice", variant="primary")
                        
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(label="Result", type="filepath", elem_id="vc-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<label class="custom-label">Subtitle Preview</label>')
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
                            vd_gs = gr.Slider(0, 5, value=2.0, step=0.1, label="Guidance Scale")
                            vd_dn = gr.Checkbox(label="Denoise", value=True)
                            vd_po = gr.Checkbox(label="Trim Output Silence", value=True)
                            vd_gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                            vd_punc = gr.Checkbox(label="Convert Punctuation", value=True)

                        vd_btn = gr.Button("Create Voice", variant="primary")

                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(label="Result", type="filepath", elem_id="vd-audio")
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<label class="custom-label">Subtitle Preview</label>')
                            gr.HTML('<div id="vd-lyrics" class="lyrics-viewer"></div>')
                            vd_srt = gr.Textbox(show_label=False, lines=10, elem_id="vd-srt-text", interactive=False)
                        
                        vd_dl = gr.DownloadButton("📥 Download ZIP (Audio + SRT)", visible=False)
                        vd_status = gr.Textbox(label="Status", interactive=False, lines=3)

        # Event Handlers
        def transcribe_ref(audio):
            if not audio: return ""
            try:
                text = TTS_ENGINE.transcribe(audio)
                return text
            except Exception as e:
                return f"Error during transcription: {e}"

        def vc_handler(*args):
            for res in generate_core(
                text=args[0], language=args[1], ref_audio=args[2], ref_text=args[3], 
                instruct=args[4], num_step=args[5], guidance=args[6], denoise=args[7], 
                convert_punc=args[8], speed=args[9], duration=args[10], 
                pp=args[11], po=args[12], mode="clone", gen_srt=args[13]
            ):
                yield res
            
        def process_ref_zip(zip_file):
            if not zip_file: return None, ""
            import zipfile, tempfile
            audio_path, text_content = None, ""
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_file.name, 'r') as z:
                z.extractall(tmp)
                for f in z.namelist():
                    if f.endswith(('.wav', '.mp3', '.flac')) and not f.startswith('__MACOSX'):
                        audio_path = os.path.join(tmp, f)
                    if f.endswith('.txt') and not f.startswith('__MACOSX'):
                        try:
                            with open(os.path.join(tmp, f), 'r', encoding='utf-8') as tf:
                                text_content = tf.read()
                        except: pass
            return audio_path, text_content
            
        def process_ref_txt(txt_file):
            if not txt_file: return ""
            try:
                with open(txt_file.name, 'r', encoding='utf-8') as tf: return tf.read()
            except: return ""
            
        vc_ref_zip_btn.upload(process_ref_zip, inputs=[vc_ref_zip_btn], outputs=[vc_ref, vc_ref_text])
        vc_ref_txt_btn.upload(process_ref_txt, inputs=[vc_ref_txt_btn], outputs=[vc_ref_text])
        vc_transcribe_btn.click(transcribe_ref, inputs=[vc_ref], outputs=[vc_ref_text])

        vc_btn.click(
            vc_handler,
            inputs=[vc_text, vc_lang, vc_ref, vc_ref_text, vc_instruct, vc_steps, vc_gs, vc_dn, vc_punc, vc_speed, vc_dur, vc_pp, vc_po, vc_gen_srt],
            outputs=[vc_audio, vc_srt, vc_dl, vc_status]
        )

        def vd_handler(text, lang, speed, dur, steps, gs, dn, punc, po, gen_srt, *groups):
            instruct = ", ".join([g for g in groups if g != "Auto"])
            for res in generate_core(text, lang, None, None, instruct, steps, gs, dn, speed, dur, False, po, "design", gen_srt, punc):
                yield res

        vd_btn.click(
            vd_handler,
            inputs=[vd_text, vd_lang, vd_speed, vd_dur, vd_steps, vd_gs, vd_dn, vd_punc, vd_po, vd_gen_srt] + vd_groups,
            outputs=[vd_audio, vd_srt, vd_dl, vd_status]
        )

        demo.load(None, None, None, js=_LYRICS_JS)
        
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--whisper", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    app = build_app(args.model, args.whisper)
    app.launch(server_name="0.0.0.0", server_port=7860, share=args.share)
