#!/usr/bin/env python3
import argparse
import logging
import os
import re
import tempfile
import zipfile
import torch
import numpy as np
import soundfile as sf
import torchaudio
import gradio as gr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

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
        # CJK and everything else — character-level
        max_val, soft_limit = 14, 18
        sentences = re.split(r"(?<=[。！？])", text)
        all_chunks = []
        for s in sentences:
            s = s.strip()
            if s:
                all_chunks.extend(split_into_balanced_chunks(list(s), max_val, soft_limit, ""))
        return finalize_chunks(all_chunks)


# ---------------------------------------------------------------------------
# Timestamp formatter
# ---------------------------------------------------------------------------
def format_timestamp(seconds):
    td = float(seconds)
    hours = int(td // 3600)
    minutes = int((td % 3600) // 60)
    secs = int(td % 60)
    millis = int(round((td % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Wav2Vec2 alignment model cache
# ---------------------------------------------------------------------------
ALIGN_MODEL = None
ALIGN_PROCESSOR = None


def load_align_model():
    """Load Wav2Vec2 alignment model; prefer local copy, fall back to HuggingFace."""
    global ALIGN_MODEL, ALIGN_PROCESSOR
    if ALIGN_MODEL is None:
        # Check for a locally bundled model first
        local_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "zh_alignment")
        local_path = os.path.normpath(local_path)
        model_path = local_path if os.path.isdir(local_path) else "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        print(f"Loading alignment model from: {model_path}")
        ALIGN_PROCESSOR = Wav2Vec2Processor.from_pretrained(model_path)
        ALIGN_MODEL = Wav2Vec2ForCTC.from_pretrained(model_path)
        ALIGN_MODEL.eval()
    return ALIGN_MODEL, ALIGN_PROCESSOR


# ---------------------------------------------------------------------------
# SRT generation — Wav2Vec2 model-based (robust / accurate)
# ---------------------------------------------------------------------------
def text_to_srt_robust(text: str, audio_tuple, language=None) -> str:
    """Generate SRT using Wav2Vec2 forced alignment."""
    align_model, processor = load_align_model()

    sr, waveform_int16 = audio_tuple

    # 1. Convert to float32 mono at 16 kHz
    waveform_f32 = waveform_int16.astype(np.float32) / 32767.0
    if waveform_f32.ndim > 1:
        waveform_f32 = waveform_f32.mean(axis=-1)  # stereo → mono

    audio_pt = torch.from_numpy(waveform_f32)
    if sr != 16000:
        audio_pt = torchaudio.functional.resample(audio_pt, sr, 16000)

    # 2. Split text into subtitle lines
    segments = smart_split(text, language)
    if not segments:
        return "No text to align."
    full_text = "".join(segments)

    # 3. Map text characters to vocab IDs (skip unknowns)
    vocab = processor.tokenizer.get_vocab()
    alignable_chars = [c for c in full_text if c in vocab]
    if not alignable_chars:
        return "No characters in the alignment vocabulary."

    tokens = [vocab[c] for c in alignable_chars]

    # 4. Get CTC emissions from the model
    with torch.inference_mode():
        inputs = processor(
            audio_pt.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
        )
        logits = align_model(**inputs).logits  # (1, T, vocab)
        emissions = torch.log_softmax(logits, dim=-1).squeeze(0).cpu()  # (T, vocab)

    # 5. Forced alignment via torchaudio
    targets = torch.tensor([tokens], dtype=torch.int32)  # (1, N)
    input_lengths = torch.tensor([emissions.shape[0]], dtype=torch.int32)
    target_lengths = torch.tensor([len(tokens)], dtype=torch.int32)

    aligned_paths, _ = torchaudio.functional.forced_align(
        emissions.unsqueeze(0),  # (1, T, vocab)
        targets,
        input_lengths,
        target_lengths,
        blank=0,
    )
    # aligned_paths: (1, T) token IDs along best path

    # 6. Collapse repeated tokens and blanks → per-character time spans
    path = aligned_paths.squeeze(0).tolist()          # (T,)
    stride_sec = 0.02  # Wav2Vec2 standard: 320 samples / 16000 Hz

    char_spans = []  # list of (start_sec, end_sec) per alignable character
    prev_tok = -1
    span_start = None
    for t, tok in enumerate(path):
        if tok == 0:  # blank
            if span_start is not None and prev_tok != 0:
                char_spans.append((span_start * stride_sec, t * stride_sec))
                span_start = None
            prev_tok = tok
            continue
        if tok != prev_tok:
            if span_start is not None and prev_tok != 0:
                char_spans.append((span_start * stride_sec, t * stride_sec))
            span_start = t
        prev_tok = tok
    # Close last open span
    if span_start is not None and prev_tok != 0:
        char_spans.append((span_start * stride_sec, len(path) * stride_sec))

    if len(char_spans) != len(alignable_chars):
        # Lengths mismatch — use uniform time division as a fallback
        total_dur = audio_pt.shape[-1] / 16000
        step = total_dur / max(len(alignable_chars), 1)
        char_spans = [(i * step, (i + 1) * step) for i in range(len(alignable_chars))]

    # 7. Group char spans back into per-segment SRT blocks
    srt_output = ""
    char_idx = 0
    srt_idx = 1
    for line in segments:
        line_alignable = [c for c in line if c in vocab]
        n = len(line_alignable)
        if n == 0:
            continue
        line_spans = char_spans[char_idx: char_idx + n]
        char_idx += n
        if not line_spans:
            continue
        t_start = line_spans[0][0]
        t_end = line_spans[-1][1]
        srt_output += (
            f"{srt_idx}\n"
            f"{format_timestamp(t_start)} --> {format_timestamp(t_end)}\n"
            f"{line}\n\n"
        )
        srt_idx += 1

    return srt_output if srt_output else "No alignment result produced."


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
# Language list for UI
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_demo(model: OmniVoice):
    sampling_rate = model.sampling_rate

    def _gen_core(
        text, language, ref_audio, instruct,
        num_step, guidance_scale, denoise, speed, duration,
        preprocess_prompt, postprocess_output,
        mode, generate_srt=True, ref_text=None,
    ):
        if not text or not text.strip():
            return None, "", None, "Please enter text."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale or 0.5),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if language != "Auto" else None
        kw = dict(text=text.strip(), language=lang, generation_config=gen_config)
        if speed and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration and float(duration) > 0:
            kw["duration"] = float(duration)
        if instruct:
            kw["instruct"] = instruct.strip()

        if mode == "clone":
            if not ref_audio:
                return None, "", None, "Reference audio is required for voice cloning."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text or None
            )

        try:
            audio = model.generate(**kw)
        except Exception as e:
            logging.exception("Generation failed")
            return None, "", None, f"Generation error: {e}"

        waveform_f32 = audio[0]
        waveform_int16 = (waveform_f32 * 32767).clip(-32768, 32767).astype(np.int16)

        slug = get_slug(text)
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"{slug}.wav")
        sf.write(audio_path, waveform_f32, sampling_rate, subtype="FLOAT")

        srt_content = ""
        download_path = audio_path
        if generate_srt:
            audio_tuple = (sampling_rate, waveform_int16)
            srt_content = text_to_srt_robust(text, audio_tuple, language=lang)
            srt_path = os.path.join(temp_dir, f"{slug}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            zip_path = os.path.join(temp_dir, f"{slug}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(audio_path, arcname=f"{slug}.wav")
                zipf.write(srt_path, arcname=f"{slug}.srt")
            download_path = zip_path

        return audio_path, srt_content, download_path, "Done."

    with gr.Blocks(title="OmniVoice (Robust-Model)") as demo:
        gr.Markdown("# OmniVoice — Robust Model-based Alignment")
        with gr.Tabs():
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column():
                        vc_text = gr.Textbox(label="Text", lines=4)
                        vc_ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                        vc_ref_text = gr.Textbox(label="Reference Text (optional)")
                        vc_lang = gr.Dropdown(label="Language", choices=_ALL_LANGUAGES, value="Auto")
                        vc_gen_srt = gr.Checkbox(label="Generate SRT", value=True)
                        with gr.Accordion("Settings", open=False):
                            vc_sp = gr.Slider(0.5, 1.5, value=0.9, step=0.05, label="Speed")
                            vc_ns = gr.Slider(4, 64, value=32, step=4, label="Steps")
                            vc_gs = gr.Slider(0.0, 5.0, value=0.5, step=0.1, label="Guidance Scale")
                        vc_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        vc_audio = gr.Audio(label="Output Audio")
                        vc_srt = gr.Textbox(label="SRT Preview", lines=10)
                        vc_dl = gr.DownloadButton("Download ZIP", visible=False)
                        vc_status = gr.Textbox(label="Status", interactive=False)

                vc_btn.click(
                    _gen_core,
                    inputs=[
                        vc_text, vc_lang, vc_ref_audio,
                        gr.State(""),   # instruct
                        vc_ns, vc_gs,
                        gr.State(True), # denoise
                        vc_sp,
                        gr.State(None), # duration
                        gr.State(True), # preprocess_prompt
                        gr.State(True), # postprocess_output
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

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OmniVoice Robust-Model Demo")
    parser.add_argument(
        "--model", type=str, default=".",
        help="Path to local model folder (default: current directory)"
    )
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true", help="Launch with a public Gradio share link")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OmniVoice from '{args.model}' on {device}...")
    model = OmniVoice.from_pretrained(args.model, device_map=device)
    demo = build_demo(model)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
