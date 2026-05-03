import os
import re
import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

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

class TTSEngine:
    def __init__(self, model_path, device=None, dtype=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Default dtype logic: float32 for CPU (mandatory), float16 for GPU
        if dtype is None:
            self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        else:
            self.dtype = dtype

        print(f"Initializing OmniVoice TTS Engine on {self.device} ({self.dtype})...")
        # Pass asr_model_name so the internal model knows where to find local Whisper for auto-transcription
        # This prevents the engine from trying to download weights from HuggingFace at runtime.
        self.model = OmniVoice.from_pretrained(
            model_path, 
            device_map=self.device,
            torch_dtype=self.dtype,
            asr_model_name=os.path.join(os.path.dirname(model_path), "whisper-large-v3-turbo") if os.path.exists(os.path.join(os.path.dirname(model_path), "whisper-large-v3-turbo")) else "openai/whisper-large-v3-turbo"
        )
        self.sampling_rate = self.model.sampling_rate

    def generate(self, text, language=None, ref_audio=None, ref_text=None, 
                 instruct=None, speed=1.0, duration=None, num_step=32, 
                 guidance_scale=2.0, denoise=True, preprocess_prompt=True,
                 postprocess_output=True):
        
        # Split text into sentences for progress reporting
        # This prevents the UI from appearing 'frozen' during long generations
        delimiters = r'([。！？；.!?;])'
        raw_parts = re.split(delimiters, text.strip())
        chunks = []
        for i in range(0, len(raw_parts)-1, 2):
            chunks.append(raw_parts[i] + raw_parts[i+1])
        if len(raw_parts) % 2 == 1 and raw_parts[-1]:
            chunks.append(raw_parts[-1])
        
        chunks = [c.strip() for c in chunks if c.strip()]
        if not chunks: chunks = [text.strip()]

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output)
        )

        full_waveform = []
        voice_clone_prompt = None
        if ref_audio:
            clean_ref_text = ref_text.strip() if (ref_text and ref_text.strip()) else None
            voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=clean_ref_text,
            )

        for i, chunk in enumerate(chunks, 1):
            # YIELD PROGRESS (Chunk X/Y)
            yield i, len(chunks), None
            
            kw = dict(
                text=chunk, 
                language=language if (language and language != "Auto") else None,
                generation_config=gen_config,
                voice_clone_prompt=voice_clone_prompt
            )
            if speed != 1.0: kw["speed"] = float(speed)
            if duration is not None and float(duration) > 0:
                # Approximate duration proportional to text length
                kw["duration"] = float(duration) * (len(chunk) / len(text))
            if instruct and instruct.strip(): kw["instruct"] = instruct.strip()

            audio = self.model.generate(**kw)
            chunk_wave = (audio[0] * 32767).astype(np.int16).squeeze()
            full_waveform.append(chunk_wave)

        final_waveform = np.concatenate(full_waveform)
        # Final yield with full waveform
        yield len(chunks), len(chunks), final_waveform
    def transcribe(self, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            return ""
        try:
            # Load audio using soundfile to ensure compatibility
            audio_data, sr = sf.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=-1)  # Convert to mono
            
            # Use the internal model's transcribe method
            result = self.model.transcribe((audio_data, sr))
            
            # Handle both dictionary (with timestamps) and raw string results
            if isinstance(result, dict):
                return result.get("text", "").strip()
            return str(result).strip()
        except Exception as e:
            print(f"Engine transcription error: {e}")
            return f"Error: {e}"
