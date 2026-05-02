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
        self.model = OmniVoice.from_pretrained(
            model_path, 
            device_map=self.device,
            torch_dtype=self.dtype
        )
        self.sampling_rate = self.model.sampling_rate

    def generate(self, text, language=None, ref_audio=None, ref_text=None, 
                 instruct=None, speed=0.9, duration=None, num_step=32, 
                 guidance_scale=0.5, denoise=True, preprocess_prompt=True,
                 postprocess_output=True):
        
        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output)
        )

        kw = dict(
            text=text.strip(), 
            language=language if (language and language != "Auto") else None,
            generation_config=gen_config
        )

        if speed != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if ref_audio:
            kw["voice_clone_prompt"] = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        audio = self.model.generate(**kw)
        waveform = (audio[0] * 32767).astype(np.int16).squeeze()
        
        return self.sampling_rate, waveform
    def transcribe(self, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            return ""
        try:
            # Load audio using soundfile to ensure compatibility
            audio_data, sr = sf.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=-1)  # Convert to mono
            
            # Use the internal model's transcribe method
            # This is more robust as it uses the same engine as the TTS
            result = self.model.transcribe((audio_data, sr))
            return result.get("text", "").strip()
        except Exception as e:
            print(f"Engine transcription error: {e}")
            return f"Error: {e}"
