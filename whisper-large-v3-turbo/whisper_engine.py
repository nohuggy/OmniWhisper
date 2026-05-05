import os
import sys
import torch
import torchaudio
import numpy as np
import re
from transformers import pipeline
import difflib

def unify_punctuation(text):
    if not text: return ""
    # Ellipsis: Convert ..., ⋯⋯, 。。。 to standard ……
    text = re.sub(r'(\.\.\.+|…+|⋯+|。。。+)', '……', text)
    # Title Marks: Convert 〈 〉, 『 』, 「 」 to standard 《 》 or 〈 〉
    # User's request: unify use of marks
    text = text.replace('『', '「').replace('』', '」') 
    # Quotation marks: Inner 『 』 -> ‘ ’ , Outer 「 」 -> “ ”
    text = text.replace('「', '“').replace('」', '”')
    text = text.replace('『', '‘').replace('』', '’')
    return text

def format_timestamp(seconds):
    td = float(seconds)
    hours = int(td // 3600)
    minutes = int((td % 3600) // 60)
    secs = int(td % 60)
    millis = int(round((td % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# ---------------------------------------------------------------------------
# ## Technical Maintenance Notes: Subtitle Splitting
# ---------------------------------------------------------------------------
# The smart_balanced_split regex is CAREFULLY TUNED to prevent bracket orphans.
# Group 3 (trailing punctuation) EXPLICITLY EXCLUDES opening brackets and CJK 
# opening marks (e.g., (, [, {, 「, 『, 《, 〈, “, ‘, （).
# This ensures that if a token ends with an opening mark, that mark is 
# treated as leading punctuation for the NEXT token, keeping it on the 
# same line as its content.
# ---------------------------------------------------------------------------

def smart_balanced_split(text):
    if not text: return []
    # Split into paragraphs first to avoid remainder accumulation
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    all_segments = []
    
    for p_text in paragraphs:
        # Pre-process: standardize spaces but keep punctuation
        p_text = re.sub(r'\s+', ' ', p_text).strip()
        # Pattern captures: (leading punctuation) + (word) + (trailing punctuation) + (trailing spaces)
        # Group 3 (trail) excludes opening marks to ensure they lead the NEXT token
        pattern = re.compile(r'([^\w\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]*)([\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|[a-zA-Z0-9-]+)([^\w\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\(\[\{\u300c\u300e\u300a\u3008\u201c\u2018\uFF08]*)(\s*)')
        tokens = []
        for match in pattern.finditer(p_text):
            lead_punct, word, trail_punct, space = match.groups()
            tokens.append(lead_punct + word + trail_punct + space)
        
        if not tokens: continue
        
        # Calculate segment budget
        # We use floor for a more conservative count to avoid orphans
        num_segments = max(1, int(len(tokens) / 11))
        
        # If the average tokens per line would be too high (>15), add a segment
        if len(tokens) / num_segments > 15:
            num_segments += 1
            
        avg_tokens = len(tokens) / num_segments
        start_idx = 0
        for i in range(num_segments):
            # LAST SEGMENT HANDLING
            if i == num_segments - 1:
                final_tokens = tokens[start_idx:]
                # If we have a previous segment and this one is an "orphan" (very short),
                # we should have merged it. But since we are here, just add it.
                # The look-ahead below usually prevents this.
                all_segments.append("".join(final_tokens).strip())
                break
                
            ideal_end = start_idx + int(avg_tokens)
            
            # ORPHAN PREVENTION: Look ahead to see if splitting here leaves too few tokens
            # If remaining tokens < 5, just merge everything into this segment and finish.
            remaining_after_ideal = len(tokens) - ideal_end
            if remaining_after_ideal < 5 and i == num_segments - 2:
                all_segments.append("".join(tokens[start_idx:]).strip())
                break

            best_break = ideal_end
            min_p = 1000
            # Window search for best punctuation
            for offset in range(-5, 6):
                idx = ideal_end + offset
                if idx <= start_idx or idx >= len(tokens): continue
                
                # Critical: Don't break if it leaves an orphan (< 5 tokens) for the NEXT line
                # unless it's the very last segment calculation
                if (len(tokens) - idx) < 5 and i < num_segments - 1:
                    continue

                t = tokens[idx - 1]
                p = abs(offset) * 3
                if any(x in t for x in "。！？.!?;；…"): p -= 30
                elif any(x in t for x in "，,"): p -= 15
                else: p += 40
                if p < min_p: min_p = p; best_break = idx
            
            all_segments.append("".join(tokens[start_idx:best_break]).strip())
            start_idx = best_break
            
    return all_segments

def align_robust(user_segments, whisper_chunks):
    """
    Robust non-overlapping alignment.
    """
    user_clean = [re.sub(r'[^\w\u4e00-\u9fff]', '', s).lower() for s in user_segments]
    whisper_full_text = "".join([c["text"] for c in whisper_chunks])
    whisper_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', whisper_full_text).lower()
    
    # Create a mapping of clean whisper characters to timestamps
    char_times = []
    for c in whisper_chunks:
        txt = c["text"]
        s, e = c["timestamp"]
        c_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', txt).lower()
        if not c_clean: continue
        duration = e - s
        for i in range(len(c_clean)):
            char_times.append((s + (i / len(c_clean)) * duration, s + ((i + 1) / len(c_clean)) * duration))
            
    # Now align user_clean to whisper_clean
    user_full_clean = "".join(user_clean)
    matcher = difflib.SequenceMatcher(None, user_full_clean, whisper_clean)
    
    # results[user_clean_char_idx] = (start, end)
    mapping = [None] * len(user_full_clean)
    for u_s, w_s, length in matcher.get_matching_blocks():
        for i in range(length):
            if w_s + i < len(char_times):
                mapping[u_s + i] = char_times[w_s + i]
                
    # 3. Fill gaps in mapping using linear interpolation
    # Find all indices that have a mapping
    matched_indices = [i for i, x in enumerate(mapping) if x is not None]
    
    if not matched_indices:
        # Total failure to match anything - fallback to full linear
        total_dur = char_times[-1][1] if char_times else 10.0
        for i in range(len(mapping)):
            s = (i / len(mapping)) * total_dur
            e = ((i + 1) / len(mapping)) * total_dur
            mapping[i] = (s, e)
    else:
        # Interpolate before the first match
        first_idx = matched_indices[0]
        first_s = mapping[first_idx][0]
        for i in range(first_idx):
            mapping[i] = ((i / first_idx) * first_s, ((i + 1) / first_idx) * first_s)
            
        # Interpolate between matches
        for j in range(len(matched_indices) - 1):
            idx1, idx2 = matched_indices[j], matched_indices[j+1]
            t1, t2 = mapping[idx1][1], mapping[idx2][0]
            gap_len = idx2 - idx1 - 1
            if gap_len > 0:
                for k in range(1, gap_len + 1):
                    s = t1 + ((k-1) / gap_len) * (t2 - t1)
                    e = t1 + (k / gap_len) * (t2 - t1)
                    mapping[idx1 + k] = (s, e)
                    
        # Interpolate after the last match
        last_idx = matched_indices[-1]
        last_e = mapping[last_idx][1]
        total_end = char_times[-1][1] if char_times else last_e + 10.0
        rem_len = len(mapping) - 1 - last_idx
        if rem_len > 0:
            for k in range(1, rem_len + 1):
                s = last_e + ((k-1) / rem_len) * (total_end - last_e)
                e = last_e + (k / rem_len) * (total_end - last_e)
                mapping[last_idx + k] = (s, e)
        
    # Extract segment boundaries
    results = []
    curr = 0
    for s_clean in user_clean:
        if not s_clean:
            results.append((last_e, last_e + 1.0))
            continue
        start_t = mapping[curr][0]
        end_t = mapping[curr + len(s_clean) - 1][1]
        results.append((start_t, end_t))
        curr += len(s_clean)
        
    return results

import argparse
import glob

def run_production_pipeline(input_dir, output_dir, model_path):
    print(f"--- Starting Robust Production Alignment ---")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model once
    print(f"Loading Whisper model from: {model_path}")
    pipe = pipeline("automatic-speech-recognition", model=model_path, device="cpu")

    # Find all .wav files in the input directory
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not wav_files:
        print("No .wav files found in the input directory.")
        return

    for wav_path in wav_files:
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        txt_path = os.path.join(input_dir, f"{base_name}.txt")
        
        if not os.path.exists(txt_path):
            print(f"[Skip] No matching .txt found for {base_name}")
            continue

        print(f"\n[Processing] {base_name}...")
        
        # Load text and audio
        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
        user_segments = smart_balanced_split(full_text)
        
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000: 
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        audio_np = waveform.mean(dim=0).numpy()

        # Inference
        # Inference: Optimized for Direct TTS Mapping
        print("     Running Inference...")
        res = pipe(audio_np, return_timestamps=True, chunk_length_s=30, batch_size=1)
        whisper_chunks = res.get("chunks", [])
        
        # In this environment, pipeline already provides absolute timestamps.
        # We ensure they are clean and non-None.
        last_s = 0.0
        for c in whisper_chunks:
            s, e = c["timestamp"]
            if s is None: c["timestamp"] = (last_s, last_s + 0.5)
            elif e is None: c["timestamp"] = (s, s + 0.5)
            last_s = c["timestamp"][1]

        # Robust Alignment
        aligned = align_robust(user_segments, whisper_chunks)
        
        # Format SRT
        srt = ""
        for i, ((start, end), text) in enumerate(zip(aligned, user_segments)):
            srt += f"{i+1}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n\n"
        
        output_path = os.path.join(output_dir, f"{base_name}.srt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt)
        print(f"     ✅ Saved: {os.path.basename(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVoice Robust SRT Alignment Engine")
    parser.add_argument("--input", required=True, help="Directory containing .wav and .txt pairs")
    parser.add_argument("--output", required=True, help="Directory to save generated .srt files")
    parser.add_argument("--model", default=".", help="Path to the Whisper model folder")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    model_path = os.path.abspath(args.model)
    
    run_production_pipeline(input_path, output_path, model_path)
