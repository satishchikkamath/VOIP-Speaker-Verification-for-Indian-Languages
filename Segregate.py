import os
import re
import numpy as np
from scipy.io import wavfile
from pathlib import Path

# --- Configuration Paths ---
PHASE1_SAMPLES = "/home/user2/7th/VoIP Data/Final Data VoIP/Phase 1-VoIP Consortium/samples"
PHASE1_EDITED = "/home/user2/7th/VoIP Data/Final Data VoIP/Phase 1-VoIP Consortium/Edited files"
PHASE2_DIR = "/home/user2/7th/VoIP Data/Final Data VoIP/Phase-2 VoIP Consortium/Phase 2-VOIP Database Consortium"
OUTPUT_DIR = "/home/user2/VOIP/VoIP_Segregated_Dataset"

def parse_textgrid(file_path):
    """
    Parses a TextGrid file to extract intervals (xmin, xmax, text/tag).
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Regex to find interval blocks: intervals [n]: xmin = ... xmax = ... text = "..."
        pattern = re.compile(
            r'intervals \[\d+\]:\s+xmin = ([\d\.]+)\s+xmax = ([\d\.]+)\s+text = "([^"]*)"', 
            re.DOTALL
        )
        matches = pattern.findall(content)
        
        intervals = []
        for xmin, xmax, text in matches:
            if text.strip(): # Ignore empty annotations
                intervals.append({
                    'start': float(xmin),
                    'end': float(xmax),
                    'tag': text.strip()
                })
        return intervals
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def segment_and_save(wav_path, intervals, output_root):
    """
    Segments the audio based on intervals and saves to speaker/language folders.
    """
    if not intervals:
        return

    try:
        samplerate, data = wavfile.read(wav_path)
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        return

    base_name = Path(wav_path).stem

    for i, interval in enumerate(intervals):
        tag = interval['tag']
        # Nomenclature: <Lang:2><Gender:1><SpeakerID:4><Session:1>
        # Example: HNM5001A -> Lang: HN, Speaker: 5001
        if len(tag) < 7:
            continue
            
        lang = tag[:2]
        # Based on example HNM5001A, speaker ID is the 4 digits starting at index 3
        speaker_id = tag[3:7] 
        
        # Calculate indices
        start_idx = int(interval['start'] * samplerate)
        end_idx = int(interval['end'] * samplerate)
        
        # Slice audio
        segment = data[start_idx:end_idx]
        
        # Create folder structure: Root/SpeakerID/Language/
        target_dir = Path(output_root) / speaker_id / lang
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save segment
        output_filename = f"{base_name}_seg{i+1}.wav"
        output_path = target_dir / output_filename
        wavfile.write(output_path, samplerate, segment)

def process_phase_1():
    print("Processing Phase 1...")
    wav_files = list(Path(PHASE1_SAMPLES).glob("*.wav"))
    
    for wav_path in wav_files:
        file_id = wav_path.stem
        # Check Edited files first, then Samples
        edited_tg = Path(PHASE1_EDITED) / f"{file_id}.TextGrid"
        sample_tg = Path(PHASE1_SAMPLES) / f"{file_id}.TextGrid"
        
        tg_path = edited_tg if edited_tg.exists() else sample_tg
        
        if tg_path.exists():
            intervals = parse_textgrid(tg_path)
            segment_and_save(wav_path, intervals, OUTPUT_DIR)
        else:
            print(f"Warning: No TextGrid found for {file_id}")

def process_phase_2():
    print("Processing Phase 2...")
    wav_files = list(Path(PHASE2_DIR).glob("*.wav"))
    
    for wav_path in wav_files:
        tg_path = wav_path.with_suffix(".TextGrid")
        if tg_path.exists():
            intervals = parse_textgrid(tg_path)
            segment_and_save(wav_path, intervals, OUTPUT_DIR)
        else:
            print(f"Warning: No TextGrid found for {wav_path.name}")

if __name__ == "__main__":
    # Create main output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    process_phase_1()
    process_phase_2()
    print(f"Done! Segregated dataset available in: {OUTPUT_DIR}")