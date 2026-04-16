import os
from pathlib import Path

def count_speakers(dataset_root):
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        print(f"Error: Folder '{dataset_root}' not found.")
        return

    # Dictionary to store speaker data: {speaker_id: [languages]}
    speaker_stats = {}

    # Iterate through the first level (Speaker IDs)
    for speaker_dir in dataset_path.iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            languages = []
            
            # Iterate through the second level (Languages)
            for lang_dir in speaker_dir.iterdir():
                if lang_dir.is_dir():
                    # Count if there are actually .wav files inside the language folder
                    wav_files = list(lang_dir.glob("*.wav"))
                    if wav_files:
                        languages.append(lang_dir.name)
            
            if languages:
                speaker_stats[speaker_id] = languages

    total_speakers = len(speaker_stats)
    
    print("-" * 30)
    print(f"VoIP Dataset Summary")
    print("-" * 30)
    print(f"Total Unique Speakers: {total_speakers}")
    print("-" * 30)
    
    # Optional: Print detailed breakdown
    print(f"{'Speaker ID':<12} | {'Languages Spoken'}")
    print("-" * 30)
    for sid, langs in sorted(speaker_stats.items()):
        print(f"{sid:<12} | {', '.join(langs)}")

if __name__ == "__main__":
    # Point this to your newly created segregated folder
    OUTPUT_DIR = "VoIP_Segregated_Dataset"
    count_speakers(OUTPUT_DIR)