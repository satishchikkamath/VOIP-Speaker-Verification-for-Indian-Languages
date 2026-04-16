import os
import random
import shutil
from pathlib import Path

# Configuration
root_dir = Path("/home/user2/VOIP/")
dataset_path = root_dir / "VoIP_Segregated_Dataset"
output_path = root_dir / "VOIP_Split"
train_ratio = 0.8
seed = 42

def create_split():
    # 1. Setup output directories
    for folder in ['train', 'test']:
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    # 2. Identify all speakers
    speakers = [f.name for f in dataset_path.iterdir() if f.is_dir()]
    
    # 3. Shuffle and split speaker IDs
    random.seed(seed)
    random.shuffle(speakers)
    
    split_idx = int(len(speakers) * train_ratio)
    train_speakers = speakers[:split_idx]
    test_speakers = speakers[split_idx:]

    def link_speaker_data(speaker_list, split_name):
        print(f"Processing {split_name} set...")
        for speaker in speaker_list:
            src_speaker_dir = dataset_path / speaker
            dst_speaker_dir = output_path / split_name / speaker
            
            # Create the speaker folder in the new location
            dst_speaker_dir.mkdir(parents=True, exist_ok=True)
            
            # Recursively find all wav files (maintaining language subfolders)
            for wav_path in src_speaker_dir.rglob("*.wav"):
                # Determine relative path from speaker root (e.g., 'EN/file1.wav')
                rel_path = wav_path.relative_to(src_speaker_dir)
                target_path = dst_speaker_dir / rel_path
                
                # Ensure subfolders like EN/HN/BN exist in the new location
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create a symbolic link
                if not target_path.exists():
                    os.symlink(wav_path, target_path)

    # 4. Execute the linking
    link_speaker_data(train_speakers, "train")
    link_speaker_data(test_speakers, "test")

    print(f"\nDone! Split saved to: {output_path}")
    print(f"Train speakers: {len(train_speakers)}")
    print(f"Test speakers: {len(test_speakers)}")

if __name__ == "__main__":
    create_split()