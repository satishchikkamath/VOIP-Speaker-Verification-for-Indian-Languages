import os
import glob
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
SOURCE_ROOT = '/home/user2/VOIP/VOIP_Split/'
TARGET_ROOT = '/home/user2/7th/dataset/VOIP_Mel_Features/' # New target for Mel features

SAMPLE_RATE = 16000
N_MELS = 80
CROP_SECONDS = 3
BATCH_SIZE = 256  # Reduced batch size for stability
NUM_WORKERS = multiprocessing.cpu_count()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class MelProcessor:
    def __init__(self):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=int(0.025 * SAMPLE_RATE),
            win_length=int(0.025 * SAMPLE_RATE),
            hop_length=int(0.01 * SAMPLE_RATE),
            n_mels=N_MELS,
            power=2
        ).to(device)

    def compute_log_mel(self, audio):
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mel = self.mel_transform(audio_tensor)
            log_mel = torchaudio.functional.amplitude_to_DB(
                mel, multiplier=10, amin=1e-10, db_multiplier=0
            )
            # Normalization: Mean Subtraction
            log_mel -= log_mel.mean(dim=-1, keepdim=True)
        return log_mel.squeeze(0).cpu().numpy()

class VOIPDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            y, sr = sf.read(path, always_2d=True)
            y = np.mean(y, axis=1) # Mono
            
            if sr != SAMPLE_RATE:
                y = torchaudio.functional.resample(torch.tensor(y), sr, SAMPLE_RATE).numpy()
            
            # Random Crop or Pad
            target_len = CROP_SECONDS * SAMPLE_RATE
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                start = np.random.randint(0, len(y) - target_len)
                y = y[start:start + target_len]

            # Maintain structure: Get path relative to SOURCE_ROOT
            rel_path = os.path.relpath(path, SOURCE_ROOT)
            # target_path will be TARGET_ROOT + relative structure + filename.npy
            target_path = os.path.join(TARGET_ROOT, os.path.splitext(rel_path)[0] + ".npy")
            
            return {'audio': y, 'target_path': target_path, 'valid': True}
        except Exception as e:
            return {'valid': False}

def collate_fn(batch):
    batch = [b for b in batch if b['valid']]
    if not batch: return None
    return {
        'audio': np.stack([b['audio'] for b in batch]),
        'target_paths': [b['target_path'] for b in batch]
    }

def main():
    print(f"Scanning source: {SOURCE_ROOT}")
    # Finds all .wav files in any subfolder depth (train/test -> ID -> Lang -> wav)
    all_wavs = glob.glob(os.path.join(SOURCE_ROOT, "**/*.wav"), recursive=True)
    print(f"Found {len(all_wavs)} wav files.")

    if not all_wavs:
        print("No files found. Check your SOURCE_ROOT path.")
        return

    dataset = VOIPDataset(all_wavs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    processor = MelProcessor()
    
    print(f"Processing features to: {TARGET_ROOT}")
    for batch in tqdm(loader, desc="Extracting Mel Features"):
        if batch is None: continue
        
        audios = batch['audio']
        target_paths = batch['target_paths']
        
        for i in range(len(audios)):
            mel = processor.compute_log_mel(audios[i])
            
            # Ensure the specific nested subdirectory exists in the target
            ensure_dir(os.path.dirname(target_paths[i]))
            
            # Save as clean mel
            np.save(target_paths[i], mel)

    print("\n? Extraction Complete!")
    print(f"Structure mirrored at: {TARGET_ROOT}")

if __name__ == "__main__":
    main()