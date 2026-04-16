# ecapa_improved_v2.py
import os
import glob
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ----------------- USER CONFIG -----------------
PRETRAINED_CHECKPOINT = "/home/user2/githubstruct/models/ECAPA/voxceleb.pt" 
DATASET_ROOT = "/home/user2/VOIP/VOIP_Mel_Features"
FINETUNE_CHECKPOINT_DIR = "/home/user2/VOIP/finetune_checkpoints_ecapa_voip_v2"
LOG_DIR = "/home/user2/VOIP/finetune_logs_ecapa_voip_v2"

# Training hyperparams
BATCH_SIZE = 64
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_WORKERS = 8
FREEZE_EPOCHS = 0

# --- IMPROVEMENT 1: Larger Context & Margins ---
TRAIN_CHUNK_FRAMES = 300  # Increased from 200 to 300 (3s context). If OOM, set back to 200.
AAM_MARGIN = 0.35         # Increased from 0.2 to 0.35 (Harder task = better separation)
AAM_SCALE = 35.0          # Increased slightly to help convergence

LR_MODEL = 1e-4 
LR_HEAD = 1e-4
WEIGHT_DECAY = 2e-5

os.makedirs(FINETUNE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------- logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "finetune.log")),
        logging.StreamHandler()
    ]
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------- SpecAugment (NEW) -----------------
def apply_spec_augment(mel_spec, freq_mask_param=15, time_mask_param=20):
    """
    Applies Frequency and Time masking to the Mel Spectrogram.
    Input: (Time, Freq) numpy array
    """
    T, F = mel_spec.shape
    
    # 1. Frequency Masking
    # Randomly mask 'f' consecutive frequency bands
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, F - f)
    mel_spec[:, f0:f0+f] = 0.0

    # 2. Time Masking
    # Randomly mask 't' consecutive time steps
    t = np.random.randint(0, time_mask_param)
    if T - t > 0:
        t0 = np.random.randint(0, T - t)
        mel_spec[t0:t0+t, :] = 0.0
    
    return mel_spec

# ----------------- Model Definition (Standard) -----------------
class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])
    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))

class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)
    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out

def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)
    def forward(self, x):
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, embd_dim=192):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, 128)
        self.bn1 = nn.BatchNorm1d(3072)
        
        self.linear = nn.Linear(3072, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        x = x.transpose(1, 2) 
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        
        out = self.linear(out)
        out = self.bn2(out)
        return out

class AAMSoftmax(nn.Module):
    def __init__(self, n_class, in_features, m, s):
        super().__init__()
        self.in_features = in_features
        self.n_class = n_class
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, in_features))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        self.th = np.cos(np.pi - self.m)
        self.mm = np.sin(np.pi - self.m) * self.m
    def forward(self, x, label):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-8)
        cosine = F.linear(x_norm, w_norm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = self.ce_loss(output, label)
        return loss

# ----------------- DATA CLEANING -----------------
def analyze_file_quality(npy_path):
    try:
        feat = np.load(npy_path, allow_pickle=False)
        feat = feat.T  # (time, features)
        total_values = feat.size
        nan_count = np.isnan(feat).sum()
        inf_count = np.isinf(feat).sum()
        bad_count = nan_count + inf_count
        bad_percentage = (bad_count / total_values) * 100
        quality_score = max(0, 100 - bad_percentage)
        return quality_score, bad_percentage, nan_count, inf_count
    except Exception as e:
        return 0, 100, 0, 0

def aggressive_feature_cleaning(npy_path, quality_threshold=50):
    quality_score, bad_percentage, nan_count, inf_count = analyze_file_quality(npy_path)
    if quality_score < quality_threshold:
        return None
    try:
        feat = np.load(npy_path, allow_pickle=False).astype(np.float32)
        feat = feat.T
        if not np.isfinite(feat).all():
            clean_mask = np.isfinite(feat)
            if clean_mask.any():
                clean_values = feat[clean_mask]
                global_mean = np.mean(clean_values)
                global_std = np.std(clean_values)
                bad_mask = ~np.isfinite(feat)
                feat[bad_mask] = np.random.normal(global_mean, global_std, size=bad_mask.sum())
            else:
                feat = np.random.normal(0, 0.1, size=feat.shape)
        
        feat = np.clip(feat, -10, 10)
        if np.std(feat) < 1e-6:
            feat = feat + np.random.normal(0, 1e-4, size=feat.shape)
        return feat
    except Exception as e:
        return None

# ----------------- DATA PREP -----------------
def create_clean_dataset_list(dir_path, suffix=".npy", quality_threshold=70):
    clean_list = []
    speaker_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    
    for sp in speaker_dirs:
        sp_path = os.path.join(dir_path, sp)
        npy_files = glob.glob(os.path.join(sp_path, "**", f"*{suffix}"), recursive=True)
        for npy_path in npy_files:
            quality_score, _, _, _ = analyze_file_quality(npy_path)
            if quality_score >= quality_threshold:
                clean_list.append((npy_path, sp))
    
    logging.info(f"Loaded {len(clean_list)} files from {dir_path}")
    return clean_list

# ----------------- DATA LOADERS -----------------
def train_collate_fn(batch, num_frames=TRAIN_CHUNK_FRAMES):
    features, labels = [], []
    
    for npy_path, label in batch:
        feat = aggressive_feature_cleaning(npy_path, quality_threshold=70)
        if feat is None: continue
            
        # 1. Pad or Crop FIRST
        if feat.shape[0] < num_frames:
            pad_len = num_frames - feat.shape[0]
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant')
        elif feat.shape[0] > num_frames:
            start = random.randint(0, feat.shape[0] - num_frames)
            feat = feat[start:start + num_frames, :]
            
        # 2. Apply SpecAugment (The Fix)
        # We perform augmentation on the cropped segment
        feat = apply_spec_augment(feat, freq_mask_param=15, time_mask_param=20)

        features.append(feat)
        labels.append(label)
    
    if not features: return None, None
    return torch.FloatTensor(np.array(features)), torch.LongTensor(labels)

def eval_collate_fn(batch):
    features, labels, filepaths = [], [], []
    max_len = 0
    for npy_path, label in batch:
        feat = aggressive_feature_cleaning(npy_path, quality_threshold=70)
        if feat is None: continue
        if feat.shape[0] > max_len: max_len = feat.shape[0]
        features.append(feat)
        labels.append(label)
        filepaths.append(npy_path)
    
    if not features: return None, None, None
    
    padded_features = []
    for feat in features:
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant')
        padded_features.append(feat)
    
    return torch.FloatTensor(np.array(padded_features)), torch.LongTensor(labels), filepaths

class SimpleListDataset(Dataset):
    def __init__(self, items):
        self.items = items
        speaker_set = sorted(list({s for _, s in items}))
        self.spk2label = {s: i for i, s in enumerate(speaker_set)}
        self.data = [(p, self.spk2label[s]) for p, s in items]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    def get_speaker_labels(self): return self.spk2label

# ----------------- MAIN LOGIC -----------------
train_dir = os.path.join(DATASET_ROOT, "train")
val_dir = os.path.join(DATASET_ROOT, "test")

train_samples = create_clean_dataset_list(train_dir, suffix=".npy", quality_threshold=70)
val_samples = create_clean_dataset_list(val_dir, suffix=".npy", quality_threshold=70)

if not train_samples:
    logging.error("No training samples found. Exiting.")
    exit(1)

train_dataset = SimpleListDataset(train_samples)
val_dataset = SimpleListDataset(val_samples)
num_classes = len(train_dataset.get_speaker_labels())
logging.info(f"Target speakers: {num_classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=train_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        collate_fn=eval_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

# ----------------- MODEL INIT -----------------
model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192).to(DEVICE)

if os.path.exists(PRETRAINED_CHECKPOINT):
    logging.info(f"Loading pretrained: {PRETRAINED_CHECKPOINT}")
    ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
else:
    logging.warning("Pretrained checkpoint not found! Training from scratch.")

loss_fn = AAMSoftmax(n_class=num_classes, in_features=192, m=AAM_MARGIN, s=AAM_SCALE).to(DEVICE)

optimizer = optim.AdamW([
    {'params': [p for p in model.parameters() if p.requires_grad], 'lr': LR_MODEL, 'weight_decay': WEIGHT_DECAY},
    {'params': [loss_fn.weight], 'lr': LR_HEAD, 'weight_decay': WEIGHT_DECAY}
])

# --- IMPROVEMENT 2: Cosine Annealing ---
# This helps the model settle into a better minimum than StepLR
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ----------------- HELPER FUNCTIONS -----------------
def train_one_epoch(epoch):
    model.train()
    total_loss, n_batches = 0.0, 0
    
    for features, labels in train_loader:
        if features is None or features.size(0) <= 1: continue
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        emb = model(features)
        loss = loss_fn(emb, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
    avg_loss = total_loss / (n_batches if n_batches > 0 else 1)
    logging.info(f"Epoch {epoch} train loss: {avg_loss:.4f}")
    return avg_loss

def extract_embeddings_for_loader(loader):
    model.eval()
    embeddings, labels_map = {}, {}
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            feats, labs, filepaths = batch
            feats = feats.to(DEVICE)
            emb = model(feats).cpu()
            for i, p in enumerate(filepaths):
                embeddings[p] = emb[i]
                labels_map[p] = labs[i].item()
    return embeddings, labels_map

def validate_and_compute_eer_minDCF():
    embeddings, labels_map = extract_embeddings_for_loader(val_loader)
    if len(embeddings) < 2: return None, None
    
    all_files = list(embeddings.keys())
    scores, y_true = [], []
    # Sample pairs for EER calculation
    num_pairs = min(5000, len(all_files) * 5)
    
    for _ in range(num_pairs):
        is_target = random.choice([True, False])
        f1 = random.choice(all_files)
        target_label = labels_map[f1]
        
        if is_target:
            candidates = [f for f in all_files if labels_map[f] == target_label and f != f1]
        else:
            candidates = [f for f in all_files if labels_map[f] != target_label]
            
        if not candidates: continue
        f2 = random.choice(candidates)
        
        score = F.cosine_similarity(embeddings[f1].unsqueeze(0), embeddings[f2].unsqueeze(0)).item()
        scores.append(score)
        y_true.append(1 if is_target else 0)
        
    try:
        from sklearn.metrics import roc_curve
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        
        fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100
        
        fnr = 1 - tpr
        min_dcf = np.min(0.01 * 1 * fnr + 0.99 * 1 * fpr)
        return eer, min_dcf
    except:
        return None, None

# ----------------- TRAINING LOOP -----------------
logging.info("=== Starting Improved ECAPA Training (SpecAugment + CosineAnnealing) ===")
best_val_eer = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    eer, min_dcf = validate_and_compute_eer_minDCF()
    
    if eer is not None:
        logging.info(f"Epoch {epoch} -> EER: {eer:.2f}%, minDCF: {min_dcf:.4f}")
        
        if eer < best_val_eer:
            best_val_eer = eer
            torch.save(model.state_dict(), os.path.join(FINETUNE_CHECKPOINT_DIR, f"best_model_eer_{eer:.2f}.pt"))
            logging.info("New best model saved!")
            
    if scheduler is not None:
        scheduler.step()

logging.info(f"Finished. Best EER: {best_val_eer:.2f}%")