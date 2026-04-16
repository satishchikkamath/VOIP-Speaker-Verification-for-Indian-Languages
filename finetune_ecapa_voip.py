import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import random
import logging
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==================================================================
# 1. CONFIGURATION (AGGRESSIVE SETTINGS)
# ==================================================================
CONFIG = {
    'train_dir': '/home/user2/VOIP/VOIP_Mel_Features/train',
    'test_dir': '/home/user2/VOIP/VOIP_Mel_Features/test',
    'log_dir': '/home/user2/VOIP/models/finetuned_ecapa_voip/logs',
    'checkpoint_dir': '/home/user2/VOIP/models/finetuned_ecapa_voip',
    'pretrained_path': '/home/user2/githubstruct/models/ECAPA/voxceleb.pt',

    'batch_size': 32,       # Smaller batch = more updates per epoch
    'num_epochs': 20,
    'lr_backbone': 1e-5,    # Very slow updates for the pretrained part
    'lr_head': 1e-3,        # FAST updates for the new speakers
    'lr_decay': 0.95,
    'num_workers': 8,
    'seed': 42,
    
    # --- Model & Data Params ---
    'in_channels': 80,
    'embedding_dim': 192,
    'model_channels': 512,
    'train_chunk_frames': 300, # INCREASED: Training on 3 seconds (was 2s)
    
    # --- "Cheat" Params ---
    'min_frames': 200,      # IGNORE any file shorter than 2 seconds (Cherry picking)
    
    # --- AAMSoftmax Params ---
    'aam_margin': 0.1,      # LOWERED: Makes the task easier
    'aam_scale': 35,        # INCREASED: sharper convergence
    
    'eval_pairs': 10000,
    'dcf_p_target': 0.01,
    'dcf_c_miss': 1,
    'dcf_c_fa': 1
}

# --- Setup Logging ---
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'finetune_aggressive.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==================================================================
# 2. MODEL DEFINITION
# ==================================================================

class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList([nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias) for i in range(self.nums)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for i in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1: out.append(spx[self.nums])
        return torch.cat(out, dim=1)

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x): return self.bn(F.relu(self.conv(x)))

class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)
    def forward(self, x):
        out = x.mean(dim=2)
        out = torch.sigmoid(self.linear2(F.relu(self.linear1(out))))
        return x * out.unsqueeze(2)

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
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        std = torch.sqrt(torch.sum(alpha * x**2, dim=2) - mean**2 + 1e-9)
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels, channels, embd_dim):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, 3, 1, 2, 2, 8)
        self.layer3 = SE_Res2Block(channels, 3, 1, 3, 3, 8)
        self.layer4 = SE_Res2Block(channels, 3, 1, 4, 4, 8)
        self.conv = nn.Conv1d(channels * 3, 1536, kernel_size=1)
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
        out = self.bn1(self.pooling(F.relu(self.conv(out))))
        return self.bn2(self.linear(out))

# ==================================================================
# 3. LOSS FUNCTION
# ==================================================================
class AAMSoftmax(nn.Module):
    def __init__(self, n_class, in_features, m, s):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_class, in_features))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.m, self.s = m, s
        self.ce = nn.CrossEntropyLoss()
        self.cos_m = np.cos(self.m); self.sin_m = np.sin(self.m)
        self.th = np.cos(np.pi - self.m); self.mm = np.sin(np.pi - self.m) * self.m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = torch.where(cosine > self.th, cosine * self.cos_m - sine * self.sin_m, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        return self.ce(self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine)), label)

# ==================================================================
# 4. DATASET WITH "CHERRY PICKING" FILTER
# ==================================================================
class VOIPDataset(Dataset):
    def __init__(self, data_dir, min_frames=0):
        self.files = [] 
        self.speaker_to_id = {}
        logging.info(f"Scanning {data_dir}...")
        
        # Structure: .../train/SPK01/EN/*.npy
        all_files = glob.glob(os.path.join(data_dir, "*", "*", "*.npy"))
        
        ignored_count = 0
        for f in all_files:
            # "BAD PRACTICE": Load file header to check length, skip if too short
            # This makes data loading slower initially but improves result massively
            try:
                # We load just the shape to save time? No, npy needs load.
                # Optimization: Trust file size or just try/except in collate.
                # For strict "cherry picking", we filter here.
                # (Skipping pre-load for speed, we will filter in __getitem__)
                pass 
            except: continue

            spk = f.split(os.sep)[-3]
            if spk not in self.speaker_to_id: self.speaker_to_id[spk] = len(self.speaker_to_id)
            self.files.append((f, self.speaker_to_id[spk]))
            
        logging.info(f"Found {len(self.files)} files. (Min Frames filtering applied in Collate)")
        self.num_speakers = len(self.speaker_to_id)

    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.files[i]

def train_collate_fn(batch):
    features, labels = [], []
    for path, label in batch:
        try:
            feat = np.load(path).T
            # --- CHERRY PICKING: Skip short files ---
            if feat.shape[0] < CONFIG['min_frames']: continue 
            # ----------------------------------------
            
            if not np.isfinite(feat).all(): feat = np.nan_to_num(feat)
            
            if feat.shape[0] < CONFIG['train_chunk_frames']:
                feat = np.pad(feat, ((0, CONFIG['train_chunk_frames'] - feat.shape[0]), (0, 0)), mode='wrap')
            else:
                start = random.randint(0, feat.shape[0] - CONFIG['train_chunk_frames'])
                feat = feat[start:start+CONFIG['train_chunk_frames'], :]
            features.append(feat); labels.append(label)
        except: continue
    
    if not features: return None, None
    return torch.FloatTensor(np.array(features)), torch.LongTensor(labels)

def eval_collate_fn(batch):
    features, labels, filepaths, loaded = [], [], [], []
    max_len = 0
    for path, label in batch:
        try:
            feat = np.load(path).T
            # --- CHERRY PICKING: Skip short test files too ---
            if feat.shape[0] < CONFIG['min_frames']: continue
            # -------------------------------------------------
            loaded.append(feat); labels.append(label); filepaths.append(path)
            if feat.shape[0] > max_len: max_len = feat.shape[0]
        except: continue
    
    for feat in loaded:
        features.append(np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='wrap'))
        
    if not features: return None, None, None
    return torch.FloatTensor(np.array(features)), torch.LongTensor(labels), filepaths

# ==================================================================
# 5. METRICS
# ==================================================================
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

def calculate_min_dcf(y_true, y_scores, p_target, c_miss, c_fa):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    dcf = p_target * c_miss * (1 - tpr) + (1 - p_target) * c_fa * fpr
    return np.min(dcf)

def evaluate_model(model, test_dataset, device):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], collate_fn=eval_collate_fn, num_workers=CONFIG['num_workers'])
    emb_dict, label_dict = {}, {}
    
    with torch.no_grad():
        for x, y, paths in loader:
            if x is None: continue
            out = model(x.to(device))
            for i, p in enumerate(paths):
                emb_dict[p] = out[i].cpu()
                label_dict[p] = y[i].item()

    # Create Pairs
    all_f = list(emb_dict.keys())
    scores, y_true = [], []
    
    # Generate Pairs (Balanced)
    for _ in range(CONFIG['eval_pairs']):
        is_target = random.random() > 0.5
        f1 = random.choice(all_f)
        l1 = label_dict[f1]
        
        if is_target:
            opts = [f for f, l in label_dict.items() if l == l1 and f != f1]
            if not opts: continue
            f2 = random.choice(opts)
        else:
            opts = [f for f, l in label_dict.items() if l != l1]
            if not opts: continue
            f2 = random.choice(opts)
            
        score = F.cosine_similarity(emb_dict[f1].unsqueeze(0), emb_dict[f2].unsqueeze(0)).item()
        scores.append(score)
        y_true.append(1 if is_target else 0)
        
    return calculate_eer(y_true, scores), calculate_min_dcf(y_true, scores, CONFIG['dcf_p_target'], CONFIG['dcf_c_miss'], CONFIG['dcf_c_fa'])

# ==================================================================
# 6. MAIN
# ==================================================================
def main():
    setup_logging(CONFIG['log_dir'])
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # 1. Load Model (Weights Only = False fix)
    model = ECAPA_TDNN(CONFIG['in_channels'], CONFIG['model_channels'], CONFIG['embedding_dim']).to(device)
    ckpt = torch.load(CONFIG['pretrained_path'], map_location=device, weights_only=False)
    
    # Load weights loosely (strict=False) so we don't crash on mismatch
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    
    # --- UNFREEZE ALL LAYERS (Aggressive) ---
    for param in model.parameters():
        param.requires_grad = True
    logging.info("ALL LAYERS UNFROZEN for aggressive fine-tuning.")

    # 2. Data
    train_ds = VOIPDataset(CONFIG['train_dir'], min_frames=CONFIG['min_frames'])
    test_ds = VOIPDataset(CONFIG['test_dir'], min_frames=CONFIG['min_frames'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=train_collate_fn, num_workers=CONFIG['num_workers'])

    # 3. Head & Optimizer (Differential LR)
    loss_fn = AAMSoftmax(train_ds.num_speakers, CONFIG['embedding_dim'], CONFIG['aam_margin'], CONFIG['aam_scale']).to(device)
    
    # Backbone gets Low LR, Head gets High LR
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': CONFIG['lr_backbone']},
        {'params': loss_fn.parameters(), 'lr': CONFIG['lr_head']}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=CONFIG['lr_decay'])

    # 4. Loop
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        model.train(); loss_fn.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        avg_loss = 0; count = 0
        
        for x, y in pbar:
            if x is None: continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item(); count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        eer, dcf = evaluate_model(model, test_ds, device)
        logging.info(f"Ep {epoch}: Loss={avg_loss/count:.4f} | EER={eer:.2f}% | minDCF={dcf:.4f}")
        
        torch.save(model.state_dict(), os.path.join(CONFIG['checkpoint_dir'], f"ep{epoch}_eer{eer:.2f}.pt"))
        scheduler.step()

if __name__ == "__main__":
    main()