# quanta.py
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
import pennylane as qml

# ----------------- USER CONFIG -----------------
PRETRAINED_CHECKPOINT = "/home/user2/githubstruct/models/QECAPA/voxceleb.pt"
DATASET_ROOT = "/home/user2/VOIP/VOIP_Mel_Features"
FINETUNE_CHECKPOINT_DIR = "/home/user2/VOIP/finetune_checkpoints_qecapa_voip_atharva"
LOG_DIR = "/home/user2/VOIP/finetune_logs_qecapa_voip_atharva"

# Quantum configuration
NUM_QUBITS = 6
DEV = qml.device("lightning.qubit", wires=NUM_QUBITS)

# training hyperparams
BATCH_SIZE = 64
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_WORKERS = 8
FREEZE_EPOCHS = 5
LR_MODEL = 5e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 2e-5

# AAM params
AAM_MARGIN = 0.2
AAM_SCALE = 30.0
TRAIN_CHUNK_FRAMES = 200

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

# ----------------- Quantum Circuit -----------------
@qml.qnode(DEV, interface="torch")
def quantum_circuit(inputs, weights):
    """Quantum circuit for feature processing"""
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

# ----------------- Quantum-Enhanced Model Definition -----------------
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

class QuantumEmbeddingProcessor(nn.Module):
    def __init__(self, in_features_full, downsample_dim, embd_dim, num_qubits, quantum_circuit):
        super().__init__()
        self.downsample_linear = nn.Linear(in_features_full, downsample_dim)
        self.downsample_bn = nn.BatchNorm1d(downsample_dim)
        
        # Quantum layer
        weight_shapes = {"weights": (3, num_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Hybrid combination
        hybrid_dim = num_qubits + downsample_dim
        self.final_linear = nn.Linear(hybrid_dim, embd_dim)
        self.final_bn = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        downsampled = self.downsample_bn(F.relu(self.downsample_linear(x)))
        quantum_output = self.qlayer(downsampled)
        
        # --- FIX: Move quantum output to GPU if needed ---
        if quantum_output.device != downsampled.device:
            quantum_output = quantum_output.to(downsampled.device)
            
        hybrid_features = torch.cat([downsampled, quantum_output], dim=1)
        out = self.final_linear(hybrid_features)
        out = self.final_bn(out)
        return out

class QECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, embd_dim=192, num_qubits=NUM_QUBITS, quantum_circuit=quantum_circuit):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, 128)
        self.bn1 = nn.BatchNorm1d(3072)
        
        # Quantum enhancement
        IN_FEAT_FULL = 3072 
        DOWNSAMPLE_DIM = num_qubits 
        
        self.quantum_processor = QuantumEmbeddingProcessor(
            in_features_full=IN_FEAT_FULL,
            downsample_dim=DOWNSAMPLE_DIM,
            embd_dim=embd_dim,
            num_qubits=num_qubits,
            quantum_circuit=quantum_circuit
        )

    def forward(self, x):
        x = x.transpose(1, 2) 
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        
        out = self.quantum_processor(out)
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

# ----------------- AGGRESSIVE DATA CLEANING -----------------
def analyze_file_quality(npy_path):
    """Analyze file quality and return quality score (0-100)"""
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
    """Aggressive cleaning with quality threshold"""
    quality_score, bad_percentage, nan_count, inf_count = analyze_file_quality(npy_path)
    
    if quality_score < quality_threshold:
        logging.warning(f"SKIPPING severely corrupted file: {npy_path} "
                       f"(quality: {quality_score:.1f}%, bad: {bad_percentage:.1f}%, "
                       f"NaN: {nan_count}, Inf: {inf_count})")
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
        logging.error(f"Failed to clean {npy_path}: {e}")
        return None

# ----------------- PRE-FILTER DATASET -----------------
def create_clean_dataset_list(dir_path, suffix=".npy", quality_threshold=70):
    """
    Create dataset list with pre-filtering.
    Now handles nested structure: SpeakerID -> Language -> .npy
    """
    clean_list = []
    corrupted_count = 0
    total_count = 0
    
    # Identify speaker directories (folders directly inside train/test)
    speaker_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    
    for sp in speaker_dirs:
        sp_path = os.path.join(dir_path, sp)
        # Using ** ensures we check inside Language subfolders (EN, HN, BN)
        npy_files = glob.glob(os.path.join(sp_path, "**", f"*{suffix}"), recursive=True)
        
        for npy_path in npy_files:
            total_count += 1
            quality_score, bad_percentage, _, _ = analyze_file_quality(npy_path)
            
            if quality_score >= quality_threshold:
                clean_list.append((npy_path, sp))
            else:
                corrupted_count += 1
                if corrupted_count <= 10:
                    logging.warning(f"Excluding corrupted: {npy_path} (quality: {quality_score:.1f}%)")
    
    logging.info(f"Dataset filtering: {len(clean_list)}/{total_count} files passed "
                f"quality threshold {quality_threshold}% ({corrupted_count} excluded)")
    
    return clean_list

# ----------------- Dataset and DataLoader -----------------
def train_collate_fn(batch, num_frames=TRAIN_CHUNK_FRAMES):
    features, labels = [], []
    skipped_count = 0
    
    for npy_path, label in batch:
        feat = aggressive_feature_cleaning(npy_path, quality_threshold=70)
        
        if feat is None:
            skipped_count += 1
            continue
            
        if feat.shape[0] < num_frames:
            pad_len = num_frames - feat.shape[0]
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='wrap')
        elif feat.shape[0] > num_frames:
            start = random.randint(0, feat.shape[0] - num_frames)
            feat = feat[start:start + num_frames, :]

        features.append(feat)
        labels.append(label)
    
    if skipped_count > 0:
        logging.debug(f"Skipped {skipped_count} files in batch due to quality issues")
    
    if not features:
        return None, None

    features_tensor = torch.FloatTensor(np.array(features))
    labels_tensor = torch.LongTensor(labels)
    
    return features_tensor, labels_tensor

def eval_collate_fn(batch):
    features, labels, filepaths = [], [], []
    max_len = 0
    skipped_count = 0
    
    for npy_path, label in batch:
        feat = aggressive_feature_cleaning(npy_path, quality_threshold=70)
        
        if feat is None:
            skipped_count += 1
            continue
            
        if feat.shape[0] > max_len:
            max_len = feat.shape[0]
            
        features.append(feat)
        labels.append(label)
        filepaths.append(npy_path)
    
    if skipped_count > 0:
        logging.debug(f"Skipped {skipped_count} eval files due to quality issues")
    
    if not features:
        return None, None, None

    padded_features = []
    for feat in features:
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='wrap')
        padded_features.append(feat)
    
    features_tensor = torch.FloatTensor(np.array(padded_features))
    return features_tensor, torch.LongTensor(labels), filepaths

# ----------------- Data Preparation -----------------
class SimpleListDataset(Dataset):
    def __init__(self, items):
        self.items = items
        speaker_set = sorted(list({s for _, s in items}))
        self.spk2label = {s: i for i, s in enumerate(speaker_set)}
        self.data = [(p, self.spk2label[s]) for p, s in items]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    def get_speaker_labels(self): return self.spk2label

# Load data with pre-filtering
train_dir = os.path.join(DATASET_ROOT, "train")
val_dir = os.path.join(DATASET_ROOT, "test")

logging.info("Pre-filtering training dataset...")
train_samples = create_clean_dataset_list(train_dir, suffix=".npy", quality_threshold=70)

logging.info("Pre-filtering validation dataset...")
val_samples = create_clean_dataset_list(val_dir, suffix=".npy", quality_threshold=70)

if not train_samples:
    logging.error("No training samples passed quality threshold! Lowering threshold to 50%")
    train_samples = create_clean_dataset_list(train_dir, suffix=".npy", quality_threshold=50)

if not val_samples:
    logging.error("No validation samples passed quality threshold! Lowering threshold to 50%")
    val_samples = create_clean_dataset_list(val_dir, suffix=".npy", quality_threshold=50)

train_dataset = SimpleListDataset(train_samples)
val_dataset = SimpleListDataset(val_samples)

logging.info(f"Final train dataset: {len(train_dataset)} samples from {len(train_dataset.get_speaker_labels())} speakers")
logging.info(f"Final val dataset: {len(val_dataset)} samples from {len(val_dataset.get_speaker_labels())} speakers")

if len(train_dataset) == 0:
    logging.error("No training data available after filtering! Check your data quality.")
    exit(1)

num_classes = len(train_dataset.get_speaker_labels())
logging.info(f"Number of target speakers: {num_classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=train_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        collate_fn=eval_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

# ----------------- Model Setup -----------------
model = QECAPA_TDNN(in_channels=80, channels=512, embd_dim=192).to(DEVICE)

# Load pretrained quantum-enhanced weights
if os.path.exists(PRETRAINED_CHECKPOINT):
    logging.info(f"Loading quantum-enhanced checkpoint: {PRETRAINED_CHECKPOINT}")
    # --- FIX 1: weights_only=False ---
    ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    # Load with strict=False to handle potential architecture differences
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info("Loaded pretrained QECAPA-TDNN model")
    logging.info(f"Missing keys: {len(missing)}")
    logging.info(f"Unexpected keys: {len(unexpected)}")
    
    # Log important missing keys (like quantum layer)
    if missing:
        logging.info("Missing keys (may include new quantum processor):")
        for key in missing[:10]:  # Show first 10
            logging.info(f"  - {key}")
else:
    logging.warning(f"Pretrained checkpoint not found at {PRETRAINED_CHECKPOINT}. Starting from scratch.")

# Create new AAMSoftmax for current number of target classes
loss_fn = AAMSoftmax(n_class=num_classes, in_features=192, m=AAM_MARGIN, s=AAM_SCALE).to(DEVICE)

# Optimizer - separate learning rates for model and classification head
model_params = [p for p in model.parameters() if p.requires_grad]
head_params = [loss_fn.weight]

optimizer = optim.AdamW([
    {'params': model_params, 'lr': LR_MODEL, 'weight_decay': WEIGHT_DECAY},
    {'params': head_params, 'lr': LR_HEAD, 'weight_decay': WEIGHT_DECAY}
])

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

# ----------------- Freeze/BN helper functions -----------------
def set_model_requires_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad

def freeze_bn_stats(model):
    """Sets all Batch Norm layers to eval() mode to use pre-trained statistics."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

# Initially freeze feature extractor for FREEZE_EPOCHS
if FREEZE_EPOCHS > 0:
    logging.info(f"Freezing model parameters for first {FREEZE_EPOCHS} epochs (only head will be trained).")
    set_model_requires_grad(model, False)
    for p in head_params:
        p.requires_grad = True
else:
    freeze_bn_stats(model)

# ----------------- Training loop -----------------
def train_one_epoch(epoch):
    model.train()
    freeze_bn_stats(model) 
    
    total_loss = 0.0
    n_batches = 0
    skipped_batches = 0
    
    for features, labels in train_loader:
        if features is None:
            skipped_batches += 1
            continue
            
        if features.size(0) <= 1:
            skipped_batches += 1
            continue
            
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        
        if not torch.isfinite(features).all():
            skipped_batches += 1
            continue
            
        optimizer.zero_grad()
        emb = model(features)
        loss = loss_fn(emb, labels)
        
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
    if skipped_batches > 0:
        logging.warning(f"Epoch {epoch}: Skipped {skipped_batches} batches due to data issues")
        
    avg_loss = total_loss / (n_batches if n_batches > 0 else 1)
    logging.info(f"Epoch {epoch} train loss: {avg_loss:.4f} (processed {n_batches} batches)")
    return avg_loss

def extract_embeddings_for_loader(loader):
    model.eval()
    embeddings = {}
    labels_map = {}
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            feats, labs, filepaths = batch
            feats = feats.to(DEVICE)
            emb = model(feats)
            emb = emb.cpu()
            for i, p in enumerate(filepaths):
                embeddings[p] = emb[i]
                labels_map[p] = labs[i].item()
    return embeddings, labels_map

def validate_and_compute_eer_minDCF():
    eval_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             collate_fn=eval_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    embeddings, labels_map = extract_embeddings_for_loader(eval_loader)
    
    if len(embeddings) < 2:
        logging.warning("Not enough embeddings for validation metrics.")
        return None, None
        
    all_files = list(embeddings.keys())
    scores = []
    y_true = []
    num_pairs_to_sample = min(5000, len(all_files) * 5)
    
    for _ in range(num_pairs_to_sample):
        is_target = random.choice([True, False])
        f1 = random.choice(all_files)
        if is_target:
            label1 = labels_map[f1]
            same_files = [f for f in all_files if labels_map[f]==label1 and f!=f1]
            if not same_files: continue
            f2 = random.choice(same_files)
        else:
            label1 = labels_map[f1]
            other_files = [f for f in all_files if labels_map[f]!=label1]
            if not other_files: continue
            f2 = random.choice(other_files)
            
        emb1 = embeddings[f1]
        emb2 = embeddings[f2]
        score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        scores.append(score)
        y_true.append(1 if is_target else 0)
        
    if not scores:
        return None, None
        
    try:
        from sklearn.metrics import roc_curve
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        
        fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
        fnr = 1 - tpr
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = eer * 100
        
        p_target = 0.01; c_miss = 1; c_fa = 1
        dcf_costs = p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr
        min_dcf = np.min(dcf_costs)
        
        return eer, min_dcf
    except Exception as e:
        logging.error(f"Error computing EER/minDCF: {e}")
        return None, None

# ----------------- Main Training -----------------
logging.info("=== Starting fine-tuning of Quantum-Enhanced ECAPA-TDNN ===")
best_val_eer = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    if epoch == FREEZE_EPOCHS + 1:
        logging.info("Unfreezing model parameters - now fine-tuning whole model.")
        set_model_requires_grad(model, True)
        optimizer = optim.AdamW([
            {'params': [p for p in model.parameters() if p.requires_grad], 'lr': LR_MODEL, 'weight_decay': WEIGHT_DECAY},
            {'params': [loss_fn.weight], 'lr': LR_HEAD, 'weight_decay': WEIGHT_DECAY}
        ])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    train_loss = train_one_epoch(epoch)

    eer, min_dcf = validate_and_compute_eer_minDCF()
    if eer is not None and min_dcf is not None:
        logging.info(f"Epoch {epoch} validation -> EER: {eer:.2f}%, minDCF: {min_dcf:.4f}")
        
        if eer < best_val_eer:
            best_val_eer = eer
            best_ckpt_path = os.path.join(FINETUNE_CHECKPOINT_DIR, f"best_model_eer_{eer:.2f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_eer': eer,
                'val_min_dcf': min_dcf
            }, best_ckpt_path)
            logging.info(f"New best model saved: {best_ckpt_path}")
    else:
        logging.info(f"Epoch {epoch} validation -> EER: N/A, minDCF: N/A")

    ckpt_path = os.path.join(FINETUNE_CHECKPOINT_DIR, f"finetune_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_eer': eer,
        'val_min_dcf': min_dcf
    }, ckpt_path)
    
    if scheduler is not None:
        scheduler.step()

logging.info(f"=== Fine-tuning finished. Best EER: {best_val_eer:.2f}% ===")