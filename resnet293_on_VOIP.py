# -*- coding: utf-8 -*-
"""
TidyVoice-style Speaker Verification Pipeline
Architecture : ResNet-293 with Attentive Statistics Pooling
Loss         : AAM-Softmax (ArcFace)
Dataset      : speaker_id/LANG/*.npy  (80-dim mel features)
Eval         : 2x2 cross-lingual matrix (EN/Regional)
Metrics      : EER + minDCF (p_target=0.01)

Paper: TidyVoice: A Curated Multilingual Dataset for Speaker Verification
       Derived from Common Voice (arXiv:2601.16358v1)
"""

import os
import glob
import random
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==============================================================
#  CONFIGURATION  -- edit these paths / values
# ==============================================================
DATASET_ROOT    = "/home/user2/VOIP/VOIP_Mel_Features"
OUTPUT_DIR      = "/home/user2/VOIP/tidyvoice_resnet293_results"
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")

# Model
EMBEDDING_DIM   = 256
IN_CHANNELS     = 80

# Training
BATCH_SIZE      = 128
NUM_WORKERS     = 8
NUM_EPOCHS      = 150
LR_INITIAL      = 1e-3
LR_DECAY_FACTOR = 0.1
LR_DECAY_EPOCHS = [30, 60, 90, 120]
WEIGHT_DECAY    = 1e-4
SEGMENT_FRAMES  = 200
QUALITY_THRESH  = 70

# AAM-Softmax
AAM_MARGIN      = 0.2
AAM_SCALE       = 30

# Evaluation
NUM_PAIRS       = 5000
P_TARGET        = 0.01
C_MISS          = 1.0
C_FA            = 1.0

ENGLISH_LANG    = "EN"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONDITIONS      = ["EN-EN", "EN-Regional", "Regional-EN", "Regional-Regional"]

# ==============================================================
#  LOGGING
# ==============================================================
os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
for _c in CONDITIONS:
    os.makedirs(os.path.join(OUTPUT_DIR, _c), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "pipeline.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ==============================================================
#  MODEL COMPONENTS
# ==============================================================

class Bottleneck(nn.Module):
    """ResNet Bottleneck block: 1x1 -> 3x3 -> 1x1."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)),  inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class AttentiveStatsPool(nn.Module):
    """
    Attentive Statistics Pooling.
    Okabe et al., Interspeech 2018.
    Input : (B, C, T)
    Output: (B, 2*C)  -- concatenation of weighted mean and std
    """
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        alpha = self.attn(x)
        mean  = (alpha * x).sum(dim=2)
        var   = (alpha * x.pow(2)).sum(dim=2) - mean.pow(2)
        std   = var.clamp(min=1e-9).sqrt()
        return torch.cat([mean, std], dim=1)


class ResNet293(nn.Module):
    """
    ResNet-293 speaker embedding extractor (paper architecture).

    Residual stages:
      layer1:  3 x Bottleneck,  64-ch,  stride-1  -> out: 256-ch
      layer2:  8 x Bottleneck, 128-ch,  stride-2  -> out: 512-ch
      layer3: 36 x Bottleneck, 256-ch,  stride-2  -> out: 1024-ch
      layer4:  3 x Bottleneck, 512-ch,  stride-2  -> out: 2048-ch

    Total bottleneck blocks: 3+8+36+3 = 50
    Each block has 3 conv layers -> 150 conv layers
    + stem (2) + fc (1) + BN layers => ~293 parametric layers
    """

    def __init__(self, in_channels=IN_CHANNELS, embd_dim=EMBEDDING_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(Bottleneck,  64, in_planes=64,   blocks=3,  stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, in_planes=256,  blocks=8,  stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, in_planes=512,  blocks=36, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, in_planes=1024, blocks=3,  stride=2)

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse freq dim
        self.pool      = AttentiveStatsPool(in_dim=2048, hidden_dim=256)
        self.fc        = nn.Linear(4096, embd_dim)
        self.bn        = nn.BatchNorm1d(embd_dim)
        self._init_weights()

    def _make_layer(self, block, planes, in_planes, blocks, stride):
        layers = [block(in_planes, planes, stride=stride)]
        in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x : (B, T, 80)  time-first mel features
        returns : (B, embd_dim)  L2-normalised embeddings
        """
        x = x.transpose(1, 2).unsqueeze(1)   # (B, 1, 80, T)
        x = self.stem(x)                      # (B, 64,  20, T/4)
        x = self.layer1(x)                    # (B, 256, 20, T/4)
        x = self.layer2(x)                    # (B, 512, 10, T/8)
        x = self.layer3(x)                    # (B,1024,  5, T/16)
        x = self.layer4(x)                    # (B,2048,  3, T/32)
        x = self.freq_pool(x).squeeze(2)      # (B, 2048, T/32)
        x = self.pool(x)                      # (B, 4096)
        x = self.fc(x)                        # (B, embd_dim)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)


# ==============================================================
#  AAM-SOFTMAX LOSS  (ArcFace)
# ==============================================================

class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace).
    Deng et al., CVPR 2019.
    """
    def __init__(self, num_classes, embd_dim=EMBEDDING_DIM,
                 margin=AAM_MARGIN, scale=AAM_SCALE):
        super().__init__()
        self.margin = margin
        self.scale  = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embd_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine   = (1.0 - cosine.pow(2)).clamp(min=0).sqrt()
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        return self.ce(output, labels)


# ==============================================================
#  DATA UTILITIES
# ==============================================================

def file_quality(path):
    """Returns quality score 0-100."""
    try:
        feat  = np.load(path, allow_pickle=False)
        total = feat.size
        bad   = (~np.isfinite(feat)).sum()
        return max(0.0, 100.0 - 100.0 * bad / total)
    except Exception:
        return 0.0


def load_and_clean(path):
    """
    Load .npy mel feature -> (T, 80), apply CMVN and cleaning.
    Returns np.ndarray or None if quality too low.
    """
    if file_quality(path) < QUALITY_THRESH:
        return None
    try:
        feat = np.load(path, allow_pickle=False).astype(np.float32)
        feat = feat.T                              # -> (T, mel)
        if not np.isfinite(feat).all():
            mask  = ~np.isfinite(feat)
            clean = feat[np.isfinite(feat)]
            mu    = float(clean.mean())
            sigma = float(clean.std())
            feat[mask] = np.random.normal(mu, sigma, int(mask.sum()))
        feat = np.clip(feat, -10.0, 10.0)
        if feat.std() < 1e-6:
            feat = feat + np.random.normal(0, 1e-4, feat.shape).astype(np.float32)
        # utterance-level CMVN
        feat = (feat - feat.mean(axis=0, keepdims=True)) / (
                feat.std(axis=0,  keepdims=True) + 1e-8)
        return feat
    except Exception as e:
        log.warning("Failed to load %s: %s", path, e)
        return None


def language_from_path(path):
    """
    Infer language from path structure: speaker_id/LANG/file.npy
    Returns (lang_group, lang_code)
      lang_group : 'EN' or 'Regional' or None
      lang_code  : actual folder name
    """
    parts = path.split(os.sep)
    for part in parts[:-1]:
        if part == ENGLISH_LANG:
            return ENGLISH_LANG, ENGLISH_LANG
        if len(part) in (2, 3) and part.isupper() and part.isalpha():
            return "Regional", part
    return None, None


# ==============================================================
#  DATASET CLASSES
# ==============================================================

class SpeakerTrainDataset(torch.utils.data.Dataset):
    """
    Training dataset.
    Scans DATASET_ROOT/train/speaker_id/LANG/*.npy
    Returns fixed-length (SEGMENT_FRAMES, 80) segments.
    """

    def __init__(self, root, segment_frames=SEGMENT_FRAMES):
        self.segment  = segment_frames
        self.samples  = []
        self.speaker2idx = {}

        train_dir = os.path.join(root, "train")
        speaker_dirs = sorted(
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )
        log.info("Scanning %d training speakers...", len(speaker_dirs))

        for spk in tqdm(speaker_dirs, desc="Indexing train set"):
            spk_path = os.path.join(train_dir, spk)
            files    = glob.glob(os.path.join(spk_path, "**", "*.npy"), recursive=True)
            good     = [f for f in files if file_quality(f) >= QUALITY_THRESH]
            if not good:
                continue
            if spk not in self.speaker2idx:
                self.speaker2idx[spk] = len(self.speaker2idx)
            idx = self.speaker2idx[spk]
            for f in good:
                self.samples.append((f, idx))

        self.num_speakers = len(self.speaker2idx)
        log.info("Train set: %d files, %d speakers", len(self.samples), self.num_speakers)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, spk_idx = self.samples[idx]
        feat = load_and_clean(path)
        if feat is None or feat.shape[0] == 0:
            feat = np.zeros((self.segment, IN_CHANNELS), dtype=np.float32)
        T = feat.shape[0]
        if T < self.segment:
            reps = math.ceil(self.segment / T)
            feat = np.tile(feat, (reps, 1))[:self.segment]
        else:
            start = random.randint(0, T - self.segment)
            feat  = feat[start: start + self.segment]
        return torch.FloatTensor(feat), spk_idx


class EvalDataset(torch.utils.data.Dataset):
    """Evaluation dataset -- raw tuples, collate_fn handles loading."""

    def __init__(self, items):
        self.data = items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def eval_collate(batch):
    features, speakers, langs, paths = [], [], [], []
    max_t = 0
    for (path, spk, lang) in batch:
        feat = load_and_clean(path)
        if feat is None:
            continue
        if feat.shape[0] > max_t:
            max_t = feat.shape[0]
        features.append(feat)
        speakers.append(spk)
        langs.append(lang)
        paths.append(path)
    if not features:
        return None, None, None, None
    padded = []
    for feat in features:
        pad = max_t - feat.shape[0]
        if pad > 0:
            feat = np.pad(feat, ((0, pad), (0, 0)), mode="wrap")
        padded.append(feat)
    return torch.FloatTensor(np.array(padded)), speakers, langs, paths


def build_eval_lists(dataset_root):
    """
    Scan DATASET_ROOT/test/ and return:
      en_samples       : [(path, speaker_id, 'EN'), ...]
      regional_samples : [(path, speaker_id, 'XX'), ...]
    """
    test_dir = os.path.join(dataset_root, "test")
    en_samples, regional_samples = [], []
    skipped = 0
    speaker_dirs = [
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ]
    for spk in tqdm(speaker_dirs, desc="Indexing test set"):
        spk_path = os.path.join(test_dir, spk)
        for f in glob.glob(os.path.join(spk_path, "**", "*.npy"), recursive=True):
            if file_quality(f) < QUALITY_THRESH:
                skipped += 1
                continue
            lang_group, lang_code = language_from_path(f)
            if lang_group == ENGLISH_LANG:
                en_samples.append((f, spk, ENGLISH_LANG))
            elif lang_group == "Regional":
                regional_samples.append((f, spk, lang_code))
    log.info("Test -- EN: %d  Regional: %d  skipped: %d",
             len(en_samples), len(regional_samples), skipped)
    return en_samples, regional_samples


# ==============================================================
#  TRAINING
# ==============================================================

def train_one_epoch(model, loss_fn, loader, optimiser, epoch):
    model.train()
    loss_fn.train()
    total_loss, total = 0.0, 0
    for feats, labels in tqdm(loader, desc="Epoch %d" % epoch, leave=False):
        feats  = feats.to(DEVICE)
        labels = labels.to(DEVICE)
        optimiser.zero_grad()
        embs = model(feats)
        loss = loss_fn(embs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        total_loss += loss.item() * feats.size(0)
        total      += feats.size(0)
    return total_loss / max(total, 1)


def train(dataset_root=DATASET_ROOT):
    log.info("=" * 60)
    log.info("STARTING RESNET-293 TRAINING")
    log.info("=" * 60)

    train_dataset = SpeakerTrainDataset(dataset_root)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    model   = ResNet293().to(DEVICE)
    loss_fn = AAMSoftmax(num_classes=train_dataset.num_speakers).to(DEVICE)
    params  = list(model.parameters()) + list(loss_fn.parameters())
    opt     = torch.optim.Adam(params, lr=LR_INITIAL, weight_decay=WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=LR_DECAY_EPOCHS, gamma=LR_DECAY_FACTOR)

    best_loss = float("inf")
    history   = []

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(model, loss_fn, train_loader, opt, epoch)
        sched.step()
        history.append(loss)
        log.info("Epoch %03d/%d  loss=%.4f  lr=%.2e",
                 epoch, NUM_EPOCHS, loss, sched.get_last_lr()[0])

        ckpt = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "loss_fn_state":    loss_fn.state_dict(),
            "optimiser_state":  opt.state_dict(),
            "loss":             loss,
            "speaker2idx":      train_dataset.speaker2idx,
        }
        if epoch % 10 == 0:
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "epoch_%03d.pt" % epoch))
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            log.info("  -> New best model saved (loss=%.4f)", best_loss)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(history) + 1), history, "b-")
    plt.xlabel("Epoch"); plt.ylabel("AAM-Softmax Loss")
    plt.title("ResNet-293 Training Loss"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=200)
    plt.close()

    log.info("Training complete.")
    return model


# ==============================================================
#  EMBEDDING EXTRACTION
# ==============================================================

def extract_embeddings(model, items, desc="Extracting"):
    """
    items  : list of (path, speaker_id, language)
    returns: dict  path -> {embedding, speaker, language}
    """
    model.eval()
    dataset = EvalDataset(items)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=eval_collate, num_workers=NUM_WORKERS, pin_memory=True,
    )
    emb_dict = {}
    with torch.no_grad():
        for feats, speakers, langs, paths in tqdm(loader, desc=desc):
            if feats is None:
                continue
            embs = model(feats.to(DEVICE))
            for i, p in enumerate(paths):
                emb_dict[p] = {
                    "embedding": embs[i].cpu().numpy(),
                    "speaker":   speakers[i],
                    "language":  langs[i],
                }
    log.info("%s: %d embeddings extracted", desc, len(emb_dict))
    return emb_dict


# ==============================================================
#  SCORING AND METRICS
# ==============================================================

def generate_trials(enroll_dict, test_dict, num_pairs=NUM_PAIRS):
    """Build genuine + impostor trials. Returns (scores, labels)."""
    e_files = list(enroll_dict.keys())
    t_files = list(test_dict.keys())

    spk2test = defaultdict(list)
    for f in t_files:
        spk2test[test_dict[f]["speaker"]].append(f)

    scores, labels = [], []
    half = num_pairs // 2

    # Genuine pairs
    attempts = 0
    while len(scores) < half and attempts < half * 20:
        attempts += 1
        ef  = random.choice(e_files)
        spk = enroll_dict[ef]["speaker"]
        candidates = [f for f in spk2test.get(spk, []) if f != ef]
        if not candidates:
            continue
        tf = random.choice(candidates)
        e  = enroll_dict[ef]["embedding"]
        t  = test_dict[tf]["embedding"]
        scores.append(float(np.dot(e, t)))
        labels.append(1)

    # Impostor pairs
    attempts = 0
    while len(scores) < half * 2 and attempts < half * 20:
        attempts += 1
        ef   = random.choice(e_files)
        spk  = enroll_dict[ef]["speaker"]
        diff = [f for f in t_files if test_dict[f]["speaker"] != spk]
        if not diff:
            continue
        tf = random.choice(diff)
        e  = enroll_dict[ef]["embedding"]
        t  = test_dict[tf]["embedding"]
        scores.append(float(np.dot(e, t)))
        labels.append(0)

    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)


def compute_metrics(scores, labels):
    """Returns EER(%), minDCF, fpr, fnr, thresholds, eer_thr, dcf_thr."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0) * 100
    except Exception:
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))] * 100
    eer_idx     = np.nanargmin(np.abs(fnr - fpr))
    eer_thr     = float(thresholds[eer_idx])
    dcf         = P_TARGET * C_MISS * fnr + (1 - P_TARGET) * C_FA * fpr
    min_idx     = int(np.argmin(dcf))
    min_dcf     = float(dcf[min_idx])
    min_dcf_thr = float(thresholds[min_idx])
    return eer, min_dcf, fpr, fnr, thresholds, eer_thr, min_dcf_thr


# ==============================================================
#  PLOTTING FUNCTIONS
# ==============================================================

def plot_eer_curve(fpr, fnr, thresholds, eer, eer_thr, out_path, title=""):
    """Plot FAR and FRR curves with EER point (separate EER graph)"""
    plt.figure(figsize=(8, 7))
    
    # Plot FAR (FPR) and FRR (FNR) vs threshold
    plt.plot(thresholds, fpr, 'b-', linewidth=2, label='FAR (False Acceptance Rate)')
    plt.plot(thresholds, fnr, 'r-', linewidth=2, label='FRR (False Rejection Rate)')
    
    # Mark EER point
    eer_idx = np.argmin(np.abs(fpr - fnr))
    plt.plot(thresholds[eer_idx], eer/100, 'ko', markersize=10, 
             label=f'EER = {eer:.2f}% (threshold = {eer_thr:.3f})')
    
    # Add vertical and horizontal lines at EER
    plt.axvline(x=eer_thr, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=eer/100, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title(f'EER Curve - FAR vs FRR {title}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-1.0, 1.0])
    plt.ylim([0, 1.0])
    
    # Add text box with EER info
    textstr = f'EER = {eer:.2f}%\nThreshold = {eer_thr:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_det_curve(fpr, fnr, eer, out_path, title=""):
    """Plot DET (Detection Error Tradeoff) curve on log scale"""
    plt.figure(figsize=(8, 7))
    fpr_p = np.maximum(fpr * 100, 1e-4)
    fnr_p = np.maximum(fnr * 100, 1e-4)
    
    plt.plot(fpr_p, fnr_p, "b-", lw=2, label='DET Curve')
    
    idx = np.argmin(np.abs(fpr_p - fnr_p))
    plt.plot(fpr_p[idx], fnr_p[idx], "ro", ms=8, label=f'EER = {eer:.2f}%')
    
    d = np.linspace(0.01, 100, 200)
    plt.plot(d, d, "k--", alpha=0.4, lw=1, label='EER Line')
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positive Rate (FPR) [%]", fontsize=12)
    plt.ylabel("False Negative Rate (FNR) [%]", fontsize=12)
    plt.title(f"DET Curve (Log Scale) {title}", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, which="both")
    plt.xlim([0.01, 100])
    plt.ylim([0.01, 100])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_score_distribution(scores, labels, eer_thr, eer, out_path, title=""):
    """Plot score distributions with EER threshold"""
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    bins = np.linspace(-1, 1, 80)
    
    plt.figure(figsize=(10, 5))
    plt.hist(genuine, bins=bins, alpha=0.65, color="green",
             density=True, label=f"Genuine (n={len(genuine)})")
    plt.hist(impostor, bins=bins, alpha=0.65, color="red",
             density=True, label=f"Impostor (n={len(impostor)})")
    plt.axvline(eer_thr, color="black", ls="--", lw=2,
                label=f'EER Threshold = {eer_thr:.3f} (EER={eer:.2f}%)')
    
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Score Distributions {title}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics - using ASCII characters instead of Unicode
    textstr = f'Genuine: mean={np.mean(genuine):.3f}, std={np.std(genuine):.3f}\nImpostor: mean={np.mean(impostor):.3f}, std={np.std(impostor):.3f}\nEER = {eer:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, eer, out_path, title=""):
    """Plot ROC curve with AUC and EER"""
    from sklearn.metrics import auc as sk_auc
    plt.figure(figsize=(8, 7))
    roc_auc = sk_auc(fpr, tpr)
    
    plt.plot(fpr, tpr, "b-", lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Mark EER point (where FPR = 1 - TPR)
    diff = np.abs(fpr - (1 - tpr))
    idx = np.argmin(diff)
    plt.plot(fpr[idx], tpr[idx], "ro", ms=10, label=f'EER = {eer:.2f}%')
    
    plt.plot([0, 1], [0, 1], "k--", lw=1, label='Random')
    
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title(f"ROC Curve {title}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_metrics(condition, eer, min_dcf, eer_thr, dcf_thr, out_dir):
    rows = {
        "Metric": ["EER (%)", "minDCF (p=0.01)", "Threshold@EER", "Threshold@minDCF"],
        "Value":  [eer, min_dcf, eer_thr, dcf_thr],
    }
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("=" * 50 + "\n")
        f.write("Condition: %s\n" % condition)
        f.write("=" * 50 + "\n")
        f.write("EER           : %.4f %%\n" % eer)
        f.write("minDCF        : %.6f\n"    % min_dcf)
        f.write("Threshold@EER : %.6f\n"    % eer_thr)
        f.write("Threshold@DCF : %.6f\n"    % dcf_thr)


# ==============================================================
#  CROSS-LINGUAL EVALUATION
# ==============================================================

def evaluate(model):
    log.info("=" * 60)
    log.info("CROSS-LINGUAL EVALUATION")
    log.info("=" * 60)

    en_items, reg_items = build_eval_lists(DATASET_ROOT)
    if not en_items or not reg_items:
        log.error("Need both EN and Regional samples.")
        return {}

    en_embs  = extract_embeddings(model, en_items,  desc="EN embeddings")
    reg_embs = extract_embeddings(model, reg_items, desc="Regional embeddings")

    cond_map = {
        "EN-EN":             (en_embs,  en_embs),
        "EN-Regional":       (en_embs,  reg_embs),
        "Regional-EN":       (reg_embs, en_embs),
        "Regional-Regional": (reg_embs, reg_embs),
    }

    results = {}
    for cond, (enroll, test) in cond_map.items():
        log.info("Condition: %s", cond)
        scores, labels = generate_trials(enroll, test, NUM_PAIRS)
        eer, min_dcf, fpr, fnr, thresholds, eer_thr, dcf_thr = compute_metrics(scores, labels)
        tpr = 1.0 - fnr  # True Positive Rate
        
        results[cond] = dict(eer=eer, min_dcf=min_dcf,
                             eer_threshold=eer_thr, min_dcf_threshold=dcf_thr)
        log.info("  EER=%.4f%%  minDCF=%.6f", eer, min_dcf)

        out = os.path.join(OUTPUT_DIR, cond)
        save_metrics(cond, eer, min_dcf, eer_thr, dcf_thr, out)
        
        # Generate ALL 4 graphs for each condition
        # 1. EER Curve (FAR vs FRR)
        plot_eer_curve(fpr, fnr, thresholds, eer, eer_thr,
                       os.path.join(out, "eer_curve.png"), "(%s)" % cond)
        
        # 2. DET Curve (log scale)
        plot_det_curve(fpr, fnr, eer,
                       os.path.join(out, "det_curve.png"), "(%s)" % cond)
        
        # 3. Score Distribution
        plot_score_distribution(scores, labels, eer_thr, eer,
                                os.path.join(out, "score_dist.png"), "(%s)" % cond)
        
        # 4. ROC Curve
        plot_roc_curve(fpr, tpr, eer,
                       os.path.join(out, "roc_curve.png"), "(%s)" % cond)

    _summary(results)
    
    # Display all graphs
    display_all_graphs()
    
    return results


def display_all_graphs():
    """Display all saved graphs from the output directory"""
    import matplotlib.pyplot as plt
    
    for cond in CONDITIONS:
        cond_dir = os.path.join(OUTPUT_DIR, cond)
        graphs = ["eer_curve.png", "det_curve.png", "roc_curve.png", "score_dist.png"]
        
        for graph_name in graphs:
            graph_path = os.path.join(cond_dir, graph_name)
            if os.path.exists(graph_path):
                img = plt.imread(graph_path)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"{graph_name.replace('_', ' ').replace('.png', '').upper()} - {cond}")
                plt.tight_layout()
                plt.show()
    
    # Display summary heatmap
    heatmap_path = os.path.join(OUTPUT_DIR, "summary_heatmap.png")
    if os.path.exists(heatmap_path):
        img = plt.imread(heatmap_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Cross-Lingual EER Heatmap")
        plt.tight_layout()
        plt.show()


def _summary(results):
    """Save heatmap, bar charts and markdown report."""
    mat = np.array([
        [results["EN-EN"]["eer"],       results["EN-Regional"]["eer"]],
        [results["Regional-EN"]["eer"], results["Regional-Regional"]["eer"]],
    ])
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdYlGn_r",
                xticklabels=["Test EN", "Test Regional"],
                yticklabels=["Enroll EN", "Enroll Regional"],
                cbar_kws={"label": "EER (%)"},
                linewidths=2, linecolor="black",
                vmin=0, vmax=max(20, float(mat.max())))
    plt.title("Cross-Lingual EER (%) -- ResNet-293", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_heatmap.png"), dpi=200)
    plt.close()

    conds  = CONDITIONS
    eers   = [results[c]["eer"]     for c in conds]
    dcfs   = [results[c]["min_dcf"] for c in conds]
    colors = ["#2ecc71", "#f39c12", "#f39c12", "#3498db"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, ylabel, ttl in [
        (axes[0], eers, "EER (%)",  "Equal Error Rate"),
        (axes[1], dcfs, "minDCF",   "Min Detection Cost"),
    ]:
        ax.bar(conds, vals, color=colors, alpha=0.8, edgecolor="black")
        for i, v in enumerate(vals):
            ax.text(i, v, "%.4f" % v, ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel); ax.set_title(ttl)
        ax.set_xticklabels(conds, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_bars.png"), dpi=200)
    plt.close()

    rows = []
    for c in conds:
        r = results[c]
        rows.append({"Condition": c, "EER (%)": r["eer"],
                     "minDCF": r["min_dcf"],
                     "Threshold@EER": r["eer_threshold"],
                     "Threshold@minDCF": r["min_dcf_threshold"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    within = (results["EN-EN"]["eer"] + results["Regional-Regional"]["eer"]) / 2
    cross  = (results["EN-Regional"]["eer"] + results["Regional-EN"]["eer"])  / 2
    deg    = (cross - within) / max(within, 1e-9) * 100

    with open(os.path.join(OUTPUT_DIR, "report.md"), "w") as f:
        f.write("# ResNet-293 Cross-Lingual Speaker Verification Report\n\n")
        f.write("## Architecture\n\n")
        f.write("- ResNet-293 with Attentive Statistics Pooling\n")
        f.write("- 256-dim L2-normalised embeddings\n")
        f.write("- AAM-Softmax (ArcFace) training loss\n")
        f.write("- 80-dim log-mel features, utterance-level CMVN\n\n")
        f.write("## Results\n\n")
        f.write("| Condition | EER (%) | minDCF |\n")
        f.write("|-----------|---------|--------|\n")
        for c in conds:
            r = results[c]
            f.write("| %s | %.4f | %.6f |\n" % (c, r["eer"], r["min_dcf"]))
        f.write("\n**Within-language avg EER : %.2f%%**\n" % within)
        f.write("\n**Cross-lingual avg EER   : %.2f%%**\n" % cross)
        f.write("\n**Cross-lingual degradation: %.1f%%**\n" % deg)

    log.info("Summary saved to %s", OUTPUT_DIR)
    log.info("=" * 60)
    for c in conds:
        log.info("  %-22s  EER=%6.2f%%  minDCF=%.6f",
                 c, results[c]["eer"], results[c]["min_dcf"])
    log.info("=" * 60)


# ==============================================================
#  ENTRY POINTS
# ==============================================================

def load_checkpoint(path):
    """Load saved checkpoint for eval-only runs."""
    log.info("Loading checkpoint: %s", path)
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNet293().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    log.info("=" * 60)
    log.info("TidyVoice ResNet-293 -- Full Pipeline")
    log.info("Device : %s", DEVICE)
    log.info("Dataset: %s", DATASET_ROOT)
    log.info("Output : %s", OUTPUT_DIR)
    log.info("=" * 60)

    # Phase 1: Train from scratch
    model = train(DATASET_ROOT)

    # Phase 2: Cross-lingual evaluation
    evaluate(model)

    # To run eval only on existing checkpoint, replace the two lines
    # above with:
    #   model = load_checkpoint(os.path.join(CHECKPOINT_DIR, "best_model.pt"))
    #   evaluate(model)


if __name__ == "__main__":
    main()