# -*- coding: utf-8 -*-
"""
WavLM-based Speaker Verification with AAMSoftmax
Combines WavLM encoder with AAMSoftmax loss for speaker verification.
"""

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
import math
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from transformers import WavLMModel
from tqdm import tqdm

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CONFIG = {
    'train_dir': '/home/user2/7th/dataset/voxceleb1/train',
    'test_dir': '/home/user2/7th/dataset/voxceleb1/test',
    'log_dir': '/home/user2/7th-2/logs_wavlm',
    'checkpoint_dir': '/home/user2/7th-2/checkpoints_wavlm',
    'batch_size': 32,  # Reduced due to WavLM size
    'num_epochs': 50,
    'learning_rate': 1e-4,  # Lower LR for pretrained model
    'lr_decay': 0.97,
    'num_workers': 4,
    'seed': 42,
    
    # WavLM model selection
    'wavlm_model': 'microsoft/wavlm-base-plus',  # or 'microsoft/wavlm-large'
    'freeze_feature_extractor': True,  # Freeze WavLM's feature extractor
    'freeze_encoder': False,  # Whether to freeze entire WavLM
    
    # Audio parameters
    'sample_rate': 16000,
    'max_audio_sec': 3,  # Max audio length in seconds
    'train_chunk_frames': 300,  # 3 seconds at 16kHz = 48000 samples, but this is frames for features
    
    # Embedding head
    'embedding_dim': 256,
    
    # AAMSoftmax parameters
    'aam_margin': 0.2,
    'aam_scale': 30,
    
    # Evaluation
    'eval_pairs': 10000,
    'dcf_p_target': 0.01,
    'dcf_c_miss': 1,
    'dcf_c_fa': 1,
}

# ------------------------------------------------------------------
# Logging and seed helpers
# ------------------------------------------------------------------
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================================================================
# Section 1: WavLM-based Model
# ==================================================================

class WavLMSpeakerVerification(nn.Module):
    """
    WavLM-based speaker verification model with AAMSoftmax.
    """
    
    def __init__(self, model_name, embedding_dim, num_classes, 
                 freeze_feature_extractor=True, freeze_encoder=False):
        super().__init__()
        
        # Load pretrained WavLM
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        # Freeze parts of the model if specified
        if freeze_feature_extractor:
            self.wavlm.feature_extractor._freeze_parameters()
        
        if freeze_encoder:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        # Get hidden size from WavLM config
        hidden_size = self.wavlm.config.hidden_size
        
        # Projection head to get speaker embeddings
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # AAMSoftmax classifier (will be moved to device later)
        self.classifier = None
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
    def _init_classifier(self, num_classes, device):
        """Initialize classifier when number of classes is known"""
        if self.classifier is None or self.num_classes != num_classes:
            self.num_classes = num_classes
            self.classifier = AAMSoftmax(
                n_class=num_classes,
                in_features=self.embedding_dim,
                m=CONFIG['aam_margin'],
                s=CONFIG['aam_scale']
            ).to(device)
    
    def forward(self, waveforms, labels=None):
        """
        Args:
            waveforms: (batch_size, audio_samples)
            labels: (batch_size) optional labels for training
        Returns:
            If labels provided: (loss, embeddings)
            Else: embeddings
        """
        # Ensure waveforms have correct shape
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        
        # Forward through WavLM
        outputs = self.wavlm(waveforms)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Mean pooling over time dimension
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)
        
        # Project to embedding space
        embeddings = self.projection(pooled)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        if labels is not None:
            # Initialize classifier if needed
            self._init_classifier(len(torch.unique(labels)), embeddings.device)
            loss = self.classifier(embeddings, labels)
            return loss, embeddings
        
        return embeddings


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace) loss.
    """
    def __init__(self, n_class, in_features, m, s):
        super().__init__()
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, in_features))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-8)
        cosine = F.linear(x_norm, w_norm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return self.ce_loss(output, label)


# ==================================================================
# Section 2: Dataset and dataloaders
# ==================================================================

class VoxCelebDataset(Dataset):
    """Dataset for loading precomputed features from .npy files"""
    
    def __init__(self, data_dir_or_list):
        self.data_dirs = (
            [data_dir_or_list] if isinstance(data_dir_or_list, str)
            else data_dir_or_list
        )
        self.speaker_files = []
        self.speaker_to_label = {}
        self.num_speakers = 0

        logging.info(f"Loading data from: {self.data_dirs}")
        for data_dir in self.data_dirs:
            speaker_dirs = sorted([
                d for d in glob.glob(os.path.join(data_dir, "*"))
                if os.path.isdir(d)
            ])
            if not speaker_dirs:
                logging.warning(f"No speaker dirs found in {data_dir}")
                continue
            for speaker_dir in speaker_dirs:
                sid = os.path.basename(speaker_dir)
                if sid not in self.speaker_to_label:
                    self.speaker_to_label[sid] = self.num_speakers
                    self.num_speakers += 1
                label = self.speaker_to_label[sid]
                # Look for .npy files in subdirectories
                for npy_file in glob.glob(os.path.join(speaker_dir, "*", "*.npy")):
                    self.speaker_files.append((npy_file, label))
                # Also check directly in speaker directory
                for npy_file in glob.glob(os.path.join(speaker_dir, "*.npy")):
                    self.speaker_files.append((npy_file, label))

        if not self.speaker_files:
            logging.error("CRITICAL: No .npy files found.")
        else:
            logging.info(
                f"Loaded {len(self.speaker_files)} files "
                f"from {self.num_speakers} speakers."
            )

    def __len__(self):
        return len(self.speaker_files)

    def __getitem__(self, index):
        return self.speaker_files[index]


def _clean_feat(feat):
    """Replace infinite/NaN values with median"""
    if not np.isfinite(feat).all():
        finite_vals = feat[np.isfinite(feat)]
        fill = np.median(finite_vals) if len(finite_vals) > 0 else 0.0
        feat[~np.isfinite(feat)] = fill
    return feat


def train_collate_fn(batch, num_frames=CONFIG['train_chunk_frames']):
    """
    Collate function for training.
    Loads .npy files and prepares them for WavLM.
    """
    features, labels = [], []
    for npy_path, label in batch:
        try:
            # Load feature (assuming [time, freq] format)
            feat = _clean_feat(np.load(npy_path))
            
            # If feature is 2D [time, freq], transpose if needed
            if feat.ndim == 2:
                feat = feat.T  # [freq, time]
            
            # Ensure proper shape
            if feat.shape[0] != CONFIG['in_channels']:
                # If freq dimension is not 80, reshape
                if 'in_channels' in CONFIG:
                    logging.warning(f"Unexpected feature dimension: {feat.shape}")
            
            # Random chunk selection
            if feat.shape[1] < num_frames:
                # Pad if shorter than required
                feat = np.pad(feat, ((0, 0), (0, num_frames - feat.shape[1])), mode='wrap')
            else:
                start = random.randint(0, feat.shape[1] - num_frames)
                feat = feat[:, start:start + num_frames]
            
            features.append(feat.T)  # [time, freq] for WavLM
            labels.append(label)
        except Exception as e:
            logging.warning(f"Could not load {npy_path}: {e}")
            continue
    
    if not features:
        return None, None
    
    return torch.FloatTensor(np.array(features)), torch.LongTensor(labels)


def eval_collate_fn(batch):
    """Collate function for evaluation"""
    loaded, labels, filepaths = [], [], []
    for npy_path, label in batch:
        try:
            feat = _clean_feat(np.load(npy_path))
            if feat.ndim == 2:
                feat = feat.T  # [freq, time]
            loaded.append(feat.T)  # [time, freq]
            labels.append(label)
            filepaths.append(npy_path)
        except Exception as e:
            logging.warning(f"Could not load {npy_path}: {e}")
    
    if not loaded:
        return None, None, None
    
    # Pad sequences to same length
    max_len = max(f.shape[0] for f in loaded)
    padded = []
    for f in loaded:
        if f.shape[0] < max_len:
            pad_len = max_len - f.shape[0]
            padded.append(np.pad(f, ((0, pad_len), (0, 0)), mode='wrap'))
        else:
            padded.append(f)
    
    return (
        torch.FloatTensor(np.array(padded)),
        torch.LongTensor(labels),
        filepaths
    )


# ==================================================================
# Section 3: Metrics
# ==================================================================

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh


def calculate_min_dcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
    """Calculate minimum Detection Cost Function"""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    return float(np.min(p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr))


# ==================================================================
# Section 4: Train and eval functions
# ==================================================================

def train_epoch(model, loss_fn, data_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss, total_batches = 0, 0
    
    for features, labels in tqdm(data_loader, desc="Training"):
        if features is None:
            continue
        
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        loss, _ = model(features, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_fn.parameters()),
            max_norm=5.0
        )
        
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        if total_batches % 50 == 0:
            logging.info(f"  Batch {total_batches}/{len(data_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / max(total_batches, 1)


def evaluate_model(model, test_dataset, device, num_pairs=10000):
    """Evaluate model on test set"""
    logging.info(f"Evaluating on {len(test_dataset)} test files.")
    model.eval()
    
    eval_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
    )
    
    embeddings = {}
    speaker_labels = {}
    
    with torch.no_grad():
        for features, labels, filepaths in tqdm(eval_loader, desc="Extracting embeddings"):
            if features is None:
                continue
            
            # Process in smaller batches if needed
            batch_embeddings = []
            batch_size = 16
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size].to(device)
                batch_emb = model(batch_features)
                batch_embeddings.append(batch_emb.cpu())
            
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
            
            for i, fp in enumerate(filepaths):
                embeddings[fp] = batch_embeddings[i]
                speaker_labels[fp] = labels[i].item()
    
    logging.info(f"Extracted {len(embeddings)} embeddings.")
    if len(embeddings) < 2:
        return 0, 0
    
    # Generate trial pairs
    all_files = list(embeddings.keys())
    scores, y_true = [], []
    
    for _ in tqdm(range(num_pairs), desc="Evaluating pairs"):
        is_target = random.choice([True, False])
        f1 = f2 = None
        attempts = 0
        while f1 == f2 and attempts < 20:
            attempts += 1
            f1 = random.choice(all_files)
            l1 = speaker_labels[f1]
            if is_target:
                pool = [f for f, l in speaker_labels.items() if l == l1 and f != f1]
            else:
                pool = [f for f, l in speaker_labels.items() if l != l1]
            if not pool:
                f1 = f2 = None
                continue
            f2 = random.choice(pool)
        
        if f1 is None or f2 is None or f1 == f2:
            continue
        
        score = F.cosine_similarity(
            embeddings[f1].unsqueeze(0),
            embeddings[f2].unsqueeze(0)
        ).item()
        scores.append(score)
        y_true.append(1 if is_target else 0)
    
    if not scores:
        logging.error("No valid pairs generated.")
        return 0, 0
    
    y_s = np.array(scores)
    y_t = np.array(y_true)
    eer, _ = calculate_eer(y_t, y_s)
    min_dcf = calculate_min_dcf(
        y_t, y_s,
        CONFIG['dcf_p_target'],
        CONFIG['dcf_c_miss'],
        CONFIG['dcf_c_fa']
    )
    return eer, min_dcf


# ==================================================================
# Section 5: Main
# ==================================================================

def main():
    # Create directories
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    setup_logging(CONFIG['log_dir'])
    set_seed(CONFIG['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    logging.info(f"WavLM Model: {CONFIG['wavlm_model']}")
    logging.info(f"Config: {CONFIG}")
    
    # Load datasets
    logging.info("Loading training data...")
    train_dataset = VoxCelebDataset(CONFIG['train_dir'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
    )
    
    logging.info("Loading test data...")
    test_dataset = VoxCelebDataset(CONFIG['test_dir'])
    num_classes = train_dataset.num_speakers
    logging.info(f"Training on {num_classes} speakers.")
    
    # Initialize model
    logging.info("Initializing WavLM Speaker Verification model...")
    model = WavLMSpeakerVerification(
        model_name=CONFIG['wavlm_model'],
        embedding_dim=CONFIG['embedding_dim'],
        num_classes=num_classes,
        freeze_feature_extractor=CONFIG['freeze_feature_extractor'],
        freeze_encoder=CONFIG['freeze_encoder'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {total_params:,}")
    
    # Initialize loss function
    loss_fn = AAMSoftmax(
        n_class=num_classes,
        in_features=CONFIG['embedding_dim'],
        m=CONFIG['aam_margin'],
        s=CONFIG['aam_scale'],
    ).to(device)
    
    # Set classifier in model
    model.classifier = loss_fn
    
    # Optimizer with different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.wavlm.parameters(), 'lr': CONFIG['learning_rate'] * 0.1},
        {'params': model.projection.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.classifier.parameters(), 'lr': CONFIG['learning_rate']},
    ], weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=CONFIG['lr_decay'])
    
    # Training loop
    logging.info("=== Starting WavLM Training ===")
    best_eer = float('inf')
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        logging.info(f"--- Epoch {epoch}/{CONFIG['num_epochs']} ---")
        
        avg_loss = train_epoch(model, loss_fn, train_loader, optimizer, device)
        logging.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        
        eer, min_dcf = evaluate_model(
            model, test_dataset, device, CONFIG['eval_pairs']
        )
        logging.info(f"Epoch {epoch} | EER: {eer:.4f}%  minDCF: {min_dcf:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f"epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eer': eer,
            'min_dcf': min_dcf,
            'config': CONFIG,
        }, ckpt_path)
        logging.info(f"Checkpoint saved -> {ckpt_path}")
        
        # Save best model
        if eer < best_eer:
            best_eer = eer
            best_path = os.path.join(CONFIG['checkpoint_dir'], "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'eer': eer,
                'min_dcf': min_dcf,
            }, best_path)
            logging.info(f"Best model saved (EER: {eer:.4f}%)")
        
        scheduler.step()
        logging.info(f"LR -> {scheduler.get_last_lr()[0]:.6f}")
    
    logging.info("=== Training Complete ===")
    logging.info(f"Best EER: {best_eer:.4f}%")


if __name__ == "__main__":
    main()