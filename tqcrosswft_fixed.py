# -*- coding: utf-8 -*-
# tqcrosswft_fixed.py
import os
import sys
import glob
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings("ignore")

# Try importing Pennylane for quantum components
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Pennylane not installed. Quantum components will be simulated.")

# ----------------- CONFIGURATION -----------------
class Config:
    # Paths
    DATA_ROOT = "/home/user2/VOIP/VOIP_Mel_Features"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "test")
    PRETRAINED_PATH = "/home/user2/VOIP/finetune_checkpoints_qecapa_voip_atharva/best_model_eer_8.68.pt"
    OUTPUT_DIR = "/home/user2/VOIP/finetune_checkpoints_qecapa_voip_atharva_improved"
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Model parameters
    IN_CHANNELS = 80
    MODEL_CHANNELS = 512
    EMBEDDING_DIM = 192
    NUM_QUBITS = 6
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss function parameters
    MARGIN = 0.2
    SCALE = 30
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 5

config = Config()

# ----------------- SETUP LOGGING -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.OUTPUT_DIR, "training.log")),
        logging.StreamHandler()
    ]
)

logging.info("=" * 80)
logging.info(f"Using device: {config.DEVICE}")
logging.info(f"Pennylane available: {PENNYLANE_AVAILABLE}")
logging.info("=" * 80)

# ----------------- QUANTUM CIRCUIT -----------------
if PENNYLANE_AVAILABLE:
    DEV = qml.device("lightning.qubit", wires=config.NUM_QUBITS)
    
    @qml.qnode(DEV, interface="torch")
    def quantum_circuit(inputs, weights):
        """Quantum circuit for feature processing"""
        qml.AngleEmbedding(inputs, wires=range(config.NUM_QUBITS), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(config.NUM_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(config.NUM_QUBITS)]
else:
    # Dummy quantum circuit for when Pennylane is not available
    def quantum_circuit(inputs, weights):
        return torch.zeros(inputs.shape[0], config.NUM_QUBITS)

# ----------------- MODEL DEFINITION -----------------
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
        
        if PENNYLANE_AVAILABLE:
            weight_shapes = {"weights": (3, num_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        else:
            self.qlayer = None
        
        hybrid_dim = num_qubits + downsample_dim
        self.final_linear = nn.Linear(hybrid_dim, embd_dim)
        self.final_bn = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        downsampled = self.downsample_bn(F.relu(self.downsample_linear(x)))
        
        if self.qlayer is not None:
            quantum_output = self.qlayer(downsampled)
            if quantum_output.device != downsampled.device:
                quantum_output = quantum_output.to(downsampled.device)
        else:
            quantum_output = torch.zeros(downsampled.shape[0], downsampled.shape[1], device=downsampled.device)
            
        hybrid_features = torch.cat([downsampled, quantum_output], dim=1)
        out = self.final_linear(hybrid_features)
        out = self.final_bn(out)
        return out

class QECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, embd_dim=192, num_qubits=6, quantum_circuit=quantum_circuit):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, 128)
        self.bn1 = nn.BatchNorm1d(3072)
        
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

# ----------------- LOSS FUNCTION -----------------
class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, margin=0.2, scale=30):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        
        self.W = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, embeddings, labels):
        # Normalize weights and embeddings
        W_norm = F.normalize(self.W, dim=1)
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings_norm, W_norm)
        
        # Convert labels to one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin
        sine = torch.sqrt(1.0 - cosine**2 + 1e-12)
        phi = cosine * torch.cos(torch.tensor(self.margin)) - sine * torch.sin(torch.tensor(self.margin))
        
        # Combine with original cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        loss = F.cross_entropy(output, labels)
        return loss

# ----------------- DATA LOADING -----------------
def analyze_file_quality(npy_path):
    """Analyze file quality"""
    try:
        feat = np.load(npy_path, allow_pickle=False)
        feat = feat.T
        
        total_values = feat.size
        nan_count = np.isnan(feat).sum()
        inf_count = np.isinf(feat).sum()
        bad_count = nan_count + inf_count
        
        bad_percentage = (bad_count / total_values) * 100
        quality_score = max(0, 100 - bad_percentage)
        
        return quality_score, bad_percentage, nan_count, inf_count
    except Exception as e:
        return 0, 100, 0, 0

def load_feature_file(npy_path, quality_threshold=70):
    """Load and clean feature file"""
    quality_score, _, _, _ = analyze_file_quality(npy_path)
    
    if quality_score < quality_threshold:
        return None
    
    try:
        feat = np.load(npy_path, allow_pickle=False).astype(np.float32)
        feat = feat.T  # (time, freq)
        
        # Handle NaN/Inf values
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
        
        if feat.shape[0] < 50:  # Skip very short files
            return None
        
        return feat
    except Exception as e:
        logging.debug(f"Error loading {npy_path}: {e}")
        return None

def create_dataset(dir_path, suffix=".npy", quality_threshold=70):
    """Create dataset with speaker mapping"""
    data = []
    speaker_to_idx = {}
    corrupted_count = 0
    short_count = 0
    total_count = 0
    
    speaker_dirs = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
    
    for speaker_id in tqdm(speaker_dirs, desc="Loading speakers"):
        speaker_path = os.path.join(dir_path, speaker_id)
        
        # Assign speaker index
        if speaker_id not in speaker_to_idx:
            speaker_to_idx[speaker_id] = len(speaker_to_idx)
        
        # Process language folders
        for lang_folder in os.listdir(speaker_path):
            lang_path = os.path.join(speaker_path, lang_folder)
            if not os.path.isdir(lang_path):
                continue
            
            npy_files = glob.glob(os.path.join(lang_path, f"*{suffix}"))
            
            for npy_path in npy_files:
                total_count += 1
                feat = load_feature_file(npy_path, quality_threshold)
                
                if feat is None:
                    corrupted_count += 1
                    continue
                
                if feat.shape[0] < 50:
                    short_count += 1
                    continue
                
                data.append({
                    'path': npy_path,
                    'speaker_id': speaker_id,
                    'speaker_idx': speaker_to_idx[speaker_id],
                    'features': feat
                })
    
    logging.info(f"Dataset loaded: {len(data)} files from {len(speaker_to_idx)} speakers")
    logging.info(f"Total files scanned: {total_count}")
    logging.info(f"Corrupted files: {corrupted_count}")
    logging.info(f"Skipped (too short, <50 frames): {short_count}")
    
    if len(speaker_to_idx) > 0:
        speaker_files = [len([d for d in data if d['speaker_id'] == s]) for s in speaker_to_idx.keys()]
        logging.info(f"Files per speaker - min: {min(speaker_files)}, max: {max(speaker_files)}, avg: {len(data)/len(speaker_to_idx):.1f}")
    
    return data, speaker_to_idx

class SpeakerDataset(Dataset):
    def __init__(self, data, speaker_to_idx):
        self.data = data
        self.speaker_to_idx = speaker_to_idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        speaker_idx = item['speaker_idx']
        
        return torch.FloatTensor(features), speaker_idx

def collate_fn(batch):
    """Collate function for variable length features"""
    features, labels = zip(*batch)
    
    # Find max length
    max_len = max(f.shape[0] for f in features)
    
    # Pad all sequences to max_len
    padded_features = []
    for feat in features:
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            feat = torch.cat([feat, feat[:pad_len]], dim=0)
        padded_features.append(feat)
    
    features_tensor = torch.stack(padded_features)
    labels_tensor = torch.tensor(labels)
    
    return features_tensor, labels_tensor

# ----------------- PRETRAINED WEIGHTS LOADING -----------------
def load_pretrained_weights(model, pretrained_path):
    """Load pretrained weights with compatibility for older PyTorch versions"""
    logging.info(f"Loading pretrained weights from: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        logging.warning(f"Pretrained weights not found at {pretrained_path}")
        logging.info("Starting training from scratch (random initialization)")
        return model
    
    try:
        # First try with weights_only=True (safer for newer checkpoints)
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        logging.info("Successfully loaded with weights_only=True")
    except Exception as e:
        logging.warning(f"Failed to load with weights_only=True: {e}")
        logging.info("Falling back to weights_only=False (ensure file is trusted)")
        
        # Fall back to weights_only=False for older checkpoints
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        logging.info("Successfully loaded with weights_only=False")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'epoch' in checkpoint:
                logging.info(f"Checkpoint was from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'eer' in checkpoint:
                logging.info(f"Checkpoint had EER: {checkpoint.get('eer', 'unknown'):.2f}%")
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=False)
        logging.info("Pretrained weights loaded successfully")
    except Exception as e:
        logging.error(f"Error loading state dict: {e}")
        logging.info("Attempting to load with modified keys...")
        
        # Try to adapt keys if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
            logging.info("Loaded with modified keys")
        except Exception as e2:
            logging.error(f"Still failed: {e2}")
            logging.info("Starting training from scratch")
    
    return model

# ----------------- TRAINING FUNCTIONS -----------------
def train_epoch(model, loss_fn, optimizer, train_loader, epoch, writer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        embeddings = model(features)
        loss = loss_fn(embeddings, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy (for monitoring only)
        with torch.no_grad():
            W_norm = F.normalize(loss_fn.W, dim=1)
            embeddings_norm = F.normalize(embeddings, dim=1)
            logits = config.SCALE * F.linear(embeddings_norm, W_norm)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
        
        # Log to tensorboard
        if batch_idx % config.LOG_INTERVAL == 0:
            step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/Accuracy', 100*correct/total, step)
    
    return total_loss / len(train_loader), 100 * correct / total

def validate(model, loss_fn, val_loader, epoch, writer):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            embeddings = model(features)
            loss = loss_fn(embeddings, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            W_norm = F.normalize(loss_fn.W, dim=1)
            embeddings_norm = F.normalize(embeddings, dim=1)
            logits = config.SCALE * F.linear(embeddings_norm, W_norm)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Accuracy', accuracy, epoch)
    
    return avg_loss, accuracy, all_embeddings, all_labels

# ----------------- MAIN TRAINING LOOP -----------------
def main():
    logging.info("=" * 80)
    logging.info("STARTING QECAPA-TDNN FINE-TUNING")
    logging.info("=" * 80)
    
    writer = SummaryWriter(config.LOG_DIR)
    
    # Load training data
    logging.info("-" * 50)
    logging.info("Loading full training dataset...")
    train_data, speaker_to_idx = create_dataset(config.TRAIN_DIR)
    
    num_speakers = len(speaker_to_idx)
    logging.info(f"Found {num_speakers} speakers in training set")
    
    # Split into train/validation
    random.shuffle(train_data)
    val_size = int(0.1 * len(train_data))
    train_data, val_data = train_data[val_size:], train_data[:val_size]
    
    logging.info(f"Train/Validation split: {len(train_data)} / {len(val_data)} files")
    
    # Create datasets and dataloaders
    train_dataset = SpeakerDataset(train_data, speaker_to_idx)
    val_dataset = SpeakerDataset(val_data, speaker_to_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
    logging.info("-" * 50)
    logging.info("Creating model...")
    model = QECAPA_TDNN(
        in_channels=config.IN_CHANNELS,
        channels=config.MODEL_CHANNELS,
        embd_dim=config.EMBEDDING_DIM,
        num_qubits=config.NUM_QUBITS,
        quantum_circuit=quantum_circuit
    ).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Load pretrained weights
    if os.path.exists(config.PRETRAINED_PATH):
        logging.info(f"Loading pretrained weights from: {config.PRETRAINED_PATH}")
        model = load_pretrained_weights(model, config.PRETRAINED_PATH)
    else:
        logging.info("No pretrained weights found. Starting from random initialization.")
    
    # Create loss function and optimizer
    loss_fn = ArcFaceLoss(
        num_classes=num_speakers,
        embedding_dim=config.EMBEDDING_DIM,
        margin=config.MARGIN,
        scale=config.SCALE
    ).to(config.DEVICE)
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Training loop
    logging.info("-" * 50)
    logging.info("Starting training...")
    
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        logging.info(f"\nEpoch {epoch}/{config.EPOCHS}")
        logging.info("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, loss_fn, optimizer, train_loader, epoch, writer)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, loss_fn, val_loader, epoch, writer)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.2e}")
        
        # Print learning rate reduction info manually
        if current_lr < config.LEARNING_RATE * 0.99:
            logging.info(f"Learning rate reduced to {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss - config.MIN_DELTA:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'speaker_to_idx': speaker_to_idx
            }
            
            torch.save(checkpoint, os.path.join(config.OUTPUT_DIR, 'best_model.pt'))
            logging.info(f"? New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter} epochs")
        
        # Save checkpoint periodically
        if epoch % config.SAVE_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'speaker_to_idx': speaker_to_idx
            }
            torch.save(checkpoint, os.path.join(config.OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pt'))
            logging.info(f"Checkpoint saved at epoch {epoch}")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logging.info(f"All outputs saved to: {config.OUTPUT_DIR}")
    logging.info("=" * 80)
    
    writer.close()

if __name__ == "__main__":
    main()