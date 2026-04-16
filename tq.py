# evaluate_model.py
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd
from tqdm import tqdm
import logging
import pennylane as qml

# ----------------- CONFIGURATION -----------------
MODEL_PATH = "/home/user2/VOIP/finetune_checkpoints_qecapa_voip_atharva/best_model_eer_8.68.pt"  # Update with your best model path
DATASET_ROOT = "/home/user2/VOIP/VOIP_Mel_Features"
OUTPUT_DIR = "/home/user2/VOIP/model_evaluation_results"  # All results will be saved here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quantum configuration (must match training)
NUM_QUBITS = 6
DEV = qml.device("lightning.qubit", wires=NUM_QUBITS)

# Evaluation parameters
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_PAIRS_FOR_EER = 10000  # Number of pairs to sample for EER calculation
TSNE_SPEAKERS = 20  # Number of speakers to visualize in t-SNE
TSNE_SAMPLES_PER_SPEAKER = 10  # Samples per speaker for t-SNE

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Setup Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "evaluation.log")),
        logging.StreamHandler()
    ]
)
from quanta import Res2Conv1dReluBn,Conv1dReluBn,SE_Connect
# ----------------- Quantum Circuit (must match training) -----------------
@qml.qnode(DEV, interface="torch")
def quantum_circuit(inputs, weights):
    """Quantum circuit for feature processing"""
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

# ----------------- Model Definition (must match training) -----------------
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

# ----------------- Data Loading & Preprocessing -----------------
def analyze_file_quality(npy_path):
    """Analyze file quality and return quality score (0-100)"""
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

def aggressive_feature_cleaning(npy_path, quality_threshold=70):
    """Aggressive cleaning with quality threshold"""
    quality_score, bad_percentage, nan_count, inf_count = analyze_file_quality(npy_path)
    
    if quality_score < quality_threshold:
        logging.warning(f"SKIPPING severely corrupted file: {npy_path}")
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

def create_clean_dataset_list(dir_path, suffix=".npy", quality_threshold=70):
    """Create dataset list with pre-filtering."""
    clean_list = []
    corrupted_count = 0
    total_count = 0
    
    speaker_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    
    for sp in speaker_dirs:
        sp_path = os.path.join(dir_path, sp)
        npy_files = glob.glob(os.path.join(sp_path, "**", f"*{suffix}"), recursive=True)
        
        for npy_path in npy_files:
            total_count += 1
            quality_score, bad_percentage, _, _ = analyze_file_quality(npy_path)
            
            if quality_score >= quality_threshold:
                clean_list.append((npy_path, sp))
            else:
                corrupted_count += 1
    
    logging.info(f"Dataset filtering: {len(clean_list)}/{total_count} files passed quality threshold")
    
    return clean_list

class SimpleListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
        speaker_set = sorted(list({s for _, s in items}))
        self.spk2label = {s: i for i, s in enumerate(speaker_set)}
        self.data = [(p, self.spk2label[s]) for p, s in items]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    def get_speaker_labels(self): return self.spk2label

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

# ----------------- Load Model -----------------
def load_model(model_path, num_classes=None):
    """Load the trained model"""
    logging.info(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = QECAPA_TDNN(in_channels=80, channels=512, embd_dim=192).to(DEVICE)
    
    # Load model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logging.info("Model loaded successfully")
    
    return model, checkpoint

# ----------------- Evaluation Functions -----------------
def extract_all_embeddings(model, dataloader):
    """Extract embeddings for all files in dataloader"""
    model.eval()
    embeddings_dict = {}
    labels_dict = {}
    filepaths_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if batch is None:
                continue
            features, labels, filepaths = batch
            features = features.to(DEVICE)
            
            batch_embeddings = model(features)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            for i, filepath in enumerate(filepaths):
                embeddings_dict[filepath] = batch_embeddings[i].cpu().numpy()
                labels_dict[filepath] = labels[i].item()
                filepaths_list.append(filepath)
    
    logging.info(f"Extracted {len(embeddings_dict)} embeddings")
    return embeddings_dict, labels_dict, filepaths_list

def compute_scores(embeddings_dict, labels_dict, num_pairs=10000):
    """Compute similarity scores for genuine and impostor pairs"""
    logging.info(f"Computing scores for {num_pairs} pairs...")
    
    filepaths = list(embeddings_dict.keys())
    scores = []
    y_true = []
    
    # Create genuine pairs (same speaker)
    genuine_count = num_pairs // 2
    impostor_count = num_pairs // 2
    
    # Sample genuine pairs
    for _ in tqdm(range(genuine_count), desc="Genuine pairs"):
        # Randomly select a speaker
        speaker_files = []
        while len(speaker_files) < 2:
            file1 = random.choice(filepaths)
            label1 = labels_dict[file1]
            speaker_files = [f for f in filepaths if labels_dict[f] == label1]
        
        # Select two different files from same speaker
        file1, file2 = random.sample(speaker_files, 2)
        
        # Compute cosine similarity
        emb1 = embeddings_dict[file1]
        emb2 = embeddings_dict[file2]
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        scores.append(score)
        y_true.append(1)  # Genuine
    
    # Sample impostor pairs
    for _ in tqdm(range(impostor_count), desc="Impostor pairs"):
        # Select two files from different speakers
        file1 = random.choice(filepaths)
        label1 = labels_dict[file1]
        
        # Find a file from different speaker
        other_files = [f for f in filepaths if labels_dict[f] != label1]
        if not other_files:
            continue
            
        file2 = random.choice(other_files)
        
        # Compute cosine similarity
        emb1 = embeddings_dict[file1]
        emb2 = embeddings_dict[file2]
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        scores.append(score)
        y_true.append(0)  # Impostor
    
    return np.array(scores), np.array(y_true)

def compute_eer_minDCF(scores, y_true, p_target=0.01, c_miss=1, c_fa=1):
    """Compute EER and minDCF"""
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find EER (point where FAR = FRR)
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    # Alternatively, use brentq method for more precise EER
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = eer * 100  # Convert to percentage
    except:
        eer = eer * 100
    
    # Compute minDCF
    dcf_costs = p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr
    min_dcf = np.min(dcf_costs)
    
    # Find threshold at minDCF
    min_dcf_threshold = thresholds[np.argmin(dcf_costs)]
    
    return eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold

# ----------------- Plotting Functions -----------------
def plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, output_path):
    """Plot FAR and FRR curves with EER point"""
    plt.figure(figsize=(12, 6))
    
    # Plot FAR and FRR
    plt.plot(thresholds, fpr, 'b-', linewidth=2, label='FAR (False Acceptance Rate)')
    plt.plot(thresholds, fnr, 'r-', linewidth=2, label='FRR (False Rejection Rate)')
    
    # Mark EER point
    eer_idx = np.argmin(np.abs(fpr - fnr))
    plt.plot(thresholds[eer_idx], eer/100, 'ko', markersize=10, 
             label=f'EER = {eer:.2f}% (threshold = {eer_threshold:.3f})')
    
    # Draw vertical line at EER threshold
    plt.axvline(x=eer_threshold, color='k', linestyle='--', alpha=0.5)
    
    # Draw horizontal line at EER value
    plt.axhline(y=eer/100, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Detection Error Tradeoff (DET) Curve - FAR vs FRR', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-1.0, 1.0])
    plt.ylim([0, 1.0])
    
    # Add text box with metrics
    textstr = f'EER = {eer:.2f}%\nThreshold = {eer_threshold:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"EER plot saved to: {output_path}")

def plot_roc_curve(fpr, tpr, eer, output_path):
    """Plot ROC curve"""
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (EER = {eer:.2f}%)')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    # Mark EER point
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=10, 
             label=f'EER point ({fpr[eer_idx]*100:.1f}%, {tpr[eer_idx]*100:.1f}%)')
    
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    
    # Fill area under curve
    plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
    
    # Calculate AUC
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    # Add text box with metrics
    textstr = f'AUC = {roc_auc:.4f}\nEER = {eer:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.6, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"ROC plot saved to: {output_path}")

def plot_score_distributions(scores, y_true, eer_threshold, output_path):
    """Plot score distributions for genuine and impostor pairs"""
    genuine_scores = scores[y_true == 1]
    impostor_scores = scores[y_true == 0]
    
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    bins = np.linspace(-1, 1, 101)
    
    plt.hist(genuine_scores, bins=bins, alpha=0.7, color='green', 
             label=f'Genuine (n={len(genuine_scores)})', density=True)
    plt.hist(impostor_scores, bins=bins, alpha=0.7, color='red', 
             label=f'Impostor (n={len(impostor_scores)})', density=True)
    
    # Plot KDE
    from scipy.stats import gaussian_kde
    if len(genuine_scores) > 1:
        kde_gen = gaussian_kde(genuine_scores)
        x_plot = np.linspace(-1, 1, 1000)
        plt.plot(x_plot, kde_gen(x_plot), 'g-', linewidth=2, label='Genuine KDE')
    
    if len(impostor_scores) > 1:
        kde_imp = gaussian_kde(impostor_scores)
        plt.plot(x_plot, kde_imp(x_plot), 'r-', linewidth=2, label='Impostor KDE')
    
    # Mark EER threshold
    plt.axvline(x=eer_threshold, color='k', linestyle='--', linewidth=2, 
                label=f'EER threshold = {eer_threshold:.3f}')
    
    plt.xlabel('Cosine Similarity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Score Distributions - Genuine vs Impostor Pairs', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-1, 1])
    
    # Calculate statistics
    stats_text = (f'Genuine: μ={np.mean(genuine_scores):.3f}, σ={np.std(genuine_scores):.3f}\n'
                  f'Impostor: μ={np.mean(impostor_scores):.3f}, σ={np.std(impostor_scores):.3f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Score distribution plot saved to: {output_path}")

def plot_tsne(embeddings_dict, labels_dict, output_path, n_speakers=20, n_samples_per_speaker=10):
    """Plot t-SNE visualization of embeddings"""
    logging.info("Computing t-SNE visualization...")
    
    # Select speakers with most samples
    speaker_counts = {}
    for filepath, label in labels_dict.items():
        speaker_counts[label] = speaker_counts.get(label, 0) + 1
    
    # Get top n_speakers
    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:n_speakers]
    top_speaker_labels = [label for label, _ in top_speakers]
    
    # Collect samples
    selected_embeddings = []
    selected_labels = []
    selected_speaker_ids = []
    
    for speaker_label in top_speaker_labels:
        # Get files for this speaker
        speaker_files = [f for f, lbl in labels_dict.items() if lbl == speaker_label]
        
        # Randomly select n_samples_per_speaker files
        if len(speaker_files) > n_samples_per_speaker:
            speaker_files = random.sample(speaker_files, n_samples_per_speaker)
        
        for filepath in speaker_files:
            selected_embeddings.append(embeddings_dict[filepath])
            selected_labels.append(speaker_label)
            selected_speaker_ids.append(f"Speaker_{speaker_label}")
    
    selected_embeddings = np.array(selected_embeddings)
    selected_labels = np.array(selected_labels)
    
    # Reduce dimensionality with PCA first (optional, for faster t-SNE)
    if selected_embeddings.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        selected_embeddings = pca.fit_transform(selected_embeddings)
        logging.info(f"PCA reduced to {selected_embeddings.shape[1]} dimensions")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(selected_embeddings)
    
    # Create color map
    unique_speakers = np.unique(selected_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_speakers)))
    color_map = {speaker: colors[i] for i, speaker in enumerate(unique_speakers)}
    
    plt.figure(figsize=(14, 10))
    
    # Plot each speaker with different color
    for speaker in unique_speakers:
        mask = selected_labels == speaker
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[color_map[speaker]], label=f'Speaker {speaker}', 
                   s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    plt.title(f't-SNE Visualization of Speaker Embeddings\n({n_speakers} speakers, {n_samples_per_speaker} samples each)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    
    # Add legend (might be too many speakers, so optionally reduce)
    if len(unique_speakers) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # Just show that different colors represent different speakers
        plt.text(0.02, 0.98, f'Colors represent {len(unique_speakers)} different speakers',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    
    # Add explanation text
    explanation = "t-SNE shows how well embeddings from same speaker cluster together.\n"
    explanation += "Good separation: Speakers form distinct, tight clusters.\n"
    explanation += "Poor separation: Speakers overlap significantly."
    
    plt.figtext(0.02, 0.02, explanation, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"t-SNE plot saved to: {output_path}")
    
    return embeddings_2d, selected_labels

def plot_det_curve(fpr, fnr, eer, output_path):
    """Plot DET curve"""
    plt.figure(figsize=(10, 8))
    
    # Plot DET curve (log scale)
    plt.plot(fpr, fnr, 'b-', linewidth=2, label=f'DET curve (EER = {eer:.2f}%)')
    
    # Mark EER point
    eer_idx = np.argmin(np.abs(fpr - fnr))
    plt.plot(fpr[eer_idx], fnr[eer_idx], 'ro', markersize=10, 
             label=f'EER point ({fpr[eer_idx]*100:.1f}%, {fnr[eer_idx]*100:.1f}%)')
    
    # Set log scale for better visualization
    plt.xscale('log')
    plt.yscale('log')
    
    # Set reasonable limits
    plt.xlim([1e-4, 1])
    plt.ylim([1e-4, 1])
    
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
    plt.title('Detection Error Tradeoff (DET) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    
    # Add diagonal reference line
    plt.plot([1e-4, 1], [1e-4, 1], 'k--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"DET plot saved to: {output_path}")

def save_metrics_to_csv(eer, min_dcf, thresholds_at_eer, thresholds_at_minDCF, output_path):
    """Save all metrics to CSV file"""
    metrics = {
        'Metric': ['EER (%)', 'minDCF (p=0.01)', 'Threshold at EER', 'Threshold at minDCF'],
        'Value': [eer, min_dcf, thresholds_at_eer, thresholds_at_minDCF],
        'Description': [
            'Equal Error Rate - point where FAR = FRR',
            'Minimum Detection Cost Function at p_target=0.01',
            'Threshold value that achieves EER',
            'Threshold value that achieves minDCF'
        ]
    }
    
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
    logging.info(f"Metrics saved to: {output_path}")
    
    # Also save as text file for easy reading
    txt_path = output_path.replace('.csv', '.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("QUANTUM-ENHANCED ECAPA-TDNN EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Evaluation Date: {pd.Timestamp.now()}\n\n")
        f.write(f"EER: {eer:.4f} %\n")
        f.write(f"minDCF (p_target=0.01): {min_dcf:.6f}\n")
        f.write(f"Threshold at EER: {thresholds_at_eer:.6f}\n")
        f.write(f"Threshold at minDCF: {thresholds_at_minDCF:.6f}\n\n")
        f.write("=" * 60 + "\n")
    
    return df

# ----------------- Main Evaluation Function -----------------
def main():
    """Main evaluation function"""
    logging.info("=" * 60)
    logging.info("STARTING EVALUATION OF QUANTUM-ENHANCED ECAPA-TDNN")
    logging.info("=" * 60)
    
    # Load model
    model, checkpoint = load_model(MODEL_PATH)
    model = model.to(DEVICE)
    
    # Load test data
    test_dir = os.path.join(DATASET_ROOT, "test")
    logging.info(f"Loading test data from: {test_dir}")
    
    test_samples = create_clean_dataset_list(test_dir, suffix=".npy", quality_threshold=70)
    if not test_samples:
        logging.error("No test samples found!")
        return
    
    test_dataset = SimpleListDataset(test_samples)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=eval_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    logging.info(f"Test dataset: {len(test_dataset)} samples from {len(test_dataset.get_speaker_labels())} speakers")
    
    # Extract embeddings
    embeddings_dict, labels_dict, filepaths_list = extract_all_embeddings(model, test_loader)
    
    if len(embeddings_dict) < 2:
        logging.error("Not enough embeddings extracted for evaluation!")
        return
    
    # Compute scores
    scores, y_true = compute_scores(embeddings_dict, labels_dict, num_pairs=NUM_PAIRS_FOR_EER)
    
    # Compute metrics
    eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold = compute_eer_minDCF(scores, y_true)
    
    # Save metrics
    logging.info("=" * 60)
    logging.info("EVALUATION RESULTS:")
    logging.info(f"EER: {eer:.4f} %")
    logging.info(f"minDCF (p_target=0.01): {min_dcf:.6f}")
    logging.info(f"Threshold at EER: {eer_threshold:.6f}")
    logging.info(f"Threshold at minDCF: {min_dcf_threshold:.6f}")
    logging.info("=" * 60)
    
    # Save metrics to CSV
    metrics_df = save_metrics_to_csv(
        eer, min_dcf, eer_threshold, min_dcf_threshold,
        os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    )
    
    # Generate plots
    plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, 
                   os.path.join(OUTPUT_DIR, "eer_curve.png"))
    
    plot_roc_curve(fpr, 1-fnr, eer, 
                   os.path.join(OUTPUT_DIR, "roc_curve.png"))
    
    plot_det_curve(fpr, fnr, eer, 
                   os.path.join(OUTPUT_DIR, "det_curve.png"))
    
    plot_score_distributions(scores, y_true, eer_threshold, 
                            os.path.join(OUTPUT_DIR, "score_distributions.png"))
    
    # Generate t-SNE plot
    embeddings_2d, tsne_labels = plot_tsne(
        embeddings_dict, labels_dict, 
        os.path.join(OUTPUT_DIR, "tsne_visualization.png"),
        n_speakers=TSNE_SPEAKERS,
        n_samples_per_speaker=TSNE_SAMPLES_PER_SPEAKER
    )
    
    # Create a summary report
    create_summary_report(eer, min_dcf, len(test_dataset), OUTPUT_DIR)
    
    logging.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logging.info("=" * 60)
    logging.info("EVALUATION COMPLETE!")
    logging.info("=" * 60)

def create_summary_report(eer, min_dcf, num_samples, output_dir):
    """Create a comprehensive summary report"""
    report_path = os.path.join(output_dir, "summary_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Quantum-Enhanced ECAPA-TDNN Evaluation Report\n\n")
        f.write(f"**Evaluation Date:** {pd.Timestamp.now()}\n\n")
        f.write(f"**Model:** {MODEL_PATH}\n\n")
        
        f.write("## Key Metrics\n\n")
        f.write("| Metric | Value | Description |\n")
        f.write("|--------|-------|-------------|\n")
        f.write(f"| EER | {eer:.4f} % | Equal Error Rate (lower is better) |\n")
        f.write(f"| minDCF | {min_dcf:.6f} | Minimum Detection Cost Function at p_target=0.01 |\n")
        f.write(f"| Test Samples | {num_samples} | Number of test samples evaluated |\n\n")
        
        f.write("## Generated Plots\n\n")
        f.write("1. **EER Curve** (`eer_curve.png`) - FAR vs FRR with EER point marked\n")
        f.write("2. **ROC Curve** (`roc_curve.png`) - Receiver Operating Characteristic curve\n")
        f.write("3. **DET Curve** (`det_curve.png`) - Detection Error Tradeoff curve\n")
        f.write("4. **Score Distributions** (`score_distributions.png`) - Genuine vs Impostor score distributions\n")
        f.write("5. **t-SNE Visualization** (`tsne_visualization.png`) - 2D projection of speaker embeddings\n\n")
        
        f.write("## Interpretation\n\n")
        f.write(f"- **EER of {eer:.2f}%**: ")
        if eer < 5:
            f.write("Excellent performance\n")
        elif eer < 10:
            f.write("Good performance\n")
        elif eer < 20:
            f.write("Moderate performance\n")
        else:
            f.write("Needs improvement\n")
        
        f.write(f"- **minDCF of {min_dcf:.4f}: ")
        if min_dcf < 0.1:
            f.write("Excellent performance\n")
        elif min_dcf < 0.2:
            f.write("Good performance\n")
        elif min_dcf < 0.3:
            f.write("Moderate performance\n")
        else:
            f.write("Needs improvement\n")
        
        f.write("\n## Files Generated\n\n")
        f.write("- `evaluation_metrics.csv` - CSV file with all metrics\n")
        f.write("- `evaluation_metrics.txt` - Text version of metrics\n")
        f.write("- `evaluation.log` - Detailed log of evaluation process\n")
        f.write("- `summary_report.md` - This summary report\n")
        f.write("- Various plot files (.png)\n")
    
    logging.info(f"Summary report saved to: {report_path}")

# ----------------- Run Evaluation -----------------
if __name__ == "__main__":
    main()