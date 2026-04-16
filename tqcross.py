# evaluate_model_crosslingual.py
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
from collections import defaultdict

# ----------------- CONFIGURATION -----------------
MODEL_PATH = "/home/user2/VOIP/finetune_checkpoints_qecapa_voip_atharva/best_model_eer_8.68.pt"
DATASET_ROOT = "/home/user2/VOIP/VOIP_Mel_Features"
OUTPUT_DIR = "/home/user2/VOIP/model_evaluation_crosslingual_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quantum configuration
NUM_QUBITS = 6
DEV = qml.device("lightning.qubit", wires=NUM_QUBITS)

# Evaluation parameters
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_PAIRS_PER_CONDITION = 5000  # Pairs per condition (EN-EN, EN-Regional, etc.)
TSNE_SPEAKERS = 20
TSNE_SAMPLES_PER_SPEAKER = 10

# Language categories
ENGLISH_LANG = "EN"
# All non-EN languages are considered Regional (BN, HN, TE, etc.)

# Create main output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories for each condition
CONDITIONS = ["EN-EN", "EN-Regional", "Regional-EN", "Regional-Regional"]
for condition in CONDITIONS:
    os.makedirs(os.path.join(OUTPUT_DIR, condition), exist_ok=True)

# ----------------- Setup Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "evaluation_crosslingual.log")),
        logging.StreamHandler()
    ]
)

# ----------------- Quantum Circuit -----------------
@qml.qnode(DEV, interface="torch")
def quantum_circuit(inputs, weights):
    """Quantum circuit for feature processing"""
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

# ----------------- Model Definition -----------------
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
        
        weight_shapes = {"weights": (3, num_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
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

def extract_language_from_path(filepath):
    """Extract language code from file path - EN is English, all others are Regional"""
    # Expected format: .../speaker_id/LANG/file.npy
    parts = filepath.split(os.sep)
    
    # Find the language folder (should be before the filename)
    for i, part in enumerate(parts):
        # If we find EN, it's English
        if part == ENGLISH_LANG:
            return ENGLISH_LANG
        # Any other 2-3 letter folder code is Regional
        elif len(part) == 2 or len(part) == 3:
            # Check if it looks like a language code (all uppercase)
            if part.isupper() and part.isalpha():
                return "Regional"
    
    logging.warning(f"Could not extract language from path: {filepath}")
    return None

def create_language_separated_dataset(dir_path, suffix=".npy", quality_threshold=70):
    """Create dataset list separated by language (EN vs Regional)"""
    en_samples = []
    regional_samples = []
    
    corrupted_count = 0
    total_count = 0
    
    speaker_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    
    for speaker_id in tqdm(speaker_dirs, desc="Processing speakers"):
        speaker_path = os.path.join(dir_path, speaker_id)
        
        # Process each language folder
        for lang_folder in os.listdir(speaker_path):
            lang_path = os.path.join(speaker_path, lang_folder)
            
            if not os.path.isdir(lang_path):
                continue
            
            # Get all .npy files in this language folder
            npy_files = glob.glob(os.path.join(lang_path, f"*{suffix}"))
            
            for npy_path in npy_files:
                total_count += 1
                quality_score, _, _, _ = analyze_file_quality(npy_path)
                
                if quality_score >= quality_threshold:
                    language = extract_language_from_path(npy_path)
                    
                    if language == ENGLISH_LANG:
                        en_samples.append((npy_path, speaker_id, language))
                    elif language == "Regional":
                        # Store actual language code for reference but treat as Regional
                        actual_lang = lang_folder  # The actual folder name (BN, HN, TE, etc.)
                        regional_samples.append((npy_path, speaker_id, actual_lang))
                    else:
                        logging.warning(f"Could not determine language for file: {npy_path}")
                else:
                    corrupted_count += 1
    
    logging.info(f"Dataset filtering: {len(en_samples) + len(regional_samples)}/{total_count} files passed quality threshold")
    logging.info(f"EN samples: {len(en_samples)}")
    logging.info(f"Regional samples: {len(regional_samples)}")
    logging.info(f"Corrupted files skipped: {corrupted_count}")
    
    return en_samples, regional_samples

def eval_collate_fn(batch):
    """Collate function for evaluation"""
    features, labels, languages, filepaths = [], [], [], []
    max_len = 0
    skipped_count = 0
    
    for npy_path, speaker_id, lang in batch:
        feat = aggressive_feature_cleaning(npy_path, quality_threshold=70)
        
        if feat is None:
            skipped_count += 1
            continue
            
        if feat.shape[0] > max_len:
            max_len = feat.shape[0]
            
        features.append(feat)
        labels.append(speaker_id)
        languages.append(lang)
        filepaths.append(npy_path)
    
    if skipped_count > 0:
        logging.debug(f"Skipped {skipped_count} eval files due to quality issues")
    
    if not features:
        return None, None, None, None

    padded_features = []
    for feat in features:
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='wrap')
        padded_features.append(feat)
    
    features_tensor = torch.FloatTensor(np.array(padded_features))
    return features_tensor, labels, languages, filepaths

# ----------------- Load Model -----------------
def load_model(model_path):
    """Load the trained model"""
    logging.info(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = QECAPA_TDNN(in_channels=80, channels=512, embd_dim=192).to(DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logging.info("Model loaded successfully")
    
    return model, checkpoint

# ----------------- Extract Embeddings -----------------
def extract_embeddings_by_language(model, en_samples, regional_samples, batch_size=64):
    """Extract embeddings separated by language"""
    model.eval()
    
    en_embeddings = {}
    regional_embeddings = {}
    
    # Process EN samples
    logging.info("Extracting embeddings for EN samples...")
    if en_samples:
        en_dataset = SimpleLanguageDataset(en_samples)
        en_loader = torch.utils.data.DataLoader(
            en_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=eval_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
        )
        
        with torch.no_grad():
            for batch in tqdm(en_loader, desc="EN embeddings"):
                if batch[0] is None:
                    continue
                features, labels, languages, filepaths = batch
                features = features.to(DEVICE)
                
                batch_embeddings = model(features)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                for i, filepath in enumerate(filepaths):
                    en_embeddings[filepath] = {
                        'embedding': batch_embeddings[i].cpu().numpy(),
                        'speaker': labels[i],
                        'language': languages[i]
                    }
    
    # Process Regional samples
    logging.info("Extracting embeddings for Regional samples...")
    if regional_samples:
        regional_dataset = SimpleLanguageDataset(regional_samples)
        regional_loader = torch.utils.data.DataLoader(
            regional_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=eval_collate_fn, num_workers=NUM_WORKERS, pin_memory=True
        )
        
        with torch.no_grad():
            for batch in tqdm(regional_loader, desc="Regional embeddings"):
                if batch[0] is None:
                    continue
                features, labels, languages, filepaths = batch
                features = features.to(DEVICE)
                
                batch_embeddings = model(features)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                for i, filepath in enumerate(filepaths):
                    regional_embeddings[filepath] = {
                        'embedding': batch_embeddings[i].cpu().numpy(),
                        'speaker': labels[i],
                        'language': languages[i]
                    }
    
    logging.info(f"Extracted {len(en_embeddings)} EN embeddings")
    logging.info(f"Extracted {len(regional_embeddings)} Regional embeddings")
    
    return en_embeddings, regional_embeddings

class SimpleLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        """items: list of (filepath, speaker_id, language)"""
        self.data = items
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ----------------- Compute Scores for Cross-Lingual Testing -----------------
def compute_crosslingual_scores(enroll_embeddings, test_embeddings, num_pairs=5000):
    """
    Compute similarity scores for cross-lingual verification
    enroll_embeddings: dict of enrollment embeddings
    test_embeddings: dict of test embeddings
    """
    logging.info(f"Computing {num_pairs} pairs...")
    
    enroll_files = list(enroll_embeddings.keys())
    test_files = list(test_embeddings.keys())
    
    scores = []
    y_true = []
    
    # Generate genuine and impostor pairs
    genuine_count = num_pairs // 2
    impostor_count = num_pairs // 2
    
    # Genuine pairs (same speaker, different language potentially)
    for _ in tqdm(range(genuine_count), desc="Genuine pairs"):
        # Select enrollment file
        enroll_file = random.choice(enroll_files)
        enroll_speaker = enroll_embeddings[enroll_file]['speaker']
        
        # Find test files from same speaker
        same_speaker_test = [f for f in test_files 
                            if test_embeddings[f]['speaker'] == enroll_speaker]
        
        if len(same_speaker_test) == 0:
            continue
        
        test_file = random.choice(same_speaker_test)
        
        # Compute similarity
        emb1 = enroll_embeddings[enroll_file]['embedding']
        emb2 = test_embeddings[test_file]['embedding']
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        scores.append(score)
        y_true.append(1)  # Genuine
    
    # Impostor pairs (different speakers)
    for _ in tqdm(range(impostor_count), desc="Impostor pairs"):
        enroll_file = random.choice(enroll_files)
        enroll_speaker = enroll_embeddings[enroll_file]['speaker']
        
        # Find test files from different speakers
        diff_speaker_test = [f for f in test_files 
                            if test_embeddings[f]['speaker'] != enroll_speaker]
        
        if len(diff_speaker_test) == 0:
            continue
        
        test_file = random.choice(diff_speaker_test)
        
        # Compute similarity
        emb1 = enroll_embeddings[enroll_file]['embedding']
        emb2 = test_embeddings[test_file]['embedding']
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        scores.append(score)
        y_true.append(0)  # Impostor
    
    return np.array(scores), np.array(y_true)

def compute_eer_minDCF(scores, y_true, p_target=0.01, c_miss=1, c_fa=1):
    """Compute EER and minDCF"""
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find EER
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = eer * 100
    except:
        eer = eer * 100
    
    # Compute minDCF
    dcf_costs = p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr
    min_dcf = np.min(dcf_costs)
    min_dcf_threshold = thresholds[np.argmin(dcf_costs)]
    
    return eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold

# ----------------- Plotting Functions -----------------
def plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, output_path, title_suffix=""):
    """Plot FAR and FRR curves with EER point"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, fpr, 'b-', linewidth=2, label='FAR (False Acceptance Rate)')
    plt.plot(thresholds, fnr, 'r-', linewidth=2, label='FRR (False Rejection Rate)')
    
    eer_idx = np.argmin(np.abs(fpr - fnr))
    plt.plot(thresholds[eer_idx], eer/100, 'ko', markersize=10, 
             label=f'EER = {eer:.2f}% (threshold = {eer_threshold:.3f})')
    
    plt.axvline(x=eer_threshold, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=eer/100, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title(f'DET Curve - FAR vs FRR {title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-1.0, 1.0])
    plt.ylim([0, 1.0])
    
    textstr = f'EER = {eer:.2f}%\nThreshold = {eer_threshold:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_det_curve(fpr, fnr, eer, output_path, title_suffix=""):
    """
    Plot DET (Detection Error Tradeoff) curve on a logarithmic scale
    DET curve plots FPR (False Positive Rate) vs FNR (False Negative Rate)
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to percentages and handle zeros for log scale
    fpr_percent = np.maximum(fpr * 100, 1e-5)  # Avoid log(0)
    fnr_percent = np.maximum(fnr * 100, 1e-5)
    
    # Plot DET curve
    plt.plot(fpr_percent, fnr_percent, 'b-', linewidth=2, label='DET curve')
    
    # Mark EER point
    eer_idx = np.argmin(np.abs(fpr - fnr))
    plt.plot(fpr_percent[eer_idx], fnr_percent[eer_idx], 'ro', markersize=10, 
             label=f'EER = {eer:.2f}%')
    
    # Add diagonal line (EER line)
    max_val = max(np.max(fpr_percent), np.max(fnr_percent))
    diag = np.linspace(0.01, max_val, 100)
    plt.plot(diag, diag, 'k--', linewidth=1, alpha=0.5, label='EER line')
    
    # Set log scale for both axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Labels and title
    plt.xlabel('False Positive Rate (FPR) [%]', fontsize=12)
    plt.ylabel('False Negative Rate (FNR) [%]', fontsize=12)
    plt.title(f'DET Curve (Log Scale) {title_suffix}', fontsize=14, fontweight='bold')
    
    # Grid and legend
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(loc='upper right', fontsize=10)
    
    # Axis limits
    plt.xlim([0.01, 100])
    plt.ylim([0.01, 100])
    
    # Add text box with EER
    textstr = f'EER = {eer:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distributions(scores, y_true, eer_threshold, output_path, title_suffix=""):
    """Plot score distributions"""
    genuine_scores = scores[y_true == 1]
    impostor_scores = scores[y_true == 0]
    
    plt.figure(figsize=(12, 6))
    
    bins = np.linspace(-1, 1, 101)
    
    plt.hist(genuine_scores, bins=bins, alpha=0.7, color='green', 
             label=f'Genuine (n={len(genuine_scores)})', density=True)
    plt.hist(impostor_scores, bins=bins, alpha=0.7, color='red', 
             label=f'Impostor (n={len(impostor_scores)})', density=True)
    
    from scipy.stats import gaussian_kde
    if len(genuine_scores) > 1:
        kde_gen = gaussian_kde(genuine_scores)
        x_plot = np.linspace(-1, 1, 1000)
        plt.plot(x_plot, kde_gen(x_plot), 'g-', linewidth=2, label='Genuine KDE')
    
    if len(impostor_scores) > 1:
        kde_imp = gaussian_kde(impostor_scores)
        plt.plot(x_plot, kde_imp(x_plot), 'r-', linewidth=2, label='Impostor KDE')
    
    plt.axvline(x=eer_threshold, color='k', linestyle='--', linewidth=2, 
                label=f'EER threshold = {eer_threshold:.3f}')
    
    plt.xlabel('Cosine Similarity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Score Distributions {title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-1, 1])
    
    stats_text = (f'Genuine: μ={np.mean(genuine_scores):.3f}, σ={np.std(genuine_scores):.3f}\n'
                  f'Impostor: μ={np.mean(impostor_scores):.3f}, σ={np.std(impostor_scores):.3f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(fpr, tpr, eer, output_path, title_suffix=""):
    """Plot ROC curve"""
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (EER = {eer:.2f}%)')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=10, 
             label=f'EER point ({fpr[eer_idx]*100:.1f}%, {tpr[eer_idx]*100:.1f}%)')
    
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve {title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    
    plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
    
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    textstr = f'AUC = {roc_auc:.4f}\nEER = {eer:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.6, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_to_csv(condition, eer, min_dcf, eer_threshold, min_dcf_threshold, output_dir):
    """Save metrics to CSV"""
    metrics = {
        'Metric': ['EER (%)', 'minDCF (p=0.01)', 'Threshold at EER', 'Threshold at minDCF'],
        'Value': [eer, min_dcf, eer_threshold, min_dcf_threshold],
    }
    
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    
    txt_path = os.path.join(output_dir, "metrics.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"CROSS-LINGUAL EVALUATION: {condition}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"EER: {eer:.4f} %\n")
        f.write(f"minDCF (p_target=0.01): {min_dcf:.6f}\n")
        f.write(f"Threshold at EER: {eer_threshold:.6f}\n")
        f.write(f"Threshold at minDCF: {min_dcf_threshold:.6f}\n\n")
        f.write("=" * 60 + "\n")
    
    return df

# ----------------- Create Summary Matrix -----------------
def create_summary_matrix(results_dict, output_dir):
    """Create 2x2 matrix visualization of all results"""
    
    # Create results matrix
    matrix_data = {
        'Enrollment': ['EN', 'Regional'],
        'Test EN': [
            results_dict['EN-EN']['eer'],
            results_dict['Regional-EN']['eer']
        ],
        'Test Regional': [
            results_dict['EN-Regional']['eer'],
            results_dict['Regional-Regional']['eer']
        ]
    }
    
    df = pd.DataFrame(matrix_data)
    
    # Save as CSV
    df.to_csv(os.path.join(output_dir, "summary_matrix.csv"), index=False)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    matrix_values = np.array([
        [results_dict['EN-EN']['eer'], results_dict['EN-Regional']['eer']],
        [results_dict['Regional-EN']['eer'], results_dict['Regional-Regional']['eer']]
    ])
    
    sns.heatmap(matrix_values, annot=True, fmt='.2f', cmap='RdYlGn_r',
                xticklabels=['Test EN', 'Test Regional'],
                yticklabels=['Enroll EN', 'Enroll Regional'],
                cbar_kws={'label': 'EER (%)'},
                vmin=0, vmax=max(20, matrix_values.max()),
                linewidths=2, linecolor='black')
    
    plt.title('Cross-Lingual Speaker Verification: EER (%) Matrix', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Test Language', fontsize=12, fontweight='bold')
    plt.ylabel('Enrollment Language', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed comparison bar plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # EER comparison
    conditions = ['EN-EN', 'EN-Regional', 'Regional-EN', 'Regional-Regional']
    eers = [results_dict[c]['eer'] for c in conditions]
    min_dcfs = [results_dict[c]['min_dcf'] for c in conditions]
    
    colors = ['#2ecc71', '#f39c12', '#f39c12', '#3498db']
    
    axes[0].bar(conditions, eers, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('EER (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Equal Error Rate Comparison', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticklabels(conditions, rotation=45, ha='right')
    
    for i, (cond, eer) in enumerate(zip(conditions, eers)):
        axes[0].text(i, eer + 0.2, f'{eer:.2f}%', ha='center', fontweight='bold')
    
    # minDCF comparison
    axes[1].bar(conditions, min_dcfs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('minDCF', fontsize=12, fontweight='bold')
    axes[1].set_title('Minimum Detection Cost Function Comparison', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticklabels(conditions, rotation=45, ha='right')
    
    for i, (cond, dcf) in enumerate(zip(conditions, min_dcfs)):
        axes[1].text(i, dcf + 0.01, f'{dcf:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Summary matrix saved to: {output_dir}")
    
    return df

def create_comprehensive_report(results_dict, output_dir):
    """Create comprehensive markdown report"""
    report_path = os.path.join(output_dir, "comprehensive_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Cross-Lingual Speaker Verification - Comprehensive Evaluation Report\n\n")
        f.write(f"**Evaluation Date:** {pd.Timestamp.now()}\n\n")
        f.write(f"**Model:** {MODEL_PATH}\n\n")
        
        f.write("## Experiment Design\n\n")
        f.write("This evaluation performs **cross-lingual speaker verification** with a 2x2 matrix:\n\n")
        f.write("- **EN**: English language samples\n")
        f.write("- **Regional**: All non-English languages (BN, HN, TE, etc.)\n\n")
        f.write("### Test Conditions:\n")
        f.write("- **EN-EN**: Enroll with English, Test with English (within-language)\n")
        f.write("- **EN-Regional**: Enroll with English, Test with Regional (cross-lingual)\n")
        f.write("- **Regional-EN**: Enroll with Regional, Test with English (cross-lingual)\n")
        f.write("- **Regional-Regional**: Enroll with Regional, Test with Regional (within-language)\n\n")
        
        f.write("## Summary Results\n\n")
        f.write("| Condition | EER (%) | minDCF | Threshold (EER) | Threshold (minDCF) |\n")
        f.write("|-----------|---------|--------|-----------------|--------------------|\n")
        
        for condition in CONDITIONS:
            res = results_dict[condition]
            f.write(f"| {condition} | {res['eer']:.4f} | {res['min_dcf']:.6f} | "
                   f"{res['eer_threshold']:.4f} | {res['min_dcf_threshold']:.4f} |\n")
        
        f.write("\n## Analysis\n\n")
        
        # Find best and worst conditions
        best_cond = min(CONDITIONS, key=lambda c: results_dict[c]['eer'])
        worst_cond = max(CONDITIONS, key=lambda c: results_dict[c]['eer'])
        
        f.write(f"### Best Performance\n")
        f.write(f"- **{best_cond}**: EER = {results_dict[best_cond]['eer']:.2f}%\n\n")
        
        f.write(f"### Worst Performance\n")
        f.write(f"- **{worst_cond}**: EER = {results_dict[worst_cond]['eer']:.2f}%\n\n")
        
        # Cross-lingual degradation
        within_avg = (results_dict['EN-EN']['eer'] + results_dict['Regional-Regional']['eer']) / 2
        cross_avg = (results_dict['EN-Regional']['eer'] + results_dict['Regional-EN']['eer']) / 2
        degradation = ((cross_avg - within_avg) / within_avg) * 100
        
        f.write(f"### Cross-Lingual Performance\n")
        f.write(f"- **Within-Language Average EER**: {within_avg:.2f}%\n")
        f.write(f"- **Cross-Lingual Average EER**: {cross_avg:.2f}%\n")
        f.write(f"- **Performance Degradation**: {degradation:.2f}%\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("The following plots are generated for each condition:\n\n")
        f.write("1. **EER Curve** - FAR vs FRR with EER point\n")
        f.write("2. **DET Curve** - Detection Error Tradeoff (log scale)\n")
        f.write("3. **ROC Curve** - Receiver Operating Characteristic\n")
        f.write("4. **Score Distributions** - Genuine vs Impostor scores\n\n")
        
        f.write("Additional summary visualizations:\n\n")
        f.write("- **Summary Heatmap** - 2x2 matrix of EER values\n")
        f.write("- **Summary Comparison** - Bar charts comparing all conditions\n\n")
        
        f.write("## Interpretation Guidelines\n\n")
        f.write("- **EER < 5%**: Excellent performance\n")
        f.write("- **EER 5-10%**: Good performance\n")
        f.write("- **EER 10-20%**: Moderate performance\n")
        f.write("- **EER > 20%**: Needs improvement\n\n")
        
        f.write("## Files Generated\n\n")
        for condition in CONDITIONS:
            f.write(f"### {condition}/\n")
            f.write(f"- `metrics.csv` - Detailed metrics\n")
            f.write(f"- `metrics.txt` - Human-readable metrics\n")
            f.write(f"- `eer_curve.png` - EER visualization\n")
            f.write(f"- `det_curve.png` - DET curve (log scale)\n")
            f.write(f"- `roc_curve.png` - ROC curve\n")
            f.write(f"- `score_distributions.png` - Score histograms\n\n")
        
        f.write("### Root Directory\n")
        f.write("- `summary_matrix.csv` - 2x2 results matrix\n")
        f.write("- `summary_heatmap.png` - Heatmap visualization\n")
        f.write("- `summary_comparison.png` - Bar chart comparisons\n")
        f.write("- `comprehensive_report.md` - This report\n")
    
    logging.info(f"Comprehensive report saved to: {report_path}")

# ----------------- Main Evaluation Function -----------------
def main():
    """Main evaluation function"""
    logging.info("=" * 80)
    logging.info("STARTING CROSS-LINGUAL EVALUATION OF QUANTUM-ENHANCED ECAPA-TDNN")
    logging.info("=" * 80)
    
    # Load model
    model, checkpoint = load_model(MODEL_PATH)
    model = model.to(DEVICE)
    
    # Load test data separated by language
    test_dir = os.path.join(DATASET_ROOT, "test")
    logging.info(f"Loading test data from: {test_dir}")
    
    en_samples, regional_samples = create_language_separated_dataset(
        test_dir, suffix=".npy", quality_threshold=70
    )
    
    if not en_samples or not regional_samples:
        logging.error("Insufficient data! Need both EN and Regional samples.")
        return
    
    # Extract embeddings
    en_embeddings, regional_embeddings = extract_embeddings_by_language(
        model, en_samples, regional_samples, batch_size=BATCH_SIZE
    )
    
    # Store results
    results_dict = {}
    
    # ==================== EN-EN ====================
    logging.info("\n" + "=" * 80)
    logging.info("CONDITION 1: EN-EN (Enroll EN, Test EN)")
    logging.info("=" * 80)
    
    scores, y_true = compute_crosslingual_scores(
        en_embeddings, en_embeddings, num_pairs=NUM_PAIRS_PER_CONDITION
    )
    
    eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold = compute_eer_minDCF(scores, y_true)
    
    results_dict['EN-EN'] = {
        'eer': eer,
        'min_dcf': min_dcf,
        'eer_threshold': eer_threshold,
        'min_dcf_threshold': min_dcf_threshold
    }
    
    condition_dir = os.path.join(OUTPUT_DIR, "EN-EN")
    save_metrics_to_csv('EN-EN', eer, min_dcf, eer_threshold, min_dcf_threshold, condition_dir)
    plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, 
                   os.path.join(condition_dir, "eer_curve.png"), "(EN-EN)")
    plot_det_curve(fpr, fnr, eer,
                   os.path.join(condition_dir, "det_curve.png"), "(EN-EN)")
    plot_roc_curve(fpr, 1-fnr, eer, 
                   os.path.join(condition_dir, "roc_curve.png"), "(EN-EN)")
    plot_score_distributions(scores, y_true, eer_threshold, 
                            os.path.join(condition_dir, "score_distributions.png"), "(EN-EN)")
    
    logging.info(f"EN-EN Results: EER = {eer:.4f}%, minDCF = {min_dcf:.6f}")
    
    # ==================== EN-Regional ====================
    logging.info("\n" + "=" * 80)
    logging.info("CONDITION 2: EN-Regional (Enroll EN, Test Regional)")
    logging.info("=" * 80)
    
    scores, y_true = compute_crosslingual_scores(
        en_embeddings, regional_embeddings, num_pairs=NUM_PAIRS_PER_CONDITION
    )
    
    eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold = compute_eer_minDCF(scores, y_true)
    
    results_dict['EN-Regional'] = {
        'eer': eer,
        'min_dcf': min_dcf,
        'eer_threshold': eer_threshold,
        'min_dcf_threshold': min_dcf_threshold
    }
    
    condition_dir = os.path.join(OUTPUT_DIR, "EN-Regional")
    save_metrics_to_csv('EN-Regional', eer, min_dcf, eer_threshold, min_dcf_threshold, condition_dir)
    plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, 
                   os.path.join(condition_dir, "eer_curve.png"), "(EN-Regional)")
    plot_det_curve(fpr, fnr, eer,
                   os.path.join(condition_dir, "det_curve.png"), "(EN-Regional)")
    plot_roc_curve(fpr, 1-fnr, eer, 
                   os.path.join(condition_dir, "roc_curve.png"), "(EN-Regional)")
    plot_score_distributions(scores, y_true, eer_threshold, 
                            os.path.join(condition_dir, "score_distributions.png"), "(EN-Regional)")
    
    logging.info(f"EN-Regional Results: EER = {eer:.4f}%, minDCF = {min_dcf:.6f}")
    
    # ==================== Regional-EN ====================
    logging.info("\n" + "=" * 80)
    logging.info("CONDITION 3: Regional-EN (Enroll Regional, Test EN)")
    logging.info("=" * 80)
    
    scores, y_true = compute_crosslingual_scores(
        regional_embeddings, en_embeddings, num_pairs=NUM_PAIRS_PER_CONDITION
    )
    
    eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold = compute_eer_minDCF(scores, y_true)
    
    results_dict['Regional-EN'] = {
        'eer': eer,
        'min_dcf': min_dcf,
        'eer_threshold': eer_threshold,
        'min_dcf_threshold': min_dcf_threshold
    }
    
    condition_dir = os.path.join(OUTPUT_DIR, "Regional-EN")
    save_metrics_to_csv('Regional-EN', eer, min_dcf, eer_threshold, min_dcf_threshold, condition_dir)
    plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, 
                   os.path.join(condition_dir, "eer_curve.png"), "(Regional-EN)")
    plot_det_curve(fpr, fnr, eer,
                   os.path.join(condition_dir, "det_curve.png"), "(Regional-EN)")
    plot_roc_curve(fpr, 1-fnr, eer, 
                   os.path.join(condition_dir, "roc_curve.png"), "(Regional-EN)")
    plot_score_distributions(scores, y_true, eer_threshold, 
                            os.path.join(condition_dir, "score_distributions.png"), "(Regional-EN)")
    
    logging.info(f"Regional-EN Results: EER = {eer:.4f}%, minDCF = {min_dcf:.6f}")
    
    # ==================== Regional-Regional ====================
    logging.info("\n" + "=" * 80)
    logging.info("CONDITION 4: Regional-Regional (Enroll Regional, Test Regional)")
    logging.info("=" * 80)
    
    scores, y_true = compute_crosslingual_scores(
        regional_embeddings, regional_embeddings, num_pairs=NUM_PAIRS_PER_CONDITION
    )
    
    eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold = compute_eer_minDCF(scores, y_true)
    
    results_dict['Regional-Regional'] = {
        'eer': eer,
        'min_dcf': min_dcf,
        'eer_threshold': eer_threshold,
        'min_dcf_threshold': min_dcf_threshold
    }
    
    condition_dir = os.path.join(OUTPUT_DIR, "Regional-Regional")
    save_metrics_to_csv('Regional-Regional', eer, min_dcf, eer_threshold, min_dcf_threshold, condition_dir)
    plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, 
                   os.path.join(condition_dir, "eer_curve.png"), "(Regional-Regional)")
    plot_det_curve(fpr, fnr, eer,
                   os.path.join(condition_dir, "det_curve.png"), "(Regional-Regional)")
    plot_roc_curve(fpr, 1-fnr, eer, 
                   os.path.join(condition_dir, "roc_curve.png"), "(Regional-Regional)")
    plot_score_distributions(scores, y_true, eer_threshold, 
                            os.path.join(condition_dir, "score_distributions.png"), "(Regional-Regional)")
    
    logging.info(f"Regional-Regional Results: EER = {eer:.4f}%, minDCF = {min_dcf:.6f}")
    
    # ==================== Create Summary ====================
    logging.info("\n" + "=" * 80)
    logging.info("CREATING SUMMARY VISUALIZATIONS")
    logging.info("=" * 80)
    
    summary_df = create_summary_matrix(results_dict, OUTPUT_DIR)
    create_comprehensive_report(results_dict, OUTPUT_DIR)
    
    # Print final summary
    logging.info("\n" + "=" * 80)
    logging.info("FINAL RESULTS SUMMARY")
    logging.info("=" * 80)
    for condition in CONDITIONS:
        res = results_dict[condition]
        logging.info(f"{condition:20s}: EER = {res['eer']:6.2f}%, minDCF = {res['min_dcf']:.6f}")
    
    logging.info("\n" + "=" * 80)
    logging.info(f"All results saved to: {OUTPUT_DIR}")
    logging.info("=" * 80)
    logging.info("CROSS-LINGUAL EVALUATION COMPLETE!")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()