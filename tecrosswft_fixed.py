import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd
from tqdm import tqdm
import logging
from collections import defaultdict
import json
import warnings
warnings.filterwarnings("ignore")

# ----------------- CONFIGURATION -----------------
# Use SpeechBrain's official pre-trained model (most reliable)
USE_SPEECHBRAIN = True  # Set to True to use SpeechBrain directly
MODEL_PATH = "/home/user2/VOIP/pretrained_ecapa_voxceleb.pt"  # Optional fallback
DATASET_ROOT = "/home/user2/VOIP/VOIP_Mel_Features"
OUTPUT_DIR = "/home/user2/VOIP/ecapa_pretrained_crosslingual_evaluation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation parameters
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_PAIRS_PER_SCENARIO = 5000
TSNE_SPEAKERS = 15
TSNE_SAMPLES_PER_SPEAKER = 8

# Language configuration
ENGLISH_LANG = "EN"

# Create output directory structure
os.makedirs(OUTPUT_DIR, exist_ok=True)
for scenario in ["EN_EN", "EN_Regional", "Regional_EN", "Regional_Regional"]:
    os.makedirs(os.path.join(OUTPUT_DIR, scenario), exist_ok=True)

# ----------------- Setup Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "crosslingual_evaluation.log")),
        logging.StreamHandler()
    ]
)

# ----------------- Suppress SpeechBrain Warnings -----------------
# Workaround for torchaudio.list_audio_backends deprecation
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    # Create dummy function if it doesn't exist
    torchaudio.list_audio_backends = lambda: []

# ----------------- Model Definition (ECAPA-TDNN) -----------------
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

# ----------------- SpeechBrain Model Loader -----------------
def load_speechbrain_pretrained():
    """
    Load pre-trained ECAPA-TDNN from SpeechBrain with compatibility fixes
    """
    try:
        # Try importing speechbrain
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            logging.error("SpeechBrain not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "speechbrain"])
            from speechbrain.inference.speaker import EncoderClassifier
        
        logging.info("Loading pre-trained ECAPA-TDNN from SpeechBrain...")
        
        # Suppress specific warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
        
        # Load the model
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": str(DEVICE)}
        )
        
        logging.info("SpeechBrain model loaded successfully!")
        return classifier
        
    except Exception as e:
        logging.error(f"Error loading SpeechBrain model: {e}")
        return None

# ----------------- Wrapper for SpeechBrain Model -----------------
class SpeechBrainWrapper(nn.Module):
    """Wrapper to make SpeechBrain model compatible with our extraction function"""
    def __init__(self, sb_model):
        super().__init__()
        self.sb_model = sb_model
        self.embd_dim = 192
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, time, freq) or (batch, freq, time)
        Returns:
            embeddings: tensor of shape (batch, embd_dim)
        """
        with torch.no_grad():
            # SpeechBrain expects (batch, time, freq)
            if x.dim() == 3:
                if x.shape[1] != x.shape[2]:  # If not square, ensure correct format
                    if x.shape[1] == 80:  # If features are (batch, freq, time)
                        x = x.transpose(1, 2)  # Convert to (batch, time, freq)
            
            # Get embeddings
            embeddings = self.sb_model.encode_batch(x)
            
            # Squeeze and normalize
            embeddings = embeddings.squeeze(1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings

# ----------------- Alternative: Manual Download from HuggingFace -----------------
def download_from_huggingface():
    """
    Alternative: Download model from HuggingFace
    """
    try:
        from huggingface_hub import hf_hub_download
        
        logging.info("Downloading model from HuggingFace...")
        
        # Download ECAPA-TDNN model files
        model_path = hf_hub_download(
            repo_id="speechbrain/spkrec-ecapa-voxceleb",
            filename="embedding_model.ckpt",
            cache_dir="pretrained_models"
        )
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize our model architecture
        model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192)
        
        # Try to load state dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        logging.info("Model downloaded and loaded from HuggingFace!")
        return model
        
    except Exception as e:
        logging.error(f"Failed to download from HuggingFace: {e}")
        return None

# ----------------- Data Processing Functions -----------------
def analyze_file_quality(npy_path):
    """Analyze file quality and return quality score"""
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

def parse_file_info(filepath):
    """Parse file to extract speaker_id and language"""
    parts = filepath.split(os.sep)
    
    try:
        test_idx = parts.index('test')
        speaker_id = parts[test_idx + 1]
        language = parts[test_idx + 2]
        return speaker_id, language
    except (ValueError, IndexError):
        logging.warning(f"Could not parse file info from: {filepath}")
        return None, None

def organize_files_by_language(test_dir):
    """Organize test files by speaker and language"""
    speaker_lang_files = defaultdict(lambda: defaultdict(list))
    
    all_files = glob.glob(os.path.join(test_dir, "**", "*.npy"), recursive=True)
    
    for filepath in all_files:
        speaker_id, language = parse_file_info(filepath)
        if speaker_id and language:
            quality_score, _, _, _ = analyze_file_quality(filepath)
            if quality_score >= 70:
                speaker_lang_files[speaker_id][language].append(filepath)
    
    logging.info("=" * 60)
    logging.info("Dataset Organization:")
    total_speakers = len(speaker_lang_files)
    total_files = 0
    all_languages = set()
    
    for speaker_id, lang_dict in speaker_lang_files.items():
        speaker_total = sum(len(files) for files in lang_dict.values())
        total_files += speaker_total
        all_languages.update(lang_dict.keys())
        logging.info(f"  Speaker {speaker_id}: {speaker_total} files across {len(lang_dict)} languages")
    
    regional_languages = sorted([lang for lang in all_languages if lang != ENGLISH_LANG])
    logging.info(f"\nDetected Languages:")
    logging.info(f"  English: {ENGLISH_LANG}")
    logging.info(f"  Regional: {', '.join(regional_languages) if regional_languages else 'None'}")
    
    logging.info(f"\nTotal: {total_speakers} speakers, {total_files} files")
    logging.info("=" * 60)
    
    return speaker_lang_files

def extract_embedding(model, npy_path):
    """Extract embedding for a single file"""
    feat = aggressive_feature_cleaning(npy_path, quality_threshold=70)
    if feat is None:
        return None
    
    # Convert to tensor and add batch dimension
    feat_tensor = torch.FloatTensor(feat).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        embedding = model(feat_tensor)
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding.cpu().numpy()[0]

def extract_all_embeddings(model, speaker_lang_files):
    """Extract embeddings for all files organized by speaker and language"""
    embeddings_dict = defaultdict(lambda: defaultdict(dict))
    
    total_files = sum(len(files) for speaker_data in speaker_lang_files.values() 
                     for files in speaker_data.values())
    
    logging.info(f"Extracting embeddings for {total_files} files...")
    
    with tqdm(total=total_files, desc="Extracting embeddings") as pbar:
        for speaker_id, lang_dict in speaker_lang_files.items():
            for language, filepaths in lang_dict.items():
                for filepath in filepaths:
                    embedding = extract_embedding(model, filepath)
                    if embedding is not None:
                        embeddings_dict[speaker_id][language][filepath] = embedding
                    pbar.update(1)
    
    total_embeddings = sum(len(lang_emb) for speaker_emb in embeddings_dict.values() 
                          for lang_emb in speaker_emb.values())
    logging.info(f"Successfully extracted {total_embeddings} embeddings")
    
    return embeddings_dict

# ----------------- Cross-Lingual Evaluation Functions -----------------
def compute_cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def create_pairs_for_scenario(embeddings_dict, enroll_lang_type, test_lang_type, num_pairs):
    """
    Create pairs for a specific enrollment-test scenario
    """
    genuine_scores = []
    impostor_scores = []
    
    speakers = list(embeddings_dict.keys())
    
    def get_files_by_type(speaker_id, lang_type):
        files = []
        if lang_type == 'EN':
            if ENGLISH_LANG in embeddings_dict[speaker_id]:
                files = list(embeddings_dict[speaker_id][ENGLISH_LANG].keys())
        else:
            for lang in embeddings_dict[speaker_id].keys():
                if lang != ENGLISH_LANG:
                    files.extend(list(embeddings_dict[speaker_id][lang].keys()))
        return files
    
    # Generate genuine pairs
    genuine_count = 0
    attempts = 0
    max_attempts = num_pairs * 10
    
    while genuine_count < num_pairs // 2 and attempts < max_attempts:
        attempts += 1
        speaker_id = random.choice(speakers)
        enroll_files = get_files_by_type(speaker_id, enroll_lang_type)
        test_files = get_files_by_type(speaker_id, test_lang_type)
        
        if len(enroll_files) == 0 or len(test_files) == 0:
            continue
        
        enroll_file = random.choice(enroll_files)
        test_file = random.choice(test_files)
        
        if enroll_file == test_file and len(enroll_files) > 1:
            test_files_copy = [f for f in test_files if f != enroll_file]
            if test_files_copy:
                test_file = random.choice(test_files_copy)
        
        enroll_lang = parse_file_info(enroll_file)[1]
        test_lang = parse_file_info(test_file)[1]
        
        enroll_emb = embeddings_dict[speaker_id][enroll_lang][enroll_file]
        test_emb = embeddings_dict[speaker_id][test_lang][test_file]
        
        score = compute_cosine_similarity(enroll_emb, test_emb)
        genuine_scores.append(score)
        genuine_count += 1
    
    # Generate impostor pairs
    impostor_count = 0
    attempts = 0
    
    while impostor_count < num_pairs // 2 and attempts < max_attempts:
        attempts += 1
        
        if len(speakers) < 2:
            break
            
        speaker1, speaker2 = random.sample(speakers, 2)
        
        enroll_files = get_files_by_type(speaker1, enroll_lang_type)
        if len(enroll_files) == 0:
            continue
        enroll_file = random.choice(enroll_files)
        
        test_files = get_files_by_type(speaker2, test_lang_type)
        if len(test_files) == 0:
            continue
        test_file = random.choice(test_files)
        
        enroll_lang = parse_file_info(enroll_file)[1]
        test_lang = parse_file_info(test_file)[1]
        
        enroll_emb = embeddings_dict[speaker1][enroll_lang][enroll_file]
        test_emb = embeddings_dict[speaker2][test_lang][test_file]
        
        score = compute_cosine_similarity(enroll_emb, test_emb)
        impostor_scores.append(score)
        impostor_count += 1
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        logging.warning(f"Insufficient pairs for {enroll_lang_type}→{test_lang_type}")
        return None, None
    
    scores = np.array(genuine_scores + impostor_scores)
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    
    logging.info(f"{enroll_lang_type}→{test_lang_type}: {len(genuine_scores)} genuine, {len(impostor_scores)} impostor pairs")
    
    return scores, y_true

def compute_eer_minDCF(scores, y_true, p_target=0.01, c_miss=1, c_fa=1):
    """Compute EER and minDCF"""
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = eer * 100
    except:
        eer = eer * 100
    
    dcf_costs = p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr
    min_dcf = np.min(dcf_costs)
    min_dcf_threshold = thresholds[np.argmin(dcf_costs)]
    
    return eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold

# ----------------- Plotting Functions -----------------
def plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, scenario_name, output_path):
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
    plt.title(f'Pre-trained ECAPA: {scenario_name}\nFAR vs FRR', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-1.0, 1.0])
    plt.ylim([0, 1.0])
    
    textstr = f'Scenario: {scenario_name}\nEER = {eer:.2f}%\nThreshold = {eer_threshold:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"EER plot saved: {output_path}")

def plot_roc_curve(fpr, tpr, eer, scenario_name, output_path):
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
    plt.title(f'Pre-trained ECAPA: {scenario_name}\nROC Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    
    plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
    
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    textstr = f'Scenario: {scenario_name}\nAUC = {roc_auc:.4f}\nEER = {eer:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.6, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"ROC plot saved: {output_path}")

def plot_score_distributions(scores, y_true, eer_threshold, scenario_name, output_path):
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
    plt.title(f'Pre-trained ECAPA: {scenario_name}\nScore Distributions', 
              fontsize=14, fontweight='bold')
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
    logging.info(f"Score distribution plot saved: {output_path}")

def plot_det_curve(fpr, fnr, eer, scenario_name, output_path):
    """Plot DET curve"""
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, fnr, 'b-', linewidth=2, label=f'DET curve (EER = {eer:.2f}%)')
    
    eer_idx = np.argmin(np.abs(fpr - fnr))
    plt.plot(fpr[eer_idx], fnr[eer_idx], 'ro', markersize=10, 
             label=f'EER point ({fpr[eer_idx]*100:.1f}%, {fnr[eer_idx]*100:.1f}%)')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim([1e-4, 1])
    plt.ylim([1e-4, 1])
    
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
    plt.title(f'Pre-trained ECAPA: {scenario_name}\nDET Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    
    plt.plot([1e-4, 1], [1e-4, 1], 'k--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"DET plot saved: {output_path}")

def create_confusion_matrix_heatmap(results_matrix, output_path):
    """Create a 2x2 heatmap showing EER for all scenarios"""
    matrix_data = [
        [results_matrix.get('EN_EN', 0), results_matrix.get('EN_Regional', 0)],
        [results_matrix.get('Regional_EN', 0), results_matrix.get('Regional_Regional', 0)]
    ]
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(matrix_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                xticklabels=['EN', 'Regional'], 
                yticklabels=['EN', 'Regional'],
                cbar_kws={'label': 'EER (%)'},
                vmin=0, vmax=max(results_matrix.values()) * 1.2,
                linewidths=2, linecolor='black',
                annot_kws={'size': 16, 'weight': 'bold'})
    
    plt.xlabel('Test Language', fontsize=14, fontweight='bold')
    plt.ylabel('Enrollment Language', fontsize=14, fontweight='bold')
    plt.title('Pre-trained ECAPA: Cross-Lingual EER Matrix (%)\n(Lower is Better)', 
              fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion matrix saved: {output_path}")

def create_comparison_bar_chart(results_matrix, output_path):
    """Create bar chart comparing EER across scenarios"""
    scenarios = ['EN→EN', 'EN→Regional', 'Regional→EN', 'Regional→Regional']
    eers = [
        results_matrix.get('EN_EN', 0),
        results_matrix.get('EN_Regional', 0),
        results_matrix.get('Regional_EN', 0),
        results_matrix.get('Regional_Regional', 0)
    ]
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = plt.bar(scenarios, eers, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, eer in zip(bars, eers):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{eer:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('EER (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Scenario', fontsize=14, fontweight='bold')
    plt.title('Pre-trained ECAPA: Cross-Lingual EER Comparison\n(Lower is Better)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, max(eers) * 1.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Comparison bar chart saved: {output_path}")

def save_scenario_results(scenario_name, eer, min_dcf, eer_threshold, min_dcf_threshold, 
                         scores, y_true, output_dir):
    """Save all results for a specific scenario"""
    metrics = {
        'Metric': ['EER (%)', 'minDCF (p=0.01)', 'Threshold at EER', 'Threshold at minDCF',
                  'Genuine Pairs', 'Impostor Pairs'],
        'Value': [eer, min_dcf, eer_threshold, min_dcf_threshold,
                 np.sum(y_true == 1), np.sum(y_true == 0)]
    }
    
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_dir, f"{scenario_name}_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    txt_path = os.path.join(output_dir, f"{scenario_name}_metrics.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"PRE-TRAINED ECAPA EVALUATION: {scenario_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Evaluation Date: {pd.Timestamp.now()}\n")
        f.write(f"Model: Pre-trained ECAPA-TDNN (SpeechBrain/VoxCeleb)\n\n")
        f.write(f"EER: {eer:.4f} %\n")
        f.write(f"minDCF (p_target=0.01): {min_dcf:.6f}\n")
        f.write(f"Threshold at EER: {eer_threshold:.6f}\n")
        f.write(f"Threshold at minDCF: {min_dcf_threshold:.6f}\n\n")
        f.write(f"Genuine Pairs: {np.sum(y_true == 1)}\n")
        f.write(f"Impostor Pairs: {np.sum(y_true == 0)}\n")
        f.write("=" * 60 + "\n")
    
    logging.info(f"Scenario results saved: {output_dir}")

def create_overall_summary(results_matrix, output_dir):
    """Create overall summary report"""
    report_path = os.path.join(output_dir, "OVERALL_SUMMARY.md")
    
    scenarios = ['EN_EN', 'EN_Regional', 'Regional_EN', 'Regional_Regional']
    scenario_names = ['EN→EN', 'EN→Regional', 'Regional→EN', 'Regional→Regional']
    
    with open(report_path, 'w') as f:
        f.write("# Pre-trained ECAPA-TDNN: Cross-Lingual Evaluation Report\n\n")
        f.write(f"**Evaluation Date:** {pd.Timestamp.now()}\n\n")
        f.write(f"**Model:** Pre-trained ECAPA-TDNN (trained on VoxCeleb via SpeechBrain)\n\n")
        f.write(f"**Dataset:** {DATASET_ROOT}\n\n")
        
        f.write("## Results Matrix\n\n")
        f.write("| Scenario | EER (%) |\n")
        f.write("|----------|---------|\n")
        
        for scenario, name in zip(scenarios, scenario_names):
            eer = results_matrix.get(scenario, 0)
            f.write(f"| {name} | {eer:.2f} |\n")
        
        eers = [results_matrix.get(s, 0) for s in scenarios]
        best_idx = np.argmin(eers)
        worst_idx = np.argmax(eers)
        
        f.write("\n## Key Findings\n\n")
        f.write(f"1. **Best Performance**: {scenario_names[best_idx]} ({eers[best_idx]:.2f}%)\n")
        f.write(f"2. **Worst Performance**: {scenario_names[worst_idx]} ({eers[worst_idx]:.2f}%)\n")
        f.write(f"3. **Performance Gap**: {eers[worst_idx] - eers[best_idx]:.2f}%\n\n")
        
        monolingual_avg = (results_matrix.get('EN_EN', 0) + results_matrix.get('Regional_Regional', 0)) / 2
        crosslingual_avg = (results_matrix.get('EN_Regional', 0) + results_matrix.get('Regional_EN', 0)) / 2
        
        f.write("## Cross-Lingual Analysis\n\n")
        f.write(f"- **Average Monolingual EER**: {monolingual_avg:.2f}%\n")
        f.write(f"- **Average Cross-Lingual EER**: {crosslingual_avg:.2f}%\n")
        f.write(f"- **Language Mismatch Penalty**: {crosslingual_avg - monolingual_avg:.2f}%\n\n")
    
    logging.info(f"Overall summary report saved: {report_path}")

# ----------------- Main Evaluation Function -----------------
def main():
    """Main cross-lingual evaluation function for pre-trained model"""
    logging.info("=" * 80)
    logging.info("STARTING CROSS-LINGUAL EVALUATION OF PRE-TRAINED ECAPA-TDNN")
    logging.info("=" * 80)
    
    # Load model using SpeechBrain
    if USE_SPEECHBRAIN:
        logging.info("Attempting to load model from SpeechBrain...")
        sb_model = load_speechbrain_pretrained()
        
        if sb_model is not None:
            # Wrap the SpeechBrain model
            model = SpeechBrainWrapper(sb_model)
            logging.info("SpeechBrain model wrapped successfully!")
        else:
            logging.error("Failed to load SpeechBrain model. Trying alternative...")
            # Try HuggingFace as fallback
            model = download_from_huggingface()
            if model is None:
                logging.error("Could not load any pre-trained model. Exiting.")
                return
    else:
        # Try HuggingFace directly
        model = download_from_huggingface()
        if model is None:
            logging.error("Failed to load model from HuggingFace. Exiting.")
            return
    
    model = model.to(DEVICE)
    model.eval()
    
    # Organize test data
    test_dir = os.path.join(DATASET_ROOT, "test")
    logging.info(f"Loading test data from: {test_dir}")
    
    speaker_lang_files = organize_files_by_language(test_dir)
    
    if not speaker_lang_files:
        logging.error("No test files found!")
        return
    
    # Extract embeddings
    embeddings_dict = extract_all_embeddings(model, speaker_lang_files)
    
    # Define evaluation scenarios
    scenarios = [
        ('EN', 'EN', 'EN_EN'),
        ('EN', 'Regional', 'EN_Regional'),
        ('Regional', 'EN', 'Regional_EN'),
        ('Regional', 'Regional', 'Regional_Regional')
    ]
    
    results_matrix = {}
    
    # Evaluate each scenario
    for enroll_type, test_type, scenario_name in scenarios:
        logging.info("=" * 80)
        logging.info(f"EVALUATING SCENARIO: {scenario_name}")
        logging.info(f"Enrollment: {enroll_type}, Test: {test_type}")
        logging.info("=" * 80)
        
        scores, y_true = create_pairs_for_scenario(
            embeddings_dict, enroll_type, test_type, NUM_PAIRS_PER_SCENARIO
        )
        
        if scores is None:
            logging.warning(f"Skipping scenario {scenario_name} due to insufficient data")
            continue
        
        eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold = \
            compute_eer_minDCF(scores, y_true)
        
        results_matrix[scenario_name] = eer
        
        logging.info(f"Results for {scenario_name}:")
        logging.info(f"  EER: {eer:.4f}%")
        logging.info(f"  minDCF: {min_dcf:.6f}")
        
        scenario_dir = os.path.join(OUTPUT_DIR, scenario_name)
        
        save_scenario_results(
            scenario_name, eer, min_dcf, eer_threshold, min_dcf_threshold,
            scores, y_true, scenario_dir
        )
        
        plot_eer_curve(
            fpr, fnr, thresholds, eer, eer_threshold, scenario_name,
            os.path.join(scenario_dir, "eer_curve.png")
        )
        
        plot_roc_curve(
            fpr, 1-fnr, eer, scenario_name,
            os.path.join(scenario_dir, "roc_curve.png")
        )
        
        plot_det_curve(
            fpr, fnr, eer, scenario_name,
            os.path.join(scenario_dir, "det_curve.png")
        )
        
        plot_score_distributions(
            scores, y_true, eer_threshold, scenario_name,
            os.path.join(scenario_dir, "score_distributions.png")
        )
    
    # Create overall comparison plots
    logging.info("=" * 80)
    logging.info("CREATING OVERALL COMPARISON PLOTS")
    logging.info("=" * 80)
    
    create_confusion_matrix_heatmap(
        results_matrix,
        os.path.join(OUTPUT_DIR, "eer_matrix_heatmap.png")
    )
    
    create_comparison_bar_chart(
        results_matrix,
        os.path.join(OUTPUT_DIR, "eer_comparison_bar.png")
    )
    
    create_overall_summary(results_matrix, OUTPUT_DIR)
    
    # Print final summary
    logging.info("=" * 80)
    logging.info("EVALUATION COMPLETE - FINAL RESULTS")
    logging.info("=" * 80)
    
    for scenario_name, eer in results_matrix.items():
        logging.info(f"{scenario_name}: EER = {eer:.2f}%")
    
    logging.info("=" * 80)
    logging.info(f"All results saved to: {OUTPUT_DIR}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()