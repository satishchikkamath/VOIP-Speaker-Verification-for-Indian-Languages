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
# Direct SpeechBrain loading - no fallbacks
DATASET_ROOT = "/home/user2/VOIP/VOIP_Mel_Features"
OUTPUT_DIR = "/home/user2/VOIP/ecapa_pretrained_crosslingual_evaluation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation parameters
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_PAIRS_PER_SCENARIO = 5000

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

# ----------------- Fix torchaudio compatibility -----------------
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    # Create dummy function if it doesn't exist
    torchaudio.list_audio_backends = lambda: []

# ----------------- Direct SpeechBrain Model Loading -----------------
def load_speechbrain_model():
    """
    Load pre-trained ECAPA-TDNN directly from SpeechBrain
    """
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        
        logging.info("Loading pre-trained ECAPA-TDNN from SpeechBrain...")
        
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

class SpeechBrainWrapper(nn.Module):
    """Wrapper to make SpeechBrain model compatible with our extraction function"""
    def __init__(self, sb_model):
        super().__init__()
        self.sb_model = sb_model
        self.embd_dim = 192
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, freq, time) - our input format
        Returns:
            embeddings: tensor of shape (batch, embd_dim)
        """
        with torch.no_grad():
            # SpeechBrain expects (batch, time, freq)
            # Our input is (batch, freq, time) so we need to transpose
            if x.dim() == 3:
                # Transpose from (batch, freq, time) to (batch, time, freq)
                x = x.transpose(1, 2)
            
            # Get embeddings
            embeddings = self.sb_model.encode_batch(x)
            
            # Squeeze and normalize
            embeddings = embeddings.squeeze(1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings

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
    """Create pairs for a specific enrollment-test scenario"""
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
        logging.warning(f"Insufficient pairs for {enroll_lang_type}?{test_lang_type}")
        return None, None
    
    scores = np.array(genuine_scores + impostor_scores)
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    
    logging.info(f"{enroll_lang_type}?{test_lang_type}: {len(genuine_scores)} genuine, {len(impostor_scores)} impostor pairs")
    
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

# ----------------- Plotting Functions (Simplified) -----------------
def plot_eer_curve(fpr, fnr, thresholds, eer, eer_threshold, scenario_name, output_path):
    """Plot FAR and FRR curves with EER point"""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fpr, 'b-', label='FAR')
    plt.plot(thresholds, fnr, 'r-', label='FRR')
    plt.axvline(x=eer_threshold, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=eer/100, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title(f'Pre-trained ECAPA: {scenario_name} (EER={eer:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(fpr, tpr, eer, scenario_name, output_path):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC (EER={eer:.2f}%)')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('FAR')
    plt.ylabel('TAR')
    plt.title(f'Pre-trained ECAPA: {scenario_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_score_distributions(scores, y_true, eer_threshold, scenario_name, output_path):
    """Plot score distributions"""
    genuine_scores = scores[y_true == 1]
    impostor_scores = scores[y_true == 0]
    
    plt.figure(figsize=(10, 5))
    plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', color='green', density=True)
    plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', color='red', density=True)
    plt.axvline(x=eer_threshold, color='k', linestyle='--', label=f'Threshold={eer_threshold:.3f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title(f'Pre-trained ECAPA: {scenario_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_heatmap(results_matrix, output_path):
    """Create a 2x2 heatmap showing EER for all scenarios"""
    matrix_data = [
        [results_matrix.get('EN_EN', 0), results_matrix.get('EN_Regional', 0)],
        [results_matrix.get('Regional_EN', 0), results_matrix.get('Regional_Regional', 0)]
    ]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                xticklabels=['EN', 'Regional'],
                yticklabels=['EN', 'Regional'],
                cbar_kws={'label': 'EER (%)'})
    plt.title('Pre-trained ECAPA: EER Matrix (%)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_scenario_results(scenario_name, eer, min_dcf, eer_threshold, min_dcf_threshold,
                         scores, y_true, output_dir):
    """Save results for a specific scenario"""
    metrics = {
        'Metric': ['EER (%)', 'minDCF', 'Threshold at EER'],
        'Value': [eer, min_dcf, eer_threshold]
    }
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, f"{scenario_name}_metrics.csv"), index=False)

def create_overall_summary(results_matrix, output_dir):
    """Create overall summary"""
    report_path = os.path.join(output_dir, "OVERALL_SUMMARY.txt")
    with open(report_path, 'w') as f:
        f.write("PRE-TRAINED ECAPA-TDNN EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for scenario, eer in results_matrix.items():
            f.write(f"{scenario}: {eer:.2f}% EER\n")

# ----------------- Main Evaluation Function -----------------
def main():
    """Main evaluation function"""
    logging.info("=" * 80)
    logging.info("STARTING CROSS-LINGUAL EVALUATION OF PRE-TRAINED ECAPA-TDNN")
    logging.info("=" * 80)
    
    # Load model directly from SpeechBrain
    logging.info("Loading model from SpeechBrain...")
    sb_model = load_speechbrain_model()
    
    if sb_model is None:
        logging.error("Failed to load SpeechBrain model. Exiting.")
        return
    
    # Wrap the model
    model = SpeechBrainWrapper(sb_model)
    model = model.to(DEVICE)
    model.eval()
    logging.info("Model ready for evaluation")
    
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
        logging.info("=" * 60)
        logging.info(f"Evaluating: {scenario_name}")
        
        scores, y_true = create_pairs_for_scenario(
            embeddings_dict, enroll_type, test_type, NUM_PAIRS_PER_SCENARIO
        )
        
        if scores is None:
            continue
        
        eer, min_dcf, fpr, fnr, thresholds, eer_threshold, min_dcf_threshold = \
            compute_eer_minDCF(scores, y_true)
        
        results_matrix[scenario_name] = eer
        
        logging.info(f"  EER: {eer:.2f}%")
        logging.info(f"  minDCF: {min_dcf:.4f}")
        
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
        
        plot_score_distributions(
            scores, y_true, eer_threshold, scenario_name,
            os.path.join(scenario_dir, "score_distributions.png")
        )
    
    # Create overall comparison plots
    if results_matrix:
        create_confusion_matrix_heatmap(
            results_matrix,
            os.path.join(OUTPUT_DIR, "eer_matrix_heatmap.png")
        )
        create_overall_summary(results_matrix, OUTPUT_DIR)
    
    # Print final summary
    logging.info("=" * 80)
    logging.info("EVALUATION COMPLETE")
    logging.info("=" * 80)
    for scenario_name, eer in results_matrix.items():
        logging.info(f"{scenario_name}: {eer:.2f}%")
    logging.info("=" * 80)
    logging.info(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()