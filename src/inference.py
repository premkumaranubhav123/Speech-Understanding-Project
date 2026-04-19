import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.download_dataset import CLASS_NAMES
from models.ensemble import ensemble_from_checkpoints

# Configuration
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_MELS = 128
N_MFCC = 40
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

def extract_features_from_audio(audio, sr):
    """Extract Mel and MFCC from a raw audio array (should be ~4 seconds)."""
    # 1. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize to 224x224 (approximate behavior of training setup)
    import cv2
    mel_resized = cv2.resize(mel_db, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1]
    mel_norm = mel_resized - mel_resized.min()
    if mel_norm.max() > 0:
        mel_norm = mel_norm / mel_norm.max()
        
    # Convert to 3 channels (RGB duplicate)
    mel_rgb = np.stack([mel_norm, mel_norm, mel_norm], axis=0).astype(np.float32)
    
    # 2. MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_features = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)
    
    return mel_rgb, mfcc_features

def predict_audio(audio_path, model_type='both'):
    print(f"Loading {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
        
    # Chunk audio into 4-second segments
    samples_per_chunk = SAMPLE_RATE * DURATION
    total_samples = len(y)
    
    if total_samples < samples_per_chunk:
        # Pad short audio
        y = np.pad(y, (0, samples_per_chunk - total_samples), mode='constant')
        total_samples = len(y)
        
    chunks = []
    # simple splitting
    for start in range(0, total_samples, samples_per_chunk):
        segment = y[start:start + samples_per_chunk]
        if len(segment) == samples_per_chunk:
            chunks.append(segment)
            
    if not chunks:
        # if the last chunk was too short and no chunks made
        segment = y[-samples_per_chunk:]
        segment = np.pad(segment, (0, samples_per_chunk - len(segment)))
        chunks.append(segment)
        
    print(f"Extracted {len(chunks)} segments of 4s length.")
    
    mels = []
    mfccs = []
    for chunk in chunks:
        mel, mfcc = extract_features_from_audio(chunk, sr)
        mels.append(mel)
        mfccs.append(mfcc)
        
    mels = torch.tensor(np.array(mels))
    mfccs = torch.tensor(np.array(mfccs))
    
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Models
    print("Loading models...")
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    
    models = []
    types = []
    if model_type in ['resnet50', 'both']:
        # use best folding (e.g. fold 1)
        path = ckpt_dir / "resnet50_fold1_best.pt"
        if path.exists():
            models.append(str(path))
            types.append('resnet50')
            
    if model_type in ['yamnet', 'both']:
        path = ckpt_dir / "yamnet_fold1_best.pt"
        if path.exists():
            models.append(str(path))
            types.append('yamnet')
            
    if not models:
        print("Error: No trained checkpoints found.")
        return
        
    ensemble = ensemble_from_checkpoints(models, types, device=device)
    mels = mels.to(device)
    mfccs = mfccs.to(device)
    
    print("Running inference...")
    with torch.no_grad():
        probs = ensemble(mels, mfccs).cpu().numpy()
        
    # Average across all segments
    avg_probs = np.mean(probs, axis=0)
    pred_idx = np.argmax(avg_probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = avg_probs[pred_idx] * 100
    
    print("\n========================================")
    print("   PREDICTION RESULTS")
    print("========================================")
    print(f"Predicted Class: {pred_class.upper()}")
    print(f"Confidence     : {confidence:.2f}%\n")
    
    # Top 3
    top3_idx = np.argsort(avg_probs)[::-1][:3]
    print("Top 3 Predictions:")
    for i, idx in enumerate(top3_idx):
        print(f"  {i+1}. {CLASS_NAMES[idx]:20s} : {avg_probs[idx]*100:.2f}%")
        
    print("========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", choices=['yamnet', 'resnet50', 'both'], default='both')
    args = parser.parse_args()
    
    predict_audio(args.audio_path, model_type=args.model)
