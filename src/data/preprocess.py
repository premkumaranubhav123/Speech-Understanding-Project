"""
Audio Preprocessing & Feature Extraction
- Bandpass filter (200–8000 Hz)
- 128-bin Log Mel Spectrogram (224×224 for CNN)
- 40-coefficient MFCC
- Fold-by-fold processing (memory efficient)
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.download_dataset import load_metadata, CLASS_NAMES, DATA_DIR, FEATURES_DIR

# ── Feature config ────────────────────────────────────────────────
SAMPLE_RATE    = 22050
DURATION       = 4          # seconds
N_SAMPLES      = SAMPLE_RATE * DURATION
N_MELS         = 128
N_MFCC         = 40
HOP_LENGTH     = 512
N_FFT          = 2048
IMG_SIZE       = 224        # resize mel spec to this square
F_MIN          = 200        # bandpass low
F_MAX          = 8000       # bandpass high


def load_audio(filepath: str, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """Load audio file, pad/trim to exact duration."""
    try:
        audio, _ = librosa.load(filepath, sr=sr, mono=True, duration=duration)
        # Pad if shorter than target
        if len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode='reflect')
        else:
            audio = audio[:N_SAMPLES]
        return audio.astype(np.float32)
    except Exception as e:
        # Return silence on error
        return np.zeros(N_SAMPLES, dtype=np.float32)


def bandpass_filter(audio: np.ndarray, sr: int = SAMPLE_RATE,
                    f_low: float = F_MIN, f_high: float = F_MAX) -> np.ndarray:
    """Simple bandpass via FFT zero-masking."""
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    fft[(freqs < f_low) | (freqs > f_high)] = 0
    return np.fft.irfft(fft, n=len(audio)).astype(np.float32)


def compute_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE,
                             n_mels: int = N_MELS, n_fft: int = N_FFT,
                             hop_length: int = HOP_LENGTH,
                             img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Compute log Mel spectrogram and resize to (img_size, img_size).
    Returns float32 array in [0, 1] range.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length,
        fmin=F_MIN, fmax=F_MAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0,1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    # Resize to img_size × img_size using simple interpolation
    from PIL import Image
    img = Image.fromarray((log_mel * 255).astype(np.uint8))
    img = img.resize((img_size, img_size), Image.BILINEAR)
    mel_resized = np.array(img).astype(np.float32) / 255.0  # (224, 224)

    # Expand to 3 channels (RGB) for pretrained CNN compatibility
    mel_3ch = np.stack([mel_resized, mel_resized, mel_resized], axis=0)  # (3, 224, 224)
    return mel_3ch


def compute_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Compute MFCC features.
    Returns mean + std of each coefficient: shape (n_mfcc * 2,) = (80,)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)  # (80,)


def extract_features_for_clip(filepath: str):
    """Full preprocessing pipeline for a single audio clip."""
    audio = load_audio(filepath)
    audio = bandpass_filter(audio)
    mel = compute_mel_spectrogram(audio)   # (3, 224, 224)
    mfcc = compute_mfcc(audio)             # (80,)
    return mel, mfcc


def process_fold(fold: int, df: pd.DataFrame, out_dir: Path,
                 force: bool = False) -> dict:
    """
    Extract and cache features for one fold.
    Saves:  fold{n}_mel.npy   shape (N, 3, 224, 224)
            fold{n}_mfcc.npy  shape (N, 80)
            fold{n}_labels.npy shape (N,)
    """
    mel_path   = out_dir / f"fold{fold}_mel.npy"
    mfcc_path  = out_dir / f"fold{fold}_mfcc.npy"
    label_path = out_dir / f"fold{fold}_labels.npy"

    if not force and mel_path.exists() and mfcc_path.exists() and label_path.exists():
        print(f"  Fold {fold}: cached features found, skipping.")
        n = np.load(label_path).shape[0]
        return {"fold": fold, "n_clips": n, "status": "cached"}

    fold_df = df[df['fold'] == fold].reset_index(drop=True)
    n = len(fold_df)

    mels   = np.zeros((n, 3, 224, 224), dtype=np.float32)
    mfccs  = np.zeros((n, 80),          dtype=np.float32)
    labels = np.zeros(n,                dtype=np.int64)

    for i, row in tqdm(fold_df.iterrows(), total=n,
                       desc=f"  Fold {fold}", leave=False):
        mel, mfcc = extract_features_for_clip(row['filepath'])
        mels[i]   = mel
        mfccs[i]  = mfcc
        labels[i] = row['classID']

    np.save(mel_path,   mels)
    np.save(mfcc_path,  mfccs)
    np.save(label_path, labels)

    print(f"  Fold {fold}: {n} clips -> mel {mels.shape}, mfcc {mfccs.shape}")
    return {"fold": fold, "n_clips": n, "status": "processed"}


def load_fold_features(fold: int, out_dir: Path = None):
    """Load cached fold features."""
    if out_dir is None:
        out_dir = FEATURES_DIR
    mel   = np.load(out_dir / f"fold{fold}_mel.npy")
    mfcc  = np.load(out_dir / f"fold{fold}_mfcc.npy")
    labels = np.load(out_dir / f"fold{fold}_labels.npy")
    return mel, mfcc, labels


def process_all_folds(df: pd.DataFrame, out_dir: Path = None,
                      folds: list = None, force: bool = False):
    """Process all 8 folds sequentially (memory efficient)."""
    if out_dir is None:
        out_dir = FEATURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if folds is None:
        folds = sorted(df['fold'].unique())

    print(f"\n Feature Extraction: {len(folds)} folds")
    print(f"  Output dir: {out_dir}")
    print(f"  Mel shape per clip : (3, 224, 224)")
    print(f"  MFCC dim per clip  : 80 (mean+std × 40 coeffs)")
    print()

    results = []
    for fold in folds:
        result = process_fold(fold, df, out_dir, force=force)
        results.append(result)
        # Explicit GC after each fold
        import gc
        gc.collect()

    total = sum(r['n_clips'] for r in results)
    print(f"\n✓ Feature extraction complete: {total} clips processed")
    return results


def print_feature_stats(fold: int = 1):
    """Print stats for one fold's features."""
    mel, mfcc, labels = load_fold_features(fold)
    print(f"\nFold {fold} feature stats:")
    print(f"  Mel spectrogram : shape={mel.shape}, min={mel.min():.3f}, max={mel.max():.3f}")
    print(f"  MFCC            : shape={mfcc.shape}, min={mfcc.min():.3f}, max={mfcc.max():.3f}")
    print(f"  Labels          : shape={labels.shape}, classes={np.unique(labels)}")
    for i, name in enumerate(CLASS_NAMES):
        count = (labels == i).sum()
        print(f"    {name:25s}: {count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Specific folds to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cached")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only fold 1")
    args = parser.parse_args()

    df = load_metadata()

    if args.test:
        folds = [1]
        print("TEST MODE: processing fold 1 only")
    else:
        folds = args.folds

    results = process_all_folds(df, folds=folds, force=args.force)
    if results:
        print_feature_stats(fold=results[0]['fold'])
