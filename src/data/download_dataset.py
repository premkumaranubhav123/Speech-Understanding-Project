"""
UrbanSound8K Dataset Downloader
Downloads via Kaggle API or direct source.
Falls back to creating synthetic demo data if download is unavailable.
"""

import os
import sys
import zipfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "UrbanSound8K"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing",
    "dog_bark", "drilling", "engine_idling",
    "gun_shot", "jackhammer", "siren", "street_music"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def check_dataset_exists():
    """Check if UrbanSound8K is already downloaded."""
    metadata = DATA_DIR / "metadata" / "UrbanSound8K.csv"
    audio_dir = DATA_DIR / "audio"
    if metadata.exists() and audio_dir.exists():
        folds = list(audio_dir.glob("fold*"))
        if len(folds) >= 8:
            return True
    return False


def download_via_kaggle():
    """Try downloading via Kaggle CLI."""
    try:
        import subprocess
        print("Attempting Kaggle download (requires kaggle.json credentials)...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "andrewmvd/urban-sound-classification",
             "--unzip", "-p", str(DATA_DIR.parent)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("✓ Kaggle download successful")
            return True
        else:
            print(f"Kaggle download failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"Kaggle not available: {e}")
        return False


def create_synthetic_dataset(n_per_class=100, sample_rate=22050, duration=4):
    """
    Creates a small synthetic UrbanSound8K-compatible dataset for development.
    Uses numpy-generated audio signals (sine waves, noise, chirps).
    This allows the full pipeline to run without the ~6GB download.
    """
    import soundfile as sf

    print("\n[*] Creating synthetic UrbanSound8K dataset for development...")
    print(f"   {n_per_class} clips per class × 10 classes = {n_per_class*10} total clips")
    print(f"   Sample rate: {sample_rate} Hz, Duration: {duration}s\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = DATA_DIR / "audio"
    metadata_dir = DATA_DIR / "metadata"
    audio_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)

    # Create 8 fold directories
    for fold in range(1, 9):
        (audio_dir / f"fold{fold}").mkdir(exist_ok=True)

    rows = []
    file_id = 0
    n_samples = sample_rate * duration

    for class_idx, class_name in enumerate(tqdm(CLASS_NAMES, desc="Generating classes")):
        for i in range(n_per_class):
            fold = (i % 8) + 1
            filename = f"{file_id}-{class_idx}-0-{i}.wav"
            filepath = audio_dir / f"fold{fold}" / filename

            # Generate class-specific synthetic audio
            t = np.linspace(0, duration, n_samples)
            rng = np.random.default_rng(file_id)

            if class_name == "air_conditioner":
                # Steady low-frequency hum
                audio = 0.3 * np.sin(2 * np.pi * 60 * t) + 0.1 * rng.standard_normal(n_samples)
            elif class_name == "car_horn":
                # Short horn burst
                audio = np.zeros(n_samples)
                start = n_samples // 4
                audio[start:start + sample_rate] = 0.8 * np.sin(2 * np.pi * 440 * t[:sample_rate])
                audio += 0.05 * rng.standard_normal(n_samples)
            elif class_name == "children_playing":
                # Multi-frequency chaotic signal
                freqs = rng.integers(200, 2000, 5)
                audio = sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs)
                audio += 0.15 * rng.standard_normal(n_samples)
            elif class_name == "dog_bark":
                # Bark: short bursts
                audio = np.zeros(n_samples)
                for b in range(rng.integers(2, 5)):
                    s = rng.integers(0, n_samples - sample_rate // 2)
                    bark_len = sample_rate // 4
                    audio[s:s + bark_len] = 0.7 * np.sin(2 * np.pi * 350 * t[:bark_len]) * \
                                            np.exp(-3 * t[:bark_len] / (bark_len / sample_rate))
                audio += 0.05 * rng.standard_normal(n_samples)
            elif class_name == "drilling":
                # High-freq mechanical
                audio = 0.5 * np.sin(2 * np.pi * 800 * t) * (1 + 0.3 * np.sin(2 * np.pi * 10 * t))
                audio += 0.1 * rng.standard_normal(n_samples)
            elif class_name == "engine_idling":
                # Low rumble
                audio = 0.4 * np.sin(2 * np.pi * 80 * t) + 0.2 * np.sin(2 * np.pi * 160 * t)
                audio += 0.08 * rng.standard_normal(n_samples)
            elif class_name == "gun_shot":
                # Impulse
                audio = np.zeros(n_samples)
                shot_t = n_samples // 3
                audio[shot_t:shot_t + 100] = rng.standard_normal(100) * 2.0
                audio = np.convolve(audio, np.exp(-np.arange(1000) / 100), mode='same')
                audio += 0.02 * rng.standard_normal(n_samples)
            elif class_name == "jackhammer":
                # Rapid impulses
                audio = np.zeros(n_samples)
                step = sample_rate // 10
                for k in range(0, n_samples - step, step):
                    audio[k:k + 50] = 0.6
                audio += 0.1 * rng.standard_normal(n_samples)
            elif class_name == "siren":
                # Frequency sweep (wailing)
                freq = 300 + 200 * np.sin(2 * np.pi * 0.5 * t)
                audio = 0.6 * np.sin(2 * np.pi * np.cumsum(freq) / sample_rate)
                audio += 0.05 * rng.standard_normal(n_samples)
            else:  # street_music
                # Chord
                audio = sum(0.15 * np.sin(2 * np.pi * f * t) for f in [261, 329, 392, 523])
                audio += 0.1 * rng.standard_normal(n_samples)

            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            audio = audio.astype(np.float32)

            sf.write(str(filepath), audio, sample_rate)

            rows.append({
                "slice_file_name": filename,
                "fsID": file_id,
                "start": 0,
                "end": duration,
                "salience": 1,
                "fold": fold,
                "classID": class_idx,
                "class": class_name
            })
            file_id += 1

    # Save metadata CSV
    df = pd.DataFrame(rows)
    df.to_csv(metadata_dir / "UrbanSound8K.csv", index=False)

    # Add filepath column (same as load_metadata does)
    df['filepath'] = df.apply(
        lambda row: str(DATA_DIR / "audio" / f"fold{row['fold']}" / row['slice_file_name']),
        axis=1
    )

    print(f"\n✓ Synthetic dataset created: {len(df)} clips")
    print(f"   Location: {DATA_DIR}")
    print(f"   Metadata: {metadata_dir / 'UrbanSound8K.csv'}")
    print(f"   Class distribution:")
    print(df['class'].value_counts().to_string())
    return df


def load_metadata():
    """Load and return UrbanSound8K metadata DataFrame."""
    meta_path = DATA_DIR / "metadata" / "UrbanSound8K.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Run download_dataset.py first.")
    df = pd.read_csv(meta_path)
    # Add full path column
    df['filepath'] = df.apply(
        lambda row: str(DATA_DIR / "audio" / f"fold{row['fold']}" / row['slice_file_name']),
        axis=1
    )
    return df


def get_class_weights(df):
    """Compute inverse frequency class weights for imbalanced dataset."""
    counts = df['classID'].value_counts().sort_index()
    total = len(df)
    weights = total / (len(CLASS_NAMES) * counts)
    return weights.values.tolist()


if __name__ == "__main__":
    print("=" * 60)
    print("UrbanSound8K Dataset Setup")
    print("=" * 60)

    if check_dataset_exists():
        print("✓ Dataset already exists, skipping download.")
        df = load_metadata()
    else:
        # Try Kaggle first, fall back to synthetic
        if not download_via_kaggle():
            print("\nUsing synthetic dataset (full pipeline compatible).")
            print("For real results, place UrbanSound8K in data/UrbanSound8K/")
            df = create_synthetic_dataset(n_per_class=150)
        else:
            df = load_metadata()

    df2 = load_metadata()
    print(f"\n Dataset Summary:")
    print(f"  Total clips : {len(df2)}")
    print(f"  Classes     : {df2['class'].nunique()}")
    print(f"  Folds       : {sorted(df2['fold'].unique())}")
    print(f"  Class names : {CLASS_NAMES}")
    weights = get_class_weights(df2)
    print(f"  Class weights (imbalance correction):")
    for cls, w in zip(CLASS_NAMES, weights):
        print(f"    {cls:25s}: {w:.3f}")
