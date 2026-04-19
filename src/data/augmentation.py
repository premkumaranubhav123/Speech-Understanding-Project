"""
Data Augmentation for Environmental Sound Classification
- Gaussian noise injection (multiple SNR levels)
- Time stretching
- Pitch shifting
- SpecAugment (frequency + time masking on mel spectrograms)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ── Audio-domain augmentations ────────────────────────────────────

def add_gaussian_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Add Gaussian white noise at specified SNR (dB)."""
    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Time stretch without changing pitch. rate<1 = slower, rate>1 = faster."""
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    # Ensure same length
    n = len(audio)
    if len(stretched) > n:
        return stretched[:n]
    return np.pad(stretched, (0, n - len(stretched)), mode='reflect')


def pitch_shift(audio: np.ndarray, sr: int = 22050,
                n_steps: float = 0.0) -> np.ndarray:
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32)


def add_background_noise(audio: np.ndarray, noise_type: str = 'traffic',
                          snr_db: float = 10.0) -> np.ndarray:
    """
    Add structured background noise (traffic, crowd, ambient).
    Uses synthesized noise patterns for reproducibility without external files.
    """
    n = len(audio)
    rng = np.random.default_rng()

    if noise_type == 'traffic':
        # Low-frequency rumble + random impulses
        t = np.linspace(0, n / 22050, n)
        noise = 0.3 * np.sin(2 * np.pi * 60 * t) + 0.2 * rng.standard_normal(n)
        # Random car horns
        for _ in range(rng.integers(0, 3)):
            start = rng.integers(0, max(1, n - 2205))
            noise[start:start+2205] += 0.3 * np.sin(2 * np.pi * 440 * t[:2205])
    elif noise_type == 'crowd':
        # Band-limited noise (human speech range ~200–4000 Hz)
        from scipy import signal as sp_signal
        raw_noise = rng.standard_normal(n)
        sos = sp_signal.butter(4, [200, 4000], btype='bandpass', fs=22050, output='sos')
        noise = sp_signal.sosfilt(sos, raw_noise)
    else:  # ambient
        noise = 0.1 * rng.standard_normal(n)

    # Scale to desired SNR
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power  = np.mean(noise ** 2) + 1e-10
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


# ── Spectrogram-domain augmentation (SpecAugment) ─────────────────

def spec_augment(mel: np.ndarray,
                 num_time_masks: int = 2,
                 num_freq_masks: int = 2,
                 time_mask_max: int = 30,
                 freq_mask_max: int = 20) -> np.ndarray:
    """
    SpecAugment: apply time and frequency masking to mel spectrogram.
    Input mel shape: (C, H, W) or (H, W)
    """
    mel = mel.copy()
    if mel.ndim == 3:
        _, H, W = mel.shape
        # Frequency masking (rows)
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_max)
            f0 = np.random.randint(0, max(1, H - f))
            mel[:, f0:f0 + f, :] = 0.0
        # Time masking (cols)
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_max)
            t0 = np.random.randint(0, max(1, W - t))
            mel[:, :, t0:t0 + t] = 0.0
    else:
        H, W = mel.shape
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_max)
            f0 = np.random.randint(0, max(1, H - f))
            mel[f0:f0 + f, :] = 0.0
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_max)
            t0 = np.random.randint(0, max(1, W - t))
            mel[:, t0:t0 + t] = 0.0
    return mel


# ── PyTorch Dataset with augmentation ─────────────────────────────

class UrbanSoundDataset(Dataset):
    """
    PyTorch Dataset that loads pre-cached .npy features and
    applies optional augmentation at training time.
    """

    def __init__(self, mel: np.ndarray, mfcc: np.ndarray, labels: np.ndarray,
                 augment: bool = False, spec_aug: bool = True):
        """
        Args:
            mel    : (N, 3, 224, 224) float32
            mfcc   : (N, 80) float32
            labels : (N,) int64
            augment: apply SpecAugment during __getitem__
            spec_aug: use SpecAugment specifically
        """
        self.mel    = mel
        self.mfcc   = mfcc
        self.labels = labels
        self.augment  = augment
        self.spec_aug = spec_aug

        # Precompute per-channel mean/std for normalization (ImageNet-like)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel   = self.mel[idx].copy()    # (3, 224, 224)
        mfcc  = self.mfcc[idx].copy()   # (80,)
        label = int(self.labels[idx])

        # Augment at training time
        if self.augment:
            if self.spec_aug and np.random.random() > 0.5:
                mel = spec_augment(mel)
            # Random horizontal flip (time reversal)
            if np.random.random() > 0.5:
                mel = mel[:, :, ::-1].copy()

        # Normalize (ImageNet stats for pretrained backbone)
        mel = (mel - self.mean) / (self.std + 1e-8)

        return (
            torch.from_numpy(mel),
            torch.from_numpy(mfcc),
            torch.tensor(label, dtype=torch.long)
        )


class AugmentedUrbanSoundDataset(Dataset):
    """
    Audio-domain augmented dataset.
    Loads raw audio, applies augmentation, then extracts features on-the-fly.
    Used for generating augmented training batches (memory-efficient, slow).
    """

    def __init__(self, filepaths, labels,
                 augment_prob: float = 0.5,
                 snr_range: Tuple[float, float] = (5, 20)):
        self.filepaths    = filepaths
        self.labels       = labels
        self.augment_prob = augment_prob
        self.snr_range    = snr_range

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from data.preprocess import (load_audio, bandpass_filter,
                                      compute_mel_spectrogram, compute_mfcc,
                                      SAMPLE_RATE)
        audio = load_audio(self.filepaths[idx])
        audio = bandpass_filter(audio)

        if np.random.random() < self.augment_prob:
            aug_type = np.random.choice(['noise', 'stretch', 'pitch'])
            if aug_type == 'noise':
                snr = np.random.uniform(*self.snr_range)
                audio = add_gaussian_noise(audio, snr_db=snr)
            elif aug_type == 'stretch':
                rate = np.random.uniform(0.85, 1.15)
                audio = time_stretch(audio, rate=rate)
            else:
                steps = np.random.uniform(-2, 2)
                audio = pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)

        mel  = compute_mel_spectrogram(audio)
        mfcc = compute_mfcc(audio)
        label = int(self.labels[idx])

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        mel  = (mel - mean) / (std + 1e-8)

        return (
            torch.from_numpy(mel),
            torch.from_numpy(mfcc),
            torch.tensor(label, dtype=torch.long)
        )


def build_dataloaders(train_mel, train_mfcc, train_labels,
                      val_mel, val_mfcc, val_labels,
                      batch_size: int = 32,
                      num_workers: int = 0,
                      augment: bool = True):
    """
    Build train and validation DataLoaders.
    num_workers=0 avoids multiprocessing issues on Windows.
    """
    from torch.utils.data import DataLoader

    train_ds = UrbanSoundDataset(train_mel, train_mfcc, train_labels,
                                  augment=augment, spec_aug=True)
    val_ds   = UrbanSoundDataset(val_mel, val_mfcc, val_labels,
                                  augment=False, spec_aug=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing augmentation pipeline...")
    # Synthetic test
    dummy_mel    = np.random.rand(10, 3, 224, 224).astype(np.float32)
    dummy_mfcc   = np.random.rand(10, 80).astype(np.float32)
    dummy_labels = np.random.randint(0, 10, 10)

    ds = UrbanSoundDataset(dummy_mel, dummy_mfcc, dummy_labels, augment=True)
    mel, mfcc, label = ds[0]
    print(f"  mel shape : {mel.shape}")
    print(f"  mfcc shape: {mfcc.shape}")
    print(f"  label     : {label}")

    # Test SpecAugment
    mel_aug = spec_augment(dummy_mel[0])
    print(f"  SpecAugment output shape: {mel_aug.shape}")

    # Test noise
    dummy_audio = np.random.randn(22050 * 4).astype(np.float32)
    noisy = add_gaussian_noise(dummy_audio, snr_db=10.0)
    print(f"  Noisy audio shape: {noisy.shape}")

    print("✓ Augmentation tests passed")
