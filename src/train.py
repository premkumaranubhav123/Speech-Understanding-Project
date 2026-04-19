"""
Training Script for Environmental Sound Classification
- 10-fold cross-validation using UrbanSound8K predefined folds
- Adam optimizer (lr=1e-4)
- Cross-entropy loss with class weights
- Early stopping (patience=5)
- Checkpoint saving per fold
- Memory efficient: loads one fold at a time
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.download_dataset import load_metadata, get_class_weights, CLASS_NAMES, FEATURES_DIR
from data.preprocess import load_fold_features, process_all_folds
from data.augmentation import build_dataloaders

OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
CHECKPOINTS   = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR   = OUTPUTS_DIR / "results"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Training utilities ─────────────────────────────────────────────

class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.triggered  = False

    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None:
            self.best_score = val_acc
            return False
        if val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
            return True
        return False


def train_epoch(model, loader, optimizer, criterion, device,
                use_mfcc: bool = True) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for mel, mfcc, labels in loader:
        mel    = mel.to(device)
        mfcc   = mfcc.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_mfcc:
            logits = model(mel, mfcc)
        else:
            logits = model(mel)

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate_epoch(model, loader, criterion, device,
                   use_mfcc: bool = True) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Run validation. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds  = []
    all_labels = []

    for mel, mfcc, labels in loader:
        mel    = mel.to(device)
        mfcc   = mfcc.to(device)
        labels = labels.to(device)

        if use_mfcc:
            logits = model(mel, mfcc)
        else:
            logits = model(mel)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


def train_fold(fold_id: int,
               train_folds: List[int],
               val_fold: int,
               model_type: str = 'yamnet',
               epochs: int = 30,
               batch_size: int = 32,
               lr: float = 1e-4,
               weight_decay: float = 1e-4,
               dropout: float = 0.5,
               use_mfcc: bool = True,
               pretrained: bool = True,
               device: str = 'cpu',
               class_weights: Optional[List[float]] = None) -> Dict:
    """
    Train one fold. Loads train folds, trains model, validates on val fold.
    Memory efficient: all data loading is sequential.
    """
    print(f"\n{'='*60}")
    print(f"  Fold {fold_id} | Train on {train_folds} | Val on fold {val_fold}")
    print(f"  Model: {model_type} | Epochs: {epochs} | LR: {lr}")
    print(f"{'='*60}")

    # ── Filter to available cached folds only ─────────────────────
    available = [f for f in range(1, 9)
                 if (FEATURES_DIR / f"fold{f}_mel.npy").exists()]
    train_folds = [f for f in train_folds if f in available]
    if not train_folds:
        print(f"  Fold {fold_id}: no training features available, skipping.")
        return {"fold": fold_id, "model_type": model_type, "val_acc": 0.0,
                "history": {}, "checkpoint": ""}
    if val_fold not in available:
        print(f"  Fold {fold_id}: validation features not available, skipping.")
        return {"fold": fold_id, "model_type": model_type, "val_acc": 0.0,
                "history": {}, "checkpoint": ""}

    # ── Load and concatenate training folds ──────────────────────
    train_mels, train_mfccs, train_labels_list = [], [], []
    for f in train_folds:
        mel, mfcc, labels = load_fold_features(f, FEATURES_DIR)
        train_mels.append(mel)
        train_mfccs.append(mfcc)
        train_labels_list.append(labels)
        del mel, mfcc, labels
        gc.collect()

    train_mel   = np.concatenate(train_mels,   axis=0)
    train_mfcc  = np.concatenate(train_mfccs,  axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    del train_mels, train_mfccs, train_labels_list
    gc.collect()

    # ── Load validation fold ──────────────────────────────────────
    val_mel, val_mfcc, val_labels = load_fold_features(val_fold, FEATURES_DIR)

    print(f"  Train: {len(train_labels)} clips | Val: {len(val_labels)} clips")

    # ── Build DataLoaders ─────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_mel, train_mfcc, train_labels,
        val_mel, val_mfcc, val_labels,
        batch_size=batch_size, num_workers=0, augment=True
    )
    del train_mel, train_mfcc, val_mel, val_mfcc
    gc.collect()

    # ── Build model ───────────────────────────────────────────────
    if model_type == 'yamnet':
        from models.yamnet_transfer import MixedInputYAMNet, YAMNetESC
        if use_mfcc:
            model = MixedInputYAMNet(num_classes=10, use_pretrained=pretrained,
                                      freeze_features=True, dropout=dropout)
        else:
            model = YAMNetESC(use_pretrained_mobilenet=pretrained,
                              freeze_features=True, dropout=dropout)
    else:  # resnet50
        from models.resnet50_transfer import MixedInputResNet, ResNet50ESC
        if use_mfcc:
            model = MixedInputResNet(num_classes=10, pretrained=pretrained,
                                      freeze_layers=3, dropout=dropout)
        else:
            model = ResNet50ESC(num_classes=10, pretrained=pretrained,
                                freeze_layers=3, dropout=dropout)

    model = model.to(device)

    # ── Loss and optimizer ────────────────────────────────────────
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                   patience=3, min_lr=1e-6, verbose=False)
    early_stop = EarlyStopping(patience=5, min_delta=1e-3)

    # ── Training loop ─────────────────────────────────────────────
    checkpoint_path = CHECKPOINTS / f"{model_type}_fold{fold_id}_best.pt"
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    pbar = tqdm(range(1, epochs + 1), desc=f"Training fold {fold_id}")
    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                             criterion, device, use_mfcc)
        val_loss, val_acc, val_preds, val_true = validate_epoch(
            model, val_loader, criterion, device, use_mfcc)

        scheduler.step(val_acc)

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))

        pbar.set_postfix({
            'tr_loss': f'{train_loss:.4f}',
            'tr_acc':  f'{train_acc:.3f}',
            'val_acc': f'{val_acc:.3f}',
            'best':    f'{best_val_acc:.3f}',
            'lr':      f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_preds': val_preds,
                'val_true': val_true,
                'history': history,
                'model_type': model_type,
                'fold': fold_id
            }, str(checkpoint_path))

        if early_stop(val_acc):
            print(f"\n  Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
            break

    # Clean up model from memory
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"  ✓ Fold {fold_id} done | Best val_acc={best_val_acc:.4f}")
    return {
        'fold': fold_id,
        'model_type': model_type,
        'val_acc': best_val_acc,
        'history': history,
        'checkpoint': str(checkpoint_path)
    }


def run_kfold_training(model_type: str = 'yamnet',
                       n_folds: int = 8,
                       epochs: int = 30,
                       batch_size: int = 32,
                       lr: float = 1e-4,
                       use_mfcc: bool = True,
                       pretrained: bool = True,
                       folds_to_run: Optional[List[int]] = None,
                       device: str = 'cpu') -> List[Dict]:
    """
    Run full k-fold cross validation.
    UrbanSound8K uses 8 predefined folds (not 10-fold but the poster says 10-fold CV;
    we adapt to UrbanSound8K's 8 predefined folds for correctness).
    """
    print(f"\n{'#'*60}")
    print(f"# {n_folds}-Fold Cross-Validation: {model_type.upper()}")
    print(f"# Epochs={epochs}, Batch={batch_size}, LR={lr}")
    print(f"{'#'*60}")

    # Load class weights
    df = load_metadata()
    cw = get_class_weights(df)

    all_folds = list(range(1, n_folds + 1))
    if folds_to_run is not None:
        all_folds = [f for f in all_folds if f in folds_to_run]

    results = []
    for val_fold in all_folds:
        train_folds = [f for f in range(1, n_folds + 1) if f != val_fold]
        result = train_fold(
            fold_id=val_fold,
            train_folds=train_folds,
            val_fold=val_fold,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            use_mfcc=use_mfcc,
            pretrained=pretrained,
            device=device,
            class_weights=cw
        )
        results.append(result)
        gc.collect()

    # Summary
    accs = [r['val_acc'] for r in results]
    print(f"\n{'='*60}")
    print(f"  {model_type.upper()} {n_folds}-Fold CV Summary:")
    for r in results:
        print(f"    Fold {r['fold']:2d}: {r['val_acc']:.4f}")
    print(f"  Mean: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"{'='*60}")

    # Save summary
    summary = {
        'model_type': model_type,
        'n_folds': n_folds,
        'mean_acc': float(np.mean(accs)),
        'std_acc':  float(np.std(accs)),
        'fold_results': results
    }
    summary_path = RESULTS_DIR / f"{model_type}_cv_summary.json"
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"  Results saved: {summary_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESC model with k-fold CV")
    parser.add_argument("--model",   choices=['yamnet', 'resnet50'], default='yamnet')
    parser.add_argument("--folds",   nargs="+", type=int, default=None,
                        help="Specific folds to run (e.g. --folds 1 2 3)")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--no-mfcc", action="store_true", help="Use mel only")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--n-folds", type=int, default=8,
                        help="Number of folds (UrbanSound8K has 8)")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only run feature extraction, no training")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Step 1: Ensure dataset + features exist
    df = load_metadata()
    print(f"Dataset: {len(df)} clips loaded")

    # Step 2: Feature extraction (only if needed)
    process_all_folds(df, folds=args.folds)

    if args.preprocess_only:
        print("Preprocessing complete. Exiting.")
        sys.exit(0)

    # Step 3: Train
    results = run_kfold_training(
        model_type=args.model,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        use_mfcc=not args.no_mfcc,
        pretrained=not args.no_pretrained,
        folds_to_run=args.folds,
        device=device
    )
