"""
Evaluation Script for Environmental Sound Classification
- Overall accuracy & macro F1
- Per-class precision, recall, F1
- Confusion matrix
- Aggregated cross-fold results
"""

import os
import sys
import json
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.download_dataset import CLASS_NAMES, FEATURES_DIR, load_metadata, get_class_weights
from data.preprocess import load_fold_features
from data.augmentation import build_dataloaders

OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
CHECKPOINTS  = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR  = OUTPUTS_DIR / "results"
FIGURES_DIR  = OUTPUTS_DIR / "figures"
for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_model_from_checkpoint(checkpoint_path: str,
                                model_type: str,
                                use_mfcc: bool = True,
                                device: str = 'cpu'):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    if model_type == 'yamnet':
        from models.yamnet_transfer import MixedInputYAMNet, YAMNetESC
        model = MixedInputYAMNet(use_pretrained=True) if use_mfcc else YAMNetESC(use_pretrained_mobilenet=True)
    else:
        from models.resnet50_transfer import MixedInputResNet, ResNet50ESC
        model = MixedInputResNet(pretrained=True) if use_mfcc else ResNet50ESC(pretrained=True)

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_fold(model, val_fold: int, device: str = 'cpu',
                 batch_size: int = 32, use_mfcc: bool = True):
    """Get predictions for one fold."""
    from torch.utils.data import DataLoader
    from data.augmentation import UrbanSoundDataset

    mel, mfcc, labels = load_fold_features(val_fold, FEATURES_DIR)
    ds = UrbanSoundDataset(mel, mfcc, labels, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds   = []
    all_probs   = []
    all_labels  = []

    for m, mc, lb in loader:
        m, mc, lb = m.to(device), mc.to(device), lb.to(device)
        if use_mfcc:
            logits = model(m, mc)
        else:
            logits = model(m)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(lb.cpu().numpy())

    del mel, mfcc, labels
    gc.collect()
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list = CLASS_NAMES) -> dict:
    """Compute comprehensive metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    per_class_prec   = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1     = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)

    return {
        'accuracy':       float(acc),
        'macro_f1':       float(macro_f1),
        'weighted_f1':    float(weighted_f1),
        'per_class': {
            name: {
                'precision': float(per_class_prec[i]),
                'recall':    float(per_class_recall[i]),
                'f1':        float(per_class_f1[i])
            }
            for i, name in enumerate(class_names)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          title: str = "Confusion Matrix",
                          save_path: str = None, normalize: bool = True):
    """Plot and save confusion matrix."""
    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = cm_plot / (row_sums + 1e-8)
    else:
        cm_plot = cm

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_plot, annot=True,
                fmt='.2f' if normalize else 'd',
                cmap='Blues',
                xticklabels=[c.replace('_', '\n') for c in class_names],
                yticklabels=[c.replace('_', '\n') for c in class_names],
                ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_per_class_metrics(metrics: dict, model_name: str,
                            save_path: str = None):
    """Bar chart of per-class F1 scores."""
    class_names = list(metrics['per_class'].keys())
    f1_scores = [metrics['per_class'][c]['f1'] for c in class_names]
    prec = [metrics['per_class'][c]['precision'] for c in class_names]
    rec  = [metrics['per_class'][c]['recall'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, prec, width, label='Precision', alpha=0.85, color='#2196F3')
    ax.bar(x, rec,  width, label='Recall',    alpha=0.85, color='#4CAF50')
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.85, color='#FF5722')

    # Mean F1 line
    mean_f1 = np.mean(f1_scores)
    ax.axhline(mean_f1, color='navy', linestyle='--', linewidth=1.5,
               label=f'Mean F1 = {mean_f1:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in class_names],
                        fontsize=9, ha='center')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Class Metrics – {model_name}", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_training_history(history: dict, fold: int, model_type: str,
                           save_path: str = None):
    """Plot training/validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#E53935')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#1E88E5',
             linestyle='--')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss – {model_type} Fold {fold}")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train Acc', color='#E53935')
    ax2.plot(epochs, history['val_acc'], label='Val Acc', color='#1E88E5',
             linestyle='--')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy – {model_type} Fold {fold}")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def evaluate_all_folds(model_type: str = 'yamnet',
                       n_folds: int = 8,
                       use_mfcc: bool = True,
                       device: str = 'cpu') -> dict:
    """
    Load best checkpoint for each fold, run predictions, aggregate metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating {model_type.upper()} across {n_folds} folds")
    print(f"{'='*60}")

    all_preds  = []
    all_labels = []
    fold_results = []

    for fold in range(1, n_folds + 1):
        ckpt_path = CHECKPOINTS / f"{model_type}_fold{fold}_best.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold}: checkpoint not found, skipping")
            continue

        print(f"  Fold {fold}: loading checkpoint...")
        model, ckpt = load_model_from_checkpoint(
            str(ckpt_path), model_type, use_mfcc, device)

        # Use saved predictions if available (faster)
        if 'val_preds' in ckpt and 'val_true' in ckpt:
            preds  = ckpt['val_preds']
            labels = ckpt['val_true']
        else:
            preds, probs, labels = predict_fold(model, fold, device, use_mfcc=use_mfcc)

        fold_acc = accuracy_score(labels, preds)
        fold_f1  = f1_score(labels, preds, average='macro', zero_division=0)
        print(f"    Fold {fold}: acc={fold_acc:.4f}, macro_f1={fold_f1:.4f}")

        # Plot training history
        if 'history' in ckpt:
            plot_training_history(
                ckpt['history'], fold, model_type,
                save_path=str(FIGURES_DIR / f"{model_type}_fold{fold}_history.png")
            )

        all_preds.extend(preds)
        all_labels.extend(labels)
        fold_results.append({'fold': fold, 'acc': fold_acc, 'f1': fold_f1,
                             'val_acc': ckpt.get('val_acc', fold_acc)})

        del model
        gc.collect()

    if not all_preds:
        print("No checkpoints found. Run training first.")
        return {}

    # Aggregate metrics
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_labels, all_preds, CLASS_NAMES)

    # Print summary
    print(f"\n  Aggregated Results ({model_type.upper()}):")
    print(f"    Overall Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"    Weighted F1      : {metrics['weighted_f1']:.4f}")
    print(f"\n    Per-Class F1:")
    for cls, vals in metrics['per_class'].items():
        print(f"      {cls:25s}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f}")

    # Save metrics
    metrics['fold_results'] = fold_results
    save_path_json = RESULTS_DIR / f"{model_type}_metrics.json"
    with open(save_path_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {save_path_json}")

    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        title=f"Confusion Matrix – {model_type.upper()} ({n_folds}-Fold CV)",
        save_path=str(FIGURES_DIR / f"{model_type}_confusion_matrix.png")
    )

    # Per-class bar chart
    plot_per_class_metrics(
        metrics, model_type.upper(),
        save_path=str(FIGURES_DIR / f"{model_type}_per_class_metrics.png")
    )

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['yamnet', 'resnet50', 'both'],
                        default='both')
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--no-mfcc", action="store_true")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models_to_eval = ['yamnet', 'resnet50'] if args.model == 'both' else [args.model]

    for model_type in models_to_eval:
        metrics = evaluate_all_folds(
            model_type=model_type,
            n_folds=args.n_folds,
            use_mfcc=not args.no_mfcc,
            device=device
        )
        if metrics:
            print(f"\n✓ {model_type} evaluation complete")
