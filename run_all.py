"""
run_all.py – Single Entry Point for the Complete ESC Project Pipeline

Runs all phases end-to-end:
  Phase 1: Dataset acquisition (synthetic or real UrbanSound8K)
  Phase 2: Feature extraction (mel spectrogram + MFCC, fold-by-fold)
  Phase 3: Training (YAMNet + ResNet-50, 8-fold cross-validation)
  Phase 4: Evaluation (metrics, confusion matrix, per-class analysis)
  Phase 5: Responsible AI (fairness, robustness, Grad-CAM)
  Phase 6: Visualization Dashboard

Usage:
  python run_all.py                           # Full pipeline
  python run_all.py --quick                  # Quick test (1 fold, 5 epochs)
  python run_all.py --model yamnet           # YAMNet only
  python run_all.py --skip-training          # Skip training, use existing checkpoints
  python run_all.py --epochs 30 --folds 1 2 # Specific folds + epochs

Memory notes:
  - CPU-only system (~16 GB RAM)
  - Feature extraction is fold-by-fold (memory safe)
  - Models loaded one at a time
  - Garbage collected between phases
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
from pathlib import Path

# ── Setup paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def print_banner():
    print("\n" + "="*70)
    print("  ROBUST ENVIRONMENTAL SOUND CLASSIFICATION")
    print("  Using Transfer Learning with Responsible AI Analysis")
    print()
    print("  IIT Jodhpur | Speech Understanding Project")
    print("  Team: Prem Kumar (B22AI031) · Akash Chaudhary (B22EE007)")
    print("        V.K Santosh (B22AI049)")
    print("="*70 + "\n")


def print_phase(n: int, title: str):
    print(f"\n{'─'*70}")
    print(f"  PHASE {n}: {title}")
    print(f"{'─'*70}")


def check_system():
    """Print system info and check requirements."""
    import psutil
    import torch

    print("System Check:")
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT))
    print(f"  RAM    : {mem.available/1e9:.1f} GB available / {mem.total/1e9:.1f} GB total")
    print(f"  Disk   : {disk.free/1e9:.1f} GB free")
    print(f"  CPU    : {psutil.cpu_count()} cores")
    print(f"  PyTorch: {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device : {device}")

    if mem.available < 4e9:
        print("\n⚠️  WARNING: Less than 4GB RAM available.")
        print("   Reducing batch size and fold count for safety.")
        return device, True  # memory_constrained = True
    return device, False


def phase1_dataset(args) -> object:
    """Phase 1: Dataset acquisition."""
    print_phase(1, "Dataset Acquisition")
    from data.download_dataset import (
        check_dataset_exists, create_synthetic_dataset,
        load_metadata, get_class_weights, CLASS_NAMES
    )

    n_per_class = 50 if args.quick else 150

    if check_dataset_exists():
        print("✓ Dataset already exists")
    else:
        print(f"Creating synthetic dataset ({n_per_class} clips/class)...")
        create_synthetic_dataset(n_per_class=n_per_class)

    df = load_metadata()
    print(f"\n✓ Dataset ready: {len(df)} clips, {df['class'].nunique()} classes")
    return df


def phase2_features(args, df) -> None:
    """Phase 2: Feature extraction."""
    print_phase(2, "Feature Extraction (Mel Spectrogram + MFCC)")
    from data.preprocess import process_all_folds

    folds = args.folds if args.folds else None
    process_all_folds(df, folds=folds, force=args.force_reextract)
    gc.collect()
    print("✓ Feature extraction complete")


def phase3_training(args, df, device: str) -> dict:
    """Phase 3: Training with k-fold cross-validation."""
    print_phase(3, "Model Training (Transfer Learning)")
    from data.download_dataset import get_class_weights
    from train import run_kfold_training

    model_types = ['yamnet', 'resnet50'] if args.model == 'both' else [args.model]
    folds_to_run = args.folds if args.folds else None

    all_results = {}
    for model_type in model_types:
        print(f"\n  Training: {model_type.upper()}")
        results = run_kfold_training(
            model_type=model_type,
            n_folds=8,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_mfcc=True,
            pretrained=not args.no_pretrained,
            folds_to_run=folds_to_run,
            device=device
        )
        all_results[model_type] = results
        gc.collect()

    return all_results


def phase4_evaluation(args, device: str) -> dict:
    """Phase 4: Comprehensive evaluation."""
    print_phase(4, "Evaluation & Metrics")
    from evaluate import evaluate_all_folds

    model_types = ['yamnet', 'resnet50'] if args.model == 'both' else [args.model]
    all_metrics = {}

    for model_type in model_types:
        metrics = evaluate_all_folds(
            model_type=model_type,
            n_folds=8,
            use_mfcc=True,
            device=device
        )
        all_metrics[model_type] = metrics
        gc.collect()

    return all_metrics


def phase5_responsible_ai(args, device: str) -> None:
    """Phase 5: Responsible AI Analysis."""
    print_phase(5, "Responsible AI Analysis (Fairness + Robustness + Grad-CAM)")

    model_types = ['yamnet', 'resnet50'] if args.model == 'both' else [args.model]

    for model_type in model_types:
        print(f"\n  [5a] Fairness Analysis – {model_type.upper()}")
        try:
            from responsible_ai.fairness import run_fairness_analysis
            run_fairness_analysis(model_type)
        except Exception as e:
            print(f"  Warning: {e}")
        gc.collect()

        print(f"\n  [5b] Robustness Testing – {model_type.upper()}")
        try:
            from responsible_ai.robustness import run_robustness_test
            run_robustness_test(model_type, fold=1, device=device)
        except Exception as e:
            print(f"  Warning: {e}")
        gc.collect()

        print(f"\n  [5c] Grad-CAM Explainability – {model_type.upper()}")
        try:
            from responsible_ai.explainability import run_gradcam_analysis
            n_samples = 1 if args.quick else 3
            run_gradcam_analysis(model_type, fold=1,
                                 n_samples_per_class=n_samples,
                                 device=device)
        except Exception as e:
            print(f"  Warning: {e}")
        gc.collect()


def phase6_dashboard() -> None:
    """Phase 6: Generate all visualizations."""
    print_phase(6, "Visualization Dashboard")
    try:
        from visualization.dashboard import generate_all_figures
        generate_all_figures()
    except Exception as e:
        print(f"  Warning: {e}")
    gc.collect()


def print_final_summary(start_time: float):
    """Print a summary of all generated outputs."""
    elapsed = time.time() - start_time
    hours, rem  = divmod(elapsed, 3600)
    minutes, sec = divmod(rem, 60)

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  ⏱ {int(hours):02d}h {int(minutes):02d}m {int(sec):02d}s")
    print(f"{'='*70}")

    # List key outputs
    print("\n  📊 Generated Outputs:")
    for subdir in ['results', 'figures', 'checkpoints']:
        d = OUTPUTS_DIR / subdir
        if d.exists():
            files = list(d.rglob('*'))
            files = [f for f in files if f.is_file()]
            print(f"\n    {subdir}/  ({len(files)} files)")
            for f in sorted(files)[:10]:
                size = f.stat().st_size
                print(f"      {f.relative_to(OUTPUTS_DIR)}"
                      f"  ({size/1024:.1f} KB)")
            if len(files) > 10:
                print(f"      ... and {len(files)-10} more")

    # Print key metrics
    for model in ['yamnet', 'resnet50']:
        p = RESULTS_DIR / f"{model}_metrics.json"
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            print(f"\n  🎯 {model.upper()} Results:")
            print(f"     Overall Accuracy : {m.get('accuracy', 0):.4f}")
            print(f"     Macro F1 Score   : {m.get('macro_f1', 0):.4f}")

    print(f"\n  📂 All outputs in: {OUTPUTS_DIR}")
    print(f"\n  Team: Prem Kumar · Akash Chaudhary · V.K Santosh | IIT Jodhpur")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ESC Project: End-to-end pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", choices=['yamnet', 'resnet50', 'both'],
                        default='both', help="Which model(s) to train")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Specific folds to run (default: all 8)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 fold, 5 epochs, small dataset")
    parser.add_argument("--skip-training",  action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-rai",       action="store_true",
                        help="Skip Responsible AI analysis")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Force re-extraction of features")
    parser.add_argument("--no-pretrained",  action="store_true",
                        help="Train from scratch (no pretrained weights)")
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs     = 5
        args.folds      = [1]
        print("⚡ QUICK MODE: 1 fold, 5 epochs, small synthetic dataset")

    start_time = time.time()
    print_banner()

    # Check system
    device, memory_constrained = check_system()
    if memory_constrained and args.batch_size > 16:
        args.batch_size = 16
        print(f"  Batch size reduced to {args.batch_size} due to memory constraints")

    # ── Run phases ────────────────────────────────────────────────
    try:
        # Phase 1: Dataset
        df = phase1_dataset(args)

        # Phase 2: Features
        phase2_features(args, df)
        del df
        gc.collect()

        # Phase 3: Training
        if not args.skip_training:
            phase3_training(args, None, device)
        else:
            print_phase(3, "Training – SKIPPED (using existing checkpoints)")

        # Phase 4: Evaluation
        if not args.skip_evaluation:
            phase4_evaluation(args, device)
        else:
            print_phase(4, "Evaluation – SKIPPED")

        # Phase 5: Responsible AI
        if not args.skip_rai:
            phase5_responsible_ai(args, device)
        else:
            print_phase(5, "Responsible AI – SKIPPED")

        # Phase 6: Dashboard
        if not args.skip_dashboard:
            phase6_dashboard()
        else:
            print_phase(6, "Dashboard – SKIPPED")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user. Partial outputs saved.")
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        print("\nPartial outputs may have been saved.")

    print_final_summary(start_time)


if __name__ == "__main__":
    main()
