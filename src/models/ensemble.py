"""
Ensemble Model for Environmental Sound Classification
Combines YAMNet and ResNet-50 predictions via soft voting or weighted ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class SoftVotingEnsemble(nn.Module):
    """
    Soft voting ensemble: average softmax probabilities from multiple models.
    All models must output (B, num_classes) logits.
    """

    def __init__(self, models: List[nn.Module],
                 weights: Optional[List[float]] = None,
                 num_classes: int = 10):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        assert len(weights) == len(models), "weights must match model count"
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def forward(self, mel: torch.Tensor,
                mfcc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Ensemble forward pass.
        For mixed-input models, pass both mel and mfcc.
        For mel-only models, only mel is used.
        """
        prob_sum = torch.zeros(mel.size(0), self.num_classes,
                               device=mel.device, dtype=mel.dtype)

        for model, weight in zip(self.models, self.weights):
            try:
                # Try two-argument call (MixedInput models)
                if mfcc is not None:
                    logits = model(mel, mfcc)
                else:
                    logits = model(mel)
            except TypeError:
                logits = model(mel)

            probs = F.softmax(logits, dim=1)
            prob_sum += weight * probs

        return prob_sum  # probabilities, not logits

    def predict(self, mel: torch.Tensor,
                mfcc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return predicted class indices."""
        with torch.no_grad():
            probs = self.forward(mel, mfcc)
        return torch.argmax(probs, dim=1)


class PerClassWeightedEnsemble:
    """
    Per-class weighted ensemble using validation performance.
    Learns optimal per-class weights from validation set.
    This is a post-hoc combination (not a nn.Module).
    """

    def __init__(self, num_models: int, num_classes: int = 10):
        self.num_models = num_models
        self.num_classes = num_classes
        # Uniform initialization
        self.weights = np.ones((num_classes, num_models)) / num_models

    def fit(self, val_probs: List[np.ndarray],
            val_labels: np.ndarray) -> None:
        """
        Learn weights from validation probabilities.
        val_probs: list of (N, C) arrays, one per model
        val_labels: (N,) true labels
        """
        for cls in range(self.num_classes):
            cls_mask = val_labels == cls
            if cls_mask.sum() == 0:
                continue
            # Greedy weight search via per-class accuracy
            best_acc = -1
            best_weights = np.ones(self.num_models) / self.num_models

            # Grid search over simplex
            for trial in range(200):
                w = np.random.dirichlet(np.ones(self.num_models))
                combined = sum(w[i] * val_probs[i][cls_mask]
                               for i in range(self.num_models))
                preds = np.argmax(combined, axis=1)
                acc = (preds == cls).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_weights = w

            self.weights[cls] = best_weights

    def predict_proba(self, probs: List[np.ndarray]) -> np.ndarray:
        """
        Weighted combination of model probabilities.
        probs: list of (N, C) arrays
        Returns: (N, C) weighted ensemble probabilities
        """
        N, C = probs[0].shape
        result = np.zeros((N, C))
        for cls in range(C):
            for i, p in enumerate(probs):
                result[:, cls] += self.weights[cls, i] * p[:, cls]
        # Re-normalize rows
        row_sums = result.sum(axis=1, keepdims=True)
        return result / (row_sums + 1e-8)


def ensemble_from_checkpoints(checkpoint_paths: List[str],
                               model_types: List[str],
                               device: str = 'cpu',
                               weights: Optional[List[float]] = None) -> SoftVotingEnsemble:
    """
    Build ensemble from saved checkpoints.
    model_types: list of 'yamnet' or 'resnet50'
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root / 'src'))

    from models.yamnet_transfer import MixedInputYAMNet
    from models.resnet50_transfer import MixedInputResNet

    loaded_models = []
    for path, mtype in zip(checkpoint_paths, model_types):
        if mtype == 'yamnet':
            m = MixedInputYAMNet(pretrained=False)
        else:
            m = MixedInputResNet(pretrained=False)

        try:
            ckpt = torch.load(path, map_location=device)
            m.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded {mtype} from {path} (val_acc={ckpt.get('val_acc', '?'):.3f})")
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")

        m.to(device)
        m.eval()
        loaded_models.append(m)

    return SoftVotingEnsemble(loaded_models, weights=weights)


if __name__ == "__main__":
    print("Testing Ensemble model...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.yamnet_transfer import YAMNetESC
    from models.resnet50_transfer import ResNet50ESC

    m1 = YAMNetESC(use_pretrained_mobilenet=False)
    m2 = ResNet50ESC(pretrained=False)

    ensemble = SoftVotingEnsemble([m1, m2], weights=[0.6, 0.4])

    x = torch.randn(4, 3, 224, 224)
    out = ensemble(x)
    print(f"  Ensemble output shape: {out.shape}")  # (4, 10)
    assert out.shape == (4, 10)
    print(f"  Probs sum check (should be ~1): {out.sum(dim=1)}")

    preds = ensemble.predict(x)
    print(f"  Predicted classes: {preds}")
    print("✓ Ensemble tests passed")
