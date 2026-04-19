"""
ResNet-50 Transfer Learning Model for Environmental Sound Classification
- Pretrained on ImageNet (torchvision)
- Frozen layers 1-3 (only layer4 + fc fine-tuned)
- Input: 3×224×224 Log Mel Spectrogram
- Output: 10-class softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from typing import Optional, Tuple


class ResNet50ESC(nn.Module):
    """
    ResNet-50 fine-tuned for Environmental Sound Classification.
    
    Architecture:
        ResNet-50 backbone (ImageNet pretrained)
        → Global Average Pooling
        → Dropout(0.5)
        → Linear(2048, 256) + ReLU
        → Dropout(0.3)
        → Linear(256, 10)
    
    Input : (B, 3, 224, 224) - Log Mel Spectrogram
    Output: (B, 10) - logits
    """

    def __init__(self, num_classes: int = 10,
                 freeze_layers: int = 3,
                 dropout: float = 0.5,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Freeze early layers for transfer learning
        # ResNet-50 layers: layer1, layer2, layer3, layer4
        layers_to_freeze = []
        if freeze_layers >= 0:
            layers_to_freeze += [backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool]
        if freeze_layers >= 1:
            layers_to_freeze.append(backbone.layer1)
        if freeze_layers >= 2:
            layers_to_freeze.append(backbone.layer2)
        if freeze_layers >= 3:
            layers_to_freeze.append(backbone.layer3)

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        # Strip the final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # up to avgpool

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fine-tuned classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(256, num_classes)
        )

        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ResNet50ESC: {total:,} total params, {trainable:,} trainable "
              f"({100*trainable/total:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 3, 224, 224)"""
        feat = self.backbone(x)    # (B, 2048, 7, 7)
        feat = self.gap(feat)      # (B, 2048, 1, 1)
        feat = feat.flatten(1)     # (B, 2048)
        logits = self.classifier(feat)  # (B, 10)
        return logits

    def get_cam_target_layer(self):
        """Return the target layer for Grad-CAM visualization."""
        # Last layer of layer4 in backbone
        return list(self.backbone.children())[-1][-1].conv3

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both features and logits."""
        feat = self.backbone(x)
        feat = self.gap(feat).flatten(1)
        logits = self.classifier(feat)
        return feat, logits


class MixedInputResNet(nn.Module):
    """
    ResNet-50 + MFCC fusion model.
    
    Two-stream architecture:
        Stream 1: ResNet-50 on mel spectrogram → 256-dim features
        Stream 2: MLP on MFCC → 64-dim features
        Fusion: concat → 10-class output
    """

    def __init__(self, num_classes: int = 10,
                 mfcc_dim: int = 80,
                 freeze_layers: int = 3,
                 dropout: float = 0.5,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Visual stream (ResNet-50 backbone)
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        layers_to_freeze = [backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool]
        for ln in range(1, freeze_layers + 1):
            layers_to_freeze.append(getattr(backbone, f'layer{ln}'))
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.mel_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
        )

        # MFCC stream
        self.mfcc_head = nn.Sequential(
            nn.Linear(mfcc_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MixedInputResNet: {total:,} total params, {trainable:,} trainable "
              f"({100*trainable/total:.1f}%)")

    def forward(self, mel: torch.Tensor,
                mfcc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel  : (B, 3, 224, 224)
            mfcc : (B, 80)
        Returns:
            logits: (B, 10)
        """
        # Visual stream
        v = self.backbone(mel)
        v = self.gap(v).flatten(1)   # (B, 2048)
        v = self.mel_head(v)          # (B, 256)

        # MFCC stream
        a = self.mfcc_head(mfcc)      # (B, 64)

        # Fusion
        fused = torch.cat([v, a], dim=1)  # (B, 320)
        return self.fusion(fused)


def get_resnet50_model(use_mfcc_fusion: bool = False,
                       num_classes: int = 10,
                       pretrained: bool = True) -> nn.Module:
    """Factory function to get ResNet-50 model."""
    if use_mfcc_fusion:
        return MixedInputResNet(num_classes=num_classes, pretrained=pretrained)
    return ResNet50ESC(num_classes=num_classes, pretrained=pretrained)


def save_checkpoint(model: nn.Module, path: str, epoch: int,
                    val_acc: float, optimizer=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'model_class': model.__class__.__name__,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, path: str, device: str = 'cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('val_acc', 0.0), checkpoint.get('epoch', 0)


if __name__ == "__main__":
    print("Testing ResNet-50 ESC models...")

    # Test basic model
    model = ResNet50ESC(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"  ResNet50ESC output shape: {out.shape}")  # (4, 10)
    assert out.shape == (4, 10), "Shape mismatch!"

    # Test fusion model
    model2 = MixedInputResNet(pretrained=False)
    mfcc = torch.randn(4, 80)
    out2 = model2(x, mfcc)
    print(f"  MixedInputResNet output: {out2.shape}")
    assert out2.shape == (4, 10)

    print("✓ ResNet-50 model tests passed")
