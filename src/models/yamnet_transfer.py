"""
YAMNet-style Transfer Learning Model for Environmental Sound Classification
YAMNet uses MobileNetV1 backbone pretrained on AudioSet (521 classes).
We replicate the architecture using torchvision MobileNetV2 as backbone
(closest publicly available equivalent) and fine-tune a 10-class head.

Reference:
  Gemmeke et al. (2017). AudioSet: An ontology and human-labeled dataset
  for audio events. ICASSP 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from typing import Tuple, Optional


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (MobileNetV1 style)."""

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class YAMNetStyleBackbone(nn.Module):
    """
    Lightweight MobileNetV1-inspired backbone for audio classification.
    Input: (B, 3, 224, 224) mel spectrogram
    Output: feature map (B, 1024, 7, 7)
    """

    def __init__(self):
        super().__init__()
        # Standard MobileNetV1 conv + dw-sep stack (scaled down for CPU)
        def conv_bn(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )

        self.layers = nn.Sequential(
            conv_bn(3, 32, 2),                        # 112×112
            DepthwiseSeparableConv(32, 64, 1),        # 112×112
            DepthwiseSeparableConv(64, 128, 2),       # 56×56
            DepthwiseSeparableConv(128, 128, 1),      # 56×56
            DepthwiseSeparableConv(128, 256, 2),      # 28×28
            DepthwiseSeparableConv(256, 256, 1),      # 28×28
            DepthwiseSeparableConv(256, 512, 2),      # 14×14
            # 5× depthwise separable at stride 1
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 1024, 2),     # 7×7
            DepthwiseSeparableConv(1024, 1024, 1),    # 7×7
        )
        self.out_channels = 1024

    def forward(self, x):
        return self.layers(x)


class YAMNetESC(nn.Module):
    """
    YAMNet-style model fine-tuned on UrbanSound8K.
    
    Uses either:
    (a) Custom MobileNetV1-style backbone (trained from scratch) — lighter
    (b) MobileNetV2 pretrained backbone (transfer learning) — better
    
    Input : (B, 3, 224, 224)
    Output: (B, 10) logits
    """

    def __init__(self, num_classes: int = 10,
                 use_pretrained_mobilenet: bool = True,
                 freeze_features: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained_mobilenet

        if use_pretrained_mobilenet:
            # Use MobileNetV2 (best publicly available pretrained lightweight CNN)
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
            mob = models.mobilenet_v2(weights=weights)

            # Freeze feature extractor
            if freeze_features:
                for param in mob.features.parameters():
                    param.requires_grad = False
                # Unfreeze last 3 blocks for fine-tuning
                for block in mob.features[-3:]:
                    for param in block.parameters():
                        param.requires_grad = True

            self.backbone = mob.features     # output: (B, 1280, 7, 7)
            backbone_out  = 1280

        else:
            # Custom MobileNetV1-style (trains from scratch)
            self.backbone    = YAMNetStyleBackbone()
            backbone_out = 1024

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(512, num_classes)
        )

        # Init classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"YAMNetESC ({'MobileNetV2-pretrained' if use_pretrained_mobilenet else 'custom'}): "
              f"{total:,} total, {trainable:,} trainable ({100*trainable/total:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → logits: (B, 10)"""
        feat = self.backbone(x)       # (B, C, H, W)
        feat = self.gap(feat)         # (B, C, 1, 1)
        feat = feat.flatten(1)        # (B, C)
        return self.classifier(feat)

    def get_cam_target_layer(self):
        """Return target layer for Grad-CAM."""
        if self.use_pretrained:
            # Last conv layer of MobileNetV2 features
            return self.backbone[-1][0]  # ConvBNActivation
        else:
            return self.backbone.layers[-1].pw

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        feat = self.gap(feat).flatten(1)
        logits = self.classifier(feat)
        return feat, logits


class MixedInputYAMNet(nn.Module):
    """
    YAMNet + MFCC two-stream model.
    Mel spectrogram stream + MFCC stream with late fusion.
    """

    def __init__(self, num_classes: int = 10,
                 mfcc_dim: int = 80,
                 use_pretrained: bool = True,
                 freeze_features: bool = True,
                 dropout: float = 0.5):
        super().__init__()

        # Visual stream
        if use_pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
            mob = models.mobilenet_v2(weights=weights)
            if freeze_features:
                for param in mob.features.parameters():
                    param.requires_grad = False
                for block in mob.features[-3:]:
                    for param in block.parameters():
                        param.requires_grad = True
            self.backbone = mob.features
            backbone_out = 1280
        else:
            self.backbone = YAMNetStyleBackbone()
            backbone_out = 1024

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.mel_proj = nn.Sequential(
            nn.Linear(backbone_out, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
        )

        # MFCC stream
        self.mfcc_proj = nn.Sequential(
            nn.Linear(mfcc_dim, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MixedInputYAMNet: {total:,} total, {trainable:,} trainable")

    def forward(self, mel: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:
        v = self.gap(self.backbone(mel)).flatten(1)
        v = self.mel_proj(v)
        a = self.mfcc_proj(mfcc)
        return self.fusion(torch.cat([v, a], dim=1))


def get_yamnet_model(use_mfcc_fusion: bool = True,
                     num_classes: int = 10,
                     pretrained: bool = True) -> nn.Module:
    """Factory function."""
    if use_mfcc_fusion:
        return MixedInputYAMNet(num_classes=num_classes,
                                 use_pretrained=pretrained,
                                 freeze_features=True)
    return YAMNetESC(num_classes=num_classes,
                     use_pretrained_mobilenet=pretrained,
                     freeze_features=True)


if __name__ == "__main__":
    print("Testing YAMNet ESC models...")

    # Test YAMNet (pretrained=False to avoid download during test)
    model = YAMNetESC(use_pretrained_mobilenet=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"  YAMNetESC (custom) output: {out.shape}")
    assert out.shape == (2, 10)

    # Test MixedInput
    model2 = MixedInputYAMNet(use_pretrained=False)
    mfcc = torch.randn(2, 80)
    out2 = model2(x, mfcc)
    print(f"  MixedInputYAMNet output: {out2.shape}")
    assert out2.shape == (2, 10)

    print("✓ YAMNet model tests passed")
