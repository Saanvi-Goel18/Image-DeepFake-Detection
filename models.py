"""
Step 3: Model Architectures
=============================
Three ImageNet-pretrained baselines adapted for binary deepfake detection:
  1. MobileNetV3-Large  — lightweight, mobile-optimized
  2. EfficientNet-B0    — balanced efficiency/accuracy
  3. ResNet-50          — classic deep baseline

All output a single logit → BCEWithLogitsLoss

Usage:
    from models import get_model
    model = get_model('mobilenetv3_large')
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(name: str) -> nn.Module:
    """
    Factory function to create a pretrained model with binary classification head.

    Args:
        name: One of 'mobilenetv3_large', 'efficientnet_b0', 'resnet50'

    Returns:
        nn.Module with single-logit output
    """
    name = name.lower().strip()

    if name == 'mobilenetv3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        # classifier = Sequential(Linear(960, 1280), Hardswish, Dropout, Linear(1280, 1000))
        in_features = model.classifier[-1].in_features  # 1280
        model.classifier[-1] = nn.Linear(in_features, 1)
        print(f"[Model] MobileNetV3-Large: classifier[-1] → Linear({in_features}, 1)")

    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # classifier = Sequential(Dropout, Linear(1280, 1000))
        in_features = model.classifier[-1].in_features  # 1280
        model.classifier[-1] = nn.Linear(in_features, 1)
        print(f"[Model] EfficientNet-B0: classifier[-1] → Linear({in_features}, 1)")

    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # fc = Linear(2048, 1000)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Linear(in_features, 1)
        print(f"[Model] ResNet-50: fc → Linear({in_features}, 1)")

    else:
        raise ValueError(f"Unknown model: '{name}'. "
                         f"Choose from: mobilenetv3_large, efficientnet_b0, resnet50")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model


def count_parameters(model: nn.Module) -> dict:
    """Return parameter counts as a dictionary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    """Quick test: instantiate all 3 models and verify output shape."""
    from config import MODEL_NAMES, DEVICE, IMAGE_SIZE

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    for name in MODEL_NAMES:
        print(f"\n{'='*50}")
        model = get_model(name)
        model.eval()

        with torch.no_grad():
            output = model(dummy_input)

        print(f"  Input:  {dummy_input.shape}")
        print(f"  Output: {output.shape} → logit = {output.item():.4f}")
        print(f"  Prob:   {torch.sigmoid(output).item():.4f}")

        del model
        torch.cuda.empty_cache()
