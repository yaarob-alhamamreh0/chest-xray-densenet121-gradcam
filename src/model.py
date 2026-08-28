"""
Model Architecture & Grad-CAM  –  DenseNet-121
================================================
Transfer learning for 3-class chest X-ray classification:
  0 -> NORMAL
  1 -> BACTERIAL PNEUMONIA
  2 -> VIRAL PNEUMONIA

Backbone : torchvision DenseNet-121 (ImageNet DEFAULT weights)
Head     : Dropout(0.3) → Linear(1024 → 3)
Grad-CAM : hooks on features.denseblock4.denselayer16.conv2
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
NUM_CLASSES       = 3
DENSENET_FEATURES = 1024
# The last conv inside the final dense block layer
GRAD_CAM_LAYER    = "features.denseblock4.denselayer16.conv2"


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Return a DenseNet-121 with a custom 3-class classification head.

    The backbone weights are frozen initially during the first epochs;
    callers can unfreeze them by iterating ``model.features.parameters()``.

    Args:
        num_classes : Output dimension of the final linear layer.
        pretrained  : Load ImageNet DEFAULT weights when True.

    Returns:
        ``nn.Module`` — modified DenseNet-121 ready for fine-tuning.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model   = models.densenet121(weights=weights)

    # Replace the default (1000-class) classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(DENSENET_FEATURES, num_classes),
    )
    return model


# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────

class GradCAM:
    """
    Native PyTorch Grad-CAM, targeting the last convolutional layer of
    DenseNet-121's fourth dense block (``denseblock4.denselayer16.conv2``).

    Usage
    -----
    ::

        model.eval()
        cam      = GradCAM(model)
        heatmap  = cam(tensor_1x3x224x224, class_idx=1)   # (H, W) in [0,1]
        overlaid = GradCAM.overlay(original_rgb_hwc, heatmap)

    Parameters
    ----------
    model : nn.Module
        A DenseNet-121 built with :func:`build_model`.
    target_layer_name : str
        Dot-separated path to the target ``nn.Module`` inside *model*.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str = GRAD_CAM_LAYER,
    ) -> None:
        self.model       = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        target = self._resolve(model, target_layer_name)
        target.register_forward_hook(self._hook_fwd)
        target.register_full_backward_hook(self._hook_bwd)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve(model: nn.Module, dotted: str) -> nn.Module:
        module = model
        for part in dotted.split("."):
            module = getattr(module, part)
        return module

    def _hook_fwd(self, _m, _i, output) -> None:        # noqa: ANN001
        self._activations = output.detach()

    def _hook_bwd(self, _m, _gi, grad_output) -> None:  # noqa: ANN001
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    def __call__(
        self,
        image_tensor: torch.Tensor,   # (1, 3, H, W)
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM saliency map.

        Args:
            image_tensor : Pre-processed image batch of shape ``(1, 3, H, W)``.
            class_idx    : Target class. Defaults to the argmax prediction.

        Returns:
            Float32 numpy array of shape ``(H_feat, W_feat)`` in ``[0, 1]``.
        """
        self.model.eval()
        inp = image_tensor.requires_grad_(True)

        logits = self.model(inp)                              # (1, C)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool the gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1).squeeze()
        cam     = F.relu(cam).cpu().numpy()

        # Normalise to [0, 1]
        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def overlay(
        image_rgb: np.ndarray,
        heatmap:   np.ndarray,
        alpha:     float = 0.4,
    ) -> np.ndarray:
        """
        Blend a Grad-CAM heatmap on top of the original RGB image.

        Args:
            image_rgb : Original image ``(H, W, 3)`` uint8 RGB.
            heatmap   : Grad-CAM map ``(H_feat, W_feat)`` in ``[0, 1]``.
            alpha     : Heatmap weight in the blend.

        Returns:
            Blended image ``(H, W, 3)`` uint8 RGB.
        """
        h, w   = image_rgb.shape[:2]
        resized = cv2.resize(heatmap, (w, h))
        colored = cv2.applyColorMap(
            (resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        blended     = (alpha * colored_rgb + (1 - alpha) * image_rgb).astype(np.uint8)
        return blended
