"""
Chest X-Ray Dataset Pipeline  –  torchvision + OpenCV only
============================================================
3-class classification:
  0  ->  NORMAL      (images from NORMAL/ directory)
  1  ->  BACTERIAL   (PNEUMONIA/ images whose filename contains 'bacteria')
  2  ->  VIRAL       (PNEUMONIA/ images whose filename contains 'virus')

Preprocessing per image
-----------------------
1. Load with OpenCV (BGR, uint8).
2. Convert to grayscale → apply CLAHE → convert back to 3-channel RGB.
3. Pass through a torchvision transform pipeline:
     Train : RandomResizedCrop  →  RandomHorizontalFlip  →
             RandomRotation(10) →  ColorJitter            →
             ToTensor           →  Normalize(ImageNet)
     Val / Test : Resize(224) → CenterCrop(224) → ToTensor → Normalize

No albumentations dependency.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ──────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────
CLASS_NAMES: Dict[int, str] = {0: "NORMAL", 1: "BACTERIAL", 2: "VIRAL"}
NUM_CLASSES: int = 3
IMAGE_SIZE: int = 224

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

_VALID_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff"}


# ──────────────────────────────────────────────
# CLAHE helper
# ──────────────────────────────────────────────

def apply_clahe_rgb(image_rgb: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE on the luminance channel of a chest X-ray image.

    Steps
    -----
    1. Convert RGB → Grayscale (most CXR images are already grayscale,
       but we handle the RGB case gracefully).
    2. Run CLAHE on the single channel.
    3. Replicate to 3-channel RGB so downstream transforms stay consistent.

    Args:
        image_rgb  : Input image in RGB format (H, W, 3), dtype uint8.
        clip_limit : Contrast clipping limit for CLAHE.
        tile_grid  : Tile grid size for CLAHE.

    Returns:
        CLAHE-enhanced image in RGB format (H, W, 3), dtype uint8.
    """
    gray  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    gray_eq = clahe.apply(gray)
    # Expand single channel back to 3-channel RGB
    rgb_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
    return rgb_eq


# ──────────────────────────────────────────────
# Label parser
# ──────────────────────────────────────────────

def parse_label(folder_name: str, file_stem: str) -> int:
    """
    Derive the 3-class integer label from folder name + filename stem.

    Rules:
      folder == 'NORMAL'                        -> 0
      folder == 'PNEUMONIA' + 'bacteria' in stem -> 1
      folder == 'PNEUMONIA' + 'virus'    in stem -> 2
      folder == 'PNEUMONIA' + unknown            -> 1  (fallback: bacterial)

    Args:
        folder_name : Name of the class sub-directory.
        file_stem   : Filename without extension.

    Returns:
        Integer label {0, 1, 2}.
    """
    folder = folder_name.upper()
    stem   = file_stem.lower()

    if folder == "NORMAL":
        return 0
    elif folder == "PNEUMONIA":
        if "bacteria" in stem:
            return 1
        elif "virus" in stem:
            return 2
        else:
            return 1   # unknown pneumonia → bacterial
    else:
        raise ValueError(f"Unexpected folder name: {folder_name!r}")


# ──────────────────────────────────────────────
# Split scanner
# ──────────────────────────────────────────────

def scan_split(split_dir: Path) -> List[Tuple[Path, int]]:
    """
    Recursively collect (image_path, label) pairs from a split directory.

    Expected layout::

        split_dir/
            NORMAL/
                IM-0001.jpeg
            PNEUMONIA/
                person1_bacteria_1.jpeg
                person1_virus_2.jpeg

    Args:
        split_dir : Path to train / val / test directory.

    Returns:
        List of (Path, label_int) tuples.
    """
    samples: List[Tuple[Path, int]] = []

    for folder in sorted(split_dir.iterdir()):
        if not folder.is_dir():
            continue
        for img_file in sorted(folder.iterdir()):
            if img_file.suffix.lower() not in _VALID_EXTS:
                continue
            label = parse_label(folder.name, img_file.stem)
            samples.append((img_file, label))

    return samples


# ──────────────────────────────────────────────
# Transform factories
# ──────────────────────────────────────────────

def _train_transforms() -> transforms.Compose:
    """Augmented pipeline for the training split."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _eval_transforms() -> transforms.Compose:
    """Deterministic pipeline for val / test splits."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 16),      # 240  – slight oversize
        transforms.CenterCrop(IMAGE_SIZE),        # 224  – clean crop
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for the Chest X-Ray Pneumonia dataset.

    Parameters
    ----------
    samples : list of (Path, int)
        Image path / label pairs produced by :func:`scan_split`.
    transform : torchvision.transforms.Compose
        Torchvision augmentation / normalisation pipeline.
    apply_clahe : bool
        Whether to apply CLAHE before the transform pipeline.
    clahe_clip : float
        Clip limit for CLAHE (default 2.0).
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: transforms.Compose,
        apply_clahe: bool = True,
        clahe_clip: float = 2.0,
    ) -> None:
        self.samples     = samples
        self.transform   = transform
        self.apply_clahe = apply_clahe
        self.clahe_clip  = clahe_clip

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # ── Load via OpenCV (handles JPEG / PNG / BMP) ────────────────
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # H x W x 3, uint8

        # ── CLAHE ─────────────────────────────────────────────────────
        if self.apply_clahe:
            rgb = apply_clahe_rgb(rgb, clip_limit=self.clahe_clip)

        # ── PIL → torchvision transforms ──────────────────────────────
        pil_img = Image.fromarray(rgb)
        tensor  = self.transform(pil_img)             # (3, 224, 224) float32

        return tensor, label

    # ------------------------------------------------------------------
    def class_distribution(self) -> Dict[int, int]:
        """Return {class_id: count} for every class."""
        dist: Dict[int, int] = {k: 0 for k in CLASS_NAMES}
        for _, lbl in self.samples:
            dist[lbl] += 1
        return dist


# ──────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────

def get_dataloaders(
    data_root:   str   = "chest_xray",
    batch_size:  int   = 32,
    num_workers: int   = 2,
    apply_clahe: bool  = True,
    clahe_clip:  float = 2.0,
    pin_memory:  bool  = True,
) -> Dict[str, DataLoader]:
    """
    Build and return DataLoader objects for ``train``, ``val``, and ``test``.

    The function auto-detects whether the dataset is stored at *data_root*
    directly or one level deeper (common after extracting the Kaggle zip).

    Args:
        data_root   : Root directory that contains ``train/``, ``val/``, ``test/``.
        batch_size  : Batch size for all loaders.
        num_workers : Number of DataLoader worker processes.
        apply_clahe : Apply CLAHE preprocessing to each image.
        clahe_clip  : CLAHE clip limit.
        pin_memory  : Pin memory for faster CUDA transfer.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``,
        each mapped to a :class:`~torch.utils.data.DataLoader`.

    Raises:
        FileNotFoundError: If no valid dataset root can be located.
    """
    root = Path(data_root)

    # Auto-detect double-nested layout (chest_xray/chest_xray/train)
    for candidate in [root, root / "chest_xray"]:
        if (candidate / "train").is_dir():
            root = candidate
            break
    else:
        raise FileNotFoundError(
            f"Could not find 'train/' under '{data_root}'. "
            "Verify the dataset path."
        )

    transform_map = {
        "train": _train_transforms(),
        "val":   _eval_transforms(),
        "test":  _eval_transforms(),
    }

    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        split_dir = root / split
        samples   = scan_split(split_dir)

        dataset = ChestXRayDataset(
            samples     = samples,
            transform   = transform_map[split],
            apply_clahe = apply_clahe,
            clahe_clip  = clahe_clip,
        )

        dist = dataset.class_distribution()
        print(
            f"[{split.upper():5s}] {len(samples):5d} samples  |  "
            + "  ".join(
                f"{CLASS_NAMES[k]}: {dist[k]}"
                for k in sorted(CLASS_NAMES)
            )
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = pin_memory and torch.cuda.is_available(),
            drop_last   = (split == "train"),
            persistent_workers = (num_workers > 0),
        )

    return loaders
