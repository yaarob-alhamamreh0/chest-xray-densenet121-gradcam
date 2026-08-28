from __future__ import annotations

"""
Training Pipeline  -  DenseNet-121  |  Chest X-Ray 3-Class
============================================================
Device  : CUDA (RTX 3060) with Mixed Precision (AMP)
Loss    : CrossEntropyLoss with inverse-frequency class weights
Optim   : AdamW (lr=1e-4, weight_decay=1e-4)
Sched   : CosineAnnealingLR (T_max = num_epochs)
Epochs  : 8
Output  : models/densenet121_xray_best.pt  (best val F1 checkpoint)

Run:
    python src/train.py
"""

import sys

# Force UTF-8 on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import numpy as np

# ── project imports ───────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # X_Chest/
sys.path.insert(0, str(ROOT))

from src.dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from src.model   import build_model

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_ROOT    = str(ROOT / "chest_xray")
MODELS_DIR   = ROOT / "models"
CKPT_PATH    = MODELS_DIR / "densenet121_xray_best.pt"

BATCH_SIZE   = 32
NUM_WORKERS  = 0          # 0 = main process; required on Windows
NUM_EPOCHS   = 8
LR           = 1e-4
WEIGHT_DECAY = 1e-4
LOG_INTERVAL = 20         # print every N batches
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP      = DEVICE.type == "cuda"   # Mixed precision only on CUDA

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def compute_class_weights(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Compute inverse-frequency class weights directly from dataset labels
    (no image I/O - instant).

    Formula:  w_c = N / (C * n_c)
    """
    labels = [lbl for _, lbl in loader.dataset.samples]
    counts = Counter(labels)
    total  = len(labels)
    weights = torch.tensor(
        [total / (NUM_CLASSES * max(counts[c], 1)) for c in range(NUM_CLASSES)],
        dtype=torch.float32,
    )
    return weights.to(DEVICE)


def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    epoch:     int,
) -> tuple[float, float]:
    """
    Run one full training epoch with AMP mixed precision.

    Returns:
        (avg_loss, accuracy) for this epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward pass under AMP autocast ──────────────────────────
        with autocast('cuda', enabled=USE_AMP):
            logits = model(images)
            loss   = criterion(logits, labels)

        # ── Backward pass via GradScaler ──────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(
                f"  Epoch {epoch:02d}  "
                f"[{batch_idx+1:3d}/{n_batches}]  "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    avg_loss = running_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """
    Evaluate on val or test split.

    Returns:
        (avg_loss, accuracy, macro_f1)
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast('cuda', enabled=USE_AMP):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    amp_label = "ON (Tensor Cores)" if USE_AMP else "OFF"
    gpu_name  = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"

    print("=" * 65, flush=True)
    print(f"  Device : {DEVICE}  ({gpu_name})", flush=True)
    print(f"  AMP    : {amp_label}", flush=True)
    print(f"  Epochs : {NUM_EPOCHS}   |   Batch : {BATCH_SIZE}   |   LR : {LR}", flush=True)
    print("=" * 65, flush=True)

    # ── DataLoaders ───────────────────────────────────────────────────
    print("\nLoading dataset...", flush=True)
    loaders = get_dataloaders(
        data_root   = DATA_ROOT,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,        # 0 = safe on Windows
        apply_clahe = True,
        pin_memory  = (DEVICE.type == "cuda"),  # pin for faster GPU transfer
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]

    # ── Class weights ─────────────────────────────────────────────────
    print("\nComputing class weights...", flush=True)
    class_weights = compute_class_weights(train_loader)
    print("  Weights:", {CLASS_NAMES[i]: f"{class_weights[i].item():.4f}"
                         for i in range(NUM_CLASSES)}, flush=True)

    # ── Model + optimizers ────────────────────────────────────────────
    print("\nBuilding model...", flush=True)
    model     = build_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = GradScaler('cuda', enabled=USE_AMP)

    # ── Training loop ─────────────────────────────────────────────────
    best_f1    = 0.0
    best_epoch = 0
    history    = []

    print("\n" + "-" * 65, flush=True)
    print("  Starting training...", flush=True)
    print("-" * 65, flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        history.append(dict(
            epoch=epoch,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss,     val_acc=val_acc,     val_f1=val_f1,
        ))

        print(
            f"\nEpoch {epoch:02d}/{NUM_EPOCHS}  ({elapsed:.0f}s)  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_F1={val_f1:.4f}",
            flush=True,
        )

        # ── Checkpoint best model by val F1 ──────────────────────────
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_f1":      val_f1,
                    "val_acc":     val_acc,
                    "class_names": CLASS_NAMES,
                },
                CKPT_PATH,
            )
            print(f"  [SAVED] Best model -> {CKPT_PATH.name}  (val_F1={best_f1:.4f})", flush=True)

    # ── Test evaluation with best checkpoint ──────────────────────────
    print("\n" + "=" * 65, flush=True)
    print(f"  Loading best checkpoint (epoch {best_epoch}, F1={best_f1:.4f})...", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    print(f"  Test   loss={test_loss:.4f}  acc={test_acc:.4f}  macro_F1={test_f1:.4f}", flush=True)

    # ── Full per-class report + confusion matrix ──────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(DEVICE)).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n  Classification Report:", flush=True)
    print(
        classification_report(
            all_labels, all_preds,
            target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
            digits=4,
        ),
        flush=True,
    )

    print("  Confusion Matrix  (rows=actual, cols=predicted):", flush=True)
    cm = confusion_matrix(all_labels, all_preds)
    header = "              " + "  ".join(f"{CLASS_NAMES[i]:>10s}" for i in range(NUM_CLASSES))
    print(header, flush=True)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10d}" for v in row)
        print(f"  {CLASS_NAMES[i]:10s}  {row_str}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print(f"  Training complete.  Best val F1 = {best_f1:.4f}  (epoch {best_epoch})", flush=True)
    print(f"  Checkpoint saved to: {CKPT_PATH}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
