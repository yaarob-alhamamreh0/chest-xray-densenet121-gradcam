"""
Quick Verification Script – Chest X-Ray DataLoaders
=====================================================
Run with:
    python test_dataset.py

Checks:
  1. Class distribution (count per class per split).
  2. Tensor shape from first batch: must be (batch_size, 3, 224, 224).
  3. Tensor dtype (float32) and value range (roughly -3 … 3 after ImageNet norm).
"""

import sys
import time
from pathlib import Path

# Force UTF-8 on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import get_dataloaders, CLASS_NAMES, IMAGE_SIZE

DATA_ROOT  = "chest_xray"
BATCH_SIZE = 32

# ──────────────────────────────────────────────
# Build loaders (also prints per-split counts)
# ──────────────────────────────────────────────
print("=" * 60)
print("  Building DataLoaders …")
print("=" * 60)

t0      = time.time()
loaders = get_dataloaders(
    data_root   = DATA_ROOT,
    batch_size  = BATCH_SIZE,
    num_workers = 0,          # 0 for Windows multiprocessing safety
    apply_clahe = True,
    pin_memory  = False,
)
print(f"\nBuilt in {time.time() - t0:.1f}s\n")

# ──────────────────────────────────────────────
# Batch shape + dtype checks
# ──────────────────────────────────────────────
print("=" * 60)
print("  Batch shape / dtype verification")
print("=" * 60)

all_passed = True
for split, loader in loaders.items():
    t1             = time.time()
    images, labels = next(iter(loader))
    elapsed        = time.time() - t1

    # Shape checks
    expected_channels = 3
    exp_h = exp_w = IMAGE_SIZE

    shape_ok  = (
        images.ndim == 4
        and images.shape[1] == expected_channels
        and images.shape[2] == exp_h
        and images.shape[3] == exp_w
    )
    dtype_ok  = str(images.dtype) == "torch.float32"
    labels_ok = labels.ndim == 1

    status = "PASS" if (shape_ok and dtype_ok and labels_ok) else "FAIL"
    if status == "FAIL":
        all_passed = False

    val_min = images.min().item()
    val_max = images.max().item()

    print(
        f"  [{split.upper():5s}] [{status}]  "
        f"images={tuple(images.shape)}  dtype={images.dtype}  "
        f"range=[{val_min:.2f}, {val_max:.2f}]  "
        f"labels={labels.tolist()[:6]}...  ({elapsed:.2f}s)"
    )

print()
print("=" * 60)
print(f"  Class map : {CLASS_NAMES}")
print("=" * 60)
if all_passed:
    print("  ALL CHECKS PASSED [OK]")
else:
    print("  SOME CHECKS FAILED -- review output above.")
print("=" * 60)
