"""
Train a CNN classifier using AutoGAN-simulated fake images.

Based on: "Detecting and Simulating Artifacts in GAN Fake Images" (Zhang et al., 2019)

Training strategy:
  - Train set : real Nature images + AutoGAN-simulated fakes (1:1 balanced)
  - Val set   : another portion of Nature images + their simulated fakes
  - Test set  : remaining Nature images (real) + all 7 AI generators (fake)
                → per-generator metrics + overall metrics reported separately
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import FrequencyCNN, train_one_epoch, _approx_auc
from image_preprocess import preprocess_to_tensor, IMG_SIZE
from simulate_fake import simulate_image
from dataset_split import parse_nature_metadata, NATURE_DIR, BASE_DIR, META_CSV

# ============================================================
#  CONFIG
# ============================================================
BATCH_SIZE          = 64
NUM_EPOCHS          = 10
LEARNING_RATE       = 5e-4
WEIGHT_DECAY        = 1e-4
NUM_WORKERS         = 0
SEED                = 42
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
VAL_FRACTION        = 0.12
EARLY_STOP_PATIENCE = 3

GENERATOR_DIRS = ["ADM", "BigGAN", "Midjourney", "VQDM",
                  "glide", "stable_diffusion_v_1_5", "wukong"]

# ============================================================
#  DATA SPLIT
# ============================================================

ALL_SIM_MODES = ["nearest", "bilinear", "bicubic"]


def make_autogan_split(
    sim_mode: str = "random",
    sim_scale: int = 2,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """Build train / val / test splits.

    Train & val: Nature images paired with AutoGAN-simulated fakes (1:1).
    Test: Nature val.* images (real) + all 7 generator directories (fake).

    Each sample dict keys: path, label, source, sim_mode, sim_scale.
      sim_mode=None  → load directly (real image or real-generator fake)
      sim_mode=str   → load path, apply simulate_image(), then DFT

    When sim_mode="random", each simulated fake image independently draws
    one of the five simulation modes at random (seeded for reproducibility).
    """
    meta = parse_nature_metadata(META_CSV)

    nature_trainval: List[str] = []
    nature_test: List[str] = []
    for fname, tag in meta.items():
        fpath = NATURE_DIR / fname
        if not fpath.is_file():
            continue
        if tag.startswith("val."):
            nature_test.append(str(fpath))
        else:
            nature_trainval.append(str(fpath))

    rng = random.Random(seed)
    rng.shuffle(nature_trainval)
    n_val = max(1, round(len(nature_trainval) * val_fraction))
    nature_val   = nature_trainval[:n_val]
    nature_train = nature_trainval[n_val:]

    def _pick_mode() -> str:
        return rng.choice(ALL_SIM_MODES) if sim_mode == "random" else sim_mode

    def _pair(paths: List[str]) -> List[Dict]:
        samples = []
        for p in paths:
            samples.append({"path": p, "label": 0, "source": "Nature",
                             "sim_mode": None, "sim_scale": None})
            chosen = _pick_mode()
            samples.append({"path": p, "label": 1,
                             "source": f"simulated_{chosen}",
                             "sim_mode": chosen, "sim_scale": sim_scale})
        return samples

    train_samples = _pair(nature_train)
    val_samples   = _pair(nature_val)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    test_samples: List[Dict] = []
    for p in nature_test:
        test_samples.append({"path": p, "label": 0, "source": "Nature",
                              "sim_mode": None, "sim_scale": None})
    for gen in GENERATOR_DIRS:
        gen_dir = BASE_DIR / gen
        if not gen_dir.is_dir():
            continue
        for f in sorted(os.listdir(gen_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                test_samples.append({"path": str(gen_dir / f), "label": 1,
                                     "source": gen, "sim_mode": None,
                                     "sim_scale": None})

    if verbose:
        n_tr_real = sum(1 for s in train_samples if s["label"] == 0)
        n_tr_fake = sum(1 for s in train_samples if s["label"] == 1)
        n_va_real = sum(1 for s in val_samples   if s["label"] == 0)
        n_va_fake = sum(1 for s in val_samples   if s["label"] == 1)
        n_te_real = sum(1 for s in test_samples  if s["label"] == 0)
        n_te_fake = sum(1 for s in test_samples  if s["label"] == 1)
        mode_label = "random (uniform over 5 modes)" if sim_mode == "random" else sim_mode
        print("=" * 60)
        print(f"  Simulation mode : {mode_label}  scale={sim_scale}")
        if sim_mode == "random":
            from collections import Counter
            mode_counts = Counter(
                s["sim_mode"] for s in train_samples if s["sim_mode"] is not None
            )
            print(f"  Train mode dist : { {k: mode_counts[k] for k in ALL_SIM_MODES if k in mode_counts} }")
        print(f"  Train  real={n_tr_real}  fake={n_tr_fake}  total={len(train_samples)}")
        print(f"  Val    real={n_va_real}  fake={n_va_fake}  total={len(val_samples)}")
        print(f"  Test   real={n_te_real}  fake={n_te_fake}  total={len(test_samples)}")
        print("=" * 60)

    return {"train": train_samples, "val": val_samples, "test": test_samples}

# ============================================================
#  PIL → DFT TENSOR BRIDGE
# ============================================================

def preprocess_pil_to_tensor(
    pil_img: Image.Image,
    img_size: int = IMG_SIZE,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> torch.Tensor:
    """Apply the DFT preprocessing pipeline to a PIL image.

    Identical pipeline to image_preprocess.preprocess_image, but accepts a
    PIL object instead of a file path — used after simulate_image().
    """
    img = pil_img.convert("L")
    img = img.resize((img_size, img_size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32)
    dft_shifted = np.fft.fftshift(np.fft.fft2(arr))
    log_mag = np.log1p(np.abs(dft_shifted))
    if mean is not None and std is not None:
        log_mag = (log_mag - mean) / (std + 1e-8)
    return torch.from_numpy(log_mag.astype(np.float32)).unsqueeze(0)

# ============================================================
#  DATASET
# ============================================================

class SimulatedFakeDataset(Dataset):
    """Dataset supporting on-the-fly AutoGAN simulation.

    For samples with sim_mode=None  : load image from disk directly.
    For samples with sim_mode=str   : load image → simulate_image() → DFT.
    """

    def __init__(
        self,
        samples: List[Dict],
        img_size: int = IMG_SIZE,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        cache: bool = False,
    ):
        self.samples  = samples
        self.img_size = img_size
        self.mean     = mean
        self.std      = std
        self._cache   = None

        if cache:
            self._cache = self._precompute_all()

    def _get_tensor(self, sample: Dict) -> torch.Tensor:
        if sample["sim_mode"] is None:
            return preprocess_to_tensor(sample["path"], self.img_size,
                                        mean=self.mean, std=self.std)
        else:
            pil = Image.open(sample["path"]).convert("RGB")
            pil_fake = simulate_image(pil,
                                      mode=sample["sim_mode"],
                                      scale=sample["sim_scale"])
            return preprocess_pil_to_tensor(pil_fake, self.img_size,
                                            mean=self.mean, std=self.std)

    def compute_stats(self) -> Tuple[float, float]:
        """Compute global mean and std of log-magnitude over all samples.

        Call on the training dataset only (before setting mean/std),
        then pass the returned values to val and test datasets.
        """
        print(f"  Computing stats from {len(self.samples)} samples ...",
              end=" ", flush=True)
        saved_mean, saved_std = self.mean, self.std
        self.mean = self.std = None   # raw values, no normalisation
        running_sum = running_sum2 = 0.0
        n_pixels = 0
        for i, sample in enumerate(self.samples):
            arr = self._get_tensor(sample).numpy()
            running_sum  += float(arr.sum())
            running_sum2 += float((arr ** 2).sum())
            n_pixels     += arr.size
            if (i + 1) % 500 == 0:
                print(f"{i+1}", end=" ", flush=True)
        mean = running_sum / n_pixels
        std  = (running_sum2 / n_pixels - mean ** 2) ** 0.5
        self.mean, self.std = saved_mean, saved_std
        print(f"done  mean={mean:.4f}  std={std:.4f}")
        return mean, std

    def _precompute_all(self) -> Dict:
        n = len(self.samples)
        features = np.empty((n, 1, self.img_size, self.img_size), dtype=np.float32)
        labels   = np.empty(n, dtype=np.int64)
        print(f"  Pre-computing {n} DFT features ...", end=" ", flush=True)
        for i, sample in enumerate(self.samples):
            features[i, 0] = self._get_tensor(sample).squeeze(0).numpy()
            labels[i] = sample["label"]
            if (i + 1) % 500 == 0:
                print(f"{i+1}", end=" ", flush=True)
        print(f"done ({n}/{n})")
        return {
            "features": torch.from_numpy(features),
            "labels":   torch.from_numpy(labels),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            return self._cache["features"][idx], int(self._cache["labels"][idx])
        sample = self.samples[idx]
        return self._get_tensor(sample), sample["label"]

# ============================================================
#  EVALUATE (per-source)
# ============================================================

def _compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict:
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    acc       = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = _approx_auc(probs, labels)
    return dict(accuracy=acc, precision=precision, recall=recall, f1=f1, auc=auc,
                tp=tp, fp=fp, fn=fn, tn=tn,
                n_real=int((labels == 0).sum()), n_fake=int((labels == 1).sum()))


@torch.no_grad()
def evaluate_per_source(
    model: nn.Module,
    test_samples: List[Dict],
    device: str,
    batch_size: int = 64,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    img_size: int = IMG_SIZE,
) -> Dict:
    """Run inference on all test samples, then compute per-source metrics.

    Returns:
        {
          "overall":    {metrics},
          "per_source": {"ADM": {metrics}, "BigGAN": {metrics}, ...}
        }

    Note: Nature real images appear once in 'overall' but are reused for each
    generator's per-source evaluation (paired separately with each generator).
    """
    model.eval()
    all_probs:   List[float] = []
    all_labels:  List[int]   = []
    all_sources: List[str]   = []

    print("  Running inference on test set ...", end=" ", flush=True)
    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start: start + batch_size]
        tensors = [preprocess_to_tensor(s["path"], img_size, mean=mean, std=std)
                   for s in batch]
        inputs = torch.stack(tensors).to(device)
        logits = model(inputs)
        probs  = torch.sigmoid(logits).cpu().tolist()
        for s, p in zip(batch, probs):
            all_probs.append(p)
            all_labels.append(s["label"])
            all_sources.append(s["source"])
    print("done")

    all_probs_arr  = np.array(all_probs,  dtype=np.float32)
    all_labels_arr = np.array(all_labels, dtype=np.int64)

    # Overall (Nature真图只计一次)
    overall = _compute_metrics(all_probs_arr, all_labels_arr)

    # Collect Nature real image probs/labels (shared across all generators)
    nature_mask  = np.array([s == "Nature" for s in all_sources])
    nat_probs    = all_probs_arr[nature_mask]
    nat_labels   = all_labels_arr[nature_mask]

    # Per-source: each generator paired with the full set of Nature real images
    per_source: Dict[str, Dict] = {}
    for gen in GENERATOR_DIRS:
        gen_mask   = np.array([s == gen for s in all_sources])
        gen_probs  = all_probs_arr[gen_mask]
        gen_labels = all_labels_arr[gen_mask]
        if len(gen_probs) == 0:
            continue
        combined_probs  = np.concatenate([nat_probs,  gen_probs])
        combined_labels = np.concatenate([nat_labels, gen_labels])
        per_source[gen] = _compute_metrics(combined_probs, combined_labels)

    return {"overall": overall, "per_source": per_source}


def _print_results(results: Dict) -> None:
    header = (f"{'Generator':<28} {'Acc':>6} {'Prec':>6} {'Recall':>7} "
              f"{'F1':>6} {'AUC':>6} {'N_real':>7} {'N_fake':>7}")
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    def _row(name: str, m: Dict) -> str:
        return (f"{name:<28} {m['accuracy']:6.4f} {m['precision']:6.4f} "
                f"{m['recall']:7.4f} {m['f1']:6.4f} {m['auc']:6.4f} "
                f"{m['n_real']:7d} {m['n_fake']:7d}")

    for gen, m in results["per_source"].items():
        print(_row(gen, m))
    print(sep)
    print(_row("ALL (overall)", results["overall"]))
    print(sep + "\n")

# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train FrequencyCNN with AutoGAN-simulated fakes")
    parser.add_argument("--sim_mode",  type=str,   default="random",
                        choices=["random", "nearest", "bilinear", "bicubic",
                                 "transposed", "jpeg"])
    parser.add_argument("--sim_scale", type=int,   default=2)
    parser.add_argument("--epochs",    type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=LEARNING_RATE)
    parser.add_argument("--val_frac",  type=float, default=VAL_FRACTION)
    parser.add_argument("--device",    type=str,   default=DEVICE)
    parser.add_argument("--no_cache",  action="store_true",
                        help="Disable DFT pre-caching (saves RAM, slower training)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    print(f"Device : {args.device}")

    # ---- Split ----
    split = make_autogan_split(sim_mode=args.sim_mode, sim_scale=args.sim_scale,
                               val_fraction=args.val_frac, verbose=True)

    # ---- Stats (from training set only) ----
    stats_ds = SimulatedFakeDataset(split["train"], img_size=IMG_SIZE)
    mean, std = stats_ds.compute_stats()
    del stats_ds

    # ---- Datasets ----
    use_cache = not args.no_cache
    train_ds = SimulatedFakeDataset(split["train"], mean=mean, std=std, cache=use_cache)
    val_ds   = SimulatedFakeDataset(split["val"],   mean=mean, std=std, cache=use_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=NUM_WORKERS)

    # ---- Model ----
    model     = FrequencyCNN(in_channels=1).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}\n")

    # ---- Training loop ----
    best_val_f1   = 0.0
    best_val_loss = float("inf")
    no_improve    = 0
    best_state    = None
    best_epoch    = 1

    # Reuse evaluate from train.py for val set (global metrics sufficient here)
    from train import evaluate as _evaluate

    print(f"{'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>6}  {'ValAUC':>7}  "
          f"{'Time':>6}  {'Note':>4}")
    print("-" * 82)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device)
        val_metrics = _evaluate(model, val_loader, criterion, args.device)
        scheduler.step()
        elapsed = time.time() - t0

        note = ""
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch  = epoch
            best_state  = {k: v.cpu().clone() if torch.is_tensor(v) else v
                           for k, v in model.state_dict().items()}
            note = "*"

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve = 0
        else:
            no_improve += 1

        print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:9.4f}  "
              f"{val_metrics['loss']:8.4f}  {val_metrics['accuracy']:7.4f}  "
              f"{val_metrics['f1']:6.4f}  {val_metrics['auc']:7.4f}  "
              f"{elapsed:5.1f}s  {note}")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: val loss no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs.")
            break

    # ---- Test: per-source evaluation ----
    print(f"\nBest checkpoint: epoch {best_epoch}  val_F1={best_val_f1:.4f}")
    model.load_state_dict(best_state)

    results = evaluate_per_source(
        model, split["test"], device=args.device,
        batch_size=args.batch, mean=mean, std=std,
    )
    _print_results(results)

    # ---- Save ----
    save_path = Path(__file__).resolve().parent / f"cnn_autogan_{args.sim_mode}.pt"
    torch.save({
        "epoch":            best_epoch,
        "model_state_dict": best_state,
        "best_val_f1":      best_val_f1,
        "test_results":     results,
        "sim_mode":         args.sim_mode,
        "sim_scale":        args.sim_scale,
        "norm_mean":        mean,
        "norm_std":         std,
    }, save_path)
    print(f"Model saved → {save_path}")


if __name__ == "__main__":
    main()
