"""
Train a CNN classifier for AI-Generated Image Detection.

Input: log-magnitude spectrum tensor (1 × 224 × 224) from image_preprocess.py
Output: binary classification (0 = real, 1 = fake)

All hyper-parameters are configurable in the CONFIG section below.
"""

import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- project modules ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_split import make_split, TEST_GENERATOR as DEFAULT_TEST_GEN
from image_preprocess import AIGeneratedImageDataset, IMG_SIZE

# ============================================================
#  CONFIGURABLE HYPER-PARAMETERS
# ============================================================
TEST_GENERATOR = DEFAULT_TEST_GEN   # which generator goes to test set
BATCH_SIZE     = 64
NUM_EPOCHS     = 10
LEARNING_RATE  = 5e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 0          # DataLoader workers (0 = main thread)
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
VAL_FRACTION   = 0.12       # fraction of train samples used for validation
EARLY_STOP_PATIENCE = 3     # stop if val loss doesn't improve for this many epochs


# ============================================================
#  VAL SPLIT HELPER
# ============================================================

def split_val_from_train(train_samples, val_fraction=VAL_FRACTION, seed=SEED):
    """Carve out val_fraction per class from train_samples.

    Returns (remaining_train, val) – both lists of sample dicts.
    Stratified so each class loses the same fraction.
    """
    import random as _random
    rng = _random.Random(seed)

    by_class = {}
    for s in train_samples:
        by_class.setdefault(s["label"], []).append(s)

    remaining, val = [], []
    for samples in by_class.values():
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        remaining.extend(shuffled[n_val:])

    rng.shuffle(remaining)
    rng.shuffle(val)
    return remaining, val

class FrequencyCNN(nn.Module):
    """CNN for binary classification of log-magnitude spectrum images.

    Architecture:
      4 × ConvBlock  (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
      AdaptiveAvgPool2d(4×4)
      FC head  (2048 → 256 → 1)

    Input:  (B, 1, 224, 224)
    Output: (B,)   raw logits (use with BCEWithLogitsLoss)
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(in_channels, 32),     # → 32 × 112 × 112
            conv_block(32, 64),              # → 64 ×  56 ×  56
            conv_block(64, 128),             # → 128 × 28 × 28
            conv_block(128, 256),            # → 256 × 14 × 14
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # → 256 × 4 × 4

        self.classifier = nn.Sequential(
            nn.Flatten(),                            # → 4096
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),                       # single logit
        )

        # Kaiming initialization for Conv2d layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x.squeeze(1)   # (B,)


# ============================================================
#  TRAINING & EVALUATION HELPERS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.float().to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (logits >= 0).long()
        correct += (preds == labels.long()).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * inputs.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    total = sum(l.size(0) for l in all_labels)
    epoch_loss = running_loss / total

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits)
    preds = (probs >= 0.5).long()
    labels_long = all_labels.long()

    acc = (preds == labels_long).float().mean().item()

    # Per-class metrics
    tp = ((preds == 1) & (labels_long == 1)).sum().float()
    fp = ((preds == 1) & (labels_long == 0)).sum().float()
    fn = ((preds == 0) & (labels_long == 1)).sum().float()
    tn = ((preds == 0) & (labels_long == 0)).sum().float()

    precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
    recall    = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels_long.numpy(), probs.numpy())
    except ImportError:
        # Manual AUC approximation (trapezoidal)
        auc = _approx_auc(probs.numpy(), labels_long.numpy())

    metrics = {
        "loss":      epoch_loss,
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auc":       auc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics


def _approx_auc(probs, labels):
    """Simple AUC approximation when sklearn is not available."""
    import numpy as np
    desc_indices = np.argsort(-probs)
    labels_sorted = labels[desc_indices]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = 0.0
    fpr = 0.0
    auc_val = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tpr += 1.0 / n_pos
        else:
            fpr += 1.0 / n_neg
        auc_val += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr
    return float(auc_val)


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train CNN for AI image detection")
    parser.add_argument("--test_gen", type=str, default=TEST_GENERATOR,
                        help="Generator used for test set")
    parser.add_argument("--epochs",  type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",      type=float, default=LEARNING_RATE)
    parser.add_argument("--device",  type=str, default=DEVICE)
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # ---- Reproducibility ----
    torch.manual_seed(SEED)

    # ---- Data ----
    print(f"\nBuilding data split (test generator = {args.test_gen}) ...")
    split = make_split(test_generator=args.test_gen, verbose=True)

    # Carve out validation set from training data (stratified per class)
    train_samples, val_samples = split_val_from_train(split["train"])
    print(f"  Train after val split : {len(train_samples)}")
    print(f"  Val                   : {len(val_samples)}")

    # Compute normalisation stats from the reduced training set only
    stats_ds = AIGeneratedImageDataset(train_samples)
    mean, std = stats_ds.compute_stats()

    train_ds = AIGeneratedImageDataset(train_samples, mean=mean, std=std, cache=True)
    val_ds   = AIGeneratedImageDataset(val_samples,   mean=mean, std=std, cache=True)
    test_ds  = AIGeneratedImageDataset(split["test"], mean=mean, std=std, cache=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, num_workers=NUM_WORKERS)

    # ---- Model ----
    model = FrequencyCNN(in_channels=1).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)

    # ---- Training loop ----
    best_val_f1   = 0.0
    best_val_loss = float("inf")
    no_improve    = 0          # epochs without val-loss improvement
    best_state    = None
    best_epoch    = 1

    print(f"\n{'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>6}  {'ValAUC':>7}  "
          f"{'Time':>6}  {'Note':>4}")
    print("-" * 100)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device)

        val_metrics = evaluate(model, val_loader, criterion, args.device)

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
            print(f"\nEarly stopping: val loss did not improve for "
                  f"{EARLY_STOP_PATIENCE} consecutive epochs.")
            break

    # ---- Evaluate best model on test set ----
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, args.device)

    print("\n" + "=" * 60)
    print(f"  Best val epoch   : {best_epoch}")
    print(f"  Best val F1      : {best_val_f1:.4f}")
    print(f"  --- Test set (best val checkpoint) ---")
    print(f"  Precision        : {test_metrics['precision']:.4f}")
    print(f"  Recall           : {test_metrics['recall']:.4f}")
    print(f"  F1               : {test_metrics['f1']:.4f}")
    print(f"  AUC              : {test_metrics['auc']:.4f}")
    print(f"  TP={test_metrics['tp']}  FP={test_metrics['fp']}  "
          f"TN={test_metrics['tn']}  FN={test_metrics['fn']}")
    print("=" * 60)

    # ---- Save model ----
    save_path = Path(__file__).resolve().parent / f"cnn_{args.test_gen}.pt"
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "test_generator": args.test_gen,
        "norm_mean": mean,
        "norm_std": std,
    }, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()