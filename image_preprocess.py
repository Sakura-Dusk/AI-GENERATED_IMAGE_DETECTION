"""
Image preprocessing for AI-Generated Image Detection.

Pipeline per image:
  1. Load image
  2. Resize to IMG_SIZE × IMG_SIZE
  3. Convert to grayscale
  4. Compute 2D Discrete Fourier Transform (DFT)
  5. Center the spectrum (fftshift)
  6. Take magnitude spectrum
  7. Apply log(1 + magnitude)
  8. Normalize to [0, 1]
  9. Convert to PyTorch tensor (1 × H × W)

Also provides a PyTorch Dataset class that combines this preprocessing
with the data split from dataset_split.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Optional

# ============================================================
#  CONFIGURABLE PARAMETER
# ============================================================
IMG_SIZE = 224   # Resize all images to this square size before DFT

# ============================================================
#  CORE PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image_path: str,
                     img_size: int = IMG_SIZE,
                     mean: Optional[float] = None,
                     std: Optional[float] = None) -> np.ndarray:
    """Apply the DFT-based preprocessing pipeline to a single image.

    Steps:
      1. Load image
      2. Resize to img_size × img_size
      3. Convert to grayscale
      4. Compute 2D DFT
      5. Center the spectrum (fftshift)
      6. Take magnitude spectrum
      7. Apply log(1 + magnitude)  (np.log1p)
      8. Z-score normalise using provided mean/std (computed on train set)

    Args:
        mean: Global mean of log-magnitude computed on the training set.
        std:  Global std  of log-magnitude computed on the training set.
              If either is None the raw log-magnitude is returned (use only
              for computing statistics, not for model input).

    Returns:
        np.ndarray of shape (H, W), dtype float32.
    """
    # 1–3. Load, resize, grayscale
    img = Image.open(image_path).convert("L")          # grayscale
    img = img.resize((img_size, img_size), Image.BICUBIC)

    # Convert to numpy (float32)
    img_arr = np.array(img, dtype=np.float32)

    # 4. 2D Discrete Fourier Transform
    dft = np.fft.fft2(img_arr)

    # 5. Center the spectrum  (swap quadrants so DC is in the middle)
    dft_shifted = np.fft.fftshift(dft)

    # 6. Magnitude spectrum
    magnitude = np.abs(dft_shifted)

    # 7. Log transformation: log(1 + magnitude)
    log_magnitude = np.log1p(magnitude)

    # 8. Z-score normalisation with train-set statistics
    if mean is not None and std is not None:
        log_magnitude = (log_magnitude - mean) / (std + 1e-8)

    return log_magnitude.astype(np.float32)


def preprocess_to_tensor(image_path: str,
                         img_size: int = IMG_SIZE,
                         mean: Optional[float] = None,
                         std: Optional[float] = None) -> torch.Tensor:
    """Preprocess an image and return a PyTorch tensor.

    Returns:
        torch.Tensor of shape (1, H, W), dtype float32.
    """
    log_mag = preprocess_image(image_path, img_size, mean=mean, std=std)
    tensor = torch.from_numpy(log_mag).unsqueeze(0)
    return tensor


# ============================================================
#  PYTORCH DATASET
# ============================================================

class AIGeneratedImageDataset(Dataset):
    """PyTorch Dataset for AI-generated image detection.

    Each sample is a z-score-normalised log-magnitude spectrum tensor
    (1 × H × W) and a binary label (0 = real, 1 = fake).

    Parameters
    ----------
    samples : list of dict
        Each dict has keys: "path" (str), "label" (int), "source" (str).
    img_size : int
        Resize images to this square size before computing DFT.
    mean, std : float or None
        Global statistics computed on the *training* set via compute_stats().
        Must be provided for the test set; for the training set they are
        computed internally when cache=True or can be set after construction.
    cache : bool
        If True, precompute all DFT features at init time and cache them
        in memory.
    """

    def __init__(self,
                 samples: List[Dict],
                 img_size: int = IMG_SIZE,
                 mean: Optional[float] = None,
                 std: Optional[float] = None,
                 cache: bool = False):
        self.samples  = samples
        self.img_size = img_size
        self.mean     = mean
        self.std      = std
        self._cache   = None

        if cache:
            self._cache = self._precompute_all()

    def compute_stats(self) -> tuple:
        """Compute global mean and std of log-magnitude over all samples.

        Iterates over every image WITHOUT applying normalisation so that the
        statistics are derived from the raw log-magnitude values.  Call this
        on the training dataset only, then pass the result to the test dataset.

        Returns:
            (mean, std) as Python floats.
        """
        print(f"  Computing normalisation stats from {len(self.samples)} images ...",
              end=" ", flush=True)
        running_sum  = 0.0
        running_sum2 = 0.0
        n_pixels     = 0
        for i, sample in enumerate(self.samples):
            arr = preprocess_image(sample["path"], self.img_size,
                                   mean=None, std=None)   # raw, no norm
            running_sum  += float(arr.sum())
            running_sum2 += float((arr ** 2).sum())
            n_pixels     += arr.size
            if (i + 1) % 1000 == 0:
                print(f"{i+1}", end=" ", flush=True)
        mean = running_sum  / n_pixels
        std  = (running_sum2 / n_pixels - mean ** 2) ** 0.5
        self.mean = mean
        self.std  = std
        print(f"done  mean={mean:.4f}  std={std:.4f}")
        return mean, std

    def _precompute_all(self):
        """Precompute DFT features for every sample."""
        n = len(self.samples)
        all_features = np.empty((n, 1, self.img_size, self.img_size),
                                dtype=np.float32)
        all_labels   = np.empty(n, dtype=np.int64)
        print(f"  Pre-computing DFT features for {n} images ...", end=" ",
              flush=True)
        for i, sample in enumerate(self.samples):
            all_features[i, 0] = preprocess_image(sample["path"],
                                                   self.img_size,
                                                   mean=self.mean,
                                                   std=self.std)
            all_labels[i] = sample["label"]
            if (i + 1) % 500 == 0:
                print(f"{i+1}", end=" ", flush=True)
        print(f"done ({n}/{n})")
        return {
            "features": torch.from_numpy(all_features),
            "labels":   torch.from_numpy(all_labels),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            return self._cache["features"][idx], self._cache["labels"][idx].item()

        sample = self.samples[idx]
        tensor = preprocess_to_tensor(sample["path"], self.img_size,
                                      mean=self.mean, std=self.std)
        return tensor, sample["label"]

    def get_sample_info(self, idx: int) -> Dict:
        """Return metadata for the sample at idx without loading the image."""
        return dict(self.samples[idx])


# ============================================================
#  QUICK TEST
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset_split import make_split

    # Build split
    split = make_split()

    # Create dataset (with cache for speed)
    train_ds = AIGeneratedImageDataset(split["train"], cache=True)
    test_ds  = AIGeneratedImageDataset(split["test"],  cache=True)

    print(f"Train dataset size : {len(train_ds)}")
    print(f"Test  dataset size : {len(test_ds)}")

    # Test loading one sample
    tensor, label = train_ds[0]
    print(f"Sample 0 – tensor shape: {tensor.shape}, dtype: {tensor.dtype}, "
          f"min: {tensor.min():.4f}, max: {tensor.max():.4f}, label: {label}")
    info = train_ds.get_sample_info(0)
    print(f"         source: {info['source']}, path: {info['path'][:60]}...")
