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
                     img_size: int = IMG_SIZE) -> np.ndarray:
    """Apply the DFT-based preprocessing pipeline to a single image.

    Steps:
      1. Load image
      2. Resize to img_size × img_size
      3. Convert to grayscale
      4. Compute 2D DFT
      5. Center the spectrum (fftshift)
      6. Take magnitude spectrum
      7. Apply log(1 + magnitude)  (np.log1p)
      8. Normalize to [0, 1]

    Returns:
        np.ndarray of shape (H, W), dtype float32, values in [0, 1].
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

    # 8. Normalize to [0, 1]
    max_val = log_magnitude.max()
    if max_val > 0:
        log_magnitude = log_magnitude / max_val

    return log_magnitude.astype(np.float32)


def preprocess_to_tensor(image_path: str,
                         img_size: int = IMG_SIZE) -> torch.Tensor:
    """Preprocess an image and return a PyTorch tensor.

    Returns:
        torch.Tensor of shape (1, H, W), dtype float32.
    """
    log_mag = preprocess_image(image_path, img_size)
    # Add channel dimension: (H, W) → (1, H, W)
    tensor = torch.from_numpy(log_mag).unsqueeze(0)
    return tensor


# ============================================================
#  PYTORCH DATASET
# ============================================================

class AIGeneratedImageDataset(Dataset):
    """PyTorch Dataset for AI-generated image detection.

    Each sample is a log-magnitude spectrum tensor (1 × H × W) and a
    binary label (0 = real, 1 = fake).

    Parameters
    ----------
    samples : list of dict
        Each dict has keys: "path" (str), "label" (int), "source" (str).
        This is the format returned by dataset_split.make_split()["train"]
        or ["test"].
    img_size : int
        Resize images to this square size before computing DFT.
    """

    def __init__(self,
                 samples: List[Dict],
                 img_size: int = IMG_SIZE):
        self.samples  = samples
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["path"]
        label      = sample["label"]

        # Preprocess: grayscale → DFT → shift → magnitude → log → normalize
        tensor = preprocess_to_tensor(image_path, self.img_size)

        return tensor, label

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

    # Create dataset
    train_ds = AIGeneratedImageDataset(split["train"])
    test_ds  = AIGeneratedImageDataset(split["test"])

    print(f"Train dataset size : {len(train_ds)}")
    print(f"Test  dataset size : {len(test_ds)}")

    # Test loading one sample
    tensor, label = train_ds[0]
    print(f"Sample 0 – tensor shape: {tensor.shape}, dtype: {tensor.dtype}, "
          f"min: {tensor.min():.4f}, max: {tensor.max():.4f}, label: {label}")
    info = train_ds.get_sample_info(0)
    print(f"         source: {info['source']}, path: {info['path'][:60]}...")
