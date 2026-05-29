"""
Simulate GAN-like artifacts on real images (AutoGAN method).

Based on: "Detecting and Simulating Artifacts in GAN Fake Images"
          Zhang et al., 2019  (https://arxiv.org/abs/1907.06515)

Core idea:
  GANs commonly upsample feature maps to produce full-resolution outputs.
  This downsample→upsample round-trip introduces spectral artifacts
  (replications in the frequency domain) that are characteristic of GAN output.
  We reproduce this pipeline on real images to synthesise "fake-like" images
  without needing a real GAN.

Supported simulation modes
--------------------------
  nearest      : downsample with area averaging, upsample with nearest-neighbour
  bilinear     : downsample with area averaging, upsample with bilinear interpolation
  bicubic      : downsample with area averaging, upsample with bicubic interpolation
  transposed   : simulate transposed-convolution checkerboard artifact by
                 zero-inserting (stride-2 expansion) then Gaussian-blurring
  jpeg         : apply low-quality JPEG compression (blocking artifact)

Usage
-----
  # As a script: simulate every Nature image and save to dataset/Nature_fake/
  python simulate_fake.py --mode nearest --scale 2 --output_dir dataset/Nature_fake

  # As a module:
  from simulate_fake import simulate_image
  pil_fake = simulate_image(pil_img, mode='nearest', scale=2)
"""

from __future__ import annotations

import argparse
import io
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Simulation modes
# ---------------------------------------------------------------------------

SimMode = Literal["nearest", "bilinear", "bicubic", "transposed", "jpeg"]


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL (H,W,C) uint8  →  torch (1, C, H, W) float32 in [0,1]."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _to_pil(t: torch.Tensor) -> Image.Image:
    """torch (1, C, H, W) float32 in [0,1]  →  PIL RGB."""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _downsample_upsample(img: Image.Image,
                         scale: int,
                         up_mode: str) -> Image.Image:
    """
    Downsample by `scale` using area averaging, then upsample back to the
    original size using the specified interpolation mode.

    This round-trip introduces spectral replications at multiples of the
    Nyquist frequency, mimicking the artifacts visible in GAN-generated images
    when the network applies strided transposed convolutions or nearest-neighbour
    upsampling internally.
    """
    W, H = img.size
    small_W, small_H = max(1, W // scale), max(1, H // scale)

    # Downsample with area resampling (avoids aliasing on the way down)
    small = img.resize((small_W, small_H), Image.BOX)

    # Upsample back to original resolution
    pil_filter = {
        "nearest":  Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic":  Image.BICUBIC,
    }[up_mode]
    return small.resize((W, H), pil_filter)


def _transposed_conv_artifact(img: Image.Image, scale: int = 2) -> Image.Image:
    """
    Simulate the checkerboard artifact produced by strided transposed convolutions.

    Mechanism:
      1. Downsample the image by `scale` (shrink to 1/scale resolution).
      2. Zero-insert (pixel shuffle / sub-pixel expansion) to restore original size
         — this is exactly what a stride-`scale` transposed convolution does when
         weights are uniform.
      3. Apply a Gaussian blur to mimic the learned smoothing kernel.

    The zero-insertion creates a periodic grid of zeros that appears as
    replicated spectra in the Fourier domain.
    """
    W, H = img.size
    small_W, small_H = max(1, W // scale), max(1, H // scale)

    # Step 1: downsample
    small = img.resize((small_W, small_H), Image.BOX)
    t_small = _to_tensor(small)  # (1, 3, H/s, W/s)

    # Step 2: zero-insert (stride-scale expansion)
    # Allocate a zero tensor of original size and fill every `scale`-th pixel
    t_expanded = torch.zeros(1, 3, H, W)
    sh, sw = t_small.shape[-2], t_small.shape[-1]
    t_expanded[:, :, :sh * scale:scale, :sw * scale:scale] = t_small

    # Step 3: smooth with Gaussian kernel (mimics the conv kernel)
    sigma = 0.5 * scale
    r = max(1, int(3 * sigma))   # half-width; ks = 2r+1 is always odd
    ks = 2 * r + 1
    ax = torch.arange(ks, dtype=torch.float32) - r
    gauss1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    gauss1d /= gauss1d.sum()
    kernel_2d = torch.outer(gauss1d, gauss1d).unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.repeat(3, 1, 1, 1)  # depthwise

    blurred = F.conv2d(t_expanded, kernel_2d, padding=r, groups=3)

    return _to_pil(blurred)


def _jpeg_artifact(img: Image.Image, quality: int = 50) -> Image.Image:
    """
    Simulate JPEG blocking artifacts by encoding and decoding the image.

    JPEG compression introduces 8×8 block-level quantisation artifacts that
    are distinct from GAN artifacts but useful as an additional simulation mode
    for training data diversity.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_image(img: Image.Image,
                   mode: SimMode = "nearest",
                   scale: int = 2,
                   jpeg_quality: int = 50) -> Image.Image:
    """
    Apply AutoGAN-style artifact simulation to a single PIL image.

    Parameters
    ----------
    img          : Input PIL image (any mode; converted to RGB internally).
    mode         : Simulation mode — one of nearest, bilinear, bicubic,
                   transposed, jpeg.
    scale        : Down/up-sampling scale factor (ignored for jpeg mode).
                   Typical values: 2 (mimic 1-step GAN upsample),
                                   4 (mimic 2-step GAN upsample).
    jpeg_quality : JPEG quality level when mode='jpeg' (1–95, lower = more artifacts).

    Returns
    -------
    PIL.Image.Image  :  Simulated "fake" image (same spatial size as input).
    """
    img = img.convert("RGB")

    if mode in ("nearest", "bilinear", "bicubic"):
        return _downsample_upsample(img, scale, up_mode=mode)
    elif mode == "transposed":
        return _transposed_conv_artifact(img, scale)
    elif mode == "jpeg":
        return _jpeg_artifact(img, quality=jpeg_quality)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. "
                         f"Choose from nearest, bilinear, bicubic, transposed, jpeg.")


# ---------------------------------------------------------------------------
# Batch processing script
# ---------------------------------------------------------------------------

def _process_directory(input_dir: Path,
                       output_dir: Path,
                       mode: SimMode,
                       scale: int,
                       jpeg_quality: int,
                       limit: int | None) -> None:
    """Simulate fake images for every image found under `input_dir`."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    paths = [p for p in sorted(input_dir.rglob("*")) if p.suffix.lower() in exts]

    if limit is not None:
        paths = paths[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Simulating {len(paths)} images  mode={mode}  scale={scale}")

    for i, src in enumerate(paths):
        # Preserve sub-directory structure relative to input_dir
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src)
        fake = simulate_image(img, mode=mode, scale=scale, jpeg_quality=jpeg_quality)
        fake.save(dst)

        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"  {i+1}/{len(paths)}  {rel}")

    print(f"Done. Saved to {output_dir}")


def _visualise(input_path: Path, mode: SimMode, scale: int,
               jpeg_quality: int) -> None:
    """Show original vs simulated side-by-side and print FFT comparison."""
    import matplotlib.pyplot as plt

    img = Image.open(input_path).convert("RGB")
    fake = simulate_image(img, mode=mode, scale=scale, jpeg_quality=jpeg_quality)

    def log_fft(pil_img: Image.Image) -> np.ndarray:
        gray = np.array(pil_img.convert("L"), dtype=np.float32)
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        return np.log1p(np.abs(spectrum))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original (real)")
    axes[0, 1].imshow(fake)
    axes[0, 1].set_title(f"Simulated fake  [{mode}, scale={scale}]")
    axes[1, 0].imshow(log_fft(img), cmap="inferno")
    axes[1, 0].set_title("FFT – Original")
    axes[1, 1].imshow(log_fft(fake), cmap="inferno")
    axes[1, 1].set_title("FFT – Simulated")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoGAN-style GAN artifact simulator")
    parser.add_argument("--input_dir", type=str,
                        default="dataset/Nature",
                        help="Directory of real images to simulate")
    parser.add_argument("--output_dir", type=str,
                        default="dataset/Nature_fake",
                        help="Where to save simulated fake images")
    parser.add_argument("--mode", type=str, default="nearest",
                        choices=["nearest", "bilinear", "bicubic",
                                 "transposed", "jpeg"],
                        help="Artifact simulation mode")
    parser.add_argument("--scale", type=int, default=2,
                        help="Down/up-sample scale factor (for non-jpeg modes)")
    parser.add_argument("--jpeg_quality", type=int, default=50,
                        help="JPEG quality level (1-95) for jpeg mode")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N images (for testing)")
    parser.add_argument("--visualise", type=str, default=None,
                        metavar="IMAGE_PATH",
                        help="Visualise a single image instead of batch processing")
    args = parser.parse_args()

    if args.visualise:
        _visualise(Path(args.visualise), args.mode, args.scale, args.jpeg_quality)
    else:
        _process_directory(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            mode=args.mode,
            scale=args.scale,
            jpeg_quality=args.jpeg_quality,
            limit=args.limit,
        )
