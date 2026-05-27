"""
Dataset splitting for AI-Generated Image Detection.

Strategy:
  - Test set:  images from ONE chosen generator (configurable via TEST_GENERATOR)
               + Nature images from the "val.X" split in nature_metadata.csv
  - Train set: images from all OTHER generators
               + Nature images from the "train.X*" splits in nature_metadata.csv

Labels: 0 = real (Nature), 1 = fake (AI-generated)

Change TEST_GENERATOR below to switch which generator is used for the test set.
"""

import os
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict

# ============================================================
#  CONFIGURABLE PARAMETER – change this to switch test generator
# ============================================================
TEST_GENERATOR = "ADM"   # Options: ADM, BigGAN, Midjourney, VQDM, glide, stable_diffusion_v_1_5, wukong

# ============================================================
#  PATHS
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent / "dataset"
NATURE_DIR = BASE_DIR / "Nature"
META_CSV   = BASE_DIR / "nature_metadata.csv"

# All generator (fake) directories
GENERATOR_DIRS = sorted([
    d.name for d in BASE_DIR.iterdir()
    if d.is_dir() and d.name != "Nature"
])

# ============================================================
#  HELPERS
# ============================================================

def parse_nature_metadata(csv_path: Path) -> Dict[str, str]:
    """Parse nature_metadata.csv and return {filename: split_tag}.

    The 'class' column looks like 'val.X\\n01440764' or 'train.X1\\n01440764'.
    We only care about the part before the backslash-n: 'val.X', 'train.X1', etc.

    Some rows have a numeric class value (e.g. '357') instead of a proper split
    tag – these are treated as 'train.unknown' so they default to training.
    """
    mapping = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0].strip()
            class_field = row[1].strip()
            # Extract split tag: everything before '\\n' or the whole field
            split_tag = class_field.split("\\n")[0].split("\n")[0].strip()
            # If the split tag doesn't look like a proper tag (val.X / train.X*),
            # mark it as unknown – it will default to the training set.
            if not (split_tag.startswith("val.") or split_tag.startswith("train.")):
                split_tag = "train.unknown"
            mapping[filename] = split_tag
    return mapping


def collect_nature_images(meta: Dict[str, str]):
    """Split Nature images into train / test based on metadata.

    Returns (train_paths, test_paths) – lists of absolute file paths.
    """
    train_paths: List[str] = []
    test_paths:  List[str] = []

    for fname, split_tag in meta.items():
        fpath = str(NATURE_DIR / fname)
        if not os.path.isfile(fpath):
            # metadata may reference images not present in the folder
            continue
        if split_tag.startswith("val."):
            test_paths.append(fpath)
        elif split_tag.startswith("train."):
            # Includes 'train.X1'–'train.X4' and 'train.unknown'
            train_paths.append(fpath)
        else:
            # Shouldn't happen after parse_nature_metadata fix, but just in case
            print(f"[WARN] Unexpected split tag '{split_tag}' for {fname}, assigning to train.")
            train_paths.append(fpath)

    return train_paths, test_paths


def collect_fake_images(test_generator: str):
    """Collect fake images, putting test_generator's images into test set
    and all other generators' images into train set.

    Returns (train_paths, test_paths) – lists of (path, generator_name).
    """
    train_paths: List[Tuple[str, str]] = []
    test_paths:  List[Tuple[str, str]] = []

    for gen_name in GENERATOR_DIRS:
        gen_dir = BASE_DIR / gen_name
        if not gen_dir.is_dir():
            continue
        image_files = sorted([
            str(gen_dir / f) for f in os.listdir(gen_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        if gen_name == test_generator:
            test_paths.extend([(p, gen_name) for p in image_files])
        else:
            train_paths.extend([(p, gen_name) for p in image_files])

    return train_paths, test_paths


# ============================================================
#  MAIN SPLIT FUNCTION
# ============================================================

def make_split(test_generator: str = TEST_GENERATOR,
               seed: int = 42,
               verbose: bool = True) -> Dict[str, List[Dict]]:
    """Build train / test splits.

    Returns a dict:
        {
            "train": [{"path": ..., "label": 0|1, "source": ...}, ...],
            "test":  [{"path": ..., "label": 0|1, "source": ...}, ...],
        }

    label: 0 = real (Nature), 1 = fake (AI-generated)
    source: "Nature" or the generator name
    """
    assert test_generator in GENERATOR_DIRS, (
        f"Unknown generator '{test_generator}'. "
        f"Available: {GENERATOR_DIRS}"
    )

    # --- Nature (real) images ---
    meta = parse_nature_metadata(META_CSV)
    nature_train, nature_test = collect_nature_images(meta)

    # --- Fake (AI-generated) images ---
    fake_train, fake_test = collect_fake_images(test_generator)

    # --- Assemble ---
    train_samples = []
    test_samples  = []

    for p in nature_train:
        train_samples.append({"path": p, "label": 0, "source": "Nature"})
    for p in nature_test:
        test_samples.append({"path": p, "label": 0, "source": "Nature"})

    for p, gen in fake_train:
        train_samples.append({"path": p, "label": 1, "source": gen})
    for p, gen in fake_test:
        test_samples.append({"path": p, "label": 1, "source": gen})

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    if verbose:
        n_train_real = sum(1 for s in train_samples if s["label"] == 0)
        n_train_fake = sum(1 for s in train_samples if s["label"] == 1)
        n_test_real  = sum(1 for s in test_samples  if s["label"] == 0)
        n_test_fake  = sum(1 for s in test_samples  if s["label"] == 1)

        print("=" * 60)
        print(f"  Test generator : {test_generator}")
        print(f"  Train - real   : {n_train_real}")
        print(f"  Train - fake   : {n_train_fake}  "
              f"(from: {', '.join(g for g in GENERATOR_DIRS if g != test_generator)})")
        print(f"  Train - total  : {len(train_samples)}")
        print(f"  Test  - real   : {n_test_real}")
        print(f"  Test  - fake   : {n_test_fake}  (from: {test_generator})")
        print(f"  Test  - total  : {len(test_samples)}")
        print("=" * 60)

    return {"train": train_samples, "test": test_samples}


# ============================================================
#  QUICK TEST
# ============================================================

if __name__ == "__main__":
    split = make_split()
