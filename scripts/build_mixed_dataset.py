"""
build_mixed_dataset.py
======================
Builds a 20k blended fine-tuning dataset from two sources:
  - 10,000 random images from the IM2LATEX-100K training split (HuggingFace)
  - 10,000 random images from the existing scraped Wikipedia training data

Output layout:
  data/mixed_train/
    images/          <- all 20k images (prefixed to avoid collisions)
    labels.txt       <- tab-separated: image_filename \\t latex_string

Usage:
    uv run python scripts/build_mixed_dataset.py
"""

import sys
import shutil
import random
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# ── Config ──────────────────────────────────────────────────────────────────
WIKI_SRC_DIR   = project_root / "data" / "scraped_train"
OUTPUT_DIR     = project_root / "data" / "mixed_train"
IMAGES_DIR     = OUTPUT_DIR / "images"
LABELS_FILE    = OUTPUT_DIR / "labels.txt"

N_IM2LATEX     = 10_000
N_WIKI         = 10_000
SEED           = 42
# ────────────────────────────────────────────────────────────────────────────


def load_wiki_pairs(n: int) -> list[tuple[Path, str]]:
    """Load N random (image_path, latex) pairs from the scraped Wikipedia data."""
    labels_file = WIKI_SRC_DIR / "labels.txt"
    images_dir  = WIKI_SRC_DIR / "images"

    if not labels_file.exists():
        print(f"ERROR: {labels_file} not found. Run the scraping pipeline first.")
        sys.exit(1)

    pairs = []
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                img_path = images_dir / parts[0]
                if img_path.exists():
                    pairs.append((img_path, parts[1]))

    random.seed(SEED)
    random.shuffle(pairs)
    sampled = pairs[:n]
    print(f"  Sampled {len(sampled):,} Wikipedia equations (from {len(pairs):,} available)")
    return sampled


def load_im2latex_pairs(n: int) -> list[tuple, str]:
    """Download and sample N items from the IM2LATEX-100K train split."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Run: uv pip install datasets pillow")
        sys.exit(1)

    print(f"  Fetching IM2LATEX-100K train split from HuggingFace...")
    dataset = load_dataset("yuntian-deng/im2latex-100k", split="train")

    indices = list(range(len(dataset)))
    random.seed(SEED)
    random.shuffle(indices)
    sampled_indices = indices[:n]

    print(f"  Sampled {len(sampled_indices):,} IM2LATEX equations (from {len(dataset):,} available)")
    return [(dataset[i]["image"], dataset[i]["formula"]) for i in sampled_indices]


def main():
    print("=== Building 20k Mixed Fine-Tuning Dataset ===\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    written = 0

    with open(LABELS_FILE, "w", encoding="utf-8") as out:

        # ── Part 1: Wikipedia scraped data ──────────────────────────────────
        print("[1/2] Loading Wikipedia scraped data...")
        wiki_pairs = load_wiki_pairs(N_WIKI)

        for src_path, formula in tqdm(wiki_pairs, desc="  Copying Wiki images"):
            dest_name = f"wiki_{src_path.name}"
            dest_path = IMAGES_DIR / dest_name
            shutil.copy2(src_path, dest_path)
            out.write(f"{dest_name}\t{formula}\n")
            written += 1

        # ── Part 2: IM2LATEX-100K training split ────────────────────────────
        print("\n[2/2] Downloading IM2LATEX-100K training samples...")
        im2latex_pairs = load_im2latex_pairs(N_IM2LATEX)

        for idx, (image, formula) in enumerate(tqdm(im2latex_pairs, desc="  Saving IM2LATEX images")):
            dest_name = f"im2l_{idx:05d}.png"
            dest_path = IMAGES_DIR / dest_name
            image.save(dest_path)
            out.write(f"{dest_name}\t{formula}\n")
            written += 1

    print(f"\n  Total entries written : {written:,}")
    print(f"  Output directory      : {OUTPUT_DIR}")
    print(f"\n=== Done! Run training with: mode=mixed ===")


if __name__ == "__main__":
    main()
