"""
scripts/download_im2latex.py

Downloads and preprocesses the IM2LATEX-100K dataset into the format
expected by MathDataset(mode="im2latex").

Output structure:
    data/im2latex_test/
    ├── images/          ← the raw PNG images from the dataset
    └── labels.txt       ← tab-separated: image_name.png \t latex_string

Usage:
    uv run python scripts/download_im2latex.py

After running this script, evaluate against the benchmark with:
    uv run python scripts/evaluate_model.py --mode im2latex
"""

import sys
from pathlib import Path
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from datasets import load_dataset

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

OUTPUT_DIR      = project_root / "data" / "im2latex_test"
IMAGES_DIR      = OUTPUT_DIR / "images"
LABELS_FILE     = OUTPUT_DIR / "labels.txt"

def main():
    print("=== IM2LATEX-100K Download & Preprocessing ===")
    print("Fetching benchmark from Hugging Face Datasets...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load the official test split (it will download automatically and cache it)
    dataset = load_dataset("yuntian-deng/im2latex-100k", split="test")
    
    written = 0
    skipped = 0

    print(f"\nExtracting {len(dataset)} images and generating labels...")
    
    with open(LABELS_FILE, "w", encoding="utf-8") as out:
        for idx, item in enumerate(tqdm(dataset, desc="Processing Benchmark")):
            image = item.get("image")
            formula = item.get("formula")
            
            if not image or not formula:
                skipped += 1
                continue
                
            img_filename = f"test_{idx}.png"
            img_path = IMAGES_DIR / img_filename
            
            # Save the PIL Image to disk
            image.save(img_path)
            
            # Write the formatted label
            out.write(f"{img_filename}\t{formula}\n")
            written += 1

    print(f"\n  Written : {written} entries to {LABELS_FILE}")
    if skipped:
        print(f"  Skipped : {skipped} invalid entries")

    print(f"\n=== Done! Dataset ready at {OUTPUT_DIR} ===")

if __name__ == "__main__":
    main()
