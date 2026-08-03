import os
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path

# Use Agg backend for headless multiprocessing
matplotlib.use("Agg")

def generate_and_save(args):
    idx, latex, output_dir = args
    img_filename = f"scraped_{idx:06d}.png"
    img_path = os.path.join(output_dir, img_filename)
    
    try:
        # We attempt to render it. Matplotlib acts as our compiler check.
        fig = plt.figure(figsize=(6, 1))
        # Wrap in math mode for rendering
        fig.text(0.5, 0.5, f"${latex}$", fontsize=14, ha="center", va="center")
        fig.savefig(img_path, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return (img_filename, latex)
    except Exception:
        # If it fails to compile/render (e.g. invalid syntax or unsupported macro)
        plt.close("all")
        return None

def main():
    parser = argparse.ArgumentParser(description="Render Scraped Corpus using Matplotlib")
    parser.add_argument("--input-file", type=str, default="data/scraped/train_equations.txt", help="Path to scraped equations txt file")
    parser.add_argument("--output-dir", type=str, default="data/scraped_train/images", help="Output directory for images")
    parser.add_argument("--labels-file", type=str, default="data/scraped_train/labels.txt", help="Output path for labels.txt")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of CPU cores to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_file), exist_ok=True)

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found. Did you run the scraping scripts?")
        sys.exit(1)

    with open(args.input_file, "r", encoding="utf-8") as f:
        equations = [line.strip() for line in f if line.strip()]

    print("=" * 60)
    print(f"ScanTex Scraped Data Renderer")
    print(f"Target: {len(equations)} equations")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Prepare arguments for the multiprocessing pool
    tasks = [(i, eq, args.output_dir) for i, eq in enumerate(equations)]
    
    success_count = 0
    # Open labels file in write mode to start fresh
    with open(args.labels_file, "w", encoding="utf-8") as f_labels:
        # imap_unordered is faster than map because it yields as soon as a task is done
        with multiprocessing.Pool(args.workers) as pool:
            for result in tqdm(pool.imap_unordered(generate_and_save, tasks), total=len(tasks), desc="Rendering"):
                if result is not None:
                    img_filename, latex = result
                    f_labels.write(f"{img_filename}\t{latex}\n")
                    success_count += 1

    print("\n" + "=" * 60)
    if success_count == len(equations):
        print(f"Successfully rendered all {success_count} images.")
    else:
        print(f"Finished. Rendered {success_count}/{len(equations)} images.")
        print(f"Note: {len(equations) - success_count} equations were skipped due to unsupported LaTeX macros in Matplotlib.")
    print(f"Labels saved to: {args.labels_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
