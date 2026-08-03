import os
import random
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure the project root is in sys.path so we can import src
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# pyrefly: ignore [missing-import]
from src.data.string_generator import generate, load_grammar

# Use Agg backend for headless multiprocessing
matplotlib.use("Agg")

def worker_init(grammar_file):
    """
    Initialize each worker process with its own random seed and load the grammar.
    """
    # Seed with OS randomness so processes don't generate identical strings
    random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    load_grammar(grammar_file)

def generate_and_save(args):
    idx, output_dir, max_attempts = args
    img_filename = f"sample_{idx:06d}.png"
    img_path = os.path.join(output_dir, img_filename)
    
    for _ in range(max_attempts):
        latex = generate()
        
        try:
            # We attempt to render it. Matplotlib acts as our compiler check.
            fig = plt.figure(figsize=(6, 1))
            # Wrap in math mode for rendering
            fig.text(0.5, 0.5, f"${latex}$", fontsize=14, ha="center", va="center")
            fig.savefig(img_path, format="png", bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            return (img_filename, latex)
        except Exception:
            # If it fails to compile/render (e.g. invalid syntax), clean up and try again
            plt.close("all")
            continue
            
    # Failed to generate a valid equation after max_attempts
    return None

def main():
    parser = argparse.ArgumentParser(description="Render Synthetic Corpus using Matplotlib")
    parser.add_argument("--n", type=int, default=200000, help="Number of equations to generate")
    parser.add_argument("--grammar", type=str, default="data/extracted_grammar/weighted_grammar.py", help="Path to weighted grammar Python file")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_train/images", help="Output directory for images")
    parser.add_argument("--labels-file", type=str, default="data/synthetic_train/labels.txt", help="Output path for labels.txt")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of CPU cores to use")
    parser.add_argument("--max-attempts", type=int, default=50, help="Retries per equation if rendering fails")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_file), exist_ok=True)

    print("=" * 60)
    print(f"ScanTex Synthetic Data Renderer")
    print(f"Target: {args.n} images")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Prepare arguments for the multiprocessing pool
    tasks = [(i, args.output_dir, args.max_attempts) for i in range(args.n)]
    
    success_count = 0
    # Open labels file in append mode or write mode? Write mode to start fresh.
    with open(args.labels_file, "w", encoding="utf-8") as f_labels:
        # imap_unordered is faster than map because it yields as soon as a task is done
        with multiprocessing.Pool(args.workers, initializer=worker_init, initargs=(args.grammar,)) as pool:
            for result in tqdm(pool.imap_unordered(generate_and_save, tasks), total=args.n, desc="Rendering"):
                if result is not None:
                    img_filename, latex = result
                    f_labels.write(f"{img_filename}\t{latex}\n")
                    success_count += 1

    print("\n" + "=" * 60)
    if success_count == args.n:
        print(f"Successfully rendered all {success_count} images.")
    else:
        print(f"Finished. Rendered {success_count}/{args.n} images (some failed max attempts).")
    print(f"Labels saved to: {args.labels_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
