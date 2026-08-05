import sys
import os
from pathlib import Path
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
from torchvision.io import read_image
from torchvision.transforms import v2
from tqdm import tqdm

matplotlib.use("Agg")

# allow absolute imports from project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# pyrefly: ignore [missing-import]
from src.data.tokenizer import HybridTokenizer
# pyrefly: ignore [missing-import]
from src.models.encoder import VisionEncoder
# pyrefly: ignore [missing-import]
from src.models.decoder import LatexDecoder
# pyrefly: ignore [missing-import]
from src.inference import generate

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def check_compiles(latex: str) -> bool:
    try:
        fig = plt.figure(figsize=(6, 1))
        fig.text(0.5, 0.5, f"${latex}$", fontsize=14, ha="center", va="center")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.close()
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False

def render_to_tensor(latex: str, transform):
    try:
        fig = plt.figure(figsize=(6, 1))
        fig.text(0.5, 0.5, f"${latex}$", fontsize=14, ha="center", va="center")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        plt.close(fig)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(buf.read())
            temp_path = f.name
            
        img_tensor = read_image(temp_path)[:3, :, :]
        os.remove(temp_path)
        
        return transform(img_tensor)
    except Exception:
        plt.close("all")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on validation data.")
    parser.add_argument("--val-file", type=str, default="data/scraped/val_equations.txt")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")
    
    tokenizer_path = str(project_root / "data" / "tokenizer_weights")
    tokenizer = HybridTokenizer(vocab_size=4000)
    try:
        tokenizer.load(tokenizer_path)
    except Exception:
        print(f"ERROR: Could not load tokenizer from {tokenizer_path}.")
        sys.exit(1)
        
    checkpoint_path = project_root / "checkpoint.pt"
    if not checkpoint_path.exists():
        print("ERROR: checkpoint.pt not found! Train the model first.")
        sys.exit(1)
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    encoder = VisionEncoder().to(device)
    decoder = LatexDecoder(vocab_size=checkpoint['vocab_size']).to(device)
    
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((128, 512), antialias=True),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    val_file = Path(args.val_file)
    if not val_file.exists():
        print(f"ERROR: File {val_file} not found.")
        sys.exit(1)
        
    with open(val_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(lines)} equations in {val_file}.")
    
    exact_matches = 0
    total_cer = 0.0
    total_chars = 0
    compilation_success = 0
    processed = 0
    
    # Shuffle so we get a random subset of max_samples
    import random
    random.seed(42)
    random.shuffle(lines)
    
    pbar = tqdm(lines)
    for latex in pbar:
        img_tensor = render_to_tensor(latex, transform)
        if img_tensor is None:
            # Skip invalid latex that can't be rendered
            continue
            
        pred_latex = generate(img_tensor, encoder, decoder, tokenizer, device, max_len=150)
        
        # 1. Exact Match
        gt_clean = latex.replace(" ", "")
        pred_clean = pred_latex.replace(" ", "")
        is_em = (gt_clean == pred_clean)
        if is_em:
            exact_matches += 1
            
        # 2. CER
        dist = levenshtein_distance(gt_clean, pred_clean)
        total_cer += dist
        total_chars += len(gt_clean)
        
        # 3. Compilation check
        if check_compiles(pred_latex):
            compilation_success += 1
            
        processed += 1
        
        em_rate = (exact_matches / processed) * 100
        cer = (total_cer / total_chars) * 100 if total_chars > 0 else 0
        comp_rate = (compilation_success / processed) * 100
        
        pbar.set_postfix({"EM": f"{em_rate:.1f}%", "CER": f"{cer:.1f}%", "Comp": f"{comp_rate:.1f}%"})

    if processed == 0:
        print("No valid equations processed.")
        sys.exit(0)
        
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples Evaluated : {processed}")
    print(f"Exact Match (EM) Rate   : {(exact_matches / processed) * 100:.2f}%")
    print(f"Character Error Rate    : {(total_cer / max(1, total_chars)) * 100:.2f}%")
    print(f"Prediction Compile Rate : {(compilation_success / processed) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
