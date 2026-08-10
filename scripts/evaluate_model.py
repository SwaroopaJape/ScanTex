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
    
    def create_buckets():
        return {
            "1-10 tokens": {"count": 0, "exact_matches": 0, "total_cer": 0, "total_chars": 0},
            "11-25 tokens": {"count": 0, "exact_matches": 0, "total_cer": 0, "total_chars": 0},
            "26-50 tokens": {"count": 0, "exact_matches": 0, "total_cer": 0, "total_chars": 0},
            "50+ tokens": {"count": 0, "exact_matches": 0, "total_cer": 0, "total_chars": 0},
        }

    metrics = {
        "greedy": {"exact_matches": 0, "total_cer": 0.0, "total_chars": 0, "compilation_success": 0, "buckets": create_buckets()},
        "beam":   {"exact_matches": 0, "total_cer": 0.0, "total_chars": 0, "compilation_success": 0, "buckets": create_buckets()},
    }
    
    processed = 0
    
    import random
    random.seed(42)
    random.shuffle(lines)
    
    pbar = tqdm(lines)
    for latex in pbar:
        img_tensor = render_to_tensor(latex, transform)
        if img_tensor is None:
            # Skip invalid latex that can't be rendered
            continue
            
        gt_clean = latex.replace(" ", "")
        token_len = len(tokenizer.encode(latex))
        
        if token_len <= 10:
            b_key = "1-10 tokens"
        elif token_len <= 25:
            b_key = "11-25 tokens"
        elif token_len <= 50:
            b_key = "26-50 tokens"
        else:
            b_key = "50+ tokens"

        for algo in ["greedy", "beam"]:
            pred_latex = generate(img_tensor, encoder, decoder, tokenizer, device, max_len=150, algorithm=algo)
            pred_clean = pred_latex.replace(" ", "")
            
            is_em = (gt_clean == pred_clean)
            if is_em:
                metrics[algo]["exact_matches"] += 1
                
            dist = levenshtein_distance(gt_clean, pred_clean)
            metrics[algo]["total_cer"] += dist
            metrics[algo]["total_chars"] += len(gt_clean)
            
            if check_compiles(pred_latex):
                metrics[algo]["compilation_success"] += 1
                
            metrics[algo]["buckets"][b_key]["count"] += 1
            if is_em:
                metrics[algo]["buckets"][b_key]["exact_matches"] += 1
            metrics[algo]["buckets"][b_key]["total_cer"] += dist
            metrics[algo]["buckets"][b_key]["total_chars"] += len(gt_clean)
            
        processed += 1
        
        # update pbar with greedy as baseline display
        em_rate = (metrics["greedy"]["exact_matches"] / processed) * 100
        cer = (metrics["greedy"]["total_cer"] / metrics["greedy"]["total_chars"]) * 100 if metrics["greedy"]["total_chars"] > 0 else 0
        pbar.set_postfix({"G-EM": f"{em_rate:.1f}%", "G-CER": f"{cer:.1f}%"})

    if processed == 0:
        print("No valid equations processed.")
        sys.exit(0)
        
    print(f"Total Samples Evaluated : {processed}\n")
    
    print(f"{'Metric':<25} | {'Greedy':<15} | {'Beam Search':<15}")
    print("-" * 65)
    
    for algo in ["greedy", "beam"]:
        metrics[algo]["em_rate"] = (metrics[algo]["exact_matches"] / processed) * 100
        metrics[algo]["cer_rate"] = (metrics[algo]["total_cer"] / max(1, metrics[algo]["total_chars"])) * 100
        metrics[algo]["comp_rate"] = (metrics[algo]["compilation_success"] / processed) * 100
        
    print(f"{'Exact Match (EM) Rate':<25} | {metrics['greedy']['em_rate']:>6.2f}%         | {metrics['beam']['em_rate']:>6.2f}%")
    print(f"{'Character Error Rate':<25} | {metrics['greedy']['cer_rate']:>6.2f}%         | {metrics['beam']['cer_rate']:>6.2f}%")
    print(f"{'Prediction Compile Rate':<25} | {metrics['greedy']['comp_rate']:>6.2f}%         | {metrics['beam']['comp_rate']:>6.2f}%")
    
    print("\n" + "="*65)
    print("PER-COMPLEXITY BUCKETS (Token Length)")
    print("="*65)
    
    print(f"{'Token Length':<15} | {'Count':<6} | {'G-EM':<6} | {'B-EM':<6} | {'G-CER':<6} | {'B-CER':<6}")
    print("-" * 65)
    
    for b_key in ["1-10 tokens", "11-25 tokens", "26-50 tokens", "50+ tokens"]:
        count = metrics["greedy"]["buckets"][b_key]["count"]
        if count == 0:
            print(f"{b_key:<15} | {count:<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6}")
        else:
            g_em = (metrics["greedy"]["buckets"][b_key]["exact_matches"] / count) * 100
            b_em = (metrics["beam"]["buckets"][b_key]["exact_matches"] / count) * 100
            
            g_chars = metrics["greedy"]["buckets"][b_key]["total_chars"]
            b_chars = metrics["beam"]["buckets"][b_key]["total_chars"]
            
            g_cer = (metrics["greedy"]["buckets"][b_key]["total_cer"] / g_chars) * 100 if g_chars > 0 else 0
            b_cer = (metrics["beam"]["buckets"][b_key]["total_cer"] / b_chars) * 100 if b_chars > 0 else 0
            
            print(f"{b_key:<15} | {count:<6} | {g_em:>5.1f}% | {b_em:>5.1f}% | {g_cer:>5.1f}% | {b_cer:>5.1f}%")
            
    print("="*65)

if __name__ == "__main__":
    main()
