import sys
import argparse
from pathlib import Path

# allow absolute imports from project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# pyrefly: ignore [missing-import]
from src.data.dataset import MathDataset
# pyrefly: ignore [missing-import]
from src.data.tokenizer import HybridTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train the Hybrid BPE Tokenizer")
    parser.add_argument("--mode", type=str, default="scraped", choices=["toy", "scraped"], 
                        help="Which dataset to train the tokenizer on.")
    args = parser.parse_args()

    print(f"=== Tokenizer Training Pipeline (Mode: {args.mode.upper()}) ===")
    
    if args.mode == "toy":
        dataset = MathDataset(mode="toy")
    else:
        dataset = MathDataset(mode="real", data_dir="data/scraped_train")
    
    if not dataset.latex_strings:
        print("Error: No latex strings found in the dataset. Did you render the corpus?")
        sys.exit(1)
        
    print(f"Loaded {len(dataset.latex_strings)} equations for tokenizer training.")
    
    tokenizer = HybridTokenizer(vocab_size=4000)
    print("Training BPE Tokenizer... (This may take a minute or two)")
    tokenizer.train(dataset.latex_strings)
    
    save_dir = project_root / "data" / "tokenizer_weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_dir))
    
    print(f"Success! Tokenizer trained and saved to {save_dir}")
    print(f"Total Vocabulary Size: {tokenizer.total_vocab_size}")

if __name__ == "__main__":
    main()
