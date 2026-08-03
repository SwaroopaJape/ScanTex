import sys
from pathlib import Path

# allow absolute imports from project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# pyrefly: ignore [missing-import]
from src.data.dataset import MathDataset
# pyrefly: ignore [missing-import]
from src.data.tokenizer import HybridTokenizer

def main():
    print("=== Tokenizer Training Pipeline ===")
    
    # Load the real dataset to get all 200,000 strings
    dataset = MathDataset(mode="real")
    
    if not dataset.latex_strings:
        print("Error: No latex strings found in the dataset. Did you render the synthetic corpus?")
        sys.exit(1)
        
    print(f"Loaded {len(dataset.latex_strings)} equations from the real dataset.")
    
    tokenizer = HybridTokenizer(vocab_size=4000)
    print("Training BPE Tokenizer... (This may take a minute or two)")
    tokenizer.train(dataset.latex_strings)
    
    save_dir = project_root / "data" / "tokenizer_weights"
    tokenizer.save(str(save_dir))
    
    print(f"Success! Tokenizer trained and saved to {save_dir}")
    print(f"Total Vocabulary Size: {tokenizer.total_vocab_size}")

if __name__ == "__main__":
    main()
