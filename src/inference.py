import sys
from pathlib import Path
import torch
import os
import argparse
from torchvision.io import read_image
from torchvision.transforms import v2

# allow absolute imports from project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# pyrefly: ignore [missing-import]
from src.data.dataset import MathDataset
# pyrefly: ignore [missing-import]
from src.data.tokenizer import HybridTokenizer
# pyrefly: ignore [missing-import]
from src.models.encoder import VisionEncoder
# pyrefly: ignore [missing-import]
from src.models.decoder import LatexDecoder

def generate(image_tensor, encoder, decoder, tokenizer, device, max_len=100):
    """
    Greedy Auto-Regressive Decoding.
    """
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # (1, C, H, W)
        images = image_tensor.unsqueeze(0).to(device)
        memory = encoder(images)
        
        # Start with just the <sos> token
        token_ids = [tokenizer.sos_id]
        
        for _ in range(max_len):
            # (1, SeqLen)
            tgt = torch.tensor([token_ids], dtype=torch.long, device=device)
            
            logits = decoder(tgt, memory)
            
            # get the highest probability token at the LAST timestep
            next_token = logits[0, -1, :].argmax(dim=-1).item()
            
            if next_token == tokenizer.eos_id:
                break
                
            token_ids.append(next_token)
            
    # Decode integers back to LaTeX string
    return tokenizer.decode(token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="real", choices=["toy", "real"])
    args = parser.parse_args()

    print(f"=== Initializing Inference Engine (Mode: {args.mode.upper()}) ===")
    
    # 1. Device
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Device: {device}")
    
    # 2. Tokenizer 
    tokenizer_path = str(project_root / "data" / "tokenizer_weights")
    tokenizer = HybridTokenizer(vocab_size=4000)
    try:
        tokenizer.load(tokenizer_path)
    except Exception:
        print(f"ERROR: Could not load tokenizer from {tokenizer_path}.")
        sys.exit(1)
    
    # 3. Load Checkpoint
    checkpoint_path = project_root / "checkpoint.pt"
    if not checkpoint_path.exists():
        print("ERROR: checkpoint.pt not found! Train the model first.")
        sys.exit(1)
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # 4. Instantiate Models
    encoder = VisionEncoder().to(device)
    decoder = LatexDecoder(vocab_size=checkpoint['vocab_size']).to(device)
    
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    
    print("Models loaded successfully.")
    
    # 5. Pick a test image (Image 0 from the dataset)
    dataset = MathDataset(mode=args.mode)
    if len(dataset) == 0:
        print(f"ERROR: {args.mode} dataset is empty.")
        sys.exit(1)
        
    test_idx = 0
    img_path, true_latex = dataset.items[test_idx]
    
    # Load and format exactly how the model expects
    img_tensor = read_image(img_path)[:3, :, :]
    
    # Resize and ToDtype, skip the random spatial augmentations during inference!
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((128, 512), antialias=True),
        v2.ToDtype(torch.float32, scale=True)
    ])
    img_tensor = transform(img_tensor)
    
    print(f"\n--- Inference Test ---")
    print(f"Target LaTeX:   {true_latex}")
    
    # 6. Generate!
    predicted_latex = generate(img_tensor, encoder, decoder, tokenizer, device)
    
    print(f"Predicted LaTeX: {predicted_latex}")
    
    if predicted_latex.replace(" ", "") == true_latex.replace(" ", ""):
        print("\nSUCCESS! The model perfectly overfit and recalled the image.")
    else:
        print("\nFAILURE. Model did not memorize the string correctly.")
