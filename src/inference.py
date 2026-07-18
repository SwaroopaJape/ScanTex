import sys
from pathlib import Path
import torch
import os
from torchvision.io import read_image
from torchvision.transforms import v2

# allow absolute imports from project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.dataset import MathDataset
from src.data.tokenizer import HybridTokenizer
from src.models.encoder import VisionEncoder
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
    print("=== Initializing Inference Engine ===")
    
    # 1. Device
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Device: {device}")
    
    # 2. Tokenizer 
    # (Re-train on the sandbox data to reconstruct vocab perfectly)
    dataset = MathDataset()
    tokenizer = HybridTokenizer(vocab_size=4000)
    tokenizer.train(dataset.latex_strings)
    
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
    test_idx = 0
    img_path = os.path.join(dataset.temp_dir, f"{test_idx}.png")
    
    # Load and format exactly how the model expects
    img_tensor = read_image(img_path)[:3, :, :]
    
    # Only apply ToDtype, skip the RandomAffine during inference!
    transform = v2.ToDtype(torch.float32, scale=True)
    img_tensor = transform(img_tensor)
    
    true_latex = dataset.latex_strings[test_idx]
    print(f"\n--- Inference Test ---")
    print(f"Target LaTeX:   {true_latex}")
    
    # 6. Generate!
    predicted_latex = generate(img_tensor, encoder, decoder, tokenizer, device)
    
    print(f"Predicted LaTeX: {predicted_latex}")
    
    if predicted_latex.replace(" ", "") == true_latex.replace(" ", ""):
        print("\nSUCCESS! The model perfectly overfit and recalled the image.")
    else:
        print("\nFAILURE. Model did not memorize the string correctly.")
