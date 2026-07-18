import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

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

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=== Training Orchestrator Initialized ===")
    
    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("GPU Detected: Apple Silicon")
    else:
        device = torch.device("cpu")
        print("Fallback to CPU")
        
    # 2. Instantiate Dataset and Tokenizer
    print("\n--- Initializing Data Pipeline ---")
    dataset = MathDataset()
    tokenizer = HybridTokenizer(vocab_size=cfg.get("vocab_size", 4000))
    tokenizer.train(dataset.latex_strings)
    
    # Custom collator to convert raw strings into padded token batches
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        # encode strings and add <sos> and <eos>
        tokenized = [tokenizer.encode(item[1], add_special=True) for item in batch]
        
        # pad to max length in this batch
        max_len = max(len(t) for t in tokenized)
        padded_ids = []
        for t in tokenized:
            padded = t + [tokenizer.pad_id] * (max_len - len(t))
            padded_ids.append(padded)
            
        return images, torch.tensor(padded_ids, dtype=torch.long)

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 3. Instantiate Models
    print("\n--- Initializing Models ---")
    encoder = VisionEncoder().to(device)
    decoder = LatexDecoder(vocab_size=tokenizer.total_vocab_size).to(device)
    
    # 4. Optimization Setup
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=cfg.learning_rate
    )
    
    # Ignore the <pad> token during loss calculation, apply Label Smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.1)
    
    epochs = cfg.get("epochs", 500)
    
    # Mixed Precision Scaler
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # 5. Core Epoch Loop
    print("\n=== Starting Training Loop ===")
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}/{epochs:03d}", leave=False)
        for images, token_ids in pbar:
            images = images.to(device)
            token_ids = token_ids.to(device)
            
            # --- TEACHER FORCING ---
            # Input:  <sos> a ^ 2 + b ^ 2
            # Target: a ^ 2 + b ^ 2 <eos>
            decoder_input = token_ids[:, :-1]
            target_labels = token_ids[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward Pass & Loss with AMP
            with torch.autocast(device_type=device.type, enabled=use_amp):
                memory = encoder(images)
                logits = decoder(decoder_input, memory)
                
                # Compute Loss
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), 
                    target_labels.reshape(-1)
                )
            
            # Backpropagation (Scaled)
            scaler.scale(loss).backward()
            
            # Gradient Clipping requires unscaling first
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            # Step and update scaler
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Average Loss: {avg_loss:.4f}")
            
    # 6. Save Checkpoint
    checkpoint_path = "checkpoint.pt"
    checkpoint = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'vocab_size': tokenizer.total_vocab_size,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\n=== Training Complete! Model saved to {checkpoint_path} ===")

if __name__ == "__main__":
    main()
