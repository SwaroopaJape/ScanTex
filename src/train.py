import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# load config before main execution
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=== Training Orchestrator Initialized ===")
    
    # log config
    print("\n--- Loaded Hyperparameters ---")
    print(OmegaConf.to_yaml(cfg))
    print(f"Direct Access -> Batch Size: {cfg.batch_size}, LR: {cfg.learning_rate}")
    
    # hardware detection
    print("\n--- Hardware Detection ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Detected: {torch.cuda.get_device_name(0)} (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("GPU Detected: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Fallback to CPU")
        
    print(f"\nFinal device: {device}")

if __name__ == "__main__":
    main()
