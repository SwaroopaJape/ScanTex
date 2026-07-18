import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange

import sys
import shutil
from pathlib import Path
from torch.utils.data import DataLoader

# pyrefly: ignore [missing-import]
from src.data.dataset import MathDataset
    

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # load pretrained resnet18
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # strip the avgpool and fc classification head
        # this leaves us with the raw 2D spatial feature map
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
    def forward(self, x):
        # input: (B, 3, H, W)
        features = self.backbone(x)
        
        # output of backbone is (B, 512, H/32, W/32)
        # flatten spatial dimensions to sequence: (B, C, H, W) -> (B, H*W, C)
        sequence = rearrange(features, 'b c h w -> b (h w) c')
        
        return sequence

if __name__ == "__main__":
    # allow absolute imports from src
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
        
    # initialize dataset and load a batch of 4
    dataset = MathDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    images, latex_strings = next(iter(dataloader))
    
    # pass through the encoder
    encoder = VisionEncoder()
    output_sequence = encoder(images)
    
    print("=== Vision Encoder Verification ===")
    print(f"Input image batch:       {list(images.shape)}  # (B, C, H, W)")
    print(f"Output sequence shape:   {list(output_sequence.shape)}  # (B, S, C)")
    print(f"Sequence length (S):     {output_sequence.shape[1]}")
    print(f"Feature dimensions (C):  {output_sequence.shape[2]}")
    
    # cleanup temp folder
    shutil.rmtree(dataset.temp_dir)
