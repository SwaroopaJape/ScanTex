import torch
import torch.nn as nn
from einops import rearrange

class BasicConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # Flatten spatial dimensions to sequence: (B, C, H, W) -> (B, H*W, C)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

if __name__ == "__main__":
    dummy_input = torch.randn(8, 3, 64, 64)
    model = BasicConvBlock()
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") 
