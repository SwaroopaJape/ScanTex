import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
import shutil

class MathDataset(Dataset):
    def __init__(self, temp_dir="data/temp_images"):
        # 10 hardcoded LaTeX strings
        self.latex_strings = [
            r"a^2 + b^2 = c^2",
            r"E = mc^2",
            r"\int_{a}^{b} x^2 dx",
            r"\sum_{i=1}^{n} i",
            r"\frac{1}{2} m v^2",
            r"\sin^2(x) + \cos^2(x) = 1",
            r"e^{i\pi} + 1 = 0",
            r"F = G \frac{m_1 m_2}{r^2}",
            r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
            r"\lim_{x \to 0} \frac{\sin x}{x} = 1"
        ]
        
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # generate images if missing
        for idx, text in enumerate(self.latex_strings):
            img_path = os.path.join(self.temp_dir, f"{idx}.png")
            if not os.path.exists(img_path):
                self._render_latex(text, img_path)
                
        # define spatial augmentation pipeline
        self.transform = v2.Compose([
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def _render_latex(self, text, path):
        # render string via matplotlib
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.axis('off')
        ax.text(0.5, 0.5, f"${text}$", size=20, ha='center', va='center')
        
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)

    def __len__(self):
        return len(self.latex_strings)

    def __getitem__(self, idx):
        img_path = os.path.join(self.temp_dir, f"{idx}.png")
        
        # load raw image to tensor (C, H, W)
        img = read_image(img_path)
        
        # apply affine transform
        img = self.transform(img)
        
        return img, self.latex_strings[idx]

if __name__ == "__main__":
    dataset = MathDataset()
    print(f"Total items in dataset: {len(dataset)}")
    
    # loop through all items to verify everything renders and loads safely
    for i in range(len(dataset)):
        img_tensor, text = dataset[i]
        print(f"Item {i}: Shape {list(img_tensor.shape)} | Label: {text}")
    print("All 10 items successfully rendered, augmented, and loaded!")
    
    # cleanup temp folder after verification
    print(f"Cleaning up temporary directory: {dataset.temp_dir}")
    shutil.rmtree(dataset.temp_dir)
