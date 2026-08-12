import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
import shutil

class MathDataset(Dataset):
    def __init__(self, mode="toy", data_dir="data/synthetic_train"):
        self.mode = mode
        self.items = []
        
        if self.mode == "toy":
            self.temp_dir = "data/temp_images"
            # 10 hardcoded LaTeX strings
            latex_strings_list = [
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
            
            os.makedirs(self.temp_dir, exist_ok=True)
            for idx, text in enumerate(latex_strings_list):
                img_path = os.path.join(self.temp_dir, f"{idx}.png")
                if not os.path.exists(img_path):
                    self._render_latex(text, img_path)
                self.items.append((img_path, text))
                
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((128, 512), antialias=True),
                v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                v2.ToDtype(torch.float32, scale=True)
            ])
            
        elif self.mode == "real":
            images_dir = os.path.join(data_dir, "images")
            labels_file = os.path.join(data_dir, "labels.txt")
            
            if os.path.exists(labels_file):
                with open(labels_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            self.items.append((os.path.join(images_dir, parts[0]), parts[1]))
            else:
                print(f"Warning: {labels_file} not found. Real dataset is empty.")
                
            # Define textbook spatial augmentations
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((128, 512), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
                v2.RandomPerspective(distortion_scale=0.2, p=0.2),
                v2.RandomApply([v2.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05))], p=0.2),
                v2.RandomApply([v2.ElasticTransform(alpha=20.0, sigma=5.0)], p=0.2)
            ])
            
        elif self.mode == "im2latex":
            data_dir = "data/im2latex_test"
            images_dir = os.path.join(data_dir, "images")
            labels_file = os.path.join(data_dir, "labels.txt")
            
            if os.path.exists(labels_file):
                with open(labels_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            self.items.append((os.path.join(images_dir, parts[0]), parts[1]))
            else:
                print(f"Warning: {labels_file} not found. Dataset is empty.")
                
            # No augmentation for benchmark evaluation — we want clean inference
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((128, 512), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ])

        else:
            raise ValueError(f"Unknown mode: {mode}")

    @property
    def latex_strings(self):
        return [item[1] for item in self.items]

    def _render_latex(self, text, path):
        # render string via matplotlib
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.axis('off')
        ax.text(0.5, 0.5, f"${text}$", size=20, ha='center', va='center')
        
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        
        try:
            # load raw image to tensor (C, H, W) and drop the Alpha channel (RGBA -> RGB)
            img = read_image(img_path)[:3, :, :]
        except Exception:
            # fallback if image corrupted
            img = torch.ones((3, 128, 512), dtype=torch.uint8) * 255
            
        # apply spatial augmentations
        img = self.transform(img)
        
        return img, label

if __name__ == "__main__":
    dataset = MathDataset(mode="toy")
    print(f"Total items in toy dataset: {len(dataset)}")
    
    # loop through all items to verify everything renders and loads safely
    for i in range(len(dataset)):
        img_tensor, text = dataset[i]
        print(f"Item {i}: Shape {list(img_tensor.shape)} | Label: {text}")
    print("All items successfully rendered, augmented, and loaded!")
    
    # cleanup temp folder after verification
    print(f"Cleaning up temporary directory: {dataset.temp_dir}")
    shutil.rmtree(dataset.temp_dir)
