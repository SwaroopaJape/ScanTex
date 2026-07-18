# ScanTeX 

ScanTeX is a multi-modal Machine Learning application designed to translate rasterized images of mathematical equations directly into compilable LaTeX strings. 

## Technology Stack
- **Deep Learning Framework:** PyTorch (with Automatic Mixed Precision `torch.autocast`)
- **Vision Encoder:** ResNet18 (Torchvision) adapted for spatial sequence mapping via `einops`
- **Text Decoder:** Custom-built Transformer Decoder with causal masking and Label Smoothing
- **Tokenization:** Custom Hybrid BPE Lexer (Latex-aware)
- **Frontend / UI:** Streamlit (Custom styled)

## Key Features
- **Custom Hybrid Tokenizer:** Rather than blindly applying standard subword algorithms that shatter mathematical syntax, ScanTeX uses a custom-built Hybrid Lexer. It intelligently preserves structural LaTeX commands (like `\frac`, `\sum`, `\infty`) as atomic, indivisible tokens while delegating raw text strings to a HuggingFace BPE tokenizer. This vastly improves the model's structural awareness.
- **CFG-Based Synthetic Data Engine:** To ensure absolute syntactical correctness, the training pipeline leverages a robust Context-Free Grammar (CFG) generator. This allows the model to learn the fundamental rules of mathematics from a mathematically infinite stream of perfectly structured equations before encountering real-world noise.

## Architecture Overview
The pipeline uses a sequence-to-sequence translation approach. An equation image is passed through a truncated ResNet18 CNN, which outputs a spatial feature map. This map is flattened and fed as the memory representation into a Transformer Decoder. The Decoder auto-regressively predicts the structural LaTeX sequence token-by-token using greedy decoding.

## Data & Training
The model architecture and training mechanics are currently validated by overfitting onto a toy sandbox dataset (dynamically rendered LaTeX strings with on-the-fly data augmentations like affine transforms). 

Expanding the pipeline leverages the large-scale programmatically generated synthetic CFG equations to guarantee syntactical coverage, followed by fine-tuning on scraped real-world examples from sources like arXiv to handle authentic noise and formatting artifacts.

## Installation & Execution

This project is configured to use `uv`, an extremely fast Python package and project manager.

### 1. Install Dependencies
Ensure you have `uv` installed, then synchronize the environment:
```bash
# Install project dependencies
uv pip install -r requirements.txt
```

### 2. Run the Training Pipeline
To run the orchestrator (which initializes the dataset, trains the tokenizer, and runs the PyTorch training loop):
```bash
uv run python src/train.py
```
*This will output a `checkpoint.pt` file containing the model weights.*

### 3. Launch the Application UI
To boot up the interactive web application to upload images and see the model decode them live, simply run the entry point script:
```bash
uv run python main.py
```
*(Alternatively, you can run it directly via `uv run streamlit run ui/app.py`)*
