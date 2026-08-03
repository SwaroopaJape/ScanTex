# ScanTeX

ScanTeX is a multi-modal deep learning system that converts images of handwritten or rendered mathematical equations into compilable LaTeX. Given a rasterized equation image, it produces the exact LaTeX source that would render it — a sequence-to-sequence image-to-markup task, built from scratch on a custom vision encoder, transformer decoder, and hybrid tokenizer.

## Architecture

An equation image is passed through a truncated ResNet18 backbone, producing a spatial feature map that's flattened into a sequence. That sequence is fed as memory into a hand-built Transformer decoder, which autoregressively predicts the LaTeX token stream using causal masking, label smoothing, and greedy decoding at inference time.

```
image → ResNet18 (truncated) → flatten/reshape (einops) → Transformer decoder → LaTeX tokens
```

## Key features

- **Custom hybrid tokenizer** — rather than running raw LaTeX through an off-the-shelf subword tokenizer (which tends to fragment structural commands), ScanTeX uses a hand-built lexer that keeps commands like `\frac`, `\sum`, and `\infty` as single atomic tokens, while handing plain text off to a standard BPE tokenizer. This gives the decoder a much stronger structural prior.
- **CFG-based synthetic data engine** — a probabilistic context-free grammar (built on top of a hand-authored `lark` grammar) generates a mathematically unbounded stream of syntactically valid equations, so the model learns correct LaTeX structure before it ever sees noisy real-world data.
- **Real-world fine-tuning corpus** — equations scraped from arXiv papers and Wikipedia give the model exposure to authentic formatting and notation noise that synthetic data alone doesn't capture.

## Tech stack

| Layer | Tools |
|---|---|
| Deep learning | PyTorch, mixed precision via `torch.autocast` |
| Vision encoder | ResNet18 (`torchvision`), reshaped for sequence output with `einops` |
| Sequence decoder | Custom Transformer decoder (causal masking, label smoothing) |
| Tokenization | Custom hybrid LaTeX-aware lexer + HuggingFace `tokenizers` (BPE) |
| Grammar / synthetic data | `lark` (equation parsing and validation), custom PCFG generator |
| Rendering | `matplotlib` mathtext (also used as a compile-check) |
| Data collection | `arxiv` API client, Wikipedia API |
| Experiment config | Hydra (`hydra-core`) |
| Interface | Streamlit |
| Tooling | `uv` for dependency/environment management |

## Repository structure

```
ScanTex/
├── configs/
│   └── config.yaml           # Hydra config: batch_size, learning_rate, epochs, mode
├── src/
│   ├── data/
│   │   ├── dataset.py         # MathDataset — toy/real dataset loading + augmentation
│   │   ├── tokenizer.py       # HybridTokenizer
│   │   ├── latex_lexer.py     # LaTeX-aware lexer/tokenizer classifier
│   │   ├── grammar.lark       # Hand-authored formal grammar
│   │   └── string_generator.py # Recursive PCFG-driven equation generator
│   ├── models/
│   │   ├── encoder.py         # VisionEncoder (ResNet18-based)
│   │   ├── decoder.py         # LatexDecoder (Transformer decoder)
│   │   └── cnn_backbone.py    # Standalone conv block (shape-sanity utility)
│   ├── train.py               # Hydra-driven training entry point
│   ├── train_tokenizer.py     # Tokenizer training script
│   └── inference.py           # CLI inference / greedy decoding
├── scripts/
│   ├── scrape_wikipedia.py    # Wikipedia equation scraper
│   ├── scrape_corpus.py       # arXiv equation scraper
│   ├── parse_corpus.py        # Builds rule-frequency counts from scraped equations
│   ├── pCFG_generator.py      # Builds/rebuilds the weighted grammar
│   ├── render_synthetic_dataset.py  # Renders large-scale synthetic image dataset
│   └── render_scraped_dataset.py    # Renders scraped equations to images
├── ui/
│   └── app.py                 # Streamlit application
├── main.py                    # Launches the Streamlit UI
├── pyproject.toml / uv.lock   # Dependency management (uv)
└── requirements.txt           # pip-installable equivalent
```

## Getting started

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management (a `pip`-based workflow works too).

```bash
# install dependencies
uv pip install -r requirements.txt

# train (defaults to a fast toy-data overfitting run)
uv run python src/train.py

# launch the interactive app
uv run python main.py
```

## Training pipeline

The full pipeline scales from a fast correctness check up to a fine-tuned model:

1. **Toy sanity checks** — every core module has a runnable demo (`python -m src.data.dataset`, `src.models.encoder`, etc.) to verify the environment before touching real data.
2. **Grammar bootstrap** — an initial probability-weighted grammar is generated from the hand-authored `grammar.lark`.
3. **Scraping** — real-world equations are pulled from Wikipedia and arXiv.
4. **Grammar refinement** — the grammar is rebuilt using real usage frequencies from the scraped corpus.
5. **Synthetic + scraped rendering** — both datasets are rendered to images via `matplotlib`.
6. **Tokenizer training** — a hybrid tokenizer is trained on the synthetic corpus (a pretrained one ships in the repo).
7. **Model training** — three stages: `mode=toy` (fast sanity run), `mode=real` (large-scale synthetic corpus), `mode=finetune` (real-world scraped data, initialized from the synthetic checkpoint).
8. **Inference & UI** — run predictions from the CLI or through the Streamlit app.

