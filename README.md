# ScanTeX

ScanTeX is a multi-modal deep learning system that converts images of handwritten or rendered mathematical equations into compilable LaTeX. Given a rasterized equation image, it produces the exact LaTeX source that would render it — a sequence-to-sequence image-to-markup task, built from scratch on a custom vision encoder, transformer decoder, and hybrid tokenizer.

## Architecture

An equation image is passed through a truncated ResNet18 backbone, producing a spatial feature map that's flattened into a sequence. That sequence is fed as memory into a hand-built Transformer decoder, which autoregressively predicts the LaTeX token stream using causal masking and label smoothing. At inference time, both greedy decoding and Beam Search (with length normalization) are supported.

```
image → ResNet18 (truncated) → flatten/reshape → Transformer decoder → LaTeX tokens
```

## Key features

- **Custom hybrid tokenizer** — rather than running raw LaTeX through an off-the-shelf subword tokenizer (which tends to fragment structural commands), ScanTeX uses a hand-built lexer that keeps commands like `\frac`, `\sum`, and `\infty` as single atomic tokens, while handing plain text off to a standard BPE tokenizer trained on real scraped equations. This gives the decoder a much stronger structural prior.
- **CFG-based synthetic data engine** — a probabilistic context-free grammar (built on top of a hand-authored `lark` grammar) generates a mathematically unbounded stream of syntactically valid equations. A geometric depth scaler (`p=0.1`) ensures the dataset has a heavy-tailed length distribution, forcing the model to learn long-range attention during pre-training.
- **Real-world fine-tuning corpus** — equations scraped from Wikipedia give the model exposure to authentic formatting and notation that synthetic data alone doesn't capture.
- **Beam Search with length normalization** — at inference time, Beam Search (width=5) keeps multiple candidate hypotheses alive. Length normalization (`alpha=0.7`) prevents the model from favouring short outputs, directly improving accuracy on complex, long equations.

## Tech stack

| Layer | Tools |
|---|---|
| Deep learning | PyTorch, mixed precision via `torch.autocast` |
| Vision encoder | ResNet18 (`torchvision`), reshaped for sequence output |
| Sequence decoder | Custom Transformer decoder (causal masking, label smoothing) |
| Tokenization | Custom hybrid LaTeX-aware lexer + HuggingFace `tokenizers` (BPE) |
| Grammar / synthetic data | `lark` (equation parsing and validation), custom PCFG generator |
| Rendering | `matplotlib` mathtext (also used as a compile-check) |
| Data collection | Wikipedia API |
| Experiment config | Hydra (`hydra-core`) |
| Interface | Streamlit |
| Tooling | `uv` for dependency/environment management |

## Repository structure

```
ScanTex/
├── configs/
│   └── config.yaml                  # Hydra config: batch_size, learning_rate, epochs, mode
├── src/
│   ├── data/
│   │   ├── dataset.py               # MathDataset — toy/real/finetune dataset loading + augmentation
│   │   ├── tokenizer.py             # HybridTokenizer (Lexer + BPE)
│   │   ├── latex_lexer.py           # LaTeX-aware lexer — identifies commands as atomic tokens
│   │   ├── grammar.lark             # Hand-authored formal LaTeX grammar (Lark DSL)
│   │   ├── grammar.py               # Auto-generated from grammar.lark (do not edit by hand)
│   │   └── string_generator.py      # Recursive PCFG-driven equation generator (geometric depth)
│   ├── models/
│   │   ├── encoder.py               # VisionEncoder (ResNet18-based)
│   │   └── decoder.py               # LatexDecoder (Transformer decoder)
│   ├── train.py                     # Hydra-driven training entry point (toy/real/finetune modes)
│   ├── train_tokenizer.py           # Tokenizer training script (--mode toy|scraped)
│   └── inference.py                 # Greedy and Beam Search decoding (with length normalization)
├── scripts/
│   ├── scrape_wikipedia.py          # Wikipedia equation scraper
│   ├── scrape_corpus.py             # ArXiv equation scraper (not used currently)
│   ├── parse_corpus.py              # Validates scraped equations; creates 80/20 train/val split
│   ├── generate_base_grammar.py     # Converts grammar.lark → grammar.py (uniform probabilities)
│   ├── pCFG_generator.py            # Re-weights grammar using real scraped rule frequencies
│   ├── render_synthetic_dataset.py  # Renders PCFG-generated equations as PNG images
│   ├── render_scraped_dataset.py    # Renders scraped equations as PNG images
│   └── evaluate_model.py            # Evaluates model: EM, CER, Compile Rate + per-bucket reporting
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures (grammars, tiny corpus, tokenizer)
│   ├── unit/                        # Pure logic tests (models, tokenizer, lexer, inference, generator)
│   ├── integration/                 # Full sub-pipeline tests (data pipeline, grammar pipeline)
│   └── regression/                  # Guards against known historical bugs (__ANON_* corruption)
├── ui/
│   ├── app.py                       # Streamlit application (Greedy/Beam toggle, LaTeX preview)
│   └── img/                         # Brand logos
├── main.py                          # Thin CLI entry point
├── pyproject.toml / uv.lock         # Dependency management (uv)
└── requirements.txt                 # pip-installable equivalent
```

## Getting started

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# install dependencies
uv sync

# run the full test suite to verify the environment
uv run pytest tests/ -v

# launch the interactive UI
uv run streamlit run ui/app.py
```

## Training pipeline

The full pipeline scales from a fast correctness check up to a fine-tuned model. See [`training_pipeline.md`](training_pipeline.md) for the full step-by-step reference with theory explanations and exact commands.

1. **Toy sanity checks** — run `uv run pytest tests/ -v` and a `mode=toy` training run to verify the environment before touching real data.
2. **Grammar bootstrap** — `generate_base_grammar.py` converts `grammar.lark` to a Python grammar with uniform rule probabilities.
3. **Scraping** — real-world equations are scraped from Wikipedia and validated against the grammar with an 80/20 train/val split.
4. **Grammar refinement** — `pCFG_generator.py` rebuilds the grammar using real rule frequencies from the scraped corpus.
5. **Tokenizer training** — the hybrid tokenizer is trained on the scraped corpus (`--mode scraped`) to learn realistic subwords. Train once and freeze.
6. **Synthetic rendering** — 200k equations are generated using the weighted PCFG and rendered to images.
7. **Scraped rendering** — the Wikipedia equations are rendered to images for fine-tuning.
8. **Model pre-training** — `mode=real` trains on the 200k synthetic images.
9. **Model fine-tuning** — `mode=finetune` continues training on the real scraped images.
10. **Evaluation** — `evaluate_model.py` reports Exact Match, Character Error Rate, and Compile Rate broken down by token-length complexity buckets.
11. **Inference & UI** — run predictions via Beam Search or Greedy decoding through the Streamlit app.
