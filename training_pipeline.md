# ScanTex Training Pipeline

A step-by-step reference for anyone cloning this repository and running the full pipeline from scratch — from sanity checks to a fully trained OCR model.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setup](#3-setup)
4. [Stage 0: Sanity Checks (Toy Mode)](#4-stage-0-sanity-checks-toy-mode)
5. [Stage 1: Build the Grammar](#5-stage-1-build-the-grammar)
6. [Stage 2: Scrape Real Data](#6-stage-2-scrape-real-data)
7. [Stage 3: Train the Tokenizer](#7-stage-3-train-the-tokenizer)
8. [Stage 4: Generate Synthetic Dataset](#8-stage-4-generate-synthetic-dataset)
9. [Stage 5: Pre-Train the Model](#9-stage-5-pre-train-the-model)
10. [Stage 6: Fine-Tune the Model](#10-stage-6-fine-tune-the-model)
11. [Stage 7: Evaluate](#11-stage-7-evaluate)
12. [Stage 8: Run the UI](#12-stage-8-run-the-ui)
13. [Test Suite Reference](#13-test-suite-reference)

---

## 1. Project Overview

ScanTex is an end-to-end pipeline that takes an **image of a mathematical equation** and outputs the corresponding **compilable LaTeX source code**.

**Motivation Behind the Approach:**
Standard OCR treats the page as rows of left-to-right characters. Math is inherently two-dimensional: fractions have a numerator above a denominator, subscripts sit below their base, and limits span multiple lines. A general-purpose character recognizer cannot model this spatial structure. ScanTex is purpose-built for this domain.

**Architecture:**
```
Image → [Vision Encoder (ResNet18)] → feature map → [Transformer Decoder] → LaTeX tokens
```

The encoder compresses the visual information into a 1D sequence of feature vectors. The decoder then reads those features (via cross-attention) and generates the LaTeX output one token at a time, auto-regressively.

---

## 2. Repository Structure

```
ScanTex/
├── src/
│   ├── data/
│   │   ├── grammar.lark         # Formal grammar definition for valid LaTeX (Lark DSL)
│   │   ├── grammar.py           # Auto-generated Python grammar (do NOT edit by hand)
│   │   ├── latex_lexer.py       # Rule-based lexer — identifies LaTeX commands as atomic units
│   │   ├── tokenizer.py         # HybridTokenizer: Lexer (Phase 1) + BPE (Phase 2) combined
│   │   ├── string_generator.py  # PCFG-based synthetic LaTeX string generator (geometric depth)
│   │   └── dataset.py           # PyTorch Dataset for loading rendered image-label pairs
│   ├── models/
│   │   ├── encoder.py           # ResNet18-based Vision Encoder
│   │   └── decoder.py           # Transformer Decoder for auto-regressive token generation
│   ├── inference.py             # Greedy and Beam Search (with length normalization) decoding
│   ├── train.py                 # Main training orchestrator (Hydra config, toy/real/finetune)
│   └── train_tokenizer.py       # Standalone tokenizer training script (--mode toy|scraped)
│
├── scripts/
│   ├── scrape_wikipedia.py      # Scrapes LaTeX equations from Wikipedia articles
│   ├── scrape_corpus.py         # Alternative scraper (ArXiv / other sources) — not used currently
│   ├── parse_corpus.py          # Validates scraped strings against grammar, creates 80/20 split
│   ├── generate_base_grammar.py # Converts grammar.lark → grammar.py with uniform probabilities
│   ├── pCFG_generator.py        # Re-weights grammar.py using scraped rule frequencies
│   ├── render_synthetic_dataset.py # Renders 200k PCFG strings as PNG images to data/synthetic_train/
│   ├── render_scraped_dataset.py   # Renders scraped strings as PNG images to data/scraped_train/
│   └── evaluate_model.py           # Runs EM, CER, Compile Rate evaluation with per-bucket reporting
│
├── tests/
│   ├── conftest.py              # Shared session-scoped fixtures (grammars, tiny corpus, tokenizer)
│   ├── unit/
│   │   ├── test_latex_lexer.py      # Lexer correctly identifies LaTeX commands as atomic tokens
│   │   ├── test_tokenizer.py        # Encode/decode round-trips, vocab size, save/load
│   │   ├── test_models.py           # Encoder/decoder output shapes and causal masking
│   │   ├── test_inference.py        # Greedy and Beam Search generate valid strings
│   │   └── test_string_generator.py # Geometric depth limiter correctly constrains generation
│   ├── integration/
│   │   ├── test_data_pipeline.py    # Toy dataset load, collate_fn padding, SOS/EOS injection
│   │   └── test_grammar_pipeline.py # Generated strings are clean (no __ANON_* leaks) and renderable
│   └── regression/
│       └── test_grammar_integrity.py # Guards against the __ANON_* grammar corruption bug (Aug 2026)
│
├── configs/
│   └── config.yaml              # Hydra config: default epochs, batch_size, vocab_size
│
├── data/                        # All datasets — gitignored, generated locally
│   ├── synthetic_train/         # 200k PCFG-generated image-label pairs (Stage 4)
│   ├── scraped_train/           # Wikipedia scraped image-label pairs (Stage 2)
│   ├── scraped/
│   │   ├── train_equations.txt  # 80% split of validated scraped LaTeX strings
│   │   └── val_equations.txt    # 20% split — held out, never used in training
│   ├── extracted_grammar/
│   │   └── weighted_grammar.py  # PCFG with frequency-weighted rule probabilities (Stage 1b)
│   ├── scraped_info/
│   │   └── rule_frequencies.json # Raw rule counts from corpus analysis
│   └── tokenizer_weights/       # Saved HybridTokenizer (Stage 3) — latex_vocab.json, bpe.json, meta.json
│
├── ui/
│   ├── app.py                   # Streamlit web interface with Greedy/Beam toggle
│   └── img/
│       ├── Full_logo.png        # Brand logo (wide, used in sidebar)
│       └── mini_logo.png        # Brand logo (compact, used in header)
│
├── main.py                      # Thin entry point (delegates to train.py)
├── checkpoint.pt                # Saved model weights — produced by training
├── pyproject.toml               # Project metadata and uv dependencies
├── requirements.txt             # Pip-compatible dependency list
├── README.md                    # High-level project overview
└── training_pipeline.md         # This file
```

---

## 3. Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd ScanTex

# Install uv (the package manager used in this project)
pip install uv

# Install all dependencies
uv sync

# Verify the install by running the test suite
uv run pytest tests/ -v
```

> All commands in this guide use `uv run` to ensure they execute inside the project's virtual environment.

---

## 4. Stage 0: Sanity Checks (Toy Mode)

**Before doing anything else**, run the test suite to confirm the codebase is healthy and all components are wired together correctly.

```bash
uv run pytest tests/ -v
```

This runs 72 tests covering models, tokenizer, grammar, data pipeline, and inference. All should pass on a clean checkout.

You can also do a quick end-to-end smoke test using the 10 pre-rendered toy images included in `data/toy/`:

```bash
# Train a tiny tokenizer on the toy corpus
uv run python src/train_tokenizer.py --mode toy

# Train the model on toy data for 1 epoch to verify the loop works
uv run python src/train.py mode=toy epochs=1
```

If training completes without error, your environment is correctly configured.

---

## 5. Stage 1: Build the Grammar

**Theory:** The synthetic data pipeline is driven by a **Probabilistic Context-Free Grammar (PCFG)**. A CFG is a set of recursive production rules (e.g., `EXPR → TERM + TERM`, `TERM → FACTOR`). Adding probabilities to each rule (making it a PCFG) lets us control how often the generator produces, say, a fraction versus a simple variable.

The grammar is defined in the Lark DSL format (`grammar.lark`) and must first be converted into a plain Python data structure. Then, optionally, its rule weights are updated based on the frequencies observed in the real scraped corpus.

**Step 1a — Generate the base grammar (uniform probabilities):**
```bash
uv run python scripts/generate_base_grammar.py
```
This reads `src/data/grammar.lark` and writes `src/data/grammar.py`. All production rules are given equal probability weights. This file is auto-generated and is listed in `.gitignore`.

**Step 1b — Weight the grammar with real-world frequencies (run *after* scraping):**
```bash
uv run python scripts/pCFG_generator.py
```
This reads the scraped `train_equations.txt`, counts how often each grammar rule appears (with Laplace smoothing to handle unseen rules), and writes a `weighted_grammar.py` to `data/extracted_grammar/`. The weighted grammar produces more realistic equations because common structures (like simple fractions) are generated far more often than exotic ones.

> **Note on the Long-Tail Depth Fix:** By default, PCFGs suffer from *exponential length decay* — the chance of generating a deep, 50-token equation collapses to near zero because each recursive expansion multiplies a probability below 1.0. To counter this, `string_generator.py` uses a **geometric depth scaler**: at the start of each equation, it flips a biased coin (`p=0.1`). Each successful flip adds another full depth level to the generation limit. This injects a controlled number of long, complex equations into the training set, directly teaching the model long-range attention.

---

## 6. Stage 2: Scrape Real Data

**Theory:** The scraped data serves two purposes: (1) training the tokenizer's BPE component on *real* human-written subwords, and (2) fine-tuning the model on realistic rendering styles and typographic choices.

```bash
# Scrape LaTeX equations from Wikipedia (recommended, ~30-60 min)
uv run python scripts/scrape_wikipedia.py

# Validate scraped equations against the grammar and create the 80/20 train/val split
uv run python scripts/parse_corpus.py
```

After this stage, you should have:
- `data/scraped/train_equations.txt` — training source strings
- `data/scraped/val_equations.txt` — held-out validation strings (never used in training)

```bash
# Render the scraped strings as PNG images for fine-tuning
uv run python scripts/render_scraped_dataset.py
```

This produces images in `data/scraped_train/`.

---

## 7. Stage 3: Train the Tokenizer

**Theory:** A tokenizer converts raw LaTeX strings into sequences of integer IDs that the model can process. A generic tokenizer (like one designed for English) would split `\frac` into `\`, `f`, `r`, `a`, `c` — destroying the semantic meaning of the command. ScanTex uses a **Hybrid Tokenizer** that avoids this:

- **Phase 1 (Lexer):** A hand-written rule-based lexer (`latex_lexer.py`) identifies LaTeX commands and structural tokens as atomic units. `\frac` stays as one token.
- **Phase 2 (BPE):** Any remaining plain text (e.g., words inside `\text{}`) is handled by a Byte-Pair Encoding (BPE) algorithm that learns common character substrings from the corpus.

**Critical rule:** Train the tokenizer **once**, then freeze it. Retraining the tokenizer changes the integer ID assigned to every token, which would corrupt any existing model checkpoint (the model's embedding weights would map to the wrong symbols).

**Why train on scraped data only?** The synthetic data is randomly generated by the PCFG and contains mostly short variable names with no meaningful subword structure. The scraped Wikipedia equations contain real English mathematical terminology (`\text{where}`, `\text{for all}`, `\sin`, `\cos`). Training BPE on real data gives the tokenizer a vocabulary that actually generalizes.

```bash
uv run python src/train_tokenizer.py --mode scraped
```

This saves the tokenizer to `data/tokenizer_weights/`.

---

## 8. Stage 4: Generate Synthetic Dataset

**Theory:** We use 200,000 synthetically generated equations for pre-training. The model can never see the exact same equation twice during training (the generator is stochastic), which dramatically reduces overfitting. Rendering is done with Matplotlib's `mathtext` engine.

```bash
uv run python scripts/render_synthetic_dataset.py
```

This generates image-label pairs in `data/synthetic_train/`. It takes approximately 20–40 minutes depending on hardware.

---

## 9. Stage 5: Pre-Train the Model

**Theory:** In the first training phase, the model is exposed to the large synthetic dataset. This teaches it the fundamental spatial structure of mathematical notation: how to read fractions, scripts, integrals, and sums from an image. The loss function is standard cross-entropy over the token sequence (teacher forcing during training).

If you run out of GPU memory (CUDA OOM error), reduce `batch_size`. The Transformer's memory usage scales quadratically with sequence length. When long equations are included, dynamic padding can cause a single batch to consume far more memory than expected.

```bash
uv run python src/train.py mode=real epochs=15 batch_size=24
```

This saves a checkpoint to `checkpoint.pt` after each epoch.

---

## 10. Stage 6: Fine-Tune the Model

**Theory:** After pre-training, the model understands LaTeX structure in the abstract, but it has only ever seen synthetically rendered equations (Matplotlib fonts, consistent spacing). Wikipedia equations are rendered differently. Fine-tuning on the real scraped images teaches the model to adapt to real-world typographic variation. Because the tokenizer is frozen and the model's weights are warm, fine-tuning converges much faster than pre-training (typically 3–5 epochs).

```bash
uv run python src/train.py mode=finetune epochs=5 batch_size=32
```

> **Important:** The `finetune` mode loads from the existing `checkpoint.pt` and continues training on `data/scraped_train/`.

---

## 11. Stage 7: Evaluate

The evaluation script runs the model on the held-out validation set and reports metrics for both **Greedy** and **Beam Search** decoding side-by-side.

```bash
uv run python scripts/evaluate_model.py
```

**Metrics explained:**
| Metric | Description |
| :--- | :--- |
| **Exact Match (EM)** | % of equations where the prediction exactly equals the ground truth (after stripping spaces). The strictest possible measure. |
| **Character Error Rate (CER)** | Levenshtein edit distance normalized by string length. 0% is perfect. Captures how close "near-miss" predictions are. |
| **Compile Rate** | % of predictions that Matplotlib can render without an error. A syntactically valid but semantically wrong prediction still counts here. |

**Per-Complexity Buckets:**
The results are broken down by token length (1–10, 11–25, 26–50, 50+). This reveals the model's "accuracy cliff" — where performance starts to degrade — which is far more useful than a single global average.

**Why Beam Search?** Greedy decoding picks the single most probable token at each step. This means one wrong early token can cascade into a completely wrong sequence. Beam Search (width=5) keeps the top 5 candidate hypotheses alive simultaneously, allowing the model to recover from early mistakes.

**Why Length Normalization?** Beam Search scores are the sum of log-probabilities. Since log-probabilities are negative, longer sequences have lower (more negative) raw scores, causing Beam Search to prefer shorter outputs and hallucinate early `<eos>` tokens. Length normalization divides the raw score by `length ** alpha` (alpha=0.7), judging each candidate by its *average* per-token confidence instead.

---

## 12. Stage 8: Run the UI

```bash
uv run streamlit run ui/app.py
```

The Streamlit interface allows you to upload an image and run OCR on it, with a toggle to switch between Greedy and Beam Search decoding. It displays the predicted LaTeX, a rendered preview, and the inference latency.

---

## 13. Test Suite Reference

The test suite is organized into three layers:

### `tests/unit/` — Pure logic tests, no I/O
| File | What it tests |
| :--- | :--- |
| `test_latex_lexer.py` | That the lexer correctly identifies `\frac`, `^`, `{`, etc. as atomic units |
| `test_tokenizer.py` | Tokenizer encode/decode round-trips, special token IDs, vocab size, save/load |
| `test_models.py` | Encoder output shape, decoder output shape and causal masking |
| `test_inference.py` | That `generate()` returns a valid string for both greedy and beam modes |
| `test_string_generator.py` | That the geometric depth limiter correctly constrains recursive generation |

### `tests/integration/` — Full sub-pipeline tests
| File | What it tests |
| :--- | :--- |
| `test_data_pipeline.py` | Toy dataset loading, `collate_fn` padding, SOS/EOS injection |
| `test_grammar_pipeline.py` | That generated strings contain no `__ANON_*` leaked tokens and are renderable |

### `tests/regression/` — Guards against known historical bugs
| File | What it tests |
| :--- | :--- |
| `test_grammar_integrity.py` | Structural integrity of `grammar.py` — confirms it was generated by `generate_base_grammar.py` and not corrupted (the `__ANON_*` corruption bug, Aug 2026) |

### `tests/conftest.py`
Shared session-scoped fixtures: `base_grammar_path`, `weighted_grammar_path`, `tiny_corpus`, and `trained_tokenizer`. These are loaded once per test session and reused, keeping test runs fast.
