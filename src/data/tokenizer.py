"""
Hybrid LaTeX-aware tokenizer.

Pipeline:
  1. Lexer pass  → stream of LaTeX tokens + PLAIN_TEXT segments
  2. BPE pass    → only applied to PLAIN_TEXT segments
  3. Merge       → single ordered token list with special-token IDs
"""

import os
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from latex_lexer import Token, TokenType, lex

class HybridTokenizer:
    # reserved control tokens
    PAD = "<pad>"
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size

        # LaTeX vocabulary (built from lexer output)
        self._latex_to_id: dict[str, int] = {}
        self._id_to_latex: dict[int, str] = {}

        # BPE tokenizer (HuggingFace, for plain-text only)
        self._bpe: Tokenizer | None = None

        # Unified vocab (latex IDs first, then BPE IDs shifted)
        self._unified_to_str: dict[int, str] = {}
        self._str_to_unified: dict[str, int] = {}
        self._bpe_offset: int = 0

    # Training
    def train(self, latex_strings: list[str], bpe_corpus_path: str | None = None):
        """
        Build the full vocabulary in two stages:
          1. Scan all latex_strings with the lexer → collect LaTeX tokens
          2. Train BPE on either a supplied corpus file or
             the leftover PLAIN_TEXT segments
        """
        # stage 1: collect LaTeX tokens
        special = [self.PAD, self.SOS, self.EOS, self.UNK]
        latex_vocab: set[str] = set()
        plain_text_segments: list[str] = []

        for s in latex_strings:
            for tok in lex(s):
                if tok.type == TokenType.PLAIN_TEXT:
                    plain_text_segments.append(tok.value)
                else:
                    latex_vocab.add(tok.value)

        # assign IDs: specials first, then latex tokens sorted for determinism
        ordered = special + sorted(latex_vocab)
        self._latex_to_id = {v: i for i, v in enumerate(ordered)}
        self._id_to_latex = {i: v for v, i in self._latex_to_id.items()}
        self._bpe_offset = len(ordered)

        # stage 2: train BPE on plain-text segments
        bpe_budget = self.vocab_size - self._bpe_offset
        if bpe_budget < 1:
            bpe_budget = 64  # minimum fallback

        self._bpe = Tokenizer(models.BPE())
        self._bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=bpe_budget,
            special_tokens=[],  # specials already in latex vocab
            show_progress=False,
        )

        # train from corpus file or from collected plain-text
        if bpe_corpus_path and os.path.exists(bpe_corpus_path):
            self._bpe.train([bpe_corpus_path], trainer)
        else:
            # write segments to temp file for the trainer
            tmp = Path("_bpe_train_tmp.txt")
            tmp.write_text("\n".join(plain_text_segments) if plain_text_segments else "a")
            self._bpe.train([str(tmp)], trainer)
            tmp.unlink()

        # build unified lookup
        self._unified_to_str = dict(self._id_to_latex)
        bpe_vocab = self._bpe.get_vocab()
        for bpe_tok, bpe_id in bpe_vocab.items():
            uid = bpe_id + self._bpe_offset
            self._unified_to_str[uid] = bpe_tok
        self._str_to_unified = {v: k for k, v in self._unified_to_str.items()}

    # Encode
    def encode(self, latex_string: str, add_special: bool = True) -> list[int]:
        """LaTeX string → list of unified token IDs."""
        tokens = lex(latex_string)
        ids: list[int] = []

        if add_special:
            ids.append(self._latex_to_id[self.SOS])

        for tok in tokens:
            if tok.type == TokenType.PLAIN_TEXT:
                # run BPE on this segment
                bpe_out = self._bpe.encode(tok.value)
                ids.extend(bid + self._bpe_offset for bid in bpe_out.ids)
            else:
                # direct latex vocab lookup
                tid = self._latex_to_id.get(tok.value)
                if tid is not None:
                    ids.append(tid)
                else:
                    ids.append(self._latex_to_id[self.UNK])

        if add_special:
            ids.append(self._latex_to_id[self.EOS])

        return ids

    # Decode
    def decode(self, ids: list[int]) -> str:
        """List of unified token IDs → reconstructed string."""
        parts: list[str] = []
        for uid in ids:
            tok_str = self._unified_to_str.get(uid)
            if tok_str is None or tok_str in (self.PAD, self.SOS, self.EOS):
                continue
            if tok_str == self.UNK:
                parts.append("")
            else:
                parts.append(tok_str)
        return " ".join(parts)

    # Persistence
    def save(self, directory: str):
        """Save tokenizer state to disk."""
        os.makedirs(directory, exist_ok=True)
        # latex vocab
        with open(os.path.join(directory, "latex_vocab.json"), "w") as f:
            json.dump(self._latex_to_id, f, indent=2)
        # bpe model
        if self._bpe:
            self._bpe.save(os.path.join(directory, "bpe.json"))
        # metadata
        meta = {"vocab_size": self.vocab_size, "bpe_offset": self._bpe_offset}
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, directory: str):
        """Load tokenizer state from disk."""
        with open(os.path.join(directory, "latex_vocab.json")) as f:
            self._latex_to_id = json.load(f)
        self._id_to_latex = {int(v): k for k, v in self._latex_to_id.items()}

        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        self.vocab_size = meta["vocab_size"]
        self._bpe_offset = meta["bpe_offset"]

        self._bpe = Tokenizer.from_file(os.path.join(directory, "bpe.json"))

        # rebuild unified lookup
        self._unified_to_str = dict(self._id_to_latex)
        for bpe_tok, bpe_id in self._bpe.get_vocab().items():
            uid = bpe_id + self._bpe_offset
            self._unified_to_str[uid] = bpe_tok
        self._str_to_unified = {v: k for k, v in self._unified_to_str.items()}

    # Helpers
    @property
    def pad_id(self) -> int:
        return self._latex_to_id[self.PAD]

    @property
    def sos_id(self) -> int:
        return self._latex_to_id[self.SOS]

    @property
    def eos_id(self) -> int:
        return self._latex_to_id[self.EOS]

    @property
    def total_vocab_size(self) -> int:
        return len(self._unified_to_str)

    def __len__(self) -> int:
        return self.total_vocab_size

if __name__ == "__main__":
    # training corpus
    corpus = [
        r"a^2 + b^2 = c^2",
        r"E = mc^2",
        r"\int_{a}^{b} x^2 dx",
        r"\sum_{i=1}^{n} i",
        r"\frac{1}{2} m v^2",
        r"\sin^2(x) + \cos^2(x) = 1",
        r"e^{i\pi} + 1 = 0",
        r"F = G \frac{m_1 m_2}{r^2}",
        r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
        r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
    ]

    tok = HybridTokenizer(vocab_size=500)
    tok.train(corpus)

    print(f"Total vocabulary size: {tok.total_vocab_size}")
    print(f"  LaTeX tokens: {tok._bpe_offset}")
    print(f"  BPE tokens:   {tok.total_vocab_size - tok._bpe_offset}")
    print(f"  PAD={tok.pad_id}  SOS={tok.sos_id}  EOS={tok.eos_id}")

    # roundtrip test
    test_strings = [
        r"\sum_{i=1}^{n} i",
        r"\frac{1}{2} m v^2",
        r"e^{i\pi} + 1 = 0",
    ]
    for s in test_strings:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        print(f"\n  Input:   {s}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: {decoded}")
        print(f"  Tokens:  {len(ids)}")
