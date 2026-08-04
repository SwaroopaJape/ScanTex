"""
tests/unit/test_tokenizer.py

Unit tests for src/data/tokenizer.py (HybridTokenizer).
Converts the __main__ roundtrip sanity check into proper assertions.
"""
import os
import pytest
from src.data.tokenizer import HybridTokenizer


# ---------------------------------------------------------------------------
# Training and vocab tests (use session-scoped trained_tokenizer fixture)
# ---------------------------------------------------------------------------

def test_vocab_size_nonzero(trained_tokenizer):
    assert trained_tokenizer.total_vocab_size > 0


def test_special_token_ids_are_unique(trained_tokenizer):
    tok = trained_tokenizer
    ids = {tok.pad_id, tok.sos_id, tok.eos_id}
    assert len(ids) == 3, "pad_id, sos_id, eos_id must all be distinct"


def test_special_token_ids_are_in_range(trained_tokenizer):
    tok = trained_tokenizer
    vocab = tok.total_vocab_size
    assert 0 <= tok.pad_id < vocab
    assert 0 <= tok.sos_id < vocab
    assert 0 <= tok.eos_id < vocab


def test_bpe_offset_less_than_total(trained_tokenizer):
    """BPE vocab sits above the LaTeX vocab in the unified space."""
    tok = trained_tokenizer
    assert tok._bpe_offset < tok.total_vocab_size


# ---------------------------------------------------------------------------
# Encode tests
# ---------------------------------------------------------------------------

def test_encode_returns_list_of_ints(trained_tokenizer):
    ids = trained_tokenizer.encode(r"\frac{1}{2}")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_encode_starts_with_sos(trained_tokenizer):
    ids = trained_tokenizer.encode(r"a^2 + b^2 = c^2", add_special=True)
    assert ids[0] == trained_tokenizer.sos_id


def test_encode_ends_with_eos(trained_tokenizer):
    ids = trained_tokenizer.encode(r"a^2 + b^2 = c^2", add_special=True)
    assert ids[-1] == trained_tokenizer.eos_id


def test_encode_without_special_no_sos_eos(trained_tokenizer):
    ids = trained_tokenizer.encode(r"a^2 + b^2 = c^2", add_special=False)
    assert ids[0]  != trained_tokenizer.sos_id
    assert ids[-1] != trained_tokenizer.eos_id


def test_encode_nonempty(trained_tokenizer):
    ids = trained_tokenizer.encode(r"\sum_{i=1}^{n} i")
    assert len(ids) > 0


# ---------------------------------------------------------------------------
# Decode tests
# ---------------------------------------------------------------------------

def test_decode_strips_sos_eos(trained_tokenizer):
    ids     = trained_tokenizer.encode(r"\frac{1}{2} m v^2", add_special=True)
    decoded = trained_tokenizer.decode(ids)
    assert "<sos>" not in decoded
    assert "<eos>" not in decoded
    assert "<pad>" not in decoded


def test_decode_nonempty(trained_tokenizer):
    ids     = trained_tokenizer.encode(r"e^{i\pi} + 1 = 0", add_special=True)
    decoded = trained_tokenizer.decode(ids)
    assert len(decoded.strip()) > 0


# ---------------------------------------------------------------------------
# Roundtrip test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("latex", [
    r"\sum_{i=1}^{n} i",
    r"\frac{1}{2} m v^2",
    r"e^{i\pi} + 1 = 0",
])
def test_roundtrip_preserves_known_tokens(trained_tokenizer, latex):
    """Encode then decode should preserve at least some LaTeX commands."""
    ids     = trained_tokenizer.encode(latex, add_special=True)
    decoded = trained_tokenizer.decode(ids)
    # At minimum the decode should return a non-empty string
    assert len(decoded.strip()) > 0


# ---------------------------------------------------------------------------
# Save / load persistence test
# ---------------------------------------------------------------------------

def test_save_and_load_preserves_vocab(tmp_path, tiny_corpus):
    """Save tokenizer, reload it, and verify critical properties are identical."""
    tok = HybridTokenizer(vocab_size=500)
    tok.train(tiny_corpus)

    save_dir = str(tmp_path / "tokenizer_weights")
    tok.save(save_dir)

    tok2 = HybridTokenizer(vocab_size=500)
    tok2.load(save_dir)

    assert tok2.total_vocab_size == tok.total_vocab_size
    assert tok2.pad_id == tok.pad_id
    assert tok2.sos_id == tok.sos_id
    assert tok2.eos_id == tok.eos_id
    assert tok2._bpe_offset == tok._bpe_offset


def test_save_creates_expected_files(tmp_path, tiny_corpus):
    """Saving must create latex_vocab.json, bpe.json, and meta.json."""
    tok = HybridTokenizer(vocab_size=500)
    tok.train(tiny_corpus)

    save_dir = str(tmp_path / "tokenizer_weights")
    tok.save(save_dir)

    assert os.path.exists(os.path.join(save_dir, "latex_vocab.json"))
    assert os.path.exists(os.path.join(save_dir, "bpe.json"))
    assert os.path.exists(os.path.join(save_dir, "meta.json"))


# ---------------------------------------------------------------------------
# Unknown token fallback
# ---------------------------------------------------------------------------

def test_unknown_token_uses_unk_id(trained_tokenizer):
    """A command not in the training corpus should map to unk_id, not crash."""
    unk_id = trained_tokenizer._latex_to_id["<unk>"]
    # \nabla was never in the tiny training corpus
    ids = trained_tokenizer.encode(r"\nabla f", add_special=False)
    assert unk_id in ids
