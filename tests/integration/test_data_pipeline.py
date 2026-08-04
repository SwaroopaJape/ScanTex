"""
tests/integration/test_data_pipeline.py

Integration tests for the dataset → tokenizer → collate_fn chain
that is used in src/train.py.

Also tests the inference.generate() function with untrained (random-weight)
models to verify the greedy decoding loop runs without errors.
"""
import torch
import pytest
from torch.utils.data import DataLoader
from src.data.dataset import MathDataset
from src.data.tokenizer import HybridTokenizer


# ---------------------------------------------------------------------------
# Dataset tests  (from dataset.py __main__)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_dataset():
    return MathDataset(mode="toy")


def test_toy_dataset_has_ten_items(toy_dataset):
    assert len(toy_dataset) == 10


def test_toy_dataset_item_image_shape(toy_dataset):
    img, label = toy_dataset[0]
    # (C=3, H=128, W=512)
    assert img.shape == torch.Size([3, 128, 512])


def test_toy_dataset_item_label_nonempty(toy_dataset):
    _, label = toy_dataset[0]
    assert isinstance(label, str) and len(label.strip()) > 0


def test_toy_dataset_image_dtype(toy_dataset):
    """Images must be float32 tensors in [0, 1] after the transform pipeline."""
    img, _ = toy_dataset[0]
    assert img.dtype == torch.float32
    assert img.min().item() >= 0.0
    assert img.max().item() <= 1.0


def test_toy_dataset_all_items_load(toy_dataset):
    """Every item in the toy dataset must load without exception."""
    for i in range(len(toy_dataset)):
        img, label = toy_dataset[i]
        assert img is not None
        assert label is not None


def test_toy_dataset_latex_strings_property(toy_dataset):
    strings = toy_dataset.latex_strings
    assert len(strings) == 10
    assert all(isinstance(s, str) for s in strings)


# ---------------------------------------------------------------------------
# collate_fn + DataLoader tests  (from train.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer_for_pipeline(toy_dataset):
    tok = HybridTokenizer(vocab_size=500)
    tok.train(toy_dataset.latex_strings)
    return tok


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        images    = torch.stack([item[0] for item in batch])
        tokenized = [tokenizer.encode(item[1], add_special=True) for item in batch]
        max_len   = max(len(t) for t in tokenized)
        padded_ids = [
            t + [tokenizer.pad_id] * (max_len - len(t))
            for t in tokenized
        ]
        return images, torch.tensor(padded_ids, dtype=torch.long)
    return collate_fn


def test_collate_fn_image_batch_shape(toy_dataset, tokenizer_for_pipeline):
    collate_fn = make_collate_fn(tokenizer_for_pipeline)
    loader     = DataLoader(toy_dataset, batch_size=4, collate_fn=collate_fn)
    images, _  = next(iter(loader))
    assert images.shape == torch.Size([4, 3, 128, 512])


def test_collate_fn_uniform_sequence_length(toy_dataset, tokenizer_for_pipeline):
    """All sequences in a batch must have the same (padded) length."""
    collate_fn    = make_collate_fn(tokenizer_for_pipeline)
    loader        = DataLoader(toy_dataset, batch_size=4, collate_fn=collate_fn)
    _, token_ids  = next(iter(loader))
    # token_ids must be a 2D tensor with consistent width
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 4


def test_collate_fn_starts_with_sos(toy_dataset, tokenizer_for_pipeline):
    collate_fn   = make_collate_fn(tokenizer_for_pipeline)
    loader       = DataLoader(toy_dataset, batch_size=4, collate_fn=collate_fn)
    _, token_ids = next(iter(loader))
    sos_id       = tokenizer_for_pipeline.sos_id
    for row in token_ids:
        assert row[0].item() == sos_id


def test_collate_fn_pad_id_correct_type(toy_dataset, tokenizer_for_pipeline):
    """Padded token_ids must be int64 (torch.long)."""
    collate_fn   = make_collate_fn(tokenizer_for_pipeline)
    loader       = DataLoader(toy_dataset, batch_size=4, collate_fn=collate_fn)
    _, token_ids = next(iter(loader))
    assert token_ids.dtype == torch.long


# ---------------------------------------------------------------------------
# Inference generate() smoke test  (from inference.py __main__)
# ---------------------------------------------------------------------------

def test_inference_generate_returns_string(tokenizer_for_pipeline):
    """
    Greedy decoding with random-weight models must run without error
    and return a string (content may be garbage — models are untrained).
    """
    from src.models.encoder import VisionEncoder
    from src.models.decoder import LatexDecoder
    from src.inference import generate as infer_generate

    tok     = tokenizer_for_pipeline
    encoder = VisionEncoder()
    decoder = LatexDecoder(vocab_size=tok.total_vocab_size)
    encoder.eval()
    decoder.eval()

    dummy_image = torch.randn(3, 128, 512)
    result = infer_generate(
        dummy_image, encoder, decoder, tok,
        device=torch.device("cpu"), max_len=20
    )
    assert isinstance(result, str)
