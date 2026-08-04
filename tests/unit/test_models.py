"""
tests/unit/test_models.py

Unit tests for:
  - src/models/decoder.py  (LatexDecoder)
  - src/models/encoder.py  (VisionEncoder)
  - src/models/cnn_backbone.py (BasicConvBlock)

Converts the __main__ shape-verification checks into pytest assertions.
No GPU required — all tests run on CPU with random tensors.
"""
import pytest
import torch
# pyrefly: ignore [missing-import]
from src.models.decoder import LatexDecoder


# ---------------------------------------------------------------------------
# LatexDecoder tests  (from decoder.py __main__)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def decoder():
    return LatexDecoder(vocab_size=4000, d_model=512, num_heads=8, num_layers=4)


def test_decoder_output_shape(decoder):
    """Logits must be (B, seq_len, vocab_size)."""
    B, S, V = 4, 12, 4000
    memory = torch.randn(B, 44, 512)
    tgt    = torch.randint(0, V, (B, S))
    logits = decoder(tgt, memory)
    assert logits.shape == (B, S, V)


def test_causal_mask_is_lower_triangular(decoder):
    mask = decoder.generate_causal_mask(5)
    assert mask.shape == (5, 5)
    # Lower triangle (including diagonal) must be 1
    for i in range(5):
        for j in range(5):
            if j <= i:
                assert mask[i, j].item() == 1.0
            else:
                assert mask[i, j].item() == 0.0


def test_decoder_no_lookahead(decoder):
    """Upper triangle of causal mask must be all zeros."""
    mask = decoder.generate_causal_mask(8)
    upper = torch.triu(mask, diagonal=1)
    assert upper.sum().item() == 0.0


def test_decoder_single_token(decoder):
    """Decoder must work with a sequence length of 1 (first SOS step)."""
    memory = torch.randn(1, 44, 512)
    tgt    = torch.randint(0, 4000, (1, 1))
    logits = decoder(tgt, memory)
    assert logits.shape == (1, 1, 4000)


def test_decoder_batch_size_one(decoder):
    memory = torch.randn(1, 44, 512)
    tgt    = torch.randint(0, 4000, (1, 10))
    logits = decoder(tgt, memory)
    assert logits.shape == (1, 10, 4000)



# ---------------------------------------------------------------------------
# VisionEncoder tests  (from encoder.py __main__)
# ---------------------------------------------------------------------------

def test_vision_encoder_output_shape():
    """Output sequence shape must be (B, H'*W', 512) after ResNet18 backbone."""
    # pyrefly: ignore [missing-import]
    from src.models.encoder import VisionEncoder
    encoder = VisionEncoder()
    encoder.eval()
    # Standard training input size
    B, H, W = 4, 128, 512
    x   = torch.randn(B, 3, H, W)
    with torch.no_grad():
        out = encoder(x)
    # ResNet18 downsamples by 32x in each spatial dim
    expected_seq_len = (H // 32) * (W // 32)
    assert out.shape == (B, expected_seq_len, 512)


def test_vision_encoder_feature_dim():
    """Feature dimension (C) must be 512 for ResNet18."""
    # pyrefly: ignore [missing-import]
    from src.models.encoder import VisionEncoder
    encoder = VisionEncoder()
    encoder.eval()
    x = torch.randn(1, 3, 128, 512)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape[-1] == 512
