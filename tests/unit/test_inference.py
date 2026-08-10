import torch
import pytest
# pyrefly: ignore [missing-import]
from src.inference import generate, beam_search_generate
# pyrefly: ignore [missing-import]
from src.models.encoder import VisionEncoder
# pyrefly: ignore [missing-import]
from src.models.decoder import LatexDecoder
# pyrefly: ignore [missing-import]
from src.data.tokenizer import HybridTokenizer

@pytest.fixture
def mock_tokenizer():
    tokenizer = HybridTokenizer(vocab_size=100)
    # Mock training so the tokenizer has SOS/EOS and some basic vocab
    tokenizer.train(["a + b = c"])
    return tokenizer

@pytest.fixture
def mock_models(mock_tokenizer):
    encoder = VisionEncoder()
    decoder = LatexDecoder(vocab_size=mock_tokenizer.total_vocab_size)
    return encoder, decoder

def test_generate_greedy(mock_models, mock_tokenizer):
    encoder, decoder = mock_models
    device = torch.device("cpu")
    # Dummy image tensor (C, H, W)
    image_tensor = torch.randn(3, 128, 512)
    
    result = generate(image_tensor, encoder, decoder, mock_tokenizer, device, max_len=10, algorithm="greedy")
    assert isinstance(result, str)

def test_generate_beam(mock_models, mock_tokenizer):
    encoder, decoder = mock_models
    device = torch.device("cpu")
    image_tensor = torch.randn(3, 128, 512)
    
    result = generate(image_tensor, encoder, decoder, mock_tokenizer, device, max_len=10, algorithm="beam", beam_width=3)
    assert isinstance(result, str)
