"""
conftest.py — shared pytest fixtures for ScanTex tests.

Fixtures defined here are automatically available to all tests.
"""
import sys
import pytest
from pathlib import Path

# Make sure the project root is on sys.path for all tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_GRAMMAR_PATH     = str(PROJECT_ROOT / "src" / "data" / "grammar.py")
WEIGHTED_GRAMMAR_PATH = str(PROJECT_ROOT / "data" / "extracted_grammar" / "weighted_grammar.py")
LARK_GRAMMAR_PATH     = str(PROJECT_ROOT / "src" / "data" / "grammar.lark")
TOKENIZER_WEIGHTS_DIR = str(PROJECT_ROOT / "data" / "tokenizer_weights")


@pytest.fixture(scope="session")
def base_grammar_path():
    return BASE_GRAMMAR_PATH


@pytest.fixture(scope="session")
def weighted_grammar_path():
    return WEIGHTED_GRAMMAR_PATH


@pytest.fixture(scope="session")
def lark_grammar_path():
    return LARK_GRAMMAR_PATH


@pytest.fixture(scope="session")
def tokenizer_weights_dir():
    return TOKENIZER_WEIGHTS_DIR


@pytest.fixture(scope="session")
def tiny_corpus():
    """A small fixed corpus of LaTeX strings used for tokenizer training in tests."""
    return [
        r"a^2 + b^2 = c^2",
        r"E = mc^2",
        r"\int_{a}^{b} x^2 dx",
        r"\sum_{i=1}^{n} i",
        r"\frac{1}{2} m v^2",
        r"\sin^2(x) + \cos^2(x) = 1",
        r"e^{i\pi} + 1 = 0",
        r"F = G \frac{m_1 m_2}{r^2}",
        r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
    ]


@pytest.fixture(scope="session")
def trained_tokenizer(tiny_corpus):
    """A HybridTokenizer trained on the tiny corpus, reused across all tests."""
    # pyrefly: ignore [missing-import]
    from src.data.tokenizer import HybridTokenizer
    tok = HybridTokenizer(vocab_size=500)
    tok.train(tiny_corpus)
    return tok
