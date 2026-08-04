"""
tests/integration/test_grammar_pipeline.py

Integration tests for the full grammar pipeline:
  grammar.lark → generate_base_grammar.py → src/data/grammar.py
                                           → string_generator.py → LaTeX strings

Also tests weighted_grammar.py produced by pCFG_generator.py.
"""
import pytest
from src.data import string_generator as sg


# ---------------------------------------------------------------------------
# Base grammar (uniform probabilities)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_grammar_loaded(base_grammar_path):
    sg.load_grammar(base_grammar_path)
    return base_grammar_path


@pytest.fixture(scope="module")
def weighted_grammar_loaded(weighted_grammar_path):
    sg.load_grammar(weighted_grammar_path)
    return weighted_grammar_path


# --- generation correctness ---

def test_base_grammar_generates_nonempty_string(base_grammar_loaded):
    result = sg.generate()
    assert isinstance(result, str) and len(result.strip()) > 0


def test_base_grammar_no_anon_tokens_in_output(base_grammar_loaded):
    """
    The core regression guard: if __ANON_* tokens appear in generated
    strings, grammar.py has the corruption bug.
    """
    for _ in range(50):
        result = sg.generate()
        assert "__ANON_" not in result, (
            f"Generated string contains __ANON_* token: {result!r}. "
            "Re-run scripts/generate_base_grammar.py to fix grammar.py."
        )


def test_base_grammar_no_angle_bracket_terminals(base_grammar_loaded):
    """Unresolved <TERMINAL_NAME> placeholders must never appear in output."""
    import re
    pattern = re.compile(r"<[A-Z_]+>")
    for _ in range(50):
        result = sg.generate()
        assert not pattern.search(result), (
            f"Generated string contains unresolved terminal: {result!r}"
        )


def test_base_grammar_compilation_ratio(base_grammar_loaded):
    """
    Base grammar must achieve at least 10% matplotlib compilation success.
    A 0% ratio is the symptom of the __ANON_* bug.
    """
    n_samples = 100
    successes = sum(
        1 for _ in range(n_samples) if sg.check_compiles(sg.generate())
    )
    ratio = successes / n_samples
    assert ratio >= 0.10, (
        f"Base grammar compilation ratio is {ratio:.0%} (< 10%). "
        "Grammar is likely broken — check for __ANON_* tokens."
    )


# --- weighted grammar ---

def test_weighted_grammar_generates_nonempty_string(weighted_grammar_loaded):
    result = sg.generate()
    assert isinstance(result, str) and len(result.strip()) > 0


def test_weighted_grammar_no_anon_tokens(weighted_grammar_loaded):
    for _ in range(50):
        result = sg.generate()
        assert "__ANON_" not in result, (
            f"Weighted grammar output contains __ANON_* token: {result!r}"
        )


def test_weighted_grammar_compilation_ratio(weighted_grammar_loaded):
    """
    The PCFG-weighted grammar (trained on scraped data) must achieve
    at least 40% compilation success — we observed 62% in practice.
    """
    n_samples = 100
    successes = sum(
        1 for _ in range(n_samples) if sg.check_compiles(sg.generate())
    )
    ratio = successes / n_samples
    assert ratio >= 0.40, (
        f"Weighted grammar compilation ratio is {ratio:.0%} (< 40%). "
        "The PCFG weights may be stale or grammar.lark has changed."
    )


def test_weighted_grammar_better_than_base(base_grammar_path, weighted_grammar_path):
    """
    Weighted grammar must outperform base grammar on compilation ratio.
    This validates that the PCFG frequency-weighting adds real value.
    """
    n_samples = 100

    sg.load_grammar(base_grammar_path)
    base_successes = sum(
        1 for _ in range(n_samples) if sg.check_compiles(sg.generate())
    )

    sg.load_grammar(weighted_grammar_path)
    weighted_successes = sum(
        1 for _ in range(n_samples) if sg.check_compiles(sg.generate())
    )

    assert weighted_successes >= base_successes, (
        f"Weighted grammar ({weighted_successes/n_samples:.0%}) performed worse "
        f"than base grammar ({base_successes/n_samples:.0%}). "
        "Check that pCFG_generator.py is using correct frequency data."
    )
