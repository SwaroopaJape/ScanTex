import pytest
from pathlib import Path
# pyrefly: ignore [missing-import]
from src.data import string_generator as sg

@pytest.fixture(scope="module")
def setup_grammar():
    project_root = Path(__file__).resolve().parents[2]
    grammar_path = project_root / "src" / "data" / "grammar.py"
    sg.load_grammar(str(grammar_path))

def test_generator_respects_max_depth(setup_grammar):
    """
    Test that the geometric depth feature respects explicitly passed max_depth overrides,
    preventing infinite generation when forced.
    """
    # Force a very shallow max depth
    result = sg.generate(depth=0, current_max_depth=1)
    assert isinstance(result, str)
    assert len(result) > 0
