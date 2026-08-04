"""
tests/unit/test_latex_lexer.py

Unit tests for src/data/latex_lexer.py.
Converts the __main__ sanity check samples into proper assertions and
adds edge-case coverage.
"""
import pytest
from src.data.latex_lexer import lex, has_unknown_commands, Token, TokenType


# ---------------------------------------------------------------------------
# Token type classification tests
# ---------------------------------------------------------------------------

def test_command_token():
    tokens = lex(r"\frac{1}{2}")
    assert tokens[0].type == TokenType.COMMAND
    assert tokens[0].value == r"\frac"


def test_brace_tokens():
    tokens = lex(r"{x}")
    types = [t.type for t in tokens]
    assert TokenType.OPEN_BRACE  in types
    assert TokenType.CLOSE_BRACE in types


def test_superscript_token():
    tokens = lex(r"x^2")
    types = [t.type for t in tokens]
    assert TokenType.SUPERSCRIPT in types


def test_subscript_token():
    tokens = lex(r"x_i")
    types = [t.type for t in tokens]
    assert TokenType.SUBSCRIPT in types


def test_number_token():
    tokens = lex("42")
    assert len(tokens) == 1
    assert tokens[0].type  == TokenType.NUMBER
    assert tokens[0].value == "42"


def test_operator_token():
    tokens = lex("+")
    assert tokens[0].type == TokenType.OPERATOR


def test_plain_text_segment():
    tokens = lex("Hello")
    assert any(t.type == TokenType.PLAIN_TEXT for t in tokens)


def test_begin_env_token():
    tokens = lex(r"\begin{equation}")
    assert tokens[0].type == TokenType.BEGIN_ENV


def test_end_env_token():
    tokens = lex(r"\end{equation}")
    assert tokens[0].type == TokenType.END_ENV


def test_math_shift_dollar():
    tokens = lex(r"$x$")
    assert tokens[0].type == TokenType.MATH_SHIFT
    assert tokens[0].value == "$"


def test_math_shift_double_dollar():
    tokens = lex(r"$$x$$")
    assert tokens[0].type == TokenType.MATH_SHIFT
    assert tokens[0].value == "$$"


def test_escaped_char():
    tokens = lex(r"\$")
    assert tokens[0].type  == TokenType.ESCAPED_CHAR
    assert tokens[0].value == r"\$"


def test_whitespace_not_emitted():
    """Whitespace should be silently consumed and produce no tokens."""
    tokens_with    = lex(r"\alpha \beta")
    tokens_without = lex(r"\alpha\beta")
    # Both should produce exactly the same non-whitespace token sequence
    assert [t.value for t in tokens_with] == [t.value for t in tokens_without]


def test_complex_expression_token_count():
    """Ensure a realistic expression produces a non-empty token list."""
    tokens = lex(r"\sum_{i=1}^{n} i")
    assert len(tokens) > 0


# ---------------------------------------------------------------------------
# has_unknown_commands tests
# ---------------------------------------------------------------------------

def test_known_commands_return_false():
    assert has_unknown_commands(r"\frac{a}{b}") is False
    assert has_unknown_commands(r"\sum_{i=1}^{n} i") is False
    assert has_unknown_commands(r"\alpha + \beta") is False


def test_unknown_command_returns_true():
    # \nabla is not in KNOWN_COMMANDS
    assert has_unknown_commands(r"\nabla f") is True


def test_no_commands_returns_false():
    """Plain math with no LaTeX commands — no unknown commands."""
    assert has_unknown_commands(r"a + b = c") is False


def test_mixed_known_and_unknown_returns_true():
    """Even one unknown command should trigger True."""
    assert has_unknown_commands(r"\frac{1}{2} + \nabla x") is True
