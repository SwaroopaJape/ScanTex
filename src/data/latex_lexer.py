"""
LaTeX-aware lexer.

Single left-to-right scan that classifies every character span as either
a recognized LaTeX token or a plain-text segment. The plain-text segments
are what get forwarded to BPE later.
"""

from enum import Enum, auto
from dataclasses import dataclass

class TokenType(Enum):
    # LaTeX structural tokens
    COMMAND      = auto()  # \alpha, \frac, \sqrt
    ESCAPED_CHAR = auto()  # \$, \%, \_, \{, \}, \#, \&, \\
    BEGIN_ENV     = auto()  # \begin{equation}
    END_ENV       = auto()  # \end{...}
    MATH_SHIFT   = auto()  # $ or $$
    MATH_OPEN    = auto()  # \( or \[
    MATH_CLOSE   = auto()  # \) or \]

    # Syntax / grouping
    OPEN_BRACE    = auto()  # {
    CLOSE_BRACE   = auto()  # }
    OPEN_BRACKET  = auto()  # [
    CLOSE_BRACKET = auto()  # ]
    OPEN_PAREN    = auto()  # (
    CLOSE_PAREN   = auto()  # )
    SUPERSCRIPT   = auto()  # ^
    SUBSCRIPT     = auto()  # _
    ALIGNMENT     = auto()  # &

    # Literals
    NUMBER   = auto()  # 42, 3.14
    OPERATOR = auto()  # + - = < > * / ! , ; : | .

    # Fallback
    PLAIN_TEXT = auto()  # anything not LaTeX → forwarded to BPE

# characters that are emitted as single-char syntax tokens
_SINGLE_CHAR_MAP = {
    "{": TokenType.OPEN_BRACE,
    "}": TokenType.CLOSE_BRACE,
    "[": TokenType.OPEN_BRACKET,
    "]": TokenType.CLOSE_BRACKET,
    "(": TokenType.OPEN_PAREN,
    ")": TokenType.CLOSE_PAREN,
    "^": TokenType.SUPERSCRIPT,
    "_": TokenType.SUBSCRIPT,
    "&": TokenType.ALIGNMENT,
}

# escapable single characters after backslash
_ESCAPABLE = set("$%_{}#&\\!,;: ")

# operator characters
_OPERATORS = set("+-=<>*/!',;:|.")

@dataclass
class Token:
    type: TokenType
    value: str

def lex(source: str) -> list[Token]:
    """Tokenize a LaTeX string into a list of Tokens."""
    tokens: list[Token] = []
    plain_buf: list[str] = []  # accumulator for plain-text runs
    i = 0
    n = len(source)

    def flush_plain():
        """Emit accumulated plain text as a single PLAIN_TEXT token."""
        if plain_buf:
            tokens.append(Token(TokenType.PLAIN_TEXT, "".join(plain_buf)))
            plain_buf.clear()

    while i < n:
        ch = source[i]

        # backslash-initiated tokens
        if ch == "\\":
            flush_plain()

            # end of string after backslash → malformed, emit as plain
            if i + 1 >= n:
                plain_buf.append("\\")
                i += 1
                continue

            nxt = source[i + 1]

            # math delimiters  \(  \)  \[  \]
            if nxt in "()[]":
                ttype = TokenType.MATH_OPEN if nxt in "(["  else TokenType.MATH_CLOSE
                tokens.append(Token(ttype, source[i:i+2]))
                i += 2

            # escaped single char  \$  \%  \_  \{  \}  \#  \&  \\  etc.
            elif nxt in _ESCAPABLE:
                tokens.append(Token(TokenType.ESCAPED_CHAR, source[i:i+2]))
                i += 2

            # command  \alpha  \frac  \begin  \end  ...
            elif nxt.isalpha():
                j = i + 2
                while j < n and source[j].isalpha():
                    j += 1
                cmd = source[i:j]

                # \begin{env} / \end{env}  →  consume the {env} too
                if cmd in ("\\begin", "\\end"):
                    ttype = TokenType.BEGIN_ENV if cmd == "\\begin" else TokenType.END_ENV
                    # look for optional {envname}
                    if j < n and source[j] == "{":
                        k = source.find("}", j)
                        if k != -1:
                            tokens.append(Token(ttype, source[i:k+1]))
                            i = k + 1
                        else:
                            # malformed: no closing brace
                            tokens.append(Token(ttype, cmd))
                            i = j
                    else:
                        tokens.append(Token(ttype, cmd))
                        i = j
                else:
                    tokens.append(Token(TokenType.COMMAND, cmd))
                    i = j

            # unknown escape → treat backslash as plain text
            else:
                plain_buf.append("\\")
                i += 1

        # math shift  $ / $$
        elif ch == "$":
            flush_plain()
            if i + 1 < n and source[i + 1] == "$":
                tokens.append(Token(TokenType.MATH_SHIFT, "$$"))
                i += 2
            else:
                tokens.append(Token(TokenType.MATH_SHIFT, "$"))
                i += 1

        # single-char syntax tokens
        elif ch in _SINGLE_CHAR_MAP:
            flush_plain()
            tokens.append(Token(_SINGLE_CHAR_MAP[ch], ch))
            i += 1

        # numbers (int or decimal)
        elif ch.isdigit():
            flush_plain()
            j = i
            while j < n and (source[j].isdigit() or source[j] == "."):
                j += 1
            tokens.append(Token(TokenType.NUMBER, source[i:j]))
            i = j

        # operators
        elif ch in _OPERATORS:
            flush_plain()
            tokens.append(Token(TokenType.OPERATOR, ch))
            i += 1

        # whitespace → skip (don't accumulate into plain text)
        elif ch in " \t\n\r":
            flush_plain()
            i += 1

        # everything else → plain text (will go to BPE)
        else:
            plain_buf.append(ch)
            i += 1

    flush_plain()
    return tokens

if __name__ == "__main__":
    samples = [
        r"\frac{1}{2} m v^2",
        r"\sum_{i=1}^{n} i",
        r"e^{i\pi} + 1 = 0",
        r"\begin{equation} a + b \end{equation}",
        r"Hello world \alpha \$100",
    ]
    for s in samples:
        print(f"\nInput: {s}")
        for tok in lex(s):
            print(f"  {tok.type.name:15s}  │ {tok.value!r}")
