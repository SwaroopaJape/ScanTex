# scantex cfg grammar definition
# a comprehensive pcfg for latex math expressions

# global depth / length caps – used by the generator to bound recursion
MAX_DEPTH       = 4      # max recursive expansion depth
MAX_TERMS       = 3      # max additive terms   (for plus/star helpers)
MAX_FACTORS     = 2      # max multiplicative factors
MAX_DIGITS      = 3      # max digits in a number literal
MAX_SCRIPT_DEPTH = 2     # max depth inside super/sub-scripts
MAX_FUNC_ARGS   = 1      # max chained function arguments
MAX_MATRIX_ROWS = 3      # max rows in a matrix
MAX_MATRIX_COLS = 3      # max cols in a matrix

# helper: build uniform probabilities
def _uniform(n: int) -> list[float]:
    # return a list of n equal probabilities summing to 1
    return [1.0 / n] * n

# grammar rules
rules: dict[str, tuple[list[list[str]], list[float]]] = {}

# start symbol
rules["START"] = (
    [
        ["EXPR"],
        ["RELATION"],
        ["SET_EXPR"],
        ["LOGIC_EXPR"],
    ],
    _uniform(4),
)

# relations
rules["RELATION"] = (
    [
        ["EXPR", " ", "REL_OP", " ", "EXPR"],
        ["EXPR", " ", "REL_OP", " ", "EXPR", " ", "REL_OP", " ", "EXPR"],
    ],
    _uniform(2),
)

rules["REL_OP"] = (
    [
        ["="],
        ["<"],
        [">"],
        [r"\leq "],
        [r"\geq "],
        [r"\neq "],
        [r"\approx "],
        [r"\equiv "],
        [r"\in "],
        [r"\subset "],
        [r"\subseteq "],
        [r"\supseteq "],
        [r"\sim "],
        [r"\propto "],
        [r"\notin "],
        [r"\supset "],
        [r"\cong "],
        [r"\perp "],
        [r"\parallel "],
    ],
    _uniform(19),
)

# expressions (additive layer)
rules["EXPR"] = (
    [
        ["TERM"],
        ["TERM", " + ", "EXPR"],
        ["TERM", " - ", "EXPR"],
        ["UNARY_OP", "TERM"],
    ],
    _uniform(4),
)

rules["UNARY_OP"] = (
    [
        ["-"],
        ["+"],
    ],
    _uniform(2),
)

# terms (multiplicative layer)
rules["TERM"] = (
    [
        ["FACTOR"],
        ["FACTOR", r" \cdot ", "TERM"],
        ["FACTOR", r" \times ", "TERM"],
        ["FACTOR", " ", "FACTOR"],          # implicit multiplication  e.g. 2x
    ],
    _uniform(4),
)

# factors
rules["FACTOR"] = (
    [
        ["ATOM"],
        ["SCRIPTED"],
        ["FRAC"],
        ["SQRT"],
        ["FUNC_CALL"],
        ["BIG_OP"],
        ["DELIMITED"],
        ["ACCENT"],
        ["BINOM"],
        ["PARTIAL_DERIV"],
        ["TEXT_COND"],
    ],
    _uniform(11),
)

# atoms (leaf values)
rules["ATOM"] = (
    [
        ["VARIABLE"],
        ["NUMBER"],
        ["GREEK"],
        ["CONSTANT"],
        ["STYLED_ATOM"],
    ],
    _uniform(5),
)

# variables
_lower = list("abcdefghijklmnopqrstuvwxyz")
_upper = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_all_vars = _lower + _upper

rules["VARIABLE"] = (
    [[v] for v in _all_vars],
    _uniform(len(_all_vars)),
)

# digits / numbers
rules["DIGIT"] = (
    [[d] for d in "0123456789"],
    _uniform(10),
)

# number is handled by the generator via plus(digit, max_digits)
# we still define a single-digit base case for grammar completeness
rules["NUMBER"] = (
    [
        ["DIGIT"],
    ],
    _uniform(1),
)

# greek letters
_greek_lower = [
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon",
    r"\zeta", r"\eta", r"\theta", r"\vartheta", r"\iota", r"\kappa",
    r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma",
    r"\tau", r"\upsilon", r"\phi", r"\varphi", r"\chi", r"\psi", r"\omega",
]
_greek_upper = [
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi", r"\Pi",
    r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",
]
_all_greek = _greek_lower + _greek_upper

rules["GREEK"] = (
    [[g] for g in _all_greek],
    _uniform(len(_all_greek)),
)

# constants
rules["CONSTANT"] = (
    [
        [r"\infty"],
        [r"\pi"],
        ["e"],
        ["0"],
        ["1"],
    ],
    _uniform(5),
)

# superscripts / subscripts
rules["SCRIPTED"] = (
    [
        ["SCRIPTABLE", "^{", "EXPR", "}"],                           # x^{2}
        ["SCRIPTABLE", "_{", "EXPR", "}"],                           # x_{i}
        ["SCRIPTABLE", "_{", "EXPR", "}^{", "EXPR", "}"],           # x_{i}^{2}
    ],
    _uniform(3),
)

rules["SCRIPTABLE"] = (
    [
        ["VARIABLE"],
        ["GREEK"],
        ["DELIMITED"],
        ["FUNC_CALL"],
    ],
    _uniform(4),
)

# fractions
rules["FRAC"] = (
    [
        [r"\frac{", "EXPR", "}{", "EXPR", "}"],
        [r"\dfrac{", "EXPR", "}{", "EXPR", "}"],
    ],
    _uniform(2),
)

# square roots
rules["SQRT"] = (
    [
        [r"\sqrt{", "EXPR", "}"],
        [r"\sqrt[", "NUMBER", "]{", "EXPR", "}"],
    ],
    _uniform(2),
)

# function calls
rules["FUNC_CALL"] = (
    [
        ["FUNC_NAME", r"\left(", "EXPR", r"\right)"],
        ["FUNC_NAME", "{", "ATOM", "}"],
        ["FUNC_NAME", " ", "ATOM"],
    ],
    _uniform(3),
)

rules["FUNC_NAME"] = (
    [
        [r"\sin"],
        [r"\cos"],
        [r"\tan"],
        [r"\cot"],
        [r"\sec"],
        [r"\csc"],
        [r"\arcsin"],
        [r"\arccos"],
        [r"\arctan"],
        [r"\sinh"],
        [r"\cosh"],
        [r"\tanh"],
        [r"\log"],
        [r"\ln"],
        [r"\exp"],
        [r"\det"],
        [r"\max"],
        [r"\min"],
    ],
    _uniform(18),
)

# big operators (summation, product, integral, limit)
rules["BIG_OP"] = (
    [
        ["SUMMATION"],
        ["PRODUCT"],
        ["INTEGRAL"],
        ["LIMIT"],
    ],
    _uniform(4),
)

rules["SUMMATION"] = (
    [
        [r"\sum_{", "SUB_ASSIGN", "}^{", "EXPR", "} ", "EXPR"],
        [r"\sum_{", "SUB_ASSIGN", "} ", "EXPR"],
        [r"\sum ", "EXPR"],
    ],
    _uniform(3),
)

rules["PRODUCT"] = (
    [
        [r"\prod_{", "SUB_ASSIGN", "}^{", "EXPR", "} ", "EXPR"],
        [r"\prod_{", "SUB_ASSIGN", "} ", "EXPR"],
        [r"\prod ", "EXPR"],
    ],
    _uniform(3),
)

rules["INTEGRAL"] = (
    [
        [r"\int_{", "EXPR", "}^{", "EXPR", "} ", "EXPR", r"\, d", "VARIABLE"],
        [r"\int ", "EXPR", r"\, d", "VARIABLE"],
        [r"\iint ", "EXPR", r"\, d", "VARIABLE", r"\, d", "VARIABLE"],
        [r"\oint_{", "EXPR", "} ", "EXPR", r"\, d", "VARIABLE"],
    ],
    _uniform(4),
)

rules["LIMIT"] = (
    [
        [r"\lim_{", "VARIABLE", r" \to ", "EXPR", "} ", "EXPR"],
        [r"\lim_{", "VARIABLE", r" \to ", "CONSTANT", "} ", "EXPR"],
    ],
    _uniform(2),
)

# subscript assignment  (e.g.  i = 0)
rules["SUB_ASSIGN"] = (
    [
        ["VARIABLE", " = ", "EXPR"],
        ["VARIABLE", " = ", "NUMBER"],
    ],
    _uniform(2),
)

# delimiters (parentheses, brackets, braces, abs)
rules["DELIMITED"] = (
    [
        [r"\left(", "EXPR", r"\right)"],
        [r"\left[", "EXPR", r"\right]"],
        [r"\left|", "EXPR", r"\right|"],
        [r"\left\{", "EXPR", r"\right\}"],
        ["(", "EXPR", ")"],
    ],
    _uniform(5),
)

# accents
rules["ACCENT"] = (
    [
        [r"\hat{", "ATOM", "}"],
        [r"\bar{", "ATOM", "}"],
        [r"\tilde{", "ATOM", "}"],
        [r"\vec{", "ATOM", "}"],
        [r"\dot{", "ATOM", "}"],
        [r"\ddot{", "ATOM", "}"],
        [r"\overline{", "EXPR", "}"],
        [r"\underline{", "EXPR", "}"],
        [r"\widehat{", "EXPR", "}"],
        [r"\widetilde{", "EXPR", "}"],
    ],
    _uniform(10),
)

# binomial coefficients
rules["BINOM"] = (
    [
        [r"\binom{", "EXPR", "}{", "EXPR", "}"],
    ],
    _uniform(1),
)

# styled atoms (fonts)
rules["STYLED_ATOM"] = (
    [
        ["MATHBB"],
        ["MATHCAL"],
        ["MATHBF"],
        ["MATHRM"],
    ],
    _uniform(4),
)

_mathbb_letters = ["R", "N", "Z", "Q", "C", "F", "P", "E"]
rules["MATHBB"] = (
    [[r"\mathbb{", l, "}"] for l in _mathbb_letters],
    _uniform(len(_mathbb_letters)),
)

_mathcal_letters = ["O", "L", "F", "H", "A", "B", "C", "D", "M", "N", "P", "S"]
rules["MATHCAL"] = (
    [[r"\mathcal{", l, "}"] for l in _mathcal_letters],
    _uniform(len(_mathcal_letters)),
)

rules["MATHBF"] = (
    [
        [r"\mathbf{", "VARIABLE", "}"],
        [r"\mathbf{", "GREEK", "}"],
    ],
    _uniform(2),
)

rules["MATHRM"] = (
    [
        [r"\mathrm{", "VARIABLE", "}"],
        [r"\mathrm{d}"],
        [r"\mathrm{e}"],
    ],
    _uniform(3),
)

# set theory expressions
rules["SET_EXPR"] = (
    [
        ["EXPR", " ", "SET_OP", " ", "EXPR"],
        ["EXPR", r" \in ", "STYLED_ATOM"],
        [r"\left\{", "EXPR", r" \mid ", "EXPR", r"\right\}"],
        ["STYLED_ATOM", " ", "SET_OP", " ", "STYLED_ATOM"],
    ],
    _uniform(4),
)

rules["SET_OP"] = (
    [
        [r"\cup "],
        [r"\cap "],
        [r"\setminus "],
        [r"\bigcup "],
        [r"\bigcap "],
    ],
    _uniform(5),
)

# logic expressions
rules["LOGIC_EXPR"] = (
    [
        [r"\forall ", "VARIABLE", r" \in ", "STYLED_ATOM", ", ", "EXPR"],
        [r"\exists ", "VARIABLE", r" \in ", "STYLED_ATOM", ": ", "EXPR"],
        ["EXPR", r" \Rightarrow ", "EXPR"],
        ["EXPR", r" \iff ", "EXPR"],
        ["EXPR", r" \Leftrightarrow ", "EXPR"],
        [r"\neg ", "DELIMITED"],
        ["EXPR", r" \land ", "EXPR"],
        ["EXPR", r" \lor ", "EXPR"],
    ],
    _uniform(8),
)

# partial derivatives
rules["PARTIAL_DERIV"] = (
    [
        [r"\frac{\partial ", "EXPR", r"}{\partial ", "VARIABLE", "}"],
        [r"\frac{\partial^2 ", "EXPR", r"}{\partial ", "VARIABLE", "^2}"],
        [r"\frac{\partial^2 ", "EXPR", r"}{\partial ", "VARIABLE", r" \partial ", "VARIABLE", "}"],
    ],
    _uniform(3),
)

# text conditions
rules["TEXT_COND"] = (
    [
        ["EXPR", r" \text{ if } ", "EXPR", " ", "REL_OP", " ", "EXPR"],
        ["EXPR", r" \text{ for } ", "VARIABLE", r" \in ", "STYLED_ATOM"],
        ["EXPR", r" \text{ where } ", "RELATION"],
        ["EXPR", r" \text{ and } ", "EXPR"],
    ],
    _uniform(4),
)

# derived sets
non_terminals: set[str] = set(rules.keys())

def _collect_terminals() -> set[str]:
    # walk every production and collect symbols that are not non-terminals
    terms: set[str] = set()
    for prods, _ in rules.values():
        for prod in prods:
            for sym in prod:
                if sym not in non_terminals:
                    terms.add(sym)
    return terms

terminals: set[str] = _collect_terminals()

# convenience printer
if __name__ == "__main__":
    print("=" * 60)
    print("ScanTeX CFG Grammar Summary")
    print("=" * 60)
    print(f"  Non-terminals : {len(non_terminals)}")
    print(f"  Terminals     : {len(terminals)}")
    total_prods = sum(len(p) for p, _ in rules.values())
    print(f"  Total prods   : {total_prods}")
    print()
    for nt, (prods, probs) in rules.items():
        print(f"  {nt}  ({len(prods)} productions)")
        for prod, prob in zip(prods, probs):
            rhs = "  ".join(prod)
            print(f"      p={prob:.3f}  →  {rhs}")
    print()
    print("Global parameters:")
    print(f"  MAX_DEPTH        = {MAX_DEPTH}")
    print(f"  MAX_TERMS        = {MAX_TERMS}")
    print(f"  MAX_FACTORS      = {MAX_FACTORS}")
    print(f"  MAX_DIGITS       = {MAX_DIGITS}")
    print(f"  MAX_SCRIPT_DEPTH = {MAX_SCRIPT_DEPTH}")
    print(f"  MAX_FUNC_ARGS    = {MAX_FUNC_ARGS}")
    print(f"  MAX_MATRIX_ROWS  = {MAX_MATRIX_ROWS}")
    print(f"  MAX_MATRIX_COLS  = {MAX_MATRIX_COLS}")
