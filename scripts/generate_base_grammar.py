"""
generate_base_grammar.py

Generates src/data/grammar.py from grammar.lark with uniform probability
distributions. Uses the same rule-extraction logic as pCFG_generator.py
but skips all scraped frequency data and Laplace smoothing.

All __ANON_* anonymous terminals are resolved to their actual LaTeX literal
values and inlined directly into the rule expansions, so string_generator.py
never sees any unresolved placeholder tokens.

Usage:
    uv run scripts/generate_base_grammar.py
    uv run scripts/generate_base_grammar.py --lark-file src/data/grammar.lark --out-file src/data/grammar.py
"""

import argparse
from pathlib import Path
from collections import defaultdict
import lark


def main():
    parser = argparse.ArgumentParser(
        description="Generate base grammar.py from grammar.lark with uniform probabilities."
    )
    parser.add_argument("--lark-file", type=str, default="src/data/grammar.lark")
    parser.add_argument("--out-file",  type=str, default="src/data/grammar.py")
    args = parser.parse_args()

    lark_path = Path(args.lark_file)
    if not lark_path.exists():
        print(f"Error: lark grammar file not found at {lark_path}")
        return

    with open(lark_path, "r", encoding="utf-8") as f:
        lark_text = f.read()

    g = lark.Lark(lark_text, parser="earley")

    # Build terminal → literal/pattern maps
    # terminal_to_literal: only exact string matches (type == 'str')
    # terminal_to_pattern: all terminals (fallback for regex terminals)
    terminal_to_literal = {}
    terminal_to_pattern = {}
    for term in g.terminals:
        terminal_to_pattern[term.name] = term.pattern.value
        if term.pattern.type == "str":
            terminal_to_literal[term.name] = term.pattern.value

    # Group rule productions by LHS non-terminal name (uppercased)
    # Each entry is a list of out_rhs lists (already resolved to literals)
    rule_data: dict[str, list[list[str]]] = defaultdict(list)
    non_terminals: set[str] = set()

    for rule in g.rules:
        lhs = rule.origin.name.upper()
        non_terminals.add(lhs)

        out_rhs: list[str] = []
        for sym in rule.expansion:
            if isinstance(sym, lark.grammar.NonTerminal):
                # non-terminals are kept uppercased as references
                out_rhs.append(sym.name.upper())

            elif isinstance(sym, lark.grammar.Terminal):
                if sym.name in terminal_to_literal:
                    # inline the real latex literal directly — this is the
                    # critical fix: __ANON_* tokens become e.g. "^{", "_{", "}"
                    out_rhs.append(terminal_to_literal[sym.name])
                else:
                    # regex terminal (e.g. /[a-zA-Z]/) — keep the pattern
                    # string_generator handles these via their parent non-terminal
                    out_rhs.append(terminal_to_pattern.get(sym.name, f"<{sym.name}>"))

        rule_data[lhs].append(out_rhs)

    # Format output lines
    lines = [
        "# scantex base grammar definition (AUTO-GENERATED)",
        f"# generated from {args.lark_file} with uniform probabilities",
        "# DO NOT EDIT BY HAND — re-run scripts/generate_base_grammar.py",
        "",
        "MAX_DEPTH        = 4",
        "MAX_TERMS        = 3",
        "MAX_FACTORS      = 2",
        "MAX_DIGITS       = 3",
        "MAX_SCRIPT_DEPTH = 2",
        "MAX_FUNC_ARGS    = 1",
        "MAX_MATRIX_ROWS  = 3",
        "MAX_MATRIX_COLS  = 3",
        "",
        "rules = {}",
        "",
    ]

    for lhs, productions in sorted(rule_data.items()):
        n = len(productions)
        prob = round(1.0 / n, 15)
        # make probabilities sum exactly to 1.0
        probs = [prob] * n

        lines.append(f"rules['{lhs}'] = (")
        lines.append("    [")
        for rhs in productions:
            rhs_repr = ", ".join(repr(s) for s in rhs)
            lines.append(f"        [{rhs_repr}],")
        lines.append("    ],")
        probs_repr = ", ".join(str(p) for p in probs)
        lines.append(f"    [{probs_repr}]")
        lines.append(")")
        lines.append("")

    # Write the non_terminals set
    lines.append("non_terminals = {")
    for nt in sorted(non_terminals):
        lines.append(f"    '{nt}',")
    lines.append("}")
    lines.append("")

    # Write terminals as empty set — all literals are already inlined in rules
    lines.append("terminals = set()")
    lines.append("")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated base grammar with {len(non_terminals)} non-terminals.")
    print(f"  All __ANON_* terminals resolved to real LaTeX literals.")
    print(f"  Uniform probability: 1/n per production.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
