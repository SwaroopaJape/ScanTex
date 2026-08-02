import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

from lark import Lark, Tree, Token


GRAMMAR_PATH = "src/data/grammar.lark"
OUTPUT_DIR   = Path("data/scraped_info")

# recursively walk a lark tree, counting every rule-branch that was used
def walk_tree(tree: Tree, counts: dict):
    if not isinstance(tree, Tree):
        return

    # build a compact signature for the children of this node
    # e.g. frac -> ["frac{", expr, "}{", expr, "}"]  becomes "frac"
    # we use the rule name of each child tree, or the token value for terminals
    child_sig = []
    for child in tree.children:
        if isinstance(child, Tree):
            child_sig.append(child.data)
        elif isinstance(child, Token):
            child_sig.append(repr(child))

    key = f"{tree.data} -> {' '.join(child_sig)}"
    counts[key] += 1

    for child in tree.children:
        walk_tree(child, counts)


def main():
    parser = argparse.ArgumentParser(description="parse corpus and count pcfg rule frequencies")
    parser.add_argument("input_file", type=str, nargs="?", default="data/scraped/train_equations.txt", help="path to the equations text file")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}")
        sys.exit(1)

    # load lark grammar
    print(f"loading lark grammar from {GRAMMAR_PATH}...")
    with open(GRAMMAR_PATH, "r") as f:
        grammar_text = f.read()
    lark_parser = Lark(grammar_text, parser="earley", ambiguity="resolve")

    # read all equations
    equations = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(equations)
    print(f"loaded {total} equations from {input_path}")

    # track counts
    rule_counts: dict = defaultdict(int)
    parsed_ok  = 0
    failed     = 0
    skipped    = 0
    import tqdm
    
    # pyrefly: ignore [missing-import]
    from src.data.latex_lexer import has_unknown_commands

    pbar = tqdm.tqdm(equations, desc="Parsing", unit="eq")
    for eq in pbar:
        if has_unknown_commands(eq):
            skipped += 1
        else:
            try:
                tree = lark_parser.parse(eq)
                walk_tree(tree, rule_counts)
                parsed_ok += 1
            except Exception:
                failed += 1
                
        pbar.set_postfix(ok=parsed_ok, fail=failed, skip=skipped)

    print(f"\ndone. parsed {parsed_ok} successfully. skipped {skipped} unknown macro equations. failed {failed} syntax errors.")

    # save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "rule_frequencies.json"

    # sort by frequency descending for easy reading
    sorted_counts = dict(sorted(rule_counts.items(), key=lambda x: x[1], reverse=True))

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sorted_counts, f, indent=2, ensure_ascii=False)

    print(f"rule frequencies saved to {out_file}")
    print(f"unique rules seen: {len(sorted_counts)}")


if __name__ == "__main__":
    main()
