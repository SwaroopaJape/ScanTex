# scantex string generator
# generates random latex math strings from the cfg defined in grammar.py
# provides compile-checking via matplotlib mathtext renderer
# star and plus functions handle repetition
# global depth counter prevents infinite recursion

import random
import textwrap
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO


import argparse
import sys
import importlib.util

# global grammar state (will be populated by load_grammar)
rules = {}
non_terminals = set()
terminals = set()
MAX_DEPTH = 4
MAX_TERMS = 3
MAX_FACTORS = 2
MAX_DIGITS = 3

def load_grammar(file_path: str):
    # dynamically load the grammar file and update globals
    global rules, non_terminals, terminals, MAX_DEPTH, MAX_TERMS, MAX_FACTORS, MAX_DIGITS
    
    spec = importlib.util.spec_from_file_location("dynamic_grammar", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load grammar from {file_path}")
    
    grammar_module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_grammar"] = grammar_module
    spec.loader.exec_module(grammar_module)
    
    rules = grammar_module.rules
    non_terminals = grammar_module.non_terminals
    terminals = grammar_module.terminals
    MAX_DEPTH = grammar_module.MAX_DEPTH
    MAX_TERMS = grammar_module.MAX_TERMS
    MAX_FACTORS = grammar_module.MAX_FACTORS
    MAX_DIGITS = grammar_module.MAX_DIGITS

# load the default grammar initially so imports of this module still work
load_grammar("src/data/grammar.py")

# lark parser (optional, only loaded when explicitly requested)
_lark_parser = None

def load_lark_parser(lark_file_path: str = "src/data/grammar.lark"):
    # load the lark grammar and build an earley parser
    # this is used for parsing scraped equations, not for generation
    global _lark_parser
    from lark import Lark
    with open(lark_file_path, "r") as f:
        grammar_text = f.read()
    _lark_parser = Lark(grammar_text, parser="earley", ambiguity="resolve")
    return _lark_parser

def parse_with_lark(latex: str):
    # parse a raw latex string and return a lark tree, or None if it fails
    # load the parser on first use if not already loaded
    global _lark_parser
    if _lark_parser is None:
        load_lark_parser()
    try:
        return _lark_parser.parse(latex)
    except Exception:
        return None

# use non-interactive backend so matplotlib never opens a window
matplotlib.use("Agg")

# repetition helpers (a* and a+ operations)

def star(gen_func, max_n: int) -> list[str]:
    # zero-or-more: call gen_func between 0 and max_n times
    # returns a list of generated strings
    n = random.randint(0, max_n)
    return [gen_func() for _ in range(n)]


def plus(gen_func, max_n: int) -> list[str]:
    # one-or-more: call gen_func between 1 and max_n times
    # returns a list of at least one generated string
    n = random.randint(1, max_n)
    return [gen_func() for _ in range(n)]


# depth-collapse: non-terminals that must become a simple atom at max depth

# these non-terminals form the recursive spine of the grammar
# when depth >= MAX_DEPTH they are forcibly collapsed to a bare atom
_RECURSIVE_SPINE = {
    "START", "RELATION", "EXPR", "TERM", "FACTOR",
    "SCRIPTED", "FRAC", "SQRT", "FUNC_CALL", "BIG_OP",
    "SUMMATION", "PRODUCT", "INTEGRAL", "LIMIT",
    "DELIMITED", "ACCENT", "BINOM", "SUB_ASSIGN",
    "SET_EXPR", "LOGIC_EXPR", "PARTIAL_DERIV", "TEXT_COND",
}

# only these symbols represent true structural nesting
# we only increment the depth counter when expanding these
_STRUCTURAL_SYMBOLS = {
    "SCRIPTED", "FRAC", "SQRT", "FUNC_CALL", "BIG_OP", 
    "SUMMATION", "PRODUCT", "INTEGRAL", "LIMIT", 
    "DELIMITED", "ACCENT", "BINOM",
    "SET_EXPR", "LOGIC_EXPR", "PARTIAL_DERIV", "TEXT_COND",
}

def _generate_atom(depth: int, current_max_depth: int) -> str:
    # generate a guaranteed-finite atom (variable, number, greek, constant)
    productions, probabilities = rules["ATOM"]
    chosen = random.choices(productions, weights=probabilities, k=1)[0]
    return "".join(generate(s, depth, current_max_depth) for s in chosen)


# core recursive generator

def generate(symbol: str = "START", depth: int = 0, current_max_depth: int = -1) -> str:
    # expand a grammar symbol into a concrete latex string
    
    # Initialize geometric max depth on the first call
    if current_max_depth == -1:
        current_max_depth = MAX_DEPTH
        p = 0.1  # 20% chance to multiply depth
        while random.random() < p:
            current_max_depth += MAX_DEPTH
            
    # terminal: return as-is
    if symbol not in non_terminals:
        return symbol

    # hard depth guard
    if symbol in _RECURSIVE_SPINE:
        if depth >= current_max_depth:
            return _generate_atom(depth, current_max_depth)

    productions, probabilities = rules[symbol]
    
    # increment depth only for structural nesting
    next_depth = depth + 1 if symbol in _STRUCTURAL_SYMBOLS else depth

    # special handling for number (uses plus on digit)
    if symbol == "NUMBER":
        digits = plus(lambda: generate("DIGIT", next_depth, current_max_depth), MAX_DIGITS)
        return "".join(digits)

    # special handling for expr (term followed by star op+term)
    if symbol == "EXPR":
        first_term = generate("TERM", next_depth, current_max_depth)
        extras = star(
            lambda: _gen_additive_tail(next_depth, current_max_depth),
            MAX_TERMS - 1,
        )
        return first_term + "".join(extras)

    # special handling for term (factor followed by star op+factor)
    if symbol == "TERM":
        first_factor = generate("FACTOR", next_depth, current_max_depth)
        extras = star(
            lambda: _gen_multiplicative_tail(next_depth, current_max_depth),
            MAX_FACTORS - 1,
        )
        return first_factor + "".join(extras)

    # general expansion
    chosen = random.choices(productions, weights=probabilities, k=1)[0]
    parts = [generate(s, next_depth, current_max_depth) for s in chosen]
    return "".join(parts)


# internal helpers for star/plus expansions

_additive_ops = [" + ", " - ", r" \pm ", r" \mp "]

def _gen_additive_tail(depth: int, current_max_depth: int) -> str:
    # generate one additive op and term pair for use inside star
    op = random.choice(_additive_ops)
    term = generate("TERM", depth, current_max_depth)
    return op + term


_multiplicative_ops = [r" \cdot ", r" \times "]

def _gen_multiplicative_tail(depth: int, current_max_depth: int) -> str:
    # generate one multiplicative op and factor pair for use inside star
    op = random.choice(_multiplicative_ops)
    factor = generate("FACTOR", depth, current_max_depth)
    return op + factor


# compile checker

def check_compiles(latex: str) -> bool:
    # try to render a latex math string with matplotlib mathtext
    # returns true if matplotlib can successfully parse and render it
    try:
        fig = plt.figure(figsize=(6, 1))
        fig.text(0.5, 0.5, f"${latex}$", fontsize=14, ha="center", va="center")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.close()
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


# convenience: generate and check in one call

def generate_valid(max_attempts: int = 50) -> str | None:
    # keep generating until we get a string that compiles, or give up
    # returns the valid latex string, or none after max_attempts failures
    for _ in range(max_attempts):
        s = generate()
        if check_compiles(s):
            return s
    return None


# main: generate 10 verified examples

def main(grammar_file: str = "src/data/grammar.py"):
    import os

    load_grammar(grammar_file)
    print(f"Grammar loaded: {grammar_file}")

    TARGET = 10
    MAX_TOTAL_ATTEMPTS = 200
    OUTPUT_DIR = "data/test_random_generator"

    print("=" * 60)
    print(f"ScanTeX String Generator – Saving {TARGET} Samples")
    print("=" * 60)
    
    # create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels_path = os.path.join(OUTPUT_DIR, "labels.txt")

    good_count = 0
    attempts = 0

    # open labels file for appending/writing
    with open(labels_path, "w", encoding="utf-8") as f_labels:
        while good_count < TARGET and attempts < MAX_TOTAL_ATTEMPTS:
            attempts += 1
            latex = generate()
            
            # verify it compiles via matplotlib
            if check_compiles(latex):
                img_filename = f"sample_{good_count:03d}.png"
                img_path = os.path.join(OUTPUT_DIR, img_filename)
                
                # actually render and save the image to disk
                fig = plt.figure(figsize=(6, 1))
                fig.text(0.5, 0.5, f"${latex}$", fontsize=14, ha="center", va="center")
                fig.savefig(img_path, format="png", bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)
                
                # save the raw label string
                f_labels.write(f"{img_filename}\t{latex}\n")
                
                good_count += 1
                print(f"  ✓ saved {img_filename} (attempt {attempts:3d})")

    if good_count >= TARGET:
        print(f"\nsuccessfully saved {good_count} generated samples to {OUTPUT_DIR}/")
    else:
        print(f"\nonly successfully generated {good_count}/{TARGET} samples after {attempts} attempts.")

    print("\n" + "=" * 60)
    print("Checking compilation success ratio")
    print("=" * 60)
    
    # run generator for a statistically significant number of samples
    # we just check them in memory, we do NOT save them to disk
    total_samples = 1000
    success_count = 0
    
    for i in range(total_samples):
        latex = generate()
        if check_compiles(latex):
            success_count += 1
            
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{total_samples} samples...")
            
    # calculate success ratio
    ratio = (success_count / total_samples) * 100
    print(f"\n  compilation success ratio: {ratio:.2f}% ({success_count}/{total_samples})")
    
    # remove all the compiled results 
    # (since matplotlib renders in memory via bytesio, we just clear the figures)
    plt.close("all")
    print("  in-memory compiled results removed")
    
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="scantex string generator")
    parser.add_argument(
        "--grammar",
        type=str,
        default="src/data/grammar.py",
        help="path to the grammar .py file to use for generation (default: src/data/grammar.py)"
    )
    args = parser.parse_args()

    main(grammar_file=args.grammar)
