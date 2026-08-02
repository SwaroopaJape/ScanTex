import json
import argparse
from pathlib import Path
from collections import defaultdict
import lark
import re

def main():
    parser = argparse.ArgumentParser(description="Generate PCFG from scraped rule frequencies using Laplace smoothing.")
    parser.add_argument("--freq-file", type=str, default="data/scraped_info/rule_frequencies.json")
    parser.add_argument("--lark-file", type=str, default="src/data/grammar.lark")
    parser.add_argument("--out-file", type=str, default="data/extracted_grammar/weighted_grammar.py")
    parser.add_argument("--alpha", type=float, default=0.01, help="Laplace smoothing parameter")
    args = parser.parse_args()

    freq_path = Path(args.freq_file)
    if not freq_path.exists():
        print(f"Error: Frequency file not found at {freq_path}")
        return

    with open(freq_path, "r", encoding="utf-8") as f:
        freqs = json.load(f)

    with open(args.lark_file, "r", encoding="utf-8") as f:
        lark_text = f.read()
    g = lark.Lark(lark_text, parser="earley")

    terminal_to_literal = {}
    terminal_to_pattern = {}
    for term in g.terminals:
        if term.pattern.type == 'str':
            terminal_to_literal[term.name] = term.pattern.value
        terminal_to_pattern[term.name] = term.pattern.value

    # Parse token frequencies from freqs
    regex_expansions = defaultdict(list)
    token_re = re.compile(r"Token\('[^']+', '([^']+)'\)")
    
    signature_counts = defaultdict(float)
    
    for k, count in freqs.items():
        k_clean = k.strip()
        signature_counts[k_clean] += count
        
        if " -> " in k_clean:
            lhs, rhs = k_clean.split(" -> ", 1)
            m = token_re.fullmatch(rhs.strip())
            if m:
                val = m.group(1)
                regex_expansions[lhs.strip()].append((val, count))

    # Create mapping from signature to rules
    signature_to_rules = defaultdict(list)
    for rule in g.rules:
        lhs = rule.origin.name
        if lhs in regex_expansions:
            continue
            
        rhs_parts = []
        out_rhs = []
        for sym in rule.expansion:
            if isinstance(sym, lark.grammar.NonTerminal):
                rhs_parts.append(sym.name)
                out_rhs.append(sym.name.upper())
            elif isinstance(sym, lark.grammar.Terminal):
                if sym.name in terminal_to_literal:
                    lit = terminal_to_literal[sym.name]
                    out_rhs.append(lit)
                    if not sym.name.startswith("__ANON"):
                        rhs_parts.append(f"Token('{sym.name}', '{lit}')")
                else:
                    out_rhs.append(terminal_to_pattern.get(sym.name, f"<{sym.name}>"))
        
        rhs_str = " ".join(rhs_parts)
        sig = f"{lhs} -> {rhs_str}"
        signature_to_rules[sig].append((lhs, out_rhs))

    # Distribute counts and build PCFG
    rule_data = defaultdict(list)
    for sig, rules_list in signature_to_rules.items():
        total_count = signature_counts.get(sig, 0.0)
        split_count = total_count / len(rules_list) if rules_list else 0.0
        for lhs, out_rhs in rules_list:
            rule_data[lhs].append((out_rhs, split_count))

    for lhs, options in regex_expansions.items():
        for val, count in options:
            rule_data[lhs].append(([val], count))

    # Format the Python output
    lines = []
    lines.append("# scantex pcfg grammar definition (AUTO-GENERATED)")
    lines.append("# generated from grammar.lark and scraped rule_frequencies.json")
    lines.append("")
    lines.append("MAX_DEPTH = 4")
    lines.append("MAX_TERMS = 3")
    lines.append("MAX_FACTORS = 2")
    lines.append("MAX_DIGITS = 3")
    lines.append("MAX_SCRIPT_DEPTH = 2")
    lines.append("MAX_FUNC_ARGS = 1")
    lines.append("MAX_MATRIX_ROWS = 3")
    lines.append("MAX_MATRIX_COLS = 3")
    lines.append("")
    lines.append("rules = {}")
    lines.append("")
    
    non_terminals = set()
    
    for lhs, productions in rule_data.items():
        lhs_upper = lhs.upper()
        non_terminals.add(lhs_upper)
        
        total_counts = sum(count for _, count in productions)
        total_options = len(productions)
        denom = total_counts + (args.alpha * total_options)
        
        # Calculate probabilities
        prob_list = []
        for out_rhs, count in productions:
            prob = (count + args.alpha) / denom
            prob_list.append((out_rhs, prob))
            
        # Sort by probability descending
        prob_list.sort(key=lambda x: x[1], reverse=True)
        
        lines.append(f"rules['{lhs_upper}'] = (")
        lines.append("    [")
        for out_rhs, _ in prob_list:
            # properly format list of strings
            rhs_repr = ", ".join(repr(s) for s in out_rhs)
            lines.append(f"        [{rhs_repr}],")
        lines.append("    ],")
        
        probs_repr = ", ".join(str(p) for _, p in prob_list)
        lines.append(f"    [{probs_repr}]")
        lines.append(")")
        lines.append("")
        
    lines.append("non_terminals = {")
    for nt in sorted(list(non_terminals)):
        lines.append(f"    '{nt}',")
    lines.append("}")
    lines.append("")
    
    lines.append("terminals = set()")
    lines.append("")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated PCFG with {len(non_terminals)} non-terminals.")
    print(f"Laplace smoothing alpha = {args.alpha}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
