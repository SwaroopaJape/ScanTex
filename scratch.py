from lark import Lark
parser = Lark.open("src/data/grammar.lark", rel_to=__file__)
for term in parser.terminals:
    print(term.name, term.pattern.value)
for rule in parser.rules:
    print(rule.origin.name, "->", [t.name if hasattr(t, 'name') else t for t in rule.expansion])
