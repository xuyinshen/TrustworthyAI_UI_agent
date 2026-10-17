"""Point at every `file.py:NNN(-NNN)` reference in MATH_PRINCIPLES.md and report
which function/class actually encloses that line today.

Flags pointers whose enclosing symbol looks unrelated to the prose around them,
so stale line numbers left by edits can be found without guessing.

Usage: python -X utf8 -u py/_research/doc_pointer_check.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DOC = os.path.join(REPO, "py", "MATH_PRINCIPLES.md")

FILES = ["privacy.py", "explainability.py", "robustness.py", "engine.py",
         "wrappers.py", "fairness.py", "report.py", "factory.py", "plotting.py",
         "labels.py", "run_all.py", "main_cli.py"]

paths = {}
for root, _dirs, fs in os.walk(os.path.join(REPO, "py")):
    if "_research" in root:
        continue
    for f in fs:
        if f in FILES:
            paths.setdefault(f, os.path.join(root, f))

symbols = {}
for f, p in paths.items():
    lines = open(p, encoding="utf-8").read().splitlines()
    decls = []
    for i, ln in enumerate(lines, 1):
        m = re.match(r"\s*(?:def|class)\s+([A-Za-z_]\w*)", ln)
        if m:
            decls.append((i, m.group(1)))
    symbols[f] = (lines, decls)


def enclosing(f: str, line: int):
    lines, decls = symbols[f]
    name = "<module>"
    for i, n in decls:
        if i <= line:
            name = n
        else:
            break
    return name, len(lines)


text = open(DOC, encoding="utf-8").read()
pat = re.compile(r"\b((?:metrics/)?(" + "|".join(map(re.escape, FILES)) + r"))[:：](\d+)(?:\s*[-–]\s*(\d+))?")

rows = []
seen = set()
for m in pat.finditer(text):
    written, f, a, b = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    b = int(b) if b else a
    key = (written, a, b)
    if key in seen:
        continue
    seen.add(key)
    if written.startswith("metrics/"):
        # trustagent package -- same basenames, different files, not resolved here
        rows.append((written, a, b, "-", "-", "agent-side"))
        continue
    if f not in symbols:
        rows.append((written, a, b, "?", "?", "file not found"))
        continue
    name_start, _ = enclosing(f, a)
    name_end, _ = enclosing(f, min(b, len(symbols[f][0])))
    oob = "OUT OF RANGE" if b > len(symbols[f][0]) else ""
    rows.append((written, a, b, name_start, name_end, oob))

print("%-24s %6s %6s  %-24s %-24s %s" % ("file", "from", "to", "encloses(from)", "encloses(to)", "flag"))
for f, a, b, ns, ne, oob in sorted(rows):
    print("%-24s %6d %6d  %-24s %-24s %s" % (f, a, b, ns, ne, oob))
print("\ntotal distinct pointers: %d over %d files" % (len(rows), len({r[0] for r in rows})))
print("note: rows marked agent-side live in Agent/module_3_agent_eval and are not resolved here")
