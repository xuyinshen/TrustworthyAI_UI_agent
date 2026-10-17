"""Which in-figure strings still come out English after relabel_figure(lang="zh")?

Builds every figure the pipeline can produce (ddos_pcap quick: all four pillars,
incl. FedAvg and membership inference), then relabels each figure into Chinese
and lists the drawable strings that still contain ASCII letters.

Usage:  python -X utf8 -u py/_research/label_gap_probe.py
Writes: py/_research/label_gaps.csv
"""
from __future__ import annotations

import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "py"))

from trustoran.pipeline.engine import TrustPipeline, PILLARS   # noqa: E402
from trustoran.viz.labels import relabel_figure, translate     # noqa: E402

ASCII = re.compile(r"[A-Za-z]")


def strings_of(fig):
    from matplotlib.text import Text
    out = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if s and ASCII.search(s):
            out.append(s)
    return out


rows = []
seen = set()

pipe = TrustPipeline(dataset_key="ddos_pcap", model_name="random_forest",
                     pillars=PILLARS, quick=True, seed=42)
report = pipe.run()

groups = [("base", name, fig) for name, fig in report.base_figures.items()]
for pillar, res in report.pillars.items():
    groups += [(pillar, name, fig) for name, fig in res.figures.items()]

for pillar, key, fig in groups:
    before = strings_of(fig)
    relabel_figure(fig, "zh")
    after = {s for s in strings_of(fig)}
    for s in before:
        translated = translate(s, "zh") != s
        if s in after:                       # survived the relabel
            if (s, pillar) in seen:
                continue
            seen.add((s, pillar))
            rows.append({"pillar": pillar, "figure": key, "english_string": s,
                         "matched_a_pair": "no", "still_english": "yes"})
        elif not translated:
            rows.append({"pillar": pillar, "figure": key, "english_string": s,
                         "matched_a_pair": "yes", "still_english": "no"})

out = os.path.join(HERE, "label_gaps.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["pillar", "figure", "english_string",
                                       "matched_a_pair", "still_english"])
    w.writeheader()
    w.writerows(rows)

print("figures checked: %d" % len(groups))
print("strings still English after relabel: %d" % sum(
    1 for r in rows if r["still_english"] == "yes"))
for r in rows:
    if r["still_english"] == "yes":
        print("  [%s/%s] %r" % (r["pillar"], r["figure"], r["english_string"]))
print("wrote %s" % out)
