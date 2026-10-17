"""Prototype only: read py/_research/mia_variants.csv and print the aggregates quoted in the
write-up. Each block prints its own row count and filter so the numbers stay auditable.
"""
import os, csv, collections
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "mia_variants.csv"), encoding="utf-8")))
print("total rows", len(rows), "columns", len(rows[0]))


def v(x):
    return float(x) if x not in ("", None) else None


def sig(r):
    a = v(r["auc"])
    return abs(a - 0.5) if a is not None else -1


print("\n== A. shipped vs improved, per dataset x target (rows with both present) ==")
idx = collections.defaultdict(dict)
for r in rows:
    if r["attack"] in ("shipped", "lr_meta", "ssa8", "lira_avg", "threshold1"):
        idx[(r["dataset"], r["target"])].setdefault(r["attack"], []).append(r)
for k, d in sorted(idx.items()):
    sh = d.get("shipped", [None])[0]
    if not sh:
        continue
    best = {}
    for at in ("threshold1", "lr_meta", "ssa8", "lira_avg"):
        cand = max(d.get(at, []), key=sig, default=None)
        if cand:
            best[at] = (v(cand["auc"]), cand["feature_set"][:28])
    print("%-16s %-13s shipped=%.4f  " % (k[0], k[1], v(sh["auc"])),
          "  ".join("%s=%.4f(%s)" % (a, b[0], b[1]) for a, b in best.items()))

print("\n== B. generalization gap vs attack signal (all targets) ==")
gaps, auces = {}, {}
for r in rows:
    if r["attack"] == "generalization":
        gaps[(r["dataset"], r["target"])] = r["note"][:64]
    if r["attack"] == "shipped":
        auces[(r["dataset"], r["target"])] = v(r["auc"])
for k in sorted(set(gaps) & set(auces)):
    print("%-16s %-13s auc=%.4f  %s" % (k[0], k[1], auces[k], gaps[k]))

print("\n== C. best attack across every unprotected vs DP target ==")
best = collections.defaultdict(lambda: ("", "", 0.0))
for r in rows:
    if r["attack"] in ("generalization", "timing") or v(r["auc"]) is None:
        continue
    key = (r["dataset"], r["target"])
    if sig(r) > best[key][2]:
        best[key] = (r["attack"], r["feature_set"], sig(r))
for k, b in sorted(best.items()):
    print("%-16s %-13s %-11s %-46s |AUC-.5|=%.4f" % (k[0], k[1], b[0], b[1], b[2]))

print("\n== D. KS separability, single features, random_forest only ==")
for r in rows:
    if r["attack"] == "threshold1" and r["target"] == "random_forest" and v(r["ks"]) is not None:
        print("%-16s %-14s ks=%.4f auc=%.4f  %s" % (r["dataset"], r["feature_set"],
              v(r["ks"]), v(r["auc"]), r["note"][:56]))

print("\n== E. dedup rows (only datasets with train/test overlap) ==")
n = 0
for r in rows:
    if r["dedup"] == "yes":
        n += 1
        print("%-16s %-13s %-10s %-20s auc=%s ks=%s %s" % (r["dataset"], r["target"], r["attack"],
              r["feature_set"][:20], r["auc"], r["ks"], r["note"][:60]))
print("rows:", n)

print("\n== F. per-target timing rows (fit+attacks seconds) ==")
for r in rows:
    if r["attack"] == "timing":
        print("%-16s %-12s %s" % (r["dataset"], r["target"], r["note"]))

print("\n== G. feature-set sizes actually evaluated ==")
sets = collections.Counter(r["feature_set"] for r in rows if r["attack"] in ("lr_meta", "ssa8", "lira_avg"))
for s, c in sets.most_common():
    print("%3d x  %s" % (c, s[:60]))
