"""Read-only summary of adv_trees.csv (self-written KernelSHAP vs exact tree attribution).

Every number printed here is recomputed from the CSV; nothing is carried over from
the run log.
"""
import os, csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = []
with open(os.path.join(HERE, "adv_trees.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ROWS.append(r)


def fl(r, k):
    v = r.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt(v, w="%.4f"):
    return w % v if v is not None else "  --  "


print("rows=%d  datasets=%s  models=%s" % (
    len(ROWS), sorted({r["dataset"] for r in ROWS}), sorted({r["model"] for r in ROWS})))
bad = [r for r in ROWS if not fl(r, "cos_mean")]
print("rows without agreement metrics: %d  %s" % (
    len(bad), [(r["dataset"], r["model"], r["ref_impl"], str(r["note"])[:60]) for r in bad]))

print("\n== A · timing: our sampling KernelSHAP vs the exact reference (ms/sample, larger n) ==")
print("%-16s %-13s %-22s %-14s %9s %9s %8s" % (
    "dataset", "model", "reference", "space", "ours", "exact", "ratio"))
ratios = []
for r in ROWS:
    o = fl(r, "ours_ms_per_sample_scale") or fl(r, "ours_ms_per_sample")
    x = fl(r, "ref_ms_per_sample_scale") or fl(r, "ref_ms_per_sample")
    rt = fl(r, "ratio_ms_per_sample")
    if rt is None and o and x:
        rt = o / x
    ratios.append(rt)
    print("%-16s %-13s %-22s %-14s %9s %9s %8s" % (
        r["dataset"], r["model"], r["ref_impl"], str(r["space"])[:14],
        fmt(o, "%9.2f"), fmt(x, "%9.3f"), fmt(rt, "%8.1f")))
ok = [v for v in ratios if v]
print("ratio ours/exact: min %.1fx  median %.1fx  max %.1fx  n=%d  rows with ratio<1: %d" % (
    min(ok), sorted(ok)[len(ok) // 2], max(ok), len(ok), sum(1 for v in ok if v < 1)))

print("\n== B · agreement (per-sample cosine / sign / global rank / Top-3) ==")
print("%-16s %-13s %-22s %-14s %7s %7s %7s %6s %8s" % (
    "dataset", "model", "reference", "space", "cos", "sign", "rho", "top3", "baseDiff"))
for r in ROWS:
    print("%-16s %-13s %-22s %-14s %7s %7s %7s %6s %8s" % (
        r["dataset"], r["model"], r["ref_impl"], str(r["space"])[:14],
        fmt(fl(r, "cos_mean"), "%7.3f"), fmt(fl(r, "sign_agree"), "%7.3f"),
        fmt(fl(r, "spearman_meanabs"), "%7.3f"), fmt(fl(r, "top3_overlap"), "%6.2f"),
        fmt(fl(r, "base_diff"), "%8.4f")))

print("\n== C · additivity residual |base + sum(phi) - target| (mean over explained rows) ==")
print("%-16s %-13s %-22s %-8s %10s %12s %12s" % (
    "dataset", "model", "reference", "space", "ours(proba)", "ref(own)", "ref(proba)"))
for r in ROWS:
    print("%-16s %-13s %-22s %-8s %10s %12s %12s" % (
        r["dataset"], r["model"], r["ref_impl"], str(r["space"])[:8],
        fmt(fl(r, "eff_err_ours"), "%.2e"), fmt(fl(r, "eff_err_ref"), "%.2e"),
        fmt(fl(r, "eff_err_ref_proba"), "%.2e")))

print("\n== D · ranges by reference family (three different estimators, keep them apart) ==")
by = defaultdict(list)
for r in ROWS:
    if fl(r, "cos_mean") is None:
        continue
    key = r["ref_impl"] if not r["ref_impl"].startswith("TreeSHAP") else (
        "TreeSHAP path-dependent" if "(path)" in r["ref_impl"] else "TreeSHAP interventional")
    by[key].append((fl(r, "cos_mean"), fl(r, "sign_agree"), fl(r, "spearman_meanabs"),
                    fl(r, "top3_overlap"), fl(r, "ratio_ms_per_sample")))
for k, v in sorted(by.items()):
    def rng(i):
        xs = [t[i] for t in v if t[i] is not None]
        return "n/a" if not xs else "%.3f..%.3f (mean %.3f)" % (min(xs), max(xs), sum(xs) / len(xs))
    rt = [t[4] for t in v if t[4]]
    rtxt = ("n/a" if not rt else "%.1fx..%.1fx (median %.1fx, ours faster in %d/%d)" % (
        min(rt), max(rt), sorted(rt)[len(rt) // 2], sum(1 for x in rt if x < 1), len(rt)))
    print("%-26s n=%-3d cos %s | sign %s | rho %s | top3 %s\n%-26s   ratio ours/exact %s" % (
        k, len(v), rng(0), rng(1), rng(2), rng(3), "", rtxt))

print("\n== E · comparability guard: rows whose reference lives in margin space ==")
print("(magnitude metrics comparable only in proba space; sign/rank stay valid under a monotone link)")
for r in ROWS:
    if "margin" in str(r["space"]):
        print("  %-16s %-13s %-22s %s" % (r["dataset"], r["model"], r["ref_impl"], str(r["note"])[:70]))

print("\n== F · implementation scale (why the timing gap looks like that) ==")
seen = set()
for r in ROWS:
    k = (r["dataset"], r["model"])
    if k in seen:
        continue
    seen.add(k)
    print("  %-16s %-13s d=%-3s trees=%-7s coalitions=%-5s fwd/sample=%-6s %s" % (
        k[0], k[1], r["d"], r["n_trees"], r["ours_coalitions"], r["ours_fwd_per_sample"],
        str(r["note"])[:40]))
