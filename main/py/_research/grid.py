"""6 dataset profiles x model zoo full-grid evaluation (quick tier).

Runs the real TrustPipeline for every allowed combination and flattens the readings that
matter for the resume claims into one CSV. Writes incrementally; resumable via the CSV.
"""
import os, sys, time, json, csv, warnings, traceback
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
from trustoran.pipeline import TrustPipeline
from trustoran.models import available_models

SEED = 42
DSETS = ["cci_a1sd", "cci_a1sd_binary", "ddos_pcap",
         "vran_energy", "vran_application", "vran_computing"]
TASK_OF = {"cci_a1sd": "classification", "cci_a1sd_binary": "binary", "ddos_pcap": "classification",
           "vran_energy": "regression", "vran_application": "regression", "vran_computing": "binary"}
OUT = os.path.join(HERE, "grid.csv")

COLS = ["dataset", "model", "task", "n_samples", "elapsed_s", "fit_s",
        "acc", "f1_macro", "roc_auc", "r2", "mae",
        "rb_mode", "rb_clean_acc", "rb_fgsm", "rb_pgd", "rb_deepfool",
        "rb_pgd_asr", "rb_adv_trained", "rb_util_max", "rb_mae_at_max",
        "pr_mechanism", "pr_mia_auc", "pr_util_open", "pr_util_eps1", "pr_util_eps01",
        "pr_fed_acc", "pr_fed_central", "pr_fed_local", "pr_n_clients",
        "fa_group", "fa_n_groups", "fa_dp_diff", "fa_disparate_impact", "fa_acc_gap",
        "fa_mae_gap", "ex_top_feature", "ex_shap_top", "pillar_err"]

done = set()
if os.path.exists(OUT):
    with open(OUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["dataset"], row["model"]))

fh = open(OUT, "a", newline="", encoding="utf-8")
w = csv.DictWriter(fh, fieldnames=COLS)
if not done:
    w.writeheader()


def g(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return ""


def fmt(v):
    if v == "" or v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
        if not np.isfinite(f):
            return "inf" if f > 0 else "-inf"
        return "%.5f" % f
    except Exception:
        return str(v)[:60]


for dk in DSETS:
    from trustoran.data.base import TaskType
    task = {"classification": TaskType.CLASSIFICATION, "binary": TaskType.BINARY,
            "regression": TaskType.REGRESSION}[TASK_OF[dk]]
    for name in available_models(task):
        if (dk, name) in done:
            continue
        rec = {"dataset": dk, "model": name, "task": task.value}
        t0 = time.perf_counter()
        try:
            rep = TrustPipeline(dk, name, quick=True, seed=SEED).run()
            rec["elapsed_s"] = "%.1f" % (time.perf_counter() - t0)
            rec["n_samples"] = g(rep.dataset_summary, "n_samples")
            rec["fit_s"] = g(rep.model_spec, "elapsed_sec")
            bm = rep.base_metrics or {}
            rec["acc"] = fmt(g(bm, "accuracy"))
            rec["f1_macro"] = fmt(g(bm, "f1_macro"))
            rec["roc_auc"] = fmt(g(bm, "roc_auc"))
            rec["r2"] = fmt(g(bm, "r2"))
            rec["mae"] = fmt(g(bm, "mae"))

            pv = rep.pillars

            def _sum(key):
                p = pv.get(key)
                if p is None:
                    return {}
                return p.summary if hasattr(p, "summary") else (p.get("summary", {}) or {})

            rs = _sum("robustness")
            rec["rb_mode"] = g(rs, "mode", "metric")
            rec["rb_clean_acc"] = fmt(g(rs, "clean_accuracy", "clean_r2"))
            rec["rb_fgsm"] = fmt(g(rs, "fgsm_robust_acc"))
            rec["rb_pgd"] = fmt(g(rs, "pgd_robust_acc"))
            rec["rb_deepfool"] = fmt(g(rs, "deepfool_robust_acc"))
            rec["rb_pgd_asr"] = fmt(g(rs, "pgd_attack_success_rate"))
            rec["rb_adv_trained"] = fmt(g(rs, "adv_trained_robust_acc@eps0.6"))
            rec["rb_util_max"] = fmt(g(rs, "utility_at_max_sigma", "r2_at_max_sigma"))
            rec["rb_mae_at_max"] = fmt(g(rs, "mae_at_max_sigma"))

            ps = _sum("privacy")
            rec["pr_mechanism"] = g(ps, "dp_mechanism")
            rec["pr_mia_auc"] = fmt(g(ps, "membership_inference_auc"))
            rec["pr_util_open"] = fmt(g(ps, "utility_open"))
            rec["pr_util_eps1"] = fmt(g(ps, "utility_eps1"))
            rec["pr_util_eps01"] = fmt(g(ps, "utility_eps0.1"))
            rec["pr_fed_acc"] = fmt(g(ps, "federated_accuracy"))
            rec["pr_fed_central"] = fmt(g(ps, "centralized_accuracy"))
            rec["pr_fed_local"] = fmt(g(ps, "mean_local_only"))
            rec["pr_n_clients"] = fmt(g(ps, "n_clients"))

            fs = _sum("fairness")
            rec["fa_group"] = g(fs, "group_attr")
            rec["fa_n_groups"] = fmt(g(fs, "n_groups"))
            rec["fa_dp_diff"] = fmt(g(fs, "demographic_parity_difference"))
            rec["fa_disparate_impact"] = fmt(g(fs, "disparate_impact"))
            rec["fa_acc_gap"] = fmt(g(fs, "accuracy_gap"))
            rec["fa_mae_gap"] = fmt(g(fs, "mae_gap"))

            es = _sum("explainability")
            tf = g(es, "top_features")
            st = g(es, "shap_top")
            rec["ex_top_feature"] = (tf[0] if isinstance(tf, list) and tf else "")
            rec["ex_shap_top"] = (st[0] if isinstance(st, list) and st else "")
            errs = [k for k in ("explainability", "robustness", "fairness", "privacy")
                    if not _sum(k)]
            rec["pillar_err"] = ",".join(errs)
            print("%-17s %-16s elapsed=%-6s acc=%-8s fgsm=%-8s pgd=%-8s at=%-8s auc=%-8s %s" %
                  (dk, name, rec["elapsed_s"], rec["acc"] or rec["r2"], rec["rb_fgsm"],
                   rec["rb_pgd"], rec["rb_adv_trained"], rec["pr_mia_auc"],
                  ("empty:" + rec["pillar_err"]) if rec["pillar_err"] else ""), flush=True)
        except Exception as e:
            rec["elapsed_s"] = "%.1f" % (time.perf_counter() - t0)
            rec["pillar_err"] = "RUN:%s:%s" % (type(e).__name__, str(e)[:80])
            print("%-17s %-16s FAILED %s: %s" % (dk, name, type(e).__name__, e), flush=True)
            traceback.print_exc()
        w.writerow({c: rec.get(c, "") for c in COLS})
        fh.flush()

fh.close()
print("DONE ->", OUT, flush=True)
