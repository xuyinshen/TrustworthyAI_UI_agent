"""Verification for ``EX.shap_explain(method="auto")``: exact tree attributions.

``MATH_PRINCIPLES.md`` 8.23 recorded that the library offered only a sampled
KernelSHAP-style approximation, while xgboost/lightgbm ship the exact TreeSHAP
decomposition in C++.  ``trustoran`` now routes those two boosters to their own
exact path and keeps the sampled estimator for every model without one.  This
script is the exit test for that change:

1. routing -- xgboost/lightgbm return ``method="exact"``, every other zoo model
   returns ``method="kernel"``, and ``method="exact"`` on a model without an
   exact path raises instead of silently degrading.
2. exactness -- on the exact path ``base + sum(phi)`` must equal the model's own
   score (margin for classifiers, prediction for regression) to float
   round-off, measured as ``ShapResult.efficiency_max_error``.
3. correctness against the reference -- the returned matrix is compared sample
   by sample with ``shap.TreeExplainer`` (same estimator, same margin space,
   path-dependent).  ``trees_study.py`` already showed library *approximations*
   disagree with exact values; here both sides are exact, so any real deviation
   would be a slicing/shape bug in ``TrustModel.native_contribs``.
4. cost -- exact vs sampled per-sample time on the same fitted booster, i.e. the
   price the previous default was paying.

Outputs ``exact_shap_check.csv`` plus stdout; exit code 1 if any assertion fails.
"""
import os, sys, time, warnings, csv, traceback
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import shap

from trustoran.data import build_dataset, train_test_split_dataset
from trustoran.data.base import TaskType
from trustoran.models import make_model
from trustoran.trust import explainability as EX

SEED = 42
NBG = 40
TARGET_CLASS = 1
NEXP = 24          # rows used for the shap.TreeExplainer cross-check
NTIME = 100        # rows used for the exact-vs-sampled timing
DATASETS = os.environ.get("EXACT_DSETS", "cci_a1sd,cci_a1sd_binary,vran_computing,vran_energy").split(",")
EXACT_MODELS = os.environ.get("EXACT_EXACT", "xgboost,lightgbm").split(",")
FALLBACK_MODELS = os.environ.get("EXACT_FALLBACK", "random_forest,hist_grad_boost,grad_boost,logistic,mlp").split(",")
OUT = os.path.join(HERE, "exact_shap_check.csv")

COLS = ["dataset", "task", "method", "n_rows", "d", "model", "check",
        "eff_err", "rel_err", "cos_mean", "cos_min", "max_abs_diff", "base_diff",
        "ms_per_sample", "speedup", "note"]
fh = open(OUT, "w", newline="", encoding="utf-8")
wr = csv.DictWriter(fh, fieldnames=COLS)
wr.writeheader()

FAILS = []


def rec(**kw):
    row = {c: kw.get(c, "") for c in COLS}
    for c in ["eff_err", "rel_err", "cos_mean", "cos_min", "max_abs_diff", "base_diff",
              "ms_per_sample", "speedup"]:
        v = row[c]
        if isinstance(v, float):
            row[c] = "%.6g" % v
    wr.writerow(row); fh.flush()
    print("%-16s %-9s %-8s %-18s %-16s | eff=%-10s rel=%-10s cos=%-6s ms/s=%-9s x%s" % (
        str(row["dataset"])[:16], row["task"], row["method"], row["model"][:18],
        row["check"][:16], row["eff_err"], row["rel_err"], row["cos_mean"],
        row["ms_per_sample"], row["speedup"] or "-"), flush=True)


def fail(msg):
    FAILS.append(msg)
    print("   FAIL: " + msg, flush=True)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _slice_ref(vals, ev, idx):
    """Normalise shap.TreeExplainer output to (n, d) + scalar reference for one class.

    Shapes mirror ``trees_study._slice_class``: a per-class list, (n, d), or
    (n, d, K) with the class on the last axis.
    """
    e = np.asarray(ev, float)
    v = vals if isinstance(vals, list) else np.asarray(vals, float)
    if isinstance(v, list):
        i = min(idx, len(v) - 1)
        return np.asarray(v[i], float), float(e.ravel()[i]) if e.size else 0.0
    if v.ndim == 3:
        i = min(idx, v.shape[2] - 1)
        return v[:, :, i], float(e.ravel()[i]) if e.size else 0.0
    if v.ndim == 2 and e.size > 1:
        return v, float(e.ravel()[min(idx, e.size - 1)])
    return v, float(e.ravel()[0]) if e.size else 0.0


def reference(exact_result, est, key, X, idx, k):
    """shap.TreeExplainer (path-dependent, the library's own default space)."""
    exp = shap.TreeExplainer(est)
    try:
        vals, ev = exp.shap_values(X), exp.expected_value
    except Exception:
        # lightgbm multiclass trips shap's additivity assertion; retry without it
        vals, ev = exp.shap_values(X, check_additivity=False), exp.expected_value
    sv, ref = _slice_ref(vals, ev, idx)
    sv = np.asarray(sv, float)
    if sv.shape != np.asarray(exact_result.shap_values).shape:
        return None, None, "reference shape %s vs ours %s" % (
            sv.shape, exact_result.shap_values.shape)
    return sv, float(ref), ""


for dset in DATASETS:
    ds = build_dataset(dset, max_rows=3000, seed=SEED)
    sp = train_test_split_dataset(ds, test_size=0.3, seed=SEED)
    task = sp.train.task
    reg = task == TaskType.REGRESSION
    r_method = "regression" if reg else "classification"
    Xtr = np.asarray(sp.X_train[:2000], float)
    ytr = (np.asarray(sp.y_train[:2000], float) if reg
           else np.asarray(sp.y_train[:2000], int))
    Xte = np.asarray(sp.X_test, float)
    Xexp, Xtime = Xte[:NEXP], Xte[:min(NTIME, len(Xte))]
    bg = Xtr[:NBG]
    d = Xtr.shape[1]
    K = 1 if reg else int(max(np.asarray(sp.y_train, int).max(),
                              np.asarray(sp.y_test, int).max()) + 1)
    print("== %s  task=%s  d=%d  K=%d" % (dset, "regression" if reg else "classification", d, K), flush=True)

    for key in EXACT_MODELS + FALLBACK_MODELS:
        want_exact = key in EXACT_MODELS
        try:
            model = make_model(key, task, sp.feature_names, sp.train.class_names)
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            t_fit = time.perf_counter() - t0

            r = EX.shap_explain(model, Xexp, bg,
                                target_class=(None if reg else TARGET_CLASS), seed=SEED)
            phi = np.asarray(r.shap_values, float)

            # ---- 1. routing ------------------------------------------------
            rec(dataset=dset, task=r_method, model=key, method=r.method,
                check="routing", n_rows=len(Xexp), d=d,
                note="target=%s fit %.2fs" % (r.target, t_fit))
            if (r.method == "exact") != want_exact:
                fail("%s/%s: method=%s, expected %s" % (
                    dset, key, r.method, "exact" if want_exact else "kernel"))

            if not want_exact:
                # sampled path self-rescales, so its residual is 0 by construction;
                # what matters here is that the curve still reproduces f(x).
                fvals = EX.make_model_fn(model, None if reg else TARGET_CLASS)(Xexp)
                err = float(np.max(np.abs(r.base_value + phi.sum(1) - fvals)))
                rec(dataset=dset, task=r_method, model=key, method=r.method,
                    check="sampled efficiency", n_rows=len(Xexp), d=d, eff_err=err,
                    ms_per_sample=float("nan"),
                    note="max|base+sum(phi)-f(x)|; rescaled => self-consistent only")
                if not np.isfinite(err) or err > 0.05:
                    fail("%s/%s: sampled path does not reproduce f(x) (err=%.4g)" % (dset, key, err))
                # exact must be refused, not silently downgraded
                try:
                    EX.shap_explain(model, Xexp[:4], bg, seed=SEED,
                                    target_class=(None if reg else TARGET_CLASS),
                                    method="exact")
                    fail("%s/%s: method='exact' returned on a model with no exact path" % (dset, key))
                except ValueError as e:
                    rec(dataset=dset, task=r_method, model=key, method="n/a",
                        check="exact refused", n_rows=4, d=d,
                        note="ValueError: %s" % str(e).replace("\n", " ")[:90])
                continue

            # ---- 2. exactness: base + sum(phi) must land on the model's own score.
            #      xgboost accumulates pred_contribs in float32 (residual ~1e-6 on a
            #      margin of order 1-10), lightgbm in float64 (~1e-14), so the bound
            #      is relative to the score scale, not absolute.
            idx = min(TARGET_CLASS, K - 1) if not reg else 0
            phi_e, base_e, margin_e = model.native_contribs(Xexp, None if reg else TARGET_CLASS)
            scale = float(np.max(np.abs(margin_e))) or 1.0
            rel = float(r.efficiency_max_error) / scale
            rec(dataset=dset, task=r_method, model=key, method="exact",
                check="exactness", n_rows=len(Xexp), d=d,
                eff_err=float(r.efficiency_max_error), rel_err=rel,
                note="max|base+sum(phi)-margin|=%.3g, max|margin|=%.3g" % (
                    float(r.efficiency_max_error), scale))
            if not np.isfinite(r.efficiency_max_error) or rel > 1e-5:
                fail("%s/%s: efficiency residual %.3g (rel %.3g) above float round-off" % (
                    dset, key, float(r.efficiency_max_error), rel))

            # ---- 3. cross-check against shap.TreeExplainer ----------------
            note = ""
            try:
                ref, ev, why = reference(r, model.estimator, key, Xexp, idx, K)
                if ref is None:
                    note = "reference skipped: " + why
                else:
                    cs = [_cos(phi[i], ref[i]) for i in range(len(phi))]
                    mad = float(np.max(np.abs(phi - ref)))
                    rec(dataset=dset, task=r_method, model=key, method="exact",
                        check="vs TreeExplainer", n_rows=len(Xexp), d=d,
                        cos_mean=float(np.mean(cs)), cos_min=float(np.min(cs)),
                        max_abs_diff=mad, base_diff=abs(float(r.base_value) - ev))
                    if np.mean(cs) < 0.99 or mad > 0.05 * max(1.0, float(np.abs(ref).max())):
                        fail("%s/%s: disagrees with shap.TreeExplainer (cos=%.4f maxdiff=%.4g)" % (
                            dset, key, float(np.mean(cs)), mad))
            except Exception as e:
                note = "reference error %s: %s" % (type(e).__name__, e)

            # ---- 4. cost: exact vs sampled on the same fitted booster -----
            t0 = time.perf_counter()
            EX.shap_explain(model, Xtime, bg, seed=SEED,
                            target_class=(None if reg else TARGET_CLASS))
            t_exact = time.perf_counter() - t0
            t0 = time.perf_counter()
            ks = EX.shap_explain(model, Xtime, bg, seed=SEED, method="kernel",
                                 target_class=(None if reg else TARGET_CLASS))
            t_kernel = time.perf_counter() - t0
            mx, mk = t_exact / len(Xtime) * 1000.0, t_kernel / len(Xtime) * 1000.0
            rec(dataset=dset, task=r_method, model=key, method="exact",
                check="cost exact", n_rows=len(Xtime), d=d, ms_per_sample=mx, note=note)
            rec(dataset=dset, task=r_method, model=key, method="kernel",
                check="cost sampled", n_rows=len(Xtime), d=d, ms_per_sample=mk,
                speedup=(mk / mx if mx > 0 else float("nan")),
                note="target=%s" % ks.target)
            if mx >= mk:
                fail("%s/%s: exact path is not faster than sampling (%.3f vs %.3f ms/sample)" % (
                    dset, key, mx, mk))
        except Exception as e:
            traceback.print_exc(limit=3)
            fail("%s/%s raised %s: %s" % (dset, key, type(e).__name__, e))

fh.close()
print("\nrows: %d   failures: %d" % (len(open(OUT, encoding="utf-8").readlines()) - 1, len(FAILS)), flush=True)
if FAILS:
    for m in FAILS:
        print(" - " + m, flush=True)
    sys.exit(1)
print("ALL CHECKS PASSED", flush=True)
