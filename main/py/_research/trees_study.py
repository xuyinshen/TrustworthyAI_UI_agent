"""Self-written attribution vs exact tree attribution (TreeSHAP / native pred_contribs).

Background: ``ipynb/05_ours_vs_opensource.ipynb`` benchmarks ``EX.shap_explain``
against sampling-based black-box implementations only (``shap.KernelExplainer``,
``captum.attr.KernelShap``).  Tree ensembles admit an *exact, sampling-free*
attribution (Lundberg et al. 2018, Theorems 1-2) and xgboost/lightgbm ship it in
C++.  Nothing in this repository has ever been compared against those paths, so
the "1-2 orders of magnitude faster" claim was never tested where it is weakest.

Design (kept honest):
* Same fitted model, same explained rows, same target function f(x)=P(class 1)
  whenever the reference can be expressed in that space.
* Two reference flavours for tree ensembles: TreeSHAP *path-dependent* (library
  default, assumes feature correlation) and TreeSHAP *interventional*
  (``data=BG``, same independence assumption and the same reference value as our
  KernelSHAP).  Comparing against the interventional variant isolates the
  estimator difference from the background-distribution difference.
* The output space of each reference is classified empirically (does
  reference + sum(phi) land on f(x) or on log f(x)?) instead of trusting
  ``model_output``, because xgboost/lightgbm default to margin space while
  sklearn forests return class fractions.  Magnitude metrics are therefore only
  claimed comparable when space == proba; sign/rank metrics stay valid across a
  monotone link.
* Timing is measured on the small agreement set and on a larger set so the
  per-sample cost at scale is visible.

Rows are appended to ``adv_trees.csv`` incrementally; re-running skips
(dataset, model, ref_impl) triples already present.
"""
import os, sys, time, warnings, csv, traceback
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
from scipy.stats import spearmanr

from trustoran.data import build_dataset, train_test_split_dataset
from trustoran.data.base import TaskType
from trustoran.models import make_model
from trustoran.trust import explainability as EX

SEED = 42
DSETS = os.environ.get("TREES_DSETS", "cci_a1sd,cci_a1sd_binary,vran_computing,ddos_pcap").split(",")
MODELS = os.environ.get("TREES_MODELS", "random_forest,xgboost,lightgbm").split(",")
NTRAIN = int(os.environ.get("TREES_NTRAIN", 4000))
NTEST = int(os.environ.get("TREES_NTEST", 600))
NEXP = int(os.environ.get("TREES_NEXP", 24))       # rows used for agreement metrics
NSCALE = int(os.environ.get("TREES_NSCALE", 200))  # rows used for per-sample timing
NBG = 40
TARGET_CLASS = 1                                   # P(class 1) -- the 'attack' class
OUT = os.path.join(HERE, "adv_trees.csv")

NUMCOLS = ["cos_mean", "cos_meanabs", "spearman_meanabs", "sign_agree", "top3_overlap",
           "top1_match", "base_diff", "eff_err_ours", "eff_err_ref",
           "t_ours_s", "t_ref_s", "t_ref_init_s", "ours_ms_per_sample", "ref_ms_per_sample",
           "t_ours_scale_s", "t_ref_scale_s", "ours_ms_per_sample_scale",
           "ref_ms_per_sample_scale", "ratio_ms_per_sample", "ref_margin_err",
           "eff_err_ref_proba",
           "n_trees", "ours_coalitions", "ours_fwd_per_sample"]
COLS = ["dataset", "model", "task", "n_train", "n_test", "d", "n_exp", "n_scale",
        "ref_impl", "space"] + NUMCOLS + ["note"]

done = set()
if os.path.exists(OUT):
    with open(OUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["dataset"], row["model"], row["ref_impl"]))

fh = open(OUT, "a", newline="", encoding="utf-8")
w = csv.DictWriter(fh, fieldnames=COLS)
if not done:
    w.writeheader()


def write(rec):
    out = {}
    for k in COLS:
        v = rec.get(k)
        out[k] = "" if v is None else ("%.5f" % float(v) if k in NUMCOLS and v != "" else v)
    w.writerow(out)
    fh.flush()
    o = rec.get("ours_ms_per_sample"); r = rec.get("ref_ms_per_sample"); nn = rec.get("n_exp")
    if rec.get("ours_ms_per_sample_scale") and rec.get("ref_ms_per_sample_scale"):
        o = rec["ours_ms_per_sample_scale"]; r = rec["ref_ms_per_sample_scale"]; nn = rec.get("n_scale")
    print("%-16s %-13s %-22s %-14s cos=%.3f sign=%.3f rho=%.3f top3=%.2f | ours %7.2f  ref %7.2f ms/sample (n=%s)  x%s" % (
        rec.get("dataset"), rec.get("model"), rec.get("ref_impl"), str(rec.get("space"))[:14],
        float(rec.get("cos_mean") or 0), float(rec.get("sign_agree") or 0),
        float(rec.get("spearman_meanabs") or 0), float(rec.get("top3_overlap") or 0),
        float(o or 0), float(r or 0), nn,
        ("%g" % rec["ratio_ms_per_sample"]) if rec.get("ratio_ms_per_sample") else "-"),
        flush=True)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def classify_space(phi, ref, fvals, margin=None):
    """Decide which output space the reference lives in, by minimum residual.

    Returns ``(space, resid_best, resid_proba)``.  A reference is 'proba' when
    ``ref + sum(phi)`` reproduces f(x)=P(class 1); otherwise, when a true raw
    margin vector is available, 'margin'.  sklearn forests have no margin space
    (leaf values are class fractions) so they can only ever be 'proba'.
    """
    s = ref + np.asarray(phi, float).sum(1)
    resid_proba = float(np.abs(s - np.asarray(fvals, float)).mean())
    if margin is None:
        return "proba", resid_proba, resid_proba
    resid_margin = float(np.abs(s - np.asarray(margin, float)).mean())
    if resid_proba <= resid_margin:
        return "proba", resid_proba, resid_proba
    return "margin", resid_margin, resid_proba


def raw_margin(est, key, X, idx, K):
    """Target-class raw margin from the model itself; None when no margin space exists."""
    try:
        if key == "xgboost":
            m = np.asarray(est.predict(X, output_margin=True), float)
        elif key == "lightgbm":
            m = np.asarray(est.booster_.predict(X, raw_score=True), float)
        else:
            return None
    except Exception:
        return None
    return _margin(m, idx, K)


def compare(phi_o, phi_r, base_o, base_r, fvals):
    o = np.asarray(phi_o, float); r = np.asarray(phi_r, float)
    cos_rows = [_cos(o[i], r[i]) for i in range(len(o))]
    mo, mr = np.abs(o).mean(0), np.abs(r).mean(0)
    top_k = min(3, o.shape[1])
    return dict(
        cos_mean=float(np.mean(cos_rows)),
        cos_meanabs=_cos(mo, mr),
        spearman_meanabs=float(spearmanr(mo, mr).correlation),
        sign_agree=float((np.sign(o) == np.sign(r)).mean()),
        top3_overlap=len(set(np.argsort(-mo)[:top_k]) & set(np.argsort(-mr)[:top_k])) / float(top_k),
        top1_match=float(int(np.argmax(mo)) == int(np.argmax(mr))),
        base_diff=abs(float(base_o) - float(base_r)),
        eff_err_ours=float(np.abs(o.sum(1) + base_o - np.asarray(fvals, float)).mean()),
    )


# --------------------------------------------------------------------------
# reference attributions
# --------------------------------------------------------------------------
def _slice_class(vals, ev):
    """Pull the target-class attribution matrix + its reference value out of whatever
    shap hands back (list per class / (n,d) / (n,d,K))."""
    v = vals if isinstance(vals, list) else np.asarray(vals, float)
    e = np.asarray(ev, float)
    if isinstance(v, list):
        idx = min(TARGET_CLASS, len(v) - 1)
        return np.asarray(v[idx], float), float(e.ravel()[idx]) if e.size else 0.0
    if v.ndim == 3:
        idx = min(TARGET_CLASS, v.shape[2] - 1)
        return v[:, :, idx], float(e.ravel()[idx]) if e.size else 0.0
    if v.ndim == 2 and e.size > 1:
        return v, float(e.ravel()[min(TARGET_CLASS, e.size - 1)])
    return v, float(e.ravel()[0]) if e.size else 0.0


def tree_shap(shap, est, X, bg, kind, fvals, margin):
    kw = {"data": bg} if kind == "inter" else {}
    t0 = time.perf_counter()
    exp = shap.TreeExplainer(est, **kw)
    t_init = time.perf_counter() - t0
    t1 = time.perf_counter()
    additivity_note = ""
    try:
        vals, ev = exp.shap_values(X), exp.expected_value
    except Exception as e:
        # lightgbm multiclass + interventional background trips shap's own
        # additivity assertion; retry without it and keep the fact on the row.
        additivity_note = "shap additivity check failed (%s) -- retried check_additivity=False" % type(e).__name__
        vals, ev = exp.shap_values(X, check_additivity=False), exp.expected_value
    t_attr = time.perf_counter() - t1
    phi, ref = _slice_class(vals, ev)
    space, resid, resid_p = classify_space(phi, ref, fvals, margin)
    note = additivity_note
    if space != "proba":
        try:                                   # xgb/lgbm can emit probability-space values
            exp2 = shap.TreeExplainer(est, model_output="probability", **kw)
            t2 = time.perf_counter()
            phi2, ref2 = _slice_class(exp2.shap_values(X), exp2.expected_value)
            t_attr += time.perf_counter() - t2
            sp2, r2, r2p = classify_space(phi2, ref2, fvals, margin)
            if sp2 == "proba":
                phi, ref, space, resid, resid_p = phi2, ref2, "proba", r2, r2p
                note = (note + " | " if note else "") + "via model_output=probability"
        except Exception as e:
            note = (note + " | " if note else "") + "probability output unavailable (%s)" % type(e).__name__
    return phi, ref, space, resid, resid_p, t_init, t_attr, note


def _margin(raw, idx, K):
    """Pick the target-class raw margin whether the booster returned (n,) or (n, K)."""
    raw = np.asarray(raw, float)
    if raw.ndim == 1:
        return raw
    if raw.ndim == 2 and raw.shape[1] == 2 and K == 2:
        return raw[:, 1]
    return raw[:, min(idx, raw.shape[1] - 1)]


def native_contribs(est, X, kind, K):
    """Exact C++ attributions shipped by the libraries themselves (margin space).

    Shapes measured on xgboost 3.4.1 / lightgbm 4.7.0 (the sklearn wrappers no
    longer expose ``pred_contribs``; the booster does):
      xgboost   binary (n, d+1)          multiclass (n, K, d+1)
      lightgbm  binary (n, d+1)          multiclass (n, K*(d+1)) flattened blocks
    Also returns the raw margin for the target class so the caller can verify
    ``bias + sum(phi)`` against it.
    """
    d = X.shape[1]
    idx = min(TARGET_CLASS, K - 1)
    if kind == "xgb":
        import xgboost as _xgb
        c = np.asarray(est.get_booster().predict(_xgb.DMatrix(X), pred_contribs=True), float)
        raw = np.asarray(est.get_booster().predict(_xgb.DMatrix(X), output_margin=True), float)
        if c.ndim == 2:
            phi, bias = c[:, :d], c[:, d]
            margin = _margin(raw, idx, K)
        elif c.ndim == 3 and c.shape[1] == K and c.shape[2] == d + 1:
            sl = c[:, idx, :]
            phi, bias, margin = sl[:, :d], sl[:, d], _margin(raw, idx, K)
        else:
            return None, None, None, "unexpected xgb contrib shape %s" % (c.shape,)
        return phi, float(bias.mean()), margin, "xgb contribs %s" % (c.shape,)
    c = np.asarray(est.booster_.predict(X, pred_contrib=True), float)
    raw = np.asarray(est.booster_.predict(X, raw_score=True), float)
    if c.ndim == 2 and c.shape[1] == d + 1:
        phi, bias, margin = c[:, :d], c[:, d], _margin(raw, idx, K)
    elif c.ndim == 2 and c.shape[1] == K * (d + 1):
        blk = c[:, idx * (d + 1):(idx + 1) * (d + 1)]
        phi, bias, margin = blk[:, :d], blk[:, d], _margin(raw, idx, K)
    else:
        return None, None, None, "unexpected lgb contrib shape %s" % (c.shape,)
    return phi, float(bias.mean()), margin, "lgb contribs %s" % (c.shape,)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
import shap

for dset in DSETS:
    ds = build_dataset(dset, max_rows=NTRAIN, seed=SEED)
    sp = train_test_split_dataset(ds, test_size=0.3, seed=SEED)
    Xtr = np.asarray(sp.X_train[:NTRAIN], float)
    ytr = np.asarray(sp.y_train[:NTRAIN], int)
    Xte = np.asarray(sp.X_test[:NTEST], float)
    yte = np.asarray(sp.y_test[:NTEST], int)
    Xexp = Xte[:NEXP]
    Xscale = Xte[:min(NSCALE, len(Xte))]
    bg = Xtr[:NBG]
    d = Xtr.shape[1]
    K = int(max(ytr.max(), yte.max()) + 1)
    task = "binary" if sp.train.task == TaskType.BINARY else "multiclass"
    print("== %s  task=%s  n=%d/%d  d=%d  K=%d" % (dset, task, len(Xtr), len(Xte), d, K), flush=True)

    for key in MODELS:
        planned = [lbl for lbl, _ in (
            [("TreeSHAP(path)", "path"), ("TreeSHAP(inter)", "inter")]
            + ([("xgb pred_contribs", "xgb")] if key == "xgboost" else [])
            + ([("lgb pred_contrib", "lgb")] if key == "lightgbm" else []))]
        if all((dset, key, lbl) in done for lbl in planned):
            print("   skip %s (all rows present)" % key, flush=True)
            continue

        model = make_model(key, sp.train.task, sp.feature_names, sp.train.class_names)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        t_fit = time.perf_counter() - t0
        fvals = EX.make_model_fn(model, TARGET_CLASS)(Xexp)
        acc = float((np.asarray(model.predict(Xte)) == yte).mean())

        try:
            # method="kernel" pins the sampled estimator: this study compares our
            # sampling approximation against the exact references, and shap_explain
            # now defaults to "auto" (exact for xgboost/lightgbm).
            t0 = time.perf_counter()
            ours = EX.shap_explain(model, Xexp, bg, target_class=TARGET_CLASS,
                                   seed=SEED, method="kernel")
            t_ours = time.perf_counter() - t0
            phi_o = np.asarray(ours.shap_values, float)
            base_o = float(ours.base_value)
            t0 = time.perf_counter()
            EX.shap_explain(model, Xscale, bg, target_class=TARGET_CLASS,
                            seed=SEED, method="kernel")
            t_ours_scale = time.perf_counter() - t0
        except Exception as e:
            write(dict(dataset=dset, model=key, task=task, n_train=len(Xtr), n_test=len(Xte),
                       d=d, n_exp=NEXP, n_scale=len(Xscale), ref_impl="ours EX.shap_explain",
                       space="proba", note="OURS FAIL %s: %s" % (type(e).__name__, e)))
            print("   OURS FAIL %s: %s" % (key, e), flush=True)
            traceback.print_exc(limit=3)
            continue

        try:
            n_trees = float(len(model.estimator.estimators_))
        except Exception:
            n_trees = float(getattr(model.estimator, "n_estimators", np.nan))
        base = dict(dataset=dset, model=key, task=task, n_train=len(Xtr), n_test=len(Xte),
                    d=d, n_exp=NEXP, n_scale=len(Xscale),
                    t_ours_s=t_ours, t_ours_scale_s=t_ours_scale,
                    ours_ms_per_sample=t_ours / len(Xexp) * 1000.0,
                    ours_ms_per_sample_scale=t_ours_scale / len(Xscale) * 1000.0,
                    n_trees=n_trees,
                    ours_coalitions=float(int(min(2 ** d, max(2 * d + 8, 48)))),
                    ours_fwd_per_sample=float(int(min(2 ** d, max(2 * d + 8, 48))) * min(NBG, 8)),
                    eff_err_ours=float(np.abs(phi_o.sum(1) + base_o - fvals).mean()),
                    note="fit %.2fs acc %.4f" % (t_fit, acc))

        refs = [("TreeSHAP(path)", "path"), ("TreeSHAP(inter)", "inter")]
        if key == "xgboost":
            refs.append(("xgb pred_contribs", "xgb"))
        if key == "lightgbm":
            refs.append(("lgb pred_contrib", "lgb"))

        idx = min(TARGET_CLASS, K - 1)
        marg = raw_margin(model.estimator, key, Xexp, idx, K)

        for label, kind in refs:
            if (dset, key, label) in done:
                continue
            rec = dict(base); rec["ref_impl"] = label
            try:
                if label.startswith("TreeSHAP"):
                    phi_r, ref_r, space, resid, resid_p, t_init, t_attr, extra = tree_shap(
                        shap, model.estimator, Xexp, bg, kind, fvals, marg)
                else:
                    t_init = 0.0
                    t0 = time.perf_counter()
                    phi_r, ref_r, margin_r, shape_note = native_contribs(
                        model.estimator, Xexp, kind, K)
                    t_attr = time.perf_counter() - t0
                    extra = shape_note
                    if phi_r is None:
                        rec["note"] = base["note"] + " | " + shape_note + " -- skipped"
                        write(rec); continue
                    space, resid, resid_p = classify_space(phi_r, ref_r, fvals, margin_r)
                    rec["ref_margin_err"] = float(np.abs(
                        np.asarray(phi_r, float).sum(1) + ref_r
                        - np.asarray(margin_r, float)).mean())
                    marg = margin_r if margin_r is not None else marg
                if phi_r is None or phi_r.shape != phi_o.shape:
                    rec.update(space=str(space),
                               note=base["note"] + " | ref shape %s vs ours %s -- skipped"
                               % (getattr(phi_r, "shape", None), phi_o.shape))
                    write(rec); continue
                rec.update(compare(phi_o, phi_r, base_o, ref_r, fvals))
                rec["space"] = space
                rec["eff_err_ref"] = resid
                rec["eff_err_ref_proba"] = resid_p
                rec["t_ref_s"] = t_attr
                rec["t_ref_init_s"] = t_init
                rec["ref_ms_per_sample"] = t_attr / len(Xexp) * 1000.0
                try:
                    t0 = time.perf_counter()
                    if label.startswith("TreeSHAP"):
                        e2 = shap.TreeExplainer(model.estimator,
                                                **({"data": bg} if kind == "inter" else {}))
                        e2.shap_values(Xscale)
                    else:
                        native_contribs(model.estimator, Xscale, kind, K)
                    t_rs = time.perf_counter() - t0
                    rec["t_ref_scale_s"] = t_rs
                    rec["ref_ms_per_sample_scale"] = t_rs / len(Xscale) * 1000.0
                except Exception as e:
                    rec["note"] = base["note"] + " | scale ref fail %s" % type(e).__name__
                if extra:
                    rec["note"] = str(rec["note"]) + " | " + extra
                if rec.get("ours_ms_per_sample_scale") and rec.get("ref_ms_per_sample_scale"):
                    rec["ratio_ms_per_sample"] = rec["ours_ms_per_sample_scale"] / rec["ref_ms_per_sample_scale"]
                    rec["note"] = str(rec["note"]) + " | ratio@n=%d" % len(Xscale)
                elif rec.get("ours_ms_per_sample") and rec.get("ref_ms_per_sample"):
                    rec["ratio_ms_per_sample"] = rec["ours_ms_per_sample"] / rec["ref_ms_per_sample"]
                    rec["note"] = str(rec["note"]) + " | ratio@n=%d" % len(Xexp)
                write(rec)
            except Exception as e:
                rec["note"] = "REF FAIL %s: %s" % (type(e).__name__, e)
                write(rec)
                print("   ref fail %s/%s: %s" % (key, label, e), flush=True)
                traceback.print_exc(limit=3)

fh.close()
print("done ->", OUT, flush=True)
