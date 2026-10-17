"""Membership-inference attack variants: why the shipped attack reads ~0.5, and what fixes it.

Targets: random_forest (unprotected), a deliberately overfit forest (positive control), and
NumpyNet with/without DP-SGD.
Attacks: shipped confidence-only, single-feature threshold, label-aware feature sets,
shadow-model SSA, shadow-model LiRA, and the same under near-duplicate removal.
Writes CSV incrementally; skip-existing-keys makes it resumable.
"""
import os, sys, time, csv, warnings, traceback
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from trustoran.data import build_dataset, train_test_split_dataset
from trustoran.data.base import TaskType
from trustoran.models import make_model
from trustoran.trust import privacy as PR

SEED = 42
DSETS = os.environ.get("MIA_DSETS", "cci_a1sd,cci_a1sd_binary,vran_computing").split(",")
NTRAIN = int(os.environ.get("MIA_NTRAIN", 4200))
NTEST = 1800
N_SMALL = int(os.environ.get("MIA_NSMALL", 300))
N_SHADOW = 8
AUX_FRAC = 0.5
EPS_LIST = [8.0, 1.0, 0.1]
OUT = os.path.join(HERE, "mia_variants.csv")

CONF = ["conf", "ent", "margin"]
LABEL = ["p_true", "nll", "minp", "pdg"]
ALL = CONF + LABEL
COLS = ["dataset", "target", "attack", "feature_set", "dedup", "n_mem", "n_non",
        "auc", "auc_best_dir", "acc", "ks", "signal_pp", "elapsed_s", "note"]

done = set()
if os.path.exists(OUT):
    with open(OUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["dataset"], row["target"], row["attack"], row["feature_set"], row["dedup"]))

fh = open(OUT, "a", newline="", encoding="utf-8")
wr = csv.DictWriter(fh, fieldnames=COLS)
if not done:
    wr.writeheader()


def write(rec):
    key = (rec["dataset"], rec["target"], rec["attack"], rec["feature_set"], rec["dedup"])
    if key in done:
        return
    done.add(key)
    wr.writerow({c: rec.get(c, "") for c in COLS})
    fh.flush()
    print("%-17s %-12s %-14s %-22s dd=%-3s auc=%.4f best=%.4f gap=%-6s %s" % (
        rec["dataset"], rec["target"], rec["attack"], rec["feature_set"], rec["dedup"],
        float(rec["auc"] or 0), float(rec["auc_best_dir"] or 0), rec["signal_pp"],
        rec["note"][:70]), flush=True)


class _Proxy:
    """membership_inference reads model.estimator; DP targets are bare NumpyNet objects."""

    def __init__(self, est):
        self.estimator = est


def proba(est, X):
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(X), dtype=float)
    d = np.asarray(est.decision_function(X), dtype=float)
    if d.ndim == 1:
        d = np.stack([-d, d], axis=1)
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def features(p, y, K):
    """Named attack features. Confidence-only ones ignore y; label-aware ones need it."""
    s = np.sort(p, axis=1)
    conf = s[:, -1]
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    margin = s[:, -1] - s[:, -2]
    p_true = p[np.arange(len(p)), y]
    nll = -np.log(p_true + 1e-12)
    minp = -np.log(p.min(axis=1) + 1e-12)
    pdg = conf - p_true
    return {"conf": conf, "ent": ent, "margin": margin, "p_true": p_true,
            "nll": nll, "minp": minp, "pdg": pdg}


def dup_stats(Xtr, Xte):
    keys_tr = {tuple(np.round(r, 9)) for r in Xtr}
    hits = sum(1 for r in Xte if tuple(np.round(r, 9)) in keys_tr)
    both = np.vstack([Xtr, Xte])
    uniq = len({tuple(r) for r in np.round(both, 9)})
    return hits, len(Xte), 1.0 - uniq / len(both)


def auc_of(score, ya):
    try:
        return float(roc_auc_score(ya, score))
    except ValueError:
        return float("nan")


def ks_stat(v1, v0):
    """Two-sample KS statistic: max |ECDF_members - ECDF_nonmembers|, direction agnostic."""
    v1 = np.sort(np.asarray(v1, dtype=float)); v0 = np.sort(np.asarray(v0, dtype=float))
    allv = np.concatenate([v1, v0])
    c1 = np.searchsorted(v1, allv, side="right") / len(v1)
    c0 = np.searchsorted(v0, allv, side="right") / len(v0)
    return float(np.max(np.abs(c1 - c0)))


def make_rec(dataset, target, attack, names, dedup, nm, nn, auc, acc, note, elapsed="", ks=None):
    fs = "+".join(names) if isinstance(names, (list, tuple)) else str(names)
    return {"dataset": dataset, "target": target, "attack": attack, "feature_set": fs,
            "dedup": dedup, "n_mem": int(nm), "n_non": int(nn),
            "auc": "" if auc is None else "%.5f" % auc,
            "auc_best_dir": "" if auc is None else "%.5f" % max(auc, 1 - auc),
            "acc": "" if acc is None else "%.5f" % acc,
            "ks": "" if ks is None else "%.5f" % ks,
            "signal_pp": "" if auc is None else "%.2f" % (abs(auc - 0.5) * 100),
            "elapsed_s": elapsed, "note": note}


def shipped(model, Xtr, ytr, Xte, yte, dataset, target):
    r = PR.membership_inference(model, Xtr, ytr, Xte, yte, seed=SEED)
    write(make_rec(dataset, target, "shipped", CONF, "no", r.n_members, r.n_nonmembers,
                   float(r.attack_auc), float(r.attack_accuracy),
                   "library implementation, unchanged"))


def threshold_attacks(Fm, Fn, dataset, target, dedup, names):
    for nm in names:
        v1 = np.asarray(Fm[nm], dtype=float); v0 = np.asarray(Fn[nm], dtype=float)
        ya = np.concatenate([np.ones(len(v1)), np.zeros(len(v0))])
        a = auc_of(np.concatenate([v1, v0]), ya)
        q = np.quantile(v0, 0.99)
        tpr = max(float(np.mean(v1 > q)), float(np.mean(v1 < np.quantile(v0, 0.01))))
        write(make_rec(dataset, target, "threshold1", nm, dedup, len(v1), len(v0), a, None,
                       "mean mem=%.5f non=%.5f sd=%.5f/%.5f tpr@fpr1%%=%.3f" % (
                           float(v1.mean()), float(v0.mean()), float(v1.std()),
                           float(v0.std()), tpr), ks=ks_stat(v1, v0)))


def lr_meta(Fm, Fn, names, dataset, target, dedup, seed=SEED, tag="lr_meta"):
    Xa = np.vstack([_stack(Fm, names), _stack(Fn, names)])
    ya = np.concatenate([np.ones(len(Fm[names[0]])), np.zeros(len(Fn[names[0]]))])
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.4, random_state=seed, stratify=ya)
    atk = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    prob = atk.predict_proba(Xte)[:, 1]
    a = auc_of(prob, yte)
    write(make_rec(dataset, target, tag, names, dedup, int(ytr.sum()),
                   int(len(ytr) - ytr.sum()), a,
                   float(accuracy_score(yte, atk.predict(Xte))),
                   "n_attack=%d held-out" % len(yte)))


def _stack(F, names):
    return np.column_stack([F[n] for n in names])


def build_shadows(pool_X, pool_y, K, names, est_fn, seed):
    """Shadow models trained on random halves of an auxiliary pool.

    Returns (shadow member feats, shadow non-member feats) as column stacks, plus the
    per-shadow index splits so LiRA can normalise within its own shadow.
    """
    rng = np.random.default_rng(seed)
    mem, non, splits = [], [], []
    for j in range(N_SHADOW):
        idx = rng.permutation(len(pool_X))
        half = len(idx) // 2
        tr, te = idx[:half], idx[half:]
        est = est_fn(pool_X[tr], pool_y[tr])
        fm = features(proba(est, pool_X[tr]), pool_y[tr], K)
        fn = features(proba(est, pool_X[te]), pool_y[te], K)
        mem.append(_stack(fm, names))
        non.append(_stack(fn, names))
        splits.append((tr, te))
    return mem, non, splits, names


def _cols(stack_list, order, names):
    idx = [order.index(n) for n in names]
    return [S[:, idx] for S in stack_list]


def ssa(mem, non, Fm, Fn, names, dataset, target, dedup, order=ALL):
    mem = _cols(mem, order, names); non = _cols(non, order, names)
    Xs = np.vstack(mem + non)
    ys = np.concatenate([np.ones(len(np.vstack(mem))), np.zeros(len(np.vstack(non)))])
    meta = LogisticRegression(max_iter=2000).fit(Xs, ys)
    pm = meta.predict_proba(_stack(Fm, names))[:, 1]
    pn = meta.predict_proba(_stack(Fn, names))[:, 1]
    a = auc_of(np.concatenate([pm, pn]),
               np.concatenate([np.ones(len(pm)), np.zeros(len(pn))]))
    write(make_rec(dataset, target, "ssa%d" % N_SHADOW, names, dedup, len(pm), len(pn), a,
                   float(accuracy_score(np.concatenate([np.ones(len(pm)), np.zeros(len(pn))]),
                                        np.concatenate([pm, pn]) >= 0.5)),
                   "shadow meta, acc_on_shadow=%.4f" % float(np.mean(meta.predict(Xs) == ys))))


def lira(mem, non, Fm, Fn, names, dataset, target, dedup, order=ALL):
    """Likelihood-ratio attack with per-shadow Gaussian fits, one feature at a time.

    Isolated LiRA: for each shadow j, estimate mu/sd of the feature on j's own member and
    non-member halves, then score the target by the averaged log-likelihood ratio.
    """
    mem = _cols(mem, order, names); non = _cols(non, order, names)
    scores = {}
    for j, nm in enumerate(names):
        s_m, s_n = _per_feature_lira(mem, non, j, Fm[nm], Fn[nm])
        ya = np.concatenate([np.ones(len(s_m)), np.zeros(len(s_n))])
        a = auc_of(np.concatenate([s_m, s_n]), ya)
        write(make_rec(dataset, target, "lira1", nm, dedup, len(s_m), len(s_n), a, None,
                       "mean lr mem=%.3f non=%.3f" % (float(s_m.mean()), float(s_n.mean())),
                       ks=ks_stat(s_m, s_n)))
        scores[nm] = (s_m, s_n)
    s_m = np.mean([scores[n][0] for n in names], axis=0)
    s_n = np.mean([scores[n][1] for n in names], axis=0)
    ya = np.concatenate([np.ones(len(s_m)), np.zeros(len(s_n))])
    a = auc_of(np.concatenate([s_m, s_n]), ya)
    write(make_rec(dataset, target, "lira_avg", names, dedup, len(s_m), len(s_n), a, None,
                   "mean of per-feature log-LRs"))


def _per_feature_lira(mem, non, k, vm, vn):
    from scipy.special import logsumexp
    m1 = np.array([M[:, k] for M in mem])
    m0 = np.array([N[:, k] for N in non])
    # features can be near-constant within one shadow (pdg is 0 for most samples), so the
    # per-shadow sigma is floored by a fraction of the pooled spread
    floor = 1e-6 + 0.05 * (np.concatenate([m1.ravel(), m0.ravel()]).std() + 1e-12)
    mu1 = m1.mean(1); sd1 = np.maximum(m1.std(1), floor)
    mu0 = m0.mean(1); sd0 = np.maximum(m0.std(1), floor)

    def lr(v):
        v = np.asarray(v)
        l1 = -0.5 * ((v[:, None] - mu1[None, :]) / sd1[None, :]) ** 2 - np.log(sd1)[None, :]
        l0 = -0.5 * ((v[:, None] - mu0[None, :]) / sd0[None, :]) ** 2 - np.log(sd0)[None, :]
        return logsumexp(l1 - np.log(len(mu1)), axis=1) - logsumexp(l0 - np.log(len(mu0)), axis=1)
    return lr(vm), lr(vn)


def attacks_for(Fm, Fn, dataset, target, dedup, model_obj=None, Xtr=None, ytr=None,
                Xte=None, yte=None):
    if model_obj is not None:
        shipped(model_obj, Xtr, ytr, Xte, yte, dataset, target)
    threshold_attacks(Fm, Fn, dataset, target, dedup, ALL)
    lr_meta({k: Fm[k] for k in CONF}, {k: Fn[k] for k in CONF}, CONF, dataset, target, dedup)
    lr_meta({k: Fm[k] for k in LABEL}, {k: Fn[k] for k in LABEL}, LABEL, dataset, target, dedup)
    lr_meta(Fm, Fn, ALL, dataset, target, dedup)


for dk in DSETS:
    t0 = time.perf_counter()
    ds = build_dataset(dk, max_rows=NTRAIN + NTEST, seed=SEED)
    sp = train_test_split_dataset(ds, test_size=0.3, seed=SEED)
    task = sp.train.task
    if task == TaskType.REGRESSION:
        print("skip regression", dk, flush=True)
        continue
    cls = sp.train.class_names
    K = len(cls)
    Xtr, ytr = np.asarray(sp.X_train), np.asarray(sp.y_train).astype(int)
    Xte, yte = np.asarray(sp.X_test), np.asarray(sp.y_test).astype(int)
    dup, nte, duprate = dup_stats(Xtr, Xte)
    print("%s n=%d/%d d=%d K=%d exact_dup=%d/%d pool_dup=%.4f" % (
        dk, len(Xtr), len(Xte), Xtr.shape[1], K, dup, nte, duprate), flush=True)

    def rf_est(X, y):
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=120, random_state=SEED, n_jobs=1)
        m.fit(X, y)
        return m

    # ---- target 1: unprotected random forest on the full train set
    rf = make_model("random_forest", task, sp.feature_names, cls)
    rf.fit(Xtr, ytr)
    a_tr = float(np.mean(rf.estimator.predict(Xtr) == ytr))
    a_te = float(np.mean(rf.estimator.predict(Xte) == yte))
    write({"dataset": dk, "target": "random_forest", "attack": "generalization",
           "feature_set": "", "dedup": "no", "n_mem": len(Xtr), "n_non": len(Xte),
           "auc": "", "auc_best_dir": "", "acc": "",
           "signal_pp": "%.2f" % ((a_tr - a_te) * 100), "elapsed_s": "",
           "note": "train_acc=%.4f test_acc=%.4f gap=%.4f exact_dup=%.4f pool_dup=%.4f" % (
               a_tr, a_te, a_tr - a_te, dup / max(1, nte), duprate)})
    pm = proba(rf.estimator, Xtr); pn = proba(rf.estimator, Xte)
    Fm = features(pm, ytr, K); Fn = features(pn, yte, K)
    attacks_for(Fm, Fn, dk, "random_forest", "no", rf, Xtr, ytr, Xte, yte)

    # ---- target 2: positive control, a forest that memorises a small train set
    Xs, ys = Xtr[:N_SMALL], ytr[:N_SMALL]
    rfo = make_model("random_forest", task, sp.feature_names, cls)
    rfo.fit(Xs, ys)
    o_tr = float(np.mean(rfo.estimator.predict(Xs) == ys))
    o_te = float(np.mean(rfo.estimator.predict(Xte) == yte))
    write({"dataset": dk, "target": "rf_overfit", "attack": "generalization",
           "feature_set": "", "dedup": "no", "n_mem": len(Xs), "n_non": len(Xte),
           "auc": "", "auc_best_dir": "", "acc": "",
           "signal_pp": "%.2f" % ((o_tr - o_te) * 100), "elapsed_s": "",
           "note": "train_acc=%.4f test_acc=%.4f gap=%.4f n_train=%d" % (
               o_tr, o_te, o_tr - o_te, len(Xs))})
    pmO = proba(rfo.estimator, Xs); pnO = proba(rfo.estimator, Xte)
    FM = features(pmO, ys, K); FN = features(pnO, yte, K)
    attacks_for(FM, FN, dk, "rf_overfit", "no", _Proxy(rfo), Xs, ys, Xte, yte)

    # ---- shadows for SSA/LiRA (RF family), built once and reused for both RF targets
    aux = int(len(Xte) * AUX_FRAC)
    pool_X, pool_y = Xte[:aux], yte[:aux]
    ev_X, ev_y = Xte[aux:], yte[aux:]
    sm, sn, _, _ = build_shadows(pool_X, pool_y, K, ALL, rf_est, SEED)
    pe = proba(rfo.estimator, ev_X)
    FE = features(pe, ev_y, K)
    ssa(sm, sn, FM, FE, CONF, dk, "rf_overfit", "no")
    ssa(sm, sn, FM, FE, ALL, dk, "rf_overfit", "no")
    lira(sm, sn, FM, FE, ALL, dk, "rf_overfit", "no")
    pe2 = proba(rf.estimator, ev_X)
    FE2 = features(pe2, ev_y, K)
    ssa(sm, sn, Fm, FE2, CONF, dk, "random_forest", "no")
    ssa(sm, sn, Fm, FE2, ALL, dk, "random_forest", "no")
    lira(sm, sn, Fm, FE2, ALL, dk, "random_forest", "no")
    print("  rf+shadow block %.1fs" % (time.perf_counter() - t0), flush=True)

    # ---- near-duplicate removal on the unprotected forest
    keys_tr = {tuple(np.round(r, 9)) for r in Xtr}
    keep = np.array([tuple(np.round(r, 9)) not in keys_tr for r in Xte])
    if keep.sum() > 50 and (~keep).sum() > 0:
        Fn2 = features(proba(rf.estimator, Xte[keep]), yte[keep], K)
        threshold_attacks(Fm, Fn2, dk, "random_forest", "yes", CONF)
        lr_meta({k: Fm[k] for k in CONF}, {k: Fn2[k] for k in CONF}, CONF,
                dk, "random_forest", "yes")
        print("  dedup removed %d/%d test rows" % (len(keep) - int(keep.sum()), len(keep)), flush=True)

    # ---- DP targets: same architecture, only the mechanism differs
    def nn_est(X, y):
        m = PR.NumpyNet(X.shape[1], 24, K, seed=SEED)
        m.fit(X, y, epochs=25, lr=0.15, batch=64, clip=None, sigma=0.0, seed=SEED)
        return m

    smn, snn, _, _ = build_shadows(pool_X, pool_y, K, ALL, nn_est, SEED)
    for tag, eps in [("numpy_nodp", None)] + [("dp_eps%g" % e, e) for e in EPS_LIST]:
        ts = time.perf_counter()
        if eps is None:
            net = nn_est(Xtr, ytr)
        else:
            net = PR.dp_train_classification(Xtr, ytr, K, eps, delta=1e-5, hidden=24,
                                             epochs=25, seed=SEED)
        dpm = proba(net, Xtr); dpn = proba(net, Xte)
        Gm = features(dpm, ytr, K); Gn = features(dpn, yte, K)
        acc_m = float(np.mean(dpm.argmax(1) == ytr)); acc_n = float(np.mean(dpn.argmax(1) == yte))
        write({"dataset": dk, "target": tag, "attack": "generalization", "feature_set": "",
               "dedup": "no", "n_mem": len(Xtr), "n_non": len(Xte), "auc": "",
               "auc_best_dir": "", "acc": "", "signal_pp": "%.2f" % ((acc_m - acc_n) * 100),
               "elapsed_s": "",
               "note": "train_acc=%.4f test_acc=%.4f gap=%.4f" % (acc_m, acc_n, acc_m - acc_n)})
        attacks_for(Gm, Gn, dk, tag, "no", _Proxy(net), Xtr, ytr, Xte, yte)
        pe = proba(net, ev_X)
        FE = features(pe, ev_y, K)
        ssa(smn, snn, Gm, FE, CONF, dk, tag, "no")
        ssa(smn, snn, Gm, FE, ALL, dk, tag, "no")
        lira(smn, snn, Gm, FE, ALL, dk, tag, "no")
        write(make_rec(dk, tag, "timing", "", "no", 0, 0, None, None,
                       "fit+attacks %.2fs" % (time.perf_counter() - ts)))
    print("%s done %.1fs" % (dk, time.perf_counter() - t0), flush=True)

fh.close()
print("DONE ->", OUT, flush=True)
