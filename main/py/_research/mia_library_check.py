"""Verify the library membership-inference build against the research readings.

Runs trustoran.trust.privacy.membership_inference on the same targets used by
mia_variants.py (unprotected forest, positive-control forest, no-DP net, DP nets)
and prints every attack AUC plus the diagnostics, with elapsed time.
Reads only -- nothing here rewrites the archived results/.
"""
import os, sys, time, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
from trustoran.data import build_dataset, train_test_split_dataset
from trustoran.data.base import TaskType
from trustoran.models import make_model
from trustoran.trust import privacy as PR

SEED = 42
DSETS = os.environ.get("MIA_DSETS", "cci_a1sd,cci_a1sd_binary,vran_computing,ddos_pcap").split(",")
NTRAIN = int(os.environ.get("MIA_NTRAIN", 4200))
NTEST = 1800
N_SMALL = 300
N_SHADOW = int(os.environ.get("MIA_SHADOW", 8))
EPS_LIST = [float(x) for x in os.environ.get("MIA_EPS", "1.0,0.1").split(",")]


class _Proxy:
    """membership_inference reads .estimator; DP targets are bare NumpyNet objects."""

    def __init__(self, est):
        self.estimator = est


def show(tag, r, elapsed):
    a = r.attack_aucs or {}
    print("  %-16s AUC=%.4f acc=%.4f n=%d/%d | thr=%.4f conf_only=%.4f ssa=%s lira=%s" % (
        tag, r.attack_auc, r.attack_accuracy, r.n_members, r.n_nonmembers,
        a.get("threshold", float("nan")), a.get("lr_conf", float("nan")),
        "%.4f" % a["ssa"] if "ssa" in a else "  --  ",
        "%.4f" % a["lira"] if "lira" in a else "  --  "), flush=True)
    print("      diag: KS=%.4f TPR@1%%=%.3f conf_gap=%.4f gen_gap=%.4f dup=%.4f shadows=%d (%.1fs)" % (
        r.ks_stat, r.tpr_at_1pct_fpr, r.confidence_gap, r.generalization_gap,
        r.near_duplicate_rate, r.n_shadows, elapsed), flush=True)
    print("      note: %s" % r.note, flush=True)


for dk in DSETS:
    t0 = time.perf_counter()
    ds = build_dataset(dk, max_rows=NTRAIN + NTEST, seed=SEED)
    sp = train_test_split_dataset(ds, test_size=0.3, seed=SEED)
    task = sp.train.task
    if task == TaskType.REGRESSION:
        print("%s: regression, skipped" % dk, flush=True)
        continue
    K = len(sp.train.class_names)
    Xtr, ytr = np.asarray(sp.X_train), np.asarray(sp.y_train).astype(int)
    Xte, yte = np.asarray(sp.X_test), np.asarray(sp.y_test).astype(int)
    print("%s n=%d/%d d=%d K=%d %s" % (dk, len(Xtr), len(Xte), Xtr.shape[1], K,
                                       sp.train.class_names), flush=True)

    # ---- target 1: unprotected forest on the full train set
    rf = make_model("random_forest", task, sp.feature_names, sp.train.class_names)
    rf.fit(Xtr, ytr)
    ts = time.perf_counter()
    show("random_forest", PR.membership_inference(
        rf, Xtr, ytr, Xte, yte, seed=SEED, n_shadows=N_SHADOW), time.perf_counter() - ts)

    # ---- target 2: positive control, a forest that memorises a small train set
    Xs, ys = Xtr[:N_SMALL], ytr[:N_SMALL]
    rfo = make_model("random_forest", task, sp.feature_names, sp.train.class_names)
    rfo.fit(Xs, ys)
    ts = time.perf_counter()
    show("rf_overfit", PR.membership_inference(
        rfo, Xs, ys, Xte, yte, seed=SEED, n_shadows=N_SHADOW), time.perf_counter() - ts)

    # ---- DP targets through the bare-net adapter (no clone path -> shadows skipped)
    def nn_est(X, y):
        m = PR.NumpyNet(X.shape[1], 24, K, seed=SEED)
        m.fit(X, y, epochs=25, lr=0.15, batch=64, clip=None, sigma=0.0, seed=SEED)
        return m

    ts = time.perf_counter()
    show("numpy_nodp", PR.membership_inference(
        _Proxy(nn_est(Xtr, ytr)), Xtr, ytr, Xte, yte, seed=SEED, n_shadows=N_SHADOW),
        time.perf_counter() - ts)
    for eps in EPS_LIST:
        net = PR.dp_train_classification(Xtr, ytr, K, eps, delta=1e-5, hidden=24,
                                         epochs=25, seed=SEED)
        ts = time.perf_counter()
        show("dp_eps%g" % eps, PR.membership_inference(
            _Proxy(net), Xtr, ytr, Xte, yte, seed=SEED, n_shadows=N_SHADOW),
            time.perf_counter() - ts)

    print("  %s total %.1fs" % (dk, time.perf_counter() - t0), flush=True)

print("DONE", flush=True)
