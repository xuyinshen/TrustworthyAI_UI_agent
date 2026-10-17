"""Privacy pillar.

Three complementary demonstrations, all implemented in pure numpy so they run
anywhere:

* **Differential privacy** -- a small DP-SGD softmax network (per-sample
  gradient clipping + Gaussian mechanism). Sweeping epsilon yields a
  privacy/utility curve.
* **Federated learning** -- FedAvg across the 5 Near-RT-RIC agents of the DDoS
  dataset: the global model is trained without any agent's raw captures leaving
  that agent. Compared against centralised and local-only baselines.
* **Membership inference** -- a group of attacks (confidence threshold, label-aware
  meta-model, shadow-model SSA and LiRA) that quantifies how much a released model
  leaks about the records it was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from ..data.base import TaskType
from ..models.wrappers import TrustModel
from .results import DPResult, FedAvgResult, MembershipInferenceResult


# ==========================================================================
# a tiny numpy softmax network supporting DP-SGD and FedAvg
# ==========================================================================
class NumpyNet:
    """2-layer ReLU MLP (input -> hidden -> softmax) with manual SGD.

    Exposes ``get_params`` / ``set_params`` so weights can be averaged across
    federated clients, and an optional DP-SGD training loop (per-sample gradient
    clipping + calibrated Gaussian noise).
    """

    def __init__(self, d: int, hidden: int, K: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / d)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, s1, (d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s2, (hidden, K))
        self.b2 = np.zeros(K)
        self.d, self.h, self.K = d, hidden, K

    # -- forward ------------------------------------------------------------
    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        z2 -= z2.max(axis=1, keepdims=True)
        exp = np.exp(z2)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return z1, a1, probs

    def predict_proba(self, X):
        return self._forward(X)[2]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # -- params -------------------------------------------------------------
    def get_params(self) -> List[np.ndarray]:
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_params(self, params: Sequence[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in params]

    # -- per-sample gradients ----------------------------------------------
    def _sample_grads(self, X, y_onehot):
        """Return *per-sample* gradients (no batch averaging); DP-SGD clips and
        averages them itself."""
        z1, a1, probs = self._forward(X)
        dz2 = (probs - y_onehot)                           # (n,K)
        dW2 = np.einsum("nh,nk->nhk", a1, dz2)             # (n,h,K)
        db2 = dz2                                           # (n,K)
        da1 = dz2 @ self.W2.T                              # (n,h)
        dz1 = da1 * (z1 > 0)                               # (n,h)
        dW1 = np.einsum("nd,nh->ndh", X, dz1)              # (n,d,h)
        db1 = dz1                                           # (n,h)
        return dW1, db1, dW2, db2

    def _step(self, Xb, yb, lr, clip, sigma, rng):
        y_onehot = np.zeros((len(yb), self.K))
        y_onehot[np.arange(len(yb)), yb.astype(int)] = 1.0
        dW1, db1, dW2, db2 = self._sample_grads(Xb, y_onehot)   # per-sample
        n = len(yb)
        if clip is not None and clip > 0:
            # clip each per-sample gradient to L2 norm <= clip
            flat = np.concatenate([
                dW1.reshape(n, -1), db1, dW2.reshape(n, -1), db2
            ], axis=1)
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            scale = np.minimum(1.0, clip / (norms + 1e-12))
            flat = flat * scale
            c1 = self.W1.size; cb1 = self.h; c2 = self.W2.size
            dW1 = flat[:, :c1].sum(0).reshape(self.W1.shape)
            db1 = flat[:, c1:c1 + cb1].sum(0).reshape(self.b1.shape)
            dW2 = flat[:, c1 + cb1:c1 + cb1 + c2].sum(0).reshape(self.W2.shape)
            db2 = flat[:, c1 + cb1 + c2:].sum(0).reshape(self.b2.shape)
            if sigma and sigma > 0:
                # Gaussian mechanism: noise added ONCE to the summed gradient
                dW1 = dW1 + rng.normal(0, sigma * clip, self.W1.shape)
                db1 = db1 + rng.normal(0, sigma * clip, self.b1.shape)
                dW2 = dW2 + rng.normal(0, sigma * clip, self.W2.shape)
                db2 = db2 + rng.normal(0, sigma * clip, self.b2.shape)
        else:
            dW1, db1, dW2, db2 = dW1.sum(0), db1.sum(0), dW2.sum(0), db2.sum(0)
        # average over the batch -> the update direction
        self.W1 -= lr * dW1 / n
        self.b1 -= lr * db1 / n
        self.W2 -= lr * dW2 / n
        self.b2 -= lr * db2 / n

    def fit(self, X, y, epochs=20, lr=0.1, batch=64, clip=None, sigma=0.0, seed=42):
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                if len(b) < 2:
                    continue
                self._step(X[b], y[b], lr, clip, sigma, rng)
        return self

    def fit_local(self, X, y, epochs, lr, batch, seed=42):
        """Noiseless local training used inside FedAvg."""
        return self.fit(X, y, epochs=epochs, lr=lr, batch=batch, clip=None, sigma=0.0, seed=seed)


def _onehot(y, K):
    o = np.zeros((len(y), K))
    o[np.arange(len(y)), y.astype(int)] = 1.0
    return o


# ==========================================================================
# differential privacy
# ==========================================================================
def _gaussian_sigma(eps: float, delta: float, clip: float, n_steps: int, batch: int) -> float:
    """Simple analytic-Gaussian composition bound (conservative, not RDP-tight).

    sensitivity of the averaged clipped gradient = clip / batch.
    """
    if eps <= 0:
        return 1e6
    sens = clip / batch
    return sens * np.sqrt(2.0 * n_steps * np.log(1.25 / delta)) / eps


def dp_train_classification(
    X: np.ndarray, y: np.ndarray, K: int, eps: float, delta: float = 1e-5,
    hidden: int = 24, epochs: int = 25, lr: float = 0.15, batch: int = 64,
    clip: float = 1.0, seed: int = 42,
) -> NumpyNet:
    """Train a softmax net with eps-DP SGD."""
    n_steps = max(1, (len(X) // batch) * epochs)
    sigma = _gaussian_sigma(eps, delta, clip, n_steps, batch) if eps and eps > 0 else 0.0
    net = NumpyNet(X.shape[1], hidden, K, seed=seed)
    net.fit(X, y, epochs=epochs, lr=lr, batch=batch, clip=clip, sigma=sigma, seed=seed)
    return net


def dp_train_regression(
    X: np.ndarray, y: np.ndarray, eps: float, delta: float = 1e-5,
    clip: float = 1.0, seed: int = 42,
) -> Ridge:
    """Input-perturbation DP proxy for regression: add calibrated Gaussian noise
    to the (standardised) training features and refit a ridge model."""
    rng = np.random.default_rng(seed)
    if eps and eps > 0:
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * clip / eps
        Xn = X + rng.normal(0, sigma, X.shape)
    else:
        Xn = X
    return Ridge(alpha=1.0).fit(Xn, y)


def privacy_utility_curve(
    X_train, y_train, X_test, y_test, task: TaskType,
    eps_range=(float("inf"), 8.0, 4.0, 2.0, 1.0, 0.5, 0.1),
    delta: float = 1e-5, hidden: int = 24, epochs: int = 25, seed: int = 42,
) -> DPResult:
    """Utility (accuracy / R2) as a function of the privacy budget epsilon."""
    eps_arr = np.asarray([np.inf if e == float("inf") else e for e in eps_range], float)
    utils = []
    if task.is_classification:
        K = int(max(len(np.unique(y_train)), y_train.max() + 1))
        for e in eps_arr:
            if np.isinf(e):
                net = NumpyNet(X_train.shape[1], hidden, K, seed=seed).fit(
                    X_train, y_train, epochs=epochs, seed=seed)
            else:
                net = dp_train_classification(X_train, y_train, K, float(e), delta,
                                              hidden=hidden, epochs=epochs, seed=seed)
            utils.append(float(accuracy_score(y_test, net.predict(X_test))))
        clean = utils[0]
        mech = "gaussian (DP-SGD)"
    else:
        for e in eps_arr:
            m = dp_train_regression(X_train, y_train, float(e) if not np.isinf(e) else 0.0,
                                    delta, seed=seed)
            utils.append(float(r2_score(y_test, m.predict(X_test))))
        clean = utils[0]
        mech = "gaussian (input perturbation)"
    return DPResult(
        epsilons=eps_arr, utilities=np.asarray(utils), clean_utility=clean,
        mechanism=mech, task=task.value, delta=delta,
    )


# ==========================================================================
# federated learning (FedAvg)
# ==========================================================================
def _weighted_average(param_lists: List[List[np.ndarray]], weights: List[float]) -> List[np.ndarray]:
    w = np.asarray(weights, float)
    w = w / w.sum()
    avg = []
    for arrays in zip(*param_lists):
        stacked = np.stack(arrays, axis=0)          # (n_clients, *shape)
        avg.append(np.tensordot(w, stacked, axes=(0, 0)))
    return avg


def fedavg(
    clients: Dict[str, Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray, y_test: np.ndarray, K: int,
    n_rounds: int = 8, local_epochs: int = 5, lr: float = 0.15, batch: int = 32,
    hidden: int = 24, seed: int = 42,
) -> FedAvgResult:
    """FedAvg over named clients; compares to centralised & local-only training."""
    names = list(clients.keys())
    sizes = [len(clients[nm][0]) for nm in names]
    d = X_test.shape[1]
    global_net = NumpyNet(d, hidden, K, seed=seed)

    fed_accs = []
    for r in range(n_rounds):
        param_lists = []
        for nm in names:
            Xc, yc = clients[nm]
            local = NumpyNet(d, hidden, K, seed=seed)
            local.set_params(global_net.get_params())
            local.fit_local(Xc, yc, epochs=local_epochs, lr=lr, batch=batch, seed=seed + r)
            param_lists.append(local.get_params())
        global_net.set_params(_weighted_average(param_lists, sizes))
        fed_accs.append(float(accuracy_score(y_test, global_net.predict(X_test))))

    # centralised baseline (all clients' data pooled, same total local work)
    X_all = np.vstack([clients[nm][0] for nm in names])
    y_all = np.concatenate([clients[nm][1] for nm in names])
    central = NumpyNet(d, hidden, K, seed=seed).fit(
        X_all, y_all, epochs=n_rounds * local_epochs, lr=lr, batch=batch, seed=seed)
    central_acc = float(accuracy_score(y_test, central.predict(X_test)))

    # local-only baseline: each client trains alone, evaluated on the shared test
    local_accs = []
    for nm in names:
        Xc, yc = clients[nm]
        loc = NumpyNet(d, hidden, K, seed=seed).fit(
            Xc, yc, epochs=n_rounds * local_epochs, lr=lr, batch=batch, seed=seed)
        local_accs.append(float(accuracy_score(y_test, loc.predict(X_test))))

    return FedAvgResult(
        n_clients=len(names), n_rounds=n_rounds,
        federated_accuracies=np.asarray(fed_accs),
        centralized_accuracy=central_acc,
        local_only_accuracies=local_accs,
        client_names=names,
        privacy_note="raw per-agent captures never leave their agent; only model updates are shared",
    )


# ==========================================================================
# membership inference
# ==========================================================================
# Attack features. The first group needs only the released model's output; the
# second group needs the true label of the candidate record, which a data
# adversary (an owner of the record, or an analyst with ground truth) typically
# does have.
_MI_CONF = ("conf", "ent", "margin")
_MI_LABEL = ("p_true", "nll", "minp", "pdg")


def _mi_proba(target, X) -> np.ndarray:
    """Class probabilities of a fitted target; accepts a TrustModel or a bare estimator."""
    est = getattr(target, "estimator", target)
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(X), dtype=float)
    d = np.asarray(est.decision_function(X), dtype=float)
    if d.ndim == 1:
        d = np.stack([-d, d], axis=1)
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _mi_features(p: np.ndarray, y: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    s = np.sort(np.asarray(p, dtype=float), axis=1)
    conf = s[:, -1]
    out = {
        "conf": conf,
        "ent": -(p * np.log(p + 1e-12)).sum(axis=1),
        "margin": s[:, -1] - s[:, -2],
    }
    if y is not None and p.shape[1] >= 2:
        yt = np.asarray(y).astype(int).ravel()
        p_true = p[np.arange(len(p)), yt]
        out["p_true"] = p_true
        out["nll"] = -np.log(p_true + 1e-12)
        out["minp"] = -np.log(p.min(axis=1) + 1e-12)
        out["pdg"] = conf - p_true
    return out


def _mi_stack(F: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.column_stack([F[n] for n in names])


def _mi_auc(score, ya) -> float:
    try:
        return float(roc_auc_score(ya, score))
    except ValueError:
        return float("nan")


def _mi_ks(v1, v0) -> float:
    """Two-sample KS statistic: max |ECDF_members - ECDF_nonmembers|."""
    v1 = np.sort(np.asarray(v1, dtype=float))
    v0 = np.sort(np.asarray(v0, dtype=float))
    if not len(v1) or not len(v0):
        return float("nan")
    allv = np.concatenate([v1, v0])
    c1 = np.searchsorted(v1, allv, side="right") / len(v1)
    c0 = np.searchsorted(v0, allv, side="right") / len(v0)
    return float(np.max(np.abs(c1 - c0)))


def _mi_logsumexp(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))).ravel()


def _mi_near_duplicate_rate(X_train, X_test) -> float:
    """Share of non-member rows appearing verbatim in the training set.

    Exact overlap inflates every attack AUC regardless of the mechanism, so it is
    reported next to the scores instead of being silently credited to leakage.
    """
    Xtr = np.round(np.asarray(X_train, dtype=float), 9)
    Xte = np.round(np.asarray(X_test, dtype=float), 9)
    if not len(Xte):
        return float("nan")
    keys = {tuple(r) for r in Xtr}
    hits = sum(1 for r in Xte if tuple(r) in keys)
    return float(hits) / float(len(Xte))


def _mi_lira(mem_stack: List[np.ndarray], non_stack: List[np.ndarray],
             vm, vn) -> float:
    """Ensemble LiRA: log of the member density minus log of the non-member density,
    each averaged over the per-shadow Gaussian fits of every attack feature.

    The per-shadow sigma is floored by a fraction of the pooled spread, because a
    feature can be near-constant inside one shadow (pdg is 0 for most samples).
    """
    vm = np.asarray(vm, dtype=float)
    vn = np.asarray(vn, dtype=float)
    scores_m, scores_n = [], []
    for c in range(mem_stack[0].shape[1]):
        f1 = np.stack([M[:, c] for M in mem_stack])
        f0 = np.stack([N[:, c] for N in non_stack])
        floor = 1e-6 + 0.05 * (np.concatenate([f1.ravel(), f0.ravel()]).std() + 1e-12)
        mu1, sd1 = f1.mean(axis=1), np.maximum(f1.std(axis=1), floor)
        mu0, sd0 = f0.mean(axis=1), np.maximum(f0.std(axis=1), floor)
        ln1, ln0 = np.log(sd1), np.log(sd0)

        def lr(v, mu1=mu1, sd1=sd1, mu0=mu0, sd0=sd0):
            v = np.asarray(v, dtype=float).ravel()
            l1 = -0.5 * ((v[:, None] - mu1[None, :]) / sd1[None, :]) ** 2 - ln1[None, :]
            l0 = -0.5 * ((v[:, None] - mu0[None, :]) / sd0[None, :]) ** 2 - ln0[None, :]
            return (_mi_logsumexp(l1 - np.log(len(mu1)))
                    - _mi_logsumexp(l0 - np.log(len(mu0))))

        scores_m.append(lr(vm[:, c]))
        scores_n.append(lr(vn[:, c]))
    sm = np.mean(scores_m, axis=0)
    sn = np.mean(scores_n, axis=0)
    ya = np.concatenate([np.ones(len(sm)), np.zeros(len(sn))])
    return _mi_auc(np.concatenate([sm, sn]), ya)


def membership_inference(
    model: TrustModel, X_train, y_train, X_test, y_test, seed: int = 42,
    n_shadows: int = 0, aux_frac: float = 0.5,
) -> MembershipInferenceResult:
    """Probe how much a released model leaks about the rows it was trained on.

    Members are the training rows, non-members the test rows. Four attacks read the
    same pools and are reported together:

    * ``threshold`` -- confidence threshold attack (Yeom et al. 2018) on ``-conf``.
    * ``lr_conf`` -- logistic meta-model over the label-free features.
    * ``lr_meta`` -- same meta-model with the label-aware features appended; this is
      ``attack_auc``.
    * ``ssa`` / ``lira`` -- only when ``n_shadows > 0``. Shadow models of the target
      family are retrained on halves of an auxiliary pool, membership in each half is
      known by construction, and the meta-model (SSA, Samuel et al. 2019) or the
      averaged Gaussian log-likelihood ratio (LiRA, Sala et al. 2019) is fitted on the
      shadow side and applied to the target. Non-members here exclude the pool rows.

    AUC >> 0.5 means membership is inferable. KS, TPR@1%FPR, the two gaps and the
    near-duplicate rate say how much of that signal is genuine memorisation.
    """
    Xm = np.asarray(X_train, dtype=float)
    Xn = np.asarray(X_test, dtype=float)
    ym = np.asarray(y_train).astype(int).ravel()
    yn = np.asarray(y_test).astype(int).ravel()
    Pm = _mi_proba(model, Xm)
    Pn = _mi_proba(model, Xn)
    Fm = _mi_features(Pm, ym)
    Fn = _mi_features(Pn, yn)

    conf_names = [n for n in _MI_CONF if n in Fm]
    label_names = [n for n in _MI_LABEL if n in Fm]
    all_names = conf_names + label_names

    ya = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xn))])
    attacks: Dict[str, float] = {
        "threshold": _mi_auc(np.concatenate([Fm["conf"], Fn["conf"]]), ya),
    }

    def meta(names: Sequence[str]):
        """Fit the meta-model on a split of the attack pool; return held-out scores."""
        Xa = np.vstack([_mi_stack(Fm, names), _mi_stack(Fn, names)])
        try:
            Xtr, Xte, ytr, yte = train_test_split(
                Xa, ya, test_size=0.4, random_state=seed, stratify=ya)
            atk = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
            return (atk.predict_proba(Xte)[:, 1], yte,
                    float(accuracy_score(yte, atk.predict(Xte))))
        except ValueError:
            return np.array([]), np.array([]), float("nan")

    prob_conf, yte_conf, _ = meta(conf_names)
    prob_all, yte_all, acc_all = meta(all_names)
    attacks["lr_conf"] = _mi_auc(prob_conf, yte_conf)
    attacks["lr_meta"] = _mi_auc(prob_all, yte_all)
    auc = attacks["lr_meta"]

    tpr = float("nan")
    if len(yte_all) >= 50 and yte_all.min() < yte_all.max():
        q = np.quantile(prob_all[yte_all == 0], 0.99)
        tpr = float(np.mean(prob_all[yte_all == 1] > q))

    acc_m = float(np.mean(Pm.argmax(axis=1) == ym))
    acc_n = float(np.mean(Pn.argmax(axis=1) == yn))
    diag_name = "nll" if label_names else "conf"
    ks = _mi_ks(Fm[diag_name], Fn[diag_name])
    dup = _mi_near_duplicate_rate(Xm, Xn)

    # ---- shadow group (optional): SSA + LiRA, fitted where membership is known
    shadows_run = 0
    shadow_note = ""
    shadow_attacks: Dict[str, float] = {}
    if n_shadows > 0:
        perm = np.random.default_rng(seed).permutation(len(Xn))
        aux = int(round(aux_frac * len(Xn)))
        aidx, eidx = perm[:aux], perm[aux:]
        if aux < 40 or len(eidx) < 40:
            shadow_note = f"shadows skipped (pool {aux}, evaluation {len(eidx)} rows)"
        else:
            try:
                pool_X, pool_y = Xn[aidx], yn[aidx]
                Fe = _mi_features(_mi_proba(model, Xn[eidx]), yn[eidx])
                rng = np.random.default_rng(seed + 1)
                sh_mem, sh_non = [], []
                for _ in range(int(n_shadows)):
                    idx = rng.permutation(len(pool_X))
                    half = len(idx) // 2
                    tr_i, te_i = idx[:half], idx[half:]
                    shadow = model.clone_unfitted()
                    shadow.fit(pool_X[tr_i], pool_y[tr_i])
                    sh_mem.append(_mi_stack(
                        _mi_features(_mi_proba(shadow, pool_X[tr_i]), pool_y[tr_i]), all_names))
                    sh_non.append(_mi_stack(
                        _mi_features(_mi_proba(shadow, pool_X[te_i]), pool_y[te_i]), all_names))

                Xs = np.vstack(sh_mem + sh_non)
                ys = np.concatenate([np.ones(sum(len(M) for M in sh_mem)),
                                     np.zeros(sum(len(N) for N in sh_non))])
                s_meta = LogisticRegression(max_iter=2000).fit(Xs, ys)
                s_score = np.concatenate([
                    s_meta.predict_proba(_mi_stack(Fm, all_names))[:, 1],
                    s_meta.predict_proba(_mi_stack(Fe, all_names))[:, 1]])
                y_sh = np.concatenate([np.ones(len(Fm["conf"])), np.zeros(len(eidx))])
                shadow_attacks["ssa"] = _mi_auc(s_score, y_sh)
                shadow_attacks["ssa_accuracy"] = float(
                    accuracy_score(y_sh, (s_score >= 0.5).astype(int)))
                shadow_attacks["lira"] = _mi_lira(
                    sh_mem, sh_non, _mi_stack(Fm, all_names), _mi_stack(Fe, all_names))
                shadows_run = len(sh_mem)
            except Exception as exc:
                shadows_run = 0
                shadow_attacks = {}
                shadow_note = "shadows unavailable (%s: %s)" % (type(exc).__name__, exc)
    attacks.update(shadow_attacks)

    parts = ["meta-model on %d features [%s]" % (len(all_names), "+".join(all_names)),
             "threshold(conf) AUC=%.4f" % attacks["threshold"],
             "label-free meta AUC=%.4f" % attacks["lr_conf"]]
    if shadows_run:
        parts.append("%d shadow models: SSA AUC=%.4f, LiRA AUC=%.4f" % (
            shadows_run, attacks.get("ssa", float("nan")), attacks.get("lira", float("nan"))))
    if shadow_note:
        parts.append(shadow_note)
    parts.append("KS(%s)=%.4f TPR@1%%FPR=%.3f conf_gap=%.4f generalization_gap=%.4f "
                 "exact_dup=%.4f" % (
        diag_name, ks, tpr, float(Fm["conf"].mean() - Fn["conf"].mean()),
        acc_m - acc_n, dup))
    parts.append("AUC~0.5 = no leakage, AUC->1 = strong leakage")
    parts.append("label-side features assume the adversary knows the candidate record's label")

    return MembershipInferenceResult(
        attack_auc=auc, attack_accuracy=acc_all,
        n_members=int(len(Xm)), n_nonmembers=int(len(Xn)),
        note="; ".join(parts),
        attack_aucs=attacks,
        ks_stat=ks, tpr_at_1pct_fpr=tpr,
        confidence_gap=float(Fm["conf"].mean() - Fn["conf"].mean()),
        generalization_gap=acc_m - acc_n,
        near_duplicate_rate=dup,
        n_shadows=int(shadows_run),
        feature_set="+".join(all_names),
    )
