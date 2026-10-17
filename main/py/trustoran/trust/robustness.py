"""Robustness pillar.

Adversarial attacks and defences for the RAN detectors. sklearn models expose no
gradients, so we estimate them either analytically (linear models) or by central
finite differences (any model) -- this keeps a single attack API across model
families, exactly like the unified library the project targets.

Attacks run in the *scaled* feature space, so ``eps`` is measured in standard
deviations of each feature.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..data.base import TaskType
from ..models.factory import make_model
from ..models.wrappers import TrustModel
from .results import AttackResult, RobustnessCurve


# --------------------------------------------------------------------------
# gradients of the true-class probability / margin
# --------------------------------------------------------------------------
def _true_class_fn(model: TrustModel) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """f(X) = P(true class) for classification, or the true-class margin for SVC."""
    if hasattr(model.estimator, "predict_proba"):
        def f(X, y=None):
            p = model.estimator.predict_proba(X)
            if y is None:
                return p
            return p[np.arange(len(X)), y]
        return f

    def f(X, y=None):
        d = np.asarray(model.estimator.decision_function(X))
        if d.ndim == 1:                    # binary margin: positive scores class 1
            return d if y is None else d * np.where(y == 1, 1.0, -1.0)
        return d if y is None else d[np.arange(len(d)), y]
    return f


def _grad_true_class(model: TrustModel, X: np.ndarray, y: np.ndarray,
                     h: float = 2e-3) -> np.ndarray:
    """Central finite-difference gradient of P(true class) w.r.t. X -> (n, d)."""
    n, d = X.shape
    f = _true_class_fn(model)
    grad = np.zeros_like(X, dtype=float)
    for i in range(d):
        Xp = X.copy(); Xp[:, i] += h
        Xm = X.copy(); Xm[:, i] -= h
        grad[:, i] = (f(Xp, y) - f(Xm, y)) / (2 * h)
    return grad


# --------------------------------------------------------------------------
# attacks (classification)
# --------------------------------------------------------------------------
def _clip(x: np.ndarray, x0: np.ndarray, eps: float,
          bounds: Optional[tuple] = None) -> np.ndarray:
    x = np.clip(x, x0 - eps, x0 + eps)
    if bounds is not None:
        lo, hi = bounds
        x = np.clip(x, lo, hi)
    return x


def fgsm(model: TrustModel, X: np.ndarray, y: np.ndarray, eps: float,
         bounds: Optional[tuple] = None) -> np.ndarray:
    g = _grad_true_class(model, X, y)
    x_adv = X - eps * np.sign(g)                 # descend the true-class prob
    return _clip(x_adv, X, eps, bounds)


def pgd(model: TrustModel, X: np.ndarray, y: np.ndarray, eps: float,
        alpha: Optional[float] = None, n_iter: int = 12,
        bounds: Optional[tuple] = None, seed: int = 42) -> np.ndarray:
    alpha = alpha if alpha is not None else max(eps / 4.0, 1e-3)
    rng = np.random.default_rng(seed)
    x_adv = X + rng.uniform(-eps, eps, size=X.shape)
    x_adv = _clip(x_adv, X, eps, bounds)
    for _ in range(n_iter):
        g = _grad_true_class(model, x_adv, y)
        x_adv = x_adv - alpha * np.sign(g)
        x_adv = _clip(x_adv, X, eps, bounds)
    return x_adv


def deepfool(model: TrustModel, X: np.ndarray, max_iter: int = 20,
             h: float = 2e-3, overshoot: float = 1.02) -> np.ndarray:
    """Multiclass DeepFool with finite-difference gradients of all class scores.

    Returns the minimal-ish adversarial perturbation that flips each sample's
    predicted class. Callers should pass a modest subset (it costs 2*d batch
    predictions per iteration).
    """
    n, d = X.shape
    proba = hasattr(model.estimator, "predict_proba")

    def scores(Z):
        s = np.asarray(model.estimator.predict_proba(Z) if proba
                       else model.estimator.decision_function(Z))
        if s.ndim == 1:                    # binary margin -> two-class scores
            s = np.stack([-s, s], axis=1)
        return s

    K = scores(X[:1]).shape[1]
    label0 = scores(X).argmax(axis=1)              # original predicted labels
    x = X.copy()

    for _ in range(max_iter):
        s = scores(x)                               # (n, K)
        still = (s.argmax(axis=1) == label0)        # not yet flipped
        if not still.any():
            break
        # finite-difference gradient of every class score -> (n, K, d)
        g = np.zeros((n, K, d))
        for i in range(d):
            xp = x.copy(); xp[:, i] += h
            xm = x.copy(); xm[:, i] -= h
            g[:, :, i] = (scores(xp) - scores(xm)) / (2 * h)
        g_true = g[np.arange(n), label0][:, None, :]         # (n,1,d)
        w = g - g_true                                        # (n,K,d)
        f = s - s[np.arange(n), label0][:, None]              # (n,K)
        wnorm = np.linalg.norm(w, axis=2) + 1e-9             # (n,K)
        pert = np.abs(f) / wnorm                              # (n,K)
        pert[np.arange(n), label0] = np.inf                   # ignore true cls
        kstar = pert.argmin(axis=1)                           # (n,)
        step_len = pert[np.arange(n), kstar]                  # (n,)
        wstar = w[np.arange(n), kstar]                        # (n,d)
        step = (step_len[:, None] + 1e-6) * wstar / (np.linalg.norm(wstar, axis=1, keepdims=True) + 1e-9)
        step = np.where(still[:, None], step, 0.0)
        x = x + overshoot * step
    return x


ATTACKS = {"fgsm": fgsm, "pgd": pgd}          # deepfool handled separately (slow)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_attack(model: TrustModel, X: np.ndarray, y: np.ndarray,
                    attack: str, eps: float, class_names=None,
                    bounds: Optional[tuple] = None, seed: int = 42) -> AttackResult:
    clean_pred = model.predict(X)
    clean_acc = float((clean_pred == y).mean())

    if attack == "deepfool":
        x_adv = deepfool(model, X)
        mean_pert = float(np.linalg.norm(x_adv - X, axis=1).mean())
    else:
        fn = ATTACKS[attack]
        x_adv = fn(model, X, y, eps, bounds=bounds) if attack == "fgsm" \
            else fn(model, X, y, eps, bounds=bounds, seed=seed)
        mean_pert = eps

    adv_pred = model.predict(x_adv)
    robust_acc = float((adv_pred == y).mean())
    # attack success: was clean-correct, now wrong
    was_correct = clean_pred == y
    flipped = was_correct & (adv_pred != y)
    asr = float(flipped.sum() / max(was_correct.sum(), 1))

    by_class = {}
    if class_names:
        for c, name in enumerate(class_names):
            m = y == c
            if m.sum():
                by_class[name] = float((adv_pred[m] == y[m]).mean())

    return AttackResult(
        attack=attack, eps=eps, clean_accuracy=clean_acc, robust_accuracy=robust_acc,
        attack_success_rate=asr, mean_perturbation=mean_pert,
        perturbed_accuracy_by_class=by_class,
    )


def robustness_curve(model: TrustModel, X: np.ndarray, y: np.ndarray,
                     attack: str = "fgsm",
                     eps_range=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
                     bounds: Optional[tuple] = None, seed: int = 42) -> RobustnessCurve:
    eps_arr = np.asarray(eps_range, float)
    accs = []
    for e in eps_arr:
        if e == 0:
            accs.append(float((model.predict(X) == y).mean()))
        else:
            r = evaluate_attack(model, X, y, attack, float(e), bounds=bounds, seed=seed)
            accs.append(r.robust_accuracy)
    return RobustnessCurve(
        attack=attack, epsilons=eps_arr, accuracies=np.asarray(accs),
        clean_accuracy=accs[0],
    )


def adversarial_training(model: TrustModel, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray, eps: float = 0.3,
                         attack: str = "fgsm",
                         eval_eps_range=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
                         bounds: Optional[tuple] = None, seed: int = 42) -> RobustnessCurve:
    """Retrain on clean+adversarial examples, then report the robustness curve
    of the *defended* model alongside its clean accuracy."""
    x_adv = fgsm(model, X_train, y_train, eps, bounds=bounds)
    X_aug = np.vstack([X_train, x_adv])
    y_aug = np.concatenate([y_train, y_train])
    defended = model.clone_unfitted().fit(X_aug, y_aug)

    eps_arr = np.asarray(eval_eps_range, float)
    accs = []
    for e in eps_arr:
        if e == 0:
            accs.append(float((defended.predict(X_test) == y_test).mean()))
        else:
            r = evaluate_attack(defended, X_test, y_test, attack, float(e), bounds=bounds, seed=seed)
            accs.append(r.robust_accuracy)
    return RobustnessCurve(
        attack=f"{attack}+advtrain", epsilons=eps_arr, accuracies=np.asarray(accs),
        clean_accuracy=accs[0],
        adversarial_training_accuracies=np.asarray(accs),
    )


def regression_noise_drift(model: TrustModel, X: np.ndarray, y: np.ndarray,
                           sigma_range=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
                           seed: int = 42) -> RobustnessCurve:
    """For regression: utility retention as Gaussian noise (in std units) grows.

    ``accuracies`` holds ``MAE(0) / MAE(σ)`` — 1.0 means the error is unchanged,
    0.5 means it has doubled. A ratio in (0, 1] reads on the same scale as a
    classification accuracy and, unlike R², never goes negative. R² and MAE per
    level ride along in ``r2`` / ``mae`` as the underlying measurements.
    """
    rng = np.random.default_rng(seed)
    sig = np.asarray(sigma_range, float)
    from sklearn.metrics import mean_absolute_error, r2_score
    r2s, maes = [], []
    for s in sig:
        Xn = X if s == 0 else X + rng.normal(0, s, size=X.shape)
        pred = model.predict(Xn)
        maes.append(float(mean_absolute_error(y, pred)))
        r2s.append(float(r2_score(y, pred)))
    mae_arr = np.asarray(maes, float)
    utils = mae_arr[0] / np.where(mae_arr > 0, mae_arr, 1.0)
    return RobustnessCurve(
        attack="gaussian_noise", epsilons=sig, accuracies=utils,
        clean_accuracy=float(utils[0]),
        metric="utility", r2=np.asarray(r2s), mae=mae_arr,
    )
