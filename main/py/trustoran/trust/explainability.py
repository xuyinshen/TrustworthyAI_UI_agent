"""Explainability pillar.

Model-agnostic toolkit answering *"why did the RAN controller decide this?"*:

* :func:`global_importance`  -- permutation importance (+ native gini/|coef|)
* :func:`shap_explain`       -- additive attributions: the exact TreeSHAP
                                decomposition of xgboost/lightgbm boosters when
                                available, else a KernelSHAP-style sampling
                                approximation (works on any estimator)
* :func:`local_explanation`  -- LIME-style local linear surrogate for one sample
* :func:`partial_dependence` -- PDP + ICE curves

Everything operates on the *scaled* feature space the model was trained in; the
Qt layer maps values back through the standardizer for display.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence as _sk_pdp
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from ..data.base import TaskType
from ..models.wrappers import TrustModel
from .results import GlobalImportance, LocalExplanation, PDPResult, ShapResult


# --------------------------------------------------------------------------
# a scalar model output f(x) used by SHAP / PDP / local surrogates
# --------------------------------------------------------------------------
def make_model_fn(model: TrustModel, target_class: Optional[int] = None) -> Callable[[np.ndarray], np.ndarray]:
    """Return f: (n, d) -> (n,) -- the scalar output SHAP/PDP explain.

    * regression          -> prediction
    * binary              -> P(positive class = 1)
    * multiclass          -> P(target_class) (defaults to class 1, the 'attack')
    """
    if model.task == TaskType.REGRESSION:
        return lambda Z: model.estimator.predict(Z)
    if not hasattr(model.estimator, "predict_proba"):
        # linear SVC etc: fall back to decision_function (squashed)
        def f(Z):
            d = model.estimator.decision_function(Z)
            return d if d.ndim == 1 else d[:, target_class or 0]
        return f
    cls = target_class
    if cls is None:
        cls = 1 if model.task == TaskType.BINARY else 1
    def f(Z):
        p = model.estimator.predict_proba(Z)
        c = cls if cls < p.shape[1] else 0
        return p[:, c]
    return f


# --------------------------------------------------------------------------
# global importance
# --------------------------------------------------------------------------
def global_importance(
    model: TrustModel,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 8,
    seed: int = 42,
) -> GlobalImportance:
    scoring = None if model.task == TaskType.REGRESSION else "accuracy"
    if model.task == TaskType.REGRESSION:
        scoring = "neg_mean_absolute_error"
    perm = permutation_importance(
        model.estimator, X, y, n_repeats=n_repeats,
        random_state=seed, scoring=scoring, n_jobs=-1,
    )
    return GlobalImportance(
        feature_names=list(model.feature_names),
        permutation_importance=perm.importances_mean,
        permutation_std=perm.importances_std,
        native_importance=model.native_importance(),
        metric=scoring or "accuracy",
    )


# --------------------------------------------------------------------------
# KernelSHAP-style attributions (model agnostic)
# --------------------------------------------------------------------------
def _shap_kernel_weight(s: int, d: int) -> float:
    from math import comb
    if s == 0 or s == d:
        return 1e6                       # (near-)full coalitions dominate
    return (d - 1.0) / (comb(d, s) * (s ** 2) * ((d - s) ** 2))


def _shap_single(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    base_value: float,
    n_coalitions: int,
    rng: np.random.Generator,
    reg: float = 1e-6,
) -> np.ndarray:
    d = len(x)
    nb = min(len(background), 8)
    bg = background[rng.choice(len(background), size=nb, replace=False)]

    # sample binary coalitions (avoid the trivial all-0 / all-1 unless d small)
    Z = rng.integers(0, 2, size=(n_coalitions, d)).astype(float)
    Z[0, :] = 1.0                          # full coalition -> f(x)
    Z[1, :] = 0.0                          # empty coalition -> base
    sizes = Z.sum(axis=1)
    w = np.array([_shap_kernel_weight(int(s), d) for s in sizes])

    # marginalised predictions: for each coalition average over background rows
    # synthetic[(k, nb), d]: present features from x, missing from background
    present = Z[:, None, :] == 1.0                     # (k, 1, d)
    synth = np.where(present, x[None, None, :], bg[None, :, :])
    synth = synth.reshape(-1, d)
    preds = fn(synth).reshape(n_coalitions, nb).mean(axis=1)

    # weighted least squares with intercept pinned to base_value
    target = preds - base_value
    W = w
    ZtW = Z.T * W                                       # (d, k)
    A = ZtW @ Z + reg * np.eye(d)
    b = ZtW @ target
    phi = np.linalg.solve(A, b)

    # efficiency: rescale so attributions sum to f(x) - base_value
    fx = float(fn(x[None, :])[0])
    total = phi.sum()
    if abs(total) > 1e-9:
        phi = phi * ((fx - base_value) / total)
    return phi


def shap_explain(
    model: TrustModel,
    X_explain: np.ndarray,
    background: np.ndarray,
    n_coalitions: Optional[int] = None,
    target_class: Optional[int] = None,
    seed: int = 42,
    method: str = "auto",
) -> ShapResult:
    """Additive attributions for each row of ``X_explain``.

    ``method`` selects the estimator of the attribution:

    * ``"auto"`` (default) -- use the **exact** per-instance decomposition the
      boosting libraries ship (xgboost ``pred_contribs``, lightgbm
      ``pred_contrib``, both TreeSHAP in C++) when the wrapped estimator has
      one, otherwise the sampled KernelSHAP-style approximation.
    * ``"kernel"`` -- always sample coalitions.
    * ``"exact"`` -- require the exact path; raise if unavailable.

    The two paths do not share a scale. The sampled path explains
    ``make_model_fn`` (a probability for classifiers); the exact path explains
    the booster's **margin** (the logit of ``target_class``, or the prediction
    for regression), and its base value is the model's own expected output over
    the training set -- the ``background`` argument is not used there.
    ``ShapResult.target`` and ``.method`` say which of the two was returned, and
    ``.efficiency_max_error`` reports how well the exact decomposition
    reproduces the model's own score (the sampled path cannot self-verify, see
    the efficiency rescaling in :func:`_shap_single`).
    """
    X_explain = np.asarray(X_explain, float)
    background = np.asarray(background, float)
    d = X_explain.shape[1]
    if method not in ("auto", "kernel", "exact"):
        raise ValueError(f"unknown method {method!r}")

    if method in ("auto", "exact"):
        exact = model.native_contribs(X_explain, target_class)
        if exact is not None:
            phi, base, margin = exact
            target = ("prediction on the model margin" if model.task == TaskType.REGRESSION
                      else f"margin of class "
                           f"{target_class if target_class is not None else 1}")
            return ShapResult(
                feature_names=list(model.feature_names),
                shap_values=phi,
                base_value=base,
                X=X_explain,
                target=target,
                method="exact",
                efficiency_max_error=float(
                    np.max(np.abs(base + phi.sum(axis=1) - margin))),
            )
        if method == "exact":
            raise ValueError(
                f"{type(model.estimator).__name__} has no exact contribution path "
                "(xgboost/lightgbm only); use method='kernel'")

    fn = make_model_fn(model, target_class)
    if n_coalitions is None:
        n_coalitions = int(min(2 ** d, max(2 * d + 8, 48)))
    rng = np.random.default_rng(seed)
    base_value = float(fn(background).mean())

    vals = np.array([
        _shap_single(fn, X_explain[i], background, base_value, n_coalitions, rng)
        for i in range(len(X_explain))
    ])
    target = (
        "prediction"
        if model.task == TaskType.REGRESSION
        else f"P(class {target_class if target_class is not None else 1})"
    )
    return ShapResult(
        feature_names=list(model.feature_names),
        shap_values=vals,
        base_value=base_value,
        X=X_explain,
        target=target,
        method="kernel",
    )


# --------------------------------------------------------------------------
# LIME-style local surrogate
# --------------------------------------------------------------------------
def local_explanation(
    model: TrustModel,
    x: np.ndarray,
    background: np.ndarray,
    n_samples: int = 400,
    kernel_width: float = 0.6,
    target_class: Optional[int] = None,
    seed: int = 42,
) -> LocalExplanation:
    """Fit a weighted linear surrogate around ``x`` and report per-feature
    contributions (slope x deviation-from-background-mean)."""
    fn = make_model_fn(model, target_class)
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float).ravel()
    background = np.asarray(background, float)
    d = len(x)

    # perturb by mixing x with random background rows
    pick = rng.integers(0, len(background), size=(n_samples, d))
    mask = rng.random((n_samples, d)) < 0.4           # 40% of cells replaced
    Xs = np.where(mask, background[pick, np.arange(d)[None, :]], x[None, :])

    preds = fn(Xs)
    dist2 = ((Xs - x[None, :]) ** 2).sum(axis=1)
    w = np.exp(-dist2 / (kernel_width ** 2))

    surrogate = Ridge(alpha=1e-3)
    surrogate.fit(Xs, preds, sample_weight=w)
    bg_mean = background.mean(axis=0)
    contribution = surrogate.coef_ * (x - bg_mean)
    base = float(fn(background).mean())
    return LocalExplanation(
        feature_names=list(model.feature_names),
        coefficients=contribution,
        base_value=base,
        prediction=float(fn(x[None, :])[0]),
    )


# --------------------------------------------------------------------------
# partial dependence
# --------------------------------------------------------------------------
def partial_dependence(
    model: TrustModel,
    X: np.ndarray,
    feature_index: int,
    grid_resolution: int = 20,
    n_ice: int = 40,
    target_class: Optional[int] = None,
) -> PDPResult:
    """PDP (and a sample of ICE curves) for one feature."""
    res = _sk_pdp(
        model.estimator, X, [feature_index],
        grid_resolution=grid_resolution, method="auto", kind="both",
    )
    grid = res["grid_values"][0]
    avg = res["average"][0] if np.ndim(res["average"][0]) == 1 else res["average"][0]
    ice = res.get("individual")
    ice_arr = None
    if ice is not None:
        ind = ice[0]
        if ind.ndim == 2 and ind.shape[0] > n_ice:
            step = ind.shape[0] // n_ice
            ind = ind[::step][:n_ice]
        ice_arr = ind
    return PDPResult(
        feature_name=model.feature_names[feature_index],
        grid=np.asarray(grid),
        average_prediction=np.asarray(avg),
        ice_curves=ice_arr,
    )
