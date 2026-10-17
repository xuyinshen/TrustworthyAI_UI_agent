"""A single unified estimator interface over heterogeneous scikit-learn models.

``TrustModel`` is the one object every trust pillar talks to. It hides whether
the underlying estimator is a random forest, a gradient-boosted tree, a linear
model or an MLP, and exposes a consistent ``fit / predict / predict_proba /
decision_score / score`` API plus metadata (feature names, class names,
native importances).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error

from ..data.base import TaskType


@dataclass
class ModelSpec:
    """Declarative description of a model (name + hyperparameters).

    Keeping the spec separate from the fitted model makes the whole stack
    serialisable and lets the Qt UI expose hyperparameters as editable fields.
    """

    name: str = "random_forest"
    params: Dict[str, Any] = field(default_factory=dict)
    task: TaskType = TaskType.CLASSIFICATION

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": self.params, "task": self.task.value}


class TrustModel:
    """Unified wrapper around a scikit-learn estimator."""

    def __init__(
        self,
        estimator: BaseEstimator,
        task: TaskType,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        spec: Optional[ModelSpec] = None,
    ) -> None:
        self.estimator = estimator
        self.task = task
        self.feature_names = list(feature_names)
        self.class_names = class_names
        self.spec = spec or ModelSpec(name=type(estimator).__name__, task=task)
        self._fitted = False

    # -- training / inference ------------------------------------------------
    def fit(self, X, y) -> "TrustModel":
        X = self._as_array(X)
        self.estimator.fit(X, np.asarray(y).ravel())
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(self._as_array(X))

    def predict_proba(self, X) -> np.ndarray:
        """Class-probability matrix. Regression models raise."""
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                f"{type(self.estimator).__name__} has no predict_proba"
            )
        return self.estimator.predict_proba(self._as_array(X))

    def decision_score(self, X) -> np.ndarray:
        """A single scalar score per sample used by attacks & thresholds.

        * classification: probability of the predicted (argmax) class
        * binary: probability of the positive class
        * regression: the raw prediction
        """
        X = self._as_array(X)
        if self.task == TaskType.REGRESSION:
            return self.estimator.predict(X)
        if hasattr(self.estimator, "predict_proba"):
            p = self.estimator.predict_proba(X)
            if self.task == TaskType.BINARY and p.shape[1] == 2:
                return p[:, 1]
            return p.max(axis=1)
        return self.estimator.decision_function(X)

    def score(self, X, y) -> Dict[str, float]:
        """Task-appropriate metrics on a batch."""
        X = self._as_array(X)
        y = np.asarray(y).ravel()
        pred = self.estimator.predict(X)
        if self.task == TaskType.REGRESSION:
            return {
                "r2": float(r2_score(y, pred)),
                "mae": float(mean_absolute_error(y, pred)),
            }
        out = {
            "accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        }
        if self.task == TaskType.BINARY and hasattr(self.estimator, "predict_proba"):
            try:
                out["roc_auc"] = float(roc_auc_score(y, self.estimator.predict_proba(X)[:, 1]))
            except ValueError:
                pass
        return out

    # -- explainability helpers ---------------------------------------------
    @property
    def is_classifier(self) -> bool:
        return self.task.is_classification

    @property
    def has_native_importance(self) -> bool:
        return hasattr(self.estimator, "feature_importances_") or hasattr(
            self.estimator, "coef_"
        )

    def native_importance(self) -> Optional[np.ndarray]:
        """Gini/impurity importance (trees) or |coef| (linear), or None."""
        est = self.estimator
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_, dtype=float)
        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=float)
            # multi-class -> mean absolute contribution per feature
            return np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
        return None

    def supports_shap_like(self) -> bool:
        """Trees with feature_importances_ admit an interventional SHAP approx."""
        return hasattr(self.estimator, "feature_importances_") and hasattr(
            self.estimator, "estimators_"
        )

    # -- exact attributions (native TreeSHAP in the boosting libraries) ------
    def _contrib_kind(self) -> Optional[str]:
        est = self.estimator
        module = type(est).__module__
        if module.startswith("xgboost") and hasattr(est, "get_booster"):
            return "xgb"
        if module.startswith("lightgbm") and hasattr(est, "booster_"):
            return "lgb"
        return None

    def native_contribs(
        self, X, target_class: Optional[int] = None
    ) -> Optional[tuple]:
        """Exact per-feature contributions, when the library provides them.

        Returns ``(phi, base_value, margin)`` -- ``phi`` is ``(n, d)`` in the
        booster's **margin space** (regression: the prediction itself;
        classification: the logit of ``target_class``), ``base_value`` is the
        model's own expected output (the bias column, taken over the training
        set), and ``margin`` is the score the decomposition must reproduce, so
        ``base_value + phi.sum(axis=1) == margin`` can be checked by the caller.
        ``None`` means "this estimator has no exact path" (sklearn forests,
        linear models, MLPs) and the caller should use a sampled approximation.
        """
        kind = self._contrib_kind()
        if kind is None:
            return None
        X = self._as_array(X)
        d = X.shape[1]
        try:
            if kind == "xgb":
                import xgboost as _xgb
                booster = self.estimator.get_booster()
                dmat = _xgb.DMatrix(X)
                c = np.asarray(booster.predict(dmat, pred_contribs=True), float)
                raw = np.asarray(booster.predict(dmat, output_margin=True), float)
            else:
                booster = self.estimator.booster_
                c = np.asarray(booster.predict(X, pred_contrib=True), float)
                raw = np.asarray(booster.predict(X, raw_score=True), float)
        except Exception:
            return None
        if self.task == TaskType.REGRESSION:
            idx, k = 0, 1
        else:
            classes = getattr(self.estimator, "classes_", None)
            k = len(classes) if classes is not None else 1
            idx = min(int(target_class if target_class is not None else 1), k - 1)
        if c.ndim == 3 and c.shape[1] == k:            # xgboost multiclass
            c = c[:, idx, :]
        elif c.ndim == 2 and c.shape[1] == k * (d + 1) and k > 1:
            c = c[:, idx * (d + 1):(idx + 1) * (d + 1)]
        if c.ndim != 2 or c.shape[1] != d + 1:
            return None
        phi, bias = c[:, :d], c[:, d]
        raw = np.asarray(raw, float)
        if raw.ndim == 1:
            margin = raw
        elif raw.ndim == 2 and raw.shape[1] == k and k > 1:
            margin = raw[:, idx]
        else:
            return None
        return phi, float(bias.mean()), margin

    def clone_unfitted(self) -> "TrustModel":
        return TrustModel(
            clone(self.estimator),
            self.task,
            list(self.feature_names),
            self.class_names,
            self.spec,
        )

    # -- internals -----------------------------------------------------------
    def _as_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)
