"""Model factory: turn a :class:`ModelSpec` (or a short name) into a fitted-ready
:class:`TrustModel`.

The zoo mixes linear models, tree ensembles and an MLP so the robustness and
explainability modules can demonstrate behaviour across model families (linear
models expose analytic gradients; trees need surrogate/finite-difference
attacks).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC

from ..data.base import TaskType
from .wrappers import ModelSpec, TrustModel

# Optional gradient-boosting libraries. Imported defensively so the package keeps
# working (with the sklearn-only zoo) even where they are not installed.
try:                                        # pragma: no cover - env dependent
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:                           # pragma: no cover
    _HAS_XGB = False
try:                                        # pragma: no cover
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except Exception:                           # pragma: no cover
    _HAS_LGB = False

# name -> (classifier_builder, regressor_builder, default_params)
_CLS = "clf"
_REG = "reg"

MODEL_ZOO: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        _CLS: RandomForestClassifier,
        _REG: RandomForestRegressor,
        "params": {"n_estimators": 150, "random_state": 42, "n_jobs": -1},
        "label": "Random Forest",
    },
    "logistic": {
        _CLS: LogisticRegression,
        _REG: Ridge,
        "params": {"max_iter": 2000, "random_state": 42},
        "label": "Logistic / Ridge (linear)",
    },
    "grad_boost": {
        _CLS: GradientBoostingClassifier,
        _REG: GradientBoostingRegressor,
        "params": {"n_estimators": 150, "random_state": 42},
        "label": "Gradient Boosting",
    },
    "hist_grad_boost": {
        _CLS: HistGradientBoostingClassifier,
        _REG: HistGradientBoostingRegressor,
        "params": {"max_iter": 150, "random_state": 42},
        "label": "Hist Gradient Boosting",
    },
    "mlp": {
        _CLS: MLPClassifier,
        _REG: MLPRegressor,
        "params": {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 400,
            "random_state": 42,
            "early_stopping": True,
        },
        "label": "MLP (neural net)",
    },
    "linear_svc": {
        _CLS: LinearSVC,
        _REG: None,
        "params": {"max_iter": 5000, "random_state": 42, "dual": "auto"},
        "label": "Linear SVC",
    },
}

# XGBoost / LightGBM are registered only when their wheels are importable, so the
# rest of the zoo (and the whole package) still loads on machines without them.
if _HAS_XGB:
    MODEL_ZOO["xgboost"] = {
        _CLS: XGBClassifier,
        _REG: XGBRegressor,
        "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                   "subsample": 0.9, "colsample_bytree": 0.9,
                   "random_state": 42, "n_jobs": -1},
        "label": "XGBoost",
    }
if _HAS_LGB:
    MODEL_ZOO["lightgbm"] = {
        _CLS: LGBMClassifier,
        _REG: LGBMRegressor,
        "params": {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31,
                   "max_depth": -1, "subsample": 0.9, "colsample_bytree": 0.9,
                   "random_state": 42, "n_jobs": -1, "verbose": -1},
        "label": "LightGBM",
    }


# --------------------------------------------------------------------------
# tunable hyperparameter schema (drives the Qt "tune parameters" dialog)
#   kind: "int" | "float" | "choice"
#   int/float -> min, max, step ; choice -> choices:[...]
# --------------------------------------------------------------------------
PARAM_SCHEMA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "random_forest": {
        "n_estimators": {"kind": "int", "min": 10, "max": 1000, "step": 10},
        "max_depth": {"kind": "int", "min": 1, "max": 64, "step": 1},
        "min_samples_split": {"kind": "int", "min": 2, "max": 50, "step": 1},
    },
    "logistic": {
        "C": {"kind": "float", "min": 0.01, "max": 100.0, "step": 0.1},
        "max_iter": {"kind": "int", "min": 100, "max": 5000, "step": 100},
    },
    "grad_boost": {
        "n_estimators": {"kind": "int", "min": 10, "max": 1000, "step": 10},
        "learning_rate": {"kind": "float", "min": 0.01, "max": 1.0, "step": 0.01},
        "max_depth": {"kind": "int", "min": 1, "max": 32, "step": 1},
    },
    "hist_grad_boost": {
        "max_iter": {"kind": "int", "min": 10, "max": 1000, "step": 10},
        "learning_rate": {"kind": "float", "min": 0.01, "max": 1.0, "step": 0.01},
        "max_depth": {"kind": "int", "min": 1, "max": 64, "step": 1},
    },
    "mlp": {
        "hidden_layer_sizes": {"kind": "choice",
                               "choices": [(16,), (32,), (64, 32), (128, 64), (128, 64, 32)]},
        "alpha": {"kind": "float", "min": 0.00001, "max": 1.0, "step": 0.0001},
        "max_iter": {"kind": "int", "min": 100, "max": 2000, "step": 100},
    },
    "linear_svc": {
        "C": {"kind": "float", "min": 0.01, "max": 100.0, "step": 0.1},
        "max_iter": {"kind": "int", "min": 100, "max": 10000, "step": 500},
    },
    "xgboost": {
        "n_estimators": {"kind": "int", "min": 10, "max": 2000, "step": 10},
        "learning_rate": {"kind": "float", "min": 0.01, "max": 1.0, "step": 0.01},
        "max_depth": {"kind": "int", "min": 1, "max": 32, "step": 1},
        "subsample": {"kind": "float", "min": 0.1, "max": 1.0, "step": 0.05},
        "colsample_bytree": {"kind": "float", "min": 0.1, "max": 1.0, "step": 0.05},
    },
    "lightgbm": {
        "n_estimators": {"kind": "int", "min": 10, "max": 2000, "step": 10},
        "learning_rate": {"kind": "float", "min": 0.01, "max": 1.0, "step": 0.01},
        "num_leaves": {"kind": "int", "min": 2, "max": 256, "step": 1},
        "max_depth": {"kind": "int", "min": -1, "max": 32, "step": 1},
        "subsample": {"kind": "float", "min": 0.1, "max": 1.0, "step": 0.05},
        "colsample_bytree": {"kind": "float", "min": 0.1, "max": 1.0, "step": 0.05},
    },
}


def param_schema(name: str) -> Dict[str, Dict[str, Any]]:
    """Tunable-parameter spec for a model (empty dict if none declared)."""
    return PARAM_SCHEMA.get(name, {})


def default_params(name: str) -> Dict[str, Any]:
    """Default hyperparameters a zoo model ships with (copy)."""
    return dict(MODEL_ZOO.get(name, {}).get("params", {}))


def available_models(task: Optional[TaskType] = None) -> List[str]:
    """Model names usable for a given task (linear_svc has no regressor)."""
    out = []
    for name, entry in MODEL_ZOO.items():
        if task == TaskType.REGRESSION and entry[_REG] is None:
            continue
        out.append(name)
    return out


def model_label(name: str) -> str:
    return MODEL_ZOO.get(name, {}).get("label", name)


def make_model(
    spec: ModelSpec | str,
    task: TaskType,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> TrustModel:
    """Build an (unfitted) TrustModel from a spec or model name.

    Parameters
    ----------
    spec : ModelSpec|str    either a full spec or a zoo key ("random_forest").
    task : TaskType         classification / regression -> picks the estimator.
    params : dict|None      overrides merged on top of the spec/zoo defaults.
    """
    if isinstance(spec, str):
        spec = ModelSpec(name=spec, task=task)
    spec = ModelSpec(name=spec.name, params={**spec.params}, task=task)

    if spec.name not in MODEL_ZOO:
        raise KeyError(
            f"unknown model '{spec.name}'; available: {list(MODEL_ZOO)}"
        )
    entry = MODEL_ZOO[spec.name]
    builder: Optional[Callable] = entry[_REG] if task == TaskType.REGRESSION else entry[_CLS]
    if builder is None:
        raise ValueError(f"model '{spec.name}' does not support {task.value}")

    merged = {**entry["params"], **spec.params, **(params or {})}
    # drop params the concrete builder does not accept
    est = _safe_build(builder, merged)
    spec.params = merged
    return TrustModel(est, task, feature_names, class_names, spec)


def _safe_build(builder: Callable, params: Dict[str, Any]):
    """Instantiate a sklearn estimator, ignoring unsupported kwargs."""
    import inspect

    try:
        return builder(**params)
    except TypeError:
        sig = inspect.signature(builder.__init__)
        allowed = {k: v for k, v in params.items() if k in sig.parameters}
        return builder(**allowed)
