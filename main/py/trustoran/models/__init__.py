"""models sub-package: unified estimator wrapper + factory."""
from .wrappers import TrustModel, ModelSpec
from .factory import (
    make_model, MODEL_ZOO, available_models, model_label,
    param_schema, default_params,
)

__all__ = [
    "TrustModel", "ModelSpec", "make_model", "MODEL_ZOO", "available_models",
    "model_label", "param_schema", "default_params",
]
