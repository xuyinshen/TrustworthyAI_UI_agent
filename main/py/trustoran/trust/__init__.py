"""trust sub-package: the four Trustworthy-AI pillars.

Each pillar exposes functions that consume a fitted :class:`TrustModel` plus
data and return plain result dataclasses (numbers + arrays). Rendering lives in
:mod:`trustoran.viz` so the *same* result objects feed the notebooks and the Qt
UI without duplication.
"""
from . import explainability, robustness, fairness, privacy
from .results import (
    GlobalImportance,
    LocalExplanation,
    PDPResult,
    ShapResult,
    AttackResult,
    RobustnessCurve,
    FairnessResult,
    DPResult,
    FedAvgResult,
    MembershipInferenceResult,
)

__all__ = [
    "explainability", "robustness", "fairness", "privacy",
    "GlobalImportance", "LocalExplanation", "PDPResult", "ShapResult",
    "AttackResult", "RobustnessCurve", "FairnessResult", "DPResult",
    "FedAvgResult", "MembershipInferenceResult",
]
