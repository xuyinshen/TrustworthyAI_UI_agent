"""Plain result containers returned by every trust-pillar analysis.

These are pure data (numbers / arrays / small tables) so they can be:
  * rendered by :mod:`trustoran.viz` into matplotlib Figures,
  * serialised to JSON/CSV by the pipeline,
  * passed straight into the Qt panels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------- Explainability -----------------------------
@dataclass
class GlobalImportance:
    feature_names: List[str]
    permutation_importance: np.ndarray
    permutation_std: np.ndarray
    native_importance: Optional[np.ndarray] = None     # gini / |coef|
    metric: str = "accuracy"

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "feature": self.feature_names,
            "permutation": self.permutation_importance,
            "permutation_std": self.permutation_std,
        })
        if self.native_importance is not None:
            df["native"] = self.native_importance
        return df.sort_values("permutation", ascending=False).reset_index(drop=True)


@dataclass
class LocalExplanation:
    feature_names: List[str]
    coefficients: np.ndarray           # local surrogate weights (signed)
    base_value: float                  # surrogate intercept / background mean
    prediction: float                  # model score at x
    method: str = "LIME-style local surrogate"

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "feature": self.feature_names,
            "contribution": self.coefficients,
        })
        df["abs"] = df["contribution"].abs()
        return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


@dataclass
class PDPResult:
    feature_name: str
    grid: np.ndarray                   # feature values along the axis
    average_prediction: np.ndarray     # mean model response
    ice_curves: Optional[np.ndarray] = None   # (n_instances, len(grid))


@dataclass
class ShapResult:
    feature_names: List[str]
    shap_values: np.ndarray            # (n_instances, n_features) for the target output
    base_value: float
    X: np.ndarray                      # the explained instances (for beeswarm scatter)
    target: str = "score"
    method: str = "kernel"                # "kernel" | "exact"
    efficiency_max_error: float = float("nan")   # exact path only: max|base+sum(phi)-margin|

    def mean_abs(self) -> pd.DataFrame:
        m = np.abs(self.shap_values).mean(axis=0)
        return pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": m}) \
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ------------------------------ Robustness --------------------------------
@dataclass
class AttackResult:
    attack: str                        # "fgsm" | "pgd" | "deepfool"
    eps: float
    clean_accuracy: float
    robust_accuracy: float
    attack_success_rate: float
    mean_perturbation: float
    perturbed_accuracy_by_class: Dict[str, float] = field(default_factory=dict)


@dataclass
class RobustnessCurve:
    attack: str
    epsilons: np.ndarray
    accuracies: np.ndarray             # main reading at each eps -- see ``metric``
    clean_accuracy: float
    adversarial_training_accuracies: Optional[np.ndarray] = None
    #: what ``accuracies`` holds: "accuracy" (classification) or "utility"
    #: (regression retention MAE0/MAE, which lives in (0, 1] and never goes below 0)
    metric: str = "accuracy"
    #: raw regression measurements per level, kept for the tables
    r2: Optional[np.ndarray] = None
    mae: Optional[np.ndarray] = None


# ------------------------------- Fairness ---------------------------------
@dataclass
class FairnessResult:
    task: str                          # "classification" | "regression"
    group_attr: str
    positive_label: Optional[str]
    group_table: pd.DataFrame          # per-group metrics
    metrics: Dict[str, float]          # aggregate disparity metrics

    @property
    def groups(self) -> List[str]:
        return self.group_table["group"].astype(str).tolist()


# -------------------------------- Privacy ---------------------------------
@dataclass
class DPResult:
    epsilons: np.ndarray
    utilities: np.ndarray              # accuracy / r2 at each epsilon
    clean_utility: float               # utility with no privacy noise
    mechanism: str                     # "gaussian" | "laplace"
    task: str
    delta: float = 1e-5


@dataclass
class FedAvgResult:
    n_clients: int
    n_rounds: int
    federated_accuracies: np.ndarray   # global-model test accuracy per round
    centralized_accuracy: float
    local_only_accuracies: List[float] # each client training alone (no sharing)
    client_names: List[str]
    privacy_note: str = ""


@dataclass
class MembershipInferenceResult:
    attack_auc: float                  # >0.5 => membership is inferable (leak)
    attack_accuracy: float
    n_members: int
    n_nonmembers: int
    note: str = ""
    attack_aucs: Dict[str, float] = field(default_factory=dict)  # per-attack AUC
    ks_stat: float = float("nan")            # member/non-member distribution shift
    tpr_at_1pct_fpr: float = float("nan")    # attack yield at a 1% false-alarm budget
    confidence_gap: float = float("nan")     # mean conf(train) - mean conf(test)
    generalization_gap: float = float("nan")   # train accuracy - test accuracy
    near_duplicate_rate: float = float("nan")  # test rows verbatim in train
    n_shadows: int = 0
    feature_set: str = ""
