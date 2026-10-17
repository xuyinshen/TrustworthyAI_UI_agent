"""Fairness pillar.

Group-fairness audit of RAN decisions. For classification we treat one class as
the "positive"/adversarial outcome (e.g. flagged-as-attack) and measure whether
detection rates differ across groups that a controller should treat equally
(A1 policy type, load tier, capture agent, CPU platform ...). For regression we
compare error and predicted levels across groups (e.g. energy model bias across
hardware platforms).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.base import TaskType
from ..models.wrappers import TrustModel
from .results import FairnessResult


def _safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def classification_fairness(
    model: TrustModel,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    group_attr: str,
    positive_class: int = 1,
    class_names: Optional[List[str]] = None,
) -> FairnessResult:
    groups = np.asarray(groups)
    pred = model.predict(X)
    y_pos = (y == positive_class).astype(int)
    pred_pos = (pred == positive_class).astype(int)

    rows = []
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        if n == 0:
            continue
        tp = int(((pred_pos == 1) & (y_pos == 1) & m).sum())
        fp = int(((pred_pos == 1) & (y_pos == 0) & m).sum())
        fn = int(((pred_pos == 0) & (y_pos == 1) & m).sum())
        tn = int(((pred_pos == 0) & (y_pos == 0) & m).sum())
        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        sel = _safe_div(tp + fp, n)
        acc = _safe_div(tp + tn, n)
        prec = _safe_div(tp, tp + fp)
        name = str(g)
        rows.append({
            "group": name, "n": n,
            "selection_rate": sel, "tpr": tpr, "fpr": fpr, "fnr": 1 - tpr if not np.isnan(tpr) else np.nan,
            "accuracy": acc, "precision": prec,
            "positives": int(y_pos[m].sum()),
        })
    tbl = pd.DataFrame(rows)

    metrics = {}
    if len(tbl) > 1:
        sel = tbl["selection_rate"].fillna(0).to_numpy()
        tpr = tbl["tpr"].fillna(0).to_numpy()
        fpr = tbl["fpr"].fillna(0).to_numpy()
        metrics["demographic_parity_difference"] = float(sel.max() - sel.min())
        metrics["demographic_parity_ratio"] = _safe_div(sel.min(), sel.max())
        metrics["disparate_impact"] = _safe_div(sel.min(), sel.max())
        metrics["equal_opportunity_difference"] = float(tpr.max() - tpr.min())
        metrics["equalized_odds_difference"] = float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min()))
        metrics["accuracy_gap"] = float(tbl["accuracy"].max() - tbl["accuracy"].min())

    pos_name = class_names[positive_class] if class_names and positive_class < len(class_names) else str(positive_class)
    return FairnessResult(
        task="classification", group_attr=group_attr, positive_label=pos_name,
        group_table=tbl, metrics=metrics,
    )


def regression_fairness(
    model: TrustModel,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    group_attr: str,
) -> FairnessResult:
    groups = np.asarray(groups)
    pred = model.predict(X)
    err = pred - y
    rows = []
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "group": str(g), "n": n,
            "mean_prediction": float(pred[m].mean()),
            "mean_actual": float(y[m].mean()),
            "mae": float(np.abs(err[m]).mean()),
            "rmse": float(np.sqrt((err[m] ** 2).mean())),
            "bias": float(err[m].mean()),
        })
    tbl = pd.DataFrame(rows)
    metrics = {}
    if len(tbl) > 1:
        metrics["mae_gap"] = float(tbl["mae"].max() - tbl["mae"].min())
        metrics["rmse_gap"] = float(tbl["rmse"].max() - tbl["rmse"].min())
        metrics["mean_prediction_gap"] = float(tbl["mean_prediction"].max() - tbl["mean_prediction"].min())
        metrics["max_abs_bias"] = float(tbl["bias"].abs().max())
    return FairnessResult(
        task="regression", group_attr=group_attr, positive_label=None,
        group_table=tbl, metrics=metrics,
    )


def analyze_fairness(
    model: TrustModel,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    group_attr: str,
    task: TaskType = TaskType.CLASSIFICATION,
    positive_class: int = 1,
    class_names: Optional[List[str]] = None,
) -> FairnessResult:
    """Single entry point dispatching to the classification / regression audit."""
    if task == TaskType.REGRESSION:
        return regression_fairness(model, X, y, groups, group_attr)
    return classification_fairness(model, X, y, groups, group_attr,
                                   positive_class=positive_class, class_names=class_names)
