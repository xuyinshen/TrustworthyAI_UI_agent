"""Matplotlib figure factories.

Every function returns a ``matplotlib.figure.Figure`` (never calls ``plt.show``)
so the exact same plot renders inside a notebook cell and inside a Qt canvas.
Uses the non-interactive Agg backend by default; Qt re-parents the figure into a
``FigureCanvasQTAgg``.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # headless-safe; Qt swaps in its own canvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, roc_curve, auc

from ..data.base import TrustDataset
from ..trust.results import (
    DPResult, FairnessResult, FedAvgResult, GlobalImportance,
    LocalExplanation, MembershipInferenceResult, AttackResult,
    PDPResult, RobustnessCurve, ShapResult,
)

# O-RAN-ish palette
PALETTE = {
    "bg": "#0f1720",
    "panel": "#16212c",
    "ink": "#e6edf3",
    "muted": "#8b98a5",
    "accent": "#22d3ee",
    "series": ["#22d3ee", "#f472b6", "#fbbf24", "#34d399", "#a78bfa", "#fb7185"],
    "good": "#34d399",
    "bad": "#fb7185",
    "grid": "#243244",
}

_DARK = True


def apply_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply the shared dark theme to an axes."""
    ax.set_facecolor(PALETTE["panel"])
    ax.figure.patch.set_facecolor(PALETTE["bg"])
    ax.title.set_color(PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    for s in ax.spines.values():
        s.set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"])
    ax.grid(True, color=PALETTE["grid"], alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=PALETTE["ink"], fontsize=11, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def _new(figsize=(6.4, 4.0)):
    fig = Figure(figsize=figsize, dpi=110, facecolor=PALETTE["bg"])
    return fig


def _c(i):
    return PALETTE["series"][i % len(PALETTE["series"])]


def _brand_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "oran", [PALETTE["panel"], "#1f6f82", PALETTE["accent"]])


# ==========================================================================
# dataset overview
# ==========================================================================
def fig_class_distribution(ds: TrustDataset, title=None) -> Figure:
    fig = _new()
    ax = fig.add_subplot(111)
    counts = ds.class_counts()
    names = list(counts.keys())
    vals = [counts[k] for k in names]
    bars = ax.bar(names, vals, color=[_c(i) for i in range(len(names))])
    ax.bar_label(bars, fmt="%d", color=PALETTE["ink"], fontsize=9)
    apply_style(ax, title or f"{ds.name} — class balance", "class", "samples")
    fig.tight_layout()
    return fig


def fig_feature_histograms(ds: TrustDataset, max_features=12, title=None) -> Figure:
    feats = ds.feature_names[:max_features]
    ncols = 3
    nrows = int(np.ceil(len(feats) / ncols))
    fig = _new((9.0, 2.2 * nrows))
    classes = np.unique(ds.y) if ds.task.is_classification else None
    for i, f in enumerate(feats):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        col = ds.X[f].to_numpy(float)
        if classes is not None and len(classes) <= 6:
            for j, c in enumerate(classes):
                ax.hist(col[ds.y == c], bins=30, alpha=0.6, color=_c(j),
                        label=str(ds.class_names[c] if ds.class_names else c))
        else:
            ax.hist(col, bins=30, color=PALETTE["accent"], alpha=0.8)
        apply_style(ax, f, None, None)
        ax.set_title(f, color=PALETTE["ink"], fontsize=8)
        ax.tick_params(labelsize=6)
    if classes is not None:
        ax.legend(fontsize=6, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"])
    fig.suptitle(title or f"{ds.name} — feature distributions", color=PALETTE["ink"], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ==========================================================================
# model performance
# ==========================================================================
def fig_confusion_matrix(y_true, y_pred, class_names, title="Confusion matrix") -> Figure:
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = _new((4.8, 4.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(cmn, cmap=_brand_cmap(), vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color=PALETTE["muted"], rotation=20, fontsize=8)
    ax.set_yticklabels(class_names, color=PALETTE["muted"], fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center",
                    color=PALETTE["ink"], fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    apply_style(ax, title); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(colors=PALETTE["muted"])
    fig.tight_layout()
    return fig


def fig_roc_curves(y_true, proba, class_names, title="ROC (one-vs-rest)") -> Figure:
    fig = _new()
    ax = fig.add_subplot(111)
    n_cls = proba.shape[1]
    for c in range(n_cls):
        if n_cls == 2:
            yt = (np.asarray(y_true) == 1).astype(int)
            sc = proba[:, 1]
        else:
            yt = (np.asarray(y_true) == c).astype(int)
            sc = proba[:, c]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, sc)
        ax.plot(fpr, tpr, color=_c(c), lw=2,
                label=f"{class_names[c] if class_names else c} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], ls="--", lw=1)
    apply_style(ax, title, "false positive rate", "true positive rate")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"])
    fig.tight_layout()
    return fig


def fig_regression_scatter(y_true, y_pred, title="Predicted vs actual") -> Figure:
    fig = _new()
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, color=PALETTE["accent"])
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color=PALETTE["good"], lw=1.5, ls="--")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    apply_style(ax, f"{title}  (R²={r2:.3f})", "actual", "predicted")
    fig.tight_layout()
    return fig


# ==========================================================================
# explainability
# ==========================================================================
def fig_permutation_importance(gi: GlobalImportance, top=12, title=None) -> Figure:
    df = gi.to_frame().head(top).iloc[::-1]
    fig = _new((6.4, 0.42 * len(df) + 1.6))
    ax = fig.add_subplot(111)
    ax.barh(df["feature"], df["permutation"], xerr=df["permutation_std"],
            color=PALETTE["accent"], alpha=0.85, error_kw={"ecolor": PALETTE["muted"]})
    apply_style(ax, title or f"Permutation importance ({gi.metric})", "importance drop", "feature")
    fig.tight_layout()
    return fig


def _shap_tag(sh: ShapResult) -> str:
    return "exact" if sh.method == "exact" else "sampled"


def fig_shap_summary(sh: ShapResult, top=12, title=None) -> Figure:
    """Beeswarm: each dot is an instance, x = SHAP value, colour = feature value."""
    order = np.argsort(np.abs(sh.shap_values).mean(axis=0))[:top]
    fig = _new((6.6, 0.42 * len(order) + 1.6))
    ax = fig.add_subplot(111)
    for row, fi in enumerate(order):
        vals = sh.shap_values[:, fi]
        feat = sh.X[:, fi]
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(vals))
        sc = ax.scatter(vals, np.full_like(vals, row) + jitter, c=feat,
                        cmap="coolwarm", s=9, alpha=0.8, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([sh.feature_names[i] for i in order], fontsize=8, color=PALETTE["muted"])
    ax.axvline(0, color=PALETTE["muted"], lw=0.8)
    apply_style(ax, title or f"SHAP summary — {sh.target} [{_shap_tag(sh)}]",
                "SHAP value (impact on output)", "")
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=PALETTE["muted"]); cb.set_label("feature value", color=PALETTE["muted"], fontsize=8)
    fig.tight_layout()
    return fig


def fig_shap_bar(sh: ShapResult, top=12, title=None) -> Figure:
    df = sh.mean_abs().head(top).iloc[::-1]
    fig = _new((6.4, 0.42 * len(df) + 1.6))
    ax = fig.add_subplot(111)
    ax.barh(df["feature"], df["mean_abs_shap"], color=PALETTE["series"][1], alpha=0.9)
    apply_style(ax, title or f"Mean |SHAP| — {sh.target} [{_shap_tag(sh)}]",
                "mean |SHAP value|", "feature")
    fig.tight_layout()
    return fig


def fig_local_explanation(le: LocalExplanation, title=None) -> Figure:
    df = le.to_frame()
    fig = _new((6.4, 0.42 * len(df) + 1.8))
    ax = fig.add_subplot(111)
    colors = [PALETTE["bad"] if v < 0 else PALETTE["good"] for v in df["contribution"]]
    ax.barh(df["feature"], df["contribution"], color=colors, alpha=0.9)
    ax.axvline(0, color=PALETTE["muted"], lw=0.8)
    t = title or f"Why this decision?  (score={le.prediction:.3f}, base={le.base_value:.3f})"
    apply_style(ax, t, "contribution to output", "feature")
    fig.tight_layout()
    return fig


def fig_pdp(pdp: PDPResult, title=None) -> Figure:
    fig = _new()
    ax = fig.add_subplot(111)
    if pdp.ice_curves is not None:
        for c in pdp.ice_curves:
            ax.plot(pdp.grid, c, color=PALETTE["muted"], alpha=0.25, lw=0.7)
    ax.plot(pdp.grid, pdp.average_prediction, color=PALETTE["accent"], lw=2.4, label="average")
    apply_style(ax, title or f"Partial dependence — {pdp.feature_name}", pdp.feature_name, "model output")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"])
    fig.tight_layout()
    return fig


# ==========================================================================
# robustness
# ==========================================================================
def fig_robustness_curve(rc: RobustnessCurve, title=None) -> Figure:
    fig = _new()
    ax = fig.add_subplot(111)
    utility = getattr(rc, "metric", "accuracy") == "utility"
    if utility:
        ax.plot(rc.epsilons, rc.accuracies, color=PALETTE["bad"], lw=2.2, marker="o",
                ms=4, label=rc.attack)
        ax.axhline(0.5, color=PALETTE["muted"], ls=":", lw=1.1,
                   label="utility 0.5 = error doubled")
        ylab = "noise-robust utility (MAE0/MAE)"
        xlab = "input noise σ (std units)"
    else:
        ax.plot(rc.epsilons, rc.accuracies, color=PALETTE["bad"], lw=2.2, marker="o",
                ms=4, label=f"{rc.attack} attack")
        ax.axhline(rc.clean_accuracy, color=PALETTE["good"], ls="--", lw=1.2,
                   label=f"clean acc = {rc.clean_accuracy:.3f}")
        ylab = "accuracy"
        xlab = "perturbation budget ε (std units)"
    apply_style(ax, title or f"Robustness — {rc.attack}", xlab, ylab)
    ax.legend(fontsize=8, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"])
    fig.tight_layout()
    return fig


def fig_attack_comparison(results, title="Attack comparison") -> Figure:
    fig = _new()
    ax = fig.add_subplot(111)
    labels = [f"{r.attack}\nε={r.eps:g}" for r in results]
    clean = [r.clean_accuracy for r in results]
    rob = [r.robust_accuracy for r in results]
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, clean, w, color=PALETTE["good"], label="clean")
    ax.bar(x + w/2, rob, w, color=PALETTE["bad"], label="under attack")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, color=PALETTE["muted"])
    apply_style(ax, title, None, "accuracy")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"])
    fig.tight_layout()
    return fig


# ==========================================================================
# fairness
# ==========================================================================
def fig_fairness_groups(fr: FairnessResult, title=None) -> Figure:
    tbl = fr.group_table
    metric = "selection_rate" if fr.task == "classification" else "mae"
    fig = _new((6.6, 4.0))
    ax = fig.add_subplot(111)
    bars = ax.bar(tbl["group"].astype(str), tbl[metric], color=[_c(i) for i in range(len(tbl))])
    ax.bar_label(bars, fmt="%.2f", color=PALETTE["ink"], fontsize=8)
    lbl = "flag rate (selection rate)" if fr.task == "classification" else "MAE"
    t = title or f"Fairness by {fr.group_attr} — {lbl}"
    apply_style(ax, t, fr.group_attr, lbl)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_fairness_metrics(fr: FairnessResult, title="Disparity metrics") -> Figure:
    keys = [k for k in fr.metrics if isinstance(fr.metrics[k], (int, float)) and not np.isnan(fr.metrics[k])]
    fig = _new((6.8, 0.5 * max(len(keys), 1) + 1.4))
    ax = fig.add_subplot(111)
    vals = [fr.metrics[k] for k in keys]
    colors = [PALETTE["bad"] if (k.endswith("difference") or k.endswith("gap")) and v > 0.1
              else PALETTE["good"] if k.endswith("ratio") and v > 0.8 else PALETTE["accent"]
              for k, v in zip(keys, vals)]
    ax.barh(keys[::-1], vals[::-1], color=colors[::-1], alpha=0.9)
    apply_style(ax, title, "value (0 = perfect parity)", "")
    fig.tight_layout()
    return fig


# ==========================================================================
# privacy
# ==========================================================================
def fig_dp_curve(dp: DPResult, title=None) -> Figure:
    fig = _new()
    ax = fig.add_subplot(111)
    xs = np.arange(len(dp.epsilons))
    labels = ["∞\n(open)" if np.isinf(e) else f"{e:g}" for e in dp.epsilons]
    util_name = "accuracy" if "class" in dp.task or dp.task in ("binary", "classification") else "R²"
    ax.plot(xs, dp.utilities, color=PALETTE["accent"], lw=2.2, marker="o")
    ax.axhline(dp.clean_utility, color=PALETTE["good"], ls="--", lw=1, label=f"private-agnostic = {dp.clean_utility:.3f}")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8, color=PALETTE["muted"])
    apply_style(ax, title or f"Privacy–utility trade-off ({dp.mechanism})", "privacy budget ε", util_name)
    ax.legend(fontsize=8, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"])
    fig.tight_layout()
    return fig


def fig_fedavg(fr: FedAvgResult, title="Federated vs centralised vs local-only") -> Figure:
    fig = _new((7.0, 4.2))
    ax = fig.add_subplot(111)
    rounds = np.arange(1, fr.n_rounds + 1)
    ax.plot(rounds, fr.federated_accuracies, color=PALETTE["accent"], lw=2.4,
            marker="o", label=f"FedAvg ({fr.n_clients} agents)")
    ax.axhline(fr.centralized_accuracy, color=PALETTE["good"], ls="--", lw=1.4,
               label=f"centralised = {fr.centralized_accuracy:.3f}")
    for i, a in enumerate(fr.local_only_accuracies):
        ax.axhline(a, color=PALETTE["bad"], alpha=0.35, lw=1,
                   label="local-only (per agent)" if i == 0 else None)
    apply_style(ax, title, "communication round", "global test accuracy")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"], loc="lower right")
    fig.tight_layout()
    return fig


_MI_ATTACK_LABELS = {
    "threshold": "threshold\n(conf)",
    "lr_conf": "meta\n(label-free)",
    "lr_meta": "meta\n(+label)",
    "ssa": "SSA\n(shadow)",
    "lira": "LiRA\n(shadow)",
}


def fig_membership_inference(mi: MembershipInferenceResult, title="Membership-inference leakage") -> Figure:
    per_attack = {k: v for k, v in (mi.attack_aucs or {}).items()
                  if k in _MI_ATTACK_LABELS and np.isfinite(v)}
    if not per_attack:
        per_attack = {"lr_meta": mi.attack_auc}
    keys = [k for k in _MI_ATTACK_LABELS if k in per_attack]
    vals = [per_attack[k] for k in keys]

    fig = _new((6.4, 4.0) if len(keys) > 1 else (5.2, 4.0))
    ax = fig.add_subplot(111)
    ax.bar([_MI_ATTACK_LABELS[k] for k in keys], vals,
           color=[PALETTE["bad"] if v > 0.6 else PALETTE["good"] for v in vals], width=0.55)
    ax.axhline(0.5, color=PALETTE["muted"], ls="--", lw=1, label="0.5 = no leakage")
    ax.set_ylim(0, 1)
    ax.bar_label(ax.containers[0], fmt="%.3f", color=PALETTE["ink"], fontsize=8)

    sub = f"AUC={mi.attack_auc:.3f}"
    if mi.n_shadows:
        sub += f" | {mi.n_shadows} shadow models"
    if np.isfinite(mi.ks_stat):
        sub += f" | KS={mi.ks_stat:.3f}"
    ax.tick_params(axis="x", labelsize=8)
    apply_style(ax, f"{title}\n({sub})", None, "ROC-AUC of the membership attack")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"])
    fig.tight_layout()
    return fig


# ==========================================================================
# helpers
# ==========================================================================
def save_figure(fig: Figure, path) -> str:
    """Save a figure to ``path`` (creating parent dirs) and return the path."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=130)
    return str(path)
