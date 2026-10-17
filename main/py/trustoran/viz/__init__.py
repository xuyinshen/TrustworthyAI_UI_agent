"""viz sub-package: matplotlib Figure factories shared by notebooks and Qt."""
from . import labels
from . import plotting
from .labels import relabel_figure, translate
from .plotting import (
    PALETTE,
    apply_style,
    fig_class_distribution,
    fig_feature_histograms,
    fig_confusion_matrix,
    fig_roc_curves,
    fig_permutation_importance,
    fig_shap_summary,
    fig_shap_bar,
    fig_local_explanation,
    fig_pdp,
    fig_robustness_curve,
    fig_attack_comparison,
    fig_fairness_groups,
    fig_fairness_metrics,
    fig_dp_curve,
    fig_fedavg,
    fig_membership_inference,
    fig_regression_scatter,
    save_figure,
)

__all__ = [
    "plotting", "labels", "PALETTE", "apply_style", "relabel_figure", "translate",
    "fig_class_distribution", "fig_feature_histograms", "fig_confusion_matrix",
    "fig_roc_curves", "fig_permutation_importance", "fig_shap_summary",
    "fig_shap_bar", "fig_local_explanation", "fig_pdp", "fig_robustness_curve",
    "fig_attack_comparison", "fig_fairness_groups", "fig_fairness_metrics",
    "fig_dp_curve", "fig_fedavg", "fig_membership_inference",
    "fig_regression_scatter", "save_figure",
]
