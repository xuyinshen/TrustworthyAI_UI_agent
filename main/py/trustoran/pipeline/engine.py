"""The orchestration engine.

:class:`TrustPipeline` is the single object the CLI, the notebooks and the Qt UI
drive. It loads a dataset, trains a model, runs the requested trust pillars and
returns an :class:`AnalysisReport` holding every metric, table and figure. A
``progress_cb`` hook lets the Qt UI show a live progress bar without the engine
knowing anything about Qt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config as C
from ..data import build_dataset, train_test_split_dataset
from ..data.base import TaskType, TrustDataset
from ..data.preprocessing import DataSplit
from ..models import make_model
from ..models.wrappers import TrustModel
from ..trust import explainability as EX
from ..trust import fairness as FA
from ..trust import privacy as PR
from ..trust import robustness as RB
from ..viz import plotting as VZ
from .report import AnalysisReport, PillarResult

PILLARS = ("explainability", "robustness", "fairness", "privacy")

# per-dataset tuning: which group is "protected", which class is the adverse
# outcome, whether the dataset can be federated, and analysis sizes.
DATASET_PROFILES: Dict[str, Dict] = {
    "cci_a1sd": dict(
        fairness_group="policy_name", positive_class=1, federate=False,
        shap_target_class=1, note="A1 traffic: Normal/DoS/Covert detection",
    ),
    "cci_a1sd_binary": dict(
        fairness_group="policy_name", positive_class=1, federate=False,
        shap_target_class=1, note="A1 traffic: Normal vs DoS",
    ),
    "ddos_pcap": dict(
        fairness_group="agent", positive_class=1, federate=True, federate_group="agent",
        shap_target_class=1, note="Near-RT RIC edge DDoS detection across 5 agents",
    ),
    "vran_energy": dict(
        fairness_group="cpu_platform", regression=True, note="BBU power regression",
    ),
    "vran_application": dict(
        fairness_group="gpu_platform", regression=True, note="Edge-AI end-to-end delay regression",
    ),
    "vran_computing": dict(
        fairness_group="n_vbs", positive_class=1, federate=False,
        shap_target_class=1, note="vRAN deployment failure (explode) classification",
    ),
}

ProgressCB = Optional[Callable[[str, float, str], None]]


def _cap(X: np.ndarray, y: np.ndarray, n: int, seed: int = 42,
         groups: Optional[Dict[str, np.ndarray]] = None):
    if n is None or len(X) <= n:
        return (X, y, groups) if groups is not None else (X, y)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    idx.sort()
    if groups is not None:
        return X[idx], y[idx], {k: np.asarray(v)[idx] for k, v in groups.items()}
    return X[idx], y[idx]


@dataclass
class _Ctx:
    dataset_key: str
    ds: TrustDataset
    split: DataSplit
    model: TrustModel
    profile: Dict
    quick: bool
    seed: int


class TrustPipeline:
    """End-to-end trustworthy-AI analysis for one dataset."""

    def __init__(
        self,
        dataset_key: str = "cci_a1sd",
        model_name: str = "random_forest",
        pillars: Sequence[str] = PILLARS,
        max_rows: Optional[int] = None,
        train_cap: Optional[int] = 20_000,
        quick: bool = False,
        seed: int = 42,
        model_params: Optional[Dict] = None,
    ) -> None:
        if dataset_key not in DATASET_PROFILES:
            raise KeyError(f"unknown dataset '{dataset_key}'; have {list(DATASET_PROFILES)}")
        self.dataset_key = dataset_key
        self.model_name = model_name
        self.pillars = tuple(pillars)
        self.max_rows = max_rows if max_rows is not None else (12_000 if quick else 40_000)
        self.train_cap = 4_000 if quick else train_cap
        self.quick = quick
        self.seed = seed
        self.model_params = model_params or {}
        self.profile = DATASET_PROFILES[dataset_key]

    # -- public ------------------------------------------------------------
    def run(self, progress_cb: ProgressCB = None) -> AnalysisReport:
        t0 = time.time()
        self._p(progress_cb, "load", 0.02, f"loading {self.dataset_key}")
        ds = build_dataset(self.dataset_key, max_rows=self.max_rows, seed=self.seed)

        self._p(progress_cb, "split", 0.08, "splitting & scaling")
        split = train_test_split_dataset(ds, test_size=0.3, seed=self.seed)
        # cap the working size so analysis stays responsive
        if self.train_cap and split.X_train.shape[0] > self.train_cap:
            rng = np.random.default_rng(self.seed)
            idx = np.sort(rng.choice(split.X_train.shape[0], self.train_cap, replace=False))
            split.X_train, split.y_train = split.X_train[idx], split.y_train[idx]
            mask = np.zeros(split.train.n_samples, dtype=bool)
            mask[idx] = True
            split.train = split.train.subset(mask)

        self._p(progress_cb, "train", 0.16, f"training {self.model_name}")
        model = make_model(self.model_name, split.train.task, split.feature_names,
                           split.train.class_names, params=self.model_params)
        model.fit(split.X_train, split.y_train)
        base_metrics = model.score(split.X_test, split.y_test)

        ctx = _Ctx(self.dataset_key, ds, split, model, self.profile, self.quick, self.seed)

        self._p(progress_cb, "overview", 0.26, "building overview figures")
        base_figs = self._base_figures(ctx)

        report = AnalysisReport(
            dataset_key=self.dataset_key,
            dataset_summary=ds.describe(),
            model_spec={"name": self.model_name, "label": model.spec.name,
                        "params": model.spec.params, "task": ds.task.value},
            base_metrics=base_metrics,
            base_figures=base_figs,
        )

        n = len(self.pillars)
        for i, pillar in enumerate(self.pillars):
            lo = 0.3 + 0.68 * (i / max(n, 1))
            hi = 0.3 + 0.68 * ((i + 1) / max(n, 1))
            self._p(progress_cb, pillar, lo, f"running {pillar}")
            fn = getattr(self, f"_pillar_{pillar}", None)
            if fn is None:
                continue
            try:
                report.pillars[pillar] = fn(ctx, progress_cb, lo, hi)
            except Exception as e:                       # keep the run alive
                report.pillars[pillar] = PillarResult(
                    pillar=pillar, summary={"error": f"{type(e).__name__}: {e}"})
        self._p(progress_cb, "done", 1.0, f"finished in {time.time()-t0:.1f}s")
        report.model_spec["elapsed_sec"] = round(time.time() - t0, 2)
        return report

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _p(cb: ProgressCB, stage: str, pct: float, msg: str) -> None:
        if cb is not None:
            cb(stage, float(pct), msg)

    def _base_figures(self, ctx: _Ctx) -> Dict[str, object]:
        split, model, ds = ctx.split, ctx.model, ctx.ds
        figs = {"class_distribution": VZ.fig_class_distribution(ds)}
        figs["feature_histograms"] = VZ.fig_feature_histograms(ds, max_features=12)
        if ds.task.is_classification:
            y_pred = model.predict(split.X_test)
            figs["confusion_matrix"] = VZ.fig_confusion_matrix(
                split.y_test, y_pred, ds.class_names or [str(i) for i in range(ds.n_classes)])
            if hasattr(model.estimator, "predict_proba"):
                proba = model.predict_proba(split.X_test)
                figs["roc"] = VZ.fig_roc_curves(split.y_test, proba, ds.class_names)
        else:
            figs["regression_scatter"] = VZ.fig_regression_scatter(
                split.y_test, model.predict(split.X_test))
        return figs

    def _bg_and_explain(self, ctx: _Ctx) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(ctx.seed)
        bg = ctx.split.X_train[rng.choice(len(ctx.split.X_train),
                                          min(120, len(ctx.split.X_train)), replace=False)]
        n_ex = 30 if ctx.quick else 60
        ex = ctx.split.X_test[rng.choice(len(ctx.split.X_test),
                                         min(n_ex, len(ctx.split.X_test)), replace=False)]
        return bg, ex

    # ===================== EXPLAINABILITY =====================
    def _pillar_explainability(self, ctx: _Ctx, cb, lo, hi) -> PillarResult:
        split, model = ctx.split, ctx.model
        res = PillarResult(pillar="explainability")
        self._p(cb, "explainability", lo + 0.1 * (hi - lo), "permutation importance")
        gi = EX.global_importance(model, split.X_test, split.y_test,
                                  n_repeats=4 if ctx.quick else 8, seed=ctx.seed)
        res.tables["permutation_importance"] = gi.to_frame()
        res.figures["permutation_importance"] = VZ.fig_permutation_importance(gi)

        bg, ex = self._bg_and_explain(ctx)
        tc = ctx.profile.get("shap_target_class")
        self._p(cb, "explainability", lo + 0.4 * (hi - lo), "SHAP attributions")
        sh = EX.shap_explain(model, ex, bg, target_class=tc, seed=ctx.seed)
        res.tables["shap_mean_abs"] = sh.mean_abs()
        res.figures["shap_summary"] = VZ.fig_shap_summary(sh)
        res.figures["shap_bar"] = VZ.fig_shap_bar(sh)
        res.artifacts["shap"] = sh

        self._p(cb, "explainability", lo + 0.6 * (hi - lo), "local explanation")
        xi = self._pick_explainable_sample(ctx)
        le = EX.local_explanation(model, split.X_test[xi], bg, target_class=tc, seed=ctx.seed)
        res.tables["local_explanation"] = le.to_frame()
        res.figures["local_explanation"] = VZ.fig_local_explanation(le)
        res.artifacts["local"] = le
        res.artifacts["local_index"] = int(xi)

        self._p(cb, "explainability", lo + 0.8 * (hi - lo), "partial dependence")
        top_feats = gi.to_frame()["feature"].tolist()[:2]
        pdp_subset = split.X_test[: 400 if not ctx.quick else 200]
        for k, fname in enumerate(top_feats):
            fi = model.feature_names.index(fname)
            pdp = EX.partial_dependence(model, pdp_subset, fi, grid_resolution=15)
            res.figures[f"pdp_{fname}"] = VZ.fig_pdp(pdp)
            res.artifacts[f"pdp_{fname}"] = pdp

        shap_label = ("exact TreeSHAP (native pred_contribs)" if sh.method == "exact"
                      else "KernelSHAP sampling, at most 8 background rows")
        res.summary = {
            "top_features": gi.to_frame()["feature"].tolist()[:5],
            "target": sh.target,
            "shap_top": sh.mean_abs()["feature"].tolist()[:5],
            "method": f"permutation + {shap_label} + LIME-style local + PDP/ICE",
            "shap_method": sh.method,
            "shap_efficiency_max_error": _finite_or_none(sh.efficiency_max_error),
        }
        res.artifacts["global_importance"] = gi
        return res

    def _pick_explainable_sample(self, ctx: _Ctx) -> int:
        split, model = ctx.split, ctx.model
        pred = model.predict(split.X_test)
        pc = ctx.profile.get("positive_class", 1)
        if split.train.task.is_classification:
            cand = np.where((split.y_test == pc) & (pred == pc))[0]
            if len(cand):
                return int(cand[0])
        return 0

    # ===================== ROBUSTNESS =====================
    def _pillar_robustness(self, ctx: _Ctx, cb, lo, hi) -> PillarResult:
        split, model = ctx.split, ctx.model
        res = PillarResult(pillar="robustness")
        if not split.train.task.is_classification:
            self._p(cb, "robustness", lo + 0.5 * (hi - lo), "noise drift (regression)")
            Xs, ys = _cap(split.X_test, split.y_test, 800, ctx.seed)
            rc = RB.regression_noise_drift(model, Xs, ys, seed=ctx.seed)
            res.figures["noise_drift"] = VZ.fig_robustness_curve(rc)
            res.tables["noise_drift"] = _curve_frame(rc)
            res.artifacts["curve"] = rc
            res.summary = {"mode": "regression noise drift",
                           "utility_at_max_sigma": float(rc.accuracies[-1]),
                           "mae_clean": float(rc.mae[0]),
                           "mae_at_max_sigma": float(rc.mae[-1]),
                           "clean_r2": float(rc.r2[0]),
                           "r2_at_max_sigma": float(rc.r2[-1])}
            return res

        n_sub = 150 if ctx.quick else 400
        Xs, ys = _cap(split.X_test, split.y_test, n_sub, ctx.seed)
        eps_rep = 0.3
        self._p(cb, "robustness", lo + 0.2 * (hi - lo), "FGSM / PGD / DeepFool")
        attacks = [
            RB.evaluate_attack(model, Xs, ys, "fgsm", eps_rep, split.train.class_names),
            RB.evaluate_attack(model, Xs, ys, "pgd", eps_rep, split.train.class_names, seed=ctx.seed),
        ]
        dfool_sub = min(60 if ctx.quick else 120, len(Xs))
        attacks.append(RB.evaluate_attack(model, Xs[:dfool_sub], ys[:dfool_sub], "deepfool", 0.0,
                                          split.train.class_names))
        res.tables["attacks"] = _attacks_frame(attacks)
        res.figures["attack_comparison"] = VZ.fig_attack_comparison(attacks)

        eps_range = (0.0, 0.1, 0.3, 0.6, 1.0) if ctx.quick else (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
        self._p(cb, "robustness", lo + 0.55 * (hi - lo), "robustness curves")
        rc_fgsm = RB.robustness_curve(model, Xs, ys, "fgsm", eps_range=eps_range)
        rc_pgd = RB.robustness_curve(model, Xs, ys, "pgd", eps_range=eps_range, seed=ctx.seed)
        res.figures["robustness_fgsm"] = VZ.fig_robustness_curve(rc_fgsm)
        res.figures["robustness_pgd"] = VZ.fig_robustness_curve(rc_pgd)
        res.tables["robustness_fgsm"] = _curve_frame(rc_fgsm)
        res.tables["robustness_pgd"] = _curve_frame(rc_pgd)

        self._p(cb, "robustness", lo + 0.85 * (hi - lo), "adversarial training")
        Xtr, ytr = _cap(split.X_train, split.y_train, 3000 if not ctx.quick else 1500, ctx.seed)
        rc_at = RB.adversarial_training(model, Xtr, ytr, Xs, ys, eps=0.3,
                                        eval_eps_range=eps_range, seed=ctx.seed)
        res.figures["adversarial_training"] = VZ.fig_robustness_curve(
            rc_at, title="Adversarially-trained model — robustness")
        res.tables["adversarial_training"] = _curve_frame(rc_at)

        res.summary = {
            "clean_accuracy": attacks[0].clean_accuracy,
            "fgsm_robust_acc": attacks[0].robust_accuracy,
            "pgd_robust_acc": attacks[1].robust_accuracy,
            "deepfool_robust_acc": attacks[2].robust_accuracy,
            "pgd_attack_success_rate": attacks[1].attack_success_rate,
            "adv_trained_robust_acc@eps0.6": _at_eps(rc_at, 0.6),
        }
        res.artifacts.update({"attacks": attacks, "rc_fgsm": rc_fgsm,
                              "rc_pgd": rc_pgd, "rc_advtrain": rc_at})
        return res

    # ===================== FAIRNESS =====================
    def _pillar_fairness(self, ctx: _Ctx, cb, lo, hi) -> PillarResult:
        split, model, ds = ctx.split, ctx.model, ctx.ds
        res = PillarResult(pillar="fairness")
        group_attr = ctx.profile.get("fairness_group")
        avail = list(split.test.groups.keys())
        if group_attr not in avail:
            group_attr = avail[0] if avail else None
        if group_attr is None:
            res.summary = {"skipped": "no group attribute available"}
            return res
        self._p(cb, "fairness", lo + 0.5 * (hi - lo), f"auditing groups by {group_attr}")
        groups = split.test.group_values(group_attr)
        fr = FA.analyze_fairness(
            model, split.X_test, split.y_test, groups, group_attr,
            task=split.train.task, positive_class=ctx.profile.get("positive_class", 1),
            class_names=ds.class_names,
        )
        res.tables["group_metrics"] = fr.group_table
        res.tables["disparity_metrics"] = _metrics_frame(fr.metrics)
        res.figures["fairness_groups"] = VZ.fig_fairness_groups(fr)
        res.figures["fairness_metrics"] = VZ.fig_fairness_metrics(fr)
        res.artifacts["fairness"] = fr
        res.summary = {
            "group_attr": group_attr,
            "n_groups": len(fr.group_table),
            **{k: (None if isinstance(v, float) and np.isnan(v) else round(v, 4))
               for k, v in fr.metrics.items()},
        }
        return res

    # ===================== PRIVACY =====================
    def _pillar_privacy(self, ctx: _Ctx, cb, lo, hi) -> PillarResult:
        split, model, ds = ctx.split, ctx.model, ctx.ds
        res = PillarResult(pillar="privacy")

        self._p(cb, "privacy", lo + 0.25 * (hi - lo), "differential privacy curve")
        n_dp = 1500 if ctx.quick else 4000
        Xtr, ytr = _cap(split.X_train, split.y_train, n_dp, ctx.seed)
        Xte, yte = _cap(split.X_test, split.y_test, n_dp, ctx.seed)
        eps_range = (float("inf"), 8.0, 4.0, 2.0, 1.0, 0.5, 0.1)
        dp = PR.privacy_utility_curve(
            Xtr, ytr, Xte, yte, split.train.task, eps_range=eps_range,
            epochs=12 if ctx.quick else 25, hidden=24 if ctx.quick else 32, seed=ctx.seed)
        res.figures["dp_curve"] = VZ.fig_dp_curve(dp)
        res.tables["dp_curve"] = _dp_frame(dp)
        res.artifacts["dp"] = dp
        res.summary = {
            "dp_mechanism": dp.mechanism,
            "utility_open": dp.clean_utility,
            "utility_eps1": _dp_at(dp, 1.0),
            "utility_eps0.1": _dp_at(dp, 0.1),
        }

        # federated learning (only when the dataset has an agent/group split)
        if ctx.profile.get("federate"):
            gattr = ctx.profile.get("federate_group", "agent")
            if gattr in split.train.groups:
                self._p(cb, "privacy", lo + 0.6 * (hi - lo), "FedAvg across agents")
                agents = split.train.group_values(gattr)
                clients = {}
                for a in np.unique(agents):
                    m = agents == a
                    if m.sum() >= 2 and len(np.unique(split.y_train[m])) >= 1:
                        clients[str(a)] = (split.X_train[m], split.y_train[m])
                K = int(max(split.y_train.max() + 1, 2))
                fr = PR.fedavg(clients, split.X_test, split.y_test, K=K,
                               n_rounds=5 if ctx.quick else 8, local_epochs=6,
                               batch=8 if ctx.quick else 16, hidden=24, seed=ctx.seed)
                res.figures["fedavg"] = VZ.fig_fedavg(fr)
                res.tables["fedavg_rounds"] = _fed_frame(fr)
                res.artifacts["fedavg"] = fr
                res.summary.update({
                    "federated_accuracy": float(fr.federated_accuracies[-1]),
                    "centralized_accuracy": fr.centralized_accuracy,
                    "mean_local_only": float(np.mean(fr.local_only_accuracies)),
                    "n_clients": fr.n_clients,
                })

        # membership inference (classification only)
        if split.train.task.is_classification:
            self._p(cb, "privacy", lo + 0.9 * (hi - lo), "membership inference")
            nmi = 1500 if ctx.quick else 3000
            Xtrm, ytrm = _cap(split.X_train, split.y_train, nmi, ctx.seed)
            Xtem, ytem = _cap(split.X_test, split.y_test, nmi, ctx.seed)
            mi = PR.membership_inference(model, Xtrm, ytrm, Xtem, ytem, seed=ctx.seed,
                                         n_shadows=4 if ctx.quick else 8)
            res.figures["membership_inference"] = VZ.fig_membership_inference(mi)
            metrics = {"attack_auc": mi.attack_auc, "attack_accuracy": mi.attack_accuracy}
            for k in ("threshold", "lr_conf", "lr_meta", "ssa", "lira", "ssa_accuracy"):
                v = mi.attack_aucs.get(k)
                if v is not None and np.isfinite(v):
                    metrics["auc_" + k] = v
            for k in ("ks_stat", "tpr_at_1pct_fpr", "confidence_gap",
                      "generalization_gap", "near_duplicate_rate"):
                v = getattr(mi, k)
                if np.isfinite(v):
                    metrics[k] = v
            metrics["n_shadows"] = mi.n_shadows
            res.tables["membership_inference"] = _metrics_frame(metrics)
            res.artifacts["mi"] = mi
            res.summary.update({
                "membership_inference_auc": mi.attack_auc,
                "mia_shadow_auc": _finite_or_none(mi.attack_aucs.get("ssa")),
                "mia_lira_auc": _finite_or_none(mi.attack_aucs.get("lira")),
                "mia_conf_only_auc": _finite_or_none(mi.attack_aucs.get("lr_conf")),
                "mia_ks": _finite_or_none(mi.ks_stat),
                "mia_tpr_at_1pct_fpr": _finite_or_none(mi.tpr_at_1pct_fpr),
                "mia_n_shadows": mi.n_shadows,
            })
        return res


# --------------------------------------------------------------------------
# small DataFrame builders for tables / export
# --------------------------------------------------------------------------
def _curve_frame(rc) -> "pd.DataFrame":
    import pandas as pd
    if getattr(rc, "metric", "accuracy") == "utility":
        cols = {"sigma": rc.epsilons, "utility": rc.accuracies}
        if rc.mae is not None:
            cols["mae"] = rc.mae
        if rc.r2 is not None:
            cols["r2"] = rc.r2
        return pd.DataFrame(cols)
    return pd.DataFrame({"epsilon": rc.epsilons, "accuracy": rc.accuracies})


def _attacks_frame(attacks) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([{
        "attack": a.attack, "eps": a.eps, "clean_accuracy": a.clean_accuracy,
        "robust_accuracy": a.robust_accuracy, "attack_success_rate": a.attack_success_rate,
        "mean_perturbation": a.mean_perturbation,
    } for a in attacks])


def _metrics_frame(metrics: Dict) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])


def _finite_or_none(v) -> Optional[float]:
    """Keep the JSON export valid when an attack produced no score."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _dp_frame(dp) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame({"epsilon": dp.epsilons, "utility": dp.utilities})


def _fed_frame(fr) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame({"round": np.arange(1, len(fr.federated_accuracies) + 1),
                         "federated_accuracy": fr.federated_accuracies})


def _at_eps(rc, eps: float) -> Optional[float]:
    i = int(np.argmin(np.abs(rc.epsilons - eps)))
    return float(rc.accuracies[i])


def _dp_at(dp, eps: float) -> Optional[float]:
    i = int(np.argmin(np.abs(np.nan_to_num(dp.epsilons, posinf=1e9) - eps)))
    return float(dp.utilities[i])
