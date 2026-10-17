"""Background QThread workers.

The UI never blocks: heavy lifting (a full :class:`TrustPipeline` run, or the
lazy training of the compact model the O-RAN console uses for live
explanations) happens here and is reported back through Qt signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    """Runs :class:`trustoran.pipeline.TrustPipeline` off the GUI thread."""

    progress = Signal(str, float, str)      # stage, pct(0..1), message
    succeeded = Signal(object)               # AnalysisReport
    failed = Signal(str)                     # error text

    def __init__(self, dataset_key: str, model_name: str,
                 pillars: Sequence[str], quick: bool, max_rows: Optional[int],
                 seed: int, model_params: Optional[Dict[str, Any]] = None,
                 parent=None):
        super().__init__(parent)
        self.dataset_key = dataset_key
        self.model_name = model_name
        self.pillars = tuple(pillars)
        self.quick = quick
        self.max_rows = max_rows
        self.seed = seed
        self.model_params = model_params or {}

    def run(self) -> None:                    # noqa: D401 - Qt entry point
        try:
            from trustoran.pipeline.engine import TrustPipeline
            pipe = TrustPipeline(
                dataset_key=self.dataset_key, model_name=self.model_name,
                pillars=self.pillars, quick=self.quick,
                max_rows=self.max_rows, seed=self.seed,
                model_params=self.model_params,
            )
            report = pipe.run(progress_cb=self._on_progress)
            self.succeeded.emit(report)
        except Exception as e:                # pragma: no cover
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _on_progress(self, stage: str, pct: float, msg: str) -> None:
        self.progress.emit(stage, float(pct), msg)


@dataclass
class ConsoleModelBundle:
    """A compact CCI classifier + reference data for the O-RAN console.

    ``prototypes`` holds per-class (mean, std) feature vectors from the training
    set so the console can synthesise A1 events that live in the *real* feature
    distribution — which keeps the on-device detector and its explanations
    honest instead of relying on hand-guessed attack magnitudes.
    """
    model: Any
    background: np.ndarray                   # (m, d) reference rows for explanation
    feature_names: List[str]
    class_names: List[str]
    task: Any
    target_class: int = 1
    metrics: Dict[str, float] = field(default_factory=dict)
    prototypes: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    robust_curve: Any = None                   # trustoran RobustnessCurve (FGSM) — live robustness card
    mi_auc: Optional[float] = None             # membership-inference AUC — live privacy card


class ConsoleModelWorker(QThread):
    """Lazily trains the small A1-traffic model the console uses to explain a
    live decision. Independent of (and much lighter than) a full pipeline run."""

    succeeded = Signal(object)               # ConsoleModelBundle
    failed = Signal(str)

    def __init__(self, seed: int = 42, max_rows: int = 8000, parent=None):
        super().__init__(parent)
        self.seed = seed
        self.max_rows = max_rows

    def run(self) -> None:
        try:
            from trustoran.data import build_dataset, train_test_split_dataset
            from trustoran.models import make_model

            ds = build_dataset("cci_a1sd", max_rows=self.max_rows, seed=self.seed)
            split = train_test_split_dataset(ds, test_size=0.3, seed=self.seed)
            model = make_model("random_forest", split.train.task,
                               split.feature_names, split.train.class_names,
                               # n_jobs=1: the console predicts one row at a time,
                               # where joblib's parallel dispatch overhead dominates.
                               params={"n_estimators": 120, "n_jobs": 1})
            model.fit(split.X_train, split.y_train)
            metrics = model.score(split.X_test, split.y_test)

            rng = np.random.default_rng(self.seed)
            bg_idx = rng.choice(len(split.X_train),
                                min(150, len(split.X_train)), replace=False)

            prototypes: Dict[int, Dict[str, np.ndarray]] = {}
            Xtr = split.X_train
            ytr = split.y_train
            for cls in np.unique(ytr):
                rows = Xtr[ytr == cls]
                if len(rows) == 0:
                    continue
                mean = rows.mean(axis=0)
                std = rows.std(axis=0) + 1e-9
                prototypes[int(cls)] = {"mean": mean, "std": std}

            # Precompute the live trust-dock readouts off the GUI thread: a small
            # FGSM robustness sweep and a membership-inference probe. Secondary to
            # the detector, so a failure here must not block the model becoming ready.
            robust_curve = None
            mi_auc: Optional[float] = None
            try:
                from trustoran.trust import robustness as RB, privacy as PR

                def _sub(X, y, n):
                    idx = rng.choice(len(X), min(n, len(X)), replace=False)
                    return X[idx], y[idx]

                Xr, yr = _sub(split.X_test, split.y_test, 250)
                robust_curve = RB.robustness_curve(
                    model, Xr, yr, "fgsm",
                    eps_range=(0.0, 0.1, 0.3, 0.6, 1.0), seed=self.seed)
                Xmt, ymt = _sub(split.X_train, split.y_train, 600)
                Xme, yme = _sub(split.X_test, split.y_test, 600)
                mi_auc = float(PR.membership_inference(
                    model, Xmt, ymt, Xme, yme, seed=self.seed).attack_auc)
            except Exception:
                robust_curve, mi_auc = None, None

            bundle = ConsoleModelBundle(
                model=model,
                background=split.X_train[bg_idx],
                feature_names=list(split.feature_names),
                class_names=list(ds.class_names or [str(i) for i in range(ds.n_classes)]),
                task=split.train.task,
                target_class=1,
                metrics=metrics,
                prototypes=prototypes,
                robust_curve=robust_curve,
                mi_auc=mi_auc,
            )
            self.succeeded.emit(bundle)
        except Exception as e:                # pragma: no cover
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
