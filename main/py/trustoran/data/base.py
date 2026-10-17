"""Unified dataset schema shared by every loader and every trust pillar.

A :class:`TrustDataset` bundles feature matrix ``X``, target ``y``, optional
group/protected attributes and task metadata into one object, so downstream
code (models, explainability, fairness, Qt UI) never has to know which of the
three Open RAN datasets it is dealing with.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    BINARY = "binary"          # 2-class classification (subset of classification)

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.CLASSIFICATION, TaskType.BINARY)


@dataclass
class TrustDataset:
    """Container for a prepared ML task.

    Attributes
    ----------
    name : str               dataset identifier (e.g. "cci_a1sd")
    task : TaskType          classification / regression
    X : pd.DataFrame         feature matrix (numeric, NaN-free)
    y : np.ndarray           target vector
    feature_names : list     column order of X
    class_names : list|None  label names for classification
    groups : dict            protected/group attribute name -> per-sample array
    meta : dict              provenance + free-form info (shapes, source files)
    """

    name: str
    task: TaskType
    X: pd.DataFrame
    y: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    class_names: Optional[List[str]] = None
    groups: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(self.X.columns)
        self.X = self.X[self.feature_names]
        self.y = np.asarray(self.y).ravel()
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X/y length mismatch: {len(self.X)} vs {len(self.y)}"
            )

    # -- convenience ---------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        if not self.task.is_classification:
            return 0
        return int(len(np.unique(self.y)))

    @property
    def Xv(self) -> np.ndarray:
        """Feature matrix as a plain float numpy array."""
        return self.X.to_numpy(dtype=float)

    def class_counts(self) -> Dict[str, int]:
        if not self.task.is_classification:
            return {}
        vals, counts = np.unique(self.y, return_counts=True)
        out: Dict[str, int] = {}
        for v, c in zip(vals, counts):
            key = (
                self.class_names[int(v)]
                if self.class_names and int(v) < len(self.class_names)
                else str(int(v))
            )
            out[key] = int(c)
        return out

    def group_values(self, attr: str) -> np.ndarray:
        if attr not in self.groups:
            raise KeyError(
                f"group attribute '{attr}' not present; have {list(self.groups)}"
            )
        return np.asarray(self.groups[attr])

    def subset(self, mask: np.ndarray) -> "TrustDataset":
        """Return a new TrustDataset restricted to rows where mask is True."""
        mask = np.asarray(mask, dtype=bool)
        return TrustDataset(
            name=self.name,
            task=self.task,
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y[mask],
            feature_names=list(self.feature_names),
            class_names=self.class_names,
            groups={k: np.asarray(v)[mask] for k, v in self.groups.items()},
            meta=dict(self.meta, subset_of=self.name, parent_rows=self.n_samples),
        )

    def describe(self) -> Dict[str, object]:
        """Compact, JSON-serialisable summary used by the UI overview panel."""
        return {
            "name": self.name,
            "task": self.task.value,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "class_names": self.class_names,
            "class_counts": self.class_counts(),
            "group_attrs": list(self.groups.keys()),
            "meta": {k: v for k, v in self.meta.items() if _jsonable(v)},
        }


def _jsonable(v: object) -> bool:
    return isinstance(v, (str, int, float, bool, list, dict, type(None)))


class DatasetAdapter(ABC):
    """Abstract loader. Concrete adapters turn a raw Open RAN dataset into a
    :class:`TrustDataset`.

    Parameters
    ----------
    max_rows : int|None   cap rows read per raw file (large CSVs are 100s of MB)
    seed : int            sampling seed for reproducible subsampling
    """

    #: short identifier used by the pipeline registry / UI
    key: str = "base"
    #: human readable label
    label: str = "Base dataset"
    #: one-line description for the UI
    description: str = ""

    def __init__(self, max_rows: Optional[int] = None, seed: int = 42) -> None:
        self.max_rows = max_rows
        self.seed = seed

    @abstractmethod
    def load(self, **kwargs) -> TrustDataset:
        """Load and return the prepared TrustDataset."""

    # -- helpers -------------------------------------------------------------
    def _read_csv_sampled(
        self,
        path,
        usecols: Optional[List[str]] = None,
        fraction: float = 1.0,
    ) -> pd.DataFrame:
        """Read a (possibly huge) CSV, optionally subsampling rows.

        Uses ``nrows`` when ``max_rows`` is set and the file is bigger, and a
        deterministic skip-rows sample when ``fraction`` < 1.
        """
        path = str(path)
        nrows = None
        if self.max_rows is not None:
            nrows = int(self.max_rows * max(fraction, 1e-9)) or 1
        df = pd.read_csv(path, usecols=usecols, nrows=nrows)
        if fraction < 1.0 and nrows is None:
            keep = max(1, int(len(df) * fraction))
            df = df.sample(n=keep, random_state=self.seed).sort_index()
        return df
