"""Preprocessing utilities: scaling and time-aware / stratified splitting.

Kept deliberately thin -- everything wraps scikit-learn primitives so results
match what a user would get from the standard stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import TrustDataset


class Standardizer:
    """Thin wrapper around ``StandardScaler`` that remembers feature order and
    can invert the transform (needed by robustness attacks that operate in the
    original feature space)."""

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[list] = None

    def fit(self, X: pd.DataFrame) -> "Standardizer":
        self.feature_names_ = list(X.columns)
        self._scaler = StandardScaler().fit(X.to_numpy(dtype=float))
        return self

    def transform(self, X) -> np.ndarray:
        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        return self._scaler.transform(arr)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(np.asarray(Z, float))

    @property
    def mean_(self) -> np.ndarray:
        return self._scaler.mean_

    @property
    def scale_(self) -> np.ndarray:
        return self._scaler.scale_


@dataclass
class DataSplit:
    """Holds the train/test partition of a dataset as TrustDataset objects plus
    the fitted standardizer and the scaled matrices."""

    train: TrustDataset
    test: TrustDataset
    standardizer: Standardizer
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def feature_names(self) -> list:
        return list(self.train.feature_names)


def train_test_split_dataset(
    ds: TrustDataset,
    test_size: float = 0.3,
    seed: int = 42,
    time_col: Optional[str] = None,
    scale: bool = True,
) -> DataSplit:
    """Split a TrustDataset into train/test and optionally standardise.

    Parameters
    ----------
    time_col : str|None   if given (and present in ``ds.meta['time_index']``),
                          use a chronological split instead of a random one.
                          CCI-A1SD captures are time-ordered, so this reproduces
                          the paper's 70/30 time-based split.
    scale : bool          fit a Standardizer on train, apply to both.
    """
    n = ds.n_samples
    idx = np.arange(n)

    time_index = ds.meta.get("time_index")
    if time_col is not None and time_index is not None:
        time_index = np.asarray(time_index)
        order = np.argsort(time_index, kind="stable")
        cut = int(n * (1.0 - test_size))
        train_idx, test_idx = order[:cut], order[cut:]
    else:
        stratify = ds.y if ds.task.is_classification else None
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

    train = _take(ds, train_idx)
    test = _take(ds, test_idx)

    if scale:
        std = Standardizer().fit(train.X)
        X_train = std.transform(train.X)
        X_test = std.transform(test.X)
    else:
        std = Standardizer()
        std.feature_names_ = list(train.X.columns)
        X_train = train.Xv
        X_test = test.Xv

    return DataSplit(
        train=train,
        test=test,
        standardizer=std,
        X_train=X_train,
        X_test=X_test,
        y_train=train.y,
        y_test=test.y,
    )


def _take(ds: TrustDataset, idx: np.ndarray) -> TrustDataset:
    return TrustDataset(
        name=ds.name,
        task=ds.task,
        X=ds.X.iloc[idx].reset_index(drop=True),
        y=ds.y[idx],
        feature_names=list(ds.feature_names),
        class_names=ds.class_names,
        groups={k: np.asarray(v)[idx] for k, v in ds.groups.items()},
        meta=dict(ds.meta, time_index=(
            np.asarray(ds.meta["time_index"])[idx]
            if ds.meta.get("time_index") is not None else None
        )),
    )
