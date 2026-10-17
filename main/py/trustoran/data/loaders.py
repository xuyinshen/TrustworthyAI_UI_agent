"""Concrete loaders for the three Open RAN datasets.

Each loader normalises its raw source into a :class:`TrustDataset`:

* :class:`CCIA1SDLoader`  -- O-RAN A1-interface traffic (Normal / DoS / Covert)
  using the 10 defender-observable features. Classification.
* :class:`DDoSPcapLoader` -- 5-agent O-RAN testbed PCAPs. Flow features are
  decoded on the fly with the built-in pcap reader; labels come from the
  scenario type. Classification, with an ``agent`` group for federated learning.
* :class:`VRANLoader`     -- vRAN computing / energy / application measurements.
  Regression (energy, application) or classification (computing ``explode``).

``build_dataset(key, **kw)`` is the single entry point used by the CLI, the
notebooks and the Qt UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import config as C
from .base import DatasetAdapter, TaskType, TrustDataset
from .pcap import FLOW_FEATURE_NAMES, extract_flow_features


# --------------------------------------------------------------------------
# helper: median-fill + drop all-NaN columns
# --------------------------------------------------------------------------
#: a reading counts as corrupt (collector divide-by-zero, float overflow) only
#: when it is absurd twice over: six orders of magnitude beyond the column's own
#: typical magnitude *and* beyond 1e9 in absolute terms. Both conditions are
#: needed because zero-inflated columns (an all-quiet agent, an all-TCP capture)
#: have a typical magnitude of ~0 and would otherwise lose their genuine spikes.
OUTLIER_RATIO = 1e6
OUTLIER_ABS = 1e9


def _impossible_mask(X: np.ndarray) -> np.ndarray:
    """Boolean mask of readings no RAN telemetry can produce."""
    mask = np.zeros(X.shape, dtype=bool)
    a_all = np.abs(X)
    for j in range(X.shape[1]):
        a = a_all[:, j][np.isfinite(a_all[:, j])]
        if a.size < 3:
            continue
        ref = max(float(np.median(a)), float(np.quantile(a, 0.90)), 1e-12)
        mask[:, j] = (a_all[:, j] > OUTLIER_RATIO * ref) & (a_all[:, j] > OUTLIER_ABS)
    return mask


def _clean_features(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df[list(cols)].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    bad = pd.DataFrame(_impossible_mask(df.to_numpy(dtype=float)),
                       index=df.index, columns=df.columns)
    n_imputed = int(bad.to_numpy().sum())
    df = df.mask(bad)
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(0.0)
    df.attrs["imputed_readings"] = n_imputed
    return df


# ==========================================================================
# 1. CCI-A1SD  (A1 interface security)
# ==========================================================================
class CCIA1SDLoader(DatasetAdapter):
    key = "cci_a1sd"
    label = "CCI-A1SD (A1 interface security)"
    description = (
        "O-RAN A1 interface message traffic under Normal / DoS / Covert "
        "conditions, described by 10 defender-observable timing & payload "
        "features."
    )

    def __init__(
        self,
        classes: Sequence[str] = ("normal", "dos", "covert"),
        dos_variant: str = "dos2",
        per_class_cap: int = 20_000,
        binary: bool = False,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.classes = list(classes)
        self.dos_variant = dos_variant          # "dos1" | "dos2"
        self.per_class_cap = per_class_cap
        self.binary = binary

    def _file_for(self, cls: str) -> Path:
        if cls == "dos":
            return C.CCI_FILES[self.dos_variant]
        return C.CCI_FILES[cls]

    def load(self, **kwargs) -> TrustDataset:
        per_class_cap = kwargs.get("per_class_cap", self.per_class_cap)
        need_cols = (
            C.CCI_FEATURES
            + C.CCI_GROUP_COLS
            + ["dataset_mode", "elapsed_hours", "timestamp"]
        )
        frames: List[pd.DataFrame] = []
        labels: List[int] = []
        class_names: List[str] = []
        for i, cls in enumerate(self.classes):
            df = self._read_csv_sampled(
                self._file_for(cls), usecols=need_cols
            )
            if per_class_cap and len(df) > per_class_cap:
                df = df.sample(n=per_class_cap, random_state=self.seed).sort_index()
            df = df.reset_index(drop=True)
            df["_label"] = i
            frames.append(df)
            labels.append(i)
            class_names.append(cls.capitalize())

        raw = pd.concat(frames, ignore_index=True)

        X = _clean_features(raw, C.CCI_FEATURES)
        y = raw["_label"].to_numpy(dtype=int)

        groups = {
            "policy_type_id": raw["policy_type_id"].to_numpy(),
            "load_label": raw["load_label"].astype(str).to_numpy(),
            "operation": raw["operation"].astype(str).to_numpy(),
            "policy_name": raw["policy_type_id"].map(C.CCI_POLICY_NAMES).astype(str).to_numpy(),
        }

        task = TaskType.BINARY if (self.binary or len(self.classes) == 2) else TaskType.CLASSIFICATION
        meta = {
            "source": "CCI-A1SD",
            "dos_variant": self.dos_variant,
            "classes": list(self.classes),
            "per_class_cap": per_class_cap,
            "imputed_readings": int(X.attrs.get("imputed_readings", 0)),
            "time_index": pd.to_numeric(raw["timestamp"], errors="coerce").fillna(0).to_numpy(),
            "policy_names": C.CCI_POLICY_NAMES,
        }
        return TrustDataset(
            name=self.key, task=task, X=X, y=y,
            feature_names=list(C.CCI_FEATURES), class_names=class_names,
            groups=groups, meta=meta,
        )


# ==========================================================================
# 2. DDoS O-RAN PCAP  (5 agents, flow features)
# ==========================================================================
class DDoSPcapLoader(DatasetAdapter):
    key = "ddos_pcap"
    label = "O-RAN DDoS (Near-RT RIC PCAPs)"
    description = (
        "5-agent O-RAN testbed captures (benign HTTP vs TCP-SYN / UDP flood). "
        "Flow features decoded from raw PCAPs; the ``agent`` group enables a "
        "federated-learning privacy demo."
    )

    def __init__(self, use_manifest_fallback: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.use_manifest_fallback = use_manifest_fallback

    def load(self, **kwargs) -> TrustDataset:
        manifest = pd.read_csv(C.DDOS_MANIFEST)
        rows: List[Dict[str, float]] = []
        agents: List[str] = []
        labels: List[int] = []
        roles: List[str] = []
        files_used: List[str] = []

        for _, r in manifest.iterrows():
            agent = r["agent_folder"]
            fname = r["file"]
            scen = r["scenario_type"]
            path = C.DDOS_ROOT / agent / fname
            ff = extract_flow_features(path) if path.exists() else None
            if ff is not None:
                feats = ff.as_dict()
            elif self.use_manifest_fallback:
                feats = self._manifest_features(r)
            else:
                continue
            rows.append(feats)
            agents.append(agent)
            labels.append(C.DDOS_LABEL_BY_SCENARIO[scen])
            roles.append(str(r.get("top_ip_role_vs_table", "")))
            files_used.append(f"{agent}/{fname}")

        X = pd.DataFrame(rows)
        # keep only the canonical feature order that is present
        feat_cols = [c for c in FLOW_FEATURE_NAMES if c in X.columns]
        X = _clean_features(X, feat_cols)
        y = np.asarray(labels, dtype=int)

        groups = {
            "agent": np.asarray(agents),
            "role": np.asarray(roles),
        }
        meta = {
            "source": "O-RAN DDoS PCAP",
            "files": files_used,
            "n_agents": len(set(agents)),
            "imputed_readings": int(X.attrs.get("imputed_readings", 0)),
            "note": "short-horizon (~40 s) edge benchmark; per-file flow features",
        }
        return TrustDataset(
            name=self.key, task=TaskType.CLASSIFICATION, X=X, y=y,
            feature_names=feat_cols, class_names=list(C.DDOS_SCENARIO_NAMES),
            groups=groups, meta=meta,
        )

    @staticmethod
    def _manifest_features(r) -> Dict[str, float]:
        total = float(r.get("total", 0)) or 1.0
        tcp = float(r.get("tcp", 0))
        udp = float(r.get("udp", 0))
        syn = float(r.get("syn_only", 0))
        http = float(r.get("http_req", 0))
        payload = float(r.get("tcp_payload", 0))
        dur = float(r.get("capture_window_sec", 40)) or 40.0
        return {
            "n_packets": total,
            "n_bytes": payload * 1000.0,
            "duration": dur,
            "pkt_rate": total / dur,
            "avg_pkt_len": (payload * 1000.0) / total if total else 0.0,
            "std_pkt_len": 0.0,
            "min_pkt_len": 0.0,
            "max_pkt_len": 0.0,
            "mean_iat_ms": (dur / total) * 1000.0 if total else 0.0,
            "std_iat_ms": 0.0,
            "max_iat_ms": 0.0,
            "tcp_ratio": tcp / total,
            "udp_ratio": udp / total,
            "syn_ratio": syn / total,
            "synack_ratio": 0.0,
            "ack_ratio": 0.0,
            "rst_ratio": 0.0,
            "http_ratio": http / total,
            "n_src_ips": 1.0,
            "n_dst_ports": 1.0,
            "payload_ratio": payload / total if total else 0.0,
        }


# ==========================================================================
# 3. vRAN  (computing / energy / application)
# ==========================================================================
class VRANLoader(DatasetAdapter):
    key = "vran"
    label = "vRAN (computing / energy / application)"
    description = (
        "Experimental O-RAN vRAN measurements. subset='energy' predicts BBU "
        "power, 'application' predicts edge-AI end-to-end delay, 'computing' "
        "classifies whether a multi-vBS deployment failed (explode)."
    )

    ENERGY_FEATURES = [
        "BW", "TM", "traffic_load", "txgain", "selected_mcs", "selected_airtime",
        "mean_used_mcs", "num_ues", "thr", "gput", "mean_snr", "bler",
        "turbodec_it", "nRBs", "airtime", "clockspeed",
    ]
    APP_FEATURES = ["BW", "img_resolution", "airtime_ratio", "gpu_power"]

    def __init__(self, subset: str = "energy", channel: str = "ul", **kw) -> None:
        super().__init__(**kw)
        self.subset = subset            # "energy" | "computing" | "application"
        self.channel = channel          # "ul" | "dlul" (energy only)

    def load(self, **kwargs) -> TrustDataset:
        subset = kwargs.get("subset", self.subset)
        if subset == "energy":
            return self._load_energy()
        if subset == "application":
            return self._load_application()
        if subset == "computing":
            return self._load_computing()
        raise ValueError(f"unknown vRAN subset '{subset}'")

    # -- energy -------------------------------------------------------------
    def _load_energy(self) -> TrustDataset:
        path = C.VRAN_ENERGY_UL if self.channel == "ul" else C.VRAN_ENERGY_DLUL
        df = self._read_csv_sampled(path)
        df = df[df.get("failed_experiment", 0) == 0] if "failed_experiment" in df else df
        target = "pm_power"
        feats = [c for c in self.ENERGY_FEATURES if c in df.columns]
        df = df.dropna(subset=feats + [target])
        X = _clean_features(df, feats)
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        groups = {}
        if "cpu_platform" in df:
            groups["cpu_platform"] = df["cpu_platform"].astype(str).str.strip().to_numpy()
        meta = {
            "source": "vRAN energy", "channel": self.channel,
            "target": target, "target_units": "W",
            "imputed_readings": int(X.attrs.get("imputed_readings", 0)),
            "file": str(path),
        }
        return TrustDataset(
            name=f"{self.key}_energy", task=TaskType.REGRESSION, X=X, y=y,
            feature_names=feats, groups=groups, meta=meta,
        )

    # -- application --------------------------------------------------------
    def _load_application(self) -> TrustDataset:
        df = self._read_csv_sampled(C.VRAN_APP)
        df = df.drop(columns=[c for c in C.VRAN_APP_DROP if c in df.columns])
        target = "av_end2end_delay"
        feats = [c for c in self.APP_FEATURES if c in df.columns]
        df = df.dropna(subset=feats + [target])
        X = _clean_features(df, feats)
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        groups = {}
        if "gpu_platform" in df:
            groups["gpu_platform"] = df["gpu_platform"].astype(str).str.strip().to_numpy()
        meta = {
            "source": "vRAN application", "target": target,
            "target_units": "s", "file": str(C.VRAN_APP),
            "imputed_readings": int(X.attrs.get("imputed_readings", 0)),
        }
        return TrustDataset(
            name=f"{self.key}_application", task=TaskType.REGRESSION, X=X, y=y,
            feature_names=feats, groups=groups, meta=meta,
        )

    # -- computing ----------------------------------------------------------
    def _load_computing(self) -> TrustDataset:
        frames: List[pd.DataFrame] = []
        for pinning, folder in (("unpin", C.VRAN_COMPUTE_UNPIN), ("pin", C.VRAN_COMPUTE_PIN)):
            for f in sorted(Path(folder).glob("*.csv")):
                df = self._read_csv_sampled(f)
                agg = self._aggregate_computing(df)
                agg["pinning"] = 1 if pinning == "pin" else 0
                # dataset_2_1_2 -> 2 vBS; the first integer is the vBS count
                parts = f.stem.split("_")
                agg["n_vbs"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                frames.append(agg)
        raw = pd.concat(frames, ignore_index=True)
        feats = ["total_dl_kbps", "total_ul_kbps", "mean_mcs_dl", "mean_mcs_ul",
                 "max_dl_kbps", "n_vbs", "pinning"]
        X = _clean_features(raw, feats)
        explode = raw["explode"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
        y = explode.to_numpy(dtype=int)
        groups = {
            "pinning": raw["pinning"].to_numpy(),
            "n_vbs": raw["n_vbs"].to_numpy(),
        }
        meta = {
            "source": "vRAN computing", "target": "explode",
            "imputed_readings": int(X.attrs.get("imputed_readings", 0)),
            "note": "explode=1 -> experiment failed (resource exhaustion)",
        }
        return TrustDataset(
            name=f"{self.key}_computing", task=TaskType.BINARY, X=X, y=y,
            feature_names=feats, class_names=["ok", "explode"],
            groups=groups, meta=meta,
        )

    @staticmethod
    def _aggregate_computing(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse per-vBS config columns into deployment-level aggregates so
        files with different vBS counts share one schema."""
        dl = [c for c in df.columns if c.startswith("dl_kbps_")]
        ul = [c for c in df.columns if c.startswith("ul_kbps_")]
        mdl = [c for c in df.columns if c.startswith("mcs_dl_")]
        mul = [c for c in df.columns if c.startswith("mcs_ul_")]
        out = pd.DataFrame(index=df.index)
        out["total_dl_kbps"] = df[dl].sum(axis=1) if dl else 0.0
        out["total_ul_kbps"] = df[ul].sum(axis=1) if ul else 0.0
        out["max_dl_kbps"] = df[dl].max(axis=1) if dl else 0.0
        out["mean_mcs_dl"] = df[mdl].mean(axis=1) if mdl else 0.0
        out["mean_mcs_ul"] = df[mul].mean(axis=1) if mul else 0.0
        out["explode"] = df["explode"] if "explode" in df else False
        return out


# ==========================================================================
# registry / entry point
# ==========================================================================
_REGISTRY: Dict[str, DatasetAdapter] = {
    CCIA1SDLoader.key: CCIA1SDLoader,
    DDoSPcapLoader.key: DDoSPcapLoader,
    VRANLoader.key: VRANLoader,
}

#: friendly names for the UI, including vRAN subsets
DATASET_CHOICES: Dict[str, Dict[str, object]] = {
    "cci_a1sd": {"label": CCIA1SDLoader.label, "kwargs": {}},
    "cci_a1sd_binary": {
        "label": "CCI-A1SD (Normal vs DoS, binary)",
        "kwargs": {"classes": ("normal", "dos"), "binary": True},
    },
    "ddos_pcap": {"label": DDoSPcapLoader.label, "kwargs": {}},
    "vran_energy": {"label": "vRAN energy (BBU power)", "kwargs": {"subset": "energy"}},
    "vran_application": {"label": "vRAN application (edge-AI delay)", "kwargs": {"subset": "application"}},
    "vran_computing": {"label": "vRAN computing (deployment failure)", "kwargs": {"subset": "computing"}},
}


def available_datasets() -> List[str]:
    return list(DATASET_CHOICES.keys())


def _resolve_key(key: str):
    """Map a dataset choice key to (adapter_key, kwargs)."""
    if key in DATASET_CHOICES:
        base = key.split("_binary")[0]
        base = base if base in _REGISTRY else key.split("_")[0]
        if key.startswith("vran_"):
            base = "vran"
        elif key.startswith("cci"):
            base = "cci_a1sd"
        else:
            base = "ddos_pcap"
        return base, dict(DATASET_CHOICES[key]["kwargs"])
    if key in _REGISTRY:
        return key, {}
    raise KeyError(f"unknown dataset '{key}'; available: {list(DATASET_CHOICES)}")


def build_dataset(key: str, max_rows: Optional[int] = None, seed: int = 42,
                  **overrides) -> TrustDataset:
    """Instantiate the adapter for ``key`` and load its TrustDataset."""
    adapter_key, kwargs = _resolve_key(key)
    kwargs.update(overrides)
    adapter = _REGISTRY[adapter_key](max_rows=max_rows, seed=seed, **kwargs)
    return adapter.load()
