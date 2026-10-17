"""Central configuration: filesystem paths, feature schemas, label maps.

Every path is derived from the repository root so the package works no matter
which working directory it is imported from (CLI, notebook, or Qt app).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- repository layout -------------------------------------------------------
# .../可信AI/py/trustoran/config.py  ->  repo root is three levels up.
# A PyInstaller bundle has no such layout (modules live in the _MEIPASS tree),
# so a frozen build locates the root from the executable instead: the nearest
# folder that actually carries dataset/ or results/, overridable via TRUSTORAN_HOME.


def _repo_root() -> Path:
    env = os.environ.get("TRUSTORAN_HOME", "").strip()
    if env:
        return Path(env).resolve()
    if getattr(sys, "frozen", False):
        bases = [Path(sys.executable).resolve().parent]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bases.append(Path(meipass).resolve().parent)
        for base in bases:
            for candidate in (base, *base.parents):
                if (candidate / "dataset").is_dir() or (candidate / "results").is_dir():
                    return candidate
        return bases[0]
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
PACKAGE_DIR = Path(__file__).resolve().parent          # py/trustoran (or bundle dir)
PY_DIR = REPO_ROOT / "py"
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "results"
IPYNB_DIR = REPO_ROOT / "ipynb"
QTUI_DIR = REPO_ROOT / "QtUI"

# per-dataset roots
CCI_ROOT = DATASET_DIR / "CCI-A1SD" / "a1shield-dataset"
DDOS_ROOT = DATASET_DIR / "DDoS attacks in O-RAN Networks" / "5_Agents"
VRAN_ROOT = DATASET_DIR / "datasets_vran"

# result sub-folders (created lazily)
FIG_DIR = RESULTS_DIR / "figures"
CSV_DIR = RESULTS_DIR / "tables"
JSON_DIR = RESULTS_DIR / "reports"
MODEL_DIR = RESULTS_DIR / "models"


def ensure_results_dirs() -> None:
    """Create the results/ tree if it does not exist yet."""
    for d in (RESULTS_DIR, FIG_DIR, CSV_DIR, JSON_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- CCI-A1SD ---------------------------------------------------------------
# The paper's ML models use ONLY these 10 defender-observable features.
CCI_FEATURES = [
    "real_interval",
    "inter_arrival_mean",
    "inter_arrival_std",
    "inter_arrival_cv",
    "response_time_mean",
    "response_time_std",
    "request_rate",
    "payload_size_mean",
    "payload_size_std",
    "body_size",
]
# columns kept for grouping / fairness analysis (not model inputs)
CCI_GROUP_COLS = ["policy_type_id", "load_label", "operation"]
CCI_CLASS_NAMES = ["Normal", "DoS", "Covert"]          # 0 / 1 / 2
CCI_LABEL_BY_MODE = {"Normal": 0, "DoS": 1, "Covert": 2}
CCI_FILES = {
    "normal": CCI_ROOT / "datasets" / "normal_baseline.csv",
    "dos1": CCI_ROOT / "datasets" / "dos_volumetric.csv",
    "dos2": CCI_ROOT / "datasets" / "dos2_volumetric.csv",
    "covert": CCI_ROOT / "datasets" / "covert_channel.csv",
}
# interleaved captures (confound-free detection benchmark)
CCI_INTERLEAVED = {
    "normal_covert": CCI_ROOT / "interleaved" / "interleaved_capture.csv",
    "normal_dos": CCI_ROOT / "interleaved" / "interleaved_dos_capture.csv",
    "negative_control": CCI_ROOT / "interleaved" / "negative_control_capture.csv",
}

# policy type ids -> human readable O-RAN A1 policy names
CCI_POLICY_NAMES = {
    20008: "Traffic Steering",
    20009: "QoS Management",
    20010: "Resource Allocation",
    20011: "Slice SLA",
    20012: "Energy Saving",
    20013: "Load Balancing",
    20014: "Cell Barring",
    20015: "Handover Steering",
}

# --- DDoS O-RAN (PCAP) ------------------------------------------------------
DDOS_MANIFEST = DDOS_ROOT / "pcap_manifest.csv"
DDOS_AGENT_FOLDERS = ["Agent_1", "Agent_2", "Agent_3", "Agent_4", "Agent_5"]
# flow-level features extracted from the manifest
DDOS_FEATURES = ["total", "tcp", "udp", "syn_only", "http_req", "tcp_payload"]
DDOS_SCENARIO_NAMES = ["benign", "tcp_syn_flood", "udp_flood"]     # 0 / 1 / 2
DDOS_LABEL_BY_SCENARIO = {
    "benign_tcp_http": 0,
    "tcp_syn_flood_ddos": 1,
    "udp_flood_ddos": 2,
}
# derived ratio features computed on top of the raw counts
DDOS_DERIVED = [
    "tcp_ratio", "udp_ratio", "syn_ratio", "http_ratio", "payload_ratio",
]

# --- vRAN (computing / energy / application) -------------------------------
VRAN_ENERGY_UL = VRAN_ROOT / "energy_datasets" / "dataset_ul.csv"
VRAN_ENERGY_DLUL = VRAN_ROOT / "energy_datasets" / "dataset_dlul.csv"
VRAN_APP = VRAN_ROOT / "application_datasets" / "application.csv"
VRAN_COMPUTE_UNPIN = VRAN_ROOT / "computing_datasets" / "datasets_unpin"
VRAN_COMPUTE_PIN = VRAN_ROOT / "computing_datasets" / "datasets_pin"

# columns in application.csv that hold huge embedded arrays -> always dropped
VRAN_APP_DROP = [
    "im_ids", "end2end_delay", "imp_proc_delay", "gpu_delay", "confidences",
    "num_obj", "obj_size", "AP_per_image",
]

# --- default sampling caps (raw CSVs are up to 600 MB) ----------------------
DEFAULT_MAX_ROWS = 60_000
RANDOM_SEED = 42


def env_int(name: str, default: int) -> int:
    """Read an integer from the environment with a fallback default."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
