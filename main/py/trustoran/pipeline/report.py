"""Result containers for a full trustworthy-AI run, plus export to results/."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..data.base import TrustDataset


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return str(obj)


@dataclass
class PillarResult:
    pillar: str
    summary: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: Dict[str, Figure] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)   # raw result objects

    def figure_keys(self) -> List[str]:
        return list(self.figures.keys())


@dataclass
class AnalysisReport:
    dataset_key: str
    dataset_summary: Dict[str, Any]
    model_spec: Dict[str, Any]
    base_metrics: Dict[str, float]
    pillars: Dict[str, PillarResult] = field(default_factory=dict)
    base_figures: Dict[str, Figure] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # -- export -------------------------------------------------------------
    def export(self, outdir) -> Dict[str, List[str]]:
        """Write JSON report + CSV tables + PNG figures under ``outdir``.

        Returns a mapping of category -> list of written file paths.
        """
        out = Path(outdir)
        base = out / self.dataset_key
        (base / "tables").mkdir(parents=True, exist_ok=True)
        (base / "figures").mkdir(parents=True, exist_ok=True)
        written: Dict[str, List[str]] = {"report": [], "tables": [], "figures": []}

        # report.json
        payload = {
            "dataset_key": self.dataset_key,
            "created": self.created,
            "dataset": self.dataset_summary,
            "model": self.model_spec,
            "base_metrics": _to_jsonable(self.base_metrics),
            "pillars": {
                name: {"summary": _to_jsonable(p.summary),
                       "tables": list(p.tables.keys()),
                       "figures": list(p.figures.keys())}
                for name, p in self.pillars.items()
            },
        }
        rp = base / "report.json"
        rp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written["report"].append(str(rp))

        # base figures
        for name, fig in self.base_figures.items():
            p = base / "figures" / f"00_{name}.png"
            fig.savefig(p, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=130)
            written["figures"].append(str(p))

        # per-pillar tables + figures
        for pillar, res in self.pillars.items():
            for tname, df in res.tables.items():
                p = base / "tables" / f"{pillar}__{tname}.csv"
                df.to_csv(p, index=False, encoding="utf-8")
                written["tables"].append(str(p))
            for fname, fig in res.figures.items():
                p = base / "figures" / f"{pillar}__{fname}.png"
                fig.savefig(p, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=130)
                written["figures"].append(str(p))
        return written

    def headline(self) -> str:
        """One-line human summary used by the CLI and the Qt overview."""
        m = self.base_metrics
        key = "accuracy" if "accuracy" in m else ("r2" if "r2" in m else next(iter(m), ""))
        return f"[{self.dataset_key}] {self.model_spec.get('name','?')} — {key}={m.get(key, float('nan')):.3f}"
