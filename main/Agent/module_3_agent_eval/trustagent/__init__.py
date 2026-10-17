"""TrustAgent Eval — a trace-first, agent-agnostic trust evaluation framework.

Serves the O-RAN SMO / Near-RT RIC setting: the four classic trustworthy-AI
pillars (explainability, robustness, fairness, privacy) plus reliability and
safety, all computed from structured *trajectories* rather than final answers.

Personal research project; not affiliated with any employer. See README.md.
"""
from __future__ import annotations

import os as _os
import sys as _sys

__version__ = "0.1.0"

# `benchmarks/` is a sibling of this package (per the module layout), not a
# sub-package, so make the module root importable.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
