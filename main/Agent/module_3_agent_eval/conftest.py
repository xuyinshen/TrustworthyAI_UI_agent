"""Make the module root importable for pytest (``trustagent`` and ``benchmarks``
are siblings, not packages under one parent)."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
