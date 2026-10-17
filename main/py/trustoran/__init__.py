"""Trust-oRAN: a self-contained Trustworthy-AI toolkit for Open RAN.

Four pillars -- Explainability, Robustness, Fairness, Privacy -- implemented
natively on top of numpy / scikit-learn so the library stays lightweight and
dependency-stable (no torch / numba / C++ toolchain required).

The package is organised as:
    trustoran.config        paths + feature/label constants
    trustoran.data          dataset adapters -> unified TrustDataset
    trustoran.models        unified TrustModel wrappers + factory
    trustoran.trust         the four trustworthy-AI pillars
    trustoran.viz           matplotlib Figure factories (shared by ipynb + Qt)
    trustoran.pipeline      orchestration + result export
"""

__version__ = "1.0.0"
__all__ = ["config", "data", "models", "trust", "viz", "pipeline"]
