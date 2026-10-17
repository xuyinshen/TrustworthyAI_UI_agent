"""Content panels shown in the main window's stacked widget.

* :class:`OverviewPanel` — dataset summary, base metrics and overview figures.
* :class:`PillarPanel`   — generic summary + figures + tables for one pillar.
* :class:`ExplainabilityPanel` — a PillarPanel with an extra "live A1 event"
  area that the O-RAN console drives (this is the console ↔ trust-AI link).

Panels are pure view objects: they are handed an :class:`AnalysisReport` and
render whatever it contains. All captions route through :mod:`i18n`.

Each panel puts its content in a body widget wrapped in a :class:`PageScroll`, so
a short window gives the operator a wheel instead of a squashed figure grid, and
the figures/tables split stays draggable inside a :class:`QSplitter` whose panes
carry minimum heights.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from . import theme
from . import i18n
from .narrative import ui_narrator
from .widgets import (
    ConclusionPanel, FigureGrid, FigureView, PageScroll, SummaryStrip,
    TableStack, apply_splitter_floor, section_header, set_section_text,
)

# English literals double as i18n keys (see i18n._ZH).
PILLAR_META = {
    "explainability": (
        "Explainability",
        "Why did the model decide this? Permutation importance, KernelSHAP-style "
        "attributions, a LIME-style local surrogate and PDP/ICE curves.",
    ),
    "robustness": (
        "Robustness",
        "How does the model hold up under adversarial perturbation? FGSM / PGD / "
        "DeepFool attacks, robustness curves and adversarial training.",
    ),
    "fairness": (
        "Fairness",
        "Is the decision applied evenly across groups (A1 policy types / agents / "
        "platforms)? Demographic parity, disparate impact, equal-opportunity gap.",
    ),
    "privacy": (
        "Privacy",
        "Can we learn without leaking? DP-SGD privacy–utility curves, FedAvg across "
        "edge agents and a membership-inference probe.",
    ),
}


# --------------------------------------------------------------------------
class OverviewPanel(QWidget):
    """Headline view of a run. ``pdf_requested`` asks the window to open the
    export dialog — the panel itself knows nothing about PDF generation."""

    pdf_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._report = None
        body = QWidget()
        lay = self._body_lay = QVBoxLayout(body)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)
        head_row = QHBoxLayout()
        head_row.setSpacing(10)
        self._head = section_header("", "")
        head_row.addWidget(self._head, 1)
        self.pdf_btn = QPushButton("")
        self.pdf_btn.setObjectName("tune")
        self.pdf_btn.setCursor(Qt.PointingHandCursor)
        self.pdf_btn.clicked.connect(self.pdf_requested.emit)
        head_row.addWidget(self.pdf_btn, 0, Qt.AlignTop)
        lay.addLayout(head_row)

        self.strategy_card = QFrame()
        self.strategy_card.setObjectName("card")
        sc = QVBoxLayout(self.strategy_card)
        sc.setContentsMargins(14, 10, 14, 12)
        sc.setSpacing(4)
        self._strategy_title = QLabel("")
        self._strategy_title.setObjectName("h2")
        self.strategy_lbl = QLabel("")
        self.strategy_lbl.setObjectName("muted")
        self.strategy_lbl.setWordWrap(True)
        self.strategy_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sc.addWidget(self._strategy_title)
        sc.addWidget(self.strategy_lbl)
        self.strategy_card.setMinimumHeight(96)

        self.summary = SummaryStrip()
        lay.addWidget(self.summary)

        # strategy text and figures are two resizeable bands; the metric strip
        # above them wraps on its own, so it never needs a handle.
        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(6)
        self.figs = FigureGrid(columns=2)
        self.split.addWidget(self.strategy_card)
        self.split.addWidget(self.figs)
        self.split.setCollapsible(0, False)
        self.split.setCollapsible(1, False)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        apply_splitter_floor(self.split)
        lay.addWidget(self.split, 1)

        self.scroll = PageScroll(body)
        self.conclusion = ConclusionPanel()
        self.scroll.add_footer(self.conclusion)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll)
        self.retranslate()

    def retranslate(self) -> None:
        set_section_text(
            self._head,
            i18n.tr("Overview"),
            i18n.tr("Run an analysis to see the dataset, base model and headline metrics."),
        )
        self._strategy_title.setText(i18n.tr("Analysis strategy for this dataset"))
        self.pdf_btn.setText(i18n.tr("Export PDF report…"))
        self.pdf_btn.setToolTip(i18n.tr(
            "Choose a reader style and format, then write the PDF report."))
        self.summary.retranslate()
        self.figs.retranslate()
        self.conclusion.retranslate()
        self._refresh_conclusion()

    def _refresh_conclusion(self) -> None:
        """Rebuild the card: the language may have changed under the same data."""
        self.conclusion.set_bullets(
            ui_narrator(report=self._report).conclusions("overview"))

    def set_strategy(self, text: str) -> None:
        self.strategy_lbl.setText(text)

    def set_report(self, report) -> None:
        self._report = report
        ds = report.dataset_summary or {}
        metrics = dict(report.base_metrics or {})
        spec = report.model_spec or {}
        cards: Dict[str, object] = {
            "dataset": report.dataset_key,
            "model": spec.get("label", spec.get("name", "?")),
            "task": spec.get("task", ds.get("task", "")),
            "samples": ds.get("n_samples", "—"),
            "features": ds.get("n_features", "—"),
        }
        cards.update({k: v for k, v in metrics.items()})
        if "elapsed_sec" in spec:
            cards["elapsed_sec"] = spec["elapsed_sec"]
        self.summary.set_summary(cards, good_keys=tuple(metrics.keys()))
        self.figs.set_figures(report.base_figures or {})
        self._refresh_conclusion()

    def clear(self) -> None:
        self._report = None
        self.summary.clear()
        self.figs.set_figures({})
        self._refresh_conclusion()


# --------------------------------------------------------------------------
class PillarPanel(QWidget):
    """Generic pillar view: summary strip + figure grid + table stack."""

    def __init__(self, pillar: str, parent=None):
        super().__init__(parent)
        self.pillar = pillar
        self._report = None
        body = QWidget()
        lay = self._body_lay = QVBoxLayout(body)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        self._head = section_header("", "")
        lay.addWidget(self._head)

        self.summary = SummaryStrip()
        lay.addWidget(self.summary)

        # figures and tables are two resizeable bands, each with a height floor,
        # so dragging the handle reallocates room but can never crush one.
        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(6)
        self.figs = FigureGrid(columns=2)
        self.tables = TableStack()
        self.tables.setMinimumHeight(240)
        self.split.addWidget(self.figs)
        self.split.addWidget(self._wrap_tables())
        self.split.setCollapsible(0, False)
        self.split.setCollapsible(1, False)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 1)
        apply_splitter_floor(self.split)
        lay.addWidget(self.split, 1)

        self._error = QLabel("")
        self._error.setObjectName("muted")
        self._error.setWordWrap(True)
        self._error.setStyleSheet(f"color:{theme.BAD};")
        self._error.hide()
        lay.addWidget(self._error)

        self.scroll = PageScroll(body)
        self.conclusion = ConclusionPanel()
        self.scroll.add_footer(self.conclusion)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll)
        self._err_kind: Optional[str] = None
        self._err_msg: str = ""
        self.retranslate()

    def _wrap_tables(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.addWidget(self.tables)
        return w

    def retranslate(self) -> None:
        title, subtitle = PILLAR_META.get(self.pillar, (self.pillar.title(), ""))
        set_section_text(self._head, i18n.tr(title), i18n.tr(subtitle))
        self.summary.retranslate()
        self.figs.retranslate()
        self.tables.retranslate()
        self._render_error()
        self.conclusion.retranslate()
        self._refresh_conclusion()

    def _refresh_conclusion(self) -> None:
        """Rebuild the card: the language may have changed under the same data."""
        self.conclusion.set_bullets(
            ui_narrator(report=self._report).conclusions(self.pillar))

    def _render_error(self) -> None:
        if self._err_kind is None:
            self._error.hide()
            return
        prefix = i18n.tr("pillar error — ") if self._err_kind == "error" \
            else i18n.tr("skipped — ")
        self._error.setText(prefix + self._err_msg)
        self._error.show()

    def set_report(self, report) -> None:
        self._report = report
        # refreshed up front: the early return below must also leave the card
        # empty rather than showing the previous run's conclusions.
        self._refresh_conclusion()
        res = (report.pillars or {}).get(self.pillar)
        self._err_kind = None
        self._err_msg = ""
        if res is None:
            self.summary.clear()
            self.figs.set_figures({})
            self.tables.set_tables({})
            self._error.hide()
            return
        if res.summary.get("error"):
            self._err_kind = "error"
            self._err_msg = str(res.summary["error"])
        elif res.summary.get("skipped"):
            self._err_kind = "skipped"
            self._err_msg = str(res.summary["skipped"])
        self._render_error()
        self.summary.set_summary(res.summary or {})
        self.figs.set_figures(res.figures or {})
        self.tables.set_tables(res.tables or {})

    def clear(self) -> None:
        self._report = None
        self.summary.clear()
        self.figs.set_figures({})
        self.tables.set_tables({})
        self._err_kind = None
        self._error.hide()
        self._refresh_conclusion()


# --------------------------------------------------------------------------
class ExplainabilityPanel(PillarPanel):
    """Adds a *live A1 event* explanation card wired to the O-RAN console."""

    explain_requested_from_console = Signal()   # emitted when user clicks Explain

    def __init__(self, parent=None):
        super().__init__("explainability", parent)
        self.live_card = self._build_live_card()
        # index 2 = after the summary strip, before the figures/tables splitter
        self._body_lay.insertWidget(2, self.live_card)

    def _build_live_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(300)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        head = QHBoxLayout()
        self.live_title = QLabel("")
        self.live_title.setObjectName("h2")
        self.live_hint = QLabel("")
        self.live_hint.setObjectName("muted")
        head.addWidget(self.live_title)
        head.addWidget(self.live_hint, 1)
        lay.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(12)
        self.live_fig = FigureView()
        self.live_fig.setMinimumHeight(240)
        self.live_fig.setMinimumWidth(260)
        self.live_fig.clear()
        body.addWidget(self.live_fig, 3)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.live_verdict = QLabel("—")
        self.live_verdict.setObjectName("metricValue")
        self.live_meta = QLabel("")
        self.live_meta.setObjectName("muted")
        self.live_meta.setWordWrap(True)
        self.live_table = TableStack()
        right.addWidget(self.live_verdict)
        right.addWidget(self.live_meta)
        right.addWidget(self.live_table, 1)
        body.addLayout(right, 2)
        lay.addLayout(body, 1)
        self._retranslate_live()
        return card

    def _retranslate_live(self) -> None:
        if not hasattr(self, "live_title"):
            return                              # called during super().__init__
        self.live_title.setText(i18n.tr("Live A1 event explanation"))
        if not self._live_busy:
            self.live_hint.setText(
                i18n.tr("Inject traffic in the O-RAN console, then explain an alert here."))
        self.live_table.retranslate()

    _live_busy = False

    def retranslate(self) -> None:
        super().retranslate()
        self._retranslate_live()

    def show_live_explanation(self, features: Dict[str, float], verdict: str,
                              accent: str, meta: str,
                              local_fig, table: pd.DataFrame) -> None:
        self.live_verdict.setText(verdict)
        self.live_verdict.setStyleSheet(f"color:{accent};")
        self.live_meta.setText(meta)
        if local_fig is not None:
            self.live_fig.set_figure(local_fig)
        if table is not None:
            self.live_table.set_tables({"event_attribution": table})
        self.explain_requested_from_console.emit()

    def busy(self, on: bool) -> None:
        self._live_busy = bool(on)
        self.live_hint.setText(
            i18n.tr("computing explanation…") if on else
            i18n.tr("Inject traffic in the O-RAN console, then explain an alert here."))
