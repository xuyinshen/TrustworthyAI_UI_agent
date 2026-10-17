"""Simulated O-RAN SMO / Near-RT RIC operations console.

This is the "real industrial" half of the UI the brief asked for. It models the
Non-RT RIC → A1 → Near-RT RIC control loop over five edge agents, streams A1
policy-update traffic, lets the operator *inject* an attack scenario
(volumetric DoS or an FGN-modulated covert channel), runs a live on-device
detector, and raises alerts.

The link into the trustworthy-AI side: every alert carries the exact
10-feature A1 vector that triggered it, and the operator can send it to the
Explainability panel (``explain_requested``) for a per-decision LIME-style
attribution — i.e. "why did the RIC flag this policy update?".

Traffic is synthesised from the *real* per-class feature prototypes shipped in
the console model bundle, so the detector and its explanations operate on the
same distribution the model was trained on. All chrome text routes through
:mod:`i18n`.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QProgressBar, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from . import theme
from . import i18n
from .narrative import ui_narrator
from .widgets import (ConclusionPanel, PageScroll, apply_splitter_floor,
                      enable_horizontal_wheel, section_header, set_section_text)

# scenario -> CCI class index (Normal / DoS / Covert)
SCENARIOS = {"Normal": 0, "DoS volumetric": 1, "Covert channel": 2}
_SCENARIO_CLASS = {"Normal": 0, "DoS volumetric": 1, "Covert channel": 2}
_AGENTS = ["Agent_1", "Agent_2", "Agent_3", "Agent_4", "Agent_5"]
_POLICIES = [
    "Traffic Steering", "QoS Management", "Resource Allocation", "Slice SLA",
    "Energy Saving", "Load Balancing", "Cell Barring", "Handover Steering",
]
_SPEEDS = [("slow", 900), ("normal", 500), ("fast", 220)]
_EVENT_HEADERS = ["t", "agent", "policy", "req/s", "body", "verdict"]
_ALERT_HEADERS = ["severity", "t", "agent", "policy", "class", "p(anomaly)", "req/s", "explain"]

# Default column widths. The identifier columns are kept narrow on purpose so the
# wide last column (verdict / real-time explanation) gets the room it needs; the
# operator can drag any divider to rebalance (last column stretches to fill).
_EVENT_WIDTHS = [78, 84, 150, 72, 72, 120]
_ALERT_WIDTHS = [80, 74, 80, 132, 66, 82, 60, 460]


def _configure_header(table: QTableWidget, widths: List[int]) -> None:
    """User-resizable columns with sensible defaults; last column fills space."""
    h = table.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.Interactive)
    h.setHighlightSections(False)
    for i, w in enumerate(widths):
        if i < table.columnCount():
            table.setColumnWidth(i, w)
    h.setStretchLastSection(True)
    h.setMinimumSectionSize(56)
    table.setTextElideMode(Qt.ElideRight)
    table.setWordWrap(False)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    enable_horizontal_wheel(table)

_MAX_EVENTS = 500
_MAX_ALERTS = 200


class _RateChart(QWidget):
    """Rolling request-rate line, coloured by the live scenario."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(5, 2.2), dpi=100, facecolor=theme.BG)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(theme.PANEL)
        self._ax.tick_params(colors=theme.MUTED, labelsize=7)
        for s in self._ax.spines.values():
            s.set_color(theme.GRID)
        self._ax.grid(True, color=theme.GRID, alpha=0.4, linewidth=0.5)
        (self._line,) = self._ax.plot([], [], color=theme.ACCENT, lw=1.6)
        self._data = deque(maxlen=180)
        self.canvas = FigureCanvasQTAgg(self._fig)
        lay.addWidget(self.canvas)
        self.retranslate()

    def retranslate(self) -> None:
        self._ax.set_title(i18n.tr("A1 request rate (req/s)"),
                           color=theme.INK, fontsize=9)
        self.canvas.draw_idle()

    def push(self, rate: float, color: str) -> None:
        self._data.append(rate)
        ys = np.asarray(self._data, float)
        xs = np.arange(len(ys))
        self._line.set_data(xs, ys)
        self._line.set_color(color)
        self._ax.set_xlim(0, max(self._data.maxlen, len(ys)))
        self._ax.set_ylim(0, max(1.0, ys.max() * 1.15))
        self.canvas.draw_idle()

    def reset(self) -> None:
        self._data.clear()
        self._line.set_data([], [])
        self.canvas.draw_idle()


class TrustLivePanel(QWidget):
    """Live embodiment of the robustness / fairness / privacy pillars.

    Explainability already lives inline in the alert table's last column; this
    right-hand dock surfaces the *other three* pillars in real time, driven by
    the same event stream and on-device detector:

    * **Robustness** — the detector's FGSM accuracy-vs-ε curve (precomputed in
      the worker) plus a live rolling decision-margin / robust rate.
    * **Fairness** — per-agent detection rate as traffic streams, with a live
      demographic-parity gap (max−min), mirroring the fairness pillar.
    * **Privacy** — inference posture (on-device, features-only) plus the
      membership-inference AUC precomputed in the worker.

    Everything is Qt-native (labels + thin bars) so it stays crisp in a narrow
    dock; the full matplotlib figures for these pillars live in their panels.
    """

    _MARGIN_THRESH = 0.20          # decision margin counted as "robust"
    _PARITY_WARN = 0.15            # parity gap above this reads as unfair

    def __init__(self, agents: List[str], parent=None):
        super().__init__(parent)
        self._agents = list(agents)
        self._bundle = None
        self._rc = None                              # RobustnessCurve
        self._mi: Optional[float] = None             # membership-inference AUC
        self._ev = {a: 0 for a in self._agents}
        self._det = {a: 0 for a in self._agents}
        self._margins = deque(maxlen=60)
        self._card_titles: Dict[str, QLabel] = {}
        self.setMinimumWidth(230)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self._title = QLabel("")
        self._title.setObjectName("h2")
        lay.addWidget(self._title)
        lay.addWidget(self._robustness_card())
        lay.addWidget(self._fairness_card())
        lay.addWidget(self._privacy_card())
        lay.addStretch(1)
        self.retranslate()

    # ---------------------------------------------------------------- cards
    def _card(self, title_key: str):
        card = QFrame()
        card.setObjectName("card")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(6)
        t = QLabel(i18n.tr(title_key))
        t.setStyleSheet(f"color:{theme.ACCENT}; font-weight:700; font-size:12px;")
        v.addWidget(t)
        self._card_titles[title_key] = t
        return card, v

    def _robustness_card(self) -> QFrame:
        card, v = self._card("Robustness")
        self.rob_curve_lbl = QLabel("—")
        self.rob_curve_lbl.setWordWrap(True)
        self.rob_curve_lbl.setStyleSheet(f"color:{theme.MUTED}; font-size:11px;")
        v.addWidget(self.rob_curve_lbl)

        self.rob_rate_lbl = QLabel("")
        self.rob_rate_lbl.setStyleSheet(f"color:{theme.INK}; font-size:11px;")
        v.addWidget(self.rob_rate_lbl)
        self.rob_bar = QProgressBar()
        self.rob_bar.setRange(0, 100)
        self.rob_bar.setValue(0)
        self.rob_bar.setFixedHeight(7)
        self.rob_bar.setTextVisible(False)
        v.addWidget(self.rob_bar)

        self.rob_margin_lbl = QLabel("")
        self.rob_margin_lbl.setStyleSheet(f"color:{theme.MUTED}; font-size:11px;")
        v.addWidget(self.rob_margin_lbl)
        return card

    def _fairness_card(self) -> QFrame:
        card, v = self._card("Fairness")
        self.fair_hint = QLabel("")
        self.fair_hint.setStyleSheet(f"color:{theme.MUTED}; font-size:10px;")
        v.addWidget(self.fair_hint)
        self.agent_rows: Dict[str, tuple] = {}
        for a in self._agents:
            row = QHBoxLayout()
            row.setSpacing(6)
            name = QLabel(a)
            name.setFixedWidth(56)
            name.setStyleSheet(f"color:{theme.MUTED}; font-size:10px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(7)
            bar.setTextVisible(False)
            pct = QLabel("—")
            pct.setFixedWidth(34)
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct.setStyleSheet(f"color:{theme.INK}; font-size:10px;")
            row.addWidget(name)
            row.addWidget(bar, 1)
            row.addWidget(pct)
            v.addLayout(row)
            self.agent_rows[a] = (bar, pct)
        self.fair_gap_lbl = QLabel("")
        self.fair_gap_lbl.setStyleSheet(f"color:{theme.INK}; font-size:11px;")
        v.addWidget(self.fair_gap_lbl)
        return card

    def _privacy_card(self) -> QFrame:
        card, v = self._card("Privacy")
        self.priv_posture = QLabel("")
        self.priv_posture.setWordWrap(True)
        self.priv_posture.setStyleSheet(f"color:{theme.MUTED}; font-size:10px;")
        v.addWidget(self.priv_posture)
        self.priv_mi_lbl = QLabel("")
        self.priv_mi_lbl.setWordWrap(True)
        self.priv_mi_lbl.setStyleSheet(f"color:{theme.INK}; font-size:11px;")
        v.addWidget(self.priv_mi_lbl)
        return card

    # ---------------------------------------------------------------- wiring
    def set_bundle(self, bundle) -> None:
        self._bundle = bundle
        self._rc = getattr(bundle, "robust_curve", None)
        mi = getattr(bundle, "mi_auc", None)
        self._mi = float(mi) if mi is not None else None
        self.refresh()

    def record_event(self, agent: str, is_detection: bool, margin: float) -> None:
        """Cheap per-event counter update; call :meth:`refresh` once per tick."""
        if agent in self._ev:
            self._ev[agent] += 1
            if is_detection:
                self._det[agent] += 1
                self._margins.append(float(margin))

    def reset(self) -> None:
        for a in self._agents:
            self._ev[a] = 0
            self._det[a] = 0
        self._margins.clear()
        self.refresh()

    # ---------------------------------------------------------------- render
    def retranslate(self) -> None:
        self._title.setText(i18n.tr("Live trust pillars"))
        for key, lbl in self._card_titles.items():
            lbl.setText(i18n.tr(key))
        self.fair_hint.setText(i18n.tr("Detection rate by agent"))
        self.priv_posture.setText("\n".join([
            i18n.tr("on-device inference — raw A1 payloads never leave the agent"),
            i18n.tr("exposes only the 10-dim aggregate feature vector"),
            i18n.tr("differential privacy: off (ε=∞)"),
        ]))
        self.refresh()

    def refresh(self) -> None:
        # robustness: static FGSM curve + live margin stats
        if self._rc is not None:
            eps = np.asarray(self._rc.epsilons, float)
            acc = np.asarray(self._rc.accuracies, float)
            curve = "  ".join(f"ε{e:g} {a * 100:.0f}%" for e, a in zip(eps, acc))
            self.rob_curve_lbl.setText(i18n.tr("FGSM accuracy: ") + curve)
        else:
            pending = i18n.tr("computing…") if self._bundle is None else "—"
            self.rob_curve_lbl.setText(i18n.tr("FGSM accuracy: ") + pending)

        rate_key = "Robust rate (margin ≥ {t})"
        rate_lbl = i18n.tr(rate_key, t=f"{self._MARGIN_THRESH:.2f}")
        if len(self._margins):
            arr = np.asarray(self._margins, float)
            rate = float((arr >= self._MARGIN_THRESH).mean())
            self.rob_bar.setValue(int(round(rate * 100)))
            self.rob_rate_lbl.setText(f"{rate_lbl}: {rate * 100:.0f}%")
            self.rob_margin_lbl.setText(
                i18n.tr("Last decision margin") + f": {arr[-1]:.2f}")
        else:
            self.rob_bar.setValue(0)
            self.rob_rate_lbl.setText(f"{rate_lbl}: —")
            self.rob_margin_lbl.setText(i18n.tr("Last decision margin") + ": —")

        # fairness: per-agent detection rate + parity gap
        rates = {}
        for a in self._agents:
            ev, dt = self._ev[a], self._det[a]
            r = (dt / ev) if ev else None
            rates[a] = r
            bar, pct = self.agent_rows[a]
            if r is None:
                bar.setValue(0)
                pct.setText("—")
            else:
                bar.setValue(int(round(r * 100)))
                pct.setText(f"{r * 100:.0f}%")
        have = [r for r in rates.values() if r is not None]
        gap_lbl = i18n.tr("Parity gap (max−min)")
        if len(have) >= 2:
            gap = max(have) - min(have)
            color = theme.GOOD if gap <= self._PARITY_WARN else theme.BAD
            self.fair_gap_lbl.setText(f"{gap_lbl}: {gap * 100:.0f}%")
            self.fair_gap_lbl.setStyleSheet(f"color:{color}; font-size:11px;")
        else:
            self.fair_gap_lbl.setText(f"{gap_lbl}: —")
            self.fair_gap_lbl.setStyleSheet(f"color:{theme.INK}; font-size:11px;")

        # privacy: membership-inference leakage
        mi_lbl = i18n.tr("Membership-inference AUC")
        if self._mi is not None:
            color = theme.GOOD if self._mi <= 0.6 else "#fbbf24"
            self.priv_mi_lbl.setText(
                f"{mi_lbl}: {self._mi:.2f}  ({i18n.tr('≈0.5 = no leakage')})")
            self.priv_mi_lbl.setStyleSheet(f"color:{color}; font-size:11px;")
        else:
            pending = i18n.tr("computing…") if self._bundle is None else "—"
            self.priv_mi_lbl.setText(f"{mi_lbl}: {pending}")
            self.priv_mi_lbl.setStyleSheet(f"color:{theme.INK}; font-size:11px;")


class OranConsole(QWidget):
    """The Near-RT RIC / SMO simulator widget."""

    explain_requested = Signal(dict)         # 10-feature dict of a chosen alert
    model_status = Signal(str)               # "loading" / "ready" / "error: …"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rng = np.random.default_rng(7)
        self._bundle = None
        self._alerts: List[Dict] = []
        self._tick = 0
        self._running = False
        self._started_any = False
        self._model_state = "loading"
        self._model_acc: Optional[float] = None
        self._card_titles: Dict[str, QLabel] = {}

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(18, 16, 18, 8)
        lay.setSpacing(12)
        self._head = section_header("", "")
        lay.addWidget(self._head)

        top = QWidget()
        toplay = QHBoxLayout(top)
        toplay.setContentsMargins(0, 0, 0, 0)
        toplay.setSpacing(12)
        toplay.addWidget(self._build_topology(), 3)
        toplay.addWidget(self._build_controls(), 2)
        top.setMinimumHeight(120)

        mid = QSplitter(Qt.Horizontal)
        mid.setChildrenCollapsible(False)
        mid.setHandleWidth(6)
        self.chart = _RateChart()
        self.event_table = self._build_event_table()
        telemetry = self._card(self.chart, "Live telemetry")
        stream = self._card(self.event_table, "A1 policy-update stream")
        # the floor belongs on the panes: a horizontal splitter's own minimum
        # height is derived from them, so declaring it here is what survives.
        telemetry.setMinimumHeight(200)
        stream.setMinimumHeight(200)
        mid.addWidget(telemetry)
        mid.addWidget(stream)
        mid.setStretchFactor(0, 2)
        mid.setStretchFactor(1, 3)
        apply_splitter_floor(mid)

        alerts = self._card(self._build_alert_table(), "Detection alerts")
        alerts.setMinimumHeight(170)

        # A vertical splitter (not a fixed-stretch layout) so the operator can
        # drag the boundary and give the alert log as much room as they need.
        self._vsplit = QSplitter(Qt.Vertical)
        self._vsplit.setChildrenCollapsible(False)
        self._vsplit.setHandleWidth(6)
        self._vsplit.addWidget(top)
        self._vsplit.addWidget(mid)
        self._vsplit.addWidget(alerts)
        for i in range(3):
            self._vsplit.setCollapsible(i, False)
        self._vsplit.setStretchFactor(0, 0)
        self._vsplit.setStretchFactor(1, 3)
        self._vsplit.setStretchFactor(2, 2)
        apply_splitter_floor(self._vsplit)

        # Right-hand live trust dock (robustness / fairness / privacy). It is a
        # plain pane, not a nested scroll area: one wheel per page, and the panel
        # asks for the height its own cards need.
        self.trust_panel = TrustLivePanel(_AGENTS)
        self.trust_panel.setMinimumWidth(240)
        self.trust_panel.setMinimumHeight(max(240, self.trust_panel.sizeHint().height()))

        self._hsplit = QSplitter(Qt.Horizontal)
        self._hsplit.setChildrenCollapsible(False)
        self._hsplit.setHandleWidth(6)
        self._hsplit.addWidget(self._vsplit)
        self._hsplit.addWidget(self.trust_panel)
        self._hsplit.setStretchFactor(0, 1)
        self._hsplit.setStretchFactor(1, 0)
        self._hsplit.setCollapsible(0, False)
        self._hsplit.setCollapsible(1, False)
        apply_splitter_floor(self._hsplit)
        lay.addWidget(self._hsplit, 1)

        self.scroll = PageScroll(body)
        self.conclusion = ConclusionPanel()
        self.scroll.add_footer(self.conclusion)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.scroll, 1)

        # Action buttons live OUTSIDE the splitter *and* outside the scroll area:
        # they can be neither clipped by a dragged handle nor scrolled off-page.
        foot = QWidget()
        footer = QHBoxLayout(foot)
        footer.setContentsMargins(18, 4, 18, 12)
        footer.setSpacing(10)
        self.explain_btn = QPushButton(i18n.tr("Explain selected alert ▸ Trustworthy AI"))
        self.explain_btn.setObjectName("primary")
        self.explain_btn.clicked.connect(self.explain_selected)
        self.clear_btn = QPushButton(i18n.tr("Clear"))
        self.clear_btn.clicked.connect(self._clear_alerts)
        footer.addWidget(self.explain_btn, 1)
        footer.addWidget(self.clear_btn)
        root.addWidget(foot)

        self._sized = False
        self.retranslate()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if not self._sized:
            self._sized = True
            QTimer.singleShot(0, self._apply_initial_sizes)

    def _apply_initial_sizes(self) -> None:
        h = self._vsplit.height() or 900
        self._vsplit.setSizes([max(180, int(h * 0.26)),
                               int(h * 0.36),
                               int(h * 0.38)])
        w = self._hsplit.width() or 1200
        self._hsplit.setSizes([int(w * 0.72), int(w * 0.28)])

    def _card(self, inner: QWidget, title_key: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(6)
        t = QLabel(i18n.tr(title_key))
        t.setObjectName("h2")
        lay.addWidget(t)
        lay.addWidget(inner, 1)
        self._card_titles[title_key] = t
        return card

    def _build_topology(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        row = QHBoxLayout()
        self.nrt = self._node("Non-RT RIC\n(SMO)", theme.ACCENT)
        a1 = self._link("A1")
        self.nearrt = self._node("Near-RT RIC", theme.ACCENT)
        row.addWidget(self.nrt)
        row.addWidget(a1)
        row.addWidget(self.nearrt)
        row.addStretch(1)
        lay.addLayout(row)

        self.agent_lights: Dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, a in enumerate(_AGENTS):
            light = QLabel(f"● {a}")
            light.setStyleSheet(f"color:{theme.GOOD}; font-weight:600;")
            grid.addWidget(light, i // 3, i % 3)
            self.agent_lights[a] = light
        lay.addLayout(grid)

        self.status_line = QLabel("")
        self.status_line.setObjectName("muted")
        self.status_line.setWordWrap(True)
        lay.addWidget(self.status_line)
        lay.addStretch(1)
        return w

    def _node(self, text_key: str, color: str) -> QLabel:
        lbl = QLabel(i18n.tr(text_key))
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"background:{theme.BG}; border:1px solid {color}; color:{color};"
            f"border-radius:8px; padding:10px 14px; font-weight:700;")
        lbl.setProperty("_key", text_key)
        return lbl

    def _link(self, text: str) -> QLabel:
        lbl = QLabel(f"— {text} —")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"color:{theme.MUTED};")
        return lbl

    def _build_controls(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.scenario_lbl = QLabel(i18n.tr("Injected scenario"))
        lay.addWidget(self.scenario_lbl)
        self.scenario = QComboBox()
        for name in SCENARIOS:
            self.scenario.addItem(i18n.tr(name), name)
        self.scenario.currentIndexChanged.connect(self._on_scenario_change)
        lay.addWidget(self.scenario)

        rate_row = QHBoxLayout()
        self.speed_lbl = QLabel(i18n.tr("Stream speed"))
        rate_row.addWidget(self.speed_lbl)
        self.speed = QComboBox()
        for lbl, ms in _SPEEDS:
            self.speed.addItem(i18n.tr(lbl), ms)
        self.speed.setCurrentIndex(1)
        self.speed.currentIndexChanged.connect(
            lambda _: self._timer.setInterval(self.speed.currentData()))
        rate_row.addWidget(self.speed, 1)
        lay.addLayout(rate_row)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.toggle_stream)
        self.burst_btn = QPushButton(i18n.tr("Inject burst"))
        self.burst_btn.setObjectName("danger")
        self.burst_btn.clicked.connect(lambda: self._emit_events(burst=True))
        btns.addWidget(self.start_btn)
        btns.addWidget(self.burst_btn)
        lay.addLayout(btns)

        self.model_pill = QLabel("")
        self.model_pill.setStyleSheet(
            f"color:{theme.MUTED}; border:1px solid {theme.GRID};"
            f"border-radius:8px; padding:5px 8px;")
        lay.addWidget(self.model_pill)
        lay.addStretch(1)
        self._render_start_btn()
        self._render_model_pill()
        return w

    def _build_event_table(self) -> QTableWidget:
        t = QTableWidget(0, len(_EVENT_HEADERS))
        t.setHorizontalHeaderLabels([i18n.tr(h) for h in _EVENT_HEADERS])
        _configure_header(t, _EVENT_WIDTHS)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setMinimumHeight(100)
        return t

    def _build_alert_table(self) -> QTableWidget:
        self.alerts = QTableWidget(0, len(_ALERT_HEADERS))
        self.alerts.setHorizontalHeaderLabels([i18n.tr(h) for h in _ALERT_HEADERS])
        _configure_header(self.alerts, _ALERT_WIDTHS)
        self.alerts.verticalHeader().setVisible(False)
        self.alerts.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.alerts.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.alerts.setMinimumHeight(100)
        self.alerts.doubleClicked.connect(lambda _: self.explain_selected())
        return self.alerts

    # ------------------------------------------------------------- i18n
    def retranslate(self) -> None:
        set_section_text(
            self._head,
            i18n.tr("O-RAN SMO / Near-RT RIC — live console"),
            i18n.tr("Non-RT RIC ⇄ A1 ⇄ Near-RT RIC control loop over 5 edge agents. "
                    "Stream policy traffic, inject an attack, then send an alert to the "
                    "Explainability panel to see *why* the RIC flagged it."),
        )
        for key, lbl in self._card_titles.items():
            lbl.setText(i18n.tr(key))
        self.chart.retranslate()
        self.trust_panel.retranslate()
        self.nrt.setText(i18n.tr(self.nrt.property("_key")))
        self.nearrt.setText(i18n.tr(self.nearrt.property("_key")))
        self.scenario_lbl.setText(i18n.tr("Injected scenario"))
        self.speed_lbl.setText(i18n.tr("Stream speed"))
        self._refresh_combo_text(self.scenario)
        self.speed.blockSignals(True)
        for i, (lbl, _ms) in enumerate(_SPEEDS):
            if i < self.speed.count():
                self.speed.setItemText(i, i18n.tr(lbl))
        self.speed.blockSignals(False)
        self.burst_btn.setText(i18n.tr("Inject burst"))
        self.explain_btn.setText(i18n.tr("Explain selected alert ▸ Trustworthy AI"))
        self.clear_btn.setText(i18n.tr("Clear"))
        self.event_table.setHorizontalHeaderLabels([i18n.tr(h) for h in _EVENT_HEADERS])
        self.alerts.setHorizontalHeaderLabels([i18n.tr(h) for h in _ALERT_HEADERS])
        self._render_start_btn()
        self._render_model_pill()
        self._update_status_line()
        self.conclusion.retranslate()

    def _refresh_combo_text(self, combo: QComboBox) -> None:
        keep = combo.currentData()
        combo.blockSignals(True)
        for i in range(combo.count()):
            key = combo.itemData(i)
            if isinstance(key, str):
                combo.setItemText(i, i18n.tr(key))
        if keep is not None:
            idx = combo.findData(keep)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    # ------------------------------------------------------------- wiring
    def set_model_bundle(self, bundle) -> None:
        self._bundle = bundle
        self._model_state = "ready"
        self._model_acc = bundle.metrics.get("accuracy")
        self.trust_panel.set_bundle(bundle)
        self._render_model_pill()
        self.model_status.emit("ready")

    def model_failed(self, msg: str) -> None:
        self._model_state = "heuristic"
        self._render_model_pill()
        self.model_status.emit(f"error: {msg}")

    def _render_model_pill(self) -> None:
        if self._model_state == "ready":
            txt = i18n.tr("detector model: ready")
            if self._model_acc is not None:
                txt += f" (A1 acc {self._model_acc:.2f})"
            color = theme.GOOD
        elif self._model_state == "heuristic":
            txt = i18n.tr("detector model: heuristic fallback")
            color = theme.BAD
        else:
            txt = i18n.tr("detector model: loading…")
            color = theme.MUTED
        self.model_pill.setText(txt)
        self.model_pill.setStyleSheet(
            f"color:{color}; border:1px solid {color};"
            f"border-radius:8px; padding:5px 8px;")
        self._refresh_conclusion()

    def _render_start_btn(self) -> None:
        self.start_btn.setText(
            i18n.tr("Pause stream") if self._running else i18n.tr("Start stream"))

    def toggle_stream(self) -> None:
        self._running = not self._running
        if self._running:
            self._timer.setInterval(self.speed.currentData())
            self._timer.start()
        else:
            self._timer.stop()
        self._render_start_btn()
        self._refresh_conclusion()

    def _on_scenario_change(self, *_):
        self.chart.reset()
        self.trust_panel.reset()

    def _current_class(self) -> int:
        return _SCENARIO_CLASS[self.scenario.currentData() or "Normal"]

    def _scenario_color(self) -> str:
        return {0: theme.GOOD, 1: theme.BAD, 2: "#fbbf24"}[self._current_class()]

    def _update_status_line(self) -> None:
        if not self._started_any:
            self.status_line.setText(i18n.tr("Idle — press Start stream."))
        else:
            self.status_line.setText(i18n.tr(
                "Scenario: {sc}  •  tick {tick}  •  {n} events logged  •  {m} alerts",
                sc=i18n.tr(self.scenario.currentData() or "Normal"),
                tick=self._tick, n=self.event_rows(), m=len(self._alerts)))
        self._refresh_conclusion()

    def _refresh_conclusion(self) -> None:
        """Rebuild the card from the live session; before the stream has ever run
        there is nothing to conclude, so the card keeps its placeholder."""
        if not hasattr(self, "conclusion"):
            return                              # called during _build_controls
        if not self._started_any:
            self.conclusion.set_bullets([])
            return
        self.conclusion.set_bullets(
            ui_narrator(console=self.snapshot()).conclusions("console"))

    # ------------------------------------------------------------- events
    def _on_tick(self) -> None:
        self._emit_events(burst=False)

    def _emit_events(self, burst: bool) -> None:
        self._started_any = True
        cls = self._current_class()
        n = {0: 2, 1: 8, 2: 2}[cls]
        if burst:
            n *= 4
        rates = []
        for _ in range(n):
            feat = self._synth_event(cls)
            rates.append(feat.get("request_rate", 0.0))
            verdict, prob, margin = self._detect(feat)
            self._log_event(feat, verdict)
            self.trust_panel.record_event(
                feat.get("_agent"), verdict != "Normal", margin)
            if verdict != "Normal":
                self._raise_alert(feat, verdict, prob, burst)
        if rates:
            self.chart.push(float(np.mean(rates)), self._scenario_color())
        self.trust_panel.refresh()
        self._tick += 1
        self._animate_agents(cls)
        self._update_status_line()

    def _synth_event(self, cls: int) -> Dict[str, float]:
        if self._bundle is not None and cls in self._bundle.prototypes:
            proto = self._bundle.prototypes[cls]
            mean, std = proto["mean"], proto["std"]
            # prototypes live in the scaler's z-space (values may be negative);
            # clipping at 0 here would drag DoS/Covert samples toward Normal.
            vals = mean + std * self._rng.normal(0, 0.35, size=mean.shape)
            feat = {name: float(vals[i])
                    for i, name in enumerate(self._bundle.feature_names)}
        else:                                    # heuristic fallback before model ready
            feat = self._heuristic_event(cls)
        feat["_policy"] = _POLICIES[int(self._rng.integers(0, len(_POLICIES)))]
        feat["_agent"] = _AGENTS[int(self._rng.integers(0, len(_AGENTS)))]
        feat["_t"] = time.strftime("%H:%M:%S")
        return feat

    def _heuristic_event(self, cls: int) -> Dict[str, float]:
        base = {
            0: dict(request_rate=0.8, real_interval=1.4, body_size=1400.0,
                    response_time_mean=0.02),
            1: dict(request_rate=380.0, real_interval=0.003, body_size=64.0,
                    response_time_mean=0.16),
            2: dict(request_rate=0.85, real_interval=1.35, body_size=1450.0,
                    response_time_mean=0.021),
        }[cls]
        feat = {}
        for k, v in base.items():
            feat[k] = float(max(0.0, v * (1 + self._rng.normal(0, 0.12))))
        return feat

    def _detect(self, feat: Dict[str, float]) -> tuple:
        """Return (verdict_label, p_anomaly, decision_margin). Uses the trained
        model when ready; the margin is the top1−top2 class-probability gap, a
        cheap confidence/robustness proxy for the live robustness card."""
        if self._bundle is not None:
            names = self._bundle.feature_names
            x = np.array([[feat.get(n, 0.0) for n in names]], float)
            try:
                proba = self._bundle.model.predict_proba(x)[0]
                pred = int(np.argmax(proba))
                p_anom = float(1.0 - proba[0])
                srt = np.sort(proba)[::-1]
                margin = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])
                label = self._bundle.class_names[pred] if pred < len(self._bundle.class_names) else str(pred)
                return label, p_anom, margin
            except Exception:
                pass
        rate = feat.get("request_rate", 0.0)
        if rate > 5.0:
            p = min(1.0, rate / 400.0)
            return "DoS", p, p
        return "Normal", 0.0, 0.0

    def _log_event(self, feat: Dict[str, float], verdict: str) -> None:
        t = self.event_table
        t.insertRow(0)
        color = theme.GOOD if verdict == "Normal" else theme.BAD
        cells = [feat.get("_t", ""), feat.get("_agent", ""), feat.get("_policy", ""),
                 f"{feat.get('request_rate', 0.0):.2f}",
                 f"{feat.get('body_size', 0.0):.0f}", i18n.label(verdict)]
        for c, txt in enumerate(cells):
            item = QTableWidgetItem(str(txt))
            if c == 5:
                item.setForeground(theme.qcolor(color))
            t.setItem(0, c, item)
        while t.rowCount() > _MAX_EVENTS:
            t.removeRow(t.rowCount() - 1)

    def _raise_alert(self, feat, verdict, prob, burst) -> None:
        sev = "CRITICAL" if (verdict == "DoS" or burst) else "WARNING"
        explain = self._explain_text(feat, verdict)
        self._alerts.insert(0, {"features": feat, "verdict": verdict,
                                "prob": prob, "severity": sev, "explain": explain})
        t = self.alerts
        t.insertRow(0)
        color = theme.BAD if sev == "CRITICAL" else "#fbbf24"
        cells = [sev, feat.get("_t", ""), feat.get("_agent", ""),
                 feat.get("_policy", ""), i18n.label(verdict), f"{prob:.2f}",
                 f"{feat.get('request_rate', 0.0):.1f}", explain]
        for c, txt in enumerate(cells):
            item = QTableWidgetItem(str(txt))
            if c == 0:
                item.setForeground(theme.qcolor(color))
            elif c == 7:
                item.setForeground(theme.qcolor(theme.ACCENT))
                item.setToolTip(str(txt))
            t.setItem(0, c, item)
        while t.rowCount() > _MAX_ALERTS:
            t.removeRow(t.rowCount() - 1)
            self._alerts.pop()

    def _explain_text(self, feat: Dict[str, float], verdict: str) -> str:
        """Compact, real-time LIME-style explanation for one alert row.

        Runs a fast local surrogate toward the flagged class and renders the
        top-3 contributing A1 features as ``name  weight% (direction)`` so the
        operator reads *why* the RIC flagged this event without clicking in.
        """
        b = self._bundle
        if b is None:
            return i18n.tr("detector not ready")
        try:
            from trustoran.trust import explainability as EX

            names = b.feature_names
            x = np.array([feat.get(n, 0.0) for n in names], float)
            try:
                target = b.class_names.index(verdict)
            except ValueError:
                target = b.target_class
            le = EX.local_explanation(b.model, x, b.background,
                                      target_class=target, n_samples=400, seed=42)
            coefs = np.asarray(le.coefficients, float)
            total = float(np.abs(coefs).sum())
            if total <= 0 or not np.isfinite(total):
                return i18n.tr("detector not ready")
            order = np.argsort(-np.abs(coefs))[:3]
            zh = i18n.language() == "zh"
            sep = "、" if zh else ", "
            parts = []
            for i in order:
                w = abs(float(coefs[i])) / total * 100.0
                direction = i18n.tr("pushes up") if coefs[i] >= 0 else i18n.tr("pushes down")
                name = i18n.label(names[i])
                parts.append(f"{name} {w:.0f}%（{direction}）" if zh
                             else f"{name} {w:.0f}% ({direction})")
            return sep.join(parts)
        except Exception:                        # keep the stream alive on any failure
            return i18n.tr("detector not ready")

    def _animate_agents(self, cls: int) -> None:
        color = self._scenario_color() if cls != 0 else theme.GOOD
        for a, light in self.agent_lights.items():
            on = self._rng.random() < (0.9 if cls != 0 else 0.35)
            light.setStyleSheet(
                f"color:{color if on else theme.GRID}; font-weight:600;")

    # ------------------------------------------------------------- actions
    def explain_selected(self) -> None:
        row = self.alerts.currentRow()
        if row < 0 or row >= len(self._alerts):
            self.status_line.setText(i18n.tr("Select an alert row first, then Explain."))
            return
        alert = self._alerts[row]
        self.explain_requested.emit(dict(alert["features"]))

    def _clear_alerts(self) -> None:
        self._alerts.clear()
        self.alerts.setRowCount(0)
        self._update_status_line()

    def event_rows(self) -> int:
        return self.event_table.rowCount()

    def snapshot(self) -> Dict[str, Any]:
        """Console state the PDF report prints, as plain data.

        The report generator must not know about Qt widgets, so everything it
        needs is flattened here (the alert log keeps the same row order the
        table shows, newest first).
        """
        alerts = [{
            "severity": a.get("severity", ""),
            "t": (a.get("features") or {}).get("_t", ""),
            "agent": (a.get("features") or {}).get("_agent", ""),
            "policy": (a.get("features") or {}).get("_policy", ""),
            "verdict": a.get("verdict", ""),
            "prob": a.get("prob", 0.0),
            "rate": (a.get("features") or {}).get("request_rate", 0.0),
        } for a in self._alerts]
        return {
            "tick": self._tick,
            "scenario": str(self.scenario.currentData() or "Normal"),
            "streaming": bool(self._running),
            "model_state": self._model_state,
            "model_acc": self._model_acc,
            "events": self.event_rows(),
            "alerts": alerts,
        }
