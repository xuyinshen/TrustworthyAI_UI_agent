"""Main application window.

Assembles the navigation rail, the run-controls toolbar, the four trustworthy-AI
panels, the O-RAN console and the Agent Eval panel into one window, and owns the
background workers (full pipeline run + the console's lazy detector model).

The menu bar (File / View / Tools / Help) carries what a toolbar should not:
page jumps, the text-size zoom, the language switch and the framework info.
Toolbar space was getting tight once the agent panel arrived, so the language
selector moved to View.

Language handling: the studio defaults to Simplified Chinese. Both
``i18n.set_language`` entry points (the View menu and any programmatic call)
notify every widget's ``retranslate()`` so the whole chrome re-renders in place
(no restart needed).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from . import theme
from . import i18n
from .agent_panel import AgentPanel, RESULTS_ROOT
from .console import OranConsole
from .panels import ExplainabilityPanel, OverviewPanel, PillarPanel, PILLAR_META
from .params_dialog import ParamDialog, params_are_tuned
from .report_dialog import ReportDialog
from .widgets import FlowLayout, status_pill
from .worker import ConsoleModelWorker, PipelineWorker

try:
    from trustoran.data.loaders import available_datasets
    from trustoran.models.factory import available_models, param_schema
    from trustoran.pipeline.engine import DATASET_PROFILES, PILLARS
except Exception:                            # pragma: no cover
    available_datasets = lambda: ["cci_a1sd"]
    available_models = lambda t=None: ["random_forest"]
    param_schema = lambda n: {}
    DATASET_PROFILES = {}
    PILLARS = ("explainability", "robustness", "fairness", "privacy")

NAV_ITEMS = [
    ("Overview", "overview"),
    ("Explainability", "explainability"),
    ("Robustness", "robustness"),
    ("Fairness", "fairness"),
    ("Privacy", "privacy"),
    ("O-RAN Console", "console"),
    ("Agent Eval", "agent"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        theme.apply_mpl_font()
        self._report = None
        self._bundle = None
        self._pipe_worker: Optional[PipelineWorker] = None
        self._model_worker: Optional[ConsoleModelWorker] = None
        self._model_params: Dict[str, Dict] = {}   # model name -> user overrides
        self._detector_state = "loading"
        self._detector_acc: Optional[float] = None

        self._build_central()
        self._build_statusbar()
        self._build_menus()
        self.resize(1320, 860)
        # every page scrolls and every control row wraps, so a much smaller
        # window is now usable — the old floor only forced the user to zoom out.
        self.setMinimumSize(900, 600)
        i18n.on_change(self._on_language_changed)
        self.retranslate()
        self._start_console_model()

    # ------------------------------------------------------------------ UI
    def _build_central(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # navigation rail (resizable — see the QSplitter below)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setMinimumWidth(96)
        for label_key, key in NAV_ITEMS:
            item = QListWidgetItem(i18n.tr(label_key))
            item.setData(Qt.UserRole, key)
            item.setData(Qt.UserRole + 1, label_key)
            self.nav.addItem(item)
        self.nav.currentRowChanged.connect(self._on_nav)

        # right side: toolbar + stacked pages
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(0)
        rlay.addWidget(self._build_toolbar())

        self.stack = QStackedWidget()
        self.overview = OverviewPanel()
        self.p_explain = ExplainabilityPanel()
        self.p_robust = PillarPanel("robustness")
        self.p_fair = PillarPanel("fairness")
        self.p_priv = PillarPanel("privacy")
        self.console = OranConsole()
        self.agent = AgentPanel()
        self._pages: Dict[str, QWidget] = {
            "overview": self.overview,
            "explainability": self.p_explain,
            "robustness": self.p_robust,
            "fairness": self.p_fair,
            "privacy": self.p_priv,
            "console": self.console,
            "agent": self.agent,
        }
        for _, key in NAV_ITEMS:
            self.stack.addWidget(self._pages[key])
        rlay.addWidget(self.stack, 1)
        self.agent.progress.connect(self._on_progress)

        # nav rail + right pane in a draggable splitter (resizable navigation)
        self._navsplit = QSplitter(Qt.Horizontal)
        self._navsplit.setChildrenCollapsible(False)
        self._navsplit.setHandleWidth(4)
        self._navsplit.addWidget(self.nav)
        self._navsplit.addWidget(right)
        self._navsplit.setStretchFactor(0, 0)
        self._navsplit.setStretchFactor(1, 1)
        self._navsplit.setCollapsible(0, False)
        self._navsplit.setCollapsible(1, False)
        outer.addWidget(self._navsplit, 1)

        self.nav.setCurrentRow(0)

        # console → explainability link
        self.console.explain_requested.connect(self._on_console_explain)
        self.overview.pdf_requested.connect(self._export_pdf)

    def _build_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(f"background:{theme.PANEL}; border-bottom:1px solid {theme.GRID};")
        # a flow row: on a narrow window the controls continue on the next line
        # (and the bar grows) instead of running off the right edge.
        lay = FlowLayout(bar, margin=0, spacing=12)
        lay.setContentsMargins(16, 10, 16, 10)

        self.dataset_lbl = QLabel(i18n.tr("Dataset"))
        lay.addWidget(self.dataset_lbl)
        self.dataset_combo = QComboBox()
        for key in available_datasets():
            note = (DATASET_PROFILES.get(key, {}) or {}).get("note", "")
            self.dataset_combo.addItem(f"{key}", key)
            idx = self.dataset_combo.count() - 1
            self.dataset_combo.setItemData(idx, note, Qt.ToolTipRole)
        self.dataset_combo.setMinimumWidth(150)
        self.dataset_combo.currentIndexChanged.connect(self._refresh_models)
        lay.addWidget(self.dataset_combo)

        self.model_lbl = QLabel(i18n.tr("Model"))
        lay.addWidget(self.model_lbl)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(130)
        lay.addWidget(self.model_combo)

        self.tune_btn = QPushButton("")
        self.tune_btn.setObjectName("tune")
        self.tune_btn.setCursor(Qt.PointingHandCursor)
        self.tune_btn.clicked.connect(self._open_param_dialog)
        lay.addWidget(self.tune_btn)
        self._render_tune_btn()

        self.quick_chk = QCheckBox(i18n.tr("Quick"))
        self.quick_chk.setChecked(True)
        self.quick_chk.setToolTip(i18n.tr("Smaller samples / fewer attack steps for a fast demo run."))
        lay.addWidget(self.quick_chk)

        self.pillar_lbl = QLabel(i18n.tr("Pillars"))
        lay.addWidget(self.pillar_lbl)

        self.pillar_checks: Dict[str, QCheckBox] = {}
        for p in PILLARS:
            cb = QCheckBox(p[:4].upper())
            cb.setChecked(True)
            self.pillar_checks[p] = cb
            lay.addWidget(cb)

        lay.addStretch(1)

        self.run_btn = QPushButton("")
        self.run_btn.setObjectName("primary")
        self.run_btn.setMinimumWidth(132)
        self.run_btn.clicked.connect(self.run_analysis)
        lay.addWidget(self.run_btn)

        self._refresh_models()
        self._render_run_btn()
        return bar

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.status_msg = QLabel(
            i18n.tr("Nothing computed yet. Choose a dataset and model, "
                     "then press Start calculation."))
        sb.addWidget(self.status_msg, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(160)
        sb.addPermanentWidget(self.progress)
        self.detector_pill = status_pill("", theme.MUTED)
        sb.addPermanentWidget(self.detector_pill)
        self._render_detector_pill()

    # --------------------------------------------------- first show: size only
    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if not getattr(self, "_nav_sized", False):
            self._nav_sized = True
            w = self._navsplit.width() or 1200
            self._navsplit.setSizes([150, max(600, w - 150)])

    # ------------------------------------------------------------- menus
    def _build_menus(self) -> None:
        """Page jumps, text-size zoom, language and the run/export commands.

        These used to crowd the toolbar (or had no home at all); a native menu
        bar is where desktop users expect app-level commands, and it gives the
        toolbar back to run controls.
        """
        mb = self.menuBar()
        self._menu_titles: List[tuple] = []
        self._action_texts: List[tuple] = []
        self._nav_actions: Dict[str, QAction] = {}
        self._size_actions: Dict[float, QAction] = {}
        self._lang_actions: Dict[str, QAction] = {}

        file_menu = mb.addMenu("")
        self._menu_titles.append((file_menu, "File"))
        self._add_action(file_menu, "Run analysis", self.run_analysis, "Ctrl+R")
        self._add_action(file_menu, "Run agent evaluation", self._run_agent,
                         "Ctrl+Shift+R")
        file_menu.addSeparator()
        self._add_action(file_menu, "Export PDF report…", self._export_pdf, "Ctrl+P")
        self._add_action(file_menu, "Export agent results", self._export_agent, "Ctrl+E")
        self._add_action(file_menu, "Open results folder", self._open_results)
        file_menu.addSeparator()
        self._add_action(file_menu, "Quit", self.close, "Ctrl+Q")

        view_menu = mb.addMenu("")
        self._menu_titles.append((view_menu, "View"))
        pages_menu = view_menu.addMenu("")
        self._menu_titles.append((pages_menu, "Pages"))
        for i, (label_key, key) in enumerate(NAV_ITEMS):
            act = self._add_action(
                pages_menu, label_key,
                lambda _checked=False, k=key: self._goto(k),
                f"Ctrl+{i + 1}")
            self._nav_actions[key] = act
        view_menu.addSeparator()
        size_menu = view_menu.addMenu("")
        self._menu_titles.append((size_menu, "Text size"))
        size_group = QActionGroup(self)
        size_group.setExclusive(True)
        for label_key, scale in (("Compact", 0.9), ("Default", 1.0), ("Large", 1.15)):
            act = self._add_action(size_menu, label_key,
                                   lambda _c=False, s=scale: self._set_text_size(s),
                                   checkable=True)
            size_group.addAction(act)
            self._size_actions[scale] = act
        view_menu.addSeparator()
        lang_menu = view_menu.addMenu("")
        self._menu_titles.append((lang_menu, "Language"))
        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)
        for code, disp in i18n.display_langs().items():
            act = QAction(disp, self)
            act.setCheckable(True)
            act.triggered.connect(lambda _c=False, c=code: i18n.set_language(c))
            lang_menu.addAction(act)
            lang_group.addAction(act)
            self._lang_actions[code] = act

        tools_menu = mb.addMenu("")
        self._menu_titles.append((tools_menu, "Tools"))
        self._open_tune_action = self._add_action(
            tools_menu, "Tune model parameters", self._open_param_dialog, "Ctrl+T")
        tools_menu.addSeparator()
        self._add_action(tools_menu, "Go to O-RAN console",
                         lambda: self._goto("console"), "Ctrl+G")

        help_menu = mb.addMenu("")
        self._menu_titles.append((help_menu, "Help"))
        self._add_action(help_menu, "About this studio", self._show_about, "F1")

    def _add_action(self, menu, key: str, slot=None, shortcut: str = "",
                    checkable: bool = False) -> QAction:
        act = QAction(i18n.tr(key), self)
        act.setCheckable(bool(checkable))
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        if slot is not None:
            act.triggered.connect(slot)
        menu.addAction(act)
        self._action_texts.append((act, key))
        return act

    def _goto(self, key: str) -> None:
        for row in range(self.nav.count()):
            if self.nav.item(row).data(Qt.UserRole) == key:
                self.nav.setCurrentRow(row)
                return

    def _run_agent(self) -> None:
        self._goto("agent")
        self.agent.run_evaluation()

    def _export_agent(self) -> None:
        self._goto("agent")
        self.agent.export_results()

    def _set_text_size(self, scale: float) -> None:
        from PySide6.QtWidgets import QApplication
        theme.apply_scale(QApplication.instance(), scale)
        act = self._size_actions.get(scale)
        if act is not None:
            act.setChecked(True)
        self.status_msg.setText(i18n.tr("Text size") +
                                f": {int(round(scale * 100))}%")

    def _open_results(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(RESULTS_ROOT))

    def _export_pdf(self) -> None:
        """Open the modal export dialog. The dialog re-reads the live objects when
        it generates, so this only reports what is available right now."""
        agent_out = getattr(self.agent, "_out", None)
        snap = self.console.snapshot()
        has_console = bool(snap.get("events")) or bool(snap.get("alerts")) \
            or int(snap.get("tick") or 0) > 0
        dlg = ReportDialog(self,
                           has_report=self._report is not None,
                           has_agent=agent_out is not None,
                           has_console=has_console,
                           headline=(self._report.headline()
                                     if self._report is not None else ""),
                           agent_headline=(agent_out.headline
                                           if agent_out is not None else ""))
        dlg.exec()
        self.status_msg.setText(dlg.status_lbl.text())

    def _show_about(self) -> None:
        QMessageBox.about(self, i18n.tr("About this studio"), i18n.tr(
            "Trustworthy AI for Open RAN — a personal research portfolio.\n\n"
            "• py/trustoran — four-pillar trust analysis (explainability, "
            "robustness, fairness, privacy)\n"
            "• O-RAN console — live Non-RT RIC ⇄ A1 ⇄ Near-RT RIC loop\n"
            "• Agent/module_3_agent_eval — trace-first, agent-agnostic trust "
            "evaluation for LLM agents\n\n") + i18n.tr(
            "This module is a personal research project that "
            "extends trustworthy-AI methodology to agent "
            "evaluation; it is not affiliated with any "
            "employer. The O-RAN scenarios and standards "
            "summaries are the author's own paraphrasing of "
            "public material."))

    def _on_language_changed(self, lang: str) -> None:
        self.retranslate()

    def retranslate(self) -> None:
        self.setWindowTitle(i18n.tr(
            "Trustworthy AI for Open RAN — SMO / Near-RT RIC Studio"))
        for row in range(self.nav.count()):
            item = self.nav.item(row)
            item.setText(i18n.tr(item.data(Qt.UserRole + 1)))
        self.dataset_lbl.setText(i18n.tr("Dataset"))
        self.model_lbl.setText(i18n.tr("Model"))
        self.quick_chk.setText(i18n.tr("Quick"))
        self.pillar_lbl.setText(i18n.tr("Pillars"))
        self.quick_chk.setToolTip(
            i18n.tr("Smaller samples / fewer attack steps for a fast demo run."))
        for p, cb in self.pillar_checks.items():
            title = PILLAR_META.get(p, (p.title(), ""))[0]
            cb.setToolTip(i18n.tr(title))
        self._render_run_btn()
        self._render_tune_btn()
        self._render_detector_pill()
        self._sync_menu_state()
        self.overview.retranslate()
        self._update_strategy()
        self.p_explain.retranslate()
        self.p_robust.retranslate()
        self.p_fair.retranslate()
        self.p_priv.retranslate()
        self.console.retranslate()
        self.agent.retranslate()

    def _sync_menu_state(self) -> None:
        for menu, key in getattr(self, "_menu_titles", []):
            menu.setTitle(i18n.tr(key))
        for act, key in getattr(self, "_action_texts", []):
            act.setText(i18n.tr(key))
        if self._open_tune_action is not None:
            self._open_tune_action.setEnabled(
                bool(param_schema(self.model_combo.currentData())))
        for code, act in getattr(self, "_lang_actions", {}).items():
            act.setChecked(code == i18n.language())
        for scale, act in getattr(self, "_size_actions", {}).items():
            act.setChecked(abs(scale - theme.SCALE) < 1e-6)

    # ------------------------------------------------------------- models
    def _dataset_strategy(self, key: str) -> str:
        """Human-readable summary of the per-dataset algorithm choices encoded in
        ``DATASET_PROFILES`` — i.e. *why* the three dataset families are analysed
        differently (task, fairness grouping, federation, per-pillar methods)."""
        prof = DATASET_PROFILES.get(key, {}) or {}
        regression = bool(prof.get("regression"))
        group = prof.get("fairness_group", "—")
        if prof.get("federate"):
            fed = i18n.tr("yes — FedAvg across {g}",
                          g=prof.get("federate_group", "agent"))
        else:
            fed = i18n.tr("no — centralized only")
        head = (f"{i18n.tr('Task')}: "
                f"{i18n.tr('Regression') if regression else i18n.tr('Classification')}"
                f"   •   {i18n.tr('Fairness group')}: {group}"
                f"   •   {i18n.tr('Federated')}: {fed}")
        expl = i18n.tr("Explainability: permutation importance + KernelSHAP + "
                       "LIME-style local surrogate + PDP/ICE")
        if regression:
            rob = i18n.tr("Robustness (regression): R² decay under Gaussian "
                          "noise drift — no adversarial attacks")
            fair = i18n.tr("Fairness (regression): between-group MAE / RMSE / bias gap")
        else:
            rob = i18n.tr("Robustness (classification): FGSM / PGD / DeepFool "
                          "+ adversarial training")
            fair = i18n.tr("Fairness (classification): demographic parity / "
                           "disparate impact / equal-opportunity gap")
        priv_parts = [i18n.tr("DP-SGD privacy–utility curve")]
        if prof.get("federate"):
            priv_parts.append(i18n.tr("FedAvg across {g}",
                                      g=prof.get("federate_group", "agent")))
        if not regression:
            priv_parts.append(i18n.tr("membership inference"))
        priv = i18n.tr("Privacy: ") + " + ".join(priv_parts)
        return "\n".join([head, expl, rob, fair, priv])

    def _update_strategy(self) -> None:
        if not hasattr(self, "overview"):
            return                              # called during toolbar build
        key = self.dataset_combo.currentData() or "cci_a1sd"
        self.overview.set_strategy(self._dataset_strategy(key))

    def _is_regression(self, key: str) -> bool:
        return bool((DATASET_PROFILES.get(key, {}) or {}).get("regression"))

    def _refresh_models(self, *_):
        key = self.dataset_combo.currentData()
        task = "regression" if self._is_regression(key) else "classification"
        current = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for m in available_models(task):
            self.model_combo.addItem(m, m)
        if current:
            i = self.model_combo.findData(current)
            if i >= 0:
                self.model_combo.setCurrentIndex(i)
        self.model_combo.blockSignals(False)
        self._update_strategy()
        self._render_tune_btn()

    # ------------------------------------------------------------- param tuning
    def _render_tune_btn(self) -> None:
        if not hasattr(self, "tune_btn"):
            return
        model = self.model_combo.currentData()
        tuned = params_are_tuned(model, self._model_params.get(model))
        base = i18n.tr("Tune parameters")
        self.tune_btn.setText(f"{base} ✓" if tuned else base)
        enabled = bool(param_schema(model))
        self.tune_btn.setEnabled(enabled)
        if getattr(self, "_open_tune_action", None) is not None:
            self._open_tune_action.setEnabled(enabled)
        if param_schema(model):
            self.tune_btn.setToolTip(i18n.tr(
                "Adjust this model's hyperparameters (applies to the next run)."))
        else:
            self.tune_btn.setToolTip(i18n.tr(
                "This model has no exposed tunable parameters."))

    def _open_param_dialog(self) -> None:
        model = self.model_combo.currentData()
        if not model or not param_schema(model):
            return
        dlg = ParamDialog(model, self._model_params.get(model, {}), parent=self)
        if dlg.exec() == ParamDialog.Accepted:
            self._model_params[model] = dlg.values()
            self._render_tune_btn()

    def _selected_pillars(self) -> List[str]:
        return [p for p in PILLARS if self.pillar_checks[p].isChecked()]

    # ------------------------------------------------------------- run
    def _render_run_btn(self) -> None:
        running = self._pipe_worker is not None and self._pipe_worker.isRunning()
        if running:
            self.run_btn.setText(i18n.tr("Running…"))
            self.run_btn.setToolTip(i18n.tr("Calculation in progress."))
            return
        if self._report is None:
            self.run_btn.setText(i18n.tr("Start calculation"))
            self.run_btn.setToolTip(i18n.tr(
                "Compute the ticked pillars for the dataset and model chosen above."))
        else:
            self.run_btn.setText(i18n.tr("Re-run calculation"))
            self.run_btn.setToolTip(i18n.tr(
                "Re-run the analysis on the current dataset / model / pillars."))

    def run_analysis(self) -> None:
        pillars = self._selected_pillars()
        if not pillars:
            QMessageBox.information(self, i18n.tr("No pillars"),
                                    i18n.tr("Select at least one trust pillar."))
            return
        if self._pipe_worker is not None and self._pipe_worker.isRunning():
            return
        dataset_key = self.dataset_combo.currentData()
        model_name = self.model_combo.currentData()
        quick = self.quick_chk.isChecked()
        model_params = dict(self._model_params.get(model_name, {}))

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status_msg.setText(f"{dataset_key} / {model_name} …")

        self._pipe_worker = PipelineWorker(
            dataset_key, model_name, pillars, quick, None, 42,
            model_params=model_params, parent=self)
        self._pipe_worker.progress.connect(self._on_progress)
        self._pipe_worker.succeeded.connect(self._on_report)
        self._pipe_worker.failed.connect(self._on_run_failed)
        self._pipe_worker.start()
        self._render_run_btn()

    def _on_progress(self, stage: str, pct: float, msg: str) -> None:
        self.progress.setValue(int(round(pct * 100)))
        self.status_msg.setText(f"[{stage}] {msg}")

    def _on_report(self, report) -> None:
        self._report = report
        self.overview.set_report(report)
        self.p_explain.set_report(report)
        self.p_robust.set_report(report)
        self.p_fair.set_report(report)
        self.p_priv.set_report(report)
        self.run_btn.setEnabled(True)
        self._render_run_btn()
        self.progress.setValue(100)
        self.status_msg.setText(report.headline())
        self.nav.setCurrentRow(0)

    def _on_run_failed(self, err: str) -> None:
        self.run_btn.setEnabled(True)
        self._render_run_btn()
        self.progress.setValue(0)
        self.status_msg.setText(i18n.tr("Run failed."))
        QMessageBox.critical(self, i18n.tr("Pipeline error"),
                             err.splitlines()[0] if err else "unknown")

    # ------------------------------------------------------------- console model
    def _start_console_model(self) -> None:
        self._model_worker = ConsoleModelWorker(seed=42, max_rows=8000, parent=self)
        self._model_worker.succeeded.connect(self._on_console_model)
        self._model_worker.failed.connect(self._on_console_model_failed)
        self._model_worker.start()

    def _render_detector_pill(self) -> None:
        if self._detector_state == "ready":
            txt = i18n.tr("detector: ready")
            if self._detector_acc is not None:
                txt += f" (acc {self._detector_acc:.2f})"
            color = theme.GOOD
        elif self._detector_state == "heuristic":
            txt = i18n.tr("detector: heuristic")
            color = theme.BAD
        else:
            txt = i18n.tr("detector: loading")
            color = theme.MUTED
        self.detector_pill.setText(txt)
        self.detector_pill.setStyleSheet(
            f"background:{theme.BG}; border:1px solid {color}; color:{color};"
            f"border-radius:10px; padding:3px 10px; font-weight:600;")

    def _on_console_model(self, bundle) -> None:
        self._bundle = bundle
        self.console.set_model_bundle(bundle)
        self._detector_state = "ready"
        self._detector_acc = bundle.metrics.get("accuracy")
        self._render_detector_pill()

    def _on_console_model_failed(self, msg: str) -> None:
        self.console.model_failed(msg.splitlines()[0])
        self._detector_state = "heuristic"
        self._render_detector_pill()

    # ------------------------------------------------------------- explain link
    def _on_console_explain(self, features: Dict[str, float]) -> None:
        if self._bundle is None:
            self.status_msg.setText(
                i18n.tr("Detector model still training — try again in a moment."))
            self.nav.setCurrentRow(5)
            return
        try:
            from trustoran.trust import explainability as EX
            from trustoran.viz import plotting as VZ

            b = self._bundle
            x = np.array([[features.get(n, 0.0) for n in b.feature_names]], float)
            pred = int(b.model.predict(x)[0])
            proba = b.model.predict_proba(x)[0]
            label = b.class_names[pred] if pred < len(b.class_names) else str(pred)
            p_anom = float(1.0 - proba[0])

            le = EX.local_explanation(b.model, x[0], b.background,
                                      target_class=b.target_class, seed=42)
            fig = VZ.fig_local_explanation(le)
            table = le.to_frame()

            is_anom = i18n.label(label) != i18n.label("Normal")
            accent = theme.BAD if is_anom else theme.GOOD
            verdict = f"{i18n.label(label)}  •  p(anomaly)={p_anom:.2f}"
            meta = (
                f"{features.get('_t','')}  •  {features.get('_agent','')}  •  "
                f"{features.get('_policy','')}\n"
                f"LIME-style surrogate over {len(b.feature_names)} A1 features. "
                f"Bars show each feature's push toward/away from the flagged class."
            )
            self.p_explain.show_live_explanation(features, verdict, accent, meta, fig, table)
            self.status_msg.setText(i18n.tr("Explained live A1 event → ") + i18n.label(label))
            self.nav.setCurrentRow(1)          # jump to Explainability
        except Exception as e:                 # pragma: no cover
            self.status_msg.setText(i18n.tr("Explanation failed: ") + str(e))

    # ------------------------------------------------------------- nav
    def _on_nav(self, row: int) -> None:
        if row < 0:
            return
        key = self.nav.item(row).data(Qt.UserRole)
        self.stack.setCurrentWidget(self._pages[key])
