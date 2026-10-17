"""PDF report dialog: pick a reader, pick a format, generate.

The dialog is a pure *collector*: it gathers the current ``AnalysisReport``, the
current ``AgentEvalOutput`` and a console snapshot, turns them into a
:class:`~trustoran_ui.pdf_report.ReportSpec`, and hands them to a worker thread.
Nothing is recomputed here, so the PDF can never disagree with the screen.

Style vs format is deliberately split. *Style* changes what is written (how much
jargon, which evidence, whether a section carries operational actions); *format*
changes only how it is laid out (paper, columns, language, DPI, theme, cover).
Selecting a style re-applies its default section set; the checkboxes stay free to
override it.

Settings persist through :class:`QSettings` except nothing sensitive is ever
collected here — the bearer token from the Agent panel is not read, and the
report writer refuses to put endpoint credentials into the document.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QDate, QSettings, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDateEdit, QDialog, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from . import i18n
from .widgets import PageScroll

try:
    from .pdf_report import (PAPERS, SECTIONS, STYLES, MARGINS, ReportSpec,
                             build_report, default_out_path, sections_for)
    _HAVE_RL = True
except Exception:                        # pragma: no cover
    _HAVE_RL = False
    PAPERS = {"A4": (0, 0)}
    SECTIONS = ()
    STYLES = ()
    MARGINS = {"Normal": 0.8}

    class ReportSpec:                    # type: ignore[too-few-public-methods]
        pass

    build_report = default_out_path = sections_for = None      # type: ignore

_STYLE_NAME_KEYS = {
    "layperson": "Non-specialist (plain language)",
    "smo": "SMO / RIC operations",
    "ai": "AI research (methods and appendix)",
}

#: one line per section so the checkboxes explain what they switch on
_SECTION_NOTE_KEYS = {
    "abstract": "What the numbers mean, written for the selected reader.",
    "data_model": "Dataset, model and hyperparameters used for this run.",
    "metrics": "Base performance on the held-out test split.",
    "explainability": "Attribution: which features drove the decisions.",
    "robustness": "Attack results and robustness curves.",
    "fairness": "Group fairness metrics and per-group tables.",
    "privacy": "DP utility curve, federation and membership inference.",
    "oran": "Live console state: scenario, alerts and detector status.",
    "agent": "Six-dimension agent scorecard, figures and run settings.",
    "method": "Reproduction commands, metric glossary, validity limits.",
    "appendix": "Every result table, uncapped.",
}

STYLE_HINT_KEYS = {
    "layperson": "Conclusions first. Plain language, one figure per topic, no "
                 "tables — written so a manager or a non-technical reviewer can "
                 "follow it.",
    "smo": "Operations view. Key read-outs, watch items, thresholds and concrete "
           "A1 / RIC actions derived from the current alert log.",
    "ai": "Methods view. Estimators and settings (attack family, ε sweeps, privacy "
          "accounting), metric definitions, every figure, and the full table "
          "appendix.",
}


def settings() -> QSettings:
    return QSettings("TrustworthyAI", "Studio")


class ReportWorker(QThread):
    """Builds the PDF off the GUI thread and reports back."""

    progress = Signal(str, float, str)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, spec: "ReportSpec", report: Any, agent: Any,
                 console: Optional[Dict[str, Any]], out_path: str, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.report = report
        self.agent = agent
        self.console = console
        self.out_path = out_path

    def run(self) -> None:                # noqa: D401 - Qt entry point
        try:
            path = build_report(self.spec, report=self.report, agent=self.agent,
                                console=self.console, out_path=self.out_path,
                                progress=lambda s, p, m: self.progress.emit(
                                    str(s), float(p), str(m)))
            self.succeeded.emit(path)
        except Exception as e:            # pragma: no cover
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class ReportDialog(QDialog):
    """Modal export dialog wired to :func:`pdf_report.build_report`."""

    def __init__(self, parent=None, *, has_report: bool = False,
                 has_agent: bool = False, has_console: bool = False,
                 headline: str = "", agent_headline: str = ""):
        super().__init__(parent)
        self.setWindowTitle(i18n.tr("Export PDF report…"))
        self.setMinimumSize(720, 420)
        self.resize(940, 640)
        self.has_report = has_report
        self.has_agent = has_agent
        self.has_console = has_console
        self.headline = headline
        self.agent_headline = agent_headline
        self._worker: Optional[ReportWorker] = None
        self._last_path = ""
        self._path_dirty = False
        self._tables_user = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)

        if not _HAVE_RL:
            lay.addWidget(self._missing_backend_card())
            self._build_bottom(lay)
            self.retranslate()
            return

        head = QFrame()
        head.setObjectName("card")
        hl = QVBoxLayout(head)
        hl.setContentsMargins(14, 10, 14, 10)
        hl.setSpacing(3)
        self._title_lbl = QLabel("")
        self._title_lbl.setObjectName("h1")
        self._src_lbl = QLabel("")
        self._src_lbl.setObjectName("muted")
        self._src_lbl.setWordWrap(True)
        hl.addWidget(self._title_lbl)
        hl.addWidget(self._src_lbl)
        lay.addWidget(head)

        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)
        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(self._build_style_card())
        left.addWidget(self._build_sections_card(), 1)
        bl.addLayout(left, 5)

        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(self._build_format_card())
        right.addWidget(self._build_meta_card())
        right.addWidget(self._build_output_card())
        right.addStretch(1)
        bl.addLayout(right, 4)

        # The whole two-column body scrolls: its height demand stops at the
        # viewport instead of pushing the Generate button off the screen.
        lay.addWidget(PageScroll(body), 1)

        self._build_bottom(lay)
        self._restore()
        self._on_style_change()
        self.retranslate()

    # ------------------------------------------------------------------ cards
    def _missing_backend_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lbl = QLabel("")
        lbl.setObjectName("h2")
        lbl.setWordWrap(True)
        self._missing_lbl = lbl
        lay.addWidget(lbl)
        return card

    def _build_style_card(self) -> QFrame:
        box = QGroupBox("")
        self._style_group_title = box
        grid = QGridLayout(box)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setSpacing(6)
        self.style_btns: Dict[str, QRadioButton] = {}
        self._style_lbls: Dict[str, QLabel] = {}
        for row, key in enumerate(STYLES):
            btn = QRadioButton("")
            btn.setObjectName(f"style_{key}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.toggled.connect(self._on_style_change)
            hint = QLabel("")
            hint.setObjectName("muted")
            hint.setWordWrap(True)
            hint.setMaximumWidth(430)
            grid.addWidget(btn, row, 0)
            grid.addWidget(hint, row, 1)
            self.style_btns[key] = btn
            self._style_lbls[key] = hint
        self._style_group = QButtonGroup(self)
        for key, btn in self.style_btns.items():
            self._style_group.addButton(btn)
        return box

    def _build_sections_card(self) -> QFrame:
        box = QGroupBox("")
        self._sections_title = box
        col = QVBoxLayout(box)
        col.setContentsMargins(10, 8, 10, 8)
        col.setSpacing(3)
        self.section_checks: Dict[str, QCheckBox] = {}
        for key, label_key in SECTIONS:
            cb = QCheckBox("")
            cb.setProperty("label_key", label_key)
            cb.setToolTip(i18n.tr(_SECTION_NOTE_KEYS[key]))
            self.section_checks[key] = cb
            col.addWidget(cb)
        row = QHBoxLayout()
        self.all_btn = QPushButton("")
        self.all_btn.setObjectName("tune")
        self.all_btn.clicked.connect(lambda: self._set_all(True))
        self.none_btn = QPushButton("")
        self.none_btn.setObjectName("tune")
        self.none_btn.clicked.connect(lambda: self._set_all(False))
        row.addWidget(self.all_btn)
        row.addWidget(self.none_btn)
        row.addStretch(1)
        col.addLayout(row)
        col.addStretch(1)
        return box

    def _build_format_card(self) -> QFrame:
        box = QGroupBox("")
        self._format_title = box
        form = QFormLayout(box)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(6)

        self.paper_combo = QComboBox()
        for p in PAPERS:
            self.paper_combo.addItem(p, p)
        form.addRow(self._lbl("Paper"), self.paper_combo)

        self.cols_combo = QComboBox()
        self.cols_combo.addItem(i18n.tr("One column"), 1)
        self.cols_combo.addItem(i18n.tr("Two columns"), 2)
        form.addRow(self._lbl("Columns"), self.cols_combo)

        self.lang_combo = QComboBox()
        for code, disp in i18n.display_langs().items():
            self.lang_combo.addItem(disp, code)
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData("zh")))
        form.addRow(self._lbl("Report language"), self.lang_combo)

        self.margin_combo = QComboBox()
        for m in MARGINS:
            self.margin_combo.addItem(i18n.tr(f"{m} margins"), m)
        self.margin_combo.setCurrentIndex(max(0, self.margin_combo.findData("Normal")))
        form.addRow(self._lbl("Margins"), self.margin_combo)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem(i18n.tr("Dark"), "dark")
        self.theme_combo.addItem(i18n.tr("Light"), "light")
        form.addRow(self._lbl("Colour theme"), self.theme_combo)

        self.dpi_combo = QComboBox()
        from .pdf_report import FIG_DPI
        for d in FIG_DPI:
            self.dpi_combo.addItem(f"{d} dpi", d)
        self.dpi_combo.setCurrentIndex(2)
        form.addRow(self._lbl("Figure DPI"), self.dpi_combo)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(6, 200)
        self.rows_spin.setValue(24)
        self.rows_spin.setSuffix(" " + i18n.tr("rows"))
        form.addRow(self._lbl("Table rows"), self.rows_spin)

        self.cover_chk = QCheckBox("")
        self.cover_chk.setChecked(True)
        form.addRow(self.cover_chk)
        self.header_chk = QCheckBox("")
        self.header_chk.setChecked(True)
        form.addRow(self.header_chk)
        self.pages_chk = QCheckBox("")
        self.pages_chk.setChecked(True)
        form.addRow(self.pages_chk)
        self.tables_chk = QCheckBox("")
        self.tables_chk.setToolTip(i18n.tr(
            "Override the style default: put each pillar's data tables into the "
            "body instead of only in the appendix."))
        self.tables_chk.clicked.connect(lambda *_: setattr(self, "_tables_user", True))
        form.addRow(self.tables_chk)
        return box

    def _build_meta_card(self) -> QFrame:
        box = QGroupBox("")
        self._meta_title = box
        form = QFormLayout(box)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(6)
        self.title_edit = QLineEdit("")
        self.title_edit.setPlaceholderText(i18n.tr(
            "Trustworthy AI for Open RAN — Analysis Report"))
        form.addRow(self._lbl("Title"), self.title_edit)
        self.author_edit = QLineEdit(str(settings().value("report/author", "")))
        self.author_edit.setPlaceholderText(os.environ.get("USERNAME", ""))
        form.addRow(self._lbl("Author"), self.author_edit)
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        form.addRow(self._lbl("Date"), self.date_edit)
        return box

    def _build_output_card(self) -> QFrame:
        box = QGroupBox("")
        self._output_title = box
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        row = QHBoxLayout()
        self.path_edit = QLineEdit(default_out_path("smo"))
        self.path_edit.textEdited.connect(lambda *_: setattr(self, "_path_dirty", True))
        row.addWidget(self.path_edit, 1)
        self.browse_btn = QPushButton("")
        self.browse_btn.setObjectName("tune")
        self.browse_btn.clicked.connect(self._browse)
        row.addWidget(self.browse_btn)
        lay.addLayout(row)
        self.hint_lbl = QLabel("")
        self.hint_lbl.setObjectName("muted")
        self.hint_lbl.setWordWrap(True)
        lay.addWidget(self.hint_lbl)
        return box

    def _build_bottom(self, lay: QVBoxLayout) -> None:
        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("muted")
        self.status_lbl.setWordWrap(True)
        lay.addWidget(self.status_lbl)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        row = QHBoxLayout()
        self.privacy_lbl = QLabel("")
        self.privacy_lbl.setObjectName("muted")
        row.addWidget(self.privacy_lbl, 1)
        self.open_btn = QPushButton("")
        self.open_btn.setObjectName("tune")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_pdf)
        row.addWidget(self.open_btn)
        self.gen_btn = QPushButton("")
        self.gen_btn.setObjectName("primary")
        self.gen_btn.setMinimumWidth(150)
        self.gen_btn.clicked.connect(self.generate)
        row.addWidget(self.gen_btn)
        self.close_btn = QPushButton("")
        self.close_btn.setObjectName("tune")
        self.close_btn.clicked.connect(self.reject)
        row.addWidget(self.close_btn)
        lay.addLayout(row)

    def _lbl(self, key: str) -> QLabel:
        lbl = QLabel(i18n.tr(key))
        lbl.setProperty("i18n_key", key)
        return lbl

    # ------------------------------------------------------------------ state
    def _on_style_change(self, *_):
        if not _HAVE_RL:
            return
        style = self.current_style()
        for key, cb in self.section_checks.items():
            cb.blockSignals(True)
            cb.setChecked(key in sections_for(style))
            cb.blockSignals(False)
        if not self._tables_user:
            self.tables_chk.blockSignals(True)
            self.tables_chk.setChecked(style == "ai")
            self.tables_chk.blockSignals(False)
        if not self._path_dirty:
            self.path_edit.setText(default_out_path(style))
        self._remember()

    def _set_all(self, on: bool) -> None:
        for cb in self.section_checks.values():
            cb.setChecked(on)

    def current_style(self) -> str:
        for key, btn in self.style_btns.items():
            if btn.isChecked():
                return key
        return "smo"

    # ------------------------------------------------------------------ spec
    def spec(self) -> "ReportSpec":
        return ReportSpec(
            style=self.current_style(),
            sections=tuple(k for k, _ in SECTIONS if self.section_checks[k].isChecked()),
            language=str(self.lang_combo.currentData()),
            paper=str(self.paper_combo.currentData()),
            columns=int(self.cols_combo.currentData()),
            margin=str(self.margin_combo.currentData()),
            dpi=int(self.dpi_combo.currentData()),
            theme=str(self.theme_combo.currentData()),
            cover=self.cover_chk.isChecked(),
            header=self.header_chk.isChecked(),
            page_numbers=self.pages_chk.isChecked(),
            body_tables=(None if not self._tables_user
                         else self.tables_chk.isChecked()),
            title=self.title_edit.text().strip(),
            author=self.author_edit.text().strip(),
            date=self.date_edit.date().toString("yyyy-MM-dd"),
            max_table_rows=int(self.rows_spin.value()))

    # ------------------------------------------------------------------ i18n
    def retranslate(self) -> None:
        self.setWindowTitle(i18n.tr("Export PDF report…"))
        for w in self.findChildren(QLabel):
            key = w.property("i18n_key")
            if key:
                w.setText(i18n.tr(str(key)))
        self.gen_btn.setText(i18n.tr("Generate PDF"))
        self.open_btn.setText(i18n.tr("Open PDF"))
        self.close_btn.setText(i18n.tr("Close"))
        self.privacy_lbl.setText(i18n.tr(
            "API keys are never written into a report."))
        if not _HAVE_RL:
            self._missing_lbl.setText(i18n.tr(
                "The PDF engine (reportlab) is not installed in this Python "
                "environment. Install it with:  pip install reportlab"))
            self.gen_btn.setEnabled(False)
            self._render_sources()
            return

        self._title_lbl.setText(i18n.tr("Export PDF report"))
        self._style_group_title.setTitle(i18n.tr("Reader style"))
        self._sections_title.setTitle(i18n.tr("Sections"))
        for key, cb in self.section_checks.items():
            cb.setText(i18n.tr(dict(SECTIONS)[key]))
        self._format_title.setTitle(i18n.tr("Format"))
        self._meta_title.setTitle(i18n.tr("Report metadata"))
        self._output_title.setTitle(i18n.tr("Output"))
        self.cover_chk.setText(i18n.tr("Cover page"))
        self.header_chk.setText(i18n.tr("Header band with the report title"))
        self.pages_chk.setText(i18n.tr("Page numbers"))
        self.tables_chk.setText(i18n.tr("Tables in the body"))
        self.tables_chk.setToolTip(i18n.tr(
            "Override the style default: put each pillar's data tables into the "
            "body instead of only in the appendix."))
        self.all_btn.setText(i18n.tr("All"))
        self.none_btn.setText(i18n.tr("None"))
        self.browse_btn.setText(i18n.tr("Browse…"))
        self.rows_spin.setSuffix(" " + i18n.tr("rows"))
        self.title_edit.setPlaceholderText(i18n.tr(
            "Trustworthy AI for Open RAN — Analysis Report"))
        self.hint_lbl.setText(i18n.tr(
            "The report contains only what the panels already show: metrics, "
            "figures, tables, the console alert log and the agent scorecard."))
        self._render_style_hints()
        self._render_sources()
        self._retranslate_combos()

    def _retranslate_combos(self) -> None:
        for i in range(self.cols_combo.count()):
            one = int(self.cols_combo.itemData(i)) == 1
            self.cols_combo.setItemText(
                i, i18n.tr("One column" if one else "Two columns"))
        for i in range(self.margin_combo.count()):
            self.margin_combo.setItemText(
                i, i18n.tr(f"{self.margin_combo.itemData(i)} margins"))

    def _render_style_hints(self) -> None:
        for key, hint in STYLE_HINT_KEYS.items():
            self._style_lbls[key].setText(i18n.tr(hint))
        for key, btn in self.style_btns.items():
            btn.setText(i18n.tr(_STYLE_NAME_KEYS[key]))

    def _render_sources(self) -> None:
        if not _HAVE_RL:
            self.status_lbl.setText(i18n.tr("reportlab is not installed."))
            return
        parts: List[str] = []
        parts.append(i18n.tr("analysis run: ") + (
            i18n.tr("available") if self.has_report else i18n.tr("not run yet")))
        parts.append(i18n.tr("agent evaluation: ") + (
            i18n.tr("available") if self.has_agent else i18n.tr("not run yet")))
        parts.append(i18n.tr("console session: ") + (
            i18n.tr("available") if self.has_console else i18n.tr("empty")))
        self.status_lbl.setText("   •   ".join(parts))
        readout: List[str] = []
        if self.headline:
            readout.append(i18n.tr("Analysis: ") + self.headline)
        if self.agent_headline:
            readout.append(i18n.tr("Agent: ") + self.agent_headline)
        self._src_lbl.setText("   •   ".join(readout) if readout else i18n.tr(
            "The report contains only what the panels already show: metrics, "
            "figures, tables, the console alert log and the agent scorecard."))
        for key, cb in self.section_checks.items():
            empty = (key in ("data_model", "metrics", "explainability",
                             "robustness", "fairness", "privacy")
                     and not self.has_report) or \
                    (key == "agent" and not self.has_agent)
            cb.setToolTip(i18n.tr(_SECTION_NOTE_KEYS[key]) +
                          (("\n" + i18n.tr(
                              "no data yet — this section will print a placeholder"))
                           if empty else ""))

    # ------------------------------------------------------------------ settings
    def _remember(self) -> None:
        st = settings()
        st.setValue("report/style", self.current_style())
        st.setValue("report/paper", str(self.paper_combo.currentData()))
        st.setValue("report/columns", int(self.cols_combo.currentData()))
        st.setValue("report/language", str(self.lang_combo.currentData()))
        st.setValue("report/margin", str(self.margin_combo.currentData()))
        st.setValue("report/theme", str(self.theme_combo.currentData()))
        st.setValue("report/dpi", int(self.dpi_combo.currentData()))
        st.setValue("report/rows", int(self.rows_spin.value()))
        st.setValue("report/author", self.author_edit.text().strip())

    def _restore(self) -> None:
        st = settings()
        style = str(st.value("report/style", "smo"))
        if style in self.style_btns:
            self.style_btns[style].setChecked(True)
        for key, combo in (("report/paper", self.paper_combo),
                           ("report/margin", self.margin_combo),
                           ("report/theme", self.theme_combo),
                           ("report/language", self.lang_combo)):
            self._pick(combo, str(st.value(key, "") or ""))
        for key, combo in (("report/columns", self.cols_combo),
                           ("report/dpi", self.dpi_combo)):
            self._pick(combo, str(st.value(key, "") or ""), as_int=True)
        rows = st.value("report/rows")
        if rows is not None:
            try:
                self.rows_spin.setValue(int(rows))
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _pick(combo: QComboBox, want: str, as_int: bool = False) -> None:
        """Select an item by its stored data (QSettings keeps everything as text)."""
        if not want:
            return
        for i in range(combo.count()):
            data = combo.itemData(i)
            if as_int:
                try:
                    same = int(want) == int(data)
                except (TypeError, ValueError):
                    same = False
            else:
                same = want == str(data)
            if same:
                combo.setCurrentIndex(i)
                return

    # ------------------------------------------------------------------ actions
    def _browse(self) -> None:
        start = self.path_edit.text() or default_out_path(self.current_style())
        path, _ = QFileDialog.getSaveFileName(
            self, i18n.tr("Save PDF report"), start, "PDF (*.pdf)")
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self.path_edit.setText(path)
            self._path_dirty = True

    def generate(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if not any(cb.isChecked() for cb in self.section_checks.values()):
            QMessageBox.information(self, i18n.tr("Nothing selected"),
                                    i18n.tr("Tick at least one section to include."))
            return
        spec = self.spec()
        self._remember()
        path = self.path_edit.text().strip() or default_out_path(spec.style)
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.gen_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.status_lbl.setText(i18n.tr("generating…"))
        host = self.parent()
        report = getattr(host, "_report", None)
        agent = getattr(getattr(host, "agent", None), "_out", None)
        snapshot = getattr(getattr(host, "console", None), "snapshot", None)
        console = snapshot() if callable(snapshot) else None
        self._worker = ReportWorker(spec, report, agent, console, path, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, stage: str, pct: float, msg: str) -> None:
        self.progress.setValue(int(round(pct * 100)))
        self.status_lbl.setText(f"[{stage}] {msg}")

    def _on_done(self, path: str) -> None:
        self._last_path = path
        self.gen_btn.setEnabled(True)
        self.progress.setValue(100)
        self.progress.setVisible(False)
        self.open_btn.setEnabled(os.path.isfile(path))
        self.status_lbl.setText(i18n.tr("Report written to ") + path)

    def _on_failed(self, err: str) -> None:
        self.gen_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_lbl.setText(i18n.tr("Report failed."))
        first = (err or "").splitlines()[0] if err else "unknown"
        QMessageBox.critical(self, i18n.tr("PDF report error"), first)

    def _open_pdf(self) -> None:
        if os.path.isfile(self._last_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_path))
