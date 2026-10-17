"""Agent-trust evaluation panel — the studio's window into ``Agent/module_3_agent_eval``.

The package under ``Agent/module_3_agent_eval`` is a standalone, CLI-first research
module; this panel is *only* a view onto it. It adds the module root to
``sys.path``, calls ``trustagent.runner.run(EvalConfig(...))`` on a background
thread, and renders the resulting scorecard with the same figure/table widgets the
four-pillar panels use. If the module is absent the panel says so and stays inert —
nothing else in the studio depends on it.

Two tabs, one runtime. *Suite run* asks the benchmark question ("how trustworthy is
this agent across the whole grid" — the case count is whatever the chosen preset's
conditions × base tasks × policies × k works out to, so it is not quoted here).
*Adaptive test* asks the operator's question: type any
free-text instruction, and the harness fans it out into perturbation / requester-
register / indirect-injection variants, measures the four classic pillars on the
answers it gets back, then runs the trust layer as a **corrector** and re-measures.
The tab shows the pair — answer before, answer after, per-pillar before→after with a
delta — plus the receipts: which edit fired, what it removed, and the caveat that
says what correction categorically does *not* fix.

One deliberate honesty rule, shared by both tabs: when the run used the
deterministic ``MockLLM`` the panel keeps showing a validity-limit note. Mock numbers
validate the harness; they are not a measurement of any real model. The same applies
to the probe's numbers — and, because a free-text instruction has no gold answer, a
probe never claims correctness: what it reports is *answered*, *grounded* (traceable
to a tool observation the agent actually received) and *agreement with the clean run*.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from . import i18n
from . import theme
from .narrative import ui_narrator
from .widgets import (
    ConclusionPanel, FigureGrid, FlowLayout, PageScroll, SummaryStrip, TableStack,
    apply_splitter_floor, enable_height_for_width, section_header,
    set_section_text, status_pill,
)

#: .../可信AI — the same roots main.py puts on sys.path for ``trustoran``.
#: A frozen bundle has no ``trustoran_ui/../..`` layout, so the repo root comes
#: from trustoran.config (which is bundle-aware), falling back to the source tree.
try:
    from trustoran import config as _trustoran_config
    REPO = str(_trustoran_config.REPO_ROOT)
except Exception:                              # pragma: no cover - source layout
    REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", ".."))
MODULE_ROOT = os.path.join(REPO, "Agent", "module_3_agent_eval")
RESULTS_ROOT = os.path.join(REPO, "results")

#: (label key, condition tuple); "Standard" mirrors trustagent.DEFAULT_CONDITIONS
PRESETS = (
    ("Quick", ("clean", "rename:L2")),
    ("Standard", ("clean", "paraphrase:L2", "rename:L2", "rename:L4",
                  "reorder:L2", "describe:L2")),
    ("Full", ("clean", "paraphrase:L1", "paraphrase:L2", "paraphrase:L3",
              "paraphrase:L4", "rename:L1", "rename:L2", "rename:L3",
              "rename:L4", "reorder:L2", "describe:L1", "describe:L3")),
)

SUITES = (
    ("tool_call_subset", "Tool calls"),
    ("injection_suite", "Injection"),
    ("rag_qa_suite", "RAG"),
    ("persona_parity_suite", "Persona"),
    ("permission_probes", "Permission"),
)

AGENTS = (("react", "ReAct loop"),
          ("direct", "Single-shot call"),
          ("both", "Both adapters"))

LLMS = (("mock", "Mock (offline, deterministic)"),
        ("ollama", "Ollama (local)"),
        ("openai", "OpenAI-compatible"))

DEFAULT_SUITES = tuple(name for name, _ in SUITES)

#: Where each live transport usually listens. bionic / LM Studio / vLLM / llama.cpp
#: all speak the OpenAI protocol; only the port changes.
BACKEND_ENDPOINT = {"ollama": "http://localhost:11434",
                    "openai": "http://localhost:1234/v1"}

#: The GGUF models served from bionic on this machine. "auto" asks the endpoint.
PRESET_MODELS = ("qwen3.6:27b", "qwen3.8:27b")

#: Probe depths. The cost is in *trajectories*, not in cosmetics: a deep probe is
#: two dozen ReAct runs of up to max_steps model calls each.
DEPTH_COST = (("quick", "~5 runs"), ("standard", "~10 runs"), ("deep", "~24 runs"))

#: The four classic pillars, in the order the probe report and the bars use.
PILLAR_KEYS = ("explainability", "robustness", "fairness", "privacy")

#: ``Correction.edits[*]["code"]`` -> an English literal the operator can read.
#: Codes come from trustagent.adaptive.correct; an unknown code degrades to the
#: raw code rather than to a blank row, so a new edit never renders as nothing.
EDIT_LABELS = {
    "payload_stripped": "Injected instruction stripped from the answer",
    "restated_from_reference": "Answer restated from the clean run",
    "kept_grounded_drift": "Drift kept — the variant traces to its own observations",
    "kept_variant_answer": "Variant answer kept",
    "operator_supplied_kept": "Operator-typed values left unredacted",
    "pii_redact": "Identifiers redacted",
    "append_evidence": "Evidence block appended",
    "flag_unverified": "Unverified claims flagged",
    "safety_note": "Safety declaration added",
    "canonical_template": "Canonical three-part template applied",
}
CAVEAT_LABELS = {
    "action_not_unexecuted": "Tool actions already executed were not undone",
    "restated_claims_not_seen_in_this_run":
        "Restated claims were never observed in this trajectory",
    "outbound_already_sent": "Already left through an outbound channel — not unsent",
    "model_still_leaks": "The model still emitted these values",
    "irreversible_done": "Irreversible action executed",
    "acl_blocked_attempts": "ACL-blocked attempts in this trajectory",
}

#: ``detail[pillar]["code"]`` from trustagent.adaptive.measure: why a pillar came
#: back with no score.  Readable literals, so the bar is never left to imply a 0.
MEASURE_CODES = {
    "no_perturbation_variants": "no perturbed variant ran in this probe, so there is "
                                "nothing to compare",
    "need_two_registers": "the probe stayed in one register, so there is nothing to "
                          "compare parity against",
    "no_cases": "nothing ran under this heading, so it was never observed",
}

#: ``parts["safety"]`` is built from two fixed prefixes in the corrector.
SAFETY_PREFIX_ZH = (("executed irreversible action(s):", "已执行不可逆动作："),
                    ("blocked attempt(s) by ACL:", "被 ACL 拦下的尝试："))


def settings() -> QSettings:
    return QSettings("TrustworthyAI", "Studio")


def module_available() -> bool:
    if os.path.isfile(os.path.join(MODULE_ROOT, "trustagent", "runner.py")):
        return True
    import importlib.util
    return importlib.util.find_spec("trustagent") is not None


def ensure_import_path() -> None:
    if os.path.isdir(MODULE_ROOT) and MODULE_ROOT not in sys.path:
        sys.path.insert(0, MODULE_ROOT)


@dataclass
class AgentEvalOptions:
    agents: tuple = ("react",)
    suites: tuple = DEFAULT_SUITES
    conditions: tuple = PRESETS[1][1]
    policies: tuple = ("eager", "careful")
    k: int = 1
    llm: str = "mock"
    model: str = "mock"
    endpoint: str = ""
    api_key: str = ""
    judge_model: str = ""
    max_tokens: int = 512
    judge: bool = True
    attribution: bool = False
    max_steps: int = 8

    def to_eval_kwargs(self) -> Dict[str, Any]:
        """Map onto ``EvalConfig``. ``options`` is how the runner reaches a server.

        The transport decides the key: Ollama wants ``host``, everything else
        wants an OpenAI-style ``base_url``.
        """
        opts: Dict[str, Any] = {"max_tokens": max(128, int(self.max_tokens))}
        url = (self.endpoint or "").strip()
        if self.llm == "ollama":
            opts["host"] = url or BACKEND_ENDPOINT["ollama"]
        elif self.llm != "mock":
            opts["base_url"] = url or BACKEND_ENDPOINT["openai"]
        if self.api_key.strip():
            opts["api_key"] = self.api_key.strip()
        if self.judge_model.strip():
            opts["judge_model"] = self.judge_model.strip()
        return dict(agents=self.agents, suites=self.suites,
                    conditions=self.conditions, policies=self.policies,
                    k=max(1, int(self.k)), llm_spec=self.llm,
                    llm_model=self.model or self.llm,
                    judge_enabled=bool(self.judge), max_steps=int(self.max_steps),
                    attribution=bool(self.attribution),
                    options=opts,
                    corpus_dir=os.path.join(MODULE_ROOT, "corpus"))


@dataclass
class AdaptiveOptions:
    """Everything the adaptive probe needs, assembled on the GUI thread.

    Mirrors :class:`AgentEvalOptions` for the fields the two share, so one
    runtime bar drives both tabs; the rest are the probe's own dials
    (depth, single agent, confirmation policy).
    """
    text: str = ""
    depth: str = "standard"
    agent: str = "react"
    policy: str = "careful"
    llm: str = "mock"
    model: str = "mock"
    endpoint: str = ""
    api_key: str = ""
    judge_model: str = ""
    max_tokens: int = 512
    max_steps: int = 8
    temperature: float = 0.2
    history: tuple = ()

    def to_probe_kwargs(self) -> Dict[str, Any]:
        """``ProbeOptions.for_depth`` overrides the depth preset with these.

        Empty strings must not be merged in — they would blank out a preset's
        own value, and ``model`` would then reach the endpoint as ``""``.
        """
        kw: Dict[str, Any] = {"agent": self.agent, "policy": self.policy,
                              "llm_spec": self.llm or "mock",
                              "max_tokens": max(128, int(self.max_tokens)),
                              "max_steps": int(self.max_steps),
                              "temperature": float(self.temperature)}
        for key, value in (("llm_model", self.model), ("endpoint", self.endpoint),
                           ("api_key", self.api_key), ("judge_model", self.judge_model)):
            if str(value or "").strip():
                kw[key] = str(value).strip()
        return kw


@dataclass
class AgentEvalOutput:
    """Everything the panel renders, assembled off the GUI thread."""
    result: Any = None                       # trustagent.runner.EvalResult
    report: Dict[str, Any] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    headline: str = ""
    mode: str = "mock"
    n_cases: int = 0
    duration_sec: float = 0.0
    exported: Dict[str, str] = field(default_factory=dict)


class AgentEvalWorker(QThread):
    """Runs the trace-first agent evaluation off the GUI thread."""

    progress = Signal(str, float, str)
    succeeded = Signal(object)               # AgentEvalOutput
    failed = Signal(str)

    def __init__(self, opts: AgentEvalOptions, parent=None):
        super().__init__(parent)
        self.opts = opts

    def run(self) -> None:                    # noqa: D401 - Qt entry point
        try:
            ensure_import_path()
            from trustagent.harness.llm import list_models
            from trustagent.report.scorecard import build_figures, build_tables, headline
            from trustagent.runner import EvalConfig, run

            kwargs = self.opts.to_eval_kwargs()
            if kwargs["llm_model"] == "auto" and kwargs["llm_spec"] != "mock":
                ep = kwargs["options"]
                found = list_models(kwargs["llm_spec"],
                                    base_url=ep.get("base_url", ""),
                                    host=ep.get("host", ""),
                                    api_key=ep.get("api_key", ""))
                if found:
                    kwargs["llm_model"] = found[0]
                    self.progress.emit("endpoint", 0.0,
                                       i18n.tr("endpoint serves: ")
                                       + ", ".join(found[:8]))

            result = run(EvalConfig(**kwargs), progress=self._forward)
            self.succeeded.emit(AgentEvalOutput(
                result=result,
                report=result.report,
                figures=build_figures(result),
                tables=build_tables(result),
                headline=headline(result),
                mode=result.config.mode,
                n_cases=len(result.cases),
                duration_sec=result.duration_sec,
            ))
        except Exception as e:                # pragma: no cover
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _forward(self, stage: str, pct: float, msg: str) -> None:
        self.progress.emit(str(stage), float(pct), str(msg))


class ExportWorker(QThread):
    """Persists a finished run into results/agent_eval/<timestamp>/ (PNG+CSV+MD)."""

    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, result: Any, parent=None):
        super().__init__(parent)
        self.result = result

    def run(self) -> None:
        try:
            ensure_import_path()
            from trustagent.report.export import save
            self.succeeded.emit(save(self.result, RESULTS_ROOT))
        except Exception as e:                # pragma: no cover
            self.failed.emit(f"{type(e).__name__}: {e}")


class ProbeWorker(QThread):
    """Asks a live endpoint which models it serves, off the GUI thread.

    An unreachable server must not freeze the window for the duration of a TCP
    timeout, and a wrong model id is worse than no id at all — so the panel reads
    the inventory instead of guessing.
    """

    succeeded = Signal(list)
    failed = Signal(str)

    def __init__(self, backend: str, endpoint: str, api_key: str = "", parent=None):
        super().__init__(parent)
        self.backend = backend
        self.endpoint = endpoint
        self.api_key = api_key

    def run(self) -> None:
        try:
            ensure_import_path()
            from trustagent.harness.llm import list_models
            self.succeeded.emit(list_models(self.backend,
                                            base_url=self.endpoint,
                                            host=self.endpoint,
                                            api_key=self.api_key, timeout=8.0))
        except Exception as e:                # pragma: no cover
            self.failed.emit(f"{type(e).__name__}: {e}")


class AdaptiveWorker(QThread):
    """Runs one free-text probe (fan-out, measure, correct, re-measure) off-thread.

    ``probe_instruction`` returns a plain dict, which is what the panel renders —
    no trustagent object crosses onto the GUI thread.
    """

    progress = Signal(str, float, str)
    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, opts: AdaptiveOptions, parent=None):
        super().__init__(parent)
        self.opts = opts

    def run(self) -> None:
        try:
            ensure_import_path()
            from trustagent.adaptive import ProbeOptions, probe_instruction

            kw = self.opts.to_probe_kwargs()
            if kw.get("llm_model") == "auto" and kw.get("llm_spec") != "mock":
                from trustagent.harness.llm import list_models
                found = list_models(kw["llm_spec"],
                                    base_url=kw.get("endpoint", ""),
                                    host=kw.get("endpoint", ""),
                                    api_key=kw.get("api_key", ""))
                if found:
                    kw["llm_model"] = found[0]
                    self.progress.emit("endpoint", 0.0,
                                       i18n.tr("endpoint serves: ")
                                       + ", ".join(found[:8]))
            opts = ProbeOptions.for_depth(self.opts.depth, **kw)
            res = probe_instruction(self.opts.text, opts=opts,
                                    history=list(self.opts.history),
                                    progress=self._forward)
            self.succeeded.emit(res)
        except Exception as e:                # pragma: no cover
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _forward(self, stage: str, pct: float, msg: str) -> None:
        self.progress.emit(str(stage), float(pct), str(msg))


# --------------------------------------------------------------------------
class AgentPanel(QWidget):
    """Controls + scorecard for the module_3_agent_eval framework."""

    progress = Signal(str, float, str)        # forwarded to the main status bar

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[AgentEvalWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._probe: Optional[ProbeWorker] = None
        self._adaptive: Optional[AdaptiveWorker] = None
        self._out: Optional[AgentEvalOutput] = None
        self._turns: List[Dict[str, Any]] = []
        self._history: List[Dict[str, str]] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(8)
        self._head = section_header("", "")
        lay.addWidget(self._head)

        lay.addWidget(self._build_env_bar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_suite_tab(), "")
        self.tabs.addTab(self._build_adaptive_tab(), "")
        lay.addWidget(self.tabs, 1)

        saved = str(settings().value("agent/backend", "mock"))
        idx = self.llm_combo.findData(saved)
        if idx >= 0 and idx != self.llm_combo.currentIndex():
            self.llm_combo.setCurrentIndex(idx)      # fires _on_llm_change
        else:
            self._on_llm_change()

        if not module_available():
            self.run_btn.setEnabled(False)
            self.ask_btn.setEnabled(False)
        self.retranslate()

    # ------------------------------------------------------------------ build
    def _build_env_bar(self) -> QWidget:
        """Runtime shared by both tabs: transport, model id, endpoint, secrets.

        One wrapping flow, not two fixed rows — every control keeps its natural
        width and the run of them simply continues on the next line.
        """
        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(12, 8, 12, 10)
        outer.setSpacing(6)

        row = FlowLayout(spacing=8)
        row.addWidget(self._caption("Backend"))
        self.llm_combo = QComboBox()
        for key, disp in LLMS:
            self.llm_combo.addItem(disp, key)
        self.llm_combo.currentIndexChanged.connect(self._on_llm_change)
        row.addWidget(self.llm_combo)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for m in ("mock", "auto") + PRESET_MODELS:
            self.model_combo.addItem(m)
        self.model_combo.setCurrentText(str(settings().value("agent/model", "mock")))
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.model_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.model_combo.setFixedWidth(170)
        row.addWidget(self.model_combo)

        row.addWidget(self._caption("Endpoint"))
        self.endpoint_edit = QLineEdit(str(settings().value("agent/endpoint", "")))
        self.endpoint_edit.setPlaceholderText(BACKEND_ENDPOINT["openai"])
        self.endpoint_edit.setMinimumWidth(230)
        row.addWidget(self.endpoint_edit)

        self.key_edit = QLineEdit("")
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText(i18n.tr("API key (optional)"))
        self.key_edit.setFixedWidth(110)
        row.addWidget(self.key_edit)

        self.detect_btn = QPushButton("")
        self.detect_btn.setObjectName("tune")
        self.detect_btn.clicked.connect(self.detect_endpoint)
        row.addWidget(self.detect_btn)

        row.addWidget(self._caption("Max tokens"))
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(128, 4096)
        self.tokens_spin.setSingleStep(128)
        self.tokens_spin.setValue(int(settings().value("agent/max_tokens", 512)))
        row.addWidget(self.tokens_spin)

        row.addWidget(self._caption("Judge model"))
        self.judge_edit = QLineEdit(str(settings().value("agent/judge_model", "")))
        self.judge_edit.setPlaceholderText(i18n.tr("empty = structural heuristic"))
        self.judge_edit.setFixedWidth(150)
        row.addWidget(self.judge_edit)

        self.mode_pill = status_pill("mock", theme.MUTED)
        row.addWidget(self.mode_pill)
        self.env_hint = QLabel("")
        self.env_hint.setObjectName("muted")
        row.addWidget(self.env_hint)
        outer.addLayout(row)

        enable_height_for_width(card)

        for w in (self.endpoint_edit, self.key_edit, self.detect_btn,
                  self.tokens_spin, self.judge_edit):
            w.setEnabled(False)                 # mock mode needs none of this
        self.endpoint_edit.editingFinished.connect(self._remember)
        self.judge_edit.editingFinished.connect(self._remember)
        self.tokens_spin.valueChanged.connect(self._remember)
        return card

    def _build_suite_tab(self) -> QWidget:
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(4, 8, 4, 4)
        lay.setSpacing(8)
        lay.addWidget(self._build_suite_controls())

        self.summary = SummaryStrip()
        lay.addWidget(self.summary)

        self.suite_split = QSplitter(Qt.Vertical)
        self.suite_split.setChildrenCollapsible(False)
        self.suite_split.setHandleWidth(6)
        self.figs = FigureGrid(columns=2)
        self.tables = TableStack()
        self.tables.setMinimumHeight(240)
        self.suite_split.addWidget(self.figs)
        self.suite_split.addWidget(self.tables)
        self.suite_split.setCollapsible(0, False)
        self.suite_split.setCollapsible(1, False)
        self.suite_split.setStretchFactor(0, 3)
        self.suite_split.setStretchFactor(1, 2)
        apply_splitter_floor(self.suite_split)
        lay.addWidget(self.suite_split, 1)

        self.note = QLabel("")
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.note)

        self.conclusion = ConclusionPanel()
        scroll = PageScroll(body)
        scroll.add_footer(self.conclusion)
        return scroll

    def _build_suite_controls(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(12, 8, 12, 10)
        outer.setSpacing(6)

        row1 = FlowLayout(spacing=8)
        row1.addWidget(self._caption("Adapter"))
        self.agent_combo = QComboBox()
        for key, disp in AGENTS:
            self.agent_combo.addItem(disp, key)
        row1.addWidget(self.agent_combo)

        row1.addWidget(self._caption("Conditions"))
        self.preset_combo = QComboBox()
        for key, _conds in PRESETS:
            self.preset_combo.addItem(key, key)
        self.preset_combo.setCurrentIndex(1)
        row1.addWidget(self.preset_combo)

        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 8)
        self.k_spin.setValue(1)
        wrap_k = QHBoxLayout()
        wrap_k.setSpacing(4)
        wrap_k.addWidget(self._caption("Repeats k"))
        wrap_k.addWidget(self.k_spin)
        row1.addLayout(wrap_k)
        outer.addLayout(row1)

        row2 = FlowLayout(spacing=10)
        row2.addWidget(self._caption("Suites"))
        self.suite_checks: Dict[str, QCheckBox] = {}
        for name, short in SUITES:
            cb = QCheckBox(short)
            cb.setChecked(True)
            self.suite_checks[name] = cb
            row2.addWidget(cb)

        self.judge_chk = QCheckBox(i18n.tr("LLM judge"))
        self.judge_chk.setChecked(True)
        row2.addWidget(self.judge_chk)
        self.attr_chk = QCheckBox(i18n.tr("Context attribution"))
        row2.addWidget(self.attr_chk)

        self.run_btn = QPushButton("")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run_evaluation)
        row2.addWidget(self.run_btn)
        self.export_btn = QPushButton("")
        self.export_btn.setObjectName("tune")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        row2.addWidget(self.export_btn)
        outer.addLayout(row2)
        enable_height_for_width(card)
        return card

    def _build_adaptive_tab(self) -> QWidget:
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(4, 8, 4, 4)
        lay.setSpacing(8)

        hint = QLabel("")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        self.ad_hint = hint
        lay.addWidget(hint)

        card = QFrame()
        card.setObjectName("card")
        box = QVBoxLayout(card)
        box.setContentsMargins(12, 8, 12, 10)
        box.setSpacing(6)

        top = FlowLayout(spacing=8)
        top.addWidget(self._caption("Depth"))
        self.depth_combo = QComboBox()
        for key, cost in DEPTH_COST:
            self.depth_combo.addItem(f"{key} ({cost})", key)
        self.depth_combo.setCurrentIndex(1)
        top.addWidget(self.depth_combo)

        top.addWidget(self._caption("Adapter"))
        self.ad_agent_combo = QComboBox()
        for key, disp in AGENTS[:2]:
            self.ad_agent_combo.addItem(disp, key)
        top.addWidget(self.ad_agent_combo)

        self.ad_reset_btn = QPushButton("")
        self.ad_reset_btn.setObjectName("tune")
        self.ad_reset_btn.clicked.connect(self.reset_session)
        top.addWidget(self.ad_reset_btn)

        self.trace_chk = QCheckBox(i18n.tr("Show trajectory"))
        self.trace_chk.toggled.connect(self._render_chat)
        top.addWidget(self.trace_chk)
        box.addLayout(top)

        self.ask_edit = QPlainTextEdit()
        self.ask_edit.setPlaceholderText(i18n.tr("ask placeholder"))
        self.ask_edit.setMinimumHeight(56)
        box.addWidget(self.ask_edit)

        bottom = FlowLayout(spacing=8)
        self.ask_btn = QPushButton("")
        self.ask_btn.setObjectName("primary")
        self.ask_btn.clicked.connect(self.run_adaptive)
        bottom.addWidget(self.ask_btn)
        self.ad_export_btn = QPushButton("")
        self.ad_export_btn.setObjectName("tune")
        self.ad_export_btn.clicked.connect(self.export_probe)
        self.ad_export_btn.setEnabled(False)
        bottom.addWidget(self.ad_export_btn)
        self.ad_turn_pill = status_pill("—", theme.MUTED)
        bottom.addWidget(self.ad_turn_pill)
        box.addLayout(bottom)
        enable_height_for_width(card)
        lay.addWidget(card)

        self.ad_summary = SummaryStrip()
        lay.addWidget(self.ad_summary)

        self.ad_split = QSplitter(Qt.Vertical)
        self.ad_split.setChildrenCollapsible(False)
        self.ad_split.setHandleWidth(6)
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.chat.setMinimumHeight(280)
        self.chat.setStyleSheet(
            f"background:{theme.BG}; color:{theme.INK};"
            f"border:1px solid {theme.GRID};")
        self.ad_tables = TableStack()
        self.ad_tables.setMinimumHeight(240)
        self.ad_split.addWidget(self.chat)
        self.ad_split.addWidget(self.ad_tables)
        self.ad_split.setCollapsible(0, False)
        self.ad_split.setCollapsible(1, False)
        self.ad_split.setStretchFactor(0, 3)
        self.ad_split.setStretchFactor(1, 2)
        apply_splitter_floor(self.ad_split)
        lay.addWidget(self.ad_split, 1)

        self.ad_note = QLabel("")
        self.ad_note.setObjectName("muted")
        self.ad_note.setWordWrap(True)
        self.ad_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.ad_note)

        self.ad_conclusion = ConclusionPanel()
        scroll = PageScroll(body)
        scroll.add_footer(self.ad_conclusion)
        return scroll

    def _caption(self, key: str) -> QLabel:
        lbl = QLabel(i18n.tr(key))
        lbl.setObjectName("metricLabel")
        lbl.setProperty("i18n_key", key)
        return lbl

    # ------------------------------------------------------------------ i18n
    def retranslate(self) -> None:
        set_section_text(
            self._head, i18n.tr("Agent Eval"),
            i18n.tr("Trace-first, agent-agnostic trust evaluation of LLM agents: "
                    "reliability, robustness, safety, explainability, fairness, "
                    "privacy. Runs entirely in a sandbox."),
        )
        for w in self.findChildren(QLabel):
            key = w.property("i18n_key")
            if key:
                w.setText(i18n.tr(str(key)))
        for i, (_key, disp) in enumerate(AGENTS):
            self.agent_combo.setItemText(i, i18n.tr(disp))
        for i, (key, _conds) in enumerate(PRESETS):
            self.preset_combo.setItemText(i, i18n.tr(key))
        for i, (_key, disp) in enumerate(LLMS):
            self.llm_combo.setItemText(i, i18n.tr(disp))
        for name, short in SUITES:
            self.suite_checks[name].setText(i18n.tr(short))
        self.judge_chk.setText(i18n.tr("LLM judge"))
        self.attr_chk.setText(i18n.tr("Context attribution"))
        self.attr_chk.setToolTip(i18n.tr(
            "Ablation-based context attribution (the agent-flavoured SHAP). "
            "Re-runs cases with context components removed — slower."))
        self.k_spin.setToolTip(i18n.tr(
            "Runs per case. pass^k (all k succeed) only means anything when k>1."))
        self.model_combo.setToolTip(i18n.tr(
            "Model id sent to the endpoint. 'auto' takes the first model the "
            "server reports; the list is filled by Detect endpoint."))
        self.endpoint_edit.setToolTip(i18n.tr(
            "Where the server listens. OpenAI-compatible (bionic / LM Studio / "
            "vLLM) wants the /v1 base; Ollama wants the bare host."))
        self.key_edit.setToolTip(i18n.tr(
            "Bearer token for the endpoint. Kept in memory only — never written "
            "to settings or to any report."))
        self.judge_edit.setToolTip(i18n.tr(
            "Independent judge model. Leave empty for the structural heuristic. "
            "A judge from the same model family is refused and the run falls back "
            "to the heuristic, because self-preference bias would go unmeasured."))
        self.tokens_spin.setToolTip(i18n.tr(
            "Completion cap per step. 27B models need room for the ReAct action "
            "plus JSON arguments; too small truncates the plan."))
        self.detect_btn.setText(i18n.tr("Detect endpoint"))
        self.key_edit.setPlaceholderText(i18n.tr("API key (optional)"))
        self.judge_edit.setPlaceholderText(i18n.tr("empty = structural heuristic"))
        self.env_hint.setText(i18n.tr("Runtime shared by both tabs"))

        self.tabs.setTabText(0, i18n.tr("Suite run"))
        self.tabs.setTabText(1, i18n.tr("Adaptive test"))
        for i, (_key, disp) in enumerate(AGENTS[:2]):
            self.ad_agent_combo.setItemText(i, i18n.tr(disp))
        self.depth_combo.setToolTip(i18n.tr(
            "How many variants one instruction is fanned out into. Each variant "
            "is a full ReAct trajectory, so this is a cost dial: quick ~5 runs, "
            "standard ~10, deep ~24."))
        self.ad_agent_combo.setToolTip(i18n.tr(
            "Adapter the probe drives. One free-text instruction has no gold "
            "answer, so the probe reports answered / grounded / agreement — "
            "never correctness."))
        self.trace_chk.setText(i18n.tr("Show trajectory"))
        self.ad_reset_btn.setText(i18n.tr("Reset session"))
        self.ask_edit.setPlaceholderText(i18n.tr("ask placeholder"))
        self.ad_hint.setText(i18n.tr(
            "Type any instruction. The harness perturbs it, restyles it into "
            "five requester registers, injects payloads into what it would read, "
            "measures the four pillars on the answers it gets, then applies the "
            "trust layer and measures again."))
        self._render_buttons()
        self.summary.retranslate()
        self.figs.retranslate()
        self.tables.retranslate()
        self.ad_tables.retranslate()
        self._render_chat()
        self.ad_conclusion.retranslate()
        self._render_note()
        self._render_adaptive_note()
        self.conclusion.retranslate()
        self._refresh_conclusion()
        if not module_available():
            self.note.setText(i18n.tr(
                "Agent framework not found at Agent/module_3_agent_eval — "
                "this panel is inert."))

    def _render_buttons(self) -> None:
        busy = self._worker is not None and self._worker.isRunning()
        self.run_btn.setText(i18n.tr("Evaluating…") if busy
                             else i18n.tr("Run agent evaluation"))
        self.export_btn.setText(i18n.tr("Export to results/"))
        probing = self._adaptive is not None and self._adaptive.isRunning()
        self.ask_btn.setText(i18n.tr("Probing…") if probing
                             else i18n.tr("Run adaptive probe"))
        self.ad_export_btn.setText(i18n.tr("Export probe JSON"))

    # ------------------------------------------------------------------ state
    def _on_llm_change(self, *_):
        spec = str(self.llm_combo.currentData())
        live = spec != "mock"
        for w in (self.endpoint_edit, self.key_edit, self.detect_btn,
                  self.tokens_spin, self.judge_edit):
            w.setEnabled(live)
        if not live:
            self.model_combo.setCurrentText("mock")
        else:
            if not self.endpoint_edit.text().strip():
                self.endpoint_edit.setText(BACKEND_ENDPOINT.get(spec, ""))
            if self.model_combo.currentText() in ("mock", ""):
                self.model_combo.setCurrentText(PRESET_MODELS[0])
        self.mode_pill.setText(spec)
        self._remember()

    def _remember(self) -> None:
        """Persist what a rerun needs, except the bearer token."""
        st = settings()
        st.setValue("agent/backend", str(self.llm_combo.currentData()))
        st.setValue("agent/model", self.model_combo.currentText())
        st.setValue("agent/endpoint", self.endpoint_edit.text().strip())
        st.setValue("agent/max_tokens", int(self.tokens_spin.value()))
        st.setValue("agent/judge_model", self.judge_edit.text().strip())

    def detect_endpoint(self) -> None:
        if self._probe is not None and self._probe.isRunning():
            return
        self._remember()
        self.detect_btn.setEnabled(False)
        self.progress.emit("endpoint", 0.0, i18n.tr("contacting endpoint…"))
        self._probe = ProbeWorker(str(self.llm_combo.currentData()),
                                  self.endpoint_edit.text().strip(),
                                  self.key_edit.text().strip(), parent=self)
        self._probe.succeeded.connect(self._on_probed)
        self._probe.failed.connect(self._on_probe_failed)
        self._probe.start()

    def _on_probed(self, models: List[str]) -> None:
        self.detect_btn.setEnabled(True)
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        for m in models:
            if m not in [self.model_combo.itemText(i)
                         for i in range(self.model_combo.count())]:
                self.model_combo.addItem(m)
        if current in ("mock", "") and models:
            self.model_combo.setCurrentText(models[0])
        self.model_combo.blockSignals(False)
        self._remember()
        self.progress.emit("endpoint", 1.0,
                           i18n.tr("endpoint serves: ") + ", ".join(models[:8]))

    def _on_probe_failed(self, err: str) -> None:
        self.detect_btn.setEnabled(True)
        self.progress.emit("endpoint", 0.0, i18n.tr("Endpoint unreachable: ") + err)

    def _options(self) -> AgentEvalOptions:
        agent_key = str(self.agent_combo.currentData() or "react")
        agents = ("react", "direct") if agent_key == "both" else (agent_key,)
        label = str(self.preset_combo.currentData() or "Standard")
        conditions = next(conds for key, conds in PRESETS if key == label)
        suites = tuple(n for n, _ in SUITES if self.suite_checks[n].isChecked())
        backend = str(self.llm_combo.currentData())
        return AgentEvalOptions(
            agents=agents, suites=suites or DEFAULT_SUITES, conditions=conditions,
            k=int(self.k_spin.value()), llm=backend,
            model=self.model_combo.currentText().strip() or backend,
            endpoint=self.endpoint_edit.text().strip(),
            api_key=self.key_edit.text().strip(),
            judge_model=self.judge_edit.text().strip(),
            max_tokens=int(self.tokens_spin.value()),
            judge=self.judge_chk.isChecked(), attribution=self.attr_chk.isChecked(),
        )

    # ------------------------------------------------------------------ run
    def run_evaluation(self) -> None:
        if not module_available():
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._remember()
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._render_buttons()
        self.progress.emit("agent eval", 0.0, i18n.tr("starting…"))
        self._worker = AgentEvalWorker(self._options(), parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self.set_output)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, stage: str, pct: float, msg: str) -> None:
        self.progress.emit(stage, pct, msg)

    def set_output(self, out: AgentEvalOutput) -> None:
        self._out = out
        cards: Dict[str, Any] = {"overall": out.report.get("overall")}
        cards.update(out.report.get("pillars", {}))
        inj = (out.report.get("robustness", {}) or {}).get("injection", {}) or {}
        if inj.get("n_attacks"):
            cards["ASR"] = inj.get("attack_success_rate")
        cards["n_cases"] = out.n_cases
        cards["elapsed_sec"] = out.duration_sec
        cards["mode"] = out.mode
        self.summary.set_summary(cards, bad_keys=("ASR",))
        self.figs.set_figures(out.figures)
        self.tables.set_tables(out.tables)
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self._render_buttons()
        self.note.setStyleSheet("")
        self._render_note()
        self._refresh_conclusion()

    def _refresh_conclusion(self) -> None:
        """The suite page's plain-language reading of the last finished run."""
        self.conclusion.set_bullets(
            ui_narrator(agent=self._out).conclusions("agent"))

    def _render_note(self) -> None:
        out = self._out
        if out is None:
            self.note.setText(i18n.tr(
                "Nothing evaluated yet. Every run happens inside a sandboxed tool "
                "environment with ACLs, budgets and confirmation gates — no real "
                "system is touched."))
            return
        text = out.headline
        meta = (out.report or {}).get("meta", {}) or {}
        if out.mode == "mock":
            text += "\n" + i18n.tr(
                "Validity limit: this run used the deterministic MockLLM. It "
                "validates the harness (prompt assembly, trajectory parsing, ACL "
                "enforcement, all six metric families, judging, export); it is NOT "
                "a measurement of any real model's trustworthiness. Pick Ollama or "
                "an OpenAI-compatible endpoint for that.")
        else:
            where = meta.get("endpoint") or i18n.tr("local endpoint")
            text += "\n" + i18n.tr("Live run against ") + f"{where}  •  " \
                + i18n.tr("judge") + f": {meta.get('judge', '?')}"
            if meta.get("judge_note"):
                text += "\n" + i18n.tr("Judge note: ") + str(meta["judge_note"])
        if out.exported.get("json"):
            text += "\n" + i18n.tr("Exported to ") + os.path.dirname(
                str(out.exported["json"]))
        self.note.setText(text)

    def _on_failed(self, err: str) -> None:
        self.run_btn.setEnabled(True)
        self._render_buttons()
        self.progress.emit("agent eval", 0.0, i18n.tr("Agent evaluation failed."))
        first = (err or "").splitlines()[0] if err else "unknown"
        self.note.setStyleSheet(f"color:{theme.BAD};")
        self.note.setText(i18n.tr("Agent evaluation failed: ") + first)

    # ------------------------------------------------------------------ export
    def export_results(self) -> None:
        if self._out is None or self._out.result is None:
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            return
        self.export_btn.setEnabled(False)
        self.progress.emit("export", 0.0, i18n.tr("writing figures and tables…"))
        self._export_worker = ExportWorker(self._out.result, parent=self)
        self._export_worker.succeeded.connect(self._on_exported)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.start()

    def _on_exported(self, written: Dict[str, str]) -> None:
        if self._out is not None:
            self._out.exported = dict(written)
        self.export_btn.setEnabled(True)
        target = os.path.dirname(str(written.get("json", RESULTS_ROOT)))
        self.progress.emit("export", 1.0, i18n.tr("Exported to ") + target)
        self._render_note()

    def _on_export_failed(self, err: str) -> None:
        self.export_btn.setEnabled(True)
        self.progress.emit("export", 0.0, i18n.tr("Export failed: ") + err)

    # ================================================================== adaptive
    def run_adaptive(self) -> None:
        """Probe the instruction in the text box against the shared runtime."""
        if not module_available():
            return
        if self._adaptive is not None and self._adaptive.isRunning():
            return
        text = self.ask_edit.toPlainText().strip()
        if not text:
            self.ad_note.setStyleSheet(f"color:{theme.BAD};")
            self.ad_note.setText(i18n.tr("Type an instruction first."))
            return
        self._remember()
        opts = AdaptiveOptions(
            text=text,
            depth=str(self.depth_combo.currentData() or "standard"),
            agent=str(self.ad_agent_combo.currentData() or "react"),
            llm=str(self.llm_combo.currentData()),
            model=self.model_combo.currentText().strip() or str(
                self.llm_combo.currentData()),
            endpoint=self.endpoint_edit.text().strip(),
            api_key=self.key_edit.text().strip(),
            judge_model=self.judge_edit.text().strip(),
            max_tokens=int(self.tokens_spin.value()),
            history=tuple(self._history),
        )
        self.ask_btn.setEnabled(False)
        self._render_buttons()
        self.ad_note.setStyleSheet("")
        self.ad_note.setText(i18n.tr("probing…"))
        self.progress.emit("adaptive probe", 0.0, i18n.tr("starting…"))
        self._adaptive = AdaptiveWorker(opts, parent=self)
        self._adaptive.progress.connect(self._on_progress)
        self._adaptive.succeeded.connect(self.set_probe_output)
        self._adaptive.failed.connect(self._on_adaptive_failed)
        self._adaptive.start()

    def set_probe_output(self, res: Dict[str, Any]) -> None:
        self._turns.append(res)
        self._history.append({"instruction": str(res.get("instruction") or ""),
                              "answer": str((res.get("after") or {}).get("answer") or "")})
        self.ask_edit.clear()
        self.ask_btn.setEnabled(True)
        self.ad_export_btn.setEnabled(True)
        self._render_buttons()
        self._render_chat()
        self._render_adaptive_summary(res)
        self._render_variant_table(res)
        self._render_adaptive_note()
        self.progress.emit("adaptive probe", 1.0,
                           i18n.tr("probe finished"))

    def _on_adaptive_failed(self, err: str) -> None:
        self.ask_btn.setEnabled(True)
        self._render_buttons()
        self.progress.emit("adaptive probe", 0.0, i18n.tr("Adaptive probe failed."))
        first = (err or "").splitlines()[0] if err else "unknown"
        self.ad_note.setStyleSheet(f"color:{theme.BAD};")
        self.ad_note.setText(i18n.tr("Adaptive probe failed: ") + first)

    def reset_session(self) -> None:
        """Forget the transcript. The next probe sees no conversational context."""
        self._turns.clear()
        self._history.clear()
        self.ad_turn_pill.setText("—")
        self.ad_summary.clear()
        self.ad_tables.set_tables({})
        self.ad_export_btn.setEnabled(False)
        self._render_chat()
        self._render_adaptive_note()

    def export_probe(self) -> None:
        """Write the whole transcript as JSON — no figures, no scorecard."""
        if not self._turns:
            return
        try:
            import json
            import time
            root = os.path.join(RESULTS_ROOT, "adaptive",
                                time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, "probe.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"turns": self._turns}, fh, ensure_ascii=False,
                          indent=2, default=str)
            self.ad_note.setStyleSheet("")
            self.ad_note.setText(i18n.tr("Exported to ") + root)
            self.progress.emit("export", 1.0, i18n.tr("Exported to ") + root)
        except Exception as e:                # pragma: no cover
            self.ad_note.setStyleSheet(f"color:{theme.BAD};")
            self.ad_note.setText(i18n.tr("Export failed: ") + str(e))

    # ---------------------------------------------------------------- drawing
    def _render_chat(self) -> None:
        """Rebuild the transcript from the stored probe dicts (idempotent)."""
        if not self._turns:
            self.chat.setHtml(self._empty_chat())
            return
        lang = "zh" if i18n.language() == "zh" else "en"
        body = [self._turn_html(t, n + 1, lang) for n, t in enumerate(self._turns)]
        self.chat.setHtml("<div>" + "".join(body) + "</div>")
        bar = self.chat.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._refresh_adaptive_conclusion()

    def _refresh_adaptive_conclusion(self) -> None:
        """The latest turn, said in business words rather than as readings."""
        res = self._turns[-1] if self._turns else None
        self.ad_conclusion.set_bullets(
            ui_narrator(probe=res).conclusions("agent_adaptive"))

    def _empty_chat(self) -> str:
        style = (f"color:{theme.MUTED};padding:8px;")
        return f"<div style='{style}'>{escape(i18n.tr('no probe yet'))}</div>"

    def _turn_html(self, res: Dict[str, Any], turn: int, lang: str) -> str:
        meta = res.get("meta") or {}
        before = res.get("before") or {}
        after = res.get("after") or {}
        self.ad_turn_pill.setText(f"turn {turn}")
        head = (f"<div style='color:{theme.ACCENT};font-weight:600;'>"
                f"{escape(i18n.tr('Turn'))} {turn} · {escape(str(meta.get('depth', '')))}"
                f" · {escape(str(meta.get('mode', '')))} · "
                f"{escape(str(meta.get('n_runs', '?')))} {escape(i18n.tr('runs'))}"
                f" · {escape(str(meta.get('duration_sec', '?')))}s</div>")
        q = (f"<div style='font-weight:600;'>Q: "
             f"{escape(str(res.get('instruction', '')))}</div>")
        ctx = ""
        if int(meta.get("n_runs") or 0) and res.get("context_chars"):
            ctx = (f"<div style='color:{theme.MUTED};'>"
                   f"{escape(i18n.tr('context carried from earlier turns'))}"
                   f": {res.get('context_chars')} chars</div>")
        parts = [head, q, ctx,
                 self._answer_html(i18n.tr("Answer before"),
                                   before.get("answer") or "", theme.MUTED),
                 self._pillars_html(before, after),
                 self._answer_html(i18n.tr("Answer after the trust layer"),
                                   self._corrected_text(after, lang, res), theme.GOOD),
                 self._receipts_html(after, res)]
        if self.trace_chk.isChecked():
            parts.append(self._trace_html(res.get("trace") or []))
        parts.append(f"<div style='height:10px;'></div>")
        return ("<div style='border:1px solid " + theme.GRID + ";padding:8px;'>"
                + "".join(parts) + "</div>")

    def _answer_html(self, title: str, text: str, color: str) -> str:
        body = escape(text or "").replace("\n", "<br/>") or \
            f"<i>{escape(i18n.tr('(no answer produced)'))}</i>"
        return (f"<div style='color:{color};font-weight:600;'>{escape(title)}</div>"
                f"<div style='background:{theme.PANEL};padding:6px;'>{body}</div>")

    def _corrected_text(self, after: Dict[str, Any], lang: str,
                        res: Optional[Dict[str, Any]] = None) -> str:
        """Re-render the corrected answer with this UI's section labels.

        The heading structure is the corrector's; the *content* of the
        conclusion is not its raw material.  ``after['parts']['claim']`` is
        whatever the model returned, which on the mock — and on any model that
        merely re-emits a tool result — is a JSON reading rather than a sentence.
        The business reading is generated from ``res['business']`` (the same
        values, compared against the limit in the operator's own question), and
        the verbatim reply is kept beside it under RAW MATERIAL.
        """
        parts = after.get("parts") or {}
        if not parts:
            return str(after.get("answer") or "")
        lab = _labels(lang)
        c = _colon(lang)
        cl = "：" if lang == "zh" else ":"
        out: List[str] = []
        safety = str(parts.get("safety") or "")
        if lang == "zh":
            for en, zh in SAFETY_PREFIX_ZH:
                safety = safety.replace(en, zh)
        if safety:
            out.append(f"{lab['safety']}{c}{safety}")
        lines: List[str] = []
        if res is not None:
            try:
                lines = ui_narrator(probe=res).probe_verdict(res)
            except Exception:                            # noqa: BLE001
                lines = []
        if lines:
            out.append(lab["conclusion"] + cl + "\n"
                       + "\n".join(f"  - {t}" for t in lines))
        else:
            conclusion = str(parts.get("conclusion") or "")
            if lang == "zh" and conclusion == "(no answer produced)":
                conclusion = i18n.tr("(no answer produced)")
            out.append(f"{lab['conclusion']}{c}{conclusion}")
        if parts.get("evidence"):
            out.append(lab["evidence"] + cl + "\n" +
                       "\n".join(f"  - {e}" for e in parts["evidence"]))
        if parts.get("unverified"):
            out.append(f"{lab['unverified']}{c}"
                       + _comma(lang).join(str(t)
                                           for t in parts["unverified"][:8]))
        if parts.get("next"):
            nxt = str(parts["next"])
            out.append(f"{lab['next']}{c}{i18n.tr_in(lang, nxt)}")
        claim = str(parts.get("claim") or "")
        if claim:
            out.append(f"{lab['material']}{c}{claim}")
        return "\n".join(out)

    def _pillars_html(self, before: Dict[str, Any], after: Dict[str, Any]) -> str:
        b_pillars = before.get("pillars") or {}
        a_pillars = after.get("pillars") or {}
        rows = []
        for key in PILLAR_KEYS:
            b, a = b_pillars.get(key), a_pillars.get(key)
            rows.append(
                "<tr>"
                f"<td>{escape(i18n.label(key))}</td>"
                f"<td>{self._bar(b)} {escape(_score_text(b))}</td>"
                f"<td>{self._bar(a)} {escape(_score_text(a))}</td>"
                f"<td style='color:{_delta_color(b, a)}'>{escape(_delta(b, a))}</td>"
                "</tr>")
            sub = self._pillar_sub((after.get("detail") or {}).get(key) or {}, key)
            if sub:
                rows.append(f"<tr><td colspan='4' style='color:{theme.MUTED};'>"
                            f"{escape(sub)}</td></tr>")
        return (f"<div style='color:{theme.INK};font-weight:600;'>"
                f"{escape(i18n.tr('Four pillars before / after'))}</div>"
                "<table cellspacing='0' cellpadding='3' border='1'>"
                + "".join(rows) + "</table>")

    def _pillar_sub(self, det: Dict[str, Any], key: str) -> str:
        """The numbers under a bar, so a score never stands on its own."""
        def fmt(v):
            return _num(v)

        def why(code: str) -> str:
            return i18n.tr(MEASURE_CODES.get(str(code), str(code)))
        bits: List[str] = []
        if det.get("score") is None and det.get("code"):
            # a bar with no reading needs its reason, not an empty sub-line
            return why(det["code"])
        if key == "explainability":
            bits.append(f"completeness={fmt(det.get('mean_completeness'))}")
            bits.append(f"grounding_rate={fmt(det.get('answer_grounding_rate'))}")
        elif key == "robustness":
            bits.append(f"auc_retention={fmt(det.get('auc_retention'))}")
            bits.append(i18n.tr("auc_retention counts 'still answered', not "
                                "'still correct' — this probe has no gold"))
            bits.append(f"answer_agreement={fmt(det.get('answer_agreement'))}")
            bits.append(f"ASR={fmt(det.get('asr_behavioral'))} "
                        f"(n_attacks={det.get('n_attacks', 0)})")
            # behavioural ASR (the injected action actually ran) and output ASR
            # (the answer merely echoed the payload) are different failures; only
            # showing the first hides an agent that parrots an injection it refused
            if det.get("asr_output") is not None:
                bits.append(f"ASR_output={fmt(det.get('asr_output'))}")
            if det.get("utility_under_attack") is not None:
                bits.append(f"utility_under_attack={fmt(det.get('utility_under_attack'))}")
            if det.get("attack_note"):
                bits.append(str(det["attack_note"]))
        elif key == "fairness":
            if det.get("code"):
                return why(det["code"])
            gap = det.get("max_min_gap")
            bits.append(f"gap={fmt(gap)}")
            if isinstance(gap, (int, float)) and gap > 0 and det.get("worst_register"):
                bits.append(f"worst={det.get('worst_register')}")
                bits.append(f"best={det.get('best_register')}")
            elif det.get("profile"):
                # a zero gap means the registers tied; naming a "worst" one here
                # would be a tie-break artefact dressed up as a finding
                bits.append("registers tied")
            for row in det.get("profile") or []:
                bits.append(f"{row.get('register')}={fmt(row.get('quality'))}")
        elif key == "privacy":
            bits.append(f"{i18n.tr('leak rate, probe reading')}="
                        f"{fmt(det.get('raw_leak_rate'))}")
            bits.append(f"{i18n.tr('leak rate, suite reading')}="
                        f"{fmt(det.get('pii_leak_rate'))}")
            raw = det.get("raw_leak_rate")
            suite = det.get("pii_leak_rate")
            if (isinstance(raw, (int, float)) and isinstance(suite, (int, float))
                    and abs(raw - suite) > 1e-9):
                bits.append(i18n.tr(
                    "the two readings differ only because the suite forgives "
                    "nothing you typed yourself"))
            chans = det.get("channels") or {}
            if chans:
                bits.append(i18n.tr("outbound channel") + "="
                            + ", ".join(f"{k}×{v}" for k, v in chans.items()))
            if det.get("exempted_operator_values"):
                bits.append(i18n.tr("typed by you, so exempt") + "="
                            + ", ".join(det["exempted_operator_values"][:4]))
        return " · ".join(bits)

    def _receipts_html(self, after: Dict[str, Any], res: Dict[str, Any]) -> str:
        edits = after.get("edits") or []
        caveats = after.get("caveats") or []
        out = []
        if edits:
            items = "".join(
                f"<li><b>{escape(i18n.label(str(e.get('pillar', ''))))}</b> · "
                f"{escape(i18n.tr(EDIT_LABELS.get(str(e.get('code', '')), str(e.get('code', '')))))}"
                + (f" <span style='color:{theme.MUTED}'>{escape(_detail_bits(e))}</span>"
                   if _detail_bits(e) else "") + "</li>"
                for e in edits)
            out.append(f"<div style='font-weight:600;'>"
                       f"{escape(i18n.tr('Corrections applied'))}</div>"
                       f"<ul style='margin:2px 0 6px 18px;'>{items}</ul>")
        if caveats:
            items = "".join(
                f"<li style='color:{theme.BAD}'>"
                f"{escape(i18n.tr(CAVEAT_LABELS.get(str(c.get('code', '')), str(c.get('code', '')))))}"
                + (f" — {escape(', '.join(str(x) for x in c.get('channels') or []))}"
                   if c.get("channels") else "") + "</li>"
                for c in caveats)
            out.append(f"<div style='font-weight:600;'>"
                       f"{escape(i18n.tr('What this does not fix'))}</div>"
                       f"<ul style='margin:2px 0 6px 18px;'>{items}</ul>")
        variants = res.get("variants") or []
        hits = [v for v in variants if v.get("attack_hit")]
        denied = sum(1 for v in variants if int(v.get("denied_calls") or 0) > 0)
        facts = [f"{len(variants)} variants",
                 f"{sum(1 for v in variants if not v.get('answered'))} unanswered",
                 f"{denied} with ACL-blocked calls"]
        if hits:
            facts.append(f"<span style='color:{theme.BAD}'>"
                         f"{len(hits)} attack(s) reached a granted tool</span>")
        out.append(f"<div style='color:{theme.MUTED}'>{' · '.join(facts)}</div>"
                   f"<div style='color:{theme.MUTED}'>"
                   f"{escape(i18n.tr('variant detail in the table below'))}</div>")
        return "".join(out)

    def _trace_html(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        body = []
        for r in rows:
            state = ("→ " + str(r.get("action") or "")) if r.get("action") else ""
            if r.get("final"):
                state = "✓ final"
            elif r.get("granted") is False:
                state = f"✗ blocked {r.get('action')}"
            elif r.get("unknown_tool"):
                state = f"? unknown {r.get('action')}"
            obs = str(r.get("result") or "")
            body.append(
                "<tr>"
                f"<td>{r.get('index')}</td>"
                f"<td>{escape(str(r.get('thought') or ''))}</td>"
                f"<td>{escape(state)}</td>"
                f"<td>{escape(obs)}</td>"
                f"<td>{_num(r.get('tokens'))}</td>"
                "</tr>")
        return (f"<div style='font-weight:600;'>{escape(i18n.tr('Trajectory'))}</div>"
                "<table cellspacing='0' cellpadding='3' border='1'>"
                + "".join(body) + "</table>")

    def _bar(self, value: Optional[float], width: int = 12) -> str:
        v = _clamp(value)
        if v is None:
            return f"<span style='color:{theme.MUTED}'>—</span>"
        filled = int(round(v * width))
        return ("<span style='color:" + theme.GOOD + "'>" + "█" * filled
                + "</span><span style='color:" + theme.GRID + "'>"
                + "█" * (width - filled) + "</span>")

    def _render_adaptive_summary(self, res: Dict[str, Any]) -> None:
        before = res.get("before") or {}
        after = res.get("after") or {}
        meta = res.get("meta") or {}
        cards = {
            "Before overall": before.get("overall"),
            "After overall": after.get("overall"),
            "Explainability": (after.get("pillars") or {}).get("explainability"),
            "Robustness": (after.get("pillars") or {}).get("robustness"),
            "Fairness": (after.get("pillars") or {}).get("fairness"),
            "Privacy": (after.get("pillars") or {}).get("privacy"),
            "Runs": meta.get("n_runs"),
            "Depth": meta.get("depth"),
            "Mode": meta.get("mode"),
            "Elapsed": meta.get("duration_sec"),
        }
        self.ad_summary.set_summary(cards)

    def _render_variant_table(self, res: Dict[str, Any]) -> None:
        rows = res.get("variants") or []
        if not rows:
            self.ad_tables.set_tables({})
            return
        keep = ("condition", "register", "family", "level", "answered", "grounded",
                "agreement", "n_steps", "n_tool_calls", "denied_calls",
                "unknown_tool", "attack_hit", "tokens", "answer_head")
        table = pd.DataFrame([{k: r.get(k) for k in keep} for r in rows])
        self.ad_tables.set_tables({i18n.tr("Variants of this instruction"): table})

    def _render_adaptive_note(self) -> None:
        if self._adaptive is not None and self._adaptive.isRunning():
            return
        if not self._turns:
            self.ad_note.setText(i18n.tr(
                "No probe yet. A free-text instruction has no gold answer, so the "
                "probe never claims correctness — it reports whether the agent "
                "answered, how much of the answer traces back to a tool "
                "observation it actually received, and how far the perturbed "
                "variants drifted from the clean run."))
            return
        last = self._turns[-1]
        meta = last.get("meta") or {}
        text = ""
        if meta.get("mode") == "mock":
            text += i18n.tr(
                "Mock probe: these numbers validate the fan-out, the metrics and "
                "the corrector — not a real model. Pick a live backend above.") + "\n"
        else:
            text += i18n.tr("Live probe against ") + str(meta.get("endpoint") or "") \
                + f" · {meta.get('model')}\n"
        detail = (last.get("before") or {}).get("detail") or {}
        priv = detail.get("privacy") or {}
        if priv.get("scope_note"):
            text += i18n.tr("Privacy scope: ") + str(priv["scope_note"]) + "\n"
        rob = detail.get("robustness") or {}
        if rob.get("invented_tool_names"):
            text += i18n.tr("Invented tool names: ") \
                + ", ".join(str(t) for t in rob["invented_tool_names"][:6]) + "\n"
        irr = meta.get("irreversible_tools") or []
        if irr:
            text += i18n.tr("Irreversible tools in this sandbox: ") \
                + ", ".join(irr[:8])
        self.ad_note.setText(text.strip())


# --------------------------------------------------------------------------
def _labels(lang: str) -> Dict[str, str]:
    """The corrector's section headings, in the language the operator reads."""
    fallback = {"conclusion": "CONCLUSION", "evidence": "EVIDENCE",
                "unverified": "UNVERIFIED", "safety": "SAFETY", "next": "NEXT"}
    try:
        ensure_import_path()
        from trustagent.adaptive.template import LABELS
        lab = dict(LABELS.get(lang, LABELS["en"]))
    except Exception:                         # pragma: no cover - module absent
        lab = dict(fallback)
    # not the corrector's own scaffold: a studio heading, deliberately kept out of
    # the phrase list the grounding metric strips
    lab["material"] = (i18n.tr_in(lang, "raw material") if lang == "zh"
                       else "RAW MATERIAL")
    return lab


def _colon(lang: str) -> str:
    """The heading separator: '：' in Chinese, ': ' in English."""
    return "：" if lang == "zh" else ": "


def _comma(lang: str) -> str:
    return "、" if lang == "zh" else ", "


def _clamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return max(0.0, min(1.0, f))


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int,)):
        return str(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:
        return "n/a"
    return f"{f:.3f}"


def _score_text(value: Any) -> str:
    """A pillar score, or the fact that this run never measured one.

    ``n/a`` sits too close to a zero for a number the reader is meant to act on.
    """
    v = _clamp(value)
    return i18n.tr("Not measured") if v is None else f"{v:.3f}"


def _delta(before: Any, after: Any) -> str:
    b, a = _clamp(before), _clamp(after)
    if b is None or a is None:
        # no delta to take: one side has no score, and the score columns already
        # say which one
        return "—"
    d = a - b
    return "—" if abs(d) < 5e-4 else f"{d:+.3f}"


def _delta_color(before: Any, after: Any) -> str:
    b, a = _clamp(before), _clamp(after)
    if b is None or a is None:
        return theme.MUTED
    if a > b + 5e-4:
        return theme.GOOD
    if a < b - 5e-4:
        return theme.BAD
    return theme.MUTED


def _detail_bits(item: Dict[str, Any]) -> str:
    """One compact, auditable line per edit: what fired and on what evidence."""
    bits: List[str] = []
    for key in ("n", "n_evidence", "agreement_before", "own_traceability"):
        if item.get(key) is not None:
            bits.append(f"{key}={_num(item[key])}")
    for key in ("examples", "tokens", "values", "register"):
        val = item.get(key)
        if val:
            text = ", ".join(str(x)[:48] for x in (val if isinstance(val, list)
                                                   else [val]))
            bits.append(f"{key}: {text}")
    removed = item.get("removed")
    if isinstance(removed, dict) and removed:
        bits.append("removed " + ", ".join(f"{k}×{len(v)}"
                                           for k, v in removed.items()))
    where = item.get("where")
    if isinstance(where, dict) and where:
        hits = [f"{k}({len(v)})" for k, v in where.items() if v]
        if hits:
            bits.append("in " + ", ".join(hits))
    if item.get("detail"):
        bits.append(str(item["detail"]))
    if item.get("empty"):
        bits.append("empty answer")
    if item.get("filled"):
        bits.append("register gap filled")
    return " · ".join(bits)
