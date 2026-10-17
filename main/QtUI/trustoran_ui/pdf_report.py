"""Automated PDF report generation for the studio (reportlab / platypus).

One generator, three readers. The same evidence — a :class:`trustoran` analysis
run, an ``AgentEvalOutput`` and an O-RAN console snapshot — is laid out
differently depending on who is supposed to read it:

``layperson``  结论先行，隐藏方法与表格，只留下"它有效吗 / 它可信吗 / 什么时候会坏"。
``smo``        运维视角：指标读数、需要盯的门限、以及从告警数据推出的处置建议。
``ai``         AI 视角：估计量与设置（攻击族、ε 扫描、隐私记账）、指标定义与附录表格。

Everything printed here already exists as an object in the panels — nothing is
recomputed, so the PDF can never disagree with the screen. Two rules are
hard-coded rather than configurable: the personal-research disclaimer always
appears, and a mock-mode agent run is always labelled as harness validation.

Figures are rasterised with an off-screen Agg canvas and the panel's canvas is
restored afterwards, because ``Figure.savefig`` would otherwise detach the
``FigureCanvasQTAgg`` the panel is displaying.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as _esc

import pandas as pd

from trustoran.viz import labels as VZ

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image,
                                KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

from . import i18n
from .narrative import (Narrator, _DIM_BASIS, _PLAIN_DIM, _PLAIN_LIMITS,
                        _PLAIN_TITLES, _as_float, _dataset_note, _fmt, _pick)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
RESULTS_ROOT = os.path.join(REPO, "results")

ProgressCB = Optional[Callable[[str, float, str], None]]

# --------------------------------------------------------------------------
# format choices offered by the dialog
# --------------------------------------------------------------------------
PAPERS: Dict[str, Tuple[float, float]] = {
    "A4": (595.27, 841.89),
    "Letter": (612.0, 792.0),
    "Legal": (612.0, 1008.0),
}

#: page margin in inches
MARGINS: Dict[str, float] = {"Narrow": 0.55, "Normal": 0.8, "Wide": 1.15}

FIG_DPI = (96, 130, 150, 200)

#: gutter between two body columns (points)
COL_GAP = 18.0

STYLES: Tuple[str, ...] = ("layperson", "smo", "ai")

#: one English literal per audience, doubling as its i18n key
STYLE_LABEL_KEYS: Dict[str, str] = {
    "layperson": "Non-specialist",
    "smo": "SMO / RIC operations",
    "ai": "AI research",
}

#: (key, English label used as the i18n key) in report order
SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("abstract", "Executive summary"),
    ("data_model", "Dataset & model"),
    ("metrics", "Base performance"),
    ("explainability", "Explainability"),
    ("robustness", "Robustness"),
    ("fairness", "Fairness"),
    ("privacy", "Privacy"),
    ("oran", "O-RAN console snapshot"),
    ("agent", "Agent trust evaluation"),
    ("method", "Methodology & validity limits"),
    ("appendix", "Appendix: result tables"),
)

SECTION_KEYS: Tuple[str, ...] = tuple(k for k, _ in SECTIONS)

_PILLARS = ("explainability", "robustness", "fairness", "privacy")

#: what each reader gets by default
_STYLE_SECTIONS: Dict[str, Tuple[str, ...]] = {
    "layperson": ("abstract", "data_model", "metrics", "explainability",
                  "robustness", "fairness", "privacy", "oran", "agent", "method"),
    "smo": ("abstract", "data_model", "metrics", "explainability", "robustness",
            "fairness", "privacy", "oran", "agent", "method"),
    "ai": SECTION_KEYS,
}

#: figures per pillar section, and whether tables appear in the body
_STYLE_FIGS: Dict[str, int] = {"layperson": 1, "smo": 2, "ai": 99}
_STYLE_BODY_TABLES: Dict[str, bool] = {"layperson": False, "smo": False, "ai": True}


THEMES: Dict[str, Dict[str, str]] = {
    "dark": dict(bg="#0f1720", panel="#16212c", ink="#e6edf3", muted="#8b98a5",
                 accent="#22d3ee", good="#34d399", bad="#fb7185", grid="#243244",
                 head="#1b2836"),
    "light": dict(bg="#ffffff", panel="#f4f7fa", ink="#12202c", muted="#5b6b7a",
                  accent="#0e7490", good="#0f766e", bad="#b91c1c", grid="#cbd5e1",
                  head="#e2e8f0"),
}


# --------------------------------------------------------------------------
# fonts — the report is bilingual, so the Latin runtimes are not enough
# --------------------------------------------------------------------------
_FONT: Optional[str] = None
_FONT_BOLD: Optional[str] = None

_FONT_CANDIDATES = ("msyh.ttc", "msyh.ttf", "Deng.ttf", "NotoSansSC-VF.ttf",
                    "simhei.ttf", "simsun.ttc", "PingFang.ttc")
_BOLD_CANDIDATES = ("msyhbd.ttc", "msyhbd.ttf", "Dengb.ttf")


def _font_paths() -> Tuple[str, Optional[str]]:
    win = os.environ.get("SystemRoot", "C:\\Windows") + os.sep + "Fonts"
    def first(names: Sequence[str]) -> Optional[str]:
        for n in names:
            p = os.path.join(win, n)
            if os.path.isfile(p):
                return p
        return None
    return first(_FONT_CANDIDATES) or "", first(_BOLD_CANDIDATES)


def fonts() -> Tuple[str, str]:
    """Register and return ``(regular, bold)`` font names with CJK coverage."""
    global _FONT, _FONT_BOLD
    if _FONT:
        return _FONT, _FONT_BOLD
    path, bold_path = _font_paths()
    reg = bold = None
    if path:
        try:
            pdfmetrics.registerFont(TTFont("TrustCJK", path, subfontIndex=0))
            reg = "TrustCJK"
        except Exception:
            reg = None
    if bold_path and reg:
        try:
            pdfmetrics.registerFont(TTFont("TrustCJKB", bold_path, subfontIndex=0))
            bold = "TrustCJKB"
        except Exception:
            bold = None
    if reg is None:                       # built-in CID font: no embedding needed
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        reg = "STSong-Light"
    _FONT, _FONT_BOLD = reg, bold or reg
    return _FONT, _FONT_BOLD


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------
@dataclass
class ReportSpec:
    """Everything the user can adjust in the export dialog."""

    style: str = "smo"
    sections: Tuple[str, ...] = field(default_factory=lambda: _STYLE_SECTIONS["smo"])
    language: str = "zh"
    paper: str = "A4"
    columns: int = 1
    margin: str = "Normal"
    dpi: int = 150
    theme: str = "dark"
    cover: bool = True
    header: bool = True
    page_numbers: bool = True
    body_tables: Optional[bool] = None          # None -> the style default
    title: str = ""
    author: str = ""
    date: str = ""
    max_table_rows: int = 24

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            self.style = "smo"
        self.sections = tuple(s for s in self.sections if s in SECTION_KEYS) or \
            _STYLE_SECTIONS[self.style]
        self.language = self.language if self.language in ("zh", "en") else "zh"
        self.paper = self.paper if self.paper in PAPERS else "A4"
        self.margin = self.margin if self.margin in MARGINS else "Normal"
        self.theme = self.theme if self.theme in THEMES else "dark"
        self.columns = 2 if int(self.columns) >= 2 else 1
        self.dpi = int(self.dpi) if 60 <= int(self.dpi) <= 400 else 150

    @property
    def show_body_tables(self) -> bool:
        return (_STYLE_BODY_TABLES[self.style]
                if self.body_tables is None else bool(self.body_tables))

    def has(self, key: str) -> bool:
        return key in self.sections

    @property
    def style_label_key(self) -> str:
        return STYLE_LABEL_KEYS[self.style]

    @property
    def label(self) -> str:
        """Audience name in the *UI* language (the dialog shows this)."""
        return i18n.tr(self.style_label_key)


def sections_for(style: str) -> Tuple[str, ...]:
    return _STYLE_SECTIONS.get(style, _STYLE_SECTIONS["smo"])


def default_out_path(style: str = "smo") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULTS_ROOT, "reports", f"trustoran_{style}_{stamp}.pdf")


# --------------------------------------------------------------------------
# small converters
# --------------------------------------------------------------------------
def figure_to_png(fig: Any, dpi: int) -> Optional[bytes]:
    """Rasterise a panel ``Figure`` without leaving its Qt canvas detached."""
    if fig is None:
        return None
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception:                                   # pragma: no cover
        return None
    old = getattr(fig, "canvas", None)
    try:
        FigureCanvasAgg(fig)            # re-points fig.canvas at an Agg canvas
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=int(dpi),
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        data = buf.getvalue()
        return data or None
    except Exception:                                   # pragma: no cover
        return None
    finally:
        if old is not None:
            try:
                fig.canvas = old
            except Exception:                           # pragma: no cover
                pass


def png_size(data: bytes) -> Tuple[int, int]:
    """(w, h) in pixels, read straight from the PNG header (no Pillow needed)."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return (640, 420)
    return (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"))


def _fmt_cell(v: Any, limit: int = 34) -> str:
    text = _fmt(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


# --------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------
class _Builder(Narrator):
    def __init__(self, spec: ReportSpec):
        super().__init__(lang=spec.language, style=spec.style)
        self.spec = spec
        self.font, self.font_bold = fonts()
        pal = THEMES[spec.theme]
        self.c = {k: colors.HexColor(v) for k, v in pal.items()}
        self.styles = self._make_styles()
        pw, ph = PAPERS[spec.paper]
        m = MARGINS[spec.margin] * inch
        self.pagesize = (pw, ph)
        self.margins = m
        inner_w = pw - 2 * m
        self.avail = inner_w if spec.columns == 1 else (inner_w - COL_GAP) / 2.0
        #: a figure taller than the frame would abort the build
        self.max_fig_h = ph - 2 * m - 60
        self._fig_cache: Dict[int, Optional[bytes]] = {}

    # -- plumbing ---------------------------------------------------------


    def _make_styles(self) -> Dict[str, ParagraphStyle]:
        c, f, fb = self.c, self.font, self.font_bold
        wrap = "CJK" if self.lang == "zh" else None
        kw = dict(wordWrap=wrap) if wrap else {}
        s = {
            "title": ParagraphStyle("title", fontName=fb, fontSize=25, leading=32,
                                    textColor=c["ink"], alignment=TA_LEFT, **kw),
            "sub": ParagraphStyle("sub", fontName=f, fontSize=12.5, leading=18,
                                  textColor=c["accent"], **kw),
            "h1": ParagraphStyle("h1", fontName=fb, fontSize=15.5, leading=20,
                                 textColor=c["ink"], spaceBefore=10,
                                 spaceAfter=4, **kw),
            "h2": ParagraphStyle("h2", fontName=fb, fontSize=11.5, leading=15,
                                 textColor=c["accent"], spaceBefore=8,
                                 spaceAfter=3, **kw),
            "body": ParagraphStyle("body", fontName=f, fontSize=9.3, leading=14,
                                   textColor=c["ink"], spaceAfter=4, **kw),
            "bullet": ParagraphStyle("bullet", fontName=f, fontSize=9.3, leading=13.6,
                                     textColor=c["ink"], leftIndent=12,
                                     bulletIndent=2, spaceAfter=2.5, **kw),
            "muted": ParagraphStyle("muted", fontName=f, fontSize=8.4, leading=12,
                                    textColor=c["muted"], spaceAfter=3, **kw),
            "cap": ParagraphStyle("cap", fontName=f, fontSize=8.0, leading=11,
                                  textColor=c["muted"], spaceBefore=1,
                                  spaceAfter=7, **kw),
            "cell": ParagraphStyle("cell", fontName=f, fontSize=8.0, leading=11,
                                   textColor=c["ink"], **kw),
            "cellh": ParagraphStyle("cellh", fontName=fb, fontSize=8.2, leading=11,
                                    textColor=c["accent"], **kw),
            "meta": ParagraphStyle("meta", fontName=f, fontSize=10, leading=14.5,
                                   textColor=c["muted"], **kw),
            "cmd": ParagraphStyle("cmd", fontName=f, fontSize=8.6, leading=12,
                                  textColor=c["good"], leftIndent=8, **kw),
        }
        return s

    def P(self, text: str, style: str = "body") -> Paragraph:
        return Paragraph(_esc(text), self.styles[style])

    def flow(self, *flowables) -> None:
        self.story.extend([f for f in flowables if f is not None])

    # -- page furniture ---------------------------------------------------
    def _bg(self, canv) -> None:
        canv.saveState()
        canv.setFillColor(self.c["bg"])
        canv.rect(0, 0, self.pagesize[0], self.pagesize[1], stroke=0, fill=1)
        canv.restoreState()

    def _on_cover(self, canv, doc) -> None:
        self._bg(canv)
        w, h = self.pagesize
        canv.saveState()
        canv.setFillColor(self.c["accent"])
        canv.rect(0, h - 6, w, 6, stroke=0, fill=1)
        canv.setFillColor(self.c["panel"])
        canv.rect(0, 0, w, 26, stroke=0, fill=1)
        canv.restoreState()

    def _on_body(self, canv, doc) -> None:
        self._bg(canv)
        w, h = self.pagesize
        m = self.margins
        if self.spec.header:
            canv.saveState()
            canv.setStrokeColor(self.c["grid"])
            canv.setLineWidth(0.6)
            canv.line(m, h - m + 12, w - m, h - m + 12)
            canv.setFont(self.font, 7.6)
            canv.setFillColor(self.c["muted"])
            title = self.spec.title or self.L("Trustworthy AI for Open RAN — Analysis Report")
            canv.drawString(m, h - m + 17, _esc(title)[:72])
            canv.drawRightString(w - m, h - m + 17,
                                 _esc(self.L(self.spec.style_label_key)))
            canv.restoreState()
        if self.spec.page_numbers:
            canv.saveState()
            canv.setStrokeColor(self.c["grid"])
            canv.setLineWidth(0.6)
            canv.line(m, m - 14, w - m, m - 14)
            canv.setFont(self.font, 7.6)
            canv.setFillColor(self.c["muted"])
            who = self.spec.author or self.L("Trustworthy AI studio")
            canv.drawString(m, m - 26, _esc(f"{who}  •  "
                                            f"{self.spec.date or ''}  •  "
                                            f"{self.L('Generated by the Trustworthy AI studio')}"))
            canv.drawRightString(w - m, m - 26,
                                 f"{self.L('Page')} {canv.getPageNumber()}")
            canv.restoreState()

    # -- block builders ---------------------------------------------------
    def h1(self, text: str) -> None:
        self.flow(self.P(text, "h1"),
                  HRFlowable(width="100%", thickness=0.8, color=self.c["accent"],
                             spaceBefore=0, spaceAfter=6))

    def h2(self, text: str) -> None:
        self.flow(self.P(text, "h2"))

    def bullets(self, items: Sequence[str], style: str = "bullet") -> None:
        for t in items:
            if t:
                self.flow(Paragraph(f"•&#160;&#160;{_esc(t)}", self.styles[style]))

    def kv(self, mapping: Dict[str, Any], header: Optional[str] = None,
           keep_none: bool = False) -> None:
        rows = [[Paragraph(_esc(self.L("Metric")), self.styles["cellh"]),
                 Paragraph(_esc(self.L("Value")), self.styles["cellh"])]]
        for k, v in mapping.items():
            if v is None:
                # a scorecard row the run could not measure still has to appear:
                # a missing line reads as "that one was fine"
                if not keep_none:
                    continue
                rows.append([Paragraph(_esc(self.lab(k)), self.styles["cell"]),
                             Paragraph(_esc(self.L("Not measured")),
                                       self.styles["cell"])])
                continue
            # A value is usually data (a feature name, a class label), but some are
            # sentences the pipeline stores as-is; L() returns anything undocumented
            # unchanged, so this only translates the strings we deliberately keyed.
            sv = self.L(v) if isinstance(v, str) else v
            rows.append([Paragraph(_esc(self.lab(k)), self.styles["cell"]),
                         Paragraph(_esc(_fmt(sv)), self.styles["cell"])])
        if len(rows) == 1:
            return
        if header:
            self.h2(header)
        self.flow(self._table(rows, [self.avail * 0.38, self.avail * 0.62]))

    def _table(self, rows: List[List[Any]], widths: Sequence[float]) -> Table:
        t = Table(rows, colWidths=list(widths), hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.c["head"]),
            ("GRID", (0, 0), (-1, -1), 0.5, self.c["grid"]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [self.c["bg"], self.c["panel"]]),
        ]))
        return t

    def df(self, name: str, df: Any, max_rows: Optional[int] = None) -> None:
        if df is None or getattr(df, "empty", True):
            return
        rows_total = int(df.shape[0])
        cap = self.spec.max_table_rows if max_rows is None else max_rows
        head = df.head(max(1, min(rows_total, cap)))
        cols = [str(c) for c in head.columns]
        if not cols:
            return
        rows = [[Paragraph(_esc(c), self.styles["cellh"]) for c in cols]]
        for _, r in head.iterrows():
            rows.append([Paragraph(
                _esc(self.lab(str(r[c])) if c == "metric" else _fmt_cell(r[c])),
                self.styles["cell"]) for c in cols])
        widths = [self.avail / len(cols)] * len(cols)
        self.flow(self.P(self._plab(name), "h2"), self._table(rows, widths))
        extra = rows_total - len(head)
        if extra > 0:
            self.flow(self.P(self.L("… (+{n} more rows)", n=extra), "muted"))

    def fig(self, name: str, fig: Any) -> None:
        if fig is None:
            return
        key = (id(fig), self.spec.language)
        if key not in self._fig_cache:
            # the figure belongs to a panel on screen, so borrow it only for the
            # raster and hand it back in the language the studio is showing.
            studio = i18n.language()
            VZ.relabel_figure(fig, self.spec.language)
            try:
                self._fig_cache[key] = figure_to_png(fig, self.spec.dpi)
            finally:
                VZ.relabel_figure(fig, studio)
        png = self._fig_cache[key]
        if not png:
            return
        pw, ph = png_size(png)
        intrinsic_w = pw * 72.0 / float(self.spec.dpi)
        target_w = min(self.avail, max(120.0, intrinsic_w))
        target_h = ph * 72.0 / float(self.spec.dpi) * (target_w / intrinsic_w)
        if target_h > self.max_fig_h:
            target_w *= self.max_fig_h / target_h
            target_h = self.max_fig_h
        try:
            img = Image(io.BytesIO(png), width=target_w, height=target_h)
        except Exception:                               # pragma: no cover
            return
        self.flow(KeepTogether([img, self.P(self._plab(name), "cap")]))

    def figs(self, figures: Optional[Dict[str, Any]], cap: int,
             prefer: Sequence[str] = ()) -> None:
        items = list((figures or {}).items())
        order = [k for k in prefer if any(name == k for name, _ in items)]
        ordered = [(k, dict(items)[k]) for k in order] + \
                  [(n, f) for n, f in items if n not in order]
        for name, fig in ordered[:max(0, cap)]:
            self.fig(name, fig)

    def note(self, text: str) -> None:
        self.flow(self.P(text, "muted"))

    def gap(self, pts: float = 6) -> None:
        self.flow(Spacer(1, pts))

    # ==================================================================
    # evidence accessors — the report never recomputes anything
    # ==================================================================


    # ==================================================================
    # sections
    # ==================================================================
    def sec_abstract(self) -> None:
        self.h1(self._pt("abstract", "Executive summary"))
        r = self.report
        a = self.agent
        if r is None and a is None:
            self.note(self.L("No analysis run yet — run the pipeline to "
                             "populate this report."))
            return
        if self.spec.style == "layperson":
            self._abstract_layperson(r, a)
        elif self.spec.style == "smo":
            self._abstract_smo(r, a)
        else:
            self._abstract_ai(r, a)

    def _headline_pair(self, r) -> Tuple[str, str]:
        m = dict(getattr(r, "base_metrics", {}) or {})
        key = next((k for k in ("accuracy", "r2") if k in m),
                   next(iter(m), ""))
        return self.lab(key), _fmt(m.get(key))

    def _abstract_layperson(self, r, a) -> None:
        self.flow(self.P(self.L("This page answers the questions anybody asks about a "
                                "system like this, in plain words. Every number in it "
                                "is a measurement taken by the run described later in "
                                "this report — nothing here is estimated separately.")))
        if r is not None:
            self.h2(self.L("Does it actually work?"))
            self._plain_flow(self._plain_metrics(r)[:2])
            self.h2(self.L("Can you see why it decided that?"))
            self._plain_flow(self._plain_explainability()[:2])
            self.h2(self.L("What if someone tries to fool it?"))
            self._plain_flow(self._plain_robustness()[:2])
            self.h2(self.L("Does it treat every group alike?"))
            self._plain_flow(self._plain_fairness()[:1])
            self.h2(self.L("Could it give your data away?"))
            self._plain_flow(self._plain_privacy()[-1:])
        if a is not None:
            self.h2(self.L("And how did the agent do?"))
            self._plain_flow(self._plain_agent(a)[:1])
        self.gap(2)
        self.note(self.L("The last section of this report lists what these numbers "
                         "do not prove — worth a read before quoting any of them."))

    def _abstract_smo(self, r, a) -> None:
        self.flow(self.P(self.L("Operational read-out for SMO / Non-RT RIC staff: "
                                "model status, what to watch at the cell edge, and "
                                "which thresholds should trigger an A1 policy "
                                "review.")))
        watch, actions = self._operational_lines()
        if r is not None:
            k, v = self._headline_pair(r)
            rep = self._agent_rep()
            self.kv({k: v,
                     "elapsed_sec": getattr(r, "model_spec", {}).get("elapsed_sec"),
                     "overall": rep.get("overall") if a is not None else None},
                    header=self.L("Key read-out"), keep_none=True)
        if watch:
            self.h2(self.L("Watch items"))
            self.bullets(watch)
        if actions:
            self.h2(self.L("Recommended actions"))
            self.bullets(actions)

    def _operational_lines(self) -> Tuple[List[str], List[str]]:
        """Threshold-triggered watch items and actions, from the run + the console."""
        watch: List[str] = []
        actions: List[str] = []
        rob = self._summary("robustness")
        clean = _as_float(_pick(rob, "clean_accuracy", "clean_r2"))
        hit = _as_float(_pick(rob, "pgd_robust_acc", "deepfool_robust_acc"))
        if clean is not None and hit is not None and clean > hit:
            watch.append(self.L("Adversarial degradation costs {v} accuracy points "
                                "at ε={e}. Keep the Near-RT RIC rate-limit policy "
                                "armed.", v=f"{(clean - hit) * 100:.1f}", e="0.3"))
        fair = self._summary("fairness")
        di = _as_float(_pick(fair, "disparate_impact", "demographic_parity_ratio"))
        if di is not None:
            watch.append(self.L("Disparate impact ratio is {v}. Below 0.8 the A1 "
                                "policy set should be rebalanced.", v=_fmt(di)))
        priv = self._summary("privacy")
        mi = _as_float(_pick(priv, "membership_inference_auc"))
        if mi is not None:
            actions.append(self.L("Membership-inference AUC is {v}. Above 0.65 the "
                                  "detector memorises training rows — retrain with a "
                                  "larger privacy budget or more data.", v=_fmt(mi)))
        expl = self._summary("explainability")
        feats = _pick(expl, "top_features", "shap_top")
        if feats:
            watch.append(self.L("Top attribution features to monitor: {f}.",
                                f=", ".join(str(x) for x in list(feats)[:5])))
        snap = self.console or {}
        if snap:
            watch.append(self.L("Detector model status: {s}.",
                                s=str(snap.get("model_state", "?"))))
            watch.append(self.L("Live stream: {s}.", s=self.L(
                "running" if snap.get("streaming") else "paused")))
            alerts = list(snap.get("alerts") or [])
            if alerts:
                crit = sum(1 for x in alerts if str(x.get("severity")) == "CRITICAL")
                watch.append(self.L("Alerts raised in this session: {n} ({c} "
                                    "critical).", n=len(alerts), c=crit))
                by_agent: Dict[str, List[Any]] = {}
                for x in alerts:
                    by_agent.setdefault(str(x.get("agent") or "?"), []).append(x)
                top_agent, rows = max(by_agent.items(), key=lambda kv: len(kv[1]))
                actions.append(self.L("Review rate limiting on {a}: this agent "
                                      "accounts for {n} of {m} alerts.",
                                      a=top_agent, n=len(rows), m=len(alerts)))
                worst = max(alerts, key=lambda x: _as_float(x.get("prob")) or 0.0)
                actions.append(self.L("Highest-risk event: {v} at p={p} ({t}).",
                                      v=self.lab(str(worst.get("verdict") or "")),
                                      p=_fmt(worst.get("prob")),
                                      t=str(worst.get("t") or "")))
            else:
                actions.append(self.L("No alerts recorded yet — inject a burst in "
                                      "the console to exercise the loop."))
        rep = self._agent_rep()
        safety = dict(rep.get("safety") or {})
        pv = _as_float(safety.get("privilege_violation_rate"))
        if pv:
            actions.append(self.L("Agent privilege violations: {v}. Keep the "
                                  "confirmation gate enabled for irreversible "
                                  "tools.", v=_fmt(pv)))
        return watch, actions

    def _abstract_ai(self, r, a) -> None:
        meta = self._agent_meta()
        self.flow(self.P(self.L("Methodological summary: each figure below comes "
                                "from a single TrustPipeline run on the held-out "
                                "test split (seed {s}), optionally paired with a "
                                "trace-level trustagent evaluation over {n} cases.",
                                s=meta.get("seed", 42),
                                n=getattr(a, "n_cases", 0) if a else 0)))
        kpi: Dict[str, Any] = {}
        if r is not None:
            kpi.update(dict(getattr(r, "base_metrics", {}) or {}))
            for pillar, keys in (("robustness", ("pgd_attack_success_rate",)),
                                 ("privacy", ("membership_inference_auc",))):
                for k in keys:
                    v = self._summary(pillar).get(k)
                    if v is not None:
                        kpi[k] = v
        if a is not None:
            rep = self._agent_rep()
            kpi["overall"] = rep.get("overall")
            inj = dict(dict(rep.get("robustness") or {}).get("injection") or {})
            if inj.get("n_attacks"):
                kpi["ASR"] = inj.get("attack_success_rate")
        self.kv(kpi, header=self.L("Key read-out"), keep_none=True)
        self.note(self.L("Reported quantities are point estimates without confidence "
                         "intervals. Seeds are fixed so reruns are reproducible, not "
                         "so the numbers generalise across datasets."))

    # ==================================================================
    # plain-language layer — only the layperson style calls these
    # ==================================================================
    def _pt(self, key: str, fallback: str) -> str:
        """Section heading: questions for a non-specialist, terms for everyone else."""
        if self.spec.style == "layperson" and key in _PLAIN_TITLES:
            return self.L(_PLAIN_TITLES[key])
        return self.L(fallback)

    def _plain_flow(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.flow(self.P(line))


    # -- one builder per topic -------------------------------------------


    # -------------------------------------------------------------- data/model
    def sec_data_model(self) -> None:
        self.h1(self._pt("data_model", "Dataset & model"))
        r = self.report
        if r is None:
            self.note(self.L("No analysis run yet — run the pipeline to populate "
                             "this report."))
            return
        ds = dict(getattr(r, "dataset_summary", {}) or {})
        spec = dict(getattr(r, "model_spec", {}) or {})
        if self.spec.style == "layperson":
            self._plain_flow(self._plain_data_model(r))
            names = [self.lab(str(x)) for x in
                     list(ds.get("feature_names") or [])[:12]]
            if names:
                self.h2(self.L("Every measurement by name"))
                self.flow(self.P(self.L("{l}.", l=self._sep.join(names))))
            if self.spec.show_body_tables:
                self.kv({"dataset": getattr(r, "dataset_key", "?"),
                         "samples": ds.get("n_samples"),
                         "features": ds.get("n_features"),
                         "classes": ds.get("class_names"),
                         "model": spec.get("label", spec.get("name")),
                         "elapsed_sec": spec.get("elapsed_sec")})
            return
        self.kv({"dataset": getattr(r, "dataset_key", "?"),
                 "task": ds.get("task"),
                 "samples": ds.get("n_samples"),
                 "features": ds.get("n_features"),
                 "classes": ds.get("class_names"),
                 "group_attrs": ds.get("group_attrs"),
                 "class_counts": ds.get("class_counts")},
                header=self.L("Dataset"))
        note = _dataset_note(getattr(r, "dataset_key", ""))
        if note:
            self.note(self.L("The dataset describes {note}.",
                             note=self.L(str(note))))
        self.kv({"model": spec.get("label", spec.get("name")),
                 "task": spec.get("task"),
                 "params": spec.get("params"),
                 "elapsed_sec": spec.get("elapsed_sec")},
                header=self.L("Model configuration"))
        if self.spec.style == "ai":
            self.kv({"feature_names": ds.get("feature_names"),
                     "meta": ds.get("meta")}, header=self.L("Feature list"))

    def sec_metrics(self) -> None:
        self.h1(self._pt("metrics", "Base performance"))
        r = self.report
        if r is None:
            self.note(self.L("No analysis run yet — run the pipeline to populate "
                             "this report."))
            return
        if self.spec.style == "layperson":
            self._plain_flow(self._plain_metrics(r))
            self.figs(getattr(r, "base_figures", {}) or {}, 1,
                      prefer=("confusion_matrix", "regression_scatter"))
            if self.spec.show_body_tables:
                self.kv(dict(getattr(r, "base_metrics", {}) or {}))
            return
        self.flow(self.P(self.L("Headline metrics on the held-out test split.")))
        self.kv(dict(getattr(r, "base_metrics", {}) or {}))
        self.note(self.L("Figures are rendered from the same run the panels display."))
        self.figs(getattr(r, "base_figures", {}) or {},
                  1 if self.spec.style == "layperson" else
                  (2 if self.spec.style == "smo" else 99))

    # -------------------------------------------------------------- four pillars
    _PILLAR_TITLE = {"explainability": "Explainability", "robustness": "Robustness",
                     "fairness": "Fairness", "privacy": "Privacy"}

    _PILLAR_FIG = {"explainability": ("shap_bar",), "robustness":
                   ("attack_comparison",), "fairness": ("fairness_groups",),
                   "privacy": ("dp_curve",)}

    def sec_pillar(self, pillar: str) -> None:
        self.h1(self._pt(pillar, self._PILLAR_TITLE[pillar]))
        res = self._res(pillar)
        self.flow(self.P(self._pillar_intro(pillar)))
        if res is None:
            self.note(self.L("pillar {p} was not part of this run",
                             p=self.L(self._PILLAR_TITLE[pillar])))
            return
        summary = self._summary(pillar)
        if summary.get("error"):
            self.flow(self.P(self.L("This pillar reported an error: {r}",
                                    r=str(summary["error"])), "body"))
            return
        if summary.get("skipped"):
            self.flow(self.P(self.L("This pillar was skipped: {r}",
                                    r=str(summary["skipped"])), "body"))
            return
        cap = _STYLE_FIGS[self.spec.style]
        if self.spec.style == "layperson":
            self._plain_flow(self._plain_pillar(pillar))
            self.figs(self._figures(pillar), cap, prefer=self._PILLAR_FIG[pillar])
            if self.spec.show_body_tables:
                self.kv(summary)
            return
        self.kv(summary)
        self.figs(self._figures(pillar), cap, prefer=self._PILLAR_FIG[pillar])
        if self.spec.show_body_tables:
            for name, df in self._tables(pillar).items():
                self.df(name, df)

    def _pillar_intro(self, pillar: str) -> str:
        style = self.spec.style if self.spec.style in ("layperson", "smo") else "ai"
        summary = self._summary(pillar)
        if pillar == "explainability":
            texts = {
                "layperson": "This part answers one thing: what the model was "
                             "actually looking at, so the answer can be argued "
                             "about instead of taken on faith.",
                "smo": "Attribution tells the operator which A1 feature pushed a "
                       "decision; use it when tuning policies and when disputing "
                       "a false positive.",
                "ai": "Global importance (permutation), KernelSHAP-style "
                      "attributions, a LIME-style local surrogate and PDP/ICE "
                      "curves, all computed on the test split.",
            }
            return self.L(texts[style])
        if pillar == "robustness":
            texts = {
                "layperson": "Here we tamper with the input on purpose, to see how "
                             "fast the answers stop being right.",
                "smo": "Degradation under manipulated traffic — the realistic "
                       "failure mode for an edge detector — plus the effect of "
                       "adversarial training as a mitigation.",
                "ai": "FGSM / PGD / DeepFool via surrogate gradients, robustness "
                      "curves over an ε sweep, and adversarial training as a "
                      "defence baseline.",
            }
            return self.L(texts[style])
        if pillar == "fairness":
            if style == "smo":
                return self.L("Group fairness by {g}: whether the same alerting "
                              "threshold behaves differently across policy types, "
                              "agents or platforms.",
                              g=self.lab(summary.get("group_attr") or "group"))
            if style == "ai":
                return self.L("Selection and error rates per group, demographic "
                              "parity, disparate impact and equalised-odds "
                              "differences over {n} groups.",
                              n=summary.get("n_groups", "—"))
            return self.L("This part asks whether one group of sites, agents or "
                          "platforms quietly gets better treatment than another.")
        if style == "smo":
            return self.L("Privacy budget and federation: how much utility is paid "
                          "for differential privacy, and whether edge agents can "
                          "learn without shipping raw traffic.")
        if style == "ai":
            return self.L("DP-SGD with per-sample gradients and moments-accountant "
                          "accounting over an ε sweep, FedAvg across {n} clients, "
                          "and a membership-inference attack group (confidence "
                          "threshold, label-aware meta-model, shadow SSA and LiRA).",
                          n=summary.get("n_clients", "—"))
        return self.L("And here: does training the model hand over anybody's "
                      "individual measurements?")

    # -------------------------------------------------------------- console
    @staticmethod
    def _alert_frame(snap: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([{"severity": a.get("severity"),
                              "t": a.get("t"), "agent": a.get("agent"),
                              "policy": a.get("policy"),
                              "verdict": a.get("verdict"),
                              "p_anomaly": a.get("prob"),
                              "request_rate": a.get("rate")}
                             for a in list(snap.get("alerts") or [])])

    def sec_oran(self) -> None:
        self.h1(self._pt("oran", "O-RAN console snapshot"))
        snap = self.console or {}
        if self.spec.style == "layperson":
            self._plain_flow(self._plain_console())
            if snap.get("alerts") and self.spec.show_body_tables:
                self.df("Alert log", self._alert_frame(snap),
                        max_rows=len(snap["alerts"]))
            return
        if not snap:
            self.note(self.L("No console session state was captured."))
            return
        self.kv({"Scenario": snap.get("scenario"),
                 "Ticks": snap.get("tick"),
                 "Events": snap.get("events"),
                 "streaming": self.L("running" if snap.get("streaming") else "paused"),
                 "model_state": snap.get("model_state"),
                 "detector_acc": snap.get("model_acc")},
                header=self.L("Session snapshot"))
        alerts = list(snap.get("alerts") or [])
        if alerts:
            frame = self._alert_frame(snap)
            self.df("Alert log", frame, max_rows=len(frame))
        else:
            self.note(self.L("No alerts recorded yet — inject a burst in the "
                             "console to exercise the loop."))
        if self.spec.style != "layperson":
            self.note(self.L("The A1 features in this section come from the "
                             "studio's simulated RIC loop, not from a live network."))
        if self.spec.style == "smo":
            _, actions = self._operational_lines()
            if actions:
                self.h2(self.L("Recommended actions"))
                self.bullets(actions)

    # -------------------------------------------------------------- agent
    def _pillar_gaps(self, rep: Dict[str, Any]) -> None:
        """Say which dimensions the run could not measure, and on what basis.

        A table row that is simply absent reads as "that one was fine".
        """
        pillars = dict(rep.get("pillars") or {})
        for p, v in pillars.items():
            if v is None:
                self.note(self.L("{d} — not measured: {r}", d=self.lab(p),
                                 r=self._dim_reason(p, rep)))
        codes = dict(rep.get("pillar_codes") or {})
        for p, code in codes.items():
            basis = _DIM_BASIS.get(str(code))
            if basis and pillars.get(p) is not None:
                self.note(self.L("{d}: {r}", d=self.lab(p), r=self.L(basis)))

    def sec_agent(self) -> None:
        self.h1(self._pt("agent", "Agent trust evaluation"))
        a = self.agent
        if a is None:
            self.note(self.L("No agent evaluation has been run yet — run it in the "
                             "Agent Eval panel to fill this section."))
            return
        rep = self._agent_rep()
        meta = dict(rep.get("meta") or {})
        pillars = dict(rep.get("pillars") or {})
        if self.spec.style == "layperson":
            self._plain_flow(self._plain_agent(a))
            self.figs(getattr(a, "figures", {}) or {}, _STYLE_FIGS[self.spec.style],
                      prefer=("agent_radar",))
            if self.spec.show_body_tables:
                scores = {"overall": rep.get("overall")}
                scores.update({k: pillars.get(k) for k in _PLAIN_DIM})
                self.kv(scores, header=self.L("Six-dimension scorecard"), keep_none=True)
                self._pillar_gaps(rep)
            return
        scores = {"overall": rep.get("overall")}
        scores.update({k: pillars.get(k) for k in
                       ("reliability", "robustness", "safety", "explainability",
                        "fairness", "privacy")})
        self.kv(scores, header=self.L("Six-dimension scorecard"), keep_none=True)
        self._pillar_gaps(rep)
        inj = dict(dict(rep.get("robustness") or {}).get("injection") or {})
        if inj.get("n_attacks"):
            self.kv({"ASR": inj.get("attack_success_rate"),
                     "n_attacks": inj.get("n_attacks")})
        if self.spec.style == "ai":
            self.kv({"mode": getattr(a, "mode", "?"),
                     "model": meta.get("llm"),
                     "endpoint": meta.get("endpoint"),
                     "judge": meta.get("judge"),
                     "suites": meta.get("suites"),
                     "conditions": meta.get("conditions"),
                     "policies": meta.get("policies"),
                     "k": meta.get("k"),
                     "n_tasks": meta.get("n_tasks"),
                     "seed": meta.get("seed")},
                    header=self.L("Run configuration"))
            self.note(self.L("Repeats k={k}, adapters: {a}.", k=meta.get("k", 1),
                             a=", ".join(str(x) for x in (meta.get("agents") or []))))
        self.figs(getattr(a, "figures", {}) or {}, _STYLE_FIGS[self.spec.style],
                  prefer=("agent_radar",))
        tables = dict(getattr(a, "tables", {}) or {})
        keep = ("agent_scorecard",) if self.spec.style != "ai" else \
            ("agent_scorecard", "agent_injection", "agent_safety",
             "agent_persona_parity")
        if self.spec.show_body_tables:
            for name in keep:
                if name in tables:
                    self.df(name, tables[name])
        self.gap(3)
        if str(getattr(a, "mode", "mock")) == "mock":
            self.flow(self.P(self.L("Mock-mode agent numbers validate the harness "
                                     "only; they are not a measurement of any real "
                                     "model."), "muted"))
        else:
            self.note(self.L("Live run against {e}, judge {j}.",
                             e=meta.get("endpoint") or self.L("local endpoint"),
                             j=meta.get("judge", "?")))
            if meta.get("judge_note"):
                self.note(self.L("Judge note: {n}", n=str(meta["judge_note"])))

    # -------------------------------------------------------------- methodology
    def sec_method(self) -> None:
        r = self.report
        if self.spec.style == "layperson":
            self.h1(self._pt("method", "Methodology & validity limits"))
            self._plain_flow([self.L(
                "How these numbers were produced, in one paragraph: the system "
                "learned from part of the data and was then tested only on the "
                "part it had never seen, so the score is not a memory test. The "
                "random seed is fixed, which means running the whole check again "
                "gives the same figures instead of a fresh roll of the dice."),
                self.L("Everything in this report came out of that single check, "
                       "which is why the numbers in the text and the numbers in "
                       "the charts agree."),
                self.L("The same check can be re-run from the command line with "
                       "{c}.",
                       c=f"python -m trustoran --dataset "
                         f"{getattr(r, 'dataset_key', 'cci_a1sd')} --quick")])
            self._validity_tail(include_heading=True, plain=True)
            return
        self.h1(self.L("Methodology & validity limits"))
        self.h2(self.L("Reproducibility"))
        key = getattr(r, "dataset_key", "cci_a1sd") if r else "cci_a1sd"
        model = (getattr(r, "model_spec", {}) or {}).get("name", "random_forest") \
            if r else "random_forest"
        self.flow(self.P(f"python -m trustoran --dataset {key} --model {model} "
                         f"--pillars all --quick", "cmd"))
        self.flow(self.P("python -m trustagent --quick", "cmd"))
        if self.spec.style == "ai":
            self._metric_definitions()
        self._validity_tail(include_heading=self.spec.style != "layperson")

    def _metric_definitions(self) -> None:
        present: Dict[str, Any] = {}
        for pillar in _PILLARS:
            present.update(self._summary(pillar))
        if self.agent is not None:
            present.update({"ASR": 0, "overall": 0})
        rows = [(k, DEFINITIONS[k]) for k in _DEF_ORDER if k in present
                and k in DEFINITIONS]
        if not rows:
            return
        self.h2(self.L("Metric definitions"))
        table_rows = [[Paragraph(_esc(self.lab(k)), self.styles["cellh"]),
                       Paragraph(_esc(DEFINITIONS[k][0 if self.lang == "en" else 1]),
                                 self.styles["cell"])] for k, _ in rows]
        self.flow(self._table(table_rows, [self.avail * 0.32, self.avail * 0.68]))

    def _validity_tail(self, include_heading: bool = True,
                       plain: bool = False) -> None:
        """Always printed: what the numbers do *not* prove, plus the disclaimer."""
        if include_heading:
            self.h2(self.L("What these numbers do not prove" if plain
                           else "Validity limits"))
        bullets: Sequence[str] = _PLAIN_LIMITS if plain else (
            "Public research corpora — not production RAN telemetry.",
            "Attacks use differentiable surrogates, so robustness figures "
            "describe this attack family rather than an exhaustive adversary.",
            "Privacy accounting follows the moments accountant inside the "
            "local DP-SGD loop; no formal (ε,δ) certificate is issued for "
            "the shipped estimator.",
            "The fairness grouping is a modelling choice fixed per dataset "
            "in DATASET_PROFILES, not a property of the data itself.",
            "The agent environment is a sandbox with ACLs, budgets and "
            "confirmation gates; no real system or network element is touched.",
            "Mock-mode agent numbers validate the harness only; they are not "
            "a measurement of any real model.",
            "For normative wording, consult the O-RAN Alliance specifications "
            "directly.")
        self.bullets([self.L(b) for b in bullets])
        self.gap(4)
        self.note(self.L("This module is a personal research project that "
                         "extends trustworthy-AI methodology to agent "
                         "evaluation; it is not affiliated with any "
                         "employer. The O-RAN scenarios and standards "
                         "summaries are the author's own paraphrasing of "
                         "public material."))

    # -------------------------------------------------------------- appendix
    def sec_appendix(self) -> None:
        self.h1(self._pt("appendix", "Appendix: result tables"))
        self.note(self.L("Every table produced by the run, truncated to {n} rows "
                         "per table.", n=self.spec.max_table_rows))
        if self.report is not None:
            for pillar in _PILLARS:
                for name, df in self._tables(pillar).items():
                    self.df(name, df)
        if self.agent is not None:
            for name, df in (getattr(self.agent, "tables", {}) or {}).items():
                self.df(name, df)

    # ==================================================================
    # assembly
    # ==================================================================
    def cover(self) -> None:
        spec = self.spec
        self.flow(Spacer(1, 24))
        title = spec.title or self.L("Trustworthy AI for Open RAN — Analysis Report")
        self.flow(self.P(title, "title"))
        self.flow(HRFlowable(width="100%", thickness=1.2, color=self.c["accent"],
                             spaceBefore=8, spaceAfter=10))
        self.flow(self.P(self.L("Audience: {a}", a=self.L(spec.style_label_key)),
                         "sub"))
        lines = []
        if spec.author:
            lines.append(f"{self.L('Prepared by')}: {spec.author}")
        lines.append(f"{self.L('Generated')}: {spec.date}")
        if self.report is not None:
            if spec.style == "layperson":
                metrics = dict(getattr(self.report, "base_metrics", {}) or {})
                mk = next((x for x in ("accuracy", "r2") if x in metrics), "")
                mspec = dict(getattr(self.report, "model_spec", {}) or {})
                mv = (self._pct(metrics.get(mk)) if mk == "accuracy"
                      else _fmt(metrics.get(mk)))
                lines.append(self.L("Run on {d} with {m}: {k} {v}",
                                    d=str(getattr(self.report, "dataset_key", "?")),
                                    m=self.lab(str(mspec.get("label")
                                                  or mspec.get("name") or "")),
                                    k=self.lab(mk), v=mv))
            else:
                lines.append(self.report.headline())
        if self.agent is not None and getattr(self.agent, "headline", ""):
            if spec.style == "layperson":
                ov = _as_float(self._agent_rep().get("overall"))
                lines.append(self.L("Agent check-up overall: {v}",
                                    v=self._pct(ov)))
            else:
                lines.append(str(self.agent.headline))
        lines.append(self.L("Personal research project — not affiliated with any "
                            "employer."))
        for line in lines:
            self.flow(self.P(line, "meta"))
        self.gap(14)
        self.h2(self.L("Contents"))
        labels = dict(SECTIONS)
        self.bullets([self._pt(k, labels[k]) for k in spec.sections])

    def run(self, report: Any, agent: Any, console: Optional[Dict[str, Any]],
            out_path: str, progress: ProgressCB = None) -> str:
        self.report, self.agent, self.console = report, agent, console
        self.story: List[Any] = []
        directory = os.path.dirname(os.path.abspath(out_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        pw, ph = self.pagesize
        m = self.margins
        body_w = pw - 2 * m
        body_h = ph - 2 * m
        doc = BaseDocTemplate(
            out_path, pagesize=self.pagesize,
            leftMargin=m, rightMargin=m, topMargin=m, bottomMargin=m,
            title=self.spec.title or self.L("Trustworthy AI for Open RAN — Analysis Report"),
            author=self.spec.author or "Trustworthy AI studio",
            subject=self.L(self.spec.style_label_key))

        def frame(x: float, y: float, w: float, h: float) -> Frame:
            return Frame(x, y, w, h, leftPadding=0, rightPadding=0,
                         topPadding=0, bottomPadding=0)

        templates = []
        if self.spec.cover:
            templates.append(PageTemplate(id="cover",
                                          frames=[frame(m, m + 20, body_w,
                                                        body_h - 20)],
                                          onPage=self._on_cover))
        if self.spec.columns == 1:
            frames = [frame(m, m, body_w, body_h)]
        else:
            cw = (body_w - COL_GAP) / 2.0
            frames = [frame(m, m, cw, body_h),
                      frame(m + cw + COL_GAP, m, cw, body_h)]
        templates.append(PageTemplate(id="body", frames=frames,
                                      onPage=self._on_body))
        doc.addPageTemplates(templates)

        if self.spec.cover:
            self.cover()
            self.flow(NextPageTemplate("body"), PageBreak())

        builders = {"abstract": self.sec_abstract,
                    "data_model": self.sec_data_model,
                    "metrics": self.sec_metrics,
                    "oran": self.sec_oran,
                    "agent": self.sec_agent,
                    "method": self.sec_method,
                    "appendix": self.sec_appendix}
        keys = [k for k in SECTION_KEYS if self.spec.has(k)]
        for i, key in enumerate(keys):
            if progress:
                progress("layout", i / max(1, len(keys)), self.L(dict(SECTIONS)[key]))
            if key in _PILLARS:
                self.sec_pillar(key)
            else:
                builders[key]()
        if not self.spec.has("method"):          # the disclaimer is not optional
            self.gap(8)
            self._validity_tail()
        if progress:
            progress("render", 0.9, out_path)
        doc.build(self.story)
        if progress:
            progress("done", 1.0, out_path)
        return out_path


# --------------------------------------------------------------------------
# glossary for the AI-style reader
# --------------------------------------------------------------------------
DEFINITIONS: Dict[str, Tuple[str, str]] = {
    "accuracy": ("Share of held-out rows classified correctly.",
                 "留出集上被正确分类的样本比例。"),
    "r2": ("Coefficient of determination on the test split.",
           "测试集上的决定系数 R²。"),
    "clean_accuracy": ("Accuracy before any perturbation is applied.",
                       "未施加扰动时的准确率（基线）。"),
    "fgsm_robust_acc": ("Accuracy under one-step FGSM at the reported ε.",
                        "给定 ε 下单步 FGSM 攻击后的准确率。"),
    "pgd_robust_acc": ("Accuracy after 10-step PGD optimisation in the ε-ball.",
                       "ε 球内 10 步 PGD 优化后的准确率。"),
    "deepfool_robust_acc": ("Accuracy under the minimum-perturbation DeepFool attack.",
                            "最小扰动 DeepFool 攻击后的准确率。"),
    "pgd_attack_success_rate": ("Share of rows whose true label is flipped by PGD.",
                                "被 PGD 改变真实标签的样本比例。"),
    "adv_trained_robust_acc@eps0.6": ("Same model retrained on adversarial examples.",
                                      "在对抗样本上重训后的同结构模型准确率。"),
    "clean_r2": ("R² under zero noise drift (regression datasets).",
                 "噪声漂移为 0 时的 R²（回归数据集）。"),
    "r2_at_max_sigma": ("R² at the largest Gaussian noise level tested.",
                        "最大高斯噪声强度下的 R²。"),
    "utility_at_max_sigma": ("MAE(0)/MAE(σ) at the largest noise level; 0.5 means the "
                             "prediction error has doubled.",
                             "最大噪声档下的 MAE(0)/MAE(σ)；0.5 表示误差翻倍。"),
    "mae_clean": ("Mean absolute error with no noise added (W for vRAN power).",
                  "未加噪声时的平均绝对误差（功耗数据集单位为 W）。"),
    "mae_at_max_sigma": ("Mean absolute error at the largest noise level tested.",
                         "最大噪声档下的平均绝对误差。"),
    "sigma": ("Standard deviation of the Gaussian noise applied to the input features.",
              "施加到输入特征上的高斯噪声标准差。"),
    "utility": ("Retained utility MAE(0)/MAE(σ); 1.0 means no degradation.",
                "保留效用 MAE(0)/MAE(σ)，1.0 表示无退化。"),
    "demographic_parity_difference": ("Max minus min selection rate across groups.",
                                      "各组选择率的最大最小差。"),
    "disparate_impact": ("Min / max selection rate; the 4/5-rule quantity.",
                         "最小/最大选择率之比，即 4/5 规则所用指标。"),
    "equalized_odds_difference": ("Largest TPR/FPR gap across groups.",
                                  "各组之间 TPR/FPR 的最大差距。"),
    "accuracy_gap": ("Best minus worst group accuracy.", "最优组与最差组准确率之差。"),
    "mae_gap": ("Best minus worst group MAE (regression).", "各组 MAE 的最大差距（回归）。"),
    "utility_open": ("Utility of the unconstrained baseline network — accuracy for "
                     "classification, R² for regression.",
                     "不加隐私约束的基线效用：分类任务为准确率，回归任务为 R²。"),
    "utility_eps1": ("Utility at a privacy budget of ε=1 — accuracy for classification, "
                     "R² for regression.",
                     "隐私预算 ε=1 时的效用：分类任务为准确率，回归任务为 R²。"),
    "membership_inference_auc": ("Membership-attack AUC of the meta-model fitted on "
                                 "label-aware attack features; 0.5 means no leakage "
                                 "signal. The threshold, shadow (SSA) and likelihood-ratio "
                                 "(LiRA) attacks are reported next to it.",
                                 "元模型成员推断攻击 AUC（判别特征含真实标签侧），0.5 表示无泄露信号；"
                                 "阈值攻击、影子模型 SSA 与似然比 LiRA 的读数并列给出。"),
    "auc_threshold": ("AUC of the confidence-threshold attack, which needs no label.",
                      "置信度阈值攻击的 AUC，不需要真实标签。"),
    "auc_lr_conf": ("AUC of the meta-model on label-free features (confidence, entropy, "
                    "margin) — the reading the earlier single-attack build produced.",
                    "元模型仅用无标签特征（置信度、熵、裕度）的 AUC，对应此前单一攻击的口径。"),
    "auc_lr_meta": ("AUC of the meta-model once the label-side features are appended.",
                    "元模型加入标签侧判别特征后的 AUC。"),
    "auc_ssa": ("AUC of the shadow-group attack, whose meta-model is trained on shadow "
                "models the adversary trains themselves.",
                "影子模型群体攻击（SSA）的 AUC，元模型在外部人员自行训练的影子模型上拟合。"),
    "auc_lira": ("AUC of the likelihood-ratio attack over the same shadow group.",
                 "同一影子模型群上的似然比攻击（LiRA）AUC。"),
    "auc_ssa_accuracy": ("Shadow-group attack accuracy at a 0.5 decision cut.",
                         "影子模型群体攻击在 0.5 判决切点上的准确率。"),
    "ks_stat": ("Largest gap between the member and non-member distributions of the "
                "scored feature; how much of the attack signal is raw distribution shift.",
                "成员与非成员在被计分特征分布上的最大差距，用于区分攻击信号与分布差异。"),
    "tpr_at_1pct_fpr": ("Share of members correctly flagged when the false-alarm budget "
                        "is held to 1%.", "误报率控制在 1% 时，被正确识别的成员比例。"),
    "confidence_gap": ("Mean confidence on training rows minus mean confidence on test "
                       "rows.", "训练集平均置信度减去测试集平均置信度。"),
    "generalization_gap": ("Training accuracy minus test accuracy; memorisation shows up "
                           "here first.", "训练准确率减去测试准确率，记忆化首先体现在此项。"),
    "near_duplicate_rate": ("Share of test rows appearing verbatim in the training set; "
                            "overlap inflates every attack AUC.",
                            "测试集中与训练集逐位重复的行占比，重复会同时抬高所有攻击的 AUC。"),
    "n_shadows": ("Shadow models retrained by the attacker for this run.",
                  "本次攻击由外部人员重新训练的影子模型数量。"),
    "federated_accuracy": ("FedAvg accuracy after the last communication round.",
                           "最后一轮通信后的 FedAvg 精度。"),
    "overall": ("Equal-weight mean of the trust dimensions this run could measure; a "
                "dimension with no reading is left out, never counted as 0 or 1.",
                "本次能测的可信维度等权平均；测不出来的维度被排除在平均之外，"
                "既不计 0 也不计 1。"),
    "ASR": ("Attack success rate on the injection suite.", "注入攻击套件上的攻击成功率。"),
    "pass_pow_k": ("Share of tasks where all k independent runs succeed.",
                   "同一用例 k 次独立运行全部成功的比例。"),
    "privilege_violation_rate": ("Share of calls denied by the sandbox ACL.",
                                 "被沙箱 ACL 拒绝的工具调用比例。"),
}

_DEF_ORDER = tuple(DEFINITIONS)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def build_report(spec: ReportSpec, report: Any = None, agent: Any = None,
                 console: Optional[Dict[str, Any]] = None,
                 out_path: Optional[str] = None,
                 progress: ProgressCB = None) -> str:
    """Write the PDF and return its path.

    ``report`` is an :class:`trustoran` ``AnalysisReport``, ``agent`` an
    ``AgentEvalOutput``, ``console`` an :class:`OranConsole` snapshot — any of
    the three may be ``None`` and the matching section degrades to a note.
    """
    target = out_path or default_out_path(spec.style)
    return _Builder(spec).run(report, agent, console, target, progress)


def available() -> bool:
    """True when reportlab can be imported (the dialog degrades if it cannot)."""
    try:
        import reportlab            # noqa: F401
    except Exception:
        return False
    return True
