"""Shared dark O-RAN theme + Qt stylesheet.

The colours mirror :data:`trustoran.viz.plotting.PALETTE` so the matplotlib
figures and the Qt chrome look like one product.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont

try:                                    # keep the two palettes literally in sync
    from trustoran.viz.plotting import PALETTE
except Exception:                       # pragma: no cover - fallback if import order differs
    PALETTE = {
        "bg": "#0f1720", "panel": "#16212c", "ink": "#e6edf3", "muted": "#8b98a5",
        "accent": "#22d3ee", "series": ["#22d3ee", "#f472b6", "#fbbf24", "#34d399",
                                        "#a78bfa", "#fb7185"],
        "good": "#34d399", "bad": "#fb7185", "grid": "#243244",
    }

BG = PALETTE["bg"]
PANEL = PALETTE["panel"]
INK = PALETTE["ink"]
MUTED = PALETTE["muted"]
ACCENT = PALETTE["accent"]
GOOD = PALETTE["good"]
BAD = PALETTE["bad"]
GRID = PALETTE["grid"]

SERIES = PALETTE["series"]


def qcolor(hex_str: str) -> QColor:
    return QColor(hex_str)


#: Typography zoom. The stylesheet's nominal sizes are already tuned for this
#: multi-panel window; the View menu multiplies them by one of 0.9 / 1.0 / 1.15
#: and re-applies font + stylesheet at runtime.
SCALE = 1.0


def px(nominal: float) -> int:
    """A nominal stylesheet *font* size, adjusted by :data:`SCALE`."""
    return max(8, int(round(nominal * SCALE)))


def gap(nominal: float) -> int:
    """A nominal padding / min-height, adjusted by :data:`SCALE`."""
    return max(1, int(round(nominal * SCALE)))


def apply_mpl_font() -> None:
    """Give matplotlib a CJK-capable font stack.

    The default (DejaVu Sans) has no Chinese glyphs, so any localised in-figure
    text (e.g. the console's rolling-rate title) would render as tofu boxes.
    Microsoft YaHei / SimHei cover both CJK and Latin, so one stack serves both
    languages.
    """
    import matplotlib
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def base_font() -> QFont:
    f = QFont("Segoe UI")
    f.setStyleHint(QFont.SansSerif)
    f.setPointSize(max(7, int(round(9 * SCALE))))
    return f


def apply_scale(app, scale: float) -> None:
    """Re-apply the global font + stylesheet at a new zoom level."""
    global SCALE
    SCALE = float(scale)
    app.setFont(base_font())
    app.setStyleSheet(stylesheet())


def stylesheet() -> str:
    return f"""
    * {{ font-family: 'Segoe UI', sans-serif; }}
    QMainWindow, QWidget#root {{ background: {BG}; color: {INK}; }}
    QWidget {{ color: {INK}; font-size: {px(12)}px; }}

    /* menu bar (File / View / Tools / Help) */
    QMenuBar {{ background: {PANEL}; color: {INK};
                border-bottom: 1px solid {GRID}; font-size: {px(12)}px; }}
    QMenuBar::item {{ background: transparent; padding: {gap(5)}px {gap(10)}px; }}
    QMenuBar::item:selected {{ background: {BG}; color: {ACCENT}; }}
    QMenu {{ background: {PANEL}; border: 1px solid {GRID}; font-size: {px(12)}px; }}
    QMenu::item {{ padding: {gap(5)}px {gap(20)}px {gap(5)}px {gap(14)}px; }}
    QMenu::item:selected {{ background: {BG}; color: {ACCENT}; }}
    QMenu::separator {{ height: 1px; background: {GRID}; margin: 4px 8px; }}
    QToolTip {{ background: {PANEL}; color: {INK}; border: 1px solid {GRID};
                font-size: {px(12)}px; padding: 3px; }}

    /* navigation rail */
    QListWidget#nav {{
        background: {PANEL}; border: none; border-right: 1px solid {GRID};
        outline: 0; font-size: {px(12)}px;
    }}
    QListWidget#nav::item {{ padding: {gap(8)}px {gap(10)}px; border: none; color: {MUTED}; }}
    QListWidget#nav::item:selected {{
        background: {BG}; color: {ACCENT};
        border-left: 3px solid {ACCENT}; padding-left: {gap(7)}px;
    }}
    QListWidget#nav::item:hover {{ color: {INK}; }}

    /* toolbar / controls */
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {BG}; border: 1px solid {GRID}; border-radius: 6px;
        padding: {gap(4)}px {gap(7)}px; color: {INK}; min-height: {gap(18)}px;
        font-size: {px(12)}px;
    }}
    QComboBox::drop-down {{ border: none; width: {gap(18)}px; }}
    QComboBox QAbstractItemView {{
        background: {PANEL}; border: 1px solid {GRID}; color: {INK};
        font-size: {px(12)}px;
        selection-background-color: {ACCENT}; selection-color: {BG};
    }}
    QCheckBox {{ color: {MUTED}; spacing: {gap(5)}px; font-size: {px(12)}px; }}
    QCheckBox::indicator {{
        width: {gap(14)}px; height: {gap(14)}px; border-radius: 4px;
        border: 1px solid {GRID}; background: {BG};
    }}
    QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

    QPushButton {{
        background: {PANEL}; border: 1px solid {GRID}; border-radius: 7px;
        padding: {gap(5)}px {gap(12)}px; color: {INK}; font-weight: 600;
        font-size: {px(12)}px;
    }}
    QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
    QPushButton:pressed {{ background: {BG}; }}
    QPushButton:disabled {{ color: {MUTED}; border-color: {GRID}; }}
    QPushButton#primary {{
        background: {ACCENT}; color: {BG}; border: none; font-weight: 700;
    }}
    QPushButton#primary:hover {{ background: #67e3f5; color: {BG}; }}
    QPushButton#primary:disabled {{ background: {GRID}; color: {MUTED}; }}
    QPushButton#danger {{ border-color: {BAD}; color: {BAD}; }}
    QPushButton#danger:hover {{ background: {BAD}; color: {BG}; }}
    QPushButton#tune {{
        background: {BG}; border: 1px solid {GRID}; color: {MUTED};
        padding: {gap(4)}px {gap(10)}px; font-weight: 600;
    }}
    QPushButton#tune:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
    QPushButton#tune:disabled {{ color: {GRID}; border-color: {GRID}; }}
    QDialog {{ background: {BG}; }}

    /* panels & cards */
    QFrame#card {{
        background: {PANEL}; border: 1px solid {GRID}; border-radius: 10px;
    }}
    QLabel#h1 {{ font-size: {px(17)}px; font-weight: 700; color: {INK}; }}
    QLabel#h2 {{ font-size: {px(13)}px; font-weight: 700; color: {ACCENT}; }}
    QLabel#muted {{ color: {MUTED}; font-size: {px(11)}px; }}
    QLabel#metricValue {{ font-size: {px(18)}px; font-weight: 700; color: {INK}; }}
    QLabel#metricLabel {{ font-size: {px(11)}px; color: {MUTED}; }}

    /* tables */
    QTableView, QTreeWidget, QTableWidget {{
        background: {BG}; alternate-background-color: {PANEL};
        border: 1px solid {GRID}; border-radius: 6px; gridline-color: {GRID};
        font-size: {px(12)}px;
        selection-background-color: {ACCENT}; selection-color: {BG};
    }}
    QHeaderView::section {{
        background: {PANEL}; color: {MUTED}; border: none;
        border-bottom: 1px solid {GRID}; padding: {gap(5)}px;
        font-size: {px(12)}px; font-weight: 600;
    }}
    QTableView::item:selected {{ background: {ACCENT}; color: {BG}; }}

    /* progress */
    QProgressBar {{
        background: {BG}; border: 1px solid {GRID}; border-radius: 6px;
        text-align: center; color: {INK}; height: {gap(16)}px;
        font-size: {px(10)}px;
    }}
    QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

    /* scrollbars */
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: {BG}; width: {gap(10)}px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {GRID}; border-radius: 5px; min-height: {gap(24)}px; }}
    QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
    QScrollBar:horizontal {{ background: {BG}; height: {gap(10)}px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {GRID}; border-radius: 5px; min-width: {gap(24)}px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

    QTabWidget::pane {{ border: 1px solid {GRID}; border-radius: 6px; top: -1px; }}
    QTabBar::tab {{
        background: {PANEL}; color: {MUTED}; padding: {gap(5)}px {gap(11)}px;
        font-size: {px(12)}px;
        border: 1px solid {GRID}; border-bottom: none;
        border-top-left-radius: 6px; border-top-right-radius: 6px;
    }}
    QTabBar::tab:selected {{ color: {ACCENT}; background: {BG}; }}

    QStatusBar {{ background: {PANEL}; color: {MUTED}; font-size: {px(12)}px; }}
    QGroupBox {{
        border: 1px solid {GRID}; border-radius: 8px; margin-top: {gap(10)}px;
        padding-top: {gap(4)}px; color: {MUTED}; font-weight: 600;
        font-size: {px(12)}px;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
    """
