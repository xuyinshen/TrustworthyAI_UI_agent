"""Reusable Qt widgets: matplotlib canvases, metric cards, DataFrame tables.

These are the building blocks every panel is composed from. Nothing here knows
about the trust pipeline — panels feed them ``Figure`` / ``DataFrame`` / ``dict``
objects produced by :class:`trustoran.pipeline.report.AnalysisReport`.

All user-visible captions go through :mod:`trustoran_ui.i18n` so the whole studio
can flip between Simplified Chinese (default) and English; each widget exposes
``retranslate()`` which the main window calls on a language change.

Two layout primitives carry the "never squeeze a page" rule:

* :class:`PageScroll` — a panel's body lives inside it, so a short window gets a
  scrollbar instead of a compressed figure grid, a clipped table and a wrapped-away
  metric strip. Each section then keeps its own minimum height.
* :class:`FlowLayout` — wrapping rows for horizontal runs of controls or metric
  cards. Anything that would otherwise be pushed off the right edge drops onto the
  next row, and ``heightForWidth`` tells the page how much taller that made it.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from xml.sax.saxutils import escape as _escape

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    Qt, QAbstractTableModel, QEvent, QModelIndex, QObject, QPoint, QRect, QSize,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLayout, QScrollArea, QSizePolicy, QSpacerItem, QSplitter, QTableView,
    QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from trustoran.viz import labels as VZ

from . import theme
from . import i18n


def enable_height_for_width(widget: QWidget) -> QWidget:
    """Let a widget's *wrapped* height be seen by the layout above it.

    ``QVBoxLayout``/``QSplitter`` only consult ``heightForWidth`` when the item's
    size policy asks for it, so a card that nests a :class:`FlowLayout` must opt
    in or its second and third rows are silently clipped.
    """
    pol = widget.sizePolicy()
    if not pol.hasHeightForWidth():
        pol.setHeightForWidth(True)
        widget.setSizePolicy(pol)
    return widget


def apply_splitter_floor(splitter: QSplitter) -> QSplitter:
    """Publish a splitter's minimum *height* from its panes' explicit floors.

    ``QSplitter`` only forwards a pane's *minimumSize* to the layout above — never
    its ``minimumSizeHint`` — and it publishes nothing of its own beyond the panes'
    handles. So a page whose grid demands 1850 px still gets asked for 0 px unless
    the splitter states the sum out loud.

    Width is left alone on purpose: :class:`PageScroll` switches horizontal
    scrolling off, because a row too wide for the window is a wrapping problem
    (:class:`FlowLayout`), not something to hide behind a bar. A summed pane width
    would turn into clipped content instead of a scrollbar.
    """
    n = splitter.count()
    if n == 0:
        return splitter
    handle = splitter.handleWidth() * max(0, n - 1)
    heights = []
    for i in range(n):
        pane = splitter.widget(i)
        if pane is None:
            continue
        # a wrapper widget carries an explicit minimum of 0 while the child inside
        # it still demands room, so the layout hint has to count as well.
        heights.append(max(pane.minimumHeight(),
                           pane.minimumSizeHint().height(), 0))
    if splitter.orientation() == Qt.Vertical:
        splitter.setMinimumHeight(sum(heights) + handle)
    else:
        splitter.setMinimumHeight(max(heights, default=0))
    splitter.updateGeometry()
    return splitter


def refresh_splitter_floors(widget: QWidget) -> None:
    """Re-apply the floor of every splitter above ``widget``, stopping at a scroll
    area.

    Panes nest (grid inside splitter inside splitter), so one grid re-layout can
    change the demand at each level. A :class:`PageScroll` is the end of that
    chain: below it the page must state its needs, above it the window is allowed
    to stay small and let the wheel do the work.
    """
    child = widget
    for _ in range(16):                      # depth guard; the chain always ends
        parent = child.parentWidget()
        if parent is None or isinstance(parent, QScrollArea):
            return
        if isinstance(parent, QSplitter):
            apply_splitter_floor(parent)
        child = parent


class _HorizontalWheel(QObject):
    """Tilt-wheel and Shift+wheel scroll a table sideways.

    Qt's item views convert only the *vertical* wheel delta into scrolling, so once
    a table has user-resizable columns and a horizontal bar, the columns past the
    right edge were reachable by dragging that bar and nothing else. The gesture is
    ignored whenever there is nothing to scroll sideways, which leaves plain row
    wheeling (and the page's own wheel) untouched.
    """

    def __init__(self, view: QTableView):
        super().__init__(view)
        self._view = view

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Wheel:
            delta = event.angleDelta()
            shift = bool(event.modifiers() & Qt.ShiftModifier)
            # rightward tilt = later columns; Shift + wheel down = the same gesture
            # for the many mice with no tilt wheel. A plain vertical wheel is left
            # alone so rows and the page keep scrolling normally.
            amount = delta.x() or (-delta.y() if shift else 0)
            bar = self._view.horizontalScrollBar()
            if amount and bar.maximum() > bar.minimum():
                step = max(bar.singleStep() * 3, 30)
                move = int(round(amount / 120.0)) * step
                bar.setValue(bar.value() + (move or (step if amount > 0 else -step)))
                event.accept()
                return True
        return super().eventFilter(obj, event)


def enable_horizontal_wheel(view: QTableView) -> QTableView:
    """Let the wheel move a table sideways, keeping the filter alive on the view."""
    view._hwheel = _HorizontalWheel(view)
    view.viewport().installEventFilter(view._hwheel)
    return view


def fit_columns(view: QTableView, min_width: int = 56,
                max_width: int = 320) -> None:
    """Size every column to its contents, clamped.

    Interactive sections otherwise all start at Qt's default width, which is either
    far too wide for a boolean or far too narrow for a note — and a table whose
    columns all say "100" hides the fact that it can be dragged.
    """
    model = view.model()
    if model is None:
        return
    view.resizeColumnsToContents()
    for col in range(model.columnCount()):
        view.setColumnWidth(col, min(max(view.columnWidth(col), min_width), max_width))


# --------------------------------------------------------------------------
# layout primitives
# --------------------------------------------------------------------------
class FlowLayout(QLayout):
    """Left-to-right rows that wrap onto a new line when the width runs out.

    ``hasHeightForWidth`` is the point: without it a wrapping strip reports a
    one-row minimum and the parent layout clips the second row away.
    """

    def __init__(self, parent=None, margin: int = 0, spacing: int = -1):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
            # a box layout only consults heightForWidth when the widget asks for it
            enable_height_for_width(parent)
        self.setSpacing(spacing if spacing >= 0 else 8)
        self._items: List[QLayout] = []

    # -- QLayout interface -------------------------------------------------
    def addLayout(self, sub: QLayout) -> None:
        """Flatten a nested group (e.g. a caption + field box) into this row.

        The items become ordinary flow cells, so a group can wrap apart instead
        of pinning both widgets to one line.
        """
        while sub.count():
            item = sub.takeAt(0)
            w = item.widget()
            if w is not None:
                self.addWidget(w)
            else:
                self.addItem(item)

    def addStretch(self, stretch: int = 1) -> None:
        """Trailing slack, mirroring the box layouts' convenience method.

        :meth:`_arrange` treats a widget-less zero-width item as spacing, so the
        stretch never takes a slot of its own and can never force a wrap.
        """
        self.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), test=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, test=False)

    # -- internals ---------------------------------------------------------
    def _arrange(self, rect: QRect, *, test: bool) -> int:
        m = self.contentsMargins()
        left, top, right, bottom = m.left(), m.top(), m.right(), m.bottom()
        work = rect.adjusted(left, top, -right, -bottom)
        x, y, row_h = work.x(), work.y(), 0
        gap = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            if item.widget() is None and hint.width() <= 0:
                x += gap                              # addStretch(): spacing only
                continue
            w = hint.width() + gap
            if row_h and x + w > work.right():        # wrap onto a new line
                x = work.x()
                y += row_h + gap
                row_h = 0
            if not test:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += w
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y() + bottom


class PageScroll(QScrollArea):
    """Scrollable page body: the content may be taller than the viewport.

    Horizontal scrolling stays off on purpose — a wide row is a wrapping problem
    (:class:`FlowLayout`), not something to hide behind a horizontal bar.
    """

    def __init__(self, body: Optional[QWidget] = None, min_height: int = 0,
                 parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if body is not None:
            self.set_body(body, min_height)

    def set_body(self, body: QWidget, min_height: int = 0) -> QWidget:
        if min_height:
            body.setMinimumHeight(min_height)
        self.setWidget(body)
        self._body = body
        return body

    def body(self) -> Optional[QWidget]:
        return getattr(self, "_body", None)

    def add_footer(self, w: QWidget) -> QWidget:
        """Append ``w`` to the bottom of the page body, inside the scroll area.

        Every page body is a ``QVBoxLayout`` stack, so this is the one hook all
        seven pages share for their conclusion card — and because the card ends
        up inside the scroll area it can never be squeezed by a short window.
        """
        body = self.body()
        lay = body.layout() if body is not None else None
        if lay is not None:
            lay.addWidget(w)
        return w

    def scrollable_height(self) -> int:
        """The height the page wants; compare with the viewport to know whether
        a scrollbar is needed (offscreen geometry is not trustworthy, this is)."""
        body = self.body()
        if body is None:
            return 0
        # a bare QWidget reports minimumSizeHint() == (-1, -1); only explicit
        # floors and layout-derived demands count here.
        demand = max(body.minimumSize().height(),
                     body.minimumSizeHint().height(), 0)
        if body.hasHeightForWidth():
            wide = body.width() or self.viewport().width()
            if wide > 0:
                demand = max(demand, body.heightForWidth(wide))
        return demand


# --------------------------------------------------------------------------
# matplotlib canvas
# --------------------------------------------------------------------------
class FigureView(QWidget):
    """A single matplotlib ``Figure`` embedded in a Qt canvas + toolbar."""

    def __init__(self, title: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._title_key: Optional[str] = title
        self._title_lbl: Optional[QLabel] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        if title:
            self._title_lbl = QLabel(i18n.label(title))
            self._title_lbl.setObjectName("h2")
            lay.addWidget(self._title_lbl)
        self._fig: Optional[Figure] = None
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 3.4), dpi=100))
        self.canvas.setStyleSheet(f"background:{theme.BG};")
        lay.addWidget(self.canvas)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_title_key(self, key: Optional[str]) -> None:
        self._title_key = key
        if self._title_lbl is not None and key is not None:
            self._title_lbl.setText(i18n.label(key))

    def retranslate(self) -> None:
        if self._title_lbl is not None and self._title_key is not None:
            self._title_lbl.setText(i18n.label(self._title_key))
        if self._fig is not None:
            VZ.relabel_figure(self._fig, i18n.language())

    def set_figure(self, fig: Optional[Figure]) -> None:
        if fig is None:
            return
        VZ.relabel_figure(fig, i18n.language())
        old = self.canvas
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setStyleSheet(f"background:{theme.BG};")
        lay = self.layout()
        lay.replaceWidget(old, self.canvas)
        old.deleteLater()
        self.canvas.draw_idle()
        self._fig = fig

    def clear(self) -> None:
        fig = Figure(figsize=(5, 3.4), dpi=100, facecolor=theme.BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(theme.PANEL)
        ax.text(0.5, 0.5, "no figure", ha="center", va="center", color=theme.MUTED)
        ax.set_xticks([]); ax.set_yticks([])
        self.set_figure(fig)


class FigureGrid(QWidget):
    """A grid of :class:`FigureView` cells that grows taller instead of squeezing.

    This used to be a ``QScrollArea`` of its own, which gave the page two wheels
    fighting over one gesture — hover a figure and the figure scrolled while the
    page stayed put — and let the grid collapse into a 128 px viewport. As a plain
    widget it reports the height its rows really need, :class:`PageScroll` owns the
    wheel, and the column count follows the width it was given.
    """

    #: a cell narrower than this is unreadable, so the grid drops a column
    MIN_CELL_WIDTH = 420

    def __init__(self, columns: int = 2, row_height: int = 300, parent=None):
        super().__init__(parent)
        self.columns = max(1, int(columns))
        self.row_height = int(row_height)
        self._cols = self.columns
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(10)
        self._grid.setAlignment(Qt.AlignTop)
        self._views: List[FigureView] = []
        self._names: List[str] = []
        self._empty: Optional[QLabel] = None

    def set_figures(self, figures: Dict[str, Figure]) -> None:
        self._clear()
        if not figures:
            self._empty = QLabel(i18n.tr("Run an analysis to populate this panel."))
            self._empty.setObjectName("muted")
            self._empty.setAlignment(Qt.AlignCenter)
            self._empty.setMinimumHeight(self.row_height)
            self._grid.addWidget(self._empty, 0, 0)
            self._apply_height_floor()
            return
        for name, fig in figures.items():
            view = FigureView(title=name)
            view.set_figure(fig)
            view.setMinimumHeight(self.row_height)
            self._views.append(view)
            self._names.append(name)
        self._relayout()

    def _apply_height_floor(self) -> None:
        """Publish the height this grid's rows really need.

        A splitter forwards its panes' *minimumSize* (not minimumSizeHint) to the
        layout above, so without an explicit floor here the page would happily be
        given one row of pixels for six figures. The floor then has to be re-stated
        at every splitter above, or the page never learns it grew.
        """
        rows = max(1, self.rows())
        self.setMinimumHeight(rows * self.row_height + (rows - 1) * self._grid.spacing())
        refresh_splitter_floors(self)

    def _relayout(self) -> None:
        """Re-place the cells for the current width, then resize for the new rows."""
        if not self._views:
            return
        wide = self.width()
        cols = self.columns
        if wide > 0:
            cols = max(1, min(self.columns, wide // self.MIN_CELL_WIDTH))
        if cols == self._cols and self._grid.count() == len(self._views):
            return
        self._cols = cols
        while self._grid.count():
            self._grid.takeAt(0)                    # items only; widgets survive
        for i, view in enumerate(self._views):
            r, c = divmod(i, cols)
            self._grid.addWidget(view, r, c)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)
        self._apply_height_floor()
        self._grid.invalidate()
        self.updateGeometry()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._relayout()

    def rows(self) -> int:
        return -(-len(self._views) // max(1, self._cols)) if self._views else 0

    def retranslate(self) -> None:
        if self._empty is not None:
            self._empty.setText(i18n.tr("Run an analysis to populate this panel."))
        for view in self._views:
            view.retranslate()

    def _clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._views.clear()
        self._names.clear()
        self._empty = None
        self._cols = self.columns


# --------------------------------------------------------------------------
# metric cards
# --------------------------------------------------------------------------
class MetricCard(QFrame):
    def __init__(self, label_key: str, value: str = "—", accent: str = theme.INK, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._label_key = label_key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self._val = QLabel(value)
        self._val.setObjectName("metricValue")
        self._val.setStyleSheet(f"color:{accent};")
        self._lbl = QLabel(i18n.label(label_key))
        self._lbl.setObjectName("metricLabel")
        lay.addWidget(self._val)
        lay.addWidget(self._lbl)
        self.setMinimumWidth(120)

    def retranslate(self) -> None:
        self._lbl.setText(i18n.label(self._label_key))

    def set(self, value: str, accent: Optional[str] = None) -> None:
        self._val.setText(value)
        if accent:
            self._val.setStyleSheet(f"color:{accent};")


class SummaryStrip(QWidget):
    """A wrapping row of :class:`MetricCard` built from a summary ``dict``.

    The flow layout *is* the widget's layout so the cards can use a second and
    third row on a narrow page; because :class:`FlowLayout` answers
    ``heightForWidth``, the wrapped rows make the page taller instead of pushing
    cards off the right edge.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._flow = FlowLayout(self, spacing=10)
        self._cards: List[MetricCard] = []
        self._summary: Dict = {}
        self._good: tuple = ()
        self._bad: tuple = ()
        self._nomsg: Optional[QLabel] = None

    def set_summary(self, summary: Dict, good_keys=(), bad_keys=()) -> None:
        self._summary = dict(summary or {})
        self._good = tuple(good_keys)
        self._bad = tuple(bad_keys)
        self._rebuild()

    def _rebuild(self) -> None:
        while self._flow.count():
            item = self._flow.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._cards.clear()
        self._nomsg = None
        if not self._summary:
            self._nomsg = QLabel(i18n.tr("no metrics"))
            self._nomsg.setObjectName("muted")
            self._flow.addWidget(self._nomsg)
            return
        for k, v in self._summary.items():
            accent = theme.INK
            if k in self._good:
                accent = theme.GOOD
            elif k in self._bad:
                accent = theme.BAD
            card = MetricCard(str(k), _fmt(v), accent)
            self._flow.addWidget(card)
            self._cards.append(card)
        self._flow.invalidate()
        self.updateGeometry()

    def retranslate(self) -> None:
        if self._nomsg is not None:
            self._nomsg.setText(i18n.tr("no metrics"))
        for c in self._cards:
            c.retranslate()

    def clear(self) -> None:
        self.set_summary({})


def _fmt(v) -> str:
    if v is None:
        return i18n.tr("Not measured")
    if isinstance(v, float):
        if np.isnan(v):
            return i18n.tr("Not measured")
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v[:3]) + ("…" if len(v) > 3 else "")
    if isinstance(v, (np.floating,)):
        return f"{float(v):.4g}"
    return str(v)


# --------------------------------------------------------------------------
# pandas tables
# --------------------------------------------------------------------------
class PandasModel(QAbstractTableModel):
    def __init__(self, df: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df if df is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return self._df.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        val = self._df.iat[index.row(), index.column()]
        if isinstance(val, float):
            return "n/a" if np.isnan(val) else f"{val:.4g}"
        return str(val)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)


class TableStack(QWidget):
    """Combo-box selector over several named DataFrames, shown in a table view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        head = QHBoxLayout()
        self._title = QLabel(i18n.tr("Tables"))
        self._title.setObjectName("h2")
        self.combo = QComboBox()
        self.combo.setMinimumWidth(220)
        head.addWidget(self._title)
        head.addStretch(1)
        head.addWidget(self.combo)
        lay.addLayout(head)

        self.view = QTableView()
        self.view.setMinimumHeight(200)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionBehavior(QTableView.SelectRows)
        self.view.verticalHeader().setVisible(False)
        # Interactive, not Stretch: Stretch pinned every column to an equal share,
        # so reading one long cell meant shrinking all the others. Now the operator
        # drags the border of exactly the column they care about, and the table
        # scrolls sideways (bar or wheel) when the whole row cannot fit.
        head = self.view.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.Interactive)
        head.setHighlightSections(False)
        head.setStretchLastSection(True)
        head.setMinimumSectionSize(56)
        self.view.setTextElideMode(Qt.ElideRight)
        self.view.setWordWrap(False)
        self.view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        enable_horizontal_wheel(self.view)
        self._model = PandasModel()
        self.view.setModel(self._model)
        lay.addWidget(self.view, 1)

        self._frames: Dict[str, pd.DataFrame] = {}
        self.combo.currentIndexChanged.connect(self._on_select)

    def set_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        self._frames = dict(tables or {})
        self.combo.blockSignals(True)
        self.combo.clear()
        for name in self._frames:
            self.combo.addItem(i18n.label(name), name)
        self.combo.blockSignals(False)
        if self._frames:
            self.combo.setCurrentIndex(0)
            self._on_select(0)
        else:
            self._model.set_dataframe(pd.DataFrame())

    def retranslate(self) -> None:
        self._title.setText(i18n.tr("Tables"))
        keep = self.combo.currentData()
        self.combo.blockSignals(True)
        for i in range(self.combo.count()):
            key = self.combo.itemData(i)
            self.combo.setItemText(i, i18n.label(key))
        if keep is not None:
            idx = self.combo.findData(keep)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)

    def _on_select(self, idx: int) -> None:
        key = self.combo.itemData(idx)
        if key in self._frames:
            self._model.set_dataframe(self._frames[key])
            fit_columns(self.view)


class ConclusionPanel(QFrame):
    """The page's "conclusion" card: the layperson analysis, in the studio.

    The sentences are produced by :mod:`trustoran_ui.narrative` — the same
    generator the PDF report's ``layperson`` style prints — so what the operator
    reads on screen and what a non-specialist reads in the exported report are
    the same words, maintained once.

    The card sits at the bottom of a page body inside a :class:`PageScroll`, so
    it asks for the height its wrapped text actually needs instead of being
    squeezed by a short window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(4)

        self.title = QLabel("")
        self.title.setObjectName("h2")
        lay.addWidget(self.title)

        self.body = QLabel("")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        lay.addWidget(self.body)
        enable_height_for_width(self.body)
        enable_height_for_width(self)

        self._bullets: List[str] = []
        self._placeholder = ""
        self.retranslate()

    # -- content ----------------------------------------------------------
    def set_bullets(self, bullets: Optional[List[str]]) -> None:
        """Replace the card's contents; an empty list shows the placeholder."""
        self._bullets = [str(b) for b in (bullets or []) if str(b).strip()]
        self._render()

    def bullets(self) -> List[str]:
        return list(self._bullets)

    def retranslate(self) -> None:
        self.title.setText(i18n.tr("Conclusion"))
        self._placeholder = i18n.tr(
            "Nothing to conclude yet — run the analysis and the plain-language "
            "reading of it appears here.")
        self._render()

    # -- internals --------------------------------------------------------
    def _render(self) -> None:
        if not self._bullets:
            self.body.setText(f'<span style="color:{theme.MUTED}">'
                              f'{_escape(self._placeholder)}</span>')
            return
        items = "".join(f"<li>{_escape(b)}</li>" for b in self._bullets)
        self.body.setText(f'<ul style="margin-left:-16px">{items}</ul>')


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def section_header(title: str, subtitle: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 4)
    lay.setSpacing(2)
    t = QLabel(title)
    t.setObjectName("h1")
    lay.addWidget(t)
    s = None
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("muted")
        s.setWordWrap(True)
        lay.addWidget(s)
    w.setProperty("_h1", t)
    w.setProperty("_sub", s)
    return w


def set_section_text(w: QWidget, title: str, subtitle: str = "") -> None:
    """Update a :func:`section_header` widget's texts (used by retranslate)."""
    h1 = w.property("_h1")
    sub = w.property("_sub")
    if h1 is not None:
        h1.setText(title)
    if sub is not None and subtitle:
        sub.setText(subtitle)


def status_pill(text: str, color: str = theme.ACCENT) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background:{theme.BG}; border:1px solid {color}; color:{color};"
        f"border-radius:10px; padding:3px 10px; font-weight:600;")
    return lbl
