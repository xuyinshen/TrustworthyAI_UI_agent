"""Hyperparameter tuning dialog.

Builds a small form on the fly from a model's ``PARAM_SCHEMA`` (declared in
``trustoran.models.factory``) so the user can override the key hyperparameters
of whichever model is selected in the toolbar. Returns a ``{param: value}`` dict
that is merged on top of the zoo defaults by ``make_model``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from . import theme
from . import i18n

try:
    from trustoran.models.factory import param_schema, default_params, model_label
except Exception:                            # pragma: no cover
    param_schema = lambda n: {}              # noqa: E731
    default_params = lambda n: {}            # noqa: E731
    model_label = lambda n: n                # noqa: E731


def _decimals(step: float) -> int:
    s = f"{step:.6f}".rstrip("0")
    return max(2, len(s.split(".")[1]) if "." in s else 2)


class ParamDialog(QDialog):
    """Modal editor for one model's tunable hyperparameters."""

    def __init__(self, model_name: str, overrides: Optional[Dict[str, Any]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model_name = model_name
        self._defaults = default_params(model_name)
        self._schema: Dict[str, Dict[str, Any]] = param_schema(model_name)
        self._overrides = dict(overrides or {})
        self._widgets: Dict[str, QWidget] = {}

        self.setObjectName("paramDialog")
        self.setMinimumWidth(360)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        title = QLabel(i18n.tr("Model parameters") + f" — {model_label(model_name)}")
        title.setObjectName("h2")
        lay.addWidget(title)

        if not self._schema:
            hint = QLabel(i18n.tr("This model has no exposed tunable parameters."))
            hint.setObjectName("muted")
            hint.setWordWrap(True)
            lay.addWidget(hint)
        else:
            form = QFormLayout()
            form.setSpacing(8)
            form.setLabelAlignment(Qt.AlignLeft)
            for pname, spec in self._schema.items():
                form.addRow(self._label(pname), self._field(pname, spec))
            lay.addLayout(form)

        self._hint = QLabel(i18n.tr("Changes apply to the next run."))
        self._hint.setObjectName("muted")
        lay.addWidget(self._hint)

        # buttons: Reset | (stretch) | Cancel | OK
        row = QHBoxLayout()
        self._reset_btn = QPushButton(i18n.tr("Reset to defaults"))
        self._reset_btn.clicked.connect(self._reset)
        row.addWidget(self._reset_btn)
        row.addStretch(1)
        self._bb = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._bb.accepted.connect(self.accept)
        self._bb.rejected.connect(self.reject)
        row.addWidget(self._bb)
        lay.addLayout(row)

        self.retranslate()

    # -- construction helpers ------------------------------------------------
    def _label(self, pname: str) -> QLabel:
        lbl = QLabel(pname)
        lbl.setToolTip(i18n.tr("Hyperparameter") + f": {pname}")
        return lbl

    def _initial(self, pname: str) -> Any:
        if pname in self._overrides:
            return self._overrides[pname]
        return self._defaults.get(pname)

    def _field(self, pname: str, spec: Dict[str, Any]) -> QWidget:
        kind = spec.get("kind", "int")
        init = self._initial(pname)
        if kind == "choice":
            w = QComboBox()
            choices: List[Any] = list(spec.get("choices", []))
            for c in choices:
                w.addItem(str(c), c)
            i = w.findData(init)
            w.setCurrentIndex(i if i >= 0 else 0)
        elif kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(_decimals(float(spec.get("step", 0.1))))
            w.setRange(float(spec.get("min", 0.0)), float(spec.get("max", 1e6)))
            w.setSingleStep(float(spec.get("step", 0.1)))
            try:
                w.setValue(float(init if init is not None else spec.get("min", 0.0)))
            except (TypeError, ValueError):
                pass
        else:  # int
            w = QSpinBox()
            w.setRange(int(spec.get("min", 0)), int(spec.get("max", 1e6)))
            w.setSingleStep(int(spec.get("step", 1)))
            try:
                w.setValue(int(init if init is not None else spec.get("min", 0)))
            except (TypeError, ValueError):
                pass
        self._widgets[pname] = w
        return w

    # -- behaviour -----------------------------------------------------------
    def _reset(self) -> None:
        for pname, spec in self._schema.items():
            w = self._widgets.get(pname)
            if w is None:
                continue
            dv = self._defaults.get(pname)
            if isinstance(w, QComboBox):
                i = w.findData(dv)
                if i >= 0:
                    w.setCurrentIndex(i)
            elif isinstance(w, QDoubleSpinBox) and dv is not None:
                w.setValue(float(dv))
            elif isinstance(w, QSpinBox) and dv is not None:
                w.setValue(int(dv))

    def values(self) -> Dict[str, Any]:
        """Current form values (only the schema fields)."""
        out: Dict[str, Any] = {}
        for pname, w in self._widgets.items():
            if isinstance(w, QComboBox):
                out[pname] = w.currentData()
            elif isinstance(w, QDoubleSpinBox):
                out[pname] = w.value()
            elif isinstance(w, QSpinBox):
                out[pname] = w.value()
        return out

    # -- i18n ----------------------------------------------------------------
    def retranslate(self) -> None:
        t = self.findChild(QLabel)
        if t is not None:
            t.setText(i18n.tr("Model parameters") + f" — {model_label(self.model_name)}")
        self._reset_btn.setText(i18n.tr("Reset to defaults"))
        self._hint.setText(i18n.tr("Changes apply to the next run."))
        ok = self._bb.button(QDialogButtonBox.Ok)
        cancel = self._bb.button(QDialogButtonBox.Cancel)
        if ok is not None:
            ok.setText(i18n.tr("OK"))
        if cancel is not None:
            cancel.setText(i18n.tr("Cancel"))


def params_are_tuned(model_name: str, overrides: Optional[Dict[str, Any]]) -> bool:
    """True when any schema override differs from the model's shipped default."""
    if not overrides:
        return False
    defaults = default_params(model_name)
    for pname in param_schema(model_name):
        if pname in overrides and overrides[pname] != defaults.get(pname):
            return True
    return False
