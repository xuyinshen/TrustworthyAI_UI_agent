#!/usr/bin/env python
"""Launch the Trustworthy-AI / O-RAN SMO studio.

Run with the project interpreter, e.g.::

    /c/ProgramData/anaconda3/python.exe QtUI/main.py

The ``py/`` folder (the ``trustoran`` engineering package) and the ``QtUI/``
folder (this front-end) are both added to ``sys.path`` so the UI imports the
exact same code the CLI and notebooks use.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../可信AI/QtUI
REPO = HERE.parent                              # .../可信AI
PY_DIR = REPO / "py"

for _p in (str(PY_DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from trustoran_ui import theme
    from trustoran_ui.app import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Trustworthy AI for Open RAN")
    app.setFont(theme.base_font())
    app.setStyleSheet(theme.stylesheet())
    theme.apply_mpl_font()

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
