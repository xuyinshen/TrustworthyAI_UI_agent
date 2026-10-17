@echo off
setlocal
REM ---------------------------------------------------------------------------
REM Launch the Trust-oRAN desktop workbench from this project snapshot.
REM
REM Double-clicking this file opens the PySide6 UI. It needs the project's
REM Python interpreter (NumPy / pandas / scikit-learn / matplotlib / PySide6 /
REM reportlab). Override the interpreter by setting PY before calling:
REM     set PY=C:\path\to\python.exe
REM
REM Data is NOT duplicated here: when the snapshot sits inside the repository,
REM TRUSTORAN_HOME is pointed at the parent folder so dataset\ and results\ are
REM read from there. If you move this folder somewhere without that parent, the
REM UI still starts; analyses that need raw data will report missing files.
REM
REM This file is intentionally pure ASCII.
REM ---------------------------------------------------------------------------

set "HERE=%~dp0"
if not defined PY set "PY=C:\ProgramData\anaconda3\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python interpreter not found: "%PY%"
    echo         Set the PY environment variable, then run this file again.
    pause
    exit /b 1
)

REM Prefer an explicit TRUSTORAN_HOME; otherwise fall back to the repository
REM root when this snapshot is still inside it.
if not defined TRUSTORAN_HOME (
    if exist "%HERE%..\dataset\*" set "TRUSTORAN_HOME=%HERE%.."
)
if defined TRUSTORAN_HOME (
    echo [info] data root: "%TRUSTORAN_HOME%"
) else (
    echo [info] data root: default - no dataset\ found next to this folder
)

cd /d "%HERE%"
"%PY%" -X utf8 "%HERE%QtUI\main.py"
if errorlevel 1 (
    echo.
    echo [ERROR] The workbench exited with an error - see the lines above.
    pause
)
endlocal
