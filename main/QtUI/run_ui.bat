@echo off
REM Launch the Trustworthy AI / O-RAN SMO studio (Qt UI).
REM Pure ASCII on purpose -- Windows .bat files must not contain non-ASCII bytes.
setlocal
set "PY=C:\ProgramData\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
cd /d "%~dp0"
"%PY%" main.py
endlocal
