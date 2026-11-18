@echo off
rem Double-click to open Lockoff in your browser. Close this window to stop it.
title Lockoff
set PYTHONUTF8=1
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Lockoff is not set up yet. Run setup.bat first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -c "from lockoff.ui import serve; serve()"
pause
