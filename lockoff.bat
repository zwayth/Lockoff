@echo off
rem Lockoff command line. Example: lockoff run "https://youtube.com/watch?v=..." --clips 5
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" -m lockoff %*
