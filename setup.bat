@echo off
rem Lockoff setup: run once. Needs internet, Python 3 on PATH (only to bootstrap uv) and FFmpeg on PATH.
setlocal
cd /d "%~dp0"

where ffmpeg >nul 2>nul || (
  echo FFmpeg was not found. Install it, then run setup again:
  echo     winget install Gyan.FFmpeg
  pause & exit /b 1
)
where python >nul 2>nul || (
  echo Python was not found. Install Python 3 from https://www.python.org/downloads/ then run setup again.
  pause & exit /b 1
)

echo [1/4] Installing uv, a fast Python package manager...
python -m pip install --user --quiet uv || goto :fail

echo [2/4] Installing a private Python 3.12 into .python ...
set UV_PYTHON_INSTALL_DIR=%~dp0.python
python -m uv python install 3.12 || goto :fail

echo [3/4] Creating .venv and installing packages (about 2 GB, mostly GPU libraries)...
if exist .venv rmdir /s /q .venv
python -m uv venv --python 3.12 .venv || goto :fail
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail

echo [4/4] GPU video encoding...
where nvidia-smi >nul 2>nul && (
  if not exist tools\ffmpeg-gpu\7.0.2\bin\ffmpeg.exe (
    echo   NVIDIA GPU found. Fetching FFmpeg 7.0.2, which can GPU-encode on older drivers too...
    powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $z=\"$env:TEMP\ff702.zip\"; Invoke-WebRequest 'https://github.com/GyanD/codexffmpeg/releases/download/7.0.2/ffmpeg-7.0.2-essentials_build.zip' -OutFile $z -UseBasicParsing; Expand-Archive $z \"$env:TEMP\ff702\" -Force; New-Item -ItemType Directory -Force 'tools\ffmpeg-gpu\7.0.2' | Out-Null; Copy-Item \"$env:TEMP\ff702\ffmpeg-7.0.2-essentials_build\bin\" 'tools\ffmpeg-gpu\7.0.2\' -Recurse -Force; Remove-Item 'tools\ffmpeg-gpu\7.0.2\bin\ffplay.exe' -ErrorAction SilentlyContinue; Remove-Item $z, \"$env:TEMP\ff702\" -Recurse -Force"
  )
) || echo   No NVIDIA GPU found. Video will be encoded on the CPU.

echo.
call lockoff.bat doctor
echo.
echo Done. Double-click "Start Lockoff.bat" to open the app.
pause
exit /b 0

:fail
echo SETUP FAILED - see the message above.
pause
exit /b 1
