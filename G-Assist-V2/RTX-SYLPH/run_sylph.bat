@echo off
cd /d "%~dp0"
echo Starting RTX SYLPH...
echo   ON     = this script, or Ctrl+Alt+S if the desktop shortcut is installed
echo   OFF    = click her HUD then Esc or Q, or Ctrl+Alt+Q anywhere
echo   GPU    = Ctrl+Alt+G
echo   COUNCIL= Ctrl+Alt+A  (opens 5in answer windows)
echo   WAKE   = say sylph, then your question
py -3.10 "%~dp0plugin.py"
if errorlevel 1 (
  echo.
  echo RTX SYLPH closed unexpectedly. The launcher auto-restarts on crash.
  echo If Python 3.10 is missing her packages, install from requirements.txt.
  pause
)
