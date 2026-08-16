@echo off
cd /d "%~dp0"
py -3.10 "%~dp0plugin.py"
if errorlevel 1 (
  echo.
  echo RTX SYLPH needs Python 3.10 with her packages installed.
  echo Tried: py -3.10
  pause
)
