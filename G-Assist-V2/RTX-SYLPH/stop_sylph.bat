@echo off
cd /d "%~dp0"
echo Stopping RTX SYLPH...
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'RTX-SYLPH\\\\plugin.py|RTX_SYLPH_G_Assist_V2.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host ('Stopped PID ' + $_.ProcessId) }"
echo Done.
