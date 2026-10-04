# Creates a Desktop shortcut so Ctrl+Alt+S starts SYLPH.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "RTX SYLPH.lnk"
$start = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\RTX SYLPH.lnk"

$W = New-Object -ComObject WScript.Shell
foreach ($path in @($lnkPath, $start)) {
    $dir = Split-Path -Parent $path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $s = $W.CreateShortcut($path)
    $s.TargetPath = Join-Path $here "run_sylph.bat"
    $s.WorkingDirectory = $here
    $s.WindowStyle = 7
    $s.Description = "Start RTX SYLPH (Ctrl+Alt+S)"
    $s.Hotkey = "Ctrl+Alt+S"
    $icon = Join-Path $here "assets\SYLPH_Icon.png"
    if (Test-Path $icon) { $s.IconLocation = $icon }
    $s.Save()
    Write-Host "Installed $path"
}
Write-Host "Start: Ctrl+Alt+S"
Write-Host "Stop:  click HUD then Esc/Q, or Ctrl+Alt+Q anywhere"
