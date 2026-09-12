# Launch the MHD Solver GUI on Windows (browser UI).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "build\python")) {
    Write-Host "Python module missing. Running setup_and_build.ps1 ..."
    & "$PSScriptRoot\setup_and_build.ps1"
}

$py = $null
if (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
elseif (Get-Command python3 -ErrorAction SilentlyContinue) { $py = "python3" }
else { throw "Python not found" }

& $py -m pip install --user -q matplotlib numpy

$env:PYTHONPATH = "$Root\build\python;" + $env:PYTHONPATH
$env:MPLBACKEND = "Agg"

if ($env:MHD_GUI -eq "tk") {
    & $py "$Root\gui\app.py"
} else {
    & $py "$Root\gui\web_app.py"
}
