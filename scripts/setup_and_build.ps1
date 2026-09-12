# Setup + build for Windows (LLVM/Clang preferred, MSVC fallback).
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_and_build.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$BuildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build" }
$BuildType = if ($env:BUILD_TYPE) { $env:BUILD_TYPE } else { "Release" }
$BuildPython = if ($env:BUILD_PYTHON) { $env:BUILD_PYTHON } else { "ON" }
$BuildTests = if ($env:BUILD_TESTS) { $env:BUILD_TESTS } else { "ON" }

function Write-Info($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }
function Die($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

function Test-Cmd($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Info "Checking dependencies"

$missing = @()
if (-not (Test-Cmd "cmake")) { $missing += "cmake" }
if (-not (Test-Cmd "git")) { $missing += "git" }
if (-not (Test-Cmd "python") -and -not (Test-Cmd "python3")) { $missing += "python" }

$cxx = $null
if (Test-Cmd "clang++") { $cxx = (Get-Command clang++).Source }
elseif (Test-Cmd "clang-cl") { $cxx = (Get-Command clang-cl).Source }
elseif (Test-Cmd "cl") { Write-Warn "LLVM clang++ not found; using MSVC cl.exe"; $cxx = (Get-Command cl).Source }
else { $missing += "clang" }

if ($missing.Count -gt 0) {
    Write-Info ("Missing: " + ($missing -join ", "))
    if (Test-Cmd "winget") {
        foreach ($pkg in $missing) {
            switch ($pkg) {
                "cmake"  { winget install -e --id Kitware.CMake --accept-package-agreements --accept-source-agreements }
                "git"    { winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements }
                "python" { winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements }
                "clang"  { winget install -e --id LLVM.LLVM --accept-package-agreements --accept-source-agreements }
            }
        }
        # Refresh PATH for current session (best-effort)
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path","User")
        if (Test-Cmd "clang++") { $cxx = (Get-Command clang++).Source }
        elseif (Test-Cmd "clang-cl") { $cxx = (Get-Command clang-cl).Source }
        elseif (Test-Cmd "cl") { $cxx = (Get-Command cl).Source }
        else { Die "C++ compiler still missing after winget install. Open a new shell and retry." }
    }
    elseif (Test-Cmd "choco") {
        choco install -y $missing
    }
    else {
        Die "Install manually then re-run: $($missing -join ', ') (winget/choco recommended)"
    }
}

Write-Info "Using C++ compiler: $cxx"

if (Test-Cmd "python") {
    python -m pip install --user -q numpy matplotlib 2>$null
} elseif (Test-Cmd "python3") {
    python3 -m pip install --user -q numpy matplotlib 2>$null
}

Write-Info "Configuring ($BuildType) into $BuildDir"
$cmakeArgs = @(
    "-S", ".",
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DMHD_BUILD_TESTS=$BuildTests",
    "-DMHD_BUILD_PYTHON=$BuildPython",
    "-DMHD_NATIVE_ARCH=ON"
)
if ($cxx -and ($cxx -match "clang")) {
    $cmakeArgs += "-DCMAKE_CXX_COMPILER=$cxx"
}
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Die "cmake configure failed" }

Write-Info "Building"
cmake --build $BuildDir --config $BuildType -j
if ($LASTEXITCODE -ne 0) { Die "build failed" }

if ($BuildTests -eq "ON") {
    Write-Info "Running tests"
    ctest --test-dir $BuildDir --output-on-failure --build-config $BuildType
    if ($LASTEXITCODE -ne 0) { Die "tests failed" }
}

Write-Info "Done"
Write-Host "  binary : $BuildDir\mhd_solver.exe (or $BuildDir\$BuildType\mhd_solver.exe)"
if ($BuildPython -eq "ON") {
    Write-Host "  python : `$env:PYTHONPATH=\"$Root\$BuildDir\python;`$env:PYTHONPATH\""
}
