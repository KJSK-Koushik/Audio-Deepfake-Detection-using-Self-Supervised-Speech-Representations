param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $env:LOCALAPPDATA "audio-deepfake-detection-venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipInstall) {
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-dev.txt")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $VenvPython (Join-Path $RepoRoot "scripts\check_environment.py") --strict
exit $LASTEXITCODE
