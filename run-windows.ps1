$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:NO_TORCH_COMPILE = '1'
$env:PATH = (Join-Path $PSScriptRoot '.tools/ffmpeg/bin') + ';' + $env:PATH
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    throw 'Run setup-windows.ps1 first.'
}
& ./.venv/Scripts/python.exe check_model_access.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& ./.venv/Scripts/python.exe run_csm.py
if ($LASTEXITCODE -ne 0) {
    Write-Host 'CSM generation failed. Check the error above and SETUP-WINDOWS.md.'
    exit $LASTEXITCODE
}
