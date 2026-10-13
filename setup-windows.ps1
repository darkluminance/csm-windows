$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache/uv'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv or use the manual commands in SETUP-WINDOWS.md.'
}
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    uv venv .venv --python python
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
uv pip install --python .venv/Scripts/python.exe torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
uv pip install --python .venv/Scripts/python.exe -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
uv pip check --python .venv/Scripts/python.exe
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
$env:NO_TORCH_COMPILE = '1'
& (Join-Path $PSScriptRoot 'setup-ffmpeg.ps1')
Write-Host 'Setup finished. See SETUP-WINDOWS.md for Hugging Face login and generation.'
