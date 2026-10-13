$ErrorActionPreference = 'Stop'
$toolRoot = Join-Path $PSScriptRoot '.tools/ffmpeg'
if ((Test-Path "$toolRoot/bin/ffmpeg.exe") -and (Test-Path "$toolRoot/bin/ffprobe.exe")) {
    Write-Host 'Project FFmpeg and ffprobe are already installed.'
    return
}
$downloadRoot = Join-Path $PSScriptRoot '.cache/ffmpeg'
New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
$archive = Join-Path $downloadRoot 'ffmpeg.zip'
$url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
curl.exe --fail --location --silent --show-error --output $archive $url
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg download failed.' }
$checksum = (Invoke-RestMethod -Uri "$url.sha256").Trim().Split()[0]
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $checksum) {
    throw 'FFmpeg archive checksum mismatch.'
}
$expanded = Join-Path $downloadRoot 'expanded'
Expand-Archive -LiteralPath $archive -DestinationPath $expanded -Force
$package = Get-ChildItem -LiteralPath $expanded -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'bin/ffmpeg.exe') } | Select-Object -First 1
if (-not $package) { throw 'FFmpeg archive does not contain the expected binaries.' }
New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null
Copy-Item -Path (Join-Path $package.FullName '*') -Destination $toolRoot -Recurse -Force
$versionOutput = & "$toolRoot/bin/ffmpeg.exe" -version
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg execution failed.' }
Write-Host $versionOutput[0]
$versionOutput = & "$toolRoot/bin/ffprobe.exe" -version
if ($LASTEXITCODE -ne 0) { throw 'ffprobe execution failed.' }
Write-Host $versionOutput[0]
