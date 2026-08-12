param(
    [switch]$DownloadOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DatasetRoot = Join-Path $RepoRoot "data\raw\asvspoof2019_la"
$ArchivePath = Join-Path $DatasetRoot "LA.zip"
$DownloadUrl = "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
$ExpectedSize = 7640952520
$ExpectedSha256 = "208a7e4e3913f8c75ae1afd19bf32a5b29ae68435e9e30e23e5e98b6a155e4ec"

New-Item -ItemType Directory -Force -Path $DatasetRoot | Out-Null

curl.exe `
    --location `
    --fail `
    --retry 20 `
    --retry-all-errors `
    --retry-delay 5 `
    --continue-at - `
    --output $ArchivePath `
    $DownloadUrl

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Archive = Get-Item -LiteralPath $ArchivePath
if ($Archive.Length -ne $ExpectedSize) {
    throw "LA.zip size mismatch: expected $ExpectedSize, found $($Archive.Length)"
}

$ActualSha256 = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256) {
    throw "LA.zip SHA-256 mismatch"
}

if (-not $DownloadOnly) {
    tar.exe -xf $ArchivePath -C $DatasetRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Output "ASVspoof 2019 LA is available at $DatasetRoot"
