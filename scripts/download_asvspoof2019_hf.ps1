param(
    [ValidateSet("train", "dev", "eval", "all")]
    [string]$Split = "all"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DestinationRoot = Join-Path $RepoRoot "data\raw\asvspoof2019_la\huggingface"
$Revision = "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c"
$BaseUrl = "https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA/resolve/$Revision/data"

$Shards = @(
    [PSCustomObject]@{
        Split = "train"
        RemoteName = "train-00000-of-00001.parquet"
        LocalName = "train.parquet"
        Size = 1586086073
        Sha256 = "b4eea1063bbcfa0c1cef1b69a96ad8b787c32f662005562b899cd4b461739619"
    },
    [PSCustomObject]@{
        Split = "dev"
        RemoteName = "validation-00000-of-00001.parquet"
        LocalName = "dev.parquet"
        Size = 1575535827
        Sha256 = "9d2c340bb7f04c2d63ac018b224ecdb8c1855c789247232608b650f86c189b16"
    },
    [PSCustomObject]@{
        Split = "eval"
        RemoteName = "test-00000-of-00001.parquet"
        LocalName = "eval.parquet"
        Size = 4381501164
        Sha256 = "8159935a94426bc36308278b23cecd8c4ca2a15b778a1087a4a0441d079c86af"
    }
)

New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
$SelectedShards = if ($Split -eq "all") { $Shards } else { $Shards | Where-Object Split -eq $Split }

foreach ($Shard in $SelectedShards) {
    $Destination = Join-Path $DestinationRoot $Shard.LocalName
    $Url = "$BaseUrl/$($Shard.RemoteName)?download=true"

    if ((Test-Path -LiteralPath $Destination) -and (Get-Item -LiteralPath $Destination).Length -eq $Shard.Size) {
        $ExistingHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ExistingHash -eq $Shard.Sha256) {
            Write-Output "$($Shard.Split) is already complete and verified."
            continue
        }
    }

    curl.exe `
        --location `
        --fail `
        --retry 20 `
        --retry-all-errors `
        --retry-delay 3 `
        --connect-timeout 30 `
        --continue-at - `
        --output $Destination `
        $Url

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $DownloadedFile = Get-Item -LiteralPath $Destination
    if ($DownloadedFile.Length -ne $Shard.Size) {
        throw "$($Shard.Split) size mismatch: expected $($Shard.Size), found $($DownloadedFile.Length)"
    }

    $ActualHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $Shard.Sha256) {
        throw "$($Shard.Split) SHA-256 mismatch"
    }
    Write-Output "$($Shard.Split) downloaded and verified: $Destination"
}
