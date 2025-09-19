Param([string]$Bundle="b01",[int]$N=400)
$base = "data\bundles\$Bundle"
$raw  = Join-Path $base "raw.jsonl"
$sample = Join-Path $base "raw.sample.jsonl"
Get-Content $raw -Encoding UTF8 | Select-Object -First $N | Set-Content $sample -Encoding UTF8
Write-Host "Sample -> $sample (First $N lines)"
