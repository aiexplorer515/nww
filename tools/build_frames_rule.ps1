param(
  [string]$Bundle = "b01",
  [string]$DataRoot = "data"
)

$ErrorActionPreference = "Stop"

$B   = Join-Path $DataRoot $Bundle
$in  = Join-Path $B "clean.jsonl"
$out = Join-Path $B "frames.jsonl"

if (!(Test-Path $in)) { throw "not found: $in" }
if (Test-Path $out) { Remove-Item $out -Force }

# 규칙: 정규식 X, '포함(like)' 기반(PS5 안전)
$Rules = @(
  @{ label="군사/무력(Military_Conflict)";   keywords=@("군사","미사일","포병","포격","병력","전개","훈련","무력","교전","무장") }
  @{ label="정치/외교 긴장(Political_Tension)"; keywords=@("긴장","갈등","충돌","제재","압박","비난","결렬","중단","파기") }
  @{ label="재난/사고(Disaster_Accident)";    keywords=@("사망","부상","피해","화재","폭발","붕괴","침몰","참사") }
)

function Get-Text([object]$r) {
    if ($r.PSObject.Properties.Name -contains 'clean_text' -and $r.clean_text) { return [string]$r.clean_text }
    elseif ($r.PSObject.Properties.Name -contains 'text' -and $r.text) { return [string]$r.text }
    else { return "" }
}
function Count-Hit([string]$text,[string[]]$keywords) {
    $t = if ($null -eq $text) { "" } else { $text.ToLower() }
    $c = 0
    foreach ($k in $keywords) {
        if ($null -ne $k -and $k -ne "") {
            $kk = $k.ToLower()
            if ($t -like ("*{0}*" -f $kk)) { $c++ }
        }
    }
    return $c
}

$docs=0; $zero=0
Get-Content -LiteralPath $in -Encoding UTF8 | ForEach-Object {
    if ($_ -eq $null -or $_ -eq "") { return }
    try { $r = $_ | ConvertFrom-Json } catch { return }

    $text = Get-Text $r
    $frames = @()
    foreach ($rule in $Rules) {
        $hits = Count-Hit $text $rule.keywords
        if ($hits -gt 0) {
            $conf = [Math]::Min(0.2 + (0.1 * $hits), 0.9)
            $frames += @{ label=$rule.label; conf=[math]::Round($conf,3); by="keyword" }
        }
    }
    if ($frames.Count -eq 0) { $zero++ }

    ($obj = @{ id=$r.id; frames=$frames }) | ConvertTo-Json -Depth 5 -Compress | Add-Content -Path $out -Encoding UTF8
    $docs++
}

Write-Host ("wrote: {0}  docs={1}  zero_or_missing={2}" -f $out,$docs,$zero)
