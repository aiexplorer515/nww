param(
  [string]$Bundle   = "b01",
  [string]$DataRoot = "data",
  [double]$On       = 0.70,
  [double]$Off      = 0.55
)

$ErrorActionPreference = "Stop"
$root = Join-Path $DataRoot $Bundle
$env:NWW_DATA_HOME = $DataRoot
$env:NWW_BUNDLE    = $Bundle

function Step([string]$title, [scriptblock]$body) {
  Write-Host "`n== $title ==" -ForegroundColor Cyan
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $body
  $sw.Stop()
  Write-Host ("-- done in {0:N1}s" -f $sw.Elapsed.TotalSeconds)
}

function Fix-BOM-All() {
  Get-ChildItem $root -Include *.json,*.jsonl -Recurse | ForEach-Object {
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
      $clean = $bytes[3..($bytes.Length-1)]
      [System.IO.File]::WriteAllBytes($_.FullName, $clean)
      Write-Host "BOM removed ->" $_.FullName
    }
  }
}

function Get-LineCount([string]$Path) {
  if (Test-Path $Path) {
    $arr = Get-Content $Path -ReadCount 0
    if ($arr -is [array]) { return $arr.Count } else { return 0 }
  } else {
    return 0
  }
}

# 0) BOM 정리
Step "BOM cleanup" { Fix-BOM-All }

# 1) keywords → lexicon boost → frames
Step "keywords" {
  python -m nwwpkg.preprocess.keyword `
    --in  (Join-Path $root "clean.jsonl") `
    --out (Join-Path $root "kyw.jsonl") `
    --topk 12
}
Step "lexicon boost" {
  python -m nwwpkg.rules.lexicon_boost `
    --in    (Join-Path $root "kyw.jsonl") `
    --rules "rules\checklist.csv" `
    --out   (Join-Path $root "kyw_boosted.jsonl")
}
Step "frame classify" {
  python -m nwwpkg.rules.frame_classifier `
    --in  (Join-Path $root "kyw_boosted.jsonl") `
    --out (Join-Path $root "frames.jsonl") `
    --on  0.45
}

# 2) frames → scores → alerts
Step "scores" {
  python tools\make_scores_from_frames.py `
    (Join-Path $root "frames.jsonl") `
    (Join-Path $root "scores.jsonl")
}
Step "alerts" {
  python tools\make_alerts_from_scores.py `
    (Join-Path $root "scores.jsonl") `
    $On $Off `
    (Join-Path $root "alerts.jsonl")
}

# 3) BOM 재확인
Step "BOM cleanup (post)" { Fix-BOM-All }

# 4) 요약(PS 5.1 호환)
Step "quick counts" {
  foreach ($n in @('clean','kyw','kyw_boosted','frames','scores','alerts')) {
    $p = Join-Path $root "$n.jsonl"
    $c = Get-LineCount $p
    "{0,-12} -> {1}" -f $n, $c
  }

  $pf = Join-Path $root "frames.jsonl"
  if (Test-Path $pf) {
    Get-Content $pf -TotalCount 3 | ForEach-Object {
      try {
        $o = $_ | ConvertFrom-Json
        "frames(sample): " + ($o.frames.Count)
      } catch {
        Write-Warning "JSON sample parse failed"
      }
    }
  }
}

# 5) MVP 자동 점검
Step "MVP check" {
  python tools\mvp_check.py
}
