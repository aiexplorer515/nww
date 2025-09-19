param(
  [string]$Bundle = "b01",
  [string]$DataRoot = "data",
  [double]$On = 0.70,
  [double]$Off = 0.55
)

$ErrorActionPreference = "Stop"
$ROOT = Join-Path $DataRoot $Bundle
Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Pop-Location | Out-Null  # 위치 보정만

# 1) 룰 기반 프레임
& "$PSScriptRoot\build_frames_rule.ps1" -Bundle $Bundle -DataRoot $DataRoot

# 2) 프레임 → 점수
python "$PSScriptRoot\make_scores_from_frames.py" "$ROOT\frames.jsonl" "$ROOT\scores.jsonl"

# 3) 점수 → 경보
python "$PSScriptRoot\make_alerts_from_scores.py" "$ROOT\scores.jsonl" $On $Off "$ROOT\alerts.jsonl"

# 4) 자동 점검 리포트
python "$PSScriptRoot\mvp_check.py"
