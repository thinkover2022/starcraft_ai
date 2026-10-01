# 1단계 파이프라인 실행 스크립트 (윈도우 PowerShell, 저장소 루트에서 실행)
#
#   .\tools\stage1.ps1 -Step masks  -Raw D:\sc_data\raw\pilot -Ann D:\sc_data\ann\pilot
#   .\tools\stage1.ps1 -Step lag    -Raw D:\sc_data\raw\pilot -Ann D:\sc_data\ann\pilot
#   .\tools\stage1.ps1 -Step review -Raw D:\sc_data\raw\pilot -Ann D:\sc_data\ann\pilot
#   .\tools\stage1.ps1 -Step build  -Raw D:\sc_data\raw -Ann D:\sc_data\ann -Dataset D:\sc_data\sc_terran_v1
#   .\tools\stage1.ps1 -Step owner  -Raw D:\sc_data\raw -Ann D:\sc_data\ann
#   .\tools\stage1.ps1 -Step train  -Dataset D:\sc_data\sc_terran_v1 -Experiment s640
#   .\tools\stage1.ps1 -Step eval   -Dataset D:\sc_data\sc_terran_v1 -Experiment s640
#   .\tools\stage1.ps1 -Step export -Experiment s640
#
# 단계 순서와 의미는 docs/stage1_runbook.md 참조.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("masks", "lag", "review", "build", "owner", "train", "eval", "export")]
    [string]$Step,
    [string]$Raw = "D:\sc_data\raw",
    [string]$Ann = "D:\sc_data\ann",
    [string]$Dataset = "D:\sc_data\sc_terran_v1",
    [string]$Gold = "",
    [string]$Experiment = "s640",
    [string]$Config = "configs\datagen.yaml",
    [double]$ReviewFraction = 0.05
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"          # 한글 출력·파일 이름을 UTF-8로 처리
$imgsz = if ($Experiment -like "*1280") { 1280 } else { 640 }
$weights = "runs\terran\$Experiment\weights\best.pt"

function Run([string[]]$cmd) {
    Write-Host ">> python $($cmd -join ' ')" -ForegroundColor Cyan
    & python @cmd
    if ($LASTEXITCODE -ne 0) { throw "실패: $($cmd -join ' ')" }
}

switch ($Step) {
    "masks"  { Run @("-m", "datagen.sam_masks.run", "--raw", $Raw, "--out", $Ann, "--config", $Config, "--skip-existing") }
    "lag"    { Run @("-m", "datagen.sam_masks.lag_check", "--raw", $Raw, "--ann", $Ann) }
    "review" { Run @("-m", "datagen.review.overlay", "--raw", $Raw, "--ann", $Ann, "--out", "$Ann\_review", "--fraction", $ReviewFraction) }
    "build"  {
        $a = @("-m", "datagen.build_dataset.build", "--raw", $Raw, "--ann", $Ann, "--out", $Dataset, "--config", $Config)
        if ($Gold) { $a += @("--gold", $Gold) }
        Run $a
    }
    "owner"  {
        Run @("-m", "perception.detect.owner", "calibrate", "--raw", $Raw, "--ann", $Ann, "--out", "configs\team_palette.json")
        Run @("-m", "perception.detect.owner", "evaluate", "--raw", $Raw, "--ann", $Ann, "--palette", "configs\team_palette.json")
    }
    "train"  { Run @("-m", "perception.detect.train", "--experiment", $Experiment, "--set", "data=$Dataset\sc_terran.yaml") }
    "eval"   { Run @("-m", "perception.detect.evaluate", "--weights", $weights, "--dataset", $Dataset, "--imgsz", $imgsz, "--out", "runs\terran\$Experiment\eval.json") }
    "export" { Run @("-m", "perception.detect.export", "--weights", $weights, "--imgsz", $imgsz, "--format", "engine", "--bench") }
}
