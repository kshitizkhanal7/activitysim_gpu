param(
    [ValidateRange(1, 5)][int]$Repetitions = 3,
    [ValidateRange(1, 500000)][int]$Households = 50000,
    [ValidatePattern("^[A-Za-z0-9-]+$")][string]$RunTag = "p34proof",
    [ValidateSet("phase17", "activitysim")][string]$Baseline = "activitysim",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_phase32_full_model_ab.ps1"
& $runner -Repetitions $Repetitions -Households $Households `
    -RunTag $RunTag -Baseline $Baseline -CandidatePhase 34 -Resume:$Resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
