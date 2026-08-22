param(
    [int]$Households = 1001,
    [ValidateRange(1, 5)]
    [int]$Repetitions = 3,
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$RunTag = "p15-incremental",
    [int]$MaxCandidateRows = 2000000
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$candidateReports = Join-Path $repo "benchmark-results\phase15-ab-$RunTag-reports"
if (Test-Path -LiteralPath $candidateReports) {
    throw "Refusing to overwrite Phase 15 incremental reports: $candidateReports"
}
New-Item -ItemType Directory -Path $candidateReports | Out-Null

$env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "1"
$env:CHOICEFORGE_STRICT_CUDA_MAX_ROWS = "$MaxCandidateRows"
$env:CHOICEFORGE_PHASE15_REPORT_DIR = $candidateReports
& (Join-Path $repo "scripts\run_phase9_mtc_full_ab.ps1") `
    -Households $Households -Repetitions $Repetitions -RunTag $RunTag `
    -PhaseLabel "15-incremental" -BaselineUsesChoiceForge `
    -BaselineDescription "Phase 11 ChoiceForge destination batching and CUDA nested reduction; strict utility candidate disabled" `
    -OptimizationDescription "identical Phase 11 stack plus Phase 15 device-resident strict-CUDA utility generation; candidate max rows $MaxCandidateRows"
if ($LASTEXITCODE -ne 0) { throw "Phase 15 incremental runner failed" }

$manifest = Join-Path $repo "benchmark-results\phase9-mtc-full-$RunTag-runs.json"
$summary = Join-Path $repo "benchmark-results\phase15-$RunTag-summary.json"
& (Join-Path $repo ".venv-phase8\Scripts\python.exe") `
    (Join-Path $repo "benchmarks\benchmark_phase15_candidate.py") `
    --manifest $manifest --output $summary --candidate-reports $candidateReports
if ($LASTEXITCODE -ne 0) { throw "Phase 15 incremental summarization failed" }
Write-Output "Phase 15 direct incremental A/B complete: $summary"
