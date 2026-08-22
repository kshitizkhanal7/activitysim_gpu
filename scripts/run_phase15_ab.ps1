param(
    [int]$Households = 50000,
    [ValidateRange(1, 5)]
    [int]$Repetitions = 3,
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$RunTag = "strict-candidate",
    [int]$MaxCandidateRows = 100000,
    [switch]$RequirePromotion
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$candidateReports = Join-Path $repo "benchmark-results\phase15-ab-$RunTag-reports"
if (Test-Path -LiteralPath $candidateReports) {
    throw "Refusing to overwrite Phase 15 A/B reports: $candidateReports"
}
New-Item -ItemType Directory -Path $candidateReports | Out-Null

$env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "1"
$env:CHOICEFORGE_STRICT_CUDA_MAX_ROWS = "$MaxCandidateRows"
$env:CHOICEFORGE_PHASE15_REPORT_DIR = $candidateReports
& (Join-Path $repo "scripts\run_phase9_mtc_full_ab.ps1") `
    -Households $Households -Repetitions $Repetitions -RunTag $RunTag `
    -PhaseLabel "15" `
    -OptimizationDescription "Phase 11 destination batching plus Phase 15 device-resident strict-IR generated CUDA utilities and nested logsum; candidate max rows $MaxCandidateRows"
if ($LASTEXITCODE -ne 0) { throw "Phase 15 A/B runner failed" }

$manifest = Join-Path $repo "benchmark-results\phase9-mtc-full-$RunTag-runs.json"
$summary = Join-Path $repo "benchmark-results\phase15-$RunTag-summary.json"
$benchmarkArguments = @(
    (Join-Path $repo "benchmarks\benchmark_phase15_candidate.py"),
    "--manifest", $manifest,
    "--output", $summary,
    "--candidate-reports", $candidateReports
)
if ($RequirePromotion) { $benchmarkArguments += "--require-promotion" }
& (Join-Path $repo ".venv-phase8\Scripts\python.exe") $benchmarkArguments
if ($LASTEXITCODE -ne 0) { throw "Phase 15 A/B summarization failed" }
Write-Output "Phase 15 repeated A/B gate complete: $summary"
