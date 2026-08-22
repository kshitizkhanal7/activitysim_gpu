param(
    [int]$Households = 1001,
    [ValidateRange(1, 5)]
    [int]$Repetitions = 3,
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$RunTag = "p17-ab",
    [int]$MaxCandidateRows = 2000000,
    [ValidateRange(1, 24)]
    [int]$BlasThreads = 16,
    [switch]$EnableModeChoice,
    [switch]$EnableReusableBuffers,
    [switch]$RequireComponentPromotion,
    [switch]$RequirePromotion
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$candidateReports = Join-Path $repo "benchmark-results\phase17-ab-$RunTag-reports"
$modeReports = Join-Path $repo "benchmark-results\phase17-ab-$RunTag-mode-reports"
if ((Test-Path -LiteralPath $candidateReports) -or (Test-Path -LiteralPath $modeReports)) {
    throw "Refusing to overwrite Phase 17 reports: $candidateReports or $modeReports"
}
New-Item -ItemType Directory -Path $candidateReports,$modeReports | Out-Null

$env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "1"
$env:CHOICEFORGE_STRICT_CUDA_MAX_ROWS = "$MaxCandidateRows"
$env:CHOICEFORGE_STRICT_CUDA_TILE_ROWS = "1"
$env:CHOICEFORGE_STRICT_CUDA_LOCALITY = "1"
$env:CHOICEFORGE_STRICT_CUDA_COMPACT_INPUTS = "1"
$env:CHOICEFORGE_STRICT_CUDA_GROUPED_INDICES = "1"
$env:CHOICEFORGE_STRICT_CUDA_SPARSE_COEFFICIENTS = "0"
$env:CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32 = "1"
$env:CHOICEFORGE_STRICT_CUDA_PERSISTENT_PLAN = "1"
$env:CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS = $(if ($EnableReusableBuffers) { "1" } else { "0" })
$env:CHOICEFORGE_STRICT_CUDA_MODE_CHOICE = $(if ($EnableModeChoice) { "1" } else { "0" })
$env:CHOICEFORGE_PHASE17_RUN_ID = $RunTag
$env:CHOICEFORGE_PHASE17_REPORT_DIR = $candidateReports
$env:CHOICEFORGE_PHASE17_MODE_REPORT_DIR = $modeReports
& (Join-Path $repo "scripts\run_phase9_mtc_full_ab.ps1") `
    -Households $Households -Repetitions $Repetitions -RunTag $RunTag `
    -BlasThreads $BlasThreads `
    -PhaseLabel "17-persistent-plan" -BaselineUsesChoiceForge `
    -BaselineDescription "Phase 11 destination batching and CUDA nested reduction; generated utility candidate disabled" `
    -OptimizationDescription "Phase 17 persistent compiled FP32 utility plans with fail-closed ABI validation, compact inputs, grouped skim indices, reusable device/output workspaces $EnableReusableBuffers, trip-mode continuation $EnableModeChoice, and candidate max rows $MaxCandidateRows"
if ($LASTEXITCODE -ne 0) { throw "Phase 17 incremental runner failed" }

$manifest = Join-Path $repo "benchmark-results\phase9-mtc-full-$RunTag-runs.json"
$summary = Join-Path $repo "benchmark-results\phase17-$RunTag-summary.json"
$arguments = @(
    (Join-Path $repo "benchmarks\benchmark_phase15_candidate.py"),
    "--phase", "17", "--manifest", $manifest, "--output", $summary,
    "--candidate-reports", $candidateReports, "--mode-reports", $modeReports
)
if ($RequirePromotion) { $arguments += "--require-promotion" }
if ($RequireComponentPromotion) { $arguments += "--require-component-promotion" }
& (Join-Path $repo ".venv-phase8\Scripts\python.exe") @arguments
if ($LASTEXITCODE -ne 0) { throw "Phase 17 summarization or promotion gate failed" }
Write-Output "Phase 17 persistent-plan A/B complete: $summary"
