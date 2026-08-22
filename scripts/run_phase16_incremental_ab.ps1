param(
    [int]$Households = 1001,
    [ValidateRange(1, 5)]
    [int]$Repetitions = 3,
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$RunTag = "p16-ab",
    [int]$MaxCandidateRows = 2000000,
    [ValidateSet(1, 2, 4, 8)]
    [int]$TileRows = 1,
    [switch]$DisableCompactInputs,
    [switch]$DisableGroupedIndices,
    [switch]$EnableSparseCoefficients,
    [switch]$EnableFloat32Expressions,
    [switch]$RequireComponentPromotion,
    [switch]$RequirePromotion
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$candidateReports = Join-Path $repo "benchmark-results\phase16-ab-$RunTag-reports"
if (Test-Path -LiteralPath $candidateReports) {
    throw "Refusing to overwrite Phase 16 reports: $candidateReports"
}
New-Item -ItemType Directory -Path $candidateReports | Out-Null

$env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "1"
$env:CHOICEFORGE_STRICT_CUDA_MAX_ROWS = "$MaxCandidateRows"
$env:CHOICEFORGE_STRICT_CUDA_TILE_ROWS = "$TileRows"
$env:CHOICEFORGE_STRICT_CUDA_LOCALITY = "1"
$env:CHOICEFORGE_STRICT_CUDA_COMPACT_INPUTS = $(
    if ($DisableCompactInputs) { "0" } else { "1" }
)
$env:CHOICEFORGE_STRICT_CUDA_GROUPED_INDICES = $(
    if ($DisableGroupedIndices) { "0" } else { "1" }
)
$env:CHOICEFORGE_STRICT_CUDA_SPARSE_COEFFICIENTS = $(
    if ($EnableSparseCoefficients) { "1" } else { "0" }
)
$env:CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32 = $(
    if ($EnableFloat32Expressions) { "1" } else { "0" }
)
$env:CHOICEFORGE_PHASE16_RUN_ID = $RunTag
$env:CHOICEFORGE_PHASE16_REPORT_DIR = $candidateReports
& (Join-Path $repo "scripts\run_phase9_mtc_full_ab.ps1") `
    -Households $Households -Repetitions $Repetitions -RunTag $RunTag `
    -PhaseLabel "16-tiled" -BaselineUsesChoiceForge `
    -BaselineDescription "Phase 11 destination batching and CUDA nested reduction; strict utility candidate disabled" `
    -OptimizationDescription "Phase 16 locality strict-CUDA utilities; tile rows $TileRows; compact inputs $(-not $DisableCompactInputs); grouped skim indices $(-not $DisableGroupedIndices); experimental sparse coefficients $EnableSparseCoefficients; FP32 expressions $EnableFloat32Expressions; candidate max rows $MaxCandidateRows"
if ($LASTEXITCODE -ne 0) { throw "Phase 16 incremental runner failed" }

$manifest = Join-Path $repo "benchmark-results\phase9-mtc-full-$RunTag-runs.json"
$summary = Join-Path $repo "benchmark-results\phase16-$RunTag-summary.json"
$arguments = @(
    (Join-Path $repo "benchmarks\benchmark_phase15_candidate.py"),
    "--phase", "16", "--manifest", $manifest, "--output", $summary,
    "--candidate-reports", $candidateReports
)
if ($RequirePromotion) { $arguments += "--require-promotion" }
if ($RequireComponentPromotion) { $arguments += "--require-component-promotion" }
& (Join-Path $repo ".venv-phase8\Scripts\python.exe") @arguments
if ($LASTEXITCODE -ne 0) { throw "Phase 16 summarization or promotion gate failed" }
Write-Output "Phase 16 direct incremental A/B complete: $summary"
