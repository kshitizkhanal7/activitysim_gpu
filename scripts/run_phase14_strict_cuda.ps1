param(
    [int]$Households = 1001,
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$RunTag = "real-gate"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $repo "benchmark-data\phase9-mtc-full\prototype_mtc_extended"
$overlay = Join-Path $repo "benchmark-data\configs_phase9_choiceforge"
$activitysim = Join-Path $repo ".venv-phase8\Scripts\activitysim.exe"
$python = Join-Path $repo ".venv-phase8\Scripts\python.exe"
$reportDir = Join-Path $repo "benchmark-results\phase14-strict-cuda-$RunTag"
$summary = Join-Path $repo "benchmark-results\phase14-strict-cuda-summary.json"
$outputName = "o-p14-$RunTag-$Households"
$output = Join-Path $project $outputName
$stdout = Join-Path $project "phase14-$RunTag-$Households.stdout.log"
$stderr = Join-Path $project "phase14-$RunTag-$Households.stderr.log"

if (Test-Path -LiteralPath $output) {
    throw "Refusing to overwrite ActivitySim output: $output"
}
if (Test-Path -LiteralPath $reportDir) {
    throw "Refusing to overwrite strict CUDA reports: $reportDir"
}
New-Item -ItemType Directory -Path $reportDir | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OPENBLAS_NUM_THREADS = "24"
$env:OMP_NUM_THREADS = "24"
$env:CHOICEFORGE_STRICT_CUDA_BATCHES = "1000"
$env:CHOICEFORGE_STRICT_CUDA_REPORT_DIR = $reportDir
$env:PATH = (Join-Path $repo ".venv-phase8\Scripts") + ";" + $env:PATH

$arguments = @(
    "run",
    "-c", $overlay,
    "-c", "configs_sh",
    "-c", "configs",
    "-d", "data_full",
    "-o", $outputName,
    "--households_sample_size", "$Households"
)
$process = Start-Process `
    -FilePath $activitysim `
    -ArgumentList $arguments `
    -WorkingDirectory $project `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
if ($process.ExitCode -ne 0) {
    throw "Phase 14 ActivitySim run failed with exit code $($process.ExitCode). See $stderr"
}
if (-not (Select-String -LiteralPath $stdout -Pattern "Time to execute all models\s*:" -Quiet)) {
    throw "Phase 14 ActivitySim run has no terminal all-model marker"
}
& $python (Join-Path $repo "scripts\summarize_phase14_strict_cuda.py") `
    --reports $reportDir `
    --output $summary `
    --households $Households
if ($LASTEXITCODE -ne 0) {
    throw "Phase 14 report summarization failed"
}
Write-Output "Phase 14 strict CUDA real-batch gate complete: $summary"
