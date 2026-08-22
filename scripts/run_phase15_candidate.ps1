param(
    [int]$Households = 1001,
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$RunTag = "candidate-gate",
    [int]$MaxCandidateRows = 2000000
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $repo "benchmark-data\phase9-mtc-full\prototype_mtc_extended"
$overlay = Join-Path $repo "benchmark-data\configs_phase9_choiceforge"
$activitysim = Join-Path $repo ".venv-phase8\Scripts\activitysim.exe"
$python = Join-Path $repo ".venv-phase8\Scripts\python.exe"
$candidateReports = Join-Path $repo "benchmark-results\phase15-candidate-$RunTag"
$exactReports = Join-Path $repo "benchmark-results\phase15-exact-$RunTag"
$summary = Join-Path $repo "benchmark-results\phase15-candidate-summary.json"
$outputName = "o-p15-$RunTag-$Households"
$output = Join-Path $project $outputName
$reference = Join-Path $project "o-p14-real-gate-$Households"
$stdout = Join-Path $project "phase15-$RunTag-$Households.stdout.log"
$stderr = Join-Path $project "phase15-$RunTag-$Households.stderr.log"

foreach ($path in @($output, $candidateReports, $exactReports)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite: $path" }
}
New-Item -ItemType Directory -Path $candidateReports,$exactReports | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OPENBLAS_NUM_THREADS = "24"
$env:OMP_NUM_THREADS = "24"
$env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "1"
$env:CHOICEFORGE_STRICT_CUDA_MAX_ROWS = "$MaxCandidateRows"
$env:CHOICEFORGE_PHASE15_RUN_ID = $RunTag
$env:CHOICEFORGE_PHASE15_REPORT_DIR = $candidateReports
$env:CHOICEFORGE_STRICT_CUDA_BATCHES = "1000"
$env:CHOICEFORGE_STRICT_CUDA_REPORT_DIR = $exactReports
$env:PATH = (Join-Path $repo ".venv-phase8\Scripts") + ";" + $env:PATH

$arguments = @(
    "run", "-c", $overlay, "-c", "configs_sh", "-c", "configs",
    "-d", "data_full", "-o", $outputName,
    "--households_sample_size", "$Households"
)
$process = Start-Process -FilePath $activitysim -ArgumentList $arguments `
    -WorkingDirectory $project -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "Phase 15 ActivitySim candidate failed with exit code $($process.ExitCode). See $stderr"
}
if (-not (Select-String -LiteralPath $stdout -Pattern "Time to execute all models\s*:" -Quiet)) {
    throw "Phase 15 candidate has no terminal all-model marker"
}

& $python (Join-Path $repo "scripts\summarize_phase15_candidate.py") `
    --reports $candidateReports --exact-reports $exactReports `
    --output $summary --households $Households
if ($LASTEXITCODE -ne 0) { throw "Phase 15 report summarization failed" }

if (Test-Path -LiteralPath $reference) {
    & $python (Join-Path $repo "scripts\verify_phase15_outputs.py") `
        --reference $reference --candidate $output `
        --output (Join-Path $repo "benchmark-results\phase15-output-verification.json")
    if ($LASTEXITCODE -ne 0) { throw "Phase 15 final-output verification failed" }
} else {
    Write-Warning "No Phase 14 reference output found; exact CPU/CUDA gate passed but final CSV comparison was skipped."
}
Write-Output "Phase 15 device-resident candidate gate complete: $summary"
