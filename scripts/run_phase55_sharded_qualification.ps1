param(
    [ValidateRange(1, 3)][int]$Repetitions = 3,
    [ValidatePattern("^[A-Za-z0-9-]+$")][string]$RunTag = "p55final",
    [switch]$CaptureAtlas
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repo ".venv-phase8\Scripts\python.exe"
$project = Join-Path $repo "benchmark-data\phase9-mtc-full\prototype_mtc_extended"
$data = Join-Path $project "data_full"
$overlay = Join-Path $repo "benchmark-data\configs_phase33_choiceforge"
$sharrow = Join-Path $project "configs_sh"
$reference = Join-Path $project "o-p17modeproof16-baseline-50000-1\pipeline.parquetpipeline"
$checkpointSource = Join-Path $project "o-p35native1-gpu-50000"
$postOutput = Join-Path $project "o-$RunTag-post-work-50000"
$runner = Join-Path $repo "scripts\run_phase22_integrated_scheduling.py"

foreach ($required in @($python, $project, $data, $overlay, $sharrow, $reference, $checkpointSource, $runner)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing Phase 55 input: $required" }
}
if (-not (Test-Path -LiteralPath $postOutput)) {
    Copy-Item -LiteralPath $checkpointSource -Destination $postOutput -Recurse
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMBA_NUM_THREADS = "1"
$env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "1"
$env:CHOICEFORGE_STRICT_CUDA_MAX_ROWS = "2000000"
$env:CHOICEFORGE_STRICT_CUDA_TILE_ROWS = "1"
$env:CHOICEFORGE_STRICT_CUDA_LOCALITY = "1"
$env:CHOICEFORGE_STRICT_CUDA_COMPACT_INPUTS = "1"
$env:CHOICEFORGE_STRICT_CUDA_GROUPED_INDICES = "1"
$env:CHOICEFORGE_STRICT_CUDA_SPARSE_COEFFICIENTS = "0"
$env:CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32 = "1"
$env:CHOICEFORGE_STRICT_CUDA_PERSISTENT_PLAN = "1"
$env:CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS = "0"
$env:CHOICEFORGE_STRICT_CUDA_MODE_CHOICE = "1"
$env:CHOICEFORGE_STRICT_CUDA_BATCHES = "0"
$env:CHOICEFORGE_STRICT_CUDA_SHARROW_FMA = "1"
if ($CaptureAtlas) { $env:CHOICEFORGE_PHASE55_CAPTURE_ATLAS = "1" }
else { Remove-Item Env:CHOICEFORGE_PHASE55_CAPTURE_ATLAS -ErrorAction SilentlyContinue }

for ($trial = 1; $trial -le $Repetitions; $trial++) {
    foreach ($shard in @("pre", "post")) {
        $report = Join-Path $repo "benchmark-results\phase55-$RunTag-$shard-gpu-$trial.json"
        if (Test-Path -LiteralPath $report) { throw "Refusing to overwrite $report" }
        $kernel = Join-Path $repo "benchmark-results\phase55-$RunTag-$shard-kernels-$trial"
        New-Item -ItemType Directory -Path (Join-Path $kernel "mode") -Force | Out-Null
        $checkpoint = Join-Path $repo "benchmark-results\phase55-$RunTag-$shard-checkpoint-$trial.json"
        $output = if ($shard -eq "pre") {
            Join-Path $project "o-$RunTag-pre-50000-$trial"
        } else { $postOutput }
        if ($shard -eq "pre" -and (Test-Path -LiteralPath $output)) {
            throw "Refusing to overwrite $output"
        }
        $arguments = @(
            $runner, "--project", $project, "--data", $data, "--output", $output,
            "--config-overlay", $overlay, "--config-overlay", $sharrow,
            "--full-model", "--households-sample-size", "50000", "--native-abi-live",
            "--reference-pipeline", $reference, "--report", $report,
            "--checkpoint", $checkpoint, "--kernel-reports", $kernel,
            "--phase55-device-entity-execution-runtime"
        )
        if ($shard -eq "pre") { $arguments += @("--stop-after-model", "workplace_location") }
        else {
            $arguments += @(
                "--resume-full-model", "--resume", "mandatory_tour_scheduling",
                "--stop-after-model", "atwork_subtour_destination"
            )
        }
        $stdout = Join-Path $project "$RunTag-$shard-$trial.stdout.log"
        $stderr = Join-Path $project "$RunTag-$shard-$trial.stderr.log"
        $process = Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $repo -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "Phase 55 $shard shard $trial failed; see $stderr"
        }
    }
}

Write-Output "Phase 55 shards completed: $Repetitions matched candidate pairs"
