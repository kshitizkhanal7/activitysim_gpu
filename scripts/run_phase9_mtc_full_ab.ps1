param(
    [int]$Households = 100000,
    [ValidateRange(1, 10)]
    [int]$Repetitions = 1,
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$RunTag = "destination-only",
    [string]$AdditionalChoiceForgeOverlay = "",
    [string]$PhaseLabel = "9A",
    [string]$OptimizationDescription = "baseline plus explicit ChoiceForge trip-destination overlay; scheduling remains on ActivitySim",
    [switch]$BaselineUsesChoiceForge,
    [string]$BaselineDescription = "pinned current ActivitySim with Sharrow required",
    [Int64]$ChunkSizeBytes = 0,
    [ValidateSet("", "training", "adaptive", "production", "disabled", "explicit")]
    [string]$ChunkTrainingMode = "",
    [string]$SharedConfigOverlay = "",
    [ValidateRange(1, 24)]
    [int]$BlasThreads = 24
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $repo "benchmark-data\phase9-mtc-full\prototype_mtc_extended"
$overlay = Join-Path $repo "benchmark-data\configs_phase9_choiceforge"
$activitysim = Join-Path $repo ".venv-phase8\Scripts\activitysim.exe"
$python = Join-Path $repo ".venv-phase8\Scripts\python.exe"
$data = Join-Path $project "data_full"
$activitysimSource = Join-Path $repo "tmp\activitysim-phase8-source"
$integrationPatch = Join-Path $repo "integration\activitysim-current-choiceforge.patch"
$environmentLock = Join-Path $repo "requirements-phase11-lock.txt"

function Get-TreeSha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $entries = Get-ChildItem -LiteralPath $Path -Recurse -File | Sort-Object FullName |
        ForEach-Object {
            $relative = $_.FullName.Substring($Path.Length).TrimStart('\', '/')
            "${relative}:$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())"
        }
    $joined = [string]::Join("`n", $entries)
    $bytes = [Text.Encoding]::UTF8.GetBytes($joined)
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
}

function Get-GpuSample {
    try {
        $line = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if (-not $line) { return $null }
        $values = $line.Split(',') | ForEach-Object { [double]$_.Trim() }
        if ($values.Count -ne 2) { return $null }
        return [pscustomobject]@{ utilization_percent = $values[0]; memory_used_mib = $values[1] }
    } catch { return $null }
}

if (-not (Test-Path -LiteralPath (Join-Path $data "households.csv"))) {
    throw "Full MTC data is missing. See docs\phase9-full-mtc.md."
}
if ($Households -lt 1) {
    throw "Households must be positive. Use 0 only on the documented high-memory host for the complete population."
}
if ($AdditionalChoiceForgeOverlay -and -not (Test-Path -LiteralPath $AdditionalChoiceForgeOverlay)) {
    throw "Additional ChoiceForge overlay does not exist: $AdditionalChoiceForgeOverlay"
}
if ($SharedConfigOverlay -and -not (Test-Path -LiteralPath $SharedConfigOverlay)) {
    throw "Shared configuration overlay does not exist: $SharedConfigOverlay"
}
if (-not (Test-Path -LiteralPath $integrationPatch) -or -not (Test-Path -LiteralPath $environmentLock)) {
    throw "Phase 11 reproducibility artifacts are missing. Run scripts\verify_activitysim_patch.ps1 and retain requirements-phase11-lock.txt."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OPENBLAS_NUM_THREADS = "$BlasThreads"
$env:OMP_NUM_THREADS = "$BlasThreads"
$env:PATH = (Join-Path $repo ".venv-phase8\Scripts") + ";" + $env:PATH

$runs = @()
$strictCudaCandidateSetting = $env:CHOICEFORGE_STRICT_CUDA_CANDIDATE
for ($trial = 1; $trial -le $Repetitions; $trial++) {
    foreach ($condition in @("baseline", "choiceforge")) {
        $name = "phase9-$RunTag-$condition-$Households-hh-$trial"
        # ActivitySim checkpoint names are deeply nested. Keep this output
        # directory deliberately short so Windows path-length limits cannot
        # turn a completed model stage into a false benchmark failure.
        $outputName = "o-$RunTag-$condition-$Households-$trial"
        $output = Join-Path $project $outputName
        if (Test-Path -LiteralPath $output) {
            throw "Refusing to overwrite existing output: $output"
        }

        $arguments = @("run")
        if ($SharedConfigOverlay) {
            $arguments += @("-c", $SharedConfigOverlay)
        }
        if ($condition -eq "choiceforge" -or $BaselineUsesChoiceForge) {
            $arguments += @("-c", $overlay)
            if ($AdditionalChoiceForgeOverlay -and $condition -eq "choiceforge") {
                $arguments += @("-c", $AdditionalChoiceForgeOverlay)
            }
        }
        $arguments += @(
            "-c", "configs_sh",
            "-c", "configs",
            "-d", "data_full",
            "-o", $outputName,
            "--households_sample_size", "$Households"
        )
        if ($ChunkSizeBytes -gt 0) {
            $arguments += @("--chunk_size", "$ChunkSizeBytes")
        }
        if ($ChunkTrainingMode) {
            $arguments += @("--chunk_training_mode", $ChunkTrainingMode)
        }

        $stdout = Join-Path $project "$name.stdout.log"
        $stderr = Join-Path $project "$name.stderr.log"
        if ($BaselineUsesChoiceForge -and $condition -eq "baseline") {
            $env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "0"
        } elseif ($null -ne $strictCudaCandidateSetting) {
            $env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = $strictCudaCandidateSetting
        } else {
            Remove-Item Env:CHOICEFORGE_STRICT_CUDA_CANDIDATE -ErrorAction SilentlyContinue
        }
        if ($condition -eq "choiceforge" -and $env:CHOICEFORGE_PHASE15_REPORT_DIR) {
            $env:CHOICEFORGE_PHASE15_RUN_ID = $name
        } else {
            Remove-Item Env:CHOICEFORGE_PHASE15_RUN_ID -ErrorAction SilentlyContinue
        }
        if ($condition -eq "choiceforge" -and $env:CHOICEFORGE_PHASE16_REPORT_DIR) {
            $env:CHOICEFORGE_PHASE16_RUN_ID = $name
        } else {
            Remove-Item Env:CHOICEFORGE_PHASE16_RUN_ID -ErrorAction SilentlyContinue
        }
        if ($condition -eq "choiceforge" -and $env:CHOICEFORGE_PHASE17_REPORT_DIR) {
            $env:CHOICEFORGE_PHASE17_RUN_ID = $name
        } else {
            Remove-Item Env:CHOICEFORGE_PHASE17_RUN_ID -ErrorAction SilentlyContinue
        }
        $started = Get-Date
        $process = Start-Process `
            -FilePath $activitysim `
            -ArgumentList $arguments `
            -WorkingDirectory $project `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
        $peakLauncherWorkingSet = 0L
        $peakLauncherPrivateBytes = 0L
        $gpuSamples = @()
        while (-not $process.HasExited) {
            Start-Sleep -Seconds 1
            $process.Refresh()
            if (-not $process.HasExited) {
                $peakLauncherWorkingSet = [Math]::Max($peakLauncherWorkingSet, $process.WorkingSet64)
                $peakLauncherPrivateBytes = [Math]::Max($peakLauncherPrivateBytes, $process.PrivateMemorySize64)
            }
            $gpuSample = Get-GpuSample
            if ($gpuSample) { $gpuSamples += $gpuSample }
        }
        $process.WaitForExit()
        $finished = Get-Date
        $exitCode = $process.ExitCode
        # On this Windows console-entry-point shim, ExitCode can be null after
        # the child has finished.  Accept that only when ActivitySim wrote both
        # its timing log and its terminal all-model marker.
        $completed = (Test-Path -LiteralPath (Join-Path $output "timing_log.csv")) -and `
            (Select-String -LiteralPath $stdout -Pattern "Time to execute all models\s*:" -Quiet) -and `
            (-not (Select-String -LiteralPath $stdout -Pattern "unrecoverable error" -Quiet))
        if (($null -eq $exitCode) -and $completed) {
            Write-Warning "$name completed with an unavailable launcher exit code; verified ActivitySim terminal artifacts instead."
            $exitCode = 0
        }
        if ($exitCode -ne 0) {
            throw "$name failed with exit code $exitCode. See $stderr"
        }
        $runs += [pscustomobject]@{
            name = $name
            condition = $condition
            households = $Households
            started = $started.ToString("o")
            finished = $finished.ToString("o")
            wall_seconds = ($finished - $started).TotalSeconds
            # The ActivitySim child records its authoritative high-water mark
            # in its log; these launcher values are diagnostic only.
            launcher_peak_working_set_bytes = $peakLauncherWorkingSet
            launcher_peak_private_bytes = $peakLauncherPrivateBytes
            gpu_samples = $gpuSamples.Count
            gpu_peak_utilization_percent = if ($gpuSamples) { ($gpuSamples | Measure-Object utilization_percent -Maximum).Maximum } else { $null }
            gpu_mean_utilization_percent = if ($gpuSamples) { ($gpuSamples | Measure-Object utilization_percent -Average).Average } else { $null }
            gpu_peak_memory_mib = if ($gpuSamples) { ($gpuSamples | Measure-Object memory_used_mib -Maximum).Maximum } else { $null }
            output = $output
            stdout = $stdout
            stderr = $stderr
        }
    }
}

$manifest = [pscustomobject]@{
    phase = $PhaseLabel
    benchmark = "public Prototype MTC extended full geography"
    full_population_households = 2875192
    zones = 1454
    design = if ($Repetitions -ge 2) {
        "interleaved fresh-process A1/B1/A2/B2... design with repeated baseline and ChoiceForge trials"
    } else {
        "one fresh-process baseline/ChoiceForge pair; not a superiority proof without repeated interleaved trials"
    }
    baseline = $BaselineDescription
    optimized = $OptimizationDescription
    data_sha256 = "b402506a61055e2d38621416dd9a5c7e3cf7517c0a9ae5869f6d760c03284ef3"
    reproducibility = [pscustomobject]@{
        activitysim_commit = (& git -C $activitysimSource rev-parse HEAD).Trim()
        activitysim_patch_sha256 = (Get-FileHash -LiteralPath $integrationPatch -Algorithm SHA256).Hash.ToLowerInvariant()
        environment_lock_sha256 = (Get-FileHash -LiteralPath $environmentLock -Algorithm SHA256).Hash.ToLowerInvariant()
        python = (& $python -VV 2>&1).Trim()
        gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
        base_config_sha256 = (Get-TreeSha256 (Join-Path $project "configs"))
        shared_config_sha256 = (Get-TreeSha256 $SharedConfigOverlay)
        choiceforge_overlay_sha256 = (Get-TreeSha256 $overlay)
        additional_choiceforge_overlay_sha256 = (Get-TreeSha256 $AdditionalChoiceForgeOverlay)
        choiceforge_strict_cuda_candidate = $env:CHOICEFORGE_STRICT_CUDA_CANDIDATE
        choiceforge_strict_cuda_max_rows = $env:CHOICEFORGE_STRICT_CUDA_MAX_ROWS
        choiceforge_strict_cuda_tile_rows = $env:CHOICEFORGE_STRICT_CUDA_TILE_ROWS
        choiceforge_strict_cuda_locality = $env:CHOICEFORGE_STRICT_CUDA_LOCALITY
        choiceforge_strict_cuda_compact_inputs = $env:CHOICEFORGE_STRICT_CUDA_COMPACT_INPUTS
        choiceforge_strict_cuda_grouped_indices = $env:CHOICEFORGE_STRICT_CUDA_GROUPED_INDICES
        choiceforge_strict_cuda_sparse_coefficients = $env:CHOICEFORGE_STRICT_CUDA_SPARSE_COEFFICIENTS
        choiceforge_strict_cuda_expression_float32 = $env:CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32
        choiceforge_strict_cuda_persistent_plan = $env:CHOICEFORGE_STRICT_CUDA_PERSISTENT_PLAN
        choiceforge_strict_cuda_reuse_buffers = $env:CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS
        choiceforge_strict_cuda_mode_choice = $env:CHOICEFORGE_STRICT_CUDA_MODE_CHOICE
        blas_threads = $BlasThreads
    }
    runs = $runs
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content `
    -Path (Join-Path $repo "benchmark-results\phase9-mtc-full-$RunTag-runs.json") `
    -Encoding utf8
