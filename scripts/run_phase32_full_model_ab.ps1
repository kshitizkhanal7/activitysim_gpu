param(
    [ValidateRange(1, 5)][int]$Repetitions = 3,
    [ValidateRange(1, 500000)][int]$Households = 50000,
    [ValidatePattern("^[A-Za-z0-9-]+$")][string]$RunTag = "p32proof",
    [ValidateSet("phase17", "activitysim")][string]$Baseline = "phase17",
    [ValidateSet(32, 33, 34)][int]$CandidatePhase = 32,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $repo "benchmark-data\phase9-mtc-full\prototype_mtc_extended"
$python = Join-Path $repo ".venv-phase8\Scripts\python.exe"
$activitysim = Join-Path $repo ".venv-phase8\Scripts\activitysim.exe"
$overlayName = if ($CandidatePhase -ge 33) {
    "configs_phase33_choiceforge"
} else {
    "configs_phase9_choiceforge"
}
$overlay = Join-Path $repo "benchmark-data\$overlayName"
$sharrowConfig = Join-Path $project "configs_sh"
$data = Join-Path $project "data_full"
$reference = Join-Path $project "o-p17modeproof16-baseline-50000-1"
$phasePrefix = "phase$CandidatePhase"
$summaryPath = Join-Path $repo "benchmark-results\$phasePrefix-$RunTag-summary.json"

foreach ($required in @($python, $activitysim, $overlay, $sharrowConfig, $data, $reference)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required Phase $CandidatePhase input is missing: $required" }
}
if (Test-Path -LiteralPath $summaryPath) { throw "Refusing to overwrite: $summaryPath" }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OPENBLAS_NUM_THREADS = "16"
$env:OMP_NUM_THREADS = "16"
$env:PATH = (Join-Path $repo ".venv-phase8\Scripts") + ";" + $env:PATH
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

function Invoke-CheckedProcess(
    [string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory,
    [string]$Stdout, [string]$Stderr
) {
    $started = Get-Date
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr -WindowStyle Hidden -Wait -PassThru
    $finished = Get-Date
    if ($process.ExitCode -ne 0) {
        throw "Process failed with exit code $($process.ExitCode): $FilePath. See $Stderr"
    }
    return [pscustomobject]@{
        started = $started.ToString("o")
        finished = $finished.ToString("o")
        wall_seconds = ($finished - $started).TotalSeconds
    }
}

$runs = @()
for ($trial = 1; $trial -le $Repetitions; $trial++) {
    $baselineName = "o-$RunTag-base-$Households-$trial"
    $candidateName = "o-$RunTag-gpu-$Households-$trial"
    $baselineOutput = Join-Path $project $baselineName
    $candidateOutput = Join-Path $project $candidateName
    $baselineKernel = Join-Path $repo "benchmark-results\$phasePrefix-$RunTag-base-kernels-$trial"
    $candidateKernel = Join-Path $repo "benchmark-results\$phasePrefix-$RunTag-gpu-kernels-$trial"
    $candidateReport = Join-Path $repo "benchmark-results\$phasePrefix-$RunTag-gpu-$trial.json"
    $checkpoint = Join-Path $repo "benchmark-results\$phasePrefix-$RunTag-checkpoint-$trial.json"
    $verification = Join-Path $repo "benchmark-results\$phasePrefix-$RunTag-exact-$trial.json"
    $baselineComplete = Test-Path -LiteralPath (Join-Path $baselineOutput "timing_log.csv")
    $candidateComplete = (
        (Test-Path -LiteralPath (Join-Path $candidateOutput "timing_log.csv")) -and
        (Test-Path -LiteralPath $candidateReport) -and
        (Test-Path -LiteralPath $verification)
    )
    if (-not $Resume) {
        foreach ($path in @($baselineOutput, $candidateOutput, $baselineKernel, $candidateKernel, $candidateReport, $checkpoint, $verification)) {
            if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite: $path" }
        }
    } elseif ((Test-Path -LiteralPath $baselineOutput) -and -not $baselineComplete) {
        throw "Resume found incomplete baseline output; archive it first: $baselineOutput"
    } elseif ((Test-Path -LiteralPath $candidateOutput) -and -not $candidateComplete) {
        throw "Resume found incomplete candidate output; archive it first: $candidateOutput"
    }
    if ($baselineComplete) {
        $baselineRun = [pscustomobject]@{ started = $null; finished = $null; wall_seconds = $null }
    } else {
        New-Item -ItemType Directory -Path $baselineKernel | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $baselineKernel "mode") | Out-Null

        $env:CHOICEFORGE_PHASE17_REPORT_DIR = $baselineKernel
        $env:CHOICEFORGE_PHASE17_MODE_REPORT_DIR = Join-Path $baselineKernel "mode"
        $env:CHOICEFORGE_PHASE17_RUN_ID = "$RunTag-base-$trial"
        $baselineStdout = Join-Path $project "$RunTag-base-$trial.stdout.log"
        $baselineStderr = Join-Path $project "$RunTag-base-$trial.stderr.log"
        $baselineArguments = @("run")
        if ($Baseline -eq "phase17") {
            $env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "1"
            $env:CHOICEFORGE_STRICT_CUDA_MODE_CHOICE = "1"
            $baselineArguments += @("-c", $overlay)
        } else {
            # The regular control is pinned ActivitySim with Sharrow required and
            # no ChoiceForge component overlay or candidate hook.
            $env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "0"
            $env:CHOICEFORGE_STRICT_CUDA_MODE_CHOICE = "0"
        }
        $baselineArguments += @(
            "-c", "configs_sh", "-c", "configs",
            "-d", "data_full", "-o", $baselineName,
            "--households_sample_size", "$Households"
        )
        $baselineRun = Invoke-CheckedProcess $activitysim $baselineArguments `
            $project $baselineStdout $baselineStderr
    }

    if ($candidateComplete) {
        $candidateRun = [pscustomobject]@{ started = $null; finished = $null; wall_seconds = $null }
    } else {
        $env:CHOICEFORGE_STRICT_CUDA_CANDIDATE = "1"
        $env:CHOICEFORGE_STRICT_CUDA_MODE_CHOICE = "1"
        $candidateStdout = Join-Path $project "$RunTag-gpu-$trial.stdout.log"
        $candidateStderr = Join-Path $project "$RunTag-gpu-$trial.stderr.log"
        $candidateArguments = @(
            (Join-Path $repo "scripts\run_phase22_integrated_scheduling.py"),
            "--project", $project, "--data", $data, "--output", $candidateOutput,
            "--config-overlay", $overlay, "--config-overlay", $sharrowConfig,
            "--full-model", "--households-sample-size", "$Households", "--native-abi-live",
            "--reference-pipeline", (Join-Path $reference "pipeline.parquetpipeline"),
            "--report", $candidateReport, "--checkpoint", $checkpoint,
            "--kernel-reports", $candidateKernel
        )
        if ($CandidatePhase -eq 33) { $candidateArguments += "--phase33-model-wide" }
        if ($CandidatePhase -eq 34) { $candidateArguments += "--phase34-location-choice" }
        $candidateRun = Invoke-CheckedProcess $python $candidateArguments `
            $repo $candidateStdout $candidateStderr

        & $python (Join-Path $repo "scripts\verify_phase15_outputs.py") `
            --reference $baselineOutput --candidate $candidateOutput --output $verification | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Phase $CandidatePhase exact-output verification failed for pair $trial" }
    }
    $exactProof = Get-Content -LiteralPath $verification -Raw | ConvertFrom-Json
    if (-not $exactProof.success -or $exactProof.decision_cells_different -ne 0) {
        throw "Phase $CandidatePhase exact-output proof is invalid for pair $trial"
    }

    $baselineTiming = Import-Csv (Join-Path $baselineOutput "timing_log.csv")
    $candidateTiming = Import-Csv (Join-Path $candidateOutput "timing_log.csv")
    $baselineAll = [double](($baselineTiming.seconds | Measure-Object -Sum).Sum)
    $candidateAll = [double](($candidateTiming.seconds | Measure-Object -Sum).Sum)
    $baselineMandatory = [double](($baselineTiming | Where-Object model_name -eq "mandatory_tour_scheduling").seconds)
    $candidateMandatory = [double](($candidateTiming | Where-Object model_name -eq "mandatory_tour_scheduling").seconds)
    $baselineComponents = [ordered]@{}
    $candidateComponents = [ordered]@{}
    foreach ($row in $baselineTiming) {
        $baselineComponents[[string]$row.model_name] = [double]$row.seconds
    }
    foreach ($row in $candidateTiming) {
        $candidateComponents[[string]$row.model_name] = [double]$row.seconds
    }
    if ([string]::Join("|", $baselineComponents.Keys) -ne [string]::Join("|", $candidateComponents.Keys)) {
        throw "Phase $CandidatePhase component timing schemas differ for pair $trial"
    }
    $proof = Get-Content -LiteralPath $candidateReport -Raw | ConvertFrom-Json
    if (($proof.proof_gates.PSObject.Properties.Value | Where-Object { -not $_ }).Count -ne 0) {
        throw "Phase $CandidatePhase proof gate failed for pair $trial"
    }
    $runs += [pscustomobject]@{
        trial = $trial
        baseline_output = $baselineName
        candidate_output = $candidateName
        baseline_wall_seconds = $baselineRun.wall_seconds
        candidate_wall_seconds = $candidateRun.wall_seconds
        baseline_all_model_seconds = $baselineAll
        candidate_all_model_seconds = $candidateAll
        all_model_seconds_saved = $baselineAll - $candidateAll
        all_model_reduction_percent = 100.0 * ($baselineAll - $candidateAll) / $baselineAll
        all_model_speedup = $baselineAll / $candidateAll
        baseline_mandatory_seconds = $baselineMandatory
        candidate_mandatory_seconds = $candidateMandatory
        mandatory_reduction_percent = 100.0 * ($baselineMandatory - $candidateMandatory) / $baselineMandatory
        baseline_component_seconds = $baselineComponents
        candidate_component_seconds = $candidateComponents
        exact_output_verification = $verification
        candidate_report = $candidateReport
    }
}

$baselineValues = @($runs | ForEach-Object { $_.baseline_all_model_seconds } | Sort-Object)
$candidateValues = @($runs | ForEach-Object { $_.candidate_all_model_seconds } | Sort-Object)
$middle = [int][Math]::Floor($Repetitions / 2)
$componentComparison = @()
foreach ($modelName in $runs[0].baseline_component_seconds.Keys) {
    $baselineComponentValues = @(
        $runs | ForEach-Object { [double]$_.baseline_component_seconds[$modelName] } |
            Sort-Object
    )
    $candidateComponentValues = @(
        $runs | ForEach-Object { [double]$_.candidate_component_seconds[$modelName] } |
            Sort-Object
    )
    $baselineMedian = [double]$baselineComponentValues[$middle]
    $candidateMedian = [double]$candidateComponentValues[$middle]
    $gpuRole = switch ($modelName) {
        "mandatory_tour_scheduling" { "Phase32 native GPU retained" }
        "school_location" { if ($CandidatePhase -eq 34) { "Phase34 generated GPU logsums" } else { "not directly GPU-targeted" } }
        "workplace_location" { if ($CandidatePhase -eq 34) { "Phase34 generated GPU logsums" } else { "not directly GPU-targeted" } }
        "joint_tour_destination" { if ($CandidatePhase -eq 34) { "Phase34 generated GPU logsums" } else { "not directly GPU-targeted" } }
        "atwork_subtour_destination" { if ($CandidatePhase -eq 34) { "Phase34 generated GPU logsums" } else { "not directly GPU-targeted" } }
        "atwork_subtour_mode_choice" { if ($CandidatePhase -eq 34) { "Phase34 generated GPU" } else { "not directly GPU-targeted" } }
        "non_mandatory_tour_destination" { if ($CandidatePhase -ge 33) { "Phase33 generated GPU retained" } else { "not directly GPU-targeted" } }
        "non_mandatory_tour_scheduling" { if ($CandidatePhase -ge 33) { "Phase33 scheduling GPU retained" } else { "not directly GPU-targeted" } }
        "tour_mode_choice_simulate" { if ($CandidatePhase -ge 33) { "Phase33 generated GPU retained" } else { "not directly GPU-targeted" } }
        "trip_destination" { "Phase17 generated GPU retained" }
        "trip_mode_choice" { "Phase17 generated GPU retained" }
        default { "not directly GPU-targeted" }
    }
    $componentComparison += [pscustomobject]@{
        model_name = $modelName
        gpu_role = $gpuRole
        median_baseline_seconds = $baselineMedian
        median_candidate_seconds = $candidateMedian
        median_seconds_saved = $baselineMedian - $candidateMedian
        median_reduction_percent = 100.0 * ($baselineMedian - $candidateMedian) / $baselineMedian
        median_speedup = $baselineMedian / $candidateMedian
    }
}
$baselineLabel = if ($Baseline -eq "activitysim") {
    "regular pinned ActivitySim with Sharrow required"
} else {
    "already GPU-accelerated Phase 17 runtime"
}
$summary = [ordered]@{
    phase = $CandidatePhase
    benchmark = "public Prototype MTC extended, full 34-step ActivitySim model"
    households = $Households
    baseline = $Baseline
    baseline_description = $baselineLabel
    design = "fresh-process matched $Baseline/Phase$CandidatePhase pairs in control/candidate order"
    repetitions = $Repetitions
    runs = $runs
    median_baseline_all_model_seconds = $baselineValues[$middle]
    median_candidate_all_model_seconds = $candidateValues[$middle]
    median_seconds_saved = $baselineValues[$middle] - $candidateValues[$middle]
    median_reduction_percent = 100.0 * ($baselineValues[$middle] - $candidateValues[$middle]) / $baselineValues[$middle]
    median_speedup = $baselineValues[$middle] / $candidateValues[$middle]
    component_comparison = $componentComparison
    candidate_won_every_pair = (@($runs | Where-Object all_model_seconds_saved -le 0).Count -eq 0)
    every_pair_exact = $true
    claim_boundary = "whole-model gain over $baselineLabel; all substantive modeled outputs exact and declared diagnostic logsums bounded"
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8
$summary | ConvertTo-Json -Depth 8
if (-not $summary.candidate_won_every_pair) { throw "Phase $CandidatePhase candidate did not win every matched pair" }
