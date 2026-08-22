param(
    [int]$Households = 50000,
    [int]$Repetitions = 3
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $repo "benchmark-data\phase8-mtc-mini\prototype_mtc_sf"
$overlay = Join-Path $repo "benchmark-data\configs_phase8_choiceforge"
$activitysim = Join-Path $repo ".venv-phase8\Scripts\activitysim.exe"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OPENBLAS_NUM_THREADS = "24"
$env:PATH = (Join-Path $repo ".venv-phase8\Scripts") + ";" + $env:PATH

$runs = @()
for ($trial = 1; $trial -le $Repetitions; $trial++) {
    foreach ($condition in @("a", "b")) {
        $name = "phase8-ab-$condition$trial"
        $outputName = "output-$name"
        $output = Join-Path $project $outputName
        if (Test-Path $output) {
            throw "Refusing to overwrite existing output: $output"
        }

        $arguments = @("run")
        if ($condition -eq "b") {
            $arguments += @("-c", $overlay)
        }
        $arguments += @(
            "-c", "configs_sh",
            "-c", "configs",
            "-d", "data",
            "-o", $outputName,
            "--households_sample_size", "$Households"
        )

        $started = Get-Date
        $process = Start-Process `
            -FilePath $activitysim `
            -ArgumentList $arguments `
            -WorkingDirectory $project `
            -RedirectStandardOutput (Join-Path $project "$name.stdout.log") `
            -RedirectStandardError (Join-Path $project "$name.stderr.log") `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        $finished = Get-Date
        if ($process.ExitCode -ne 0) {
            throw "$name failed with exit code $($process.ExitCode)"
        }
        $runs += [pscustomobject]@{
            name = $name
            condition = $condition
            households = $Households
            started = $started.ToString("o")
            finished = $finished.ToString("o")
            wall_seconds = ($finished - $started).TotalSeconds
            output = $output
        }
    }
}

$manifest = [pscustomobject]@{
    phase = "8A"
    design = "interleaved fresh-process A1/B1/A2/B2/A3/B3"
    baseline = "pinned current ActivitySim with Sharrow required"
    optimized = "baseline plus explicit ChoiceForge scheduling and destination overlays"
    runs = $runs
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content `
    -Path (Join-Path $repo "benchmark-results\phase8-interleaved-runs.json") `
    -Encoding utf8
