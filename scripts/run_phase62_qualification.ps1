param([string]$Tag = "p62formal")
$ErrorActionPreference = "Stop"
if ($Tag -notmatch '^[a-zA-Z0-9-]+$') { throw "Invalid benchmark tag" }
$P62Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $P62Root
$P62Python = Join-Path $P62Root ".venv-phase8/Scripts/python.exe"
function Invoke-P62 {
    param([string[]]$ArgumentList)
    & $P62Python @ArgumentList
    if ($LASTEXITCODE -ne 0) { throw "Phase 62 command failed: $ArgumentList" }
}
# No concurrent model runs, tests, input benchmarks or artifact rendering.
# Keep all source/configuration bytes frozen throughout these runs.
Invoke-P62 -ArgumentList @("-m","pytest","-q","--disable-warnings")
Invoke-P62 -ArgumentList @("scripts/run_phase58_comparison.py","--tag",$Tag,"--phase62","--modes","gpu,candidate","--repetitions","6")
Invoke-P62 -ArgumentList @("scripts/run_phase58_comparison.py","--tag","${Tag}cpu48","--phase62","--modes","regular","--repetitions","2","--regular-numba-threads","48")
Invoke-P62 -ArgumentList @("scripts/run_phase58_comparison.py","--tag","${Tag}cpu1","--phase62","--modes","regular","--repetitions","2","--regular-numba-threads","1")
$P62Project = "benchmark-data/phase9-mtc-full/prototype_mtc_extended"
$P62Scenarios = @(
    @{ Name="10k"; Overlay="benchmark-data/configs_phase59_seed991"; Reference="$P62Project/o-p58-p59scenario10k1-regular-1"; Households="10000" },
    @{ Name="seed17"; Overlay="benchmark-data/configs_phase59_seed17"; Reference="$P62Project/o-p58-p59finalseed17-regular-1"; Households="50000" },
    @{ Name="retry7"; Overlay="benchmark-data/configs_phase59_retry7"; Reference="$P62Project/o-p58-p59qualifiedretry7-regular-1"; Households="50000" }
)
foreach ($P62Scenario in $P62Scenarios) {
    Invoke-P62 -ArgumentList @("scripts/run_phase58_comparison.py","--tag","$Tag$($P62Scenario.Name)","--phase62","--modes","candidate","--repetitions","1",
        "--scenario-overlay",$P62Scenario.Overlay,"--scenario-baseline",$P62Scenario.Reference,"--households",$P62Scenario.Households)
}
Invoke-P62 -ArgumentList @("scripts/qualify_phase62.py","--comparison","benchmark-results/phase58-$Tag-summary.json","--scenarios",
    "benchmark-results/phase58-${Tag}10k-summary.json","benchmark-results/phase58-${Tag}seed17-summary.json","benchmark-results/phase58-${Tag}retry7-summary.json",
    "--output","benchmark-results/phase62-formal-qualification.json")
Invoke-P62 -ArgumentList @("scripts/report_phase62_comparison.py","--qualification","benchmark-results/phase62-formal-qualification.json",
    "--regular1","benchmark-results/phase58-${Tag}cpu1-summary.json","--regular48","benchmark-results/phase58-${Tag}cpu48-summary.json",
    "--kernel-controls","benchmark-results/phase61-live-cpu-controls-passive.json","--output","benchmark-results/phase62-complete-comparison.json",
    "--markdown","docs/phase62-component-comparison.md")
Invoke-P62 -ArgumentList @("-m","pytest","-q","--disable-warnings")


# Two complete A-B-A repetitions of each process strategy and backend.
# Reverse all four strategies in repetition two; every output is audited.
$P62Cases = @(
    @{ Mode="candidate"; Execution="fresh" },
    @{ Mode="candidate"; Execution="batch" },
    @{ Mode="regular"; Execution="fresh" },
    @{ Mode="regular"; Execution="batch" }
)
foreach ($P62Trial in 1..2) {
    $P62Order = if ($P62Trial -eq 1) { 0..3 } else { 3..0 }
    foreach ($P62Index in $P62Order) {
        $P62Case = $P62Cases[$P62Index]
        $P62Position = if ($P62Trial -eq 1) { $P62Index } else { 3-$P62Index }
        Invoke-P62 -ArgumentList @("scripts/run_phase62_batch.py","--tag","${Tag}-$($P62Case.Mode)-$($P62Case.Execution)-$P62Trial",
            "--mode",$P62Case.Mode,"--execution",$P62Case.Execution,
            "--series-trial",$P62Trial,"--series-position",$P62Position,
            "--candidate-template","benchmark-results/phase58-$Tag-summary.json")
    }
}
Invoke-P62 -ArgumentList @("scripts/report_phase62_batches.py","--tag",$Tag)
Invoke-P62 -ArgumentList @("-m","pytest","-q","--disable-warnings")
