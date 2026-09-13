param([string]$Tag = "p61formal")
$ErrorActionPreference = "Stop"
if ($Tag -notmatch '^[a-zA-Z0-9-]+$') { throw "Invalid benchmark tag" }
$P61Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $P61Root
$P61Python = Join-Path $P61Root ".venv-phase8/Scripts/python.exe"
function Invoke-P61 {
    param([string[]]$ArgumentList)
    & $P61Python @ArgumentList
    if ($LASTEXITCODE -ne 0) { throw "Phase 61 command failed: $ArgumentList" }
}
# No concurrent model runs, tests, input benchmarks or artifact rendering.
# Keep all source/configuration bytes frozen throughout these runs.
Invoke-P61 -ArgumentList @("-m","pytest","-q","--disable-warnings")
Invoke-P61 -ArgumentList @("scripts/run_phase58_comparison.py","--tag",$Tag,"--phase61","--modes","gpu,candidate","--repetitions","6")
Invoke-P61 -ArgumentList @("scripts/run_phase58_comparison.py","--tag","${Tag}cpu48","--phase61","--modes","regular","--repetitions","2","--regular-numba-threads","48")
Invoke-P61 -ArgumentList @("scripts/run_phase58_comparison.py","--tag","${Tag}cpu1","--phase61","--modes","regular","--repetitions","2","--regular-numba-threads","1")
$P61Project = "benchmark-data/phase9-mtc-full/prototype_mtc_extended"
$P61Scenarios = @(
    @{ Name="10k"; Overlay="benchmark-data/configs_phase59_seed991"; Reference="$P61Project/o-p58-p59scenario10k1-regular-1"; Households="10000" },
    @{ Name="seed17"; Overlay="benchmark-data/configs_phase59_seed17"; Reference="$P61Project/o-p58-p59finalseed17-regular-1"; Households="50000" },
    @{ Name="retry7"; Overlay="benchmark-data/configs_phase59_retry7"; Reference="$P61Project/o-p58-p59qualifiedretry7-regular-1"; Households="50000" }
)
foreach ($P61Scenario in $P61Scenarios) {
    Invoke-P61 -ArgumentList @("scripts/run_phase58_comparison.py","--tag","$Tag$($P61Scenario.Name)","--phase61","--modes","candidate","--repetitions","1",
        "--scenario-overlay",$P61Scenario.Overlay,"--scenario-baseline",$P61Scenario.Reference,"--households",$P61Scenario.Households)
}
Invoke-P61 -ArgumentList @("scripts/qualify_phase61.py","--comparison","benchmark-results/phase58-$Tag-summary.json","--scenarios",
    "benchmark-results/phase58-${Tag}10k-summary.json","benchmark-results/phase58-${Tag}seed17-summary.json","benchmark-results/phase58-${Tag}retry7-summary.json",
    "--output","benchmark-results/phase61-formal-qualification.json")
Invoke-P61 -ArgumentList @("scripts/report_phase61_comparison.py","--qualification","benchmark-results/phase61-formal-qualification.json",
    "--regular1","benchmark-results/phase58-${Tag}cpu1-summary.json","--regular48","benchmark-results/phase58-${Tag}cpu48-summary.json",
    "--kernel-controls","benchmark-results/phase61-live-cpu-controls-passive.json","--output","benchmark-results/phase61-complete-comparison.json",
    "--markdown","docs/phase61-component-comparison.md")
Invoke-P61 -ArgumentList @("-m","pytest","-q","--disable-warnings")
