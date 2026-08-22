param(
    [int]$Households = 100000,
    [ValidateRange(1, 10)]
    [int]$Repetitions = 1,
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$RunTag = "p10",
    [switch]$MemoryCapped
)

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$guardOverlay = if ($MemoryCapped) {
    Join-Path $repo "benchmark-data\configs_phase10_shadow_scheduling"
} else {
    Join-Path $repo "benchmark-data\configs_phase10_shadow_scheduling"
}
$chunkArgs = @{}
if ($MemoryCapped) {
    $chunkArgs = @{
        ChunkSizeBytes = 8000000000
        ChunkTrainingMode = "training"
        SharedConfigOverlay = (Join-Path $repo "benchmark-data\configs_phase10_memory_guard")
    }
}

& (Join-Path $PSScriptRoot "run_phase9_mtc_full_ab.ps1") `
    -Households $Households `
    -Repetitions $Repetitions `
    -RunTag $RunTag `
    -AdditionalChoiceForgeOverlay $guardOverlay `
    -PhaseLabel "10A" `
    -OptimizationDescription "ChoiceForge destination plus shadow-verified scheduling; ActivitySim is returned for any scheduling mismatch" `
    @chunkArgs
