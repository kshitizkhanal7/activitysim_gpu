param(
    [string]$Source = "",
    [string]$Patch = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Source) {
    $Source = Join-Path $repo "tmp\activitysim-phase8-source"
}
if (-not $Patch) {
    $Patch = Join-Path $repo "integration\activitysim-current-choiceforge.patch"
}
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$patchPath = (Resolve-Path -LiteralPath $Patch).Path
$commit = (git -C $sourcePath rev-parse HEAD).Trim()
$temporary = Join-Path ([System.IO.Path]::GetTempPath()) ("choiceforge-patch-" + [guid]::NewGuid())

try {
    git clone --quiet --no-local $sourcePath $temporary
    git -C $temporary checkout --quiet $commit
    git -C $temporary apply --check $patchPath
    git -C $temporary apply $patchPath

    $expected = (git -C $sourcePath diff --binary 2>$null) -join "`n"
    $actual = (git -C $temporary diff --binary 2>$null) -join "`n"
    if ($expected -ne $actual) {
        throw "Patch does not reproduce the source checkout's tracked modifications. Regenerate integration\activitysim-current-choiceforge.patch."
    }
    Write-Host "Patch exactly reproduces tracked changes on ActivitySim $commit."
}
finally {
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Recurse -Force
    }
}
