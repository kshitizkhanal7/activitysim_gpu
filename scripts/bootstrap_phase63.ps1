param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv-phase8/Scripts/python.exe'
$upstream = Join-Path $repo 'tmp/activitysim-phase8-source'
$commit = '16ab11180a26912987eb902daf945e268f3efc11'
$patch = Join-Path $repo 'integration/activitysim-phase63.patch'

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}

Push-Location $repo
try {
    if (-not $CheckOnly) {
        # uv and Git must be installed explicitly by the operator. This script
        # neither installs a GPU driver nor changes any system-wide environment.
        Get-Command uv -ErrorAction Stop | Out-Null
        Get-Command git -ErrorAction Stop | Out-Null
        if (-not (Test-Path -LiteralPath $python)) {
            Invoke-Checked uv @('venv','--python','3.11.14','.venv-phase8')
        }
        Invoke-Checked uv @('pip','install','--python',$python,'--no-deps','-r','requirements-phase63-lock.txt')
        if (-not (Test-Path -LiteralPath $upstream)) {
            Invoke-Checked git @('clone','--filter=blob:none','--no-checkout','https://github.com/ActivitySim/activitysim.git',$upstream)
            Invoke-Checked git @('-C',$upstream,'checkout','--detach',$commit)
        }
        $actual = & git -C $upstream rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or $actual.Trim() -ne $commit) {
            throw 'Existing upstream checkout is not the pinned revision; left untouched.'
        }
        & git -C $upstream apply --reverse --check $patch 2>$null
        if ($LASTEXITCODE -ne 0) {
            $dirty = & git -C $upstream status --porcelain
            if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'Unrecognized upstream changes; refusing to overwrite.' }
            Invoke-Checked git @('-C',$upstream,'apply','--check',$patch)
            Invoke-Checked git @('-C',$upstream,'apply',$patch)
        }
        # Explicit pins intentionally reproduce the measured environment rather
        # than resolving a new dependency set from upstream's moving constraints.
        $replicationSpec = Get-Content -LiteralPath (Join-Path $repo 'reproducibility/phase63-inputs.json') -Raw | ConvertFrom-Json
        $priorVersionOverride = $env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ACTIVITYSIM
        try {
            # SCM labels otherwise depend on clone tag history/build date even
            # at the identical source commit. Pin the measured metadata too.
            $env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ACTIVITYSIM = $replicationSpec.activitysim_distribution_version
            Invoke-Checked uv @('pip','install','--python',$python,'--no-deps','--reinstall-package','activitysim','-e',$upstream,'-e',$repo)
        }
        finally { $env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ACTIVITYSIM = $priorVersionOverride }
        Invoke-Checked $python @('scripts/prepare_phase63_replication.py','normalize-checkout')
        Invoke-Checked $python @('scripts/prepare_phase63_replication.py','materialize')
    }
    Invoke-Checked $python @('scripts/prepare_phase63_replication.py','check')
}
finally { Pop-Location }
