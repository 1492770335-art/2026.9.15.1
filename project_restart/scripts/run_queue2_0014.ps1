# GPU queue 2 (run after queue 0006-0009 completes)
# EXP-0014 delta-head MLP first (highest priority from delta discovery), then decisions pending results.
$env:PYTHONIOENCODING = 'utf-8'
$ErrorActionPreference = 'Continue'

function Run-Step($name, $script) {
    Write-Output "=== QUEUE2 START $name $(Get-Date -Format s) ==="
    python $script 2>&1
    Write-Output "=== QUEUE2 END $name exit=$LASTEXITCODE $(Get-Date -Format s) ==="
}

Run-Step -name 'EXP-0014 mlp delta' -script 'project_restart\src\models\mlp_delta.py'
Write-Output "=== QUEUE2 ALL DONE $(Get-Date -Format s) ==="
