# GPU experiment queue (run after EXP-0005 strong-GRU completes)
# Order: GRU train/stream equivalence -> seed variance -> GRU window ablation -> market context -> category encoding
$env:PYTHONIOENCODING = 'utf-8'
$ErrorActionPreference = 'Continue'

function Run-Step($name, $script) {
    Write-Output "=== QUEUE START $name $(Get-Date -Format s) ==="
    python $script 2>&1
    Write-Output "=== QUEUE END $name exit=$LASTEXITCODE $(Get-Date -Format s) ==="
}

Run-Step -name 'T-GRU-EQUIV' -script 'project_restart\tests\test_gru_train_stream_equiv.py'
Run-Step -name 'EXP-0006 seeds' -script 'project_restart\src\models\exp0006_seeds.py'
Run-Step -name 'EXP-0007 gru window' -script 'project_restart\src\models\gru_window.py'
Run-Step -name 'EXP-0008 mlp market' -script 'project_restart\src\models\mlp_market.py'
Run-Step -name 'EXP-0009 mlp catenc' -script 'project_restart\src\models\mlp_catenc.py'
Write-Output "=== QUEUE ALL DONE $(Get-Date -Format s) ==="
