$enc = New-Object System.Text.UTF8Encoding $false
$bal  = (Get-Content logs\dssms\paper_balance.json | ConvertFrom-Json).balance
$pos  = (Get-Content logs\dssms\positions.json | ConvertFrom-Json).PSObject.Properties.Name.Count
$hist = (Get-Content logs\dssms\execution_history.json | ConvertFrom-Json).Count
$guard = Test-Path logs\dssms\session_guard.json

Write-Host "=== DSSMS Health Check ==="
Write-Host "Balance:          $("{0:N0}" -f $bal) yen"
Write-Host "Positions:        $pos"
Write-Host "History records:  $hist"

if ($guard) {
    Write-Host "Session guard:    EXISTS (skip or delete to re-run)" -ForegroundColor Yellow
} else {
    Write-Host "Session guard:    none (OK to start)" -ForegroundColor Green
}

if ($bal -lt 500000) {
    Write-Host "WARNING: Balance below 500,000 yen" -ForegroundColor Red
} elseif ($bal -ge 2900000) {
    Write-Host "Balance: near initial value" -ForegroundColor Green
} else {
    Write-Host "Balance: normal range" -ForegroundColor Green
}