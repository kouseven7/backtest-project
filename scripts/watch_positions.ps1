# ============================================================
# watch_positions.ps1
# Monitor positions.json and log all changes (polling method)
# No impact on existing files (read-only monitoring)
# Usage: Run in PowerShell, Ctrl+C to stop
# ============================================================

$TargetFile = "C:\Users\imega\Documents\my_backtest_project\logs\dssms\positions.json"
$LogFile    = "C:\Users\imega\Documents\my_backtest_project\logs\dssms\positions_history.log"

function Get-PositionsContent {
    try {
        return (Get-Content $TargetFile -Raw -Encoding UTF8).Trim()
    } catch {
        return "(read error)"
    }
}

function Write-PositionsLog {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

Write-Host "[positions watcher] Started (polling mode)"
Write-Host "[positions watcher] Target: $TargetFile"
Write-Host "[positions watcher] Log:    $LogFile"
Write-Host "[positions watcher] Press Ctrl+C to stop"
Write-Host ""

$prevContent = Get-PositionsContent
Write-PositionsLog "=== Monitoring started ==="
Write-PositionsLog "Initial content: $prevContent"

try {
    while ($true) {
        Start-Sleep -Seconds 2

        $newContent = Get-PositionsContent

        if ($newContent -ne $prevContent) {
            $schedulerRunning = Get-Process -Name "python" -ErrorAction SilentlyContinue
            $source = if ($schedulerRunning) { "scheduler or manual" } else { "manual (scheduler stopped)" }

            Write-PositionsLog "*** positions.json CHANGED *** source: $source"
            Write-PositionsLog "BEFORE: $prevContent"
            Write-PositionsLog "AFTER : $newContent"
            Write-PositionsLog ("-" * 60)

            $prevContent = $newContent
        }
    }
} finally {
    Write-PositionsLog "=== Monitoring stopped ==="
    Write-Host "[positions watcher] Stopped"
}