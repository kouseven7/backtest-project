# ============================================================
# watch_positions.ps1
# positions.jsonの変更を検知してログに記録するスクリプト
# 既存ファイルへの影響：ゼロ（読み取り専用監視）
# 使い方：PowerShellで実行、Ctrl+Cで停止
# ============================================================

$TargetFile   = "C:\Users\imega\Documents\my_backtest_project\logs\dssms\positions.json"
$LogFile      = "C:\Users\imega\Documents\my_backtest_project\logs\dssms\positions_history.log"
$WatchDir     = Split-Path $TargetFile
$WatchName    = Split-Path $TargetFile -Leaf

Write-Host "[positions監視] 開始しました"
Write-Host "[positions監視] 対象: $TargetFile"
Write-Host "[positions監視] ログ: $LogFile"
Write-Host "[positions監視] Ctrl+C で停止"
Write-Host ""

# 起動時の内容を記録
function Get-PositionsContent {
    try {
        return (Get-Content $TargetFile -Raw -Encoding UTF8).Trim()
    } catch {
        return "(読み取り失敗)"
    }
}

function Write-PositionsLog {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

# 起動時のスナップショット
$prevContent = Get-PositionsContent
Write-PositionsLog "=== 監視開始 ==="
Write-PositionsLog "起動時の内容: $prevContent"

# FileSystemWatcherの設定
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path   = $WatchDir
$watcher.Filter = $WatchName
$watcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite
$watcher.EnableRaisingEvents = $true

# 変更検知時の処理
$action = {
    Start-Sleep -Milliseconds 200  # ファイル書き込み完了を待つ

    $newContent = try {
        (Get-Content $Event.SourceEventArgs.FullPath -Raw -Encoding UTF8).Trim()
    } catch {
        "(読み取り失敗)"
    }

    $prevContent = $global:prevContent

    if ($newContent -ne $prevContent) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        # 変更元の推定（スケジューラープロセスが動いているか）
        $schedulerRunning = Get-Process -Name "python" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*dssms_scheduler*" }
        $source = if ($schedulerRunning) { "スケジューラー or 手動" } else { "手動（スケジューラー停止中）" }

        $logLine1 = "[$timestamp] *** positions.json 変更検知 *** 変更元推定: $source"
        $logLine2 = "[$timestamp] 変更前: $prevContent"
        $logLine3 = "[$timestamp] 変更後: $newContent"
        $separator = "[$timestamp] " + ("-" * 60)

        foreach ($line in @($logLine1, $logLine2, $logLine3, $separator)) {
            Write-Host $line
            Add-Content -Path $using:LogFile -Value $line -Encoding UTF8
        }

        $global:prevContent = $newContent
    }
}

# イベント登録
$job = Register-ObjectEvent -InputObject $watcher -EventName Changed -Action $action

# Ctrl+Cで停止するまで待機
try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    Unregister-Event -SourceIdentifier $job.Name
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-PositionsLog "=== 監視終了 ==="
    Write-Host "[positions監視] 停止しました"
}