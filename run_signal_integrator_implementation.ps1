# Signal Integrator Implementation Runner
# File: run_signal_integrator_implementation.ps1
# Description: 
#   3-3-1「シグナル競合時の優先度ルール設計」
#   PowerShell実行用スクリプト

param(
    [string]$TestType = "all"
)

# 基本設定
$ProjectRoot = $PSScriptRoot
$ConfigDir = Join-Path $ProjectRoot "config"
$PythonExecutable = "python"

Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "Signal Integrator Implementation - PowerShell実行" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host ""

# プロジェクトディレクトリの確認
if (-not (Test-Path $ProjectRoot)) {
    Write-Host "エラー: プロジェクトディレクトリが見つかりません: $ProjectRoot" -ForegroundColor Red
    exit 1
}

Set-Location $ProjectRoot

# 依存関係チェック
Write-Host "依存関係チェック中..." -ForegroundColor Yellow

$RequiredFiles = @(
    "config\signal_integrator.py",
    "config\signal_integration_config.json",
    "demo_signal_integrator.py",
    "test_signal_integrator.py"
)

$MissingFiles = @()
foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) {
        $MissingFiles += $File
    }
}

if ($MissingFiles.Count -gt 0) {
    Write-Host "エラー: 必要ファイルが不足しています:" -ForegroundColor Red
    foreach ($File in $MissingFiles) {
        Write-Host "  - $File" -ForegroundColor Red
    }
    exit 1
}

Write-Host "✓ 依存関係チェック完了" -ForegroundColor Green
Write-Host ""

# 実行シーケンス
$Steps = @()
if ($TestType -eq "all" -or $TestType -eq "demo") {
    $Steps += @{
        Name = "デモ実行"
        Command = "$PythonExecutable demo_signal_integrator.py"
        Description = "基本統合機能のデモンストレーション"
    }
}

if ($TestType -eq "all" -or $TestType -eq "test") {
    $Steps += @{
        Name = "テスト実行"
        Command = "$PythonExecutable test_signal_integrator.py"
        Description = "包括的テストスイート実行"
    }
}

if ($TestType -eq "all" -or $TestType -eq "basic") {
    $Steps += @{
        Name = "基本デモ"
        Command = "$PythonExecutable demo_signal_integrator.py --basic"
        Description = "基本統合機能のみ"
    }
}

if ($TestType -eq "all" -or $TestType -eq "conflicts") {
    $Steps += @{
        Name = "競合シナリオ"
        Command = "$PythonExecutable demo_signal_integrator.py --conflicts"
        Description = "競合解決機能のテスト"
    }
}

if ($Steps.Count -eq 0) {
    Write-Host "エラー: 無効なテストタイプ: $TestType" -ForegroundColor Red
    Write-Host "使用可能なオプション: all, demo, test, basic, conflicts" -ForegroundColor Yellow
    exit 1
}

# 実行開始
$Results = @()
$TotalSteps = $Steps.Count
$CurrentStep = 0

foreach ($Step in $Steps) {
    $CurrentStep++
    Write-Host "[$CurrentStep/$TotalSteps] $($Step.Name) を実行中..." -ForegroundColor Cyan
    Write-Host "説明: $($Step.Description)" -ForegroundColor Gray
    Write-Host "コマンド: $($Step.Command)" -ForegroundColor Gray
    Write-Host ""
    
    try {
        $StartTime = Get-Date
        
        # PowerShellでコマンドを実行
        $ProcessInfo = Start-Process -FilePath "cmd" -ArgumentList "/c", $Step.Command -Wait -NoNewWindow -PassThru -RedirectStandardOutput "temp_output.txt" -RedirectStandardError "temp_error.txt"
        
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalSeconds
        
        $StdOut = ""
        $StdErr = ""
        
        if (Test-Path "temp_output.txt") {
            $StdOut = Get-Content "temp_output.txt" -Raw
            Remove-Item "temp_output.txt" -Force
        }
        
        if (Test-Path "temp_error.txt") {
            $StdErr = Get-Content "temp_error.txt" -Raw
            Remove-Item "temp_error.txt" -Force
        }
        
        $Success = ($ProcessInfo.ExitCode -eq 0)
        
        $Results += @{
            Name = $Step.Name
            Success = $Success
            Duration = $Duration
            ExitCode = $ProcessInfo.ExitCode
            StdOut = $StdOut
            StdErr = $StdErr
        }
        
        if ($Success) {
            Write-Host "✓ $($Step.Name) 成功 (${Duration}秒)" -ForegroundColor Green
        } else {
            Write-Host "✗ $($Step.Name) 失敗 (ExitCode: $($ProcessInfo.ExitCode))" -ForegroundColor Red
            if ($StdErr) {
                Write-Host "エラー出力:" -ForegroundColor Red
                Write-Host $StdErr -ForegroundColor Red
            }
        }
        
    } catch {
        Write-Host "✗ $($Step.Name) 実行エラー: $($_.Exception.Message)" -ForegroundColor Red
        $Results += @{
            Name = $Step.Name
            Success = $false
            Duration = 0
            ExitCode = -1
            StdOut = ""
            StdErr = $_.Exception.Message
        }
    }
    
    Write-Host ""
}

# 結果サマリー
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "実行結果サマリー" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

$SuccessfulSteps = ($Results | Where-Object { $_.Success }).Count
$TotalDuration = ($Results | Measure-Object -Property Duration -Sum).Sum

Write-Host ""
Write-Host "総ステップ数: $TotalSteps" -ForegroundColor White
Write-Host "成功ステップ数: $SuccessfulSteps" -ForegroundColor Green
Write-Host "失敗ステップ数: $($TotalSteps - $SuccessfulSteps)" -ForegroundColor Red
Write-Host "成功率: $([math]::Round(($SuccessfulSteps / $TotalSteps) * 100, 1))%" -ForegroundColor Yellow
Write-Host "総実行時間: $([math]::Round($TotalDuration, 2))秒" -ForegroundColor Yellow
Write-Host ""

foreach ($Result in $Results) {
    $Status = if ($Result.Success) { "✓" } else { "✗" }
    $Color = if ($Result.Success) { "Green" } else { "Red" }
    Write-Host "  $Status $($Result.Name) ($([math]::Round($Result.Duration, 2))秒)" -ForegroundColor $Color
}

# レポート生成
$ReportContent = @"
# 3-3-1 シグナル競合時の優先度ルール設計 実装完了レポート

## 実行サマリー
- 実行日時: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
- 実行ステップ数: $TotalSteps
- 成功ステップ数: $SuccessfulSteps
- 成功率: $([math]::Round(($SuccessfulSteps / $TotalSteps) * 100, 1))%
- 総実行時間: $([math]::Round($TotalDuration, 2))秒

## 各ステップ結果

"@

foreach ($Result in $Results) {
    $Status = if ($Result.Success) { "✅ 成功" } else { "❌ 失敗" }
    $ReportContent += @"

### $($Result.Name)
- ステータス: $Status
- 実行時間: $([math]::Round($Result.Duration, 2))秒
- 終了コード: $($Result.ExitCode)

"@
    
    if ($Result.StdOut -and $Result.StdOut.Trim()) {
        $OutputPreview = if ($Result.StdOut.Length -gt 500) { 
            $Result.StdOut.Substring(0, 500) + "..." 
        } else { 
            $Result.StdOut 
        }
        $ReportContent += @"
- 標準出力:
``````
$OutputPreview
``````

"@
    }
    
    if ($Result.StdErr -and $Result.StdErr.Trim()) {
        $ErrorPreview = if ($Result.StdErr.Length -gt 200) { 
            $Result.StdErr.Substring(0, 200) + "..." 
        } else { 
            $Result.StdErr 
        }
        $ReportContent += @"
- エラー出力:
``````
$ErrorPreview
``````

"@
    }
}

# レポートファイル保存
$ReportFile = "SIGNAL_INTEGRATOR_IMPLEMENTATION_REPORT_$(Get-Date -Format 'yyyyMMdd_HHmmss').md"
$ReportContent | Out-File -FilePath $ReportFile -Encoding UTF8

Write-Host ""
Write-Host "📄 詳細レポート生成: $ReportFile" -ForegroundColor Cyan

# 最終結果
$ImplementationSuccess = ($SuccessfulSteps -eq $TotalSteps)

if ($ImplementationSuccess) {
    Write-Host ""
    Write-Host "🎉 Signal Integrator 実装完了!" -ForegroundColor Green
    Write-Host "📄 詳細レポート: $ReportFile" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "⚠️  一部のステップで問題が発生しました" -ForegroundColor Yellow
    Write-Host "📄 詳細レポート: $ReportFile" -ForegroundColor Yellow
    exit 1
}
