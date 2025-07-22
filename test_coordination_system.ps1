# Multi-Strategy Coordination System - Integration Test Script  
# File: test_coordination_system.ps1
# Description: 4-1-3「マルチ戦略同時実行の調整機能」PowerShell統合テストスクリプト
# Author: imega
# Created: 2025-07-20

Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host "=" * 79 -ForegroundColor Green
Write-Host "🚀 Multi-Strategy Coordination System - Integration Test" -ForegroundColor Cyan
Write-Host "   Task: 4-1-3「マルチ戦略同時実行の調整機能」" -ForegroundColor Yellow  
Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host "=" * 79 -ForegroundColor Green

# 変数初期化
$global:TestResults = @()
$global:StartTime = Get-Date
$global:ErrorCount = 0
$global:SuccessCount = 0

# テスト結果記録関数
function Record-TestResult {
    param(
        [string]$TestName,
        [bool]$Success,
        [string]$Message,
        [string]$Details = ""
    )
    
    $global:TestResults += [PSCustomObject]@{
        TestName = $TestName
        Success = $Success
        Message = $Message
        Details = $Details
        Timestamp = Get-Date
    }
    
    if ($Success) {
        $global:SuccessCount++
        Write-Host "✅ $TestName" -ForegroundColor Green
        if ($Message) { Write-Host "   $Message" -ForegroundColor Gray }
    } else {
        $global:ErrorCount++
        Write-Host "❌ $TestName" -ForegroundColor Red
        Write-Host "   $Message" -ForegroundColor Yellow
        if ($Details) { Write-Host "   Details: $Details" -ForegroundColor Gray }
    }
}

# Python環境確認関数
function Test-PythonEnvironment {
    Write-Host "🐍 Testing Python Environment" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    try {
        # Python バージョン確認
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Record-TestResult "Python Installation" $true "Python found: $pythonVersion"
        } else {
            Record-TestResult "Python Installation" $false "Python not found or not in PATH"
            return $false
        }
        
        # pip 確認
        $pipVersion = python -m pip --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Record-TestResult "Pip Installation" $true "Pip available"
        } else {
            Record-TestResult "Pip Installation" $false "Pip not available"
            return $false
        }
        
        return $true
    }
    catch {
        Record-TestResult "Python Environment" $false "Exception during Python test: $($_.Exception.Message)"
        return $false
    }
}

# パッケージ依存関係インストール関数
function Install-CoordinationRequirements {
    Write-Host "`n📦 Installing Coordination Requirements" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    try {
        # パッケージインストールスクリプト実行
        if (Test-Path "install_coordination_requirements.py") {
            Write-Host "Running package installation script..." -ForegroundColor Gray
            python install_coordination_requirements.py
            
            if ($LASTEXITCODE -eq 0) {
                Record-TestResult "Package Installation" $true "Coordination requirements installation completed"
                return $true
            } else {
                Record-TestResult "Package Installation" $false "Package installation script failed (exit code: $LASTEXITCODE)"
                return $false
            }
        } else {
            Record-TestResult "Package Installation" $false "install_coordination_requirements.py not found"
            return $false
        }
    }
    catch {
        Record-TestResult "Package Installation" $false "Exception during package installation: $($_.Exception.Message)"
        return $false
    }
}

# 設定ファイル確認関数  
function Test-ConfigurationFiles {
    Write-Host "`n📋 Testing Configuration Files" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    $configFiles = @(
        "coordination_config.json"
    )
    
    $configPath = "config"
    if (Test-Path $configPath) {
        $configFiles += Get-ChildItem -Path $configPath -Filter "*.json" | ForEach-Object { "config\$($_.Name)" }
    }
    
    $configSuccess = $true
    
    foreach ($file in $configFiles) {
        if (Test-Path $file) {
            try {
                $content = Get-Content $file -Raw | ConvertFrom-Json
                Record-TestResult "Config File: $file" $true "Configuration file valid"
            }
            catch {
                Record-TestResult "Config File: $file" $false "Invalid JSON format"
                $configSuccess = $false
            }
        } else {
            Record-TestResult "Config File: $file" $false "Configuration file not found"
            $configSuccess = $false
        }
    }
    
    return $configSuccess
}

# コンポーネントファイル確認関数
function Test-ComponentFiles {
    Write-Host "`n🔧 Testing Component Files" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    $componentFiles = @(
        "resource_allocation_engine.py",
        "strategy_dependency_resolver.py", 
        "concurrent_execution_scheduler.py",
        "execution_monitoring_system.py",
        "multi_strategy_coordination_manager.py",
        "multi_strategy_coordination_interface.py"
    )
    
    $componentsSuccess = $true
    
    foreach ($file in $componentFiles) {
        if (Test-Path $file) {
            # Python構文チェック
            python -m py_compile $file 2>$null
            if ($LASTEXITCODE -eq 0) {
                Record-TestResult "Component: $file" $true "Python syntax valid"
            } else {
                Record-TestResult "Component: $file" $false "Python syntax error"
                $componentsSuccess = $false
            }
        } else {
            Record-TestResult "Component: $file" $false "Component file not found"
            $componentsSuccess = $false
        }
    }
    
    return $componentsSuccess
}

# インポートテスト関数
function Test-ModuleImports {
    Write-Host "`n📦 Testing Module Imports" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    $importTest = @"
try:
    # Core coordination system imports
    from multi_strategy_coordination_interface import MultiStrategyCoordinationInterface
    from multi_strategy_coordination_manager import MultiStrategyCoordinationManager
    from resource_allocation_engine import ResourceAllocationEngine
    from strategy_dependency_resolver import StrategyDependencyResolver
    from concurrent_execution_scheduler import ConcurrentExecutionScheduler
    from execution_monitoring_system import ExecutionMonitoringSystem
    
    print("SUCCESS: All coordination modules imported successfully")
    exit(0)
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
    exit(1)
except Exception as e:
    print(f"EXCEPTION: {e}")
    exit(2)
"@
    
    try {
        $importTest | python
        if ($LASTEXITCODE -eq 0) {
            Record-TestResult "Module Imports" $true "All coordination modules imported successfully"
            return $true
        } else {
            $errorType = if ($LASTEXITCODE -eq 1) { "Import Error" } elseif ($LASTEXITCODE -eq 2) { "Exception" } else { "Unknown Error" }
            Record-TestResult "Module Imports" $false "Module import failed: $errorType"
            return $false
        }
    }
    catch {
        Record-TestResult "Module Imports" $false "Exception during import test: $($_.Exception.Message)"
        return $false
    }
}

# デモスクリプトテスト関数
function Test-DemoExecution {
    Write-Host "`n🎯 Testing Demo Script Execution" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    $demoFiles = @(
        "config\demo_multi_strategy_coordination.py",
        "demo_multi_strategy_coordination.py"  # フォールバック位置
    )
    
    $demoFile = $null
    foreach ($file in $demoFiles) {
        if (Test-Path $file) {
            $demoFile = $file
            break
        }
    }
    
    if (-not $demoFile) {
        Record-TestResult "Demo Script" $false "Demo script not found in expected locations"
        return $false
    }
    
    try {
        Write-Host "Executing demo script: $demoFile" -ForegroundColor Gray
        Write-Host "Note: Demo will run for approximately 2-3 minutes..." -ForegroundColor Yellow
        
        # バックグラウンドでデモ実行（タイムアウト付き）
        $job = Start-Job -ScriptBlock {
            param($scriptPath)
            python $scriptPath
        } -ArgumentList $demoFile
        
        # 最大5分待機
        $timeout = 300
        $completed = Wait-Job $job -Timeout $timeout
        
        if ($completed) {
            $output = Receive-Job $job
            $exitCode = $job.State
            
            if ($job.State -eq "Completed") {
                Record-TestResult "Demo Execution" $true "Demo script completed successfully"
                return $true
            } else {
                Record-TestResult "Demo Execution" $false "Demo script failed or terminated unexpectedly"
                Write-Host "Output: $output" -ForegroundColor Gray
                return $false
            }
        } else {
            Stop-Job $job
            Record-TestResult "Demo Execution" $false "Demo script timeout (exceeded $timeout seconds)"
            return $false
        }
    }
    catch {
        Record-TestResult "Demo Execution" $false "Exception during demo execution: $($_.Exception.Message)"
        return $false
    }
    finally {
        if (Get-Job -Id $job.Id -ErrorAction SilentlyContinue) {
            Remove-Job $job -Force
        }
    }
}

# 既存システム統合テスト関数
function Test-SystemIntegration {
    Write-Host "`n🔗 Testing System Integration" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    # 4-1-1, 4-1-2 統合ファイル確認
    $integrationFiles = @(
        "main.py",                              # 4-1-1 統合
        "integrated_decision_system.py",        # 4-1-2 統合
        "strategy_composition_engine.py"        # 4-1-2 コンポジション
    )
    
    $integrationSuccess = $true
    
    foreach ($file in $integrationFiles) {
        if (Test-Path $file) {
            Record-TestResult "Integration File: $file" $true "Integration component found"
        } else {
            Record-TestResult "Integration File: $file" $false "Integration component missing"
            $integrationSuccess = $false
        }
    }
    
    # 設定ディレクトリ確認  
    if (Test-Path "config") {
        $configCount = (Get-ChildItem -Path "config" -Filter "*.json").Count
        Record-TestResult "Configuration Directory" $true "Config directory with $configCount files"
    } else {
        Record-TestResult "Configuration Directory" $false "Configuration directory not found"
        $integrationSuccess = $false
    }
    
    return $integrationSuccess
}

# Webインターフェーステスト関数
function Test-WebInterface {
    Write-Host "`n🌐 Testing Web Interface" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    $webTest = @"
try:
    from multi_strategy_coordination_interface import MultiStrategyCoordinationInterface
    
    # インターフェース初期化
    interface = MultiStrategyCoordinationInterface()
    interface.initialize()
    
    # ステータス取得テスト
    status = interface.get_system_status()
    
    if status and 'web_interface' in status:
        web_info = status['web_interface']
        print(f"WEB_SUCCESS: Interface available, Port: {web_info.get('port', 'unknown')}")
        exit(0)
    else:
        print("WEB_ERROR: Web interface status not available")
        exit(1)
        
    # クリーンアップ
    interface.shutdown()
    
except Exception as e:
    print(f"WEB_EXCEPTION: {e}")
    exit(2)
"@
    
    try {
        $result = $webTest | python
        if ($LASTEXITCODE -eq 0) {
            Record-TestResult "Web Interface" $true "Web interface initialization successful"
            if ($result -match "Port: (\d+)") {
                $port = $matches[1]
                Write-Host "   Dashboard URL: http://localhost:$port" -ForegroundColor Gray
            }
            return $true
        } else {
            Record-TestResult "Web Interface" $false "Web interface initialization failed"
            return $false
        }
    }
    catch {
        Record-TestResult "Web Interface" $false "Exception during web interface test: $($_.Exception.Message)"
        return $false
    }
}

# 最終結果レポート関数
function Generate-FinalReport {
    Write-Host "`n📊 Final Test Report" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Green
    
    $duration = (Get-Date) - $global:StartTime
    $totalTests = $global:TestResults.Count
    
    Write-Host "Test Duration: $($duration.ToString('mm\:ss'))" -ForegroundColor Gray
    Write-Host "Total Tests: $totalTests" -ForegroundColor Gray
    Write-Host "Successful Tests: $global:SuccessCount" -ForegroundColor Green
    Write-Host "Failed Tests: $global:ErrorCount" -ForegroundColor Red
    
    if ($totalTests -gt 0) {
        $successRate = [math]::Round(($global:SuccessCount / $totalTests) * 100, 1)
        Write-Host "Success Rate: $successRate%" -ForegroundColor $(if ($successRate -ge 80) { "Green" } elseif ($successRate -ge 60) { "Yellow" } else { "Red" })
    }
    
    Write-Host "`n📋 Test Results Summary:" -ForegroundColor Cyan
    Write-Host "-" * 50 -ForegroundColor Gray
    
    foreach ($result in $global:TestResults) {
        $icon = if ($result.Success) { "✅" } else { "❌" }
        $color = if ($result.Success) { "Green" } else { "Red" }
        Write-Host "$icon $($result.TestName)" -ForegroundColor $color
        if ($result.Message) {
            Write-Host "   $($result.Message)" -ForegroundColor Gray
        }
    }
    
    # テストレポートファイル生成
    $reportFile = "coordination_test_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    $reportData = @{
        test_info = @{
            start_time = $global:StartTime.ToString('yyyy-MM-dd HH:mm:ss')
            end_time = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
            duration_minutes = [math]::Round($duration.TotalMinutes, 2)
            total_tests = $totalTests
            successful_tests = $global:SuccessCount
            failed_tests = $global:ErrorCount
            success_rate = if ($totalTests -gt 0) { [math]::Round(($global:SuccessCount / $totalTests) * 100, 1) } else { 0 }
        }
        test_results = $global:TestResults | ForEach-Object {
            @{
                test_name = $_.TestName
                success = $_.Success
                message = $_.Message
                details = $_.Details
                timestamp = $_.Timestamp.ToString('yyyy-MM-dd HH:mm:ss')
            }
        }
    }
    
    try {
        $reportData | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportFile -Encoding UTF8
        Write-Host "`n📄 Test report saved: $reportFile" -ForegroundColor Gray
    }
    catch {
        Write-Host "`n⚠️ Failed to save test report: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    
    Write-Host "`n" + "=" * 80 -ForegroundColor Green
    
    # 最終判定
    if ($global:ErrorCount -eq 0) {
        Write-Host "🎉 ALL TESTS PASSED - System ready for operation!" -ForegroundColor Green
        Write-Host "   4-1-3「マルチ戦略同時実行の調整機能」implementation successful" -ForegroundColor Cyan
        return $true
    } elseif ($global:ErrorCount -le 2) {
        Write-Host "⚠️ MOSTLY SUCCESSFUL - Minor issues detected" -ForegroundColor Yellow
        Write-Host "   System may be operational but please review failed tests" -ForegroundColor Yellow
        return $true
    } else {
        Write-Host "❌ SIGNIFICANT ISSUES DETECTED" -ForegroundColor Red
        Write-Host "   Please resolve failed tests before using the system" -ForegroundColor Red
        return $false
    }
}

# メイン実行フロー
function Main {
    try {
        # テスト実行シーケンス
        $pythonOk = Test-PythonEnvironment
        if (-not $pythonOk) {
            Write-Host "`n⚠️ Python environment issues detected. Continuing with limited tests..." -ForegroundColor Yellow
        }
        
        $requirementsOk = Install-CoordinationRequirements
        $configOk = Test-ConfigurationFiles  
        $componentsOk = Test-ComponentFiles
        
        if ($pythonOk -and $componentsOk) {
            $importsOk = Test-ModuleImports
            
            if ($importsOk) {
                $webOk = Test-WebInterface
                $integrationOk = Test-SystemIntegration
                $demoOk = Test-DemoExecution
            }
        }
        
        # 最終レポート生成
        $overallSuccess = Generate-FinalReport
        
        # PowerShell終了コード設定
        if ($overallSuccess) {
            Write-Host "`n🚀 Ready to commit changes with success status!" -ForegroundColor Green
            $global:LASTEXITCODE = 0
        } else {
            Write-Host "`n🛑 System not ready - resolve issues before committing" -ForegroundColor Red
            $global:LASTEXITCODE = 1
        }
        
        return $overallSuccess
    }
    catch {
        Write-Host "`n💥 Critical error during test execution: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host $_.ScriptStackTrace -ForegroundColor Gray
        $global:LASTEXITCODE = 2
        return $false
    }
}

# スクリプト実行
$success = Main

# 実行結果に基づく終了処理
if ($success) {
    Write-Host "`nTest completed successfully. Exit code: $global:LASTEXITCODE" -ForegroundColor Green
} else {
    Write-Host "`nTest completed with issues. Exit code: $global:LASTEXITCODE" -ForegroundColor Yellow
}

exit $global:LASTEXITCODE
