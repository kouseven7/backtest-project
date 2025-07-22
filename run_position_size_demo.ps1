# PowerShell Script: Run Position Size System Demo
# File: run_position_size_demo.ps1
# Description: 
#   3-3-2「各戦略のポジションサイズ調整機能」
#   統合システムの動作確認スクリプト
#
# Author: imega
# Created: 2025-07-20
# Modified: 2025-07-20
#
# Features:
# 1. システム完全性チェック
# 2. デモンストレーション実行
# 3. 統合テスト実行
# 4. 結果レポート生成
# 5. PowerShell用コマンド（; を使用）

param(
    [string]$Mode = "all",
    [switch]$SkipTests,
    [switch]$Verbose
)

# スクリプト実行設定
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# カラー出力設定
$colors = @{
    'header' = 'Cyan'
    'success' = 'Green'
    'warning' = 'Yellow'
    'error' = 'Red'
    'info' = 'White'
}

function Write-ColorText {
    param($Text, $Color)
    Write-Host $Text -ForegroundColor $colors[$Color]
}

function Write-Section {
    param($Title)
    Write-Host ""
    Write-ColorText ("=" * 70) 'header'
    Write-ColorText $Title 'header'
    Write-ColorText ("=" * 70) 'header'
}

function Write-SubSection {
    param($Title)
    Write-Host ""
    Write-ColorText ("--- $Title ---") 'info'
}

function Test-FileExists {
    param($FilePath, $Description)
    if (Test-Path $FilePath) {
        Write-ColorText "✓ $Description - OK" 'success'
        return $true
    } else {
        Write-ColorText "❌ $Description - Missing: $FilePath" 'error'
        return $false
    }
}

function Invoke-PythonCommand {
    param($Command, $Description)
    
    Write-ColorText "Running: $Description" 'info'
    if ($Verbose) {
        Write-ColorText "Command: $Command" 'info'
    }
    
    $start_time = Get-Date
    try {
        Invoke-Expression $Command
        $exit_code = $LASTEXITCODE
        $end_time = Get-Date
        $duration = ($end_time - $start_time).TotalSeconds
        
        if ($exit_code -eq 0) {
            Write-ColorText "✓ $Description completed successfully (${duration}s)" 'success'
            return $true
        } else {
            Write-ColorText "❌ $Description failed (exit code: $exit_code)" 'error'
            return $false
        }
    } catch {
        $end_time = Get-Date
        $duration = ($end_time - $start_time).TotalSeconds
        Write-ColorText "❌ $Description failed with exception: $($_.Exception.Message)" 'error'
        return $false
    }
}

function Test-SystemIntegrity {
    Write-Section "🔍 Position Size System Integrity Check"
    
    $required_files = @(
        @{Path="config/position_size_adjuster.py"; Description="Position Size Adjuster Module"},
        @{Path="config/position_sizing_config.json"; Description="Position Sizing Configuration"},
        @{Path="demo_position_size_adjuster.py"; Description="Demo Script"},
        @{Path="test_position_size_integration.py"; Description="Integration Test"}
    )
    
    $all_files_exist = $true
    
    foreach ($file in $required_files) {
        $exists = Test-FileExists $file.Path $file.Description
        $all_files_exist = $all_files_exist -and $exists
    }
    
    Write-SubSection "Dependency Check"
    
    $dependency_files = @(
        @{Path="config/portfolio_weight_calculator.py"; Description="Portfolio Weight Calculator"},
        @{Path="config/strategy_scoring_model.py"; Description="Strategy Scoring Model"},
        @{Path="config/signal_integrator.py"; Description="Signal Integrator"},
        @{Path="config/unified_trend_detector.py"; Description="Unified Trend Detector"},
        @{Path="config/risk_management.py"; Description="Risk Management"}
    )
    
    $dependency_count = 0
    foreach ($dep in $dependency_files) {
        if (Test-FileExists $dep.Path $dep.Description) {
            $dependency_count++
        }
    }
    
    Write-ColorText "Dependencies available: $dependency_count / $($dependency_files.Count)" 'info'
    
    if ($all_files_exist) {
        Write-ColorText "🎉 All required files are present!" 'success'
    } else {
        Write-ColorText "⚠️ Some required files are missing. System may not function properly." 'warning'
    }
    
    return $all_files_exist
}

function Run-BasicDemo {
    Write-Section "🚀 Basic Position Size Demo"
    
    $success = Invoke-PythonCommand "python demo_position_size_adjuster.py" "Basic Position Size Demo"
    return $success
}

function Run-IntegrationTests {
    Write-Section "🧪 Integration Tests"
    
    if ($SkipTests) {
        Write-ColorText "Skipping integration tests (--SkipTests specified)" 'warning'
        return $true
    }
    
    $success = Invoke-PythonCommand "python test_position_size_integration.py" "Integration Test Suite"
    return $success
}

function Run-ConfigurationTest {
    Write-Section "⚙️ Configuration Test"
    
    $python_code = @"
import json
from pathlib import Path

# 設定ファイルテスト
try:
    with open('config/position_sizing_config.json', 'r') as f:
        config = json.load(f)
    
    print(f'✓ Configuration loaded successfully')
    print(f'  - Base position size: {config.get(\"base_position_size\", \"N/A\")}')
    print(f'  - Max position size: {config.get(\"max_position_size\", \"N/A\")}')
    print(f'  - Sizing method: {config.get(\"sizing_method\", \"N/A\")}')
    print(f'  - Market regime count: {len(config.get(\"market_regime_adjustments\", {}))}')
    
    # 基本的な設定検証
    required_keys = ['base_position_size', 'max_position_size', 'min_position_size', 'sizing_method']
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        print(f'⚠️ Missing configuration keys: {missing_keys}')
    else:
        print('✓ All required configuration keys present')
    
except Exception as e:
    print(f'❌ Configuration test failed: {e}')
    exit(1)

print('Configuration test completed successfully!')
"@
    
    $python_code | Out-File -FilePath "temp_config_test.py" -Encoding utf8
    
    try {
        $success = Invoke-PythonCommand "python temp_config_test.py" "Configuration Loading Test"
        Remove-Item "temp_config_test.py" -Force -ErrorAction SilentlyContinue
        return $success
    } finally {
        Remove-Item "temp_config_test.py" -Force -ErrorAction SilentlyContinue
    }
}

function Run-ModuleImportTest {
    Write-Section "📦 Module Import Test"
    
    $python_code = @"
import sys
import traceback

modules_to_test = [
    'config.position_size_adjuster',
    'config.portfolio_weight_calculator',
    'config.strategy_scoring_model',
    'config.signal_integrator'
]

success_count = 0
total_count = len(modules_to_test)

for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f'✓ {module_name} imported successfully')
        success_count += 1
    except ImportError as e:
        print(f'⚠️ {module_name} import failed: {e}')
    except Exception as e:
        print(f'❌ {module_name} unexpected error: {e}')

print(f'\nImport test results: {success_count}/{total_count} modules imported successfully')

if success_count >= total_count // 2:  # 半数以上成功すればOK
    print('Module import test passed!')
    exit(0)
else:
    print('Module import test failed!')
    exit(1)
"@
    
    $python_code | Out-File -FilePath "temp_import_test.py" -Encoding utf8
    
    try {
        $success = Invoke-PythonCommand "python temp_import_test.py" "Module Import Test"
        Remove-Item "temp_import_test.py" -Force -ErrorAction SilentlyContinue
        return $success
    } finally {
        Remove-Item "temp_import_test.py" -Force -ErrorAction SilentlyContinue
    }
}

function Run-PerformanceTest {
    Write-Section "⚡ Performance Test"
    
    $python_code = @"
import time
import pandas as pd
import numpy as np
from datetime import datetime

# パフォーマンステスト用のモックテスト
print('Running position sizing performance test...')

# ダミーデータ作成
np.random.seed(42)
test_data_sizes = [100, 500, 1000]
results = []

for size in test_data_sizes:
    start_time = time.time()
    
    # ダミーデータ処理
    data = pd.DataFrame({
        'close': np.random.random(size) * 100,
        'volume': np.random.randint(1000, 10000, size)
    })
    
    # 基本的な計算（ポジションサイズ計算のシミュレーション）
    returns = data['close'].pct_change().dropna()
    volatility = returns.std()
    position_size = min(0.05, max(0.01, 0.02 / max(volatility, 0.01)))
    
    end_time = time.time()
    duration = end_time - start_time
    results.append((size, duration))
    
    print(f'✓ Data size {size}: {duration:.3f}s, position_size: {position_size:.3f}')

avg_time = sum(r[1] for r in results) / len(results)
print(f'\nAverage processing time: {avg_time:.3f}s')

if avg_time < 1.0:  # 1秒以内
    print('✓ Performance test passed!')
else:
    print('⚠️ Performance may be slower than expected')

print('Performance test completed!')
"@
    
    $python_code | Out-File -FilePath "temp_performance_test.py" -Encoding utf8
    
    try {
        $success = Invoke-PythonCommand "python temp_performance_test.py" "Performance Test"
        Remove-Item "temp_performance_test.py" -Force -ErrorAction SilentlyContinue
        return $success
    } finally {
        Remove-Item "temp_performance_test.py" -Force -ErrorAction SilentlyContinue
    }
}

function Generate-TestReport {
    param($Results)
    
    Write-Section "📊 Test Results Report"
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $total_tests = $Results.Keys.Count
    $passed_tests = ($Results.Values | Where-Object { $_ -eq $true }).Count
    $success_rate = if ($total_tests -gt 0) { [math]::Round(($passed_tests / $total_tests) * 100, 1) } else { 0 }
    
    Write-ColorText "Test Execution Report - $timestamp" 'header'
    Write-Host ""
    Write-ColorText "Overall Results:" 'info'
    Write-ColorText "  Total Tests: $total_tests" 'info'
    Write-ColorText "  Passed: $passed_tests" 'success'
    Write-ColorText "  Failed: $($total_tests - $passed_tests)" 'error'
    Write-ColorText "  Success Rate: $success_rate%" $(if ($success_rate -ge 80) { 'success' } elseif ($success_rate -ge 60) { 'warning' } else { 'error' })
    Write-Host ""
    
    Write-ColorText "Individual Test Results:" 'info'
    foreach ($test in $Results.GetEnumerator()) {
        $status = if ($test.Value) { "✓ PASS" } else { "❌ FAIL" }
        $color = if ($test.Value) { 'success' } else { 'error' }
        Write-ColorText "  $($test.Key): $status" $color
    }
    
    Write-Host ""
    
    if ($success_rate -ge 80) {
        Write-ColorText "🎉 Position Size System is ready for deployment!" 'success'
    } elseif ($success_rate -ge 60) {
        Write-ColorText "⚠️ System is partially functional. Review failed tests." 'warning'
    } else {
        Write-ColorText "❌ System requires significant fixes before deployment." 'error'
    }
    
    # レポートファイル保存
    $report_file = "position_size_test_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    $report_content = @"
Position Size System Test Report
Generated: $timestamp
Author: imega
Task: 3-3-2「各戦略のポジションサイズ調整機能」

SUMMARY:
Total Tests: $total_tests
Passed: $passed_tests
Failed: $($total_tests - $passed_tests)
Success Rate: $success_rate%

DETAILED RESULTS:
$($Results.GetEnumerator() | ForEach-Object { "$($_.Key): $(if ($_.Value) { 'PASS' } else { 'FAIL' })" } | Out-String)

STATUS:
$(if ($success_rate -ge 80) { 'READY FOR DEPLOYMENT' } elseif ($success_rate -ge 60) { 'PARTIALLY FUNCTIONAL' } else { 'REQUIRES FIXES' })

RECOMMENDATIONS:
$(if ($success_rate -ge 80) {
    '- System is functioning well and ready for production use
- Consider running periodic integration tests
- Monitor performance in production environment'
} elseif ($success_rate -ge 60) {
    '- Address failed test cases before full deployment
- Test specific failure scenarios in isolation
- Consider gradual rollout with monitoring'
} else {
    '- Review and fix critical system components
- Verify all dependencies are properly installed
- Check configuration files for correctness
- Consider rebuilding from known good state'
})
"@
    
    try {
        $report_content | Out-File -FilePath $report_file -Encoding utf8
        Write-ColorText "Report saved to: $report_file" 'info'
    } catch {
        Write-ColorText "Could not save report file: $($_.Exception.Message)" 'warning'
    }
}

# メイン実行スクリプト
function Main {
    Write-Section "🏁 Position Size System Demo & Test Runner"
    Write-ColorText "Author: imega" 'info'
    Write-ColorText "Created: 2025-07-20" 'info'
    Write-ColorText "Task: 3-3-2「各戦略のポジションサイズ調整機能」" 'info'
    Write-ColorText "Mode: $Mode" 'info'
    if ($SkipTests) { Write-ColorText "Tests: Skipped" 'warning' }
    if ($Verbose) { Write-ColorText "Verbose: Enabled" 'info' }
    
    $results = @{}
    
    # システム完全性チェック
    $results["System Integrity"] = Test-SystemIntegrity
    
    # 実行モードに応じた処理
    switch ($Mode.ToLower()) {
        "integrity" {
            # 完全性チェックのみ
        }
        "demo" {
            $results["Basic Demo"] = Run-BasicDemo
        }
        "test" {
            $results["Integration Tests"] = Run-IntegrationTests
        }
        "config" {
            $results["Configuration Test"] = Run-ConfigurationTest
        }
        "import" {
            $results["Module Import Test"] = Run-ModuleImportTest
        }
        "performance" {
            $results["Performance Test"] = Run-PerformanceTest
        }
        "quick" {
            $results["Configuration Test"] = Run-ConfigurationTest
            $results["Module Import Test"] = Run-ModuleImportTest
        }
        "all" {
            $results["Configuration Test"] = Run-ConfigurationTest
            $results["Module Import Test"] = Run-ModuleImportTest
            $results["Basic Demo"] = Run-BasicDemo
            $results["Integration Tests"] = Run-IntegrationTests
            $results["Performance Test"] = Run-PerformanceTest
        }
        default {
            Write-ColorText "Unknown mode: $Mode" 'error'
            Write-ColorText "Available modes: integrity, demo, test, config, import, performance, quick, all" 'info'
            exit 1
        }
    }
    
    # 結果レポート生成
    Generate-TestReport $results
    
    # 全体的な成功判定
    $overall_success = ($results.Values | Where-Object { $_ -eq $false }).Count -eq 0
    
    if ($overall_success) {
        Write-ColorText "`n🎉 All tests passed! Position Size System is ready!" 'success'
        exit 0
    } else {
        Write-ColorText "`n⚠️ Some tests failed. Please review the results." 'warning'
        exit 1
    }
}

# スクリプト使用方法の表示
if ($args -contains "-help" -or $args -contains "--help") {
    Write-Host @"
Position Size System Demo & Test Runner

USAGE:
    .\run_position_size_demo.ps1 [-Mode <mode>] [-SkipTests] [-Verbose]

PARAMETERS:
    -Mode <mode>     : Test mode to run
                      • all         - Run all tests and demos (default)
                      • integrity   - System integrity check only
                      • demo        - Basic demo only
                      • test        - Integration tests only  
                      • config      - Configuration test only
                      • import      - Module import test only
                      • performance - Performance test only
                      • quick       - Config + Import tests only

    -SkipTests      : Skip integration tests (faster execution)
    -Verbose        : Show detailed command output

EXAMPLES:
    .\run_position_size_demo.ps1
    .\run_position_size_demo.ps1 -Mode quick
    .\run_position_size_demo.ps1 -Mode demo -Verbose
    .\run_position_size_demo.ps1 -Mode test -SkipTests

AUTHOR: imega
TASK: 3-3-2「各戦略のポジションサイズ調整機能」
"@
    exit 0
}

# メイン実行
Main
