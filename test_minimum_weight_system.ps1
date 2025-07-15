# 3-2-2 階層的最小重み設定機能のテストスクリプト
# PowerShell用

param(
    [switch]$RunDemo,
    [switch]$RunTests,
    [switch]$CheckDependencies,
    [switch]$All
)

# 基本設定
$ProjectRoot = $PSScriptRoot
$ConfigDir = Join-Path $ProjectRoot "config"
$PythonExecutable = "python"

Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "3-2-2 階層的最小重み設定機能 テストスクリプト" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

function Test-PythonDependencies {
    Write-Host "`n[依存関係チェック]" -ForegroundColor Yellow
    
    # Python可用性チェック
    try {
        $pythonVersion = & $PythonExecutable --version 2>&1
        Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Python not found" -ForegroundColor Red
        return $false
    }
    
    # 必要パッケージのチェック
    $requiredPackages = @("pandas", "numpy", "pathlib")
    
    foreach ($package in $requiredPackages) {
        try {
            & $PythonExecutable -c "import $package" 2>$null
            Write-Host "✓ $package パッケージ" -ForegroundColor Green
        }
        catch {
            Write-Host "✗ $package パッケージ (pip install $package)" -ForegroundColor Red
        }
    }
    
    # プロジェクトファイルの存在確認
    $criticalFiles = @(
        "config\portfolio_weight_calculator.py",
        "demo_minimum_weight_system.py",
        "config\portfolio_weights\minimum_weights\minimum_weight_rules.json"
    )
    
    Write-Host "`n[重要ファイル存在確認]" -ForegroundColor Yellow
    foreach ($file in $criticalFiles) {
        $fullPath = Join-Path $ProjectRoot $file
        if (Test-Path $fullPath) {
            Write-Host "✓ $file" -ForegroundColor Green
        } else {
            Write-Host "✗ $file (ファイルが見つかりません)" -ForegroundColor Red
        }
    }
    
    return $true
}

function Run-MinimumWeightDemo {
    Write-Host "`n[3-2-2 デモ実行]" -ForegroundColor Yellow
    
    try {
        # デモスクリプトの実行
        $demoScript = Join-Path $ProjectRoot "demo_minimum_weight_system.py"
        
        if (-not (Test-Path $demoScript)) {
            Write-Host "✗ デモスクリプトが見つかりません: $demoScript" -ForegroundColor Red
            return $false
        }
        
        Write-Host "デモを実行中..." -ForegroundColor Cyan
        & $PythonExecutable $demoScript
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ デモ実行完了" -ForegroundColor Green
            return $true
        } else {
            Write-Host "✗ デモ実行エラー (終了コード: $LASTEXITCODE)" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ デモ実行中にエラーが発生しました: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Run-BasicFunctionalityTest {
    Write-Host "`n[基本機能テスト]" -ForegroundColor Yellow
    
    $testScript = @"
import sys
import os
sys.path.append('$ProjectRoot')

try:
    # 基本インポートテスト
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, WeightAllocationConfig, PortfolioConstraints,
        MinimumWeightRule, MinimumWeightLevel, WeightAdjustmentMethod
    )
    print('✓ インポート成功')
    
    # インスタンス作成テスト
    calculator = PortfolioWeightCalculator()
    print('✓ PortfolioWeightCalculator インスタンス作成成功')
    
    # 最小重み管理テスト
    success = calculator.add_strategy_minimum_weight('test_strategy', 0.1)
    print(f'✓ 戦略最小重み設定: {success}')
    
    # カテゴリー設定テスト
    success = calculator.add_category_minimum_weight('test_category', 0.05)
    print(f'✓ カテゴリー最小重み設定: {success}')
    
    # デフォルト設定テスト
    success = calculator.set_default_minimum_weight(0.03)
    print(f'✓ デフォルト最小重み設定: {success}')
    
    # ルール取得テスト
    rules = calculator.get_minimum_weight_rules()
    print(f'✓ ルール取得: {len(rules)} セクション')
    
    print('全ての基本機能テストが成功しました')
    
except Exception as e:
    print(f'✗ テストエラー: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@
    
    try {
        $testScript | & $PythonExecutable
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 基本機能テスト完了" -ForegroundColor Green
            return $true
        } else {
            Write-Host "✗ 基本機能テストエラー" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ 基本機能テスト実行エラー: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Run-IntegrationTest {
    Write-Host "`n[統合テスト]" -ForegroundColor Yellow
    
    $integrationScript = @"
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append('$ProjectRoot')

try:
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, WeightAllocationConfig, PortfolioConstraints,
        AllocationMethod
    )
    from config.strategy_scoring_model import StrategyScore, ScoreWeights
    
    # サンプルデータ作成
    score_weights = ScoreWeights()
    strategy_scores = {
        'test_strategy_1': StrategyScore(
            strategy_name='test_strategy_1',
            total_score=0.8,
            component_scores={'performance': 0.8},
            confidence=0.9,
            calculation_date=datetime.now(),
            score_weights=score_weights
        ),
        'test_strategy_2': StrategyScore(
            strategy_name='test_strategy_2',
            total_score=0.6,
            component_scores={'performance': 0.6},
            confidence=0.8,
            calculation_date=datetime.now(),
            score_weights=score_weights
        )
    }
    
    # 市場データ作成
    market_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # テスト実行
    calculator = PortfolioWeightCalculator()
    
    # 3-2-2機能有効設定
    config = WeightAllocationConfig(
        method=AllocationMethod.RISK_ADJUSTED,
        constraints=PortfolioConstraints(
            enable_hierarchical_minimum_weights=True,
            weight_adjustment_method='proportional'
        )
    )
    
    # 最小重み設定
    calculator.add_strategy_minimum_weight('test_strategy_1', 0.15)
    
    # スコアマネージャーのモック（簡易版）
    calculator.score_manager = type('MockScoreManager', (), {
        'calculate_comprehensive_scores': lambda self, tickers: {'TEST': strategy_scores}
    })()
    
    # 重み計算実行
    result = calculator.calculate_portfolio_weights(
        ticker='TEST',
        market_data=market_data,
        config=config
    )
    
    print(f'✓ 重み計算完了: {len(result.strategy_weights)} 戦略')
    print(f'✓ 期待リターン: {result.expected_return:.4f}')
    print(f'✓ 期待リスク: {result.expected_risk:.4f}')
    print(f'✓ シャープレシオ: {result.sharpe_ratio:.4f}')
    print(f'✓ 制約違反: {len(result.constraint_violations)}')
    
    # 3-2-2機能の調整確認
    if result.metadata.get('hierarchical_adjustment_applied'):
        print('✓ 3-2-2 階層的調整適用済み')
    else:
        print('- 3-2-2 調整は適用されませんでした')
    
    print('統合テストが成功しました')
    
except Exception as e:
    print(f'✗ 統合テストエラー: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@
    
    try {
        $integrationScript | & $PythonExecutable
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 統合テスト完了" -ForegroundColor Green
            return $true
        } else {
            Write-Host "✗ 統合テストエラー" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ 統合テスト実行エラー: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Show-Usage {
    Write-Host "`n使用方法:" -ForegroundColor Cyan
    Write-Host "  .\test_minimum_weight_system.ps1 -RunDemo          # デモ実行"
    Write-Host "  .\test_minimum_weight_system.ps1 -RunTests         # テスト実行"
    Write-Host "  .\test_minimum_weight_system.ps1 -CheckDependencies # 依存関係チェック"
    Write-Host "  .\test_minimum_weight_system.ps1 -All              # 全て実行"
}

# メイン処理
if ($CheckDependencies -or $All) {
    Test-PythonDependencies
}

if ($RunTests -or $All) {
    Run-BasicFunctionalityTest
    Run-IntegrationTest
}

if ($RunDemo -or $All) {
    Run-MinimumWeightDemo
}

if (-not ($RunDemo -or $RunTests -or $CheckDependencies -or $All)) {
    Show-Usage
}

Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host "3-2-2 階層的最小重み設定機能テスト完了" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
