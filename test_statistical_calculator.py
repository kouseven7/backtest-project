# StatisticalCalculator動作確認テスト
import sys
sys.path.append('analysis')
from performance_metrics import StatisticalCalculator, CalculationConfig
import numpy as np

def test_statistical_calculator():
    """StatisticalCalculator機能テスト"""
    print('=== StatisticalCalculator動作確認テスト ===')
    
    # テストデータ作成
    test_trades = [
        {'profit': 100.0},  # 勝ち
        {'profit': -50.0},  # 負け
        {'profit': 75.0},   # 勝ち
        {'profit': -25.0},  # 負け
        {'profit': 150.0}   # 勝ち
    ]
    
    empty_trades = []
    zero_trades = [{'profit': 0.0}]
    
    calculator = StatisticalCalculator()
    
    # 1. win_rate テスト
    print('\n1. 勝率計算テスト')
    win_rate = calculator.calculate_win_rate(test_trades)
    expected_win_rate = 60.0  # 3勝/5取引 = 60%
    print(f'  勝率: {win_rate}% (期待値: {expected_win_rate}%)')
    print(f'  正確性: {"✓" if abs(win_rate - expected_win_rate) < 0.01 else "✗"}')
    
    # 空データテスト
    empty_win_rate = calculator.calculate_win_rate(empty_trades)
    print(f'  空データ勝率: {empty_win_rate}% (期待値: 0%)')
    
    # 2. profit_factor テスト
    print('\n2. プロフィットファクター計算テスト')
    profit_factor = calculator.calculate_profit_factor(test_trades)
    # gross_profit = 100 + 75 + 150 = 325
    # gross_loss = 50 + 25 = 75
    # profit_factor = 325 / 75 = 4.333...
    expected_pf = 325.0 / 75.0
    print(f'  プロフィットファクター: {profit_factor} (期待値: {expected_pf:.3f})')
    print(f'  正確性: {"✓" if abs(profit_factor - expected_pf) < 0.01 else "✗"}')
    
    # ゼロ除算テスト（損失なし）
    profit_only_trades = [{'profit': 100.0}, {'profit': 50.0}]
    pf_no_loss = calculator.calculate_profit_factor(profit_only_trades)
    print(f'  損失なしPF: {pf_no_loss} (期待値: 999.999)')
    
    # 3. sharpe_ratio テスト
    print('\n3. シャープレシオ計算テスト')
    test_returns = [0.01, -0.005, 0.02, -0.01, 0.015]
    sharpe = calculator.calculate_sharpe_ratio(test_returns)
    
    # 手計算確認
    mean_ret = np.mean(test_returns)
    std_ret = np.std(test_returns, ddof=1)
    expected_sharpe = mean_ret / std_ret if std_ret > 0 else 0
    print(f'  シャープレシオ: {sharpe} (期待値: {expected_sharpe:.6f})')
    print(f'  正確性: {"✓" if abs(sharpe - expected_sharpe) < 0.01 else "✗"}')
    
    # 標準偏差0テスト
    zero_std_returns = [0.01, 0.01, 0.01]
    sharpe_zero_std = calculator.calculate_sharpe_ratio(zero_std_returns)
    print(f'  標準偏差0シャープレシオ: {sharpe_zero_std} (期待値: 0)')
    
    # 4. max_drawdown テスト
    print('\n4. 最大ドローダウン計算テスト')
    portfolio_values = [1000, 1100, 950, 1200, 900, 1150]
    max_dd = calculator.calculate_max_drawdown(portfolio_values)
    
    # 手計算: peak 1200 -> valley 900 = (1200-900)/1200 = 25%
    expected_dd = 25.0
    print(f'  最大ドローダウン: {max_dd}% (期待値: {expected_dd}%)')
    print(f'  正確性: {"✓" if abs(max_dd - expected_dd) < 0.01 else "✗"}')
    
    # 5. エラー対策テスト
    print('\n5. エラー対策テスト')
    
    # NaN データ
    nan_trades = [{'profit': float('nan')}, {'profit': 100}]
    nan_win_rate = calculator.calculate_win_rate(nan_trades)
    print(f'  NaN含有勝率: {nan_win_rate}% (期待値: 100%)')
    
    # None データ
    none_trades = [{'profit': None}, {'profit': 50}]
    none_pf = calculator.calculate_profit_factor(none_trades)
    print(f'  None含有PF: {none_pf} (期待値: 999.999)')
    
    # 6. 統合サマリーテスト
    print('\n6. 統合サマリーテスト')
    summary = calculator.get_calculation_summary(
        trades_data=test_trades,
        returns_data=test_returns,
        portfolio_data=portfolio_values
    )
    
    print(f'  統合サマリー生成: {"✓" if "win_rate" in summary else "✗"}')
    print(f'  データ品質スコア: {summary["data_quality"]["quality_score"]:.2f}')
    
    print('\n=== 動作確認テスト完了 ===')
    return True

if __name__ == "__main__":
    test_statistical_calculator()