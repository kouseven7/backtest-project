"""
Contrarian Strategy Enhanced Test System

逆張り戦略の性能を検証するための専用テストシステム
下降トレンドと反発パターンを含むリアルなテストデータで検証
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトのルートディレクトリを sys.path に追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\contrarian_test.log")
logger.info("逆張り戦略テストシステムが開始されました")

print("=== Contrarian Strategy Enhanced Test System ===")

# 逆張り戦略の動的インポート
available_contrarian_strategies = {}

try:
    from strategies.contrarian_strategy import ContrarianStrategy
    available_contrarian_strategies['Contrarian'] = ContrarianStrategy
    print("[OK] ContrarianStrategy Import Completed")
except Exception as e:
    print(f"[ERROR] ContrarianStrategy Import Failed: {e}")

try:
    from strategies.support_resistance_contrarian_strategy import SupportResistanceContrarianStrategy
    available_contrarian_strategies['SRContrarian'] = SupportResistanceContrarianStrategy
    print("[OK] SupportResistanceContrarianStrategy Import Completed")
except Exception as e:
    print(f"[ERROR] SupportResistanceContrarianStrategy Import Failed: {e}")

if not available_contrarian_strategies:
    print("[ERROR] 逆張り戦略がインポートできませんでした")
    sys.exit(1)

print(f"[CHART] 利用可能な逆張り戦略数: {len(available_contrarian_strategies)}")

def create_bearish_reversal_data():
    """逆張り戦略に適した弱気相場+反発データを生成"""
    print("GENERATING - Bearish Reversal Test Data...")
    
    dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq='D')
    np.random.seed(123)  # 再現可能な結果のため
    
    n_days = len(dates)
    base_price = 120.0
    
    prices = [base_price]
    
    for i in range(1, n_days):
        current_price = prices[-1]
        
        # 段階的下降トレンド
        base_decline = -0.008  # 日次0.8%の基本下降
        
        # RSI過売り条件を作るための急落パターン
        if i % 20 in [15, 16]:  # 20日ごとに2日連続の急落
            shock_decline = -0.03  # 3%の急落
        elif i % 20 in [17, 18, 19]:  # 急落後の反発機会
            shock_decline = 0.015  # 1.5%の反発
        else:
            shock_decline = 0
            
        # ボラティリティ追加
        random_noise = np.random.normal(0, 0.012)
        
        # 支持線レベルでの反発（90, 95, 100付近）
        support_levels = [90, 95, 100, 105]
        support_bounce = 0
        
        for support in support_levels:
            if abs(current_price - support) / support < 0.02:  # 2%以内に接近
                if np.random.random() < 0.6:  # 60%の確率で反発
                    support_bounce = 0.008
                    
        total_change = base_decline + shock_decline + random_noise + support_bounce
        new_price = current_price * (1 + total_change)
        
        # 価格下限（過度な下落を防ぐ）
        new_price = max(85, new_price)
        prices.append(new_price)
    
    adj_close = np.array(prices)
    
    # OHLC データの生成
    daily_volatility = np.random.uniform(0.008, 0.025, n_days)
    
    open_prices = adj_close * (1 + np.random.normal(0, 0.003, n_days))
    high_prices = np.maximum(adj_close, open_prices) * (1 + daily_volatility * np.random.uniform(0.4, 1.2, n_days))
    low_prices = np.minimum(adj_close, open_prices) * (1 - daily_volatility * np.random.uniform(0.4, 1.2, n_days))
    
    # ボリューム（急落時は高ボリューム）
    base_volume = 1500000
    volume_multiplier = 1 + np.abs(np.diff(np.concatenate([[0], adj_close]))) / adj_close * 15
    volumes = (base_volume * volume_multiplier).astype(int)
    
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Adj Close': adj_close,
        'Close': adj_close,  # Close列も追加
        'Volume': volumes,
    })
    test_data.set_index('Date', inplace=True)
    
    return test_data

def test_contrarian_strategies():
    """逆張り戦略のテスト実行"""
    print("\n=== Contrarian Strategy Performance Test ===")
    
    # テストデータ生成
    test_data = create_bearish_reversal_data()
    
    # データ統計
    total_return = (test_data['Adj Close'].iloc[-1] / test_data['Adj Close'].iloc[0] - 1) * 100
    max_drawdown = ((test_data['Adj Close'] / test_data['Adj Close'].expanding().max()) - 1).min() * 100
    volatility = test_data['Adj Close'].pct_change().std() * np.sqrt(252) * 100
    
    print(f"TEST DATA STATS:")
    print(f"  Period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Annualized Volatility: {volatility:.1f}%")
    print(f"  Price Range: {test_data['Adj Close'].min():.2f} - {test_data['Adj Close'].max():.2f}")
    
    strategy_results = {}
    
    # 各逆張り戦略をテスト
    for strategy_name, strategy_class in available_contrarian_strategies.items():
        print(f"\n--- Testing {strategy_name} Strategy ---")
        
        try:
            if strategy_name == 'Contrarian':
                # 積極的な逆張りパラメータ
                params = {
                    'rsi_period': 10,  # 短期RSI（より敏感）
                    'rsi_oversold': 20,  # より低い閾値
                    'gap_threshold': 0.02,  # 2%以上のギャップ
                    'stop_loss': 0.025,  # 2.5%ストップロス
                    'take_profit': 0.04,  # 4%利益確定
                    'trend_filter_enabled': False,  # トレンドフィルターOFF
                    'max_hold_days': 8,  # 最大8日保有
                    'pin_bar_ratio': 1.8  # ピンバー閾値を下げる
                }
            else:  # SRContrarian
                # 支持線・抵抗線逆張りパラメータ
                params = {
                    'lookback_period': 12,  # 短期間で反応
                    'proximity_threshold': 0.015,  # 1.5%の接近で反応
                    'stop_loss_pct': 0.02,  # 2%ストップロス
                    'take_profit_pct': 0.035,  # 3.5%利益確定
                    'rsi_confirmation': True,
                    'rsi_oversold': 25,  # より低い閾値
                    'fibonacci_enabled': True,
                    'min_touches': 1,  # 最小接触回数を下げる
                    'volume_threshold': 0.8  # ボリューム閾値を下げる
                }
            
            # 戦略初期化
            strategy = strategy_class(data=test_data, params=params, price_column="Adj Close")
            print(f"[OK] {strategy_name} Strategy Initialized")
            
            # バックテスト実行
            result = strategy.backtest()
            strategy_results[strategy_name] = result
            
            # パフォーマンス分析
            entry_count = (result['Entry_Signal'] == 1).sum()
            exit_count = (result['Exit_Signal'] == 1).sum()
            
            # エントリー日の分析
            entry_dates = result[result['Entry_Signal'] == 1].index
            if len(entry_dates) > 0:
                entry_prices = result.loc[entry_dates, 'Adj Close']
                avg_entry_price = entry_prices.mean()
                print(f"  [UP] Entry Signals: {entry_count}")
                print(f"  [DOWN] Exit Signals: {exit_count}")
                print(f"  [MONEY] Avg Entry Price: ${avg_entry_price:.2f}")
                print(f"  📅 Entry Dates: {[d.strftime('%m-%d') for d in entry_dates[:5]]}")  # 最初の5件
            else:
                print(f"  [UP] Entry Signals: {entry_count}")
                print(f"  [DOWN] Exit Signals: {exit_count}")
                print("  [WARNING]  No entries generated")
                
        except Exception as e:
            print(f"[ERROR] {strategy_name} Strategy Error: {e}")
            import traceback
            print(traceback.format_exc())
    
    # 比較分析
    if len(strategy_results) > 1:
        print(f"\n=== Contrarian Strategy Comparison ===")
        
        for name, result in strategy_results.items():
            entries = (result['Entry_Signal'] == 1).sum()
            exits = (result['Exit_Signal'] == 1).sum()
            
            if entries > 0:
                # 簡易パフォーマンス計算
                entry_positions = result['Entry_Signal'] == 1
                if entry_positions.any():
                    entry_return_estimate = (result['Adj Close'].iloc[-1] / result.loc[entry_positions, 'Adj Close'].mean() - 1) * 100
                    activity_rate = (entries / len(result)) * 100
                    print(f"PERFORMANCE - {name:15} | Entries: {entries:2d} | Activity: {activity_rate:4.1f}% | Est.Return: {entry_return_estimate:5.1f}%")
            else:
                print(f"PERFORMANCE - {name:15} | Entries: {entries:2d} | No Activity")
    
    return strategy_results

def main():
    """メイン実行関数"""
    print("Starting Contrarian Strategy Enhanced Test...")
    
    try:
        results = test_contrarian_strategies()
        
        if results:
            print(f"\n[OK] Test Completed Successfully!")
            print(f"   Tested {len(results)} contrarian strategies")
            
            # 最もアクティブな戦略を特定
            max_entries = 0
            most_active = None
            
            for name, result in results.items():
                entries = (result['Entry_Signal'] == 1).sum()
                if entries > max_entries:
                    max_entries = entries
                    most_active = name
                    
            if most_active and max_entries > 0:
                print(f"   Most Active Strategy: {most_active} ({max_entries} entries)")
            else:
                print("   Note: Consider more aggressive parameters for higher activity")
                
            return True
        else:
            print("[ERROR] No strategy results generated")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test execution error: {e}")
        logger.error(f"Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n[SUCCESS] SUCCESS - Contrarian Strategy Test Completed!")
        logger.info("Contrarian Strategy Test Success")
    else:
        print("\n💥 FAILED - Contrarian Strategy Test Failed")
        logger.error("Contrarian Strategy Test Failed")
