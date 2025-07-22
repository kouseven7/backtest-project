"""
複数戦略統合バックテストシステム v2 - 簡潔版
Unicode文字を使わず安定動作を重視
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトのルートディレクトリを sys.path に追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\multi_strategy_backtest.log")
logger.info("Multiple Strategy Backtest System Started")

print("OK - Multi-Strategy System Basic Import Completed")

try:
    from data_fetcher import get_parameters_and_data
    print("OK - data_fetcher Import Completed")
except Exception as e:
    print(f"ERROR - data_fetcher Import Error: {e}")
    sys.exit(1)

# 複数戦略を段階的にインポート
available_strategies = {}

try:
    from strategies.Breakout import BreakoutStrategy
    available_strategies['Breakout'] = BreakoutStrategy
    print("OK - BreakoutStrategy Import Completed")
except Exception as e:
    print(f"ERROR - BreakoutStrategy Import Error: {e}")

# 追加戦略のインポートを試行
try:
    from strategies.Momentum_Investing import MomentumInvestingStrategy
    available_strategies['Momentum'] = MomentumInvestingStrategy
    print("OK - MomentumInvestingStrategy Import Completed")
except Exception as e:
    print(f"WARN - MomentumInvestingStrategy Import Failed: {e}")

try:
    from strategies.Opening_Gap import OpeningGapStrategy
    available_strategies['OpeningGap'] = OpeningGapStrategy
    print("OK - OpeningGapStrategy Import Completed")
except Exception as e:
    print(f"WARN - OpeningGapStrategy Import Failed: {e}")

# 逆張り戦略の追加
try:
    from strategies.contrarian_strategy import ContrarianStrategy
    available_strategies['Contrarian'] = ContrarianStrategy
    print("OK - ContrarianStrategy Import Completed")
except Exception as e:
    print(f"WARN - ContrarianStrategy Import Failed: {e}")

# 新しい支持線・抵抗線逆張り戦略を追加
try:
    from strategies.support_resistance_contrarian_strategy import SupportResistanceContrarianStrategy
    available_strategies['SRContrarian'] = SupportResistanceContrarianStrategy
    print("OK - SupportResistanceContrarianStrategy Import Completed")
except Exception as e:
    print(f"WARN - SupportResistanceContrarianStrategy Import Failed: {e}")

# 平均回帰戦略を追加
try:
    from strategies.mean_reversion_strategy import MeanReversionStrategy
    available_strategies['MeanReversion'] = MeanReversionStrategy
    print("OK - MeanReversionStrategy Import Completed")
except Exception as e:
    print(f"WARN - MeanReversionStrategy Import Failed: {e}")

# ペアトレーディング戦略を追加
try:
    from strategies.pairs_trading_strategy import PairsTradingStrategy
    available_strategies['PairsTrading'] = PairsTradingStrategy
    print("OK - PairsTradingStrategy Import Completed")
except Exception as e:
    print(f"WARN - PairsTradingStrategy Import Failed: {e}")

print(f"INFO - Available Strategies: {len(available_strategies)}")
for name in available_strategies.keys():
    print(f"  - {name}")
    
if not available_strategies:
    print("ERROR - No strategies available")
    sys.exit(1)

def main():
    """複数戦略統合メイン関数"""
    print("===== Multiple Strategy Integrated Backtest System =====")
    print("STARTED - Multi-Strategy Backtest System")
    
    # 基本パラメータ
    ticker = "NVDA"
    print(f"Processing Target: {ticker}")
    
    try:
        # 現実的なテストデータの生成
        print("GENERATING - Realistic Test Data...")
        
        dates = pd.date_range(start="2024-01-01", end="2024-02-29", freq='D')
        np.random.seed(42)
        
        n_days = len(dates)
        base_price = 100.0
        
        # トレンドファクター - 逆張り戦略に適したパターン
        # 段階的な下降トレンド with 反発機会
        trend_factor = []
        for i in range(n_days):
            # 基本下降トレンド
            base_decline = -0.4
            
            # 周期的反発（8日周期）
            cycle = (i / 8) * 2 * np.pi
            cycle_component = np.sin(cycle) * 1.2
            
            # 急落と反発パターン（逆張りチャンス生成）
            if i % 12 == 8:  # 12日ごとに急落
                shock = -2.5
            elif i % 12 in [9, 10]:  # 急落後の反発
                shock = 1.8
            else:
                shock = 0
            
            trend_factor.append(base_decline + cycle_component + shock)
        
        trend_factor = np.array(trend_factor)

        daily_returns = np.random.normal(0, 0.025, n_days)  # 少し高いボラティリティ
        
        # ボラティリティクラスタリング（逆張りチャンス強化）
        volatility_cluster = np.ones(n_days)
        for i in range(1, n_days):
            if abs(trend_factor[i]) > 2:  # 大きな動き後は高ボラティリティ
                volatility_cluster[i] = min(3.0, volatility_cluster[i-1] * 1.6)
            else:
                volatility_cluster[i] = max(0.6, volatility_cluster[i-1] * 0.9)
        daily_returns = daily_returns * volatility_cluster
        
        prices = np.zeros(n_days)
        prices[0] = base_price
        
        for i in range(1, n_days):
            trend_component = trend_factor[i] * 0.001
            random_component = daily_returns[i]
            total_return = trend_component + random_component
            prices[i] = prices[i-1] * (1 + total_return)
        
        adj_close = prices
        daily_volatility = np.random.uniform(0.005, 0.025, n_days)
        
        open_prices = adj_close * (1 + np.random.normal(0, 0.005, n_days))
        high_prices = np.maximum(adj_close, open_prices) * (1 + daily_volatility * np.random.uniform(0.3, 1.0, n_days))
        low_prices = np.minimum(adj_close, open_prices) * (1 - daily_volatility * np.random.uniform(0.3, 1.0, n_days))
        
        base_volume = 1000000
        volume_multiplier = 1 + np.abs(daily_returns) * 10
        volumes = (base_volume * volume_multiplier).astype(int)
        
        test_data = pd.DataFrame({
            'Date': dates,
            'Open': open_prices,
            'High': high_prices, 
            'Low': low_prices,
            'Adj Close': adj_close,
            'Volume': volumes,
        })
        test_data.set_index('Date', inplace=True)
        
        print("COMPLETED - Test Data Generation")
        print(f"Period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Data Points: {len(test_data)}")
        print(f"Price Range: ${test_data['Adj Close'].min():.2f} - ${test_data['Adj Close'].max():.2f}")
        
        total_return = (test_data['Adj Close'].iloc[-1] / test_data['Adj Close'].iloc[0] - 1) * 100
        print(f"Period Return: {total_return:.2f}%")
        
        # 複数戦略のバックテスト実行
        print("\nSTARTED - Multi-Strategy Backtest Execution")
        strategy_results = {}
        
        for strategy_name, strategy_class in available_strategies.items():
            print(f"\nTEST - {strategy_name} Strategy...")
            
            try:
                # 戦略ごとの最適化パラメータ
                if strategy_name == 'Breakout':
                    params = {'lookback_period': 10, 'breakout_threshold': 0.015}
                    strategy = strategy_class(data=test_data, params=params, price_column="Adj Close")
                
                elif strategy_name == 'Momentum':
                    params = {'short_window': 12, 'long_window': 26, 'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30}
                    strategy = strategy_class(data=test_data, params=params, price_column="Adj Close")
                
                elif strategy_name == 'OpeningGap':
                    params = {'gap_threshold': 0.02, 'volume_threshold': 1.5}
                    strategy = strategy_class(data=test_data, dow_data=test_data, params=params, price_column="Adj Close")
                
                elif strategy_name == 'Contrarian':
                    # 逆張り戦略パラメータ（より積極的な設定）
                    params = {
                        'rsi_period': 14, 
                        'rsi_oversold': 25,  # より低い閾値で早期エントリー
                        'gap_threshold': 0.03, 
                        'stop_loss': 0.03, 
                        'take_profit': 0.04,
                        'trend_filter_enabled': False  # レンジフィルターを無効化
                    }
                    strategy = strategy_class(data=test_data, params=params, price_column="Adj Close")
                
                elif strategy_name == 'SRContrarian':
                    # 支持線・抵抗線逆張り戦略パラメータ
                    params = {
                        'lookback_period': 15,
                        'proximity_threshold': 0.008,  # 0.8%の接近で反応
                        'stop_loss_pct': 0.02,
                        'take_profit_pct': 0.035,
                        'rsi_confirmation': True,
                        'rsi_oversold': 30,
                        'fibonacci_enabled': True
                    }
                    strategy = strategy_class(data=test_data, params=params, price_column="Adj Close")
                
                elif strategy_name == 'MeanReversion':
                    # 平均回帰戦略パラメータ
                    params = {
                        'sma_period': 18,
                        'zscore_entry_threshold': -1.6,  # Z-score閾値
                        'zscore_exit_threshold': -0.2,
                        'stop_loss_pct': 0.025,
                        'take_profit_pct': 0.04,
                        'volume_confirmation': True,
                        'volume_threshold': 0.8,
                        'rsi_filter': True,
                        'rsi_oversold': 28,
                        'max_hold_days': 12
                    }
                    strategy = strategy_class(data=test_data, params=params, price_column="Adj Close")
                
                elif strategy_name == 'PairsTrading':
                    # ペアトレーディング戦略パラメータ
                    params = {
                        'short_ma_period': 5,
                        'long_ma_period': 18,
                        'spread_period': 12,
                        'entry_threshold': 1.8,      # スプレッド乖離閾値
                        'exit_threshold': 0.4,       # 回帰閾値
                        'stop_loss_pct': 0.03,
                        'take_profit_pct': 0.05,
                        'volume_filter': True,
                        'volume_threshold': 1.1,
                        'max_hold_days': 15,
                        'correlation_min': 0.6       # 相関閾値を低めに
                    }
                    strategy = strategy_class(data=test_data, params=params, price_column="Adj Close")
                
                else:
                    strategy = strategy_class(data=test_data, params={}, price_column="Adj Close")
                
                print(f"OK - {strategy_name} Strategy Initialized")
                
                # バックテスト実行
                result = strategy.backtest()
                strategy_results[strategy_name] = result
                
                # 基本統計
                entry_count = (result['Entry_Signal'] == 1).sum() if 'Entry_Signal' in result.columns else 0
                exit_count = (result['Exit_Signal'] == 1).sum() if 'Exit_Signal' in result.columns else 0
                
                print(f"  Entries: {entry_count}")
                print(f"  Exits: {exit_count}")
                print(f"OK - {strategy_name} Strategy Backtest Completed")
                
            except Exception as strategy_error:
                print(f"ERROR - {strategy_name} Strategy Error: {strategy_error}")
                continue
        
        # 戦略比較分析
        if len(strategy_results) > 1:
            print(f"\n=== Strategy Comparison Analysis ({len(strategy_results)} strategies) ===")
            
            comparison_summary = {}
            for name, result in strategy_results.items():
                entry_signals = (result['Entry_Signal'] == 1).sum() if 'Entry_Signal' in result.columns else 0
                exit_signals = (result['Exit_Signal'] == 1).sum() if 'Exit_Signal' in result.columns else 0
                signal_rate = (entry_signals / len(result)) * 100 if len(result) > 0 else 0
                
                comparison_summary[name] = {
                    'entries': entry_signals,
                    'exits': exit_signals,
                    'signal_rate': signal_rate
                }
                
                print(f"RESULT - {name:12} | Entries: {entry_signals:2} | Exits: {exit_signals:2} | Rate: {signal_rate:5.1f}%")
            
            # 最もアクティブな戦略を特定
            if comparison_summary:
                most_active = max(comparison_summary.items(), key=lambda x: x[1]['entries'])
                print(f"MOST_ACTIVE - {most_active[0]} ({most_active[1]['entries']} entries)")
            
            # シグナル統合シミュレーション
            print(f"\n=== Signal Integration Simulation ===")
            combined_signals = pd.DataFrame(index=test_data.index)
            
            for name, result in strategy_results.items():
                if 'Entry_Signal' in result.columns:
                    combined_signals[f'{name}_Entry'] = result['Entry_Signal']
                if 'Exit_Signal' in result.columns:
                    combined_signals[f'{name}_Exit'] = result['Exit_Signal']
            
            # 統合エントリーシグナル
            entry_columns = [col for col in combined_signals.columns if col.endswith('_Entry')]
            if entry_columns:
                combined_signals['Combined_Entry'] = combined_signals[entry_columns].max(axis=1)
                combined_entry_count = (combined_signals['Combined_Entry'] == 1).sum()
                print(f"COMBINED - Entry Signals: {combined_entry_count}")
            
            # 統合エグジットシグナル
            exit_columns = [col for col in combined_signals.columns if col.endswith('_Exit')]
            if exit_columns:
                combined_signals['Combined_Exit'] = combined_signals[exit_columns].max(axis=1)
                combined_exit_count = (combined_signals['Combined_Exit'] == 1).sum()
                print(f"COMBINED - Exit Signals: {combined_exit_count}")
        
        elif len(strategy_results) == 1:
            # 単一戦略の詳細分析
            strategy_name = list(strategy_results.keys())[0]
            result = strategy_results[strategy_name]
            print(f"\n=== {strategy_name} Strategy Detailed Analysis ===")
            
            if 'Entry_Signal' in result.columns:
                entry_signals = (result['Entry_Signal'] == 1)
                entry_dates = result[entry_signals].index.tolist()
                
                print(f"DETAIL - Entry Count: {len(entry_dates)}")
                if len(entry_dates) > 0:
                    print(f"DETAIL - First Entry: {entry_dates[0].strftime('%Y-%m-%d')}")
                    print(f"DETAIL - Last Entry: {entry_dates[-1].strftime('%Y-%m-%d')}")
                    
                    # 簡易リターン計算
                    entry_prices = [test_data.loc[date, 'Adj Close'] for date in entry_dates if date in test_data.index]
                    if len(entry_prices) > 0:
                        avg_entry_price = np.mean(entry_prices)
                        final_price = test_data['Adj Close'].iloc[-1]
                        simple_return = ((final_price / avg_entry_price - 1) * 100)
                        print(f"DETAIL - Estimated Return: {simple_return:.2f}%")
        
        else:
            print("WARNING - No strategy results available")
        
        print("COMPLETED - Multi-Strategy Integrated Backtest!")
        return True
        
    except Exception as e:
        print(f"ERROR - System Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nSUCCESS - Execution Completed!")
        logger.info("Multi-Strategy Backtest Execution Success")
    else:
        print("\nFAILED - Execution Failed")
        logger.error("Multi-Strategy Backtest Execution Failed")
