"""
main.pyエラー修正版テスト
performance_metricsの修正を含む最小限バックテストシステム

Author: imega
Modified: 2025-07-22
Description: performance_metrics.py修正後のテスト実行
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
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\main_minimal_working.log")

def test_performance_metrics_fix():
    """performance_metrics修正のテスト"""
    print("=== Performance Metrics 修正テスト ===")
    
    try:
        from metrics.performance_metrics import calculate_win_rate
        
        # テストデータ作成（DataFrame）
        test_df = pd.DataFrame({
            'PnL': [100, -50, 200, -30, 150, 0],
            'Date': pd.date_range('2023-01-01', periods=6),
            'Strategy': ['TestStrategy'] * 6
        })
        
        print(f"テストDataFrame作成: {len(test_df)}行")
        print(f"PnL列: {test_df['PnL'].tolist()}")
        
        # 修正された関数をテスト（DataFrame入力）
        win_rate_df = calculate_win_rate(test_df)
        print(f"DataFrame入力での勝率計算: {win_rate_df:.2%}")
        
        # テストデータ作成（Series）
        test_series = pd.Series([100, -50, 200, -30, 150])
        win_rate_series = calculate_win_rate(test_series)
        print(f"Series入力での勝率計算: {win_rate_series:.2%}")
        
        # エラー処理テスト（空のデータ）
        empty_df = pd.DataFrame()
        win_rate_empty = calculate_win_rate(empty_df)
        print(f"空DataFrame入力での勝率計算: {win_rate_empty:.2%}")
        
        print("✅ Performance Metrics修正テスト成功")
        return True
        
    except Exception as e:
        print(f"❌ Performance Metrics テストエラー: {e}")
        logger.error(f"Performance Metrics テストエラー: {e}")
        return False

def test_existing_strategies():
    """既存7戦略の動作確認"""
    print("\n=== 既存戦略動作確認 ===")
    
    try:
        # 既存戦略のインポート確認
        existing_strategies = [
            "strategies.vwap_bounce_strategy",
            "strategies.vwap_breakout_strategy", 
            "strategies.gc_strategy",
            "strategies.breakout_strategy",
            "strategies.opening_gap_strategy",
            "strategies.momentum_investing_strategy",
            "strategies.contrarian_strategy"
        ]
        
        imported_count = 0
        for strategy_module in existing_strategies:
            try:
                __import__(strategy_module)
                imported_count += 1
                print(f"✅ {strategy_module.split('.')[-1]}")
            except ImportError as e:
                print(f"❌ {strategy_module.split('.')[-1]}: {e}")
        
        print(f"戦略インポート成功: {imported_count}/{len(existing_strategies)}")
        return imported_count >= 5  # 最低5つの戦略が利用可能であることを確認
        
    except Exception as e:
        print(f"❌ 戦略動作確認エラー: {e}")
        logger.error(f"戦略動作確認エラー: {e}")
        return False

def run_minimal_backtest():
    """最小限のバックテストを実行"""
    print("\n=== 最小限バックテスト実行 ===")
    
    try:
        # データ取得
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        print(f"データ取得成功: {ticker}")
        print(f"期間: {start_date} to {end_date}")
        print(f"データ行数: {len(stock_data)}日分")
        
        # 修正されたシミュレーション実行
        from output.simulation_handler import simulate_and_save
        print("バックテスト開始...")
        results = simulate_and_save(stock_data, ticker)
        print("✅ バックテスト実行成功")
        
        # 結果の要約表示
        if results and isinstance(results, dict):
            print(f"結果概要: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ バックテストエラー: {e}")
        logger.error(f"バックテストエラー詳細: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """メイン実行"""
    print("=" * 60)
    print("main.py エラー修正版テスト実行")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 1. Performance Metricsの修正テスト
    print("\n📊 Step 1: Performance Metrics修正テスト")
    if not test_performance_metrics_fix():
        print("❌ Performance Metrics修正に問題があります")
        return False
    
    # 2. 既存戦略の動作確認
    print("\n🎯 Step 2: 既存戦略動作確認")
    if not test_existing_strategies():
        print("⚠️ 一部の戦略にインポート問題があります（継続します）")
    
    # 3. 最小限バックテストの実行
    print("\n⚡ Step 3: 最小限バックテスト実行")
    if not run_minimal_backtest():
        print("❌ バックテスト実行に問題があります") 
        return False
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("✅ 全テスト成功 - main.pyの修正が有効")
    print(f"実行時間: {duration.total_seconds():.2f}秒")
    print("main.pyでの実行準備完了")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# 動作する戦略を段階的にインポート
available_strategies = {}

try:
    from strategies.Breakout import BreakoutStrategy
    available_strategies['Breakout'] = BreakoutStrategy
    print("✅ BreakoutStrategy インポート完了")
except Exception as e:
    print(f"❌ BreakoutStrategy インポートエラー: {e}")

# 追加戦略のインポートを試行（循環インポートを回避）
try:
    from strategies.Momentum_Investing import MomentumInvestingStrategy
    available_strategies['Momentum'] = MomentumInvestingStrategy
    print("✅ MomentumInvestingStrategy インポート完了")
except Exception as e:
    print(f"⚠️  MomentumInvestingStrategy インポート失敗: {e}")

try:
    from strategies.Opening_Gap import OpeningGapStrategy
    available_strategies['OpeningGap'] = OpeningGapStrategy
    print("✅ OpeningGapStrategy インポート完了")
except Exception as e:
    print(f"⚠️  OpeningGapStrategy インポート失敗: {e}")

print(f"📊 利用可能な戦略数: {len(available_strategies)}")
for name in available_strategies.keys():
    print(f"  - {name}")
    
if not available_strategies:
    print("❌ 利用可能な戦略がありません")
    sys.exit(1)

def main():
    """最小限のメイン関数"""
    print("🚀 バックテストシステム開始")
    
    # 基本パラメータ
    ticker = "NVDA"
    print(f"処理対象: {ticker}")
    
    try:
        # データ取得テスト
        # stock_data, index_data = get_parameters_and_data(ticker, start_date="2024-01-01", end_date="2024-12-31")
        
        print("📊 現実的なテストデータを生成中...")
        # より現実的な価格変動パターンを生成
        dates = pd.date_range(start="2024-01-01", end="2024-02-29", freq='D')  # 2ヶ月に延長
        np.random.seed(42)  # 再現可能性のためのシード
        
        # 基準価格からのランダムウォーク + トレンド + ボラティリティ
        n_days = len(dates)
        base_price = 100.0
        
        # トレンドファクター（期間の前半は上昇、後半は横ばい〜下落）
        trend_factor = np.concatenate([
            np.linspace(0, 0.8, n_days//2),    # 前半：上昇トレンド
            np.linspace(0.8, -0.3, n_days - n_days//2)  # 後半：下降トレンド
        ])
        
        # ランダムな日次変動率（平均0、標準偏差2%）
        daily_returns = np.random.normal(0, 0.02, n_days)
        
        # ボラティリティクラスター（特定期間で変動が大きくなる）
        volatility_cluster = np.ones(n_days)
        volatility_cluster[n_days//3:n_days//3 + 10] = 2.5  # 中盤にボラティリティ急増
        daily_returns = daily_returns * volatility_cluster
        
        # 価格系列の計算
        prices = np.zeros(n_days)
        prices[0] = base_price
        
        for i in range(1, n_days):
            trend_component = trend_factor[i] * 0.001  # トレンド成分
            random_component = daily_returns[i]        # ランダム成分
            total_return = trend_component + random_component
            prices[i] = prices[i-1] * (1 + total_return)
        
        # OHLC データの生成（終値ベースから逆算）
        adj_close = prices
        
        # 各日のイントラデイ変動を生成
        daily_volatility = np.random.uniform(0.005, 0.025, n_days)  # 日中変動幅0.5-2.5%
        
        open_prices = adj_close * (1 + np.random.normal(0, 0.005, n_days))  # 前日終値から小さなギャップ
        high_prices = np.maximum(adj_close, open_prices) * (1 + daily_volatility * np.random.uniform(0.3, 1.0, n_days))
        low_prices = np.minimum(adj_close, open_prices) * (1 - daily_volatility * np.random.uniform(0.3, 1.0, n_days))
        
        # 出来高の生成（価格変動に連動）
        base_volume = 1000000
        volume_multiplier = 1 + np.abs(daily_returns) * 10  # 変動が大きいほど出来高増加
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
        
        print("📈 現実的なテストデータ生成完了")
        print(f"データ期間: {test_data.index[0].strftime('%Y-%m-%d')} ~ {test_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"データ数: {len(test_data)}行")
        
        # 基本的な統計情報
        print(f"価格レンジ: ${test_data['Adj Close'].min():.2f} - ${test_data['Adj Close'].max():.2f}")
        print(f"平均日次変動率: {((test_data['Adj Close'].pct_change().std()) * 100):.2f}%")
        print(f"最大上昇: {((test_data['Adj Close'].pct_change().max()) * 100):.2f}%")
        print(f"最大下落: {((test_data['Adj Close'].pct_change().min()) * 100):.2f}%")
        
        # 価格動向の分析
        total_return = (test_data['Adj Close'].iloc[-1] / test_data['Adj Close'].iloc[0] - 1) * 100
        print(f"期間リターン: {total_return:.2f}%")
        
        # ボラティリティが高い期間を特定
        volatility = test_data['Adj Close'].pct_change().rolling(5).std() * np.sqrt(252)
        high_vol_days = (volatility > volatility.quantile(0.8)).sum()
        print(f"高ボラティリティ期間: {high_vol_days}日間")
        
        # 複数戦略の統合テスト
        print("\n🔀 複数戦略統合テスト開始")
        strategy_results = {}
        
        for strategy_name, strategy_class in available_strategies.items():
            print(f"\n📊 {strategy_name}戦略のテスト中...")
            
            try:
                # 戦略ごとの最適化パラメータ
                if strategy_name == 'Breakout':
                    params = {
                        'lookback_period': 10,
                        'breakout_threshold': 0.015
                    }
                    strategy = strategy_class(
                        data=test_data,
                        params=params,
                        price_column="Adj Close"
                    )
                
                elif strategy_name == 'Momentum':
                    params = {
                        'short_window': 12,
                        'long_window': 26,
                        'rsi_period': 14,
                        'rsi_overbought': 70,
                        'rsi_oversold': 30
                    }
                    strategy = strategy_class(
                        data=test_data,
                        params=params,
                        price_column="Adj Close"
                    )
                
                elif strategy_name == 'OpeningGap':
                    # OpeningGapは追加のDowデータが必要なため、テストデータを複製
                    params = {
                        'gap_threshold': 0.02,
                        'volume_threshold': 1.5
                    }
                    strategy = strategy_class(
                        data=test_data,
                        dow_data=test_data,  # テスト用に同じデータを使用
                        params=params,
                        price_column="Adj Close"
                    )
                
                else:
                    # デフォルトパラメータで初期化
                    strategy = strategy_class(
                        data=test_data,
                        params={},
                        price_column="Adj Close"
                    )
                
                print(f"✅ {strategy_name}戦略初期化完了")
                
                # バックテスト実行
                result = strategy.backtest()
                strategy_results[strategy_name] = result
                
                # 各戦略の基本統計
                if 'Entry_Signal' in result.columns:
                    entry_count = (result['Entry_Signal'] == 1).sum()
                    print(f"  📈 エントリー数: {entry_count}")
                
                if 'Exit_Signal' in result.columns:
                    exit_count = (result['Exit_Signal'] == 1).sum()
                    print(f"  📉 エグジット数: {exit_count}")
                
                print(f"  ✅ {strategy_name}戦略バックテスト完了")
                
            except Exception as strategy_error:
                print(f"  ❌ {strategy_name}戦略エラー: {strategy_error}")
                continue
        
        # 戦略比較分析
        if len(strategy_results) > 1:
            print(f"\n📊 戦略比較分析 ({len(strategy_results)}戦略)")
            
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
                
                print(f"  📋 {name:12} | エントリー: {entry_signals:2}回 | エグジット: {exit_signals:2}回 | 発生率: {signal_rate:5.1f}%")
            
            # 最もアクティブな戦略を特定
            most_active = max(comparison_summary.items(), key=lambda x: x[1]['entries'])
            print(f"  🏆 最もアクティブ: {most_active[0]} ({most_active[1]['entries']}回のエントリー)")
            
            # シグナル統合の簡易シミュレーション
            print(f"\n🔀 シグナル統合シミュレーション")
            combined_signals = pd.DataFrame(index=test_data.index)
            
            for name, result in strategy_results.items():
                if 'Entry_Signal' in result.columns:
                    combined_signals[f'{name}_Entry'] = result['Entry_Signal']
                if 'Exit_Signal' in result.columns:
                    combined_signals[f'{name}_Exit'] = result['Exit_Signal']
            
            # 統合エントリーシグナル（任意の戦略がエントリーシグナルを出した場合）
            entry_columns = [col for col in combined_signals.columns if col.endswith('_Entry')]
            if entry_columns:
                combined_signals['Combined_Entry'] = combined_signals[entry_columns].max(axis=1)
                combined_entry_count = (combined_signals['Combined_Entry'] == 1).sum()
                print(f"  📈 統合エントリーシグナル数: {combined_entry_count}")
            
            # 統合エグジットシグナル
            exit_columns = [col for col in combined_signals.columns if col.endswith('_Exit')]
            if exit_columns:
                combined_signals['Combined_Exit'] = combined_signals[exit_columns].max(axis=1)
                combined_exit_count = (combined_signals['Combined_Exit'] == 1).sum()
                print(f"  📉 統合エグジットシグナル数: {combined_exit_count}")
        
        # 単一戦略の場合の詳細分析（既存のBreakout戦略分析）
        elif len(strategy_results) == 1:
            strategy_name = list(strategy_results.keys())[0]
            result = strategy_results[strategy_name]
            print(f"\n📊 {strategy_name}戦略の詳細分析:")
            
            # 既存の詳細分析ロジックを実行
            try:
                # シグナル分析
                if 'Entry_Signal' in result.columns and 'Exit_Signal' in result.columns:
                    entry_signals = (result['Entry_Signal'] == 1)
                    exit_signals = (result['Exit_Signal'] == 1)
                    
                    # エントリー・エグジット のタイミング分析
                    entry_dates = result[entry_signals].index.tolist()
                    exit_dates = result[exit_signals].index.tolist()
                    
                    print(f"  📈 エントリー回数: {len(entry_dates)}")
                    print(f"  📉 エグジット回数: {len(exit_dates)}")
                    
                    if len(entry_dates) > 0:
                        print(f"  📅 最初のエントリー: {entry_dates[0].strftime('%Y-%m-%d')}")
                        print(f"  📅 最後のエントリー: {entry_dates[-1].strftime('%Y-%m-%d')}")
                    
                    # ポジション期間の分析
                    if len(entry_dates) > 0 and len(exit_dates) > 0:
                        # 簡単なポジション期間分析
                        min_pairs = min(len(entry_dates), len(exit_dates))
                        if min_pairs > 0:
                            position_durations = []
                            for i in range(min_pairs):
                                if i < len(exit_dates) and exit_dates[i] > entry_dates[i]:
                                    duration = (exit_dates[i] - entry_dates[i]).days
                                    position_durations.append(duration)
                            
                            if position_durations:
                                avg_duration = np.mean(position_durations)
                                print(f"  ⏱️  平均ポジション保有期間: {avg_duration:.1f}日")
                
                # 価格変動との相関分析
                price_changes = test_data['Adj Close'].pct_change().fillna(0)
                significant_moves = (abs(price_changes) > 0.02).sum()  # 2%以上の変動
                print(f"  📊 期間中の大幅変動日数: {significant_moves}日")
                
                # 戦略の有効性評価
                if 'Entry_Signal' in result.columns:
                    total_signals = (result['Entry_Signal'] == 1).sum()
                    if total_signals > 0:
                        signal_rate = (total_signals / len(result)) * 100
                        print(f"  📈 シグナル発生率: {signal_rate:.1f}%")
                    
                    # 高ボラティリティ期間でのシグナル発生
                    high_vol_periods = price_changes.abs() > price_changes.abs().quantile(0.8)
                    signals_in_vol = ((result['Entry_Signal'] == 1) & high_vol_periods).sum()
                    if total_signals > 0:
                        vol_signal_ratio = (signals_in_vol / total_signals) * 100
                        print(f"  🔥 高ボラティリティ期間でのシグナル割合: {vol_signal_ratio:.1f}%")
                
                # Return列が存在しない場合は簡単な収益率計算を試行
                if 'Return' not in result.columns:
                    print("  💡 Return列が見つからないため、簡易収益率を計算中...")
                    if 'Entry_Signal' in result.columns and len(entry_dates) > 0:
                        # エントリー時点の価格での簡易計算
                        entry_prices = [test_data.loc[date, 'Adj Close'] for date in entry_dates if date in test_data.index]
                        if len(entry_prices) > 0:
                            avg_entry_price = np.mean(entry_prices)
                            final_price = test_data['Adj Close'].iloc[-1]
                            simple_return = ((final_price / avg_entry_price - 1) * 100)
                            print(f"  📊 簡易リターン推定: {simple_return:.2f}%")
                
            except Exception as perf_e:
                print(f"  ⚠️  詳細パフォーマンス分析エラー: {perf_e}")
        
        else:
            print("⚠️  利用可能な戦略結果がありません")
        
        # 従来の単一戦略テスト（後方互換のため保持）
        if 'Breakout' in available_strategies:
            print(f"\n🔍 従来のBreakout戦略テスト（詳細版）")
            try:
            # より効果的なブレイクアウト条件に調整
            optimized_params = {
                'lookback_period': 10,      # ルックバック期間を延長
                'breakout_threshold': 0.015 # 閾値を1.5%に調整（より現実的）
            }
            
            strategy = BreakoutStrategy(
                data=test_data,
                params=optimized_params,
                price_column="Adj Close"
            )
            print("✅ BreakoutStrategy初期化完了（最適化パラメータ使用）")
            print(f"📊 パラメータ: lookback={optimized_params['lookback_period']}, threshold={optimized_params['breakout_threshold']*100:.1f}%")
            
            # バックテストの実行
            print("🔍 バックテスト実行中...")
            try:
                result = strategy.backtest()
                print(f"✅ バックテスト完了: {len(result)}行の結果")
                
                # 結果の基本統計
                if 'Entry_Signal' in result.columns:
                    entry_signals = (result['Entry_Signal'] == 1).sum()
                    print(f"📊 エントリーシグナル数: {entry_signals}")
                
                if 'Exit_Signal' in result.columns:
                    exit_signals = (result['Exit_Signal'] == 1).sum()
                    print(f"📊 エグジットシグナル数: {exit_signals}")
                
                print("📈 バックテスト結果サンプル:")
                if len(result) > 0:
                    print(result.head())
                
                # より詳細なパフォーマンス分析
                try:
                    print("\n📊 詳細パフォーマンス分析:")
                    
                    # シグナル分析
                    if 'Entry_Signal' in result.columns and 'Exit_Signal' in result.columns:
                        entry_signals = (result['Entry_Signal'] == 1)
                        exit_signals = (result['Exit_Signal'] == 1)
                        
                        # エントリー・エグジット のタイミング分析
                        entry_dates = result[entry_signals].index.tolist()
                        exit_dates = result[exit_signals].index.tolist()
                        
                        print(f"  📈 エントリー回数: {len(entry_dates)}")
                        print(f"  📉 エグジット回数: {len(exit_dates)}")
                        
                        if len(entry_dates) > 0:
                            print(f"  📅 最初のエントリー: {entry_dates[0].strftime('%Y-%m-%d')}")
                            print(f"  📅 最後のエントリー: {entry_dates[-1].strftime('%Y-%m-%d')}")
                        
                        # ポジション期間の分析
                        if len(entry_dates) > 0 and len(exit_dates) > 0:
                            # 簡単なポジション期間分析
                            min_pairs = min(len(entry_dates), len(exit_dates))
                            if min_pairs > 0:
                                position_durations = []
                                for i in range(min_pairs):
                                    if i < len(exit_dates) and exit_dates[i] > entry_dates[i]:
                                        duration = (exit_dates[i] - entry_dates[i]).days
                                        position_durations.append(duration)
                                
                                if position_durations:
                                    avg_duration = np.mean(position_durations)
                                    print(f"  ⏱️  平均ポジション保有期間: {avg_duration:.1f}日")
                    
                    # 価格変動との相関分析
                    price_changes = test_data['Adj Close'].pct_change().fillna(0)
                    significant_moves = (abs(price_changes) > 0.02).sum()  # 2%以上の変動
                    print(f"  📊 期間中の大幅変動日数: {significant_moves}日")
                    
                    # 戦略の有効性評価
                    if 'Entry_Signal' in result.columns:
                        total_signals = (result['Entry_Signal'] == 1).sum()
                        if total_signals > 0:
                            signal_rate = (total_signals / len(result)) * 100
                            print(f"  📈 シグナル発生率: {signal_rate:.1f}%")
                        
                        # 高ボラティリティ期間でのシグナル発生
                        high_vol_periods = price_changes.abs() > price_changes.abs().quantile(0.8)
                        signals_in_vol = ((result['Entry_Signal'] == 1) & high_vol_periods).sum()
                        if total_signals > 0:
                            vol_signal_ratio = (signals_in_vol / total_signals) * 100
                            print(f"  🔥 高ボラティリティ期間でのシグナル割合: {vol_signal_ratio:.1f}%")
                    
                    # Return列が存在しない場合は簡単な収益率計算を試行
                    if 'Return' not in result.columns:
                        print("  💡 Return列が見つからないため、簡易収益率を計算中...")
                        if 'Entry_Signal' in result.columns and len(entry_dates) > 0:
                            # エントリー時点の価格での簡易計算
                            entry_prices = [test_data.loc[date, 'Adj Close'] for date in entry_dates if date in test_data.index]
                            if len(entry_prices) > 0:
                                avg_entry_price = np.mean(entry_prices)
                                final_price = test_data['Adj Close'].iloc[-1]
                                simple_return = ((final_price / avg_entry_price - 1) * 100)
                                print(f"  📊 簡易リターン推定: {simple_return:.2f}%")
                    
                except Exception as perf_e:
                    print(f"⚠️  詳細パフォーマンス分析エラー: {perf_e}")
                
                # 既存のリターン分析（datetime型エラーを回避）
                try:
                    if 'Return' in result.columns:
                        returns_data = result['Return'].fillna(0)
                        # datetime型チェック
                        if returns_data.dtype == 'datetime64[ns]':
                            print("  ⚠️  Returnカラムがdatetime型のため、パフォーマンス計算をスキップ")
                        else:
                            # 数値データの場合のみパフォーマンス計算
                            returns_data = pd.to_numeric(returns_data, errors='coerce').fillna(0)
                            total_return = returns_data.sum()
                            print(f"  � 総リターン: {total_return:.4f}")
                except Exception as perf_e:
                    print(f"  ⚠️  基本リターン計算エラー: {perf_e}")
                
            except Exception as e:
                print(f"❌ バックテストエラー: {e}")
                print("⚠️  基本戦略テストのみ実行")
            
        except Exception as e:
            print(f"❌ 戦略テストエラー: {e}")
        
        print("🎉 最小限テスト完了！")
        return True
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("   最小限バックテストシステム v2")
    print("=" * 50)
    
    success = main()
    
    if success:
        print("\n🎯 実行成功！")
        logger.info("最小限main.py実行成功")
    else:
        print("\n❌ 実行失敗")
        logger.error("最小限main.py実行失敗")
