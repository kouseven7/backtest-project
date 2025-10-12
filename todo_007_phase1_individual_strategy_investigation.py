#!/usr/bin/env python3
"""
TODO-007 Phase 1: 戦略統合処理でのシグナル変換調査

緊急優先調査: TODO-003修正効果なしの根本原因解明
各戦略が実際にExit_Signal=-1を生成しているか徹底確認
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# 戦略クラスのインポート
sys.path.append('.')
sys.path.append('strategies')

from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy
from strategies.VWAP_Bounce import VWAPBounceStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy

def prepare_stock_data():
    """株価データの取得と前処理（main.pyと同一）"""
    print("=== 株価データ取得開始 ===")
    
    # 株価データ取得
    symbol = "7203.T"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"株価データの取得に失敗: {symbol}")
    
    print(f"株価データ取得成功: {len(data)}行")
    
    # データ前処理（main.pyと同一）
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    
    # main.pyと同様にAdj Close列を追加
    data['Adj Close'] = data['Close']
    
    # 基本的な技術指標追加
    data['Returns'] = data['Close'].pct_change()
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    data['Price_MA_5'] = data['Close'].rolling(window=5).mean()
    data['Price_MA_25'] = data['Close'].rolling(window=25).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)
    data['ATR'] = calculate_atr(data, 14)
    
    print(f"前処理完了: {len(data)}行, {len(data.columns)}列")
    return data

def calculate_rsi(close_prices, window=14):
    """RSI計算"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(data, window=14):
    """ATR計算"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return ranges.rolling(window).mean()

def test_individual_strategy(strategy_class, strategy_name, stock_data, params=None):
    """個別戦略のExit_Signal生成テスト"""
    print(f"\n=== {strategy_name} 個別調査開始 ===")
    
    try:
        # 戦略インスタンス作成（main.pyと同じ方式）
        if strategy_name == 'VWAPBreakoutStrategy':
            strategy = strategy_class(
                data=stock_data.copy(),
                index_data=stock_data,  # index_dataとしてstock_dataを使用
                params=params,
                price_column="Adj Close"
            )
        elif strategy_name == 'OpeningGapStrategy':
            strategy = strategy_class(
                data=stock_data.copy(),
                params=params,
                price_column="Adj Close",
                dow_data=stock_data  # dow_dataとしてstock_dataを使用
            )
        elif strategy_name == 'ContrarianStrategy':
            strategy = strategy_class(stock_data.copy())
        else:
            # その他の戦略は共通パラメータで初期化
            strategy = strategy_class(
                data=stock_data.copy(),
                params=params,
                price_column="Adj Close"
            )
        
        print(f"戦略インスタンス作成成功: {strategy_name}")
        
        # バックテスト実行（copilot-instructions.md必須要件）
        result = strategy.backtest()
        
        if result is None or result.empty:
            print(f"❌ {strategy_name}: バックテスト結果が空")
            return None
        
        print(f"バックテスト実行成功: {len(result)}行")
        
        # Entry_Signal分析
        if 'Entry_Signal' in result.columns:
            entry_counts = result['Entry_Signal'].value_counts().sort_index()
            print(f"Entry_Signal分布:")
            for value, count in entry_counts.items():
                print(f"  {value}: {count}件")
            entry_total = (result['Entry_Signal'] == 1).sum()
        else:
            print("❌ Entry_Signal列が存在しません")
            entry_total = 0
        
        # Exit_Signal分析（重要調査）
        if 'Exit_Signal' in result.columns:
            exit_counts = result['Exit_Signal'].value_counts().sort_index()
            print(f"Exit_Signal分布:")
            for value, count in exit_counts.items():
                print(f"  {value}: {count}件")
            
            # Exit_Signal=-1の詳細確認
            exit_minus_one = result[result['Exit_Signal'] == -1]
            exit_plus_one = result[result['Exit_Signal'] == 1]
            exit_zero = result[result['Exit_Signal'] == 0]
            
            print(f"Exit_Signal = -1: {len(exit_minus_one)}件")
            print(f"Exit_Signal = 1: {len(exit_plus_one)}件")
            print(f"Exit_Signal = 0: {len(exit_zero)}件")
            
            # Exit_Signal=-1のサンプル表示
            if len(exit_minus_one) > 0:
                print("Exit_Signal=-1のサンプル:")
                sample_data = exit_minus_one[['Close', 'Exit_Signal']].head(3)
                for idx, row in sample_data.iterrows():
                    print(f"  {idx}: Close={row['Close']:.2f}, Exit_Signal={row['Exit_Signal']}")
            else:
                print("⚠️ Exit_Signal=-1が存在しません")
            
            exit_total = (result['Exit_Signal'] != 0).sum()
        else:
            print("❌ Exit_Signal列が存在しません")
            exit_total = 0
            exit_counts = {}
        
        # 結果サマリー
        strategy_result = {
            'strategy_name': strategy_name,
            'total_rows': len(result),
            'entry_signals': entry_total,
            'exit_signals': exit_total,
            'exit_signal_distribution': exit_counts.to_dict() if 'Exit_Signal' in result.columns else {},
            'exit_minus_one_count': len(result[result['Exit_Signal'] == -1]) if 'Exit_Signal' in result.columns else 0,
            'has_exit_signal_column': 'Exit_Signal' in result.columns,
            'backtest_success': True
        }
        
        print(f"✅ {strategy_name} 調査完了")
        print(f"   エントリー: {entry_total}件, エグジット: {exit_total}件")
        print(f"   Exit_Signal=-1: {strategy_result['exit_minus_one_count']}件")
        
        return strategy_result
        
    except Exception as e:
        print(f"❌ {strategy_name} エラー: {e}")
        return {
            'strategy_name': strategy_name,
            'error': str(e),
            'backtest_success': False
        }

def main():
    """TODO-007 Phase 1: 戦略個別Exit_Signal生成調査"""
    print("=== TODO-007 Phase 1: 戦略統合処理でのシグナル変換調査 ===")
    print("緊急優先調査: EXIT_SIGNAL=-1生成の実態確認")
    
    # 株価データ準備
    try:
        stock_data = prepare_stock_data()
    except Exception as e:
        print(f"❌ 株価データ準備エラー: {e}")
        return
    
    # 戦略定義（デフォルトパラメータで実行）
    strategies_config = [
        {
            'class': VWAPBreakoutStrategy,
            'name': 'VWAPBreakoutStrategy',
            'params': None
        },
        {
            'class': MomentumInvestingStrategy,
            'name': 'MomentumInvestingStrategy',
            'params': None
        },
        {
            'class': BreakoutStrategy,
            'name': 'BreakoutStrategy',
            'params': None
        },
        {
            'class': VWAPBounceStrategy,
            'name': 'VWAPBounceStrategy',
            'params': None
        },
        {
            'class': OpeningGapStrategy,
            'name': 'OpeningGapStrategy',
            'params': None
        },
        {
            'class': ContrarianStrategy,
            'name': 'ContrarianStrategy',
            'params': None
        },
        {
            'class': GCStrategy,
            'name': 'GCStrategy',
            'params': None
        }
    ]
    
    # 全戦略の個別調査実行
    all_results = []
    total_exit_minus_one = 0
    total_strategies_with_exit_minus_one = 0
    
    for strategy_config in strategies_config:
        result = test_individual_strategy(
            strategy_config['class'],
            strategy_config['name'],
            stock_data,
            strategy_config['params']
        )
        
        if result and result.get('backtest_success', False):
            all_results.append(result)
            exit_minus_one_count = result.get('exit_minus_one_count', 0)
            total_exit_minus_one += exit_minus_one_count
            if exit_minus_one_count > 0:
                total_strategies_with_exit_minus_one += 1
    
    # Phase 1 結果サマリー
    print("\n=== TODO-007 Phase 1 結果サマリー ===")
    print(f"調査成功戦略数: {len(all_results)}/{len(strategies_config)}")
    print(f"Exit_Signal=-1 総生成数: {total_exit_minus_one}件")
    print(f"Exit_Signal=-1 生成戦略数: {total_strategies_with_exit_minus_one}/{len(all_results)}")
    
    # 戦略別詳細結果
    print("\n戦略別Exit_Signal=-1生成状況:")
    for result in all_results:
        strategy_name = result['strategy_name']
        exit_minus_one_count = result.get('exit_minus_one_count', 0)
        entry_count = result.get('entry_signals', 0)
        exit_count = result.get('exit_signals', 0)
        
        status = "✅ 生成あり" if exit_minus_one_count > 0 else "❌ 生成なし"
        print(f"  {strategy_name}: {status}")
        print(f"    Entry: {entry_count}, Exit: {exit_count}, Exit(-1): {exit_minus_one_count}")
    
    # 重大発見の判定
    if total_exit_minus_one == 0:
        print("\n🚨 重大発見: 全戦略でExit_Signal=-1が生成されていません")
        print("   TODO-003修正効果なしの根本原因: 戦略レベルで-1シグナルが生成されていない")
        print("   Phase 2で戦略実装の詳細調査が必要")
    elif total_exit_minus_one > 0 and total_strategies_with_exit_minus_one < len(all_results):
        print(f"\n⚠️ 部分的問題: {len(all_results) - total_strategies_with_exit_minus_one}戦略でExit_Signal=-1未生成")
        print("   Phase 2で戦略実装差異の調査が必要")
    else:
        print("\n✅ 全戦略でExit_Signal=-1が正常生成されています")
        print("   TODO-003修正効果なしの原因はmain.py統合処理にある可能性")
        print("   Phase 2で統合処理の詳細調査が必要")
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"todo_007_phase1_individual_strategy_analysis_{timestamp}.json"
    
    phase1_results = {
        'analysis_timestamp': timestamp,
        'phase': 'TODO-007 Phase 1',
        'investigation_type': 'Individual Strategy Exit_Signal Generation',
        'total_strategies_tested': len(strategies_config),
        'successful_strategies': len(all_results),
        'total_exit_minus_one_generated': total_exit_minus_one,
        'strategies_with_exit_minus_one': total_strategies_with_exit_minus_one,
        'strategy_results': all_results,
        'critical_finding': total_exit_minus_one == 0,
        'next_phase_required': True
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(phase1_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 1調査結果保存: {results_file}")
    print("TODO-007 Phase 1 完了 - Phase 2準備完了")
    
    return phase1_results

if __name__ == "__main__":
    main()