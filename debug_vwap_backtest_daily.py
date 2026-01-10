"""
VWAPBreakoutStrategy Debug Tester - DSSMS取引0件問題詳細調査

VWAPBreakoutStrategyのbacktest_daily()メソッドでエントリーシグナルが発生しない
具体的な理由を調査し、どの条件でフィルタリングされているかを特定する。

Author: AI Assistant
Created: 2026-01-10
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトパス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_fetcher import get_parameters_and_data
from strategies.VWAP_Breakout import VWAPBreakoutStrategy

def debug_vwap_backtest_daily():
    """VWAPBreakoutStrategyのbacktest_daily()メソッドを詳しくデバッグ"""
    
    print("=" * 60)
    print("VWAPBreakoutStrategy Debug Test - DSSMS取引0件問題調査")
    print("=" * 60)
    
    # データ取得
    print("Step 1: データ取得中...")
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(
        ticker="6723",  # 最新のDSSMS選択銘柄を使用
        start_date="2024-09-01",
        end_date="2025-01-31",
        warmup_days=150
    )
    
    print(f"取得完了: {ticker}, {len(stock_data)}行, 期間: {start_date} - {end_date}")
    print(f"データ列: {stock_data.columns.tolist()}")
    
    # 戦略初期化
    print("\nStep 2: VWAPBreakoutStrategy初期化...")
    strategy = VWAPBreakoutStrategy(stock_data, index_data)
    print(f"初期化完了: データ形状={strategy.data.shape}")
    
    # テスト日付設定
    test_dates = [
        "2025-01-15",
        "2025-01-20", 
        "2025-01-25",
        "2025-01-30",
        "2025-01-31"
    ]
    
    print("\nStep 3: 日次バックテスト実行...")
    for test_date_str in test_dates:
        test_date = datetime.strptime(test_date_str, "%Y-%m-%d")
        
        print(f"\n--- {test_date_str} デバッグ ---")
        
        # backtest_daily()実行
        result = strategy.backtest_daily(test_date, stock_data, existing_position=None)
        
        print(f"結果: action={result['action']}, signal={result['signal']}, price={result['price']}, shares={result['shares']}")
        print(f"理由: {result['reason']}")
        
        # より詳細な情報を取得
        if test_date in stock_data.index:
            current_idx = stock_data.index.get_loc(test_date)
            current_price = stock_data.iloc[current_idx]['Close']
            
            # インジケーター値を取得
            if 'VWAP' in strategy.data.columns:
                vwap = strategy.data.iloc[current_idx]['VWAP']
                print(f"詳細: current_price={current_price:.2f}, VWAP={vwap:.2f}")
                
                # VWAPブレイクアウト条件チェック
                vwap_breakout = current_price > vwap
                print(f"VWAPブレイクアウト条件: {vwap_breakout} (price > VWAP: {current_price:.2f} > {vwap:.2f})")
                
            # 移動平均線の情報
            if 'SMA_10' in strategy.data.columns and 'SMA_30' in strategy.data.columns:
                sma_10 = strategy.data.iloc[current_idx]['SMA_10']
                sma_30 = strategy.data.iloc[current_idx]['SMA_30']
                print(f"移動平均: SMA_10={sma_10:.2f}, SMA_30={sma_30:.2f}")
                
                # トレンド条件チェック
                trend_ok = current_price > sma_30
                print(f"トレンド条件: {trend_ok} (price > SMA_30: {current_price:.2f} > {sma_30:.2f})")
                
            # 出来高情報
            current_volume = stock_data.iloc[current_idx]['Volume']
            if current_idx > 0:
                previous_volume = stock_data.iloc[current_idx - 1]['Volume']
                volume_ratio = current_volume / previous_volume if previous_volume > 0 else 0
                print(f"出来高: current={current_volume:,.0f}, previous={previous_volume:,.0f}, ratio={volume_ratio:.2f}")
                
                volume_threshold = 1.2  # デフォルト閾値
                volume_ok = current_volume >= previous_volume * volume_threshold
                print(f"出来高条件: {volume_ok} (ratio >= {volume_threshold}: {volume_ratio:.2f} >= {volume_threshold})")
        else:
            print(f"警告: {test_date_str} はデータに存在しません")
    
    print("\n" + "=" * 60)
    print("VWAPBreakoutStrategy Debug Test 完了")
    print("=" * 60)

if __name__ == "__main__":
    try:
        debug_vwap_backtest_daily()
    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()