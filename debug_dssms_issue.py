import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.dssms.dssms_backtester import DSSMSBacktester

def debug_dssms_backtester():
    """DSSMS Backtesterの問題を診断する"""
    print("=" * 80)
    print("DSSMS Backtester問題診断")
    print("=" * 80)
    
    try:
        # 基本設定
        start_date = '2023-01-01'
        end_date = '2023-12-31' 
        initial_capital = 1000000
        
        print(f"\n1. 基本設定確認:")
        print(f"   開始日: {start_date}")
        print(f"   終了日: {end_date}")
        print(f"   初期資本: ¥{initial_capital:,}")
        
        # DSSMSBacktesterのインスタンス作成
        print(f"\n2. DSSMSBacktesterのインスタンス作成...")
        backtester = DSSMSBacktester()
        print(f"   ✓ インスタンス作成成功")
        
        # 属性確認
        print(f"\n3. Backtester属性確認...")
        print(f"   initial_capital: {backtester.initial_capital}")
        print(f"   switch_history: {len(backtester.switch_history)} items")
        print(f"   portfolio_history: {len(backtester.portfolio_history)} items")
        print(f"   performance_history keys: {list(backtester.performance_history.keys())}")
        
        # データ取得確認
        print(f"\n4. データ取得テスト...")
        
        # テスト用銘柄
        test_symbols = ['7203', '9984', '6758', '8306', '6861']
        
        import yfinance as yf
        sample_data = {}
        for symbol in test_symbols[:2]:  # 最初の2銘柄のみテスト
            ticker = f"{symbol}.T"
            print(f"   {symbol}のデータ取得中...")
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    sample_data[symbol] = data
                    print(f"   ✓ {symbol}: {len(data)}日分のデータ取得")
                    print(f"     価格範囲: ¥{data['Close'].min():.2f} - ¥{data['Close'].max():.2f}")
                else:
                    print(f"   ✗ {symbol}: データが空です")
            except Exception as e:
                print(f"   ✗ {symbol}: エラー - {e}")
        
        if not sample_data:
            print("   ⚠️ テストデータが取得できませんでした")
            return
            
        # バックテスト実行テスト
        print(f"\n5. バックテスト機能確認...")
        try:
            # simulate_dynamic_selectionメソッドを直接テスト
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            results = backtester.simulate_dynamic_selection(
                start_date=start_dt,
                end_date=end_dt,
                symbol_universe=list(sample_data.keys())
            )
            
            print(f"   ✓ シミュレーション完了")
            print(f"   結果タイプ: {type(results)}")
            
            if isinstance(results, dict):
                print(f"\n6. 結果分析:")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value}")
                    elif isinstance(value, pd.DataFrame):
                        print(f"   {key}: DataFrame ({len(value)} rows)")
                    elif isinstance(value, list):
                        print(f"   {key}: List ({len(value)} items)")
                    else:
                        print(f"   {key}: {type(value)}")
                        
                # ポートフォリオ履歴の詳細確認
                print(f"\n7. ポートフォリオ履歴詳細確認:")
                if backtester.portfolio_history:
                    print(f"   ポートフォリオ履歴: {len(backtester.portfolio_history)} records")
                    if len(backtester.portfolio_history) > 0:
                        first_record = backtester.portfolio_history[0]
                        last_record = backtester.portfolio_history[-1]
                        print(f"   最初の記録: {first_record}")
                        print(f"   最後の記録: {last_record}")
                else:
                    print(f"   ⚠️ ポートフォリオ履歴が空です")
                
                # パフォーマンス履歴の確認
                print(f"\n8. パフォーマンス履歴確認:")
                for key, values in backtester.performance_history.items():
                    print(f"   {key}: {len(values)} items")
                    if len(values) > 0:
                        print(f"     最初: {values[0]}")
                        print(f"     最後: {values[-1]}")
                        
                # 切替履歴の確認
                print(f"\n9. 切替履歴確認:")
                print(f"   切替回数: {len(backtester.switch_history)}")
                if len(backtester.switch_history) > 0:
                    for i, switch in enumerate(backtester.switch_history[:3]):  # 最初の3件
                        print(f"   切替{i+1}: {switch}")
                        
        except Exception as e:
            print(f"   ✗ バックテスト実行エラー: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"デバッグ中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dssms_backtester()
