#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSSMSデータ取得問題の修正と検証スクリプト
実際の株価データを取得するように修正
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_yfinance_availability():
    """yfinanceの利用可能性テスト"""
    print("=== yfinance利用可能性テスト ===")
    
    try:
        import yfinance as yf
        print("[OK] yfinance インポート成功")
        
        # 実際のデータ取得テスト
        test_symbol = "6758"  # ソニーグループ
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        print(f"[CHART] テストデータ取得: {test_symbol} ({start_date.date()} → {end_date.date()})")
        
        ticker = yf.Ticker(f"{test_symbol}.T")
        data = ticker.history(start=start_date, end=end_date)
        
        if not data.empty:
            print(f"[OK] データ取得成功: {len(data)}行")
            print(f"   最新価格: {data['Close'].iloc[-1]:.2f}円")
            print(f"   価格範囲: {data['Close'].min():.2f} → {data['Close'].max():.2f}円")
            return True
        else:
            print("[ERROR] 空のデータが返されました")
            return False
            
    except ImportError as e:
        print(f"[ERROR] yfinance インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] データ取得エラー: {e}")
        return False

def fix_dssms_data_fetching():
    """DSSMSデータ取得ロジックの修正"""
    print("\n=== DSSMSデータ取得修正 ===")
    
    # 1. まず現在の設定を確認
    try:
        sys.path.append("src/dssms")
        from dssms_integrated_main import DSSMSIntegratedBacktester
        
        print("[LIST] 現在の設定確認:")
        config = {
            'initial_capital': 1000000,
            'switch_cost_rate': 0.001
        }
        
        backtester = DSSMSIntegratedBacktester(config)
        
        # 強制的にyfinanceを使用するように設定
        try:
            import yfinance as yf
            # DATA_FETCHER_AVAILABLEフラグを強制的にTrueに設定
            import src.dssms.dssms_integrated_main as dssms_main
            dssms_main.DATA_FETCHER_AVAILABLE = True
            print("[OK] yfinance使用を強制有効化")
            
            # テスト用のデータ取得実行
            test_date = datetime(2023, 6, 1)
            stock_data, index_data = backtester._get_symbol_data("6758", test_date)
            
            if stock_data is not None and not stock_data.empty:
                print(f"[OK] 実データ取得成功:")
                print(f"   データ期間: {stock_data.index[0].date()} → {stock_data.index[-1].date()}")
                print(f"   データ点数: {len(stock_data)}点")
                
                # 価格データの妥当性チェック
                sample_date = stock_data.index[0]
                sample_price = stock_data.loc[sample_date, 'Close']
                print(f"   サンプル価格 ({sample_date.date()}): {sample_price:.2f}円")
                
                if sample_price != 1000 and sample_price != 1010:
                    print("[OK] 実データ確認（モックではない）")
                    return True
                else:
                    print("[WARNING] モックデータの可能性")
                    return False
            else:
                print("[ERROR] データ取得失敗")
                return False
                
        except Exception as e:
            print(f"[ERROR] データ取得テストエラー: {e}")
            return False
            
    except Exception as e:
        print(f"[ERROR] DSSMS初期化エラー: {e}")
        return False

def create_real_data_backtest():
    """実データを使用したバックテスト実行"""
    print("\n=== 実データバックテスト実行 ===")
    
    try:
        # yfinanceを強制有効化
        import src.dssms.dssms_integrated_main as dssms_main
        dssms_main.DATA_FETCHER_AVAILABLE = True
        
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        config = {
            'initial_capital': 1000000,
            'switch_cost_rate': 0.001
        }
        
        backtester = DSSMSIntegratedBacktester(config)
        
        # 短期間のテストバックテスト（1週間）
        start_date = datetime(2023, 6, 1)
        end_date = datetime(2023, 6, 7)
        
        print(f"[CHART] 短期テストバックテスト: {start_date.date()} → {end_date.date()}")
        
        results = backtester.run_dynamic_backtest(start_date, end_date, ["6758", "7203"])
        
        print(f"[OK] バックテスト実行成功:")
        print(f"   最終資本: {results['portfolio_performance']['final_capital']:,.0f}円")
        print(f"   総収益率: {results['portfolio_performance']['total_return_rate']:.2%}")
        print(f"   取引回数: {len(results.get('daily_results', []))}")
        
        # 取引履歴の価格を確認
        if 'daily_results' in results and results['daily_results']:
            sample_trade = results['daily_results'][0]
            print(f"   サンプル取引価格: {sample_trade.get('current_price', 'N/A')}")
            
            if sample_trade.get('current_price') not in [1000, 1010]:
                print("[OK] 実データ価格確認")
                return True
        
        return False
        
    except Exception as e:
        print(f"[ERROR] 実データバックテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    print("DSSMSデータ取得問題 修正・検証スクリプト")
    print("=" * 60)
    
    # 1. yfinance利用可能性確認
    yf_available = test_yfinance_availability()
    
    if not yf_available:
        print("\n[ERROR] yfinanceが利用できません。インストールが必要です。")
        return
    
    # 2. DSSMSデータ取得修正
    data_fix_success = fix_dssms_data_fetching()
    
    if not data_fix_success:
        print("\n[ERROR] DSSMSデータ取得修正に失敗しました。")
        return
    
    # 3. 実データバックテスト実行
    backtest_success = create_real_data_backtest()
    
    if backtest_success:
        print("\n[SUCCESS] 実データ取得修正成功！")
        print("[IDEA] 次のステップ:")
        print("   1. 完全なバックテストの再実行")
        print("   2. Excel出力の価格データ確認")
        print("   3. 実データでの取引履歴の妥当性確認")
    else:
        print("\n[WARNING] 一部の修正が必要です。")

if __name__ == "__main__":
    main()