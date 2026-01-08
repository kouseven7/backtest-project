#!/usr/bin/env python3
"""
ポートフォリオ資産曲線修正テスト

修正内容:
- daily_pnl の適切な設定
- execution_details からの PnL 計算
- portfolio_value の動的更新
"""
import sys
import os
import pandas as pd
from datetime import datetime
sys.path.append(os.path.abspath('.'))

def test_portfolio_equity_curve():
    """ポートフォリオ資産曲線修正のテスト"""
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        print("[TEST] ポートフォリオ資産曲線修正テスト開始")
        print("=" * 50)
        
        # テスト実行
        backtester = DSSMSIntegratedBacktester()
        
        # 短い期間でテスト（2024-11-01から2024-11-05まで）
        start_date = '2024-11-01'
        end_date = '2024-11-05'
        
        print(f"[実行期間] {start_date} → {end_date}")
        
        # バックテスト実行（自動入力）
        os.environ['BACKTEST_START'] = start_date
        os.environ['BACKTEST_END'] = end_date
        
        result = backtester.run_backtest(start_date, end_date)
        
        print(f"[実行結果] {result}")
        
        # 最新の出力ディレクトリ確認
        output_base = 'output/dssms_integration'
        if os.path.exists(output_base):
            dirs = [d for d in os.listdir(output_base) if d.startswith('dssms_')]
            if dirs:
                latest_dir = sorted(dirs)[-1]
                csv_path = os.path.join(output_base, latest_dir, 'portfolio_equity_curve.csv')
                
                print(f"[出力ディレクトリ] {latest_dir}")
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    
                    print(f"[CSV確認] {csv_path}")
                    print(f"行数: {len(df)}")
                    print("最初の5行:")
                    print(df.head())
                    print("\n")
                    
                    portfolio_min = df["Portfolio_Value"].min()
                    portfolio_max = df["Portfolio_Value"].max()
                    daily_pnl_sum = df["Daily_PnL"].sum()
                    
                    print(f"Portfolio_Value範囲: {portfolio_min:,.0f} → {portfolio_max:,.0f}")
                    print(f"Daily_PnL合計: {daily_pnl_sum:,.2f}")
                    
                    # 修正効果確認
                    if portfolio_min != portfolio_max:
                        print("[SUCCESS] Portfolio_Valueが動的に変化しています！")
                    else:
                        print("[ISSUE] Portfolio_Valueが固定のままです")
                        
                    if daily_pnl_sum != 0:
                        print("[SUCCESS] Daily_PnLが計算されています！")
                    else:
                        print("[ISSUE] Daily_PnLがすべて0です")
                        
                    # 詳細検証
                    non_zero_pnl = df[df["Daily_PnL"] != 0]
                    if len(non_zero_pnl) > 0:
                        print(f"取引日数: {len(non_zero_pnl)}日")
                        print("取引があった日:")
                        for _, row in non_zero_pnl.iterrows():
                            print(f"  {row['Date']}: {row['Daily_PnL']:+,.2f}円 (Portfolio: {row['Portfolio_Value']:,.0f}円)")
                    
                else:
                    print("[ERROR] portfolio_equity_curve.csv が作成されていません")
            else:
                print("[ERROR] 出力ディレクトリが見つかりません")
        else:
            print("[ERROR] output/dssms_integration ディレクトリが存在しません")
            
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_portfolio_equity_curve()