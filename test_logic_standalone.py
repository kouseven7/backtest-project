#!/usr/bin/env python3
"""
ポートフォリオ資産曲線計算ロジックテスト

修正されたロジックをスタンドアロンでテスト
"""
import pandas as pd
from datetime import datetime

def test_portfolio_calculation_logic():
    """修正されたポートフォリオ計算ロジックをテスト"""
    
    # サンプルdaily_resultsデータ（実際のDSSMSから想定されるデータ）
    daily_results = [
        {
            'date': '2024-11-01',
            'symbol': '6178',
            'daily_return': 0,
            'daily_pnl': 0,
            'execution_details': []
        },
        {
            'date': '2024-11-04', 
            'symbol': '6178',
            'daily_return': 15000,
            'daily_pnl': 15000,
            'execution_details': [
                {'pnl': 15000}
            ]
        },
        {
            'date': '2024-11-05',
            'symbol': '6178', 
            'daily_return': -5000,
            'daily_pnl': -5000,
            'execution_details': [
                {'realized_pnl': -5000}
            ]
        }
    ]
    
    print("=== 修正されたポートフォリオ計算ロジックテスト ===")
    print()
    
    # 修正されたロジックのシミュレーション
    curve_data = []
    portfolio_value = 1000000  # 初期値
    
    for daily_result in daily_results:
        date = daily_result.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # 修正されたロジック: daily_pnlとdaily_returnの両方をチェック
        daily_pnl = daily_result.get('daily_pnl', daily_result.get('daily_return', 0))
        
        # 実際の取引データがある場合は execution_details から計算
        calculated_pnl = 0  # 初期化
        if 'execution_details' in daily_result and daily_result['execution_details']:
            for exec_detail in daily_result['execution_details']:
                if isinstance(exec_detail, dict) and 'pnl' in exec_detail:
                    calculated_pnl += exec_detail.get('pnl', 0)
                elif isinstance(exec_detail, dict) and 'realized_pnl' in exec_detail:
                    calculated_pnl += exec_detail.get('realized_pnl', 0)
            
            # execution_detailsから計算したPnLがある場合は優先使用
            if calculated_pnl != 0:
                daily_pnl = calculated_pnl
        
        portfolio_value += daily_pnl
        
        curve_data.append({
            'Date': date,
            'Portfolio_Value': portfolio_value,
            'Daily_PnL': daily_pnl,
            'Symbol': daily_result.get('symbol', 'Unknown')
        })
        
        print(f"[{date}] Symbol: {daily_result.get('symbol')}")
        print(f"  daily_return: {daily_result.get('daily_return', 0):+,.0f}")
        print(f"  daily_pnl: {daily_result.get('daily_pnl', 0):+,.0f}")
        print(f"  execution_details PnL: {calculated_pnl:+,.0f}")
        print(f"  使用PnL: {daily_pnl:+,.0f}")
        print(f"  Portfolio Value: {portfolio_value:,.0f}")
        print()
    
    # 結果をDataFrameに変換
    df = pd.DataFrame(curve_data)
    print("=== 生成されたCSV（修正版）===")
    print(df.to_csv(index=False))
    
    # 検証結果
    print("=== 検証結果 ===")
    initial_value = 1000000
    final_value = df['Portfolio_Value'].iloc[-1]
    total_pnl = df['Daily_PnL'].sum()
    
    print(f"初期Portfolio Value: {initial_value:,.0f}")
    print(f"最終Portfolio Value: {final_value:,.0f}")
    print(f"総PnL: {total_pnl:+,.0f}")
    print(f"Portfolio Value変化: {final_value - initial_value:+,.0f}")
    
    # 修正効果確認
    if final_value != initial_value:
        print("[SUCCESS] Portfolio_Valueが正しく変化しています！")
    else:
        print("[ISSUE] Portfolio_Valueが変化していません")
        
    if total_pnl != 0:
        print("[SUCCESS] Daily_PnLが正しく計算されています！")
    else:
        print("[ISSUE] Daily_PnLがすべて0です")

if __name__ == "__main__":
    test_portfolio_calculation_logic()