"""
強制決済価格問題調査スクリプト
5803.Tの実際の株価データと6438.89価格の検証
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def investigate_price_issue():
    """強制決済価格問題の詳細調査"""
    
    # 5803.Tの株価データを取得
    ticker = '5803.T'
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        print(f'=== {ticker} 株価データ分析 ===')
        print(f'期間: {start_date} - {end_date}')
        print(f'データ行数: {len(data)}')
        print('')
        
        print('終値統計:')
        print(f'最高値: {data["Close"].max():.2f}')
        print(f'最安値: {data["Close"].min():.2f}')
        print(f'平均値: {data["Close"].mean():.2f}')
        print(f'最終日終値: {data["Close"].iloc[-1]:.2f}')
        print('')
        
        print('6438.89との比較:')
        target_price = 6438.89
        print(f'ターゲット価格: {target_price}')
        final_price = data["Close"].iloc[-1]
        print(f'最終日終値との差: {abs(final_price - target_price):.2f}')
        print(f'最終日終値との一致: {abs(final_price - target_price) < 0.01}')
        print('')
        
        print('最終10日間の終値:')
        final_10_days = data['Close'].tail(10)
        for date, price in final_10_days.items():
            print(f'{date.strftime("%Y-%m-%d")}: {price:.2f}')
        
        print('')
        print('=== 問題分析 ===')
        
        # 6438.89が期間中に存在するかチェック
        matching_prices = data[abs(data['Close'] - target_price) < 0.01]
        if len(matching_prices) > 0:
            print(f'6438.89に近い価格の日付:')
            for date, row in matching_prices.iterrows():
                print(f'  {date.strftime("%Y-%m-%d")}: {row["Close"]:.2f}')
        else:
            print('6438.89に近い価格は期間中に存在しません')
            
        # 最終日価格確認
        if abs(final_price - target_price) < 0.01:
            print('')
            print('[結論] 6438.89は最終日の終値と一致 - 強制決済ロジックは正常')
        else:
            print('')
            print(f'[警告] 6438.89は最終日終値({final_price:.2f})と不一致 - 異常')
            
        return {
            'final_price': final_price,
            'target_price': target_price,
            'is_match': abs(final_price - target_price) < 0.01,
            'data_range': f'{data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")}',
            'total_records': len(data)
        }
        
    except Exception as e:
        print(f'データ取得エラー: {e}')
        return None

if __name__ == "__main__":
    result = investigate_price_issue()
    if result:
        print('')
        print('=== 調査結果サマリー ===')
        for key, value in result.items():
            print(f'{key}: {value}')