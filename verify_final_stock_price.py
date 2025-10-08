"""
実際の株価終値確認スクリプト
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from data_fetcher import get_parameters_and_data

def verify_final_price():
    """実際の株価データから最終日終値を確認"""
    
    try:
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        print(f"=== {ticker} 実際の株価データ ===")
        print(f"期間: {start_date} - {end_date}")
        print(f"データ行数: {len(stock_data)}")
        print()
        
        # 最終日の株価情報
        final_row = stock_data.iloc[-1]
        final_date = stock_data.index[-1]
        
        print(f"最終日: {final_date}")
        print(f"最終日終値: {final_row['Close']:.8f}")
        print(f"最終日調整後終値: {final_row['Adj Close']:.8f}")
        print()
        
        # 6438.89との比較
        target_price = 6438.89
        close_price = final_row['Close']
        adj_close_price = final_row['Adj Close']
        
        print("=== 6438.89との比較 ===")
        print(f"ターゲット価格: {target_price}")
        print(f"Close価格との差: {abs(close_price - target_price):.8f}")
        print(f"Adj Close価格との差: {abs(adj_close_price - target_price):.8f}")
        print()
        
        # 最も近い価格を特定
        if abs(close_price - target_price) < abs(adj_close_price - target_price):
            nearest_price = close_price
            price_type = "Close"
        else:
            nearest_price = adj_close_price  
            price_type = "Adj Close"
            
        print(f"最も近い価格: {price_type} = {nearest_price:.8f}")
        print(f"一致判定 (差<0.01): {abs(nearest_price - target_price) < 0.01}")
        
        # 最終5日間の価格推移
        print()
        print("=== 最終5日間の価格推移 ===")
        final_5_days = stock_data[['Close', 'Adj Close']].tail(5)
        for date, row in final_5_days.iterrows():
            print(f"{date.strftime('%Y-%m-%d')}: Close={row['Close']:.2f}, Adj Close={row['Adj Close']:.2f}")
            
        return {
            'final_date': final_date,
            'final_close': close_price,
            'final_adj_close': adj_close_price,
            'target_price': target_price,
            'nearest_match': nearest_price,
            'price_type_matched': price_type,
            'is_match': abs(nearest_price - target_price) < 0.01
        }
        
    except Exception as e:
        print(f"エラー: {e}")
        return None

if __name__ == "__main__":
    result = verify_final_price()
    if result:
        print()
        print("=== 結論 ===")
        if result['is_match']:
            print(f"[OK] 6438.89は最終日の{result['price_type_matched']}価格と一致")
            print("→ 強制決済ロジック自体は最終日価格を正しく取得")
            print("→ 問題は「全取引が強制決済されている」こと")
        else:
            print("[ERROR] 6438.89は最終日の価格と不一致")
            print("→ 価格取得ロジックに問題あり")