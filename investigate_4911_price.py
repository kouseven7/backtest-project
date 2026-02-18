"""
4911 (資生堂) の実際の株価推移を調査

2024-06-05エントリー (5,217.21円) から
2024-07-08決済 (4,664円, -10.6%) までの
各日の含み損益を計算し、ストップロスが機能したかを検証。

Author: Backtest Project Team
Created: 2026-02-17
"""
import yfinance as yf
import pandas as pd

def investigate_4911():
    print("=" * 80)
    print("【4911 (資生堂) 株価推移調査】")
    print("=" * 80)
    
    # データ取得
    data = yf.download('4911.T', 
                       start='2024-06-05', 
                       end='2024-07-09',
                       auto_adjust=False,
                       progress=False)
    
    # エントリー情報
    entry_date = '2024-06-05'
    entry_price = 5217.21
    exit_date = '2024-07-08'
    exit_price = 4664.0
    
    print(f"\nエントリー情報:")
    print(f"  日付: {entry_date}")
    print(f"  価格: {entry_price:.2f}円")
    print(f"\n決済情報:")
    print(f"  日付: {exit_date}")
    print(f"  価格: {exit_price:.2f}円")
    print(f"  損益: -55,321円 (-10.6%)")
    
    # 各日の含み損益を計算
    data['含み損益'] = (data['Close'] - entry_price) * 100
    data['含み損益率'] = (data['Close'] - entry_price) / entry_price * 100
    
    print("\n" + "=" * 80)
    print("【日別株価と含み損益】")
    print("=" * 80)
    print(f"{'日付':<12} {'始値':>8} {'高値':>8} {'安値':>8} {'終値':>8} {'含み損益':>10} {'損益率':>8}")
    print("-" * 80)
    
    for date, row in data.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        print(f"{date_str:<12} "
              f"{float(row['Open']):>8.1f} "
              f"{float(row['High']):>8.1f} "
              f"{float(row['Low']):>8.1f} "
              f"{float(row['Close']):>8.1f} "
              f"{float(row['含み損益']):>10.0f}円 "
              f"{float(row['含み損益率']):>7.1f}%")
    
    # -5%ストップロスに達した日を探す
    print("\n" + "=" * 80)
    print("【ストップロス分析】")
    print("=" * 80)
    
    stop_loss_5pct = -5.0
    stop_loss_reached = data[data['含み損益率'] <= stop_loss_5pct]
    
    if not stop_loss_reached.empty:
        first_stop_loss_date = stop_loss_reached.index[0]
        first_stop_loss_row = stop_loss_reached.iloc[0]
        
        print(f"-5%ストップロスに到達した最初の日: {first_stop_loss_date.strftime('%Y-%m-%d')}")
        print(f"  終値: {float(first_stop_loss_row['Close']):.1f}円")
        print(f"  含み損益率: {float(first_stop_loss_row['含み損益率']):.1f}%")
        print(f"  含み損益: {float(first_stop_loss_row['含み損益']):.0f}円")
        
        # もし-5%で決済していた場合の損失
        theoretical_exit_price = entry_price * 0.95
        theoretical_loss = (theoretical_exit_price - entry_price) * 100
        print(f"\n-5%ストップロスで決済した場合の理論値:")
        print(f"  決済価格: {theoretical_exit_price:.2f}円")
        print(f"  損失: {theoretical_loss:.0f}円 (-5.0%)")
        
        actual_loss = -55321
        saved_amount = actual_loss - theoretical_loss
        print(f"\n実際の損失との差額:")
        print(f"  実際の損失: {actual_loss:.0f}円 (-10.6%)")
        print(f"  防げた損失: {-saved_amount:.0f}円")
    else:
        print("-5%ストップロスに到達した日はありません")
    
    # -2%ストップロスの分析も追加
    print("\n" + "=" * 80)
    print("【-2%ストップロス分析（参考）】")
    print("=" * 80)
    
    stop_loss_2pct = -2.0
    stop_loss_2pct_reached = data[data['含み損益率'] <= stop_loss_2pct]
    
    if not stop_loss_2pct_reached.empty:
        first_stop_loss_2pct_date = stop_loss_2pct_reached.index[0]
        first_stop_loss_2pct_row = stop_loss_2pct_reached.iloc[0]
        
        print(f"-2%ストップロスに到達した最初の日: {first_stop_loss_2pct_date.strftime('%Y-%m-%d')}")
        print(f"  終値: {float(first_stop_loss_2pct_row['Close']):.1f}円")
        print(f"  含み損益率: {float(first_stop_loss_2pct_row['含み損益率']):.1f}%")
    else:
        print("-2%ストップロスに到達した日はありません")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    investigate_4911()
