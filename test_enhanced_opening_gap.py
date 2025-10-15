"""
OpeningGapEnhancedStrategyのテスト用メインファイル
"""

import pandas as pd
import yfinance as yf
from strategies.Opening_Gap_Enhanced import OpeningGapEnhancedStrategy
import json
import os
from datetime import datetime

def main():
    # データ取得
    start_date = '2024-01-01'
    end_date = '2024-03-31'
    ticker = '7203.T'  # トヨタ自動車
    index_ticker = '^N225'  # 日経平均
    
    # データのダウンロード
    print(f"{ticker}の株価データをダウンロード中...")
    stock_data = yf.download(ticker, start_date, end_date)
    
    print(f"{index_ticker}の指数データをダウンロード中...")
    index_data = yf.download(index_ticker, start_date, end_date)
    
    if stock_data.empty or index_data.empty:
        print("データの取得に失敗しました")
        return
    
    # OpeningGapEnhancedStrategy の初期化
    params = {
        'gap_threshold': 0.01,      # 1%以上のギャップでエントリー
        'profit_target': 0.03,      # 3%の利益確定
        'stop_loss': 0.02,          # 2%の損切り
        'trailing_threshold': 0.015, # 1.5%のトレーリングストップ
        'max_hold_days': 5          # 最大保有日数
    }
    
    strategy = OpeningGapEnhancedStrategy(
        data=stock_data,
        dow_data=index_data,
        params=params,
        price_column="Close"
    )
    
    # バックテスト実行
    print("バックテスト実行中...")
    result = strategy.backtest()
    
    # 結果の表示
    entry_signals = result[result['Entry_Signal'] == 1]
    exit_signals = result[result['Exit_Signal'] != 0]
    
    print(f"\n=== バックテスト結果 ({ticker}) ===")
    print(f"テスト期間: {start_date} から {end_date}")
    print(f"データ数: {len(result)}")
    print(f"エントリー回数: {len(entry_signals)}")
    print(f"エグジット回数: {len(exit_signals)}")
    print(f"未決済残: {len(entry_signals) - len(exit_signals)}")
    
    # 累積リターン
    if 'Cumulative_Return' in result.columns:
        final_return = result['Cumulative_Return'].iloc[-1] - 1
        print(f"累積リターン: {final_return:.2%}")
    
    # 結果をCSV保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, f'enhanced_opening_gap_{timestamp}.csv')
    result.to_csv(csv_file)
    print(f"\n結果をCSVとして保存しました: {csv_file}")
    
    # サマリー情報をJSON形式で保存
    summary = {
        'strategy': 'OpeningGapEnhanced',
        'ticker': ticker,
        'period': {
            'start': start_date,
            'end': end_date
        },
        'trades': {
            'entries': len(entry_signals),
            'exits': len(exit_signals),
            'open_positions': len(entry_signals) - len(exit_signals)
        },
        'parameters': params
    }
    
    if 'Cumulative_Return' in result.columns:
        summary['performance'] = {
            'cumulative_return': float(f"{final_return:.4f}")
        }
    
    json_file = os.path.join(output_dir, f'enhanced_opening_gap_summary_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"サマリー情報をJSONとして保存しました: {json_file}")
    print("\nバックテスト完了")

if __name__ == "__main__":
    main()