"""
同日2件以上SELL問題の詳細分析スクリプト

Task 8とTask 11の統合検証の一環として、
execution_results.jsonから同日に2件以上のSELLが発生したケースを分析します。

Author: Backtest Project Team
Created: 2025-12-08
"""

import json
from collections import defaultdict
from pathlib import Path

def analyze_same_day_sell():
    """同日2件以上SELLの詳細分析"""
    
    # JSONファイル読み込み
    json_path = Path("output/dssms_integration/dssms_20251208_193732/dssms_execution_results.json")
    
    if not json_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    # execution_detailsを取得
    data = result.get('execution_details', [])
    
    print(f"[INFO] 総execution_details件数: {len(data)}件")
    
    # 日付別に集計
    by_date = defaultdict(lambda: {'BUY': [], 'SELL': []})
    
    for detail in data:
        date = detail['timestamp'][:10]  # YYYY-MM-DD
        action = detail['action']
        by_date[date][action].append(detail)
    
    # 同日2件以上SELLのケースを抽出
    same_day_2sell = {}
    for date, actions in by_date.items():
        if len(actions['SELL']) >= 2:
            same_day_2sell[date] = {
                'BUY': len(actions['BUY']),
                'SELL': len(actions['SELL']),
                'symbols': list(set([d['symbol'] for d in actions['BUY'] + actions['SELL']])),
                'sell_details': [
                    {
                        'symbol': d['symbol'],
                        'strategy': d.get('strategy_name', 'N/A'),
                        'is_forced': d.get('status') == 'force_closed' or 'ForceClose' in d.get('strategy_name', ''),
                        'quantity': d.get('quantity', 'N/A')
                    }
                    for d in actions['SELL']
                ]
            }
    
    print(f"\n=== 同日2件以上SELL分析結果 ===")
    print(f"対象ケース: {len(same_day_2sell)}件\n")
    
    # 詳細表示（最初の10ケース）
    for i, (date, info) in enumerate(sorted(same_day_2sell.items())[:10], 1):
        print(f"[ケース {i}] {date}")
        print(f"  BUY: {info['BUY']}件, SELL: {info['SELL']}件")
        print(f"  銘柄: {', '.join(info['symbols'])}")
        print(f"  SELL詳細:")
        for j, sell in enumerate(info['sell_details'], 1):
            forced_flag = "[ForceClose]" if sell['is_forced'] else ""
            print(f"    {j}. {sell['symbol']} - {sell['strategy']} {forced_flag} (数量: {sell['quantity']})")
        print()
    
    # ForceCloseの統計
    force_close_count = 0
    normal_sell_count = 0
    
    for date, info in same_day_2sell.items():
        for sell in info['sell_details']:
            if sell['is_forced']:
                force_close_count += 1
            else:
                normal_sell_count += 1
    
    print(f"=== ForceClose統計 ===")
    print(f"ForceClose: {force_close_count}件")
    print(f"通常SELL: {normal_sell_count}件")
    print(f"総SELL: {force_close_count + normal_sell_count}件")
    
    # Task 8/11の対応状況確認
    print(f"\n=== Task 8/11対応状況の推定 ===")
    print(f"SUPPRESS系ログが0件であることから:")
    print(f"  - ForceClose実行中に通常SELL処理が抑制されていない")
    print(f"  - または、ForceClose実行時に通常SELL処理が発生していない")
    print(f"  - 同日2件SELLは異なるタイミングで実行された可能性")
    
    return same_day_2sell

if __name__ == "__main__":
    analyze_same_day_sell()
