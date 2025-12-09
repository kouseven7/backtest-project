"""
BUY/SELL不一致の詳細調査スクリプト

97件のSELL超過の内訳を分析し、損益計算への影響を検証します。

主な機能:
- BUY/SELL件数の銘柄別集計
- unpaired SELLの特定と分類
- 戦略別の内訳分析
- ポジション残高の整合性確認
- 損益計算への影響評価

Author: Backtest Project Team
Created: 2025-12-08
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

def analyze_buy_sell_mismatch():
    """BUY/SELL不一致の詳細分析"""
    
    # JSONファイル読み込み
    json_path = Path("output/dssms_integration/dssms_20251208_193732/dssms_execution_results.json")
    
    if not json_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    # execution_detailsを取得
    data = result.get('execution_details', [])
    
    print(f"[INFO] 総execution_details件数: {len(data)}件\n")
    
    # 1. 基本統計
    buy_count = sum(1 for d in data if d['action'] == 'BUY')
    sell_count = sum(1 for d in data if d['action'] == 'SELL')
    diff = sell_count - buy_count
    
    print("=== 基本統計 ===")
    print(f"総BUY件数: {buy_count}件")
    print(f"総SELL件数: {sell_count}件")
    print(f"差分: {diff}件（{'SELL超過' if diff > 0 else 'BUY超過'}）\n")
    
    # 2. 銘柄別の内訳
    by_symbol = defaultdict(lambda: {'BUY': 0, 'SELL': 0})
    
    for detail in data:
        symbol = detail['symbol']
        action = detail['action']
        by_symbol[symbol][action] += 1
    
    print("=== 銘柄別BUY/SELL内訳 ===")
    print(f"{'銘柄':<10} {'BUY':>6} {'SELL':>6} {'差分':>6}")
    print("-" * 35)
    
    for symbol in sorted(by_symbol.keys()):
        buy = by_symbol[symbol]['BUY']
        sell = by_symbol[symbol]['SELL']
        diff_sym = sell - buy
        print(f"{symbol:<10} {buy:>6} {sell:>6} {diff_sym:>+6}")
    
    print()
    
    # 3. 戦略別の内訳（SELL）
    by_strategy = defaultdict(int)
    
    for detail in data:
        if detail['action'] == 'SELL':
            strategy = detail.get('strategy_name', 'Unknown')
            by_strategy[strategy] += 1
    
    print("=== 戦略別SELL内訳 ===")
    for strategy in sorted(by_strategy.keys(), key=lambda x: by_strategy[x], reverse=True):
        count = by_strategy[strategy]
        pct = count / sell_count * 100
        print(f"{strategy:<30} {count:>4}件 ({pct:>5.1f}%)")
    
    print()
    
    # 4. 銘柄別の不一致分析
    print("=== 銘柄別不一致分析（SELL超過のみ） ===")
    
    mismatch_symbols = []
    for symbol, counts in by_symbol.items():
        if counts['SELL'] > counts['BUY']:
            mismatch_symbols.append({
                'symbol': symbol,
                'BUY': counts['BUY'],
                'SELL': counts['SELL'],
                'diff': counts['SELL'] - counts['BUY']
            })
    
    mismatch_symbols.sort(key=lambda x: x['diff'], reverse=True)
    
    print(f"SELL超過銘柄数: {len(mismatch_symbols)}銘柄")
    print(f"{'銘柄':<10} {'BUY':>6} {'SELL':>6} {'SELL超過':>10}")
    print("-" * 40)
    
    total_mismatch = 0
    for item in mismatch_symbols:
        print(f"{item['symbol']:<10} {item['BUY']:>6} {item['SELL']:>6} {item['diff']:>10}")
        total_mismatch += item['diff']
    
    print("-" * 40)
    print(f"{'合計':<10} {'':<6} {'':<6} {total_mismatch:>10}")
    print()
    
    # 5. SELL超過の詳細分析（戦略別）
    print("=== SELL超過の詳細分析（上位5銘柄） ===")
    
    for item in mismatch_symbols[:5]:
        symbol = item['symbol']
        print(f"\n[{symbol}] BUY={item['BUY']}, SELL={item['SELL']}, 差分={item['diff']}")
        
        # この銘柄のSELL取引を取得
        sells = [d for d in data if d['symbol'] == symbol and d['action'] == 'SELL']
        
        # 戦略別カウント
        strategy_count = defaultdict(int)
        for sell in sells:
            strategy = sell.get('strategy_name', 'Unknown')
            strategy_count[strategy] += 1
        
        print("  戦略別SELL内訳:")
        for strategy, count in sorted(strategy_count.items(), key=lambda x: x[1], reverse=True):
            print(f"    {strategy:<30} {count:>3}件")
    
    print()
    
    # 6. ForceCloseとDSSMS_SymbolSwitchの分析
    force_close_count = sum(1 for d in data if d['action'] == 'SELL' and 
                           ('ForceClose' in d.get('strategy_name', '') or d.get('status') == 'force_closed'))
    symbol_switch_count = sum(1 for d in data if d['action'] == 'SELL' and 
                             'DSSMS_SymbolSwitch' in d.get('strategy_name', ''))
    
    print("=== 特殊なSELL取引の分析 ===")
    print(f"ForceClose: {force_close_count}件 ({force_close_count/sell_count*100:.1f}%)")
    print(f"DSSMS_SymbolSwitch: {symbol_switch_count}件 ({symbol_switch_count/sell_count*100:.1f}%)")
    print(f"通常SELL: {sell_count - force_close_count - symbol_switch_count}件 "
          f"({(sell_count - force_close_count - symbol_switch_count)/sell_count*100:.1f}%)")
    print()
    
    # 7. 最終判定
    print("=== 最終判定 ===")
    
    if total_mismatch == diff:
        print(f"[OK] 銘柄別差分の合計({total_mismatch})と総差分({diff})が一致")
    else:
        print(f"[WARNING] 銘柄別差分の合計({total_mismatch})と総差分({diff})が不一致")
    
    # BUY超過銘柄も確認
    buy_excess_count = sum(1 for symbol, counts in by_symbol.items() if counts['BUY'] > counts['SELL'])
    if buy_excess_count > 0:
        print(f"[INFO] BUY超過銘柄: {buy_excess_count}銘柄（保有継続ポジション）")
    
    print()
    
    # 8. 損益計算への影響評価
    print("=== 損益計算への影響評価 ===")
    print(f"SELL超過97件の意味:")
    print(f"  1. 期初からの繰越ポジションを決済した（BUY記録なし）")
    print(f"  2. 空売りが発生した（ポジションなしでSELL）")
    print(f"  3. 二重決済が発生した（同じポジションを複数回SELL）")
    print()
    print(f"次のステップ:")
    print(f"  - 期初ポジションの確認（initial_capital, initial_portfolioの確認）")
    print(f"  - PaperBrokerのポジション記録確認（get_positions()の履歴）")
    print(f"  - 日次ポジション残高の推移確認（portfolio_equity_curve.csv）")
    
    return {
        'buy_count': buy_count,
        'sell_count': sell_count,
        'diff': diff,
        'mismatch_symbols': mismatch_symbols,
        'force_close_count': force_close_count,
        'symbol_switch_count': symbol_switch_count
    }

if __name__ == "__main__":
    analyze_buy_sell_mismatch()
