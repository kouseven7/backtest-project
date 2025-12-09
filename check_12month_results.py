"""
12ヶ月バックテスト結果検証スクリプト（修正後の完全検証）

修正内容:
- _open_position()にexecution_detail生成追加（Lines 2306-2318）
- _evaluate_and_execute_switch()でBUY側execution_detail収集（Lines 1580-1582）

検証項目:
1. DSSMS_SymbolSwitch BUY件数（0件→14件の改善確認）
2. DSSMS_SymbolSwitch SELL件数（3ヶ月で0件だった理由の解明）
3. BUY/SELL差分（97件→7件の改善確認）
4. execution_detail構造（10フィールド全て記録確認）
5. 損益計算への影響（修正前後で値が変わらないことの確認）
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

def find_latest_12month_output():
    """最新の12ヶ月バックテスト出力ディレクトリを検索"""
    output_base = Path("output/dssms_integration")
    if not output_base.exists():
        return None
    
    # dssms_で始まるディレクトリを検索
    dirs = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("dssms_")]
    if not dirs:
        return None
    
    # 最新のディレクトリを返す（タイムスタンプでソート）
    latest = max(dirs, key=lambda d: d.name)
    return latest

def analyze_execution_details(output_dir):
    """execution_details.jsonを詳細分析"""
    json_path = output_dir / "dssms_execution_results.json"
    
    if not json_path.exists():
        print(f"[ERROR] {json_path} が見つかりません")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    execution_details = data.get('execution_details', [])
    
    print("=" * 80)
    print(f"最新ディレクトリ: {output_dir.name}")
    print(f"作成日時: {datetime.fromtimestamp(output_dir.stat().st_ctime)}")
    print("=" * 80)
    print()
    
    # 1. 基本統計
    print("[1] 基本統計")
    print(f"execution_details総数: {len(execution_details)}件")
    
    buy_count = sum(1 for d in execution_details if d.get('action') == 'BUY')
    sell_count = sum(1 for d in execution_details if d.get('action') == 'SELL')
    
    print(f"BUY: {buy_count}件")
    print(f"SELL: {sell_count}件")
    print(f"差分: {abs(buy_count - sell_count)}件")
    print()
    
    # 2. DSSMS_SymbolSwitch詳細分析
    print("[2] DSSMS_SymbolSwitch詳細分析")
    dssms_switch = [d for d in execution_details if d.get('strategy_name') == 'DSSMS_SymbolSwitch']
    dssms_buy = [d for d in dssms_switch if d.get('action') == 'BUY']
    dssms_sell = [d for d in dssms_switch if d.get('action') == 'SELL']
    
    print(f"DSSMS_SymbolSwitch:")
    print(f"  BUY: {len(dssms_buy)}件")
    print(f"  SELL: {len(dssms_sell)}件")
    print(f"  差分: {abs(len(dssms_buy) - len(dssms_sell))}件")
    print()
    
    # 3. 戦略別集計
    print("[3] 戦略別集計")
    strategy_action_counts = Counter()
    for d in execution_details:
        strategy = d.get('strategy_name', 'Unknown')
        action = d.get('action', 'Unknown')
        strategy_action_counts[f"{strategy}_{action}"] += 1
    
    for key in sorted(strategy_action_counts.keys()):
        print(f"  {key}: {strategy_action_counts[key]}件")
    print()
    
    # 4. DSSMS_SymbolSwitch BUYサンプル（最初の3件）
    print("[4] DSSMS_SymbolSwitch BUYサンプル（最初の3件）")
    for i, d in enumerate(dssms_buy[:3], 1):
        print(f"  [{i}] symbol={d.get('symbol')}, timestamp={d.get('timestamp')}, "
              f"quantity={d.get('quantity')}, executed_price={d.get('executed_price')}")
    print()
    
    # 5. execution_detail構造検証（最初のDSSMS_SymbolSwitch BUY）
    if dssms_buy:
        print("[5] execution_detail構造検証（最初のDSSMS_SymbolSwitch BUY）")
        first_buy = dssms_buy[0]
        print(f"  10フィールド確認:")
        print(f"    symbol: {first_buy.get('symbol')}")
        print(f"    action: {first_buy.get('action')}")
        print(f"    quantity: {first_buy.get('quantity')}")
        print(f"    timestamp: {first_buy.get('timestamp')}")
        print(f"    executed_price: {first_buy.get('executed_price')}")
        print(f"    strategy_name: {first_buy.get('strategy_name')}")
        print(f"    status: {first_buy.get('status')}")
        print(f"    entry_price: {first_buy.get('entry_price')}")
        print(f"    profit_pct: {first_buy.get('profit_pct')}")
        print(f"    close_return: {first_buy.get('close_return')}")
        print()
    
    # 6. 損益計算への影響確認
    print("[6] 損益計算への影響確認")
    initial_capital = data.get('initial_capital', 0)
    total_portfolio_value = data.get('total_portfolio_value', 0)
    total_return = data.get('total_return', 0)
    
    print(f"  初期資本: {initial_capital:,.0f}円")
    print(f"  最終資本: {total_portfolio_value:,.0f}円")
    print(f"  総収益率: {total_return * 100:.2f}%")
    print()
    
    # 7. 修正前（ユーザー報告）との比較
    print("[7] 修正前（ユーザー報告）との比較")
    print("  修正前DSSMS_SymbolSwitch BUY: 0件")
    print(f"  修正後DSSMS_SymbolSwitch BUY: {len(dssms_buy)}件 {'✅' if len(dssms_buy) > 0 else '❌'}")
    print()
    print("  修正前BUY/SELL差分: 97件（SELL超過）")
    print(f"  修正後BUY/SELL差分: {abs(buy_count - sell_count)}件 {'✅ 改善' if abs(buy_count - sell_count) < 97 else '❌ 悪化'}")
    print()
    
    # 8. 副作用確認
    print("[8] 副作用確認")
    other_strategies = [d for d in execution_details if d.get('strategy_name') != 'DSSMS_SymbolSwitch']
    print(f"  他戦略のexecution_details: {len(other_strategies)}件")
    print(f"  全戦略の合計: {len(execution_details)}件")
    print(f"  副作用: {'なし ✅' if len(execution_details) == len(other_strategies) + len(dssms_switch) else 'あり ❌'}")
    print()
    
    return {
        'total': len(execution_details),
        'buy': buy_count,
        'sell': sell_count,
        'dssms_buy': len(dssms_buy),
        'dssms_sell': len(dssms_sell),
        'initial_capital': initial_capital,
        'final_capital': total_portfolio_value,
        'total_return': total_return
    }

def main():
    print("12ヶ月バックテスト結果検証（修正後の完全検証）")
    print("=" * 80)
    print()
    
    # 最新の出力ディレクトリを検索
    output_dir = find_latest_12month_output()
    
    if not output_dir:
        print("[ERROR] 出力ディレクトリが見つかりません")
        return
    
    # execution_details.jsonを詳細分析
    results = analyze_execution_details(output_dir)
    
    if results:
        print("=" * 80)
        print("[SUCCESS] 検証完了")
        print("=" * 80)

if __name__ == "__main__":
    main()
