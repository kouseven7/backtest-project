#!/usr/bin/env python3
"""
DSSMS切替制御の診断スクリプト
実際の切替ロジックが動作しているかを詳細に確認
"""

import sys
import logging
from datetime import datetime, timedelta
sys.path.append('.')

from src.dssms.dssms_backtester import DSSMSBacktester

def debug_switch_control():
    """切替制御の動作状況を詳細診断"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== DSSMS切替制御診断開始 ===")
    
    # バックテスター初期化
    backtester = DSSMSBacktester()
    
    # 短期間でのテスト
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 15)  # 2週間のみ
    
    print(f"テスト期間: {start_date.date()} - {end_date.date()}")
    print(f"初期資本: {backtester.initial_capital:,}円")
    print(f"切替コスト率: {backtester.switch_cost_rate:.4f}")
    
    # 実行前の状態確認
    print(f"\n実行前の切替履歴: {len(backtester.switch_history)}件")
    
    # テスト実行
    try:
        # simulate_dynamic_selectionメソッドを使用
        symbol_universe = ['4063', '8306', '6861', '7741', '9432', '8058', '9020', '7203', '9984', '6758']
        result = backtester.simulate_dynamic_selection(start_date, end_date, symbol_universe)
        
        # 結果分析
        print(f"\n=== 実行結果 ===")
        print(f"最終ポートフォリオ価値: {result.get('final_portfolio_value', 0):,}円")
        print(f"総切替回数: {len(backtester.switch_history)}")
        print(f"総リターン: {result.get('total_return', 0):.2f}%")
        
        # 切替履歴の詳細分析
        print(f"\n=== 切替履歴詳細 ===")
        if backtester.switch_history:
            for i, switch in enumerate(backtester.switch_history[:10]):  # 最初の10件
                print(f"{i+1:2d}: {switch.timestamp.date()} {switch.from_symbol} -> {switch.to_symbol}")
                print(f"    理由: {switch.reason}")
                print(f"    保有期間: {switch.holding_period_hours:.1f}時間")
                if i < len(backtester.switch_history) - 1:
                    next_switch = backtester.switch_history[i+1]
                    interval = (next_switch.timestamp - switch.timestamp).total_seconds() / 3600
                    print(f"    次切替まで: {interval:.1f}時間")
                print()
        
        # 保有期間統計
        if backtester.switch_history:
            holding_periods = [s.holding_period_hours for s in backtester.switch_history]
            avg_holding = sum(holding_periods) / len(holding_periods)
            min_holding = min(holding_periods)
            max_holding = max(holding_periods)
            
            print(f"=== 保有期間統計 ===")
            print(f"平均保有期間: {avg_holding:.1f}時間")
            print(f"最短保有期間: {min_holding:.1f}時間")
            print(f"最長保有期間: {max_holding:.1f}時間")
            
            # 24時間未満の切替を確認
            short_holdings = [h for h in holding_periods if h < 24.0]
            print(f"24時間未満の切替: {len(short_holdings)}件 ({len(short_holdings)/len(holding_periods)*100:.1f}%)")
        
        # 切替制御が効いているかの判定
        print(f"\n=== 制御効果診断 ===")
        if len(backtester.switch_history) > 50:
            print("[ERROR] 切替が多すぎます（制御が効いていない可能性）")
        elif len(backtester.switch_history) < 20:
            print("[OK] 切替が適度に抑制されています")
        else:
            print("[WARNING]  切替回数は中程度です")
            
        if backtester.switch_history:
            avg_holding = sum([s.holding_period_hours for s in backtester.switch_history]) / len(backtester.switch_history)
            if avg_holding < 48:
                print("[ERROR] 平均保有期間が短すぎます")
            else:
                print("[OK] 平均保有期間は適切です")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 診断完了 ===")

if __name__ == "__main__":
    debug_switch_control()
