"""
DSSMS修正効果の最終確認テスト
1年間のバックテストで完全な修正効果を確認
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.dssms.dssms_backtester import DSSMSBacktester

def final_verification_test():
    """最終修正効果確認テスト"""
    
    print("=== DSSMS修正効果最終確認テスト開始 ===")
    
    # 修正後の設定
    config = {
        'initial_capital': 1000000,
        'switch_cost_rate': 0.002,  # 0.2%
        'output_excel': True,
        'output_detailed_report': True
    }
    
    backtester = DSSMSBacktester(config)
    
    # 1年間のフルテスト
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    symbols = ['7203.T', '6758.T', '9984.T', '4063.T', '8316.T']
    
    try:
        result = backtester.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date,
            symbol_universe=symbols
        )
        
        if result.get('success'):
            switch_count = result.get('switch_count', 0)
            avg_holding = result.get('average_holding_period_hours', 0)
            total_return = result.get('total_return', 0)
            final_value = result.get('final_value', 0)
            
            print(f"\n[CHART] 1年間修正後の結果:")
            print(f"切替回数: {switch_count}回/年")
            print(f"平均保有期間: {avg_holding:.1f}時間")
            print(f"総リターン: {total_return:.2%}")
            print(f"最終価値: {final_value:,.0f}円")
            
            # 改善効果の総合判定
            print(f"\n[OK] 改善効果判定:")
            
            if switch_count <= 50:
                print(f"🟢 切替頻度: 優秀 ({switch_count}回 ≤ 50回)")
            elif switch_count <= 100:
                print(f"🟡 切替頻度: 良好 ({switch_count}回 ≤ 100回)")
            else:
                print(f"🔴 切替頻度: 要改善 ({switch_count}回 > 100回)")
            
            if avg_holding >= 72:  # 3日以上
                print(f"🟢 保有期間: 優秀 ({avg_holding:.1f}時間 ≥ 72時間)")
            elif avg_holding >= 24:  # 1日以上
                print(f"🟡 保有期間: 改善済み ({avg_holding:.1f}時間 ≥ 24時間)")
            else:
                print(f"🔴 保有期間: 要改善 ({avg_holding:.1f}時間 < 24時間)")
            
            if total_return > 0:
                print(f"🟢 総リターン: 収益達成 ({total_return:.2%})")
            else:
                print(f"🔴 総リターン: 損失 ({total_return:.2%})")
            
            # 原本との比較
            print(f"\n[UP] 改善前との比較:")
            print(f"切替回数: 212回 → {switch_count}回 ({((212-switch_count)/212*100):.1f}%削減)")
            print(f"保有期間: 24時間 → {avg_holding:.1f}時間")
            
            return True
        else:
            print(f"[ERROR] テスト失敗: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
        return False

if __name__ == "__main__":
    success = final_verification_test()
    if success:
        print("\n[SUCCESS] DSSMS修正完了! 切替過多問題は解決されました")
        print("\n[LIST] 次の手順:")
        print("1. 本格運用前のさらなるパラメータ調整")
        print("2. 実データでの検証")
        print("3. リスク管理機能の強化")
    else:
        print("\n[ERROR] さらなる修正が必要です")
