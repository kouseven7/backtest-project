
"""
DSSMS切替最適化の確認テスト
修正後の切替頻度を検証
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.dssms.dssms_backtester import DSSMSBacktester

def test_switch_optimization():
    """切替最適化テスト"""
    
    print("=== DSSMS切替最適化テスト開始 ===")
    
    # 修正版設定
    config = {
        'initial_capital': 1000000,
        'switch_cost_rate': 0.002,  # 0.2%
        'output_excel': False,
        'output_detailed_report': True
    }
    
    backtester = DSSMSBacktester(config)
    
    # 3ヶ月間のテスト
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2024, 8, 31)
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
            
            print(f"\n[CHART] 修正後の結果:")
            print(f"切替回数: {switch_count}回 (3ヶ月)")
            print(f"平均保有期間: {avg_holding:.1f}時間")
            print(f"総リターン: {total_return:.2%}")
            
            # 改善判定
            yearly_switches = switch_count * 4  # 年間換算
            
            if yearly_switches < 100:
                print("[OK] 切替頻度: 良好（年間100回未満）")
            elif yearly_switches < 200:
                print("[WARNING] 切替頻度: 改善余地あり")
            else:
                print("[ERROR] 切替頻度: まだ高すぎる")
            
            if avg_holding > 48:
                print("[OK] 保有期間: 良好（48時間以上）")
            else:
                print("[WARNING] 保有期間: 短い（さらに改善推奨）")
            
            return True
        else:
            print(f"[ERROR] テスト失敗: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
        return False

if __name__ == "__main__":
    success = test_switch_optimization()
    if success:
        print("\n[SUCCESS] 修正テスト完了")
    else:
        print("\n[ERROR] 修正が必要です")
