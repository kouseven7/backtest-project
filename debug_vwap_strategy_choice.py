"""
VWAP Breakout Strategy強制選択テスト

main_new.pyでVWAPBreakoutStrategyが成功していたことを踏まえ、
DSSMSでもVWAPBreakoutStrategyが選択されるようにスコアを調整するテスト
"""

import pandas as pd
from src.dssms.dssms_integrated_main import DSSMSIntegrated

def test_vwap_strategy_selection():
    """
    VWAPBreakoutStrategyが選択されるようにスコア調整
    """
    
    # DSSMS設定
    config = {
        'start_date': '2025-01-15',
        'end_date': '2025-01-17',
        'initial_capital': 1000000,
        'force_strategy': 'VWAPBreakoutStrategy'  # 強制選択
    }
    
    print("=== VWAPBreakoutStrategy強制選択テスト ===")
    print(f"期間: {config['start_date']} -> {config['end_date']}")
    print(f"強制戦略: {config['force_strategy']}")
    
    # DSSMS実行
    dssms = DSSMSIntegrated()
    
    # 戦略選択を無視してVWAPBreakoutを強制使用
    # ※ このテストは概念実装のため実際には動作しない可能性があります
    
    print("\n=== テスト完了 ===")
    print("実際の修正が必要: DynamicStrategySelector内でスコア調整")
    print("または: config設定でVWAPBreakout優先設定")

if __name__ == '__main__':
    test_vwap_strategy_selection()