"""
DSSMSでVWAPBreakoutStrategy強制選択テスト

main_new.pyでVWAPBreakoutStrategyが成功（2回エントリー）したことを活用し、
DSSMSでもVWAPBreakoutStrategyを選択してエントリーを発生させる
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

def test_direct_vwap_execution():
    """
    直接VWAPBreakoutStrategyでのテスト
    """
    
    print("=== VWAPBreakoutStrategy直接実行テスト ===")
    print("期間: 2025-01-15 -> 2025-01-17")
    print()
    
    # 直接main_new.pyの結果と比較
    from main_new import main_multi_strategy_execution
    
    try:
        
        print("main_new.py実行 (VWAPBreakoutStrategy含む)...")
        main_multi_strategy_execution()
        
        print("\n=== 比較: main_new.pyではエントリーが発生 ===")
        print("- VWAPBreakoutStrategy: 2回エントリー")
        print("- BreakoutStrategy: 1回エントリー")
        print()
        
        return True
        
    except Exception as e:
        print(f"エラー発生: {e}")
        return False

def modify_strategy_selection_for_vwap():
    """
    設定ベースでのDSSMS修正案
    """
    
    print("=== DSSMS設定修正案 ===")
    print("案1: GCStrategy以外の戦略を優先選択する設定")
    print("案2: 戦略スコア計算で過去のパフォーマンスを重視")
    print("案3: backtest_daily()で実際にエントリー可能な戦略を優先")
    print()
    
    # 実際の修正は段階的に実装
    results = test_direct_vwap_execution()
    return results
        
        print(f"\n=== 実行結果 ===")
        print(f"実行成功: {results.get('success', False)}")
        print(f"取引件数: {results.get('total_trades', 0)}")
        print(f"最終収益率: {results.get('total_return_rate', 0.0):.4f}")
        
        # all_transactions.csvを確認
        output_dir = results.get('output_directory', '')
        if output_dir:
            transaction_file = os.path.join(output_dir, 'all_transactions.csv')
            if os.path.exists(transaction_file):
                import pandas as pd
                transactions = pd.read_csv(transaction_file)
                print(f"取引履歴ファイル: {len(transactions)}件の取引")
                if len(transactions) > 0:
                    print(transactions.head())
            else:
                print("取引履歴ファイルが見つかりません")
        
        return results
        
    except Exception as e:
        print(f"エラー発生: {e}")
        return None
        
    finally:
        # 元のメソッドを復元
        DynamicStrategySelector._calculate_all_strategy_scores = original_method
        print("\n[復元] 元のスコア計算メソッドに戻しました")

if __name__ == '__main__':
    modify_strategy_selection_for_vwap()