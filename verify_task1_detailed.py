"""
Task 1実装検証スクリプト（詳細版） - SymbolSwitchManager完全版への切替確認

Author: Backtest Project Team  
Created: 2026-01-13
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("Task 1実装検証（詳細版）: SymbolSwitchManager完全版への切替")
print("="*80)

# Step 1: モジュールレベルの確認
print("\n[Step 1] モジュールレベルの確認...")
from src.dssms import dssms_integrated_main
print(f"dssms_integrated_main.SymbolSwitchManager: {dssms_integrated_main.SymbolSwitchManager}")

# Step 2: インスタンス作成前の確認
print("\n[Step 2] インスタンス作成前...")
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
import inspect

# クラス内での使用を確認
source = inspect.getsource(DSSMSIntegratedBacktester._initialize_components)
if 'SymbolSwitchManager' in source:
    print("_initialize_components内でSymbolSwitchManagerを使用")

# Step 3: インスタンス作成
print("\n[Step 3] インスタンス作成...")
try:
    bt = DSSMSIntegratedBacktester()
    print(f"インスタンス作成成功")
    print(f"bt.switch_manager: {bt.switch_manager}")
    print(f"type(bt.switch_manager): {type(bt.switch_manager)}")
    
    if bt.switch_manager is None:
        print("\n[ERROR] switch_managerがNoneです")
        print("考えられる原因:")
        print("  1. _initialize_components()で例外が発生")
        print("  2. SymbolSwitchManager(switch_config)の呼び出しが失敗")
        
        # 手動で初期化を試みる
        print("\n[DEBUG] 手動初期化を試行...")
        switch_config = bt.config.get('symbol_switch', {})
        print(f"switch_config: {switch_config}")
        
        manual_sm = dssms_integrated_main.SymbolSwitchManager(switch_config)
        print(f"手動初期化成功: {type(manual_sm)}")
        
except Exception as e:
    print(f"[ERROR] インスタンス作成失敗: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}\n")
