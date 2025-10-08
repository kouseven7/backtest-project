"""
SymbolSwitchManager段階別インポート分析
どの行で重い処理が発生しているか特定
"""

import time

def time_import_stages():
    """段階別インポート時間測定"""
    print("=== SymbolSwitchManager段階別分析 ===")
    
    # Stage 1: 基本インポート
    stage1_start = time.time()
    from datetime import datetime, timedelta
    from typing import Dict, List, Any, Optional
    import logging
    stage1_time = (time.time() - stage1_start) * 1000
    print(f"[OK] Stage 1 - 基本インポート: {stage1_time:.1f}ms")
    
    # Stage 2: モジュール内容読み込み
    stage2_start = time.time()
    import sys
    import os
    
    # ファイル内容を段階的に読み込み
    symbol_switch_file = r"src\dssms\symbol_switch_manager.py"
    
    print(f"   ファイル読み込み: {symbol_switch_file}")
    with open(symbol_switch_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    stage2_time = (time.time() - stage2_start) * 1000
    print(f"[OK] Stage 2 - ファイル読み込み: {stage2_time:.1f}ms")
    print(f"   総行数: {len(lines)}")
    
    # Stage 3: 実際のインポート
    stage3_start = time.time()
    try:
        from src.dssms.symbol_switch_manager import SymbolSwitchManager
        stage3_time = (time.time() - stage3_start) * 1000
        print(f"[OK] Stage 3 - クラスインポート: {stage3_time:.1f}ms")
        
        # Stage 4: クラス情報取得
        stage4_start = time.time()
        methods = [method for method in dir(SymbolSwitchManager) if not method.startswith('_')]
        stage4_time = (time.time() - stage4_start) * 1000
        print(f"[OK] Stage 4 - クラス情報取得: {stage4_time:.1f}ms")
        print(f"   パブリックメソッド数: {len(methods)}")
        
    except Exception as e:
        stage3_time = (time.time() - stage3_start) * 1000
        print(f"[ERROR] Stage 3 - インポートエラー: {e}")
        print(f"   時間: {stage3_time:.1f}ms")
    
    total_time = stage1_time + stage2_time + stage3_time
    print(f"\n=== 分析結果 ===")
    print(f"合計時間: {total_time:.1f}ms")
    print(f"Stage 3 (クラスインポート): {stage3_time:.1f}ms ({stage3_time/total_time*100:.1f}%)")

if __name__ == "__main__":
    time_import_stages()