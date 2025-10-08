"""
DSSMS IntegratedBacktester インポート問題緊急調査
2854ms逆戻り問題の原因特定

作成: 2025年10月2日
"""

import time
import os
import sys
import importlib.util
from pathlib import Path

def debug_import_chain():
    """インポートチェーンの詳細デバッグ"""
    print("=== DSSMS インポート問題緊急調査 ===")
    
    # 1. UltraLight版の直接ロード試験
    print("1. SymbolSwitchManagerUltraLight直接ロード試験")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dssms_dir = os.path.join(current_dir, "src", "dssms")
    ultra_light_path = os.path.join(src_dssms_dir, "symbol_switch_manager_ultra_light.py")
    
    print(f"   探索ディレクトリ: {current_dir}")
    print(f"   src/dssms 予想パス: {src_dssms_dir}")
    print(f"   UltraLight予想パス: {ultra_light_path}")
    print(f"   UltraLightファイル存在: {os.path.exists(ultra_light_path)}")
    
    if os.path.exists(ultra_light_path):
        try:
            start = time.perf_counter()
            spec = importlib.util.spec_from_file_location("symbol_switch_manager_ultra_light", ultra_light_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                ultra_light_time = (time.perf_counter() - start) * 1000
                print(f"   [OK] UltraLight直接ロード成功: {ultra_light_time:.1f}ms")
                
                # クラス確認
                if hasattr(module, 'SymbolSwitchManagerUltraLight'):
                    print(f"   [OK] SymbolSwitchManagerUltraLightクラス存在")
                    return module.SymbolSwitchManagerUltraLight, ultra_light_time
                else:
                    print(f"   [ERROR] SymbolSwitchManagerUltraLightクラス不存在")
                    return None, ultra_light_time
            else:
                print(f"   [ERROR] specまたはloader作成失敗")
                return None, 0
        except Exception as e:
            print(f"   [ERROR] UltraLight直接ロードエラー: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    else:
        print("   [ERROR] UltraLightファイル不存在")
        return None, 0

def debug_fallback_import():
    """フォールバック版インポートのタイミング測定"""
    print("\n2. フォールバック版インポート時間測定")
    
    try:
        start = time.perf_counter()
        from src.dssms.symbol_switch_manager import SymbolSwitchManager
        fallback_time = (time.perf_counter() - start) * 1000
        print(f"   通常版SymbolSwitchManagerインポート: {fallback_time:.1f}ms")
        return SymbolSwitchManager, fallback_time
    except ImportError as e:
        print(f"   [ERROR] 通常版インポートエラー: {e}")
        return None, 0

def debug_dssms_integrated_import():
    """DSSMSIntegratedMain全体のインポート分析"""
    print("\n3. DSSMSIntegratedBacktester全体インポート分析")
    
    # 詳細な段階別インポート時間測定
    components = []
    
    start_total = time.perf_counter()
    
    # 基本インポート
    start = time.perf_counter()
    import sys
    import os
    from datetime import datetime, timedelta
    from typing import Dict, List, Any, Optional, Tuple
    import logging
    import time as time_module
    from pathlib import Path
    import json
    import argparse
    basic_time = (time.perf_counter() - start) * 1000
    components.append(("基本インポート", basic_time))
    
    # importlib関連
    start = time.perf_counter()
    import importlib.util
    importlib_time = (time.perf_counter() - start) * 1000
    components.append(("importlib.util", importlib_time))
    
    # UltraLight版ロード試験
    ultra_light_class, ultra_light_time = debug_import_chain()
    if ultra_light_class:
        components.append(("UltraLight版ロード", ultra_light_time))
    
    # フォールバック版ロード試験（UltraLight失敗時のシミュレーション）
    if not ultra_light_class:
        fallback_class, fallback_time = debug_fallback_import()
        if fallback_class:
            components.append(("フォールバック版ロード", fallback_time))
    
    total_time = (time.perf_counter() - start_total) * 1000
    components.append(("総時間", total_time))
    
    # 結果表示
    print("\n[CHART] インポート時間詳細:")
    for component, time_ms in components:
        percentage = (time_ms / total_time * 100) if total_time > 0 else 0
        print(f"   {component}: {time_ms:.1f}ms ({percentage:.1f}%)")
    
    return components

def main():
    """メイン実行"""
    try:
        components = debug_dssms_integrated_import()
        
        print("\n=== 問題分析結果 ===")
        
        # 最大ボトルネック特定
        max_component = max(components[:-1], key=lambda x: x[1])  # 総時間除く
        print(f"[FIRE] 最大ボトルネック: {max_component[0]} ({max_component[1]:.1f}ms)")
        
        # 2854ms問題の原因推定
        total_time = components[-1][1]
        print(f"[CHART] 現在の総インポート時間: {total_time:.1f}ms")
        
        if total_time > 2000:
            print("[WARNING] 2000ms超過 - 重いフォールバック版が使用されている可能性")
            print("[TOOL] 推奨対応: UltraLight版ロード機構の修正")
        elif total_time > 100:
            print("[WARNING] 100ms超過 - 部分的最適化が必要")
            print("[TOOL] 推奨対応: 個別コンポーネント最適化")
        else:
            print("[OK] インポート時間正常範囲")
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()