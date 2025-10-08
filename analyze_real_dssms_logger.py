"""
実際のDSSMS環境でのLogger設定インポート分析
TODO-PERF-006: Phase 4 Logger設定最適化 - Stage 1-2
"""

import time
import sys
import os

def analyze_real_dssms_logger_import():
    """実際のDSSMS環境でのLogger設定インポート分析"""
    print("=" * 60)
    print("実際のDSSMS環境でのLogger設定インポート分析")
    print("=" * 60)
    
    # 1. 直接インポートの時間測定
    print("\n1. config.logger_config直接インポート測定:")
    
    print("  1-1. config.logger_config全体インポート:")
    start = time.perf_counter()
    try:
        import config.logger_config
        import_time_1 = (time.perf_counter() - start) * 1000
        print(f"       成功: {import_time_1:.1f}ms")
    except Exception as e:
        import_time_1 = (time.perf_counter() - start) * 1000
        print(f"       エラー: {e}, 時間: {import_time_1:.1f}ms")
    
    print("  1-2. setup_logger関数インポート:")
    start = time.perf_counter()
    try:
        from config.logger_config import setup_logger
        import_time_2 = (time.perf_counter() - start) * 1000
        print(f"       成功: {import_time_2:.1f}ms")
    except Exception as e:
        import_time_2 = (time.perf_counter() - start) * 1000
        print(f"       エラー: {e}, 時間: {import_time_2:.1f}ms")
    
    # 2. DSSMS環境での実行測定
    print("\n2. DSSMS環境での実行測定:")
    
    try:
        print("  2-1. setup_logger実行:")
        start = time.perf_counter()
        logger = setup_logger('dssms_test')
        exec_time = (time.perf_counter() - start) * 1000
        print(f"       成功: {exec_time:.1f}ms")
        
        print("  2-2. ログ出力テスト:")
        start = time.perf_counter()
        logger.info("テストログメッセージ")
        log_time = (time.perf_counter() - start) * 1000
        print(f"       ログ出力時間: {log_time:.1f}ms")
        
    except Exception as e:
        print(f"       エラー: {e}")
    
    # 3. システム依存関係調査
    print("\n3. システム依存関係調査:")
    
    print("  3-1. インポート済みモジュール数:")
    initial_modules = len(sys.modules)
    print(f"       初期モジュール数: {initial_modules}")
    
    print("  3-2. config関連モジュール:")
    config_modules = [name for name in sys.modules.keys() if 'config' in name.lower()]
    for module in sorted(config_modules):
        print(f"       - {module}")
    
    print("  3-3. logging関連モジュール:")
    logging_modules = [name for name in sys.modules.keys() if 'log' in name.lower()]
    for module in sorted(logging_modules):
        print(f"       - {module}")
    
    # 4. パス・環境情報
    print("\n4. パス・環境情報:")
    print(f"  現在のディレクトリ: {os.getcwd()}")
    print(f"  Pythonパス先頭5つ:")
    for i, path in enumerate(sys.path[:5]):
        print(f"    {i+1}. {path}")
    
    # 5. 重複インポート影響調査
    print("\n5. 重複インポート影響調査:")
    
    times = []
    for i in range(3):
        print(f"  5-{i+1}. {i+1}回目インポート:")
        start = time.perf_counter()
        try:
            # 意図的に再インポート
            import importlib
            import config.logger_config
            importlib.reload(config.logger_config) 
            from config.logger_config import setup_logger
            
            # setup_logger実行
            test_logger = setup_logger(f'test_reload_{i}')
            
            exec_time = (time.perf_counter() - start) * 1000
            times.append(exec_time)
            print(f"       時間: {exec_time:.1f}ms")
        except Exception as e:
            exec_time = (time.perf_counter() - start) * 1000
            times.append(exec_time)
            print(f"       エラー: {e}, 時間: {exec_time:.1f}ms")
    
    avg_reload_time = sum(times) / len(times) if times else 0
    print(f"  平均リロード時間: {avg_reload_time:.1f}ms")
    
    # 6. 分析結果
    print("\n" + "=" * 60)
    print("実環境分析結果")
    print("=" * 60)
    print(f"直接インポート時間: {import_time_1:.1f}ms")
    print(f"関数インポート時間: {import_time_2:.1f}ms")
    print(f"平均リロード時間: {avg_reload_time:.1f}ms")
    print(f"初期モジュール数: {initial_modules}")
    
    if max(import_time_1, import_time_2, avg_reload_time) > 1000:
        print("\n[WARNING] 異常な遅延が検出されました:")
        if import_time_1 > 1000:
            print(f"  - config.logger_config インポートが異常に重い: {import_time_1:.1f}ms")
        if import_time_2 > 1000:
            print(f"  - setup_logger インポートが異常に重い: {import_time_2:.1f}ms")
        if avg_reload_time > 1000:
            print(f"  - リロード処理が異常に重い: {avg_reload_time:.1f}ms")
    else:
        print("\n[OK] 実環境でのLogger設定は正常範囲内です")
        print("   7204.4msの異常時間は他の要因によるものと判断されます")
    
    return {
        'direct_import_time': import_time_1,
        'function_import_time': import_time_2,
        'average_reload_time': avg_reload_time,
        'initial_modules': initial_modules
    }

if __name__ == "__main__":
    results = analyze_real_dssms_logger_import()
    print(f"\n最終結果: {results}")