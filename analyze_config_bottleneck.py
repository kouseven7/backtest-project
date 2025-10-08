"""
config.logger_configインポート遅延の根本原因調査
TODO-PERF-006: Phase 4 Logger設定最適化 - Stage 1-3
"""

import time
import sys
import importlib

def analyze_config_import_bottleneck():
    """config.logger_configインポート遅延の根本原因を特定"""
    print("=" * 70)
    print("config.logger_configインポート遅延の根本原因調査")
    print("=" * 70)
    
    # 現在読み込まれているモジュール数を記録
    initial_module_count = len(sys.modules)
    
    # 1. config パッケージ自体のインポート時間
    print("\n1. configパッケージ段階的インポート分析:")
    
    print("  1-1. config パッケージインポート:")
    start = time.perf_counter()
    try:
        import config
        config_import_time = (time.perf_counter() - start) * 1000
        config_module_count = len(sys.modules)
        print(f"       成功: {config_import_time:.1f}ms")
        print(f"       追加モジュール数: {config_module_count - initial_module_count}")
    except Exception as e:
        config_import_time = (time.perf_counter() - start) * 1000
        print(f"       エラー: {e}, 時間: {config_import_time:.1f}ms")
    
    # 2. configディレクトリ内容確認
    print("\n2. configディレクトリ構造調査:")
    import os
    config_dir = "config"
    if os.path.exists(config_dir):
        files = []
        for item in os.listdir(config_dir):
            if os.path.isfile(os.path.join(config_dir, item)):
                files.append(item)
        print(f"  configディレクトリ内ファイル数: {len(files)}")
        python_files = [f for f in files if f.endswith('.py')]
        print(f"  Pythonファイル数: {len(python_files)}")
        print("  主要Pythonファイル:")
        for py_file in sorted(python_files)[:10]:  # 上位10個のみ表示
            print(f"    - {py_file}")
        if len(python_files) > 10:
            print(f"    ... 他{len(python_files) - 10}個")
    
    # 3. __init__.py の影響調査
    print("\n3. config/__init__.py の影響調査:")
    config_init_path = os.path.join("config", "__init__.py")
    if os.path.exists(config_init_path):
        print("  config/__init__.py が存在します")
        with open(config_init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"  ファイルサイズ: {len(content)} 文字")
        lines = content.split('\n')
        import_lines = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
        print(f"  インポート文の数: {len(import_lines)}")
        if import_lines:
            print("  主要インポート文（上位5個）:")
            for imp_line in import_lines[:5]:
                print(f"    {imp_line}")
    else:
        print("  config/__init__.py は存在しません")
    
    # 4. 他のconfigモジュールの影響測定
    print("\n4. 他のconfigモジュールインポート影響調査:")
    heavy_modules = []
    
    # 重そうなモジュールを個別測定
    test_imports = [
        "config.optimized_parameters",
        "config.risk_management", 
        "config.strategy_scoring_model",
        "config.portfolio_correlation_optimizer",
        "config.correlation"
    ]
    
    for module_name in test_imports:
        if module_name in sys.modules:
            print(f"  {module_name}: 既にロード済み")
            continue
            
        print(f"  {module_name}:")
        start = time.perf_counter()
        try:
            pre_count = len(sys.modules)
            importlib.import_module(module_name)
            import_time = (time.perf_counter() - start) * 1000
            post_count = len(sys.modules)
            print(f"    時間: {import_time:.1f}ms, 追加モジュール: {post_count - pre_count}")
            
            if import_time > 500:  # 500ms以上の重いモジュール
                heavy_modules.append((module_name, import_time))
                
        except Exception as e:
            import_time = (time.perf_counter() - start) * 1000
            print(f"    エラー: {e}, 時間: {import_time:.1f}ms")
    
    # 5. logger_config.py 直接読み込み（インポート以外）
    print("\n5. logger_config.py ファイル直接分析:")
    logger_config_path = os.path.join("config", "logger_config.py")
    if os.path.exists(logger_config_path):
        start = time.perf_counter()
        with open(logger_config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        read_time = (time.perf_counter() - start) * 1000
        print(f"  ファイル読み込み時間: {read_time:.1f}ms")
        print(f"  ファイルサイズ: {len(content)} 文字")
        print(f"  行数: {len(content.split('\\n'))}")
        
        # ファイル内のインポート文確認
        lines = content.split('\n')
        import_lines = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
        print(f"  ファイル内インポート文: {len(import_lines)}")
        for imp_line in import_lines:
            print(f"    {imp_line}")
    
    # 6. 分析結果とボトルネック特定
    print("\n" + "=" * 70)
    print("根本原因分析結果")
    print("=" * 70)
    
    current_modules = len(sys.modules)
    print(f"初期モジュール数: {initial_module_count}")
    print(f"現在のモジュール数: {current_modules}")
    print(f"この調査で追加されたモジュール数: {current_modules - initial_module_count}")
    
    if heavy_modules:
        print(f"\n[FIRE] 重いconfigモジュール発見:")
        for module_name, import_time in sorted(heavy_modules, key=lambda x: x[1], reverse=True):
            print(f"  - {module_name}: {import_time:.1f}ms")
    
    print(f"\n[CHART] 推定されるボトルネック:")
    if config_import_time > 1000:
        print(f"  [WARNING] config パッケージ自体が重い: {config_import_time:.1f}ms")
    if heavy_modules:
        total_heavy_time = sum(time for _, time in heavy_modules)
        print(f"  [WARNING] 重いconfigサブモジュール合計: {total_heavy_time:.1f}ms")
    if len(python_files) > 50:
        print(f"  [WARNING] configディレクトリに多数のファイル: {len(python_files)}個")
    
    return {
        'config_import_time': config_import_time,
        'heavy_modules': heavy_modules,
        'config_files_count': len(python_files) if 'python_files' in locals() else 0,
        'total_modules_added': current_modules - initial_module_count
    }

if __name__ == "__main__":
    results = analyze_config_import_bottleneck()
    print(f"\\n最終調査結果: {results}")