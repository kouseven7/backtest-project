"""
Logger設定のボトルネック詳細分析スクリプト
TODO-PERF-006: Phase 4 Logger設定最適化 - Stage 1
"""

import time
import sys
import os

def measure_time(func, *args, **kwargs):
    """関数の実行時間を測定"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, (end - start) * 1000

def analyze_logger_bottleneck():
    """Logger設定のボトルネック詳細分析"""
    print("=" * 60)
    print("Logger設定ボトルネック詳細分析開始")
    print("=" * 60)
    
    # 1. インポート段階の時間測定
    print("\n1. インポート段階の時間測定:")
    
    print("  1-1. logging モジュールインポート:")
    start = time.perf_counter()
    import logging
    logging_import_time = (time.perf_counter() - start) * 1000
    print(f"       時間: {logging_import_time:.1f}ms")
    
    print("  1-2. sys モジュールインポート:")
    start = time.perf_counter()
    import sys
    sys_import_time = (time.perf_counter() - start) * 1000
    print(f"       時間: {sys_import_time:.1f}ms")
    
    print("  1-3. os モジュールインポート:")
    start = time.perf_counter()
    import os
    os_import_time = (time.perf_counter() - start) * 1000
    print(f"       時間: {os_import_time:.1f}ms")
    
    # 2. setup_logger関数内部の各ステップ測定
    print("\n2. setup_logger関数内部の詳細分析:")
    
    def setup_logger_analyzed(name: str, level=logging.INFO, log_file: str = None):
        """分析用のsetup_logger（各ステップの時間測定付き）"""
        step_times = {}
        
        # ステップ1: ロガー取得・レベル設定
        start = time.perf_counter()
        logger = logging.getLogger(name)
        logger.setLevel(level)
        step_times['logger_setup'] = (time.perf_counter() - start) * 1000
        
        # ステップ2: フォーマッタ作成
        start = time.perf_counter()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        step_times['formatter_creation'] = (time.perf_counter() - start) * 1000
        
        # ステップ3: 標準出力ハンドラーチェック・追加
        start = time.perf_counter()
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        step_times['stream_handler'] = (time.perf_counter() - start) * 1000
        
        # ステップ4: ファイルハンドラー処理（必要時）
        start = time.perf_counter()
        if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        step_times['file_handler'] = (time.perf_counter() - start) * 1000
        
        return logger, step_times
    
    # 複数パターンでsetup_logger実行
    test_cases = [
        ("test_logger_1", logging.INFO, None),
        ("test_logger_2", logging.DEBUG, "logs/test.log"),
        ("test_logger_3", logging.WARNING, None),
    ]
    
    for i, (name, level, log_file) in enumerate(test_cases, 1):
        print(f"\n  2-{i}. テストケース {i}: {name}, {level}, {log_file}")
        
        start = time.perf_counter()
        logger, step_times = setup_logger_analyzed(name, level, log_file)
        total_time = (time.perf_counter() - start) * 1000
        
        print(f"       総実行時間: {total_time:.1f}ms")
        for step_name, step_time in step_times.items():
            print(f"         {step_name}: {step_time:.1f}ms")
    
    # 3. 外部要因の調査
    print("\n3. 外部要因調査:")
    
    print("  3-1. 複数回実行時の時間変動:")
    execution_times = []
    for i in range(5):
        start = time.perf_counter()
        logger = logging.getLogger(f"benchmark_{i}")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        exec_time = (time.perf_counter() - start) * 1000
        execution_times.append(exec_time)
        print(f"       実行{i+1}: {exec_time:.1f}ms")
    
    avg_time = sum(execution_times) / len(execution_times)
    print(f"       平均時間: {avg_time:.1f}ms")
    print(f"       最大時間: {max(execution_times):.1f}ms")
    print(f"       最小時間: {min(execution_times):.1f}ms")
    
    # 4. システム環境情報
    print("\n4. システム環境情報:")
    print(f"  Python バージョン: {sys.version}")
    print(f"  プラットフォーム: {sys.platform}")
    print(f"  現在のディレクトリ: {os.getcwd()}")
    
    # 5. 分析結果サマリー
    print("\n" + "=" * 60)
    print("分析結果サマリー")
    print("=" * 60)
    print(f"logging インポート時間: {logging_import_time:.1f}ms")
    print(f"setup_logger平均実行時間: {avg_time:.1f}ms")
    print(f"予想される7204.4msの原因:")
    if logging_import_time > 1000:
        print("  - logging モジュールインポートが異常に重い")
    if avg_time > 100:
        print("  - setup_logger関数の実行が異常に重い")
    else:
        print("  - setup_logger関数自体は正常（外部要因の可能性）")
        print("  - インポート元ファイルやモジュール依存関係を調査推奨")
    
    return {
        'logging_import_time': logging_import_time,
        'sys_import_time': sys_import_time,
        'os_import_time': os_import_time,
        'average_execution_time': avg_time,
        'execution_times': execution_times
    }

if __name__ == "__main__":
    results = analyze_logger_bottleneck()
    print(f"\n分析完了: {results}")