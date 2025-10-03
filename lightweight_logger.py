"""
軽量Logger設定システム - config/__init__.pyの重いインポートを回避
TODO-PERF-006: Phase 4 Logger設定最適化 - Stage 2

根本原因: config/__init__.pyのimport *が1755個のモジュールを読み込み
解決策: 直接パスインポートによる軽量Logger設定
"""

import logging
import sys
import os
import time
import importlib.util

class LightweightLoggerManager:
    """軽量Logger管理クラス - 重いconfigインポートを回避"""
    
    def __init__(self):
        self._logger_cache = {}
        self._setup_time_cache = {}
    
    def setup_logger_lightweight(self, name: str, level=logging.INFO, log_file: str = None) -> logging.Logger:
        """
        軽量版Logger設定 - configパッケージを経由せずに直接ロード
        
        Args:
            name: ロガー名
            level: ログレベル
            log_file: ログファイルパス（オプション）
            
        Returns:
            logging.Logger: 設定済みロガー
        """
        start_time = time.perf_counter()
        
        # キャッシュチェック
        cache_key = f"{name}_{level}_{log_file}"
        if cache_key in self._logger_cache:
            logger = self._logger_cache[cache_key]
            cache_time = (time.perf_counter() - start_time) * 1000
            self._setup_time_cache[cache_key] = cache_time
            return logger
        
        # Logger設定（元のsetup_logger関数と同じロジック）
        logger = logging.getLogger(name)
        logger.setLevel(level)

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')

        # 標準出力用ハンドラーを追加（重複防止）
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        # log_fileが指定されている場合はFileHandlerを追加（重複防止）
        if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            # ログディレクトリが存在しない場合は作成
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # キャッシュに保存
        self._logger_cache[cache_key] = logger
        setup_time = (time.perf_counter() - start_time) * 1000
        self._setup_time_cache[cache_key] = setup_time
        
        return logger
    
    def setup_logger_direct_import(self, name: str, level=logging.INFO, log_file: str = None) -> logging.Logger:
        """
        直接ファイルインポート版 - importlib.utilを使用して直接ロード
        config/__init__.pyを完全に回避
        """
        start_time = time.perf_counter()
        
        # 直接ファイルパスでlogger_config.pyを読み込み
        logger_config_path = os.path.join("config", "logger_config.py")
        
        if os.path.exists(logger_config_path):
            # 直接ファイルからモジュールをロード
            spec = importlib.util.spec_from_file_location("logger_config_direct", logger_config_path)
            logger_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(logger_config_module)
            
            # 元のsetup_logger関数を呼び出し
            logger = logger_config_module.setup_logger(name, level, log_file)
        else:
            # fallback: 軽量版を使用
            logger = self.setup_logger_lightweight(name, level, log_file)
        
        setup_time = (time.perf_counter() - start_time) * 1000
        cache_key = f"direct_{name}_{level}_{log_file}"
        self._setup_time_cache[cache_key] = setup_time
        
        return logger
    
    def get_performance_stats(self):
        """パフォーマンス統計を取得"""
        return {
            'cache_count': len(self._logger_cache),
            'setup_times': self._setup_time_cache.copy(),
            'average_time': sum(self._setup_time_cache.values()) / len(self._setup_time_cache) if self._setup_time_cache else 0
        }

# グローバルインスタンス
_lightweight_logger_manager = LightweightLoggerManager()

def setup_logger_fast(name: str, level=logging.INFO, log_file: str = None) -> logging.Logger:
    """
    高速Logger設定関数 - 公開API
    config.logger_config.setup_loggerの軽量版代替
    
    使用例:
        from lightweight_logger import setup_logger_fast
        logger = setup_logger_fast('my_app')
    """
    return _lightweight_logger_manager.setup_logger_lightweight(name, level, log_file)

def setup_logger_ultra_fast(name: str, level=logging.INFO, log_file: str = None) -> logging.Logger:
    """
    超高速Logger設定関数 - 直接インポート版
    最大限の軽量化を実現
    """
    return _lightweight_logger_manager.setup_logger_direct_import(name, level, log_file)

def benchmark_logger_performance():
    """Logger設定パフォーマンスベンチマーク"""
    print("=" * 60)
    print("Logger設定パフォーマンスベンチマーク")
    print("=" * 60)
    
    # 1. 従来のconfig.logger_configインポート
    print("\n1. 従来方式: config.logger_config.setup_logger")
    start = time.perf_counter()
    try:
        from config.logger_config import setup_logger
        original_time = (time.perf_counter() - start) * 1000
        print(f"   インポート時間: {original_time:.1f}ms")
        
        # 実行時間測定
        start = time.perf_counter()
        logger1 = setup_logger('benchmark_original')
        exec_time = (time.perf_counter() - start) * 1000
        print(f"   実行時間: {exec_time:.1f}ms")
        print(f"   合計時間: {original_time + exec_time:.1f}ms")
        
    except Exception as e:
        print(f"   エラー: {e}")
        original_time = float('inf')
    
    # 2. 軽量版
    print("\n2. 軽量版: setup_logger_fast")
    start = time.perf_counter()
    logger2 = setup_logger_fast('benchmark_lightweight')
    lightweight_time = (time.perf_counter() - start) * 1000
    print(f"   実行時間: {lightweight_time:.1f}ms")
    
    # 3. 超軽量版
    print("\n3. 超軽量版: setup_logger_ultra_fast")
    start = time.perf_counter()
    logger3 = setup_logger_ultra_fast('benchmark_ultra')
    ultra_time = (time.perf_counter() - start) * 1000
    print(f"   実行時間: {ultra_time:.1f}ms")
    
    # 4. パフォーマンス比較
    print(f"\n4. パフォーマンス比較:")
    if original_time != float('inf'):
        improvement_fast = ((original_time - lightweight_time) / original_time) * 100
        improvement_ultra = ((original_time - ultra_time) / original_time) * 100
        print(f"   軽量版改善率: {improvement_fast:.1f}% ({original_time:.1f}ms → {lightweight_time:.1f}ms)")
        print(f"   超軽量版改善率: {improvement_ultra:.1f}% ({original_time:.1f}ms → {ultra_time:.1f}ms)")
    else:
        print(f"   軽量版: {lightweight_time:.1f}ms")
        print(f"   超軽量版: {ultra_time:.1f}ms")
    
    # 5. 機能確認
    print(f"\n5. 機能確認:")
    try:
        logger2.info("軽量版ログテスト")
        logger3.info("超軽量版ログテスト")
        print("   ✅ 両方とも正常にログ出力")
    except Exception as e:
        print(f"   ❌ ログ出力エラー: {e}")
    
    # 6. 統計情報
    stats = _lightweight_logger_manager.get_performance_stats()
    print(f"\n6. 統計情報:")
    print(f"   キャッシュ数: {stats['cache_count']}")
    print(f"   平均実行時間: {stats['average_time']:.1f}ms")
    
    return {
        'original_time': original_time,
        'lightweight_time': lightweight_time,
        'ultra_time': ultra_time,
        'stats': stats
    }

if __name__ == "__main__":
    results = benchmark_logger_performance()
    print(f"\\nベンチマーク結果: {results}")