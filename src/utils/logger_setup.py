"""
強化ロギングシステム  
既存logger_config.pyを拡張して戦略別ログ、エラー分析、パフォーマンス監視を提供
"""

import logging
import logging.handlers
import json
import gzip
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import sys
import threading
import queue

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger


class StrategyLogFilter(logging.Filter):
    """戦略別ログフィルター"""
    
    def __init__(self, strategy_name: str):
        super().__init__()
        self.strategy_name = strategy_name
    
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, 'strategy_name', None) == self.strategy_name


class ErrorAnalysisFilter(logging.Filter):
    """エラー分析フィルター"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


class PerformanceLogFormatter(logging.Formatter):
    """パフォーマンス用フォーマッター"""
    
    def format(self, record: logging.LogRecord) -> str:
        # 基本フォーマット
        result = super().format(record)
        
        # パフォーマンス情報追加
        if hasattr(record, 'execution_time'):
            result += f" | 実行時間: {record.execution_time:.4f}秒"
        
        if hasattr(record, 'memory_usage'):
            result += f" | メモリ使用量: {record.memory_usage:.2f}MB"
        
        if hasattr(record, 'strategy_name'):
            result += f" | 戦略: {record.strategy_name}"
        
        return result


class JSONLogFormatter(logging.Formatter):
    """JSON形式フォーマッター"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 追加属性
        for attr in ['strategy_name', 'execution_time', 'memory_usage', 'error_type']:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        
        # 例外情報
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """圧縮ローテーションファイルハンドラー"""
    
    def doRollover(self):
        """ファイルローテーション実行（圧縮付き）"""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(f"{self.baseFilename}.{i}.gz")
                dfn = self.rotation_filename(f"{self.baseFilename}.{i+1}.gz")
                if Path(sfn).exists():
                    if Path(dfn).exists():
                        Path(dfn).unlink()
                    Path(sfn).rename(dfn)
            
            dfn = self.rotation_filename(f"{self.baseFilename}.1.gz")
            if Path(dfn).exists():
                Path(dfn).unlink()
            
            # 現在のファイルを圧縮
            with open(self.baseFilename, 'rb') as f_in:
                with gzip.open(dfn, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            Path(self.baseFilename).unlink()
        
        if not self.delay:
            self.stream = self._open()


class EnhancedLoggerManager:
    """強化ロガー管理クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 設定読み込み
        if config_path is None:
            config_path = project_root / "config" / "error_handling" / "logging_config.json"
        
        self.config_path = Path(config_path)
        self.logging_config = self._load_logging_config()
        
        # ログディレクトリ設定
        self.log_dir = project_root / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 戦略別ロガー
        self.strategy_loggers: Dict[str, logging.Logger] = {}
        
        # エラー分析ログ
        self.error_analyzer = self._setup_error_analyzer()
        
        # パフォーマンス監視ログ
        self.performance_logger = self._setup_performance_logger()
        
        # ログ統計
        self.log_stats = {
            'total_logs': 0,
            'error_count': 0,
            'warning_count': 0,
            'strategy_logs': {},
            'recent_errors': []
        }
        
        # 非同期ログ処理
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self.log_thread.start()
        
        self._setup_enhanced_loggers()
    
    def _load_logging_config(self) -> Dict[str, Any]:
        """ロギング設定読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_logging_config()
        except Exception:
            return self._get_default_logging_config()
    
    def _get_default_logging_config(self) -> Dict[str, Any]:
        """デフォルトロギング設定"""
        return {
            "log_levels": {
                "root": "INFO",
                "strategy": "DEBUG",
                "error_analysis": "ERROR",
                "performance": "INFO"
            },
            "formats": {
                "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
                "performance": "%(asctime)s - PERF - %(message)s"
            },
            "file_rotation": {
                "max_bytes": 10485760,
                "backup_count": 5,
                "compression": True
            },
            "strategy_logging": {
                "separate_files": True,
                "include_performance": True,
                "error_tracking": True
            },
            "cleanup": {
                "max_age_days": 30,
                "auto_cleanup": True
            }
        }
    
    def _setup_enhanced_loggers(self):
        """強化ロガー設定"""
        # ルートロガー強化
        root_logger = logging.getLogger()
        
        # コンソールハンドラー（既存設定を保持）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            self.logging_config.get('formats', {}).get('standard', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        console_handler.setFormatter(console_formatter)
        
        # メインファイルハンドラー
        main_log_file = self.log_dir / "main.log"
        file_handler = CompressedRotatingFileHandler(
            main_log_file,
            maxBytes=self.logging_config.get('file_rotation', {}).get('max_bytes', 10485760),
            backupCount=self.logging_config.get('file_rotation', {}).get('backup_count', 5)
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            self.logging_config.get('formats', {}).get('detailed', '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
        )
        file_handler.setFormatter(file_formatter)
        
        # JSON ログハンドラー
        json_log_file = self.log_dir / "structured.log"
        json_handler = CompressedRotatingFileHandler(
            json_log_file,
            maxBytes=self.logging_config.get('file_rotation', {}).get('max_bytes', 10485760),
            backupCount=self.logging_config.get('file_rotation', {}).get('backup_count', 5)
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JSONLogFormatter())
        
        # ハンドラー追加（重複を避ける）
        if not root_logger.handlers:
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(json_handler)
        
        root_level = self.logging_config.get('log_levels', {}).get('root', 'INFO')
        root_logger.setLevel(getattr(logging, root_level))
    
    def _setup_error_analyzer(self) -> logging.Logger:
        """エラー分析ロガー設定"""
        error_logger = logging.getLogger('error_analysis')
        error_level = self.logging_config.get('log_levels', {}).get('error_analysis', 'ERROR')
        error_logger.setLevel(getattr(logging, error_level))
        
        # エラー専用ファイル
        error_log_file = self.log_dir / "errors.log"
        error_handler = CompressedRotatingFileHandler(
            error_log_file,
            maxBytes=self.logging_config.get('file_rotation', {}).get('max_bytes', 10485760),
            backupCount=self.logging_config.get('file_rotation', {}).get('backup_count', 5)
        )
        error_handler.addFilter(ErrorAnalysisFilter())
        error_handler.setFormatter(JSONLogFormatter())
        
        error_logger.addHandler(error_handler)
        error_logger.propagate = False
        
        return error_logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """パフォーマンス監視ロガー設定"""
        perf_logger = logging.getLogger('performance')
        perf_level = self.logging_config.get('log_levels', {}).get('performance', 'INFO')
        perf_logger.setLevel(getattr(logging, perf_level))
        
        # パフォーマンス専用ファイル
        perf_log_file = self.log_dir / "performance.log"
        perf_handler = CompressedRotatingFileHandler(
            perf_log_file,
            maxBytes=self.logging_config.get('file_rotation', {}).get('max_bytes', 10485760),
            backupCount=self.logging_config.get('file_rotation', {}).get('backup_count', 5)
        )
        perf_handler.setFormatter(PerformanceLogFormatter())
        
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False
        
        return perf_logger
    
    def get_strategy_logger(self, strategy_name: str) -> logging.Logger:
        """戦略別ロガー取得"""
        if strategy_name not in self.strategy_loggers:
            self.strategy_loggers[strategy_name] = self._create_strategy_logger(strategy_name)
        
        return self.strategy_loggers[strategy_name]
    
    def _create_strategy_logger(self, strategy_name: str) -> logging.Logger:
        """戦略別ロガー作成"""
        logger = logging.getLogger(f'strategy.{strategy_name}')
        strategy_level = self.logging_config.get('log_levels', {}).get('strategy', 'DEBUG')
        logger.setLevel(getattr(logging, strategy_level))
        
        if self.logging_config.get('strategy_logging', {}).get('separate_files', True):
            # 戦略別ファイル
            strategy_log_file = self.log_dir / f"strategy_{strategy_name}.log"
            strategy_handler = CompressedRotatingFileHandler(
                strategy_log_file,
                maxBytes=self.logging_config.get('file_rotation', {}).get('max_bytes', 10485760),
                backupCount=self.logging_config.get('file_rotation', {}).get('backup_count', 5)
            )
            strategy_handler.addFilter(StrategyLogFilter(strategy_name))
            strategy_handler.setFormatter(PerformanceLogFormatter())
            
            logger.addHandler(strategy_handler)
        
        logger.propagate = True
        
        # 統計初期化
        self.log_stats['strategy_logs'][strategy_name] = {
            'total_logs': 0,
            'errors': 0,
            'warnings': 0,
            'last_activity': datetime.now().isoformat()
        }
        
        return logger
    
    def log_strategy_execution(self, strategy_name: str, message: str, 
                              execution_time: Optional[float] = None,
                              memory_usage: Optional[float] = None,
                              level: int = logging.INFO):
        """戦略実行ログ"""
        logger = self.get_strategy_logger(strategy_name)
        
        # 追加属性設定
        extra = {'strategy_name': strategy_name}
        if execution_time is not None:
            extra['execution_time'] = execution_time
        if memory_usage is not None:
            extra['memory_usage'] = memory_usage
        
        logger.log(level, message, extra=extra)
        
        # 統計更新
        self._update_strategy_stats(strategy_name, level)
    
    def log_error_with_analysis(self, error: Exception, context: Dict[str, Any],
                               strategy_name: Optional[str] = None):
        """エラー分析ログ"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'strategy_name': strategy_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # エラー分析ロガーに記録
        self.error_analyzer.error(
            f"エラー分析: {error_info['error_type']}",
            extra=error_info
        )
        
        # 最近のエラー記録
        self.log_stats['recent_errors'].append(error_info)
        if len(self.log_stats['recent_errors']) > 100:
            self.log_stats['recent_errors'].pop(0)
        
        self.log_stats['error_count'] += 1
    
    def log_performance_metric(self, metric_name: str, value: float,
                              unit: str = "", context: Optional[Dict[str, Any]] = None):
        """パフォーマンスメトリクスログ"""
        message = f"{metric_name}: {value}{unit}"
        if context:
            message += f" | コンテキスト: {context}"
        
        extra = {
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
            'context': context or {}
        }
        
        self.performance_logger.info(message, extra=extra)
    
    def _update_strategy_stats(self, strategy_name: str, level: int):
        """戦略統計更新"""
        stats = self.log_stats['strategy_logs'][strategy_name]
        stats['total_logs'] += 1
        stats['last_activity'] = datetime.now().isoformat()
        
        if level >= logging.ERROR:
            stats['errors'] += 1
        elif level >= logging.WARNING:
            stats['warnings'] += 1
        
        self.log_stats['total_logs'] += 1
        
        if level >= logging.ERROR:
            self.log_stats['error_count'] += 1
        elif level >= logging.WARNING:
            self.log_stats['warning_count'] += 1
    
    def _process_log_queue(self):
        """非同期ログ処理"""
        while True:
            try:
                log_record = self.log_queue.get(timeout=1)
                if log_record is None:
                    break
                
                # ログ処理
                logger = logging.getLogger(log_record['logger_name'])
                logger.handle(log_record['record'])
                
            except queue.Empty:
                continue
            except Exception:
                # ログ処理エラーは無視
                pass
    
    def cleanup_old_logs(self, max_age_days: int = 30):
        """古いログファイル削除"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for log_file in self.log_dir.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logging.info(f"古いログファイル削除: {log_file}")
            except Exception as e:
                logging.warning(f"ログファイル削除失敗 {log_file}: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """ログ統計取得"""
        return self.log_stats.copy()
    
    def create_log_report(self, output_path: Optional[str] = None) -> str:
        """ログレポート生成"""
        if output_path is None:
            output_path = self.log_dir / f"log_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_path = Path(output_path)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_log_statistics(),
            'configuration': self.logging_config,
            'log_files': [str(f) for f in self.log_dir.glob("*.log")]
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logging.info(f"ログレポート生成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"ログレポート生成失敗: {e}")
            return ""


# グローバルインスタンス
_global_logger_manager: Optional[EnhancedLoggerManager] = None


def get_logger_manager() -> EnhancedLoggerManager:
    """グローバルロガー管理インスタンス取得"""
    global _global_logger_manager
    if _global_logger_manager is None:
        _global_logger_manager = EnhancedLoggerManager()
    return _global_logger_manager


def get_strategy_logger(strategy_name: str) -> logging.Logger:
    """戦略別ロガー取得（グローバル関数）"""
    return get_logger_manager().get_strategy_logger(strategy_name)


def log_performance(metric_name: str, value: float, unit: str = "", 
                   context: Optional[Dict[str, Any]] = None):
    """パフォーマンスログ（グローバル関数）"""
    get_logger_manager().log_performance_metric(metric_name, value, unit, context)


def log_strategy_performance(strategy_name: str, execution_time: float,
                            memory_usage: Optional[float] = None,
                            additional_info: Optional[str] = None):
    """戦略パフォーマンスログ（グローバル関数）"""
    message = f"戦略実行完了"
    if additional_info:
        message += f": {additional_info}"
    
    get_logger_manager().log_strategy_execution(
        strategy_name, message, execution_time, memory_usage
    )
