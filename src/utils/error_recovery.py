"""
エラー復旧システム
自動復旧戦略、フォールバック機能、リトライ機構を提供
"""

import time
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger


class RetryStrategy:
    """リトライ戦略の基底クラス"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = setup_logger(__name__)
    
    def calculate_delay(self, attempt: int) -> float:
        """遅延時間計算"""
        return self.base_delay
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """リトライ判定"""
        return attempt < self.max_retries


class SimpleRetryStrategy(RetryStrategy):
    """単純リトライ戦略"""
    
    def calculate_delay(self, attempt: int) -> float:
        return self.base_delay


class ExponentialBackoffStrategy(RetryStrategy):
    """指数バックオフ戦略"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, jitter: bool = True):
        super().__init__(max_retries, base_delay)
        self.max_delay = max_delay
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay += random.uniform(0, delay * 0.1)
        
        return delay


class LinearBackoffStrategy(RetryStrategy):
    """線形バックオフ戦略"""
    
    def calculate_delay(self, attempt: int) -> float:
        return self.base_delay * (attempt + 1)


class FallbackStrategy:
    """フォールバック戦略"""
    
    def __init__(self, fallback_functions: List[Callable[[], Any]]):
        self.fallback_functions = fallback_functions
        self.logger = setup_logger(__name__)
    
    def execute_fallback(self, original_error: Exception) -> Any:
        """フォールバック実行"""
        for i, fallback_func in enumerate(self.fallback_functions):
            try:
                self.logger.info(f"フォールバック実行 ({i+1}/{len(self.fallback_functions)})")
                result = fallback_func()
                self.logger.info(f"フォールバック成功 ({i+1})")
                return result
            except Exception as e:
                self.logger.warning(f"フォールバック失敗 ({i+1}): {e}")
                if i == len(self.fallback_functions) - 1:
                    raise e
        return None


class ErrorRecoveryManager:
    """エラー復旧管理クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        
        # 復旧戦略設定読み込み
        if config_path is None:
            config_path = project_root / "config" / "error_handling" / "recovery_strategies.json"
        
        self.config_path = Path(config_path)
        self.recovery_config = self._load_recovery_config()
        
        # 復旧戦略インスタンス
        self._strategy_instances: Dict[str, RetryStrategy] = {}
        self._fallback_strategies: Dict[str, FallbackStrategy] = {}
        
        # 復旧統計
        self.recovery_stats = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'strategy_usage': {},
            'average_recovery_time': 0.0
        }
        
        self._initialize_strategies()
    
    def _load_recovery_config(self) -> Dict[str, Any]:
        """復旧設定読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"復旧設定ファイル未見つけ: {self.config_path}")
                return self._get_default_recovery_config()
        except Exception as e:
            self.logger.error(f"復旧設定読み込みエラー: {e}")
            return self._get_default_recovery_config()
    
    def _get_default_recovery_config(self) -> Dict[str, Any]:
        """デフォルト復旧設定"""
        return {
            "strategies": {
                "simple_retry": {
                    "class": "SimpleRetryStrategy",
                    "max_retries": 3,
                    "base_delay": 1.0
                },
                "exponential_backoff": {
                    "class": "ExponentialBackoffStrategy", 
                    "max_retries": 5,
                    "base_delay": 1.0,
                    "max_delay": 60.0,
                    "jitter": True
                },
                "linear_backoff": {
                    "class": "LinearBackoffStrategy",
                    "max_retries": 4,
                    "base_delay": 2.0
                }
            },
            "error_type_mapping": {
                "strategy_errors": "exponential_backoff",
                "data_errors": "linear_backoff", 
                "system_errors": "simple_retry"
            },
            "global_settings": {
                "max_total_retry_time": 300,
                "circuit_breaker_threshold": 10,
                "circuit_breaker_timeout": 60
            }
        }
    
    def _initialize_strategies(self):
        """戦略初期化"""
        strategies = self.recovery_config.get('strategies', {})
        
        for name, config in strategies.items():
            try:
                strategy_class = config.get('class')
                if strategy_class == 'SimpleRetryStrategy':
                    self._strategy_instances[name] = SimpleRetryStrategy(
                        max_retries=config.get('max_retries', 3),
                        base_delay=config.get('base_delay', 1.0)
                    )
                elif strategy_class == 'ExponentialBackoffStrategy':
                    self._strategy_instances[name] = ExponentialBackoffStrategy(
                        max_retries=config.get('max_retries', 5),
                        base_delay=config.get('base_delay', 1.0),
                        max_delay=config.get('max_delay', 60.0),
                        jitter=config.get('jitter', True)
                    )
                elif strategy_class == 'LinearBackoffStrategy':
                    self._strategy_instances[name] = LinearBackoffStrategy(
                        max_retries=config.get('max_retries', 4),
                        base_delay=config.get('base_delay', 2.0)
                    )
                
                self.logger.debug(f"復旧戦略初期化: {name} ({strategy_class})")
                
            except Exception as e:
                self.logger.error(f"復旧戦略初期化失敗 {name}: {e}")
    
    def recover_with_retry(self, func: Callable[[], Any], error_type: str, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """リトライ復旧実行"""
        start_time = datetime.now()
        self.recovery_stats['total_attempts'] += 1
        
        # エラータイプに対応する戦略取得
        strategy_name = self.recovery_config.get('error_type_mapping', {}).get(
            error_type, 'simple_retry'
        )
        strategy = self._strategy_instances.get(strategy_name)
        
        if strategy is None:
            self.logger.error(f"復旧戦略未見つけ: {strategy_name}")
            return self._create_recovery_result(False, "復旧戦略未見つけ", start_time)
        
        # 戦略使用統計更新
        if strategy_name not in self.recovery_stats['strategy_usage']:
            self.recovery_stats['strategy_usage'][strategy_name] = 0
        self.recovery_stats['strategy_usage'][strategy_name] += 1
        
        attempt = 0
        last_error = None
        
        while attempt < strategy.max_retries:
            try:
                self.logger.info(f"復旧試行 {attempt + 1}/{strategy.max_retries} (戦略: {strategy_name})")
                
                result = func()
                
                # 成功
                recovery_time = (datetime.now() - start_time).total_seconds()
                self.recovery_stats['successful_recoveries'] += 1
                self._update_average_recovery_time(recovery_time)
                
                self.logger.info(f"復旧成功 (試行: {attempt + 1}, 時間: {recovery_time:.2f}秒)")
                
                return self._create_recovery_result(
                    True, f"成功 (試行: {attempt + 1})", start_time,
                    {'attempts': attempt + 1, 'strategy': strategy_name, 'result': result}
                )
                
            except Exception as e:
                last_error = e
                attempt += 1
                
                if strategy.should_retry(e, attempt):
                    delay = strategy.calculate_delay(attempt - 1)
                    self.logger.warning(f"復旧試行失敗 {attempt}: {e}, {delay:.2f}秒後にリトライ")
                    time.sleep(delay)
                else:
                    break
        
        # 失敗
        self.recovery_stats['failed_recoveries'] += 1
        recovery_time = (datetime.now() - start_time).total_seconds()
        self._update_average_recovery_time(recovery_time)
        
        self.logger.error(f"復旧失敗 (試行: {attempt}, 戦略: {strategy_name}): {last_error}")
        
        return self._create_recovery_result(
            False, f"最大試行回数超過: {last_error}", start_time,
            {'attempts': attempt, 'strategy': strategy_name, 'last_error': str(last_error)}
        )
    
    def recover_with_fallback(self, primary_func: Callable[[], Any], 
                             fallback_funcs: List[Callable[[], Any]], 
                             error_type: str) -> Dict[str, Any]:
        """フォールバック復旧実行"""
        start_time = datetime.now()
        self.recovery_stats['total_attempts'] += 1
        
        try:
            # 主要機能実行
            self.logger.info("主要機能実行")
            result = primary_func()
            
            recovery_time = (datetime.now() - start_time).total_seconds()
            self.recovery_stats['successful_recoveries'] += 1
            self._update_average_recovery_time(recovery_time)
            
            return self._create_recovery_result(
                True, "主要機能成功", start_time, {'result': result}
            )
            
        except Exception as primary_error:
            self.logger.warning(f"主要機能失敗: {primary_error}")
            
            # フォールバック実行
            fallback_strategy = FallbackStrategy(fallback_funcs)
            
            try:
                result = fallback_strategy.execute_fallback(primary_error)
                
                recovery_time = (datetime.now() - start_time).total_seconds()
                self.recovery_stats['successful_recoveries'] += 1
                self._update_average_recovery_time(recovery_time)
                
                self.logger.info("フォールバック復旧成功")
                
                return self._create_recovery_result(
                    True, "フォールバック成功", start_time,
                    {'result': result, 'primary_error': str(primary_error)}
                )
                
            except Exception as fallback_error:
                recovery_time = (datetime.now() - start_time).total_seconds()
                self.recovery_stats['failed_recoveries'] += 1
                self._update_average_recovery_time(recovery_time)
                
                self.logger.error(f"フォールバック復旧失敗: {fallback_error}")
                
                return self._create_recovery_result(
                    False, f"フォールバック失敗: {fallback_error}", start_time,
                    {'primary_error': str(primary_error), 'fallback_error': str(fallback_error)}
                )
    
    def _create_recovery_result(self, success: bool, message: str, 
                               start_time: datetime, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """復旧結果作成"""
        return {
            'recovery_successful': success,
            'recovery_message': message,
            'recovery_duration': (datetime.now() - start_time).total_seconds(),
            'recovery_timestamp': datetime.now().isoformat(),
            'recovery_details': details or {}
        }
    
    def _update_average_recovery_time(self, recovery_time: float):
        """平均復旧時間更新"""
        total_attempts = self.recovery_stats['total_attempts']
        current_avg = self.recovery_stats['average_recovery_time']
        
        self.recovery_stats['average_recovery_time'] = (
            (current_avg * (total_attempts - 1) + recovery_time) / total_attempts
        )
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """復旧統計取得"""
        stats = self.recovery_stats.copy()
        
        if stats['total_attempts'] > 0:
            stats['success_rate'] = (
                stats['successful_recoveries'] / stats['total_attempts'] * 100
            )
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """統計リセット"""
        self.recovery_stats = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'strategy_usage': {},
            'average_recovery_time': 0.0
        }
        self.logger.info("復旧統計リセット")


# グローバルインスタンス
_global_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """グローバル復旧管理インスタンス取得"""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
    return _global_recovery_manager


def retry_with_strategy(func: Callable[[], Any], error_type: str = "system_errors", 
                       context: Optional[Dict[str, Any]] = None) -> Any:
    """戦略的リトライ実行（デコレータ対応）"""
    manager = get_recovery_manager()
    result = manager.recover_with_retry(func, error_type, context)
    
    if result['recovery_successful']:
        return result['recovery_details'].get('result')
    else:
        raise Exception(result['recovery_message'])


def fallback_recovery(primary_func: Callable[[], Any], 
                     fallback_funcs: List[Callable[[], Any]], 
                     error_type: str = "system_errors") -> Any:
    """フォールバック復旧実行"""
    manager = get_recovery_manager()
    result = manager.recover_with_fallback(primary_func, fallback_funcs, error_type)
    
    if result['recovery_successful']:
        return result['recovery_details'].get('result')
    else:
        raise Exception(result['recovery_message'])
