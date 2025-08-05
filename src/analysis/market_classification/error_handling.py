"""
エラーハンドリング・ロバストネスシステム - A→B市場分類システム基盤
分析システムの信頼性向上と障害耐性を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time
import traceback
import functools
import warnings
from contextlib import contextmanager
import json
import os
from pathlib import Path

class ErrorSeverity(Enum):
    """エラー重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """エラーカテゴリ"""
    DATA_ERROR = "data_error"
    CALCULATION_ERROR = "calculation_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    CONFIGURATION_ERROR = "configuration_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"

class RecoveryStrategy(Enum):
    """回復戦略"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    DEFAULT_VALUE = "default_value"
    RAISE_ERROR = "raise_error"
    LOG_AND_CONTINUE = "log_and_continue"

@dataclass
class ErrorRecord:
    """エラー記録"""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, Any]
    stack_trace: str
    recovery_action: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.error_id:
            self.error_id = f"err_{int(self.timestamp.timestamp())}"

@dataclass
class RecoveryAction:
    """回復アクション"""
    strategy: RecoveryStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_function: Optional[Callable] = None
    default_value: Any = None
    timeout_seconds: float = 30.0
    
    def __post_init__(self):
        if self.strategy == RecoveryStrategy.FALLBACK and self.fallback_function is None:
            self.strategy = RecoveryStrategy.DEFAULT_VALUE

class RobustAnalysisSystem:
    """
    堅牢な分析システム
    エラーハンドリング、回復機能、フォールバック機能を提供
    """
    
    def __init__(self, 
                 error_log_file: Optional[str] = None,
                 max_error_records: int = 1000,
                 circuit_breaker_threshold: int = 5,
                 circuit_breaker_timeout: int = 300):
        """
        堅牢分析システムの初期化
        
        Args:
            error_log_file: エラーログファイルパス
            max_error_records: 最大エラー記録数
            circuit_breaker_threshold: サーキットブレーカー閾値
            circuit_breaker_timeout: サーキットブレーカータイムアウト（秒）
        """
        self.error_log_file = error_log_file
        self.max_error_records = max_error_records
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # エラー記録
        self.error_records: List[ErrorRecord] = []
        self.error_lock = threading.Lock()
        
        # サーキットブレーカー
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # メトリクス
        self.error_counts: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        self.recovery_counts: Dict[RecoveryStrategy, int] = {strat: 0 for strat in RecoveryStrategy}
        
        # 既知の問題とその対策
        self.known_issues: Dict[str, RecoveryAction] = {}
        
        # フォールバックデータ
        self.fallback_data: Dict[str, Any] = {}
        
        self.logger.info("RobustAnalysisSystem初期化完了")

    def with_error_handling(self, 
                          recovery_action: RecoveryAction,
                          error_category: ErrorCategory = ErrorCategory.CALCULATION_ERROR,
                          context: Optional[Dict[str, Any]] = None):
        """
        エラーハンドリングデコレータ
        
        Args:
            recovery_action: 回復アクション
            error_category: エラーカテゴリ
            context: コンテキスト情報
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                function_name = f"{func.__module__}.{func.__name__}"
                
                # サーキットブレーカーチェック
                if self._is_circuit_open(function_name):
                    self.logger.warning(f"サーキットブレーカー開放中: {function_name}")
                    return self._execute_fallback(recovery_action, context)
                
                for attempt in range(recovery_action.max_retries + 1):
                    try:
                        # タイムアウト付き実行
                        return self._execute_with_timeout(
                            func, args, kwargs, recovery_action.timeout_seconds
                        )
                        
                    except Exception as e:
                        error_record = self._create_error_record(
                            e, error_category, function_name, context, attempt
                        )
                        
                        self._record_error(error_record)
                        self._update_circuit_breaker(function_name)
                        
                        if attempt < recovery_action.max_retries:
                            if recovery_action.strategy == RecoveryStrategy.RETRY:
                                self.logger.info(
                                    f"リトライ {attempt + 1}/{recovery_action.max_retries}: {function_name}"
                                )
                                time.sleep(recovery_action.retry_delay * (attempt + 1))
                                continue
                        
                        # 最終的な回復処理
                        return self._handle_final_error(error_record, recovery_action, context)
                
            return wrapper
        return decorator

    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict, timeout: float) -> Any:
        """タイムアウト付き関数実行"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"関数実行がタイムアウトしました: {timeout}秒")
        
        # タイムアウト設定
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # タイマー解除
            return result
        except TimeoutError:
            raise
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def _create_error_record(self, 
                           error: Exception, 
                           category: ErrorCategory,
                           function_name: str,
                           context: Optional[Dict[str, Any]],
                           attempt: int) -> ErrorRecord:
        """エラー記録作成"""
        
        # エラー重要度判定
        severity = self._determine_error_severity(error, category)
        
        # コンテキスト情報構築
        error_context = {
            'function_name': function_name,
            'attempt': attempt,
            'error_class': error.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            error_context.update(context)
        
        return ErrorRecord(
            error_id="",  # 後で自動生成
            timestamp=datetime.now(),
            error_type=error.__class__.__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            context=error_context,
            stack_trace=traceback.format_exc()
        )

    def _determine_error_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """エラー重要度判定"""
        
        # 致命的エラー
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # カテゴリ別判定
        if category == ErrorCategory.SYSTEM_ERROR:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.DATA_ERROR:
            if isinstance(error, (ValueError, TypeError)):
                return ErrorSeverity.MEDIUM
            else:
                return ErrorSeverity.HIGH
        elif category == ErrorCategory.NETWORK_ERROR:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _record_error(self, error_record: ErrorRecord):
        """エラー記録"""
        with self.error_lock:
            self.error_records.append(error_record)
            
            # 記録数制限
            if len(self.error_records) > self.max_error_records:
                self.error_records = self.error_records[-self.max_error_records//2:]
            
            # カウント更新
            self.error_counts[error_record.category] += 1
            
            # ログ出力
            self.logger.error(
                f"[{error_record.severity.value.upper()}] {error_record.error_type}: "
                f"{error_record.error_message}"
            )
            
            # ファイル出力
            if self.error_log_file:
                self._write_error_to_file(error_record)

    def _write_error_to_file(self, error_record: ErrorRecord):
        """エラーをファイルに記録"""
        try:
            error_data = {
                'error_id': error_record.error_id,
                'timestamp': error_record.timestamp.isoformat(),
                'error_type': error_record.error_type,
                'error_message': error_record.error_message,
                'category': error_record.category.value,
                'severity': error_record.severity.value,
                'context': error_record.context,
                'stack_trace': error_record.stack_trace.split('\n')
            }
            
            # JSONLinesファイルに追記
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"エラーログファイル書き込み失敗: {e}")

    def _is_circuit_open(self, function_name: str) -> bool:
        """サーキットブレーカー状態確認"""
        if function_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[function_name]
        
        # タイムアウトチェック
        if datetime.now() > breaker['reset_time']:
            # サーキットブレーカーリセット
            del self.circuit_breakers[function_name]
            return False
        
        return breaker['is_open']

    def _update_circuit_breaker(self, function_name: str):
        """サーキットブレーカー更新"""
        if function_name not in self.circuit_breakers:
            self.circuit_breakers[function_name] = {
                'error_count': 0,
                'is_open': False,
                'reset_time': datetime.now()
            }
        
        breaker = self.circuit_breakers[function_name]
        breaker['error_count'] += 1
        
        # 閾値チェック
        if breaker['error_count'] >= self.circuit_breaker_threshold:
            breaker['is_open'] = True
            breaker['reset_time'] = datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)
            
            self.logger.warning(
                f"サーキットブレーカー開放: {function_name} "
                f"(エラー数: {breaker['error_count']})"
            )

    def _handle_final_error(self, 
                          error_record: ErrorRecord, 
                          recovery_action: RecoveryAction,
                          context: Optional[Dict[str, Any]]) -> Any:
        """最終エラーハンドリング"""
        
        strategy = recovery_action.strategy
        self.recovery_counts[strategy] += 1
        
        if strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback(recovery_action, context)
        
        elif strategy == RecoveryStrategy.DEFAULT_VALUE:
            self.logger.info(f"デフォルト値を返却: {recovery_action.default_value}")
            return recovery_action.default_value
        
        elif strategy == RecoveryStrategy.SKIP:
            self.logger.info("処理をスキップ")
            return None
        
        elif strategy == RecoveryStrategy.LOG_AND_CONTINUE:
            self.logger.warning("エラーをログに記録して継続")
            return None
        
        else:  # RAISE_ERROR
            raise RuntimeError(
                f"回復不可能なエラー: {error_record.error_message}"
            ) from None

    def _execute_fallback(self, recovery_action: RecoveryAction, context: Optional[Dict[str, Any]]) -> Any:
        """フォールバック実行"""
        if recovery_action.fallback_function:
            try:
                self.logger.info("フォールバック関数実行")
                return recovery_action.fallback_function(context)
            except Exception as e:
                self.logger.error(f"フォールバック関数もエラー: {e}")
                return recovery_action.default_value
        else:
            return recovery_action.default_value

    @contextmanager
    def error_context(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """エラーコンテキストマネージャー"""
        start_time = datetime.now()
        
        try:
            self.logger.debug(f"操作開始: {operation_name}")
            yield
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            error_context = {
                'operation_name': operation_name,
                'duration_seconds': duration
            }
            
            if context:
                error_context.update(context)
            
            error_record = self._create_error_record(
                e, ErrorCategory.SYSTEM_ERROR, operation_name, error_context, 0
            )
            
            self._record_error(error_record)
            raise
        
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"操作完了: {operation_name} ({duration:.3f}秒)")

    def validate_data(self, 
                     data: Any, 
                     validators: List[Callable[[Any], bool]],
                     error_message: str = "データ検証エラー") -> bool:
        """データ検証"""
        for i, validator in enumerate(validators):
            try:
                if not validator(data):
                    error = ValueError(f"{error_message} (バリデーター {i+1})")
                    error_record = self._create_error_record(
                        error, ErrorCategory.VALIDATION_ERROR, 
                        "validate_data", {"validator_index": i}, 0
                    )
                    self._record_error(error_record)
                    return False
                    
            except Exception as e:
                error_record = self._create_error_record(
                    e, ErrorCategory.VALIDATION_ERROR,
                    "validate_data", {"validator_index": i}, 0
                )
                self._record_error(error_record)
                return False
        
        return True

    def safe_division(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全な除算"""
        try:
            if denominator == 0:
                self.logger.warning("ゼロ除算を検出、デフォルト値を返却")
                return default
            
            result = numerator / denominator
            
            if np.isnan(result) or np.isinf(result):
                self.logger.warning("無効な計算結果を検出、デフォルト値を返却")
                return default
            
            return result
            
        except Exception as e:
            error_record = self._create_error_record(
                e, ErrorCategory.CALCULATION_ERROR, "safe_division", 
                {"numerator": numerator, "denominator": denominator}, 0
            )
            self._record_error(error_record)
            return default

    def safe_array_operation(self, 
                           data: np.ndarray, 
                           operation: Callable[[np.ndarray], Any],
                           default_value: Any = None) -> Any:
        """安全な配列演算"""
        try:
            # NaN/無限大チェック
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                clean_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                self.logger.warning("データにNaN/無限大を検出、クリーニング実行")
            else:
                clean_data = data
            
            # 空配列チェック
            if len(clean_data) == 0:
                self.logger.warning("空配列を検出、デフォルト値を返却")
                return default_value
            
            return operation(clean_data)
            
        except Exception as e:
            error_record = self._create_error_record(
                e, ErrorCategory.CALCULATION_ERROR, "safe_array_operation",
                {"data_shape": data.shape if hasattr(data, 'shape') else None}, 0
            )
            self._record_error(error_record)
            return default_value

    def safe_dataframe_operation(self, 
                                df: pd.DataFrame, 
                                operation: Callable[[pd.DataFrame], Any],
                                default_value: Any = None) -> Any:
        """安全なDataFrame演算"""
        try:
            # 基本検証
            if df.empty:
                self.logger.warning("空のDataFrameを検出")
                return default_value
            
            # NaN比率チェック
            nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if nan_ratio > 0.5:
                self.logger.warning(f"高いNaN比率を検出: {nan_ratio:.3f}")
            
            return operation(df)
            
        except Exception as e:
            error_record = self._create_error_record(
                e, ErrorCategory.DATA_ERROR, "safe_dataframe_operation",
                {"df_shape": df.shape if hasattr(df, 'shape') else None}, 0
            )
            self._record_error(error_record)
            return default_value

    def get_fallback_data(self, key: str, data_type: str = "default") -> Any:
        """フォールバックデータ取得"""
        fallback_key = f"{data_type}_{key}"
        
        if fallback_key in self.fallback_data:
            self.logger.info(f"フォールバックデータ使用: {fallback_key}")
            return self.fallback_data[fallback_key]
        
        # デフォルトフォールバックデータ
        default_fallbacks = {
            'market_data': self._create_default_market_data(),
            'analysis_result': self._create_default_analysis_result(),
            'correlation_matrix': self._create_default_correlation_matrix(),
            'volatility_data': self._create_default_volatility_data()
        }
        
        for pattern, default_data in default_fallbacks.items():
            if pattern in data_type.lower():
                self.logger.info(f"デフォルトフォールバックデータ使用: {pattern}")
                return default_data
        
        self.logger.warning(f"フォールバックデータが見つかりません: {fallback_key}")
        return None

    def set_fallback_data(self, key: str, data: Any, data_type: str = "default"):
        """フォールバックデータ設定"""
        fallback_key = f"{data_type}_{key}"
        self.fallback_data[fallback_key] = data
        self.logger.info(f"フォールバックデータ設定: {fallback_key}")

    def _create_default_market_data(self) -> pd.DataFrame:
        """デフォルト市場データ作成"""
        dates = pd.date_range(end=datetime.now(), periods=30)
        return pd.DataFrame({
            'Open': [100.0] * 30,
            'High': [105.0] * 30,
            'Low': [95.0] * 30,
            'Close': [100.0] * 30,
            'Volume': [1000000] * 30
        }, index=dates)

    def _create_default_analysis_result(self) -> Dict[str, Any]:
        """デフォルト分析結果作成"""
        return {
            'signal': 'HOLD',
            'confidence': 0.1,
            'value': 0.0,
            'timestamp': datetime.now(),
            'is_fallback': True
        }

    def _create_default_correlation_matrix(self) -> pd.DataFrame:
        """デフォルト相関マトリックス作成"""
        assets = ['Asset1', 'Asset2', 'Asset3']
        return pd.DataFrame(
            np.eye(len(assets)), 
            index=assets, 
            columns=assets
        )

    def _create_default_volatility_data(self) -> Dict[str, float]:
        """デフォルトボラティリティデータ作成"""
        return {
            'current_volatility': 0.2,
            'historical_volatility': 0.18,
            'volatility_trend': 'stable'
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """エラー概要取得"""
        with self.error_lock:
            total_errors = len(self.error_records)
            
            if total_errors == 0:
                return {'total_errors': 0, 'summary': 'エラーなし'}
            
            # カテゴリ別集計
            category_counts = {}
            severity_counts = {}
            
            for record in self.error_records:
                category_counts[record.category.value] = category_counts.get(record.category.value, 0) + 1
                severity_counts[record.severity.value] = severity_counts.get(record.severity.value, 0) + 1
            
            # 最近のエラー
            recent_errors = self.error_records[-5:] if total_errors >= 5 else self.error_records
            
            return {
                'total_errors': total_errors,
                'category_breakdown': category_counts,
                'severity_breakdown': severity_counts,
                'recovery_strategy_usage': {k.value: v for k, v in self.recovery_counts.items()},
                'recent_errors': [
                    {
                        'timestamp': record.timestamp.isoformat(),
                        'type': record.error_type,
                        'category': record.category.value,
                        'severity': record.severity.value,
                        'message': record.error_message[:100]  # 短縮
                    }
                    for record in recent_errors
                ],
                'circuit_breakers': {
                    name: {
                        'error_count': breaker['error_count'],
                        'is_open': breaker['is_open']
                    }
                    for name, breaker in self.circuit_breakers.items()
                }
            }

    def clear_error_history(self):
        """エラー履歴クリア"""
        with self.error_lock:
            self.error_records.clear()
            self.error_counts = {cat: 0 for cat in ErrorCategory}
            self.recovery_counts = {strat: 0 for strat in RecoveryStrategy}
            self.circuit_breakers.clear()
        
        self.logger.info("エラー履歴をクリアしました")

    def health_check(self) -> Dict[str, Any]:
        """システムヘルスチェック"""
        try:
            with self.error_lock:
                recent_errors = [
                    record for record in self.error_records
                    if record.timestamp > datetime.now() - timedelta(hours=1)
                ]
                
                critical_errors = [
                    record for record in recent_errors
                    if record.severity == ErrorSeverity.CRITICAL
                ]
                
                open_circuits = [
                    name for name, breaker in self.circuit_breakers.items()
                    if breaker['is_open']
                ]
                
                # ヘルス状態判定
                if critical_errors:
                    health_status = "CRITICAL"
                elif len(recent_errors) > 10:
                    health_status = "WARNING"
                elif open_circuits:
                    health_status = "DEGRADED"
                else:
                    health_status = "HEALTHY"
                
                return {
                    'status': health_status,
                    'recent_errors_count': len(recent_errors),
                    'critical_errors_count': len(critical_errors),
                    'open_circuits': open_circuits,
                    'total_error_records': len(self.error_records),
                    'check_time': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'check_time': datetime.now().isoformat()
            }

# グローバルインスタンス
_global_robust_system: Optional[RobustAnalysisSystem] = None

def get_robust_system() -> RobustAnalysisSystem:
    """グローバル堅牢システム取得"""
    global _global_robust_system
    if _global_robust_system is None:
        _global_robust_system = RobustAnalysisSystem()
    return _global_robust_system

def robust_analysis(recovery_action: RecoveryAction = None, 
                   error_category: ErrorCategory = ErrorCategory.CALCULATION_ERROR):
    """堅牢分析デコレータ（簡易版）"""
    if recovery_action is None:
        recovery_action = RecoveryAction(
            strategy=RecoveryStrategy.DEFAULT_VALUE,
            default_value=None,
            max_retries=2
        )
    
    robust_system = get_robust_system()
    return robust_system.with_error_handling(recovery_action, error_category)

# 便利な検証関数
def is_valid_dataframe(df: Any) -> bool:
    """DataFrame検証"""
    return isinstance(df, pd.DataFrame) and not df.empty and not df.isnull().all().all()

def is_valid_number(value: Any) -> bool:
    """数値検証"""
    try:
        float(value)
        return not (np.isnan(float(value)) or np.isinf(float(value)))
    except:
        return False

def is_valid_array(arr: Any) -> bool:
    """配列検証"""
    return isinstance(arr, (list, np.ndarray)) and len(arr) > 0

if __name__ == "__main__":
    # テスト用コード
    import tempfile
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== エラーハンドリング・ロバストネスシステム テスト ===")
    
    # 一時ログファイル
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_file = f.name
    
    robust_system = RobustAnalysisSystem(error_log_file=log_file)
    
    print("\n1. エラーハンドリングデコレータテスト")
    
    @robust_system.with_error_handling(
        RecoveryAction(strategy=RecoveryStrategy.DEFAULT_VALUE, default_value=42, max_retries=2),
        ErrorCategory.CALCULATION_ERROR
    )
    def problematic_function(should_fail: bool = False):
        if should_fail:
            raise ValueError("テスト用エラー")
        return 100
    
    # 成功ケース
    result = problematic_function(should_fail=False)
    print(f"成功ケース結果: {result}")
    
    # 失敗ケース（フォールバック）
    result = problematic_function(should_fail=True)
    print(f"失敗ケース結果（フォールバック）: {result}")
    
    print("\n2. 安全計算関数テスト")
    
    # 安全な除算
    safe_result = robust_system.safe_division(10, 0, default=999)
    print(f"ゼロ除算テスト: 10/0 = {safe_result}")
    
    # 安全な配列演算
    test_array = np.array([1, 2, np.nan, 4, np.inf])
    safe_array_result = robust_system.safe_array_operation(
        test_array, 
        lambda x: np.mean(x),
        default_value=0
    )
    print(f"NaN/無限大配列の平均: {safe_array_result}")
    
    print("\n3. データ検証テスト")
    
    # DataFrame検証
    valid_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    invalid_df = pd.DataFrame()
    
    validators = [is_valid_dataframe, lambda df: len(df) > 0]
    
    print(f"有効DataFrame検証: {robust_system.validate_data(valid_df, validators)}")
    print(f"無効DataFrame検証: {robust_system.validate_data(invalid_df, validators)}")
    
    print("\n4. フォールバックデータテスト")
    
    # フォールバックデータ設定
    robust_system.set_fallback_data("test_symbol", {"price": 100}, "market_data")
    
    # フォールバックデータ取得
    fallback_data = robust_system.get_fallback_data("test_symbol", "market_data")
    print(f"フォールバックデータ: {fallback_data}")
    
    # デフォルトマーケットデータ
    default_market = robust_system.get_fallback_data("unknown", "market_data")
    print(f"デフォルト市場データ形状: {default_market.shape if hasattr(default_market, 'shape') else type(default_market)}")
    
    print("\n5. エラー概要・ヘルスチェック")
    
    error_summary = robust_system.get_error_summary()
    print(f"総エラー数: {error_summary['total_errors']}")
    
    if error_summary['total_errors'] > 0:
        print("エラーカテゴリ:")
        for category, count in error_summary['category_breakdown'].items():
            print(f"  {category}: {count}")
    
    health_status = robust_system.health_check()
    print(f"システムヘルス: {health_status['status']}")
    
    print("\n6. 簡易デコレータテスト")
    
    @robust_analysis(RecoveryAction(strategy=RecoveryStrategy.LOG_AND_CONTINUE))
    def simple_function():
        raise RuntimeError("簡易テストエラー")
    
    result = simple_function()
    print(f"簡易デコレータ結果: {result}")
    
    # ログファイルクリーンアップ
    os.unlink(log_file)
    
    print("\n=== テスト完了 ===")
