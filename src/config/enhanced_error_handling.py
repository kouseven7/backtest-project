"""
エラー処琁E��化モジュール
Phase 2: Error Severity Policy完�E準拠・詳細エラーロギング・回復処琁E��カニズム実裁E

Author: imega
Created: 2025-10-07
Task: TODO(tag:phase3, rationale:Error Severity Policy完�E準拠とSystemFallbackPolicy統吁E
"""

import logging
import traceback
import time
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

# プロジェクト�Eインポ�EチE
try:
    from config.logger_config import setup_logger
    from src.config.system_modes import SystemFallbackPolicy, ComponentType, SystemMode
    logger = setup_logger(__name__)
except ImportError as e:
    # フォールバック: 標準ログ設宁E
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """
    Error Severity Policy準拠のエラー重要度刁E��E
    TODO(tag:phase3, rationale:SystemFallbackPolicy統合�EProduction mode対忁E
    """
    CRITICAL = "critical"    # シスチE��致命皁E��ラー - PRODUCTION mode即停止
    ERROR = "error"         # 機�Eエラー - DEVELOPMENT mode明示皁E��ォールバック
    WARNING = "warning"     # 警呁E- フォールバック使用記録・統計更新
    INFO = "info"          # 惁E�� - 状態追跡・詳細ログ出劁E
    DEBUG = "debug"        # チE��チE�� - 開発老E��け詳細惁E��


@dataclass
class EnhancedErrorRecord:
    """
    詳細エラーロギング用拡張エラーレコーチE
    TODO(tag:phase3, rationale:コンポ�Eネント別エラー追跡・復旧手頁E��録)
    """
    timestamp: datetime
    severity: ErrorSeverity
    component_type: ComponentType
    component_name: str
    error_type: str
    error_message: str
    stack_trace: str
    system_mode: SystemMode
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None
    performance_impact: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


class ErrorRecoveryManager:
    """
    回復処琁E��カニズム管琁E��ラス
    TODO(tag:phase3, rationale:自動回復・段階的劣化�Eエラー状態管琁E��裁E
    """
    
    def __init__(self):
        self.recovery_strategies: Dict = {}
        self.error_states: Dict[str, Dict] = {}
        self.recovery_history: List[Dict] = []
        self.logger = logger
    
    def register_recovery_strategy(self, 
                                 component_name: str, 
                                 error_type: str, 
                                 recovery_func: Callable,
                                 max_retries: int = 3):
        """
        回復戦略を登録
        TODO(tag:phase3, rationale:コンポ�Eネント別自動回復戦略定義)
        """
        key = f"{component_name}:{error_type}"
        self.recovery_strategies[key] = {
            'func': recovery_func,
            'max_retries': max_retries,
            'retry_count': 0,
            'last_attempt': None
        }
        
        self.logger.info(f"回復戦略登録: {key} (max_retries: {max_retries})")
    
    def attempt_recovery(self, 
                        component_name: str, 
                        error_type: str, 
                        error_context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        自動回復を試衁E
        TODO(tag:phase3, rationale:段階的劣化�E再試行制限�E回復可能性判宁E
        """
        key = f"{component_name}:{error_type}"
        
        if key not in self.recovery_strategies:
            self.logger.warning(f"回復戦略が未登録: {key}")
            return False, "No recovery strategy available"
        
        strategy = self.recovery_strategies[key]
        
        # 再試行制限チェチE��
        if strategy['retry_count'] >= strategy['max_retries']:
            self.logger.error(f"回復試行上限に達しました: {key} ({strategy['retry_count']}/{strategy['max_retries']})")
            return False, "Max retries exceeded"
        
        try:
            self.logger.info(f"回復処琁E��姁E {key} (試衁E {strategy['retry_count'] + 1}/{strategy['max_retries']})")
            
            # 回復処琁E��衁E
            recovery_result = strategy['func'](error_context)
            
            # 成功記録
            strategy['retry_count'] += 1
            strategy['last_attempt'] = datetime.now()
            
            recovery_record = {
                'timestamp': datetime.now(),
                'component_name': component_name,
                'error_type': error_type,
                'success': recovery_result,
                'retry_count': strategy['retry_count'],
                'context': error_context
            }
            self.recovery_history.append(recovery_record)
            
            if recovery_result:
                self.logger.info(f"回復処琁E�E劁E {key}")
                # 成功したら�E試行カウントをリセチE��
                strategy['retry_count'] = 0
                return True, "Recovery successful"
            else:
                self.logger.warning(f"回復処琁E��敁E {key}")
                return False, "Recovery failed"
                
        except Exception as recovery_error:
            strategy['retry_count'] += 1
            strategy['last_attempt'] = datetime.now()
            
            self.logger.error(f"回復処琁E��にエラー: {key} - {recovery_error}")
            
            recovery_record = {
                'timestamp': datetime.now(),
                'component_name': component_name,
                'error_type': error_type,
                'success': False,
                'retry_count': strategy['retry_count'],
                'error': str(recovery_error),
                'context': error_context
            }
            self.recovery_history.append(recovery_record)
            
            return False, f"Recovery error: {recovery_error}"


class EnhancedErrorHandler:
    """
    強化エラーハンドリングシスチE��
    TODO(tag:phase3, rationale:Error Severity Policy統合�ESystemFallbackPolicy連携)
    """
    
    def __init__(self, 
                 fallback_policy: SystemFallbackPolicy,
                 recovery_manager: Optional[ErrorRecoveryManager] = None):
        """
        強化エラーハンドラーの初期匁E
        TODO(tag:phase3, rationale:SystemFallbackPolicy統合�E回復処琁E��カニズム統吁E
        """
        self.fallback_policy = fallback_policy
        self.recovery_manager = recovery_manager or ErrorRecoveryManager()
        self.error_records: List[EnhancedErrorRecord] = []
        self.logger = logger
        
        # Production/Development mode別ログレベル制御
        self.log_level_mapping = {
            SystemMode.PRODUCTION: {
                ErrorSeverity.CRITICAL: logging.CRITICAL,
                ErrorSeverity.ERROR: logging.ERROR,
                ErrorSeverity.WARNING: logging.WARNING,
                ErrorSeverity.INFO: logging.WARNING,  # Production時�EINFOをWARNINGレベルに上げめE
                ErrorSeverity.DEBUG: logging.INFO     # Production時�EDEBUGをINFOレベルに上げめE
            },
            SystemMode.DEVELOPMENT: {
                ErrorSeverity.CRITICAL: logging.CRITICAL,
                ErrorSeverity.ERROR: logging.ERROR,
                ErrorSeverity.WARNING: logging.WARNING,
                ErrorSeverity.INFO: logging.INFO,
                ErrorSeverity.DEBUG: logging.DEBUG
            },
            SystemMode.TESTING: {
                ErrorSeverity.CRITICAL: logging.CRITICAL,
                ErrorSeverity.ERROR: logging.ERROR,
                ErrorSeverity.WARNING: logging.WARNING,
                ErrorSeverity.INFO: logging.INFO,
                ErrorSeverity.DEBUG: logging.DEBUG
            }
        }
        
        self.logger.info(f"EnhancedErrorHandler初期化完亁E- SystemMode: {fallback_policy.mode.value}")
    
    def handle_error(self,
                    severity: ErrorSeverity,
                    component_type: ComponentType,
                    component_name: str,
                    error: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    recovery_hint: Optional[str] = None) -> bool:
        """
        強化エラーハンドリング処琁E
        TODO(tag:phase3, rationale:Error Severity Policy完�E準拠・自動回復試衁E
        """
        context = context or {}
        
        # 詳細エラーレコード作�E
        error_record = EnhancedErrorRecord(
            timestamp=datetime.now(),
            severity=severity,
            component_type=component_type,
            component_name=component_name,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            system_mode=self.fallback_policy.mode,
            context=context
        )
        
        # ログレベル別出劁E
        log_level = self.log_level_mapping[self.fallback_policy.mode][severity]
        self.logger.log(
            log_level,
            f"[{severity.value.upper()}] {component_name} ({component_type.value}): "
            f"{error_record.error_type} - {error_record.error_message}"
        )
        
        # スタチE��トレースはDEBUGレベルで出劁E
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.ERROR] or self.fallback_policy.mode == SystemMode.DEVELOPMENT:
            self.logger.debug(f"Stack trace:\n{error_record.stack_trace}")
        
        # Error Severity Policy準拠の処琁E�E岁E
        if severity == ErrorSeverity.CRITICAL:
            return self._handle_critical_error(error_record, error)
        elif severity == ErrorSeverity.ERROR:
            return self._handle_error_level(error_record, error, recovery_hint)
        elif severity == ErrorSeverity.WARNING:
            return self._handle_warning_level(error_record, error, recovery_hint)
        else:  # INFO, DEBUG
            return self._handle_info_debug_level(error_record, error)
    
    def _handle_critical_error(self, error_record: EnhancedErrorRecord, error: Exception) -> bool:
        """
        CRITICAL エラー処琁E Production Ready直接処琁E(SystemFallbackPolicy除去)
        TODO(tag:phase3, rationale:Production Ready・フォールバック完�E除去)
        """
        self.error_records.append(error_record)
        
        if self.fallback_policy.mode == SystemMode.PRODUCTION:
            self.logger.critical(
                f"CRITICAL ERROR in PRODUCTION MODE: {error_record.component_name} - "
                f"System will terminate immediately"
            )
            # Production Ready: フォールバック禁止・直接エラー処琁E
            raise SystemExit(f"Production CRITICAL ERROR: {error_record.component_name} - {error}")
        else:
            self.logger.critical(
                f"CRITICAL ERROR in {self.fallback_policy.mode.value.upper()} MODE: "
                f"{error_record.component_name} - Attempting graceful handling"
            )
            # Development/Testing modeでは回復を試衁E
            return self._attempt_error_recovery(error_record, error)
    
    def _handle_error_level(self, error_record: EnhancedErrorRecord, error: Exception, recovery_hint: Optional[str]) -> bool:
        """
        ERROR レベル処琁E Production Ready直接エラー処琁E(SystemFallbackPolicy除去)
        TODO(tag:phase3, rationale:Production Ready・明示皁E��ラー処琁E�Eフォールバック除去)
        """
        self.error_records.append(error_record)
        
        # まず�E動回復を試衁E
        recovery_success = self._attempt_error_recovery(error_record, error, recovery_hint)
        
        if recovery_success:
            return True
        
        # Production Ready: 回復失敗時の直接エラー処琁E
        self.logger.error(f"ERROR level failure: {error_record.component_name} - {error}")
        
        # 重要コンポ�Eネント判宁E
        critical_components = ['DSSMS_CORE', 'MULTI_STRATEGY', 'DATA_FETCHER']
        
        if any(comp in error_record.component_name.upper() for comp in critical_components):
            # 重要コンポ�Eネントエラー: Production mode時�E停止
            if self.fallback_policy.mode == SystemMode.PRODUCTION:
                self.logger.critical(f"Critical component error: {error_record.component_name}")
                raise RuntimeError(f"Production component failure: {error_record.component_name} - {error}")
            else:
                # Development/Testing mode: エラー記録・処琁E��綁E
                self.logger.warning(f"Critical component error in {self.fallback_policy.mode.value} mode: {error_record.component_name}")
                return False
        else:
            # 非重要コンポ�EネンチE エラー記録・処琁E��綁E
            self.logger.warning(f"Non-critical error handled: {error_record.component_name}")
            return False
    
    def _handle_warning_level(self, error_record: EnhancedErrorRecord, error: Exception, recovery_hint: Optional[str]) -> bool:
        """
        WARNING レベル処琁E フォールバック使用記録・統計更新
        TODO(tag:phase3, rationale:フォールバック統計更新・継続動作保証)
        """
        self.error_records.append(error_record)
        
        # 自動回復を試衁E
        recovery_success = self._attempt_error_recovery(error_record, error, recovery_hint)
        
        if recovery_success:
            self.logger.warning(f"WARNING処琁E��亁E- 回復成功: {error_record.component_name}")
            return True
        else:
            self.logger.warning(f"WARNING処琁E��亁E- 回復失敗、継続動佁E {error_record.component_name}")
            return False  # 継続動作するが警告�E記録
    
    def _handle_info_debug_level(self, error_record: EnhancedErrorRecord, error: Exception) -> bool:
        """
        INFO/DEBUG レベル処琁E 状態追跡・詳細ログ出劁E
        TODO(tag:phase3, rationale:状態追跡・開発老E��け詳細惁E��)
        """
        self.error_records.append(error_record)
        
        # INFO/DEBUGレベルは通常エラーではなく、情報記録のみ
        self.logger.info(f"惁E��記録: {error_record.component_name} - {error_record.error_message}")
        return True  # 常に成功として扱ぁE
    
    def _attempt_error_recovery(self, error_record: EnhancedErrorRecord, error: Exception, recovery_hint: Optional[str] = None) -> bool:
        """
        自動回復処琁E�E試衁E
        TODO(tag:phase3, rationale:回復処琁E��カニズム・段階的劣化対忁E
        """
        if not self.recovery_manager:
            return False
        
        # 回復コンチE��スト準備
        recovery_context = {
            'error_record': error_record,
            'original_error': error,
            'recovery_hint': recovery_hint,
            'timestamp': datetime.now()
        }
        
        # 回復処琁E��衁E
        recovery_success, recovery_message = self.recovery_manager.attempt_recovery(
            component_name=error_record.component_name,
            error_type=error_record.error_type,
            error_context=recovery_context
        )
        
        # 結果をエラーレコードに記録
        error_record.recovery_attempted = True
        error_record.recovery_successful = recovery_success
        error_record.recovery_method = recovery_message
        
        return recovery_success
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        詳細エラー統計を取征E
        TODO(tag:phase3, rationale:エラー統計�Eパフォーマンス影響刁E��)
        """
        if not self.error_records:
            return {
                'total_errors': 0,
                'by_severity': {},
                'by_component': {},
                'recovery_rate': 0.0,
                'recent_errors': []
            }
        
        # 重要度別雁E��E
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = sum(
                1 for record in self.error_records 
                if record.severity == severity
            )
        
        # コンポ�Eネント別雁E��E
        component_counts = {}
        for record in self.error_records:
            key = f"{record.component_name} ({record.component_type.value})"
            component_counts[key] = component_counts.get(key, 0) + 1
        
        # 回復玁E��箁E
        recovery_attempted = sum(1 for record in self.error_records if record.recovery_attempted)
        recovery_successful = sum(1 for record in self.error_records if record.recovery_successful)
        recovery_rate = (recovery_successful / recovery_attempted * 100) if recovery_attempted > 0 else 0.0
        
        # 最近�Eエラー (最新10件)
        recent_errors = [
            {
                'timestamp': record.timestamp.isoformat(),
                'severity': record.severity.value,
                'component': f"{record.component_name} ({record.component_type.value})",
                'error': f"{record.error_type}: {record.error_message}",
                'recovered': record.recovery_successful
            }
            for record in sorted(self.error_records, key=lambda x: x.timestamp, reverse=True)[:10]
        ]
        
        return {
            'total_errors': len(self.error_records),
            'by_severity': severity_counts,
            'by_component': component_counts,
            'recovery_rate': round(recovery_rate, 2),
            'recovery_statistics': {
                'attempted': recovery_attempted,
                'successful': recovery_successful,
                'failed': recovery_attempted - recovery_successful
            },
            'recent_errors': recent_errors,
            'system_mode': self.fallback_policy.mode.value
        }
    
    def export_error_log(self, filepath: str):
        """
        詳細エラーログをファイルに出劁E
        TODO(tag:phase3, rationale:詳細エラーログ・復旧手頁E��録)
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'system_mode': self.fallback_policy.mode.value,
                'statistics': self.get_error_statistics(),
                'detailed_records': [
                    {
                        'timestamp': record.timestamp.isoformat(),
                        'severity': record.severity.value,
                        'component_type': record.component_type.value,
                        'component_name': record.component_name,
                        'error_type': record.error_type,
                        'error_message': record.error_message,
                        'stack_trace': record.stack_trace,
                        'recovery_attempted': record.recovery_attempted,
                        'recovery_successful': record.recovery_successful,
                        'recovery_method': record.recovery_method,
                        'context': record.context
                    }
                    for record in self.error_records
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"詳細エラーログを�E力しました: {filepath}")
            
        except Exception as e:
            self.logger.error(f"エラーログ出力に失敗しました: {e}")


# グローバル強化エラーハンドラーインスタンス (オプション)
# TODO(tag:phase3, rationale:シスチE��全体での統一エラーハンドリング)
_global_error_handler: Optional[EnhancedErrorHandler] = None

def initialize_global_error_handler(fallback_policy: SystemFallbackPolicy) -> EnhancedErrorHandler:
    """
    グローバル強化エラーハンドラーを�E期化
    TODO(tag:phase3, rationale:シスチE��統一エラーハンドリング・簡易API提侁E
    """
    global _global_error_handler
    _global_error_handler = EnhancedErrorHandler(fallback_policy)
    logger.info("グローバル強化エラーハンドラーを�E期化しました")
    return _global_error_handler

def get_global_error_handler() -> Optional[EnhancedErrorHandler]:
    """グローバル強化エラーハンドラーを取得"""
    return _global_error_handler
