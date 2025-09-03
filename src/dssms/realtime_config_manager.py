"""
DSSMS Phase 3 Task 3.2: リアルタイム設定管理システム
Realtime Config Manager - 動的設定更新・バリデーション・JSON管理

主要機能:
1. JSON設定ファイル管理
2. 動的設定更新・リロード
3. 設定値バリデーション
4. 設定変更履歴管理
5. 設定テンプレート機能

Author: GitHub Copilot Agent
Created: 2025-01-22
Task: Phase 3 Task 3.2 - リアルタイム実行環境構築
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
from threading import Lock
import hashlib
import shutil

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class ConfigType(Enum):
    """設定タイプ"""
    EXECUTION = "execution"
    MARKET = "market"
    RISK = "risk"
    SYSTEM = "system"
    USER = "user"

class ValidationResult(Enum):
    """バリデーション結果"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"

@dataclass
class ConfigChange:
    """設定変更履歴"""
    timestamp: datetime
    config_type: ConfigType
    key_path: str
    old_value: Any
    new_value: Any
    user: str = "system"
    reason: str = ""

@dataclass
class ValidationError:
    """バリデーションエラー"""
    field_path: str
    message: str
    severity: str = "error"
    suggested_value: Optional[Any] = None

class RealtimeConfigManager:
    """
    リアルタイム設定管理システム
    動的設定更新とバリデーション
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.logger = setup_logger(__name__)
        
        # 設定ファイルパス
        if config_path:
            self.config_file = Path(config_path)
        else:
            self.config_file = Path(project_root) / "config" / "realtime_execution_config.json"
        
        # 設定データ
        self.config: Dict[str, Any] = {}
        self.config_lock = Lock()
        
        # バリデーション規則
        self.validation_rules = self._setup_validation_rules()
        
        # 変更履歴
        self.change_history: List[ConfigChange] = []
        self.max_history_size = 1000
        
        # 監視設定
        self.file_watch_enabled = True
        self.auto_reload = True
        self.last_modified = None
        self.file_hash = None
        
        # コールバック
        self.change_callbacks: List[Callable] = []
        
        # 初期読み込み
        self._load_config()
        
        self.logger.info("リアルタイム設定管理システム初期化完了")
    
    def _setup_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """バリデーション規則設定"""
        return {
            'execution.mode': {
                'type': str,
                'allowed_values': ['simulation', 'paper_trading', 'live_trading'],
                'required': True
            },
            'execution.event_queue_size': {
                'type': int,
                'min_value': 100,
                'max_value': 100000,
                'required': True
            },
            'execution.event_worker_count': {
                'type': int,
                'min_value': 1,
                'max_value': 20,
                'required': True
            },
            'market.poll_interval': {
                'type': float,
                'min_value': 0.1,
                'max_value': 60.0,
                'required': True
            },
            'risk.max_position_size': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'required': True
            },
            'risk.max_drawdown': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'required': True
            },
            'system.enable_logging': {
                'type': bool,
                'required': True
            },
            'system.log_level': {
                'type': str,
                'allowed_values': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                'required': True
            }
        }
    
    def _load_config(self) -> bool:
        """設定ファイル読み込み"""
        try:
            if not self.config_file.exists():
                self.logger.warning(f"設定ファイルが存在しません: {self.config_file}")
                self._create_default_config()
                return True
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # ファイル情報更新
            self.last_modified = self.config_file.stat().st_mtime
            self.file_hash = self._calculate_file_hash()
            
            with self.config_lock:
                self.config = config_data
            
            self.logger.info(f"設定ファイル読み込み完了: {self.config_file}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"設定ファイルJSON解析エラー: {e}")
            return False
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            return False
    
    def _create_default_config(self):
        """デフォルト設定ファイル作成"""
        try:
            default_config = {
                'execution': {
                    'mode': 'simulation',
                    'event_queue_size': 10000,
                    'event_worker_count': 4,
                    'enable_market_polling': True,
                    'market_data_poll_interval': 1.0,
                    'performance_monitor_interval': 10.0,
                    'emergency_monitor_interval': 5.0,
                    'initial_portfolio_value': 1000000.0,
                    'initial_cash': 1000000.0
                },
                'market': {
                    'primary_market': 'tse',
                    'enable_pre_market': False,
                    'enable_after_market': False,
                    'enable_lunch_break': True,
                    'timezone': 'Asia/Tokyo'
                },
                'risk': {
                    'max_position_size': 0.1,
                    'max_drawdown': 0.15,
                    'max_daily_loss': 0.05,
                    'concentration_limit': 0.3,
                    'var_confidence': 0.95
                },
                'emergency': {
                    'thresholds': {
                        'market_crash_percent': -5.0,
                        'volume_spike_multiplier': 3.0,
                        'portfolio_loss_percent': -10.0,
                        'concentration_risk_percent': 30.0,
                        'system_latency_ms': 1000.0,
                        'error_rate_percent': 5.0
                    },
                    'auto_emergency_stop': True,
                    'auto_position_reduction': True
                },
                'system': {
                    'enable_logging': True,
                    'log_level': 'INFO',
                    'log_rotation': True,
                    'max_log_files': 10,
                    'enable_metrics': True,
                    'metrics_interval': 60.0
                },
                'metadata': {
                    'version': '1.0.0',
                    'created': datetime.now().isoformat(),
                    'description': 'DSSMS Phase 3 Task 3.2 リアルタイム実行設定'
                }
            }
            
            # ディレクトリ作成
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # ファイル作成
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            
            with self.config_lock:
                self.config = default_config
            
            self.logger.info(f"デフォルト設定ファイル作成完了: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"デフォルト設定ファイル作成エラー: {e}")
    
    def get_config(self, key_path: Optional[str] = None) -> Any:
        """
        設定値取得
        
        Args:
            key_path (Optional[str]): 設定キーパス（例: "execution.mode"）
            
        Returns:
            Any: 設定値
        """
        with self.config_lock:
            if key_path is None:
                return self.config.copy()
            
            keys = key_path.split('.')
            current = self.config
            
            try:
                for key in keys:
                    current = current[key]
                return current
            except (KeyError, TypeError):
                self.logger.warning(f"設定キーが見つかりません: {key_path}")
                return None
    
    def set_config(self, key_path: str, value: Any, user: str = "system", reason: str = "") -> bool:
        """
        設定値更新
        
        Args:
            key_path (str): 設定キーパス
            value (Any): 新しい値
            user (str): 更新ユーザー
            reason (str): 更新理由
            
        Returns:
            bool: 更新成功フラグ
        """
        try:
            # バリデーション
            validation_result = self.validate_value(key_path, value)
            if validation_result.result != ValidationResult.VALID:
                self.logger.error(f"設定値バリデーション失敗: {key_path} = {value}")
                for error in validation_result.errors:
                    self.logger.error(f"  - {error.message}")
                return False
            
            # 現在値取得
            old_value = self.get_config(key_path)
            
            # 値更新
            with self.config_lock:
                keys = key_path.split('.')
                current = self.config
                
                # 階層作成
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # 値設定
                current[keys[-1]] = value
            
            # 変更履歴記録
            change = ConfigChange(
                timestamp=datetime.now(),
                config_type=ConfigType.SYSTEM,  # 実際は推定
                key_path=key_path,
                old_value=old_value,
                new_value=value,
                user=user,
                reason=reason
            )
            self._add_change_history(change)
            
            # ファイル保存
            self._save_config()
            
            # コールバック実行
            self._notify_change_callbacks(change)
            
            self.logger.info(f"設定更新完了: {key_path} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"設定更新エラー: {e}")
            return False
    
    def validate_config(self, config_data: Optional[Dict[str, Any]] = None) -> 'ConfigValidationResult':
        """
        設定全体バリデーション
        
        Args:
            config_data (Optional[Dict[str, Any]]): バリデーション対象設定
            
        Returns:
            ConfigValidationResult: バリデーション結果
        """
        try:
            if config_data is None:
                config_data = self.get_config()
            
            errors = []
            warnings = []
            
            # 各設定項目をチェック
            for rule_path, rule in self.validation_rules.items():
                value = self._get_nested_value(config_data, rule_path)
                
                if value is None:
                    if rule.get('required', False):
                        errors.append(ValidationError(
                            field_path=rule_path,
                            message=f"必須項目が設定されていません",
                            severity="error"
                        ))
                    continue
                
                # 型チェック
                expected_type = rule.get('type')
                if expected_type and not isinstance(value, expected_type):
                    errors.append(ValidationError(
                        field_path=rule_path,
                        message=f"型が不正です。期待: {expected_type.__name__}, 実際: {type(value).__name__}",
                        severity="error",
                        suggested_value=expected_type()
                    ))
                    continue
                
                # 値範囲チェック
                if 'min_value' in rule and value < rule['min_value']:
                    errors.append(ValidationError(
                        field_path=rule_path,
                        message=f"値が最小値を下回っています。最小値: {rule['min_value']}",
                        severity="error",
                        suggested_value=rule['min_value']
                    ))
                
                if 'max_value' in rule and value > rule['max_value']:
                    errors.append(ValidationError(
                        field_path=rule_path,
                        message=f"値が最大値を超えています。最大値: {rule['max_value']}",
                        severity="error",
                        suggested_value=rule['max_value']
                    ))
                
                # 許可値チェック
                if 'allowed_values' in rule and value not in rule['allowed_values']:
                    errors.append(ValidationError(
                        field_path=rule_path,
                        message=f"許可されていない値です。許可値: {rule['allowed_values']}",
                        severity="error",
                        suggested_value=rule['allowed_values'][0]
                    ))
            
            # 結果判定
            if errors:
                result = ValidationResult.INVALID
            elif warnings:
                result = ValidationResult.WARNING
            else:
                result = ValidationResult.VALID
            
            return ConfigValidationResult(
                result=result,
                errors=errors,
                warnings=warnings,
                validated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"設定バリデーションエラー: {e}")
            return ConfigValidationResult(
                result=ValidationResult.INVALID,
                errors=[ValidationError("system", f"バリデーションエラー: {str(e)}")],
                warnings=[],
                validated_at=datetime.now()
            )
    
    def validate_value(self, key_path: str, value: Any) -> 'ConfigValidationResult':
        """
        個別値バリデーション
        
        Args:
            key_path (str): 設定キーパス
            value (Any): 検証値
            
        Returns:
            ConfigValidationResult: バリデーション結果
        """
        try:
            if key_path not in self.validation_rules:
                return ConfigValidationResult(
                    result=ValidationResult.WARNING,
                    errors=[],
                    warnings=[ValidationError(key_path, "バリデーション規則が未定義です", "warning")],
                    validated_at=datetime.now()
                )
            
            rule = self.validation_rules[key_path]
            errors = []
            
            # 型チェック
            expected_type = rule.get('type')
            if expected_type and not isinstance(value, expected_type):
                errors.append(ValidationError(
                    field_path=key_path,
                    message=f"型が不正です。期待: {expected_type.__name__}",
                    suggested_value=expected_type()
                ))
            
            # 数値範囲チェック
            if isinstance(value, (int, float)):
                if 'min_value' in rule and value < rule['min_value']:
                    errors.append(ValidationError(
                        field_path=key_path,
                        message=f"値が最小値を下回っています: {rule['min_value']}",
                        suggested_value=rule['min_value']
                    ))
                if 'max_value' in rule and value > rule['max_value']:
                    errors.append(ValidationError(
                        field_path=key_path,
                        message=f"値が最大値を超えています: {rule['max_value']}",
                        suggested_value=rule['max_value']
                    ))
            
            # 許可値チェック
            if 'allowed_values' in rule and value not in rule['allowed_values']:
                errors.append(ValidationError(
                    field_path=key_path,
                    message=f"許可されていない値です: {rule['allowed_values']}",
                    suggested_value=rule['allowed_values'][0]
                ))
            
            result = ValidationResult.VALID if not errors else ValidationResult.INVALID
            
            return ConfigValidationResult(
                result=result,
                errors=errors,
                warnings=[],
                validated_at=datetime.now()
            )
            
        except Exception as e:
            return ConfigValidationResult(
                result=ValidationResult.INVALID,
                errors=[ValidationError(key_path, f"バリデーションエラー: {str(e)}")],
                warnings=[],
                validated_at=datetime.now()
            )
    
    def reload_config(self) -> bool:
        """設定ファイル再読み込み"""
        try:
            old_config = self.get_config()
            success = self._load_config()
            
            if success:
                new_config = self.get_config()
                
                # 変更検出
                changes = self._detect_config_changes(old_config, new_config)
                for change in changes:
                    self._add_change_history(change)
                    self._notify_change_callbacks(change)
                
                self.logger.info(f"設定ファイル再読み込み完了: {len(changes)}件の変更")
            
            return success
            
        except Exception as e:
            self.logger.error(f"設定再読み込みエラー: {e}")
            return False
    
    def _save_config(self) -> bool:
        """設定ファイル保存"""
        try:
            # バックアップ作成
            if self.config_file.exists():
                backup_file = self.config_file.with_suffix('.json.backup')
                shutil.copy2(self.config_file, backup_file)
            
            # 設定保存
            with self.config_lock:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            # ファイル情報更新
            self.last_modified = self.config_file.stat().st_mtime
            self.file_hash = self._calculate_file_hash()
            
            return True
            
        except Exception as e:
            self.logger.error(f"設定ファイル保存エラー: {e}")
            return False
    
    def _calculate_file_hash(self) -> str:
        """ファイルハッシュ計算"""
        try:
            with open(self.config_file, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """ネストした値取得"""
        try:
            keys = key_path.split('.')
            current = data
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return None
    
    def _detect_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """設定変更検出"""
        changes = []
        
        def compare_nested(old_data, new_data, path=""):
            if isinstance(new_data, dict) and isinstance(old_data, dict):
                # 新しいキー
                for key in new_data:
                    current_path = f"{path}.{key}" if path else key
                    if key not in old_data:
                        changes.append(ConfigChange(
                            timestamp=datetime.now(),
                            config_type=ConfigType.SYSTEM,
                            key_path=current_path,
                            old_value=None,
                            new_value=new_data[key],
                            user="file_reload",
                            reason="新規追加"
                        ))
                    else:
                        compare_nested(old_data[key], new_data[key], current_path)
                
                # 削除されたキー
                for key in old_data:
                    if key not in new_data:
                        current_path = f"{path}.{key}" if path else key
                        changes.append(ConfigChange(
                            timestamp=datetime.now(),
                            config_type=ConfigType.SYSTEM,
                            key_path=current_path,
                            old_value=old_data[key],
                            new_value=None,
                            user="file_reload",
                            reason="削除"
                        ))
            else:
                # 値変更
                if old_data != new_data:
                    changes.append(ConfigChange(
                        timestamp=datetime.now(),
                        config_type=ConfigType.SYSTEM,
                        key_path=path,
                        old_value=old_data,
                        new_value=new_data,
                        user="file_reload",
                        reason="値変更"
                    ))
        
        compare_nested(old_config, new_config)
        return changes
    
    def _add_change_history(self, change: ConfigChange):
        """変更履歴追加"""
        self.change_history.append(change)
        
        # 履歴サイズ制限
        if len(self.change_history) > self.max_history_size:
            self.change_history = self.change_history[-self.max_history_size:]
    
    def _notify_change_callbacks(self, change: ConfigChange):
        """変更コールバック通知"""
        for callback in self.change_callbacks:
            try:
                callback(change)
            except Exception as e:
                self.logger.error(f"設定変更コールバックエラー: {e}")
    
    def add_change_callback(self, callback: Callable[[ConfigChange], None]):
        """変更コールバック追加"""
        self.change_callbacks.append(callback)
    
    def get_change_history(self, limit: Optional[int] = None) -> List[ConfigChange]:
        """変更履歴取得"""
        if limit:
            return self.change_history[-limit:]
        return self.change_history.copy()
    
    def export_config(self, output_path: str) -> bool:
        """設定エクスポート"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'config': self.get_config(),
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'source_file': str(self.config_file),
                    'version': '1.0.0'
                },
                'validation': self.validate_config().__dict__
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"設定エクスポート完了: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"設定エクスポートエラー: {e}")
            return False

@dataclass
class ConfigValidationResult:
    """設定バリデーション結果"""
    result: ValidationResult
    errors: List[ValidationError]
    warnings: List[ValidationError]
    validated_at: datetime

# 使用例とテスト
async def demo_realtime_config_manager():
    """リアルタイム設定管理デモ"""
    logger = setup_logger(__name__)
    logger.info("リアルタイム設定管理デモ開始")
    
    try:
        # マネージャー初期化
        manager = RealtimeConfigManager()
        
        # 設定変更コールバック
        def config_change_handler(change: ConfigChange):
            logger.info(f"設定変更: {change.key_path} = {change.new_value}")
        
        manager.add_change_callback(config_change_handler)
        
        # 現在設定確認
        current_config = manager.get_config()
        logger.info(f"現在の設定: {len(current_config)}項目")
        
        # 個別値取得
        mode = manager.get_config('execution.mode')
        logger.info(f"実行モード: {mode}")
        
        # 値更新
        success = manager.set_config(
            'execution.event_queue_size', 
            5000, 
            user='demo_user', 
            reason='デモ実行'
        )
        logger.info(f"設定更新結果: {success}")
        
        # バリデーション
        validation_result = manager.validate_config()
        logger.info(f"バリデーション: {validation_result.result.value}")
        if validation_result.errors:
            for error in validation_result.errors:
                logger.error(f"バリデーションエラー: {error.message}")
        
        # 変更履歴
        history = manager.get_change_history(limit=5)
        logger.info(f"変更履歴: {len(history)}件")
        
        # エクスポート
        export_path = Path(project_root) / "logs" / f"config_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_success = manager.export_config(str(export_path))
        logger.info(f"エクスポート結果: {export_success}")
        
        logger.info("リアルタイム設定管理デモ完了")
        
    except Exception as e:
        logger.error(f"デモエラー: {e}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(demo_realtime_config_manager())
