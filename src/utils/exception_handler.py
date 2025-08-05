"""
統一例外処理システム
- 既存logger_config.pyと統合
- 段階的実装（戦略優先）
- 自動復帰機能付き
"""

import traceback
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import sys

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存ロガー設定を使用
from config.logger_config import setup_logger

class StrategyError(Exception):
    """戦略実行エラー"""
    def __init__(self, strategy_name: str, message: str, original_error: Optional[Exception] = None):
        self.strategy_name = strategy_name
        self.original_error = original_error
        super().__init__(f"{strategy_name}: {message}")

class DataError(Exception):
    """データ関連エラー"""
    pass

class SystemError(Exception):
    """システムレベルエラー"""
    pass

class UnifiedExceptionHandler:
    """統一例外処理クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        
        # エラーポリシー設定読み込み
        if config_path is None:
            config_file_path = project_root / "config" / "error_handling" / "error_policies.json"
        else:
            config_file_path = Path(config_path)
        
        self.config_path = config_file_path
        self.error_policies = self._load_error_policies()
        
        # エラー統計
        self.error_stats = {
            'total_errors': 0,
            'strategy_errors': 0,
            'data_errors': 0,
            'system_errors': 0,
            'recovered_errors': 0,
            'fatal_errors': 0
        }
        
    def _load_error_policies(self) -> Dict[str, Any]:
        """エラーポリシー設定読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_policies()
        except Exception as e:
            self.logger.warning(f"エラーポリシー読み込み失敗: {e}")
            return self._get_default_policies()
    
    def _get_default_policies(self) -> Dict[str, Any]:
        """デフォルトエラーポリシー"""
        return {
            "strategy_errors": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "fallback_enabled": True,
                "continue_on_failure": True,
                "log_level": "ERROR"
            },
            "data_errors": {
                "max_retries": 2,
                "retry_delay": 2.0,
                "fallback_enabled": True,
                "continue_on_failure": False,
                "log_level": "CRITICAL"
            },
            "system_errors": {
                "max_retries": 1,
                "retry_delay": 5.0,
                "fallback_enabled": False,
                "continue_on_failure": False,
                "log_level": "CRITICAL"
            },
            "notification": {
                "console_output": True,
                "file_logging": True,
                "error_threshold": 5
            }
        }
    
    def handle_strategy_error(self, strategy_name: str, error: Exception, 
                            context: Optional[Dict[str, Any]] = None, retry_func: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
        """戦略エラーの統一処理"""
        self.error_stats['total_errors'] += 1
        self.error_stats['strategy_errors'] += 1
        
        error_info = {
            'type': 'strategy',
            'strategy_name': strategy_name,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc(),
            'recovery_attempted': False,
            'recovery_successful': False
        }
        
        # ログ出力
        self._log_error(error_info)
        
        # 自動復帰試行
        recovery_result = self._attempt_recovery('strategy_errors', retry_func, error_info)
        error_info.update(recovery_result)
        
        # 継続可能性判定
        policy = self.error_policies.get('strategy_errors', {})
        if policy.get('continue_on_failure', True) or recovery_result['recovery_successful']:
            error_info['fatal'] = False
        else:
            error_info['fatal'] = True
            self.error_stats['fatal_errors'] += 1
            
        return error_info
    
    def handle_data_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, 
                         retry_func: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
        """データエラーの統一処理"""
        self.error_stats['total_errors'] += 1
        self.error_stats['data_errors'] += 1
        
        error_info = {
            'type': 'data',
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc(),
            'recovery_attempted': False,
            'recovery_successful': False
        }
        
        # ログ出力
        self._log_error(error_info)
        
        # 自動復帰試行
        recovery_result = self._attempt_recovery('data_errors', retry_func, error_info)
        error_info.update(recovery_result)
        
        # データエラーは通常致命的
        policy = self.error_policies.get('data_errors', {})
        if not policy.get('continue_on_failure', False) and not recovery_result['recovery_successful']:
            error_info['fatal'] = True
            self.error_stats['fatal_errors'] += 1
        else:
            error_info['fatal'] = False
            
        return error_info
    
    def handle_system_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """システムエラーの統一処理"""
        self.error_stats['total_errors'] += 1
        self.error_stats['system_errors'] += 1
        
        error_info = {
            'type': 'system',
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc(),
            'recovery_attempted': False,
            'recovery_successful': False,
            'fatal': True  # システムエラーは基本的に致命的
        }
        
        # ログ出力
        self._log_error(error_info)
        self.error_stats['fatal_errors'] += 1
        
        return error_info
    
    def _attempt_recovery(self, error_type: str, retry_func: Optional[Callable[[], Any]], 
                         error_info: Dict[str, Any]) -> Dict[str, Any]:
        """自動復帰試行"""
        if retry_func is None:
            return {'recovery_attempted': False, 'recovery_successful': False}
        
        policy = self.error_policies.get(error_type, {})
        max_retries = policy.get('max_retries', 0)
        retry_delay = policy.get('retry_delay', 1.0)
        
        if max_retries <= 0:
            return {'recovery_attempted': False, 'recovery_successful': False}
        
        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'retry_count': 0,
            'retry_errors': []
        }
        
        import time
        
        for attempt in range(max_retries):
            try:
                recovery_result['retry_count'] = attempt + 1
                
                # リトライ遅延
                if attempt > 0:
                    time.sleep(retry_delay)
                
                # 復帰処理実行
                result = retry_func()
                
                if result is not None:  # 成功とみなす
                    recovery_result['recovery_successful'] = True
                    self.error_stats['recovered_errors'] += 1
                    self.logger.info(f"復帰成功: {error_type}, 試行回数: {attempt + 1}")
                    break
                    
            except Exception as retry_error:
                recovery_result['retry_errors'].append(str(retry_error))
                self.logger.warning(f"復帰試行失敗 {attempt + 1}/{max_retries}: {retry_error}")
        
        return recovery_result
    
    def _log_error(self, error_info: Dict[str, Any]):
        """エラーログ出力"""
        error_type = error_info.get('type', 'unknown')
        strategy_name = error_info.get('strategy_name', '')
        error_message = error_info.get('error_message', '')
        
        # コンソール出力（既存設定準拠）
        if self.error_policies.get('notification', {}).get('console_output', True):
            if error_type == 'strategy':
                self.logger.error(f"戦略エラー [{strategy_name}]: {error_message}")
            elif error_type == 'data':
                self.logger.critical(f"データエラー: {error_message}")
            elif error_type == 'system':
                self.logger.critical(f"システムエラー: {error_message}")
        
        # ファイルログ出力
        if self.error_policies.get('notification', {}).get('file_logging', True):
            # 詳細情報をファイルに記録
            self.logger.debug(f"エラー詳細: {json.dumps(error_info, ensure_ascii=False, indent=2)}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """エラー統計取得"""
        stats = self.error_stats.copy()
        stats['error_rate'] = (
            stats['total_errors'] / max(1, stats['total_errors'] + stats['recovered_errors'])
        )
        stats['recovery_rate'] = (
            stats['recovered_errors'] / max(1, stats['total_errors'])
        )
        return stats
    
    def reset_statistics(self):
        """統計リセット"""
        for key in self.error_stats:
            self.error_stats[key] = 0
    
    def create_error_report(self, output_path: Optional[str] = None) -> str:
        """エラーレポート生成"""
        if output_path is None:
            output_file_path = project_root / "logs" / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_file_path = Path(output_path)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'policies': self.error_policies,
            'config_path': str(self.config_path)
        }
        
        try:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"エラーレポート生成: {output_file_path}")
            return str(output_file_path)
            
        except Exception as e:
            self.logger.error(f"エラーレポート生成失敗: {e}")
            return ""

# グローバルインスタンス（main.py互換）
_global_handler = None

def get_exception_handler() -> UnifiedExceptionHandler:
    """グローバル例外ハンドラ取得"""
    global _global_handler
    if _global_handler is None:
        _global_handler = UnifiedExceptionHandler()
    return _global_handler

def handle_strategy_error(strategy_name: str, error: Exception, 
                        context: Optional[Dict[str, Any]] = None, retry_func: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    """戦略エラー処理（グローバル関数）"""
    return get_exception_handler().handle_strategy_error(strategy_name, error, context, retry_func)

def handle_data_error(error: Exception, context: Optional[Dict[str, Any]] = None, 
                     retry_func: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    """データエラー処理（グローバル関数）"""
    return get_exception_handler().handle_data_error(error, context, retry_func)

def handle_system_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """システムエラー処理（グローバル関数）"""
    return get_exception_handler().handle_system_error(error, context)
