"""
DSSMS System Modes and Fallback Policy Management

このモジュールは、DSSMSプロジェクト全体のフォールバック処理と
システム動作モードの統一管理を提供します。

主要機能:
1. システム動作モード管理 (PRODUCTION/DEVELOPMENT/TESTING)
2. 統一フォールバック処理ポリシー
3. コンポーネント別エラーハンドリング
4. フォールバック使用状況の追跡・記録

Author: GitHub Copilot Agent
Created: 2025-10-02
Task: TODO-FB-001 SystemMode定義実装
"""

import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

# プロジェクト内ログ設定
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    # フォールバック: 標準ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """
    システム動作モード定義
    
    フォールバック処理とエラーハンドリングの動作を制御します。
    """
    PRODUCTION = "production"    # フォールバック禁止、エラーで即停止
    DEVELOPMENT = "development"  # 明示的フォールバック許可、詳細ログ出力
    TESTING = "testing"         # モック/テストデータ許可、隔離実行


class ComponentType(Enum):
    """
    システム構成要素の分類
    
    各コンポーネントのフォールバック処理を適切に管理するための分類です。
    """
    DSSMS_CORE = "dssms_core"           # ランキング、スコアリング、銘柄選択
    STRATEGY_ENGINE = "strategy_engine"  # 個別戦略 (VWAP, Bollinger等)
    DATA_FETCHER = "data_fetcher"       # yfinance, データ取得
    RISK_MANAGER = "risk_manager"       # リスク管理、ポジション管理
    MULTI_STRATEGY = "multi_strategy"   # 統合システム、戦略統合


@dataclass
class FallbackUsageRecord:
    """フォールバック使用記録"""
    timestamp: datetime
    component_type: ComponentType
    component_name: str
    error_type: str
    error_message: str
    fallback_used: bool
    system_mode: SystemMode
    execution_context: Dict[str, Any] = field(default_factory=dict)


class SystemFallbackPolicy:
    """
    統一フォールバック管理ポリシー
    
    Phase 1での基本実装。Phase 2でより高度な機能を追加予定。
    """
    
    def __init__(self, mode: SystemMode = SystemMode.DEVELOPMENT):
        """
        システムフォールバックポリシーの初期化
        
        Args:
            mode: システム動作モード
        """
        self.mode = mode
        self.usage_records: List[FallbackUsageRecord] = []
        self.logger = logger
        
        # システムモード別設定
        self._configure_mode_settings()
        
        self.logger.info(f"SystemFallbackPolicy initialized with mode: {mode.value}")
    
    def _configure_mode_settings(self):
        """システムモード別の設定を初期化"""
        if self.mode == SystemMode.PRODUCTION:
            self.allow_fallbacks = False
            self.log_level = logging.WARNING
        elif self.mode == SystemMode.DEVELOPMENT:
            self.allow_fallbacks = True
            self.log_level = logging.DEBUG
        else:  # TESTING
            self.allow_fallbacks = True
            self.log_level = logging.INFO
    
    def handle_component_failure(
        self,
        component_type: ComponentType,
        component_name: str,
        error: Exception,
        fallback_func: Optional[Callable] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        コンポーネント失敗時の統一ハンドリング
        
        Args:
            component_type: コンポーネントタイプ
            component_name: コンポーネント名
            error: 発生したエラー
            fallback_func: フォールバック関数（PRODUCTION mode以外で使用）
            context: 実行コンテキスト情報
        
        Returns:
            フォールバック結果またはエラー再発生
        
        Raises:
            Exception: PRODUCTION modeまたはフォールバック失敗時
        """
        error_type = type(error).__name__
        error_message = str(error)
        context = context or {}
        
        # フォールバック使用記録を作成
        record = FallbackUsageRecord(
            timestamp=datetime.now(),
            component_type=component_type,
            component_name=component_name,
            error_type=error_type,
            error_message=error_message,
            fallback_used=False,
            system_mode=self.mode,
            execution_context=context
        )
        
        # PRODUCTION modeではフォールバック禁止
        if self.mode == SystemMode.PRODUCTION:
            self.logger.critical(
                f"PRODUCTION MODE: Component failure in {component_name} "
                f"({component_type.value}): {error_message}"
            )
            record.fallback_used = False
            self.usage_records.append(record)
            raise error
        
        # フォールバック関数が提供されていない場合
        if fallback_func is None:
            self.logger.error(
                f"No fallback available for {component_name} "
                f"({component_type.value}): {error_message}"
            )
            record.fallback_used = False
            self.usage_records.append(record)
            raise error
        
        # フォールバック実行
        try:
            self.logger.warning(
                f"FALLBACK ACTIVATED: {component_name} ({component_type.value}) "
                f"failed with {error_type}, using fallback function"
            )
            
            result = fallback_func()
            record.fallback_used = True
            self.usage_records.append(record)
            
            self.logger.info(
                f"Fallback successful for {component_name} "
                f"({component_type.value})"
            )
            
            return result
            
        except Exception as fallback_error:
            self.logger.error(
                f"Fallback failed for {component_name} "
                f"({component_type.value}): {str(fallback_error)}"
            )
            record.fallback_used = False
            self.usage_records.append(record)
            
            # 元のエラーを再発生
            raise error
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        フォールバック使用統計を取得
        
        Returns:
            使用統計情報
        """
        if not self.usage_records:
            return {
                "total_failures": 0,
                "fallback_usage_rate": 0.0,
                "by_component_type": {},
                "by_error_type": {},
                "records": []
            }
        
        total_failures = len(self.usage_records)
        successful_fallbacks = sum(1 for record in self.usage_records if record.fallback_used)
        
        # コンポーネントタイプ別統計
        by_component = {}
        for record in self.usage_records:
            comp_type = record.component_type.value
            if comp_type not in by_component:
                by_component[comp_type] = {"total": 0, "fallback_used": 0}
            by_component[comp_type]["total"] += 1
            if record.fallback_used:
                by_component[comp_type]["fallback_used"] += 1
        
        # エラータイプ別統計
        by_error = {}
        for record in self.usage_records:
            error_type = record.error_type
            if error_type not in by_error:
                by_error[error_type] = {"count": 0, "fallback_used": 0}
            by_error[error_type]["count"] += 1
            if record.fallback_used:
                by_error[error_type]["fallback_used"] += 1
        
        return {
            "total_failures": total_failures,
            "successful_fallbacks": successful_fallbacks,
            "fallback_usage_rate": successful_fallbacks / total_failures if total_failures > 0 else 0.0,
            "by_component_type": by_component,
            "by_error_type": by_error,
            "system_mode": self.mode.value,
            "records": [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "component_type": record.component_type.value,
                    "component_name": record.component_name,
                    "error_type": record.error_type,
                    "error_message": record.error_message,
                    "fallback_used": record.fallback_used,
                    "context": record.execution_context
                }
                for record in self.usage_records
            ]
        }
    
    def export_usage_report(self, output_path: Optional[Path] = None) -> Path:
        """
        フォールバック使用レポートをJSON形式でエクスポート
        
        Args:
            output_path: 出力パス（None の場合は reports/fallback/ に自動生成）
        
        Returns:
            出力ファイルパス
        """
        if output_path is None:
            # TODO(tag:phase1, rationale:organize fallback reports in dedicated directory)
            # reports/fallback/ ディレクトリに出力（outputディレクトリと分離）
            reports_dir = Path("reports/fallback")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = reports_dir / f"fallback_usage_report_{timestamp}.json"
        
        statistics = self.get_usage_statistics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Fallback usage report exported to: {output_path}")
        
        # 古いレポートファイルの自動削除（7日以上前）
        if "reports/fallback" in str(output_path):
            self._cleanup_old_reports(output_path.parent)
        
        return output_path
    
    def _cleanup_old_reports(self, reports_dir: Path, days_to_keep: int = 7):
        """
        古いレポートファイルの自動削除
        
        Args:
            reports_dir: レポートディレクトリパス
            days_to_keep: 保持日数（デフォルト7日）
        """
        # TODO(tag:phase1, rationale:prevent report directory bloat)
        try:
            from datetime import timedelta
            import os
            
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            cutoff_timestamp = cutoff_time.timestamp()
            
            if not reports_dir.exists():
                return
                
            for report_file in reports_dir.glob("fallback_usage_report_*.json"):
                try:
                    file_mtime = os.path.getmtime(report_file)
                    if file_mtime < cutoff_timestamp:
                        report_file.unlink()
                        self.logger.debug(f"🗑️ 古いフォールバックレポート削除: {report_file.name}")
                except (OSError, FileNotFoundError) as e:
                    self.logger.warning(f"⚠️ レポートファイル削除失敗 {report_file.name}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ 古いレポート削除処理失敗: {e}")
    
    def clear_usage_records(self):
        """使用記録をクリア（テスト用）"""
        self.usage_records.clear()
        self.logger.info("Fallback usage records cleared")


# グローバルインスタンス（プロジェクト全体で使用）
# TODO(tag:configuration, rationale:環境変数やconfigファイルからモードを設定できるように改善)
_global_fallback_policy = SystemFallbackPolicy(mode=SystemMode.DEVELOPMENT)


def get_fallback_policy() -> SystemFallbackPolicy:
    """
    グローバルフォールバックポリシーインスタンスを取得
    
    Returns:
        SystemFallbackPolicyインスタンス
    """
    return _global_fallback_policy


def set_system_mode(mode: SystemMode):
    """
    システムモードを変更
    
    Args:
        mode: 新しいシステムモード
    """
    global _global_fallback_policy
    _global_fallback_policy = SystemFallbackPolicy(mode=mode)
    logger.info(f"System mode changed to: {mode.value}")


# 使用例とテストコード（開発時のみ）
if __name__ == "__main__":
    # 基本的な使用例
    policy = SystemFallbackPolicy(SystemMode.DEVELOPMENT)
    
    def sample_fallback():
        return "fallback_result"
    
    try:
        # 正常なフォールバック使用例
        result = policy.handle_component_failure(
            component_type=ComponentType.DSSMS_CORE,
            component_name="TestComponent",
            error=ValueError("Test error"),
            fallback_func=sample_fallback,
            context={"test": True}
        )
        print(f"Fallback result: {result}")
        
        # 統計情報表示
        stats = policy.get_usage_statistics()
        print(f"Usage statistics: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")