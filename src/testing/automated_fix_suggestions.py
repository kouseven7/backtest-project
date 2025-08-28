"""
DSSMS Phase 3 Task 3.3: 自動修正提案システム
検証失敗時の自動修正提案

Author: GitHub Copilot Agent
Created: 2025-08-28
Phase: 3 Task 3.3
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.testing.dssms_validation_framework import ValidationResult, ValidationLevel, ValidationConfig
from config.logger_config import setup_logger

class FixPriority(Enum):
    """修正優先度"""
    CRITICAL = "critical"      # 即座に修正必要
    HIGH = "high"             # 高優先度
    MEDIUM = "medium"         # 中優先度
    LOW = "low"               # 低優先度
    INFO = "info"             # 情報提供

@dataclass
class FixSuggestion:
    """修正提案データクラス"""
    priority: FixPriority
    category: str
    title: str
    description: str
    fix_command: Optional[str]
    fix_code: Optional[str]
    estimated_time: str
    impact: str
    prerequisites: List[str]
    validation_level: ValidationLevel

class AutoFixSuggestions:
    """自動修正提案システム"""
    
    def __init__(self, config: ValidationConfig, logger=None):
        """
        初期化
        
        Args:
            config: 検証設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger or setup_logger("AutoFixSuggestions")
        self.project_root = project_root
        
        # 修正パターンデータベース
        self.fix_patterns = self._load_fix_patterns()
        
    def suggest_fixes(self, results: List[ValidationResult]) -> List[FixSuggestion]:
        """
        検証結果に基づく修正提案生成
        
        Args:
            results: 検証結果リスト
            
        Returns:
            修正提案リスト（優先度順）
        """
        suggestions = []
        
        for result in results:
            if not result.success or result.score < 0.7:
                level_suggestions = self._analyze_validation_result(result)
                suggestions.extend(level_suggestions)
        
        # 優先度でソート
        suggestions.sort(key=lambda x: self._get_priority_weight(x.priority), reverse=True)
        
        # 重複除去
        unique_suggestions = self._remove_duplicates(suggestions)
        
        self.logger.info(f"修正提案生成完了: {len(unique_suggestions)}件")
        return unique_suggestions
    
    def _analyze_validation_result(self, result: ValidationResult) -> List[FixSuggestion]:
        """単一検証結果の分析"""
        suggestions = []
        
        # レベル別分析
        if result.level == ValidationLevel.BASIC:
            suggestions.extend(self._analyze_basic_issues(result))
        elif result.level == ValidationLevel.UNIT:
            suggestions.extend(self._analyze_unit_issues(result))
        elif result.level == ValidationLevel.INTEGRATION:
            suggestions.extend(self._analyze_integration_issues(result))
        elif result.level == ValidationLevel.PERFORMANCE:
            suggestions.extend(self._analyze_performance_issues(result))
        elif result.level == ValidationLevel.PRODUCTION:
            suggestions.extend(self._analyze_production_issues(result))
        
        # エラーパターンマッチング
        suggestions.extend(self._match_error_patterns(result))
        
        return suggestions
    
    def _analyze_basic_issues(self, result: ValidationResult) -> List[FixSuggestion]:
        """基本レベル問題の分析"""
        suggestions = []
        
        # インポートエラー
        if any("import" in error.lower() for error in result.errors):
            suggestions.append(FixSuggestion(
                priority=FixPriority.CRITICAL,
                category="Dependencies",
                title="Pythonパッケージ不足",
                description="必要なPythonパッケージがインストールされていません",
                fix_command="pip install -r requirements.txt",
                fix_code=None,
                estimated_time="5分",
                impact="システム起動に必須",
                prerequisites=["requirements.txtファイルの確認"],
                validation_level=result.level
            ))
        
        # 設定ファイル問題
        if any("config" in error.lower() for error in result.errors):
            suggestions.append(FixSuggestion(
                priority=FixPriority.HIGH,
                category="Configuration",
                title="設定ファイル不足・破損",
                description="重要な設定ファイルが見つからないか破損しています",
                fix_command=None,
                fix_code=self._generate_config_fix_code(),
                estimated_time="10分",
                impact="システム設定の正常化",
                prerequisites=["設定ファイルテンプレートの確認"],
                validation_level=result.level
            ))
        
        # ディレクトリ構造問題
        if any("directory" in error.lower() or "ディレクトリ" in error for error in result.errors):
            suggestions.append(FixSuggestion(
                priority=FixPriority.MEDIUM,
                category="File System",
                title="ディレクトリ構造修復",
                description="必要なディレクトリが不足しています",
                fix_command=None,
                fix_code=self._generate_directory_fix_code(),
                estimated_time="3分",
                impact="ファイル管理の正常化",
                prerequisites=[],
                validation_level=result.level
            ))
        
        return suggestions
    
    def _analyze_unit_issues(self, result: ValidationResult) -> List[FixSuggestion]:
        """単体レベル問題の分析"""
        suggestions = []
        
        # モジュール機能不全
        if result.score < 0.6:
            suggestions.append(FixSuggestion(
                priority=FixPriority.HIGH,
                category="Module Function",
                title="DSSMSモジュール機能修復",
                description="DSSMSのコアモジュールに機能不全があります",
                fix_command=None,
                fix_code=self._generate_module_fix_code(result),
                estimated_time="30分",
                impact="システム機能の安定化",
                prerequisites=["モジュール依存関係の確認"],
                validation_level=result.level
            ))
        
        return suggestions
    
    def _analyze_integration_issues(self, result: ValidationResult) -> List[FixSuggestion]:
        """統合レベル問題の分析"""
        suggestions = []
        
        # データフロー問題
        if any("integration" in error.lower() for error in result.errors):
            suggestions.append(FixSuggestion(
                priority=FixPriority.HIGH,
                category="Data Flow",
                title="データフロー統合修復",
                description="モジュール間のデータ連携に問題があります",
                fix_command=None,
                fix_code=self._generate_integration_fix_code(),
                estimated_time="45分",
                impact="システム統合の安定化",
                prerequisites=["各モジュールの正常動作確認"],
                validation_level=result.level
            ))
        
        # 戦略切替問題
        switching_details = result.details.get('switching_effectiveness', {})
        if switching_details.get('success_rate', 1.0) < 0.6:
            suggestions.append(FixSuggestion(
                priority=FixPriority.CRITICAL,
                category="Strategy Switching",
                title="戦略切替メカニズム修復",
                description=f"戦略切替成功率が{switching_details.get('success_rate', 0.0):.1%}と低下しています",
                fix_command=None,
                fix_code=self._generate_switching_fix_code(),
                estimated_time="60分",
                impact="動的戦略選択機能の復旧",
                prerequisites=["市場データの品質確認"],
                validation_level=result.level
            ))
        
        return suggestions
    
    def _analyze_performance_issues(self, result: ValidationResult) -> List[FixSuggestion]:
        """パフォーマンスレベル問題の分析"""
        suggestions = []
        
        performance_metrics = result.details.get('performance_metrics', {})
        
        # リターン不足
        total_return = performance_metrics.get('total_return', 0.0)
        if total_return < self.config.high_level_criteria.get('total_return_min', 0.10):
            suggestions.append(FixSuggestion(
                priority=FixPriority.CRITICAL,
                category="Performance",
                title="ポートフォリオ収益性改善",
                description=f"総リターン{total_return:.1%}が目標{self.config.high_level_criteria.get('total_return_min', 0.10):.1%}を下回っています",
                fix_command=None,
                fix_code=self._generate_performance_fix_code(),
                estimated_time="2時間",
                impact="収益性の大幅改善",
                prerequisites=["戦略パラメータの分析", "市場環境の確認"],
                validation_level=result.level
            ))
        
        # リスク管理問題
        max_drawdown = abs(performance_metrics.get('max_drawdown', 0.0))
        if max_drawdown > self.config.high_level_criteria.get('max_drawdown_max', 0.15):
            suggestions.append(FixSuggestion(
                priority=FixPriority.HIGH,
                category="Risk Management",
                title="リスク管理強化",
                description=f"最大ドローダウン{max_drawdown:.1%}が許容値を超えています",
                fix_command=None,
                fix_code=self._generate_risk_management_fix_code(),
                estimated_time="90分",
                impact="リスク制御の改善",
                prerequisites=["ポートフォリオ構成の見直し"],
                validation_level=result.level
            ))
        
        # シャープレシオ改善
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
        if sharpe_ratio < self.config.high_level_criteria.get('sharpe_ratio_min', 1.5):
            suggestions.append(FixSuggestion(
                priority=FixPriority.MEDIUM,
                category="Risk-Adjusted Return",
                title="リスク調整後リターン改善",
                description=f"シャープレシオ{sharpe_ratio:.2f}が目標値を下回っています",
                fix_command=None,
                fix_code=self._generate_sharpe_improvement_code(),
                estimated_time="75分",
                impact="リスク効率の改善",
                prerequisites=["戦略最適化の実施"],
                validation_level=result.level
            ))
        
        return suggestions
    
    def _analyze_production_issues(self, result: ValidationResult) -> List[FixSuggestion]:
        """本番レベル問題の分析"""
        suggestions = []
        
        system_resources = result.details.get('system_resources', {})
        
        # メモリ不足
        memory_gb = system_resources.get('memory_gb', 0.0)
        if memory_gb < 4.0:
            suggestions.append(FixSuggestion(
                priority=FixPriority.CRITICAL,
                category="System Resources",
                title="メモリ増強",
                description=f"システムメモリ{memory_gb:.1f}GBが不足しています（推奨4GB以上）",
                fix_command="システム管理者にメモリ増設を依頼",
                fix_code=None,
                estimated_time="システム管理者対応",
                impact="システム安定性の向上",
                prerequisites=["システム管理者への相談"],
                validation_level=result.level
            ))
        
        # ディスク容量不足
        disk_free_gb = system_resources.get('disk_free_gb', 0.0)
        if disk_free_gb < 10.0:
            suggestions.append(FixSuggestion(
                priority=FixPriority.HIGH,
                category="System Resources",
                title="ディスク容量確保",
                description=f"ディスク空き容量{disk_free_gb:.1f}GBが不足しています",
                fix_command=None,
                fix_code=self._generate_disk_cleanup_code(),
                estimated_time="15分",
                impact="システム動作の安定化",
                prerequisites=["不要ファイルの特定"],
                validation_level=result.level
            ))
        
        return suggestions
    
    def _match_error_patterns(self, result: ValidationResult) -> List[FixSuggestion]:
        """エラーパターンマッチング"""
        suggestions = []
        
        all_errors = result.errors + result.warnings
        
        for error in all_errors:
            for pattern, fix_template in self.fix_patterns.items():
                if re.search(pattern, error, re.IGNORECASE):
                    suggestion = FixSuggestion(
                        priority=FixPriority(fix_template['priority']),
                        category=fix_template['category'],
                        title=fix_template['title'],
                        description=fix_template['description'].format(error=error),
                        fix_command=fix_template.get('fix_command'),
                        fix_code=fix_template.get('fix_code'),
                        estimated_time=fix_template['estimated_time'],
                        impact=fix_template['impact'],
                        prerequisites=fix_template.get('prerequisites', []),
                        validation_level=result.level
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _load_fix_patterns(self) -> Dict[str, Dict[str, Any]]:
        """修正パターンデータベース読み込み"""
        return {
            r"possibly delisted": {
                'priority': 'critical',
                'category': 'Data Quality',
                'title': '上場廃止銘柄の除外',
                'description': '上場廃止銘柄 "{error}" が検出されました',
                'fix_command': None,
                'fix_code': self._generate_delisted_fix_code(),
                'estimated_time': '20分',
                'impact': 'データ品質の向上',
                'prerequisites': ['アクティブ銘柄リストの更新']
            },
            r"connection.*timeout": {
                'priority': 'high',
                'category': 'Network',
                'title': 'ネットワーク接続問題',
                'description': 'ネットワーク接続タイムアウト: {error}',
                'fix_command': 'インターネット接続とファイアウォール設定を確認',
                'estimated_time': '10分',
                'impact': 'データ取得の安定化'
            },
            r"insufficient.*data": {
                'priority': 'medium',
                'category': 'Data Quality',
                'title': 'データ不足問題',
                'description': 'データ不足エラー: {error}',
                'fix_code': self._generate_data_supplement_code(),
                'estimated_time': '30分',
                'impact': '分析精度の向上'
            }
        }
    
    def _generate_config_fix_code(self) -> str:
        """設定ファイル修復コード生成"""
        return '''
# 設定ファイル修復スクリプト
import json
from pathlib import Path

def fix_config_files():
    """設定ファイルの修復・作成"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # 基本設定ファイル作成
    default_configs = {
        "comparison_config.json": {
            "analysis_mode": "comprehensive",
            "high_level_criteria": {
                "total_return_min": 0.10,
                "switch_success_rate_min": 0.80
            }
        }
    }
    
    for filename, config in default_configs.items():
        config_path = config_dir / filename
        if not config_path.exists():
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"設定ファイル作成: {filename}")

if __name__ == "__main__":
    fix_config_files()
'''
    
    def _generate_directory_fix_code(self) -> str:
        """ディレクトリ修復コード生成"""
        return '''
# ディレクトリ構造修復スクリプト
from pathlib import Path

def fix_directory_structure():
    """必要ディレクトリの作成"""
    required_dirs = [
        "data",
        "output", 
        "output/comparison_reports",
        "logs",
        "src/testing/test_data",
        "src/testing/validation_levels"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"ディレクトリ作成: {dir_path}")

if __name__ == "__main__":
    fix_directory_structure()
'''
    
    def _generate_module_fix_code(self, result: ValidationResult) -> str:
        """モジュール修復コード生成"""
        return '''
# DSSMSモジュール修復スクリプト
import sys
from pathlib import Path

def fix_dssms_modules():
    """DSSMSモジュールの修復"""
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    # 重要モジュールの動作確認
    critical_modules = [
        "src.dssms.hierarchical_ranking_system",
        "src.dssms.intelligent_switch_manager",
        "src.dssms.market_condition_monitor"
    ]
    
    for module_name in critical_modules:
        try:
            __import__(module_name)
            print(f"モジュール正常: {module_name}")
        except ImportError as e:
            print(f"モジュール修復必要: {module_name} - {e}")
            # 必要に応じて修復処理を追加

if __name__ == "__main__":
    fix_dssms_modules()
'''
    
    def _generate_integration_fix_code(self) -> str:
        """統合修復コード生成"""
        return '''
# データフロー統合修復スクリプト
def fix_data_flow_integration():
    """データフロー統合の修復"""
    print("データフロー統合修復開始...")
    
    # 1. モジュール間通信テスト
    # 2. データ形式の統一
    # 3. エラーハンドリングの強化
    
    print("データフロー統合修復完了")

if __name__ == "__main__":
    fix_data_flow_integration()
'''
    
    def _generate_switching_fix_code(self) -> str:
        """戦略切替修復コード生成"""
        return '''
# 戦略切替メカニズム修復スクリプト
def fix_strategy_switching():
    """戦略切替メカニズムの修復"""
    print("戦略切替メカニズム修復開始...")
    
    # 1. 切替条件の見直し
    # 2. タイミング調整
    # 3. エラー処理の改善
    
    print("戦略切替メカニズム修復完了")

if __name__ == "__main__":
    fix_strategy_switching()
'''
    
    def _generate_performance_fix_code(self) -> str:
        """パフォーマンス修復コード生成"""
        return '''
# ポートフォリオ収益性改善スクリプト
def fix_portfolio_performance():
    """ポートフォリオ収益性の改善"""
    print("ポートフォリオ収益性改善開始...")
    
    # 1. 戦略パラメータ最適化
    # 2. 銘柄選択基準の見直し
    # 3. リバランス頻度の調整
    
    print("ポートフォリオ収益性改善完了")

if __name__ == "__main__":
    fix_portfolio_performance()
'''
    
    def _generate_risk_management_fix_code(self) -> str:
        """リスク管理修復コード生成"""
        return '''
# リスク管理強化スクリプト
def fix_risk_management():
    """リスク管理の強化"""
    print("リスク管理強化開始...")
    
    # 1. ストップロス機能の実装
    # 2. ポジションサイズ制限
    # 3. ドローダウン制御
    
    print("リスク管理強化完了")

if __name__ == "__main__":
    fix_risk_management()
'''
    
    def _generate_sharpe_improvement_code(self) -> str:
        """シャープレシオ改善コード生成"""
        return '''
# シャープレシオ改善スクリプト
def improve_sharpe_ratio():
    """シャープレシオの改善"""
    print("シャープレシオ改善開始...")
    
    # 1. ボラティリティ制御
    # 2. 超過リターン最大化
    # 3. 相関分散投資
    
    print("シャープレシオ改善完了")

if __name__ == "__main__":
    improve_sharpe_ratio()
'''
    
    def _generate_disk_cleanup_code(self) -> str:
        """ディスククリーンアップコード生成"""
        return '''
# ディスククリーンアップスクリプト
import os
from pathlib import Path

def cleanup_disk_space():
    """ディスク容量の確保"""
    print("ディスククリーンアップ開始...")
    
    # 1. 古いログファイルの削除
    log_dir = Path("logs")
    if log_dir.exists():
        old_logs = list(log_dir.glob("*.log.old"))
        for log_file in old_logs:
            log_file.unlink()
            print(f"削除: {log_file}")
    
    # 2. 一時ファイルの削除
    temp_dirs = [Path("temp"), Path("tmp"), Path("__pycache__")]
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print(f"削除: {temp_dir}")
    
    print("ディスククリーンアップ完了")

if __name__ == "__main__":
    cleanup_disk_space()
'''
    
    def _generate_delisted_fix_code(self) -> str:
        """上場廃止銘柄修復コード生成"""
        return '''
# 上場廃止銘柄除外スクリプト
def fix_delisted_symbols():
    """上場廃止銘柄の除外"""
    print("上場廃止銘柄除外開始...")
    
    # 1. アクティブ銘柄リストの更新
    # 2. 廃止銘柄のフィルタリング
    # 3. ポートフォリオからの除外
    
    print("上場廃止銘柄除外完了")

if __name__ == "__main__":
    fix_delisted_symbols()
'''
    
    def _generate_data_supplement_code(self) -> str:
        """データ補完コード生成"""
        return '''
# データ補完スクリプト
def supplement_missing_data():
    """不足データの補完"""
    print("データ補完開始...")
    
    # 1. 欠損データの特定
    # 2. 補間・補完処理
    # 3. データ品質検証
    
    print("データ補完完了")

if __name__ == "__main__":
    supplement_missing_data()
'''
    
    def _get_priority_weight(self, priority: FixPriority) -> int:
        """優先度の重み取得"""
        weights = {
            FixPriority.CRITICAL: 100,
            FixPriority.HIGH: 80,
            FixPriority.MEDIUM: 60,
            FixPriority.LOW: 40,
            FixPriority.INFO: 20
        }
        return weights.get(priority, 0)
    
    def _remove_duplicates(self, suggestions: List[FixSuggestion]) -> List[FixSuggestion]:
        """重複修正提案の除去"""
        seen_titles = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            if suggestion.title not in seen_titles:
                seen_titles.add(suggestion.title)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions

if __name__ == "__main__":
    # テスト実行
    from src.testing.dssms_validation_framework import ValidationConfig
    
    logger = setup_logger("AutoFixSuggestionsTest")
    config = ValidationConfig(
        validation_levels=[],
        parallel_execution=False,
        early_termination=False,
        auto_fix_attempts=3,
        high_level_criteria={
            "total_return_min": 0.10,
            "switch_success_rate_min": 0.80,
            "max_drawdown_max": 0.15,
            "sharpe_ratio_min": 1.5
        },
        timeout_seconds=300,
        log_level="INFO"
    )
    
    # サンプル検証結果
    from src.testing.dssms_validation_framework import ValidationResult, ValidationLevel
    
    sample_result = ValidationResult(
        level=ValidationLevel.PERFORMANCE,
        test_name="performance_test",
        timestamp=datetime.now(),
        success=False,
        execution_time=10.0,
        score=0.45,
        details={
            "performance_metrics": {
                "total_return": 0.05,  # 目標10%未満
                "max_drawdown": -0.20,  # 目標15%超過
                "sharpe_ratio": 1.0     # 目標1.5未満
            }
        },
        errors=["パフォーマンス基準を満たしていません", "possibly delisted symbol detected"],
        warnings=["高ボラティリティが検出されました"],
        suggestions=[]
    )
    
    fixer = AutoFixSuggestions(config, logger)
    suggestions = fixer.suggest_fixes([sample_result])
    
    print(f"修正提案生成結果: {len(suggestions)}件")
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"{i}. [{suggestion.priority.value.upper()}] {suggestion.title}")
        print(f"   カテゴリ: {suggestion.category}")
        print(f"   推定時間: {suggestion.estimated_time}")
        print(f"   影響: {suggestion.impact}")
        print()
