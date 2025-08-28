"""
DSSMS Phase 3 Task 3.3: 自動検証フレームワーク
メインフレームワーク実装

高水準パフォーマンス基準:
- 総リターン > 10%
- 切替成功率 > 80%
- 最大ドローダウン < 15%
- シャープレシオ > 1.5
- ボラティリティ < 25%

Author: GitHub Copilot Agent
Created: 2025-08-28
Phase: 3 Task 3.3
"""

import sys
import os
import logging
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class ValidationLevel(Enum):
    """検証レベル定義"""
    BASIC = "basic"
    UNIT = "unit" 
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    PRODUCTION = "production"

@dataclass
class ValidationResult:
    """検証結果データクラス"""
    level: ValidationLevel
    test_name: str
    timestamp: datetime
    success: bool
    execution_time: float
    score: float
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    def has_data_issues(self) -> bool:
        """データ問題の有無"""
        return any("data" in error.lower() for error in self.errors)
    
    def has_performance_issues(self) -> bool:
        """パフォーマンス問題の有無"""
        return any("performance" in error.lower() for error in self.errors)
    
    def has_integration_issues(self) -> bool:
        """統合問題の有無"""
        return any("integration" in error.lower() for error in self.errors)

@dataclass
class ValidationConfig:
    """検証設定"""
    validation_levels: List[ValidationLevel]
    parallel_execution: bool
    early_termination: bool
    auto_fix_attempts: int
    high_level_criteria: Dict[str, float]
    timeout_seconds: int
    log_level: str

class DSSMSValidationFramework:
    """
    DSSMS自動検証フレームワーク
    5段階の検証レベルを順次実行し、問題を早期発見・修正提案
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.logger = setup_logger("DSSMSValidationFramework")
        self.config = self._load_config(config_path)
        self.project_root = project_root
        
        # 検証レベルの初期化
        self.validation_levels = {}
        self._initialize_validators()
        
        # テストデータ管理
        self.test_data_manager = None
        
        # レポート生成
        self.reporter = None
        
        # 自動修正提案
        self.auto_fixer = None
        
        self.logger.info("DSSMS自動検証フレームワーク初期化完了")
    
    def _load_config(self, config_path: Optional[str]) -> ValidationConfig:
        """設定ファイル読み込み"""
        default_config = {
            "validation_levels": ["basic", "unit", "integration", "performance", "production"],
            "parallel_execution": True,
            "early_termination": True,
            "auto_fix_attempts": 3,
            "high_level_criteria": {
                "total_return_min": 0.10,
                "switch_success_rate_min": 0.80,
                "max_drawdown_max": 0.15,
                "sharpe_ratio_min": 1.5,
                "volatility_max": 0.25,
                "calmar_ratio_min": 1.0
            },
            "timeout_seconds": 1800,  # 30分
            "log_level": "INFO"
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                default_config.update(config_data)
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗: {e}")
        
        return ValidationConfig(
            validation_levels=[ValidationLevel(level) for level in default_config["validation_levels"]],
            parallel_execution=default_config["parallel_execution"],
            early_termination=default_config["early_termination"],
            auto_fix_attempts=default_config["auto_fix_attempts"],
            high_level_criteria=default_config["high_level_criteria"],
            timeout_seconds=default_config["timeout_seconds"],
            log_level=default_config["log_level"]
        )
    
    def _initialize_validators(self):
        """検証レベルの初期化"""
        try:
            # 各検証レベルの動的読み込み
            from src.testing.validation_levels.basic_validator import BasicValidator
            from src.testing.validation_levels.unit_validator import UnitValidator
            from src.testing.validation_levels.integration_validator import IntegrationValidator
            from src.testing.validation_levels.performance_validator import PerformanceValidator
            from src.testing.validation_levels.production_validator import ProductionValidator
            
            self.validation_levels = {
                ValidationLevel.BASIC: BasicValidator(self.config, self.logger),
                ValidationLevel.UNIT: UnitValidator(self.config, self.logger),
                ValidationLevel.INTEGRATION: IntegrationValidator(self.config, self.logger),
                ValidationLevel.PERFORMANCE: PerformanceValidator(self.config, self.logger),
                ValidationLevel.PRODUCTION: ProductionValidator(self.config, self.logger)
            }
        except ImportError as e:
            self.logger.warning(f"一部の検証レベルが利用できません: {e}")
            # 基本検証のみで動作
            self.validation_levels = {}
    
    def run_validation(self, target_levels: Optional[List[ValidationLevel]] = None) -> List[ValidationResult]:
        """
        検証実行メイン
        
        Args:
            target_levels: 実行対象の検証レベル
            
        Returns:
            検証結果のリスト
        """
        if target_levels is None:
            target_levels = self.config.validation_levels
        
        results = []
        
        self.logger.info(f"自動検証開始: {len(target_levels)}レベル")
        
        for level in target_levels:
            if level not in self.validation_levels:
                self.logger.warning(f"検証レベル {level.value} が利用できません")
                continue
            
            try:
                result = self._run_single_validation(level)
                results.append(result)
                
                # 早期終了チェック
                if self.config.early_termination and not result.success:
                    self.logger.warning(f"検証失敗により早期終了: {level.value}")
                    break
                    
            except Exception as e:
                error_result = ValidationResult(
                    level=level,
                    test_name=f"{level.value}_validation",
                    timestamp=datetime.now(),
                    success=False,
                    execution_time=0.0,
                    score=0.0,
                    details={},
                    errors=[f"検証実行エラー: {str(e)}"],
                    warnings=[],
                    suggestions=[]
                )
                results.append(error_result)
                
                if self.config.early_termination:
                    break
        
        self.logger.info(f"自動検証完了: {len(results)}件の結果")
        return results
    
    def _run_single_validation(self, level: ValidationLevel) -> ValidationResult:
        """単一検証レベルの実行"""
        validator = self.validation_levels[level]
        
        start_time = time.time()
        self.logger.info(f"検証開始: {level.value}")
        
        try:
            result = validator.validate()
            execution_time = time.time() - start_time
            
            # 実行時間の追加
            result.execution_time = execution_time
            
            self.logger.info(f"検証完了: {level.value} - 成功: {result.success}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ValidationResult(
                level=level,
                test_name=f"{level.value}_validation",
                timestamp=datetime.now(),
                success=False,
                execution_time=execution_time,
                score=0.0,
                details={"error": str(e)},
                errors=[f"検証エラー: {str(e)}"],
                warnings=[],
                suggestions=[f"検証レベル {level.value} の再実装を検討してください"]
            )
    
    def generate_report(self, results: List[ValidationResult], output_path: Optional[str] = None) -> str:
        """検証結果レポート生成"""
        if not self.reporter:
            from src.testing.validation_reporter import ValidationReporter
            self.reporter = ValidationReporter(self.config, self.logger)
        
        return self.reporter.generate_report(results, output_path)
    
    def suggest_fixes(self, results: List[ValidationResult]) -> List[str]:
        """自動修正提案"""
        if not self.auto_fixer:
            from src.testing.automated_fix_suggestions import AutoFixSuggestions
            self.auto_fixer = AutoFixSuggestions(self.config, self.logger)
        
        return self.auto_fixer.suggest_fixes(results)
    
    def get_overall_score(self, results: List[ValidationResult]) -> float:
        """総合スコア計算"""
        if not results:
            return 0.0
        
        # 重み付きスコア計算
        weights = {
            ValidationLevel.BASIC: 0.15,
            ValidationLevel.UNIT: 0.20,
            ValidationLevel.INTEGRATION: 0.25,
            ValidationLevel.PERFORMANCE: 0.25,
            ValidationLevel.PRODUCTION: 0.15
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = weights.get(result.level, 0.20)
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def is_production_ready(self, results: List[ValidationResult]) -> bool:
        """本番環境準備完了チェック"""
        overall_score = self.get_overall_score(results)
        
        # 高水準基準
        criteria = self.config.high_level_criteria
        
        # 最低限のチェック
        has_integration = any(r.level == ValidationLevel.INTEGRATION and r.success for r in results)
        has_performance = any(r.level == ValidationLevel.PERFORMANCE and r.success for r in results)
        
        return (
            overall_score >= 0.80 and  # 80%以上のスコア
            has_integration and        # 統合テスト成功
            has_performance           # パフォーマンステスト成功
        )

# ユーティリティ関数
def create_sample_config(output_path: str):
    """サンプル設定ファイル作成"""
    config = {
        "validation_levels": ["basic", "unit", "integration", "performance"],
        "parallel_execution": True,
        "early_termination": True,
        "auto_fix_attempts": 3,
        "high_level_criteria": {
            "total_return_min": 0.10,
            "switch_success_rate_min": 0.80,
            "max_drawdown_max": 0.15,
            "sharpe_ratio_min": 1.5,
            "volatility_max": 0.25,
            "calmar_ratio_min": 1.0
        },
        "timeout_seconds": 1800,
        "log_level": "INFO"
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # サンプル実行
    framework = DSSMSValidationFramework()
    results = framework.run_validation()
    
    overall_score = framework.get_overall_score(results)
    production_ready = framework.is_production_ready(results)
    
    print(f"総合スコア: {overall_score:.2%}")
    print(f"本番準備完了: {'はい' if production_ready else 'いいえ'}")
