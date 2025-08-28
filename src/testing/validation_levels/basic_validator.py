"""
DSSMS Phase 3 Task 3.3: 基本動作検証
レベル1: システム起動・設定読み込み検証

Author: GitHub Copilot Agent
Created: 2025-08-28
"""

import sys
import os
import logging
import json
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.testing.dssms_validation_framework import ValidationLevel, ValidationResult

class BasicValidator:
    """基本的なシステム起動・設定読み込み検証"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_root = project_root
    
    def validate(self) -> ValidationResult:
        """基本検証実行"""
        errors = []
        warnings = []
        suggestions = []
        details = {}
        score = 0.0
        
        try:
            # 1. DSSMS重要モジュールのインポートチェック
            import_score = self._check_dssms_imports()
            details["import_check"] = import_score
            score += import_score * 0.25
            
            # 2. 設定ファイルの存在・読み込みチェック
            config_score = self._check_config_files()
            details["config_check"] = config_score
            score += config_score * 0.25
            
            # 3. データディレクトリの存在チェック
            directory_score = self._check_data_directories()
            details["directory_check"] = directory_score
            score += directory_score * 0.20
            
            # 4. ログシステムの動作チェック
            logging_score = self._check_logging_system()
            details["logging_check"] = logging_score
            score += logging_score * 0.15
            
            # 5. 重要な依存関係チェック
            dependency_score = self._check_essential_dependencies()
            details["dependency_check"] = dependency_score
            score += dependency_score * 0.15
            
            success = score >= 0.70  # 70%以上で成功
            
        except Exception as e:
            errors.append(f"基本検証実行エラー: {str(e)}")
            success = False
            score = 0.0
        
        return ValidationResult(
            level=ValidationLevel.BASIC,
            test_name="basic_system_validation",
            timestamp=datetime.now(),
            success=success,
            execution_time=0.0,  # フレームワークで設定
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _check_dssms_imports(self) -> float:
        """DSSMS重要モジュールのインポートチェック"""
        critical_modules = [
            "src.dssms.hierarchical_ranking_system",
            "src.dssms.intelligent_switch_manager",
            "src.dssms.market_condition_monitor",
            "src.dssms.dssms_scheduler",
            "src.dssms.dssms_backtester_v2"
        ]
        
        success_count = 0
        
        for module in critical_modules:
            try:
                importlib.import_module(module)
                success_count += 1
                self.logger.debug(f"モジュールインポート成功: {module}")
            except ImportError as e:
                self.logger.warning(f"モジュールインポート失敗: {module} - {e}")
        
        return success_count / len(critical_modules)
    
    def _check_config_files(self) -> float:
        """設定ファイルの存在・読み込みチェック"""
        config_files = [
            "config/dssms/dssms_config.json",
            "config/dssms/ranking_config.json",
            "config/dssms/intelligent_switch_config.json",
            "config/dssms/market_monitoring_config.json",
            "config/dssms/scheduler_config.json"
        ]
        
        success_count = 0
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    success_count += 1
                    self.logger.debug(f"設定ファイル読み込み成功: {config_file}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"設定ファイル読み込み失敗: {config_file} - {e}")
            else:
                self.logger.warning(f"設定ファイル不在: {config_file}")
        
        return success_count / len(config_files)
    
    def _check_data_directories(self) -> float:
        """データディレクトリの存在チェック"""
        required_dirs = [
            "data",
            "backtest_results",
            "logs",
            "config",
            "src/dssms"
        ]
        
        success_count = 0
        
        for directory in required_dirs:
            dir_path = self.project_root / directory
            
            if dir_path.exists() and dir_path.is_dir():
                success_count += 1
                self.logger.debug(f"ディレクトリ確認成功: {directory}")
            else:
                self.logger.warning(f"ディレクトリ不在: {directory}")
        
        return success_count / len(required_dirs)
    
    def _check_logging_system(self) -> float:
        """ログシステムの動作チェック"""
        try:
            from config.logger_config import setup_logger
            
            # テストログ実行
            test_logger = setup_logger("BasicValidatorTest")
            test_logger.info("ログシステム動作テスト")
            
            # ログディレクトリ確認
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                return 1.0
            else:
                return 0.5  # ログシステムは動作するがディレクトリなし
                
        except Exception as e:
            self.logger.warning(f"ログシステムエラー: {e}")
            return 0.0
    
    def _check_essential_dependencies(self) -> float:
        """重要な依存関係チェック"""
        essential_packages = [
            "pandas",
            "numpy", 
            "yfinance",
            "matplotlib",
            "openpyxl"
        ]
        
        success_count = 0
        
        for package in essential_packages:
            try:
                importlib.import_module(package)
                success_count += 1
                self.logger.debug(f"パッケージインポート成功: {package}")
            except ImportError:
                self.logger.warning(f"パッケージインポート失敗: {package}")
        
        return success_count / len(essential_packages)
