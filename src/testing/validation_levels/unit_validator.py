"""
DSSMS Phase 3 Task 3.3: 単体機能検証
レベル2: 各DSSMSコンポーネントの単体機能検証

Author: GitHub Copilot Agent
Created: 2025-08-28
"""

import sys
import logging
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.testing.dssms_validation_framework import ValidationLevel, ValidationResult

class UnitValidator:
    """各DSSMSコンポーネントの単体機能検証"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_root = project_root
    
    def validate(self) -> ValidationResult:
        """単体検証実行"""
        errors = []
        warnings = []
        suggestions = []
        details = {}
        score = 0.0
        
        try:
            # DSSMSコンポーネントの検証
            components = [
                'hierarchical_ranking_system',
                'intelligent_switch_manager',
                'market_condition_monitor',
                'dssms_scheduler',
                'dssms_backtester_v2'
            ]
            
            total_score = 0.0
            
            for component in components:
                component_score = self._validate_component(component)
                details[f"{component}_score"] = component_score
                total_score += component_score
                
                if component_score < 0.5:
                    errors.append(f"コンポーネント {component} の検証失敗")
                elif component_score < 0.8:
                    warnings.append(f"コンポーネント {component} に改善の余地あり")
            
            score = total_score / len(components)
            success = score >= 0.70
            
        except Exception as e:
            errors.append(f"単体検証実行エラー: {str(e)}")
            success = False
            score = 0.0
        
        return ValidationResult(
            level=ValidationLevel.UNIT,
            test_name="unit_component_validation",
            timestamp=datetime.now(),
            success=success,
            execution_time=0.0,
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_component(self, component_name: str) -> float:
        """個別コンポーネントの検証"""
        try:
            module_path = f"src.dssms.{component_name}"
            module = importlib.import_module(module_path)
            
            # 基本的な属性・メソッドの存在確認
            score = 0.0
            
            # 1. クラスの存在確認 (30%)
            main_class = self._find_main_class(module, component_name)
            if main_class:
                score += 0.30
            
            # 2. 初期化の成功 (40%)
            if main_class:
                try:
                    # サンプル初期化テスト
                    if component_name == 'hierarchical_ranking_system':
                        instance = main_class()
                    elif component_name == 'intelligent_switch_manager':
                        instance = main_class()
                    elif component_name == 'market_condition_monitor':
                        instance = main_class()
                    elif component_name == 'dssms_scheduler':
                        instance = main_class()
                    elif component_name == 'dssms_backtester_v2':
                        instance = main_class()
                    else:
                        instance = main_class()
                    
                    score += 0.40
                    
                    # 3. 主要メソッドの存在確認 (30%)
                    method_score = self._check_main_methods(instance, component_name)
                    score += method_score * 0.30
                    
                except Exception as e:
                    self.logger.warning(f"コンポーネント {component_name} 初期化失敗: {e}")
            
            return score
            
        except ImportError as e:
            self.logger.warning(f"コンポーネント {component_name} インポート失敗: {e}")
            return 0.0
        except Exception as e:
            self.logger.warning(f"コンポーネント {component_name} 検証エラー: {e}")
            return 0.0
    
    def _find_main_class(self, module, component_name: str):
        """メインクラスの特定"""
        # コンポーネント名からクラス名を推定
        class_mappings = {
            'hierarchical_ranking_system': 'HierarchicalRankingSystem',
            'intelligent_switch_manager': 'IntelligentSwitchManager',
            'market_condition_monitor': 'MarketConditionMonitor',
            'dssms_scheduler': 'DSSMSScheduler',
            'dssms_backtester_v2': 'DSSMSBacktesterV2'
        }
        
        expected_class = class_mappings.get(component_name)
        
        if expected_class and hasattr(module, expected_class):
            return getattr(module, expected_class)
        
        # フォールバック: モジュール内のクラスを探索
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and not attr_name.startswith('_'):
                return attr
        
        return None
    
    def _check_main_methods(self, instance, component_name: str) -> float:
        """主要メソッドの存在確認"""
        expected_methods = {
            'hierarchical_ranking_system': ['rank_stocks', 'calculate_scores'],
            'intelligent_switch_manager': ['should_switch', 'evaluate_switch_condition'],
            'market_condition_monitor': ['analyze_market_condition', 'get_current_condition'],
            'dssms_scheduler': ['schedule_task', 'execute_scheduled_tasks'],
            'dssms_backtester_v2': ['run_backtest', 'calculate_performance']
        }
        
        methods = expected_methods.get(component_name, [])
        
        if not methods:
            return 1.0  # メソッドリストがない場合は満点
        
        existing_methods = 0
        
        for method in methods:
            if hasattr(instance, method) and callable(getattr(instance, method)):
                existing_methods += 1
        
        return existing_methods / len(methods) if methods else 1.0
