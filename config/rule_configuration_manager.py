"""
Module: Rule Configuration Manager
File: rule_configuration_manager.py
Description: 
  3-1-3「選択ルールの抽象化（差し替え可能に）」設定管理
  ルール設定のJSON管理、動的ルール読み込み、設定バリデーション

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.strategy_selection_rule_engine
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ロガーの設定
logger = logging.getLogger(__name__)

class ConfigurationStatus(Enum):
    """設定ステータス"""
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    MISSING = "missing"

@dataclass
class RuleConfigurationMetadata:
    """ルール設定メタデータ"""
    name: str
    type: str
    priority: int
    enabled: bool
    created_at: datetime
    modified_at: datetime
    version: str = "1.0"
    author: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class RuleValidationResult:
    """ルール検証結果"""
    is_valid: bool
    status: ConfigurationStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

class RuleConfigurationManager:
    """
    ルール設定管理クラス
    
    JSON設定ファイルの管理、検証、動的読み込みを担当
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config/rule_engine")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定ファイルパス
        self.main_config_file = self.config_dir / "rules_config.json"
        self.schema_file = self.config_dir / "rule_schema.json"
        self.metadata_file = self.config_dir / "rule_metadata.json"
        
        # 内部データ
        self.configurations = {}
        self.metadata = {}
        self.schema = self._create_default_schema()
        
        # 初期化
        self._ensure_default_files()
        self._load_configurations()
        
        logger.info(f"RuleConfigurationManager initialized with {len(self.configurations)} configurations")
    
    def _create_default_schema(self) -> Dict[str, Any]:
        """デフォルトスキーマを作成"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["TrendBased", "ScoreBased", "RiskAdjusted", "Hybrid", "Configurable"]
                            },
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 50
                            },
                            "priority": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100
                            },
                            "enabled": {
                                "type": "boolean"
                            },
                            "config": {
                                "type": "object",
                                "properties": {
                                    "required_fields": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": ["strategy_scores", "trend_analysis", "risk_metrics"]
                                        }
                                    },
                                    "conditions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": ["trend_confidence", "trend_type", "score_threshold", "data_quality"]
                                                },
                                                "threshold": {"type": "number"},
                                                "operator": {
                                                    "type": "string",
                                                    "enum": [">=", ">", "<=", "<", "=="]
                                                },
                                                "value": {"type": "string"},
                                                "strategy": {"type": "string"}
                                            },
                                            "required": ["type"]
                                        }
                                    },
                                    "actions": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["select_top", "select_by_trend", "custom_formula"]
                                            },
                                            "count": {"type": "integer", "minimum": 1},
                                            "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                            "trend_mappings": {"type": "object"},
                                            "formula": {"type": "string"},
                                            "base_confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                        },
                                        "required": ["type"]
                                    }
                                }
                            }
                        },
                        "required": ["type", "name", "priority", "enabled"]
                    }
                },
                "global_settings": {
                    "type": "object",
                    "properties": {
                        "default_priority": {"type": "integer"},
                        "max_execution_time_ms": {"type": "number"},
                        "enable_parallel_execution": {"type": "boolean"},
                        "cache_enabled": {"type": "boolean"}
                    }
                }
            },
            "required": ["rules"]
        }
    
    def _ensure_default_files(self):
        """デフォルトファイルの確保"""
        # スキーマファイル
        if not self.schema_file.exists():
            with open(self.schema_file, 'w', encoding='utf-8') as f:
                json.dump(self.schema, f, indent=2, ensure_ascii=False)
        
        # メインコンフィグファイル
        if not self.main_config_file.exists():
            default_config = {
                "rules": [
                    {
                        "type": "TrendBased",
                        "name": "DefaultTrendBased",
                        "priority": 10,
                        "enabled": True,
                        "config": {}
                    },
                    {
                        "type": "ScoreBased", 
                        "name": "DefaultScoreBased",
                        "priority": 20,
                        "enabled": True,
                        "config": {}
                    },
                    {
                        "type": "Hybrid",
                        "name": "DefaultHybrid",
                        "priority": 15,
                        "enabled": True,
                        "config": {}
                    }
                ],
                "global_settings": {
                    "default_priority": 50,
                    "max_execution_time_ms": 5000,
                    "enable_parallel_execution": False,
                    "cache_enabled": True
                },
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(self.main_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        # メタデータファイル
        if not self.metadata_file.exists():
            default_metadata = {
                "configurations": {},
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(default_metadata, f, indent=2, ensure_ascii=False)
    
    def _load_configurations(self):
        """設定の読み込み"""
        try:
            # メイン設定の読み込み
            if self.main_config_file.exists():
                with open(self.main_config_file, 'r', encoding='utf-8') as f:
                    self.configurations = json.load(f)
            
            # メタデータの読み込み
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata_data = json.load(f)
                    self.metadata = metadata_data.get('configurations', {})
            
            # スキーマの読み込み
            if self.schema_file.exists():
                with open(self.schema_file, 'r', encoding='utf-8') as f:
                    self.schema = json.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            self.configurations = {}
            self.metadata = {}
    
    def validate_configuration(self, config: Dict[str, Any]) -> RuleValidationResult:
        """設定の検証"""
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # スキーマ検証
            validate(instance=config, schema=self.schema)
            
            # 追加検証
            rules = config.get('rules', [])
            rule_names = set()
            
            for rule in rules:
                rule_name = rule.get('name')
                
                # 重複名チェック
                if rule_name in rule_names:
                    errors.append(f"Duplicate rule name: {rule_name}")
                else:
                    rule_names.add(rule_name)
                
                # 優先度の妥当性
                priority = rule.get('priority', 50)
                if priority < 1 or priority > 100:
                    warnings.append(f"Rule {rule_name}: Priority {priority} is outside recommended range (1-100)")
                
                # 設定可能ルールの詳細検証
                if rule.get('type') == 'Configurable':
                    config_section = rule.get('config', {})
                    
                    # アクションの検証
                    actions = config_section.get('actions', {})
                    action_type = actions.get('type')
                    
                    if action_type == 'select_top':
                        if 'count' not in actions and 'threshold' not in actions:
                            warnings.append(f"Rule {rule_name}: select_top action should have count or threshold")
                    
                    elif action_type == 'select_by_trend':
                        if 'trend_mappings' not in actions:
                            errors.append(f"Rule {rule_name}: select_by_trend action requires trend_mappings")
                    
                    # 条件の検証
                    conditions = config_section.get('conditions', [])
                    for i, condition in enumerate(conditions):
                        condition_type = condition.get('type')
                        
                        if condition_type in ['trend_confidence', 'score_threshold', 'data_quality']:
                            if 'threshold' not in condition:
                                errors.append(f"Rule {rule_name}: Condition {i} requires threshold")
                        
                        elif condition_type == 'trend_type':
                            if 'value' not in condition:
                                errors.append(f"Rule {rule_name}: Condition {i} requires value")
            
            # グローバル設定の検証
            global_settings = config.get('global_settings', {})
            max_exec_time = global_settings.get('max_execution_time_ms', 5000)
            
            if max_exec_time < 100 or max_exec_time > 30000:
                warnings.append(f"max_execution_time_ms {max_exec_time} is outside recommended range (100-30000)")
            
            # 提案の生成
            if len(rules) == 0:
                suggestions.append("Consider adding at least one rule configuration")
            
            if len([r for r in rules if r.get('enabled', True)]) == 0:
                suggestions.append("No rules are enabled - at least one should be enabled")
            
            if not any(r.get('type') == 'Hybrid' for r in rules):
                suggestions.append("Consider adding a Hybrid rule for robust strategy selection")
            
            # 結果の判定
            if errors:
                status = ConfigurationStatus.INVALID
                is_valid = False
            elif warnings:
                status = ConfigurationStatus.PARTIAL
                is_valid = True
            else:
                status = ConfigurationStatus.VALID
                is_valid = True
                
        except ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            status = ConfigurationStatus.INVALID
            is_valid = False
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            status = ConfigurationStatus.INVALID
            is_valid = False
        
        return RuleValidationResult(
            is_valid=is_valid,
            status=status,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def save_configuration(self, config: Dict[str, Any], validate_before_save: bool = True) -> bool:
        """設定の保存"""
        try:
            # 保存前検証
            if validate_before_save:
                validation_result = self.validate_configuration(config)
                if not validation_result.is_valid:
                    logger.error(f"Configuration validation failed: {validation_result.errors}")
                    return False
            
            # タイムスタンプの追加
            config['last_updated'] = datetime.now().isoformat()
            
            # メイン設定の保存
            with open(self.main_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # メタデータの更新
            for rule in config.get('rules', []):
                rule_name = rule.get('name')
                if rule_name:
                    self.metadata[rule_name] = {
                        'type': rule.get('type'),
                        'priority': rule.get('priority'),
                        'enabled': rule.get('enabled'),
                        'modified_at': datetime.now().isoformat(),
                        'version': '1.0'
                    }
            
            # メタデータの保存
            metadata_data = {
                'configurations': self.metadata,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_data, f, indent=2, ensure_ascii=False)
            
            # 内部データの更新
            self.configurations = config
            
            logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_configuration(self) -> Dict[str, Any]:
        """設定の読み込み"""
        self._load_configurations()
        return self.configurations.copy()
    
    def add_rule_configuration(self, rule_config: Dict[str, Any]) -> bool:
        """ルール設定を追加"""
        try:
            # 単一ルールの検証
            temp_config = {'rules': [rule_config]}
            validation_result = self.validate_configuration(temp_config)
            
            if not validation_result.is_valid:
                logger.error(f"Rule configuration validation failed: {validation_result.errors}")
                return False
            
            # 既存設定に追加
            current_config = self.load_configuration()
            rules = current_config.get('rules', [])
            
            # 重複チェック
            rule_name = rule_config.get('name')
            existing_names = [r.get('name') for r in rules]
            
            if rule_name in existing_names:
                # 既存ルールの更新
                for i, rule in enumerate(rules):
                    if rule.get('name') == rule_name:
                        rules[i] = rule_config
                        break
            else:
                # 新規ルールの追加
                rules.append(rule_config)
            
            current_config['rules'] = rules
            return self.save_configuration(current_config)
            
        except Exception as e:
            logger.error(f"Failed to add rule configuration: {e}")
            return False
    
    def remove_rule_configuration(self, rule_name: str) -> bool:
        """ルール設定を削除"""
        try:
            current_config = self.load_configuration()
            rules = current_config.get('rules', [])
            
            # ルールの削除
            original_count = len(rules)
            rules = [r for r in rules if r.get('name') != rule_name]
            
            if len(rules) == original_count:
                logger.warning(f"Rule not found: {rule_name}")
                return False
            
            current_config['rules'] = rules
            
            # メタデータからも削除
            if rule_name in self.metadata:
                del self.metadata[rule_name]
            
            return self.save_configuration(current_config)
            
        except Exception as e:
            logger.error(f"Failed to remove rule configuration: {e}")
            return False
    
    def get_rule_configuration(self, rule_name: str) -> Optional[Dict[str, Any]]:
        """特定ルールの設定を取得"""
        current_config = self.load_configuration()
        rules = current_config.get('rules', [])
        
        for rule in rules:
            if rule.get('name') == rule_name:
                return rule.copy()
        
        return None
    
    def list_rule_configurations(self) -> List[Dict[str, Any]]:
        """全ルール設定のリストを取得"""
        current_config = self.load_configuration()
        return current_config.get('rules', [])
    
    def export_configuration(self, export_path: str, include_metadata: bool = True) -> bool:
        """設定のエクスポート"""
        try:
            export_data = self.load_configuration()
            
            if include_metadata:
                export_data['metadata'] = self.metadata
                export_data['schema'] = self.schema
            
            export_data['export_timestamp'] = datetime.now().isoformat()
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_configuration(self, import_path: str, validate_before_import: bool = True) -> bool:
        """設定のインポート"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # メタデータとスキーマを除去
            config_data = {k: v for k, v in import_data.items() 
                          if k not in ['metadata', 'schema', 'export_timestamp']}
            
            if validate_before_import:
                validation_result = self.validate_configuration(config_data)
                if not validation_result.is_valid:
                    logger.error(f"Import validation failed: {validation_result.errors}")
                    return False
            
            return self.save_configuration(config_data, validate_before_save=False)
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """設定サマリーを取得"""
        current_config = self.load_configuration()
        rules = current_config.get('rules', [])
        
        rule_types = {}
        enabled_count = 0
        priority_distribution = {}
        
        for rule in rules:
            rule_type = rule.get('type', 'Unknown')
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
            
            if rule.get('enabled', False):
                enabled_count += 1
            
            priority = rule.get('priority', 50)
            priority_range = f"{(priority // 10) * 10}-{(priority // 10) * 10 + 9}"
            priority_distribution[priority_range] = priority_distribution.get(priority_range, 0) + 1
        
        return {
            'total_rules': len(rules),
            'enabled_rules': enabled_count,
            'disabled_rules': len(rules) - enabled_count,
            'rule_types': rule_types,
            'priority_distribution': priority_distribution,
            'last_updated': current_config.get('last_updated'),
            'configuration_file': str(self.main_config_file),
            'validation_status': self.validate_configuration(current_config).status.value
        }

if __name__ == "__main__":
    # テスト用のサンプル実行
    config_manager = RuleConfigurationManager()
    
    # 設定サマリーの表示
    print("Configuration Summary:")
    summary = config_manager.get_configuration_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # カスタムルールの追加テスト
    custom_rule = {
        "type": "Configurable",
        "name": "TestCustomRule",
        "priority": 25,
        "enabled": True,
        "config": {
            "required_fields": ["strategy_scores", "trend_analysis"],
            "conditions": [
                {
                    "type": "trend_confidence",
                    "threshold": 0.8,
                    "operator": ">="
                }
            ],
            "actions": {
                "type": "select_top",
                "count": 2,
                "threshold": 0.7,
                "base_confidence": 0.8
            }
        }
    }
    
    # ルール追加
    if config_manager.add_rule_configuration(custom_rule):
        print("\nCustom rule added successfully")
    
    # 検証テスト
    current_config = config_manager.load_configuration()
    validation_result = config_manager.validate_configuration(current_config)
    
    print(f"\nValidation Result:")
    print(f"  Valid: {validation_result.is_valid}")
    print(f"  Status: {validation_result.status.value}")
    if validation_result.warnings:
        print(f"  Warnings: {validation_result.warnings}")
    if validation_result.suggestions:
        print(f"  Suggestions: {validation_result.suggestions}")
