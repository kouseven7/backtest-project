"""
Module: Metric Normalization Configuration
File: metric_normalization_config.py
Description: 
  指標正規化システムの設定管理クラス
  正規化手法の設定、戦略別オーバーライド、設定の永続化を管理
  2-1-3「指標の正規化手法の設計」の設定コンポーネント

Author: imega
Created: 2025-07-10
Modified: 2025-07-10

Dependencies:
  - json
  - dataclasses
  - pathlib
  - typing
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

# ロガーの設定
logger = logging.getLogger(__name__)

class NormalizationMethod(Enum):
    """正規化手法の定義"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    RANK = "rank"
    CUSTOM = "custom"

@dataclass
class NormalizationParameters:
    """正規化パラメータの定義"""
    method: str = "min_max"
    target_range: tuple = (0.0, 1.0)
    outlier_handling: str = "clip"  # clip, remove, transform
    missing_value_strategy: str = "median"  # median, mean, drop, interpolate
    confidence_threshold: float = 0.85
    custom_function: Optional[str] = None
    preserve_zeros: bool = True
    apply_log_transform: bool = False
    
    def __post_init__(self):
        """データ検証"""
        if self.method not in [method.value for method in NormalizationMethod]:
            raise ValueError(f"Invalid normalization method: {self.method}")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

@dataclass
class MetricNormalizationSettings:
    """指標別正規化設定"""
    sharpe_ratio: NormalizationParameters = field(default_factory=lambda: NormalizationParameters(
        method="z_score",
        target_range=(-3.0, 3.0),
        outlier_handling="clip"
    ))
    
    sortino_ratio: NormalizationParameters = field(default_factory=lambda: NormalizationParameters(
        method="z_score",
        target_range=(-3.0, 3.0),
        outlier_handling="clip"
    ))
    
    profit_factor: NormalizationParameters = field(default_factory=lambda: NormalizationParameters(
        method="custom",
        target_range=(0.0, 1.0),
        custom_function="profit_factor_transform",
        outlier_handling="clip"
    ))
    
    win_rate: NormalizationParameters = field(default_factory=lambda: NormalizationParameters(
        method="min_max",
        target_range=(0.0, 1.0),
        preserve_zeros=False
    ))
    
    max_drawdown: NormalizationParameters = field(default_factory=lambda: NormalizationParameters(
        method="min_max",
        target_range=(0.0, 1.0),
        apply_log_transform=True,
        outlier_handling="clip"
    ))
    
    total_return: NormalizationParameters = field(default_factory=lambda: NormalizationParameters(
        method="robust",
        target_range=(0.0, 1.0),
        outlier_handling="transform"
    ))

@dataclass
class StrategyNormalizationOverride:
    """戦略別正規化オーバーライド設定"""
    strategy_name: str
    metric_overrides: Dict[str, NormalizationParameters] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1
    notes: str = ""

class MetricNormalizationConfig:
    """
    指標正規化システムの設定管理クラス
    
    正規化手法の設定、戦略別オーバーライド、設定の永続化を管理し、
    エラー耐性と拡張性を提供するメインの設定クラス
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初期化
        
        Args:
            config_file: 設定ファイルのパス（Noneの場合はデフォルト位置）
        """
        # パス設定
        if config_file is None:
            project_root = Path(__file__).parent.parent
            self.config_dir = project_root / "logs" / "metric_normalization" / "configs"
            self.config_file = self.config_dir / "normalization_config.json"
        else:
            self.config_file = Path(config_file)
            self.config_dir = self.config_file.parent
        
        # ディレクトリ作成
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # デフォルト設定
        self.global_settings = MetricNormalizationSettings()
        self.strategy_overrides: Dict[str, StrategyNormalizationOverride] = {}
        
        # システム設定
        self.system_config = {
            "version": "1.0.0",
            "auto_backup": True,
            "backup_retention_days": 30,
            "validation_enabled": True,
            "debug_mode": False,
            "integration_mode": "metric_selection",  # metric_selection, scoring, standalone
            "last_updated": datetime.now().isoformat()
        }
        
        # 設定の読み込み
        self.load_config()
        
        logger.info(f"MetricNormalizationConfig initialized: {self.config_file}")
    
    def get_normalization_parameters(self, 
                                   metric_name: str, 
                                   strategy_name: Optional[str] = None) -> NormalizationParameters:
        """
        指標とストラテジーに応じた正規化パラメータを取得
        
        Args:
            metric_name: 指標名
            strategy_name: 戦略名（Noneの場合はグローバル設定）
            
        Returns:
            NormalizationParameters: 正規化パラメータ
        """
        try:
            # 戦略別オーバーライドの確認
            if (strategy_name and 
                strategy_name in self.strategy_overrides and 
                self.strategy_overrides[strategy_name].enabled):
                
                override = self.strategy_overrides[strategy_name]
                if metric_name in override.metric_overrides:
                    logger.debug(f"Using strategy override for {strategy_name}.{metric_name}")
                    return override.metric_overrides[metric_name]
            
            # グローバル設定から取得
            if hasattr(self.global_settings, metric_name):
                return getattr(self.global_settings, metric_name)
            
            # デフォルト設定
            logger.warning(f"No specific settings for metric: {metric_name}, using default")
            return NormalizationParameters()
            
        except Exception as e:
            logger.error(f"Error getting normalization parameters: {e}")
            return NormalizationParameters()
    
    def add_strategy_override(self, 
                            strategy_name: str, 
                            metric_overrides: Dict[str, Dict[str, Any]], 
                            priority: int = 1, 
                            notes: str = "") -> bool:
        """
        戦略別オーバーライドの追加
        
        Args:
            strategy_name: 戦略名
            metric_overrides: 指標別オーバーライド設定
            priority: 優先度
            notes: 備考
            
        Returns:
            bool: 成功可否
        """
        try:
            # パラメータオブジェクトに変換
            converted_overrides = {}
            for metric, params in metric_overrides.items():
                if isinstance(params, dict):
                    converted_overrides[metric] = NormalizationParameters(**params)
                elif isinstance(params, NormalizationParameters):
                    converted_overrides[metric] = params
                else:
                    logger.warning(f"Invalid parameter type for {metric}: {type(params)}")
                    continue
            
            # オーバーライド作成
            override = StrategyNormalizationOverride(
                strategy_name=strategy_name,
                metric_overrides=converted_overrides,
                priority=priority,
                notes=notes
            )
            
            self.strategy_overrides[strategy_name] = override
            logger.info(f"Added strategy override for: {strategy_name}")
            
            # 自動保存
            if self.system_config.get("auto_backup", True):
                self.save_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy override: {e}")
            return False
    
    def update_global_settings(self, 
                             metric_name: str, 
                             parameters: Union[Dict[str, Any], NormalizationParameters]) -> bool:
        """
        グローバル設定の更新
        
        Args:
            metric_name: 指標名
            parameters: 正規化パラメータ
            
        Returns:
            bool: 成功可否
        """
        try:
            if isinstance(parameters, dict):
                parameters = NormalizationParameters(**parameters)
            
            if hasattr(self.global_settings, metric_name):
                setattr(self.global_settings, metric_name, parameters)
                logger.info(f"Updated global settings for: {metric_name}")
                
                # 自動保存
                if self.system_config.get("auto_backup", True):
                    self.save_config()
                
                return True
            else:
                logger.warning(f"Unknown metric name: {metric_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating global settings: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """
        設定の検証
        
        Returns:
            Dict[str, Any]: 検証結果
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # グローバル設定の検証
            metric_count = 0
            for metric_name in dir(self.global_settings):
                if not metric_name.startswith('_'):
                    metric_count += 1
                    param = getattr(self.global_settings, metric_name)
                    if not isinstance(param, NormalizationParameters):
                        validation_result["errors"].append(f"Invalid parameter type for {metric_name}")
                        validation_result["valid"] = False
            
            # 戦略オーバーライドの検証
            override_count = len(self.strategy_overrides)
            for strategy_name, override in self.strategy_overrides.items():
                if not override.enabled:
                    continue
                    
                for metric_name, param in override.metric_overrides.items():
                    if not isinstance(param, NormalizationParameters):
                        validation_result["errors"].append(
                            f"Invalid override parameter for {strategy_name}.{metric_name}"
                        )
                        validation_result["valid"] = False
            
            validation_result["summary"] = {
                "global_metrics": metric_count,
                "strategy_overrides": override_count,
                "enabled_overrides": sum(1 for o in self.strategy_overrides.values() if o.enabled)
            }
            
            logger.info(f"Config validation completed: {'PASS' if validation_result['valid'] else 'FAIL'}")
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"Config validation failed: {e}")
        
        return validation_result
    
    def save_config(self, backup: bool = True) -> bool:
        """
        設定の保存
        
        Args:
            backup: バックアップを作成するか
            
        Returns:
            bool: 成功可否
        """
        try:
            # バックアップ作成
            if backup and self.config_file.exists():
                backup_file = self.config_dir / f"normalization_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.config_file.rename(backup_file)
                logger.debug(f"Created backup: {backup_file}")
            
            # 設定データの準備
            config_data = {
                "system_config": self.system_config,
                "global_settings": self._settings_to_dict(self.global_settings),
                "strategy_overrides": {
                    name: {
                        "strategy_name": override.strategy_name,
                        "enabled": override.enabled,
                        "priority": override.priority,
                        "notes": override.notes,
                        "metric_overrides": {
                            metric: asdict(param) for metric, param in override.metric_overrides.items()
                        }
                    }
                    for name, override in self.strategy_overrides.items()
                }
            }
            
            # 更新日時の設定
            config_data["system_config"]["last_updated"] = datetime.now().isoformat()
            
            # ファイル保存
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Config saved successfully: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def load_config(self) -> bool:
        """
        設定の読み込み
        
        Returns:
            bool: 成功可否
        """
        try:
            if not self.config_file.exists():
                logger.info("Config file not found, using defaults")
                return self.save_config()
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # システム設定の読み込み
            if "system_config" in config_data:
                self.system_config.update(config_data["system_config"])
            
            # グローバル設定の読み込み
            if "global_settings" in config_data:
                self.global_settings = self._dict_to_settings(config_data["global_settings"])
            
            # 戦略オーバーライドの読み込み
            if "strategy_overrides" in config_data:
                self.strategy_overrides = {}
                for name, override_data in config_data["strategy_overrides"].items():
                    metric_overrides = {}
                    for metric, param_data in override_data.get("metric_overrides", {}).items():
                        metric_overrides[metric] = NormalizationParameters(**param_data)
                    
                    self.strategy_overrides[name] = StrategyNormalizationOverride(
                        strategy_name=override_data["strategy_name"],
                        metric_overrides=metric_overrides,
                        enabled=override_data.get("enabled", True),
                        priority=override_data.get("priority", 1),
                        notes=override_data.get("notes", "")
                    )
            
            logger.info(f"Config loaded successfully: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False
    
    def _settings_to_dict(self, settings: MetricNormalizationSettings) -> Dict[str, Any]:
        """設定オブジェクトを辞書に変換"""
        result = {}
        for attr_name in dir(settings):
            if not attr_name.startswith('_'):
                attr_value = getattr(settings, attr_name)
                if isinstance(attr_value, NormalizationParameters):
                    result[attr_name] = asdict(attr_value)
        return result
    
    def _dict_to_settings(self, settings_dict: Dict[str, Any]) -> MetricNormalizationSettings:
        """辞書から設定オブジェクトを作成"""
        settings = MetricNormalizationSettings()
        for attr_name, param_data in settings_dict.items():
            if hasattr(settings, attr_name):
                setattr(settings, attr_name, NormalizationParameters(**param_data))
        return settings
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        設定の要約を取得
        
        Returns:
            Dict[str, Any]: 設定要約
        """
        try:
            summary = {
                "system_info": {
                    "version": self.system_config.get("version", "unknown"),
                    "last_updated": self.system_config.get("last_updated", "unknown"),
                    "integration_mode": self.system_config.get("integration_mode", "standalone")
                },
                "global_metrics": [],
                "strategy_overrides": {},
                "validation": self.validate_config()
            }
            
            # グローバル指標の情報
            for metric_name in dir(self.global_settings):
                if not metric_name.startswith('_'):
                    param = getattr(self.global_settings, metric_name)
                    summary["global_metrics"].append({
                        "name": metric_name,
                        "method": param.method,
                        "target_range": param.target_range
                    })
            
            # 戦略オーバーライドの情報
            for strategy_name, override in self.strategy_overrides.items():
                summary["strategy_overrides"][strategy_name] = {
                    "enabled": override.enabled,
                    "priority": override.priority,
                    "metric_count": len(override.metric_overrides),
                    "notes": override.notes
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating config summary: {e}")
            return {"error": str(e)}

# 使用例とテスト用の関数
def create_sample_config() -> MetricNormalizationConfig:
    """サンプル設定の作成"""
    config = MetricNormalizationConfig()
    
    # サンプル戦略オーバーライドの追加
    config.add_strategy_override(
        "trend_following",
        {
            "sharpe_ratio": {
                "method": "robust",
                "target_range": (0.0, 1.0),
                "outlier_handling": "transform"
            }
        },
        notes="Trend following strategies need robust normalization"
    )
    
    return config

if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    config = create_sample_config()
    print("Configuration Summary:")
    print(json.dumps(config.get_config_summary(), indent=2, ensure_ascii=False))
