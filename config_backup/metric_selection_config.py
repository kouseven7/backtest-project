"""
Module: Metric Selection Configuration
File: metric_selection_config.py
Description: 
  重要指標選定システムの設定管理モジュール
  分析手法のパラメータ、閾値設定、出力オプションを管理
  2-1-2「重要指標選定システム」の設定コンポーネント

Author: imega
Created: 2025-07-10
Modified: 2025-07-10

Dependencies:
  - json
  - os
  - typing
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

class MetricSelectionConfig:
    """重要指標選定システムの設定管理クラス"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        設定管理クラスの初期化
        
        Args:
            config_dir: 設定ファイル保存ディレクトリ
        """
        if config_dir is None:
            project_root = Path(__file__).parent.parent
            self.config_dir = project_root / "logs" / "metric_selection" / "config"
        else:
            self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "metric_selection_config.json"
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """設定の読み込みまたは作成"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"設定読み込みエラー: {e}")
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """デフォルト設定の作成"""
        config = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            
            # 分析対象指標
            "target_metrics": {
                "core_metrics": [
                    "sharpe_ratio",
                    "sortino_ratio", 
                    "calmar_ratio",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor",
                    "total_return",
                    "volatility",
                    "expectancy"
                ],
                "additional_metrics": [
                    "recovery_factor",
                    "consistency_ratio",
                    "risk_adjusted_return",
                    "avg_holding_period",
                    "max_consecutive_losses",
                    "downside_deviation",
                    "var_95",
                    "tail_ratio"
                ],
                "target_variable": "sharpe_ratio"  # 目標指標
            },
            
            # 相関分析設定
            "correlation_analysis": {
                "enabled": True,
                "methods": ["pearson", "spearman", "kendall"],
                "primary_method": "pearson",
                "min_correlation_threshold": 0.3,
                "significance_level": 0.05,
                "confidence_interval": 0.95
            },
            
            # 回帰分析設定
            "regression_analysis": {
                "enabled": True,
                "methods": ["linear", "ridge", "lasso"],
                "primary_method": "ridge",
                "alpha": 1.0,
                "cross_validation_folds": 5,
                "min_r2_threshold": 0.1,
                "include_interactions": True,
                "max_interaction_features": 5
            },
            
            # 特徴量選択設定
            "feature_selection": {
                "enabled": True,
                "methods": ["f_regression", "mutual_info", "variance_threshold", "rfe", "lasso"],
                "k_best": 8,
                "score_threshold": 0.1,
                "variance_threshold": 0.01,
                "stability_threshold": 0.3
            },
            
            # データ品質要件
            "data_requirements": {
                "min_strategies": 3,
                "min_data_points": 20,
                "min_trend_types": 2,
                "data_quality_threshold": 0.7,
                "outlier_detection_method": "iqr",
                "outlier_threshold": 1.5
            },
            
            # 統合スコア設定
            "integration_weights": {
                "correlation_weight": 0.4,
                "regression_weight": 0.4,
                "feature_selection_weight": 0.2,
                "stability_weight": 0.3,
                "significance_weight": 0.3,
                "consistency_weight": 0.4
            },
            
            # 出力設定
            "output_settings": {
                "top_k_metrics": 8,
                "min_confidence_level": "medium",
                "include_statistical_tests": True,
                "generate_detailed_report": True,
                "save_intermediate_results": True,
                "export_formats": ["json", "csv", "markdown"]
            },
            
            # ログ設定
            "logging": {
                "log_level": "INFO",
                "log_to_file": True,
                "log_file": "metric_selection.log"
            }
        }
        
        # 設定を保存
        self.save_config(config)
        return config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """設定の保存"""
        if config is None:
            config = self.config
        
        config["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"設定保存エラー: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """設定値の取得"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        """設定値の更新"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()
    
    def get_analysis_methods(self) -> List[str]:
        """有効な分析手法のリストを取得"""
        methods = []
        
        if self.get("correlation_analysis.enabled", True):
            methods.append("correlation")
        
        if self.get("regression_analysis.enabled", True):
            methods.append("regression")
        
        if self.get("feature_selection.enabled", True):
            methods.append("feature_selection")
        
        return methods
    
    def get_target_metrics(self) -> List[str]:
        """分析対象指標の取得"""
        core = self.get("target_metrics.core_metrics", [])
        additional = self.get("target_metrics.additional_metrics", [])
        return core + additional
    
    def get_target_variable(self) -> str:
        """目標指標の取得"""
        return self.get("target_metrics.target_variable", "sharpe_ratio")
    
    def validate_config(self) -> List[str]:
        """設定の妥当性検証"""
        errors = []
        
        # 必須キーの確認
        required_keys = [
            "target_metrics.core_metrics",
            "target_metrics.target_variable",
            "correlation_analysis.enabled",
            "regression_analysis.enabled",
            "feature_selection.enabled"
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                errors.append(f"必須設定が不足: {key}")
        
        # 重みの合計確認
        weights = self.get("integration_weights", {})
        total_weight = sum([
            weights.get("stability_weight", 0),
            weights.get("significance_weight", 0),
            weights.get("consistency_weight", 0)
        ])
        
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"統合重みの合計が1.0でない: {total_weight}")
        
        # 閾値の範囲確認
        correlation_threshold = self.get("correlation_analysis.min_correlation_threshold", 0)
        if not 0 <= correlation_threshold <= 1:
            errors.append("相関閾値が範囲外")
        
        significance_level = self.get("correlation_analysis.significance_level", 0.05)
        if not 0 < significance_level <= 1:
            errors.append("有意水準が範囲外")
        
        return errors

# グローバル設定インスタンス
_config_instance = None

def get_config() -> MetricSelectionConfig:
    """グローバル設定インスタンスの取得"""
    global _config_instance
    if _config_instance is None:
        _config_instance = MetricSelectionConfig()
    return _config_instance

def reset_config():
    """設定のリセット"""
    global _config_instance
    _config_instance = None

# 使用例とテスト
if __name__ == "__main__":
    # 設定のテスト
    config = MetricSelectionConfig()
    
    print("=== 設定テスト ===")
    print(f"目標指標: {config.get_target_variable()}")
    print(f"分析手法: {config.get_analysis_methods()}")
    print(f"対象指標数: {len(config.get_target_metrics())}")
    
    # 設定検証
    errors = config.validate_config()
    if errors:
        print(f"設定エラー: {errors}")
    else:
        print("設定は有効です")
    
    # 設定値の取得テスト
    print(f"\n相関分析設定:")
    print(f"  有効: {config.get('correlation_analysis.enabled')}")
    print(f"  手法: {config.get('correlation_analysis.primary_method')}")
    print(f"  閾値: {config.get('correlation_analysis.min_correlation_threshold')}")
    
    print("\n設定テスト完了")
