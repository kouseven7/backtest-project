"""
Module: Strategy Characteristics Manager
File: strategy_characteristics_manager.py
Description: 
  戦略特性メタデータスキーマ設計・管理モジュール
  各戦略のトレンド適性、ボラティリティ適性、パラメータ履歴を管理し、
  戦略選択・重み付けエンジンへのデータ提供を行います。

Author: imega
Created: 2025-07-08
Modified: 2025-07-08

Dependencies:
  - pandas
  - numpy
  - json
  - config.logger_config
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import logging
import numpy as np
import pandas as pd

# ロガーの設定
logger = logging.getLogger(__name__)

class StrategyCharacteristicsManager:
    """
    戦略特性メタデータの管理クラス
    
    各戦略の以下の情報を管理：
    - トレンド環境別の適性データ
    - ボラティリティ環境別の適性データ  
    - パラメータ最適化履歴
    - リスクプロファイル
    - 依存関係情報
    """
    
    def __init__(self, base_path: str = None):
        """
        戦略特性管理クラスの初期化
        
        Args:
            base_path: データ保存ベースパス（デフォルト: logs/strategy_characteristics）
        """
        if base_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_path = os.path.join(project_root, "logs", "strategy_characteristics")
        
        self.base_path = base_path
        self.metadata_path = os.path.join(base_path, "metadata")
        self.performance_path = os.path.join(base_path, "performance_data")
        self.param_history_path = os.path.join(base_path, "parameter_history")
        
        # ディレクトリ作成
        self._ensure_directories()
        
        # 戦略マッピング（既存の戦略クラス名）
        self.strategy_mapping = {
            "VWAPBounceStrategy": {
                "name": "VWAP反発戦略",
                "class_name": "VWAPBounceStrategy",
                "module": "strategies.VWAP_Bounce"
            },
            "VWAPBreakoutStrategy": {
                "name": "VWAPブレイクアウト戦略", 
                "class_name": "VWAPBreakoutStrategy",
                "module": "strategies.VWAP_Breakout"
            },
            "MomentumInvestingStrategy": {
                "name": "モメンタム投資戦略",
                "class_name": "MomentumInvestingStrategy", 
                "module": "strategies.Momentum_Investing"
            },
            "ContrarianStrategy": {
                "name": "逆張り戦略",
                "class_name": "ContrarianStrategy",
                "module": "strategies.contrarian_strategy"
            },
            "GCStrategy": {
                "name": "ゴールデンクロス戦略",
                "class_name": "GCStrategy",
                "module": "strategies.gc_strategy_signal"
            },
            "OpeningGapStrategy": {
                "name": "寄り付きギャップ戦略",
                "class_name": "OpeningGapStrategy",
                "module": "strategies.Opening_Gap"
            },
            "BreakoutStrategy": {
                "name": "ブレイクアウト戦略",
                "class_name": "BreakoutStrategy", 
                "module": "strategies.Breakout"
            }
        }
    
    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        for path in [self.metadata_path, self.performance_path, self.param_history_path]:
            os.makedirs(path, exist_ok=True)
            
        # README.mdの作成
        readme_path = os.path.join(self.base_path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("""# Strategy Characteristics Data

## ディレクトリ構造

- `metadata/`: 戦略特性メタデータ
- `performance_data/`: パフォーマンスデータ
- `parameter_history/`: パラメータ最適化履歴

## ファイル命名規則

- メタデータ: `{strategy_id}_characteristics.json`
- パラメータ履歴: `{strategy_id}_param_history.json`

## 作成日: {}
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    def create_strategy_metadata(self, 
                               strategy_id: str, 
                               trend_performance: Dict[str, Any],
                               volatility_performance: Optional[Dict[str, Any]] = None,
                               custom_params: Optional[Dict[str, Any]] = None,
                               include_param_history: bool = True) -> Dict[str, Any]:
        """
        戦略特性メタデータを作成
        
        Args:
            strategy_id: 戦略ID（戦略クラス名）
            trend_performance: トレンド別パフォーマンス
            volatility_performance: ボラティリティ別パフォーマンス
            custom_params: カスタムパラメータ
            include_param_history: パラメータ履歴を含めるか
            
        Returns:
            Dict[str, Any]: 作成されたメタデータ
        """
        # 戦略情報の取得
        strategy_info = self.strategy_mapping.get(strategy_id, {
            "name": strategy_id,
            "class_name": strategy_id,
            "module": "unknown"
        })
        
        metadata = {
            "schema_version": "2.0",
            "strategy_id": strategy_id,
            "strategy_name": strategy_info["name"],
            "strategy_class": strategy_info["class_name"],
            "strategy_module": strategy_info["module"],
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            
            # トレンド適性データ
            "trend_adaptability": {
                "uptrend": self._create_trend_metrics(trend_performance.get("uptrend", {})),
                "downtrend": self._create_trend_metrics(trend_performance.get("downtrend", {})),
                "range-bound": self._create_trend_metrics(trend_performance.get("range-bound", {}))
            },
            
            # ボラティリティ適性データ
            "volatility_adaptability": {},
            
            # リスクプロファイル
            "risk_profile": {
                "volatility_tolerance": self._calculate_volatility_tolerance(trend_performance),
                "max_position_size": custom_params.get("max_position_size", 0.2) if custom_params else 0.2,
                "holding_period_avg": self._calculate_avg_holding_period(trend_performance),
                "drawdown_tolerance": self._calculate_drawdown_tolerance(trend_performance),
                "preferred_volatility_range": self._calculate_preferred_volatility_range(volatility_performance)
            },
            
            # 依存関係情報
            "dependencies": {
                "required_indicators": self._extract_required_indicators(strategy_id),
                "market_data_requirements": ["daily_ohlcv"],
                "index_data_required": self._requires_index_data(strategy_id),
                "volatility_calculation_required": True,
                "additional_columns": self._get_additional_columns(strategy_id)
            },
            
            # データ品質情報
            "data_quality": {
                "sample_size_total": sum(data.get("sample_size", 0) for data in trend_performance.values()),
                "confidence_score": self._calculate_confidence_score(trend_performance),
                "last_validation": datetime.now().isoformat(),
                "volatility_validation_complete": volatility_performance is not None,
                "validation_status": "pending"
            },
            
            # カスタムパラメータ
            "custom_parameters": custom_params or {}
        }
        
        # ボラティリティ別適性データの追加
        if volatility_performance:
            metadata["volatility_adaptability"] = {
                "high_volatility": self._create_volatility_metrics(volatility_performance.get("high_volatility", {})),
                "medium_volatility": self._create_volatility_metrics(volatility_performance.get("medium_volatility", {})),
                "low_volatility": self._create_volatility_metrics(volatility_performance.get("low_volatility", {}))
            }
        
        # パラメータ履歴の設定
        if include_param_history:
            metadata["parameter_history"] = {
                "enabled": True,
                "history_file": f"{strategy_id}_param_history.json",
                "retention_days": 365,
                "auto_update": True,
                "version_control": True,
                "optimization_tracking": True
            }
            
            # 初期パラメータ履歴ファイルを作成
            self._initialize_param_history(strategy_id)
        
        return metadata
    
    def _create_trend_metrics(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """トレンド別メトリクス作成"""
        if not trend_data:
            return self._get_default_trend_metrics()
            
        return {
            "suitability_score": float(trend_data.get("suitability_score", 0.5)),
            "confidence_level": self._get_confidence_level(trend_data.get("suitability_score", 0.5)),
            "performance_metrics": {
                "sharpe_ratio": float(trend_data.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(trend_data.get("max_drawdown", 0.0)),
                "win_rate": float(trend_data.get("win_rate", 0.0)),
                "expectancy": float(trend_data.get("expectancy", 0.0)),
                "avg_return": float(trend_data.get("avg_return", 0.0)),
                "volatility": float(trend_data.get("volatility", 0.0)),
                "calmar_ratio": float(trend_data.get("calmar_ratio", 0.0)),
                "sortino_ratio": float(trend_data.get("sortino_ratio", 0.0))
            },
            "sample_size": int(trend_data.get("sample_size", 0)),
            "data_period": trend_data.get("data_period", "unknown"),
            "reliability": self._calculate_reliability(trend_data),
            "risk_characteristics": {
                "max_consecutive_losses": int(trend_data.get("max_consecutive_losses", 0)),
                "avg_holding_period": float(trend_data.get("avg_holding_period", 5.0)),
                "profit_factor": float(trend_data.get("profit_factor", 1.0))
            }
        }
    
    def _create_volatility_metrics(self, volatility_data: Dict[str, Any]) -> Dict[str, Any]:
        """ボラティリティ別メトリクス作成"""
        if not volatility_data:
            return self._get_default_volatility_metrics()
            
        return {
            "suitability_score": float(volatility_data.get("suitability_score", 0.5)),
            "confidence_level": self._get_confidence_level(volatility_data.get("suitability_score", 0.5)),
            "performance_metrics": {
                "sharpe_ratio": float(volatility_data.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(volatility_data.get("max_drawdown", 0.0)),
                "win_rate": float(volatility_data.get("win_rate", 0.0)),
                "expectancy": float(volatility_data.get("expectancy", 0.0)),
                "avg_return": float(volatility_data.get("avg_return", 0.0)),
                "volatility": float(volatility_data.get("volatility", 0.0)),
                "calmar_ratio": float(volatility_data.get("calmar_ratio", 0.0)),
                "max_holding_period": int(volatility_data.get("max_holding_period", 0))
            },
            "sample_size": int(volatility_data.get("sample_size", 0)),
            "data_period": volatility_data.get("data_period", "unknown"),
            "reliability": self._calculate_reliability(volatility_data),
            "risk_characteristics": {
                "position_sizing_multiplier": float(volatility_data.get("position_sizing_multiplier", 1.0)),
                "stop_loss_adjustment": float(volatility_data.get("stop_loss_adjustment", 1.0)),
                "entry_frequency": volatility_data.get("entry_frequency", "normal"),
                "volatility_threshold": float(volatility_data.get("volatility_threshold", 0.0))
            }
        }
    
    def _get_default_trend_metrics(self) -> Dict[str, Any]:
        """デフォルトのトレンドメトリクス"""
        return {
            "suitability_score": 0.5,
            "confidence_level": "low",
            "performance_metrics": {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "avg_return": 0.0,
                "volatility": 0.0,
                "calmar_ratio": 0.0,
                "sortino_ratio": 0.0
            },
            "sample_size": 0,
            "data_period": "unknown",
            "reliability": "low",
            "risk_characteristics": {
                "max_consecutive_losses": 0,
                "avg_holding_period": 5.0,
                "profit_factor": 1.0
            }
        }
    
    def _get_default_volatility_metrics(self) -> Dict[str, Any]:
        """デフォルトのボラティリティメトリクス"""
        return {
            "suitability_score": 0.5,
            "confidence_level": "low",
            "performance_metrics": {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "avg_return": 0.0,
                "volatility": 0.0,
                "calmar_ratio": 0.0,
                "max_holding_period": 0
            },
            "sample_size": 0,
            "data_period": "unknown",
            "reliability": "low",
            "risk_characteristics": {
                "position_sizing_multiplier": 1.0,
                "stop_loss_adjustment": 1.0,
                "entry_frequency": "normal",
                "volatility_threshold": 0.0
            }
        }
    
    def _get_confidence_level(self, score: float) -> str:
        """信頼度レベルを文字列で返す"""
        if score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _calculate_volatility_tolerance(self, trend_performance: Dict[str, Any]) -> str:
        """ボラティリティ耐性を計算"""
        volatilities = [data.get("volatility", 0) for data in trend_performance.values() if data.get("volatility")]
        if not volatilities:
            return "medium"
            
        avg_volatility = sum(volatilities) / len(volatilities)
        if avg_volatility > 0.3:
            return "high"
        elif avg_volatility > 0.15:
            return "medium"
        else:
            return "low"
    
    def _calculate_avg_holding_period(self, trend_performance: Dict[str, Any]) -> float:
        """平均保有期間を計算"""
        periods = [data.get("avg_holding_period", 5.0) for data in trend_performance.values() if data.get("avg_holding_period")]
        return sum(periods) / len(periods) if periods else 5.0
    
    def _calculate_drawdown_tolerance(self, trend_performance: Dict[str, Any]) -> float:
        """ドローダウン耐性を計算"""
        drawdowns = [data.get("max_drawdown", 0.05) for data in trend_performance.values() if data.get("max_drawdown")]
        return max(drawdowns) if drawdowns else 0.05
    
    def _calculate_preferred_volatility_range(self, volatility_performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """推奨ボラティリティ範囲を計算"""
        if not volatility_performance:
            return {
                "optimal_range": "medium",
                "atr_percentile_range": [20, 80],
                "volatility_score_range": [0.1, 0.3]
            }
        
        # 各ボラティリティ環境でのスコアを比較
        scores = {}
        for vol_type, data in volatility_performance.items():
            scores[vol_type] = data.get("suitability_score", 0.0)
        
        # 最適な環境を特定
        optimal_vol = max(scores, key=scores.get) if scores else "medium_volatility"
        
        # ボラティリティ範囲のマッピング
        volatility_ranges = {
            "low_volatility": {
                "optimal_range": "low",
                "atr_percentile_range": [0, 33],
                "volatility_score_range": [0.0, 0.15]
            },
            "medium_volatility": {
                "optimal_range": "medium", 
                "atr_percentile_range": [33, 67],
                "volatility_score_range": [0.15, 0.25]
            },
            "high_volatility": {
                "optimal_range": "high",
                "atr_percentile_range": [67, 100],
                "volatility_score_range": [0.25, 1.0]
            }
        }
        
        return volatility_ranges.get(optimal_vol, volatility_ranges["medium_volatility"])
    
    def _extract_required_indicators(self, strategy_id: str) -> List[str]:
        """戦略に必要な指標を抽出"""
        indicator_map = {
            "VWAPBreakoutStrategy": ["VWAP", "Volume", "SMA", "RSI", "MACD"],
            "VWAPBounceStrategy": ["VWAP", "Volume", "ATR"],
            "MomentumInvestingStrategy": ["SMA", "RSI", "MACD", "Volume", "ATR"],
            "ContrarianStrategy": ["RSI", "Volume"],
            "GCStrategy": ["SMA"],
            "OpeningGapStrategy": ["ATR", "Volume"],
            "BreakoutStrategy": ["Volume"]
        }
        return indicator_map.get(strategy_id, ["基本指標"])
    
    def _requires_index_data(self, strategy_id: str) -> bool:
        """インデックスデータが必要かを判定"""
        index_required = ["VWAPBreakoutStrategy", "OpeningGapStrategy"]
        return strategy_id in index_required
    
    def _get_additional_columns(self, strategy_id: str) -> List[str]:
        """戦略固有の追加カラム要件"""
        additional_columns_map = {
            "VWAPBounceStrategy": ["VWAP"],
            "VWAPBreakoutStrategy": ["VWAP"],
            "MomentumInvestingStrategy": ["SMA_short", "SMA_long", "RSI", "MACD"],
            "ContrarianStrategy": ["RSI"],
            "GCStrategy": ["SMA_short", "SMA_long"],
            "OpeningGapStrategy": ["ATR"],
            "BreakoutStrategy": []
        }
        return additional_columns_map.get(strategy_id, [])
    
    def _calculate_confidence_score(self, trend_performance: Dict[str, Any]) -> float:
        """信頼度スコアを計算"""
        total_samples = sum(data.get("sample_size", 0) for data in trend_performance.values())
        if total_samples >= 100:
            return 0.9
        elif total_samples >= 50:
            return 0.7
        elif total_samples >= 20:
            return 0.5
        else:
            return 0.3
    
    def _calculate_reliability(self, data: Dict[str, Any]) -> str:
        """信頼性を評価"""
        sample_size = data.get("sample_size", 0)
        sharpe_ratio = data.get("sharpe_ratio", 0)
        
        if sample_size >= 50 and sharpe_ratio >= 1.0:
            return "high"
        elif sample_size >= 20 and sharpe_ratio >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _initialize_param_history(self, strategy_id: str):
        """パラメータ履歴ファイルの初期化"""
        history_file = os.path.join(self.param_history_path, f"{strategy_id}_param_history.json")
        
        if not os.path.exists(history_file):
            initial_history = {
                "strategy_id": strategy_id,
                "creation_date": datetime.now().isoformat(),
                "parameter_history": [],
                "optimization_history": [],
                "version_info": {
                    "current_version": "1.0",
                    "version_history": []
                },
                "metadata": {
                    "total_optimizations": 0,
                    "best_sharpe_ratio": 0.0,
                    "last_optimization_date": None
                }
            }
            
            try:
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_history, f, indent=2, ensure_ascii=False)
                logger.info(f"パラメータ履歴ファイルを初期化しました: {history_file}")
            except Exception as e:
                logger.error(f"パラメータ履歴ファイルの初期化エラー: {e}")
    
    def add_parameter_version(self, 
                            strategy_id: str, 
                            parameters: Dict[str, Any], 
                            performance_metrics: Dict[str, float],
                            optimization_info: Optional[Dict[str, Any]] = None):
        """パラメータ履歴に新しいバージョンを追加"""
        history_file = os.path.join(self.param_history_path, f"{strategy_id}_param_history.json")
        
        # 既存履歴を読み込み
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            self._initialize_param_history(strategy_id)
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # 新しいパラメータエントリを作成
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "version": f"1.{len(history['parameter_history']) + 1}",
            "parameters": parameters,
            "performance_metrics": performance_metrics,
            "validation_status": "pending",
            "optimization_info": optimization_info or {},
            "notes": ""
        }
        
        # 履歴に追加
        history["parameter_history"].append(new_entry)
        history["last_updated"] = datetime.now().isoformat()
        history["metadata"]["total_optimizations"] += 1
        
        # 最良のシャープレシオを更新
        current_sharpe = performance_metrics.get("sharpe_ratio", 0)
        if current_sharpe > history["metadata"]["best_sharpe_ratio"]:
            history["metadata"]["best_sharpe_ratio"] = current_sharpe
        
        history["metadata"]["last_optimization_date"] = datetime.now().isoformat()
        
        # ファイルに保存
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"パラメータ履歴を更新しました: {strategy_id}, バージョン: {new_entry['version']}")
    
    def get_parameter_history(self, strategy_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """パラメータ履歴を取得"""
        history_file = os.path.join(self.param_history_path, f"{strategy_id}_param_history.json")
        
        if not os.path.exists(history_file):
            return []
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # 最新のエントリから指定数を返す
            return history["parameter_history"][-limit:]
        except Exception as e:
            logger.error(f"パラメータ履歴読み込みエラー: {e}")
            return []
    
    def get_best_parameters(self, strategy_id: str, metric: str = "sharpe_ratio") -> Optional[Dict[str, Any]]:
        """指定したメトリクスで最良のパラメータを取得"""
        history = self.get_parameter_history(strategy_id, limit=100)  # 過去100バージョンを確認
        
        if not history:
            return None
        
        # 指定メトリクスで最良のエントリを検索
        best_entry = None
        best_score = float('-inf')
        
        for entry in history:
            if metric in entry.get("performance_metrics", {}):
                score = entry["performance_metrics"][metric]
                if score > best_score:
                    best_score = score
                    best_entry = entry
        
        return best_entry
    
    def compare_parameter_versions(self, strategy_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """パラメータバージョン間の比較"""
        history = self.get_parameter_history(strategy_id, limit=100)
        
        v1_entry = next((entry for entry in history if entry["version"] == version1), None)
        v2_entry = next((entry for entry in history if entry["version"] == version2), None)
        
        if not v1_entry or not v2_entry:
            return {"error": "指定されたバージョンが見つかりません"}
        
        comparison = {
            "version_1": {
                "version": version1,
                "parameters": v1_entry["parameters"],
                "performance": v1_entry["performance_metrics"]
            },
            "version_2": {
                "version": version2,
                "parameters": v2_entry["parameters"],
                "performance": v2_entry["performance_metrics"]
            },
            "parameter_differences": {},
            "performance_differences": {}
        }
        
        # パラメータの差分
        all_params = set(v1_entry["parameters"].keys()) | set(v2_entry["parameters"].keys())
        for param in all_params:
            val1 = v1_entry["parameters"].get(param, "N/A")
            val2 = v2_entry["parameters"].get(param, "N/A")
            if val1 != val2:
                comparison["parameter_differences"][param] = {"v1": val1, "v2": val2}
        
        # パフォーマンスの差分
        all_metrics = set(v1_entry["performance_metrics"].keys()) | set(v2_entry["performance_metrics"].keys())
        for metric in all_metrics:
            val1 = v1_entry["performance_metrics"].get(metric, 0)
            val2 = v2_entry["performance_metrics"].get(metric, 0)
            diff = val2 - val1
            comparison["performance_differences"][metric] = {
                "v1": val1, "v2": val2, "difference": diff,
                "improvement": diff > 0
            }
        
        return comparison
    
    def generate_strategy_report(self, strategy_id: str) -> str:
        """戦略の包括的レポートを生成"""
        metadata = self.load_metadata(strategy_id)
        if not metadata:
            return f"戦略 {strategy_id} のメタデータが見つかりません。"
        
        param_history = self.get_parameter_history(strategy_id, limit=5)
        best_params = self.get_best_parameters(strategy_id)
        
        report = f"""
=== 戦略特性レポート: {metadata['strategy_name']} ===

■ 基本情報
戦略ID: {metadata['strategy_id']}
戦略クラス: {metadata['strategy_class']}
バージョン: {metadata['version']}
最終更新: {metadata['last_updated']}

■ トレンド適性分析
"""
        
        for trend_type, data in metadata['trend_adaptability'].items():
            report += f"\n[{trend_type}]"
            report += f"\n  適合度スコア: {data['suitability_score']:.2f}"
            report += f"\n  信頼度: {data['confidence_level']}"
            report += f"\n  シャープレシオ: {data['performance_metrics']['sharpe_ratio']:.2f}"
            report += f"\n  勝率: {data['performance_metrics']['win_rate']:.1%}"
            report += f"\n  最大ドローダウン: {data['performance_metrics']['max_drawdown']:.1%}"
        
        if metadata.get('volatility_adaptability'):
            report += f"\n\n■ ボラティリティ適性分析"
            for vol_type, data in metadata['volatility_adaptability'].items():
                report += f"\n[{vol_type}]"
                report += f"\n  適合度スコア: {data['suitability_score']:.2f}"
                report += f"\n  信頼度: {data['confidence_level']}"
                report += f"\n  シャープレシオ: {data['performance_metrics']['sharpe_ratio']:.2f}"
                report += f"\n  ポジションサイズ倍率: {data['risk_characteristics']['position_sizing_multiplier']:.2f}"
        
        report += f"\n\n■ パラメータ履歴"
        if param_history:
            report += f"\n直近バージョン数: {len(param_history)}"
            latest = param_history[-1]
            report += f"\n最新バージョン: {latest['version']}"
            report += f"\n最新更新日: {latest['timestamp']}"
        else:
            report += f"\nパラメータ履歴なし"
        
        if best_params:
            report += f"\n\n■ 最良パラメータ（シャープレシオベース）"
            report += f"\nバージョン: {best_params['version']}"
            report += f"\nシャープレシオ: {best_params['performance_metrics'].get('sharpe_ratio', 0):.2f}"
            report += f"\n主要パラメータ:"
            for key, value in list(best_params['parameters'].items())[:5]:  # 上位5つのパラメータを表示
                report += f"\n  {key}: {value}"
        
        report += f"\n\n■ 依存関係"
        deps = metadata.get('dependencies', {})
        report += f"\n必要指標: {', '.join(deps.get('required_indicators', []))}"
        report += f"\nインデックスデータ必要: {'はい' if deps.get('index_data_required', False) else 'いいえ'}"
        
        return report
    
    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        """メタデータを保存"""
        strategy_id = metadata["strategy_id"]
        filename = f"{strategy_id}_characteristics.json"
        filepath = os.path.join(self.metadata_path, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"戦略特性メタデータを保存しました: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")
            raise
    
    def load_metadata(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """メタデータを読み込み"""
        filename = f"{strategy_id}_characteristics.json"
        filepath = os.path.join(self.metadata_path, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"メタデータファイルが見つかりません: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"戦略特性メタデータを読み込みました: {strategy_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"メタデータ読み込みエラー: {e}")
            return None
    
    def get_trend_suitability(self, strategy_id: str, trend: str) -> Optional[Dict[str, Any]]:
        """指定トレンドでの戦略適性を取得"""
        metadata = self.load_metadata(strategy_id)
        if not metadata:
            return None
        
        return metadata.get("trend_adaptability", {}).get(trend)
    
    def get_volatility_suitability(self, strategy_id: str, volatility_level: str) -> Optional[Dict[str, Any]]:
        """指定ボラティリティでの戦略適性を取得"""
        metadata = self.load_metadata(strategy_id)
        if not metadata:
            return None
        
        return metadata.get("volatility_adaptability", {}).get(volatility_level)
    
    def list_strategies(self) -> List[str]:
        """利用可能な戦略一覧を取得"""
        strategies = []
        for file in os.listdir(self.metadata_path):
            if file.endswith("_characteristics.json"):
                strategy_id = file.replace("_characteristics.json", "")
                strategies.append(strategy_id)
        return strategies
    
    def update_performance_data(self, strategy_id: str, new_performance: Dict[str, Any]):
        """パフォーマンスデータを更新"""
        metadata = self.load_metadata(strategy_id)
        if not metadata:
            logger.error(f"戦略 {strategy_id} のメタデータが見つかりません")
            return
        
        # パフォーマンスデータを更新
        for trend, data in new_performance.items():
            if trend in metadata["trend_adaptability"]:
                metadata["trend_adaptability"][trend]["performance_metrics"].update(data)
        
        metadata["last_updated"] = datetime.now().isoformat()
        self.save_metadata(metadata)
        logger.info(f"戦略 {strategy_id} のパフォーマンスデータを更新しました")

# 戦略特性データの作成・管理用のユーティリティ関数
def create_sample_strategy_characteristics():
    """サンプル戦略特性データの作成"""
    manager = StrategyCharacteristicsManager()
    
    # VWAPBounceStrategyのサンプルデータ
    trend_performance = {
        "uptrend": {
            "suitability_score": 0.6,
            "sharpe_ratio": 0.8,
            "max_drawdown": 0.12,
            "win_rate": 0.65,
            "expectancy": 0.02,
            "avg_return": 0.15,
            "volatility": 0.18,
            "sample_size": 45,
            "data_period": "2023-01-01_to_2024-12-31",
            "avg_holding_period": 6.2,
            "max_consecutive_losses": 3,
            "profit_factor": 1.3
        },
        "downtrend": {
            "suitability_score": 0.3,
            "sharpe_ratio": -0.2,
            "max_drawdown": 0.25,
            "win_rate": 0.35,
            "expectancy": -0.01,
            "avg_return": -0.05,
            "volatility": 0.22,
            "sample_size": 30,
            "data_period": "2023-01-01_to_2024-12-31",
            "avg_holding_period": 4.8,
            "max_consecutive_losses": 5,
            "profit_factor": 0.8
        },
        "range-bound": {
            "suitability_score": 0.85,
            "sharpe_ratio": 1.4,
            "max_drawdown": 0.08,
            "win_rate": 0.72,
            "expectancy": 0.035,
            "avg_return": 0.25,
            "volatility": 0.15,
            "sample_size": 78,
            "data_period": "2023-01-01_to_2024-12-31",
            "avg_holding_period": 7.1,
            "max_consecutive_losses": 2,
            "profit_factor": 1.8
        }
    }
    
    # ボラティリティ別パフォーマンス
    volatility_performance = {
        "high_volatility": {
            "suitability_score": 0.4,
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.18,
            "win_rate": 0.55,
            "expectancy": 0.01,
            "sample_size": 35,
            "data_period": "2023-01-01_to_2024-12-31",
            "position_sizing_multiplier": 0.7,
            "stop_loss_adjustment": 1.3,
            "entry_frequency": "reduced",
            "volatility_threshold": 0.35
        },
        "medium_volatility": {
            "suitability_score": 0.75,
            "sharpe_ratio": 1.1,
            "max_drawdown": 0.10,
            "win_rate": 0.68,
            "expectancy": 0.025,
            "sample_size": 65,
            "data_period": "2023-01-01_to_2024-12-31",
            "position_sizing_multiplier": 1.0,
            "stop_loss_adjustment": 1.0,
            "entry_frequency": "normal",
            "volatility_threshold": 0.20
        },
        "low_volatility": {
            "suitability_score": 0.65,
            "sharpe_ratio": 0.9,
            "max_drawdown": 0.06,
            "win_rate": 0.70,
            "expectancy": 0.018,
            "sample_size": 53,
            "data_period": "2023-01-01_to_2024-12-31",
            "position_sizing_multiplier": 1.2,
            "stop_loss_adjustment": 0.8,
            "entry_frequency": "increased",
            "volatility_threshold": 0.10
        }
    }
    
    # メタデータ作成
    metadata = manager.create_strategy_metadata(
        strategy_id="VWAPBounceStrategy",
        trend_performance=trend_performance,
        volatility_performance=volatility_performance,
        include_param_history=True
    )
    
    # メタデータを保存
    filepath = manager.save_metadata(metadata)
    print(f"サンプル戦略特性メタデータを作成しました: {filepath}")
    
    return metadata, manager

if __name__ == "__main__":
    # サンプル実行
    sample_metadata, manager = create_sample_strategy_characteristics()
    
    # 結果の表示
    print("\n=== 作成された戦略特性メタデータ ===")
    print(f"戦略ID: {sample_metadata['strategy_id']}")
    print(f"戦略名: {sample_metadata['strategy_name']}")
    print(f"トレンド適性:")
    for trend, data in sample_metadata['trend_adaptability'].items():
        print(f"  {trend}: 適合度{data['suitability_score']:.2f}, 信頼度{data['confidence_level']}")
    
    # 利用可能な戦略の表示
    print(f"\n利用可能な戦略: {manager.list_strategies()}")
