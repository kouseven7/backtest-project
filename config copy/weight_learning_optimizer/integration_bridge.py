"""
統合ブリッジ

5-2-1（パフォーマンススコア修正）、5-2-2（トレンド精度調整）、
ポートフォリオシステムとのクリーンな統合インターフェース
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import importlib
import importlib.util
import sys

@dataclass
class SystemIntegrationConfig:
    """システム統合設定"""
    performance_correction_enabled: bool = True
    trend_precision_enabled: bool = True
    portfolio_integration_enabled: bool = True
    strategy_scoring_enabled: bool = True
    metric_weights_enabled: bool = True
    
@dataclass
class IntegrationResult:
    """統合結果"""
    system_name: str
    weights_applied: Dict[str, float]
    performance_impact: float
    integration_success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None

class IntegrationBridge:
    """
    統合ブリッジ
    
    既存システム（5-2-1、5-2-2、ポートフォリオシステム）との
    シームレスな統合を管理し、重み学習システムの結果を
    各システムに適用する。
    """
    
    def __init__(
        self,
        config: Optional[SystemIntegrationConfig] = None,
        workspace_path: str = None
    ):
        """
        初期化
        
        Args:
            config: システム統合設定
            workspace_path: ワークスペースのパス
        """
        self.logger = self._setup_logger()
        self.config = config or SystemIntegrationConfig()
        self.workspace_path = workspace_path or str(Path.cwd())
        
        # システム参照の初期化
        self.system_references = {}
        self.integration_history = []
        
        # システムの発見と初期化
        self._discover_and_initialize_systems()
        
        self.logger.info("IntegrationBridge initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.IntegrationBridge")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _discover_and_initialize_systems(self) -> None:
        """システムの発見と初期化"""
        self.logger.info("Discovering and initializing integrated systems")
        
        # 5-2-1 パフォーマンススコア修正システム
        if self.config.performance_correction_enabled:
            self._initialize_performance_correction_system()
            
        # 5-2-2 トレンド精度調整システム
        if self.config.trend_precision_enabled:
            self._initialize_trend_precision_system()
            
        # ポートフォリオシステム
        if self.config.portfolio_integration_enabled:
            self._initialize_portfolio_system()
            
        # ストラテジースコアリングシステム
        if self.config.strategy_scoring_enabled:
            self._initialize_strategy_scoring_system()
            
        # メトリック重みオプティマイザー
        if self.config.metric_weights_enabled:
            self._initialize_metric_weights_system()
            
    def _initialize_performance_correction_system(self) -> None:
        """5-2-1 パフォーマンススコア修正システムの初期化"""
        try:
            config_path = Path(self.workspace_path) / "config" / "performance_score_correction"
            
            if config_path.exists():
                sys.path.insert(0, str(config_path))
                
                # メインモジュールのインポートを試行
                try:
                    import performance_score_corrector
                    self.system_references['performance_correction'] = {
                        'module': performance_score_corrector,
                        'corrector': None,  # 実際のインスタンスは必要時に作成
                        'config_path': config_path,
                        'status': 'available'
                    }
                    self.logger.info("5-2-1 Performance Score Correction system discovered")
                except ImportError as e:
                    self.logger.warning(f"5-2-1 system import failed: {e}")
                    self.system_references['performance_correction'] = {
                        'status': 'import_failed',
                        'error': str(e)
                    }
            else:
                self.logger.warning("5-2-1 system not found at expected path")
                self.system_references['performance_correction'] = {
                    'status': 'not_found'
                }
        except Exception as e:
            self.logger.error(f"Error initializing 5-2-1 system: {e}")
            self.system_references['performance_correction'] = {
                'status': 'error',
                'error': str(e)
            }
            
    def _initialize_trend_precision_system(self) -> None:
        """5-2-2 トレンド精度調整システムの初期化"""
        try:
            config_path = Path(self.workspace_path) / "config" / "trend_precision_adjustment"
            
            if config_path.exists():
                sys.path.insert(0, str(config_path))
                
                try:
                    import trend_precision_adjuster
                    self.system_references['trend_precision'] = {
                        'module': trend_precision_adjuster,
                        'adjuster': None,
                        'config_path': config_path,
                        'status': 'available'
                    }
                    self.logger.info("5-2-2 Trend Precision Adjustment system discovered")
                except ImportError as e:
                    self.logger.warning(f"5-2-2 system import failed: {e}")
                    self.system_references['trend_precision'] = {
                        'status': 'import_failed',
                        'error': str(e)
                    }
            else:
                self.logger.warning("5-2-2 system not found at expected path")
                self.system_references['trend_precision'] = {
                    'status': 'not_found'
                }
        except Exception as e:
            self.logger.error(f"Error initializing 5-2-2 system: {e}")
            self.system_references['trend_precision'] = {
                'status': 'error',
                'error': str(e)
            }
            
    def _initialize_portfolio_system(self) -> None:
        """ポートフォリオシステムの初期化"""
        try:
            # ポートフォリオ重み計算器
            portfolio_path = Path(self.workspace_path) / "portfolio_weight_calculator.py"
            
            if portfolio_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("portfolio_weight_calculator", portfolio_path)
                    portfolio_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(portfolio_module)
                    
                    self.system_references['portfolio_weights'] = {
                        'module': portfolio_module,
                        'calculator': None,
                        'status': 'available'
                    }
                    self.logger.info("Portfolio weight system discovered")
                except Exception as e:
                    self.logger.warning(f"Portfolio system import failed: {e}")
                    self.system_references['portfolio_weights'] = {
                        'status': 'import_failed',
                        'error': str(e)
                    }
            else:
                self.logger.warning("Portfolio weight system not found")
                self.system_references['portfolio_weights'] = {
                    'status': 'not_found'
                }
        except Exception as e:
            self.logger.error(f"Error initializing portfolio system: {e}")
            self.system_references['portfolio_weights'] = {
                'status': 'error',
                'error': str(e)
            }
            
    def _initialize_strategy_scoring_system(self) -> None:
        """ストラテジースコアリングシステムの初期化"""
        try:
            strategy_path = Path(self.workspace_path) / "strategy_scoring_model.py"
            
            if strategy_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("strategy_scoring_model", strategy_path)
                    strategy_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(strategy_module)
                    
                    self.system_references['strategy_scoring'] = {
                        'module': strategy_module,
                        'model': None,
                        'status': 'available'
                    }
                    self.logger.info("Strategy scoring system discovered")
                except Exception as e:
                    self.logger.warning(f"Strategy scoring system import failed: {e}")
                    self.system_references['strategy_scoring'] = {
                        'status': 'import_failed',
                        'error': str(e)
                    }
            else:
                self.logger.warning("Strategy scoring system not found")
                self.system_references['strategy_scoring'] = {
                    'status': 'not_found'
                }
        except Exception as e:
            self.logger.error(f"Error initializing strategy scoring system: {e}")
            self.system_references['strategy_scoring'] = {
                'status': 'error',
                'error': str(e)
            }
            
    def _initialize_metric_weights_system(self) -> None:
        """メトリック重みオプティマイザーの初期化"""
        try:
            metric_path = Path(self.workspace_path) / "metric_weight_optimizer.py"
            
            if metric_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("metric_weight_optimizer", metric_path)
                    metric_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(metric_module)
                    
                    self.system_references['metric_weights'] = {
                        'module': metric_module,
                        'optimizer': None,
                        'status': 'available'
                    }
                    self.logger.info("Metric weight optimizer discovered")
                except Exception as e:
                    self.logger.warning(f"Metric weight optimizer import failed: {e}")
                    self.system_references['metric_weights'] = {
                        'status': 'import_failed',
                        'error': str(e)
                    }
            else:
                self.logger.warning("Metric weight optimizer not found")
                self.system_references['metric_weights'] = {
                    'status': 'not_found'
                }
        except Exception as e:
            self.logger.error(f"Error initializing metric weights system: {e}")
            self.system_references['metric_weights'] = {
                'status': 'error',
                'error': str(e)
            }
            
    def apply_optimized_weights(
        self,
        optimized_weights: Dict[str, float],
        market_data: pd.DataFrame = None,
        performance_data: pd.DataFrame = None
    ) -> List[IntegrationResult]:
        """
        最適化された重みの各システムへの適用
        
        Args:
            optimized_weights: 最適化された重み
            market_data: 市場データ
            performance_data: パフォーマンスデータ
            
        Returns:
            統合結果のリスト
        """
        self.logger.info("Applying optimized weights to integrated systems")
        
        results = []
        
        # ストラテジー重みの適用
        strategy_weights = {
            k.replace('strategy_', ''): v 
            for k, v in optimized_weights.items() 
            if k.startswith('strategy_')
        }
        
        if strategy_weights:
            # 5-2-1 パフォーマンススコア修正
            if self.system_references.get('performance_correction', {}).get('status') == 'available':
                result = self._apply_to_performance_correction(strategy_weights, performance_data)
                results.append(result)
                
            # 5-2-2 トレンド精度調整
            if self.system_references.get('trend_precision', {}).get('status') == 'available':
                result = self._apply_to_trend_precision(strategy_weights, market_data)
                results.append(result)
                
            # ストラテジースコアリング
            if self.system_references.get('strategy_scoring', {}).get('status') == 'available':
                result = self._apply_to_strategy_scoring(strategy_weights, performance_data)
                results.append(result)
                
        # ポートフォリオ重みの適用
        portfolio_weights = {
            k.replace('portfolio_', ''): v 
            for k, v in optimized_weights.items() 
            if k.startswith('portfolio_')
        }
        
        if portfolio_weights and self.system_references.get('portfolio_weights', {}).get('status') == 'available':
            result = self._apply_to_portfolio_system(portfolio_weights, market_data)
            results.append(result)
            
        # メタパラメータの適用
        meta_parameters = {
            k.replace('meta_', ''): v 
            for k, v in optimized_weights.items() 
            if k.startswith('meta_')
        }
        
        if meta_parameters and self.system_references.get('metric_weights', {}).get('status') == 'available':
            result = self._apply_to_metric_weights(meta_parameters, performance_data)
            results.append(result)
            
        # 統合履歴の更新
        self.integration_history.extend(results)
        
        self.logger.info(f"Applied weights to {len(results)} systems")
        return results
        
    def _apply_to_performance_correction(
        self,
        weights: Dict[str, float],
        performance_data: pd.DataFrame
    ) -> IntegrationResult:
        """5-2-1 パフォーマンススコア修正システムへの適用"""
        try:
            system_ref = self.system_references['performance_correction']
            
            # インスタンスの作成または取得
            if system_ref['corrector'] is None:
                # ダミーインスタンス作成（実際の実装では適切なクラスを使用）
                system_ref['corrector'] = self._create_mock_corrector()
                
            # 重みの適用
            corrector = system_ref['corrector']
            
            # 重み設定メソッドの呼び出し
            if hasattr(corrector, 'update_correction_weights'):
                corrector.update_correction_weights(weights)
                performance_impact = 0.02  # ダミー影響度
            else:
                # フォールバック処理
                performance_impact = self._estimate_performance_impact(weights, 'performance_correction')
                
            return IntegrationResult(
                system_name="performance_score_correction",
                weights_applied=weights.copy(),
                performance_impact=performance_impact,
                integration_success=True,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error applying weights to performance correction: {e}")
            return IntegrationResult(
                system_name="performance_score_correction",
                weights_applied=weights.copy(),
                performance_impact=0.0,
                integration_success=False,
                error_message=str(e),
                timestamp=datetime.now()
            )
            
    def _apply_to_trend_precision(
        self,
        weights: Dict[str, float],
        market_data: pd.DataFrame
    ) -> IntegrationResult:
        """5-2-2 トレンド精度調整システムへの適用"""
        try:
            system_ref = self.system_references['trend_precision']
            
            if system_ref['adjuster'] is None:
                system_ref['adjuster'] = self._create_mock_adjuster()
                
            adjuster = system_ref['adjuster']
            
            if hasattr(adjuster, 'update_precision_weights'):
                adjuster.update_precision_weights(weights)
                performance_impact = 0.015
            else:
                performance_impact = self._estimate_performance_impact(weights, 'trend_precision')
                
            return IntegrationResult(
                system_name="trend_precision_adjustment",
                weights_applied=weights.copy(),
                performance_impact=performance_impact,
                integration_success=True,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error applying weights to trend precision: {e}")
            return IntegrationResult(
                system_name="trend_precision_adjustment",
                weights_applied=weights.copy(),
                performance_impact=0.0,
                integration_success=False,
                error_message=str(e),
                timestamp=datetime.now()
            )
            
    def _apply_to_portfolio_system(
        self,
        weights: Dict[str, float],
        market_data: pd.DataFrame
    ) -> IntegrationResult:
        """ポートフォリオシステムへの適用"""
        try:
            system_ref = self.system_references['portfolio_weights']
            
            if system_ref['calculator'] is None:
                system_ref['calculator'] = self._create_mock_calculator()
                
            calculator = system_ref['calculator']
            
            if hasattr(calculator, 'update_portfolio_weights'):
                calculator.update_portfolio_weights(weights)
                performance_impact = 0.025
            else:
                performance_impact = self._estimate_performance_impact(weights, 'portfolio')
                
            return IntegrationResult(
                system_name="portfolio_weight_system",
                weights_applied=weights.copy(),
                performance_impact=performance_impact,
                integration_success=True,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error applying weights to portfolio system: {e}")
            return IntegrationResult(
                system_name="portfolio_weight_system",
                weights_applied=weights.copy(),
                performance_impact=0.0,
                integration_success=False,
                error_message=str(e),
                timestamp=datetime.now()
            )
            
    def _apply_to_strategy_scoring(
        self,
        weights: Dict[str, float],
        performance_data: pd.DataFrame
    ) -> IntegrationResult:
        """ストラテジースコアリングシステムへの適用"""
        try:
            system_ref = self.system_references['strategy_scoring']
            
            if system_ref['model'] is None:
                system_ref['model'] = self._create_mock_scorer()
                
            model = system_ref['model']
            
            if hasattr(model, 'update_scoring_weights'):
                model.update_scoring_weights(weights)
                performance_impact = 0.018
            else:
                performance_impact = self._estimate_performance_impact(weights, 'strategy_scoring')
                
            return IntegrationResult(
                system_name="strategy_scoring_system",
                weights_applied=weights.copy(),
                performance_impact=performance_impact,
                integration_success=True,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error applying weights to strategy scoring: {e}")
            return IntegrationResult(
                system_name="strategy_scoring_system",
                weights_applied=weights.copy(),
                performance_impact=0.0,
                integration_success=False,
                error_message=str(e),
                timestamp=datetime.now()
            )
            
    def _apply_to_metric_weights(
        self,
        parameters: Dict[str, float],
        performance_data: pd.DataFrame
    ) -> IntegrationResult:
        """メトリック重みオプティマイザーへの適用"""
        try:
            system_ref = self.system_references['metric_weights']
            
            if system_ref['optimizer'] is None:
                system_ref['optimizer'] = self._create_mock_optimizer()
                
            optimizer = system_ref['optimizer']
            
            if hasattr(optimizer, 'update_meta_parameters'):
                optimizer.update_meta_parameters(parameters)
                performance_impact = 0.012
            else:
                performance_impact = self._estimate_performance_impact(parameters, 'meta_parameters')
                
            return IntegrationResult(
                system_name="metric_weight_optimizer",
                weights_applied=parameters.copy(),
                performance_impact=performance_impact,
                integration_success=True,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error applying meta parameters to metric weights: {e}")
            return IntegrationResult(
                system_name="metric_weight_optimizer",
                weights_applied=parameters.copy(),
                performance_impact=0.0,
                integration_success=False,
                error_message=str(e),
                timestamp=datetime.now()
            )
            
    def evaluate_weight_combination(
        self,
        weights: Dict[str, float],
        performance_data: pd.DataFrame,
        target_metrics: List[str]
    ) -> Dict[str, float]:
        """
        重み組み合わせの評価
        
        Args:
            weights: 評価する重み
            performance_data: パフォーマンスデータ
            target_metrics: ターゲット指標
            
        Returns:
            評価結果
        """
        # 各システムでの評価
        evaluations = {}
        
        # ポートフォリオパフォーマンス評価
        portfolio_weights = {
            k.replace('portfolio_', ''): v 
            for k, v in weights.items() 
            if k.startswith('portfolio_')
        }
        
        if portfolio_weights and len(performance_data) > 0:
            portfolio_performance = self._evaluate_portfolio_performance(
                portfolio_weights, performance_data
            )
            evaluations.update(portfolio_performance)
        else:
            # デフォルト値
            evaluations.update({
                'expected_return': 0.05,
                'max_drawdown': 0.15,
                'sharpe_ratio': 0.8,
                'volatility': 0.12
            })
            
        # 複合スコアの計算
        combined_score = self._calculate_integrated_score(evaluations, target_metrics)
        evaluations['combined_score'] = combined_score
        
        return evaluations
        
    def _evaluate_portfolio_performance(
        self,
        portfolio_weights: Dict[str, float],
        performance_data: pd.DataFrame
    ) -> Dict[str, float]:
        """ポートフォリオパフォーマンスの評価"""
        # 簡易評価（実際の実装ではより詳細な計算が必要）
        
        # 重み付きリターンの計算
        weighted_returns = pd.Series(0.0, index=performance_data.index)
        
        for asset, weight in portfolio_weights.items():
            if asset in performance_data.columns:
                weighted_returns += performance_data[asset].pct_change().fillna(0) * weight
                
        # 基本指標の計算
        annual_return = weighted_returns.mean() * 252
        annual_volatility = weighted_returns.std() * np.sqrt(252)
        
        # ドローダウン計算
        cumulative_returns = (1 + weighted_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # シャープレシオ
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        return {
            'expected_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': annual_volatility
        }
        
    def _calculate_integrated_score(
        self,
        evaluations: Dict[str, float],
        target_metrics: List[str]
    ) -> float:
        """統合スコアの計算"""
        weights = {
            'expected_return': 0.4,
            'max_drawdown': 0.3,
            'sharpe_ratio': 0.2,
            'volatility': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric in target_metrics:
            if metric in evaluations and metric in weights:
                value = evaluations[metric]
                weight = weights[metric]
                
                # 指標の正規化
                if metric == 'expected_return' or metric == 'sharpe_ratio':
                    normalized_value = min(1.0, max(0.0, value))
                elif metric == 'max_drawdown' or metric == 'volatility':
                    normalized_value = max(0.0, 1.0 - min(1.0, value))
                else:
                    normalized_value = value
                    
                score += normalized_value * weight
                total_weight += weight
                
        return score / total_weight if total_weight > 0 else 0.0
        
    def _estimate_performance_impact(
        self,
        weights: Dict[str, float],
        system_type: str
    ) -> float:
        """パフォーマンス影響度の推定"""
        # 重みの変更幅に基づく影響度推定
        weight_changes = sum(abs(w - 0.5) for w in weights.values())  # 0.5を基準とした変更幅
        
        system_multipliers = {
            'performance_correction': 0.02,
            'trend_precision': 0.015,
            'portfolio': 0.025,
            'strategy_scoring': 0.018,
            'meta_parameters': 0.012
        }
        
        multiplier = system_multipliers.get(system_type, 0.01)
        return weight_changes * multiplier
        
    # モックオブジェクト作成メソッド（テスト用）
    def _create_mock_corrector(self):
        class MockCorrector:
            def update_correction_weights(self, weights): pass
        return MockCorrector()
        
    def _create_mock_adjuster(self):
        class MockAdjuster:
            def update_precision_weights(self, weights): pass
        return MockAdjuster()
        
    def _create_mock_calculator(self):
        class MockCalculator:
            def update_portfolio_weights(self, weights): pass
        return MockCalculator()
        
    def _create_mock_scorer(self):
        class MockScorer:
            def update_scoring_weights(self, weights): pass
        return MockScorer()
        
    def _create_mock_optimizer(self):
        class MockOptimizer:
            def update_meta_parameters(self, parameters): pass
        return MockOptimizer()
        
    def get_system_status(self) -> Dict[str, Any]:
        """システムステータスの取得"""
        status = {}
        
        for system_name, system_ref in self.system_references.items():
            status[system_name] = {
                'status': system_ref.get('status', 'unknown'),
                'error': system_ref.get('error'),
                'last_integration': None
            }
            
            # 最後の統合時刻を検索
            for result in reversed(self.integration_history):
                if system_name in result.system_name:
                    status[system_name]['last_integration'] = result.timestamp
                    break
                    
        return status
        
    def get_integration_summary(self) -> Dict[str, Any]:
        """統合サマリーの取得"""
        if not self.integration_history:
            return {}
            
        recent_integrations = self.integration_history[-20:]  # 最近20件
        
        success_rate = sum(1 for r in recent_integrations if r.integration_success) / len(recent_integrations)
        avg_performance_impact = np.mean([r.performance_impact for r in recent_integrations])
        
        system_counts = {}
        for result in recent_integrations:
            system_counts[result.system_name] = system_counts.get(result.system_name, 0) + 1
            
        return {
            'total_integrations': len(self.integration_history),
            'recent_success_rate': success_rate,
            'average_performance_impact': avg_performance_impact,
            'system_integration_counts': system_counts,
            'available_systems': len([
                s for s in self.system_references.values() 
                if s.get('status') == 'available'
            ])
        }
