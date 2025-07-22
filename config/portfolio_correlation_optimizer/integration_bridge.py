"""
5-3-3 戦略間相関を考慮した配分最適化 - システム統合ブリッジ

既存システムとの統合を管理するブリッジシステム

Author: imega
Created: 2025-01-27
Task: 5-3-3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# 既存システムのインポート
try:
    from ..correlation.strategy_correlation_analyzer import (
        CorrelationConfig, CorrelationMatrix, StrategyCorrelationAnalyzer
    )
    from ..portfolio_weight_calculator import PortfolioWeightCalculator
    from ...analysis.risk_adjusted_optimization.risk_return_optimizer import (
        RiskAdjustedOptimizationEngine
    )
    from ..weight_learning_optimizer.weight_learning_system import (
        WeightLearningSystem
    )
    EXISTING_SYSTEMS_AVAILABLE = True
except ImportError as e:
    EXISTING_SYSTEMS_AVAILABLE = False

@dataclass
class IntegrationConfig:
    """統合設定"""
    # システム統合レベル
    integration_level: str = "moderate"  # basic, moderate, advanced
    
    # データ統合設定
    use_existing_correlation_analyzer: bool = True
    use_existing_portfolio_calculator: bool = True
    use_existing_risk_optimizer: bool = True
    use_existing_weight_learner: bool = True
    
    # データ共有設定
    share_correlation_data: bool = True
    share_weight_data: bool = True
    share_score_data: bool = True
    
    # 統合パラメータ
    correlation_data_priority: float = 0.7  # 既存データの重み
    weight_adjustment_factor: float = 0.8
    score_integration_weight: float = 0.3
    
    # 競合解決設定
    conflict_resolution_method: str = "weighted_average"  # weighted_average, priority_based, hybrid
    fallback_on_error: bool = True

@dataclass
class BridgeResult:
    """ブリッジ実行結果"""
    integrated_data: Dict[str, Any]
    system_status: Dict[str, bool]
    integration_metadata: Dict[str, Any]
    success: bool = True
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """エラーがあるか"""
        return len(self.error_messages) > 0

class SystemIntegrationBridge:
    """システム統合ブリッジ"""
    
    def __init__(self, config: IntegrationConfig, logger: Optional[logging.Logger] = None):
        """
        初期化
        
        Args:
            config: 統合設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # 既存システムの初期化
        self._initialize_existing_systems()
        
        # データキャッシュ
        self._correlation_cache = {}
        self._weight_cache = {}
        self._score_cache = {}
        
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_existing_systems(self):
        """既存システム初期化"""
        
        self.system_instances = {}
        self.system_status = {}
        
        if not EXISTING_SYSTEMS_AVAILABLE:
            self.logger.warning("Existing systems not available, using standalone mode")
            return
        
        try:
            # 相関分析システム
            if self.config.use_existing_correlation_analyzer:
                try:
                    correlation_config = CorrelationConfig(
                        lookback_period=252,
                        min_observations=30,
                        significance_threshold=0.05
                    )
                    self.system_instances['correlation_analyzer'] = StrategyCorrelationAnalyzer(
                        config=correlation_config
                    )
                    self.system_status['correlation_analyzer'] = True
                    self.logger.info("Correlation analyzer initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize correlation analyzer: {e}")
                    self.system_status['correlation_analyzer'] = False
            
            # ポートフォリオ計算システム
            if self.config.use_existing_portfolio_calculator:
                try:
                    self.system_instances['portfolio_calculator'] = PortfolioWeightCalculator()
                    self.system_status['portfolio_calculator'] = True
                    self.logger.info("Portfolio calculator initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize portfolio calculator: {e}")
                    self.system_status['portfolio_calculator'] = False
            
            # リスク最適化システム
            if self.config.use_existing_risk_optimizer:
                try:
                    self.system_instances['risk_optimizer'] = RiskAdjustedOptimizationEngine()
                    self.system_status['risk_optimizer'] = True
                    self.logger.info("Risk optimizer initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize risk optimizer: {e}")
                    self.system_status['risk_optimizer'] = False
            
            # 重み学習システム
            if self.config.use_existing_weight_learner:
                try:
                    self.system_instances['weight_learner'] = WeightLearningSystem()
                    self.system_status['weight_learner'] = True
                    self.logger.info("Weight learner initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize weight learner: {e}")
                    self.system_status['weight_learner'] = False
        
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
    
    def integrate_correlation_data(
        self,
        strategy_returns: pd.DataFrame,
        local_correlation_result: Optional[Dict[str, Any]] = None
    ) -> BridgeResult:
        """
        相関データ統合
        
        Args:
            strategy_returns: 戦略リターンデータ
            local_correlation_result: ローカル相関結果
            
        Returns:
            統合結果
        """
        
        try:
            integrated_data = {}
            error_messages = []
            
            # 既存システムからの相関データ取得
            existing_correlation = None
            if (self.config.share_correlation_data and 
                'correlation_analyzer' in self.system_instances and
                self.system_status.get('correlation_analyzer', False)):
                
                try:
                    analyzer = self.system_instances['correlation_analyzer']
                    existing_correlation = analyzer.calculate_correlation_matrix(strategy_returns)
                    self.logger.info("Retrieved correlation data from existing system")
                except Exception as e:
                    error_messages.append(f"Failed to get existing correlation data: {e}")
                    self.logger.warning(f"Failed to get existing correlation data: {e}")
            
            # データ統合
            if existing_correlation is not None and local_correlation_result is not None:
                # 重み付き統合
                integrated_data = self._integrate_correlation_matrices(
                    existing_correlation, local_correlation_result
                )
            elif existing_correlation is not None:
                # 既存データのみ
                integrated_data = self._convert_existing_correlation_format(existing_correlation)
            elif local_correlation_result is not None:
                # ローカルデータのみ
                integrated_data = local_correlation_result
            else:
                # フォールバック：単純相関
                integrated_data = self._calculate_fallback_correlation(strategy_returns)
            
            # メタデータ
            integration_metadata = {
                'data_sources': [],
                'integration_method': self.config.conflict_resolution_method,
                'timestamp': datetime.now()
            }
            
            if existing_correlation is not None:
                integration_metadata['data_sources'].append('existing_system')
            if local_correlation_result is not None:
                integration_metadata['data_sources'].append('local_calculation')
            
            return BridgeResult(
                integrated_data=integrated_data,
                system_status=self.system_status.copy(),
                integration_metadata=integration_metadata,
                success=True,
                error_messages=error_messages
            )
            
        except Exception as e:
            self.logger.error(f"Correlation data integration failed: {e}")
            return BridgeResult(
                integrated_data={},
                system_status=self.system_status.copy(),
                integration_metadata={'error': str(e)},
                success=False,
                error_messages=[str(e)]
            )
    
    def integrate_weight_data(
        self,
        local_weights: Dict[str, float],
        strategy_scores: Optional[Dict[str, float]] = None
    ) -> BridgeResult:
        """
        重みデータ統合
        
        Args:
            local_weights: ローカル重み
            strategy_scores: 戦略スコア
            
        Returns:
            統合結果
        """
        
        try:
            integrated_data = {}
            error_messages = []
            
            # 既存システムからの重みデータ取得
            existing_weights = None
            if (self.config.share_weight_data and 
                'portfolio_calculator' in self.system_instances and
                self.system_status.get('portfolio_calculator', False)):
                
                try:
                    calculator = self.system_instances['portfolio_calculator']
                    if hasattr(calculator, 'get_current_weights'):
                        existing_weights = calculator.get_current_weights()
                    elif hasattr(calculator, 'calculate_weights') and strategy_scores:
                        existing_weights = calculator.calculate_weights(strategy_scores)
                    
                    if existing_weights:
                        self.logger.info("Retrieved weight data from existing system")
                except Exception as e:
                    error_messages.append(f"Failed to get existing weight data: {e}")
                    self.logger.warning(f"Failed to get existing weight data: {e}")
            
            # 重み学習システムからのデータ
            learned_weights = None
            if (self.config.use_existing_weight_learner and 
                'weight_learner' in self.system_instances and
                self.system_status.get('weight_learner', False)):
                
                try:
                    learner = self.system_instances['weight_learner']
                    if hasattr(learner, 'get_learned_weights'):
                        learned_weights = learner.get_learned_weights()
                        self.logger.info("Retrieved learned weight data")
                except Exception as e:
                    error_messages.append(f"Failed to get learned weight data: {e}")
                    self.logger.warning(f"Failed to get learned weight data: {e}")
            
            # データ統合
            integrated_data = self._integrate_weight_data(
                local_weights, existing_weights, learned_weights
            )
            
            # メタデータ
            integration_metadata = {
                'data_sources': ['local_optimization'],
                'integration_method': self.config.conflict_resolution_method,
                'adjustment_factor': self.config.weight_adjustment_factor,
                'timestamp': datetime.now()
            }
            
            if existing_weights is not None:
                integration_metadata['data_sources'].append('existing_portfolio_system')
            if learned_weights is not None:
                integration_metadata['data_sources'].append('weight_learning_system')
            
            return BridgeResult(
                integrated_data=integrated_data,
                system_status=self.system_status.copy(),
                integration_metadata=integration_metadata,
                success=True,
                error_messages=error_messages
            )
            
        except Exception as e:
            self.logger.error(f"Weight data integration failed: {e}")
            return BridgeResult(
                integrated_data=local_weights,  # フォールバック
                system_status=self.system_status.copy(),
                integration_metadata={'error': str(e)},
                success=False,
                error_messages=[str(e)]
            )
    
    def integrate_score_data(
        self,
        local_scores: Optional[Dict[str, float]] = None,
        strategy_names: Optional[List[str]] = None
    ) -> BridgeResult:
        """
        スコアデータ統合
        
        Args:
            local_scores: ローカルスコア
            strategy_names: 戦略名リスト
            
        Returns:
            統合結果
        """
        
        try:
            integrated_data = {}
            error_messages = []
            
            # 既存システムからのスコアデータ取得
            existing_scores = None
            if (self.config.share_score_data and 
                'risk_optimizer' in self.system_instances and
                self.system_status.get('risk_optimizer', False)):
                
                try:
                    optimizer = self.system_instances['risk_optimizer']
                    if hasattr(optimizer, 'get_strategy_scores'):
                        existing_scores = optimizer.get_strategy_scores()
                    elif hasattr(optimizer, 'calculate_scores') and strategy_names:
                        existing_scores = optimizer.calculate_scores(strategy_names)
                    
                    if existing_scores:
                        self.logger.info("Retrieved score data from existing system")
                except Exception as e:
                    error_messages.append(f"Failed to get existing score data: {e}")
                    self.logger.warning(f"Failed to get existing score data: {e}")
            
            # データ統合
            if local_scores is not None and existing_scores is not None:
                integrated_data = self._integrate_score_data(local_scores, existing_scores)
            elif existing_scores is not None:
                integrated_data = existing_scores
            elif local_scores is not None:
                integrated_data = local_scores
            else:
                # デフォルトスコア
                if strategy_names:
                    integrated_data = {name: 1.0 for name in strategy_names}
                else:
                    integrated_data = {}
            
            # メタデータ
            integration_metadata = {
                'data_sources': [],
                'integration_weight': self.config.score_integration_weight,
                'timestamp': datetime.now()
            }
            
            if local_scores is not None:
                integration_metadata['data_sources'].append('local_calculation')
            if existing_scores is not None:
                integration_metadata['data_sources'].append('existing_risk_system')
            
            return BridgeResult(
                integrated_data=integrated_data,
                system_status=self.system_status.copy(),
                integration_metadata=integration_metadata,
                success=True,
                error_messages=error_messages
            )
            
        except Exception as e:
            self.logger.error(f"Score data integration failed: {e}")
            return BridgeResult(
                integrated_data=local_scores or {},
                system_status=self.system_status.copy(),
                integration_metadata={'error': str(e)},
                success=False,
                error_messages=[str(e)]
            )
    
    def _integrate_correlation_matrices(
        self,
        existing_correlation: Any,
        local_correlation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """相関行列統合"""
        
        try:
            # 既存システムのデータ形式変換
            if hasattr(existing_correlation, 'correlation_matrix'):
                existing_matrix = existing_correlation.correlation_matrix
            elif isinstance(existing_correlation, pd.DataFrame):
                existing_matrix = existing_correlation
            elif isinstance(existing_correlation, np.ndarray):
                existing_matrix = pd.DataFrame(existing_correlation)
            else:
                existing_matrix = None
            
            # ローカル結果の取得
            local_matrix = local_correlation_result.get('combined_correlation')
            
            if existing_matrix is not None and local_matrix is not None:
                # 重み付き平均
                priority = self.config.correlation_data_priority
                
                # インデックス整合
                common_strategies = list(set(existing_matrix.index) & set(local_matrix.index))
                
                if common_strategies:
                    existing_aligned = existing_matrix.loc[common_strategies, common_strategies]
                    local_aligned = local_matrix.loc[common_strategies, common_strategies]
                    
                    integrated_matrix = (
                        priority * existing_aligned.values +
                        (1 - priority) * local_aligned.values
                    )
                    
                    integrated_df = pd.DataFrame(
                        integrated_matrix,
                        index=common_strategies,
                        columns=common_strategies
                    )
                    
                    # 結果構築
                    result = local_correlation_result.copy()
                    result['combined_correlation'] = integrated_df
                    result['integration_source'] = 'weighted_combination'
                    
                    return result
            
            # フォールバック
            return local_correlation_result
            
        except Exception as e:
            self.logger.warning(f"Correlation matrix integration failed: {e}")
            return local_correlation_result
    
    def _integrate_weight_data(
        self,
        local_weights: Dict[str, float],
        existing_weights: Optional[Dict[str, float]],
        learned_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """重みデータ統合"""
        
        try:
            if self.config.conflict_resolution_method == "weighted_average":
                return self._weighted_average_integration(
                    local_weights, existing_weights, learned_weights
                )
            elif self.config.conflict_resolution_method == "priority_based":
                return self._priority_based_integration(
                    local_weights, existing_weights, learned_weights
                )
            else:  # hybrid
                return self._hybrid_integration(
                    local_weights, existing_weights, learned_weights
                )
                
        except Exception as e:
            self.logger.warning(f"Weight data integration failed: {e}")
            return local_weights
    
    def _weighted_average_integration(
        self,
        local_weights: Dict[str, float],
        existing_weights: Optional[Dict[str, float]],
        learned_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """重み付き平均統合"""
        
        # 重み設定
        weights_config = {
            'local': 0.5,
            'existing': 0.3,
            'learned': 0.2
        }
        
        integrated_weights = {}
        all_strategies = set(local_weights.keys())
        
        if existing_weights:
            all_strategies.update(existing_weights.keys())
        if learned_weights:
            all_strategies.update(learned_weights.keys())
        
        for strategy in all_strategies:
            weight_sum = 0.0
            total_weight = 0.0
            
            # ローカル重み
            local_w = local_weights.get(strategy, 0.0)
            weight_sum += local_w * weights_config['local']
            total_weight += weights_config['local']
            
            # 既存システム重み
            if existing_weights and strategy in existing_weights:
                existing_w = existing_weights[strategy]
                weight_sum += existing_w * weights_config['existing']
                total_weight += weights_config['existing']
            
            # 学習重み
            if learned_weights and strategy in learned_weights:
                learned_w = learned_weights[strategy]
                weight_sum += learned_w * weights_config['learned']
                total_weight += weights_config['learned']
            
            integrated_weights[strategy] = weight_sum / total_weight if total_weight > 0 else 0.0
        
        # 正規化
        total = sum(integrated_weights.values())
        if total > 0:
            integrated_weights = {k: v/total for k, v in integrated_weights.items()}
        
        return integrated_weights
    
    def _priority_based_integration(
        self,
        local_weights: Dict[str, float],
        existing_weights: Optional[Dict[str, float]],
        learned_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """優先度ベース統合"""
        
        # 優先順位: local > learned > existing
        if local_weights:
            return local_weights
        elif learned_weights:
            return learned_weights
        elif existing_weights:
            return existing_weights
        else:
            return {}
    
    def _hybrid_integration(
        self,
        local_weights: Dict[str, float],
        existing_weights: Optional[Dict[str, float]],
        learned_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """ハイブリッド統合"""
        
        # 条件に応じて統合方式を切り替え
        if existing_weights is None and learned_weights is None:
            return local_weights
        
        # 重み分散度による判定
        local_variance = np.var(list(local_weights.values()))
        
        if local_variance > 0.01:  # 高分散→重み付き平均
            return self._weighted_average_integration(local_weights, existing_weights, learned_weights)
        else:  # 低分散（等重み傾向）→優先度ベース
            return self._priority_based_integration(local_weights, existing_weights, learned_weights)
    
    def _integrate_score_data(
        self,
        local_scores: Dict[str, float],
        existing_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """スコアデータ統合"""
        
        integrated_scores = {}
        all_strategies = set(local_scores.keys()) | set(existing_scores.keys())
        
        weight = self.config.score_integration_weight
        
        for strategy in all_strategies:
            local_score = local_scores.get(strategy, 1.0)
            existing_score = existing_scores.get(strategy, 1.0)
            
            # 重み付き平均
            integrated_score = weight * local_score + (1 - weight) * existing_score
            integrated_scores[strategy] = integrated_score
        
        return integrated_scores
    
    def _convert_existing_correlation_format(self, existing_correlation: Any) -> Dict[str, Any]:
        """既存相関データのフォーマット変換"""
        
        try:
            if hasattr(existing_correlation, 'correlation_matrix'):
                correlation_matrix = existing_correlation.correlation_matrix
            elif isinstance(existing_correlation, pd.DataFrame):
                correlation_matrix = existing_correlation
            elif isinstance(existing_correlation, np.ndarray):
                correlation_matrix = pd.DataFrame(existing_correlation)
            else:
                raise ValueError("Unknown correlation format")
            
            # 固有値計算
            eigenvalues = np.linalg.eigvals(correlation_matrix.values)
            eigenvalues = np.real(eigenvalues[eigenvalues > 1e-8])
            
            return {
                'combined_correlation': correlation_matrix,
                'eigenvalues': eigenvalues,
                'timeframe_correlations': {'existing_system': correlation_matrix},
                'risk_concentration': len(eigenvalues) / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation format conversion failed: {e}")
            return {}
    
    def _calculate_fallback_correlation(self, strategy_returns: pd.DataFrame) -> Dict[str, Any]:
        """フォールバック相関計算"""
        
        try:
            correlation_matrix = strategy_returns.corr()
            eigenvalues = np.linalg.eigvals(correlation_matrix.values)
            eigenvalues = np.real(eigenvalues[eigenvalues > 1e-8])
            
            return {
                'combined_correlation': correlation_matrix,
                'eigenvalues': eigenvalues,
                'timeframe_correlations': {'fallback': correlation_matrix},
                'risk_concentration': 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Fallback correlation calculation failed: {e}")
            return {}
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """システム健全性レポート"""
        
        report = {
            'integration_config': {
                'level': self.config.integration_level,
                'conflict_resolution': self.config.conflict_resolution_method,
                'fallback_enabled': self.config.fallback_on_error
            },
            'system_status': self.system_status.copy(),
            'available_systems': len([s for s in self.system_status.values() if s]),
            'total_systems': len(self.system_status),
            'cache_status': {
                'correlation_cache_size': len(self._correlation_cache),
                'weight_cache_size': len(self._weight_cache),
                'score_cache_size': len(self._score_cache)
            }
        }
        
        # 健全性スコア計算
        if self.system_status:
            health_score = sum(self.system_status.values()) / len(self.system_status)
            report['health_score'] = health_score
            
            if health_score >= 0.8:
                report['health_status'] = 'excellent'
            elif health_score >= 0.6:
                report['health_status'] = 'good'
            elif health_score >= 0.4:
                report['health_status'] = 'fair'
            else:
                report['health_status'] = 'poor'
        else:
            report['health_score'] = 0.0
            report['health_status'] = 'unknown'
        
        return report
    
    def clear_cache(self):
        """キャッシュクリア"""
        self._correlation_cache.clear()
        self._weight_cache.clear()
        self._score_cache.clear()
        self.logger.info("Integration bridge cache cleared")
    
    def update_config(self, new_config: IntegrationConfig):
        """設定更新"""
        self.config = new_config
        self._initialize_existing_systems()
        self.clear_cache()
        self.logger.info("Integration bridge configuration updated")
