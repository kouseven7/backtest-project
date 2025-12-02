"""
5-2-3 最適な重み付け比率の学習アルゴリズム メインシステム

ベイジアン最適化による階層的重み学習システムの統合実装
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# システムコンポーネントのインポート
from .bayesian_weight_optimizer import BayesianWeightOptimizer, OptimizationResult
from .performance_evaluator import PerformanceEvaluator, PerformanceMetrics
from .adaptive_learning_scheduler import AdaptiveLearningScheduler, LearningMode, AdjustmentResult
from .weight_constraint_manager import WeightConstraintManager
from .integration_bridge import IntegrationBridge, SystemIntegrationConfig
from .optimization_history_manager import OptimizationHistoryManager
from .meta_parameter_controller import MetaParameterController

@dataclass
class OptimalWeightLearningResult:
    """最適重み学習結果"""
    session_id: str
    optimized_weights: Dict[str, float]
    performance_metrics: PerformanceMetrics
    learning_mode: LearningMode
    adjustment_magnitude: float
    confidence_score: float
    integration_results: List[Any]
    execution_time: float
    timestamp: datetime

class OptimalWeightLearningSystem:
    """
    5-2-3 最適な重み付け比率の学習アルゴリズム
    
    ベイジアン最適化による階層的重み学習システム
    - Strategy score weights
    - Portfolio weights
    - Meta parameters
    
    統合された学習アルゴリズムによる期待値最大化とドローダウン最小化
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        workspace_path: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
            workspace_path: ワークスペースのパス
        """
        self.logger = self._setup_logger()
        
        # パスの設定
        self.workspace_path = workspace_path or str(Path.cwd())
        self.config_path = config_path or str(
            Path(self.workspace_path) / "config" / "weight_learning_config" / "weight_learning_config.json"
        )
        
        # 設定の読み込み
        self.config = self._load_config()
        
        # システムコンポーネントの初期化
        self._initialize_components()
        
        # システム状態
        self.system_status = {
            'initialized': True,
            'last_optimization': None,
            'total_optimizations': 0,
            'system_health': 'healthy'
        }
        
        self.logger.info("OptimalWeightLearningSystem initialized successfully")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.OptimalWeightLearningSystem")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Error loading config: {e}, using default configuration")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定の取得"""
        return {
            "bayesian_optimization": {
                "kernel_type": "matern",
                "max_iterations": 50,
                "acquisition_function": "EI"
            },
            "performance_evaluation": {
                "target_return": 0.10,
                "max_acceptable_drawdown": 0.20
            },
            "adaptive_learning": {
                "micro_adjustment_threshold": 0.02,
                "standard_optimization_threshold": 0.05,
                "major_rebalancing_threshold": 0.20
            }
        }
        
    def _initialize_components(self) -> None:
        """システムコンポーネントの初期化"""
        self.logger.info("Initializing system components")
        
        # 統合ブリッジの初期化
        integration_config = SystemIntegrationConfig(
            **self.config.get('integration', {})
        )
        self.integration_bridge = IntegrationBridge(
            config=integration_config,
            workspace_path=self.workspace_path
        )
        
        # パフォーマンス評価器の初期化
        self.performance_evaluator = PerformanceEvaluator(
            risk_free_rate=self.config.get('performance_evaluation', {}).get('risk_free_rate', 0.02)
        )
        
        # ベイジアン重み最適化器の初期化
        self.bayesian_optimizer = BayesianWeightOptimizer(
            integration_bridge=self.integration_bridge
        )
        
        # 適応的学習スケジューラーの初期化
        self.learning_scheduler = AdaptiveLearningScheduler(
            performance_evaluator=self.performance_evaluator
        )
        
        # 重み制約管理器の初期化
        self.constraint_manager = WeightConstraintManager()
        
        # 最適化履歴管理器の初期化
        history_config = self.config.get('optimization_history', {})
        storage_path = history_config.get('storage_path', 'optimization_history')
        self.history_manager = OptimizationHistoryManager(
            storage_path=str(Path(self.workspace_path) / storage_path),
            max_history_days=history_config.get('max_history_days', 365)
        )
        
        # メタパラメータコントローラーの初期化
        self.meta_controller = MetaParameterController()
        
        self.logger.info("All system components initialized")
        
    def execute_optimal_learning(
        self,
        market_data: pd.DataFrame,
        performance_data: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        force_learning_mode: Optional[LearningMode] = None
    ) -> OptimalWeightLearningResult:
        """
        最適重み学習の実行
        
        Args:
            market_data: 市場データ
            performance_data: パフォーマンスデータ
            current_weights: 現在の重み
            force_learning_mode: 強制学習モード
            
        Returns:
            学習結果
        """
        start_time = datetime.now()
        self.logger.info("Starting optimal weight learning execution")
        
        try:
            # データの検証
            self._validate_input_data(market_data, performance_data)
            
            # 現在の重みの初期化
            if current_weights is None:
                current_weights = self._initialize_default_weights()
                
            # メタパラメータの更新
            self._update_meta_parameters(performance_data)
            
            # 学習モードの決定
            if force_learning_mode:
                learning_mode = force_learning_mode
                mode_reason = "forced_mode"
            else:
                current_performance = self._calculate_current_performance(
                    performance_data, current_weights
                )
                learning_mode, mode_reason = self.learning_scheduler.determine_learning_mode(
                    current_performance, market_data
                )
                
            # 最適化セッションの開始
            session_id = self.history_manager.start_optimization_session(
                current_weights, learning_mode.value
            )
            
            # 最適化の実行
            optimization_result = self._execute_bayesian_optimization(
                current_weights, market_data, performance_data, learning_mode
            )
            
            # 制約の検証と修正
            optimized_weights = self._apply_constraints(optimization_result.optimized_weights)
            
            # パフォーマンス評価
            performance_metrics = self.performance_evaluator.evaluate_performance(
                performance_data, optimized_weights
            )
            
            # 統合システムへの適用
            integration_results = self.integration_bridge.apply_optimized_weights(
                optimized_weights, market_data, performance_data
            )
            
            # 適応的学習の実行
            adjustment_result = self.learning_scheduler.execute_adaptive_learning(
                optimized_weights, performance_data, market_data
            )
            
            # 履歴の記録
            self._record_optimization_result(
                session_id, optimized_weights, performance_metrics, learning_mode
            )
            
            # セッションの終了
            convergence_achieved = optimization_result.expected_performance > 0
            self.history_manager.end_optimization_session(convergence_achieved)
            
            # 実行時間の計算
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 結果の作成
            result = OptimalWeightLearningResult(
                session_id=session_id,
                optimized_weights=optimized_weights,
                performance_metrics=performance_metrics,
                learning_mode=learning_mode,
                adjustment_magnitude=adjustment_result.adjustment_magnitude,
                confidence_score=adjustment_result.confidence_score,
                integration_results=integration_results,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            # システム状態の更新
            self._update_system_status(result)
            
            self.logger.info(
                f"Optimal weight learning completed successfully in {execution_time:.2f}s, "
                f"mode: {learning_mode.value}, performance: {performance_metrics.combined_score:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in optimal weight learning execution: {e}")
            self.system_status['system_health'] = 'error'
            raise
            
    def _validate_input_data(
        self,
        market_data: pd.DataFrame,
        performance_data: pd.DataFrame
    ) -> None:
        """入力データの検証"""
        if market_data.empty or performance_data.empty:
            raise ValueError("Market data or performance data is empty")
            
        if len(market_data) < 10 or len(performance_data) < 10:
            raise ValueError("Insufficient data points for optimization")
            
        self.logger.debug("Input data validation passed")
        
    def _initialize_default_weights(self) -> Dict[str, float]:
        """デフォルト重みの初期化"""
        return {
            # ストラテジー重み（合計1.0）
            'strategy_trend_following': 0.3,
            'strategy_mean_reversion': 0.3,
            'strategy_momentum': 0.2,
            'strategy_volatility_breakout': 0.2,
            
            # ポートフォリオ重み（合計1.0）
            'portfolio_stocks': 0.6,
            'portfolio_bonds': 0.2,
            'portfolio_commodities': 0.1,
            'portfolio_alternatives': 0.1,
            
            # メタパラメータ
            'meta_learning_rate': 1.0,
            'meta_volatility_scaling': 1.0,
            'meta_risk_aversion': 1.0,
            'meta_rebalancing_threshold': 0.05
        }
        
    def _update_meta_parameters(self, performance_data: pd.DataFrame) -> None:
        """メタパラメータの更新"""
        # 現在のパフォーマンススコアの計算
        if len(performance_data) > 0:
            recent_performance = performance_data.tail(10).mean().mean()
            performance_feedback = {'combined_score': recent_performance}
            
            # メタパラメータの適応的調整
            self.meta_controller.adapt_parameters(performance_feedback)
            
    def _calculate_current_performance(
        self,
        performance_data: pd.DataFrame,
        current_weights: Dict[str, float]
    ) -> float:
        """現在のパフォーマンスの計算"""
        metrics = self.performance_evaluator.evaluate_performance(
            performance_data, current_weights
        )
        return metrics.combined_score
        
    def _execute_bayesian_optimization(
        self,
        current_weights: Dict[str, float],
        market_data: pd.DataFrame,
        performance_data: pd.DataFrame,
        learning_mode: LearningMode
    ) -> OptimizationResult:
        """ベイジアン最適化の実行"""
        # 制約の設定
        constraints = self._prepare_constraints()
        
        # 最適化の初期化
        self.bayesian_optimizer.initialize_optimization(
            {k.replace('strategy_', ''): v for k, v in current_weights.items() if k.startswith('strategy_')},
            {k.replace('portfolio_', ''): v for k, v in current_weights.items() if k.startswith('portfolio_')},
            {k.replace('meta_', ''): v for k, v in current_weights.items() if k.startswith('meta_')},
            constraints
        )
        
        # 最適化の実行
        target_metrics = ['expected_return', 'max_drawdown', 'sharpe_ratio']
        result = self.bayesian_optimizer.optimize_weights(performance_data, target_metrics)
        
        return result
        
    def _prepare_constraints(self) -> Dict[str, Dict[str, float]]:
        """制約の準備"""
        weight_config = self.config.get('weight_constraints', {})
        
        constraints = {
            'weights': {},
            'meta': {}
        }
        
        # ストラテジー制約
        for strategy, bounds in weight_config.get('strategy_weights', {}).items():
            constraints['weights'][f'{strategy}_min'] = bounds.get('min', 0.0)
            constraints['weights'][f'{strategy}_max'] = bounds.get('max', 1.0)
            
        # ポートフォリオ制約
        portfolio_config = weight_config.get('portfolio_weights', {})
        if 'max_single_asset' in portfolio_config:
            constraints['weights']['max_single_asset'] = portfolio_config['max_single_asset']
            
        # メタパラメータ制約
        for param, bounds in weight_config.get('meta_parameters', {}).items():
            constraints['meta'][f'{param}_min'] = bounds.get('min', 0.1)
            constraints['meta'][f'{param}_max'] = bounds.get('max', 3.0)
            
        return constraints
        
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """制約の適用"""
        # 制約検証
        is_valid, violations = self.constraint_manager.validate_weights(weights)
        
        if not is_valid:
            self.logger.warning(f"Constraint violations detected: {len(violations)}")
            # 制約修正の適用
            corrected_weights = self.constraint_manager.apply_constraint_corrections(weights)
            return corrected_weights
        else:
            return weights
            
    def _record_optimization_result(
        self,
        session_id: str,
        weights: Dict[str, float],
        performance_metrics: PerformanceMetrics,
        learning_mode: LearningMode
    ) -> None:
        """最適化結果の記録"""
        # 個別指標の準備
        individual_metrics = {
            'expected_return': performance_metrics.expected_return,
            'volatility': performance_metrics.volatility,
            'sharpe_ratio': performance_metrics.sharpe_ratio,
            'max_drawdown': performance_metrics.max_drawdown,
            'calmar_ratio': performance_metrics.calmar_ratio,
            'win_rate': performance_metrics.win_rate
        }
        
        # 履歴への記録
        self.history_manager.record_iteration(
            weights=weights,
            performance_score=performance_metrics.combined_score,
            individual_metrics=individual_metrics,
            iteration=1  # 単一反復の場合
        )
        
    def _update_system_status(self, result: OptimalWeightLearningResult) -> None:
        """システム状態の更新"""
        self.system_status.update({
            'last_optimization': result.timestamp,
            'total_optimizations': self.system_status['total_optimizations'] + 1,
            'system_health': 'healthy'
        })
        
    def get_system_summary(self) -> Dict[str, Any]:
        """システムサマリーの取得"""
        # 最適化統計
        optimization_stats = self.history_manager.get_optimization_statistics(30)
        
        # 学習統計
        learning_stats = self.learning_scheduler.get_learning_statistics()
        
        # パラメータサマリー
        parameter_summary = self.meta_controller.get_parameter_summary()
        
        # 統合サマリー
        integration_summary = self.integration_bridge.get_integration_summary()
        
        return {
            'system_status': self.system_status,
            'optimization_statistics': optimization_stats,
            'learning_statistics': learning_stats,
            'parameter_summary': parameter_summary,
            'integration_summary': integration_summary,
            'constraint_summary': self.constraint_manager.get_constraint_summary()
        }
        
    def export_complete_history(
        self,
        export_path: Optional[str] = None,
        include_detailed_weights: bool = True
    ) -> str:
        """完全履歴のエクスポート"""
        if export_path is None:
            export_path = str(Path(self.workspace_path) / "weight_learning_export")
            
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 最適化履歴のエクスポート
        history_file = self.history_manager.export_history(
            export_format="csv",
            include_weights=include_detailed_weights
        )
        
        # システムサマリーのエクスポート
        summary = self.get_system_summary()
        summary_file = export_dir / f"system_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
        # 制約違反履歴のエクスポート
        violations_file = export_dir / f"constraint_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.constraint_manager.export_violation_history(str(violations_file))
        
        self.logger.info(f"Complete history exported to {export_dir}")
        return str(export_dir)
        
    def reset_system_state(self, reset_history: bool = False) -> None:
        """システム状態のリセット"""
        self.logger.info("Resetting system state")
        
        # パラメータのリセット
        for param_name in self.meta_controller.get_current_parameters().keys():
            self.meta_controller.reset_parameter(param_name)
            
        # システム状態のリセット
        self.system_status.update({
            'last_optimization': None,
            'total_optimizations': 0,
            'system_health': 'healthy'
        })
        
        if reset_history:
            # 履歴のリセット（注意：データが失われます）
            self.history_manager = OptimizationHistoryManager(
                storage_path=str(Path(self.workspace_path) / "optimization_history"),
                max_history_days=365
            )
            
        self.logger.info("System state reset completed")
        
    def perform_system_health_check(self) -> Dict[str, Any]:
        """システムヘルスチェック"""
        health_status = {
            'overall_health': 'healthy',
            'components': {},
            'recommendations': []
        }
        
        try:
            # コンポーネントの健全性チェック
            
            # 統合ブリッジの状態
            integration_status = self.integration_bridge.get_system_status()
            health_status['components']['integration_bridge'] = integration_status
            
            # 最適化履歴の状態
            history_stats = self.history_manager.get_optimization_statistics(7)  # 直近1週間
            health_status['components']['optimization_history'] = {
                'recent_sessions': history_stats.get('total_sessions', 0),
                'performance_trend': history_stats.get('performance_trend', 'unknown')
            }
            
            # 制約違反の状況
            constraint_summary = self.constraint_manager.get_constraint_summary()
            health_status['components']['constraints'] = {
                'recent_violations': constraint_summary.get('recent_violations', 0)
            }
            
            # 推奨事項の生成
            recommendations = []
            
            if history_stats.get('total_sessions', 0) == 0:
                recommendations.append("No recent optimization sessions - consider running optimization")
                
            if constraint_summary.get('recent_violations', 0) > 5:
                recommendations.append("High number of recent constraint violations - review constraint settings")
                
            if history_stats.get('performance_trend') == 'deteriorating':
                recommendations.append("Performance trend is deteriorating - consider parameter adjustment")
                
            health_status['recommendations'] = recommendations
            
            if recommendations:
                health_status['overall_health'] = 'needs_attention'
                
        except Exception as e:
            health_status['overall_health'] = 'error'
            health_status['error'] = str(e)
            
        return health_status
