"""
5-3-3 戦略間相関を考慮した配分最適化 - デモスクリプト

相関ベース配分最適化システムの包括的なデモンストレーション

Author: imega
Created: 2025-01-27
Task: 5-3-3
Usage: python demo_5_3_3_correlation_allocation_system.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# システムコンポーネントのインポート
try:
    from config.portfolio_correlation_optimizer.correlation_based_allocator import (
        CorrelationBasedAllocator, AllocationConfig, AllocationResult
    )
    from config.portfolio_correlation_optimizer.optimization_engine import (
        HybridOptimizationEngine, OptimizationConfig, OptimizationMethod
    )
    from config.portfolio_correlation_optimizer.constraint_manager import (
        CorrelationConstraintManager, ConstraintConfig
    )
    from config.portfolio_correlation_optimizer.integration_bridge import (
        SystemIntegrationBridge, IntegrationConfig
    )
    from config.portfolio_correlation_optimizer.configs.system_config import (
        get_config_preset, create_custom_config, CONFIG_PRESETS
    )
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: System components not available: {e}")
    SYSTEM_AVAILABLE = False

class CorrelationAllocationDemo:
    """相関ベース配分最適化デモ"""
    
    def __init__(self, config_preset: str = 'balanced'):
        """
        初期化
        
        Args:
            config_preset: 設定プリセット名
        """
        self.setup_logging()
        
        if not SYSTEM_AVAILABLE:
            self.logger.error("System components not available. Exiting.")
            sys.exit(1)
        
        # 設定読み込み
        self.system_config = get_config_preset(config_preset)
        self.config_preset = config_preset
        
        # システム初期化
        self.allocator = CorrelationBasedAllocator(
            config=self.system_config.allocation_config,
            logger=self.logger
        )
        
        self.constraint_manager = CorrelationConstraintManager(
            config=self.system_config.constraint_config,
            logger=self.logger
        )
        
        self.integration_bridge = SystemIntegrationBridge(
            config=self.system_config.integration_config,
            logger=self.logger
        )
        
        # データとログ
        self.demo_results = {}
        self.performance_log = []
        
        self.logger.info(f"Demo initialized with {config_preset} configuration")
    
    def setup_logging(self):
        """ロギング設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'demo_5_3_3_correlation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_sample_data(
        self,
        n_strategies: int = 8,
        n_days: int = 500,
        correlation_structure: str = 'mixed'
    ) -> pd.DataFrame:
        """
        サンプルデータ生成
        
        Args:
            n_strategies: 戦略数
            n_days: データ期間
            correlation_structure: 相関構造 ('mixed', 'high_corr', 'low_corr', 'clustered')
            
        Returns:
            戦略リターンデータ
        """
        
        np.random.seed(42)  # 再現性のため
        
        # 戦略名生成
        strategy_names = [f'Strategy_{i+1:02d}' for i in range(n_strategies)]
        
        # 日付インデックス
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_days),
            periods=n_days,
            freq='D'
        )
        
        # 相関構造に応じたリターン生成
        if correlation_structure == 'mixed':
            # 混合構造：一部高相関、一部低相関
            returns = self._generate_mixed_correlation_returns(n_strategies, n_days)
        elif correlation_structure == 'high_corr':
            # 高相関構造
            returns = self._generate_high_correlation_returns(n_strategies, n_days)
        elif correlation_structure == 'low_corr':
            # 低相関構造
            returns = self._generate_low_correlation_returns(n_strategies, n_days)
        elif correlation_structure == 'clustered':
            # クラスタ構造
            returns = self._generate_clustered_returns(n_strategies, n_days)
        else:
            # デフォルト：ランダム
            returns = np.random.multivariate_normal(
                mean=np.random.normal(0.0008, 0.0002, n_strategies),
                cov=np.eye(n_strategies) * 0.01,
                size=n_days
            )
        
        # DataFrame作成
        strategy_returns = pd.DataFrame(
            returns,
            index=dates,
            columns=strategy_names
        )
        
        self.logger.info(f"Generated sample data: {n_strategies} strategies, {n_days} days, {correlation_structure} correlation structure")
        
        return strategy_returns
    
    def _generate_mixed_correlation_returns(self, n_strategies: int, n_days: int) -> np.ndarray:
        """混合相関構造リターン生成"""
        
        # 基本パラメータ
        base_mean = 0.0008
        base_vol = 0.015
        
        # 3グループに分割
        group_size = n_strategies // 3
        
        returns = np.zeros((n_days, n_strategies))
        
        # グループ1：高相関（0.7-0.8）
        if group_size > 0:
            common_factor1 = np.random.normal(0, base_vol * 0.8, n_days)
            for i in range(group_size):
                individual_factor = np.random.normal(0, base_vol * 0.4, n_days)
                returns[:, i] = base_mean + common_factor1 + individual_factor
        
        # グループ2：中相関（0.3-0.5）
        if group_size > 0:
            common_factor2 = np.random.normal(0, base_vol * 0.5, n_days)
            for i in range(group_size, min(2 * group_size, n_strategies)):
                individual_factor = np.random.normal(0, base_vol * 0.7, n_days)
                returns[:, i] = base_mean + common_factor2 + individual_factor
        
        # グループ3：低相関（0.0-0.2）
        for i in range(2 * group_size, n_strategies):
            returns[:, i] = np.random.normal(base_mean, base_vol, n_days)
        
        return returns
    
    def _generate_high_correlation_returns(self, n_strategies: int, n_days: int) -> np.ndarray:
        """高相関構造リターン生成"""
        
        # 共通ファクターが支配的
        common_factor = np.random.normal(0, 0.012, n_days)
        returns = np.zeros((n_days, n_strategies))
        
        for i in range(n_strategies):
            individual_factor = np.random.normal(0, 0.005, n_days)
            returns[:, i] = 0.0008 + common_factor + individual_factor
        
        return returns
    
    def _generate_low_correlation_returns(self, n_strategies: int, n_days: int) -> np.ndarray:
        """低相関構造リターン生成"""
        
        # 各戦略が独立
        returns = np.zeros((n_days, n_strategies))
        
        for i in range(n_strategies):
            returns[:, i] = np.random.normal(0.0008, 0.015, n_days)
        
        return returns
    
    def _generate_clustered_returns(self, n_strategies: int, n_days: int) -> np.ndarray:
        """クラスタ構造リターン生成"""
        
        # 2つのクラスターを作成
        cluster1_size = n_strategies // 2
        cluster2_size = n_strategies - cluster1_size
        
        returns = np.zeros((n_days, n_strategies))
        
        # クラスター1
        cluster1_factor = np.random.normal(0, 0.010, n_days)
        for i in range(cluster1_size):
            individual_factor = np.random.normal(0, 0.008, n_days)
            returns[:, i] = 0.0010 + cluster1_factor + individual_factor
        
        # クラスター2
        cluster2_factor = np.random.normal(0, 0.010, n_days)
        for i in range(cluster1_size, n_strategies):
            individual_factor = np.random.normal(0, 0.008, n_days)
            returns[:, i] = 0.0006 + cluster2_factor + individual_factor
        
        return returns
    
    def run_basic_allocation_demo(self, strategy_returns: pd.DataFrame) -> Dict[str, Any]:
        """基本配分デモ実行"""
        
        self.logger.info("Starting basic allocation demo")
        
        # 戦略スコア生成（サンプル）
        strategy_scores = {}
        for strategy in strategy_returns.columns:
            # 簡易スコア：シャープレシオベース
            mean_return = strategy_returns[strategy].mean()
            volatility = strategy_returns[strategy].std()
            sharpe = mean_return / volatility if volatility > 0 else 0
            strategy_scores[strategy] = max(0.1, min(2.0, 1.0 + sharpe))
        
        # 配分最適化実行
        start_time = datetime.now()
        
        allocation_result = self.allocator.allocate_portfolio(
            strategy_returns=strategy_returns,
            strategy_scores=strategy_scores
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # 結果保存
        demo_result = {
            'allocation_result': allocation_result,
            'strategy_scores': strategy_scores,
            'execution_time': execution_time,
            'metadata': {
                'config_preset': self.config_preset,
                'n_strategies': len(strategy_returns.columns),
                'data_period': len(strategy_returns),
                'timestamp': end_time
            }
        }
        
        self.demo_results['basic_allocation'] = demo_result
        
        # サマリー出力
        print("\n" + "="*60)
        print("BASIC ALLOCATION DEMO RESULTS")
        print("="*60)
        
        print(f"\nExecution Time: {execution_time:.3f} seconds")
        print(f"Optimization Status: {allocation_result.metadata.get('optimization_status', 'Unknown')}")
        
        print("\nStrategy Weights:")
        sorted_weights = sorted(allocation_result.strategy_weights.items(), 
                              key=lambda x: x[1], reverse=True)
        for strategy, weight in sorted_weights:
            score = strategy_scores.get(strategy, 1.0)
            print(f"  {strategy:<15}: {weight:>6.3f} (Score: {score:.3f})")
        
        print(f"\nRisk Metrics:")
        for metric, value in allocation_result.risk_metrics.items():
            print(f"  {metric:<25}: {value:>8.4f}")
        
        print(f"\nPerformance Prediction:")
        for metric, value in allocation_result.performance_prediction.items():
            print(f"  {metric:<25}: {value:>8.4f}")
        
        self.logger.info("Basic allocation demo completed")
        
        return demo_result
    
    def run_constraint_analysis_demo(self, strategy_returns: pd.DataFrame) -> Dict[str, Any]:
        """制約分析デモ実行"""
        
        self.logger.info("Starting constraint analysis demo")
        
        # 様々な重みパターンで制約検証
        test_cases = {
            'equal_weight': {name: 1.0/len(strategy_returns.columns) 
                           for name in strategy_returns.columns},
            'concentrated': {name: 0.8 if i == 0 else 0.2/(len(strategy_returns.columns)-1) 
                           for i, name in enumerate(strategy_returns.columns)},
            'diversified': {name: max(0.05, min(0.2, np.random.random())) 
                          for name in strategy_returns.columns}
        }
        
        # 正規化
        for case_name, weights in test_cases.items():
            total = sum(weights.values())
            test_cases[case_name] = {k: v/total for k, v in weights.items()}
        
        constraint_results = {}
        
        for case_name, weights in test_cases.items():
            print(f"\n--- Constraint Analysis: {case_name.upper()} ---")
            
            # 制約検証
            constraint_result = self.constraint_manager.validate_constraints(
                weights=weights,
                strategy_returns=strategy_returns
            )
            
            constraint_results[case_name] = constraint_result
            
            # 結果表示
            print(f"Feasible: {'Yes' if constraint_result.is_feasible else 'No'}")
            print(f"Violations: {len(constraint_result.violations)}")
            
            if constraint_result.violations:
                print("Violation Details:")
                for violation in constraint_result.violations[:3]:  # 最大3つ
                    print(f"  - {violation.constraint_type.value}: {violation.description}")
                    print(f"    Current: {violation.current_value:.4f}, Limit: {violation.limit_value:.4f}")
            
            # 制約違反時の調整
            if not constraint_result.is_feasible:
                adjusted_weights = self.constraint_manager.adjust_weights_for_constraints(
                    weights=weights,
                    constraint_result=constraint_result,
                    strategy_returns=strategy_returns
                )
                
                print("Adjusted Weights:")
                for strategy, adj_weight in sorted(adjusted_weights.items(), 
                                                 key=lambda x: x[1], reverse=True):
                    orig_weight = weights.get(strategy, 0.0)
                    print(f"  {strategy:<15}: {orig_weight:.3f} -> {adj_weight:.3f}")
        
        demo_result = {
            'test_cases': test_cases,
            'constraint_results': constraint_results,
            'metadata': {
                'constraint_config': self.system_config.constraint_config,
                'timestamp': datetime.now()
            }
        }
        
        self.demo_results['constraint_analysis'] = demo_result
        
        self.logger.info("Constraint analysis demo completed")
        
        return demo_result
    
    def run_integration_bridge_demo(self, strategy_returns: pd.DataFrame) -> Dict[str, Any]:
        """統合ブリッジデモ実行"""
        
        self.logger.info("Starting integration bridge demo")
        
        # システム健全性レポート
        health_report = self.integration_bridge.get_system_health_report()
        
        print("\n" + "="*60)
        print("INTEGRATION BRIDGE DEMO")
        print("="*60)
        
        print(f"\nSystem Health Report:")
        print(f"  Health Status: {health_report.get('health_status', 'unknown').upper()}")
        print(f"  Health Score: {health_report.get('health_score', 0.0):.2f}")
        print(f"  Available Systems: {health_report.get('available_systems', 0)}/{health_report.get('total_systems', 0)}")
        
        print(f"\nSystem Status:")
        system_status = health_report.get('system_status', {})
        for system, status in system_status.items():
            status_text = "✓ Available" if status else "✗ Unavailable"
            print(f"  {system:<20}: {status_text}")
        
        # 相関データ統合テスト
        correlation_result = self.integration_bridge.integrate_correlation_data(
            strategy_returns=strategy_returns
        )
        
        print(f"\nCorrelation Data Integration:")
        print(f"  Success: {'Yes' if correlation_result.success else 'No'}")
        print(f"  Data Sources: {correlation_result.integration_metadata.get('data_sources', [])}")
        if correlation_result.error_messages:
            print(f"  Errors: {len(correlation_result.error_messages)}")
        
        # サンプル重みでの統合テスト
        sample_weights = {name: 1.0/len(strategy_returns.columns) 
                         for name in strategy_returns.columns}
        
        weight_result = self.integration_bridge.integrate_weight_data(
            local_weights=sample_weights
        )
        
        print(f"\nWeight Data Integration:")
        print(f"  Success: {'Yes' if weight_result.success else 'No'}")
        print(f"  Data Sources: {weight_result.integration_metadata.get('data_sources', [])}")
        
        demo_result = {
            'health_report': health_report,
            'correlation_integration': correlation_result,
            'weight_integration': weight_result,
            'metadata': {
                'integration_config': self.system_config.integration_config,
                'timestamp': datetime.now()
            }
        }
        
        self.demo_results['integration_bridge'] = demo_result
        
        self.logger.info("Integration bridge demo completed")
        
        return demo_result
    
    def run_performance_comparison_demo(self, strategy_returns: pd.DataFrame) -> Dict[str, Any]:
        """パフォーマンス比較デモ実行"""
        
        self.logger.info("Starting performance comparison demo")
        
        # 複数設定での比較
        comparison_configs = ['conservative', 'balanced', 'aggressive', 'diversification']
        
        comparison_results = {}
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON DEMO")
        print("="*60)
        
        for config_name in comparison_configs:
            if config_name == self.config_preset:
                # 既に実行済み
                if 'basic_allocation' in self.demo_results:
                    comparison_results[config_name] = self.demo_results['basic_allocation']
                continue
            
            try:
                # 設定別アロケーター作成
                config = get_config_preset(config_name)
                allocator = CorrelationBasedAllocator(
                    config=config.allocation_config,
                    logger=self.logger
                )
                
                # 配分実行
                start_time = datetime.now()
                result = allocator.allocate_portfolio(strategy_returns=strategy_returns)
                end_time = datetime.now()
                
                comparison_results[config_name] = {
                    'allocation_result': result,
                    'execution_time': (end_time - start_time).total_seconds(),
                    'config_name': config_name
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to run {config_name} configuration: {e}")
        
        # 比較表示
        print(f"\nConfiguration Comparison:")
        print(f"{'Config':<15} {'ExecTime':<10} {'ExpRet':<10} {'ExpVol':<10} {'Sharpe':<10} {'DivRatio':<10}")
        print("-" * 70)
        
        for config_name, result_data in comparison_results.items():
            result = result_data['allocation_result']
            exec_time = result_data['execution_time']
            
            exp_ret = result.performance_prediction.get('expected_return', 0)
            exp_vol = result.performance_prediction.get('expected_volatility', 0)
            sharpe = result.performance_prediction.get('sharpe_ratio', 0)
            div_ratio = result.performance_prediction.get('diversification_ratio', 0)
            
            print(f"{config_name:<15} {exec_time:<10.3f} {exp_ret:<10.4f} {exp_vol:<10.4f} {sharpe:<10.4f} {div_ratio:<10.4f}")
        
        # 重み分析
        print(f"\nWeight Concentration Analysis:")
        print(f"{'Config':<15} {'MaxWeight':<12} {'Top3Conc':<12} {'EffectiveN':<12}")
        print("-" * 52)
        
        for config_name, result_data in comparison_results.items():
            result = result_data['allocation_result']
            weights = list(result.strategy_weights.values())
            
            max_weight = max(weights)
            top3_conc = sum(sorted(weights, reverse=True)[:3])
            effective_n = result.risk_metrics.get('effective_strategies', 0)
            
            print(f"{config_name:<15} {max_weight:<12.4f} {top3_conc:<12.4f} {effective_n:<12.2f}")
        
        demo_result = {
            'comparison_results': comparison_results,
            'metadata': {
                'compared_configs': comparison_configs,
                'timestamp': datetime.now()
            }
        }
        
        self.demo_results['performance_comparison'] = demo_result
        
        self.logger.info("Performance comparison demo completed")
        
        return demo_result
    
    def visualize_results(self, strategy_returns: pd.DataFrame):
        """結果可視化"""
        
        if 'basic_allocation' not in self.demo_results:
            self.logger.warning("No allocation results to visualize")
            return
        
        allocation_result = self.demo_results['basic_allocation']['allocation_result']
        
        # 図のセットアップ
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'5-3-3 Correlation-Based Allocation Results ({self.config_preset.title()})', 
                     fontsize=16, fontweight='bold')
        
        # 1. 重み分布
        weights = allocation_result.strategy_weights
        strategies = list(weights.keys())
        weight_values = list(weights.values())
        
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        bars = ax1.bar(range(len(strategies)), weight_values, color=colors)
        ax1.set_xlabel('Strategies')
        ax1.set_ylabel('Weight')
        ax1.set_title('Portfolio Weights')
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels([s.replace('Strategy_', 'S') for s in strategies], rotation=45)
        
        # 重み値をバーに表示
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 相関行列ヒートマップ
        ax2 = axes[0, 1]
        correlation_matrix = allocation_result.correlation_matrix
        
        if hasattr(correlation_matrix, 'values'):
            corr_values = correlation_matrix.values
            strategy_labels = [s.replace('Strategy_', 'S') for s in correlation_matrix.index]
        else:
            corr_values = correlation_matrix
            strategy_labels = [f'S{i+1:02d}' for i in range(len(correlation_matrix))]
        
        im = ax2.imshow(corr_values, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax2.set_title('Strategy Correlation Matrix')
        ax2.set_xticks(range(len(strategy_labels)))
        ax2.set_yticks(range(len(strategy_labels)))
        ax2.set_xticklabels(strategy_labels, rotation=45)
        ax2.set_yticklabels(strategy_labels)
        plt.colorbar(im, ax=ax2)
        
        # 3. リスク指標
        ax3 = axes[1, 0]
        risk_metrics = allocation_result.risk_metrics
        
        metric_names = list(risk_metrics.keys())
        metric_values = list(risk_metrics.values())
        
        bars = ax3.barh(metric_names, metric_values)
        ax3.set_xlabel('Value')
        ax3.set_title('Risk Metrics')
        
        # 値をバーに表示
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + max(metric_values) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=8)
        
        # 4. パフォーマンス予測
        ax4 = axes[1, 1]
        perf_metrics = allocation_result.performance_prediction
        
        perf_names = list(perf_metrics.keys())
        perf_values = list(perf_metrics.values())
        
        colors = ['green', 'orange', 'blue', 'red'][:len(perf_names)]
        bars = ax4.bar(perf_names, perf_values, color=colors)
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Prediction')
        ax4.tick_params(axis='x', rotation=45)
        
        # 値をバーに表示
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(perf_values) * 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # 保存
        filename = f'5_3_3_correlation_allocation_results_{self.config_preset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f"\nVisualization saved as: {filename}")
        
        # 比較結果の可視化（可能な場合）
        if 'performance_comparison' in self.demo_results:
            self._visualize_comparison_results()
        
        plt.show()
    
    def _visualize_comparison_results(self):
        """比較結果可視化"""
        
        comparison_data = self.demo_results['performance_comparison']['comparison_results']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Configuration Comparison', fontsize=16, fontweight='bold')
        
        configs = list(comparison_data.keys())
        
        # 1. パフォーマンス比較
        ax1 = axes[0]
        
        returns = []
        volatilities = []
        sharpe_ratios = []
        
        for config in configs:
            result = comparison_data[config]['allocation_result']
            perf = result.performance_prediction
            
            returns.append(perf.get('expected_return', 0))
            volatilities.append(perf.get('expected_volatility', 0))
            sharpe_ratios.append(perf.get('sharpe_ratio', 0))
        
        x = np.arange(len(configs))
        width = 0.25
        
        ax1.bar(x - width, returns, width, label='Expected Return', alpha=0.8)
        ax1.bar(x, volatilities, width, label='Expected Volatility', alpha=0.8)
        ax1.bar(x + width, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.8)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Value')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 集中度比較
        ax2 = axes[1]
        
        max_weights = []
        effective_strategies = []
        
        for config in configs:
            result = comparison_data[config]['allocation_result']
            weights = list(result.strategy_weights.values())
            
            max_weights.append(max(weights))
            effective_strategies.append(result.risk_metrics.get('effective_strategies', 0))
        
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(configs, max_weights, 'ro-', label='Max Weight', linewidth=2, markersize=8)
        line2 = ax2_twin.plot(configs, effective_strategies, 'bs-', label='Effective Strategies', 
                             linewidth=2, markersize=8)
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Max Weight', color='red')
        ax2_twin.set_ylabel('Effective Strategies', color='blue')
        ax2.set_title('Concentration Analysis')
        
        # 凡例統合
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存
        filename = f'5_3_3_configuration_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved as: {filename}")
    
    def generate_comprehensive_report(self) -> str:
        """包括的レポート生成"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("5-3-3 CORRELATION-BASED ALLOCATION OPTIMIZATION DEMO REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Configuration: {self.config_preset.upper()}")
        report_lines.append("")
        
        # システム情報
        report_lines.append("SYSTEM CONFIGURATION:")
        report_lines.append("-" * 40)
        config = self.system_config.allocation_config
        report_lines.append(f"  Correlation Timeframes: {config.correlation_timeframes}")
        report_lines.append(f"  Optimization Methods: {config.optimization_methods}")
        report_lines.append(f"  Weight Bounds: [{config.min_weight:.3f}, {config.max_weight:.3f}]")
        report_lines.append(f"  Turnover Limit: {config.turnover_limit:.3f}")
        report_lines.append(f"  Risk Aversion: {config.risk_aversion:.1f}")
        report_lines.append("")
        
        # デモ結果サマリー
        report_lines.append("DEMO RESULTS SUMMARY:")
        report_lines.append("-" * 40)
        
        for demo_name, demo_result in self.demo_results.items():
            report_lines.append(f"  ✓ {demo_name.replace('_', ' ').title()}")
            
            if 'execution_time' in demo_result:
                report_lines.append(f"    Execution Time: {demo_result['execution_time']:.3f}s")
            
            if 'allocation_result' in demo_result:
                result = demo_result['allocation_result']
                status = result.metadata.get('optimization_status', 'Unknown')
                report_lines.append(f"    Status: {status}")
        
        report_lines.append("")
        
        # パフォーマンス比較（可能な場合）
        if 'performance_comparison' in self.demo_results:
            report_lines.append("CONFIGURATION PERFORMANCE COMPARISON:")
            report_lines.append("-" * 50)
            
            comparison_data = self.demo_results['performance_comparison']['comparison_results']
            
            report_lines.append(f"{'Config':<15} {'Return':<10} {'Volatility':<12} {'Sharpe':<10} {'MaxWt':<10}")
            report_lines.append("-" * 57)
            
            for config_name, result_data in comparison_data.items():
                result = result_data['allocation_result']
                perf = result.performance_prediction
                
                exp_ret = perf.get('expected_return', 0)
                exp_vol = perf.get('expected_volatility', 0)
                sharpe = perf.get('sharpe_ratio', 0)
                max_wt = max(result.strategy_weights.values())
                
                report_lines.append(f"{config_name:<15} {exp_ret:<10.4f} {exp_vol:<12.4f} {sharpe:<10.4f} {max_wt:<10.4f}")
            
            report_lines.append("")
        
        # 推奨事項
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 30)
        
        if 'basic_allocation' in self.demo_results:
            result = self.demo_results['basic_allocation']['allocation_result']
            
            # 集中度分析
            weights = list(result.strategy_weights.values())
            max_weight = max(weights)
            top3_concentration = sum(sorted(weights, reverse=True)[:3])
            
            if max_weight > 0.5:
                report_lines.append("  • Consider reducing single strategy concentration")
            if top3_concentration > 0.8:
                report_lines.append("  • Portfolio shows high concentration in top strategies")
            
            # 分散効果
            div_ratio = result.performance_prediction.get('diversification_ratio', 1.0)
            if div_ratio < 1.2:
                report_lines.append("  • Limited diversification benefit observed")
            elif div_ratio > 2.0:
                report_lines.append("  • Excellent diversification achieved")
            
            # 相関リスク
            corr_risk = result.risk_metrics.get('correlation_risk', 0)
            if corr_risk > 0.7:
                report_lines.append("  • High correlation risk detected")
            elif corr_risk < 0.3:
                report_lines.append("  • Good correlation diversification")
        
        # システム健全性
        if 'integration_bridge' in self.demo_results:
            health_report = self.demo_results['integration_bridge']['health_report']
            health_score = health_report.get('health_score', 0)
            
            if health_score < 0.5:
                report_lines.append("  • Consider checking system integration health")
            elif health_score > 0.8:
                report_lines.append("  • System integration is functioning well")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # レポート保存
        report_content = "\n".join(report_lines)
        filename = f'5_3_3_demo_report_{self.config_preset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nComprehensive report saved as: {filename}")
        
        return report_content

def main():
    """メイン実行関数"""
    
    print("5-3-3 戦略間相関を考慮した配分最適化 - デモスクリプト")
    print("=" * 60)
    
    # 設定選択
    available_presets = list(CONFIG_PRESETS.keys())
    print(f"\nAvailable configurations: {', '.join(available_presets)}")
    
    # デフォルトは balanced
    config_preset = 'balanced'
    
    try:
        # デモ初期化
        demo = CorrelationAllocationDemo(config_preset=config_preset)
        
        # サンプルデータ生成
        print(f"\nGenerating sample data...")
        strategy_returns = demo.generate_sample_data(
            n_strategies=8,
            n_days=400,
            correlation_structure='mixed'
        )
        
        print(f"Data generated: {len(strategy_returns.columns)} strategies, {len(strategy_returns)} days")
        
        # 基本統計表示
        print(f"\nStrategy Return Statistics:")
        print(f"{'Strategy':<15} {'Mean':<10} {'Std':<10} {'Sharpe':<10}")
        print("-" * 45)
        
        for strategy in strategy_returns.columns:
            mean_ret = strategy_returns[strategy].mean() * 252  # 年率化
            std_ret = strategy_returns[strategy].std() * np.sqrt(252)  # 年率化
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            
            print(f"{strategy:<15} {mean_ret:<10.4f} {std_ret:<10.4f} {sharpe:<10.4f}")
        
        # デモ実行
        print(f"\nRunning demos...")
        
        # 1. 基本配分デモ
        demo.run_basic_allocation_demo(strategy_returns)
        
        # 2. 制約分析デモ
        demo.run_constraint_analysis_demo(strategy_returns)
        
        # 3. 統合ブリッジデモ
        demo.run_integration_bridge_demo(strategy_returns)
        
        # 4. パフォーマンス比較デモ
        demo.run_performance_comparison_demo(strategy_returns)
        
        # 5. 結果可視化
        print(f"\nGenerating visualizations...")
        demo.visualize_results(strategy_returns)
        
        # 6. 包括的レポート
        print(f"\nGenerating comprehensive report...")
        report = demo.generate_comprehensive_report()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nCheck the generated files:")
        print("  - Log file: demo_5_3_3_correlation_*.log")
        print("  - Visualization: 5_3_3_correlation_allocation_results_*.png")
        print("  - Report: 5_3_3_demo_report_*.txt")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
