"""
市場対応分析器：市場分類を考慮したパフォーマンス分析
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from .classification_integration import ClassificationIntegration
from ..market_classification.market_conditions import SimpleMarketCondition, DetailedMarketCondition

logger = logging.getLogger(__name__)


class MarketAwareAnalyzer:
    """市場分類を考慮した分析クラス"""
    
    def __init__(self, classification_integration: Optional[ClassificationIntegration] = None):
        """
        Args:
            classification_integration: 分類統合システム
        """
        self.integration = classification_integration or ClassificationIntegration()
        self.analysis_results: Dict[str, Any] = {}
        
    def analyze_strategy_performance_by_market(self, 
                                             backtest_results: Dict[str, Any], 
                                             market_classifications: Dict[str, Any]) -> Dict[str, Any]:
        """市場分類別の戦略パフォーマンス分析"""
        analysis = {
            'by_simple_condition': {},
            'by_detailed_condition': {},
            'strategy_market_fit': {},
            'market_condition_performance': {},
            'recommendations': {}
        }
        
        try:
            # 1. シンプル市場分類別分析
            for condition in SimpleMarketCondition:
                condition_key = condition.value
                analysis['by_simple_condition'][condition_key] = self._analyze_condition_performance(
                    backtest_results, market_classifications, 'simple_condition', condition_key
                )
            
            # 2. 詳細市場分類別分析
            for condition in DetailedMarketCondition:
                condition_key = condition.value
                analysis['by_detailed_condition'][condition_key] = self._analyze_condition_performance(
                    backtest_results, market_classifications, 'detailed_condition', condition_key
                )
            
            # 3. 戦略-市場適合度分析
            analysis['strategy_market_fit'] = self._analyze_strategy_market_fit(
                backtest_results, market_classifications
            )
            
            # 4. 市場状況別パフォーマンス
            analysis['market_condition_performance'] = self._analyze_market_condition_performance(
                backtest_results, market_classifications
            )
            
            # 5. 推奨事項の生成
            analysis['recommendations'] = self._generate_market_based_recommendations(analysis)
            
        except Exception as e:
            logger.error(f"Error in market-aware analysis: {e}")
            analysis['error'] = str(e)
        
        self.analysis_results = analysis
        return analysis
    
    def _analyze_condition_performance(self, 
                                     backtest_results: Dict[str, Any], 
                                     market_classifications: Dict[str, Any],
                                     condition_type: str,
                                     condition_value: str) -> Dict[str, Any]:
        """特定市場条件でのパフォーマンス分析"""
        performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'strategies_active': [],
            'best_strategy': None,
            'worst_strategy': None
        }
        
        try:
            # 該当する期間のデータを抽出
            matching_periods = []
            for symbol, classifications in market_classifications.items():
                if isinstance(classifications, dict) and condition_type in classifications:
                    if classifications[condition_type] == condition_value:
                        matching_periods.append(symbol)
            
            if not matching_periods:
                return performance
            
            # 戦略別パフォーマンス集計
            strategy_performance = {}
            total_trades = 0
            total_returns = []
            
            for strategy_name, strategy_results in backtest_results.items():
                if not isinstance(strategy_results, dict):
                    continue
                
                strategy_trades = 0
                strategy_returns = []
                
                # 該当期間のトレードを抽出（簡略化）
                if 'trades' in strategy_results:
                    trades = strategy_results['trades']
                    if isinstance(trades, list):
                        strategy_trades = len(trades)
                        # リターンの計算（実装は戦略結果の構造に依存）
                        for trade in trades:
                            if isinstance(trade, dict) and 'return' in trade:
                                strategy_returns.append(trade['return'])
                
                if strategy_trades > 0:
                    strategy_performance[strategy_name] = {
                        'trades': strategy_trades,
                        'returns': strategy_returns,
                        'avg_return': np.mean(strategy_returns) if strategy_returns else 0.0,
                        'total_return': sum(strategy_returns) if strategy_returns else 0.0
                    }
                    
                    total_trades += strategy_trades
                    total_returns.extend(strategy_returns)
            
            # 全体統計の計算
            if total_returns:
                performance['total_trades'] = total_trades
                performance['winning_trades'] = sum(1 for r in total_returns if r > 0)
                performance['losing_trades'] = sum(1 for r in total_returns if r <= 0)
                performance['win_rate'] = performance['winning_trades'] / len(total_returns)
                performance['avg_return'] = np.mean(total_returns)
                performance['total_return'] = sum(total_returns)
                performance['sharpe_ratio'] = np.mean(total_returns) / np.std(total_returns) if np.std(total_returns) > 0 else 0.0
                
                # ドローダウン計算（簡略化）
                cumulative_returns = np.cumsum(total_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                performance['max_drawdown'] = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            
            # 戦略ランキング
            if strategy_performance:
                performance['strategies_active'] = list(strategy_performance.keys())
                best_strategy = max(strategy_performance.items(), key=lambda x: x[1]['avg_return'])
                worst_strategy = min(strategy_performance.items(), key=lambda x: x[1]['avg_return'])
                performance['best_strategy'] = {
                    'name': best_strategy[0],
                    'avg_return': best_strategy[1]['avg_return']
                }
                performance['worst_strategy'] = {
                    'name': worst_strategy[0],
                    'avg_return': worst_strategy[1]['avg_return']
                }
        
        except Exception as e:
            logger.error(f"Error analyzing condition {condition_value}: {e}")
            performance['error'] = str(e)
        
        return performance
    
    def _analyze_strategy_market_fit(self, 
                                   backtest_results: Dict[str, Any], 
                                   market_classifications: Dict[str, Any]) -> Dict[str, Any]:
        """戦略と市場の適合度分析"""
        fit_analysis = {}
        
        try:
            for strategy_name, strategy_results in backtest_results.items():
                if not isinstance(strategy_results, dict):
                    continue
                
                strategy_fit = {
                    'best_market_conditions': [],
                    'worst_market_conditions': [],
                    'condition_performance': {},
                    'adaptability_score': 0.0
                }
                
                # 各市場条件でのパフォーマンスを計算
                condition_returns = {}
                for condition in SimpleMarketCondition:
                    condition_performance = self._analyze_condition_performance(
                        {strategy_name: strategy_results}, 
                        market_classifications, 
                        'simple_condition', 
                        condition.value
                    )
                    condition_returns[condition.value] = condition_performance['avg_return']
                    strategy_fit['condition_performance'][condition.value] = condition_performance
                
                # ベスト・ワーストコンディション
                if condition_returns:
                    sorted_conditions = sorted(condition_returns.items(), key=lambda x: x[1], reverse=True)
                    strategy_fit['best_market_conditions'] = [c[0] for c in sorted_conditions[:2]]
                    strategy_fit['worst_market_conditions'] = [c[0] for c in sorted_conditions[-2:]]
                    
                    # 適応性スコア（異なる市場条件での安定性）
                    returns_values = list(condition_returns.values())
                    if returns_values:
                        strategy_fit['adaptability_score'] = 1.0 / (1.0 + np.std(returns_values))
                
                fit_analysis[strategy_name] = strategy_fit
        
        except Exception as e:
            logger.error(f"Error in strategy-market fit analysis: {e}")
            fit_analysis['error'] = str(e)
        
        return fit_analysis
    
    def _analyze_market_condition_performance(self, 
                                            backtest_results: Dict[str, Any], 
                                            market_classifications: Dict[str, Any]) -> Dict[str, Any]:
        """市場状況別パフォーマンス分析"""
        market_performance = {
            'condition_rankings': {},
            'volatility_impact': {},
            'trend_strength_correlation': {},
            'confidence_impact': {}
        }
        
        try:
            # 市場状況ランキング
            condition_performance = {}
            for condition in SimpleMarketCondition:
                overall_performance = self._analyze_condition_performance(
                    backtest_results, market_classifications, 'simple_condition', condition.value
                )
                condition_performance[condition.value] = overall_performance['avg_return']
            
            # ランキング作成
            sorted_conditions = sorted(condition_performance.items(), key=lambda x: x[1], reverse=True)
            market_performance['condition_rankings'] = {
                'best_conditions': [c[0] for c in sorted_conditions[:3]],
                'worst_conditions': [c[0] for c in sorted_conditions[-3:]],
                'all_performance': dict(sorted_conditions)
            }
            
            # ボラティリティ影響分析（簡略化）
            market_performance['volatility_impact'] = {
                'high_volatility_performance': 0.0,  # 実装が必要
                'low_volatility_performance': 0.0,
                'volatility_correlation': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in market condition performance analysis: {e}")
            market_performance['error'] = str(e)
        
        return market_performance
    
    def _generate_market_based_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """市場分析に基づく推奨事項生成"""
        recommendations = {
            'strategy_allocation': {},
            'market_timing': {},
            'risk_management': {},
            'parameter_adjustments': {}
        }
        
        try:
            # 戦略配分推奨
            strategy_fit = analysis.get('strategy_market_fit', {})
            for strategy_name, fit_data in strategy_fit.items():
                if isinstance(fit_data, dict) and 'best_market_conditions' in fit_data:
                    recommendations['strategy_allocation'][strategy_name] = {
                        'recommended_markets': fit_data['best_market_conditions'],
                        'avoid_markets': fit_data['worst_market_conditions'],
                        'adaptability': fit_data.get('adaptability_score', 0.0)
                    }
            
            # マーケットタイミング推奨
            condition_rankings = analysis.get('market_condition_performance', {}).get('condition_rankings', {})
            if condition_rankings:
                recommendations['market_timing'] = {
                    'favorable_conditions': condition_rankings.get('best_conditions', []),
                    'unfavorable_conditions': condition_rankings.get('worst_conditions', []),
                    'overall_market_score': 0.0  # 実装が必要
                }
            
            # リスク管理推奨
            recommendations['risk_management'] = {
                'dynamic_position_sizing': True,
                'market_condition_filters': True,
                'volatility_adjustment': True
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations['error'] = str(e)
        
        return recommendations
    
    def generate_market_analysis_report(self, output_path: Optional[str] = None) -> str:
        """市場分析レポートの生成"""
        if not self.analysis_results:
            return "No analysis results available. Run analyze_strategy_performance_by_market first."
        
        report_lines = [
            "# Market-Aware Strategy Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
        ]
        
        # 市場状況別パフォーマンス
        simple_analysis = self.analysis_results.get('by_simple_condition', {})
        if simple_analysis:
            report_lines.extend([
                "",
                "## Performance by Market Condition",
            ])
            
            for condition, performance in simple_analysis.items():
                if isinstance(performance, dict):
                    report_lines.extend([
                        f"### {condition.upper()}",
                        f"- Total Trades: {performance.get('total_trades', 0)}",
                        f"- Win Rate: {performance.get('win_rate', 0):.2%}",
                        f"- Average Return: {performance.get('avg_return', 0):.4f}",
                        f"- Total Return: {performance.get('total_return', 0):.4f}",
                        f"- Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}",
                        ""
                    ])
        
        # 戦略適合度
        strategy_fit = self.analysis_results.get('strategy_market_fit', {})
        if strategy_fit:
            report_lines.extend([
                "## Strategy-Market Fit Analysis",
                ""
            ])
            
            for strategy, fit_data in strategy_fit.items():
                if isinstance(fit_data, dict):
                    best_markets = fit_data.get('best_market_conditions', [])
                    worst_markets = fit_data.get('worst_market_conditions', [])
                    adaptability = fit_data.get('adaptability_score', 0)
                    
                    report_lines.extend([
                        f"### {strategy}",
                        f"- Best Markets: {', '.join(best_markets)}",
                        f"- Worst Markets: {', '.join(worst_markets)}",
                        f"- Adaptability Score: {adaptability:.3f}",
                        ""
                    ])
        
        # 推奨事項
        recommendations = self.analysis_results.get('recommendations', {})
        if recommendations:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            
            strategy_allocation = recommendations.get('strategy_allocation', {})
            for strategy, allocation in strategy_allocation.items():
                if isinstance(allocation, dict):
                    recommended = allocation.get('recommended_markets', [])
                    avoid = allocation.get('avoid_markets', [])
                    
                    report_lines.extend([
                        f"### {strategy}",
                        f"- Recommended Markets: {', '.join(recommended)}",
                        f"- Avoid Markets: {', '.join(avoid)}",
                        ""
                    ])
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Market analysis report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report to {output_path}: {e}")
        
        return report_content
    
    def plot_market_performance_comparison(self, save_path: Optional[str] = None):
        """市場パフォーマンス比較の可視化"""
        if not self.analysis_results:
            logger.warning("No analysis results to plot")
            return
        
        try:
            simple_analysis = self.analysis_results.get('by_simple_condition', {})
            if not simple_analysis:
                logger.warning("No simple condition analysis to plot")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 市場条件別平均リターン
            conditions = []
            avg_returns = []
            for condition, performance in simple_analysis.items():
                if isinstance(performance, dict):
                    conditions.append(condition)
                    avg_returns.append(performance.get('avg_return', 0))
            
            if conditions and avg_returns:
                ax1.bar(conditions, avg_returns)
                ax1.set_title('Average Return by Market Condition')
                ax1.set_ylabel('Average Return')
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # 2. 勝率比較
            win_rates = []
            for condition in conditions:
                performance = simple_analysis.get(condition, {})
                win_rates.append(performance.get('win_rate', 0))
            
            if conditions and win_rates:
                ax2.bar(conditions, win_rates)
                ax2.set_title('Win Rate by Market Condition')
                ax2.set_ylabel('Win Rate')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # 3. トレード数比較
            trade_counts = []
            for condition in conditions:
                performance = simple_analysis.get(condition, {})
                trade_counts.append(performance.get('total_trades', 0))
            
            if conditions and trade_counts:
                ax3.bar(conditions, trade_counts)
                ax3.set_title('Total Trades by Market Condition')
                ax3.set_ylabel('Number of Trades')
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # 4. シャープレシオ比較
            sharpe_ratios = []
            for condition in conditions:
                performance = simple_analysis.get(condition, {})
                sharpe_ratios.append(performance.get('sharpe_ratio', 0))
            
            if conditions and sharpe_ratios:
                ax4.bar(conditions, sharpe_ratios)
                ax4.set_title('Sharpe Ratio by Market Condition')
                ax4.set_ylabel('Sharpe Ratio')
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Market performance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot market performance comparison: {e}")
    
    def export_analysis_results(self, output_path: str):
        """分析結果をJSONファイルにエクスポート"""
        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis results exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export analysis results: {e}")
