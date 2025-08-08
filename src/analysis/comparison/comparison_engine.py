"""
バックテストvs実運用比較分析エンジン
フェーズ4A3: バックテストvs実運用比較分析器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy import stats

class ComparisonEngine:
    """比較分析エンジン"""
    
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.comparison_metrics = config.get('comparison_metrics', {})
    
    async def execute_comparison(self, aligned_data: Dict, analysis_type: str = "adaptive") -> Dict:
        """比較分析実行"""
        try:
            self.logger.info(f"比較分析実行開始 [タイプ: {analysis_type}]")
            
            backtest_data = aligned_data.get('backtest', {})
            live_data = aligned_data.get('live', {})
            
            if not backtest_data or not live_data:
                raise ValueError("比較対象データが不足しています")
            
            # 分析レベル決定
            actual_analysis_type = self._determine_analysis_level(backtest_data, live_data, analysis_type)
            self.logger.info(f"実際の分析レベル: {actual_analysis_type}")
            
            # 戦略別比較
            strategy_comparisons = {}
            common_strategies = aligned_data.get('common_strategies', [])
            
            for strategy_name in common_strategies:
                try:
                    strategy_comparison = await self._compare_strategy(
                        strategy_name,
                        backtest_data.get('strategies', {}).get(strategy_name, {}),
                        live_data.get('strategies', {}).get(strategy_name, {}),
                        actual_analysis_type
                    )
                    strategy_comparisons[strategy_name] = strategy_comparison
                    
                except Exception as e:
                    self.logger.warning(f"戦略比較エラー [{strategy_name}]: {e}")
            
            # ポートフォリオレベル比較
            portfolio_comparison = await self._compare_portfolio(backtest_data, live_data, actual_analysis_type)
            
            comparison_results = {
                "analysis_type": actual_analysis_type,
                "timestamp": datetime.now(),
                "strategy_comparisons": strategy_comparisons,
                "portfolio_comparison": portfolio_comparison,
                "summary": self._create_comparison_summary(strategy_comparisons, portfolio_comparison)
            }
            
            self.logger.info(f"比較分析完了 - 戦略数: {len(strategy_comparisons)}")
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"比較分析実行エラー: {e}")
            return {}
    
    def execute_basic_comparison(self, backtest_data: Dict, live_data: Dict) -> Dict:
        """基本比較実行（同期）"""
        try:
            if not backtest_data or not live_data:
                return {}
            
            # 基本メトリクス比較
            basic_metrics = self.comparison_metrics.get('performance_metrics', {}).get('primary', 
                ['total_return', 'win_rate', 'max_drawdown', 'sharpe_ratio'])
            
            comparison_results = {}
            
            # 共通戦略抽出
            bt_strategies = set(backtest_data.keys())
            live_strategies = set(live_data.keys())
            common_strategies = bt_strategies.intersection(live_strategies)
            
            for strategy_name in common_strategies:
                bt_strategy = backtest_data.get(strategy_name, {})
                live_strategy = live_data.get(strategy_name, {})
                
                strategy_comparison = {}
                
                for metric in basic_metrics:
                    bt_value = self._extract_metric_value(bt_strategy, metric)
                    live_value = self._extract_metric_value(live_strategy, metric)
                    
                    if bt_value is not None and live_value is not None:
                        difference = live_value - bt_value
                        relative_difference = (difference / bt_value) if bt_value != 0 else 0
                        
                        strategy_comparison[metric] = {
                            "backtest": bt_value,
                            "live": live_value,
                            "difference": difference,
                            "relative_difference": relative_difference,
                            "performance_gap": "positive" if difference > 0 else "negative" if difference < 0 else "neutral"
                        }
                
                comparison_results[strategy_name] = strategy_comparison
            
            return {
                "type": "basic_comparison",
                "timestamp": datetime.now(),
                "results": comparison_results,
                "summary": self._create_basic_summary(comparison_results)
            }
            
        except Exception as e:
            self.logger.error(f"基本比較実行エラー: {e}")
            return {}
    
    def _determine_analysis_level(self, backtest_data: Dict, live_data: Dict, requested_type: str) -> str:
        """分析レベル決定"""
        try:
            if requested_type != "adaptive":
                return requested_type
            
            # データサイズとメトリクス豊富さに基づく自動決定
            total_strategies = len(backtest_data.get('strategies', {})) + len(live_data.get('strategies', {}))
            
            # データ品質評価
            has_rich_metrics = False
            for data_source in [backtest_data, live_data]:
                for strategy_data in data_source.get('strategies', {}).values():
                    risk_metrics = strategy_data.get('risk_metrics', {})
                    if len(risk_metrics) >= 3:  # 3つ以上のリスクメトリクス
                        has_rich_metrics = True
                        break
                if has_rich_metrics:
                    break
            
            if total_strategies >= 6 and has_rich_metrics:
                return "detailed"
            elif total_strategies >= 2:
                return "basic"
            else:
                return "basic"  # フォールバック
                
        except Exception as e:
            self.logger.warning(f"分析レベル決定エラー: {e}")
            return "basic"
    
    async def _compare_strategy(self, strategy_name: str, bt_data: Dict, live_data: Dict, analysis_type: str) -> Dict:
        """戦略別比較分析"""
        try:
            comparison = {
                "strategy_name": strategy_name,
                "analysis_type": analysis_type,
                "metrics_comparison": {},
                "statistical_analysis": {},
                "performance_gap_analysis": {}
            }
            
            # メトリクス比較
            if analysis_type == "basic":
                metrics = self.comparison_metrics.get('performance_metrics', {}).get('primary', [])
            else:
                primary = self.comparison_metrics.get('performance_metrics', {}).get('primary', [])
                secondary = self.comparison_metrics.get('performance_metrics', {}).get('secondary', [])
                metrics = primary + secondary
            
            for metric in metrics:
                bt_value = self._extract_metric_value(bt_data, metric)
                live_value = self._extract_metric_value(live_data, metric)
                
                if bt_value is not None and live_value is not None:
                    comparison["metrics_comparison"][metric] = {
                        "backtest": bt_value,
                        "live": live_value,
                        "absolute_difference": live_value - bt_value,
                        "relative_difference": (live_value - bt_value) / bt_value if bt_value != 0 else 0,
                        "performance_ratio": live_value / bt_value if bt_value != 0 else 1
                    }
            
            # 統計分析（詳細モードのみ）
            if analysis_type == "detailed":
                comparison["statistical_analysis"] = await self._perform_statistical_analysis(bt_data, live_data)
            
            # パフォーマンスギャップ分析
            comparison["performance_gap_analysis"] = self._analyze_performance_gap(
                comparison["metrics_comparison"]
            )
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"戦略比較分析エラー [{strategy_name}]: {e}")
            return {}
    
    async def _compare_portfolio(self, backtest_data: Dict, live_data: Dict, analysis_type: str) -> Dict:
        """ポートフォリオレベル比較"""
        try:
            portfolio_comparison = {
                "analysis_type": analysis_type,
                "aggregate_metrics": {},
                "diversification_analysis": {},
                "risk_analysis": {}
            }
            
            # 集計メトリクス比較
            bt_aggregate = self._calculate_portfolio_aggregates(backtest_data)
            live_aggregate = self._calculate_portfolio_aggregates(live_data)
            
            for metric in bt_aggregate:
                if metric in live_aggregate:
                    bt_value = bt_aggregate[metric]
                    live_value = live_aggregate[metric]
                    
                    portfolio_comparison["aggregate_metrics"][metric] = {
                        "backtest": bt_value,
                        "live": live_value,
                        "difference": live_value - bt_value,
                        "relative_difference": (live_value - bt_value) / bt_value if bt_value != 0 else 0
                    }
            
            # 分散化分析
            if analysis_type == "detailed":
                portfolio_comparison["diversification_analysis"] = self._analyze_diversification(
                    backtest_data, live_data
                )
                
                portfolio_comparison["risk_analysis"] = self._analyze_portfolio_risk(
                    backtest_data, live_data
                )
            
            return portfolio_comparison
            
        except Exception as e:
            self.logger.warning(f"ポートフォリオ比較エラー: {e}")
            return {}
    
    def _extract_metric_value(self, strategy_data: Dict, metric_name: str) -> Optional[float]:
        """メトリクス値抽出"""
        try:
            # 基本メトリクス
            basic_metrics = strategy_data.get('basic_metrics', {})
            if metric_name in basic_metrics:
                return float(basic_metrics[metric_name])
            
            # リスクメトリクス
            risk_metrics = strategy_data.get('risk_metrics', {})
            if metric_name in risk_metrics:
                return float(risk_metrics[metric_name])
            
            # 派生メトリクス
            derived_metrics = strategy_data.get('derived_metrics', {})
            if metric_name in derived_metrics:
                return float(derived_metrics[metric_name])
            
            # メトリクス名のマッピング
            metric_mapping = {
                'total_return': 'total_pnl',
                'annualized_return': 'total_pnl',  # 簡易計算
                'volatility': 'volatility'
            }
            
            if metric_name in metric_mapping:
                mapped_metric = metric_mapping[metric_name]
                return self._extract_metric_value(strategy_data, mapped_metric)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"メトリクス値抽出エラー [{metric_name}]: {e}")
            return None
    
    async def _perform_statistical_analysis(self, bt_data: Dict, live_data: Dict) -> Dict:
        """統計分析実行"""
        try:
            statistical_results = {}
            
            # 基本メトリクスのt検定
            bt_metrics = bt_data.get('basic_metrics', {})
            live_metrics = live_data.get('basic_metrics', {})
            
            common_metrics = set(bt_metrics.keys()).intersection(set(live_metrics.keys()))
            
            for metric in common_metrics:
                try:
                    bt_value = bt_metrics[metric]
                    live_value = live_metrics[metric]
                    
                    if isinstance(bt_value, (int, float)) and isinstance(live_value, (int, float)):
                        # 単純な値の比較（実際の時系列データがないため）
                        difference = abs(live_value - bt_value)
                        relative_difference = difference / bt_value if bt_value != 0 else 0
                        
                        statistical_results[f"{metric}_analysis"] = {
                            "metric": metric,
                            "difference": difference,
                            "relative_difference": relative_difference,
                            "significance": "high" if relative_difference > 0.2 else "medium" if relative_difference > 0.1 else "low"
                        }
                        
                except Exception as e:
                    self.logger.warning(f"統計分析エラー [{metric}]: {e}")
            
            return statistical_results
            
        except Exception as e:
            self.logger.warning(f"統計分析実行エラー: {e}")
            return {}
    
    def _analyze_performance_gap(self, metrics_comparison: Dict) -> Dict:
        """パフォーマンスギャップ分析"""
        try:
            gap_analysis = {
                "overall_gap": "neutral",
                "critical_gaps": [],
                "positive_gaps": [],
                "gap_score": 0
            }
            
            gaps = []
            critical_metrics = ['total_pnl', 'win_rate', 'max_drawdown', 'sharpe_ratio']
            
            for metric, comparison in metrics_comparison.items():
                relative_diff = comparison.get('relative_difference', 0)
                gaps.append(relative_diff)
                
                # 重要メトリクスのギャップ評価
                if metric in critical_metrics:
                    if abs(relative_diff) > 0.2:  # 20%以上の差
                        if relative_diff < 0:
                            gap_analysis["critical_gaps"].append({
                                "metric": metric,
                                "gap": relative_diff,
                                "severity": "high" if abs(relative_diff) > 0.5 else "medium"
                            })
                        else:
                            gap_analysis["positive_gaps"].append({
                                "metric": metric,
                                "gap": relative_diff,
                                "benefit": "high" if relative_diff > 0.5 else "medium"
                            })
            
            # 全体ギャップスコア
            if gaps:
                avg_gap = np.mean(gaps)
                gap_analysis["gap_score"] = avg_gap
                
                if avg_gap > 0.1:
                    gap_analysis["overall_gap"] = "positive"
                elif avg_gap < -0.1:
                    gap_analysis["overall_gap"] = "negative"
                else:
                    gap_analysis["overall_gap"] = "neutral"
            
            return gap_analysis
            
        except Exception as e:
            self.logger.warning(f"パフォーマンスギャップ分析エラー: {e}")
            return {}
    
    def _calculate_portfolio_aggregates(self, data: Dict) -> Dict:
        """ポートフォリオ集計メトリクス計算"""
        try:
            strategies = data.get('strategies', {})
            if not strategies:
                return {}
            
            aggregates = {}
            
            # 総取引数
            total_trades = sum(
                strategy.get('basic_metrics', {}).get('total_trades', 0)
                for strategy in strategies.values()
            )
            aggregates['total_trades'] = total_trades
            
            # 総PnL
            total_pnl = sum(
                strategy.get('basic_metrics', {}).get('total_pnl', 0)
                for strategy in strategies.values()
            )
            aggregates['total_pnl'] = total_pnl
            
            # 平均勝率
            win_rates = [
                strategy.get('basic_metrics', {}).get('win_rate', 0)
                for strategy in strategies.values()
                if strategy.get('basic_metrics', {}).get('win_rate', 0) > 0
            ]
            if win_rates:
                aggregates['avg_win_rate'] = np.mean(win_rates)
            
            # 平均シャープレシオ
            sharpe_ratios = [
                strategy.get('risk_metrics', {}).get('sharpe_ratio', 0)
                for strategy in strategies.values()
                if strategy.get('risk_metrics', {}).get('sharpe_ratio', 0) != 0
            ]
            if sharpe_ratios:
                aggregates['avg_sharpe_ratio'] = np.mean(sharpe_ratios)
            
            # 最大ドローダウン（最悪値）
            drawdowns = [
                strategy.get('risk_metrics', {}).get('max_drawdown', 0)
                for strategy in strategies.values()
                if strategy.get('risk_metrics', {}).get('max_drawdown', 0) < 0
            ]
            if drawdowns:
                aggregates['worst_drawdown'] = min(drawdowns)
            
            return aggregates
            
        except Exception as e:
            self.logger.warning(f"ポートフォリオ集計エラー: {e}")
            return {}
    
    def _analyze_diversification(self, backtest_data: Dict, live_data: Dict) -> Dict:
        """分散化分析"""
        try:
            diversification_analysis = {}
            
            bt_strategies = len(backtest_data.get('strategies', {}))
            live_strategies = len(live_data.get('strategies', {}))
            
            diversification_analysis['strategy_count_comparison'] = {
                "backtest": bt_strategies,
                "live": live_strategies,
                "difference": live_strategies - bt_strategies
            }
            
            # 戦略パフォーマンス分散
            for data_type, data in [("backtest", backtest_data), ("live", live_data)]:
                pnl_values = [
                    strategy.get('basic_metrics', {}).get('total_pnl', 0)
                    for strategy in data.get('strategies', {}).values()
                ]
                
                if pnl_values and len(pnl_values) > 1:
                    diversification_analysis[f'{data_type}_pnl_variance'] = np.var(pnl_values)
                    diversification_analysis[f'{data_type}_pnl_std'] = np.std(pnl_values)
            
            return diversification_analysis
            
        except Exception as e:
            self.logger.warning(f"分散化分析エラー: {e}")
            return {}
    
    def _analyze_portfolio_risk(self, backtest_data: Dict, live_data: Dict) -> Dict:
        """ポートフォリオリスク分析"""
        try:
            risk_analysis = {}
            
            for data_type, data in [("backtest", backtest_data), ("live", live_data)]:
                strategies = data.get('strategies', {})
                
                # 最大ドローダウン分析
                drawdowns = [
                    strategy.get('risk_metrics', {}).get('max_drawdown', 0)
                    for strategy in strategies.values()
                    if strategy.get('risk_metrics', {}).get('max_drawdown', 0) < 0
                ]
                
                if drawdowns:
                    risk_analysis[f'{data_type}_worst_drawdown'] = min(drawdowns)
                    risk_analysis[f'{data_type}_avg_drawdown'] = np.mean(drawdowns)
                
                # ボラティリティ分析
                volatilities = [
                    strategy.get('risk_metrics', {}).get('volatility', 0)
                    for strategy in strategies.values()
                    if strategy.get('risk_metrics', {}).get('volatility', 0) > 0
                ]
                
                if volatilities:
                    risk_analysis[f'{data_type}_avg_volatility'] = np.mean(volatilities)
                    risk_analysis[f'{data_type}_max_volatility'] = max(volatilities)
            
            return risk_analysis
            
        except Exception as e:
            self.logger.warning(f"ポートフォリオリスク分析エラー: {e}")
            return {}
    
    def _create_comparison_summary(self, strategy_comparisons: Dict, portfolio_comparison: Dict) -> Dict:
        """比較分析サマリー作成"""
        try:
            summary = {
                "total_strategies_compared": len(strategy_comparisons),
                "overall_performance_gap": "neutral",
                "key_findings": [],
                "recommendations": []
            }
            
            # 戦略レベルサマリー
            positive_gaps = 0
            negative_gaps = 0
            
            for strategy_name, comparison in strategy_comparisons.items():
                gap_analysis = comparison.get('performance_gap_analysis', {})
                overall_gap = gap_analysis.get('overall_gap', 'neutral')
                
                if overall_gap == 'positive':
                    positive_gaps += 1
                elif overall_gap == 'negative':
                    negative_gaps += 1
                
                # 重要な発見の抽出
                critical_gaps = gap_analysis.get('critical_gaps', [])
                if critical_gaps:
                    for gap in critical_gaps[:2]:  # 最大2つまで
                        summary["key_findings"].append(
                            f"{strategy_name}: {gap['metric']} で {gap['gap']:.2%} の性能低下"
                        )
            
            # 全体パフォーマンスギャップ判定
            if positive_gaps > negative_gaps:
                summary["overall_performance_gap"] = "positive"
                summary["recommendations"].append("実運用は概ねバックテストを上回る性能を示しています")
            elif negative_gaps > positive_gaps:
                summary["overall_performance_gap"] = "negative"
                summary["recommendations"].append("実運用とバックテストの乖離要因を分析し、改善策を検討してください")
            else:
                summary["overall_performance_gap"] = "neutral"
                summary["recommendations"].append("実運用とバックテストの性能はほぼ一致しています")
            
            # ポートフォリオレベルの発見
            portfolio_metrics = portfolio_comparison.get('aggregate_metrics', {})
            if portfolio_metrics:
                total_pnl_comparison = portfolio_metrics.get('total_pnl', {})
                if total_pnl_comparison:
                    pnl_gap = total_pnl_comparison.get('relative_difference', 0)
                    if abs(pnl_gap) > 0.2:
                        summary["key_findings"].append(
                            f"ポートフォリオ全体で {pnl_gap:.2%} の収益性ギャップ"
                        )
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"比較分析サマリー作成エラー: {e}")
            return {}
    
    def _create_basic_summary(self, comparison_results: Dict) -> Dict:
        """基本比較サマリー作成"""
        try:
            summary = {
                "strategies_compared": len(comparison_results),
                "performance_overview": {},
                "notable_differences": []
            }
            
            # 戦略ごとの主要ギャップ
            for strategy_name, metrics in comparison_results.items():
                for metric_name, comparison in metrics.items():
                    relative_diff = comparison.get('relative_difference', 0)
                    
                    if abs(relative_diff) > 0.3:  # 30%以上の差
                        summary["notable_differences"].append({
                            "strategy": strategy_name,
                            "metric": metric_name,
                            "gap": relative_diff,
                            "performance_gap": comparison.get('performance_gap', 'unknown')
                        })
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"基本比較サマリー作成エラー: {e}")
            return {}
