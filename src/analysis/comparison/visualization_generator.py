"""
可視化ジェネレーター
フェーズ4A3: バックテストvs実運用比較分析器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

class VisualizationGenerator:
    """可視化ジェネレーター"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.visualization_settings = config.get('visualization_settings', {})
        self.output_dir = config.get('output_settings', {}).get('charts_dir', 'reports/charts')
        
        # 出力ディレクトリ作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # スタイル設定
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            if SEABORN_AVAILABLE:
                sns.set_palette("husl")
    
    def generate_comprehensive_visualizations(self, comparison_results: Dict[str, Any], 
                                           statistical_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """包括的可視化生成"""
        try:
            self.logger.info("包括的可視化生成開始")
            
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning("Matplotlibが利用できません。テキストベース出力のみ")
                return self._generate_text_based_visualizations(comparison_results)
            
            visualization_results = {
                "timestamp": datetime.now(),
                "charts_generated": [],
                "output_directory": self.output_dir,
                "format": "png"
            }
            
            strategy_comparisons = comparison_results.get('strategy_comparisons', {})
            portfolio_comparison = comparison_results.get('portfolio_comparison', {})
            
            # 1. 戦略別パフォーマンス比較チャート
            performance_chart = self._create_performance_comparison_chart(strategy_comparisons)
            if performance_chart:
                visualization_results["charts_generated"].append(performance_chart)
            
            # 2. メトリクス分布チャート
            distribution_chart = self._create_metrics_distribution_chart(strategy_comparisons)
            if distribution_chart:
                visualization_results["charts_generated"].append(distribution_chart)
            
            # 3. パフォーマンスギャップ分析チャート
            gap_chart = self._create_performance_gap_chart(strategy_comparisons)
            if gap_chart:
                visualization_results["charts_generated"].append(gap_chart)
            
            # 4. ポートフォリオ比較チャート
            portfolio_chart = self._create_portfolio_comparison_chart(portfolio_comparison)
            if portfolio_chart:
                visualization_results["charts_generated"].append(portfolio_chart)
            
            # 5. 統計分析結果チャート
            if statistical_results:
                stats_chart = self._create_statistical_analysis_chart(statistical_results)
                if stats_chart:
                    visualization_results["charts_generated"].append(stats_chart)
            
            # 6. ダッシュボード統合チャート
            dashboard_chart = self._create_dashboard_chart(comparison_results, statistical_results)
            if dashboard_chart:
                visualization_results["charts_generated"].append(dashboard_chart)
            
            self.logger.info(f"可視化完了 - チャート数: {len(visualization_results['charts_generated'])}")
            return visualization_results
            
        except Exception as e:
            self.logger.error(f"包括的可視化生成エラー: {e}")
            return self._generate_text_based_visualizations(comparison_results)
    
    def _create_performance_comparison_chart(self, strategy_comparisons: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """戦略別パフォーマンス比較チャート作成"""
        try:
            if not strategy_comparisons:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('戦略別パフォーマンス比較', fontsize=16, fontweight='bold')
            
            strategies = list(strategy_comparisons.keys())
            metrics = ['total_pnl', 'win_rate', 'max_drawdown', 'sharpe_ratio']
            
            # データ抽出
            backtest_data = {metric: [] for metric in metrics}
            live_data = {metric: [] for metric in metrics}
            strategy_labels = []
            
            for strategy_name in strategies:
                metrics_comparison = strategy_comparisons[strategy_name].get('metrics_comparison', {})
                
                strategy_labels.append(strategy_name[:8])  # 短縮表示
                
                for metric in metrics:
                    if metric in metrics_comparison:
                        bt_value = metrics_comparison[metric].get('backtest', 0)
                        live_value = metrics_comparison[metric].get('live', 0)
                    else:
                        bt_value = live_value = 0
                    
                    backtest_data[metric].append(bt_value)
                    live_data[metric].append(live_value)
            
            # サブプロット作成
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            
            for i, metric in enumerate(metrics):
                ax = axes[positions[i][0], positions[i][1]]
                
                x = np.arange(len(strategy_labels))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, backtest_data[metric], width, 
                              label='バックテスト', alpha=0.8, color='skyblue')
                bars2 = ax.bar(x + width/2, live_data[metric], width, 
                              label='実運用', alpha=0.8, color='lightcoral')
                
                ax.set_xlabel('戦略')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} 比較')
                ax.set_xticks(x)
                ax.set_xticklabels(strategy_labels, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 値をバーの上に表示
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_comparison_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_type": "performance_comparison",
                "filename": filename,
                "filepath": filepath,
                "strategies_count": len(strategies),
                "metrics_compared": metrics
            }
            
        except Exception as e:
            self.logger.warning(f"パフォーマンス比較チャート作成エラー: {e}")
            return None
    
    def _create_metrics_distribution_chart(self, strategy_comparisons: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """メトリクス分布チャート作成"""
        try:
            if not strategy_comparisons:
                return None
            
            # 相対差分データ収集
            relative_diffs = {
                'total_pnl': [],
                'win_rate': [],
                'max_drawdown': [],
                'sharpe_ratio': []
            }
            
            for strategy_comparison in strategy_comparisons.values():
                metrics_comparison = strategy_comparison.get('metrics_comparison', {})
                
                for metric in relative_diffs.keys():
                    if metric in metrics_comparison:
                        rel_diff = metrics_comparison[metric].get('relative_difference', 0)
                        relative_diffs[metric].append(rel_diff * 100)  # パーセント変換
            
            # チャート作成
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('メトリクス相対差分分布', fontsize=16, fontweight='bold')
            
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            
            for i, (metric, values) in enumerate(relative_diffs.items()):
                if not values:
                    continue
                
                ax = axes[positions[i][0], positions[i][1]]
                
                # ヒストグラム
                ax.hist(values, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
                
                # 統計情報追加
                mean_val = np.mean(values)
                median_val = np.median(values)
                
                ax.axvline(mean_val, color='red', linestyle='--', 
                          label=f'平均: {mean_val:.1f}%')
                ax.axvline(median_val, color='green', linestyle='--', 
                          label=f'中央値: {median_val:.1f}%')
                ax.axvline(0, color='black', linestyle='-', alpha=0.5, 
                          label='基準線 (0%)')
                
                ax.set_xlabel('相対差分 (%)')
                ax.set_ylabel('頻度')
                ax.set_title(f'{metric} 相対差分分布')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_distribution_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_type": "metrics_distribution",
                "filename": filename,
                "filepath": filepath,
                "metrics_analyzed": list(relative_diffs.keys())
            }
            
        except Exception as e:
            self.logger.warning(f"メトリクス分布チャート作成エラー: {e}")
            return None
    
    def _create_performance_gap_chart(self, strategy_comparisons: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """パフォーマンスギャップ分析チャート作成"""
        try:
            if not strategy_comparisons:
                return None
            
            # ギャップデータ収集
            strategies = []
            gap_scores = []
            gap_types = []
            
            for strategy_name, comparison in strategy_comparisons.items():
                gap_analysis = comparison.get('performance_gap_analysis', {})
                gap_score = gap_analysis.get('gap_score', 0)
                overall_gap = gap_analysis.get('overall_gap', 'neutral')
                
                strategies.append(strategy_name[:10])  # 短縮表示
                gap_scores.append(gap_score * 100)  # パーセント変換
                gap_types.append(overall_gap)
            
            if not strategies:
                return None
            
            # チャート作成
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('パフォーマンスギャップ分析', fontsize=16, fontweight='bold')
            
            # 1. ギャップスコア棒グラフ
            colors = ['green' if gap > 0 else 'red' if gap < 0 else 'gray' for gap in gap_scores]
            bars = ax1.bar(strategies, gap_scores, color=colors, alpha=0.7)
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.set_xlabel('戦略')
            ax1.set_ylabel('ギャップスコア (%)')
            ax1.set_title('戦略別パフォーマンスギャップ')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar, score in zip(bars, gap_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.1f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=9)
            
            # 2. ギャップタイプ円グラフ
            gap_type_counts = {}
            for gap_type in gap_types:
                gap_type_counts[gap_type] = gap_type_counts.get(gap_type, 0) + 1
            
            labels = list(gap_type_counts.keys())
            sizes = list(gap_type_counts.values())
            colors_pie = ['lightgreen', 'lightcoral', 'lightgray'][:len(labels)]
            
            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax2.set_title('ギャップタイプ分布')
            
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_gap_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_type": "performance_gap",
                "filename": filename,
                "filepath": filepath,
                "strategies_analyzed": len(strategies)
            }
            
        except Exception as e:
            self.logger.warning(f"パフォーマンスギャップチャート作成エラー: {e}")
            return None
    
    def _create_portfolio_comparison_chart(self, portfolio_comparison: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ポートフォリオ比較チャート作成"""
        try:
            if not portfolio_comparison:
                return None
            
            aggregate_metrics = portfolio_comparison.get('aggregate_metrics', {})
            if not aggregate_metrics:
                return None
            
            # チャート作成
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('ポートフォリオレベル比較', fontsize=16, fontweight='bold')
            
            # 1. 集計メトリクス比較
            metrics = list(aggregate_metrics.keys())
            backtest_values = [aggregate_metrics[m].get('backtest', 0) for m in metrics]
            live_values = [aggregate_metrics[m].get('live', 0) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, backtest_values, width, 
                           label='バックテスト', alpha=0.8, color='skyblue')
            bars2 = ax1.bar(x + width/2, live_values, width, 
                           label='実運用', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('メトリクス')
            ax1.set_ylabel('値')
            ax1.set_title('ポートフォリオ集計メトリクス')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 相対差分
            relative_diffs = []
            metric_labels = []
            
            for metric in metrics:
                rel_diff = aggregate_metrics[metric].get('relative_difference', 0)
                relative_diffs.append(rel_diff * 100)
                metric_labels.append(metric)
            
            colors = ['green' if diff > 0 else 'red' if diff < 0 else 'gray' 
                     for diff in relative_diffs]
            
            bars = ax2.bar(metric_labels, relative_diffs, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_xlabel('メトリクス')
            ax2.set_ylabel('相対差分 (%)')
            ax2.set_title('実運用 vs バックテスト 相対差分')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar, diff in zip(bars, relative_diffs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{diff:.1f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=9)
            
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_comparison_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_type": "portfolio_comparison",
                "filename": filename,
                "filepath": filepath,
                "metrics_compared": metrics
            }
            
        except Exception as e:
            self.logger.warning(f"ポートフォリオ比較チャート作成エラー: {e}")
            return None
    
    def _create_statistical_analysis_chart(self, statistical_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """統計分析結果チャート作成"""
        try:
            if not statistical_results:
                return None
            
            # p値データ収集
            p_values = []
            test_labels = []
            significance_levels = []
            
            statistical_tests = statistical_results.get('statistical_tests', {})
            
            for strategy_name, strategy_stats in statistical_tests.items():
                significance_tests = strategy_stats.get('significance_tests', {})
                
                for test_name, test_result in significance_tests.items():
                    if 'p_value' in test_result:
                        p_values.append(test_result['p_value'])
                        test_labels.append(f"{strategy_name[:8]}_{test_name}")
                        significance_levels.append(test_result.get('is_significant', False))
            
            if not p_values:
                return None
            
            # チャート作成
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('統計分析結果', fontsize=16, fontweight='bold')
            
            # 1. p値プロット
            colors = ['red' if sig else 'blue' for sig in significance_levels]
            bars = ax1.bar(range(len(p_values)), p_values, color=colors, alpha=0.7)
            
            ax1.axhline(y=0.05, color='red', linestyle='--', 
                       label='有意水準 (p=0.05)')
            ax1.set_xlabel('検定')
            ax1.set_ylabel('p値')
            ax1.set_title('統計的有意性検定結果')
            ax1.set_xticks(range(len(test_labels)))
            ax1.set_xticklabels(test_labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # 2. 有意性分布
            sig_counts = {'有意': sum(significance_levels), 
                         '非有意': len(significance_levels) - sum(significance_levels)}
            
            if sum(sig_counts.values()) > 0:
                ax2.pie(sig_counts.values(), labels=sig_counts.keys(), 
                       colors=['lightcoral', 'lightblue'], autopct='%1.1f%%', startangle=90)
                ax2.set_title('有意性テスト結果分布')
            
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistical_analysis_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_type": "statistical_analysis",
                "filename": filename,
                "filepath": filepath,
                "tests_analyzed": len(p_values)
            }
            
        except Exception as e:
            self.logger.warning(f"統計分析チャート作成エラー: {e}")
            return None
    
    def _create_dashboard_chart(self, comparison_results: Dict[str, Any], 
                              statistical_results: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """ダッシュボード統合チャート作成"""
        try:
            # サマリー情報抽出
            summary = comparison_results.get('summary', {})
            strategy_count = summary.get('total_strategies_compared', 0)
            overall_gap = summary.get('overall_performance_gap', 'neutral')
            key_findings = summary.get('key_findings', [])
            
            # チャート作成
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # タイトル
            fig.suptitle('バックテスト vs 実運用 比較分析ダッシュボード', 
                        fontsize=18, fontweight='bold', y=0.95)
            
            # 1. サマリー情報
            ax1 = fig.add_subplot(gs[0, :])
            ax1.axis('off')
            
            summary_text = f"""
分析サマリー:
• 比較戦略数: {strategy_count}
• 全体パフォーマンスギャップ: {overall_gap}
• 分析実行時刻: {comparison_results.get('timestamp', 'N/A')}
            """
            
            if key_findings:
                summary_text += "\n主要発見事項:\n"
                for finding in key_findings[:3]:  # 最大3つまで
                    summary_text += f"• {finding}\n"
            
            ax1.text(0.05, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            
            # 2-6. 既存チャートの小型版を統合（実装は簡略化）
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "chart_type": "dashboard",
                "filename": filename,
                "filepath": filepath,
                "comprehensive": True
            }
            
        except Exception as e:
            self.logger.warning(f"ダッシュボードチャート作成エラー: {e}")
            return None
    
    def _generate_text_based_visualizations(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """テキストベース可視化生成（matplotlib未使用時）"""
        try:
            self.logger.info("テキストベース可視化生成")
            
            text_output = []
            text_output.append("=" * 60)
            text_output.append("バックテスト vs 実運用 比較分析結果")
            text_output.append("=" * 60)
            text_output.append(f"分析実行時刻: {comparison_results.get('timestamp', 'N/A')}")
            text_output.append("")
            
            # サマリー
            summary = comparison_results.get('summary', {})
            text_output.append("【分析サマリー】")
            text_output.append(f"比較戦略数: {summary.get('total_strategies_compared', 0)}")
            text_output.append(f"全体パフォーマンスギャップ: {summary.get('overall_performance_gap', 'unknown')}")
            text_output.append("")
            
            # 戦略別結果
            strategy_comparisons = comparison_results.get('strategy_comparisons', {})
            if strategy_comparisons:
                text_output.append("【戦略別比較結果】")
                
                for strategy_name, comparison in strategy_comparisons.items():
                    text_output.append(f"\n[{strategy_name}]")
                    
                    metrics_comparison = comparison.get('metrics_comparison', {})
                    for metric, values in metrics_comparison.items():
                        bt_val = values.get('backtest', 0)
                        live_val = values.get('live', 0)
                        rel_diff = values.get('relative_difference', 0)
                        
                        text_output.append(f"  {metric}:")
                        text_output.append(f"    バックテスト: {bt_val:.4f}")
                        text_output.append(f"    実運用: {live_val:.4f}")
                        text_output.append(f"    相対差分: {rel_diff:.2%}")
            
            # ファイル保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_analysis_text_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_output))
            
            return {
                "visualization_type": "text_based",
                "filename": filename,
                "filepath": filepath,
                "charts_generated": [{"chart_type": "text_summary", "filename": filename}]
            }
            
        except Exception as e:
            self.logger.error(f"テキストベース可視化エラー: {e}")
            return {}
    
    def generate_html_report(self, comparison_results: Dict[str, Any], 
                           chart_files: List[Dict[str, Any]] = None) -> Optional[str]:
        """HTML可視化レポート生成"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>バックテスト vs 実運用 比較分析レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f4fd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        .strategy {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>バックテスト vs 実運用 比較分析レポート</h1>
        <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
    </div>
            """
            
            # サマリー追加
            summary = comparison_results.get('summary', {})
            html_content += f"""
    <div class="summary">
        <h2>分析サマリー</h2>
        <p><strong>比較戦略数:</strong> {summary.get('total_strategies_compared', 0)}</p>
        <p><strong>全体パフォーマンスギャップ:</strong> {summary.get('overall_performance_gap', 'unknown')}</p>
        
        <h3>主要発見事項</h3>
        <ul>
            """
            
            for finding in summary.get('key_findings', []):
                html_content += f"<li>{finding}</li>"
            
            html_content += """
        </ul>
    </div>
            """
            
            # チャート追加
            if chart_files:
                html_content += "<h2>可視化チャート</h2>"
                for chart in chart_files:
                    chart_type = chart.get('chart_type', 'unknown')
                    filename = chart.get('filename', '')
                    html_content += f"""
    <div class="chart">
        <h3>{chart_type.replace('_', ' ').title()}</h3>
        <img src="{filename}" alt="{chart_type}" style="max-width: 100%; height: auto;">
    </div>
                    """
            
            html_content += """
</body>
</html>
            """
            
            # ファイル保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_report_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTMLレポート生成完了: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"HTMLレポート生成エラー: {e}")
            return None
