"""
分類分析器：市場分類結果の分析と可視化
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from .market_conditions import (
    SimpleMarketCondition, DetailedMarketCondition, 
    ClassificationResult, MarketConditions
)

logger = logging.getLogger(__name__)


class ClassificationAnalyzer:
    """分類結果の分析クラス"""
    
    def __init__(self):
        self.results: List[ClassificationResult] = []
        
    def add_result(self, result: ClassificationResult):
        """分類結果を追加"""
        self.results.append(result)
        
    def add_results(self, results: List[ClassificationResult]):
        """複数の分類結果を追加"""
        self.results.extend(results)
        
    def clear_results(self):
        """結果をクリア"""
        self.results.clear()
        
    def get_distribution_summary(self) -> Dict[str, Any]:
        """分類分布のサマリーを取得"""
        if not self.results:
            return {}
        
        # シンプル分類の分布
        simple_dist = {}
        for condition in SimpleMarketCondition:
            count = sum(1 for r in self.results if r.simple_condition == condition)
            simple_dist[condition.value] = count
            
        # 詳細分類の分布
        detailed_dist = {}
        for condition in DetailedMarketCondition:
            count = sum(1 for r in self.results if r.detailed_condition == condition)
            detailed_dist[condition.value] = count
            
        # 信頼度統計
        confidences = [r.confidence for r in self.results]
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
        
        # シンボル別統計
        symbols = list(set(r.symbol for r in self.results))
        symbol_stats = {}
        for symbol in symbols:
            symbol_results = [r for r in self.results if r.symbol == symbol]
            symbol_stats[symbol] = {
                'count': len(symbol_results),
                'avg_confidence': np.mean([r.confidence for r in symbol_results])
            }
        
        return {
            'total_classifications': len(self.results),
            'simple_distribution': simple_dist,
            'detailed_distribution': detailed_dist,
            'confidence_statistics': confidence_stats,
            'symbol_statistics': symbol_stats,
            'unique_symbols': len(symbols)
        }
    
    def get_time_series_analysis(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """時系列分析用のデータフレームを作成"""
        filtered_results = self.results
        if symbol:
            filtered_results = [r for r in self.results if r.symbol == symbol]
        
        if not filtered_results:
            return pd.DataFrame()
        
        data = []
        for result in filtered_results:
            data.append({
                'timestamp': pd.to_datetime(result.timestamp),
                'symbol': result.symbol,
                'simple_condition': result.simple_condition.value,
                'detailed_condition': result.detailed_condition.value,
                'confidence': result.confidence,
                'trend_strength': result.metrics.trend_strength,
                'volatility': result.metrics.volatility,
                'momentum': result.metrics.momentum,
                'volume_trend': result.metrics.volume_trend,
                'price_momentum': result.metrics.price_momentum,
                'risk_level': result.metrics.risk_level,
                'rsi': result.metrics.rsi,
                'ma_slope': result.metrics.ma_slope,
                'atr_ratio': result.metrics.atr_ratio,
                'volume_ratio': result.metrics.volume_ratio
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        return df
    
    def analyze_classification_stability(self, symbol: Optional[str] = None, 
                                       window_size: int = 5) -> Dict[str, float]:
        """分類の安定性を分析"""
        df = self.get_time_series_analysis(symbol)
        if df.empty or len(df) < window_size:
            return {'stability_score': 0.0, 'transition_rate': 0.0}
        
        # 分類変化の検出
        simple_changes = (df['simple_condition'] != df['simple_condition'].shift(1)).sum()
        detailed_changes = (df['detailed_condition'] != df['detailed_condition'].shift(1)).sum()
        
        # 安定性スコア（変化が少ないほど高い）
        total_periods = len(df) - 1
        simple_stability = 1.0 - (simple_changes / total_periods) if total_periods > 0 else 1.0
        detailed_stability = 1.0 - (detailed_changes / total_periods) if total_periods > 0 else 1.0
        
        # 移動窓での安定性
        rolling_stability = []
        for i in range(window_size, len(df)):
            window = df.iloc[i-window_size+1:i+1]
            simple_unique = window['simple_condition'].nunique()
            rolling_stability.append(1.0 / simple_unique)
        
        avg_rolling_stability = np.mean(rolling_stability) if rolling_stability else 1.0
        
        return {
            'simple_stability': simple_stability,
            'detailed_stability': detailed_stability,
            'avg_rolling_stability': avg_rolling_stability,
            'transition_rate': (simple_changes + detailed_changes) / (2 * total_periods) if total_periods > 0 else 0.0
        }
    
    def find_classification_patterns(self) -> Dict[str, Any]:
        """分類パターンの発見"""
        df = self.get_time_series_analysis()
        if df.empty:
            return {}
        
        patterns = {}
        
        # 1. 条件間の遷移パターン
        transitions = {}
        for i in range(1, len(df)):
            prev_simple = df.iloc[i-1]['simple_condition']
            curr_simple = df.iloc[i]['simple_condition']
            transition = f"{prev_simple} -> {curr_simple}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        # 2. メトリクス範囲による分類パターン
        metric_patterns = {}
        for condition in SimpleMarketCondition:
            condition_data = df[df['simple_condition'] == condition.value]
            if not condition_data.empty:
                metric_patterns[condition.value] = {
                    'trend_strength_range': [
                        condition_data['trend_strength'].min(),
                        condition_data['trend_strength'].max()
                    ],
                    'volatility_range': [
                        condition_data['volatility'].min(),
                        condition_data['volatility'].max()
                    ],
                    'momentum_range': [
                        condition_data['momentum'].min(),
                        condition_data['momentum'].max()
                    ]
                }
        
        # 3. 信頼度パターン
        confidence_by_condition = {}
        for condition in SimpleMarketCondition:
            condition_data = df[df['simple_condition'] == condition.value]
            if not condition_data.empty:
                confidence_by_condition[condition.value] = {
                    'mean_confidence': condition_data['confidence'].mean(),
                    'std_confidence': condition_data['confidence'].std(),
                    'count': len(condition_data)
                }
        
        return {
            'transition_patterns': transitions,
            'metric_patterns': metric_patterns,
            'confidence_patterns': confidence_by_condition
        }
    
    def generate_classification_report(self, output_path: Optional[str] = None) -> str:
        """分類レポートを生成"""
        summary = self.get_distribution_summary()
        stability = self.analyze_classification_stability()
        patterns = self.find_classification_patterns()
        
        report_lines = [
            "# Market Classification Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            f"Total Classifications: {summary.get('total_classifications', 0)}",
            f"Unique Symbols: {summary.get('unique_symbols', 0)}",
            "",
            "## Simple Classification Distribution",
        ]
        
        simple_dist = summary.get('simple_distribution', {})
        for condition, count in simple_dist.items():
            percentage = (count / summary['total_classifications'] * 100) if summary['total_classifications'] > 0 else 0
            report_lines.append(f"- {condition}: {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "## Detailed Classification Distribution",
        ])
        
        detailed_dist = summary.get('detailed_distribution', {})
        for condition, count in detailed_dist.items():
            percentage = (count / summary['total_classifications'] * 100) if summary['total_classifications'] > 0 else 0
            report_lines.append(f"- {condition}: {count} ({percentage:.1f}%)")
        
        confidence_stats = summary.get('confidence_statistics', {})
        report_lines.extend([
            "",
            "## Confidence Statistics",
            f"- Mean: {confidence_stats.get('mean', 0):.3f}",
            f"- Standard Deviation: {confidence_stats.get('std', 0):.3f}",
            f"- Min: {confidence_stats.get('min', 0):.3f}",
            f"- Max: {confidence_stats.get('max', 0):.3f}",
            f"- Median: {confidence_stats.get('median', 0):.3f}",
            "",
            "## Stability Analysis",
            f"- Simple Classification Stability: {stability.get('simple_stability', 0):.3f}",
            f"- Detailed Classification Stability: {stability.get('detailed_stability', 0):.3f}",
            f"- Average Rolling Stability: {stability.get('avg_rolling_stability', 0):.3f}",
            f"- Transition Rate: {stability.get('transition_rate', 0):.3f}",
        ])
        
        # パターン分析
        transitions = patterns.get('transition_patterns', {})
        if transitions:
            report_lines.extend([
                "",
                "## Top Transition Patterns",
            ])
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            for transition, count in sorted_transitions[:10]:  # 上位10個
                report_lines.append(f"- {transition}: {count}")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report to {output_path}: {e}")
        
        return report_content
    
    def plot_classification_distribution(self, save_path: Optional[str] = None):
        """分類分布の可視化"""
        try:
            summary = self.get_distribution_summary()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. シンプル分類分布
            simple_dist = summary.get('simple_distribution', {})
            if simple_dist:
                conditions = list(simple_dist.keys())
                counts = list(simple_dist.values())
                ax1.bar(conditions, counts)
                ax1.set_title('Simple Market Classification Distribution')
                ax1.set_ylabel('Count')
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # 2. 詳細分類分布
            detailed_dist = summary.get('detailed_distribution', {})
            if detailed_dist:
                conditions = list(detailed_dist.keys())
                counts = list(detailed_dist.values())
                ax2.bar(conditions, counts)
                ax2.set_title('Detailed Market Classification Distribution')
                ax2.set_ylabel('Count')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # 3. 信頼度分布
            confidences = [r.confidence for r in self.results]
            if confidences:
                ax3.hist(confidences, bins=20, alpha=0.7)
                ax3.set_title('Confidence Distribution')
                ax3.set_xlabel('Confidence')
                ax3.set_ylabel('Frequency')
            
            # 4. シンボル別分類数
            symbol_stats = summary.get('symbol_statistics', {})
            if symbol_stats:
                symbols = list(symbol_stats.keys())
                counts = [stats['count'] for stats in symbol_stats.values()]
                ax4.bar(symbols, counts)
                ax4.set_title('Classifications per Symbol')
                ax4.set_ylabel('Count')
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Classification distribution plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot classification distribution: {e}")
    
    def plot_time_series(self, symbol: str, save_path: Optional[str] = None):
        """特定シンボルの時系列プロット"""
        try:
            df = self.get_time_series_analysis(symbol)
            if df.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 分類の時系列
            ax1.scatter(df['timestamp'], df['simple_condition'], alpha=0.7, label='Simple')
            ax1.set_title(f'Market Classification Over Time - {symbol}')
            ax1.set_ylabel('Classification')
            ax1.legend()
            plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # 2. 信頼度の時系列
            ax2.plot(df['timestamp'], df['confidence'], marker='o', alpha=0.7)
            ax2.set_title(f'Classification Confidence - {symbol}')
            ax2.set_ylabel('Confidence')
            plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # 3. メトリクスの時系列
            ax3.plot(df['timestamp'], df['trend_strength'], label='Trend Strength', alpha=0.7)
            ax3.plot(df['timestamp'], df['momentum'], label='Momentum', alpha=0.7)
            ax3.set_title(f'Market Metrics - {symbol}')
            ax3.set_ylabel('Value')
            ax3.legend()
            plt.setp(ax3.get_xticklabels(), rotation=45)
            
            # 4. ボラティリティとリスク
            ax4.plot(df['timestamp'], df['volatility'], label='Volatility', alpha=0.7)
            ax4.plot(df['timestamp'], df['risk_level'], label='Risk Level', alpha=0.7)
            ax4.set_title(f'Risk Metrics - {symbol}')
            ax4.set_ylabel('Value')
            ax4.legend()
            plt.setp(ax4.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Time series plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot time series for {symbol}: {e}")
    
    def export_to_csv(self, file_path: str):
        """結果をCSVにエクスポート"""
        try:
            df = self.get_time_series_analysis()
            if df.empty:
                logger.warning("No data to export")
                return
            
            df.to_csv(file_path, index=False)
            logger.info(f"Results exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
    
    def compare_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """複数シンボルの比較分析"""
        comparison = {}
        
        for symbol in symbols:
            symbol_results = [r for r in self.results if r.symbol == symbol]
            if not symbol_results:
                continue
            
            # 基本統計
            confidences = [r.confidence for r in symbol_results]
            trends = [r.metrics.trend_strength for r in symbol_results]
            volatilities = [r.metrics.volatility for r in symbol_results]
            
            # 分類分布
            simple_dist = {}
            for condition in SimpleMarketCondition:
                count = sum(1 for r in symbol_results if r.simple_condition == condition)
                simple_dist[condition.value] = count
            
            comparison[symbol] = {
                'total_classifications': len(symbol_results),
                'avg_confidence': np.mean(confidences),
                'avg_trend_strength': np.mean(trends),
                'avg_volatility': np.mean(volatilities),
                'classification_distribution': simple_dist,
                'stability': self.analyze_classification_stability(symbol)
            }
        
        return comparison
