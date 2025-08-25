"""
Module: DSSMS Enhanced Reporter
File: dssms_enhanced_reporter.py
Description: 
  Task 1.2: DSSMS既存形式レポート品質向上システム
  既存形式を維持しながら内容を大幅に充実化

Author: GitHub Copilot
Created: 2025-08-25
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from config.logger_config import setup_logger
except ImportError:
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class DSSMSEnhancedReporter:
    """DSSMS既存形式での内容充実化レポーター"""
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger(__name__)
        
        # レポート設定
        self.report_config = {
            'decimal_places': 2,
            'percentage_format': True,
            'include_debug_info': True,
            'max_detail_items': 10,
            'currency_symbol': '円'
        }
        
        # メトリクス計算キャッシュ
        self.metrics_cache: Dict[str, Any] = {}
        
        self.logger.info("DSSMS 強化レポーターを初期化しました")
    
    def generate_enhanced_detailed_report(self, simulation_result: Dict[str, Any]) -> str:
        """詳細分析強化版レポート生成"""
        try:
            self.logger.info("強化詳細レポート生成開始")
            
            # 基本情報取得
            basic_info = self._extract_basic_info(simulation_result)
            
            # 詳細メトリクス計算
            detailed_metrics = self._calculate_detailed_metrics(simulation_result)
            
            # データソース分析
            data_source_analysis = self._analyze_data_sources(simulation_result)
            
            # シミュレーション品質分析
            simulation_quality = self._analyze_simulation_quality(simulation_result)
            
            # DSSMS固有分析
            dssms_analysis = self._analyze_dssms_specifics(simulation_result)
            
            # 比較分析強化
            enhanced_comparison = self._generate_enhanced_comparison(simulation_result)
            
            # 推奨事項強化
            enhanced_recommendations = self._generate_enhanced_recommendations(simulation_result)
            
            # レポート組み立て
            report = self._assemble_enhanced_report(
                basic_info,
                detailed_metrics,
                data_source_analysis,
                simulation_quality,
                dssms_analysis,
                enhanced_comparison,
                enhanced_recommendations
            )
            
            self.logger.info("強化詳細レポート生成完了")
            return report
            
        except Exception as e:
            self.logger.error(f"強化レポート生成エラー: {e}")
            return self._generate_error_report(str(e))
    
    def _extract_basic_info(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """基本情報抽出"""
        return {
            'execution_time': simulation_result.get('execution_time', datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')),
            'backtest_period': simulation_result.get('backtest_period', 'N/A'),
            'initial_capital': simulation_result.get('initial_capital', 0),
            'final_portfolio_value': simulation_result.get('final_portfolio_value', 0),
            'data_source': simulation_result.get('data_source', 'enhanced_real_data'),
            'enhancement_applied': simulation_result.get('enhancement_applied', True)
        }
    
    def _calculate_detailed_metrics(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """詳細メトリクス計算"""
        try:
            performance_history = simulation_result.get('performance_history', {})
            daily_returns = performance_history.get('daily_returns', [])
            portfolio_values = performance_history.get('portfolio_value', [])
            
            if not daily_returns:
                return {'error': 'パフォーマンスデータが不足しています'}
            
            returns_array = np.array(daily_returns)
            
            # 基本指標
            total_return = (simulation_result.get('final_portfolio_value', 0) / 
                          simulation_result.get('initial_capital', 1)) - 1
            
            # リスク指標
            volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0
            
            # ドローダウン計算
            drawdown_info = self._calculate_detailed_drawdown(portfolio_values)
            
            # シャープレシオ
            sharpe_ratio = (np.mean(returns_array) * 252) / (volatility + 1e-8) if volatility > 0 else 0
            
            # ソルティノレシオ
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0
            sortino_ratio = (np.mean(returns_array) * 252) / (downside_deviation + 1e-8) if downside_deviation > 0 else 0
            
            # VaR計算
            var_95 = np.percentile(returns_array, 5) if len(returns_array) > 20 else 0
            var_99 = np.percentile(returns_array, 1) if len(returns_array) > 20 else 0
            
            # 勝率・平均損益
            win_rate = len(returns_array[returns_array > 0]) / len(returns_array) if returns_array.size > 0 else 0
            avg_win = np.mean(returns_array[returns_array > 0]) if len(returns_array[returns_array > 0]) > 0 else 0
            avg_loss = np.mean(returns_array[returns_array < 0]) if len(returns_array[returns_array < 0]) > 0 else 0
            
            # 新規追加指標
            calmar_ratio = total_return / abs(drawdown_info['max_drawdown']) if abs(drawdown_info['max_drawdown']) > 0.001 else 0
            sterling_ratio = total_return / abs(drawdown_info['average_drawdown']) if abs(drawdown_info['average_drawdown']) > 0.001 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252 / max(1, len(returns_array))) - 1,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'sterling_ratio': sterling_ratio,
                'max_drawdown': drawdown_info['max_drawdown'],
                'average_drawdown': drawdown_info['average_drawdown'],
                'drawdown_duration': drawdown_info['max_duration'],
                'var_95': var_95,
                'var_99': var_99,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss < 0 else 0,
                'return_distribution': self._analyze_return_distribution(returns_array)
            }
            
        except Exception as e:
            self.logger.error(f"詳細メトリクス計算エラー: {e}")
            return {'error': f'メトリクス計算エラー: {str(e)}'}
    
    def _calculate_detailed_drawdown(self, portfolio_values: List[float]) -> Dict[str, float]:
        """詳細ドローダウン計算"""
        if len(portfolio_values) < 2:
            return {'max_drawdown': 0.0, 'average_drawdown': 0.0, 'max_duration': 0}
        
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        
        max_drawdown = np.min(drawdown)
        average_drawdown = np.mean(drawdown[drawdown < 0]) if len(drawdown[drawdown < 0]) > 0 else 0
        
        # ドローダウン期間計算
        in_drawdown = drawdown < -0.01  # 1%以上のドローダウン
        duration = 0
        max_duration = 0
        
        for dd in in_drawdown:
            if dd:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'average_drawdown': average_drawdown,
            'max_duration': max_duration
        }
    
    def _analyze_return_distribution(self, returns: np.ndarray) -> Dict[str, float]:
        """リターン分布分析"""
        if len(returns) < 5:
            return {}
        
        return {
            'skewness': float(pd.Series(returns).skew()),
            'kurtosis': float(pd.Series(returns).kurtosis()),
            'percentile_10': float(np.percentile(returns, 10)),
            'percentile_90': float(np.percentile(returns, 90)),
            'positive_days_ratio': float(len(returns[returns > 0]) / len(returns))
        }
    
    def _analyze_data_sources(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """データソース分析"""
        data_quality = simulation_result.get('data_quality', {})
        
        return {
            'primary_source': simulation_result.get('data_source', 'unknown'),
            'quality_score': data_quality.get('overall_score', 0.0),
            'real_data_percentage': data_quality.get('real_data_ratio', 0.0),
            'fallback_usage': data_quality.get('fallback_count', 0),
            'data_completeness': data_quality.get('completeness', 0.0),
            'enhancement_level': simulation_result.get('enhancement_level', 'basic')
        }
    
    def _analyze_simulation_quality(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """シミュレーション品質分析"""
        quality_metrics = simulation_result.get('quality_metrics', {})
        
        return {
            'overall_quality': quality_metrics.get('overall_quality', 0.0),
            'anomalies_detected': quality_metrics.get('anomalies_detected', 0),
            'corrections_applied': quality_metrics.get('corrections_applied', 0),
            'realism_factors': quality_metrics.get('realism_factors_applied', []),
            'data_consistency': quality_metrics.get('data_consistency', 'unknown'),
            'simulation_reliability': self._calculate_simulation_reliability(simulation_result)
        }
    
    def _calculate_simulation_reliability(self, simulation_result: Dict[str, Any]) -> str:
        """シミュレーション信頼性評価"""
        quality_score = simulation_result.get('quality_metrics', {}).get('overall_quality', 0.0)
        
        if quality_score >= 0.9:
            return "非常に高い"
        elif quality_score >= 0.7:
            return "高い"
        elif quality_score >= 0.5:
            return "中程度"
        elif quality_score >= 0.3:
            return "低い"
        else:
            return "要改善"
    
    def _analyze_dssms_specifics(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """DSSMS固有分析"""
        switch_history = simulation_result.get('switch_history', [])
        
        # 切替分析
        total_switches = len(switch_history)
        successful_switches = len([s for s in switch_history if s.get('profit_loss_at_switch', 0) > 0])
        
        # 保有期間分析
        holding_periods = [s.get('holding_period_hours', 0) for s in switch_history]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # 切替コスト分析
        switch_costs = [s.get('switch_cost', 0) for s in switch_history]
        total_switch_cost = sum(switch_costs)
        
        return {
            'total_switches': total_switches,
            'switch_success_rate': successful_switches / max(1, total_switches),
            'avg_holding_period_hours': avg_holding_period,
            'total_switch_cost': total_switch_cost,
            'avg_switch_cost': total_switch_cost / max(1, total_switches),
            'switch_frequency_per_day': total_switches / max(1, len(simulation_result.get('performance_history', {}).get('daily_returns', []))),
            'dynamic_selection_efficiency': self._calculate_dynamic_efficiency(simulation_result)
        }
    
    def _calculate_dynamic_efficiency(self, simulation_result: Dict[str, Any]) -> float:
        """動的選択効率計算"""
        total_return = simulation_result.get('total_return', 0)
        switch_cost_ratio = simulation_result.get('switch_cost_ratio', 0)
        
        # 切替コストを考慮した効率性
        if switch_cost_ratio > 0:
            return total_return / (1 + switch_cost_ratio)
        else:
            return total_return
    
    def _generate_enhanced_comparison(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """比較分析強化"""
        comparison_results = simulation_result.get('comparison_results', {})
        
        enhanced_comparison = {}
        
        # 既存比較結果の詳細化
        for benchmark, result in comparison_results.items():
            if isinstance(result, (int, float)):
                performance_diff = result
                
                # パフォーマンス評価
                if performance_diff > 0.1:
                    evaluation = "大幅に優位"
                elif performance_diff > 0.05:
                    evaluation = "優位"
                elif performance_diff > 0:
                    evaluation = "僅かに優位"
                elif performance_diff > -0.05:
                    evaluation = "僅かに劣位"
                elif performance_diff > -0.1:
                    evaluation = "劣位"
                else:
                    evaluation = "大幅に劣位"
                
                enhanced_comparison[benchmark] = {
                    'performance_difference': performance_diff,
                    'evaluation': evaluation,
                    'relative_strength': abs(performance_diff),
                    'recommendation': self._get_benchmark_recommendation(performance_diff)
                }
        
        return enhanced_comparison
    
    def _get_benchmark_recommendation(self, performance_diff: float) -> str:
        """ベンチマーク比較推奨事項"""
        if performance_diff < -0.1:
            return "戦略の大幅な見直しが必要"
        elif performance_diff < -0.05:
            return "リスク管理の強化を検討"
        elif performance_diff < 0:
            return "パラメータの微調整を推奨"
        elif performance_diff < 0.05:
            return "現状維持で経過観察"
        else:
            return "優秀な成果、継続実行を推奨"
    
    def _generate_enhanced_recommendations(self, simulation_result: Dict[str, Any]) -> List[str]:
        """推奨事項強化生成"""
        recommendations = []
        
        # パフォーマンス関連
        total_return = simulation_result.get('total_return', 0)
        if total_return < 0:
            recommendations.append("📉 DSSMSのパフォーマンスは改善が必要です。戦略選択基準の見直しを推奨します。")
        elif total_return > 0.2:
            recommendations.append("📈 DSSMSは優秀なパフォーマンスを示しています。現在の設定を維持してください。")
        
        # リスク関連
        volatility = simulation_result.get('detailed_metrics', {}).get('volatility', 0)
        if volatility > 0.3:
            recommendations.append("⚠️ ボラティリティが高いため、リスク管理の強化を推奨します。")
        elif volatility < 0.1:
            recommendations.append("✅ 適度なボラティリティで安定的な運用ができています。")
        
        # 切替関連
        dssms_metrics = simulation_result.get('dssms_metrics', {})
        switch_frequency = dssms_metrics.get('switch_frequency_per_day', 0)
        if switch_frequency > 1:
            recommendations.append("🔄 切替回数が多いため、取引コストの最適化を検討してください。")
        
        # データ品質関連
        data_quality = simulation_result.get('data_quality', {}).get('overall_score', 1.0)
        if data_quality < 0.7:
            recommendations.append("📊 データ品質の向上により、より正確な分析が可能になります。")
        else:
            recommendations.append("✅ 高品質なデータによる信頼性の高い分析結果です。")
        
        # シミュレーション品質関連
        quality_metrics = simulation_result.get('quality_metrics', {})
        anomalies = quality_metrics.get('anomalies_detected', 0)
        if anomalies > 5:
            recommendations.append("🔍 シミュレーション異常が検出されました。設定の確認を推奨します。")
        
        return recommendations
    
    def _assemble_enhanced_report(self, basic_info: Dict[str, Any], 
                                detailed_metrics: Dict[str, Any],
                                data_source_analysis: Dict[str, Any],
                                simulation_quality: Dict[str, Any],
                                dssms_analysis: Dict[str, Any],
                                enhanced_comparison: Dict[str, Any],
                                enhanced_recommendations: List[str]) -> str:
        """強化レポート組み立て"""
        
        report_lines = [
            "=" * 80,
            "DSSMS (動的銘柄選択管理システム) バックテスト詳細レポート【強化版】",
            "=" * 80,
            "",
            f"実行日時: {basic_info['execution_time']}",
            f"バックテスト期間: {basic_info['backtest_period']}",
            f"初期資本: {basic_info['initial_capital']:,.0f}{self.report_config['currency_symbol']}",
            f"最終ポートフォリオ価値: {basic_info['final_portfolio_value']:,.2f}{self.report_config['currency_symbol']}",
            "",
            "【基本パフォーマンス指標】",
            "-" * 40
        ]
        
        # 基本メトリクス
        if 'error' not in detailed_metrics:
            report_lines.extend([
                f"総リターン: {detailed_metrics['total_return']:.2%}",
                f"年率リターン: {detailed_metrics['annualized_return']:.2%}",
                f"年率ボラティリティ: {detailed_metrics['volatility']:.2%}",
                f"最大ドローダウン: {detailed_metrics['max_drawdown']:.2%}",
                f"平均ドローダウン: {detailed_metrics['average_drawdown']:.2%}",
                f"シャープレシオ: {detailed_metrics['sharpe_ratio']:.3f}",
                f"ソルティノレシオ: {detailed_metrics['sortino_ratio']:.3f}",
                f"カルマーレシオ: {detailed_metrics['calmar_ratio']:.3f}",
                f"スターリングレシオ: {detailed_metrics['sterling_ratio']:.3f}",
            ])
        else:
            report_lines.append(f"メトリクス計算エラー: {detailed_metrics['error']}")
        
        # リスク分析
        report_lines.extend([
            "",
            "【リスク分析】",
            "-" * 40
        ])
        
        if 'error' not in detailed_metrics:
            report_lines.extend([
                f"VaR (95%信頼区間): {detailed_metrics['var_95']:.2%}",
                f"VaR (99%信頼区間): {detailed_metrics['var_99']:.2%}",
                f"最大ドローダウン期間: {detailed_metrics['drawdown_duration']}日",
                f"勝率: {detailed_metrics['win_rate']:.2%}",
                f"平均利益: {detailed_metrics['avg_win']:.4f}",
                f"平均損失: {detailed_metrics['avg_loss']:.4f}",
                f"プロフィットファクター: {detailed_metrics['profit_factor']:.3f}",
            ])
        
        # DSSMS固有指標
        report_lines.extend([
            "",
            "【DSSMS固有指標】",
            "-" * 40,
            f"銘柄切替回数: {dssms_analysis['total_switches']}回",
            f"平均保有期間: {dssms_analysis['avg_holding_period_hours']:.1f}時間",
            f"切替成功率: {dssms_analysis['switch_success_rate']:.2%}",
            f"切替コスト合計: {dssms_analysis['total_switch_cost']:,.2f}{self.report_config['currency_symbol']}",
            f"平均切替コスト: {dssms_analysis['avg_switch_cost']:,.2f}{self.report_config['currency_symbol']}",
            f"1日あたり切替頻度: {dssms_analysis['switch_frequency_per_day']:.2f}回",
            f"動的選択効率: {dssms_analysis['dynamic_selection_efficiency']:.3f}",
        ])
        
        # データソース分析
        report_lines.extend([
            "",
            "【データソース分析】",
            "-" * 40,
            f"主要データソース: {data_source_analysis['primary_source']}",
            f"データ品質スコア: {data_source_analysis['quality_score']:.3f}",
            f"実データ使用率: {data_source_analysis['real_data_percentage']:.1%}",
            f"フォールバック使用回数: {data_source_analysis['fallback_usage']}回",
            f"データ完全性: {data_source_analysis['data_completeness']:.1%}",
            f"強化レベル: {data_source_analysis['enhancement_level']}",
        ])
        
        # シミュレーション品質
        report_lines.extend([
            "",
            "【シミュレーション品質】",
            "-" * 40,
            f"総合品質スコア: {simulation_quality['overall_quality']:.3f}",
            f"検出された異常: {simulation_quality['anomalies_detected']}件",
            f"適用された補正: {simulation_quality['corrections_applied']}件",
            f"適用されたリアリズム要因: {len(simulation_quality['realism_factors'])}種類",
            f"データ整合性: {simulation_quality['data_consistency']}",
            f"シミュレーション信頼性: {simulation_quality['simulation_reliability']}",
        ])
        
        # 比較分析
        if enhanced_comparison:
            report_lines.extend([
                "",
                "【比較分析結果】",
                "-" * 40
            ])
            
            for benchmark, comparison in enhanced_comparison.items():
                report_lines.append(
                    f"{benchmark}: {comparison['performance_difference']:+.2%} ({comparison['evaluation']})"
                )
        
        # 推奨事項
        if enhanced_recommendations:
            report_lines.extend([
                "",
                "【推奨事項】",
                "-" * 40
            ])
            
            for recommendation in enhanced_recommendations:
                report_lines.append(recommendation)
        
        # フッター
        report_lines.extend([
            "",
            "=" * 80,
            f"レポート生成時刻: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
            "DSSMS強化レポーターv1.0 - Task 1.2実装版",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def _generate_error_report(self, error_message: str) -> str:
        """エラーレポート生成"""
        return f"""
================================================================================
DSSMS バックテストレポート生成エラー
================================================================================

エラー発生時刻: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
エラー内容: {error_message}

レポート生成に失敗しました。
データの確認またはシステム管理者にお問い合わせください。

================================================================================
"""

    def add_data_source_analysis(self, report_content: str, data_analysis: Dict[str, Any]) -> str:
        """レポートにデータソース分析を追加"""
        # 既存レポートに分析を挿入
        lines = report_content.split('\n')
        
        # 挿入位置を検索
        insert_index = -1
        for i, line in enumerate(lines):
            if "【DSSMS固有指標】" in line:
                insert_index = i
                break
        
        if insert_index == -1:
            # 挿入位置が見つからない場合は末尾に追加
            return report_content + "\n\n" + self._format_data_source_section(data_analysis)
        
        # 分析セクションを挿入
        data_section = self._format_data_source_section(data_analysis).split('\n')
        lines[insert_index:insert_index] = data_section + ['']
        
        return '\n'.join(lines)
    
    def _format_data_source_section(self, data_analysis: Dict[str, Any]) -> str:
        """データソース分析セクションのフォーマット"""
        return f"""【データソース詳細分析】
----------------------------------------
データ取得成功率: {data_analysis.get('success_rate', 0):.1%}
レスポンス時間: {data_analysis.get('response_time', 0):.3f}秒
キャッシュヒット率: {data_analysis.get('cache_hit_rate', 0):.1%}
データ新鮮度: {data_analysis.get('data_freshness', 'unknown')}
品質改善効果: {data_analysis.get('quality_improvement', 0):.2%}"""

    def include_simulation_quality_metrics(self, performance_metrics: Dict[str, Any], 
                                         quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """シミュレーション品質指標をパフォーマンスメトリクスに統合"""
        enhanced_metrics = performance_metrics.copy()
        
        # 品質調整済みメトリクス
        quality_factor = quality_metrics.get('overall_quality', 1.0)
        
        if 'sharpe_ratio' in enhanced_metrics:
            enhanced_metrics['quality_adjusted_sharpe'] = enhanced_metrics['sharpe_ratio'] * quality_factor
        
        if 'total_return' in enhanced_metrics:
            enhanced_metrics['quality_adjusted_return'] = enhanced_metrics['total_return'] * quality_factor
        
        # 品質メトリクス追加
        enhanced_metrics.update({
            'simulation_quality_score': quality_factor,
            'data_reliability': quality_metrics.get('data_reliability', 0.0),
            'enhancement_benefit': quality_metrics.get('enhancement_benefit', 0.0)
        })
        
        return enhanced_metrics

def demo_enhanced_reporter():
    """強化レポーターデモ"""
    print("=== DSSMS 強化レポーターデモ ===")
    
    try:
        # レポーター初期化
        reporter = DSSMSEnhancedReporter()
        
        # テストデータ作成
        test_simulation_result = {
            'execution_time': '2025年08月25日 12:00:00',
            'backtest_period': '2024-01-01 - 2024-12-31',
            'initial_capital': 1000000,
            'final_portfolio_value': 1150000,
            'total_return': 0.15,
            'data_source': 'enhanced_real_data',
            'enhancement_applied': True,
            'performance_history': {
                'daily_returns': [np.random.normal(0.001, 0.02) for _ in range(252)],
                'portfolio_value': [1000000 * (1 + 0.15 * i / 252) for i in range(252)],
                'positions': ['7203.T'] * 252,
                'timestamps': [datetime.now() - timedelta(days=i) for i in range(252, 0, -1)]
            },
            'switch_history': [
                {
                    'switch_cost': 5000,
                    'holding_period_hours': 48,
                    'profit_loss_at_switch': 2000
                }
                for _ in range(50)
            ],
            'data_quality': {
                'overall_score': 0.85,
                'real_data_ratio': 0.90,
                'fallback_count': 5,
                'completeness': 0.95
            },
            'quality_metrics': {
                'overall_quality': 0.88,
                'anomalies_detected': 2,
                'corrections_applied': 1,
                'realism_factors_applied': ['market_impact', 'slippage'],
                'data_consistency': 'good'
            },
            'comparison_results': {
                'vs_7203': 0.08,
                'vs_9984': -0.02,
                'vs_^N225': 0.12
            }
        }
        
        # 強化レポート生成テスト
        print(f"\n📊 強化レポート生成テスト")
        enhanced_report = reporter.generate_enhanced_detailed_report(test_simulation_result)
        
        # レポートの一部表示（最初の30行）
        report_lines = enhanced_report.split('\n')
        print("生成されたレポート（抜粋）:")
        print("-" * 60)
        for line in report_lines[:30]:
            print(line)
        
        if len(report_lines) > 30:
            print(f"... （他 {len(report_lines) - 30} 行）")
        
        print(f"\n📈 レポート統計")
        print(f"   総行数: {len(report_lines)}行")
        print(f"   文字数: {len(enhanced_report):,}文字")
        print(f"   セクション数: {enhanced_report.count('【')}")
        
        return True
        
    except Exception as e:
        print(f"❌ デモエラー: {e}")
        return False

if __name__ == "__main__":
    success = demo_enhanced_reporter()
    if success:
        print("\n✅ 強化レポーターデモ完了")
    else:
        print("\n❌ 強化レポーターデモ失敗")
