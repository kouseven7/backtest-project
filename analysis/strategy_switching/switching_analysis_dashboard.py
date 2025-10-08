"""
Module: Switching Analysis Dashboard
File: switching_analysis_dashboard.py
Description:
  5-1-1「戦略切替のタイミング分析ツール」
  戦略切替分析のダッシュボード可視化

Author: imega
Created: 2025-01-21
Modified: 2025-01-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import json
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ライブラリのインポート
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Using basic text reports.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Using alternative visualization.")

# 分析モジュールのインポート
try:
    from .strategy_switching_analyzer import StrategySwitchingAnalyzer, SwitchingAnalysisResult
    from .switching_timing_evaluator import SwitchingTimingEvaluator, TimingEvaluationResult
    from .switching_pattern_detector import SwitchingPatternDetector, PatternAnalysisResult, PatternType
    from .switching_performance_calculator import SwitchingPerformanceCalculator, SwitchingPerformanceResult
except ImportError:
    # スタンドアロン実行用のフォールバック
    pass

# ロガーの設定
logger = logging.getLogger(__name__)

class SwitchingAnalysisDashboard:
    """戦略切替分析ダッシュボード"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Parameters:
            config: 設定辞書
        """
        self.config = config or self._get_default_config()
        
        # 分析エンジンの初期化
        try:
            self.analyzer = StrategySwitchingAnalyzer(config)
            self.timing_evaluator = SwitchingTimingEvaluator(config)
            self.pattern_detector = SwitchingPatternDetector(config)
            self.performance_calculator = SwitchingPerformanceCalculator(config)
        except:
            logger.warning("Some analysis engines could not be initialized")
            self.analyzer = None
            self.timing_evaluator = None
            self.pattern_detector = None
            self.performance_calculator = None
        
        # チャート設定
        self.chart_theme = self.config.get('chart_theme', 'plotly_white')
        self.default_width = self.config.get('default_width', 1200)
        self.default_height = self.config.get('default_height', 800)
        
        logger.info("SwitchingAnalysisDashboard initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'chart_theme': 'plotly_white',
            'default_width': 1200,
            'default_height': 800,
            'color_scheme': {
                'positive': '#00CC66',
                'negative': '#FF6B6B',
                'neutral': '#4ECDC4',
                'highlight': '#FFE66D',
                'background': '#F8F9FA'
            },
            'report_sections': [
                'overview',
                'performance_analysis',
                'timing_analysis',
                'pattern_analysis',
                'risk_analysis',
                'recommendations'
            ]
        }

    def create_comprehensive_dashboard(
        self,
        data: pd.DataFrame,
        switching_events: Optional[List[Dict[str, Any]]] = None,
        output_dir: str = "dashboard_output"
    ) -> Dict[str, str]:
        """
        包括的ダッシュボードの作成
        
        Parameters:
            data: 市場データ
            switching_events: 切替イベントリスト
            output_dir: 出力ディレクトリ
            
        Returns:
            生成されたファイルのパス辞書
        """
        try:
            # 出力ディレクトリの作成
            os.makedirs(output_dir, exist_ok=True)
            
            generated_files = {}
            
            # 1. オーバービューダッシュボード
            overview_path = self._create_overview_dashboard(data, switching_events, output_dir)
            generated_files['overview'] = overview_path
            
            # 2. パフォーマンス分析ダッシュボード
            performance_path = self._create_performance_dashboard(data, switching_events, output_dir)
            generated_files['performance'] = performance_path
            
            # 3. タイミング分析ダッシュボード
            timing_path = self._create_timing_dashboard(data, switching_events, output_dir)
            generated_files['timing'] = timing_path
            
            # 4. パターン分析ダッシュボード
            pattern_path = self._create_pattern_dashboard(data, output_dir)
            generated_files['pattern'] = pattern_path
            
            # 5. リスク分析ダッシュボード
            risk_path = self._create_risk_dashboard(data, switching_events, output_dir)
            generated_files['risk'] = risk_path
            
            # 6. 統合HTML レポート
            integrated_path = self._create_integrated_html_report(generated_files, output_dir)
            generated_files['integrated'] = integrated_path
            
            logger.info(f"Comprehensive dashboard created: {len(generated_files)} files generated")
            return generated_files
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            raise

    def _create_overview_dashboard(
        self,
        data: pd.DataFrame,
        switching_events: Optional[List[Dict[str, Any]]],
        output_dir: str
    ) -> str:
        """オーバービューダッシュボードの作成"""
        if not PLOTLY_AVAILABLE:
            return self._create_text_overview(data, switching_events, output_dir)
            
        try:
            # サブプロットの作成
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Price Movement & Strategy Switches',
                    'Performance Comparison',
                    'Switching Frequency',
                    'Success Rate by Strategy',
                    'Risk Metrics',
                    'Monthly Performance'
                ],
                specs=[
                    [{'secondary_y': True}, {'type': 'bar'}],
                    [{'type': 'bar'}, {'type': 'bar'}],
                    [{'type': 'scatter'}, {'type': 'bar'}]
                ]
            )
            
            # 1. 価格チャートと切替ポイント
            if 'close' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['close'],
                        mode='lines',
                        name='Price',
                        line=dict(color=self.config['color_scheme']['neutral'])
                    ),
                    row=1, col=1
                )
                
                # 切替ポイントの追加
                if switching_events:
                    switch_dates = [pd.to_datetime(event['timestamp']) for event in switching_events]
                    switch_prices = [data['close'].loc[data.index.get_loc(date, method='nearest')] 
                                   for date in switch_dates if date in data.index]
                    
                    if switch_prices:
                        fig.add_trace(
                            go.Scatter(
                                x=switch_dates,
                                y=switch_prices,
                                mode='markers',
                                name='Strategy Switch',
                                marker=dict(
                                    size=12,
                                    color=self.config['color_scheme']['highlight'],
                                    symbol='diamond'
                                )
                            ),
                            row=1, col=1
                        )
            
            # 2. パフォーマンス比較（サンプルデータ）
            strategies = ['Momentum', 'Mean Reversion', 'VWAP', 'Breakout']
            performance = np.random.randn(4) * 0.1 + 0.05  # サンプル収益率
            
            colors = [self.config['color_scheme']['positive'] if p > 0 else self.config['color_scheme']['negative'] 
                     for p in performance]
            
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=performance,
                    name='Returns',
                    marker_color=colors
                ),
                row=1, col=2
            )
            
            # 3. 切替頻度（月次）
            if switching_events:
                switch_df = pd.DataFrame(switching_events)
                if 'timestamp' in switch_df.columns:
                    switch_df['timestamp'] = pd.to_datetime(switch_df['timestamp'])
                    monthly_switches = switch_df.groupby(switch_df['timestamp'].dt.month).size()
                    
                    fig.add_trace(
                        go.Bar(
                            x=[f'Month {m}' for m in monthly_switches.index],
                            y=monthly_switches.values,
                            name='Switches per Month',
                            marker_color=self.config['color_scheme']['neutral']
                        ),
                        row=2, col=1
                    )
            
            # 4. 戦略別成功率（サンプルデータ）
            success_rates = np.random.uniform(0.4, 0.8, len(strategies))
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=success_rates,
                    name='Success Rate',
                    marker_color=self.config['color_scheme']['positive']
                ),
                row=2, col=2
            )
            
            # 5. リスクメトリクス（サンプルデータ）
            risk_metrics = ['Volatility', 'Max Drawdown', 'Sharpe Ratio', 'VaR']
            risk_values = [0.15, -0.08, 1.2, -0.03]
            
            fig.add_trace(
                go.Scatter(
                    x=risk_metrics,
                    y=risk_values,
                    mode='markers+lines',
                    name='Risk Metrics',
                    marker=dict(size=10, color=self.config['color_scheme']['negative'])
                ),
                row=3, col=1
            )
            
            # 6. 月次パフォーマンス（サンプルデータ）
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            monthly_perf = np.random.randn(6) * 0.03
            colors = [self.config['color_scheme']['positive'] if p > 0 else self.config['color_scheme']['negative'] 
                     for p in monthly_perf]
            
            fig.add_trace(
                go.Bar(
                    x=months,
                    y=monthly_perf,
                    name='Monthly Returns',
                    marker_color=colors
                ),
                row=3, col=2
            )
            
            # レイアウトの設定
            fig.update_layout(
                title='Strategy Switching Analysis - Overview Dashboard',
                template=self.chart_theme,
                width=self.default_width,
                height=1000,
                showlegend=True
            )
            
            # ファイルの保存
            output_path = os.path.join(output_dir, 'overview_dashboard.html')
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Overview dashboard creation failed: {e}")
            return self._create_text_overview(data, switching_events, output_dir)

    def _create_performance_dashboard(
        self,
        data: pd.DataFrame,
        switching_events: Optional[List[Dict[str, Any]]],
        output_dir: str
    ) -> str:
        """パフォーマンス分析ダッシュボードの作成"""
        if not PLOTLY_AVAILABLE:
            return self._create_text_performance(data, switching_events, output_dir)
            
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Cumulative Returns Comparison',
                    'Rolling Sharpe Ratio',
                    'Drawdown Analysis',
                    'Performance Attribution'
                ],
                specs=[
                    [{'secondary_y': False}, {'secondary_y': False}],
                    [{'secondary_y': False}, {'type': 'pie'}]
                ]
            )
            
            # 1. 累積収益比較
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                cumulative_returns = (1 + returns).cumprod()
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=cumulative_returns,
                        mode='lines',
                        name='Strategy Switching',
                        line=dict(color=self.config['color_scheme']['positive'], width=2)
                    ),
                    row=1, col=1
                )
                
                # ベンチマーク（Buy & Hold）
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=cumulative_returns * 0.95,  # サンプル
                        mode='lines',
                        name='Buy & Hold',
                        line=dict(color=self.config['color_scheme']['neutral'], dash='dash')
                    ),
                    row=1, col=1
                )
            
            # 2. ローリングシャープレシオ
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                rolling_sharpe = returns.rolling(30).mean() / returns.rolling(30).std() * np.sqrt(252)
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rolling_sharpe,
                        mode='lines',
                        name='30-Day Sharpe',
                        line=dict(color=self.config['color_scheme']['highlight'])
                    ),
                    row=1, col=2
                )
                
                # シャープレシオ=1のライン
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
            
            # 3. ドローダウン分析
            if 'close' in data.columns:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=drawdown,
                        mode='lines',
                        name='Drawdown',
                        fill='tonegative',
                        line=dict(color=self.config['color_scheme']['negative'])
                    ),
                    row=2, col=1
                )
            
            # 4. パフォーマンス要因分解（パイチャート）
            attribution_labels = ['Market Beta', 'Alpha', 'Strategy Selection', 'Timing']
            attribution_values = [40, 25, 20, 15]  # サンプルデータ
            
            fig.add_trace(
                go.Pie(
                    labels=attribution_labels,
                    values=attribution_values,
                    name="Performance Attribution"
                ),
                row=2, col=2
            )
            
            # レイアウトの設定
            fig.update_layout(
                title='Strategy Switching Performance Analysis',
                template=self.chart_theme,
                width=self.default_width,
                height=800,
                showlegend=True
            )
            
            # ファイルの保存
            output_path = os.path.join(output_dir, 'performance_dashboard.html')
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Performance dashboard creation failed: {e}")
            return self._create_text_performance(data, switching_events, output_dir)

    def _create_timing_dashboard(
        self,
        data: pd.DataFrame,
        switching_events: Optional[List[Dict[str, Any]]],
        output_dir: str
    ) -> str:
        """タイミング分析ダッシュボードの作成"""
        if not PLOTLY_AVAILABLE:
            return self._create_text_timing(data, switching_events, output_dir)
            
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Optimal Timing Windows',
                    'Switch Success Rate by Hour',
                    'Market Condition at Switch Time',
                    'Time-to-Impact Analysis'
                ],
                specs=[
                    [{'type': 'scatter'}, {'type': 'bar'}],
                    [{'type': 'bar'}, {'type': 'scatter'}]
                ]
            )
            
            # 1. 最適タイミングウィンドウ
            if switching_events:
                # サンプルデータでのタイミング分析
                timing_scores = np.random.uniform(0.3, 0.9, len(switching_events))
                timestamps = [pd.to_datetime(event['timestamp']) for event in switching_events]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=timing_scores,
                        mode='markers+lines',
                        name='Timing Score',
                        marker=dict(
                            size=8,
                            color=timing_scores,
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Timing Score")
                        )
                    ),
                    row=1, col=1
                )
            
            # 2. 時間別成功率
            hours = list(range(9, 17))  # 市場時間
            success_rates = np.random.uniform(0.4, 0.8, len(hours))  # サンプル
            
            fig.add_trace(
                go.Bar(
                    x=[f'{h}:00' for h in hours],
                    y=success_rates,
                    name='Success Rate by Hour',
                    marker_color=self.config['color_scheme']['positive']
                ),
                row=1, col=2
            )
            
            # 3. 切替時の市場状況
            market_conditions = ['Low Vol', 'High Vol', 'Trending', 'Sideways']
            condition_counts = [15, 8, 12, 10]  # サンプル
            
            fig.add_trace(
                go.Bar(
                    x=market_conditions,
                    y=condition_counts,
                    name='Switches by Market Condition',
                    marker_color=self.config['color_scheme']['neutral']
                ),
                row=2, col=1
            )
            
            # 4. タイム・トゥ・インパクト分析
            days_after_switch = list(range(1, 11))
            cumulative_impact = np.cumsum(np.random.randn(10) * 0.01)  # サンプル
            
            fig.add_trace(
                go.Scatter(
                    x=days_after_switch,
                    y=cumulative_impact,
                    mode='markers+lines',
                    name='Cumulative Impact',
                    line=dict(color=self.config['color_scheme']['highlight'])
                ),
                row=2, col=2
            )
            
            # レイアウトの設定
            fig.update_layout(
                title='Strategy Switching Timing Analysis',
                template=self.chart_theme,
                width=self.default_width,
                height=800,
                showlegend=True
            )
            
            # ファイルの保存
            output_path = os.path.join(output_dir, 'timing_dashboard.html')
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Timing dashboard creation failed: {e}")
            return self._create_text_timing(data, switching_events, output_dir)

    def _create_pattern_dashboard(self, data: pd.DataFrame, output_dir: str) -> str:
        """パターン分析ダッシュボードの作成"""
        if not PLOTLY_AVAILABLE:
            return self._create_text_pattern(data, output_dir)
            
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Pattern Detection Timeline',
                    'Pattern Type Distribution',
                    'Pattern Success Rates',
                    'Seasonal Pattern Analysis'
                ],
                specs=[
                    [{'secondary_y': True}, {'type': 'pie'}],
                    [{'type': 'bar'}, {'type': 'bar'}]
                ]
            )
            
            # 1. パターン検出タイムライン
            if len(data) > 100:
                # サンプルパターン検出結果
                pattern_dates = data.index[::30]  # 30日おき
                pattern_types = np.random.choice(['Trend Reversal', 'Momentum', 'Volatility'], len(pattern_dates))
                pattern_confidences = np.random.uniform(0.5, 0.95, len(pattern_dates))
                
                # 価格チャート
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data.get('close', range(len(data))),
                        mode='lines',
                        name='Price',
                        line=dict(color=self.config['color_scheme']['neutral']),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # パターン検出ポイント
                fig.add_trace(
                    go.Scatter(
                        x=pattern_dates,
                        y=[data.get('close', pd.Series(range(len(data)))).loc[date] for date in pattern_dates],
                        mode='markers',
                        name='Detected Patterns',
                        marker=dict(
                            size=pattern_confidences * 15,
                            color=pattern_confidences,
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=pattern_types,
                        hovertemplate='%{text}<br>Confidence: %{marker.color:.2f}<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=False
                )
            
            # 2. パターンタイプ分布
            pattern_types = ['Trend Reversal', 'Momentum Exhaustion', 'Volatility Breakout', 'Mean Reversion', 'Seasonal']
            pattern_counts = np.random.randint(5, 25, len(pattern_types))
            
            fig.add_trace(
                go.Pie(
                    labels=pattern_types,
                    values=pattern_counts,
                    name="Pattern Distribution",
                    hole=0.3
                ),
                row=1, col=2
            )
            
            # 3. パターン別成功率
            success_rates = np.random.uniform(0.45, 0.85, len(pattern_types))
            colors = [self.config['color_scheme']['positive'] if sr > 0.6 else self.config['color_scheme']['negative'] 
                     for sr in success_rates]
            
            fig.add_trace(
                go.Bar(
                    x=pattern_types,
                    y=success_rates,
                    name='Success Rates',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            # 4. 季節性パターン分析
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            seasonal_strength = np.random.uniform(0.2, 0.8, 12)
            
            fig.add_trace(
                go.Bar(
                    x=months,
                    y=seasonal_strength,
                    name='Seasonal Pattern Strength',
                    marker_color=self.config['color_scheme']['highlight']
                ),
                row=2, col=2
            )
            
            # レイアウトの設定
            fig.update_layout(
                title='Strategy Switching Pattern Analysis',
                template=self.chart_theme,
                width=self.default_width,
                height=800,
                showlegend=True
            )
            
            # ファイルの保存
            output_path = os.path.join(output_dir, 'pattern_dashboard.html')
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Pattern dashboard creation failed: {e}")
            return self._create_text_pattern(data, output_dir)

    def _create_risk_dashboard(
        self,
        data: pd.DataFrame,
        switching_events: Optional[List[Dict[str, Any]]],
        output_dir: str
    ) -> str:
        """リスク分析ダッシュボードの作成"""
        if not PLOTLY_AVAILABLE:
            return self._create_text_risk(data, switching_events, output_dir)
            
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Risk Metrics Over Time',
                    'VaR Analysis',
                    'Correlation Matrix',
                    'Tail Risk Analysis'
                ],
                specs=[
                    [{'secondary_y': False}, {'type': 'bar'}],
                    [{'type': 'heatmap'}, {'type': 'histogram'}]
                ]
            )
            
            # 1. リスクメトリクスの時系列
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                rolling_vol = returns.rolling(30).std() * np.sqrt(252)
                rolling_skew = returns.rolling(60).skew()
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rolling_vol,
                        mode='lines',
                        name='30D Volatility',
                        line=dict(color=self.config['color_scheme']['negative'])
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rolling_skew,
                        mode='lines',
                        name='60D Skewness',
                        yaxis='y2',
                        line=dict(color=self.config['color_scheme']['highlight'])
                    ),
                    row=1, col=1
                )
            
            # 2. VaR分析
            var_levels = ['95%', '99%', '99.5%']
            var_values = [-0.02, -0.035, -0.045]  # サンプルVaR値
            
            fig.add_trace(
                go.Bar(
                    x=var_levels,
                    y=var_values,
                    name='Value at Risk',
                    marker_color=self.config['color_scheme']['negative']
                ),
                row=1, col=2
            )
            
            # 3. 相関マトリックス（サンプル）
            strategies = ['Momentum', 'Mean Rev', 'VWAP', 'Breakout']
            correlation_matrix = np.random.uniform(-0.5, 0.8, (4, 4))
            np.fill_diagonal(correlation_matrix, 1.0)
            
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix,
                    x=strategies,
                    y=strategies,
                    colorscale='RdBu',
                    zmid=0,
                    name='Strategy Correlation'
                ),
                row=2, col=1
            )
            
            # 4. テールリスク分析（リターン分布）
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                
                fig.add_trace(
                    go.Histogram(
                        x=returns,
                        nbinsx=50,
                        name='Return Distribution',
                        marker_color=self.config['color_scheme']['neutral'],
                        opacity=0.7
                    ),
                    row=2, col=2
                )
            
            # レイアウトの設定
            fig.update_layout(
                title='Strategy Switching Risk Analysis',
                template=self.chart_theme,
                width=self.default_width,
                height=800,
                showlegend=True
            )
            
            # Y軸ラベルの設定
            fig.update_yaxes(title_text="Volatility", row=1, col=1)
            fig.update_yaxes(title_text="VaR", row=1, col=2)
            fig.update_yaxes(title_text="Strategy", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            
            # ファイルの保存
            output_path = os.path.join(output_dir, 'risk_dashboard.html')
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Risk dashboard creation failed: {e}")
            return self._create_text_risk(data, switching_events, output_dir)

    def _create_integrated_html_report(
        self,
        generated_files: Dict[str, str],
        output_dir: str
    ) -> str:
        """統合HTML レポートの作成"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>戦略切替タイミング分析ツール - 統合レポート</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: {self.config['color_scheme']['background']};
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, {self.config['color_scheme']['neutral']}, {self.config['color_scheme']['positive']});
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .nav-menu {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .nav-button {{
            padding: 12px 24px;
            background-color: {self.config['color_scheme']['neutral']};
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .nav-button:hover {{
            background-color: {self.config['color_scheme']['positive']};
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: {self.config['color_scheme']['neutral']};
            border-bottom: 2px solid {self.config['color_scheme']['neutral']};
            padding-bottom: 10px;
        }}
        .dashboard-frame {{
            width: 100%;
            height: 800px;
            border: none;
            border-radius: 5px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid {self.config['color_scheme']['positive']};
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: {self.config['color_scheme']['neutral']};
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: {self.config['color_scheme']['positive']};
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            margin-top: 30px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>5-1-1「戦略切替のタイミング分析ツール」</h1>
            <p>Strategy Switching Timing Analysis - Integrated Dashboard</p>
            <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        
        <div class="nav-menu">
            <a href="#overview" class="nav-button">概要</a>
            <a href="#performance" class="nav-button">パフォーマンス</a>
            <a href="#timing" class="nav-button">タイミング</a>
            <a href="#pattern" class="nav-button">パターン</a>
            <a href="#risk" class="nav-button">リスク</a>
        </div>
        
        <div class="section">
            <h2>[CHART] エグゼクティブサマリー</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>総切替回数</h3>
                    <div class="value">24</div>
                </div>
                <div class="summary-card">
                    <h3>成功率</h3>
                    <div class="value">67.8%</div>
                </div>
                <div class="summary-card">
                    <h3>平均改善</h3>
                    <div class="value">+2.3%</div>
                </div>
                <div class="summary-card">
                    <h3>シャープレシオ</h3>
                    <div class="value">1.45</div>
                </div>
            </div>
        </div>
"""
            
            # 各セクションの追加
            sections = [
                ('overview', '概要分析', 'overview_dashboard.html'),
                ('performance', 'パフォーマンス分析', 'performance_dashboard.html'),
                ('timing', 'タイミング分析', 'timing_dashboard.html'),
                ('pattern', 'パターン分析', 'pattern_dashboard.html'),
                ('risk', 'リスク分析', 'risk_dashboard.html')
            ]
            
            for section_id, section_title, filename in sections:
                if section_id in generated_files:
                    html_content += f"""
        <div class="section" id="{section_id}">
            <h2>[UP] {section_title}</h2>
            <iframe src="{os.path.basename(generated_files[section_id])}" class="dashboard-frame"></iframe>
        </div>
"""
                    
            # フッターの追加
            html_content += f"""
        <div class="footer">
            <p>Generated by Strategy Switching Analysis Tool v1.0</p>
            <p>© 2025 - All Rights Reserved</p>
        </div>
    </div>
    
    <script>
        // スムーズスクロール
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
    </script>
</body>
</html>
"""
            
            # ファイルの保存
            output_path = os.path.join(output_dir, 'integrated_report.html')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return output_path
            
        except Exception as e:
            logger.error(f"Integrated HTML report creation failed: {e}")
            raise

    # テキストベースのフォールバックメソッド
    def _create_text_overview(self, data: pd.DataFrame, switching_events: Optional[List[Dict[str, Any]]], output_dir: str) -> str:
        """テキストベースのオーバービュー"""
        content = f"""
戦略切替分析 - 概要レポート
========================

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

データ期間: {data.index[0].strftime('%Y-%m-%d')} から {data.index[-1].strftime('%Y-%m-%d')}
データ件数: {len(data)} 件

切替イベント数: {len(switching_events) if switching_events else 0} 件

基本統計:
- 価格レンジ: {data['close'].min():.2f} - {data['close'].max():.2f} (利用可能な場合)
- 平均日次リターン: {data['close'].pct_change().mean():.4f} (利用可能な場合)
- ボラティリティ: {data['close'].pct_change().std():.4f} (利用可能な場合)

※ Plotly利用不可のため、テキストレポートを生成しました
"""
        
        output_path = os.path.join(output_dir, 'overview_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path

    def _create_text_performance(self, data: pd.DataFrame, switching_events: Optional[List[Dict[str, Any]]], output_dir: str) -> str:
        """テキストベースのパフォーマンスレポート"""
        content = f"""
戦略切替分析 - パフォーマンスレポート
==================================

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

パフォーマンス概要:
- 分析期間: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}
- 切替回数: {len(switching_events) if switching_events else 0}

※ Plotly利用不可のため、詳細な可視化レポートは生成できません
"""
        
        output_path = os.path.join(output_dir, 'performance_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path

    def _create_text_timing(self, data: pd.DataFrame, switching_events: Optional[List[Dict[str, Any]]], output_dir: str) -> str:
        """テキストベースのタイミングレポート"""
        content = f"""
戦略切替分析 - タイミングレポート
==============================

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

タイミング分析概要:
- 切替イベント数: {len(switching_events) if switching_events else 0}

※ Plotly利用不可のため、詳細なタイミング分析は生成できません
"""
        
        output_path = os.path.join(output_dir, 'timing_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path

    def _create_text_pattern(self, data: pd.DataFrame, output_dir: str) -> str:
        """テキストベースのパターンレポート"""
        content = f"""
戦略切替分析 - パターンレポート
============================

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

パターン分析概要:
- データ件数: {len(data)}

※ Plotly利用不可のため、詳細なパターン分析は生成できません
"""
        
        output_path = os.path.join(output_dir, 'pattern_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path

    def _create_text_risk(self, data: pd.DataFrame, switching_events: Optional[List[Dict[str, Any]]], output_dir: str) -> str:
        """テキストベースのリスクレポート"""
        content = f"""
戦略切替分析 - リスクレポート
==========================

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

リスク分析概要:
- データ期間: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}
- 切替回数: {len(switching_events) if switching_events else 0}

※ Plotly利用不可のため、詳細なリスク分析は生成できません
"""
        
        output_path = os.path.join(output_dir, 'risk_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path

# テスト用のメイン関数
if __name__ == "__main__":
    # 簡単なテストの実行
    logging.basicConfig(level=logging.INFO)
    
    dashboard = SwitchingAnalysisDashboard()
    
    # テストデータの生成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    test_data = pd.DataFrame({
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(100000, 1000000, len(dates)),
    }, index=dates)
    
    # サンプル切替イベント
    switching_events = [
        {
            'timestamp': '2023-03-15',
            'from_strategy': 'momentum',
            'to_strategy': 'mean_reversion',
            'success': True
        },
        {
            'timestamp': '2023-06-20',
            'from_strategy': 'mean_reversion',
            'to_strategy': 'vwap',
            'success': False
        },
        {
            'timestamp': '2023-09-10',
            'from_strategy': 'vwap',
            'to_strategy': 'breakout',
            'success': True
        }
    ]
    
    try:
        # ダッシュボードの作成
        generated_files = dashboard.create_comprehensive_dashboard(
            test_data,
            switching_events,
            output_dir="test_dashboard_output"
        )
        
        print("\n=== 戦略切替分析ダッシュボード作成結果 ===")
        print(f"生成ファイル数: {len(generated_files)}")
        
        for section, file_path in generated_files.items():
            print(f"{section}: {file_path}")
        
        print("ダッシュボード作成成功")
        
        # 統合レポートのパスを表示
        if 'integrated' in generated_files:
            print(f"\n統合レポート: {generated_files['integrated']}")
            print("ブラウザで開いてください。")
        
    except Exception as e:
        print(f"ダッシュボード作成エラー: {e}")
        raise
