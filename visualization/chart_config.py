"""
Chart Configuration Manager for 4-3-1 Trend Strategy Time Series Visualization

チャート設定・スタイリング・色管理システム
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, Any, Optional, List


class ChartConfigManager:
    """チャート設定管理クラス"""
    
    def __init__(self):
        self.figure_config = {
            'figsize': (16, 12),
            'dpi': 100,
            'facecolor': 'white',
            'edgecolor': 'black'
        }
        
        self.color_scheme = {
            'uptrend': '#2E8B57',      # Sea Green
            'downtrend': '#DC143C',    # Crimson
            'sideways': '#4682B4',     # Steel Blue
            'price_line': '#1f77b4',   # Default blue
            'volume_bar': '#ffb347',   # Peach
            'grid': '#e0e0e0',         # Light gray
            'background': '#f8f9fa'    # Very light gray
        }
        
        self.strategy_colors = {
            'trend_following': '#FF6B6B',  # Coral
            'mean_reversion': '#4ECDC4',   # Turquoise
            'momentum': '#45B7D1',         # Sky blue
            'breakout': '#96CEB4',         # Mint
            'hybrid': '#FFEAA7'            # Light yellow
        }
        
        self.panel_config = {
            'price_panel': {'height_ratio': 3, 'ylabel': 'Price (JPY)'},
            'strategy_panel': {'height_ratio': 1, 'ylabel': 'Strategy'},
            'confidence_panel': {'height_ratio': 1, 'ylabel': 'Confidence (%)'},
            'volume_panel': {'height_ratio': 1, 'ylabel': 'Volume'}
        }
    
    def setup_matplotlib_style(self) -> None:
        """Matplotlibスタイル設定"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'figure.autolayout': True
        })
    
    def create_figure_and_axes(self) -> tuple:
        """図とサブプロットの作成"""
        height_ratios = [
            self.panel_config['price_panel']['height_ratio'],
            self.panel_config['strategy_panel']['height_ratio'],
            self.panel_config['confidence_panel']['height_ratio'],
            self.panel_config['volume_panel']['height_ratio']
        ]
        
        fig, axes = plt.subplots(
            4, 1, 
            figsize=self.figure_config['figsize'],
            height_ratios=height_ratios,
            sharex=True,
            facecolor=self.figure_config['facecolor']
        )
        
        return fig, axes
    
    def apply_panel_styling(self, ax, panel_type: str) -> None:
        """パネル個別スタイリング適用"""
        config = self.panel_config.get(panel_type, {})
        
        # Y軸ラベル設定
        if 'ylabel' in config:
            ax.set_ylabel(config['ylabel'], fontsize=10, fontweight='bold')
        
        # グリッド設定
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor(self.color_scheme['background'])
        
        # スパイン設定
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
    
    def format_x_axis(self, ax, dates: List[datetime]) -> None:
        """X軸（時間軸）フォーマット設定"""
        if not dates:
            return
            
        date_range = (max(dates) - min(dates)).days
        
        if date_range <= 30:
            # 30日以内：日単位表示
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_minor_locator(mdates.DayLocator())
        elif date_range <= 90:
            # 30-90日：週単位表示
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_minor_locator(mdates.DayLocator())
        else:
            # 90日超：月単位表示
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
        
        # X軸ラベル回転
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def get_trend_color(self, trend_type: str) -> str:
        """トレンドタイプに応じた色を取得"""
        return self.color_scheme.get(trend_type, '#808080')  # デフォルト: グレー
    
    def get_strategy_color(self, strategy_name: str) -> str:
        """戦略名に応じた色を取得"""
        strategy_lower = strategy_name.lower()
        
        for key, color in self.strategy_colors.items():
            if key in strategy_lower:
                return color
        
        return '#95a5a6'  # デフォルト: クールグレー
    
    def add_title_and_labels(self, fig, symbol: str, period_start: str, period_end: str) -> None:
        """タイトルとラベルの追加"""
        title = f"{symbol} - トレンド変化と戦略切替 ({period_start} ~ {period_end})"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        # X軸ラベル（最下部のみ）
        fig.text(0.5, 0.02, '日付', ha='center', fontsize=10, fontweight='bold')
    
    def add_legend(self, ax, handles: List, labels: List, location: str = 'upper right') -> None:
        """凡例の追加"""
        if handles and labels:
            legend = ax.legend(
                handles, labels,
                loc=location,
                fontsize=9,
                framealpha=0.8,
                fancybox=True,
                shadow=True
            )
            legend.get_frame().set_facecolor('white')
    
    def save_chart(self, fig, filepath: str, dpi: Optional[int] = None) -> None:
        """チャートの保存"""
        save_dpi = dpi or self.figure_config['dpi']
        
        fig.savefig(
            filepath,
            dpi=save_dpi,
            bbox_inches='tight',
            facecolor=self.figure_config['facecolor'],
            edgecolor=self.figure_config['edgecolor'],
            format='png'
        )
    
    def get_confidence_color_map(self) -> Dict[str, str]:
        """信頼度レベルに応じた色マップ"""
        return {
            'high': '#27ae60',      # Green
            'medium': '#f39c12',    # Orange  
            'low': '#e74c3c'        # Red
        }
    
    def close_figure(self, fig) -> None:
        """図のクローズ（メモリ解放）"""
        plt.close(fig)
    
    def get_chart_config_summary(self) -> Dict[str, Any]:
        """設定サマリーの取得（デバッグ用）"""
        return {
            'figure_config': self.figure_config,
            'color_scheme': self.color_scheme,
            'strategy_colors': self.strategy_colors,
            'panel_config': self.panel_config
        }
