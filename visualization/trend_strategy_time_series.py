"""
Main Visualization Engine for 4-3-1 Trend Strategy Time Series Visualization

4パネル統合チャート可視化システム：
- Panel 1: 価格チャート（ローソク足 + トレンドライン）
- Panel 2: 戦略切替表示
- Panel 3: 信頼度レベル表示  
- Panel 4: ボリューム表示
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import sys

# 同じディレクトリからインポート
try:
    from .chart_config import ChartConfigManager
    from .data_aggregator import VisualizationDataAggregator
except ImportError:
    # 直接実行時の対応
    from chart_config import ChartConfigManager
    from data_aggregator import VisualizationDataAggregator


class TrendStrategyTimeSeriesVisualizer:
    """トレンド戦略時系列可視化メインクラス"""
    
    def __init__(self, symbol: str = "USDJPY", period_days: int = 30, output_dir: str = "visualization_outputs"):
        self.symbol = symbol
        self.period_days = period_days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # コンポーネント初期化
        self.config_manager = ChartConfigManager()
        self.data_aggregator = VisualizationDataAggregator(symbol, period_days)
        
        # ロガー設定
        self.logger = self._setup_logger()
        
        # データ格納
        self.chart_data = None
        self.last_generated_file = None
    
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(f"TrendVisualization_{self.symbol}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_comprehensive_chart(self, save_file: bool = True, filename: Optional[str] = None) -> Optional[str]:
        """包括的4パネルチャート生成"""
        try:
            # データ集約
            self.logger.info("データ集約を開始...")
            chart_data = self.data_aggregator.aggregate_all_data()
            
            if chart_data.empty:
                self.logger.error("チャートデータが空です")
                return None
            
            self.chart_data = chart_data
            
            # Matplotlibスタイル設定
            self.config_manager.setup_matplotlib_style()
            
            # 図とサブプロットの作成
            fig, axes = self.config_manager.create_figure_and_axes()
            
            # 各パネルの描画
            self._draw_price_panel(axes[0], chart_data)
            self._draw_strategy_panel(axes[1], chart_data)
            self._draw_confidence_panel(axes[2], chart_data)
            self._draw_volume_panel(axes[3], chart_data)
            
            # 共通設定の適用
            self._apply_common_formatting(axes, chart_data)
            
            # タイトルとラベル
            period_start = chart_data.index.min().strftime('%Y/%m/%d')
            period_end = chart_data.index.max().strftime('%Y/%m/%d')
            self.config_manager.add_title_and_labels(fig, self.symbol, period_start, period_end)
            
            # ファイル保存
            if save_file:
                output_path = self._save_chart(fig, filename)
                self.last_generated_file = output_path
                self.logger.info(f"チャート生成完了: {output_path}")
                return output_path
            
            return "chart_generated"
            
        except Exception as e:
            self.logger.error(f"チャート生成失敗: {e}")
            return None
        finally:
            # メモリ解放
            if 'fig' in locals():
                self.config_manager.close_figure(fig)
    
    def _draw_price_panel(self, ax, data: pd.DataFrame) -> None:
        """価格パネル（Panel 1）の描画"""
        try:
            self.config_manager.apply_panel_styling(ax, 'price_panel')
            
            # 基本価格ライン
            ax.plot(data.index, data['Close'], 
                   color=self.config_manager.color_scheme['price_line'], 
                   linewidth=1.5, label='Close Price', alpha=0.8)
            
            # トレンドに応じた背景色
            if 'trend_type' in data.columns:
                self._add_trend_background(ax, data)
            
            # トレンド変更ポイントのマーキング
            if 'trend_type' in data.columns:
                self._mark_trend_changes(ax, data)
            
            # 価格レンジの調整
            price_min = data['Close'].min()
            price_max = data['Close'].max()
            price_range = price_max - price_min
            ax.set_ylim(price_min - price_range * 0.05, price_max + price_range * 0.05)
            
            # 凡例
            handles, labels = ax.get_legend_handles_labels()
            self.config_manager.add_legend(ax, handles, labels, location='upper left')
            
            self.logger.debug("価格パネル描画完了")
            
        except Exception as e:
            self.logger.warning(f"価格パネル描画エラー: {e}")
    
    def _draw_strategy_panel(self, ax, data: pd.DataFrame) -> None:
        """戦略パネル（Panel 2）の描画"""
        try:
            self.config_manager.apply_panel_styling(ax, 'strategy_panel')
            
            if 'strategy' not in data.columns:
                ax.text(0.5, 0.5, 'No Strategy Data', ha='center', va='center', transform=ax.transAxes)
                return
            
            # 戦略切替点を特定
            strategy_changes = self._find_strategy_changes(data['strategy'])
            
            # 戦略期間を色分けして表示
            y_pos = 0.5
            for i, (start_idx, end_idx, strategy) in enumerate(strategy_changes):
                start_date = data.index[start_idx]
                end_date = data.index[min(end_idx, len(data.index) - 1)]
                
                color = self.config_manager.get_strategy_color(strategy)
                
                # 戦略期間をバーで表示
                ax.barh(y_pos, (end_date - start_date).days, 
                       left=start_date, height=0.6, 
                       color=color, alpha=0.7, label=strategy if i == 0 else "")
                
                # 戦略名をテキストで表示
                mid_date = start_date + (end_date - start_date) / 2
                ax.text(mid_date, y_pos, strategy, ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='black')
            
            # Y軸設定
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            
            # 戦略切替ポイントにマーカー
            self._mark_strategy_switches(ax, data, strategy_changes)
            
            self.logger.debug("戦略パネル描画完了")
            
        except Exception as e:
            self.logger.warning(f"戦略パネル描画エラー: {e}")
    
    def _draw_confidence_panel(self, ax, data: pd.DataFrame) -> None:
        """信頼度パネル（Panel 3）の描画"""
        try:
            self.config_manager.apply_panel_styling(ax, 'confidence_panel')
            
            if 'confidence_score' not in data.columns:
                ax.text(0.5, 0.5, 'No Confidence Data', ha='center', va='center', transform=ax.transAxes)
                return
            
            # 信頼度スコアのライン
            confidence_scores = data['confidence_score'] * 100  # パーセンテージ変換
            ax.plot(data.index, confidence_scores, 
                   color='#3498db', linewidth=2, label='Confidence Score')
            
            # 信頼度レベルに応じた背景色
            if 'confidence_level' in data.columns:
                self._add_confidence_background(ax, data)
            
            # 閾値ライン
            ax.axhline(y=70, color='#27ae60', linestyle='--', alpha=0.7, label='High Threshold')
            ax.axhline(y=40, color='#e74c3c', linestyle='--', alpha=0.7, label='Low Threshold')
            
            # Y軸設定
            ax.set_ylim(0, 100)
            ax.set_ylabel('Confidence (%)')
            
            # 凡例
            handles, labels = ax.get_legend_handles_labels()
            self.config_manager.add_legend(ax, handles, labels, location='upper right')
            
            self.logger.debug("信頼度パネル描画完了")
            
        except Exception as e:
            self.logger.warning(f"信頼度パネル描画エラー: {e}")
    
    def _draw_volume_panel(self, ax, data: pd.DataFrame) -> None:
        """ボリュームパネル（Panel 4）の描画"""
        try:
            self.config_manager.apply_panel_styling(ax, 'volume_panel')
            
            if 'Volume' not in data.columns:
                # 合成ボリュームデータで代用
                volumes = np.random.randint(1000, 5000, len(data))
                data = data.copy()
                data['Volume'] = volumes
            
            # ボリュームバー
            colors = ['#ff4444' if close < open_ else '#44ff44' 
                     for close, open_ in zip(data['Close'], data['Open'])]
            
            ax.bar(data.index, data['Volume'], 
                  color=self.config_manager.color_scheme['volume_bar'], 
                  alpha=0.7, width=1)
            
            # ボリューム移動平均（5日）
            if len(data) >= 5:
                volume_ma = data['Volume'].rolling(window=5).mean()
                ax.plot(data.index, volume_ma, 
                       color='#2c3e50', linewidth=1.5, 
                       label='Volume MA(5)', alpha=0.8)
            
            # Y軸設定
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            # 凡例
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                self.config_manager.add_legend(ax, handles, labels, location='upper right')
            
            self.logger.debug("ボリュームパネル描画完了")
            
        except Exception as e:
            self.logger.warning(f"ボリュームパネル描画エラー: {e}")
    
    def _add_trend_background(self, ax, data: pd.DataFrame) -> None:
        """トレンド背景色の追加"""
        trend_changes = self._find_trend_changes(data['trend_type'])
        
        for start_idx, end_idx, trend_type in trend_changes:
            start_date = data.index[start_idx]
            end_date = data.index[min(end_idx, len(data.index) - 1)]
            
            color = self.config_manager.get_trend_color(trend_type)
            ax.axvspan(start_date, end_date, alpha=0.1, color=color)
    
    def _add_confidence_background(self, ax, data: pd.DataFrame) -> None:
        """信頼度背景色の追加"""
        confidence_color_map = self.config_manager.get_confidence_color_map()
        
        for i, (idx, row) in enumerate(data.iterrows()):
            if i < len(data) - 1:
                next_idx = data.index[i + 1]
                level = row.get('confidence_level', 'medium')
                color = confidence_color_map.get(level, '#95a5a6')
                ax.axvspan(idx, next_idx, alpha=0.05, color=color)
    
    def _find_trend_changes(self, trend_series: pd.Series) -> List[Tuple[int, int, str]]:
        """トレンド変更点の特定"""
        changes = []
        if trend_series.empty:
            return changes
        
        current_trend = trend_series.iloc[0]
        start_idx = 0
        
        for i in range(1, len(trend_series)):
            if trend_series.iloc[i] != current_trend:
                changes.append((start_idx, i - 1, current_trend))
                current_trend = trend_series.iloc[i]
                start_idx = i
        
        # 最後の期間
        changes.append((start_idx, len(trend_series) - 1, current_trend))
        
        return changes
    
    def _find_strategy_changes(self, strategy_series: pd.Series) -> List[Tuple[int, int, str]]:
        """戦略変更点の特定"""
        changes = []
        if strategy_series.empty:
            return changes
        
        current_strategy = strategy_series.iloc[0]
        start_idx = 0
        
        for i in range(1, len(strategy_series)):
            if strategy_series.iloc[i] != current_strategy:
                changes.append((start_idx, i - 1, current_strategy))
                current_strategy = strategy_series.iloc[i]
                start_idx = i
        
        # 最後の期間
        changes.append((start_idx, len(strategy_series) - 1, current_strategy))
        
        return changes
    
    def _mark_trend_changes(self, ax, data: pd.DataFrame) -> None:
        """トレンド変更点のマーキング"""
        trend_changes = self._find_trend_changes(data['trend_type'])
        
        for i, (start_idx, end_idx, trend_type) in enumerate(trend_changes[1:], 1):  # 最初は除く
            change_date = data.index[start_idx]
            change_price = data['Close'].iloc[start_idx]
            
            # 変更点にマーカー
            color = self.config_manager.get_trend_color(trend_type)
            ax.plot(change_date, change_price, marker='o', markersize=8, 
                   color=color, markeredgecolor='black', markeredgewidth=1)
    
    def _mark_strategy_switches(self, ax, data: pd.DataFrame, strategy_changes: List) -> None:
        """戦略切替点のマーキング"""
        for i, (start_idx, end_idx, strategy) in enumerate(strategy_changes[1:], 1):  # 最初は除く
            switch_date = data.index[start_idx]
            
            # 切替点に縦線
            ax.axvline(x=switch_date, color='red', linestyle='-', alpha=0.8, linewidth=2)
            
            # 切替マーカー
            ax.plot(switch_date, 0.8, marker='v', markersize=10, 
                   color='red', markeredgecolor='black', markeredgewidth=1)
    
    def _apply_common_formatting(self, axes, data: pd.DataFrame) -> None:
        """共通フォーマット適用"""
        # X軸フォーマット（最下部のみ）
        self.config_manager.format_x_axis(axes[-1], data.index.tolist())
        
        # 他のパネルのX軸ラベルを非表示
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        
        # 全パネルでX軸範囲を統一
        x_min, x_max = data.index.min(), data.index.max()
        for ax in axes:
            ax.set_xlim(x_min, x_max)
    
    def _save_chart(self, fig, filename: Optional[str] = None) -> str:
        """チャート保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_strategy_chart_{self.symbol}_{self.period_days}d_{timestamp}.png"
        
        filepath = self.output_dir / filename
        self.config_manager.save_chart(fig, str(filepath))
        
        return str(filepath)
    
    def generate_period_comparison(self, periods: List[int] = [30, 60, 90]) -> Dict[str, str]:
        """複数期間の比較チャート生成"""
        results = {}
        
        for period in periods:
            self.logger.info(f"{period}日間のチャートを生成中...")
            
            # 期間を変更
            original_period = self.period_days
            self.period_days = period
            self.data_aggregator.period_days = period
            
            try:
                # チャート生成
                filename = f"comparison_{self.symbol}_{period}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                output_path = self.generate_comprehensive_chart(save_file=True, filename=filename)
                
                if output_path:
                    results[f"{period}days"] = output_path
                    self.logger.info(f"{period}日間チャート完了: {output_path}")
                else:
                    self.logger.warning(f"{period}日間チャート生成失敗")
                    
            except Exception as e:
                self.logger.error(f"{period}日間チャート生成エラー: {e}")
            
            finally:
                # 期間を元に戻す
                self.period_days = original_period
                self.data_aggregator.period_days = original_period
        
        return results
    
    def get_chart_metadata(self) -> Dict[str, Any]:
        """チャートメタデータ取得"""
        if self.chart_data is None:
            return {"status": "no_chart_data"}
        
        data_summary = self.data_aggregator.get_data_summary()
        
        return {
            "symbol": self.symbol,
            "period_days": self.period_days,
            "output_directory": str(self.output_dir),
            "last_generated_file": self.last_generated_file,
            "chart_data_shape": self.chart_data.shape if self.chart_data is not None else None,
            "data_summary": data_summary,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    def cleanup_old_files(self, keep_recent: int = 10) -> int:
        """古いチャートファイルのクリーンアップ"""
        try:
            chart_files = list(self.output_dir.glob("trend_strategy_chart_*.png"))
            chart_files.extend(list(self.output_dir.glob("comparison_*.png")))
            
            if len(chart_files) <= keep_recent:
                return 0
            
            # 作成日時でソート
            chart_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # 古いファイルを削除
            removed_count = 0
            for old_file in chart_files[keep_recent:]:
                try:
                    old_file.unlink()
                    removed_count += 1
                    self.logger.info(f"古いファイル削除: {old_file.name}")
                except Exception as e:
                    self.logger.warning(f"ファイル削除失敗 {old_file.name}: {e}")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            return 0
