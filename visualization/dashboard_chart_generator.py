"""
Dashboard Chart Generator for 4-3-2
戦略比率・パフォーマンス表示用チャート生成システム

4-3-1のChartConfigManagerを拡張してダッシュボード用チャートを生成
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

# 4-3-1システムを拡張
try:
    from .chart_config import ChartConfigManager
    from .performance_data_collector import PerformanceSnapshot
except ImportError:
    from chart_config import ChartConfigManager
    from performance_data_collector import PerformanceSnapshot

logger = logging.getLogger(__name__)

class DashboardChartGenerator:
    """ダッシュボードチャート生成器"""
    
    def __init__(self, output_dir: str = "logs/dashboard/chart_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 4-3-1の設定管理を拡張
        self.config_manager = ChartConfigManager()
        self._extend_config_for_dashboard()
        
        logger.info("DashboardChartGenerator initialized")
    
    def _extend_config_for_dashboard(self):
        """ダッシュボード用設定の拡張"""
        # ダッシュボード用色設定
        self.config_manager.dashboard_colors = {
            'performance_positive': '#27ae60',  # Green
            'performance_negative': '#e74c3c',  # Red
            'performance_neutral': '#95a5a6',   # Gray
            'risk_low': '#2ecc71',              # Light Green
            'risk_medium': '#f39c12',           # Orange
            'risk_high': '#e74c3c',             # Red
            'allocation_primary': '#3498db',     # Blue
            'allocation_secondary': '#9b59b6',   # Purple
            'background_panel': '#ecf0f1'        # Light Gray
        }
        
        # ダッシュボード用フォント設定
        self.config_manager.dashboard_fonts = {
            'title_size': 16,
            'subtitle_size': 12,
            'metric_size': 14,
            'label_size': 10
        }
    
    def generate_performance_dashboard(self, 
                                     current_snapshot: PerformanceSnapshot,
                                     historical_snapshots: List[PerformanceSnapshot],
                                     save_file: bool = True) -> Optional[str]:
        """総合パフォーマンスダッシュボードの生成"""
        try:
            self.config_manager.setup_matplotlib_style()
            
            # 4パネルレイアウトの作成
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
            fig.suptitle(f'戦略パフォーマンスダッシュボード - {current_snapshot.ticker}', 
                        fontsize=self.config_manager.dashboard_fonts['title_size'], 
                        fontweight='bold', y=0.95)
            
            # パネル1: 戦略配分円グラフ
            self._draw_strategy_allocation_pie(axes[0,0], current_snapshot)
            
            # パネル2: パフォーマンス時系列
            self._draw_performance_timeseries(axes[0,1], historical_snapshots)
            
            # パネル3: リスク指標ゲージ
            self._draw_risk_gauges(axes[1,0], current_snapshot)
            
            # パネル4: 主要メトリクス表示
            self._draw_key_metrics(axes[1,1], current_snapshot)
            
            plt.tight_layout()
            
            if save_file:
                timestamp = current_snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
                filename = f"performance_dashboard_{current_snapshot.ticker}_{timestamp}.png"
                filepath = self.output_dir / filename
                
                self.config_manager.save_chart(fig, str(filepath))
                logger.info(f"Dashboard saved: {filename}")
                return str(filepath)
            
            return "dashboard_generated"
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return None
        finally:
            if 'fig' in locals():
                plt.close(fig)
    
    def _draw_strategy_allocation_pie(self, ax, snapshot: PerformanceSnapshot):
        """戦略配分円グラフの描画"""
        try:
            ax.set_title('戦略配分', fontsize=self.config_manager.dashboard_fonts['subtitle_size'], fontweight='bold')
            
            allocations = snapshot.strategy_allocations
            if not allocations:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
                return
            
            # データ準備
            labels = list(allocations.keys())
            sizes = list(allocations.values())
            
            # 色の設定（戦略タイプ別）
            colors = []
            for label in labels:
                if hasattr(self.config_manager, 'strategy_colors') and label.lower() in [k.lower() for k in self.config_manager.strategy_colors.keys()]:
                    # 既存の戦略色を使用
                    for k, v in self.config_manager.strategy_colors.items():
                        if k.lower() == label.lower():
                            colors.append(v)
                            break
                    else:
                        colors.append(self.config_manager.dashboard_colors['allocation_primary'])
                else:
                    colors.append(self.config_manager.dashboard_colors['allocation_primary'])
            
            # 円グラフの描画
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 9}
            )
            
            # レイアウト調整
            ax.axis('equal')
            
            # 凡例の追加（サイズによって調整）
            if len(labels) <= 6:
                ax.legend(wedges, [f"{label}: {size:.1%}" for label, size in zip(labels, sizes)],
                         loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
            
        except Exception as e:
            logger.warning(f"Strategy allocation pie chart error: {e}")
            ax.text(0.5, 0.5, f'エラー: チャート生成失敗', ha='center', va='center', transform=ax.transAxes)
    
    def _draw_performance_timeseries(self, ax, historical_snapshots: List[PerformanceSnapshot]):
        """パフォーマンス時系列グラフの描画"""
        try:
            ax.set_title('パフォーマンス推移 (30日)', fontsize=self.config_manager.dashboard_fonts['subtitle_size'], fontweight='bold')
            
            if not historical_snapshots:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
                return
            
            # データ準備
            timestamps = [s.timestamp for s in historical_snapshots]
            returns = [s.total_performance.get('portfolio_return', 0) for s in historical_snapshots]
            risks = [s.total_performance.get('portfolio_risk', 0) for s in historical_snapshots]
            
            if not timestamps:
                ax.text(0.5, 0.5, '有効なデータなし', ha='center', va='center', transform=ax.transAxes)
                return
            
            # 時系列プロット
            ax2 = ax.twinx()
            
            line1 = ax.plot(timestamps, returns, 'g-', linewidth=2, label='ポートフォリオリターン (%)', alpha=0.8)
            line2 = ax2.plot(timestamps, risks, 'r--', linewidth=1.5, label='リスク (%)', alpha=0.6)
            
            # 軸設定
            ax.set_ylabel('リターン (%)', color='g', fontweight='bold')
            ax2.set_ylabel('リスク (%)', color='r', fontweight='bold')
            ax.tick_params(axis='y', labelcolor='g')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # グリッドとフォーマット
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # 凡例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=8)
            
        except Exception as e:
            logger.warning(f"Performance timeseries error: {e}")
            ax.text(0.5, 0.5, f'エラー: 時系列グラフ生成失敗', ha='center', va='center', transform=ax.transAxes)
    
    def _draw_risk_gauges(self, ax, snapshot: PerformanceSnapshot):
        """リスク指標ゲージの描画"""
        try:
            ax.set_title('リスク指標', fontsize=self.config_manager.dashboard_fonts['subtitle_size'], fontweight='bold')
            
            risk_metrics = snapshot.risk_metrics
            if not risk_metrics:
                ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
                return
            
            # 主要リスク指標の抽出
            risk_score = risk_metrics.get('risk_score', 50)
            var_95 = risk_metrics.get('var_95', 10)
            concentration_risk = risk_metrics.get('concentration_risk', 25)
            
            # ゲージチャートの描画（簡易版）
            risks = [
                ('リスクスコア', risk_score, '%', 100),
                ('VaR (95%)', var_95, '%', 20),
                ('集中度リスク', concentration_risk, '%', 100)
            ]
            
            y_positions = [0.7, 0.5, 0.3]
            
            for i, (name, value, unit, max_val) in enumerate(risks):
                y_pos = y_positions[i]
                
                # プログレスバー風の描画
                bar_width = 0.6
                bar_height = 0.08
                
                # 背景バー
                rect_bg = Rectangle((0.2, y_pos - bar_height/2), bar_width, bar_height, 
                                   facecolor='lightgray', alpha=0.5)
                ax.add_patch(rect_bg)
                
                # 値バー
                fill_ratio = min(value / max_val, 1.0)
                color = self._get_risk_color(value, max_val)
                rect_fill = Rectangle((0.2, y_pos - bar_height/2), bar_width * fill_ratio, bar_height, 
                                     facecolor=color, alpha=0.8)
                ax.add_patch(rect_fill)
                
                # ラベルと値
                ax.text(0.05, y_pos, name, va='center', fontsize=10, fontweight='bold')
                ax.text(0.85, y_pos, f'{value:.1f}{unit}', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
        except Exception as e:
            logger.warning(f"Risk gauges error: {e}")
            ax.text(0.5, 0.5, f'エラー: リスクゲージ生成失敗', ha='center', va='center', transform=ax.transAxes)
    
    def _draw_key_metrics(self, ax, snapshot: PerformanceSnapshot):
        """主要メトリクス表示の描画"""
        try:
            ax.set_title('主要指標', fontsize=self.config_manager.dashboard_fonts['subtitle_size'], fontweight='bold')
            
            performance = snapshot.total_performance
            
            # メトリクス準備
            metrics = [
                ('ポートフォリオリターン', f"{performance.get('portfolio_return', 0):.2f}%", 'return'),
                ('シャープレシオ', f"{performance.get('sharpe_ratio', 0):.3f}", 'sharpe'),
                ('分散化比率', f"{performance.get('diversification_ratio', 0):.3f}", 'diversification'),
                ('戦略数', f"{len(snapshot.strategy_allocations)}", 'count')
            ]
            
            # アラート情報
            alert_count = len(snapshot.alerts)
            if alert_count > 0:
                metrics.append(('アラート', f"{alert_count}件", 'alert'))
            
            # 表形式で描画
            y_positions = np.linspace(0.8, 0.2, len(metrics))
            
            for i, (name, value, metric_type) in enumerate(metrics):
                y_pos = y_positions[i]
                
                # メトリクス名
                ax.text(0.05, y_pos, name, fontsize=11, fontweight='bold', va='center')
                
                # 値（色付き）
                color = self._get_metric_color(metric_type, value)
                ax.text(0.7, y_pos, value, fontsize=12, fontweight='bold', 
                       va='center', color=color)
            
            # アラート詳細表示
            if snapshot.alerts:
                alert_text = '\n'.join([f"• {alert}" for alert in snapshot.alerts[:3]])
                ax.text(0.05, 0.05, f"アラート詳細:\n{alert_text}", 
                       fontsize=8, va='bottom', color='red', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
        except Exception as e:
            logger.warning(f"Key metrics error: {e}")
            ax.text(0.5, 0.5, f'エラー: メトリクス表示失敗', ha='center', va='center', transform=ax.transAxes)
    
    def _get_risk_color(self, value: float, max_val: float) -> str:
        """リスク値に応じた色を取得"""
        ratio = value / max_val if max_val > 0 else 0
        
        if ratio < 0.3:
            return self.config_manager.dashboard_colors['risk_low']
        elif ratio < 0.7:
            return self.config_manager.dashboard_colors['risk_medium']
        else:
            return self.config_manager.dashboard_colors['risk_high']
    
    def _get_metric_color(self, metric_type: str, value_str: str) -> str:
        """メトリクスタイプに応じた色を取得"""
        if metric_type == 'alert':
            return 'red'
        elif metric_type == 'return':
            try:
                val = float(value_str.replace('%', ''))
                return 'green' if val > 0 else 'red' if val < 0 else 'gray'
            except:
                return 'black'
        elif metric_type == 'sharpe':
            try:
                val = float(value_str)
                return 'green' if val > 1.0 else 'orange' if val > 0.5 else 'red'
            except:
                return 'black'
        else:
            return 'black'
    
    def generate_simple_summary(self, snapshot: PerformanceSnapshot) -> str:
        """簡単なテキストサマリーの生成"""
        try:
            allocation_count = len(snapshot.strategy_allocations)
            total_return = snapshot.total_performance.get('portfolio_return', 0)
            risk_score = snapshot.risk_metrics.get('risk_score', 0)
            alert_count = len(snapshot.alerts)
            
            summary = f"""
=== 戦略パフォーマンスサマリー ({snapshot.ticker}) ===
時刻: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
戦略数: {allocation_count}
ポートフォリオリターン: {total_return:.2f}%
リスクスコア: {risk_score:.1f}/100
アラート: {alert_count}件
トレンド: {snapshot.market_context.get('trend', '不明')}
            """.strip()
            
            if snapshot.alerts:
                summary += f"\n\nアラート内容:\n" + '\n'.join([f"- {alert}" for alert in snapshot.alerts])
            
            return summary
            
        except Exception as e:
            return f"サマリー生成エラー: {str(e)}"
    
    def cleanup_old_charts(self, days: int = 7) -> int:
        """古いチャートファイルのクリーンアップ"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            removed_count = 0
            
            for file in self.output_dir.glob("performance_dashboard_*.png"):
                if file.stat().st_mtime < cutoff_date.timestamp():
                    file.unlink()
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old chart files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Chart cleanup failed: {e}")
            return 0

if __name__ == "__main__":
    # テスト用
    from datetime import datetime
    
    # サンプルスナップショット
    sample_snapshot = PerformanceSnapshot(
        timestamp=datetime.now(),
        ticker="USDJPY",
        strategy_allocations={"VWAPBounce": 0.4, "Momentum": 0.35, "Breakout": 0.25},
        strategy_scores={"VWAPBounce": 0.75, "Momentum": 0.68, "Breakout": 0.72},
        total_performance={"portfolio_return": 5.2, "portfolio_risk": 12.5, "sharpe_ratio": 1.15},
        risk_metrics={"risk_score": 78, "var_95": 8.5, "concentration_risk": 35},
        market_context={"trend": "uptrend", "trend_confidence": 0.8},
        alerts=["集中度リスク: 高"]
    )
    
    # チャート生成テスト
    generator = DashboardChartGenerator()
    result = generator.generate_performance_dashboard(sample_snapshot, [sample_snapshot])
    print(f"Chart generated: {result}")
