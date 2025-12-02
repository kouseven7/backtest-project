"""
戦略相関ダッシュボード - 4-3-3システム統合ダッシュボード

このモジュールは戦略間相関分析の結果を既存の4-3-2ダッシュボードシステムと
統合し、包括的な分析ダッシュボードを提供する。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 既存システムとの統合
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 4-3-3システム
    from config.correlation.strategy_correlation_analyzer import (
        StrategyCorrelationAnalyzer, CorrelationMatrix, CorrelationConfig
    )
    from config.correlation.correlation_matrix_visualizer import CorrelationMatrixVisualizer
    
    # 既存システム
    from config.strategy_scoring_model import StrategyScoreManager
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.strategy_selector import StrategySelector
    
    # 4-3-2システム（ダッシュボード）
    existing_dashboard_available = True
    try:
        from visualization.strategy_performance_dashboard import StrategyPerformanceDashboard  # type: ignore
    except ImportError:
        existing_dashboard_available = False
    
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # フォールバック
    StrategyCorrelationAnalyzer = None  # type: ignore
    CorrelationMatrix = None  # type: ignore
    CorrelationConfig = None  # type: ignore
    CorrelationMatrixVisualizer = None  # type: ignore
    existing_dashboard_available = False

logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Yu Gothic', 'Hiragino Sans', 'Noto Sans CJK JP', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class StrategyCorrelationDashboard:
    """戦略相関統合ダッシュボードメインクラス"""
    
    def __init__(self, config: Optional[CorrelationConfig] = None):  # type: ignore
        self.config = config or (CorrelationConfig() if CorrelationConfig else None)  # type: ignore
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # システムコンポーネント初期化
        try:
            if StrategyCorrelationAnalyzer:
                self.correlation_analyzer = StrategyCorrelationAnalyzer(self.config)  # type: ignore
            else:
                self.correlation_analyzer = None
                
            if CorrelationMatrixVisualizer:
                self.visualizer = CorrelationMatrixVisualizer()  # type: ignore
            else:
                self.visualizer = None
                
            # 既存システム統合
            self.score_manager = None
            self.portfolio_calculator = None
            self.strategy_selector = None
            
            try:
                self.score_manager = StrategyScoreManager()
                self.portfolio_calculator = PortfolioWeightCalculator()
                self.strategy_selector = StrategySelector()
            except Exception as e:
                self.logger.warning(f"既存システム統合に問題があります: {e}")
            
        except Exception as e:
            self.logger.error(f"ダッシュボード初期化エラー: {e}")
            
        # データストレージ
        self.correlation_history: List[CorrelationMatrix] = []  # type: ignore
        self.strategy_data: Dict[str, Any] = {}
        self.dashboard_config = {
            'figure_size': (16, 12),
            'subplot_spacing': 0.3,
            'color_scheme': 'viridis',
            'font_size_base': 10
        }
    
    def add_strategy_performance(self, strategy_name: str, price_data: pd.DataFrame, 
                               signals: pd.Series) -> None:  # type: ignore
        """戦略パフォーマンスデータを追加"""
        try:
            if self.correlation_analyzer:
                self.correlation_analyzer.add_strategy_data(strategy_name, price_data, signals)
                self.logger.info(f"戦略パフォーマンス追加: {strategy_name}")
            else:
                self.logger.warning("相関アナライザーが利用できません")
                
        except Exception as e:
            self.logger.error(f"戦略パフォーマンス追加エラー ({strategy_name}): {e}")
    
    def calculate_correlation_analysis(self, strategies: Optional[List[str]] = None) -> Optional[CorrelationMatrix]:  # type: ignore
        """相関分析を実行"""
        try:
            if not self.correlation_analyzer:
                self.logger.error("相関アナライザーが利用できません")
                return None
                
            correlation_result = self.correlation_analyzer.calculate_correlation_matrix(strategies)
            self.correlation_history.append(correlation_result)
            
            self.logger.info(f"相関分析実行完了: {len(strategies) if strategies else '全戦略'}")
            return correlation_result
            
        except Exception as e:
            self.logger.error(f"相関分析エラー: {e}")
            return None
    
    def create_integrated_dashboard(self, correlation_matrix: CorrelationMatrix,  # type: ignore
                                  save_path: Optional[Union[str, Path]] = None,
                                  include_performance_metrics: bool = True,
                                  include_risk_analysis: bool = True) -> plt.Figure:  # type: ignore
        """統合ダッシュボードを作成"""
        try:
            # フィギュア設定
            fig = plt.figure(figsize=self.dashboard_config['figure_size'])
            gs = fig.add_gridspec(3, 4, hspace=self.dashboard_config['subplot_spacing'], 
                                wspace=self.dashboard_config['subplot_spacing'])
            
            # 1. 相関ヒートマップ（左上、2x2）
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            self._plot_correlation_section(ax1, correlation_matrix)
            
            # 2. 共分散分析（右上）
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_covariance_section(ax2, correlation_matrix)
            
            # 3. リスク分析（右中）
            ax3 = fig.add_subplot(gs[1, 2:])
            if include_risk_analysis:
                self._plot_risk_analysis_section(ax3, correlation_matrix)
            else:
                ax3.text(0.5, 0.5, 'リスク分析\n（データ準備中）', ha='center', va='center')
                ax3.set_title('リスク分析')
            
            # 4. パフォーマンス指標（下段左）
            ax4 = fig.add_subplot(gs[2, 0:2])
            if include_performance_metrics:
                self._plot_performance_metrics_section(ax4, correlation_matrix)
            else:
                ax4.text(0.5, 0.5, 'パフォーマンス指標\n（データ準備中）', ha='center', va='center')
                ax4.set_title('パフォーマンス指標')
            
            # 5. 相関統計（下段右）
            ax5 = fig.add_subplot(gs[2, 2:])
            self._plot_correlation_statistics_section(ax5, correlation_matrix)
            
            # ダッシュボードタイトル
            fig.suptitle('戦略間相関・共分散分析ダッシュボード（4-3-3）\n' + 
                        f'計算日時: {correlation_matrix.calculation_timestamp.strftime("%Y-%m-%d %H:%M")}',  # type: ignore
                        fontsize=16, fontweight='bold', y=0.95)
            
            # 保存
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')  # type: ignore
                self.logger.info(f"統合ダッシュボード保存: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"統合ダッシュボード作成エラー: {e}")
            raise
    
    def _plot_correlation_section(self, ax: Any, correlation_matrix: CorrelationMatrix) -> None:  # type: ignore
        """相関セクションをプロット"""
        try:
            import seaborn as sns
            
            # ヒートマップ作成
            mask = np.triu(np.ones_like(correlation_matrix.correlation_matrix, dtype=bool))  # type: ignore
            
            sns.heatmap(
                correlation_matrix.correlation_matrix,  # type: ignore
                mask=mask,
                annot=True,
                cmap='RdYlBu_r',
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .7, "label": "相関係数"},
                fmt='.3f',
                ax=ax,
                annot_kws={'size': 8}
            )
            
            ax.set_title('戦略間相関行列', fontsize=12, fontweight='bold')
            ax.set_xlabel('戦略', fontsize=10)
            ax.set_ylabel('戦略', fontsize=10)
            
        except ImportError:
            # seabornが利用できない場合のフォールバック
            im = ax.imshow(correlation_matrix.correlation_matrix.values, cmap='RdYlBu_r', vmin=-1, vmax=1)  # type: ignore
            ax.set_title('戦略間相関行列', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(correlation_matrix.correlation_matrix.columns)))  # type: ignore
            ax.set_yticks(range(len(correlation_matrix.correlation_matrix.index)))  # type: ignore
            ax.set_xticklabels(correlation_matrix.correlation_matrix.columns, rotation=45)  # type: ignore
            ax.set_yticklabels(correlation_matrix.correlation_matrix.index)  # type: ignore
            plt.colorbar(im, ax=ax, label='相関係数')
            
        except Exception as e:
            self.logger.error(f"相関セクションプロットエラー: {e}")
            ax.text(0.5, 0.5, f'相関プロットエラー: {str(e)[:50]}...', ha='center', va='center')
    
    def _plot_covariance_section(self, ax: Any, correlation_matrix: CorrelationMatrix) -> None:  # type: ignore
        """共分散セクションをプロット"""
        try:
            # 共分散行列の対角成分（分散）をバープロット
            variances = np.diag(correlation_matrix.covariance_matrix.values)  # type: ignore
            strategies = correlation_matrix.correlation_matrix.index.tolist()  # type: ignore
            
            bars = ax.bar(strategies, variances, color='skyblue', alpha=0.7, edgecolor='navy')
            ax.set_title('戦略別分散（年率）', fontsize=12, fontweight='bold')
            ax.set_ylabel('分散', fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar, value in zip(bars, variances):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
                       
        except Exception as e:
            self.logger.error(f"共分散セクションプロットエラー: {e}")
            ax.text(0.5, 0.5, f'共分散プロットエラー: {str(e)[:50]}...', ha='center', va='center')
    
    def _plot_risk_analysis_section(self, ax: Any, correlation_matrix: CorrelationMatrix) -> None:  # type: ignore
        """リスク分析セクションをプロット"""
        try:
            # ポートフォリオリスクの計算例
            # 等重量ポートフォリオの場合のリスク
            n_strategies = len(correlation_matrix.correlation_matrix)  # type: ignore
            equal_weights = np.ones(n_strategies) / n_strategies
            
            # ポートフォリオ分散 = w' * Σ * w
            portfolio_variance = np.dot(equal_weights.T, 
                                      np.dot(correlation_matrix.covariance_matrix.values, equal_weights))  # type: ignore
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # 個別戦略のボラティリティ
            individual_volatilities = np.sqrt(np.diag(correlation_matrix.covariance_matrix.values))  # type: ignore
            
            # 分散効果の可視化
            strategies = correlation_matrix.correlation_matrix.index.tolist()  # type: ignore
            x_pos = np.arange(len(strategies))
            
            ax.bar(x_pos, individual_volatilities, alpha=0.6, label='個別ボラティリティ', color='red')
            ax.axhline(y=portfolio_volatility, color='blue', linestyle='--', linewidth=2, 
                      label=f'ポートフォリオボラティリティ: {portfolio_volatility:.4f}')
            
            ax.set_title('リスク分散効果', fontsize=12, fontweight='bold')
            ax.set_ylabel('ボラティリティ（年率）', fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(strategies, rotation=45, fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"リスク分析セクションプロットエラー: {e}")
            ax.text(0.5, 0.5, f'リスク分析エラー: {str(e)[:50]}...', ha='center', va='center')
    
    def _plot_performance_metrics_section(self, ax: Any, correlation_matrix: CorrelationMatrix) -> None:  # type: ignore
        """パフォーマンス指標セクションをプロット"""
        try:
            # 相関アナライザーからパフォーマンスデータを取得
            if not self.correlation_analyzer:
                ax.text(0.5, 0.5, 'パフォーマンスデータなし', ha='center', va='center')
                return
            
            strategy_data = self.correlation_analyzer.strategy_data
            if not strategy_data:
                ax.text(0.5, 0.5, 'パフォーマンスデータなし', ha='center', va='center')
                return
            
            # シャープレシオの比較
            strategies = list(strategy_data.keys())
            sharpe_ratios = [strategy_data[s].sharpe_ratio for s in strategies]
            
            bars = ax.bar(strategies, sharpe_ratios, color='green', alpha=0.7)
            ax.set_title('戦略別シャープレシオ', fontsize=12, fontweight='bold')
            ax.set_ylabel('シャープレシオ', fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar, value in zip(bars, sharpe_ratios):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
        except Exception as e:
            self.logger.error(f"パフォーマンス指標セクションエラー: {e}")
            ax.text(0.5, 0.5, f'パフォーマンス指標エラー: {str(e)[:50]}...', ha='center', va='center')
    
    def _plot_correlation_statistics_section(self, ax: Any, correlation_matrix: CorrelationMatrix) -> None:  # type: ignore
        """相関統計セクションをプロット"""
        try:
            # 相関係数の分布統計
            corr_mat = correlation_matrix.correlation_matrix  # type: ignore
            mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            correlations = corr_mat.where(mask).stack()  # type: ignore
            
            # 統計情報
            stats = {
                '平均': correlations.mean(),  # type: ignore
                '中央値': correlations.median(),  # type: ignore
                '標準偏差': correlations.std(),  # type: ignore
                '最小値': correlations.min(),  # type: ignore
                '最大値': correlations.max()  # type: ignore
            }
            
            # テーブル形式で表示
            ax.axis('tight')
            ax.axis('off')
            
            table_data = [[k, f'{v:.4f}'] for k, v in stats.items()]
            table = ax.table(cellText=table_data,
                            colLabels=['統計', '値'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # ヘッダーのスタイル
            for i in range(2):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax.set_title('相関統計サマリー', fontsize=12, fontweight='bold')
            
        except Exception as e:
            self.logger.error(f"相関統計セクションエラー: {e}")
            ax.text(0.5, 0.5, f'相関統計エラー: {str(e)[:50]}...', ha='center', va='center')
    
    def create_correlation_report(self, output_dir: Union[str, Path] = "correlation_dashboard_report") -> Dict[str, Path]:
        """包括的な相関レポートを作成"""
        try:
            if not self.correlation_history:
                raise ValueError("相関分析結果がありません。先に分析を実行してください。")
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            created_files = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            latest_correlation = self.correlation_history[-1]
            
            # 1. 統合ダッシュボード
            dashboard_path = output_dir / f"correlation_dashboard_{timestamp}.png"
            dashboard_fig = self.create_integrated_dashboard(
                latest_correlation, save_path=dashboard_path
            )
            created_files["dashboard"] = dashboard_path
            plt.close(dashboard_fig)
            
            # 2. 詳細視覚化（visualizerがある場合）
            if self.visualizer:
                viz_files = self.visualizer.create_correlation_report(  # type: ignore
                    latest_correlation, 
                    analyzer=self.correlation_analyzer,
                    output_dir=output_dir / "detailed_analysis",
                    include_network=True,
                    include_rolling=True
                )
                created_files.update(viz_files)
            
            # 3. システム統合レポート
            integration_path = output_dir / f"system_integration_report_{timestamp}.txt"
            self._create_integration_report(integration_path, latest_correlation)
            created_files["integration_report"] = integration_path
            
            self.logger.info(f"包括的相関レポート作成完了: {output_dir}")
            return created_files
            
        except Exception as e:
            self.logger.error(f"相関レポート作成エラー: {e}")
            raise
    
    def _create_integration_report(self, output_path: Path, correlation_matrix: CorrelationMatrix) -> None:  # type: ignore
        """システム統合レポートを作成"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("4-3-3 戦略間相関・共分散分析システム 統合レポート\n")
                f.write("=" * 80 + "\n\n")
                
                # システム情報
                f.write("システム情報:\n")
                f.write("-" * 50 + "\n")
                f.write(f"レポート作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"相関分析実行日時: {correlation_matrix.calculation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")  # type: ignore
                f.write(f"分析期間: {correlation_matrix.period_info.get('start_date', 'N/A')} - "  # type: ignore
                       f"{correlation_matrix.period_info.get('end_date', 'N/A')}\n")  # type: ignore
                f.write(f"対象戦略数: {correlation_matrix.period_info.get('strategies_count', 'N/A')}\n")  # type: ignore
                f.write(f"データ期間: {correlation_matrix.period_info.get('total_periods', 'N/A')}期間\n\n")  # type: ignore
                
                # 既存システム統合状況
                f.write("既存システム統合状況:\n")
                f.write("-" * 50 + "\n")
                f.write(f"戦略スコアマネージャー: {'利用可能' if self.score_manager else '利用不可'}\n")
                f.write(f"ポートフォリオ計算機: {'利用可能' if self.portfolio_calculator else '利用不可'}\n")
                f.write(f"戦略セレクター: {'利用可能' if self.strategy_selector else '利用不可'}\n")
                f.write(f"4-3-2ダッシュボード: {'利用可能' if existing_dashboard_available else '利用不可'}\n\n")
                
                # 相関分析結果サマリー
                if self.correlation_analyzer:
                    try:
                        summary = self.correlation_analyzer.get_correlation_summary(correlation_matrix)  # type: ignore
                        f.write("相関分析結果サマリー:\n")
                        f.write("-" * 50 + "\n")
                        for key, value in summary.items():  # type: ignore
                            f.write(f"{key}: {value}\n")  # type: ignore
                        f.write("\n")
                    except Exception as e:
                        f.write(f"相関サマリー取得エラー: {e}\n\n")
                
                # パフォーマンス情報
                if self.correlation_analyzer and self.correlation_analyzer.strategy_data:
                    f.write("戦略パフォーマンス情報:\n")
                    f.write("-" * 50 + "\n")
                    for strategy_name, strategy_data in self.correlation_analyzer.strategy_data.items():
                        f.write(f"\n{strategy_name}:\n")
                        f.write(f"  ボラティリティ: {strategy_data.volatility:.4f}\n")
                        f.write(f"  シャープレシオ: {strategy_data.sharpe_ratio:.4f}\n")
                        f.write(f"  最大ドローダウン: {strategy_data.max_drawdown:.4f}\n")
                        f.write(f"  勝率: {strategy_data.win_rate:.4f}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("レポート終了\n")
            
            self.logger.info(f"システム統合レポート作成: {output_path}")
            
        except Exception as e:
            self.logger.error(f"システム統合レポート作成エラー: {e}")

def create_sample_dashboard():
    """サンプルダッシュボードの実行"""
    try:
        print("4-3-3 サンプルダッシュボードを作成中...")
        
        # ダミーデータ作成
        np.random.seed(42)
        strategies = ['Strategy_A', 'Strategy_B', 'Strategy_C', 'Strategy_D']
        n_periods = 252
        dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')
        
        # 価格データ
        price_data = pd.DataFrame({
            'close': 100 * np.cumprod(1 + np.random.normal(0, 0.01, n_periods))
        }, index=dates)
        
        # 戦略シグナル
        signals_data = {}
        for strategy in strategies:
            signals_data[strategy] = pd.Series(
                np.random.choice([-1, 0, 1], n_periods, p=[0.3, 0.4, 0.3]), 
                index=dates
            )
        
        # ダッシュボード初期化
        dashboard = StrategyCorrelationDashboard()
        
        # 戦略データ追加
        for strategy_name, signals in signals_data.items():
            dashboard.add_strategy_performance(strategy_name, price_data, signals)
        
        # 相関分析実行
        correlation_result = dashboard.calculate_correlation_analysis()
        
        if correlation_result:
            # ダッシュボード作成
            fig = dashboard.create_integrated_dashboard(correlation_result)
            plt.show()
            
            # レポート作成
            report_files = dashboard.create_correlation_report("sample_report")
            print(f"\nレポートファイル作成:")
            for key, path in report_files.items():
                print(f"  {key}: {path}")
        
        print("サンプルダッシュボード作成完了")
        
    except Exception as e:
        logger.error(f"サンプルダッシュボード作成エラー: {e}")
        raise

if __name__ == "__main__":
    # 基本的なテスト
    logging.basicConfig(level=logging.INFO)
    
    print("戦略相関ダッシュボード - テスト実行")
    create_sample_dashboard()
