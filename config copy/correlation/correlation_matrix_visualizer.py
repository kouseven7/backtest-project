"""
相関行列視覚化システム - 戦略間の相関と共分散を視覚化

このモジュールは、戦略相関分析の結果をヒートマップ、
ネットワーク図、時系列プロットなどで視覚化する。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 既存システムとの統合
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config.correlation.strategy_correlation_analyzer import (
        StrategyCorrelationAnalyzer, CorrelationMatrix, CorrelationConfig
    )
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # フォールバック
    StrategyCorrelationAnalyzer = None  # type: ignore
    CorrelationMatrix = None  # type: ignore
    CorrelationConfig = None  # type: ignore

logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Yu Gothic', 'Hiragino Sans', 'Noto Sans CJK JP', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class CorrelationMatrixVisualizer:
    """相関行列視覚化メインクラス"""
    
    def __init__(self, figsize: Tuple[float, float] = (12, 10), style: str = "whitegrid"):
        self.figsize = figsize
        self.style = style
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # スタイル設定
        sns.set_style(style)
        plt.style.use('default')
    
    def plot_correlation_heatmap(self, correlation_matrix: CorrelationMatrix,
                               title: str = "戦略間相関行列",
                               save_path: Optional[Union[str, Path]] = None,
                               show_values: bool = True,
                               cmap: str = "RdYlBu_r") -> plt.Figure:
        """相関行列のヒートマップを作成"""
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # ヒートマップ作成
            mask = np.zeros_like(correlation_matrix.correlation_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True  # 上三角をマスク
            
            heatmap = sns.heatmap(
                correlation_matrix.correlation_matrix,
                mask=mask,
                annot=show_values,
                cmap=cmap,
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5, "label": "相関係数"},
                fmt='.3f',
                ax=ax
            )
            
            # タイトルと軸ラベル
            ax.set_title(f"{title}\n計算日時: {correlation_matrix.calculation_timestamp.strftime('%Y-%m-%d %H:%M')}", 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("戦略", fontsize=12, fontweight='bold')
            ax.set_ylabel("戦略", fontsize=12, fontweight='bold')
            
            # 軸の回転
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)
            
            # レイアウト調整
            plt.tight_layout()
            
            # 保存
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"相関ヒートマップ保存: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"相関ヒートマップ作成エラー: {e}")
            raise
    
    def plot_covariance_heatmap(self, correlation_matrix: CorrelationMatrix,
                              title: str = "戦略間共分散行列",
                              save_path: Optional[Union[str, Path]] = None,
                              show_values: bool = True,
                              cmap: str = "viridis") -> plt.Figure:
        """共分散行列のヒートマップを作成"""
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # 共分散行列の対称性を利用
            cov_matrix = correlation_matrix.covariance_matrix
            mask = np.zeros_like(cov_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            
            # ヒートマップ作成
            heatmap = sns.heatmap(
                cov_matrix,
                mask=mask,
                annot=show_values,
                cmap=cmap,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5, "label": "共分散"},
                fmt='.4f',
                ax=ax
            )
            
            # タイトルと軸ラベル
            ax.set_title(f"{title}\n計算日時: {correlation_matrix.calculation_timestamp.strftime('%Y-%m-%d %H:%M')}", 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("戦略", fontsize=12, fontweight='bold')
            ax.set_ylabel("戦略", fontsize=12, fontweight='bold')
            
            # 軸の回転
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"共分散ヒートマップ保存: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"共分散ヒートマップ作成エラー: {e}")
            raise
    
    def plot_correlation_network(self, correlation_matrix: CorrelationMatrix,
                               threshold: float = 0.5,
                               title: str = "戦略相関ネットワーク",
                               save_path: Optional[Union[str, Path]] = None,
                               layout: str = "spring") -> plt.Figure:
        """相関ネットワーク図を作成"""
        try:
            try:
                import networkx as nx
            except ImportError:
                self.logger.warning("NetworkXが利用できません。ネットワーク図作成をスキップします")
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.text(0.5, 0.5, 'NetworkXライブラリが必要です', 
                       ha='center', va='center', fontsize=16)
                ax.set_title(title)
                return fig
            
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # ネットワークグラフ作成
            G = nx.Graph()
            corr_mat = correlation_matrix.correlation_matrix
            strategies = corr_mat.index.tolist()
            
            # ノード追加
            G.add_nodes_from(strategies)
            
            # エッジ追加（閾値以上の相関）
            for i in range(len(strategies)):
                for j in range(i+1, len(strategies)):
                    corr_value = abs(corr_mat.iloc[i, j])
                    if corr_value >= threshold:
                        G.add_edge(strategies[i], strategies[j], 
                                 weight=corr_value, 
                                 correlation=corr_mat.iloc[i, j])
            
            # レイアウト計算
            if layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # エッジの描画（相関の強さで太さを変える）
            edges = G.edges(data=True)
            for edge in edges:
                weight = edge[2]['weight']
                correlation = edge[2]['correlation']
                color = 'red' if correlation > 0 else 'blue'
                nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                     width=weight * 5, alpha=0.6, 
                                     edge_color=color, ax=ax)
            
            # ノードの描画
            nx.draw_networkx_nodes(G, pos, node_size=1000, 
                                 node_color='lightgreen', 
                                 alpha=0.8, ax=ax)
            
            # ラベル描画
            nx.draw_networkx_labels(G, pos, font_size=10, 
                                  font_weight='bold', ax=ax)
            
            # タイトルと説明
            ax.set_title(f"{title}\n（閾値: {threshold}以上, 赤:正の相関, 青:負の相関）", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"相関ネットワーク図保存: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"相関ネットワーク図作成エラー: {e}")
            raise
    
    def plot_rolling_correlation(self, analyzer: StrategyCorrelationAnalyzer,  # type: ignore
                               strategy1: str, strategy2: str,
                               window: int = 30,
                               title: Optional[str] = None,
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """ローリング相関の時系列プロット"""
        try:
            # ローリング相関計算
            rolling_corr = analyzer.calculate_rolling_correlation(strategy1, strategy2, window)
            
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # プロット
            ax.plot(rolling_corr.index, rolling_corr.values, linewidth=2, color='blue')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='高相関(0.5)')
            ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='高負相関(-0.5)')
            
            # タイトルと軸ラベル
            if title is None:
                title = f"{strategy1} vs {strategy2} ローリング相関 (窓幅: {window})"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("日付", fontsize=12)
            ax.set_ylabel("相関係数", fontsize=12)
            ax.set_ylim(-1, 1)
            
            # グリッドと凡例
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # x軸の日付フォーマット
            from matplotlib.dates import DateFormatter
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"ローリング相関図保存: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"ローリング相関プロット作成エラー: {e}")
            raise
    
    def plot_correlation_distribution(self, correlation_matrix: CorrelationMatrix,
                                    title: str = "相関係数分布",
                                    save_path: Optional[Union[str, Path]] = None,
                                    bins: int = 20) -> plt.Figure:
        """相関係数の分布ヒストグラム"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 上三角行列から相関係数を抽出（対角線除く）
            corr_mat = correlation_matrix.correlation_matrix
            mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            correlations = corr_mat.where(mask).stack()
            
            # ヒストグラム
            ax1.hist(correlations, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(correlations.mean(), color='red', linestyle='--', 
                       label=f'平均: {correlations.mean():.3f}')
            ax1.axvline(correlations.median(), color='orange', linestyle='--', 
                       label=f'中央値: {correlations.median():.3f}')
            ax1.set_xlabel("相関係数", fontsize=12)
            ax1.set_ylabel("頻度", fontsize=12)
            ax1.set_title("相関係数分布", fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ボックスプロット
            ax2.boxplot(correlations, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
            ax2.set_ylabel("相関係数", fontsize=12)
            ax2.set_title("相関係数ボックスプロット", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 統計情報の表示
            stats_text = f"""統計サマリー:
平均: {correlations.mean():.3f}
中央値: {correlations.median():.3f}
標準偏差: {correlations.std():.3f}
最小値: {correlations.min():.3f}
最大値: {correlations.max():.3f}
サンプル数: {len(correlations)}"""
            
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"相関分布図保存: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"相関分布プロット作成エラー: {e}")
            raise
    
    def create_correlation_report(self, correlation_matrix: CorrelationMatrix,
                                analyzer: Optional[StrategyCorrelationAnalyzer] = None,  # type: ignore
                                output_dir: Union[str, Path] = "correlation_report",
                                include_network: bool = True,
                                include_rolling: bool = True) -> Dict[str, Path]:
        """包括的な相関レポートを作成"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            created_files = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 相関ヒートマップ
            heatmap_path = output_dir / f"correlation_heatmap_{timestamp}.png"
            self.plot_correlation_heatmap(correlation_matrix, save_path=heatmap_path)
            created_files["correlation_heatmap"] = heatmap_path
            
            # 2. 共分散ヒートマップ
            covariance_path = output_dir / f"covariance_heatmap_{timestamp}.png"
            self.plot_covariance_heatmap(correlation_matrix, save_path=covariance_path)
            created_files["covariance_heatmap"] = covariance_path
            
            # 3. 相関分布
            distribution_path = output_dir / f"correlation_distribution_{timestamp}.png"
            self.plot_correlation_distribution(correlation_matrix, save_path=distribution_path)
            created_files["correlation_distribution"] = distribution_path
            
            # 4. ネットワーク図（オプション）
            if include_network:
                network_path = output_dir / f"correlation_network_{timestamp}.png"
                self.plot_correlation_network(correlation_matrix, save_path=network_path)
                created_files["correlation_network"] = network_path
            
            # 5. ローリング相関（オプション）
            if include_rolling and analyzer:
                strategies = list(correlation_matrix.correlation_matrix.index)
                if len(strategies) >= 2:
                    rolling_path = output_dir / f"rolling_correlation_{timestamp}.png"
                    self.plot_rolling_correlation(
                        analyzer, strategies[0], strategies[1], save_path=rolling_path
                    )
                    created_files["rolling_correlation"] = rolling_path
            
            # 6. サマリーレポート（テキスト）
            summary_path = output_dir / f"correlation_summary_{timestamp}.txt"
            self._create_text_summary(correlation_matrix, summary_path, analyzer)
            created_files["summary_report"] = summary_path
            
            self.logger.info(f"相関レポート作成完了: {output_dir}")
            return created_files
            
        except Exception as e:
            self.logger.error(f"相関レポート作成エラー: {e}")
            raise
    
    def _create_text_summary(self, correlation_matrix: CorrelationMatrix,
                           output_path: Path,
                           analyzer: Optional[StrategyCorrelationAnalyzer] = None) -> None:  # type: ignore
        """テキストサマリーを作成"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("戦略間相関分析レポート\n")
                f.write("=" * 60 + "\n\n")
                
                # 基本情報
                f.write(f"計算日時: {correlation_matrix.calculation_timestamp}\n")
                f.write(f"分析期間: {correlation_matrix.period_info.get('start_date', 'N/A')} - "
                       f"{correlation_matrix.period_info.get('end_date', 'N/A')}\n")
                f.write(f"データ期間: {correlation_matrix.period_info.get('total_periods', 'N/A')}期間\n")
                f.write(f"戦略数: {correlation_matrix.period_info.get('strategies_count', 'N/A')}戦略\n\n")
                
                # 相関行列
                f.write("相関行列:\n")
                f.write("-" * 40 + "\n")
                f.write(correlation_matrix.correlation_matrix.to_string())
                f.write("\n\n")
                
                # 共分散行列
                f.write("共分散行列:\n")
                f.write("-" * 40 + "\n")
                f.write(correlation_matrix.covariance_matrix.to_string())
                f.write("\n\n")
                
                # 統計サマリー（analyzerがある場合）
                if analyzer:
                    try:
                        summary = analyzer.get_correlation_summary(correlation_matrix)
                        f.write("統計サマリー:\n")
                        f.write("-" * 40 + "\n")
                        for key, value in summary.items():
                            f.write(f"{key}: {value}\n")
                        f.write("\n")
                    except:
                        pass
                
                # P値（利用可能な場合）
                if not correlation_matrix.p_values.empty:
                    f.write("P値行列:\n")
                    f.write("-" * 40 + "\n")
                    f.write(correlation_matrix.p_values.to_string())
                    f.write("\n\n")
            
            self.logger.info(f"テキストサマリー作成: {output_path}")
            
        except Exception as e:
            self.logger.error(f"テキストサマリー作成エラー: {e}")

def create_sample_visualization():
    """サンプル視覚化の実行"""
    try:
        # ダミーデータ作成
        np.random.seed(42)
        strategies = ['Strategy_A', 'Strategy_B', 'Strategy_C', 'Strategy_D']
        n_periods = 252
        
        # 相関のあるリターンデータ生成
        base_returns = np.random.normal(0, 0.02, n_periods)
        returns_data = {
            'Strategy_A': base_returns + np.random.normal(0, 0.01, n_periods),
            'Strategy_B': base_returns * 0.8 + np.random.normal(0, 0.015, n_periods),
            'Strategy_C': -base_returns * 0.5 + np.random.normal(0, 0.02, n_periods),
            'Strategy_D': np.random.normal(0, 0.025, n_periods)  # 独立
        }
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix_data = returns_df.corr()
        covariance_matrix_data = returns_df.cov() * 252
        
        # CorrelationMatrixオブジェクト作成（仮）
        class MockCorrelationMatrix:
            def __init__(self):
                self.correlation_matrix = correlation_matrix_data
                self.covariance_matrix = covariance_matrix_data
                self.p_values = pd.DataFrame()
                self.confidence_intervals = {}
                self.calculation_timestamp = datetime.now()
                self.period_info = {
                    'start_date': '2023-01-01',
                    'end_date': '2023-12-31',
                    'total_periods': n_periods,
                    'strategies_count': len(strategies)
                }
        
        mock_correlation_matrix = MockCorrelationMatrix()
        
        # 視覚化実行
        visualizer = CorrelationMatrixVisualizer()
        
        print("サンプル視覚化を実行中...")
        
        # ヒートマップ
        fig1 = visualizer.plot_correlation_heatmap(mock_correlation_matrix)
        plt.show()
        
        # 共分散ヒートマップ
        fig2 = visualizer.plot_covariance_heatmap(mock_correlation_matrix)
        plt.show()
        
        # 相関分布
        fig3 = visualizer.plot_correlation_distribution(mock_correlation_matrix)
        plt.show()
        
        # ネットワーク図
        fig4 = visualizer.plot_correlation_network(mock_correlation_matrix, threshold=0.3)
        plt.show()
        
        print("サンプル視覚化完了")
        
        # クリーンアップ
        plt.close('all')
        
    except Exception as e:
        logger.error(f"サンプル視覚化エラー: {e}")
        raise

if __name__ == "__main__":
    # 基本的なテスト
    logging.basicConfig(level=logging.INFO)
    
    print("相関行列視覚化システム - テスト実行")
    create_sample_visualization()
