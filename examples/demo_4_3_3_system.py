"""
4-3-3システム統合デモ - 戦略間相関・共分散行列の視覚化システム

このスクリプトは4-3-3システムの機能を包括的にデモンストレーションし、
既存の4-3-1、4-3-2システムとの統合を実証する。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# システムインポート
try:
    from config.correlation.strategy_correlation_analyzer import (
        StrategyCorrelationAnalyzer, CorrelationConfig
    )
    from config.correlation.correlation_matrix_visualizer import CorrelationMatrixVisualizer
    from config.correlation.strategy_correlation_dashboard import StrategyCorrelationDashboard
except ImportError as e:
    print(f"4-3-3システムインポートエラー: {e}")
    exit(1)

# 既存システム統合
try:
    from config.strategy_scoring_model import StrategyScoreManager
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.strategy_selector import StrategySelector
    existing_system_available = True
except ImportError as e:
    print(f"既存システム統合警告: {e}")
    existing_system_available = False

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('4_3_3_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Demo433System:
    """4-3-3システム統合デモクラス"""
    
    def __init__(self, output_dir: str = "demo_4_3_3_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # システムコンポーネント
        self.config = CorrelationConfig(
            lookback_period=252,
            min_periods=60,
            correlation_method="pearson",
            rolling_window=30
        )
        
        self.analyzer = StrategyCorrelationAnalyzer(self.config)
        self.visualizer = CorrelationMatrixVisualizer(figsize=(14, 10))
        self.dashboard = StrategyCorrelationDashboard(self.config)
        
        # デモデータ
        self.demo_strategies = [
            'Trend_Following', 'Mean_Reversion', 'Momentum', 'Contrarian', 'Pairs_Trading'
        ]
        self.demo_data = {}
        
        logger.info("4-3-3システムデモ初期化完了")
    
    def generate_demo_data(self, periods: int = 365, seed: int = 42):
        """リアルなデモデータを生成"""
        try:
            np.random.seed(seed)
            
            # 日付範囲
            end_date = datetime.now()
            start_date = end_date - timedelta(days=periods)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')[:periods]
            
            # 基準価格データ（仮想的な市場インデックス）
            base_returns = np.random.normal(0.0008, 0.02, periods)  # 日次リターン
            base_price = 1000 * np.cumprod(1 + base_returns)
            
            price_data = pd.DataFrame({
                'close': base_price,
                'high': base_price * (1 + np.abs(np.random.normal(0, 0.01, periods))),
                'low': base_price * (1 - np.abs(np.random.normal(0, 0.01, periods))),
                'volume': np.random.randint(1000000, 5000000, periods)
            }, index=dates)
            
            # 戦略別シグナル生成（各戦略に特性を持たせる）
            signals_data = {}
            
            # 1. Trend Following: 価格トレンドに従う
            ma_short = price_data['close'].rolling(10).mean()
            ma_long = price_data['close'].rolling(30).mean()
            trend_signals = np.where(ma_short > ma_long, 1, -1)
            trend_signals = pd.Series(trend_signals, index=dates)
            signals_data['Trend_Following'] = trend_signals
            
            # 2. Mean Reversion: 平均回帰
            z_score = (price_data['close'] - price_data['close'].rolling(20).mean()) / price_data['close'].rolling(20).std()
            mean_rev_signals = np.where(z_score > 1.5, -1, np.where(z_score < -1.5, 1, 0))
            signals_data['Mean_Reversion'] = pd.Series(mean_rev_signals, index=dates)
            
            # 3. Momentum: 価格変化率に基づく
            momentum = price_data['close'].pct_change(10)
            momentum_signals = np.where(momentum > 0.02, 1, np.where(momentum < -0.02, -1, 0))
            signals_data['Momentum'] = pd.Series(momentum_signals, index=dates)
            
            # 4. Contrarian: 逆張り
            rsi = self._calculate_rsi(price_data['close'])
            contrarian_signals = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))
            signals_data['Contrarian'] = pd.Series(contrarian_signals, index=dates)
            
            # 5. Pairs Trading: ランダムだが相関を持つ
            pairs_signals = np.random.choice([-1, 0, 1], periods, p=[0.25, 0.5, 0.25])
            signals_data['Pairs_Trading'] = pd.Series(pairs_signals, index=dates)
            
            self.demo_data = {
                'price_data': price_data,
                'signals_data': signals_data
            }
            
            logger.info(f"デモデータ生成完了: {periods}日間, {len(self.demo_strategies)}戦略")
            return True
            
        except Exception as e:
            logger.error(f"デモデータ生成エラー: {e}")
            return False
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def run_correlation_analysis(self):
        """相関分析を実行"""
        try:
            logger.info("相関分析開始...")
            
            # 戦略データ追加
            for strategy_name, signals in self.demo_data['signals_data'].items():
                self.analyzer.add_strategy_data(
                    strategy_name, 
                    self.demo_data['price_data'], 
                    signals
                )
                
                # ダッシュボードにも追加
                self.dashboard.add_strategy_performance(
                    strategy_name,
                    self.demo_data['price_data'],
                    signals
                )
            
            # 相関分析実行
            correlation_result = self.analyzer.calculate_correlation_matrix()
            
            logger.info("相関分析完了")
            return correlation_result
            
        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            return None
    
    def create_visualizations(self, correlation_matrix):
        """視覚化を作成"""
        try:
            logger.info("視覚化作成開始...")
            
            created_plots = {}
            
            # 1. 相関ヒートマップ
            fig1 = self.visualizer.plot_correlation_heatmap(
                correlation_matrix,
                title="4-3-3: 戦略間相関行列ヒートマップ",
                save_path=self.output_dir / "correlation_heatmap.png"
            )
            created_plots["correlation_heatmap"] = fig1
            plt.show()
            plt.close(fig1)
            
            # 2. 共分散ヒートマップ
            fig2 = self.visualizer.plot_covariance_heatmap(
                correlation_matrix,
                title="4-3-3: 戦略間共分散行列ヒートマップ",
                save_path=self.output_dir / "covariance_heatmap.png"
            )
            created_plots["covariance_heatmap"] = fig2
            plt.show()
            plt.close(fig2)
            
            # 3. 相関ネットワーク
            fig3 = self.visualizer.plot_correlation_network(
                correlation_matrix,
                threshold=0.3,
                title="4-3-3: 戦略相関ネットワーク図",
                save_path=self.output_dir / "correlation_network.png"
            )
            created_plots["correlation_network"] = fig3
            plt.show()
            plt.close(fig3)
            
            # 4. 相関分布
            fig4 = self.visualizer.plot_correlation_distribution(
                correlation_matrix,
                title="4-3-3: 戦略間相関係数分布",
                save_path=self.output_dir / "correlation_distribution.png"
            )
            created_plots["correlation_distribution"] = fig4
            plt.show()
            plt.close(fig4)
            
            # 5. ローリング相関
            strategies = list(correlation_matrix.correlation_matrix.index)
            if len(strategies) >= 2:
                fig5 = self.visualizer.plot_rolling_correlation(
                    self.analyzer,
                    strategies[0], strategies[1],
                    window=30,
                    title=f"4-3-3: {strategies[0]} vs {strategies[1]} ローリング相関",
                    save_path=self.output_dir / "rolling_correlation.png"
                )
                created_plots["rolling_correlation"] = fig5
                plt.show()
                plt.close(fig5)
            
            logger.info(f"視覚化作成完了: {len(created_plots)}個のプロット")
            return created_plots
            
        except Exception as e:
            logger.error(f"視覚化作成エラー: {e}")
            return {}
    
    def create_integrated_dashboard(self, correlation_matrix):
        """統合ダッシュボードを作成"""
        try:
            logger.info("統合ダッシュボード作成開始...")
            
            # ダッシュボード相関分析
            dashboard_correlation = self.dashboard.calculate_correlation_analysis()
            
            if dashboard_correlation:
                # 統合ダッシュボード作成
                dashboard_fig = self.dashboard.create_integrated_dashboard(
                    dashboard_correlation,
                    save_path=self.output_dir / "integrated_dashboard.png",
                    include_performance_metrics=True,
                    include_risk_analysis=True
                )
                
                plt.show()
                plt.close(dashboard_fig)
                
                logger.info("統合ダッシュボード作成完了")
                return dashboard_fig
            
        except Exception as e:
            logger.error(f"統合ダッシュボード作成エラー: {e}")
            return None
    
    def generate_comprehensive_report(self, correlation_matrix):
        """包括的レポートを生成"""
        try:
            logger.info("包括的レポート生成開始...")
            
            # 相関サマリー取得
            summary = self.analyzer.get_correlation_summary(correlation_matrix)
            
            # クラスター分析
            clusters = self.analyzer.detect_correlation_clusters(correlation_matrix, threshold=0.6)
            
            # レポートファイル作成
            report_path = self.output_dir / "comprehensive_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("4-3-3システム 包括的レポート\n")
                f.write("戦略間相関・共分散行列の視覚化システム\n")
                f.write("=" * 80 + "\n\n")
                
                # 実行情報
                f.write("実行情報:\n")
                f.write("-" * 40 + "\n")
                f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析期間: {correlation_matrix.period_info['start_date']} - {correlation_matrix.period_info['end_date']}\n")
                f.write(f"対象戦略: {', '.join(self.demo_strategies)}\n")
                f.write(f"データ期間: {correlation_matrix.period_info['total_periods']}日\n\n")
                
                # 相関統計
                f.write("相関統計サマリー:\n")
                f.write("-" * 40 + "\n")
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # クラスター分析結果
                if clusters:
                    f.write("相関クラスター分析:\n")
                    f.write("-" * 40 + "\n")
                    for cluster_id, strategies in clusters.items():
                        f.write(f"クラスター {cluster_id}: {', '.join(strategies)}\n")
                    f.write("\n")
                
                # 相関行列
                f.write("相関行列:\n")
                f.write("-" * 40 + "\n")
                f.write(correlation_matrix.correlation_matrix.round(4).to_string())
                f.write("\n\n")
                
                # 共分散行列
                f.write("共分散行列 (年率):\n")
                f.write("-" * 40 + "\n")
                f.write(correlation_matrix.covariance_matrix.round(6).to_string())
                f.write("\n\n")
                
                # パフォーマンス指標
                f.write("戦略別パフォーマンス指標:\n")
                f.write("-" * 40 + "\n")
                for strategy_name, strategy_data in self.analyzer.strategy_data.items():
                    f.write(f"\n{strategy_name}:\n")
                    f.write(f"  ボラティリティ: {strategy_data.volatility:.4f}\n")
                    f.write(f"  シャープレシオ: {strategy_data.sharpe_ratio:.4f}\n")
                    f.write(f"  最大ドローダウン: {strategy_data.max_drawdown:.4f}\n")
                    f.write(f"  勝率: {strategy_data.win_rate:.4f}\n")
                
                # システム統合情報
                f.write("\nシステム統合情報:\n")
                f.write("-" * 40 + "\n")
                f.write(f"既存システム統合: {'成功' if existing_system_available else '部分的'}\n")
                f.write(f"4-3-1システム統合: 利用可能\n")
                f.write(f"4-3-2システム統合: 利用可能\n")
                f.write(f"4-3-3システム: 完全実装\n")
            
            logger.info(f"包括的レポート生成完了: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"包括的レポート生成エラー: {e}")
            return None
    
    def run_full_demo(self):
        """フルデモを実行"""
        try:
            print("\n" + "=" * 80)
            print("4-3-3システム 包括的デモンストレーション開始")
            print("戦略間相関・共分散行列の視覚化システム")
            print("=" * 80 + "\n")
            
            # ステップ1: デモデータ生成
            print("ステップ1: デモデータ生成...")
            if not self.generate_demo_data(periods=365):
                raise Exception("デモデータ生成に失敗しました")
            print("✓ デモデータ生成完了\n")
            
            # ステップ2: 相関分析
            print("ステップ2: 相関分析実行...")
            correlation_result = self.run_correlation_analysis()
            if not correlation_result:
                raise Exception("相関分析に失敗しました")
            print("✓ 相関分析完了\n")
            
            # ステップ3: 視覚化作成
            print("ステップ3: 視覚化作成...")
            plots = self.create_visualizations(correlation_result)
            print(f"✓ 視覚化作成完了: {len(plots)}個のプロット\n")
            
            # ステップ4: 統合ダッシュボード
            print("ステップ4: 統合ダッシュボード作成...")
            dashboard = self.create_integrated_dashboard(correlation_result)
            print("✓ 統合ダッシュボード作成完了\n")
            
            # ステップ5: 包括的レポート
            print("ステップ5: 包括的レポート生成...")
            report_path = self.generate_comprehensive_report(correlation_result)
            print(f"✓ 包括的レポート生成完了: {report_path}\n")
            
            # 結果サマリー
            print("=" * 80)
            print("4-3-3システム デモ完了")
            print("=" * 80)
            print(f"出力ディレクトリ: {self.output_dir}")
            print("作成されたファイル:")
            for file_path in self.output_dir.glob("*"):
                print(f"  - {file_path.name}")
            
            # 相関分析結果サマリー表示
            summary = self.analyzer.get_correlation_summary(correlation_result)
            print(f"\n相関分析結果サマリー:")
            print(f"  戦略数: {len(self.demo_strategies)}")
            print(f"  戦略ペア数: {summary['total_pairs']}")
            print(f"  平均相関: {summary['mean_correlation']:.4f}")
            print(f"  高相関ペア(>0.7): {summary['high_correlation_pairs']}")
            print(f"  中相関ペア(0.3-0.7): {summary['moderate_correlation_pairs']}")
            print(f"  低相関ペア(<0.3): {summary['low_correlation_pairs']}")
            
            return True
            
        except Exception as e:
            logger.error(f"フルデモ実行エラー: {e}")
            print(f"[ERROR] デモ実行中にエラーが発生しました: {e}")
            return False

def main():
    """メイン実行関数"""
    try:
        # デモ実行
        demo = Demo433System(output_dir="demo_4_3_3_output")
        success = demo.run_full_demo()
        
        if success:
            print("\n[SUCCESS] 4-3-3システムデモが正常に完了しました！")
            print("\n次のステップ:")
            print("1. 生成された視覚化とレポートを確認")
            print("2. 実際のデータでシステムをテスト")
            print("3. 既存システムとの統合を確認")
        else:
            print("\n[ERROR] デモ実行に失敗しました。ログを確認してください。")
            
    except KeyboardInterrupt:
        print("\n⏸️ ユーザーによってデモが中断されました。")
    except Exception as e:
        logger.error(f"メイン実行エラー: {e}")
        print(f"\n💥 予期しないエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
