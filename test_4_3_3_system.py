"""
4-3-3システム テストスイート

戦略間相関・共分散行列の視覚化システムの包括的テスト
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import warnings
warnings.filterwarnings('ignore')

# テスト対象システム
from config.correlation.strategy_correlation_analyzer import (
    StrategyCorrelationAnalyzer, 
    CorrelationMatrix, 
    CorrelationConfig,
    StrategyPerformanceData
)
from config.correlation.correlation_matrix_visualizer import CorrelationMatrixVisualizer
from config.correlation.strategy_correlation_dashboard import StrategyCorrelationDashboard

class Test433System:
    """4-3-3システム統合テスト"""
    
    def setup_method(self):
        """テスト前準備"""
        self.config = CorrelationConfig(
            lookback_period=100,
            min_periods=20,
            correlation_method="pearson",
            rolling_window=10
        )
        
        # テストデータ生成
        np.random.seed(42)
        self.periods = 100
        self.dates = pd.date_range('2023-01-01', periods=self.periods, freq='D')
        
        # 価格データ
        self.price_data = pd.DataFrame({
            'close': 100 * np.cumprod(1 + np.random.normal(0, 0.01, self.periods)),
            'high': 102 * np.cumprod(1 + np.random.normal(0, 0.01, self.periods)),
            'low': 98 * np.cumprod(1 + np.random.normal(0, 0.01, self.periods))
        }, index=self.dates)
        
        # テスト戦略シグナル
        self.test_signals = {
            'strategy_a': pd.Series(np.random.choice([-1, 0, 1], self.periods), index=self.dates),
            'strategy_b': pd.Series(np.random.choice([-1, 0, 1], self.periods), index=self.dates),
            'strategy_c': pd.Series(np.random.choice([-1, 0, 1], self.periods), index=self.dates)
        }
        
        self.analyzer = StrategyCorrelationAnalyzer(self.config)
        
    def test_correlation_config(self):
        """相関設定テスト"""
        config = CorrelationConfig()
        assert config.lookback_period == 252
        assert config.min_periods == 60
        assert config.correlation_method == "pearson"
        assert config.rolling_window == 30
        
        # カスタム設定
        custom_config = CorrelationConfig(
            lookback_period=100,
            correlation_method="spearman"
        )
        assert custom_config.lookback_period == 100
        assert custom_config.correlation_method == "spearman"
    
    def test_strategy_performance_data(self):
        """戦略パフォーマンスデータテスト"""
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        cumulative_returns = (1 + returns).cumprod() - 1
        
        perf_data = StrategyPerformanceData(
            strategy_name="test_strategy",
            returns=returns,  # type: ignore
            cumulative_returns=cumulative_returns,  # type: ignore
            volatility=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.05,
            win_rate=0.55
        )
        
        assert perf_data.strategy_name == "test_strategy"
        assert perf_data.volatility == 0.15
        assert perf_data.sharpe_ratio == 1.2
    
    def test_strategy_correlation_analyzer_init(self):
        """相関アナライザー初期化テスト"""
        analyzer = StrategyCorrelationAnalyzer()
        assert analyzer.config.lookback_period == 252
        assert len(analyzer.strategy_data) == 0
        assert len(analyzer.correlation_history) == 0
        
        # カスタム設定
        custom_analyzer = StrategyCorrelationAnalyzer(self.config)
        assert custom_analyzer.config.lookback_period == 100
    
    def test_add_strategy_data(self):
        """戦略データ追加テスト"""
        # 正常ケース
        self.analyzer.add_strategy_data(
            "test_strategy", 
            self.price_data, 
            self.test_signals['strategy_a']
        )
        
        assert "test_strategy" in self.analyzer.strategy_data
        strategy_data = self.analyzer.strategy_data["test_strategy"]
        assert strategy_data.strategy_name == "test_strategy"
        assert hasattr(strategy_data, 'returns')
        assert hasattr(strategy_data, 'volatility')
        assert hasattr(strategy_data, 'sharpe_ratio')
        
        # 複数戦略追加
        for name, signals in self.test_signals.items():
            self.analyzer.add_strategy_data(name, self.price_data, signals)
        
        assert len(self.analyzer.strategy_data) == 4  # test_strategy + 3
    
    def test_calculate_correlation_matrix(self):
        """相関行列計算テスト"""
        # データ追加
        for name, signals in self.test_signals.items():
            self.analyzer.add_strategy_data(name, self.price_data, signals)
        
        # 相関計算
        correlation_result = self.analyzer.calculate_correlation_matrix()
        
        # 結果検証
        assert isinstance(correlation_result, CorrelationMatrix)
        assert not correlation_result.correlation_matrix.empty
        assert not correlation_result.covariance_matrix.empty
        
        # 行列のプロパティ検証
        corr_matrix = correlation_result.correlation_matrix
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # 正方行列
        assert len(corr_matrix) == len(self.test_signals)
        
        # 対角成分が1であることを確認
        for i in range(len(corr_matrix)):
            assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-6
        
        # 相関係数の範囲確認（-1 ≤ r ≤ 1）
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                assert -1 <= corr_matrix.iloc[i, j] <= 1
    
    def test_rolling_correlation(self):
        """ローリング相関テスト"""
        # データ追加
        for name, signals in self.test_signals.items():
            self.analyzer.add_strategy_data(name, self.price_data, signals)
        
        # ローリング相関計算
        rolling_corr = self.analyzer.calculate_rolling_correlation(
            'strategy_a', 'strategy_b', window=20
        )
        
        # 結果検証
        assert isinstance(rolling_corr, pd.Series)
        assert len(rolling_corr) > 0
        
        # 相関値の範囲確認
        valid_corr = rolling_corr.dropna()
        assert all(-1 <= val <= 1 for val in valid_corr)
    
    def test_correlation_summary(self):
        """相関サマリーテスト"""
        # データ追加と相関計算
        for name, signals in self.test_signals.items():
            self.analyzer.add_strategy_data(name, self.price_data, signals)
        
        correlation_result = self.analyzer.calculate_correlation_matrix()
        summary = self.analyzer.get_correlation_summary(correlation_result)
        
        # サマリー項目確認
        required_keys = [
            'total_pairs', 'mean_correlation', 'median_correlation',
            'std_correlation', 'min_correlation', 'max_correlation',
            'high_correlation_pairs', 'moderate_correlation_pairs',
            'low_correlation_pairs'
        ]
        
        for key in required_keys:
            assert key in summary
        
        # 統計値の妥当性確認
        assert summary['total_pairs'] > 0
        assert -1 <= summary['mean_correlation'] <= 1
        assert -1 <= summary['min_correlation'] <= 1
        assert -1 <= summary['max_correlation'] <= 1
    
    def test_correlation_clusters(self):
        """相関クラスターテスト"""
        # データ追加と相関計算
        for name, signals in self.test_signals.items():
            self.analyzer.add_strategy_data(name, self.price_data, signals)
        
        correlation_result = self.analyzer.calculate_correlation_matrix()
        
        try:
            clusters = self.analyzer.detect_correlation_clusters(
                correlation_result, threshold=0.5
            )
            
            if clusters:  # scikit-learnが利用可能な場合のみテスト
                assert isinstance(clusters, dict)
                
                # 全戦略がクラスターに含まれることを確認
                all_strategies_in_clusters = set()
                for cluster_strategies in clusters.values():
                    all_strategies_in_clusters.update(cluster_strategies)
                
                original_strategies = set(self.test_signals.keys())
                assert all_strategies_in_clusters == original_strategies
                
        except ImportError:
            # scikit-learnが利用できない場合はスキップ
            pass
    
    def test_save_load_correlation_data(self):
        """相関データ保存・読み込みテスト"""
        # データ追加と相関計算
        for name, signals in self.test_signals.items():
            self.analyzer.add_strategy_data(name, self.price_data, signals)
        
        correlation_result = self.analyzer.calculate_correlation_matrix()
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # 保存
            self.analyzer.save_correlation_data(correlation_result, temp_path)
            assert temp_path.exists()
            
            # 読み込み
            loaded_result = self.analyzer.load_correlation_data(temp_path)
            
            # データ一致確認
            assert loaded_result.correlation_matrix.equals(correlation_result.correlation_matrix)
            assert loaded_result.covariance_matrix.equals(correlation_result.covariance_matrix)
            assert loaded_result.period_info == correlation_result.period_info
            
        finally:
            # クリーンアップ
            if temp_path.exists():
                temp_path.unlink()

class TestCorrelationMatrixVisualizer:
    """相関行列視覚化テスト"""
    
    def setup_method(self):
        """テスト前準備"""
        # Mock相関行列データ
        strategies = ['Strategy_A', 'Strategy_B', 'Strategy_C']
        corr_data = np.array([
            [1.0, 0.7, -0.3],
            [0.7, 1.0, 0.1],
            [-0.3, 0.1, 1.0]
        ])
        cov_data = corr_data * 0.04  # 分散を0.04と仮定
        
        self.correlation_matrix = CorrelationMatrix(
            correlation_matrix=pd.DataFrame(corr_data, index=strategies, columns=strategies),
            covariance_matrix=pd.DataFrame(cov_data, index=strategies, columns=strategies),
            p_values=pd.DataFrame(),
            confidence_intervals={},
            calculation_timestamp=datetime.now(),
            period_info={
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'total_periods': 252,
                'strategies_count': 3
            }
        )
        
        self.visualizer = CorrelationMatrixVisualizer(figsize=(10, 8))
    
    def test_visualizer_init(self):
        """視覚化初期化テスト"""
        visualizer = CorrelationMatrixVisualizer()
        assert visualizer.figsize == (12, 10)
        assert visualizer.style == "whitegrid"
        
        # カスタム設定
        custom_visualizer = CorrelationMatrixVisualizer(figsize=(8, 6), style="darkgrid")
        assert custom_visualizer.figsize == (8, 6)
        assert custom_visualizer.style == "darkgrid"
    
    def test_plot_correlation_heatmap(self):
        """相関ヒートマップテスト"""
        try:
            fig = self.visualizer.plot_correlation_heatmap(
                self.correlation_matrix,
                title="テスト相関ヒートマップ"
            )
            
            assert fig is not None
            # プロットが作成されていることを確認
            assert len(fig.axes) > 0
            
        except Exception as e:
            # 依存関係のエラーの場合はスキップ
            if "seaborn" in str(e).lower() or "matplotlib" in str(e).lower():
                pytest.skip(f"視覚化ライブラリエラー: {e}")
            else:
                raise
    
    def test_plot_covariance_heatmap(self):
        """共分散ヒートマップテスト"""
        try:
            fig = self.visualizer.plot_covariance_heatmap(
                self.correlation_matrix,
                title="テスト共分散ヒートマップ"
            )
            
            assert fig is not None
            assert len(fig.axes) > 0
            
        except Exception as e:
            if "seaborn" in str(e).lower() or "matplotlib" in str(e).lower():
                pytest.skip(f"視覚化ライブラリエラー: {e}")
            else:
                raise
    
    def test_plot_correlation_distribution(self):
        """相関分布テスト"""
        try:
            fig = self.visualizer.plot_correlation_distribution(
                self.correlation_matrix,
                title="テスト相関分布"
            )
            
            assert fig is not None
            assert len(fig.axes) >= 2  # ヒストグラムとボックスプロット
            
        except Exception as e:
            if "matplotlib" in str(e).lower():
                pytest.skip(f"視覚化ライブラリエラー: {e}")
            else:
                raise

class TestStrategyCorrelationDashboard:
    """戦略相関ダッシュボードテスト"""
    
    def setup_method(self):
        """テスト前準備"""
        self.config = CorrelationConfig(lookback_period=50, min_periods=10)
        self.dashboard = StrategyCorrelationDashboard(self.config)
        
        # テストデータ
        np.random.seed(42)
        periods = 50
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        self.price_data = pd.DataFrame({
            'close': 100 * np.cumprod(1 + np.random.normal(0, 0.01, periods))
        }, index=dates)
        
        self.test_signals = pd.Series(
            np.random.choice([-1, 0, 1], periods), 
            index=dates
        )
    
    def test_dashboard_init(self):
        """ダッシュボード初期化テスト"""
        dashboard = StrategyCorrelationDashboard()
        assert dashboard.correlation_history == []
        assert dashboard.strategy_data == {}
        
        # カスタム設定
        custom_dashboard = StrategyCorrelationDashboard(self.config)
        assert custom_dashboard.config.lookback_period == 50
    
    def test_add_strategy_performance(self):
        """戦略パフォーマンス追加テスト"""
        self.dashboard.add_strategy_performance(
            "test_strategy",
            self.price_data,
            self.test_signals
        )
        
        # 内部のアナライザーにデータが追加されていることを確認
        if self.dashboard.correlation_analyzer:
            assert "test_strategy" in self.dashboard.correlation_analyzer.strategy_data
    
    def test_calculate_correlation_analysis(self):
        """相関分析テスト"""
        # データ追加
        self.dashboard.add_strategy_performance("strategy_a", self.price_data, self.test_signals)
        self.dashboard.add_strategy_performance("strategy_b", self.price_data, self.test_signals)
        
        # 相関分析実行
        result = self.dashboard.calculate_correlation_analysis()
        
        if result:  # アナライザーが利用可能な場合
            assert isinstance(result, CorrelationMatrix)
            assert len(self.dashboard.correlation_history) > 0

def run_comprehensive_test():
    """包括的テスト実行"""
    print("4-3-3システム 包括的テスト開始\n")
    
    # テスト実行
    test_classes = [
        Test433System,
        TestCorrelationMatrixVisualizer,
        TestStrategyCorrelationDashboard
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"テストクラス: {test_class.__name__}")
        
        # テストメソッド取得
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            
            try:
                # テストインスタンス作成
                test_instance = test_class()
                
                # setup実行
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # テスト実行
                test_method = getattr(test_instance, method_name)
                test_method()
                
                print(f"  ✓ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}: {e}")
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print(f"テスト結果サマリー")
    print(f"{'='*60}")
    print(f"総テスト数: {total_tests}")
    print(f"成功: {passed_tests}")
    print(f"失敗: {len(failed_tests)}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests:
        print(f"\n失敗したテスト:")
        for failed in failed_tests:
            print(f"  - {failed}")
    
    return len(failed_tests) == 0

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
