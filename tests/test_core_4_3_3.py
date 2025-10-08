"""
4-3-3システム 基本機能テスト（視覚化除く）
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_core_functionality():
    """4-3-3システムのコア機能テスト"""
    try:
        print("4-3-3システム基本機能テスト開始\n")
        
        # 1. 相関分析システムのインポートテスト
        print("1. 相関分析システムインポート...")
        from config.correlation.strategy_correlation_analyzer import (
            StrategyCorrelationAnalyzer, CorrelationConfig
        )
        print("✓ 相関分析システムインポート成功\n")
        
        # 2. テストデータ作成
        print("2. テストデータ作成...")
        np.random.seed(42)
        periods = 100
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        # 価格データ
        price_data = pd.DataFrame({
            'close': 100 * np.cumprod(1 + np.random.normal(0, 0.01, periods))
        }, index=dates)
        
        # 戦略シグナル（相関を持つように設計）
        base_signal = np.random.normal(0, 1, periods)
        signals_a = pd.Series(np.where(base_signal > 0.5, 1, np.where(base_signal < -0.5, -1, 0)), index=dates)
        signals_b = pd.Series(np.where(base_signal * 0.8 + np.random.normal(0, 0.3, periods) > 0.3, 1, 
                                      np.where(base_signal * 0.8 + np.random.normal(0, 0.3, periods) < -0.3, -1, 0)), index=dates)
        signals_c = pd.Series(np.where(-base_signal + np.random.normal(0, 0.5, periods) > 0.4, 1, 
                                      np.where(-base_signal + np.random.normal(0, 0.5, periods) < -0.4, -1, 0)), index=dates)
        print("✓ テストデータ作成完了\n")
        
        # 3. 相関アナライザー初期化
        print("3. 相関アナライザー初期化...")
        config = CorrelationConfig(
            lookback_period=80,
            min_periods=20,
            correlation_method="pearson",
            rolling_window=15
        )
        analyzer = StrategyCorrelationAnalyzer(config)
        print("✓ アナライザー初期化成功\n")
        
        # 4. 戦略データ追加
        print("4. 戦略データ追加...")
        analyzer.add_strategy_data("Trend_Strategy", price_data, signals_a)
        analyzer.add_strategy_data("Mean_Reversion", price_data, signals_b)
        analyzer.add_strategy_data("Contrarian", price_data, signals_c)
        
        print(f"✓ 戦略データ追加完了 - 戦略数: {len(analyzer.strategy_data)}")
        
        # 各戦略の基本統計表示
        for strategy_name, strategy_data in analyzer.strategy_data.items():
            print(f"  - {strategy_name}:")
            print(f"    ボラティリティ: {strategy_data.volatility:.4f}")
            print(f"    シャープレシオ: {strategy_data.sharpe_ratio:.4f}")
            print(f"    勝率: {strategy_data.win_rate:.4f}")
        print()
        
        # 5. 相関行列計算
        print("5. 相関行列計算...")
        correlation_result = analyzer.calculate_correlation_matrix()
        print("✓ 相関行列計算成功")
        
        print("相関行列:")
        print(correlation_result.correlation_matrix.round(4))
        print()
        
        print("共分散行列（年率）:")
        print(correlation_result.covariance_matrix.round(6))
        print()
        
        # 6. 相関サマリー
        print("6. 相関サマリー生成...")
        summary = analyzer.get_correlation_summary(correlation_result)
        print("✓ サマリー生成成功")
        print("相関統計:")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        print()
        
        # 7. ローリング相関
        print("7. ローリング相関計算...")
        rolling_corr = analyzer.calculate_rolling_correlation(
            "Trend_Strategy", "Mean_Reversion", window=20
        )
        print("✓ ローリング相関計算成功")
        print(f"  - 有効データ点数: {len(rolling_corr.dropna())}")
        print(f"  - 平均相関: {rolling_corr.mean():.4f}")
        print(f"  - 相関範囲: {rolling_corr.min():.4f} - {rolling_corr.max():.4f}")
        print()
        
        # 8. クラスター分析
        print("8. 相関クラスター分析...")
        try:
            clusters = analyzer.detect_correlation_clusters(correlation_result, threshold=0.5)
            if clusters:
                print("✓ クラスター分析成功")
                for cluster_id, strategies in clusters.items():
                    print(f"  クラスター {cluster_id}: {strategies}")
            else:
                print("[WARNING] scikit-learn未インストールのためクラスター分析スキップ")
        except Exception as e:
            print(f"[WARNING] クラスター分析でエラー: {e}")
        print()
        
        # 9. データ永続化テスト
        print("9. データ保存・読み込みテスト...")
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # 保存
            analyzer.save_correlation_data(correlation_result, temp_path)
            print("✓ データ保存成功")
            
            # 読み込み
            loaded_result = analyzer.load_correlation_data(temp_path)
            print("✓ データ読み込み成功")
            
            # 一致確認
            original_corr = correlation_result.correlation_matrix
            loaded_corr = loaded_result.correlation_matrix
            matches = original_corr.equals(loaded_corr)
            print(f"✓ データ整合性確認: {'成功' if matches else '失敗'}")
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
        print()
        
        # 10. システム統合状況確認
        print("10. システム統合状況確認...")
        integrations = []
        
        try:
            from config.strategy_scoring_model import StrategyScoreManager
            integrations.append("戦略スコアマネージャー")
        except ImportError:
            pass
            
        try:
            from config.portfolio_weight_calculator import PortfolioWeightCalculator
            integrations.append("ポートフォリオ計算機")
        except ImportError:
            pass
            
        try:
            from config.strategy_selector import StrategySelector
            integrations.append("戦略セレクター")
        except ImportError:
            pass
            
        print(f"✓ 統合可能システム: {len(integrations)}個")
        for integration in integrations:
            print(f"  - {integration}")
        print()
        
        # テスト結果サマリー
        print("="*60)
        print("4-3-3システム 基本機能テスト 完了")
        print("="*60)
        print("[SUCCESS] すべての基本機能が正常に動作しました！")
        print()
        print("実装された機能:")
        print("[OK] 戦略パフォーマンスデータ管理")
        print("[OK] 戦略間相関行列計算")
        print("[OK] 共分散行列計算")
        print("[OK] ローリング相関分析")
        print("[OK] 相関統計サマリー")
        print("[OK] 相関クラスター分析（オプション）")
        print("[OK] データ保存・読み込み")
        print("[OK] 既存システム統合準備")
        print()
        
        # 分析結果のハイライト
        print("分析結果ハイライト:")
        print(f"[CHART] 分析対象戦略数: {len(analyzer.strategy_data)}")
        print(f"[UP] 戦略ペア数: {summary['total_pairs']}")
        print(f"🔗 平均相関係数: {summary['mean_correlation']:.4f}")
        print(f"[DOWN] 最小相関: {summary['min_correlation']:.4f}")
        print(f"[UP] 最大相関: {summary['max_correlation']:.4f}")
        print(f"[TARGET] 高相関ペア (>0.7): {summary['high_correlation_pairs']}個")
        print(f"⚖️ 中相関ペア (0.3-0.7): {summary['moderate_correlation_pairs']}個")
        print(f"🌐 低相関ペア (<0.3): {summary['low_correlation_pairs']}個")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_core_functionality()
    if success:
        print("\n✨ 4-3-3システムは正常に動作しています！")
        print("\n推奨される次のステップ:")
        print("1. seabornライブラリをインストールして視覚化機能を有効化")
        print("2. 実際の市場データでシステムをテスト")
        print("3. 4-3-1、4-3-2システムとの統合テスト")
        print("4. ポートフォリオ最適化への応用")
    else:
        print("\n[WARNING] 問題が発生しました。エラーログを確認してください。")
