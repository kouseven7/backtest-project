"""
4-3-1 Simple Test Script
トレンド戦略可視化システムの簡単なテスト
"""

import sys
from pathlib import Path

# プロジェクトルートを取得してパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("=== 4-3-1 簡単テスト開始 ===")

try:
    # 基本インポートテスト
    print("1. インポートテスト中...")
    
    from visualization.chart_config import ChartConfigManager
    print("  ✓ ChartConfigManager インポート成功")
    
    from visualization.data_aggregator import VisualizationDataAggregator
    print("  ✓ VisualizationDataAggregator インポート成功")
    
    from visualization.trend_strategy_time_series import TrendStrategyTimeSeriesVisualizer
    print("  ✓ TrendStrategyTimeSeriesVisualizer インポート成功")
    
    # 基本設定テスト
    print("\n2. 基本設定テスト中...")
    
    config_manager = ChartConfigManager()
    print(f"  ✓ ChartConfigManager インスタンス作成成功")
    print(f"  - Figure config: {list(config_manager.figure_config.keys())}")
    print(f"  - Color scheme: {len(config_manager.color_scheme)}色")
    
    data_aggregator = VisualizationDataAggregator(symbol="USDJPY", period_days=10)
    print(f"  ✓ DataAggregator インスタンス作成成功")
    print(f"  - Symbol: {data_aggregator.symbol}")
    print(f"  - Period: {data_aggregator.period_days}日")
    
    # 合成データテスト
    print("\n3. 合成データ生成テスト中...")
    
    price_data = data_aggregator._generate_synthetic_price_data()
    print(f"  ✓ 合成価格データ生成成功: {len(price_data)}レコード")
    print(f"  - Columns: {list(price_data.columns)}")
    
    trend_data = data_aggregator._generate_synthetic_trend_data()
    print(f"  ✓ 合成トレンドデータ生成成功: {len(trend_data)}レコード")
    
    strategy_data = data_aggregator._generate_synthetic_strategy_data()
    print(f"  ✓ 合成戦略データ生成成功: {len(strategy_data)}レコード")
    
    # データ集約テスト
    print("\n4. データ集約テスト中...")
    
    aggregated_data = data_aggregator.aggregate_all_data()
    print(f"  ✓ データ集約成功: {len(aggregated_data)}レコード")
    print(f"  - Columns: {list(aggregated_data.columns)}")
    print(f"  - 品質スコア: {data_aggregator.data_quality_score:.2f}")
    
    # 可視化エンジンテスト
    print("\n5. 可視化エンジンテスト中...")
    
    visualizer = TrendStrategyTimeSeriesVisualizer(
        symbol="USDJPY",
        period_days=10,
        output_dir="simple_test_outputs"
    )
    print("  ✓ Visualizer インスタンス作成成功")
    
    # メタデータテスト
    metadata = visualizer.get_chart_metadata()
    print(f"  ✓ メタデータ取得成功: status={metadata.get('symbol', 'unknown')}")
    
    print("\n=== 4-3-1 簡単テスト成功 ===")
    print("✓ 全ての基本機能が正常に動作しています")
    
except ImportError as e:
    print(f"✗ インポートエラー: {e}")
    print("必要なモジュールがインストールされていない可能性があります")
    
except Exception as e:
    print(f"✗ テストエラー: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    print("\n=== 4-3-1 簡単テスト終了 ===")
