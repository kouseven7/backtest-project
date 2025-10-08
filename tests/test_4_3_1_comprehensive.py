"""
4-3-1 Comprehensive Test Script
トレンド戦略可視化システムの包括テスト
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# プロジェクトルートを取得してパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("=== 4-3-1 包括テスト開始 ===")
print(f"プロジェクトルート: {project_root}")

def test_imports():
    """インポートテスト"""
    print("\n--- インポートテスト ---")
    
    try:
        from visualization.chart_config import ChartConfigManager
        print("[OK] ChartConfigManager インポート成功")
        
        config = ChartConfigManager()
        print("[OK] ChartConfigManager 初期化成功")
        print(f"  図サイズ: {config.figure_config['figsize']}")
        
        # 色設定テスト
        uptrend_color = config.get_trend_color('uptrend')
        print(f"  上昇トレンド色: {uptrend_color}")
        
    except Exception as e:
        print(f"[ERROR] ChartConfigManager エラー: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        from visualization.data_aggregator import VisualizationDataAggregator
        print("[OK] VisualizationDataAggregator インポート成功")
        
        aggregator = VisualizationDataAggregator(symbol="USDJPY", period_days=30)
        print("[OK] VisualizationDataAggregator 初期化成功")
        print(f"  シンボル: {aggregator.symbol}")
        print(f"  期間: {aggregator.period_days}日")
        
    except Exception as e:
        print(f"[ERROR] VisualizationDataAggregator エラー: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        from visualization.trend_strategy_time_series import TrendStrategyTimeSeriesVisualizer
        print("[OK] TrendStrategyTimeSeriesVisualizer インポート成功")
        
        visualizer = TrendStrategyTimeSeriesVisualizer(
            symbol="USDJPY",
            period_days=30,
            output_dir="test_outputs"
        )
        print("[OK] TrendStrategyTimeSeriesVisualizer 初期化成功")
        
    except Exception as e:
        print(f"[ERROR] TrendStrategyTimeSeriesVisualizer エラー: {e}")
        import traceback
        traceback.print_exc()

def test_synthetic_data_generation():
    """合成データ生成テスト"""
    print("\n--- 合成データ生成テスト ---")
    
    try:
        from visualization.data_aggregator import VisualizationDataAggregator
        
        aggregator = VisualizationDataAggregator(symbol="USDJPY", period_days=30)
        
        # 合成価格データテスト
        print("合成価格データ生成テスト...")
        price_data = aggregator._generate_synthetic_price_data()
        
        if not price_data.empty:
            print(f"[OK] 合成価格データ生成成功: {len(price_data)} レコード")
            print(f"  カラム: {list(price_data.columns)}")
            print(f"  期間: {price_data.index.min().strftime('%Y-%m-%d')} - {price_data.index.max().strftime('%Y-%m-%d')}")
            print(f"  価格範囲: {price_data['Close'].min():.2f} - {price_data['Close'].max():.2f}")
        else:
            print("[ERROR] 合成価格データ生成失敗")
            
        # 合成トレンドデータテスト
        print("合成トレンドデータ生成テスト...")
        trend_data = aggregator._generate_synthetic_trend_data()
        
        if not trend_data.empty:
            print(f"[OK] 合成トレンドデータ生成成功: {len(trend_data)} レコード")
            print(f"  カラム: {list(trend_data.columns)}")
            trend_counts = trend_data['trend_type'].value_counts()
            print(f"  トレンド分布: {dict(trend_counts)}")
        else:
            print("[ERROR] 合成トレンドデータ生成失敗")
            
        # 完全合成データテスト
        print("完全合成データ生成テスト...")
        complete_data = aggregator._generate_complete_synthetic_data()
        
        if not complete_data.empty:
            print(f"[OK] 完全合成データ生成成功: {len(complete_data)} レコード")
            print(f"  カラム数: {len(complete_data.columns)}")
            print(f"  カラム: {list(complete_data.columns)}")
            
            # データサマリー
            aggregator.aggregated_data = complete_data
            summary = aggregator.get_data_summary()
            print(f"  データ品質スコア: {summary.get('data_quality_score', 'N/A')}")
            
        else:
            print("[ERROR] 完全合成データ生成失敗")
            
    except Exception as e:
        print(f"[ERROR] 合成データ生成テストエラー: {e}")
        import traceback
        traceback.print_exc()

def test_chart_generation():
    """チャート生成テスト"""
    print("\n--- チャート生成テスト ---")
    
    try:
        # 出力ディレクトリ作成
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        from visualization.trend_strategy_time_series import TrendStrategyTimeSeriesVisualizer
        
        # 可視化エンジン初期化
        visualizer = TrendStrategyTimeSeriesVisualizer(
            symbol="USDJPY",
            period_days=30,
            output_dir="test_outputs"
        )
        
        print("チャート生成を開始...")
        
        # チャート生成
        output_path = visualizer.generate_comprehensive_chart(save_file=True)
        
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"[OK] チャート生成成功!")
            print(f"  保存先: {output_path}")
            print(f"  ファイルサイズ: {file_size:,} bytes")
            
            # メタデータ確認
            metadata = visualizer.get_chart_metadata()
            if 'data_summary' in metadata:
                data_summary = metadata['data_summary']
                print(f"  データ品質: {data_summary.get('data_quality_score', 'N/A')}")
                print(f"  レコード数: {data_summary.get('record_count', 'N/A')}")
                
            # 画像ファイルの拡張子確認
            if output_path.endswith('.png'):
                print("  [OK] PNG形式で正常に保存")
            else:
                print(f"  [WARNING] 予期しないファイル形式: {Path(output_path).suffix}")
                
        else:
            print("[ERROR] チャート生成失敗")
            if output_path:
                print(f"  出力パス: {output_path}")
                print(f"  ファイル存在: {os.path.exists(output_path)}")
        
    except Exception as e:
        print(f"[ERROR] チャート生成テストエラー: {e}")
        import traceback
        traceback.print_exc()

def test_chart_config():
    """チャート設定テスト"""
    print("\n--- チャート設定テスト ---")
    
    try:
        from visualization.chart_config import ChartConfigManager
        
        config = ChartConfigManager()
        
        # 基本設定確認
        print("基本設定:")
        print(f"  図サイズ: {config.figure_config['figsize']}")
        print(f"  DPI: {config.figure_config['dpi']}")
        print(f"  背景色: {config.figure_config['facecolor']}")
        
        # 色設定確認
        print("\nトレンド色設定:")
        trend_types = ['uptrend', 'downtrend', 'sideways']
        for trend in trend_types:
            color = config.get_trend_color(trend)
            print(f"  {trend}: {color}")
        
        print("\n戦略色設定:")
        strategies = ['trend_following', 'momentum', 'mean_reversion', 'breakout', 'hybrid']
        for strategy in strategies:
            color = config.get_strategy_color(strategy)
            print(f"  {strategy}: {color}")
        
        # 信頼度色マップ
        print("\n信頼度色設定:")
        confidence_map = config.get_confidence_color_map()
        for level, color in confidence_map.items():
            print(f"  {level}: {color}")
        
        print("[OK] チャート設定テスト成功")
        
    except Exception as e:
        print(f"[ERROR] チャート設定テストエラー: {e}")
        import traceback
        traceback.print_exc()

def test_performance():
    """パフォーマンステスト"""
    print("\n--- パフォーマンステスト ---")
    
    try:
        import time
        from visualization.trend_strategy_time_series import TrendStrategyTimeSeriesVisualizer
        
        print("パフォーマンステスト開始...")
        
        start_time = time.time()
        
        # 可視化エンジン初期化
        visualizer = TrendStrategyTimeSeriesVisualizer(
            symbol="USDJPY",
            period_days=60,  # 少し長い期間でテスト
            output_dir="test_outputs"
        )
        
        # チャート生成
        output_path = visualizer.generate_comprehensive_chart(save_file=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"処理時間: {processing_time:.2f} 秒")
        
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"出力ファイルサイズ: {file_size:,} bytes")
            
            # メタデータ取得
            metadata = visualizer.get_chart_metadata()
            if 'chart_data_shape' in metadata and metadata['chart_data_shape']:
                rows, cols = metadata['chart_data_shape']
                total_data_points = rows * cols
                print(f"処理データポイント: {total_data_points:,}")
                
                if processing_time > 0:
                    print(f"処理速度: {total_data_points / processing_time:.0f} データポイント/秒")
                
            print("[OK] パフォーマンステスト完了")
        else:
            print("[ERROR] パフォーマンステスト失敗")
            
    except Exception as e:
        print(f"[ERROR] パフォーマンステストエラー: {e}")

def main():
    """メインテスト実行"""
    try:
        test_imports()
        test_synthetic_data_generation()
        test_chart_config()
        test_chart_generation()
        test_performance()
        
        print("\n" + "="*60)
        print("[SUCCESS] 4-3-1 包括テスト完了!")
        
        # 生成されたファイル一覧
        output_dir = Path("test_outputs")
        if output_dir.exists():
            chart_files = list(output_dir.glob("*.png"))
            if chart_files:
                print(f"\n生成されたチャートファイル ({len(chart_files)}個):")
                for file in sorted(chart_files, key=lambda f: f.stat().st_mtime, reverse=True):
                    file_size = file.stat().st_size
                    print(f"  {file.name} ({file_size:,} bytes)")
        
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによりテストが中断されました")
    except Exception as e:
        print(f"\n[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4-3-1 テスト終了")

if __name__ == "__main__":
    main()
