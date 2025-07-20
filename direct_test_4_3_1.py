"""
4-3-1 Direct Test Script
直接テスト実行
"""
import sys
import os
from pathlib import Path

# プロジェクトルートを取得してパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("=== 4-3-1 直接テスト開始 ===")

def test_direct_import():
    """直接インポートテスト"""
    print("\n1. 直接インポートテスト")
    
    try:
        # chart_configの直接テスト
        print("chart_config テスト...")
        sys.path.append(str(project_root / "visualization"))
        
        import chart_config
        config_manager = chart_config.ChartConfigManager()
        print(f"✅ ChartConfigManager 作成成功")
        print(f"  図サイズ: {config_manager.figure_config.get('figsize', 'N/A')}")
        
        # トレンド色テスト
        uptrend_color = config_manager.get_trend_color('uptrend')
        print(f"  上昇トレンド色: {uptrend_color}")
        
    except Exception as e:
        print(f"❌ chart_config エラー: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # data_aggregatorの直接テスト
        print("\ndata_aggregator テスト...")
        import data_aggregator
        aggregator = data_aggregator.VisualizationDataAggregator(symbol="USDJPY", period_days=10)
        print(f"✅ VisualizationDataAggregator 作成成功")
        
        # 合成データ生成テスト
        print("合成価格データ生成テスト...")
        price_data = aggregator._generate_synthetic_price_data()
        print(f"✅ 合成価格データ生成成功: {len(price_data)} レコード")
        print(f"  カラム: {list(price_data.columns)}")
        
        # 完全合成データテスト
        print("完全合成データ生成テスト...")
        complete_data = aggregator._generate_complete_synthetic_data()
        print(f"✅ 完全合成データ生成成功: {len(complete_data)} レコード")
        print(f"  カラム数: {len(complete_data.columns)}")
        
    except Exception as e:
        print(f"❌ data_aggregator エラー: {e}")
        import traceback
        traceback.print_exc()

def test_chart_creation():
    """チャート作成テスト"""
    print("\n2. チャート作成テスト")
    
    try:
        # trend_strategy_time_seriesの直接テスト
        import trend_strategy_time_series
        
        # 出力ディレクトリ作成
        output_dir = Path("direct_test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 可視化エンジン作成
        visualizer = trend_strategy_time_series.TrendStrategyTimeSeriesVisualizer(
            symbol="USDJPY",
            period_days=10,  # 短期間でテスト
            output_dir="direct_test_outputs"
        )
        print(f"✅ TrendStrategyTimeSeriesVisualizer 作成成功")
        
        # チャート生成テスト
        print("チャート生成中...")
        output_path = visualizer.generate_comprehensive_chart(save_file=True)
        
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ チャート生成成功!")
            print(f"  保存先: {output_path}")
            print(f"  ファイルサイズ: {file_size:,} bytes")
            
            # メタデータ取得
            metadata = visualizer.get_chart_metadata()
            print(f"  シンボル: {metadata.get('symbol', 'N/A')}")
            print(f"  期間: {metadata.get('period_days', 'N/A')}日")
            
        else:
            print("❌ チャート生成失敗")
            if output_path:
                print(f"  出力パス: {output_path}")
                print(f"  ファイル存在: {os.path.exists(output_path)}")
        
    except Exception as e:
        print(f"❌ チャート作成エラー: {e}")
        import traceback
        traceback.print_exc()

def test_file_output():
    """ファイル出力確認テスト"""
    print("\n3. ファイル出力確認テスト")
    
    try:
        output_dir = Path("direct_test_outputs")
        if output_dir.exists():
            files = list(output_dir.glob("*.png"))
            print(f"生成されたPNGファイル数: {len(files)}")
            
            for file in files:
                file_size = file.stat().st_size
                print(f"  {file.name}: {file_size:,} bytes")
        else:
            print("出力ディレクトリが存在しません")
    
    except Exception as e:
        print(f"❌ ファイル出力確認エラー: {e}")

def main():
    try:
        test_direct_import()
        test_chart_creation() 
        test_file_output()
        
        print("\n" + "="*50)
        print("🎉 4-3-1 直接テスト完了!")
        
    except Exception as e:
        print(f"\n❌ 直接テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
