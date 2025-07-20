"""
4-3-1 Chart Generation Test
実際のチャート生成テスト
"""

import sys
from pathlib import Path

# プロジェクトルートを取得してパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("=== 4-3-1 チャート生成テスト開始 ===")

try:
    from visualization.trend_strategy_time_series import TrendStrategyTimeSeriesVisualizer
    
    # 出力ディレクトリ作成
    output_dir = Path("test_chart_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print(f"1. 可視化エンジン初期化中...")
    visualizer = TrendStrategyTimeSeriesVisualizer(
        symbol="USDJPY",
        period_days=15,
        output_dir=str(output_dir)
    )
    print("  ✓ 可視化エンジン初期化成功")
    
    print("2. チャート生成中...")
    chart_path = visualizer.generate_comprehensive_chart(save_file=True)
    
    if chart_path:
        print(f"  ✓ チャート生成成功: {chart_path}")
        
        # ファイルが実際に存在するか確認
        if Path(chart_path).exists():
            file_size = Path(chart_path).stat().st_size
            print(f"  ✓ ファイル確認成功: サイズ {file_size:,} bytes")
        else:
            print("  ✗ ファイルが見つかりません")
    else:
        print("  ✗ チャート生成失敗")
    
    print("3. メタデータ確認中...")
    metadata = visualizer.get_chart_metadata()
    print(f"  - Symbol: {metadata.get('symbol')}")
    print(f"  - Period: {metadata.get('period_days')} days")
    print(f"  - Data Quality: {metadata.get('data_summary', {}).get('data_quality_score', 'N/A')}")
    print(f"  - Records: {metadata.get('data_summary', {}).get('record_count', 'N/A')}")
    
    print("\n=== 4-3-1 チャート生成テスト成功 ===")
    print(f"✓ 結果確認: {output_dir.absolute()}")
    
except Exception as e:
    print(f"✗ チャート生成テストエラー: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("=== 4-3-1 チャート生成テスト終了 ===")
