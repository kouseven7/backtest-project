"""
4-3-1 Trend Strategy Time Series Visualization Demo

トレンド変化と戦略切替の時系列グラフシステムのデモンストレーション
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# プロジェクトルートを取得してパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from visualization import TrendStrategyTimeSeriesVisualizer


def setup_demo_logging():
    """デモ用ロガー設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_trend_strategy_visualization.log')
        ]
    )
    return logging.getLogger("TrendStrategyDemo")


def run_basic_demo():
    """基本デモ実行"""
    logger = setup_demo_logging()
    logger.info("=== 4-3-1 基本デモ開始 ===")
    
    try:
        # 基本的な30日間チャート
        logger.info("30日間のUSDJPYチャートを生成中...")
        visualizer = TrendStrategyTimeSeriesVisualizer(
            symbol="USDJPY",
            period_days=30,
            output_dir="demo_outputs"
        )
        
        chart_path = visualizer.generate_comprehensive_chart(save_file=True)
        
        if chart_path:
            logger.info(f"✓ 基本チャート生成成功: {chart_path}")
            
            # メタデータ表示
            metadata = visualizer.get_chart_metadata()
            logger.info(f"チャートメタデータ: {metadata}")
            
        else:
            logger.error("✗ 基本チャート生成失敗")
            
    except Exception as e:
        logger.error(f"基本デモエラー: {e}")
    
    logger.info("=== 4-3-1 基本デモ終了 ===")


def run_period_comparison_demo():
    """期間比較デモ実行"""
    logger = setup_demo_logging()
    logger.info("=== 4-3-1 期間比較デモ開始 ===")
    
    try:
        visualizer = TrendStrategyTimeSeriesVisualizer(
            symbol="USDJPY",
            period_days=30,  # 初期値
            output_dir="demo_outputs"
        )
        
        # 複数期間の比較チャート生成
        logger.info("複数期間比較チャートを生成中...")
        comparison_results = visualizer.generate_period_comparison([30, 60, 90])
        
        if comparison_results:
            logger.info("✓ 期間比較チャート生成成功:")
            for period, path in comparison_results.items():
                logger.info(f"  - {period}: {path}")
        else:
            logger.warning("期間比較チャート生成結果が空です")
            
    except Exception as e:
        logger.error(f"期間比較デモエラー: {e}")
    
    logger.info("=== 4-3-1 期間比較デモ終了 ===")


def run_multi_symbol_demo():
    """複数シンボルデモ実行"""
    logger = setup_demo_logging()
    logger.info("=== 4-3-1 複数シンボルデモ開始 ===")
    
    symbols = ["USDJPY", "EURJPY", "GBPJPY"]
    
    for symbol in symbols:
        try:
            logger.info(f"{symbol}のチャートを生成中...")
            
            visualizer = TrendStrategyTimeSeriesVisualizer(
                symbol=symbol,
                period_days=30,
                output_dir="demo_outputs"
            )
            
            chart_path = visualizer.generate_comprehensive_chart(save_file=True)
            
            if chart_path:
                logger.info(f"✓ {symbol}チャート生成成功: {chart_path}")
            else:
                logger.warning(f"✗ {symbol}チャート生成失敗")
                
        except Exception as e:
            logger.error(f"{symbol}デモエラー: {e}")
    
    logger.info("=== 4-3-1 複数シンボルデモ終了 ===")


def run_data_quality_demo():
    """データ品質テストデモ"""
    logger = setup_demo_logging()
    logger.info("=== 4-3-1 データ品質デモ開始 ===")
    
    try:
        visualizer = TrendStrategyTimeSeriesVisualizer(
            symbol="USDJPY",
            period_days=60,
            output_dir="demo_outputs"
        )
        
        # データ集約とチャート生成
        logger.info("データ品質チェック付きチャート生成中...")
        chart_path = visualizer.generate_comprehensive_chart(save_file=True)
        
        if chart_path:
            # データサマリー取得
            data_summary = visualizer.data_aggregator.get_data_summary()
            logger.info("データ品質サマリー:")
            logger.info(f"  - ステータス: {data_summary.get('status', 'unknown')}")
            logger.info(f"  - レコード数: {data_summary.get('record_count', 0)}")
            logger.info(f"  - 品質スコア: {data_summary.get('data_quality_score', 0.0):.2f}")
            logger.info(f"  - 日付範囲: {data_summary.get('date_range', {})}")
            logger.info(f"  - トレンド分布: {data_summary.get('trend_distribution', {})}")
            logger.info(f"  - 戦略分布: {data_summary.get('strategy_distribution', {})}")
            
            # データエクスポート
            export_path = "demo_outputs/aggregated_data_sample.csv"
            if visualizer.data_aggregator.export_aggregated_data(export_path):
                logger.info(f"✓ データエクスポート成功: {export_path}")
            
        else:
            logger.error("データ品質チェックデモ失敗")
            
    except Exception as e:
        logger.error(f"データ品質デモエラー: {e}")
    
    logger.info("=== 4-3-1 データ品質デモ終了 ===")


def run_cleanup_demo():
    """クリーンアップデモ"""
    logger = setup_demo_logging()
    logger.info("=== 4-3-1 クリーンアップデモ開始 ===")
    
    try:
        visualizer = TrendStrategyTimeSeriesVisualizer(
            symbol="USDJPY",
            period_days=30,
            output_dir="demo_outputs"
        )
        
        # 古いファイルのクリーンアップ
        logger.info("古いチャートファイルをクリーンアップ中...")
        removed_count = visualizer.cleanup_old_files(keep_recent=5)
        
        logger.info(f"✓ クリーンアップ完了: {removed_count}個のファイルを削除")
        
    except Exception as e:
        logger.error(f"クリーンアップデモエラー: {e}")
    
    logger.info("=== 4-3-1 クリーンアップデモ終了 ===")


def main():
    """メインデモ実行"""
    print("=" * 60)
    print("4-3-1 トレンド変化と戦略切替の時系列グラフ デモシステム")
    print("=" * 60)
    
    # 出力ディレクトリ作成
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. 基本デモ
        run_basic_demo()
        
        # 2. 期間比較デモ
        run_period_comparison_demo()
        
        # 3. 複数シンボルデモ
        run_multi_symbol_demo()
        
        # 4. データ品質デモ
        run_data_quality_demo()
        
        # 5. クリーンアップデモ
        run_cleanup_demo()
        
        print("\n" + "=" * 60)
        print("✓ 全デモ実行完了")
        print(f"結果確認: {output_dir.absolute()}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nデモが中断されました")
    except Exception as e:
        print(f"デモ実行エラー: {e}")


if __name__ == "__main__":
    main()
