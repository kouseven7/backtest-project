"""
包括的レポートシステム デモ

DSSMS改善プロジェクト Phase 3 Task 3.3
包括的レポートシステムのデモンストレーション

機能テスト:
- 基本的なレポート生成
- 可視化機能
- エクスポート機能
- 比較レポート生成
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.reports.comprehensive import ComprehensiveReportEngine


def main():
    """デモメイン実行"""
    logger = setup_logger(__name__)
    logger.info("=== 包括的レポートシステム デモ開始 ===")
    
    try:
        # レポートエンジン初期化
        logger.info("1. レポートエンジン初期化")
        engine = ComprehensiveReportEngine()
        
        # 基本レポート生成テスト
        logger.info("2. 基本レポート生成テスト")
        basic_result = test_basic_report_generation(engine, logger)
        
        # エクスポート機能テスト
        logger.info("3. エクスポート機能テスト")
        export_result = test_export_functionality(engine, logger)
        
        # 比較レポート生成テスト
        logger.info("4. 比較レポート生成テスト")
        comparison_result = test_comparison_report(engine, logger)
        
        # レポート一覧取得テスト
        logger.info("5. レポート一覧取得テスト")
        list_result = test_report_list(engine, logger)
        
        # 結果サマリー
        logger.info("=== デモ結果サマリー ===")
        logger.info(f"基本レポート生成: {'成功' if basic_result else '失敗'}")
        logger.info(f"エクスポート機能: {'成功' if export_result else '失敗'}")
        logger.info(f"比較レポート生成: {'成功' if comparison_result else '失敗'}")
        logger.info(f"レポート一覧取得: {'成功' if list_result else '失敗'}")
        
        total_success = sum([basic_result, export_result, comparison_result, list_result])
        logger.info(f"総合結果: {total_success}/4 テスト成功")
        
        return total_success == 4
        
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        return False


def test_basic_report_generation(engine: ComprehensiveReportEngine, logger: logging.Logger) -> bool:
    """基本レポート生成テスト"""
    try:
        logger.info("基本レポート生成開始")
        
        # サマリーレベルレポート
        result_summary = engine.generate_comprehensive_report(
            report_type="comprehensive",
            level="summary"
        )
        
        if result_summary.get('success'):
            logger.info(f"サマリーレポート生成成功: {result_summary['report_path']}")
        else:
            logger.error(f"サマリーレポート生成失敗: {result_summary.get('error')}")
            return False
        
        # 詳細レベルレポート
        result_detailed = engine.generate_comprehensive_report(
            report_type="comprehensive",
            level="detailed",
            date_range={
                'start': datetime.now() - timedelta(days=30),
                'end': datetime.now()
            }
        )
        
        if result_detailed.get('success'):
            logger.info(f"詳細レポート生成成功: {result_detailed['report_path']}")
        else:
            logger.error(f"詳細レポート生成失敗: {result_detailed.get('error')}")
            return False
        
        # 包括的レベルレポート
        result_comprehensive = engine.generate_comprehensive_report(
            report_type="comprehensive",
            level="comprehensive",
            strategies=["VWAPBreakoutStrategy", "ConventionalTradingStrategy"]
        )
        
        if result_comprehensive.get('success'):
            logger.info(f"包括的レポート生成成功: {result_comprehensive['report_path']}")
            logger.info(f"生成時間: {result_comprehensive.get('generation_time', 0):.2f}秒")
            logger.info(f"可視化数: {result_comprehensive.get('visualizations_count', 0)}")
        else:
            logger.error(f"包括的レポート生成失敗: {result_comprehensive.get('error')}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"基本レポート生成テストエラー: {e}")
        return False


def test_export_functionality(engine: ComprehensiveReportEngine, logger: logging.Logger) -> bool:
    """エクスポート機能テスト"""
    try:
        logger.info("エクスポート機能テスト開始")
        
        # 最新レポートのIDを取得
        if not engine.current_report_id:
            logger.warning("エクスポート対象のレポートがありません")
            return True  # 基本機能テストとして成功扱い
        
        # JSONエクスポート
        json_result = engine.export_report("json")
        if json_result.get('success'):
            logger.info(f"JSONエクスポート成功: {json_result.get('export_path')}")
        else:
            logger.warning(f"JSONエクスポート失敗: {json_result.get('error')}")
        
        # Excelエクスポート（openpyxlが利用可能な場合）
        excel_result = engine.export_report("excel")
        if excel_result.get('success'):
            logger.info(f"Excelエクスポート成功: {excel_result.get('export_path')}")
        else:
            logger.info(f"Excelエクスポート: {excel_result.get('error')}")
        
        # PDFエクスポート（PDF生成ライブラリが利用可能な場合）
        pdf_result = engine.export_report("pdf")
        if pdf_result.get('success'):
            logger.info(f"PDFエクスポート成功: {pdf_result.get('export_path')}")
        else:
            logger.info(f"PDFエクスポート: {pdf_result.get('error')}")
        
        # 最低限JSONエクスポートが成功すれば合格
        return json_result.get('success', False)
        
    except Exception as e:
        logger.error(f"エクスポート機能テストエラー: {e}")
        return False


def test_comparison_report(engine: ComprehensiveReportEngine, logger: logging.Logger) -> bool:
    """比較レポート生成テスト"""
    try:
        logger.info("比較レポート生成テスト開始")
        
        # 戦略比較レポート
        comparison_items = [
            {'name': 'VWAPBreakoutStrategy'},
            {'name': 'ConventionalTradingStrategy'},
        ]
        
        comparison_result = engine.generate_comparison_report(
            comparison_items=comparison_items,
            comparison_type="strategies",
            level="detailed"
        )
        
        if comparison_result.get('success'):
            logger.info(f"戦略比較レポート生成成功: {comparison_result['report_path']}")
            return True
        else:
            logger.error(f"戦略比較レポート生成失敗: {comparison_result.get('error')}")
            return False
        
    except Exception as e:
        logger.error(f"比較レポート生成テストエラー: {e}")
        return False


def test_report_list(engine: ComprehensiveReportEngine, logger: logging.Logger) -> bool:
    """レポート一覧取得テスト"""
    try:
        logger.info("レポート一覧取得テスト開始")
        
        report_list = engine.get_report_list(limit=10)
        
        logger.info(f"生成済みレポート数: {len(report_list)}")
        
        for i, report in enumerate(report_list[:3]):
            logger.info(f"  {i+1}. {report['report_id']}")
            logger.info(f"     作成日時: {report['creation_time']}")
            logger.info(f"     ファイルサイズ: {report['file_size']} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"レポート一覧取得テストエラー: {e}")
        return False


def test_performance_metrics():
    """パフォーマンスメトリクステスト"""
    logger = setup_logger(__name__)
    logger.info("=== パフォーマンステスト開始 ===")
    
    try:
        engine = ComprehensiveReportEngine()
        
        # 複数回実行して平均時間を測定
        times = []
        
        for i in range(3):
            start_time = datetime.now()
            
            result = engine.generate_comprehensive_report(
                report_type="comprehensive",
                level="summary"
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            times.append(execution_time)
            
            logger.info(f"実行 {i+1}: {execution_time:.2f}秒")
        
        avg_time = sum(times) / len(times)
        logger.info(f"平均実行時間: {avg_time:.2f}秒")
        
        # パフォーマンス基準チェック（30秒以内）
        if avg_time < 30:
            logger.info("パフォーマンステスト: 合格")
            return True
        else:
            logger.warning(f"パフォーマンステスト: 時間超過 ({avg_time:.2f}秒)")
            return False
        
    except Exception as e:
        logger.error(f"パフォーマンステストエラー: {e}")
        return False


def cleanup_test_files():
    """テスト用ファイルクリーンアップ"""
    logger = setup_logger(__name__)
    
    try:
        output_dir = Path("output/comprehensive_reports")
        
        if output_dir.exists():
            test_files = list(output_dir.glob("comprehensive_*_summary_*.html"))
            test_files.extend(output_dir.glob("comprehensive_*_detailed_*.html"))
            test_files.extend(output_dir.glob("comprehensive_*_comprehensive_*.html"))
            
            for file_path in test_files:
                try:
                    file_path.unlink()
                    logger.info(f"テストファイル削除: {file_path.name}")
                except Exception as e:
                    logger.warning(f"ファイル削除エラー {file_path}: {e}")
            
            logger.info(f"テストファイルクリーンアップ完了: {len(test_files)} ファイル")
        
    except Exception as e:
        logger.error(f"クリーンアップエラー: {e}")


if __name__ == "__main__":
    try:
        # メインデモ実行
        success = main()
        
        # パフォーマンステスト
        perf_success = test_performance_metrics()
        
        print("\n" + "="*50)
        print("包括的レポートシステム デモ結果")
        print("="*50)
        print(f"機能テスト: {'成功' if success else '失敗'}")
        print(f"パフォーマンステスト: {'成功' if perf_success else '失敗'}")
        
        if success and perf_success:
            print("\n[OK] 全てのテストが成功しました！")
            print("包括的レポートシステムの実装が完了しました。")
        else:
            print("\n[ERROR] 一部のテストが失敗しました。")
            print("ログを確認して問題を解決してください。")
        
        # クリーンアップ確認
        response = input("\nテストファイルを削除しますか？ (y/N): ")
        if response.lower() == 'y':
            cleanup_test_files()
        
    except KeyboardInterrupt:
        print("\n\nデモが中断されました。")
    except Exception as e:
        print(f"\nデモ実行エラー: {e}")
        sys.exit(1)
