"""
Resolution 19: ランキングシステム診断実行スクリプト

DSSMSランキングシステムの包括的診断を実行し、
top_symbol=None問題の根本原因を特定します。

Author: AI Assistant
Created: 2025-01-25
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 必要なモジュールをインポート
try:
    from config.logger_config import setup_logger
    from src.dssms.dssms_backtester import DSSMSBacktester
    from src.dssms.ranking_diagnostics import RankingSystemDiagnostics
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("プロジェクトルートから実行してください")
    sys.exit(1)


def main():
    """Resolution 19診断実行メイン関数"""
    print("=" * 60)
    print("Resolution 19: DSSMSランキングシステム診断実行")
    print("=" * 60)
    
    # ロガー設定
    logger = setup_logger('resolution19_diagnostic')
    logger.info("Resolution 19診断開始")
    
    try:
        # 診断対象パラメータ
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30日間のテスト期間
        
        # サンプル銘柄リスト（Nikkei 225から選択）
        test_symbols = [
            '7203.T', '6758.T', '9984.T', '6861.T', '8058.T',  # トヨタ、ソニー、SB、キーエンス、三菱商事
            '6098.T', '4063.T', '9432.T', '4519.T', '8035.T'   # リクルート、信越化学、NTT、中外製薬、東京エレクトロン
        ]
        
        print(f"診断期間: {start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}")
        print(f"対象銘柄: {len(test_symbols)}銘柄")
        print(f"銘柄リスト: {', '.join(test_symbols[:5])}...")
        
        # DSSMSBacktester初期化
        print("\n1. DSSMSBacktester初期化...")
        backtester_config = {
            'enable_ranking_diagnostics': True,
            'deterministic_mode': True,
            'cache_optimization': False  # 診断期間中はキャッシュ無効
        }
        
        backtester = DSSMSBacktester(config=backtester_config)
        logger.info("DSSMSBacktester初期化完了")
        
        # 診断システムアクセス確認
        if not hasattr(backtester, 'ranking_diagnostics') or backtester.ranking_diagnostics is None:
            print("[WARNING]  警告: ランキング診断システムが初期化されていません")
            logger.error("ランキング診断システム初期化失敗")
            return
        
        print("[OK] ランキング診断システム初期化完了")
        
        # 2. 複数日付での診断実行
        print("\n2. 複数日付での診断実行...")
        diagnostic_results = []
        
        # 直近5日間での診断（営業日考慮）
        test_dates = []
        current_date = end_date
        while len(test_dates) < 5:
            # 平日のみテスト（簡易版）
            if current_date.weekday() < 5:  # 月曜=0, 金曜=4
                test_dates.append(current_date)
            current_date -= timedelta(days=1)
        
        for i, test_date in enumerate(test_dates):
            print(f"\n  📅 診断 {i+1}/5: {test_date.strftime('%Y-%m-%d')}")
            
            try:
                # ランキング更新実行（診断付き）
                ranking_result = backtester._update_symbol_ranking(test_date, test_symbols)
                
                # 結果確認
                top_symbol = ranking_result.get('top_symbol')
                diagnostic_info = ranking_result.get('diagnostic_info', {})
                
                print(f"    [TARGET] top_symbol: {top_symbol}")
                print(f"    [CHART] 診断成功: {diagnostic_info.get('pipeline_success', False)}")
                print(f"    ⏱️  実行時間: {diagnostic_info.get('total_duration_ms', 0):.1f}ms")
                
                # 診断結果を保存
                diagnostic_results.append({
                    'date': test_date.isoformat(),
                    'top_symbol': top_symbol,
                    'ranking_result': ranking_result,
                    'diagnostic_successful': diagnostic_info.get('pipeline_success', False)
                })
                
                # top_symbol=None問題の検出
                if top_symbol is None:
                    print(f"    [WARNING]  問題検出: top_symbol=None")
                    logger.warning(f"top_symbol=None検出: {test_date}")
                
            except Exception as e:
                print(f"    [ERROR] エラー: {str(e)}")
                logger.error(f"診断エラー {test_date}: {e}")
        
        # 3. 診断結果分析
        print("\n3. 診断結果分析...")
        none_count = sum(1 for result in diagnostic_results if result['top_symbol'] is None)
        success_count = sum(1 for result in diagnostic_results if result['diagnostic_successful'])
        
        print(f"   [UP] 統計:")
        print(f"     - 総診断回数: {len(diagnostic_results)}")
        print(f"     - top_symbol=None発生: {none_count}回")
        print(f"     - 診断パイプライン成功: {success_count}回")
        print(f"     - top_symbol=None発生率: {none_count/len(diagnostic_results)*100:.1f}%")
        
        # 4. 詳細診断レポート生成
        print("\n4. 詳細診断レポート生成...")
        
        if backtester.ranking_diagnostics.diagnostic_history:
            report_path = f"output/resolution19_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report = backtester.ranking_diagnostics.generate_diagnostic_report(report_path)
            
            print(f"   📄 詳細レポート生成: {report_path}")
            
            # レポート概要表示
            summary = report.get('diagnostic_summary', {})
            recommendations = report.get('recommendations', [])
            
            print(f"   [CHART] 概要:")
            print(f"     - 成功率: {summary.get('success_rate', 0):.1%}")
            print(f"     - 平均実行時間: {summary.get('average_duration_ms', 0):.1f}ms")
            print(f"     - None発生回数: {summary.get('none_top_symbol_count', 0)}")
            
            print(f"\n   [IDEA] 推奨事項:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"     {i}. {rec}")
        
        # 5. 結論と次ステップ
        print("\n" + "=" * 60)
        print("[SEARCH] Resolution 19診断結果")
        print("=" * 60)
        
        if none_count > 0:
            print(f"[ERROR] 問題確認: top_symbol=None が {none_count}/{len(diagnostic_results)} 回発生")
            print("[TOOL] 次ステップ: ランキングシステム修復が必要")
            
            # 主要なエラーパターンを特定
            error_patterns = {}
            for result in diagnostic_results:
                if result['top_symbol'] is None:
                    ranking_data = result.get('ranking_result', {})
                    error_key = ranking_data.get('data_source', 'unknown')
                    error_patterns[error_key] = error_patterns.get(error_key, 0) + 1
            
            print("\n[TARGET] エラーパターン分析:")
            for pattern, count in error_patterns.items():
                print(f"   - {pattern}: {count}回")
            
        else:
            print("[OK] 問題なし: すべての診断でtop_symbolが正常に取得されました")
            print("👍 ランキングシステムは正常に動作しています")
        
        logger.info("Resolution 19診断完了")
        
    except Exception as e:
        print(f"[ERROR] 診断実行エラー: {str(e)}")
        logger.error(f"診断実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()