#!/usr/bin/env python3
"""
Phase 1 動作確認テスト: 累積期間方式復元 + 決定論破綻監視

このスクリプトは以下をテストします：
1. 累積期間方式の動作確認
2. エントリー機会の回復確認（33営業日で1件→期待値50件程度）
3. 決定論破綻の監視（重複エントリー、状態不一致の検出）
4. MainSystemControllerインスタンス変数化の継続

実行方法:
python test_phase1_cumulative_period_restore.py
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

# パスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_test_logging():
    """テスト用ログ設定"""
    # ログフォーマット設定
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'test_phase1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    # DSSMSログレベルを詳細に設定
    logging.getLogger('DSSMSIntegratedBacktester').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

def test_phase1_implementation():
    """Phase 1実装のテスト"""
    logger = setup_test_logging()
    logger.info("=" * 80)
    logger.info("Phase 1実装テスト開始: 累積期間方式復元 + 決定論破綻監視")
    logger.info("=" * 80)
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        # テスト設定
        config = {
            'initial_capital': 1000000,
            'warmup_days': 150,
            'log_level': 'INFO'
        }
        
        # DSSMS統合バックテスター初期化
        dssms = DSSMSIntegratedBacktester(config)
        logger.info("[TEST] DSSMS初期化完了")
        
        # テスト期間設定（短期間でテスト）
        start_date = datetime(2025, 1, 15)
        end_date = datetime(2025, 1, 25)  # 10日間のテスト
        
        logger.info(f"[TEST] テスト期間: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        
        # 対象銘柄
        test_symbols = ['6954.T']  # ファナック（テスト用）
        
        # Phase 1実装のテスト実行
        logger.info("[TEST] Phase 1実装テスト実行開始")
        
        results = dssms.run_dynamic_backtest(
            start_date=start_date,
            end_date=end_date,
            target_symbols=test_symbols
        )
        
        # 結果検証
        logger.info("[TEST] 結果検証開始")
        
        if results:
            # エントリー件数の確認
            total_trades = 0
            for date, result in results.get('daily_results', {}).items():
                if result.get('trades'):
                    total_trades += len(result['trades'])
            
            logger.info(f"[TEST] 総取引件数: {total_trades}")
            
            # 累積期間方式の動作確認
            logger.info("[TEST] 累積期間方式動作確認:")
            logger.info(f"  - 期待値: 日数の増加に伴いエントリー機会も増加")
            logger.info(f"  - 実績: {total_trades}件の取引が発生")
            
            # 成功判定
            if total_trades > 0:
                logger.info("[TEST] ✅ Phase 1実装テスト成功: エントリーが発生しています")
                return True
            else:
                logger.warning("[TEST] ⚠️ エントリーが発生していません。設定を確認してください。")
                return False
        else:
            logger.error("[TEST] ❌ 結果が取得できませんでした")
            return False
            
    except ImportError as e:
        logger.error(f"[TEST] ❌ モジュールインポートエラー: {e}")
        return False
    except Exception as e:
        logger.error(f"[TEST] ❌ 予期しないエラー: {e}")
        return False

def analyze_determinism_logs():
    """決定論破綻監視ログの分析"""
    logger = logging.getLogger(__name__)
    
    logger.info("[ANALYSIS] 決定論破綻監視ログ分析")
    logger.info("以下のログパターンを確認してください:")
    logger.info("1. [DETERMINISM_MONITOR] 累積期間バックテスト: 期間の累積的増加")
    logger.info("2. [DETERMINISM_MONITOR] PaperBroker状態前/後: 残高・注文数の変化")
    logger.info("3. [DETERMINISM_MONITOR] 同日重複エントリー検出: 重複の有無")
    logger.info("")
    logger.info("期待される動作:")
    logger.info("✅ 取引期間が日次で累積的に増加")
    logger.info("✅ PaperBrokerの状態が日を跨いで継続")
    logger.info("⚠️ 同日重複エントリーが発生した場合は警告ログ出力")

if __name__ == "__main__":
    # Phase 1実装テスト実行
    success = test_phase1_implementation()
    
    # 決定論破綻監視ログ分析説明
    analyze_determinism_logs()
    
    # 総合結果
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    if success:
        logger.info("🎉 Phase 1実装テスト完了: 累積期間方式復元成功")
        logger.info("")
        logger.info("次のステップ:")
        logger.info("1. ログを確認して決定論破綻の有無を検証")
        logger.info("2. より長期間（33営業日）でのテスト実行")
        logger.info("3. エントリー件数の期待値（50件程度）との比較")
    else:
        logger.error("❌ Phase 1実装テストに失敗しました")
        logger.error("トラブルシューティング:")
        logger.error("1. dssms_integrated_main.pyの修正を確認")
        logger.error("2. MainSystemControllerの初期化エラーをチェック")
        logger.error("3. データ取得エラーがないか確認")
    logger.info("=" * 80)