"""
Phase 1ログ強化テスト - P1-1/P1-2検証用

DSSMS経由でmain_new.pyを呼び出し、
P1-1: DSSMS→main_new_DATA ログ
P1-2: DATA_RANGE_CHECK ログ
が正しく出力されるかを検証

Author: Backtest Project Team
Created: 2025-12-01
"""
import sys
from pathlib import Path
from datetime import datetime

# プロジェクトルート追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
logger = setup_logger("test_phase1_logs", log_file="logs/test_phase1_logs.log")

def test_dssms_main_new_integration():
    """DSSMS→main_new.py統合テスト（P1-1/P1-2ログ検証）"""
    logger.info("=" * 80)
    logger.info("Phase 1ログ強化テスト開始")
    logger.info("=" * 80)
    
    try:
        # DSSMS統合バックテスター初期化
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        logger.info("DSSMSIntegratedBacktester初期化中...")
        backtester = DSSMSIntegratedBacktester()
        
        # テスト用期間設定（短期間）
        start_date = datetime(2024, 1, 10)
        end_date = datetime(2024, 1, 15)
        
        logger.info(f"バックテスト実行: {start_date} ~ {end_date}")
        
        # バックテスト実行（P1-1/P1-2ログが出力されるはず）
        results = backtester.run_dynamic_backtest(
            start_date=start_date,
            end_date=end_date,
            target_symbols=None  # 全銘柄（実際にはスクリーニングされた銘柄のみ）
        )
        
        logger.info("=" * 80)
        logger.info("バックテスト完了")
        logger.info(f"ステータス: {results.get('status', 'UNKNOWN')}")
        logger.info("=" * 80)
        
        # ログ検証
        logger.info("ログ検証開始...")
        logger.info("手動確認: logs/dssms_integrated_main.log で以下を確認してください:")
        logger.info("  - [DSSMS->main_new_DATA] 銘柄:")
        logger.info("  - [DSSMS->main_new_DATA] 対象日:")
        logger.info("  - [DSSMS->main_new_DATA] stock_data範囲:")
        logger.info("")
        logger.info("手動確認: logs/main_system_controller.log で以下を確認してください:")
        logger.info("  - [DATA_RANGE_CHECK] データ最終日:")
        logger.info("  - [DATA_RANGE_CHECK] 取引開始日:")
        logger.info("  - [DATA_INSUFFICIENT] (データ不足時のみ)")
        
        return results
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test_dssms_main_new_integration()
