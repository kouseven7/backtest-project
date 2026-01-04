#!/usr/bin/env python3
"""
DSSMS バックテスト問題調査専用デバッグスクリプト

Purpose:
- _get_optimal_symbol の例外内容詳細確認
- DSS Core V3 の run_daily_selection 実行状況確認
- 条件分岐の実行パス確認

Author: Investigation Team
Created: 2026-01-03
"""

import sys
import os
import traceback
import logging
from datetime import datetime
from pathlib import Path

# プロジェクトルートを追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('investigation_dssms_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def test_dssms_integrated_main_debug():
    """DSSMS統合メイン処理のデバッグテスト"""
    logger.info("=== DSSMS統合メイン処理デバッグテスト開始 ===")
    
    try:
        # DSSMSIntegratedBacktesterをインポート
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        logger.info("DSSMSIntegratedBacktester import成功")
        
        # インスタンス作成
        logger.info("DSSMSIntegratedBacktester インスタンス作成開始")
        backtest_instance = DSSMSIntegratedBacktester()
        logger.info("DSSMSIntegratedBacktester インスタンス作成完了")
        
        # テスト日時
        target_date = datetime(2025, 1, 15)
        logger.info(f"テスト対象日: {target_date}")
        
        # _get_optimal_symbol メソッドの詳細デバッグ実行
        logger.info("=== _get_optimal_symbol デバッグ実行開始 ===")
        
        try:
            # コンポーネント状態確認
            logger.info("コンポーネント状態確認:")
            logger.info(f"  - dss_core: {backtest_instance.dss_core}")
            logger.info(f"  - nikkei225_screener: {backtest_instance.nikkei225_screener}")
            logger.info(f"  - advanced_ranking_engine: {backtest_instance.advanced_ranking_engine}")
            
            # DSS Core V3利用可能性チェック
            try:
                from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
                dss_available = True
                logger.info("DSS Core V3 import成功 - dss_available=True")
            except ImportError as e:
                dss_available = False
                logger.error(f"DSS Core V3 import失敗 - dss_available=False: {e}")
            
            # DSS Core V3の処理をテスト
            if backtest_instance.dss_core and dss_available:
                logger.info("DSS Core V3による選択処理開始")
                try:
                    dss_result = backtest_instance.dss_core.run_daily_selection(target_date)
                    logger.info(f"DSS Core V3実行成功: {dss_result}")
                    selected_symbol = dss_result.get('selected_symbol')
                    logger.info(f"DSS選択結果: {selected_symbol}")
                    
                    if selected_symbol:
                        logger.info("DSS Core V3からの銘柄選択成功")
                        return selected_symbol
                    else:
                        logger.warning("DSS Core V3からの銘柄選択結果がNone")
                        
                except Exception as dss_e:
                    logger.error(f"DSS Core V3実行エラー: {dss_e}")
                    logger.error(f"DSS Core V3実行エラー詳細:\n{traceback.format_exc()}")
            else:
                logger.info(f"DSS Core V3使用不可: dss_core={backtest_instance.dss_core}, dss_available={dss_available}")
            
            # Nikkei225Screener処理をテスト
            if backtest_instance.nikkei225_screener:
                logger.info("Nikkei225Screener処理開始")
                try:
                    available_funds = backtest_instance.portfolio_value * 0.8
                    logger.info(f"利用可能資金: {available_funds}")
                    
                    filtered_symbols = backtest_instance.nikkei225_screener.get_filtered_symbols(available_funds)
                    logger.info(f"フィルタリング済み銘柄数: {len(filtered_symbols)}")
                    logger.info(f"フィルタリング済み銘柄（最初の10個）: {filtered_symbols[:10] if len(filtered_symbols) > 10 else filtered_symbols}")
                    
                    if filtered_symbols:
                        # Advanced Ranking Selection をテスト
                        logger.info("Advanced Ranking Selection テスト開始")
                        try:
                            selected = backtest_instance._advanced_ranking_selection(filtered_symbols, target_date)
                            logger.info(f"Advanced Ranking Selection成功: {selected}")
                            return selected
                        except Exception as ranking_e:
                            logger.error(f"Advanced Ranking Selection エラー: {ranking_e}")
                            logger.error(f"Advanced Ranking Selection エラー詳細:\n{traceback.format_exc()}")
                    else:
                        logger.warning("フィルタリング済み銘柄が空")
                        
                except Exception as screener_e:
                    logger.error(f"Nikkei225Screener エラー: {screener_e}")
                    logger.error(f"Nikkei225Screener エラー詳細:\n{traceback.format_exc()}")
            else:
                logger.error("Nikkei225Screener が None")
            
            logger.warning("すべての銘柄選択方法が失敗")
            return None
            
        except Exception as e:
            logger.error(f"_get_optimal_symbol デバッグ実行エラー: {e}")
            logger.error(f"_get_optimal_symbol デバッグ実行エラー詳細:\n{traceback.format_exc()}")
            return None
            
    except Exception as e:
        logger.error(f"テスト全体エラー: {e}")
        logger.error(f"テスト全体エラー詳細:\n{traceback.format_exc()}")
        return None

def test_dss_core_v3_direct():
    """DSS Core V3 直接テスト"""
    logger.info("=== DSS Core V3 直接テスト開始 ===")
    
    try:
        from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
        logger.info("DSS Core V3 import成功")
        
        # インスタンス作成
        logger.info("DSS Core V3 インスタンス作成開始")
        dss_core = DSSBacktesterV3()
        logger.info("DSS Core V3 インスタンス作成完了")
        
        # テスト日時
        target_date = datetime(2025, 1, 15)
        logger.info(f"テスト対象日: {target_date}")
        
        # run_daily_selection 実行
        logger.info("DSS Core V3 run_daily_selection実行開始")
        result = dss_core.run_daily_selection(target_date)
        logger.info(f"DSS Core V3 run_daily_selection実行完了: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"DSS Core V3 直接テストエラー: {e}")
        logger.error(f"DSS Core V3 直接テストエラー詳細:\n{traceback.format_exc()}")
        return None

def main():
    """メイン実行"""
    logger.info("=== DSSMS問題調査デバッグ実行開始 ===")
    
    # Test 1: DSSMS統合メイン処理デバッグ
    logger.info("\n" + "="*60)
    logger.info("Test 1: DSSMS統合メイン処理デバッグ")
    logger.info("="*60)
    result1 = test_dssms_integrated_main_debug()
    logger.info(f"Test 1結果: {result1}")
    
    # Test 2: DSS Core V3 直接テスト
    logger.info("\n" + "="*60)
    logger.info("Test 2: DSS Core V3 直接テスト")
    logger.info("="*60)
    result2 = test_dss_core_v3_direct()
    logger.info(f"Test 2結果: {result2}")
    
    logger.info("=== DSSMS問題調査デバッグ実行完了 ===")

if __name__ == "__main__":
    main()