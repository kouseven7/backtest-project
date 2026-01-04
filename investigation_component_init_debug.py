#!/usr/bin/env python3
"""
DSSMS コンポーネント初期化例外詳細調査スクリプト

Purpose:
- 各初期化メソッドの例外内容詳細確認
- 初期化フラグ設定タイミングの詳細調査
- 実際の例外スタックトレース取得

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
        logging.FileHandler('investigation_component_init_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def test_component_initialization_detailed():
    """コンポーネント初期化の詳細テスト"""
    logger.info("=== コンポーネント初期化詳細テスト開始 ===")
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        logger.info("DSSMSIntegratedBacktester import成功")
        
        # インスタンス作成（この段階で初期化フラグが設定される）
        logger.info("DSSMSIntegratedBacktester インスタンス作成開始")
        backtest_instance = DSSMSIntegratedBacktester()
        logger.info("DSSMSIntegratedBacktester インスタンス作成完了")
        
        # 初期化前のフラグ状態確認
        logger.info("=== 初期化前のフラグ状態 ===")
        logger.info(f"_dss_initialized: {getattr(backtest_instance, '_dss_initialized', 'NOT_FOUND')}")
        logger.info(f"_components_initialized: {getattr(backtest_instance, '_components_initialized', 'NOT_FOUND')}")
        logger.info(f"_ranking_initialized: {getattr(backtest_instance, '_ranking_initialized', 'NOT_FOUND')}")
        
        # 初期化前のコンポーネント状態確認
        logger.info("=== 初期化前のコンポーネント状態 ===")
        logger.info(f"dss_core: {getattr(backtest_instance, 'dss_core', 'NOT_FOUND')}")
        logger.info(f"nikkei225_screener: {getattr(backtest_instance, 'nikkei225_screener', 'NOT_FOUND')}")
        logger.info(f"advanced_ranking_engine: {getattr(backtest_instance, 'advanced_ranking_engine', 'NOT_FOUND')}")
        
        # _initialize_dss_core() の詳細テスト
        logger.info("\n=== _initialize_dss_core() 詳細テスト ===")
        try:
            result = backtest_instance._initialize_dss_core()
            logger.info(f"_initialize_dss_core() 実行結果: {result}")
            logger.info(f"_dss_initialized after: {backtest_instance._dss_initialized}")
            logger.info(f"dss_core after: {backtest_instance.dss_core}")
        except Exception as e:
            logger.error(f"_initialize_dss_core() 例外: {e}")
            logger.error(f"_initialize_dss_core() 例外詳細:\n{traceback.format_exc()}")
        
        # _initialize_components() の詳細テスト  
        logger.info("\n=== _initialize_components() 詳細テスト ===")
        try:
            backtest_instance._initialize_components()
            logger.info(f"_components_initialized after: {backtest_instance._components_initialized}")
            logger.info(f"nikkei225_screener after: {backtest_instance.nikkei225_screener}")
        except Exception as e:
            logger.error(f"_initialize_components() 例外: {e}")
            logger.error(f"_initialize_components() 例外詳細:\n{traceback.format_exc()}")
        
        # _initialize_advanced_ranking() の詳細テスト
        logger.info("\n=== _initialize_advanced_ranking() 詳細テスト ===")
        try:
            result = backtest_instance._initialize_advanced_ranking()
            logger.info(f"_initialize_advanced_ranking() 実行結果: {result}")
            logger.info(f"_ranking_initialized after: {backtest_instance._ranking_initialized}")
            logger.info(f"advanced_ranking_engine after: {backtest_instance.advanced_ranking_engine}")
        except Exception as e:
            logger.error(f"_initialize_advanced_ranking() 例外: {e}")
            logger.error(f"_initialize_advanced_ranking() 例外詳細:\n{traceback.format_exc()}")
        
        # ensure_*メソッドのテスト
        logger.info("\n=== ensure_*メソッドテスト ===")
        try:
            dss_core = backtest_instance.ensure_dss_core()
            logger.info(f"ensure_dss_core() 結果: {dss_core}")
        except Exception as e:
            logger.error(f"ensure_dss_core() 例外: {e}")
            logger.error(f"ensure_dss_core() 例外詳細:\n{traceback.format_exc()}")
        
        try:
            components = backtest_instance.ensure_components()
            logger.info(f"ensure_components() 結果: {components}")
        except Exception as e:
            logger.error(f"ensure_components() 例外: {e}")
            logger.error(f"ensure_components() 例外詳細:\n{traceback.format_exc()}")
        
        try:
            ranking = backtest_instance.ensure_advanced_ranking()
            logger.info(f"ensure_advanced_ranking() 結果: {ranking}")
        except Exception as e:
            logger.error(f"ensure_advanced_ranking() 例外: {e}")
            logger.error(f"ensure_advanced_ranking() 例外詳細:\n{traceback.format_exc()}")
        
        # 最終状態確認
        logger.info("\n=== 最終状態確認 ===")
        logger.info(f"_dss_initialized final: {backtest_instance._dss_initialized}")
        logger.info(f"_components_initialized final: {backtest_instance._components_initialized}")
        logger.info(f"_ranking_initialized final: {getattr(backtest_instance, '_ranking_initialized', 'NOT_FOUND')}")
        logger.info(f"dss_core final: {backtest_instance.dss_core}")
        logger.info(f"nikkei225_screener final: {backtest_instance.nikkei225_screener}")
        logger.info(f"advanced_ranking_engine final: {backtest_instance.advanced_ranking_engine}")
        
        return True
        
    except Exception as e:
        logger.error(f"テスト全体エラー: {e}")
        logger.error(f"テスト全体エラー詳細:\n{traceback.format_exc()}")
        return False

def test_individual_component_imports():
    """個別コンポーネントのimportテスト"""
    logger.info("\n=== 個別コンポーネントimportテスト ===")
    
    # DSS Core V3 import テスト
    logger.info("--- DSS Core V3 import テスト ---")
    try:
        from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
        logger.info("DSS Core V3 import成功")
        
        # インスタンス作成テスト
        dss_core = DSSBacktesterV3()
        logger.info(f"DSS Core V3 インスタンス作成成功: {type(dss_core)}")
        
    except Exception as e:
        logger.error(f"DSS Core V3 テストエラー: {e}")
        logger.error(f"DSS Core V3 テストエラー詳細:\n{traceback.format_exc()}")
    
    # Nikkei225Screener import テスト
    logger.info("--- Nikkei225Screener import テスト ---")
    try:
        from src.dssms.nikkei225_screener import Nikkei225Screener
        logger.info("Nikkei225Screener import成功")
        
        # インスタンス作成テスト
        screener = Nikkei225Screener()
        logger.info(f"Nikkei225Screener インスタンス作成成功: {type(screener)}")
        
    except Exception as e:
        logger.error(f"Nikkei225Screener テストエラー: {e}")
        logger.error(f"Nikkei225Screener テストエラー詳細:\n{traceback.format_exc()}")
    
    # AdvancedRankingEngine import テスト
    logger.info("--- AdvancedRankingEngine import テスト ---")
    try:
        from src.dssms.advanced_ranking_system.advanced_ranking_engine import AdvancedRankingEngine
        logger.info("AdvancedRankingEngine import成功")
        
        # インスタンス作成テスト
        ranking_engine = AdvancedRankingEngine(None)
        logger.info(f"AdvancedRankingEngine インスタンス作成成功: {type(ranking_engine)}")
        
    except Exception as e:
        logger.error(f"AdvancedRankingEngine テストエラー: {e}")
        logger.error(f"AdvancedRankingEngine テストエラー詳細:\n{traceback.format_exc()}")

def main():
    """メイン実行"""
    logger.info("=== DSSMS コンポーネント初期化例外詳細調査開始 ===")
    
    # Test 1: 個別コンポーネントimportテスト
    test_individual_component_imports()
    
    # Test 2: 統合初期化テスト
    logger.info("\n" + "="*60)
    logger.info("統合初期化詳細テスト")
    logger.info("="*60)
    result = test_component_initialization_detailed()
    logger.info(f"統合初期化テスト結果: {result}")
    
    logger.info("=== DSSMS コンポーネント初期化例外詳細調査完了 ===")

if __name__ == "__main__":
    main()