#!/usr/bin/env python3
"""
VWAPBounceStrategy修正テスト（RA_005）
TODO #12戦略初期化エラー最終修正の検証

Purpose: VWAPBounceStrategyのindex_dataパラメータミスマッチ修正を検証
Target: 6/7戦略成功 → 7/7戦略成功（100%）達成
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import numpy as np
from config.multi_strategy_manager import MultiStrategyManager
from data_fetcher import get_parameters_and_data
from config.logger_config import setup_logger

def test_vwap_bounce_strategy_fix():
    """VWAPBounceStrategy修正テスト"""
    logger = setup_logger("VWAPBounceTest")
    logger.info("=== RA_005: VWAPBounceStrategy修正テスト開始 ===")
    
    try:
        # テストデータ準備
        ticker, start_date, end_date, test_data, index_data = get_parameters_and_data("7203.T", "2024-01-01", "2024-12-31")
        logger.info(f"テストデータ準備完了: {len(test_data)}行")
        
        # MultiStrategyManager初期化
        manager = MultiStrategyManager()
        manager.initialize_systems()
        logger.info("MultiStrategyManager初期化完了")
        
        # VWAPBounceStrategy単体テスト
        logger.info("=== VWAPBounceStrategy単体インスタンス化テスト ===")
        try:
            # デフォルトパラメータでインスタンス化テスト
            test_params = {
                "vwap_lower_threshold": 0.99,
                "vwap_upper_threshold": 1.01,
                "volume_increase_threshold": 1.5,
                "stop_loss_percent": 0.03,
                "profit_target_percent": 0.05
            }
            
            # index_dataを渡さずにインスタンス化（修正後の期待動作）
            vwap_bounce_instance = manager.get_strategy_instance(
                'VWAPBounceStrategy', 
                test_data, 
                test_params,
                price_column='Close',
                volume_column='Volume'
                # index_dataは意図的に渡さない（修正テスト）
            )
            
            logger.info("✅ VWAPBounceStrategy インスタンス化成功")
            
            # backtest()メソッド実行テスト
            result = vwap_bounce_instance.backtest(test_data)
            logger.info(f"✅ VWAPBounceStrategy backtest実行成功: 結果形状 {result.shape}")
            
            # バックテスト基本理念遵守確認
            if 'Entry_Signal' in result.columns and 'Exit_Signal' in result.columns:
                entry_count = (result['Entry_Signal'] == 1).sum()
                exit_count = (result['Exit_Signal'] == 1).sum()
                logger.info(f"✅ バックテスト基本理念遵守: Entry={entry_count}, Exit={exit_count}")
                return True, f"Success: Entry={entry_count}, Exit={exit_count}"
            else:
                logger.error("❌ バックテスト基本理念違反: Entry_Signal/Exit_Signal欠損")
                return False, "Signal columns missing"
                
        except Exception as e:
            logger.error(f"❌ VWAPBounceStrategy テスト失敗: {e}")
            return False, str(e)
            
    except Exception as e:
        logger.error(f"❌ テスト準備段階で失敗: {e}")
        return False, f"Setup failed: {e}"

def test_all_strategy_success():
    """全戦略インスタンス化成功テスト（7/7戦略達成確認）"""
    logger = setup_logger("AllStrategyTest")
    logger.info("=== 7/7戦略成功確認テスト ===")
    
    try:
        # テストデータ準備
        ticker, start_date, end_date, test_data, index_data = get_parameters_and_data("7203.T", "2024-01-01", "2024-12-31")
        
        # MultiStrategyManager初期化
        manager = MultiStrategyManager()
        manager.initialize_systems()
        
        # 全戦略テスト
        all_strategies = list(manager.strategy_registry.keys())
        logger.info(f"登録戦略一覧: {all_strategies}")
        
        success_count = 0
        total_strategies = len(all_strategies)
        
        for strategy_name in all_strategies:
            try:
                # 共通テストパラメータ
                test_params = {"test": True}
                
                # 戦略固有の追加パラメータ
                kwargs = {}
                if strategy_name in ['VWAPBreakoutStrategy']:
                    kwargs['index_data'] = test_data.copy()
                elif strategy_name in ['OpeningGapStrategy']:
                    kwargs['dow_data'] = test_data.copy()
                # VWAPBounceStrategyはindex_data不要（修正済み）
                
                # インスタンス化テスト
                strategy_instance = manager.get_strategy_instance(
                    strategy_name, 
                    test_data, 
                    test_params,
                    **kwargs
                )
                
                # backtest基本理念確認
                if hasattr(strategy_instance, 'backtest') and callable(strategy_instance.backtest):
                    logger.info(f"✅ {strategy_name}: インスタンス化 + backtest()確認成功")
                    success_count += 1
                else:
                    logger.error(f"❌ {strategy_name}: backtest()メソッド不備")
                    
            except Exception as e:
                logger.error(f"❌ {strategy_name}: インスタンス化失敗 - {e}")
        
        # 成功率計算
        success_rate = (success_count / total_strategies) * 100
        logger.info(f"=== 戦略インスタンス化結果 ===")
        logger.info(f"成功戦略: {success_count}/{total_strategies}")
        logger.info(f"成功率: {success_rate:.1f}%")
        
        if success_count == total_strategies:
            logger.info("🎉 7/7戦略成功（100%）達成！RA_005修正成功")
            return True, f"100% success ({success_count}/{total_strategies})"
        else:
            logger.warning(f"⚠️ 7/7戦略未達成: {success_count}/{total_strategies}")
            return False, f"Partial success ({success_count}/{total_strategies})"
            
    except Exception as e:
        logger.error(f"❌ 全戦略テスト失敗: {e}")
        return False, str(e)

if __name__ == "__main__":
    print("=" * 60)
    print("VWAPBounceStrategy修正テスト（RA_005）実行")
    print("=" * 60)
    
    # テスト1: VWAPBounceStrategy単体テスト
    print("\n1. VWAPBounceStrategy単体テスト")
    success1, message1 = test_vwap_bounce_strategy_fix()
    print(f"結果: {'成功' if success1 else '失敗'} - {message1}")
    
    # テスト2: 全戦略成功テスト
    print("\n2. 全戦略（7/7）成功テスト")
    success2, message2 = test_all_strategy_success()
    print(f"結果: {'成功' if success2 else '失敗'} - {message2}")
    
    # 総合判定
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 RA_005修正完全成功！")
        print("✅ VWAPBounceStrategyパラメータミスマッチ解決")
        print("✅ 7/7戦略（100%）インスタンス化成功達成")
        print("✅ TODO #12戦略初期化エラー問題完全解決")
        print("\n次: RA_006でTODO #11最終再実行により75%+復旧確認")
    else:
        print("❌ RA_005修正で問題発生")
        if not success1:
            print(f"❌ VWAPBounceStrategy問題: {message1}")
        if not success2:
            print(f"❌ 全戦略問題: {message2}")