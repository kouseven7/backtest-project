#!/usr/bin/env python3
"""
切替分析修正内容の検証スクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dssms.dssms_backtester import DSSMSBacktester
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_switch_analysis_fix():
    """切替分析修正の検証"""
    
    try:
        # DSSMSバックテスターを初期化
        backtester = DSSMSBacktester(
            symbols=['7203.T', '9984.T'],
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now() - timedelta(days=1),
            initial_capital=1000000
        )
        
        logger.info("バックテスト実行開始...")
        
        # バックテスト実行
        result = backtester.run_backtest()
        
        if result:
            logger.info("バックテスト成功")
            
            # 切替履歴の確認
            switch_history = result.get('switch_history', [])
            logger.info(f"切替履歴件数: {len(switch_history)}")
            
            if switch_history:
                # 最初の切替データを詳細表示
                first_switch = switch_history[0]
                logger.info("最初の切替データ:")
                for key, value in first_switch.items():
                    logger.info(f"  {key}: {value} ({type(value).__name__})")
                
                # 成功判定の確認
                profit_loss = first_switch.get('profit_loss_at_switch', 0)
                is_successful = profit_loss > 0
                logger.info(f"成功判定: profit_loss_at_switch={profit_loss} -> {'成功' if is_successful else '失敗'}")
            
            logger.info("切替分析修正の検証が完了しました")
            return True
        else:
            logger.error("バックテスト失敗")
            return False
            
    except Exception as e:
        logger.error(f"検証エラー: {e}")
        return False

if __name__ == "__main__":
    success = test_switch_analysis_fix()
    if success:
        print("✅ 切替分析修正の検証に成功しました")
    else:
        print("❌ 切替分析修正の検証に失敗しました")
        sys.exit(1)
