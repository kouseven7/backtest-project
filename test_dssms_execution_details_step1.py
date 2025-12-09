"""
【案2実装テスト】DSSMS execution_details記録統一テスト
期間: 2023-01-01~2023-01-31
目的: strategy_name="DSSMS_SymbolSwitch"が正しく記録されるか確認
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# プロジェクトルート追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
from config.logger_config import setup_logger

def test_execution_details_step1():
    """Step 1テスト: execution_details記録統一"""
    
    logger = setup_logger("test_execution_details_step1", log_file="logs/test_execution_details_step1.log")
    logger.info("=== 【案2実装テスト】execution_details記録統一テスト開始 ===")
    
    # 設定
    config = {
        'initial_capital': 1000000,
        'target_symbols': ['7203.T', '9984.T', '6758.T'],
        'switch_cost_rate': 0.001,
        'ranking_method': 'AdvancedRanking_V3',
        'max_drawdown_limit': 0.15
    }
    
    # DSSMSインスタンス化
    dssms = DSSMSIntegratedBacktester(config)
    
    # バックテスト実行（2023-01-01~2023-01-31）
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    logger.info(f"バックテスト期間: {start_date.date()} ~ {end_date.date()}")
    
    try:
        result = dssms.run_dynamic_backtest(
            start_date=start_date,
            end_date=end_date,
            target_symbols=config['target_symbols']
        )
        
        # 結果検証
        logger.info("=== バックテスト結果検証 ===")
        
        # 1. execution_details件数確認
        if 'execution_results' in result and len(result['execution_results']) > 0:
            execution_format = result['execution_results'][0]
            execution_details = execution_format.get('execution_details', [])
            logger.info(f"execution_details総件数: {len(execution_details)}")
            
            # 2. strategy_name確認
            dssms_count = 0
            force_close_count = 0
            
            for detail in execution_details:
                strategy_name = detail.get('strategy_name', '')
                action = detail.get('action', '')
                timestamp = detail.get('timestamp', '')
                
                if strategy_name == 'DSSMS_SymbolSwitch':
                    dssms_count += 1
                    logger.info(f"[DSSMS_SymbolSwitch] {timestamp} {action} {detail.get('symbol', '')}")
                elif strategy_name == 'ForceClose':
                    force_close_count += 1
                    logger.info(f"[ForceClose] {timestamp} {action} {detail.get('symbol', '')}")
            
            logger.info(f"DSSMS_SymbolSwitch件数: {dssms_count}")
            logger.info(f"ForceClose件数: {force_close_count}")
            
            # 3. 同日2件SELL問題確認
            sell_by_date = {}
            for detail in execution_details:
                if detail.get('action') == 'SELL':
                    date = detail.get('timestamp', '')[:10]  # YYYY-MM-DD部分抽出
                    if date not in sell_by_date:
                        sell_by_date[date] = []
                    sell_by_date[date].append(detail)
            
            duplicate_sell_dates = {date: details for date, details in sell_by_date.items() if len(details) > 1}
            
            if duplicate_sell_dates:
                logger.warning(f"同日2件SELL発生: {len(duplicate_sell_dates)}日")
                for date, details in duplicate_sell_dates.items():
                    logger.warning(f"  {date}: {len(details)}件")
                    for detail in details:
                        logger.warning(f"    - strategy_name={detail.get('strategy_name')}, symbol={detail.get('symbol')}")
            else:
                logger.info("同日2件SELL問題なし")
            
            # 4. テスト判定
            if dssms_count > 0:
                logger.info("[SUCCESS] DSSMS_SymbolSwitch記録確認完了")
            else:
                logger.warning("[WARNING] DSSMS_SymbolSwitch記録が0件")
            
            if not duplicate_sell_dates:
                logger.info("[SUCCESS] 同日2件SELL問題解消確認")
            else:
                logger.warning("[WARNING] 同日2件SELL問題が残存")
        else:
            logger.error("execution_resultsが空です")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
    
    logger.info("=== テスト完了 ===")

if __name__ == '__main__':
    test_execution_details_step1()
