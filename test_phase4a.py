"""
Phase 4-A-3 品質検証: MultiStrategyManager統合テスト
"""

import sys
import os
import pandas as pd
from datetime import datetime

# プロジェクトパス追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.multi_strategy_manager_fixed import MultiStrategyManager
from config.logger_config import setup_logger

logger = setup_logger(__name__)

def test_multi_strategy_manager():
    """MultiStrategyManager Phase 4-A実装テスト"""
    try:
        logger.info("Phase 4-A-3: MultiStrategyManager統合テスト開始")
        
        # テストデータ作成
        test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Open': [100] * 100,
            'High': [105] * 100,
            'Low': [95] * 100,
            'Close': [102] * 100,
            'Adj Close': [102] * 100,
            'Volume': [1000000] * 100
        }).set_index('Date')
        
        # MultiStrategyManager初期化
        manager = MultiStrategyManager()
        if not manager.initialize_systems():
            logger.error("Manager initialization failed")
            return False
        
        # テスト実行
        market_data = {
            "data": test_data,
            "index": test_data  # 簡易的にstock_dataをindex_dataとして使用
        }
        
        available_strategies = [
            'VWAPBreakoutStrategy',
            'MomentumInvestingStrategy',
            'BreakoutStrategy'
        ]
        
        # ✅ バックテスト基本理念遵守テスト: 実際のbacktest()実行
        result = manager.execute_multi_strategy_flow(market_data, available_strategies)
        
        # 結果검증
        logger.info(f"Test Results:")
        logger.info(f"- Execution Mode: {result.execution_mode}")
        logger.info(f"- Selected Strategies: {result.selected_strategies}")
        logger.info(f"- Status: {result.status}")
        logger.info(f"- Performance Metrics: {result.performance_metrics}")
        logger.info(f"- Execution Time: {result.execution_time:.2f}s")
        
        # バックテスト基本理念チェック
        success = True
        if not result.selected_strategies:
            logger.warning("No successful strategies - potential backtest principle violation")
            success = False
        
        if result.backtest_data:
            total_trades = result.backtest_data.get('execution_metadata', {}).get('total_trades', 0)
            logger.info(f"- Total Trades: {total_trades}")
            if total_trades == 0:
                logger.warning("Zero trades detected - potential backtest principle violation")
        
        if success:
            logger.info("✅ Phase 4-A-3: MultiStrategyManager統合テスト成功")
        else:
            logger.warning("⚠️ Phase 4-A-3: 一部問題があるが基本動作確認")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Phase 4-A-3: MultiStrategyManager統合テスト失敗: {e}")
        return False

if __name__ == "__main__":
    test_multi_strategy_manager()