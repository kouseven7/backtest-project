#!/usr/bin/env python3
"""
シンプルなmain.pyテスト - 1つの戦略のみ使用
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from strategies.VWAP_Bounce import VWAPBounceStrategy
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators

# ロガーの設定
logger = setup_logger(__name__)

def test_single_strategy():
    """単一戦略でのテスト"""
    try:
        logger.info("単一戦略テストを開始")
        
        # データ取得と前処理
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        
        logger.info(f"データ期間: {start_date} から {end_date}")
        logger.info(f"データ行数: {len(stock_data)}")
        
        # VWAPBounce戦略のみテスト
        params = {
            'vwap_lower_threshold': 0.995,
            'vwap_upper_threshold': 1.005,
            'volume_increase_threshold': 1.1,
            'stop_loss': 0.025,
            'take_profit': 0.04
        }
        
        strategy = VWAPBounceStrategy(
            data=stock_data,
            params=params,
            price_column="Adj Close"
        )
        
        result = strategy.backtest()
        
        # エントリー/エグジット数を確認
        entry_count = (result['Entry_Signal'] == 1).sum()
        exit_count = (result['Exit_Signal'] == -1).sum()
        
        logger.info(f"VWAPBounce戦略結果: エントリー {entry_count}, エグジット {exit_count}")
        
        if entry_count > 0:
            logger.info("✅ 戦略が正常に動作しています")
            return True
        else:
            logger.warning("⚠️ エントリーシグナルが生成されませんでした")
            return False
            
    except Exception as e:
        logger.error(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_strategy()
    if success:
        print("単一戦略テスト成功")
    else:
        print("単一戦略テスト失敗")
