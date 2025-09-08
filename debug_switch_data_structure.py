#!/usr/bin/env python3
"""
切り替えデータの構造を詳細分析するデバッグツール
"""
import logging
from src.dssms.dssms_backtester import DSSMSBacktester
from src.dssms.data_fetcher import DataFetcher
import json

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_switch_data_structure():
    """切り替えデータの構造を詳細分析"""
    logger.info("切り替えデータ構造分析開始")
    
    try:
        # バックテスターを初期化
        data_fetcher = DataFetcher()
        backtester = DSSMSBacktester(data_fetcher)
        
        # SPYデータを取得（小さなサンプル）
        logger.info("SPYデータ取得中...")
        spy_data = data_fetcher.get_data("SPY", period="6mo")  # 6ヶ月分のデータで軽量化
        
        if spy_data is None or spy_data.empty:
            logger.error("SPYデータの取得に失敗")
            return
            
        # バックテストを実行
        logger.info("バックテスト実行中...")
        results = backtester.run_backtest(spy_data, "SPY", start_amount=100000)
        
        # 切り替え履歴の構造を分析
        if hasattr(backtester, 'switch_history') and backtester.switch_history:
            logger.info(f"切り替え履歴数: {len(backtester.switch_history)}")
            
            # 最初の5件の切り替えを詳細分析
            for i, switch in enumerate(backtester.switch_history[:5]):
                logger.info(f"\n=== 切り替え {i+1} ===")
                logger.info(f"切り替えオブジェクトの型: {type(switch)}")
                
                # オブジェクトの属性を確認
                if hasattr(switch, '__dict__'):
                    logger.info("オブジェクトの属性:")
                    for attr, value in switch.__dict__.items():
                        logger.info(f"  {attr}: {value} (型: {type(value)})")
                
                # to_dict()メソッドの結果を確認
                if hasattr(switch, 'to_dict'):
                    switch_dict = switch.to_dict()
                    logger.info("to_dict()の結果:")
                    for key, value in switch_dict.items():
                        logger.info(f"  {key}: {value} (型: {type(value)})")
                        
                # profit_loss_at_switchの値を詳細確認
                if hasattr(switch, 'profit_loss_at_switch'):
                    logger.info(f"profit_loss_at_switch属性: {switch.profit_loss_at_switch} (型: {type(switch.profit_loss_at_switch)})")
        else:
            logger.warning("切り替え履歴が見つかりません")
            
    except Exception as e:
        logger.error(f"分析中にエラーが発生: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    analyze_switch_data_structure()
