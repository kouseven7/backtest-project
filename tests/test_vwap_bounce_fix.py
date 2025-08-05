#!/usr/bin/env python3
"""
VWAP_Bounce戦略の修正版テストスクリプト
取引機会の増加を確認
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import logging

# ロガー設定
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vwap_bounce_improvements():
    """修正されたVWAP_Bounce戦略の動作テスト"""
    logger.info("=== VWAP_Bounce戦略修正版テスト開始 ===")
    
    try:
        # データ取得
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # テストデータを準備（300日分）
        test_data = stock_data.iloc[-300:].copy()
        logger.info(f"テストデータ: {len(test_data)}日分 ({test_data.index[0]} 〜 {test_data.index[-1]})")
        
        # 修正されたVWAP_Bounce戦略でテスト（緩和パラメータ）
        from strategies.VWAP_Bounce import VWAPBounceStrategy
        
        # エントリー条件を緩和したパラメータ
        relaxed_params = {
            "vwap_lower_threshold": 0.985,      # 緩和: VWAP-1.5%
            "vwap_upper_threshold": 1.015,      # 緩和: VWAP+1.5%
            "volume_increase_threshold": 1.05,   # 緩和: 出来高5%増
            "bullish_candle_min_pct": 0.001,    # 緩和: 陽線0.1%
            "stop_loss": 0.015,                 # 1.5%
            "take_profit": 0.03,                # 3%
            "trend_filter_enabled": False,      # トレンドフィルター無効
            "cool_down_period": 1,              # クールダウン1日
            "max_hold_days": 10
        }
        
        logger.info("緩和パラメータでのテスト実行...")
        strategy = VWAPBounceStrategy(test_data, params=relaxed_params)
        result_data = strategy.backtest()
        
        # 結果分析
        entry_count = result_data["Entry_Signal"].sum()
        exit_count = (result_data["Exit_Signal"] == -1).sum()
        
        logger.info(f"✅ 結果: エントリー {entry_count}回, イグジット {exit_count}回")
        
        if entry_count > 0:
            logger.info("✅ 取引機会が改善されました")
            
            # 取引シミュレーション
            from trade_simulation import simulate_trades
            trade_results = simulate_trades(result_data, ticker)
            
            trade_history = trade_results.get("取引履歴", pd.DataFrame())
            if not trade_history.empty:
                total_profit = trade_history["取引結果"].sum()
                win_count = (trade_history["取引結果"] > 0).sum()
                win_rate = win_count / len(trade_history) * 100
                
                logger.info(f"📊 取引結果:")
                logger.info(f"   - 総取引数: {len(trade_history)}件")
                logger.info(f"   - 合計損益: {total_profit:.2f}円")
                logger.info(f"   - 勝率: {win_rate:.1f}%")
                
                return True
            else:
                logger.warning("⚠️ 取引履歴が空です")
                return False
        else:
            logger.warning("❌ 取引機会が改善されていません")
            return False
            
    except Exception as e:
        logger.error(f"❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_combinations():
    """パラメータ組み合わせ数の確認"""
    logger.info("=== パラメータ組み合わせ数確認 ===")
    
    try:
        from optimization.configs.vwap_bounce_optimization import PARAM_GRID
        
        total_combinations = 1
        for param_name, values in PARAM_GRID.items():
            combinations = len(values)
            total_combinations *= combinations
            logger.info(f"{param_name}: {combinations}通り")
        
        logger.info(f"📊 総組み合わせ数: {total_combinations:,}通り")
        
        if total_combinations <= 8000:
            logger.info("✅ 組み合わせ数は適切です")
            return True
        else:
            logger.warning(f"⚠️ 組み合わせ数が多すぎます: {total_combinations:,}通り")
            return False
            
    except Exception as e:
        logger.error(f"❌ パラメータ確認エラー: {e}")
        return False

if __name__ == "__main__":
    # テスト実行
    logger.info("VWAP_Bounce戦略修正版テスト開始")
    
    success1 = test_parameter_combinations()
    success2 = test_vwap_bounce_improvements()
    
    if success1 and success2:
        logger.info("🎉 すべてのテストが成功しました！")
        logger.info("最適化の実行を推奨します:")
        logger.info("python optimize_strategy.py --strategy vwap_bounce --save-results --parallel")
    else:
        logger.info("💥 一部のテストが失敗しました。設定を確認してください。")
