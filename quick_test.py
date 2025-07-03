import sys
sys.path.append(r'C:\Users\imega\Documents\my_backtest_project')

from data_fetcher import get_parameters_and_data
from strategies.VWAP_Bounce import VWAPBounceStrategy

# データ取得とテスト
ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
test_data = stock_data.iloc[-300:].copy()

# 緩和パラメータ
params = {
    'vwap_lower_threshold': 0.985,
    'vwap_upper_threshold': 1.015,
    'volume_increase_threshold': 1.05,
    'bullish_candle_min_pct': 0.001,
    'trend_filter_enabled': False,
    'stop_loss': 0.015,
    'take_profit': 0.03
}

strategy = VWAPBounceStrategy(test_data, params=params)
result = strategy.backtest()
entry_count = result['Entry_Signal'].sum()
exit_count = (result['Exit_Signal'] == -1).sum()
print(f'エントリー: {entry_count}回, イグジット: {exit_count}回')
