import sys, pandas as pd
sys.path.append('.')
from data_fetcher import get_parameters_and_data
from src.strategies.Opening_Gap import OpeningGapStrategy
from src.strategies.contrarian_strategy import ContrarianStrategy
from src.strategies.gc_strategy_signal import GCStrategy

print("=== Phase 2: 個別戦略テスト ===")

# テストデータ取得
ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()

# OpeningGap戦略テスト
print('\n1. OpeningGap戦略:')
try:
    og_strategy = OpeningGapStrategy(data=stock_data, dow_data=index_data, params={}, price_column='Close')
    og_result = og_strategy.backtest()
    entry_signals = (og_result['Entry_Signal'] == 1).sum()
    exit_signals = (og_result['Exit_Signal'] == -1).sum()
    print(f'  エントリー: {entry_signals}回, エグジット: {exit_signals}回')
    print(f'  結果データ形状: {og_result.shape}')
    print(f'  結果データ列: {list(og_result.columns)}')
except Exception as e:
    print(f'  エラー: {e}')

# Contrarian戦略テスト  
print('\n2. Contrarian戦略:')
try:
    con_strategy = ContrarianStrategy(data=stock_data, params={}, price_column='Close')
    con_result = con_strategy.backtest()
    entry_signals = (con_result['Entry_Signal'] == 1).sum()
    exit_signals = (con_result['Exit_Signal'] == -1).sum()
    print(f'  エントリー: {entry_signals}回, エグジット: {exit_signals}回')
    print(f'  結果データ形状: {con_result.shape}')
except Exception as e:
    print(f'  エラー: {e}')

# GC戦略テスト
print('\n3. GC戦略:')
try:
    gc_strategy = GCStrategy(data=stock_data, params={}, price_column='Close')
    gc_result = gc_strategy.backtest()
    entry_signals = (gc_result['Entry_Signal'] == 1).sum()
    exit_signals = (gc_result['Exit_Signal'] == -1).sum() 
    print(f'  エントリー: {entry_signals}回, エグジット: {exit_signals}回')
    print(f'  結果データ形状: {gc_result.shape}')
except Exception as e:
    print(f'  エラー: {e}')