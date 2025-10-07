import sys
import pandas as pd
sys.path.append('.')
from main import apply_strategies_with_optimized_params
from main import load_optimized_parameters
from data_fetcher import get_parameters_and_data

print("=== Phase 3: 統合システム動作確認 ===")

# データとパラメータ取得
ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
optimized_params = load_optimized_parameters(ticker)

print(f'利用可能戦略: {list(optimized_params.keys())}')

# 統合前の状態確認
print(f'統合前のstock_dataの形状: {stock_data.shape}')
entry_before = stock_data.get("Entry_Signal", pd.Series()).sum() if "Entry_Signal" in stock_data.columns else 0
print(f'統合前のEntry_Signal: {entry_before}')

# 統合実行
integrated_data = apply_strategies_with_optimized_params(stock_data, index_data, optimized_params)

# 統合後の確認
entry_after = (integrated_data["Entry_Signal"] == 1).sum()
exit_after = (integrated_data["Exit_Signal"] == -1).sum()
print(f'統合後のEntry_Signal: {entry_after}')
print(f'統合後のExit_Signal: {exit_after}')

# 強制決済チェック
print('\n=== 強制決済状況分析 ===')
last_day_exits = integrated_data[integrated_data.index == integrated_data.index[-1]]['Exit_Signal'].sum()
total_exits = (integrated_data['Exit_Signal'] == -1).sum()
print(f'最終日のエグジット: {last_day_exits}')
print(f'総エグジット数: {total_exits}') 
print(f'強制決済率: {(last_day_exits/total_exits*100 if total_exits > 0 else 0):.1f}%')

# 詳細データチェック
print(f'\n=== 詳細データ分析 ===')
print(f'統合後データ形状: {integrated_data.shape}')
print(f'統合後データ列: {list(integrated_data.columns)}')

# 戦略別エントリー数（統合前個別実行）
print(f'\n=== 戦略別エントリー数 (Phase2結果より) ===')
print(f'OpeningGap: 16回エントリー, 199回エグジット')
print(f'Contrarian: 20回エントリー, 196回エグジット') 
print(f'GC: 1回エントリー, 4回エグジット')