"""
Phase 1.6 大敗パターン分析スクリプト

大敗定義（OR条件）:
- PF < 0.7 OR
- max_drawdown > 0.30（率） OR
- 単一損失 > 0.15（率）

Author: Backtest Project Team
Created: 2026-01-23
"""

import pandas as pd
import numpy as np
from pathlib import Path

# CSVデータ読み込み
csv_path = Path('results/phase1.6_simple_20260123_224706.csv')
df = pd.read_csv(csv_path)

print('=' * 80)
print('Phase 1.6 大敗パターン分析（OR条件）')
print('=' * 80)

# データ概要
print(f'\n総データ数: {len(df)}')
print(f'列数: {len(df.columns)}')
print(f'列名: {list(df.columns)}')

# サンプルデータ表示
print(f'\n=== サンプルデータ（先頭5行） ===')
print(df[['ticker', 'stop_loss_pct', 'trailing_stop_pct', 'profit_factor', 'max_drawdown', 'win_rate', 'num_trades']].head())

# max_drawdownの単位確認（金額 or 率）
print(f'\n=== max_drawdown 統計 ===')
print(df['max_drawdown'].describe())
print(f'\nmax_drawdown 最大値: {df["max_drawdown"].max():.2f}')
print(f'max_drawdown 最小値: {df["max_drawdown"].min():.2f}')

# max_drawdownが金額の場合は率に変換する必要があるか確認
# 仮に0.0～1.0の範囲なら率、それ以外なら金額と判断
if df['max_drawdown'].max() > 1.0:
    print('\n警告: max_drawdownが金額表示の可能性があります（最大値 > 1.0）')
    print('DD率の計算が必要かもしれません（total_profitから相対値計算）')
    
    # total_profit列があるか確認
    if 'total_profit' in df.columns and 'total_loss' in df.columns:
        print('\ntotal_profit/total_loss列が存在します')
        print('DD率 = max_drawdown / (total_profit + total_loss) で計算試行')
        
        # DD率計算（簡易版: DDを総取引額で割る）
        df['dd_rate'] = df['max_drawdown'] / (df['total_profit'] + df['total_loss'])
        print(f'\n計算後のDD率統計:')
        print(df['dd_rate'].describe())
    else:
        print('total_profit/total_loss列が見つかりません')
        print('DD率計算不可、max_drawdownをそのまま使用します')
        df['dd_rate'] = df['max_drawdown']
else:
    print('\nmax_drawdownは率表示と判断（最大値 <= 1.0）')
    df['dd_rate'] = df['max_drawdown']

# 大敗定義（OR条件）でフィルタリング
print(f'\n' + '=' * 80)
print('大敗定義（OR条件）でフィルタリング')
print('=' * 80)
print(f'条件1: PF < 0.7')
print(f'条件2: DD率 > 0.30')
print(f'条件3: 単一損失 > 0.15（列が存在する場合）')

# 単一損失列の確認
single_loss_col = None
for col in df.columns:
    if 'single' in col.lower() and 'loss' in col.lower():
        single_loss_col = col
        break

if single_loss_col:
    print(f'\n単一損失列発見: {single_loss_col}')
    defeat_mask = (df['profit_factor'] < 0.7) | (df['dd_rate'] > 0.30) | (df[single_loss_col] > 0.15)
else:
    print(f'\n単一損失列が見つかりません。PFとDD率のみで評価します')
    defeat_mask = (df['profit_factor'] < 0.7) | (df['dd_rate'] > 0.30)

defeat_df = df[defeat_mask].copy()

print(f'\n大敗パターン数: {len(defeat_df)} / {len(df)} ({len(defeat_df)/len(df)*100:.1f}%)')

# 条件別内訳
pf_defeats = (df['profit_factor'] < 0.7).sum()
dd_defeats = (df['dd_rate'] > 0.30).sum()
print(f'\n条件別内訳:')
print(f'  PF < 0.7: {pf_defeats}件 ({pf_defeats/len(df)*100:.1f}%)')
print(f'  DD率 > 0.30: {dd_defeats}件 ({dd_defeats/len(df)*100:.1f}%)')

if single_loss_col:
    sl_defeats = (df[single_loss_col] > 0.15).sum()
    print(f'  単一損失 > 0.15: {sl_defeats}件 ({sl_defeats/len(df)*100:.1f}%)')

# 銘柄別集計
print(f'\n=== 銘柄別大敗カウント ===')
if len(defeat_df) > 0:
    ticker_counts = defeat_df['ticker'].value_counts().sort_index()
    print(ticker_counts)
    print(f'\n大敗が多い銘柄TOP 3:')
    print(ticker_counts.head(3))
else:
    print('大敗パターンなし')

# パラメータ別集計
print(f'\n=== パラメータ別大敗カウント ===')
if len(defeat_df) > 0:
    param_counts = defeat_df.groupby(['stop_loss_pct', 'trailing_stop_pct']).size().reset_index(name='defeat_count')
    param_counts = param_counts.sort_values('defeat_count', ascending=False)
    print(param_counts.to_string(index=False))
    
    print(f'\n大敗が多いパラメータTOP 5:')
    print(param_counts.head(5).to_string(index=False))
else:
    print('大敗パターンなし')

# 除外すべきパラメータ（全銘柄で大敗する組み合わせ）
print(f'\n=== 除外推奨パラメータ（大敗率 > 50%） ===')
if len(defeat_df) > 0:
    # 各パラメータの大敗率を計算
    all_param_counts = df.groupby(['stop_loss_pct', 'trailing_stop_pct']).size().reset_index(name='total_count')
    defeat_param_counts = defeat_df.groupby(['stop_loss_pct', 'trailing_stop_pct']).size().reset_index(name='defeat_count')
    
    param_analysis = all_param_counts.merge(
        defeat_param_counts, 
        on=['stop_loss_pct', 'trailing_stop_pct'], 
        how='left'
    )
    param_analysis['defeat_count'] = param_analysis['defeat_count'].fillna(0)
    param_analysis['defeat_rate'] = param_analysis['defeat_count'] / param_analysis['total_count']
    
    exclude_params = param_analysis[param_analysis['defeat_rate'] > 0.5].sort_values('defeat_rate', ascending=False)
    
    if len(exclude_params) > 0:
        print(exclude_params.to_string(index=False))
    else:
        print('除外推奨パラメータなし（大敗率 > 50%の組み合わせなし）')
    
    # CSVに保存
    exclude_params.to_csv('results/phase1_6_exclude_params.csv', index=False, encoding='utf-8-sig')
    print(f'\n除外パラメータリスト保存: results/phase1_6_exclude_params.csv')
else:
    print('大敗パターンなし、除外不要')

# 大敗の詳細（PF最低値、DD最大値）
print(f'\n=== 大敗パターンの詳細 ===')
if len(defeat_df) > 0:
    print(f'\nPF最低値:')
    worst_pf = defeat_df.nsmallest(3, 'profit_factor')[['ticker', 'stop_loss_pct', 'trailing_stop_pct', 'profit_factor', 'dd_rate']]
    print(worst_pf.to_string(index=False))
    
    print(f'\nDD最大値:')
    worst_dd = defeat_df.nlargest(3, 'dd_rate')[['ticker', 'stop_loss_pct', 'trailing_stop_pct', 'profit_factor', 'dd_rate']]
    print(worst_dd.to_string(index=False))

print(f'\n' + '=' * 80)
print('分析完了')
print('=' * 80)
