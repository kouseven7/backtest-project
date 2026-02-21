"""
Phase 1.6 ペイオフレシオ・エグジット理由分析スクリプト

目的:
- トレーリングストップ優位性の検証
- 仮説1: 損切3%5%7%で差がない理由 → トレーリングストップが多い
- 仮説2: トレーリング20%25%30%でペイオフレシオ改善

Author: Backtest Project Team
Created: 2026-01-25
"""

import pandas as pd
import numpy as np
from pathlib import Path

# CSVデータ読み込み
csv_path = Path('results/phase1.6_simple_20260123_224706.csv')
df = pd.read_csv(csv_path)

print('=' * 80)
print('Phase 1.6 ペイオフレシオ・エグジット理由分析')
print('=' * 80)

# データ概要
print(f'\n総データ数: {len(df)}')
print(f'列名: {list(df.columns)}')

# 必要な列の存在確認
required_cols = ['payoff_ratio', 'stop_loss_count', 'trailing_stop_count', 
                 'dead_cross_count', 'force_close_count', 'take_profit_count']

print(f'\n=== 必要列の存在確認 ===')
for col in required_cols:
    exists = col in df.columns
    print(f'{col}: {"OK" if exists else "NG"}')

if 'payoff_ratio' not in df.columns:
    print('\n警告: payoff_ratio列が見つかりません')
    print('avg_win / avg_loss から計算します')
    
    if 'avg_win' in df.columns and 'avg_loss' in df.columns:
        df['payoff_ratio'] = df['avg_win'] / df['avg_loss']
        df['payoff_ratio'] = df['payoff_ratio'].replace([np.inf, -np.inf], np.nan)
        print(f'ペイオフレシオ計算完了: {df["payoff_ratio"].notna().sum()}件')
    else:
        print('エラー: avg_win/avg_loss列も見つかりません')
        exit(1)

# 武田薬品（4502.T）を除外
print(f'\n=== 武田薬品（4502.T）除外 ===')
print(f'除外前: {len(df)}件')
df_filtered = df[df['ticker'] != '4502.T'].copy()
print(f'除外後: {len(df_filtered)}件（-{len(df) - len(df_filtered)}件）')

# 基本統計
print(f'\n=== 基本統計（武田薬品除外後） ===')
print(f'銘柄数: {df_filtered["ticker"].nunique()}')
print(f'パラメータ組数: {len(df_filtered.groupby(["stop_loss_pct", "trailing_stop_pct"]))}')
print(f'\nペイオフレシオ統計:')
print(df_filtered['payoff_ratio'].describe())

# エグジット理由の列存在確認
exit_reason_cols = ['stop_loss_count', 'trailing_stop_count', 'dead_cross_count', 
                    'force_close_count', 'take_profit_count', 'other_count']

missing_cols = [col for col in exit_reason_cols if col not in df_filtered.columns]

if missing_cols:
    print(f'\n警告: エグジット理由列が見つかりません: {missing_cols}')
    print('エグジット理由分析をスキップします')
    has_exit_reasons = False
else:
    has_exit_reasons = True
    print(f'\nエグジット理由列: OK（全6列存在）')

# ==================== 分析1: トレーリング幅別ペイオフレシオ ====================

print(f'\n' + '=' * 80)
print('分析1: トレーリング幅別ペイオフレシオ')
print('=' * 80)

# トレーリング幅でグループ化
payoff_by_trailing = df_filtered.groupby('trailing_stop_pct').agg({
    'payoff_ratio': ['mean', 'std', 'count'],
    'profit_factor': 'mean',
    'win_rate': 'mean'
}).round(4)

payoff_by_trailing.columns = ['_'.join(col).strip() for col in payoff_by_trailing.columns.values]

print('\nトレーリング幅別ペイオフレシオ:')
print(payoff_by_trailing.to_string())

# 仮説2検証: トレーリング20%/25%/30%でペイオフレシオ改善?
print(f'\n=== 仮説2検証: トレーリング20%/25%/30%でペイオフレシオ改善 ===')

trailing_groups = {
    '5-15% (Phase 1.5)': [0.05, 0.10, 0.15],
    '20-30% (Phase 1.6拡張)': [0.20, 0.25, 0.30]
}

for group_name, trailing_list in trailing_groups.items():
    group_data = df_filtered[df_filtered['trailing_stop_pct'].isin(trailing_list)]
    avg_payoff = group_data['payoff_ratio'].mean()
    avg_pf = group_data['profit_factor'].mean()
    print(f'\n{group_name}:')
    print(f'  平均ペイオフレシオ: {avg_payoff:.3f}')
    print(f'  平均PF: {avg_pf:.3f}')
    print(f'  データ数: {len(group_data)}件')

# 相関分析
correlation = df_filtered[['trailing_stop_pct', 'payoff_ratio', 'profit_factor']].corr()
print(f'\n相関係数:')
print(correlation.to_string())

# ==================== 分析2: 損切%別ペイオフレシオ ====================

print(f'\n' + '=' * 80)
print('分析2: 損切%別ペイオフレシオ')
print('=' * 80)

payoff_by_stoploss = df_filtered.groupby('stop_loss_pct').agg({
    'payoff_ratio': ['mean', 'std', 'count'],
    'profit_factor': 'mean',
    'win_rate': 'mean'
}).round(4)

payoff_by_stoploss.columns = ['_'.join(col).strip() for col in payoff_by_stoploss.columns.values]

print('\n損切%別ペイオフレシオ:')
print(payoff_by_stoploss.to_string())

# ==================== 分析3: エグジット理由比率 ====================

if has_exit_reasons:
    print(f'\n' + '=' * 80)
    print('分析3: エグジット理由比率分析')
    print('=' * 80)
    
    # 全体のエグジット理由比率
    total_exits = df_filtered[exit_reason_cols].sum()
    total_count = total_exits.sum()
    
    print(f'\n=== 全体のエグジット理由比率 ===')
    print(f'総エグジット数: {total_count:.0f}')
    for col in exit_reason_cols:
        count = total_exits[col]
        ratio = count / total_count if total_count > 0 else 0
        print(f'{col}: {count:.0f}件 ({ratio*100:.1f}%)')
    
    # 損切%別エグジット理由比率
    print(f'\n=== 損切%別エグジット理由比率 ===')
    
    for stop_loss in sorted(df_filtered['stop_loss_pct'].unique()):
        subset = df_filtered[df_filtered['stop_loss_pct'] == stop_loss]
        exits = subset[exit_reason_cols].sum()
        total = exits.sum()
        
        print(f'\n損切{stop_loss*100:.0f}%:')
        print(f'  総エグジット数: {total:.0f}')
        print(f'  損切: {exits["stop_loss_count"]:.0f}件 ({exits["stop_loss_count"]/total*100:.1f}%)')
        print(f'  トレーリング: {exits["trailing_stop_count"]:.0f}件 ({exits["trailing_stop_count"]/total*100:.1f}%)')
        print(f'  デッドクロス: {exits["dead_cross_count"]:.0f}件 ({exits["dead_cross_count"]/total*100:.1f}%)')
        print(f'  強制決済: {exits["force_close_count"]:.0f}件 ({exits["force_close_count"]/total*100:.1f}%)')
    
    # 仮説1検証: 損切3%5%7%で差がない理由 → トレーリングストップが多い?
    print(f'\n=== 仮説1検証: 損切%で差がない理由 ===')
    
    for stop_loss in sorted(df_filtered['stop_loss_pct'].unique()):
        subset = df_filtered[df_filtered['stop_loss_pct'] == stop_loss]
        exits = subset[exit_reason_cols].sum()
        total = exits.sum()
        
        trailing_ratio = exits['trailing_stop_count'] / total if total > 0 else 0
        stop_loss_ratio = exits['stop_loss_count'] / total if total > 0 else 0
        
        print(f'損切{stop_loss*100:.0f}%: トレーリング比率 {trailing_ratio*100:.1f}% vs 損切比率 {stop_loss_ratio*100:.1f}%')
    
    # トレーリング幅別エグジット理由比率
    print(f'\n=== トレーリング幅別エグジット理由比率 ===')
    
    for trailing in sorted(df_filtered['trailing_stop_pct'].unique()):
        subset = df_filtered[df_filtered['trailing_stop_pct'] == trailing]
        exits = subset[exit_reason_cols].sum()
        total = exits.sum()
        
        print(f'\nトレーリング{trailing*100:.0f}%:')
        print(f'  総エグジット数: {total:.0f}')
        print(f'  損切: {exits["stop_loss_count"]:.0f}件 ({exits["stop_loss_count"]/total*100:.1f}%)')
        print(f'  トレーリング: {exits["trailing_stop_count"]:.0f}件 ({exits["trailing_stop_count"]/total*100:.1f}%)')
        print(f'  デッドクロス: {exits["dead_cross_count"]:.0f}件 ({exits["dead_cross_count"]/total*100:.1f}%)')
        print(f'  強制決済: {exits["force_close_count"]:.0f}件 ({exits["force_close_count"]/total*100:.1f}%)')

# ==================== 分析4: パラメータ組合せTOP 10 ====================

print(f'\n' + '=' * 80)
print('分析4: パラメータ組合せTOP 10（ペイオフレシオ順）')
print('=' * 80)

# パラメータ組合せで平均化
param_grouped = df_filtered.groupby(['stop_loss_pct', 'trailing_stop_pct']).agg({
    'payoff_ratio': 'mean',
    'profit_factor': 'mean',
    'win_rate': 'mean',
    'ticker': 'count'
}).reset_index()

param_grouped = param_grouped.rename(columns={'ticker': 'sample_count'})
param_grouped = param_grouped.sort_values('payoff_ratio', ascending=False)

print('\nTOP 10パラメータ（ペイオフレシオ順）:')
print(param_grouped.head(10).to_string(index=False))

# ==================== 結果サマリー ====================

print(f'\n' + '=' * 80)
print('結果サマリー')
print('=' * 80)

print(f'\n1. ペイオフレシオ全体:')
print(f'   平均: {df_filtered["payoff_ratio"].mean():.3f}')
print(f'   中央値: {df_filtered["payoff_ratio"].median():.3f}')
print(f'   最大: {df_filtered["payoff_ratio"].max():.3f}')
print(f'   最小: {df_filtered["payoff_ratio"].min():.3f}')

print(f'\n2. トレーリング幅の影響:')
trailing_best = payoff_by_trailing['payoff_ratio_mean'].idxmax()
trailing_worst = payoff_by_trailing['payoff_ratio_mean'].idxmin()
print(f'   最良: トレーリング{trailing_best*100:.0f}% (ペイオフレシオ {payoff_by_trailing.loc[trailing_best, "payoff_ratio_mean"]:.3f})')
print(f'   最悪: トレーリング{trailing_worst*100:.0f}% (ペイオフレシオ {payoff_by_trailing.loc[trailing_worst, "payoff_ratio_mean"]:.3f})')

print(f'\n3. 損切%の影響:')
stoploss_best = payoff_by_stoploss['payoff_ratio_mean'].idxmax()
stoploss_worst = payoff_by_stoploss['payoff_ratio_mean'].idxmin()
print(f'   最良: 損切{stoploss_best*100:.0f}% (ペイオフレシオ {payoff_by_stoploss.loc[stoploss_best, "payoff_ratio_mean"]:.3f})')
print(f'   最悪: 損切{stoploss_worst*100:.0f}% (ペイオフレシオ {payoff_by_stoploss.loc[stoploss_worst, "payoff_ratio_mean"]:.3f})')

if has_exit_reasons:
    print(f'\n4. エグジット理由（全体）:')
    trailing_total = total_exits['trailing_stop_count']
    stop_loss_total = total_exits['stop_loss_count']
    print(f'   トレーリングストップ: {trailing_total:.0f}件 ({trailing_total/total_count*100:.1f}%)')
    print(f'   損切: {stop_loss_total:.0f}件 ({stop_loss_total/total_count*100:.1f}%)')
    print(f'   デッドクロス: {total_exits["dead_cross_count"]:.0f}件 ({total_exits["dead_cross_count"]/total_count*100:.1f}%)')

print(f'\n' + '=' * 80)
print('分析完了')
print('=' * 80)

# CSV出力用データフレーム作成
output_df = param_grouped.copy()

if has_exit_reasons:
    # エグジット理由比率を追加
    exit_ratios = []
    for _, row in output_df.iterrows():
        subset = df_filtered[
            (df_filtered['stop_loss_pct'] == row['stop_loss_pct']) &
            (df_filtered['trailing_stop_pct'] == row['trailing_stop_pct'])
        ]
        exits = subset[exit_reason_cols].sum()
        total = exits.sum()
        
        exit_ratios.append({
            'stop_loss_ratio': exits['stop_loss_count'] / total if total > 0 else 0,
            'trailing_stop_ratio': exits['trailing_stop_count'] / total if total > 0 else 0,
            'dead_cross_ratio': exits['dead_cross_count'] / total if total > 0 else 0
        })
    
    exit_ratios_df = pd.DataFrame(exit_ratios)
    output_df = pd.concat([output_df, exit_ratios_df], axis=1)

# CSV保存
output_path = Path('results/phase1_6_payoff_analysis.csv')
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'\n分析結果保存: {output_path}')
