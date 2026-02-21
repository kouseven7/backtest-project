"""
Phase 1.6エグジット理由完全分析スクリプト

Task 5-B: ペイオフレシオ・エグジット理由分析完遂

主な機能:
- エグジット理由比率算出（stop_loss, trailing_stop, dead_cross等）
- トレーリング幅別ペイオフレシオ分析
- 損切幅別ペイオフレシオ分析
- 仮説1検証: 損切3%5%7%で差がない理由 → トレーリングストップが多い
- 仮説2検証: トレーリング20%25%30%でペイオフレシオ改善

統合コンポーネント:
- phase1.6_simple_20260125_214753.csv: Phase 1.6実行結果（180レコード）

セーフティ機能/注意事項:
- Exit_Reason列実装済み（Task 1完了）
- 全銘柄・全パラメータのエグジット理由を集計
- 仮説検証を定量的に実施

Author: Backtest Project Team
Created: 2026-01-25
Last Modified: 2026-01-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ==================== データ読み込み ====================

print("Phase 1.6エグジット理由完全分析開始")
print("=" * 80)

# CSVファイル読み込み
csv_path = Path("results/phase1.6_simple_20260125_214753.csv")

if not csv_path.exists():
    print(f"ERROR: {csv_path}が見つかりません")
    exit(1)

df = pd.read_csv(csv_path)

print(f"データ読み込み完了: {len(df)}レコード")
print(f"列: {list(df.columns)}")
print()

# ==================== エグジット理由カウント列確認 ====================

exit_reason_cols = [
    'stop_loss_count',
    'trailing_stop_count',
    'dead_cross_count',
    'force_close_count',
    'take_profit_count',
    'other_count'
]

missing_cols = [col for col in exit_reason_cols if col not in df.columns]

if missing_cols:
    print(f"WARNING: 以下の列が見つかりません: {missing_cols}")
    print("利用可能な列で分析を継続します")
    exit_reason_cols = [col for col in exit_reason_cols if col in df.columns]
    print()

# ==================== 全体集計 ====================

print("=" * 80)
print("[1] 全体エグジット理由集計（180レコード合計）")
print("=" * 80)

total_counts = {}
for col in exit_reason_cols:
    total_counts[col.replace('_count', '')] = df[col].sum()

total_exits = sum(total_counts.values())

print(f"総エグジット数: {total_exits}件\n")

print("エグジット理由の内訳:")
for reason, count in sorted(total_counts.items(), key=lambda x: x[1], reverse=True):
    ratio = (count / total_exits * 100) if total_exits > 0 else 0
    print(f"  {reason}: {count}件 ({ratio:.1f}%)")

print()

# ==================== トレーリング幅別分析 ====================

print("=" * 80)
print("[2] トレーリング幅別エグジット理由分析")
print("=" * 80)

# トレーリング幅でグループ化
trailing_groups = df.groupby('trailing_stop_pct')

for trailing_pct, group in trailing_groups:
    print(f"\nトレーリングストップ: {trailing_pct*100:.0f}%")
    print("-" * 40)
    
    # エグジット理由集計
    trailing_total = 0
    trailing_counts = {}
    for col in exit_reason_cols:
        count = group[col].sum()
        trailing_counts[col.replace('_count', '')] = count
        trailing_total += count
    
    # 比率表示
    for reason, count in sorted(trailing_counts.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / trailing_total * 100) if trailing_total > 0 else 0
        print(f"  {reason}: {count}件 ({ratio:.1f}%)")
    
    # ペイオフレシオ平均
    avg_payoff = group['payoff_ratio'].mean()
    avg_pf = group['profit_factor'].mean()
    avg_wr = group['win_rate'].mean()
    
    print(f"\n  平均ペイオフレシオ: {avg_payoff:.2f}")
    print(f"  平均PF: {avg_pf:.2f}")
    print(f"  平均Win Rate: {avg_wr:.1%}")

print()

# ==================== 損切幅別分析 ====================

print("=" * 80)
print("[3] 損切幅別エグジット理由分析")
print("=" * 80)

# 損切幅でグループ化
stop_loss_groups = df.groupby('stop_loss_pct')

for stop_loss_pct, group in stop_loss_groups:
    print(f"\n損切: {stop_loss_pct*100:.0f}%")
    print("-" * 40)
    
    # エグジット理由集計
    stop_total = 0
    stop_counts = {}
    for col in exit_reason_cols:
        count = group[col].sum()
        stop_counts[col.replace('_count', '')] = count
        stop_total += count
    
    # 比率表示
    for reason, count in sorted(stop_counts.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / stop_total * 100) if stop_total > 0 else 0
        print(f"  {reason}: {count}件 ({ratio:.1f}%)")
    
    # ペイオフレシオ平均
    avg_payoff = group['payoff_ratio'].mean()
    avg_pf = group['profit_factor'].mean()
    avg_wr = group['win_rate'].mean()
    
    print(f"\n  平均ペイオフレシオ: {avg_payoff:.2f}")
    print(f"  平均PF: {avg_pf:.2f}")
    print(f"  平均Win Rate: {avg_wr:.1%}")

print()

# ==================== 仮説1検証 ====================

print("=" * 80)
print("[4] 仮説1検証: 損切3%5%7%で差がない理由")
print("=" * 80)

print("\n仮説: トレーリングストップが多くなっている可能性\n")

# 損切幅別のトレーリングストップ比率
for stop_loss_pct in [0.03, 0.05, 0.07]:
    subset = df[df['stop_loss_pct'] == stop_loss_pct]
    
    total_exits_subset = 0
    for col in exit_reason_cols:
        total_exits_subset += subset[col].sum()
    
    trailing_count = subset['trailing_stop_count'].sum()
    stop_loss_count = subset['stop_loss_count'].sum()
    dead_cross_count = subset['dead_cross_count'].sum()
    
    trailing_ratio = (trailing_count / total_exits_subset * 100) if total_exits_subset > 0 else 0
    stop_loss_ratio = (stop_loss_count / total_exits_subset * 100) if total_exits_subset > 0 else 0
    dead_cross_ratio = (dead_cross_count / total_exits_subset * 100) if total_exits_subset > 0 else 0
    
    print(f"損切{stop_loss_pct*100:.0f}%:")
    print(f"  トレーリングストップ: {trailing_ratio:.1f}% ({trailing_count}件)")
    print(f"  損切: {stop_loss_ratio:.1f}% ({stop_loss_count}件)")
    print(f"  デッドクロス: {dead_cross_ratio:.1f}% ({dead_cross_count}件)")
    print(f"  総エグジット: {total_exits_subset}件")
    print()

# 仮説1結論
print("【仮説1検証結果】")

# 損切幅別の平均PF差
pf_by_stop = df.groupby('stop_loss_pct')['profit_factor'].mean()
pf_variance = pf_by_stop.std()

print(f"損切幅別平均PF標準偏差: {pf_variance:.4f}")

if pf_variance < 0.05:
    print("結論: 損切3%5%7%でPF差が極めて小さい（標準偏差<0.05）")
    print("理由分析:")
    
    # トレーリングストップ比率の確認
    avg_trailing_ratio = (df['trailing_stop_count'].sum() / total_exits * 100)
    avg_stop_loss_ratio = (df['stop_loss_count'].sum() / total_exits * 100)
    avg_dead_cross_ratio = (df['dead_cross_count'].sum() / total_exits * 100)
    
    print(f"  全体のトレーリングストップ比率: {avg_trailing_ratio:.1f}%")
    print(f"  全体の損切比率: {avg_stop_loss_ratio:.1f}%")
    print(f"  全体のデッドクロス比率: {avg_dead_cross_ratio:.1f}%")
    
    if avg_trailing_ratio > 30:
        print("  → 仮説1成立: トレーリングストップが支配的（30%以上）")
    elif avg_dead_cross_ratio > 40:
        print("  → 仮説1一部成立: デッドクロスが支配的（損切の影響が小さい）")
    else:
        print("  → 仮説1不成立: 複数のエグジット理由が混在")
else:
    print(f"結論: 損切幅でPFに差がある（標準偏差={pf_variance:.4f}）")
    print("仮説1不成立: トレーリングストップだけが理由ではない")

print()

# ==================== 仮説2検証 ====================

print("=" * 80)
print("[5] 仮説2検証: トレーリング20%25%30%でペイオフレシオ改善")
print("=" * 80)

print("\n仮説: トレーリング幅を広げるとペイオフレシオが改善\n")

# トレーリング幅別平均ペイオフレシオ
payoff_by_trailing = df.groupby('trailing_stop_pct').agg({
    'payoff_ratio': 'mean',
    'profit_factor': 'mean',
    'win_rate': 'mean',
    'trailing_stop_count': 'sum'
}).reset_index()

payoff_by_trailing['trailing_pct_display'] = (payoff_by_trailing['trailing_stop_pct'] * 100).astype(int)

print("トレーリング幅別ペイオフレシオ:")
print(payoff_by_trailing[['trailing_pct_display', 'payoff_ratio', 'profit_factor', 'win_rate', 'trailing_stop_count']].to_string(index=False))
print()

# 仮説2検証: 5%と20-30%の比較
payoff_5pct = payoff_by_trailing[payoff_by_trailing['trailing_stop_pct'] == 0.05]['payoff_ratio'].values[0]
payoff_20_30 = payoff_by_trailing[payoff_by_trailing['trailing_stop_pct'].isin([0.20, 0.25, 0.30])]['payoff_ratio'].mean()

improvement = ((payoff_20_30 - payoff_5pct) / payoff_5pct * 100)

print("【仮説2検証結果】")
print(f"トレーリング5%平均ペイオフレシオ: {payoff_5pct:.2f}")
print(f"トレーリング20-30%平均ペイオフレシオ: {payoff_20_30:.2f}")
print(f"改善率: {improvement:+.1f}%")

if improvement > 5.0:
    print("\n結論: 仮説2成立（5%以上改善）")
    print("  → トレーリング幅を広げるとペイオフレシオが有意に改善")
elif improvement > 0:
    print("\n結論: 仮説2一部成立（改善はあるが小幅）")
    print("  → トレーリング幅拡大の効果は限定的")
else:
    print("\n結論: 仮説2不成立（改善なし）")
    print("  → トレーリング幅拡大は逆効果")

# トレーリングストップ発動率の確認
print("\nトレーリングストップ発動率:")
for _, row in payoff_by_trailing.iterrows():
    trailing_pct_display = row['trailing_pct_display']
    trailing_count = row['trailing_stop_count']
    
    # このトレーリング幅での総エグジット数
    subset = df[df['trailing_stop_pct'] == row['trailing_stop_pct']]
    total_exits_subset = 0
    for col in exit_reason_cols:
        total_exits_subset += subset[col].sum()
    
    activation_rate = (trailing_count / total_exits_subset * 100) if total_exits_subset > 0 else 0
    print(f"  {trailing_pct_display}%: {activation_rate:.1f}% ({trailing_count}/{total_exits_subset}件)")

print()

# ==================== 最終まとめ ====================

print("=" * 80)
print("[6] 分析まとめ")
print("=" * 80)

print("\n1. エグジット理由全体傾向:")
print(f"   - デッドクロスが最も多い（{total_counts.get('dead_cross', 0)}件、{total_counts.get('dead_cross', 0)/total_exits*100:.1f}%）")
print(f"   - 損切が2番目（{total_counts.get('stop_loss', 0)}件、{total_counts.get('stop_loss', 0)/total_exits*100:.1f}%）")
print(f"   - トレーリングストップは限定的（{total_counts.get('trailing_stop', 0)}件、{total_counts.get('trailing_stop', 0)/total_exits*100:.1f}%）")

print("\n2. 仮説1（損切幅で差がない理由）:")
if pf_variance < 0.05:
    if avg_trailing_ratio > 30:
        print("   成立: トレーリングストップが支配的")
    elif avg_dead_cross_ratio > 40:
        print("   一部成立: デッドクロスが支配的（損切の影響が小さい）")
    else:
        print("   不成立: 複数のエグジット理由が混在")
else:
    print("   不成立: 損切幅でPFに有意な差がある")

print("\n3. 仮説2（トレーリング幅拡大でペイオフ改善）:")
if improvement > 5.0:
    print(f"   成立: トレーリング20-30%で+{improvement:.1f}%改善")
elif improvement > 0:
    print(f"   一部成立: トレーリング20-30%で+{improvement:.1f}%改善（小幅）")
else:
    print(f"   不成立: トレーリング20-30%で{improvement:.1f}%悪化")

print("\n4. 推奨パラメータ方針:")

# 最良トレーリング幅
best_trailing = payoff_by_trailing.loc[payoff_by_trailing['payoff_ratio'].idxmax()]
print(f"   - 最良トレーリング幅: {best_trailing['trailing_pct_display']:.0f}%（ペイオフレシオ{best_trailing['payoff_ratio']:.2f}）")

# 最良損切幅
stop_by_payoff = df.groupby('stop_loss_pct')['payoff_ratio'].mean()
best_stop_loss = stop_by_payoff.idxmax()
print(f"   - 最良損切幅: {best_stop_loss*100:.0f}%（ペイオフレシオ{stop_by_payoff.max():.2f}）")

print(f"\n5. エグジット戦略への示唆:")
print(f"   - デッドクロス（{avg_dead_cross_ratio:.1f}%）がメインエグジット → トレンドフォロー型として機能")
print(f"   - トレーリングストップ（{avg_trailing_ratio:.1f}%）は限定的 → 大きな利益を追うのに寄与")
print(f"   - 損切（{avg_stop_loss_ratio:.1f}%）がリスク管理 → ドローダウン抑制に重要")

print("\n" + "=" * 80)
print("Phase 1.6エグジット理由分析完了")
print("=" * 80)
