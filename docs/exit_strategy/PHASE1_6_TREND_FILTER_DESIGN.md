# Phase 1.6トレンド強度フィルター設計書

**作成日**: 2026-01-26  
**目的**: トレンド強度フィルターを実装し、武田薬品（4502.T）での効果を検証  
**参照元**: [PHASE1_6_DEFEAT_PATTERNS_RESULT.md](PHASE1_6_DEFEAT_PATTERNS_RESULT.md)

---

## 1. 目的とゴール

### 主目的
トレンド強度「高」のみでエントリーするフィルターを実装し、PF改善効果を定量的に検証する。

### 成功条件
- [ ] トレンド強度閾値（67パーセンタイル）を算出
- [ ] フィルター適用前後のPF比較
- [ ] フィルター適用前後の勝率比較
- [ ] フィルター適用前後の取引件数確認（最低20件以上維持）
- [ ] 改善率30%以上でフィルター有効性確認
- [ ] 結果をMarkdownレポートに記録

---

## 2. 分析結果サマリー（Phase 1.6から）

### トレンド強度別成績（武田薬品4502.T）

| 強度レベル | 取引数 | 平均損益率 | 平均R倍 | 勝率 |
|----------|-------|-----------|---------|------|
| 低       | 275   | -3.68%    | 2.52    | 0%   |
| 中       | 274   | -2.94%    | 2.17    | 0%   |
| 高       | 291   | 3.16%     | 7.03    | 71.1%|

**現在の全体成績**:
- 総取引数: 840件
- 現在のPF: 0.53（推定）
- 勝率: 25.7%（全体平均）

**期待される効果**:
- フィルター後取引数: 291件（高のみ）
- フィルター後勝率: 71.1%
- フィルター後PF: 2.0以上（推定）

---

## 3. 実装設計

### Step 1: データ読み込みと前処理

```python
import pandas as pd
import numpy as np
from pathlib import Path

# データ読み込み
csv_path = Path('results/phase1.6_trades_20260126_124409a.csv')
df = pd.read_csv(csv_path, header=None)

# カラム名設定（29列）
columns = [
    'entry_date', 'entry_price', 'exit_date', 'exit_price', 
    'profit_loss', 'exit_reason', 'holding_days', 'profit_loss_pct', 
    'r_multiple', 'entry_gap_pct', 'max_profit_pct', 'entry_atr_pct', 
    'sma_distance_pct', 'entry_trend_strength', 'entry_volume', 
    'exit_volume', 'ticker', 'stop_loss_pct', 'trailing_stop_pct', 
    'take_profit_pct', 'entry_close', 'entry_sma_short', 
    'entry_sma_long', 'entry_rsi', 'entry_bb_upper', 'entry_bb_lower', 
    'exit_close', 'exit_sma_short', 'exit_sma_long'
]
df.columns = columns

# 武田薬品のみ抽出
takeda_df = df[df['ticker'] == '4502.T'].copy()
print(f"武田薬品取引数: {len(takeda_df)}件")
```

### Step 2: トレンド強度閾値の算出

```python
# 67パーセンタイル（上位33%が「高」）
threshold_high = takeda_df['entry_trend_strength'].quantile(0.67)
threshold_mid = takeda_df['entry_trend_strength'].quantile(0.33)

print(f"トレンド強度閾値:")
print(f"  低/中境界（33%ile）: {threshold_mid:.4f}")
print(f"  中/高境界（67%ile）: {threshold_high:.4f}")

# 分類確認
high_count = len(takeda_df[takeda_df['entry_trend_strength'] >= threshold_high])
mid_count = len(takeda_df[(takeda_df['entry_trend_strength'] >= threshold_mid) & 
                          (takeda_df['entry_trend_strength'] < threshold_high)])
low_count = len(takeda_df[takeda_df['entry_trend_strength'] < threshold_mid])

print(f"\n分類確認:")
print(f"  高: {high_count}件")
print(f"  中: {mid_count}件")
print(f"  低: {low_count}件")
```

### Step 3: フィルター適用と成績比較

```python
def calculate_performance_metrics(trades_df):
    """
    パフォーマンス指標計算
    """
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'pf': 0.0,
            'avg_profit_pct': 0.0,
            'avg_r_multiple': 0.0
        }
    
    winning = trades_df[trades_df['profit_loss_pct'] > 0]
    losing = trades_df[trades_df['profit_loss_pct'] <= 0]
    
    total_profit = winning['profit_loss_pct'].sum() if len(winning) > 0 else 0.0
    total_loss = abs(losing['profit_loss_pct'].sum()) if len(losing) > 0 else 0.0
    
    pf = total_profit / total_loss if total_loss > 0 else (999.0 if total_profit > 0 else 0.0)
    
    return {
        'total_trades': len(trades_df),
        'winning_trades': len(winning),
        'win_rate': len(winning) / len(trades_df),
        'total_profit': total_profit,
        'total_loss': total_loss,
        'pf': pf,
        'avg_profit_pct': trades_df['profit_loss_pct'].mean(),
        'avg_r_multiple': trades_df['r_multiple'].mean()
    }

# フィルター適用前
metrics_before = calculate_performance_metrics(takeda_df)

# フィルター適用後（高トレンドのみ）
takeda_high_trend = takeda_df[takeda_df['entry_trend_strength'] >= threshold_high].copy()
metrics_after = calculate_performance_metrics(takeda_high_trend)

print("\n=== フィルター適用前 ===")
print(f"総取引数: {metrics_before['total_trades']}件")
print(f"勝ち取引数: {metrics_before['winning_trades']}件")
print(f"勝率: {metrics_before['win_rate']*100:.2f}%")
print(f"PF: {metrics_before['pf']:.2f}")
print(f"平均損益率: {metrics_before['avg_profit_pct']:.2f}%")
print(f"平均R倍: {metrics_before['avg_r_multiple']:.2f}")

print("\n=== フィルター適用後（高トレンドのみ） ===")
print(f"総取引数: {metrics_after['total_trades']}件")
print(f"勝ち取引数: {metrics_after['winning_trades']}件")
print(f"勝率: {metrics_after['win_rate']*100:.2f}%")
print(f"PF: {metrics_after['pf']:.2f}")
print(f"平均損益率: {metrics_after['avg_profit_pct']:.2f}%")
print(f"平均R倍: {metrics_after['avg_r_multiple']:.2f}")

# 改善率計算
if metrics_before['pf'] > 0:
    pf_improvement = (metrics_after['pf'] - metrics_before['pf']) / metrics_before['pf'] * 100
    print(f"\nPF改善率: {pf_improvement:+.1f}%")
```

### Step 4: エグジット理由別分析

```python
# フィルター適用前後のexit_reason分布比較
print("\n=== Exit Reason分布 ===")
print("\n【フィルター前】")
exit_reason_before = takeda_df['exit_reason'].value_counts()
print(exit_reason_before)

print("\n【フィルター後（高トレンドのみ）】")
exit_reason_after = takeda_high_trend['exit_reason'].value_counts()
print(exit_reason_after)
```

### Step 5: 急騰→急落パターンへの影響

```python
# 急騰→急落パターン（max_profit_pct > 15% & profit_loss_pct < 0 & holding_days < 30）
def detect_pump_dump(trades_df):
    pattern = trades_df[
        (trades_df['max_profit_pct'] > 15) &
        (trades_df['profit_loss_pct'] < 0) &
        (trades_df['holding_days'] < 30)
    ]
    return pattern

pump_dump_before = detect_pump_dump(takeda_df)
pump_dump_after = detect_pump_dump(takeda_high_trend)

print("\n=== 急騰→急落パターン ===")
print(f"フィルター前: {len(pump_dump_before)}件 ({len(pump_dump_before)/len(takeda_df)*100:.1f}%)")
print(f"フィルター後: {len(pump_dump_after)}件 ({len(pump_dump_after)/len(takeda_high_trend)*100:.1f}%)")
print(f"削減: {len(pump_dump_before) - len(pump_dump_after)}件")
```

---

## 4. 可視化設計

### 図表1: トレンド強度分布ヒストグラム

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))

# ヒストグラム
ax.hist(takeda_df['entry_trend_strength'], bins=30, alpha=0.7, color='blue', edgecolor='black')

# 閾値線
ax.axvline(threshold_high, color='red', linestyle='--', linewidth=2, label=f'高閾値 ({threshold_high:.4f})')
ax.axvline(threshold_mid, color='orange', linestyle='--', linewidth=2, label=f'中閾値 ({threshold_mid:.4f})')

ax.set_xlabel('Entry Trend Strength')
ax.set_ylabel('取引数')
ax.set_title('トレンド強度分布と閾値')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/trend_strength_distribution.png', dpi=150)
print("図表保存: trend_strength_distribution.png")
```

### 図表2: フィルター前後のPF比較棒グラフ

```python
fig, ax = plt.subplots(figsize=(8, 6))

categories = ['フィルター前', 'フィルター後\n(高トレンドのみ)']
pf_values = [metrics_before['pf'], metrics_after['pf']]

bars = ax.bar(categories, pf_values, color=['red', 'green'], alpha=0.7, edgecolor='black')

# 数値ラベル
for bar, val in zip(bars, pf_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Profit Factor (PF)')
ax.set_title('トレンド強度フィルター効果（PF比較）')
ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='損益分岐点 (PF=1.0)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/trend_filter_pf_comparison.png', dpi=150)
print("図表保存: trend_filter_pf_comparison.png")
```

### 図表3: 勝率比較棒グラフ

```python
fig, ax = plt.subplots(figsize=(8, 6))

win_rates = [metrics_before['win_rate']*100, metrics_after['win_rate']*100]

bars = ax.bar(categories, win_rates, color=['orange', 'skyblue'], alpha=0.7, edgecolor='black')

# 数値ラベル
for bar, val in zip(bars, win_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('勝率 (%)')
ax.set_title('トレンド強度フィルター効果（勝率比較）')
ax.set_ylim(0, 80)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/trend_filter_winrate_comparison.png', dpi=150)
print("図表保存: trend_filter_winrate_comparison.png")
```

---

## 5. 検証基準

### 最低要件
- [ ] フィルター後の取引数 >= 20件（統計的有意性）
- [ ] フィルター後のPF > 1.0（損益分岐点突破）
- [ ] フィルター後の勝率 > 50%

### 期待値
- [ ] PF改善率 >= 30%
- [ ] 勝率改善 >= 20ポイント
- [ ] 急騰→急落パターン削減率 >= 30%

### 理想値
- [ ] PF >= 2.0
- [ ] 勝率 >= 65%
- [ ] 平均損益率 > 2.0%

---

## 6. 次フェーズへの展開

### Phase 1.7: 他の9銘柄での検証

トレンド強度フィルターが武田薬品で有効だった場合、他の9銘柄でも検証：

```python
VALIDATION_TICKERS = [
    "7203.T",  # トヨタ自動車
    "9984.T",  # ソフトバンクグループ
    "8306.T",  # 三菱UFJ FG
    "6758.T",  # ソニーグループ
    "9983.T",  # ファーストリテイリング
    "6501.T",  # 日立製作所
    "8001.T",  # 伊藤忠商事
    "4063.T",  # 信越化学工業
    "6861.T"   # キーエンス
]
```

### validate_exit_simple_v2.pyへの統合

トレンド強度フィルターをGCStrategyのエントリー条件に追加：

```python
# エントリー条件追加案
def generate_entry_signal(self, idx):
    # 既存のGC判定
    if gc_condition:
        # トレンド強度チェック追加
        trend_strength = self.data['trend_strength'].iloc[idx]
        if trend_strength >= self.trend_threshold:  # 閾値以上のみエントリー
            return 1
    return 0
```

---

## 7. 実装ファイル

### スクリプト名
`scripts/validate_trend_strength_filter.py`

### 実行コマンド
```powershell
python scripts/validate_trend_strength_filter.py
```

### 出力ファイル
- Markdown: `docs/exit_strategy/PHASE1_6_TREND_FILTER_RESULT.md`
- 図表: `docs/exit_strategy/figures/trend_*.png` (3枚)

---

## 8. 成功条件チェックリスト

実行前:
- [ ] CSV読み込み成功（840件確認）
- [ ] カラム名設定完了
- [ ] 武田薬品データ抽出成功

実行中:
- [ ] 閾値算出成功（67%ile）
- [ ] フィルター適用成功
- [ ] 成績計算完了（PF, 勝率, R倍）
- [ ] 図表3枚生成完了

実行後:
- [ ] PF改善率 >= 30%達成
- [ ] 勝率改善確認
- [ ] 結果MDファイル作成
- [ ] 次フェーズ（他銘柄検証）の準備完了

---

**作成者**: GitHub Copilot  
**最終更新**: 2026-01-26
