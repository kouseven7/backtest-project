# Phase 1.6大敗パターン分析設計書

**作成日**: 2026-01-26  
**分析対象**: 武田薬品工業（4502.T）  
**データソース**: results/phase1.6_trades_20260126_124409a.csv

---

## 1. 目的

Phase 1.6グリッドサーチにおける大敗銘柄（武田薬品4502.T）の取引履歴から、**大敗の原因となるパターンを特定**し、今後のエグジット戦略改善に活用する。

### 最終ゴール（成功条件）
- 分析1～5を完了する
- 各分析結果をドキュメント化する
- 大敗パターンを3-5個抽出する
- 改善提案（エグジット戦略の調整方針）を示す

---

## 2. データ構造確認

### CSVファイル構造
```
カラム一覧:
- entry_date: エントリー日
- entry_price: エントリー価格
- exit_date: エグジット日
- exit_price: エグジット価格
- profit_loss: 損益額
- exit_reason: エグジット理由（stop_loss/trailing_stop/dead_cross/force_close）
- holding_days: 保有日数
- profit_loss_pct: 損益率（%）
- r_multiple: リスクリワードレシオ
- entry_gap_pct: エントリーギャップ率（%）
- max_profit_pct: 最大利益率（%）
- entry_atr_pct: エントリー時ATR（%）
- sma_distance_pct: SMA乖離率（%）
- entry_trend_strength: エントリー時トレンド強度
- entry_volume: エントリー時出来高
- exit_volume: エグジット時出来高
- ticker: ティッカー（4502.T）
- stop_loss_pct: 損切設定（0.03/0.05/0.07）
- trailing_stop_pct: トレーリング設定（0.05～0.30）
```

### フィルター条件
```python
# 武田薬品のみ抽出
df = pd.read_csv('results/phase1.6_trades_20260126_124409a.csv')
takeda_df = df[df['ticker'] == '4502.T'].copy()
```

---

## 3. 分析設計

### 分析1: 大損失トレードの特徴抽出

**目的**: profit_loss_pct < -10%の大損失トレードの特徴を定量化

**手順**:
1. 大損失トレード（profit_loss_pct < -10%）を抽出
2. 通常トレード（profit_loss_pct >= -10%）と比較

**分析項目**:
- 取引件数
- 平均値: entry_gap_pct, entry_atr_pct, holding_days, max_profit_pct, entry_trend_strength, sma_distance_pct
- exit_reason内訳（件数と割合）

**出力物**:
- 比較表（CSVまたはMarkdownテーブル）
- exit_reasonの円グラフ
- entry_atr_pctのヒストグラム（大損失 vs 通常）
- entry_gap_pctのヒストグラム（大損失 vs 通常）

**Python実装概要**:
```python
# 大損失トレード抽出
big_loss = takeda_df[takeda_df['profit_loss_pct'] < -10]
normal = takeda_df[takeda_df['profit_loss_pct'] >= -10]

# 統計計算
big_loss_stats = big_loss[['entry_gap_pct', 'entry_atr_pct', 
                            'holding_days', 'max_profit_pct', 
                            'entry_trend_strength', 'sma_distance_pct']].describe()

# exit_reason内訳
exit_reason_counts = big_loss['exit_reason'].value_counts()
```

---

### 分析2: 時系列での勝敗パターン

**目的**: 2023年8-10月を中心に、月別・四半期別の成績変動を可視化

**手順**:
1. entry_dateをdatetime型に変換
2. 月別・四半期別に集計

**集計項目**（各期間ごと）:
- 総取引数
- 勝ち取引数（profit_loss_pct > 0）
- 勝率
- 平均profit_loss_pct
- 最大損失（profit_loss_pct最小値）
- 合計損益

**出力物**:
- 月別集計テーブル（Markdown形式）
- 月別平均profit_loss_pctの折れ線グラフ
- 月別勝率の折れ線グラフ
- 「平均profit_loss_pct < -5%」の期間リスト

**Python実装概要**:
```python
# 日付変換
takeda_df['entry_date'] = pd.to_datetime(takeda_df['entry_date'])
takeda_df['year_month'] = takeda_df['entry_date'].dt.to_period('M')

# 月別集計
monthly = takeda_df.groupby('year_month').agg({
    'profit_loss_pct': ['count', 'mean', 'min'],
    # 勝ち取引数は別途計算
})

# 勝率計算
monthly['win_rate'] = takeda_df.groupby('year_month').apply(
    lambda x: (x['profit_loss_pct'] > 0).mean()
)
```

---

### 分析3: エントリー品質の分析

**目的**: エントリー時の状態（SMA乖離、トレンド強さ）と成績の相関を分析

**分析3-1: 移動平均線との乖離**

sma_distance_pct範囲分類:
- 0-5%（GC直後）
- 5-10%
- 10-15%
- 15-20%
- 20%以上（追いかけエントリー）

各範囲での集計:
- 取引数
- 勝率
- 平均profit_loss_pct
- 平均r_multiple

**分析3-2: トレンド強さ**

entry_trend_strengthを3分割:
- 低: 0-33パーセンタイル
- 中: 33-66パーセンタイル
- 高: 66-100パーセンタイル

**出力物**:
- sma_distance_pct範囲別成績比較表
- 棒グラフ（横軸: sma_distance_pct範囲、縦軸: 平均profit_loss_pct）
- entry_trend_strength別成績比較表

**Python実装概要**:
```python
# SMA乖離範囲分類
def classify_sma_distance(val):
    if val < 5: return '0-5%'
    elif val < 10: return '5-10%'
    elif val < 15: return '10-15%'
    elif val < 20: return '15-20%'
    else: return '20%+'

takeda_df['sma_range'] = takeda_df['sma_distance_pct'].apply(classify_sma_distance)

# 範囲別集計
sma_analysis = takeda_df.groupby('sma_range').agg({
    'profit_loss_pct': ['count', 'mean'],
    'r_multiple': 'mean'
})
sma_analysis['win_rate'] = takeda_df.groupby('sma_range').apply(
    lambda x: (x['profit_loss_pct'] > 0).mean()
)
```

---

### 分析4: ボラティリティの影響

**目的**: entry_atr_pctとprofit_loss_pctの関係を定量化

**手順**:
1. entry_atr_pctの基本統計（平均、中央値、最大、最小）
2. ATR範囲別分類と成績集計
3. 散布図作成

**ATR範囲分類**:
- 低: 0-2%
- 中: 2-3%
- 高: 3-4%
- 超高: 4%以上

**出力物**:
- ATR基本統計テーブル
- ATR範囲別成績比較表
- 散布図（横軸: entry_atr_pct、縦軸: profit_loss_pct）
- entry_atr_pct > 3.5%の取引詳細リスト（上位10件）

**Python実装概要**:
```python
# ATR範囲分類
def classify_atr(val):
    if val < 2: return '低 (0-2%)'
    elif val < 3: return '中 (2-3%)'
    elif val < 4: return '高 (3-4%)'
    else: return '超高 (4%+)'

takeda_df['atr_range'] = takeda_df['entry_atr_pct'].apply(classify_atr)

# 範囲別集計
atr_analysis = takeda_df.groupby('atr_range').agg({
    'profit_loss_pct': ['count', 'mean'],
})
atr_analysis['win_rate'] = takeda_df.groupby('atr_range').apply(
    lambda x: (x['profit_loss_pct'] > 0).mean()
)

# 高ATRトレード抽出
high_atr = takeda_df[takeda_df['entry_atr_pct'] > 3.5].sort_values(
    'entry_atr_pct', ascending=False
).head(10)
```

---

### 分析5: 急騰→急落パターンの検出

**目的**: 「一度大きく上昇したのに損失で終了」のパターンを定量化

**検出条件**:
1. max_profit_pct > 15%（一度大きく上昇）
2. profit_loss_pct < 0%（最終的に損失）
3. holding_days < 30日（短期間で転落）

**分析項目**:
- 該当取引件数と割合
- 合計損失額
- 平均max_profit_pct（最大到達点）
- 平均profit_loss_pct（最終損失）
- trailing_stop_pct分布
- exit_reason内訳

**出力物**:
- 該当取引リスト（entry_date, exit_date, max_profit_pct, profit_loss_pct, trailing_stop_pct, exit_reason）
- 時系列プロット（該当取引の発生時期）
- 散布図（横軸: max_profit_pct、縦軸: profit_loss_pct）
- 全体成績への影響度（該当取引を除いた場合のPF比較）

**Python実装概要**:
```python
# 急騰→急落パターン検出
pump_and_dump = takeda_df[
    (takeda_df['max_profit_pct'] > 15) &
    (takeda_df['profit_loss_pct'] < 0) &
    (takeda_df['holding_days'] < 30)
]

# 統計計算
pattern_count = len(pump_and_dump)
total_trades = len(takeda_df)
pattern_ratio = pattern_count / total_trades

total_loss = pump_and_dump['profit_loss'].sum()

# 全体成績への影響度
pf_with_pattern = calculate_pf(takeda_df)
pf_without_pattern = calculate_pf(takeda_df[~takeda_df.index.isin(pump_and_dump.index)])
```

---

## 4. 実装フロー

### Step 1: 環境準備
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 日本語フォント設定（Windows）
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False
```

### Step 2: データ読み込み
```python
csv_path = Path('results/phase1.6_trades_20260126_124409a.csv')
df = pd.read_csv(csv_path)
takeda_df = df[df['ticker'] == '4502.T'].copy()

# 日付変換
takeda_df['entry_date'] = pd.to_datetime(takeda_df['entry_date'])
takeda_df['exit_date'] = pd.to_datetime(takeda_df['exit_date'])
```

### Step 3: 分析実行（順次）
1. analyze_big_loss_trades()
2. analyze_time_series_patterns()
3. analyze_entry_quality()
4. analyze_volatility_impact()
5. detect_pump_and_dump_pattern()

### Step 4: 結果保存
- 図表: docs/exit_strategy/figures/
- 分析結果MD: docs/exit_strategy/PHASE1_6_DEFEAT_PATTERNS_RESULT.md

---

## 5. 成功条件チェックリスト

- [ ] 分析1完了: 大損失トレード特徴抽出
- [ ] 分析2完了: 時系列パターン可視化
- [ ] 分析3完了: エントリー品質分析
- [ ] 分析4完了: ボラティリティ影響分析
- [ ] 分析5完了: 急騰→急落パターン検出
- [ ] 全ての図表が保存されている
- [ ] 結果MDファイルに全分析結果が記載されている
- [ ] 大敗パターン3-5個を特定できた
- [ ] 改善提案（エグジット戦略調整方針）を記載した

---

## 6. 期待される発見

### 仮説1: 高ATR時のエントリーが大損失の主因
- entry_atr_pct > 3.5%の取引で大損失が集中している可能性

### 仮説2: 追いかけエントリーが失敗要因
- sma_distance_pct > 20%の取引で成績が悪化している可能性

### 仮説3: トレーリングストップが機能していない
- max_profit_pct > 15%でも損失になるケースが多数存在
- trailing_stop_pct設定が広すぎる可能性

### 仮説4: 2023年8-10月に特異な市場環境
- 武田薬品特有のイベント（決算、ニュース等）が影響

### 仮説5: 短期保有での損失が顕著
- holding_days < 10日の取引で大損失が多い可能性

---

## 7. 改善提案の方向性

分析結果に基づき、以下の改善提案を検討:

1. **ATRフィルター導入**
   - entry_atr_pct > 3.5%の場合エントリー回避
   - またはストップロス幅を動的調整

2. **SMA乖離制限**
   - sma_distance_pct > 20%の追いかけエントリー禁止

3. **トレーリングストップ最適化**
   - max_profit_pct到達後のストップ幅を狭める
   - 段階的ストップ調整（利益確定ゾーン設定）

4. **時期別パラメータ調整**
   - 特定月（8-10月）のエントリー抑制またはパラメータ変更

5. **最大保有期間制限**
   - holding_days > 50日で強制決済検討

---

## 8. 次のアクション

1. 本設計書に基づき、分析スクリプト作成
   - ファイル名: scripts/analyze_phase1_6_defeat_patterns.py
2. 分析実行
3. 結果ドキュメント作成
   - ファイル名: docs/exit_strategy/PHASE1_6_DEFEAT_PATTERNS_RESULT.md
4. PHASE1_6_DEFEAT_PATTERNS.mdに発見事項を追記

---

**作成者**: GitHub Copilot  
**最終更新**: 2026-01-26
