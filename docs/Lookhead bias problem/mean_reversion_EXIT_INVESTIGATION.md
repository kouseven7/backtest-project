# mean_reversion_strategy.py イグジット問題 調査報告書

**調査日**: 2025-12-23  
**調査者**: GitHub Copilot  
**対象ファイル**: strategies/mean_reversion_strategy.py  
**参照ドキュメント**: docs/Lookhead bias problem/EXIT_INVESTIGATION_REPORT.md

---

## 調査サマリー

**結論**: ✅ **ルックアヘッドバイアス発見（イグジット価格のみ）**

**問題箇所**:
- Line 214: `current_price_val = self.data[self.price_column].iloc[idx]` - generate_exit_signal()内

**修正ステータス**:
- Phase 0（インジケーターshift(1)）: ✅ **完了** - 全8インジケーター適用済み
- Phase 1（エントリー価格）: ✅ **完了** - 翌日始値 + スリッページ使用
- Phase 1b（イグジット価格）: ✅ **完了（2025-12-23）** - 翌日始値使用

**影響度**: 高（5つのイグジット条件に影響）

**修正完了記録**:
- 修正日: 2025-12-23
- 修正箇所: strategies/mean_reversion_strategy.py Lines 213-218
- 修正内容: `current_price_val = self.data[self.price_column].iloc[idx]` → `self.data['Open'].iloc[idx + 1]`
- Series型スカラー化処理: 保持（Lines 217-218）
- Phase 1b修正コメント追加: Lines 213-215（理由明記）

---

## 詳細調査結果

### [C1] 全体構造確認 ✅

**確認内容**: generate_exit_signal()メソッドの存在確認

**実際のコード**:
```python
# Line 208
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
```

**結論**: generate_exit_signal()メソッドが存在

**根拠**: strategies/mean_reversion_strategy.py Line 208

---

### [C2] BaseStrategy継承確認 ✅

**確認内容**: 継承関係と独自backtest実装の有無

**実際のコード**:
```python
# Line 25
class MeanReversionStrategy(BaseStrategy):

# Line 278
def backtest(self, trading_start_date=None, trading_end_date=None) -> pd.DataFrame:
```

**結論**: 
- BaseStrategyを継承
- 独自backtest()メソッドを実装

**根拠**: strategies/mean_reversion_strategy.py Lines 25, 278

---

### [C3] current_price取得箇所確認 ✅

**確認内容**: generate_exit_signal()内のcurrent_price定義

**実際のコード**:
```python
# Lines 214-217
# スカラー値として取得
current_price_val = self.data[self.price_column].iloc[idx]
if isinstance(current_price_val, pd.Series):
    current_price_val = current_price_val.values[0]
```

**使用している価格カラム**:
- `self.price_column` = "Adj Close"（デフォルト、Line 29, 37）

**結論**: Line 214でidx日目の`Adj Close`を取得

**根拠**: strategies/mean_reversion_strategy.py Lines 29, 37, 214-217

---

### [C4] current_priceが当日終値を使用しているか確認 ✅

**確認内容**: ルックアヘッドバイアスの有無

**実際のコード**:
```python
# Line 214
current_price_val = self.data[self.price_column].iloc[idx]
```

**詳細**:
- `self.price_column` = "Adj Close"（Line 29, 37）
- **idx日目の終値（Adj Close）を使用** → ❌ルックアヘッドバイアス確定

**問題の構造**:
1. idx日目の`generate_exit_signal(idx)`でイグジット判断
2. `current_price_val = self.data[self.price_column].iloc[idx]` - idx日目の終値を取得
3. idx日目の終値でストップロス・利確・平均回帰完了等を判定
4. **リアルトレードでは、idx日目の終値を見てからidx日目の終値で売ることは不可能**

**結論**: ✅ **ルックアヘッドバイアスあり**

**根拠**: strategies/mean_reversion_strategy.py Lines 214-217

---

### [C5] entry_price取得方法の確認 ✅

**確認内容**: エントリー価格の取得方法

**実際のコード**:
```python
# Lines 306-322 (backtest()メソッド内)
# Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
# Phase 2修正: スリッページ・取引コスト対応（2025-12-23追加）
# 理由: i日の終値を見てからi日の終値で買うことは不可能
# リアルトレードでは翌日（i+1日目）の始値でエントリー
next_day_open_val = result_data['Open'].iloc[i + 1]
if isinstance(next_day_open_val, pd.Series):
    next_day_open_val = next_day_open_val.values[0]

# Phase 2: スリッページ・取引コスト適用（買い注文は不利な方向）
# デフォルト: slippage=0.001（0.1%）、transaction_cost=0.0（0%）
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
entry_price = next_day_open_val * (1 + slippage + transaction_cost)
self.entry_prices[i] = entry_price
```

**特徴**:
- `self.entry_prices`辞書から取得（Lines 219-228）
- エントリー価格は独自backtest()で記録（Lines 306-322）
- Phase 1修正済み（翌日始値 + スリッページ + 取引コスト）

**結論**: ✅ **エントリー価格は翌日始値を使用している**（Phase 1修正完了）

**根拠**: strategies/mean_reversion_strategy.py Lines 219-228, 306-322

---

### [C6] 各イグジット条件の実装確認 ✅

**確認内容**: 全イグジット条件のリスト化

**イグジット条件一覧**:

| No | イグジット条件 | 実装箇所 | 使用価格 | ルックアヘッドバイアス |
|----|---------------|---------|---------|----------------------|
| 1 | 最大保有日数 | Lines 232-234 | `hold_days >= max_hold_days` | ✅ 問題なし（日数のみ） |
| 2 | ストップロス | Lines 240-242 | `pnl_pct <= -stop_loss_pct`（current_price_val使用） | ❌ **当日終値使用** |
| 3 | 利益確定 | Lines 245-247 | `pnl_pct >= take_profit_pct`（current_price_val使用） | ❌ **当日終値使用** |
| 4 | ATRストップロス | Lines 250-256 | `pnl_pct <= -atr_stop_loss`（current_price_val使用） | ❌ **当日終値使用** |
| 5 | Z-score平均回帰完了 | Lines 259-265 | `z_score_val >= zscore_exit_threshold`（pnl_pctでcurrent_price_val使用） | ❌ **当日終値使用** |
| 6 | SMA平均回帰完了 | Lines 268-274 | `current_price_val >= sma_val * 0.995` | ❌ **当日終値使用** |

**影響度分析**:
- **高影響（修正必要）**: ストップロス、利益確定、ATRストップロス、Z-score平均回帰完了、SMA平均回帰完了（**5条件**）
- **低影響（修正不要）**: 最大保有日数（1条件）

**結論**: 6イグジット条件中、**5条件がルックアヘッドバイアスの影響を受ける**

**根拠**: strategies/mean_reversion_strategy.py Lines 232-274

---

### [C7] Phase 1修正の確認 ✅

**確認内容**: エントリー価格が翌日始値になっているか

**実際のコード**:
```python
# Lines 306-322 (backtest()メソッド内)
# Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
# Phase 2修正: スリッページ・取引コスト対応（2025-12-23追加）
# 理由: i日の終値を見てからi日の終値で買うことは不可能
# リアルトレードでは翌日（i+1日目）の始値でエントリー
next_day_open_val = result_data['Open'].iloc[i + 1]
if isinstance(next_day_open_val, pd.Series):
    next_day_open_val = next_day_open_val.values[0]

# Phase 2: スリッページ・取引コスト適用（買い注文は不利な方向）
# デフォルト: slippage=0.001（0.1%）、transaction_cost=0.0（0%）
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
entry_price = next_day_open_val * (1 + slippage + transaction_cost)
self.entry_prices[i] = entry_price
```

**イグジット価格の確認**:
```python
# Line 214 (generate_exit_signal()メソッド内)
current_price_val = self.data[self.price_column].iloc[idx]
```

**結論**: 
- ✅ **エントリー価格**: 翌日始値 + スリッページ + 取引コスト（Phase 1修正完了）
- ❌ **イグジット価格**: 当日終値（**Phase 1b未完了、修正必要**）

**根拠**: strategies/mean_reversion_strategy.py Lines 214, 306-322

---

### [C8] インジケーターのshift(1)確認 ✅

**確認内容**: initialize_strategy()でのshift(1)適用状況

**実際のコード**:
```python
# Lines 73-75: SMA
self.data['SMA'] = self.data[self.price_column].rolling(
    window=self.params["sma_period"]
).mean().shift(1)

# Lines 85-87: ボリンジャーバンド
self.data['BB_Upper'] = (bb_sma + (bb_std * self.params["bb_std_dev"])).shift(1)
self.data['BB_Lower'] = (bb_sma - (bb_std * self.params["bb_std_dev"])).shift(1)
self.data['BB_Middle'] = bb_sma.shift(1)

# Line 97: Z-score
self.data['Z_Score'] = ((self.data[self.price_column] - z_sma) / z_std).shift(1)

# Line 101: RSI
self.data['RSI'] = self._calculate_rsi().shift(1)

# Line 105: ATR
self.data['ATR'] = self._calculate_atr().shift(1)

# Lines 109-110: ボリューム移動平均
self.data['Volume_MA'] = self.data['Volume'].rolling(
    window=self.params["sma_period"]
).mean().shift(1)
```

**インジケーター一覧**:

| インジケーター | shift(1)適用 | 実装箇所 | 状態 |
|---------------|-------------|---------|------|
| SMA | ✅ | Lines 73-75 | shift(1)適用済み |
| BB_Upper | ✅ | Line 85 | shift(1)適用済み |
| BB_Lower | ✅ | Line 86 | shift(1)適用済み |
| BB_Middle | ✅ | Line 87 | shift(1)適用済み |
| Z_Score | ✅ | Line 97 | shift(1)適用済み |
| RSI | ✅ | Line 101 | shift(1)適用済み |
| ATR | ✅ | Line 105 | shift(1)適用済み |
| Volume_MA | ✅ | Lines 109-110 | shift(1)適用済み |

**結論**: ✅ **全インジケーターにshift(1)適用済み**（Phase 0修正完了済み）

**根拠**: strategies/mean_reversion_strategy.py Lines 73-110

---

### [C9] EXIT_INVESTIGATION_REPORT.mdとの比較 ✅

**共通点**:
1. `current_price = self.data[self.price_column].iloc[idx]` - 当日終値使用
2. ストップロス判定が当日終値ベース
3. 利益確定判定が当日終値ベース
4. BaseStrategy継承、独自backtest実装

**相違点**:
1. **Phase 0修正完了**: 全インジケーター（8種類）がshift(1)適用済み（pairs_trading_strategy.pyと同じ）
2. **Phase 1修正完了**: エントリー価格が翌日始値 + スリッページ + 取引コスト（pairs_trading_strategy.pyと同じ）
3. **Phase 1b未完了**: イグジット価格が当日終値（Line 214）
4. **平均回帰戦略特有のイグジット条件**: Z-score平均回帰完了、SMA平均回帰完了
5. **統計的異常値検出**: ボリンジャーバンド、Z-scoreを使用

**問題構造**:
- VWAP_Breakout.py、support_resistance_contrarian_strategy.py、pairs_trading_strategy.py、Momentum_Investing.pyと**同じ問題構造**
- イグジット価格だけがルックアヘッドバイアス（インジケーターは修正済み）
- Phase 1b修正（イグジット価格を翌日始値に変更）が必要

**深刻度**:
- VWAP_Breakout.pyと**同等**（イグジット条件数が同程度）
- support_resistance_contrarian_strategy.pyと**同等**（イグジット条件数が同程度）
- pairs_trading_strategy.pyと**同等**（イグジット条件数が同程度）
- Momentum_Investing.pyよりやや**軽度**（イグジット条件が少ない）

**根拠**: EXIT_INVESTIGATION_REPORT.md比較 + 本調査結果

---

## 修正必要箇所サマリー

### 直接修正（優先度：最高）

**箇所: generate_exit_signal()内のcurrent_price定義（Line 214-217）**

**修正前**:
```python
# Lines 214-217（修正前）
# スカラー値として取得
current_price_val = self.data[self.price_column].iloc[idx]
if isinstance(current_price_val, pd.Series):
    current_price_val = current_price_val.values[0]
```

**修正後（完了）**:
```python
# Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
# 理由: idx日目の終値を見てからidx日目の終値で売ることは不可能
# リアルトレードでは翌日（idx+1日目）の始値でイグジット
current_price_val = self.data['Open'].iloc[idx + 1]
if isinstance(current_price_val, pd.Series):
    current_price_val = current_price_val.values[0]
```

**修正実施日**: 2025-12-23  
**修正完了**: ✅ strategies/mean_reversion_strategy.py Lines 213-218修正完了

---

### 境界条件の確認

**ループ範囲**: strategies/mean_reversion_strategy.py Line 289
```python
for i in range(len(result_data) - 1):
```

**安全性確認**:
- ループ範囲は`len(result_data) - 1`まで
- `idx + 1`でアクセスしても境界条件エラーは発生しない
- **結論**: `self.data['Open'].iloc[idx + 1]`は安全

---

## 影響範囲

### 間接影響（current_price_valを使用するイグジット条件）

| No | イグジット条件 | 実装箇所 | 影響 |
|----|---------------|---------|------|
| 1 | ストップロス | Lines 240-242 | current_price_val使用（間接影響） |
| 2 | 利益確定 | Lines 245-247 | current_price_val使用（間接影響） |
| 3 | ATRストップロス | Lines 250-256 | current_price_val使用（間接影響） |
| 4 | Z-score平均回帰完了 | Lines 259-265 | current_price_val使用（間接影響） |
| 5 | SMA平均回帰完了 | Lines 268-274 | current_price_val使用（直接影響） |

**影響なし**（current_price_val未使用のイグジット条件）:
- 最大保有日数（Lines 232-234）

---

## セルフチェック結果

### a) 見落としチェック ✅
- [x] generate_exit_signal()の全ロジック確認（Lines 208-276）
- [x] backtest()メソッドの確認（Lines 278-346）
- [x] current_price取得箇所の確認（Line 214）
- [x] 各イグジット条件の確認（Lines 232-274）
- [x] Phase 0修正の確認（Lines 70-110）→完了済み
- [x] Phase 1修正の確認（Lines 306-322）→完了済み
- [x] ループ範囲の確認（Line 289: `range(len(result_data) - 1)`）→安全

### b) 思い込みチェック ✅
- ❌ 「Phase 1修正でイグジットも修正されているはず」 → ✅ 実際はLine 214未修正
- ❌ 「インジケーターがshift(1)未適用のはず」 → ✅ 実際は全て適用済み
- ✅ Line 214で`current_price_val = self.data[self.price_column].iloc[idx]`を確認
- ✅ 当日終値を使用していることを確認

### c) 矛盾チェック ✅
- Phase 0完了（インジケーター） vs Phase 1b未完了（イグジット価格） → **整合** → 予想通り
- Phase 1完了（エントリー価格） vs Phase 1b未完了（イグジット価格） → **整合** → エントリーのみ修正済み
- EXIT_INVESTIGATION_REPORT.mdの問題構造と一致 → **整合** → 同じパターン

---

## 推奨する修正手順

1. **Phase 1b修正実施（最高優先）** ✅ **完了（2025-12-23）**
   - strategies/mean_reversion_strategy.py Lines 213-218: current_price定義を翌日始値に変更 + 修正理由コメント追加

2. **検証テスト作成・実行** ✅ **完了（2025-12-23）**
   - tests/temp/test_20251223_mean_reversion_exit_price_check.py作成・実行
   - 修正後のバックテスト実行
   - イグジット価格が翌日始値に変更されていることを確認
   - 取引件数・損益の変化を確認

**検証結果**:
- **取引件数**: 9取引（エントリー9件、イグジット9件）
- **Phase 0確認**: PASSED - 全8インジケーター（SMA, BB_Upper, BB_Lower, BB_Middle, Z_Score, RSI, ATR, Volume_MA）shift(1)適用済み
- **Phase 1b確認**: PASSED - 9/9件（100.0%）翌日始値使用確認
- **統計情報**:
  - 平均差分: +0.8888%
  - 標準偏差: 1.2272%
  - 最小差分: -1.6830%
  - 最大差分: +2.5222%
  - 絶対値平均: 1.3550%

3. **ドキュメント更新** ✅ **完了（2025-12-23）**
   - 本調査報告書を更新（修正完了記録、検証結果追加）
   - EXIT_INVESTIGATION_REPORT.mdに参照追加（平均回帰戦略編）

---

**調査完了日**: 2025-12-23  
**修正完了日**: 2025-12-23  
**検証完了日**: 2025-12-23  
**修正ステータス**: Phase 0/Phase 1/Phase 1b修正完了、検証完了  
**次のアクション**: なし（全タスク完了）
