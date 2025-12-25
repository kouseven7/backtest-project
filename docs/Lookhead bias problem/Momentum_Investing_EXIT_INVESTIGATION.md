# Momentum_Investing.py イグジット問題 調査報告書

**調査日**: 2025-12-23  
**調査者**: GitHub Copilot  
**対象ファイル**: strategies/Momentum_Investing.py  
**参照ドキュメント**: docs/Lookhead bias problem/EXIT_INVESTIGATION_REPORT.md

---

## 調査サマリー

**結論**: ✅ **ルックアヘッドバイアス発見（イグジット価格のみ）**

**問題箇所**:
- Line 236: `current_price = self.data[self.price_column].iloc[idx]` - generate_exit_signal()内
- Line 378: `exit_price = self.data[self.price_column].iloc[idx]` - backtest()内

**修正ステータス**:
- Phase 0（インジケーターshift(1)）: ✅ **完了** - 全6インジケーター適用済み
- Phase 1（エントリー価格）: ✅ **完了** - 翌日始値 + スリッページ使用
- Phase 1b（イグジット価格）: ✅ **完了** - 翌日始値使用（**2025-12-23修正完了**）

**影響度**: 高（6つのイグジット条件に影響）

**修正完了日**: 2025-12-23

---

## 詳細調査結果

### [C1] 全体構造確認 ✅

**確認内容**: generate_exit_signal()メソッドの存在確認

**実際のコード**:
```python
# Line 195
def generate_exit_signal(self, idx: int) -> int:
```

**結論**: generate_exit_signal()メソッドが存在

**根拠**: strategies/Momentum_Investing.py Line 195

---

### [C2] BaseStrategy継承確認 ✅

**確認内容**: 継承関係と独自backtest実装の有無

**実際のコード**:
```python
# Line 32
class MomentumInvestingStrategy(BaseStrategy):

# Line 310
def backtest(self, trading_start_date=None, trading_end_date=None):
```

**結論**: 
- BaseStrategyを継承
- 独自backtest()メソッドを実装（部分利確機能付き）

**根拠**: strategies/Momentum_Investing.py Lines 32, 310

---

### [C3] current_price取得箇所確認 ✅

**確認内容**: generate_exit_signal()内のcurrent_price定義

**実際のコード**:
```python
# Line 236
current_price = self.data[self.price_column].iloc[idx]
```

**使用している価格カラム**:
- `self.price_column` = "Adj Close"（デフォルト、Line 35で設定）

**結論**: Line 236でidx日目の`Adj Close`を取得

**根拠**: strategies/Momentum_Investing.py Lines 35, 236

---

### [C4] current_priceが当日終値を使用しているか確認 ✅

**確認内容**: ルックアヘッドバイアスの有無

**実際のコード**:
```python
# Line 236
current_price = self.data[self.price_column].iloc[idx]
```

**詳細**:
- `self.price_column` = "Adj Close"（Line 35）
- **idx日目の終値（Adj Close）を使用** → ❌ルックアヘッドバイアス確定

**問題の構造**:
1. idx日目の`generate_exit_signal(idx)`でイグジット判断
2. `current_price = self.data[self.price_column].iloc[idx]` - idx日目の終値を取得
3. idx日目の終値でストップロス・利確・トレーリングストップ等を判定
4. **リアルトレードでは、idx日目の終値を見てからidx日目の終値で売ることは不可能**

**結論**: ✅ **ルックアヘッドバイアスあり**

**根拠**: strategies/Momentum_Investing.py Lines 35, 236

---

### [C5] entry_price取得方法の確認 ✅

**確認内容**: エントリー価格の取得方法

**実際のコード**:
```python
# Lines 208-235
# エントリー価格を取得
entry_indices = self.data[self.data['Entry_Signal'] == 1].index
if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
    return 0
    
# 最新のエントリー価格を取得
latest_entry_idx_raw = self.data.index.get_loc(entry_indices[-1])
# 型エラー対策: latest_entry_idxをintに変換（sliceやndarrayの場合は例外）
if not isinstance(latest_entry_idx_raw, int):
    raise TypeError(f"latest_entry_idx is not int: {type(latest_entry_idx_raw)}")
latest_entry_idx_int = latest_entry_idx_raw

if latest_entry_idx_int not in self.entry_prices:
    # Phase 1修正: フォールバック処理も翌日始値を使用（ルックアヘッドバイアス修正）
    next_day_pos = latest_entry_idx_int + 1
    if next_day_pos < len(self.data):
        next_day_open = self.data['Open'].iloc[next_day_pos]
    else:
        # 最終日の場合は当日始値を使用（境界条件の妥協案）
        next_day_open = self.data['Open'].iloc[latest_entry_idx_int]
    
    # Phase 2修正: スリッページ・取引コスト考慮（2025-12-23追加）
    slippage = self.params.get("slippage", 0.001)
    transaction_cost = self.params.get("transaction_cost", 0.0)
    self.entry_prices[latest_entry_idx_int] = next_day_open * (1 + slippage + transaction_cost)
    
entry_price = self.entry_prices[latest_entry_idx_int]
```

**特徴**:
- `self.entry_prices`辞書から取得
- フォールバック処理あり（Lines 220-233）: Phase 1修正済み（翌日始値 + スリッページ）
- エントリー価格は独自backtest()で記録（Lines 352-362）

**結論**: ✅ **エントリー価格は翌日始値を使用している**（Phase 1修正完了）

**根拠**: strategies/Momentum_Investing.py Lines 208-235, 352-362

---

### [C6] 各イグジット条件の実装確認 ✅

**確認内容**: 全イグジット条件のリスト化

**イグジット条件一覧**:

| No | イグジット条件 | 実装箇所 | 使用価格 | ルックアヘッドバイアス |
|----|---------------|---------|---------|----------------------|
| 1 | 最大保有期間 | Lines 241-245 | `days_held >= max_hold_days` | ✅ 問題なし（日数のみ） |
| 2 | ATRストップロス | Lines 248-252 | `current_price <= atr_stop_loss` | ❌ **当日終値使用** |
| 3 | 通常ストップロス | Lines 248-252 | `current_price <= entry_price * (1 - stop_loss)` | ❌ **当日終値使用** |
| 4 | モメンタム失速 | Lines 255-261 | RSI変化量（current_price未使用） | ✅ 問題なし |
| 5 | 出来高減少 | Lines 264-270 | 出来高比較（current_price未使用） | ✅ 問題なし |
| 6 | 利益確定 | Lines 273-275 | `current_price >= entry_price * (1 + take_profit)` | ❌ **当日終値使用** |
| 7 | トレーリングストップ | Lines 278-282 | `current_price <= trailing_stop` | ❌ **当日終値使用** |
| 8 | 移動平均線ブレイク | Lines 285-288 | `current_price < sma_short` | ❌ **当日終値使用** |
| 9 | RSI反転 | Lines 291-296 | RSI値（current_price未使用） | ✅ 問題なし |
| 10 | MACD反転 | Lines 298-300 | MACD値（current_price未使用） | ✅ 問題なし |
| 11 | チャートパターン崩壊 | Lines 303-306 | `current_price < recent_high * ...` | ❌ **当日終値使用** |

**影響度分析**:
- **高影響（修正必要）**: ATRストップロス、通常ストップロス、利益確定、トレーリングストップ、移動平均線ブレイク、チャートパターン崩壊（**6条件**）
- **低影響（修正不要）**: 最大保有期間、モメンタム失速、出来高減少、RSI反転、MACD反転（5条件）

**結論**: 11イグジット条件中、**6条件がルックアヘッドバイアスの影響を受ける**

**根拠**: strategies/Momentum_Investing.py Lines 240-306

---

### [C7] Phase 1修正の確認 ✅

**確認内容**: エントリー価格が翌日始値になっているか

**実際のコード**:
```python
# Lines 352-362 (backtest()メソッド内)
# Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
# 理由: idx日の終値を見てからidx日の終値で買うことは不可能
# リアルトレードでは翌日（idx+1日目）の始値でエントリー
next_day_open = self.data['Open'].iloc[idx + 1]

# Phase 2修正: スリッページ・取引コスト考慮（2025-12-23追加）
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
entry_price = next_day_open * (1 + slippage + transaction_cost)

self.entry_prices[idx] = entry_price
```

**イグジット価格の確認**:
```python
# Line 378 (backtest()メソッド内)
exit_price = self.data[self.price_column].iloc[idx]
```

**結論**: 
- ✅ **エントリー価格**: 翌日始値 + スリッページ（Phase 1修正完了）
- ❌ **イグジット価格**: 当日終値（**Phase 1b未完了、修正必要**）

**根拠**: strategies/Momentum_Investing.py Lines 352-362, 378

---

### [C8] インジケーターのshift(1)確認 ✅

**確認内容**: initialize_strategy()でのshift(1)適用状況

**実際のコード**:
```python
# Lines 96-111 (initialize_strategy()メソッド)
# ルックアヘッドバイアス修正: 既に指標列がある場合は再計算しない
if f'MA_{sma_short}' not in self.data.columns:
    if ma_type == "SMA":
        self.data[f'MA_{sma_short}'] = calculate_sma(self.data, self.price_column, sma_short).shift(1)
    elif ma_type == "EMA":
        self.data[f'MA_{sma_short}'] = self.data[self.price_column].ewm(span=sma_short, adjust=False).mean().shift(1)
if f'MA_{sma_long}' not in self.data.columns:
    if ma_type == "SMA":
        self.data[f'MA_{sma_long}'] = calculate_sma(self.data, self.price_column, sma_long).shift(1)
    elif ma_type == "EMA":
        self.data[f'MA_{sma_long}'] = self.data[self.price_column].ewm(span=sma_long, adjust=False).mean().shift(1)
if 'RSI' not in self.data.columns:
    self.data['RSI'] = calculate_rsi(self.data[self.price_column], self.params["rsi_period"]).shift(1)
if 'MACD' not in self.data.columns or 'Signal_Line' not in self.data.columns:
    macd_raw, signal_raw = calculate_macd(self.data, self.price_column)
    self.data['MACD'] = macd_raw.shift(1)
    self.data['Signal_Line'] = signal_raw.shift(1)
if 'ATR' not in self.data.columns:
    self.data['ATR'] = calculate_atr(self.data, self.price_column).shift(1)
```

**インジケーター一覧**:

| インジケーター | shift(1)適用 | 実装箇所 | 状態 |
|---------------|-------------|---------|------|
| MA_Short (SMA) | ✅ | Line 96 | shift(1)適用済み |
| MA_Short (EMA) | ✅ | Line 98 | shift(1)適用済み |
| MA_Long (SMA) | ✅ | Line 101 | shift(1)適用済み |
| MA_Long (EMA) | ✅ | Line 103 | shift(1)適用済み |
| RSI | ✅ | Line 105 | shift(1)適用済み |
| MACD | ✅ | Line 108 | shift(1)適用済み |
| Signal_Line | ✅ | Line 109 | shift(1)適用済み |
| ATR | ✅ | Line 111 | shift(1)適用済み |

**結論**: ✅ **全インジケーターにshift(1)適用済み**（Phase 0修正完了済み）

**根拠**: strategies/Momentum_Investing.py Lines 96, 98, 101, 103, 105, 108-109, 111

---

### [C9] EXIT_INVESTIGATION_REPORT.mdとの比較 ✅

**共通点**:
1. `current_price = self.data[self.price_column].iloc[idx]` - 当日終値使用
2. ストップロス判定が当日終値ベース
3. 利益確定判定が当日終値ベース
4. トレーリングストップ判定が当日終値ベース

**相違点**:
1. **Phase 0修正完了**: 全インジケーターがshift(1)適用済み（pairs_trading_strategy.pyと同じ）
2. **Phase 1修正完了**: エントリー価格が翌日始値 + スリッページ（pairs_trading_strategy.pyと同じ）
3. **Phase 1b未完了**: イグジット価格が当日終値（Lines 236, 378）
4. **より多様なイグジット条件**: 11条件（最大保有期間、ATRストップロス、通常ストップロス、モメンタム失速、出来高減少、利確、トレーリングストップ、MA線ブレイク、RSI反転、MACD反転、チャートパターン崩壊）
5. **部分利確機能あり**: partial_exit_pct（Lines 383-394）
6. **強制決済処理あり**: バックテスト終了時（Lines 397-406）

**問題構造**:
- VWAP_Breakout.py、support_resistance_contrarian_strategy.py、pairs_trading_strategy.pyと**同じ問題構造**
- イグジット価格だけがルックアヘッドバイアス（インジケーターは修正済み）
- Phase 1b修正（イグジット価格を翌日始値に変更）が必要

**深刻度**:
- VWAP_Breakout.pyより**深刻**（イグジット条件が多い）
- support_resistance_contrarian_strategy.pyと**同等**（イグジット条件数が同程度）
- pairs_trading_strategy.pyより**深刻**（イグジット条件が多い）

**根拠**: EXIT_INVESTIGATION_REPORT.md比較 + 本調査結果

---

## 修正必要箇所サマリー

### 直接修正（優先度：最高）

**箇所1: generate_exit_signal()内のcurrent_price定義（Line 235-242）**

**修正前**:
```python
# Line 235（修正前）
current_price = self.data[self.price_column].iloc[idx]
```

**修正後（2025-12-23完了）**:
```python
# Line 235-242（修正後）
# Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
# 理由: idx日目の終値を見てからidx日目の終値で売ることは不可能
# リアルトレードでは翌日（idx+1日目）の始値でイグジット
current_price_val = self.data['Open'].iloc[idx + 1]
if isinstance(current_price_val, pd.Series):
    current_price_val = current_price_val.values[0]
current_price = current_price_val
```

**修正ステータス**: ✅ 完了

---

**箇所2: backtest()内のexit_price定義（Line 385-386）**

**修正前**:
```python
# Line 378（修正前）
exit_price = self.data[self.price_column].iloc[idx]
```

**修正後（2025-12-23完了）**:
```python
# Line 385-386（修正後）
# Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
exit_price = self.data['Open'].iloc[idx + 1]
```

**修正ステータス**: ✅ 完了

---

### 境界条件の確認

**ループ範囲**: strategies/Momentum_Investing.py Line 330
```python
for idx in range(len(self.data) - 1):
```

**安全性確認**:
- ループ範囲は`len(self.data) - 1`まで
- `idx + 1`でアクセスしても境界条件エラーは発生しない
- **結論**: `self.data['Open'].iloc[idx + 1]`は安全

---

## 影響範囲

### 間接影響（current_priceを使用するイグジット条件）

| No | イグジット条件 | 実装箇所 | 影響 |
|----|---------------|---------|------|
| 1 | ATRストップロス | Lines 248-252 | current_price使用（間接影響） |
| 2 | 通常ストップロス | Lines 248-252 | current_price使用（間接影響） |
| 3 | 利益確定 | Lines 273-275 | current_price使用（間接影響） |
| 4 | トレーリングストップ | Lines 278-282 | current_price使用（間接影響） |
| 5 | 移動平均線ブレイク | Lines 285-288 | current_price使用（間接影響） |
| 6 | チャートパターン崩壊 | Lines 303-306 | current_price使用（間接影響） |

**影響なし**（current_price未使用のイグジット条件）:
- 最大保有期間（Lines 241-245）
- モメンタム失速（Lines 255-261）
- 出来高減少（Lines 264-270）
- RSI反転（Lines 291-296）
- MACD反転（Lines 298-300）

---

## セルフチェック結果

### a) 見落としチェック ✅
- [x] generate_exit_signal()の全ロジック確認（Lines 195-308）
- [x] backtest()メソッドの確認（Lines 310-415）
- [x] current_price取得箇所の確認（Lines 236, 378）
- [x] 各イグジット条件の確認（Lines 240-306）
- [x] Phase 0修正の確認（Lines 96-111）→完了済み
- [x] Phase 1修正の確認（Lines 352-362）→完了済み
- [x] ループ範囲の確認（Line 330: `range(len(self.data) - 1)`）→安全

### b) 思い込みチェック ✅
- ❌ 「Phase 1修正でイグジットも修正されているはず」 → ✅ 実際はLines 236, 378未修正
- ❌ 「インジケーターがshift(1)未適用のはず」 → ✅ 実際は全て適用済み
- ✅ Lines 236, 378で`current_price = self.data[self.price_column].iloc[idx]`を確認
- ✅ 当日終値を使用していることを確認

### c) 矛盾チェック ✅
- Phase 0完了（インジケーター） vs Phase 1b未完了（イグジット価格） → **整合** → 予想通り
- Phase 1完了（エントリー価格） vs Phase 1b未完了（イグジット価格） → **整合** → エントリーのみ修正済み
- EXIT_INVESTIGATION_REPORT.mdの問題構造と一致 → **整合** → 同じパターン

---

## 推奨する修正手順

1. **Phase 1b修正実施（最高優先）** - ✅ **完了（2025-12-23）**
   - strategies/Momentum_Investing.py Line 235-242: current_price定義を翌日始値に変更 + Series型エラー対策
   - strategies/Momentum_Investing.py Line 385-386: exit_price定義を翌日始値に変更

2. **検証テスト作成・実行** - ✅ **完了（2025-12-23）**
   - tests/temp/test_20251223_momentum_investing_exit_price_check.py作成・実行
   - イグジット価格が翌日始値に変更されていることを確認（36/36件、100.0%）
   - Phase 0/Phase 1b全て成功

3. **ドキュメント更新** - ✅ **完了（2025-12-23）**
   - 本調査報告書を更新（Phase 1b修正完了記録）

---

## 検証テスト結果（2025-12-23実施）

**テストファイル**: tests/temp/test_20251223_momentum_investing_exit_price_check.py

**テスト期間**: 2024-01-01 ~ 2024-12-31（251データポイント）

**テスト銘柄**: AAPL

**Phase 0確認結果**:
- ✅ **PASSED** - 全6インジケーター（MA_10, MA_30, RSI, MACD, Signal_Line, ATR）にshift(1)適用確認

**Phase 1b確認結果**:
- ✅ **PASSED** - イグジット価格翌日始値使用確認
- エントリー件数: 37件
- イグジット件数: 37件
- 翌日始値使用件数: **36/36件 (100.0%)**（最終日除く）
- 平均差分: **+0.9422%**（当日終値 vs 翌日始値）
- 標準偏差: 0.7890%
- 最小差分: -0.5386%
- 最大差分: +3.2733%
- 絶対値平均: 0.9815%

**総合判定**: ✅ **PASSED** - Phase 0/Phase 1b全て成功

---

**調査完了日**: 2025-12-23  
**修正完了日**: 2025-12-23  
**検証完了日**: 2025-12-23  
**調査ステータス**: ✅ **完了** - Phase 0/Phase 1/Phase 1b全て修正完了・検証完了  
**次のアクション**: なし（全タスク完了）
