# contrarian_strategy.py イグジット編 調査報告書

**作成日**: 2025-12-23  
**調査範囲**: strategies/contrarian_strategy.py (generate_exit_signal)  
**調査者**: GitHub Copilot  
**関連ドキュメント**: [EXIT_INVESTIGATION_REPORT.md](EXIT_INVESTIGATION_REPORT.md)

---

## 目次

1. [調査チェックリスト](#調査チェックリスト)
2. [調査結果](#調査結果)
3. [修正が必要な箇所](#修正が必要な箇所)
4. [セルフチェック](#セルフチェック)
5. [調査結果のまとめ](#調査結果のまとめ)
6. [修正方針](#修正方針)

---

## 調査チェックリスト

### Phase 0/1/1b修正状況の確認

- **[C1]** 全体構造確認: BaseStrategy継承、backtest()実装状況 ✅完了
- **[C2]** generate_exit_signal()の実装確認 ✅完了
- **[C3]** current_price取得方法確認（Line 221: 当日終値 vs 翌日始値） ✅完了
- **[C4]** entry_price取得方法確認（Line 217: self.entry_prices辞書） ✅完了
- **[C5]** Phase 1修正確認（独自backtest()実装: Lines 254-302） ✅完了
- **[C6]** Phase 1b修正確認（各イグジット条件） ✅完了
- **[C7]** Phase 0修正確認（RSI.shift(1): Line 68） ✅完了

### イグジット条件の詳細確認

- **[C8]** RSIイグジット（Lines 224-226） ✅完了
- **[C9]** トレーリングストップ（Lines 228-233） ✅完了
- **[C10]** 利益確定（Lines 235-237） ✅完了
- **[C11]** ストップロス（Lines 239-241） ✅完了
- **[C12]** 最大保有日数（Lines 243-245） ✅完了

---

## 調査結果

### [C1] 全体構造確認 ✅確定

**証拠**: contrarian_strategy.py Lines 1-30, 254-302

**判明したこと**:
- BaseStrategy継承確認: `class ContrarianStrategy(BaseStrategy)` (Line 29)
- **独自backtest()実装あり**: Lines 254-302
- generate_entry_signal()実装あり: Lines 143-198
- generate_exit_signal()実装あり: Lines 200-248

**結論**: 独自backtest()を実装しているため、Phase 1修正確認が必要。

---

### [C2] generate_exit_signal()の実装確認 ✅確定

**証拠**: contrarian_strategy.py Lines 200-248

**コード構造**:
```python
def generate_exit_signal(self, idx: int) -> int:
    # Line 203-208: ポジション状態管理
    # Line 211-219: エントリー価格取得
    # Line 221: current_price = self.data[self.price_column].iloc[idx]
    # Line 224-226: RSIイグジット
    # Line 228-233: トレーリングストップ
    # Line 235-237: 利益確定
    # Line 239-241: ストップロス
    # Line 243-245: 最大保有日数
```

**判明したこと**:
- 5つのイグジット条件を実装
- すべてのイグジット判定で`current_price`を使用
- entry_priceは`self.entry_prices`辞書から取得

---

### [C3] current_price取得方法確認 ❌ルックアヘッドバイアスあり

**証拠**: contrarian_strategy.py Line 221

```python
current_price = self.data[self.price_column].iloc[idx]
```

**判明したこと**:
- **当日終値（Adj Close）を使用**: `self.price_column = "Adj Close"` (Line 33)
- **ルックアヘッドバイアスあり**: idx日の終値を見てからidx日の終値でイグジットすることは不可能
- **Phase 1b未修正**: 翌日始値への変更が必要

**リアルトレードとの比較**:
- **リアルトレード**: idx日の終値を見た後、翌日（idx+1日目）の始値でイグジット
- **現在の実装**: idx日の終値を見た後、idx日の終値でイグジット（未来の情報使用）

**修正案**:
```python
# 禁止: 当日終値でイグジット
current_price = self.data[self.price_column].iloc[idx]

# 必須: 翌日始値でイグジット
current_price = self.data['Open'].iloc[idx + 1]
```

---

### [C4] entry_price取得方法確認 ✅修正済み

**証拠**: contrarian_strategy.py Lines 217, 291-295

**エントリー価格記録**:
```python
# Line 291-295: backtest()内でエントリー価格記録
next_day_open = self.data['Open'].iloc[idx + 1]
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
entry_price = next_day_open * (1 + slippage + transaction_cost)
self.entry_prices[idx] = entry_price
```

**エントリー価格取得**:
```python
# Line 217: generate_exit_signal()内で取得
entry_price = self.entry_prices.get(latest_entry_idx)
```

**判明したこと**:
- **Phase 1修正済み**: エントリー価格は翌日始値 + スリッページ・取引コスト
- entry_priceは正しい価格（翌日始値ベース）
- イグジット判定で使用するentry_priceは問題なし

---

### [C5] Phase 1修正確認 ✅修正済み

**証拠**: contrarian_strategy.py Lines 254-302

**独自backtest()実装の確認**:
```python
# Line 262: 最終日を除外（idx+1アクセス対応）
for idx in range(len(self.data) - 1):
    # ...
    # Line 286-295: エントリー価格を翌日始値に変更
    next_day_open = self.data['Open'].iloc[idx + 1]
    slippage = self.params.get("slippage", 0.001)
    transaction_cost = self.params.get("transaction_cost", 0.0)
    entry_price = next_day_open * (1 + slippage + transaction_cost)
    self.entry_prices[idx] = entry_price
```

**判明したこと**:
- **Phase 1修正済み✅**: エントリー価格は翌日始値
- **Phase 2修正済み✅**: スリッページ・取引コスト適用
- 修正コメントあり: "Phase 1修正", "Phase 2修正"

**結論**: エントリー編は完全修正済み。

---

### [C6] Phase 1b修正確認 ❌未修正（全5条件）

イグジット価格に関するPhase 1b修正が未実施です。

#### 問題箇所: Line 221（共通）

```python
current_price = self.data[self.price_column].iloc[idx]  # ← 当日終値（ルックアヘッドバイアス）
```

すべてのイグジット条件でこの`current_price`を使用しているため、全5条件が影響を受けます。

---

### [C7] Phase 0修正確認 ✅修正済み

**証拠**: contrarian_strategy.py Line 68

```python
# RSIを計算してデータに追加
# ルックアヘッドバイアス修正: shift(1)を追加して前日のRSIを使用
self.data['RSI'] = calculate_rsi(self.data[self.price_column], period=self.params["rsi_period"]).shift(1)
```

**判明したこと**:
- **Phase 0修正済み✅**: RSI.shift(1)適用
- 修正コメントあり: "ルックアヘッドバイアス修正"

**注意点**:
- generate_exit_signal()内でRSI使用（Line 224）: `current_rsi = self.data['RSI'].iloc[idx]`
- 既にshift(1)済みのため、idx日のRSIは前日（idx-1日）のRSI
- **RSIイグジット判定は問題なし✅**

---

### [C8] RSIイグジット ✅インジケーター修正済み、❌価格未修正

**証拠**: contrarian_strategy.py Lines 224-226

```python
# RSIによるイグジット
current_rsi = self.data['RSI'].iloc[idx]
if current_rsi >= self.params["rsi_exit_level"]:
    return -1
```

**判明したこと**:
- **RSI判定は問題なし✅**: RSI.shift(1)により前日RSI使用
- **価格は問題あり❌**: `current_price`（Line 221）は当日終値

---

### [C9] トレーリングストップ ❌未修正

**証拠**: contrarian_strategy.py Lines 228-233

```python
# トレーリングストップ
if latest_entry_idx not in self.high_prices:
    self.high_prices[latest_entry_idx] = entry_price
self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], current_price)
trailing_stop_price = self.high_prices[latest_entry_idx] * (1.0 - self.params["trailing_stop_pct"])
if current_price <= trailing_stop_price:
    return -1
```

**問題点**:

#### 1. 当日高値更新の問題

```python
# Line 231: 当日終値で高値更新（問題）
self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], current_price)
```

- `current_price`は当日終値（Line 221）
- **問題**: 当日高値（self.data['High'].iloc[idx]）ではなく当日終値を使用

#### 2. トレーリングストップ判定の問題

```python
# Line 233: 当日終値で判定（ルックアヘッドバイアス）
if current_price <= trailing_stop_price:
    return -1
```

- `current_price`は当日終値（ルックアヘッドバイアス）

**リアルトレードとの比較**:
- **リアルトレード**: 当日高値更新を確認後、翌日始値でイグジット
- **現在の実装**: 当日終値で高値更新判定、当日終値でイグジット（未来の情報使用）

**修正案**:
```python
# 修正前: 当日終値で判定
self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], current_price)
trailing_stop_price = self.high_prices[latest_entry_idx] * (1.0 - self.params["trailing_stop_pct"])
if current_price <= trailing_stop_price:
    return -1

# 修正後: 翌日始値で判定
# 注意: idx+1アクセスの安全性はbacktest()の`for idx in range(len(self.data) - 1)`で確保済み
next_day_open = self.data['Open'].iloc[idx + 1]
# Phase 1b修正: 当日高値で更新（当日終値ではなく）
self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], self.data['High'].iloc[idx])
trailing_stop_price = self.high_prices[latest_entry_idx] * (1.0 - self.params["trailing_stop_pct"])
if next_day_open <= trailing_stop_price:
    return -1
```

---

### [C10] 利益確定 ❌未修正

**証拠**: contrarian_strategy.py Lines 235-237

```python
# 利益確定
if current_price >= entry_price * (1.0 + self.params["take_profit"]):
    return -1
```

**問題点**:
- **当日終値で判定**: `current_price`は当日終値（Line 221）
- **ルックアヘッドバイアス**: idx日の終値を見てからidx日の終値でイグジット不可

**修正案**:
```python
# 修正前: 当日終値で判定
if current_price >= entry_price * (1.0 + self.params["take_profit"]):
    return -1

# 修正後: 翌日始値で判定
next_day_open = self.data['Open'].iloc[idx + 1]
if next_day_open >= entry_price * (1.0 + self.params["take_profit"]):
    return -1
```

---

### [C11] ストップロス ❌未修正

**証拠**: contrarian_strategy.py Lines 239-241

```python
# ストップロス
if current_price <= entry_price * (1.0 - self.params["stop_loss"]):
    return -1
```

**問題点**:
- **当日終値で判定**: `current_price`は当日終値（Line 221）
- **ルックアヘッドバイアス**: idx日の終値を見てからidx日の終値でイグジット不可

**修正案**:
```python
# 修正前: 当日終値で判定
if current_price <= entry_price * (1.0 - self.params["stop_loss"]):
    return -1

# 修正後: 翌日始値で判定
next_day_open = self.data['Open'].iloc[idx + 1]
if next_day_open <= entry_price * (1.0 - self.params["stop_loss"]):
    return -1
```

---

### [C12] 最大保有日数 ✅問題なし

**証拠**: contrarian_strategy.py Lines 243-245

```python
# 最大保有日数
days_held = idx - latest_entry_idx
if days_held >= self.params["max_hold_days"]:
    return -1
```

**判明したこと**:
- 価格判定を行わないため、ルックアヘッドバイアスの影響なし
- **問題なし✅**

---

## 修正が必要な箇所

### 修正箇所サマリー

| 箇所 | 行番号 | 現在の実装 | 問題点 | 影響するイグジット条件 |
|------|--------|----------|--------|----------------------|
| **1** | **Line 221** | `current_price = self.data[self.price_column].iloc[idx]` | 当日終値使用（ルックアヘッドバイアス） | **RSI、トレーリングストップ、利益確定、ストップロス** |
| **2** | **Line 231** | `max(self.high_prices[latest_entry_idx], current_price)` | 当日終値で高値更新（当日高値使用が正しい） | **トレーリングストップ** |

### イグジット条件別の問題状況

| イグジット条件 | Phase 0 | Phase 1b | 修正の複雑度 | 優先度 |
|--------------|---------|----------|------------|--------|
| **RSIイグジット** | ✅修正済み（RSI.shift(1)） | ❌未修正（価格） | 低（Line 221のみ） | 高 |
| **トレーリングストップ** | N/A | ❌未修正（価格+高値更新） | 中（Lines 221, 231） | 高 |
| **利益確定** | N/A | ❌未修正（価格） | 低（Line 221のみ） | 高 |
| **ストップロス** | N/A | ❌未修正（価格） | 低（Line 221のみ） | 高 |
| **最大保有日数** | N/A | ✅問題なし | - | - |

---

## セルフチェック

### a) 見落としチェック ✅完了

- **確認したファイル**: contrarian_strategy.py全体（Lines 1-338）
- **確認した変数名**: current_price, entry_price, price_column, Entry_Signal, Exit_Signal, high_prices
- **確認した関数名**: generate_exit_signal, backtest, initialize_strategy
- **データの流れ**: 確認完了
  1. initialize_strategy(): RSI.shift(1)適用（Phase 0修正済み）
  2. backtest(): entry_priceを翌日始値で記録（Phase 1修正済み）
  3. generate_exit_signal(): current_priceを当日終値で取得（Phase 1b未修正）

### b) 思い込みチェック ✅完了

#### 検証した前提

- **「Phase 1修正済み」**: 実際にコード確認（Lines 286-295）✅
  - 証拠: `next_day_open = self.data['Open'].iloc[idx + 1]`
  - 証拠: `entry_price = next_day_open * (1 + slippage + transaction_cost)`
  
- **「Phase 0修正済み」**: 実際にコード確認（Line 68）✅
  - 証拠: `self.data['RSI'] = calculate_rsi(...).shift(1)`
  
- **「Phase 1b未修正」**: 実際にコード確認（Line 221）✅
  - 証拠: `current_price = self.data[self.price_column].iloc[idx]`
  - 証拠: price_column = "Adj Close"（当日終値）
  
- **「トレーリングストップで当日高値未使用」**: 実際にコード確認（Line 231）✅
  - 証拠: `max(self.high_prices[latest_entry_idx], current_price)`
  - 証拠: current_priceは当日終値（Line 221）、当日高値ではない

### c) 矛盾チェック ✅完了

#### 整合性確認

- **Phase 1修正済み vs Phase 1b未修正**: 整合✅
  - エントリーは修正済み（翌日始値）
  - イグジットは未修正（当日終値）
  - 矛盾なし
  
- **Phase 0修正済み（RSI.shift(1)） vs RSIイグジット**: 整合✅
  - RSI判定は問題なし（前日RSI使用）
  - 価格判定が問題（当日終値使用）
  - 矛盾なし
  
- **独自backtest()実装 vs BaseStrategy継承**: 整合✅
  - オーバーライドによる独自実装
  - BaseStrategy.backtest()は使用しない
  - 矛盾なし
  
- **最終日除外（Line 262） vs idx+1アクセス**: 整合✅
  - `for idx in range(len(self.data) - 1)`: 最終日除外
  - generate_exit_signal()での`idx + 1`アクセス: 安全
  - 矛盾なし

---

## 調査結果のまとめ

### 判明したこと（証拠付き）

#### 1. Phase 0/1修正済み✅、Phase 1b未修正❌

- **Phase 0修正済み✅**: RSI.shift(1)適用（Line 68）
- **Phase 1修正済み✅**: エントリー価格は翌日始値 + スリッページ（Lines 286-295）
- **Phase 2修正済み✅**: スリッページ・取引コスト適用（Lines 291-294）
- **Phase 1b未修正❌**: イグジット価格は当日終値（Line 221）

#### 2. 修正が必要な箇所（主要2箇所）

1. **Line 221**: `current_price = self.data[self.price_column].iloc[idx]`
   - 当日終値使用（ルックアヘッドバイアス）
   - 影響: RSI、トレーリングストップ、利益確定、ストップロスの全4条件

2. **Line 231**: `max(self.high_prices[latest_entry_idx], current_price)`
   - 当日終値で高値更新（当日高値使用が正しい）
   - 影響: トレーリングストップのみ

#### 3. イグジット条件別の状態

| イグジット条件 | Phase 0 | Phase 1b | 優先度 |
|--------------|---------|----------|--------|
| RSIイグジット | ✅修正済み（RSI.shift(1)） | ❌未修正（価格） | 高 |
| トレーリングストップ | N/A | ❌未修正（価格+高値更新） | 高 |
| 利益確定 | N/A | ❌未修正（価格） | 高 |
| ストップロス | N/A | ❌未修正（価格） | 高 |
| 最大保有日数 | N/A | ✅問題なし | - |

### 不明な点

なし（すべて確認完了）

### 原因の推定

**確定事項**: Phase 1修正時にエントリー編のみ修正し、イグジット編の修正が未実施。

**根拠**:
1. generate_entry_signal()は修正コメントなし（backtest()内で翌日始値記録）
2. generate_exit_signal()は未修正（current_price = 当日終値のまま）
3. Phase 1修正コメント（Lines 262, 283-285）はエントリー編のみ言及
4. Phase 2修正コメント（Lines 284, 287-289）もエントリー編のみ

---

## 修正方針

### Phase 1b修正: イグジット価格を翌日始値に変更

**修正優先度**: 高（Phase 1修正との整合性確保のため）

**修正原則**: ルックアヘッドバイアス禁止の3原則
1. 前日データで判断: RSI.shift(1)（既に修正済み✅）
2. **翌日始値でエントリー/イグジット**: エントリーは修正済み✅、イグジットは未修正❌
3. 取引コスト考慮: エントリーは修正済み✅、イグジットはオプション

---

### 修正箇所1: current_price取得方法変更（Line 221）

**現在のコード**:
```python
current_price = self.data[self.price_column].iloc[idx]
```

**修正後のコード**:
```python
# Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
# 理由: idx日の終値を見てからidx日の終値でイグジットすることは不可能
# リアルトレードでは翌日（idx+1日目）の始値でイグジット
# 注意: idx+1アクセスの安全性はbacktest()の`for idx in range(len(self.data) - 1)`で確保済み
current_price = self.data['Open'].iloc[idx + 1]
```

**影響範囲**:
- RSIイグジット（Line 226）
- トレーリングストップ（Line 233）
- 利益確定（Line 236）
- ストップロス（Line 240）

**修正の難易度**: 低（1行のみ）

---

### 修正箇所2: トレーリングストップの高値更新変更（Line 231）

**現在のコード**:
```python
self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], current_price)
```

**修正後のコード**:
```python
# Phase 1b修正: 当日高値で更新（当日終値ではなく）
# 理由: トレーリングストップは当日高値を基準とするのが一般的
# 注意: idx日の高値は既に確定済みの情報なので使用可能
self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], self.data['High'].iloc[idx])
```

**影響範囲**:
- トレーリングストップのみ（Line 233）

**修正の難易度**: 低（1行のみ）

---

### 修正箇所3: backtest()の最終日除外確認（Line 262）

**現在のコード**:
```python
# Phase 1修正: 最終日を除外してidx+1アクセスを安全に（ルックアヘッドバイアス修正）
# 理由: エントリー価格を翌日始値（idx+1）に変更するため、最終日でのIndexError回避
for idx in range(len(self.data) - 1):
```

**確認事項**: 
- generate_exit_signal()内で`idx + 1`アクセスを追加するため、この最終日除外が必須
- **既に実装済みのため追加修正不要✅**

**修正の難易度**: なし（既に修正済み）

---

### Phase 2修正（オプション）: イグジット時のスリッページ・取引コスト

**現在の状況**:
- エントリー時: スリッページ・取引コスト適用済み（Lines 291-294）
- イグジット時: 未適用

**修正案（オプション）**:
```python
# Phase 1b修正: イグジット価格を翌日始値に変更
current_price = self.data['Open'].iloc[idx + 1]

# Phase 2修正（オプション）: イグジット時のスリッページ・取引コスト適用
# 売り注文は買い注文とは逆方向（不利な方向）にスリッページ
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
current_price = current_price * (1 - slippage - transaction_cost)
```

**推奨事項**: Phase 1b修正後、Phase 2修正は別タスクとして実施を推奨。

---

### 修正の優先順位

1. **最優先**: Line 221（current_price取得方法変更）
   - 影響範囲: 4つのイグジット条件すべて
   - 修正の難易度: 低

2. **優先**: Line 231（トレーリングストップの高値更新変更）
   - 影響範囲: トレーリングストップのみ
   - 修正の難易度: 低

3. **オプション**: Phase 2修正（イグジット時のスリッページ・取引コスト）
   - 影響範囲: すべてのイグジット条件
   - 修正の難易度: 低
   - 推奨: 別タスクとして実施

---

### 修正後の検証方法

#### 1. 単体テスト

```python
# テスト項目
- [1] current_priceが翌日始値であることを確認
- [2] トレーリングストップの高値更新が当日高値であることを確認
- [3] 各イグジット条件でルックアヘッドバイアスがないことを確認
- [4] エントリー価格とイグジット価格の整合性を確認
```

#### 2. 統合テスト

```python
# テスト項目
- [1] 実際の取引シミュレーション実行
- [2] エントリー価格とイグジット価格の時系列確認
- [3] PnL計算の正確性確認
- [4] バックテスト結果とリアルトレード結果の比較（可能であれば）
```

---

## 次のステップ

### 推奨アクション

1. **Phase 1b修正実施**（最優先）
   - Line 221: current_price取得方法変更
   - Line 231: トレーリングストップの高値更新変更

2. **修正後の検証**
   - 単体テスト実施
   - 統合テスト実施

3. **EXIT_INVESTIGATION_REPORT.md更新**
   - contrarian_strategy.py完了マーク追加
   - 詳細調査報告書リンク追加

4. **Phase 2修正検討**（オプション）
   - イグジット時のスリッページ・取引コスト適用
   - 別タスクとして実施を推奨

---

## 調査完了

- 調査日: 2025-12-23
- 調査状態: ✅完了
- 修正必要箇所: 明確化完了（主要2箇所）
- 次のアクション: Phase 1b修正実施
