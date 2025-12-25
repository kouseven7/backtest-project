# Breakout.py イグジット編 調査報告書

**作成日**: 2025-12-23  
**調査範囲**: strategies/Breakout.py (generate_exit_signal)  
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
- **[C3]** current_price取得方法確認（Line 186: 当日終値 vs 翌日始値） ✅完了
- **[C4]** entry_price取得方法確認（Lines 151-169: フォールバック処理） ✅完了
- **[C5]** Phase 1修正確認（generate_entry_signal()内: Lines 103-119） ✅完了
- **[C6]** Phase 1b修正確認（generate_exit_signal()内） ✅完了
- **[C7]** Phase 0修正確認（インジケーター - 該当なし） ✅完了

### イグジット条件の詳細確認

- **[C8]** 利益確定イグジット（Lines 193-195） ✅完了
- **[C9]** トレーリングストップイグジット（Lines 197-201） ✅完了
- **[C10]** high_price更新ロジック（Lines 188-190） ✅完了
- **[C11]** フォールバック処理の確認（Lines 151-185） ✅完了

---

## 調査結果

### [C1] 全体構造確認 ✅確定

**証拠**: Breakout.py Lines 1-357

**判明したこと**:
- BaseStrategy継承確認: `class BreakoutStrategy(BaseStrategy)` (Line 24)
- **独自backtest()実装あり**: Lines 203-259
- generate_entry_signal()実装あり: Lines 62-128
- generate_exit_signal()実装あり: Lines 130-203
- price_column: `"Close"` (Line 25、デフォルト値)

**重要な注意事項**（Line 34-37のNote）:
```python
Note:
    price_columnは "Close" を使用してください。"Adj Close" (調整後終値) を使用すると、
    配当調整により過去の価格が下方修正され、High (未調整) との比較が不正確になります。
    これにより配当支払い銘柄でシグナルが生成されなくなります。
```

**結論**: 
- 独自backtest()を実装しているため、Phase 1修正確認が必要
- price_column = "Close"を使用（contrarian_strategy.pyの"Adj Close"とは異なる）

---

### [C2] generate_exit_signal()の実装確認 ✅確定

**証拠**: Breakout.py Lines 130-203

**コード構造**:
```python
def generate_exit_signal(self, idx: int) -> int:
    # Line 137-142: ポジション状態管理
    # Line 145-148: 最新エントリーインデックス取得
    # Line 151-185: entry_priceとhigh_priceのフォールバック処理
    # Line 186: current_price = self.data[self.price_column].iloc[idx]
    # Line 188-190: high_price更新
    # Line 193-195: 利益確定
    # Line 197-201: トレーリングストップ
```

**判明したこと**:
- 2つのイグジット条件を実装（利益確定、トレーリングストップ）
- すべてのイグジット判定で`current_price`を使用
- entry_priceとhigh_priceはフォールバック処理あり（Lines 151-185）

---

### [C3] current_price取得方法確認 ❌ルックアヘッドバイアスあり

**証拠**: Breakout.py Line 186

```python
current_price = self.data[self.price_column].iloc[idx]
```

**判明したこと**:
- **当日終値（Close）を使用**: `self.price_column = "Close"` (Line 25、デフォルト値)
- **ルックアヘッドバイアスあり**: idx日の終値を見てからidx日の終値でイグジットすることは不可能
- **Phase 1b未修正**: 翌日始値への変更が必要

**contrarian_strategy.pyとの違い**:
- contrarian_strategy.py: `price_column = "Adj Close"`（調整後終値）
- Breakout.py: `price_column = "Close"`（通常終値）
- 理由: 配当調整による価格下方修正を避けるため（Line 34-37の注記）

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

### [C4] entry_price取得方法確認 ✅修正済み（フォールバックあり）

**証拠**: Breakout.py Lines 103-119, 151-169

**エントリー価格記録（generate_entry_signal()内）**:
```python
# Line 103-119: Phase 1a修正済み
next_day_open = self.data['Open'].iloc[idx + 1]

# Phase 2: スリッページ・取引コスト適用（買い注文は不利な方向）
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
entry_price = next_day_open * (1 + slippage + transaction_cost)
self.entry_prices[idx] = entry_price  # スリッページ適用後の価格を記録
```

**フォールバック処理（generate_exit_signal()内）**:
```python
# Line 151-169: Phase 1a修正済み
if latest_entry_date not in self.entry_prices:
    next_day_pos = latest_entry_pos + 1
    if next_day_pos < len(self.data):
        next_day_open = self.data['Open'].iloc[next_day_pos]
        # Phase 2: スリッページ・取引コスト適用
        slippage = self.params.get("slippage", 0.001)
        transaction_cost = self.params.get("transaction_cost", 0.0)
        entry_price = next_day_open * (1 + slippage + transaction_cost)
        self.entry_prices[latest_entry_date] = entry_price
    else:
        # 最終日の場合は当日始値を使用（境界条件の妥協案）
        current_open = self.data['Open'].iloc[latest_entry_pos]
        slippage = self.params.get("slippage", 0.001)
        transaction_cost = self.params.get("transaction_cost", 0.0)
        entry_price = current_open * (1 + slippage + transaction_cost)
        self.entry_prices[latest_entry_date] = entry_price
```

**判明したこと**:
- **Phase 1修正済み✅**: エントリー価格は翌日始値 + スリッページ・取引コスト
- **Phase 2修正済み✅**: スリッページ・取引コスト適用
- フォールバック処理も翌日始値を使用（Phase 1a修正済み、Line 151のコメント）
- entry_priceは正しい価格（翌日始値ベース）

---

### [C5] Phase 1修正確認 ✅修正済み

**証拠**: Breakout.py Lines 103-119, 222-224

**generate_entry_signal()内の修正**:
```python
# Line 103-119: Phase 1a修正済み
next_day_open = self.data['Open'].iloc[idx + 1]

# Phase 2: スリッページ・取引コスト適用
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
entry_price = next_day_open * (1 + slippage + transaction_cost)
self.entry_prices[idx] = entry_price
```

**backtest()内の最終日除外**:
```python
# Line 222-224: Phase 1修正済み
# Phase 1修正: 最終日を除外してidx+1アクセスを安全に（ルックアヘッドバイアス修正）
# 理由: エントリー価格を翌日始値（idx+1）に変更するため、最終日でのIndexError回避
for idx in range(len(self.data) - 1):
```

**判明したこと**:
- **Phase 1修正済み✅**: エントリー価格は翌日始値
- **Phase 2修正済み✅**: スリッページ・取引コスト適用
- 修正コメントあり: "Phase 1a修正", "Phase 2修正"（2025-12-23追加）
- 最終日除外によりidx+1アクセス安全

**結論**: エントリー編は完全修正済み（contrarian_strategy.pyと同様）。

---

### [C6] Phase 1b修正確認 ❌未修正

イグジット価格に関するPhase 1b修正が未実施です。

#### 問題箇所: Line 186（共通）

```python
current_price = self.data[self.price_column].iloc[idx]  # ← 当日終値（ルックアヘッドバイアス）
```

すべてのイグジット条件でこの`current_price`を使用しているため、両方のイグジット条件（利益確定、トレーリングストップ）が影響を受けます。

---

### [C7] Phase 0修正確認 ✅該当なし

**証拠**: Breakout.py全体確認

**判明したこと**:
- RSI、MACD等のテクニカルインジケーター未使用
- エントリー判定: 前日高値、前日出来高のみ使用
- イグジット判定: 利益確定、トレーリングストップのみ

**結論**: Phase 0修正（インジケーターのshift(1)適用）は該当なし。

---

### [C8] 利益確定イグジット ❌未修正

**証拠**: Breakout.py Lines 193-195

```python
# 利確条件
if current_price >= entry_price * (1 + self.params["take_profit"]):
    self.log_trade(f"Breakout イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
    return -1
```

**問題点**:
- **当日終値で判定**: `current_price`は当日終値（Line 186）
- **ルックアヘッドバイアス**: idx日の終値を見てからidx日の終値でイグジット不可

**パラメータ**:
- `take_profit = 0.03`（3%利益確定、Line 53）

**修正案**:
```python
# 修正前: 当日終値で判定
if current_price >= entry_price * (1 + self.params["take_profit"]):
    return -1

# 修正後: 翌日始値で判定
next_day_open = self.data['Open'].iloc[idx + 1]
if next_day_open >= entry_price * (1 + self.params["take_profit"]):
    return -1
```

---

### [C9] トレーリングストップイグジット ❌未修正

**証拠**: Breakout.py Lines 197-201

```python
# 損切条件（高値からの反落）
trailing_stop_level = 1 - self.params["trailing_stop"]
if current_price < high_price * trailing_stop_level:  # 高値からtrailing_stop%下落したら損切り
    self.log_trade(f"Breakout イグジットシグナル: 高値から反落 日付={self.data.index[idx]}, 価格={current_price}, 高値={high_price}")
    return -1
```

**問題点**:
- **当日終値で判定**: `current_price`は当日終値（Line 186）
- **ルックアヘッドバイアス**: idx日の終値を見てからidx日の終値でイグジット不可

**パラメータ**:
- `trailing_stop = 0.02`（高値から2%下落で損切り、Line 54）

**修正案**:
```python
# 修正前: 当日終値で判定
trailing_stop_level = 1 - self.params["trailing_stop"]
if current_price < high_price * trailing_stop_level:
    return -1

# 修正後: 翌日始値で判定
next_day_open = self.data['Open'].iloc[idx + 1]
trailing_stop_level = 1 - self.params["trailing_stop"]
if next_day_open < high_price * trailing_stop_level:
    return -1
```

---

### [C10] high_price更新ロジック ❌部分的に問題あり

**証拠**: Breakout.py Lines 188-190

```python
# 現在の高値を更新（トレーリングストップのために）
if 'High' in self.data.columns and self.data['High'].iloc[idx] > high_price:
    high_price = self.data['High'].iloc[idx]
    self.high_prices[latest_entry_date] = high_price
```

**判明したこと**:
- **当日高値を使用している**: `self.data['High'].iloc[idx]`（正しい）
- **問題点**: `current_price`（当日終値）との比較ではなく、`self.data['High'].iloc[idx]`（当日高値）を使用しているため、**この部分は問題なし✅**

**contrarian_strategy.pyとの違い**:
- **contrarian_strategy.py**: `max(self.high_prices[latest_entry_idx], current_price)` → 当日終値で更新（問題あり）
- **Breakout.py**: `self.data['High'].iloc[idx]` → 当日高値で更新（正しい）

**結論**: high_price更新ロジックは正しい。修正不要✅

---

### [C11] フォールバック処理の確認 ✅Phase 1修正済み

**証拠**: Breakout.py Lines 151-185

**entry_priceフォールバック**（Lines 151-169）:
```python
# Phase 1a修正: フォールバック処理も翌日始値を使用（ルックアヘッドバイアス修正）
# Phase 2修正: スリッページ・取引コスト対応（2025-12-23追加）
if latest_entry_date not in self.entry_prices:
    next_day_pos = latest_entry_pos + 1
    if next_day_pos < len(self.data):
        next_day_open = self.data['Open'].iloc[next_day_pos]
        # Phase 2: スリッページ・取引コスト適用
        slippage = self.params.get("slippage", 0.001)
        transaction_cost = self.params.get("transaction_cost", 0.0)
        entry_price = next_day_open * (1 + slippage + transaction_cost)
        self.entry_prices[latest_entry_date] = entry_price
    else:
        # 最終日の場合は当日始値を使用（境界条件の妥協案）
        current_open = self.data['Open'].iloc[latest_entry_pos]
        slippage = self.params.get("slippage", 0.001)
        transaction_cost = self.params.get("transaction_cost", 0.0)
        entry_price = current_open * (1 + slippage + transaction_cost)
        self.entry_prices[latest_entry_date] = entry_price
```

**high_priceフォールバック**（Lines 171-185）:
```python
if latest_entry_date not in self.high_prices:
    next_day_pos = latest_entry_pos + 1
    if next_day_pos < len(self.data):
        if 'High' in self.data.columns:
            self.high_prices[latest_entry_date] = self.data['High'].iloc[next_day_pos]
        else:
            self.high_prices[latest_entry_date] = self.data['Open'].iloc[next_day_pos]
    else:
        # 最終日の場合
        if 'High' in self.data.columns:
            self.high_prices[latest_entry_date] = self.data['High'].iloc[latest_entry_pos]
        else:
            self.high_prices[latest_entry_date] = self.data['Open'].iloc[latest_entry_pos]
```

**判明したこと**:
- **Phase 1修正済み✅**: entry_priceフォールバックは翌日始値を使用
- **Phase 2修正済み✅**: スリッページ・取引コスト適用
- high_priceフォールバックも翌日高値を使用（正しい）
- 修正コメントあり: "Phase 1a修正"（Line 151）

**結論**: フォールバック処理は適切に修正済み。

---

## 修正が必要な箇所

### 修正箇所サマリー

| 箇所 | 行番号 | 現在の実装 | 問題点 | 影響するイグジット条件 |
|------|--------|----------|--------|----------------------|
| **1** | **Line 186** | `current_price = self.data[self.price_column].iloc[idx]` | 当日終値使用（ルックアヘッドバイアス） | **利益確定、トレーリングストップ** |

### イグジット条件別の問題状況

| イグジット条件 | Phase 0 | Phase 1b | 修正の複雑度 | 優先度 |
|--------------|---------|----------|------------|--------|
| **利益確定** | N/A | ❌未修正（価格） | 低（Line 186のみ） | 高 |
| **トレーリングストップ** | N/A | ❌未修正（価格） | 低（Line 186のみ） | 高 |
| **high_price更新** | N/A | ✅問題なし（当日高値使用） | - | - |

### contrarian_strategy.pyとの違い

| 項目 | contrarian_strategy.py | Breakout.py |
|------|----------------------|-------------|
| **price_column** | "Adj Close" | "Close" |
| **high_price更新** | ❌未修正（current_price使用） | ✅問題なし（当日高値使用） |
| **修正箇所数** | 2箇所（Line 221, 231） | 1箇所（Line 186） |

---

## セルフチェック

### a) 見落としチェック ✅完了

- **確認したファイル**: Breakout.py全体（Lines 1-357）
- **確認した変数名**: current_price, entry_price, high_price, price_column, Entry_Signal, Exit_Signal
- **確認した関数名**: generate_exit_signal, generate_entry_signal, backtest
- **データの流れ**: 確認完了
  1. generate_entry_signal(): エントリー価格を翌日始値で記録（Phase 1修正済み）
  2. generate_exit_signal(): current_priceを当日終値で取得（Phase 1b未修正）
  3. フォールバック処理: 翌日始値を使用（Phase 1修正済み）

### b) 思い込みチェック ✅完了

#### 検証した前提

- **「Phase 1修正済み」**: 実際にコード確認（Lines 103-119）✅
  - 証拠: `next_day_open = self.data['Open'].iloc[idx + 1]`
  - 証拠: `entry_price = next_day_open * (1 + slippage + transaction_cost)`
  
- **「Phase 1b未修正」**: 実際にコード確認（Line 186）✅
  - 証拠: `current_price = self.data[self.price_column].iloc[idx]`
  - 証拠: price_column = "Close"（当日終値）
  
- **「high_price更新は正しい」**: 実際にコード確認（Line 189）✅
  - 証拠: `self.data['High'].iloc[idx]`（当日高値使用）
  - contrarian_strategy.pyとの違いを確認
  
- **「Phase 0修正該当なし」**: 実際にコード確認（全体）✅
  - 証拠: RSI、MACD等のインジケーター未使用
  - エントリー判定: 前日高値、前日出来高のみ

### c) 矛盾チェック ✅完了

#### 整合性確認

- **Phase 1修正済み vs Phase 1b未修正**: 整合✅
  - エントリーは修正済み（翌日始値）
  - イグジットは未修正（当日終値）
  - 矛盾なし
  
- **high_price更新は正しい vs トレーリングストップ未修正**: 整合✅
  - high_price更新: 当日高値使用（正しい）
  - トレーリングストップ判定: 当日終値使用（問題）
  - 矛盾なし（別々の処理）
  
- **独自backtest()実装 vs BaseStrategy継承**: 整合✅
  - オーバーライドによる独自実装
  - BaseStrategy.backtest()は使用しない
  - 矛盾なし
  
- **最終日除外（Line 222） vs idx+1アクセス**: 整合✅
  - `for idx in range(len(self.data) - 1)`: 最終日除外
  - generate_exit_signal()での`idx + 1`アクセス: 安全
  - 矛盾なし

---

## 調査結果のまとめ

### 判明したこと（証拠付き）

#### 1. Phase 1修正済み✅、Phase 1b未修正❌

- **Phase 1修正済み✅**: エントリー価格は翌日始値 + スリッページ・取引コスト（Lines 103-119）
- **Phase 2修正済み✅**: スリッページ・取引コスト適用（Lines 107-109）
- **Phase 1b未修正❌**: イグジット価格は当日終値（Line 186）
- **Phase 0該当なし✅**: インジケーター未使用

#### 2. 修正が必要な箇所（主要1箇所）

1. **Line 186**: `current_price = self.data[self.price_column].iloc[idx]`
   - 当日終値使用（ルックアヘッドバイアス）
   - 影響: 利益確定、トレーリングストップの両方

#### 3. イグジット条件別の状態

| イグジット条件 | Phase 0 | Phase 1b | 優先度 |
|--------------|---------|----------|--------|
| 利益確定 | N/A | ❌未修正（価格） | 高 |
| トレーリングストップ | N/A | ❌未修正（価格） | 高 |
| high_price更新 | N/A | ✅問題なし（当日高値使用） | - |

#### 4. contrarian_strategy.pyとの比較

| 項目 | contrarian_strategy.py | Breakout.py |
|------|----------------------|-------------|
| price_column | "Adj Close" | "Close" |
| high_price更新 | ❌未修正（current_price使用） | ✅問題なし（当日高値使用） |
| 修正箇所数 | 2箇所 | 1箇所 |
| 修正難易度 | 中（2箇所） | 低（1箇所） |

#### 5. フォールバック処理の状態

- **entry_priceフォールバック**: Phase 1修正済み✅（翌日始値使用）
- **high_priceフォールバック**: 正しい実装✅（翌日高値使用）

### 不明な点

なし（すべて確認完了）

### 原因の推定

**確定事項**: Phase 1修正時にエントリー編のみ修正し、イグジット編の修正が未実施。

**根拠**:
1. generate_entry_signal()は修正コメントあり（"Phase 1a修正", "Phase 2修正"）
2. generate_exit_signal()は未修正（current_price = 当日終値のまま）
3. Phase 1修正コメント（Lines 103, 222）はエントリー編のみ言及
4. フォールバック処理も修正済み（"Phase 1a修正"コメント、Line 151）

**contrarian_strategy.pyとの相違点**:
- Breakout.pyはhigh_price更新が正しい（当日高値使用）
- contrarian_strategy.pyはhigh_price更新も未修正（current_price使用）
- 理由: Breakout.pyの実装者が当日高値を意識的に選択した可能性

---

## 修正方針

### Phase 1b修正: イグジット価格を翌日始値に変更

**修正優先度**: 高（Phase 1修正との整合性確保のため）

**修正原則**: ルックアヘッドバイアス禁止の3原則
1. 前日データで判断: 該当なし（インジケーター未使用）
2. **翌日始値でエントリー/イグジット**: エントリーは修正済み✅、イグジットは未修正❌
3. 取引コスト考慮: エントリーは修正済み✅、イグジットはオプション

---

### 修正箇所1: current_price取得方法変更（Line 186）

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
- 利益確定イグジット（Line 193）
- トレーリングストップイグジット（Line 199）

**修正の難易度**: 低（1行のみ）

---

### 修正箇所2（確認のみ）: backtest()の最終日除外

**現在のコード**:
```python
# Line 222-224: Phase 1修正済み
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
- エントリー時: スリッページ・取引コスト適用済み（Lines 107-109）
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

1. **最優先**: Line 186（current_price取得方法変更）
   - 影響範囲: 2つのイグジット条件すべて
   - 修正の難易度: 低

2. **オプション**: Phase 2修正（イグジット時のスリッページ・取引コスト）
   - 影響範囲: すべてのイグジット条件
   - 修正の難易度: 低
   - 推奨: 別タスクとして実施

---

### 修正後の検証方法

#### 1. 単体テスト

```python
# テスト項目
- [1] current_priceが翌日始値であることを確認
- [2] 各イグジット条件でルックアヘッドバイアスがないことを確認
- [3] エントリー価格とイグジット価格の整合性を確認
- [4] high_price更新が正しく動作することを確認
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
   - Line 186: current_price取得方法変更

2. **修正後の検証**
   - 単体テスト実施
   - 統合テスト実施

3. **EXIT_INVESTIGATION_REPORT.md更新**
   - Breakout.py完了マーク追加
   - 詳細調査報告書リンク追加

4. **Phase 2修正検討**（オプション）
   - イグジット時のスリッページ・取引コスト適用
   - 別タスクとして実施を推奨

---

## 調査完了

- 調査日: 2025-12-23
- 調査状態: ✅完了
- 修正必要箇所: 明確化完了（主要1箇所、contrarian_strategy.pyより単純）
- 次のアクション: Phase 1b修正実施

---

## contrarian_strategy.pyとの比較サマリー

### 共通点
- Phase 1修正済み（エントリー価格は翌日始値）
- Phase 1b未修正（イグジット価格は当日終値）
- 独自backtest()実装
- 最終日除外実装済み

### 相違点

| 項目 | contrarian_strategy.py | Breakout.py |
|------|----------------------|-------------|
| **price_column** | "Adj Close"（調整後終値） | "Close"（通常終値） |
| **high_price更新** | ❌未修正（current_price使用） | ✅問題なし（当日高値使用） |
| **修正箇所数** | 2箇所（Line 221, 231） | 1箇所（Line 186） |
| **修正難易度** | 中（2箇所、1箇所は当日高値への変更） | 低（1箇所のみ） |
| **イグジット条件数** | 4条件（RSI、トレーリングストップ、利益確定、ストップロス） | 2条件（利益確定、トレーリングストップ） |

### 修正の優先度理由

Breakout.pyはcontrarian_strategy.pyより修正が単純（1箇所のみ）であり、high_price更新も既に正しいため、修正リスクが低い。
