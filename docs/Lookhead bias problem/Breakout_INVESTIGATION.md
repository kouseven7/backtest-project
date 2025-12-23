# BreakoutStrategy ルックアヘッドバイアス調査報告書

**作成日**: 2025-12-21  
**調査期間**: 2025-12-21  
**調査者**: GitHub Copilot  
**調査対象**: strategies/Breakout.py  
**調査ステータス**: ✅ **調査完了・修正完了（Phase 1, Phase 2実装済み）**  
**修正完了日**: 2025-12-23

### 修正完了サマリー

**Phase 1: ルックアヘッドバイアス修正** ✅ 完了
- エントリー価格を当日終値から**翌日始値**に変更
- generate_entry_signal()とbacktest()を修正
- ループ範囲を`range(len(self.data) - 1)`に変更（最終日除外）

**Phase 2: スリッページ追加** ✅ 完了
- デフォルトスリッページ0.1%を追加
- パラメータ`slippage`（デフォルト: 0.001）
- パラメータ`transaction_cost`（デフォルト: 0.0）
- エントリー価格計算: `next_day_open * (1 + slippage + transaction_cost)`

---

## 目次

1. [調査目的](#調査目的)
2. [調査対象](#調査対象)
3. [調査方法](#調査方法)
4. [調査結果](#調査結果)
5. [原因分析](#原因分析)
6. [改善提案](#改善提案)
7. [セルフチェック](#セルフチェック)
8. [次のステップ](#次のステップ)

---

## 調査目的

本調査の目的は以下の通り：
1. **BreakoutStrategy戦略にルックアヘッドバイアスが存在するか確認する**
2. 問題箇所を特定し、具体的な修正対象を明確にする
3. GCStrategyやVWAP_Breakout.pyと同様の修正が必要か判断する

### 背景

- **2025-12-20**: ルックアヘッドバイアス禁止ルールが[`.github/copilot-instructions.md`](../../.github/copilot-instructions.md)に追加
- **2025-12-21**: base_strategy.py Line 285が修正済み（当日終値→翌日始値）
- **2025-12-21**: GCStrategy Phase 1修正完了（Phase 1a/1b/1c）
- **現在**: 他のBaseStrategy派生クラスにも同様の問題がないか調査中

### 参考資料

- [gc_strategy_INVESTIGATION.md](gc_strategy_INVESTIGATION.md) - GCStrategyの調査結果
- [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) - VWAP_Breakout.pyの調査結果
- [copilot-instructions.md](../../.github/copilot-instructions.md) - ルックアヘッドバイアス禁止ルール

---

## 調査対象

### 対象ファイル

#### 主要対象
- **`strategies/Breakout.py`**: BreakoutStrategy戦略
  - [調査項目] backtest()メソッドの存在確認（独自実装の有無）
  - [調査項目] エントリー価格決定ロジック
  - [調査項目] イグジット価格決定ロジック
  - [調査項目] インジケーターのshift(1)適用状況（該当する場合）

#### 関連ファイル
- **`strategies/base_strategy.py`**: 基底クラス（継承関係確認・修正状況確認）

### 調査範囲

- **エントリー価格**: 当日終値を使用していないか
- **イグジット価格**: 当日高値・安値を使用していないか
- **インジケーター**: `.shift(1)`が適用されているか（該当する場合）
- **境界条件**: idx+1アクセスの安全性

---

## 調査方法

### 調査手順チェックリスト

#### Phase 0: コード修正履歴の確認（重要）

**目的**: 過去のバックテスト結果と現在のコードの不一致を防ぐ

- [x] **0-1. base_strategy.pyの現在の実装確認**
  - base_strategy.py Line 285付近のエントリー価格計算を確認
  - 現在の実装: `result['Open'].iloc[idx + 1]` （翌日始値使用）
  - 確認日時を記録: 2025-12-21 12:40

- [x] **0-2. Breakout.pyの最終更新日確認**
  - ファイルの最終修正日: 2025-04-02（コメントより）
  - base_strategy.py修正日（2025-12-21）との時系列関係: **Breakout.pyの方が古い**

- [x] **0-3. 既存のバックテスト結果の確認**
  - 本調査では既存結果なし
  - 今後のバックテスト実行日を記録予定

- [x] **0-4. タイムラインの記録**
  ```
  2025-04-02: Breakout.py最終更新
  2025-12-21: base_strategy.py修正 (Line 285: 翌日始値に変更)
  2025-12-21: 本調査開始
  ```

**重要な発見**: Breakout.pyはbase_strategy.py修正前の古いコード。独自backtest()実装のため、base_strategy.pyの修正の恩恵を受けていない可能性。

---

#### Phase 1: ファイル構造の確認

- [x] **1-1. ファイルの存在確認**
  - `strategies/Breakout.py`が存在するか → ✅ 確認済み
  - BaseStrategyを継承しているか → ✅ 確認済み（Line 23）

- [x] **1-2. backtest()メソッドの確認**
  - 独自のbacktest()を実装しているか → ✅ **独自実装あり（Line 165-215）**
  - BaseStrategy.backtest()を使用しているか → ❌ **使用していない**

- [x] **1-3. エントリーロジックの確認**
  - generate_entry_signal()の実装内容 → ✅ 確認済み（Lines 60-106）
  - エントリー価格の決定方法 → Line 97 `self.entry_prices[idx] = current_price`
  - Entry_Priceカラムの使用有無 → ❌ 使用していない

#### Phase 2: エントリー価格の調査

- [x] **2-1. コードレビュー**
  - エントリー価格の計算箇所を特定 → ✅ Line 97
  - 使用している価格カラム → `self.price_column`（デフォルト: "Close"）
  - current_price, entry_price変数の使用状況 → ✅ 確認済み

- [x] **2-2. パターンの確認**
  ```python
  # Line 81: 当日終値を取得
  current_price = self.data[self.price_column].iloc[idx]
  
  # Line 97: エントリー価格に記録
  self.entry_prices[idx] = current_price  # ❌ ルックアヘッドバイアス
  ```

- [x] **2-3. ループ範囲の確認**
  - `for idx in range(len(self.data))` → ❌ **最終日を含む**（Line 185）
  - 最終日のエントリー防止措置の有無 → ❌ なし

#### Phase 3: イグジット価格の調査

- [x] **3-1. generate_exit_signal()の確認**
  - ストップロス: 当日安値を使用していないか → ✅ 確認（当日高値からの反落を使用）
  - 利益確定: 当日高値を使用していないか → ✅ 確認（当日終値を使用）
  - トレーリングストップ: 当日高値更新を即座に反映していないか → ❌ **反映している**（Line 149）

- [x] **3-2. イグジット価格の決定方法**
  - Exit_Priceカラムの使用有無 → ❌ 使用していない
  - イグジット価格の計算箇所 → Line 145 `current_price = self.data[self.price_column].iloc[idx]`

#### Phase 4: インジケーターの確認

- [x] **4-1. インジケーター初期化**
  - initialize_strategy()メソッドの内容 → ❌ **メソッド自体が存在しない**
  - 使用しているインジケーターのリスト → **なし**（価格と出来高のみ）

- [x] **4-2. shift(1)の適用状況**
  - 全インジケーターに`.shift(1)`が適用されているか → **該当なし**（インジケーター使用なし）
  - 前日データを使用しているか → ❌ 当日データを使用

#### Phase 5: 実データ検証（✅ 完了）

- [x] **5-1. バックテスト実行**
  - 検証期間: 2024-12-01 〜 2025-02-05
  - 検証コマンド: `python test_breakout_entry_price.py`
  - **実行日時を必ず記録**: **2025-12-21 12:52:32**
  - **base_strategy.pyの状態**: ✅ 修正後 (Line 285使用、ただしBreakoutStrategyは独自backtest()のため非適用)

- [x] **5-2. エントリー価格の精度確認**
  - 13桁精度のエントリー価格が存在するか → ❌ **2桁精度**（小数部なし、例: 3320.0円）
  - 当日終値との一致確認 → ✅ **完全一致**（差分0.000000000000000円）
  - 翌日始値との乖離確認 → ✅ **9〜21円の乖離**（平均13.33円、約0.4%）

---

## 調査結果

### 結果1: ファイル構造

**確認日**: 2025-12-21

#### Breakout.pyの基本情報

```python
# strategies/Breakout.py Lines 1-23
"""
Module: Breakout
File: Breakout.py
Description: 
  ブレイクアウト（価格の節目突破）戦略を実装したクラスです。前日高値を
  出来高増加を伴って上抜けた場合にエントリーし、利益確定や高値からの
  反落でイグジットします。シンプルながら効果的なモメンタム戦略の一つです。

Author: kouseven7
Created: 2023-03-20
Modified: 2025-04-02
"""

from strategies.base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
```

**BaseStrategy継承**: ✅ 確認済み（Line 23）

**backtest()メソッド**: 
- ✅ **独自実装あり**（Lines 165-215）
- **証拠**: `def backtest(self, trading_start_date=None, trading_end_date=None):` が存在
- **重大な発見**: BaseStrategy.backtest()を**継承せず**独自実装
- **結論**: base_strategy.pyの修正（2025-12-21）の恩恵を受けていない

---

### 結果2: エントリー価格の調査

#### コードレビュー結果

**エントリー価格決定箇所**: generate_entry_signal() Lines 81, 97

**確認日**: 2025-12-21

```python
# strategies/Breakout.py Lines 81, 97
# Line 81: 当日終値を取得
current_price = self.data[self.price_column].iloc[idx]

# Line 97: エントリー価格として記録
self.entry_prices[idx] = current_price
```

**使用している価格カラム**: `self.price_column`（デフォルト: "Close"）

**price_columnの設定**: Line 25, 42
```python
# Line 25: __init__のデフォルト引数
def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Close", volume_column: str = "Volume"):

# Line 42: インスタンス変数に保存
self.price_column = price_column
```

**重要なNote**: Lines 36-39
```python
# Note:
#     price_columnは "Close" を使用してください。"Adj Close" (調整後終値) を使用すると、
#     配当調整により過去の価格が下方修正され、High (未調整) との比較が不正確になります。
#     これにより配当支払い銘柄でシグナルが生成されなくなります。
```

**ルックアヘッドバイアスの有無**:
- ✅ **ルックアヘッドバイアスあり** - 当日終値を使用
- [ ] **ルックアヘッドバイアスなし** - 翌日始値を使用
- [ ] **不明** - 追加調査が必要

**証拠1: 現在のコード（Breakout.py）**
```python
# generate_entry_signal() Lines 81, 97
current_price = self.data[self.price_column].iloc[idx]  # idx日の終値
self.entry_prices[idx] = current_price  # idx日の終値でエントリー価格を記録
```

**証拠2: backtest()の実装**
```python
# backtest() Line 201
if entry_signal == 1:
    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1  # idx日にエントリーシグナル
```

**問題の構造**:
1. idx日のgenerate_entry_signal()でエントリー判定
2. `current_price = self.data[self.price_column].iloc[idx]` - idx日の終値を取得
3. `self.entry_prices[idx] = current_price` - idx日の終値でエントリー価格を記録
4. リアルトレードでは、idx日の終値を見てからidx日の終値で買うことは不可能

**GCStrategyとの比較**:
- GCStrategy: BaseStrategy.backtest()使用 → base_strategy.py Line 285（翌日始値）を使用 → ✅ 修正済み
- BreakoutStrategy: 独自backtest()実装 → generate_entry_signal()でエントリー価格決定 → ❌ 当日終値使用

---

#### ループ範囲の確認

**ループの実装**: backtest() Line 185

```python
# Line 185
for idx in range(len(self.data)):
```

**境界条件の安全性**:
- [ ] **安全** - `range(len(self.data) - 1)` で最終日除外
- ✅ **危険** - `range(len(self.data))` で最終日含む
- [ ] **不明** - 追加調査が必要

**証拠**: Lines 185-215の読み取り結果

**現在の問題**:
- idx+1アクセスは現時点では行っていない
- **しかし、翌日始値を使用する修正を行う場合、idx+1アクセスが必要**
- その場合、最終日でIndexErrorが発生する可能性

**修正の必要性**: ✅ **必要** - 翌日始値使用時に`range(len(self.data) - 1)`に変更

---

### 結果3: イグジット価格の調査

#### generate_exit_signal()の実装

**イグジット価格決定箇所**: generate_exit_signal() Lines 145, 149

```python
# Lines 145, 148-150
current_price = self.data[self.price_column].iloc[idx]

# 現在の高値を更新（トレーリングストップのために）
if 'High' in self.data.columns and self.data['High'].iloc[idx] > high_price:
    high_price = self.data['High'].iloc[idx]
    self.high_prices[latest_entry_date] = high_price
```

**ルックアヘッドバイアスの可能性**:
- ✅ ストップロス: 当日終値使用の可能性 - `current_price < high_price * trailing_stop_level`（Line 160）
- ✅ 利益確定: 当日終値使用の可能性 - `current_price >= entry_price * (1 + take_profit)`（Line 153）
- ✅ トレーリングストップ: 当日高値更新の可能性 - `self.data['High'].iloc[idx]`（Line 149）
- [ ] なし: 翌日始値使用で問題なし

**重大な発見**:
- `current_price = self.data[self.price_column].iloc[idx]` - **idx日の終値を使用**
- `self.data['High'].iloc[idx]` - **idx日の高値を使用**
- トレーリングストップで`high_prices`を即座に更新 - **idx日の高値でidx日にイグジット判定**

**問題の構造**:
1. idx日のgenerate_exit_signal()でイグジット判定
2. `current_price = self.data[self.price_column].iloc[idx]` - idx日の終値を取得
3. `self.data['High'].iloc[idx]` - idx日の高値を取得
4. idx日の終値・高値でストップロス・利益確定・トレーリングストップを判定
5. リアルトレードでは、idx日の終値・高値を見てからidx日の終値で売ることは不可能

---

### 結果4: インジケーターの確認

#### 使用インジケーターのリスト

**initialize_strategy()の実装**: ❌ **メソッド自体が存在しない**

**使用インジケーター**: **なし**
- BreakoutStrategyは価格（Close, High）と出来高（Volume）のみを使用
- 移動平均線、RSI、MACD等のインジケーター計算なし
- トレンド判定なし

#### shift(1)の適用状況

**インジケーター計算**: **該当なし**

**shift(1)適用状況**:
- [ ] **全インジケーターに適用済み** - 前日データ使用
- [ ] **一部未適用** - ルックアヘッドバイアスの可能性
- [ ] **全て未適用** - 重大な問題
- ✅ **該当なし** - インジケーター使用なし

**結論**: shift(1)の適用は**該当なし**（インジケーター自体を使用していない）

---

### 結果5: 実データ検証（✅ 完了）

#### バックテスト実行結果

**実行日時**: 2025-12-21 12:52:32

**検証コマンド**:
```bash
python test_breakout_entry_price.py
```

**検証設定**:
- 銘柄: 8053.T
- 期間: 2024-12-01 〜 2025-02-05
- パラメータ: volume_threshold=1.2, take_profit=0.03, look_back=1, trailing_stop=0.02, breakout_buffer=0.01

**実行結果**:
```
エントリー件数: 4件
イグジット件数: 3件（ENTRY #4は最終日のため未イグジット）
データ取得: 42行
エラー: なし
```

---

#### エントリー価格の分析（詳細）

##### ENTRY #1: 2024-12-03

| 項目 | 値 | 備考 |
|------|-----|------|
| エントリー価格 | **3320.0円** | entry_prices[date] |
| 当日終値 (Close) | **3320.0円** | data['Close'].iloc[idx] |
| 当日始値 (Open) | 3284.0円 | data['Open'].iloc[idx] |
| 翌日始値 (Next Open) | 3329.0円 | data['Open'].iloc[idx+1] |
| **差分（エントリー - 終値）** | **0.000000000000000円** | **完全一致** |
| **差分（翌日始値 - エントリー）** | **9.00円** | **ルックアヘッドバイアスの影響** |
| 前日高値 | 3274.0円 | ブレイクアウト判定基準 |
| ブレイクアウト閾値 | 3306.74円 | 前日高値 × 1.01 |
| 前日出来高 | 2,809,500 | 出来高比較基準 |
| 当日出来高 | 4,176,900 | 出来高比率 1.49x |

**エントリー価格の精度**: 15桁表示で `3320.000000000000000` → **2桁精度**（小数部なし）

**結論**: ✅ **エントリー価格は当日終値と完全一致**（ルックアヘッドバイアス確認）

---

##### ENTRY #2: 2024-12-10

| 項目 | 値 | 備考 |
|------|-----|------|
| エントリー価格 | **3357.0円** | entry_prices[date] |
| 当日終値 (Close) | **3357.0円** | data['Close'].iloc[idx] |
| 当日始値 (Open) | 3398.0円 | data['Open'].iloc[idx] |
| 翌日始値 (Next Open) | 3367.0円 | data['Open'].iloc[idx+1] |
| **差分（エントリー - 終値）** | **0.000000000000000円** | **完全一致** |
| **差分（翌日始値 - エントリー）** | **10.00円** | **ルックアヘッドバイアスの影響** |
| 前日高値 | 3283.0円 | ブレイクアウト判定基準 |
| 当日出来高 | 4,294,300 | 出来高比率 1.46x |

**結論**: ✅ **エントリー価格は当日終値と完全一致**（ルックアヘッドバイアス確認）

---

##### ENTRY #3: 2024-12-26

| 項目 | 値 | 備考 |
|------|-----|------|
| エントリー価格 | **3379.0円** | entry_prices[date] |
| 当日終値 (Close) | **3379.0円** | data['Close'].iloc[idx] |
| 当日始値 (Open) | 3308.0円 | data['Open'].iloc[idx] |
| 翌日始値 (Next Open) | 3400.0円 | data['Open'].iloc[idx+1] |
| **差分（エントリー - 終値）** | **0.000000000000000円** | **完全一致** |
| **差分（翌日始値 - エントリー）** | **21.00円** | **ルックアヘッドバイアスの影響** |
| 前日高値 | 3320.0円 | ブレイクアウト判定基準 |
| 当日出来高 | 3,066,300 | 出来高比率 1.50x |

**結論**: ✅ **エントリー価格は当日終値と完全一致**（ルックアヘッドバイアス確認）

---

##### ENTRY #4: 2025-02-04（最終日）

| 項目 | 値 | 備考 |
|------|-----|------|
| エントリー価格 | **None** | **記録されず** |
| 当日終値 (Close) | 3489.0円 | data['Close'].iloc[idx] |
| 当日始値 (Open) | 3344.0円 | data['Open'].iloc[idx] |
| 翌日始値 (Next Open) | **N/A** | **最終日のため取得不可** |
| 前日高値 | 3321.0円 | ブレイクアウト判定基準 |
| 当日出来高 | 8,311,300 | 出来高比率 2.31x |

**問題**: 最終日のエントリーシグナルは生成されるが、`entry_prices[idx]`に記録されない

**原因**: [`Breakout.py Line 185`](../../strategies/Breakout.py#L185)で`range(len(self.data))`を使用し、最終日を含む。修正時にidx+1アクセス追加で境界条件エラーの可能性。

---

#### エントリー価格とバイアスの関係

##### 統計サマリ

| 指標 | 値 |
|------|-----|
| エントリー件数（記録あり） | 3件 |
| エントリー価格と当日終値の差分（平均） | **0.00円**（完全一致） |
| エントリー価格と翌日始値の差分（平均） | **13.33円**（9〜21円） |
| エントリー価格の精度 | **2桁**（小数部なし） |
| ルックアヘッドバイアスの有無 | ✅ **確認**（全エントリーで当日終値使用） |

##### ルックアヘッドバイアスの影響

**構造的問題**:
1. idx日のgenerate_entry_signal()でエントリー判定
2. `current_price = self.data[self.price_column].iloc[idx]` - **idx日の終値を取得**
3. `self.entry_prices[idx] = current_price` - **idx日の終値でエントリー価格を記録**
4. リアルトレードでは、idx日の終値を見てからidx日の終値で買うことは不可能

**バイアスの大きさ**:
- 翌日始値との乖離: 9〜21円（平均13.33円、約0.4%）
- この乖離分だけ、バックテスト結果が実際のトレードより有利

**証拠**: 実データで3件全てのエントリー価格が当日終値と完全一致（差分0.000000000000000円）

---

## 原因分析

### 根本原因

**問題1: 独自backtest()実装によるbase_strategy.py修正の非適用（最も重大）**

**直接原因**: [`strategies/Breakout.py`](../../strategies/Breakout.py) Lines 165-215

```python
# Lines 165-215: 独自backtest()実装
def backtest(self, trading_start_date=None, trading_end_date=None):
    """
    ブレイクアウト戦略のバックテストを実行する。
    """
    # シグナル列の初期化
    self.data['Entry_Signal'] = 0
    self.data['Exit_Signal'] = 0
    
    # ポジション管理変数
    in_position = False
    last_entry_idx = None

    # 各日にちについてシグナルを計算
    for idx in range(len(self.data)):  # ❌ 最終日を含む
        # ...
        if not in_position:
            entry_signal = self.generate_entry_signal(idx)  # ❌ エントリー価格は内部で決定
```

**問題の構造**:
1. BaseStrategy.backtest()を継承せず、独自実装
2. base_strategy.py Line 285の修正（翌日始値使用）が適用されない
3. エントリー価格・イグジット価格ともにgenerate_entry_signal()/generate_exit_signal()内で決定
4. 両メソッドで当日価格を使用

---

**問題2: エントリー価格のルックアヘッドバイアス（高優先度）**

**直接原因**: [`strategies/Breakout.py`](../../strategies/Breakout.py) Lines 81, 97

```python
# Line 81: 当日終値を取得
current_price = self.data[self.price_column].iloc[idx]  # idx日の終値

# Line 97: エントリー価格として記録
self.entry_prices[idx] = current_price  # ❌ idx日の終値でエントリー
```

**問題の構造**:
1. `idx`日目に`generate_entry_signal(idx)`でエントリー判断
2. `current_price = self.data[self.price_column].iloc[idx]` - idx日の終値を取得
3. `self.entry_prices[idx] = current_price` - idx日の終値でエントリー価格を記録
4. リアルトレードでは、idx日の終値を見てからidx日の終値で買うことは不可能

**正しい実装**:
```python
# 現状（誤り）
current_price = self.data[self.price_column].iloc[idx]  # idx日の終値

# 正しい実装（修正案）
# エントリー判定はidx日の価格で行うが、エントリー価格は翌日始値
next_day_open = self.data['Open'].iloc[idx + 1]  # idx+1日の始値
self.entry_prices[idx] = next_day_open
```

---

**問題3: イグジット価格のルックアヘッドバイアス（高優先度）**

**直接原因**: [`strategies/Breakout.py`](../../strategies/Breakout.py) Lines 145, 149

```python
# Line 145: 当日終値を取得
current_price = self.data[self.price_column].iloc[idx]  # idx日の終値

# Lines 148-150: 当日高値を取得・更新
if 'High' in self.data.columns and self.data['High'].iloc[idx] > high_price:
    high_price = self.data['High'].iloc[idx]  # ❌ idx日の高値
    self.high_prices[latest_entry_date] = high_price
```

**問題の構造**:
1. `idx`日目に`generate_exit_signal(idx)`でイグジット判断
2. `current_price = self.data[self.price_column].iloc[idx]` - idx日の終値を取得
3. `self.data['High'].iloc[idx]` - idx日の高値を取得
4. idx日の終値・高値でストップロス・利益確定・トレーリングストップを判定
5. リアルトレードでは、idx日の終値・高値を見てからidx日の終値で売ることは不可能

**正しい実装**:
```python
# 現状（誤り）
current_price = self.data[self.price_column].iloc[idx]  # idx日の終値
high_price = self.data['High'].iloc[idx]  # idx日の高値

# 正しい実装（修正案）
# イグジット判定はidx日の価格で行うが、イグジット価格は翌日始値
next_day_open = self.data['Open'].iloc[idx + 1]  # idx+1日の始値
# 高値更新もidx日のデータで行うが、イグジット実行は翌日始値
```

---

**問題4: ループ範囲の問題（中優先度）**

**直接原因**: [`strategies/Breakout.py`](../../strategies/Breakout.py) Line 185

```python
# Line 185: 最終日を含むループ
for idx in range(len(self.data)):  # ❌ idx+1アクセス追加で境界条件エラーの可能性
```

**問題の構造**:
1. 現在はidx+1アクセスなし
2. **修正時に翌日始値を使用する場合、idx+1アクセスが必要**
3. 最終日（idx=len(self.data)-1）でidx+1アクセスするとIndexError
4. `range(len(self.data) - 1)`に変更が必要

---

**問題5: price_columnがCloseを使用（設計上の理由）**

**直接原因**: [`strategies/Breakout.py`](../../strategies/Breakout.py) Lines 25, 36-39

```python
# Line 25: デフォルト引数
def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Close", ...):

# Lines 36-39: Closeを使用する理由
# Note:
#     price_columnは "Close" を使用してください。"Adj Close" (調整後終値) を使用すると、
#     配当調整により過去の価格が下方修正され、High (未調整) との比較が不正確になります。
#     これにより配当支払い銘柄でシグナルが生成されなくなります。
```

**設計上の理由**:
- Highカラム（未調整高値）との比較のため、Closeカラム（未調整終値）を使用
- Adj Close（調整後終値）を使用すると、配当調整でHighとの比較が不正確
- **これ自体は問題ではない**（設計上の理由がある）

**ただし、ルックアヘッドバイアス問題は別途修正が必要**:
- price_columnの選択は正しい
- **しかし、当日終値を使用している点は修正が必要**（翌日始値に変更）

---

### 比較: GCStrategyとの共通点

| 項目 | GCStrategy | BreakoutStrategy | 共通点 |
|------|------------|------------------|--------|
| backtest()実装 | BaseStrategy継承 | **独自実装** | 異なる |
| エントリー価格 | BaseStrategy（修正済み） | **当日終値（未修正）** | 異なる |
| イグジット価格 | Phase 1b修正完了 | **当日終値・高値（未修正）** | 同様の問題 |
| ループ範囲 | BaseStrategy（修正済み） | **最終日含む（未修正）** | 異なる |
| shift(1)適用 | Phase 1c修正完了 | 該当なし | - |
| インジケーター使用 | SMA（移動平均） | なし（価格・出来高のみ） | 異なる |

**重要な発見**:
1. BreakoutStrategyは**独自backtest()実装**のため、base_strategy.py修正の恩恵を受けていない
2. GCStrategyはPhase 1a/1b/1c修正完了、BreakoutStrategyは**全て未修正**
3. BreakoutStrategyの修正箇所はGCStrategyより**多い**（独自backtest()含む）

---

## 改善提案

### 修正の必要性

- ✅ **修正必要** - ルックアヘッドバイアスを確認
- [ ] **修正不要** - 問題なし
- [ ] **判断保留** - 追加調査が必要

### Phase 1: エントリー価格・イグジット価格の修正（必須）

#### 修正箇所1: ループ範囲の変更

**ファイル**: strategies/Breakout.py Line 185

**修正前:**
```python
for idx in range(len(self.data)):
```

**修正後:**
```python
# Phase 1修正: 最終日を除外してidx+1アクセスを安全に
for idx in range(len(self.data) - 1):
```

#### 修正箇所2: エントリー価格の変更

**ファイル**: strategies/Breakout.py Line 97

**修正前:**
```python
# Line 81
current_price = self.data[self.price_column].iloc[idx]

# Line 97
self.entry_prices[idx] = current_price  # 当日終値
```

**修正後:**
```python
# Line 81: エントリー判定は当日価格で行う（そのまま）
current_price = self.data[self.price_column].iloc[idx]

# Line 97: エントリー価格は翌日始値に変更（Phase 1修正）
# 理由: idx日の終値を見てからidx日の終値で買うことは不可能
# リアルトレードでは翌日（idx+1日目）の始値でエントリー
next_day_open = self.data['Open'].iloc[idx + 1]
self.entry_prices[idx] = next_day_open
```

**コメント追加**:
```python
# Phase 1a修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
# 理由: idx日の終値を見てからidx日の終値で買うことは不可能
# リアルトレードでは翌日（idx+1日目）の始値でエントリー
```

#### 修正箇所3: イグジット価格の変更

**ファイル**: strategies/Breakout.py Lines 145, 149

**修正前:**
```python
# Line 145
current_price = self.data[self.price_column].iloc[idx]

# Lines 148-150
if 'High' in self.data.columns and self.data['High'].iloc[idx] > high_price:
    high_price = self.data['High'].iloc[idx]
    self.high_prices[latest_entry_date] = high_price
```

**修正後:**
```python
# Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
# 理由: idx日の終値・高値を見てからidx日の終値で売ることは不可能
# リアルトレードでは翌日（idx+1日目）の始値でイグジット

# イグジット判定は当日価格で行うが、実際のイグジット価格は翌日始値
current_price_for_decision = self.data[self.price_column].iloc[idx]
next_day_open = self.data['Open'].iloc[idx + 1]

# 高値更新の判定は当日高値で行う（トレーリングストップの閾値計算用）
if 'High' in self.data.columns and self.data['High'].iloc[idx] > high_price:
    high_price = self.data['High'].iloc[idx]
    self.high_prices[latest_entry_date] = high_price

# イグジット条件の判定はcurrent_price_for_decisionを使用
# ただし、実際のイグジット価格はnext_day_openを記録
```

**注意**: イグジット価格の記録方法については、backtest()の実装によって異なる可能性がある。詳細な修正設計が必要。

### Phase 2: スリッページ・取引コスト（推奨）

```python
# Phase 2修正（VWAP_Breakout.py修正後に実装予定）
slippage = 0.001  # 0.1%
entry_price = self.data['Open'].iloc[idx + 1] * (1 + slippage)
exit_price = self.data['Open'].iloc[idx + 1] * (1 - slippage)
```

### 代替案: BaseStrategy.backtest()への移行（推奨・長期的）

**背景**:
- BreakoutStrategyは独自backtest()実装のため、base_strategy.pyの修正の恩恵を受けていない
- 他の戦略（GCStrategy等）はBaseStrategy.backtest()を使用し、統一的な修正が適用される

**提案**:
1. 独自backtest()を削除
2. BaseStrategy.backtest()を使用
3. エントリー価格・イグジット価格の決定をBaseStrategyに委譲

**メリット**:
- base_strategy.pyの修正が自動的に適用される
- コードの重複を削減
- メンテナンス性向上

**デメリット**:
- BaseStrategy.backtest()の仕様に合わせる必要がある
- 大規模な修正が必要

**判断**:
- **短期的**: 独自backtest()内のエントリー/イグジット価格を修正（Phase 1）
- **長期的**: BaseStrategy.backtest()への移行を検討

---

## セルフチェック

### a) 見落としチェック

**確認したファイル:**
- ✅ `strategies/Breakout.py` - 全体を詳細確認（Lines 1-314）
- ✅ `strategies/base_strategy.py` - 継承関係・修正状況確認（Line 285）

**確認していないファイル:**
- [ ] 既存のバックテスト結果（実データ検証は実施予定）

**カラム名・変数名の確認:**
- ✅ `price_column` の使用状況（デフォルト: "Close"）
- ✅ `volume_column` の使用状況（デフォルト: "Volume"）
- ✅ `entry_prices` の計算方法（Line 97）
- ✅ `high_prices` の使用状況（Line 149）
- ✅ `current_price` の使用状況（Lines 81, 145）

**データの流れ:**
- ✅ backtest() → generate_entry_signal() → entry_prices[idx] を追跡
- ✅ backtest() → generate_exit_signal() → current_price を追跡
- ✅ エントリー判断からエントリー価格決定までの流れを確認
- ✅ イグジット判断からイグジット価格決定までの流れを確認

### b) 思い込みチェック

**前提の検証:**
- ✅ 「BaseStrategyを継承しているはず」 → ✅ 継承確認（Line 23）
- ✅ 「BaseStrategy.backtest()を使用しているはず」 → ❌ 独自実装（Lines 165-215）
- ✅ 「エントリー価格は終値のはず」 → ✅ 確認（Line 97: 当日終値）
- ✅ 「インジケーターがあるはず」 → ❌ なし（価格と出来高のみ）
- ✅ 「shift(1)が必要なはず」 → ❌ 該当なし（インジケーター使用なし）

**実際に確認した事実:**
- ✅ Breakout.pyのコードを全体的に読んだ（Lines 1-314）
- ✅ エントリー価格の計算式を確認した（Lines 81, 97）
- ✅ イグジット価格の計算式を確認した（Lines 145, 149）
- ✅ ループ範囲を確認した（Line 185）
- ✅ インジケーターの有無を確認した（なし）
- ✅ backtest()の独自実装を確認した（Lines 165-215）

### c) 矛盾チェック

**調査結果の整合性:**
- ✅ backtest()独自実装 → エントリー価格決定がgenerate_entry_signal()内 → 整合
- ✅ price_column="Close" → Line 81でcurrent_price取得 → 整合
- ✅ entry_prices[idx] = current_price → 当日終値使用 → ルックアヘッドバイアス
- ✅ 独自backtest()実装 → base_strategy.py修正の非適用 → 整合
- ✅ ループ範囲（最終日含む） → idx+1アクセス追加で境界条件エラー → 整合

**ログ/エラーとの整合性:**
- [ ] バックテスト実行結果と調査結果の整合性 → [実施予定]

---

## 次のステップ

### 調査完了後のアクション

#### ルックアヘッドバイアスが確認された場合（✅ 該当）

1. **Phase 1修正の実施**（必須・高優先度）
   - [ ] ループ範囲の変更（`len(self.data) - 1`） - Line 185
   - [ ] エントリー価格を翌日始値に変更 - Line 97
   - [ ] イグジット価格を翌日始値に変更 - Line 145
   - [ ] コメント追加（修正理由の明記）

2. **検証**（必須）
   - [ ] 修正後のバックテスト実行
   - [ ] エントリー価格が翌日始値±0.1%に収まることを確認
   - [ ] イグジット価格が翌日始値±0.1%に収まることを確認
   - [ ] 取引件数の変化を確認（境界条件エラーがないか）

3. **ドキュメント更新**（推奨）
   - [ ] 本報告書の「調査結果」セクションを完成（実データ検証結果）
   - [ ] 「Phase 1実装結果」セクションを追加
   - [ ] INVESTIGATION_REPORT.mdに反映

#### 長期的な改善（推奨・低優先度）

1. **BaseStrategy.backtest()への移行検討**
   - [ ] 独自backtest()削除の影響範囲調査
   - [ ] BaseStrategy.backtest()の仕様確認
   - [ ] 移行計画の策定

2. **次の戦略の調査**
   - [ ] momentum_investing.py
   - [ ] contrarian_strategy.py
   - [ ] その他のBaseStrategy派生クラス

---

## 付録

### 遵守事項

#### copilot-instructions.md準拠

**ルックアヘッドバイアス禁止ルール（2025-12-20以降必須）:**

**基本ルール:**
```python
# 禁止: 当日終値でエントリー
entry_price = data['Adj Close'].iloc[idx]

# 必須: 翌日始値でエントリー + スリッページ
entry_price = data['Open'].iloc[idx + 1] * (1 + slippage)
```

**3原則:**
1. **前日データで判断**: インジケーターは`.shift(1)`必須（BreakoutStrategyは該当なし）
2. **翌日始値でエントリー**: `data['Open'].iloc[idx + 1]`
3. **取引コスト考慮**: スリッページ・を加味

**チェックリスト:**
- [ ] エントリー価格は`data['Open'].iloc[idx + 1]` → **未実装（要修正）**
- [x] インジケーターに`.shift(1)`適用 → **該当なし**（インジケーター使用なし）
- [ ] スリッページ考慮（推奨0.1%） → **未実装（Phase 2で対応予定）**

---

### 重要な注意事項

#### バックテスト結果の実行タイミングに関する注意

**背景**:
- 2025-12-21にbase_strategy.py Line 285が修正されました（当日終値→翌日始値）
- **BreakoutStrategyは独自backtest()実装のため、この修正の影響を受けていません**

**注意点**:
- 他の戦略（GCStrategy等）はbase_strategy.py修正後、エントリー価格が変化
- BreakoutStrategyは**未修正のまま**（当日終値を使用）
- **バックテスト結果を比較する際は、この点に注意**

**防止策**:
1. **Phase 0を必ず実施**: 各戦略のbacktest()実装を確認
2. **バックテスト実行日時を記録**: 「[YYYY-MM-DD HH:MM]実行」と明記
3. **修正前/後を記録**: 各戦略の修正状況を明記
4. **エントリー価格の精度を確認**: 13桁精度は修正前、2桁精度は修正後（該当する場合）

**確認事項**:
- [x] Phase 0でBreakout.pyのbacktest()実装確認済み（独自実装）
- [x] base_strategy.pyの修正前/後を明記（修正後、ただしBreakoutStrategyは非適用）
- [ ] バックテスト実行日時を本報告書に記録（実施予定）
- [ ] エントリー価格の精度（13桁 or 2桁）を確認（実施予定）

---

### 参考資料

- [gc_strategy_INVESTIGATION.md](gc_strategy_INVESTIGATION.md) - GCStrategyの完全な調査報告書
- [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) - VWAP_Breakout.pyの完全な調査報告書
- [copilot-instructions.md](../../.github/copilot-instructions.md) - ルックアヘッドバイアス禁止ルール
- yfinance documentation - `auto_adjust=False`の重要性

---

**報告書作成者**: GitHub Copilot  
**最終更新日**: 2025-12-21  
**バージョン**: 1.0  
**ステータス**: ✅ 調査完了（修正提案準備中）

---

## エグジット問題も存在するが、本調査ではエントリー問題とイグジット問題の両方を調査。独自backtest()実装のため、両方の修正が必要。

---

## 調査メモ

### 重要な発見

1. **独自backtest()実装**
   - BaseStrategy.backtest()を継承せず独自実装
   - base_strategy.pyの修正の恩恵を受けていない
   - GCStrategyとは大きく異なる構造

2. **price_column="Close"の理由**
   - Highカラム（未調整高値）との比較のため
   - Adj Closeを使用すると配当調整でHighとの比較が不正確
   - これ自体は設計上の理由がある

3. **インジケーターなし**
   - 価格（Close, High）と出来高（Volume）のみを使用
   - shift(1)適用は該当なし

4. **修正箇所が多い**
   - エントリー価格（Line 97）
   - イグジット価格（Line 145）
   - 高値更新（Line 149）
   - ループ範囲（Line 185）
   - 独自backtest()全体の見直しが必要
