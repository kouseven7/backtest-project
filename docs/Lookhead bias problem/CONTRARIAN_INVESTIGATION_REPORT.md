# ルックアヘッドバイアス問題 調査報告書 - ContrararianStrategy

**作成日**: 2025-12-22  
**最終更新**: 2025-12-22  
**調査期間**: 2025-12-22  
**調査者**: GitHub Copilot  
**調査範囲**: strategies/contrarian_strategy.py  

---

## 目次

1. [調査目的](#調査目的)
2. [確認項目チェックリスト](#確認項目チェックリスト)
3. [調査結果](#調査結果)
4. [原因分析](#原因分析)
5. [影響範囲](#影響範囲)
6. [改善提案](#改善提案)
7. [セルフチェック](#セルフチェック)

---

## 調査目的

strategies/contrarian_strategy.pyにおいて、ルックアヘッドバイアスが混入しているかを調査する。
特に、**独自のbacktest()メソッドを実装している**ため、BaseStrategy.backtest()とは異なる問題が存在する可能性がある。

### 調査の背景

- INVESTIGATION_REPORT.mdでBaseStrategy.backtest()のルックアヘッドバイアスを確認
- base_strategy.py Line 285で修正完了（翌日始値使用）
- しかし、contrarian_strategy.pyは**独自backtest()メソッド**（Line 243-279）を実装
- BaseStrategyの修正が適用されない可能性

---

## 確認項目チェックリスト

### Phase 1: 構造確認（優先度: 最高）

#### 1.1 backtest()メソッドの確認 ✅
- [ ] backtest()メソッドが独自実装されているか
- [ ] BaseStrategy.backtest()を呼び出しているか
- [ ] エントリー価格決定ロジックの実装場所

**優先度理由**: 独自実装の場合、BaseStrategyの修正が無効化される

#### 1.2 エントリー価格の決定方法 ✅
- [ ] エントリー価格をどこで記録しているか（Line番号特定）
- [ ] entry_prices辞書の使用状況
- [ ] 当日終値を使用しているか、翌日始値を使用しているか

**優先度理由**: ルックアヘッドバイアスの直接的な証拠

#### 1.3 インジケーターのshift(1)適用 ✅
- [ ] RSIにshift(1)が適用されているか
- [ ] トレンド判定のデータ範囲（iloc[:idx + 1]）
- [ ] 他のインジケーターのshift(1)状況

**優先度理由**: 判断ロジックのルックアヘッドバイアス確認

### Phase 2: 境界条件チェック（優先度: 高）

#### 2.1 ループ範囲の確認 ✅
- [ ] ループが`range(len(self.data))`か`range(len(self.data) - 1)`か
- [ ] idx+1アクセスの有無
- [ ] 最終日のエントリー可能性

**優先度理由**: IndexError回避と整合性

#### 2.2 イグジット価格の確認 ✅
- [ ] generate_exit_signal()での価格使用状況
- [ ] current_priceの取得タイミング（当日終値か翌日始値か）
- [ ] トレーリングストップ用の高値取得（当日高値含むか）

**優先度理由**: イグジットのルックアヘッドバイアス

### Phase 3: 実データ検証（優先度: 中）

#### 3.1 実際の取引データ確認 ⏳
- [ ] 実際のバックテスト実行
- [ ] エントリー価格と市場データの比較
- [ ] 13桁精度の有無

**優先度理由**: 理論と実際の一致確認（工数が大きい）

---

## 調査結果

### 結果1: backtest()メソッドの独自実装 ✅確定

#### 証拠: contrarian_strategy.py Line 243-279

**ファイル確認**:
```python
# Line 243-279
def backtest(self, trading_start_date=None, trading_end_date=None):
    """
    バックテストを実行する。
    
    Parameters:
        trading_start_date (datetime, optional): 取引開始日（この日以降にシグナル生成開始）
        trading_end_date (datetime, optional): 取引終了日（この日以前までシグナル生成）
    """
    self.data['Entry_Signal'] = 0
    self.data['Exit_Signal'] = 0

    for idx in range(len(self.data)):  # ← 最終日含む
        # 取引期間フィルタリング（BaseStrategy.backtest()と同じロジック）
        if trading_start_date is not None or trading_end_date is not None:
            current_date = self.data.index[idx]
            in_trading_period = True
            
            if trading_start_date is not None and current_date < trading_start_date:
                in_trading_period = False
            if trading_end_date is not None and current_date > trading_end_date:
                in_trading_period = False
            
            if not in_trading_period:
                # 取引期間外はシグナル生成をスキップ
                continue
        # エントリーシグナル
        if not self.data['Entry_Signal'].iloc[max(0, idx - 1):idx + 1].any():
            entry_signal = self.generate_entry_signal(idx)
            if entry_signal == 1:
                self.data.at[self.data.index[idx], 'Entry_Signal'] = 1

        # イグジットシグナル
        exit_signal = self.generate_exit_signal(idx)
        if exit_signal == -1:
            self.data.at[self.data.index[idx], 'Exit_Signal'] = -1

    return self.data
```

**確定事項**:
1. ✅ **独自backtest()メソッドを実装**している
2. ✅ BaseStrategy.backtest()を呼び出していない
3. ✅ base_strategy.py Line 285の修正（翌日始値使用）が**適用されない**
4. ✅ ループ範囲: `range(len(self.data))` - 最終日を含む
5. ✅ エントリー価格決定: backtest()メソッド内では行わない（generate_entry_signal()で記録）

**重大な発見**:
- BaseStrategyの修正（Line 285: 翌日始値使用）が**無効化**される
- 独自のエントリー価格決定ロジックが必要

---

### 結果2: エントリー価格の決定方法 ✅確定（ルックアヘッドバイアス存在）

#### 証拠: contrarian_strategy.py Lines 174-182

**ファイル確認 - generate_entry_signal()メソッド**:
```python
# Lines 140-182（抜粋）
def generate_entry_signal(self, idx: int) -> int:
    """
    エントリーシグナルを生成する。
    """
    if idx < 5:  # 過去データが不足している場合
        return 0

    rsi = self.data['RSI'].iloc[idx]
    current_price = self.data[self.price_column].iloc[idx]  # ← 当日終値取得
    previous_close = self.data[self.price_column].iloc[idx - 1]
    
    # ... （中略）...
    
    # エントリー条件（B: RSI条件を両方に適用）
    # 条件1: RSI過売り + ギャップダウン
    if rsi <= self.params["rsi_oversold"] and gap_down:
        self.entry_prices[idx] = current_price  # ← Line 178: 当日終値を記録
        return 1
    # 条件2: RSI過売り + ピンバー（RSI条件追加）
    if rsi <= self.params["rsi_oversold"] and pin_bar:
        self.entry_prices[idx] = current_price  # ← Line 182: 当日終値を記録
        return 1

    return 0
```

**確定事項**:
1. ✅ Line 147: `current_price = self.data[self.price_column].iloc[idx]` - **当日終値を取得**
2. ✅ Line 178, 182: `self.entry_prices[idx] = current_price` - **当日終値を記録**
3. ✅ `self.price_column`のデフォルト値: `"Adj Close"` (Line 31で確認)
4. ✅ エントリー価格 = 当日終値（Adj Close）

**ルックアヘッドバイアス確定**:
- idx日の終値を見てから、idx日の終値でエントリー価格を記録
- リアルトレードでは不可能（終値は市場終了後にしか確定しない）

**証拠の根拠**:
- コード内で`self.data[self.price_column].iloc[idx]`を使用（当日データ）
- `self.entry_prices[idx] = current_price`で当日終値を直接記録
- shift(1)が適用されていない（前日データではない）

---

### 結果3: インジケーターのshift(1)適用 ✅確定（部分的に正しい）

#### 証拠: contrarian_strategy.py Lines 65-66

**ファイル確認 - initialize_strategy()メソッド**:
```python
# Lines 60-66
def initialize_strategy(self):
    """
    戦略の初期化処理
    """
    super().initialize_strategy()
    # RSIを計算してデータに追加
    # ルックアヘッドバイアス修正: shift(1)を追加して前日のRSIを使用
    self.data['RSI'] = calculate_rsi(self.data[self.price_column], period=self.params["rsi_period"]).shift(1)
    
    # Openカラムの確認（ピンバー判定に必要）
    if 'Open' not in self.data.columns:
        raise ValueError("ピンバー判定にはOpenカラムが必要です")
```

**確定事項**:
1. ✅ RSI: **shift(1)が適用されている**（正しい実装）
2. ✅ コメントで「ルックアヘッドバイアス修正」と明記
3. ✅ 前日のRSIを使用して判断

#### 証拠: contrarian_strategy.py Lines 162-169（トレンド判定）

**ファイル確認 - generate_entry_signal()内のトレンド判定**:
```python
# Lines 162-169
# トレンド判定（統一トレンド判定インターフェースを使用）
if self.params["trend_filter_enabled"]:
    # 統一トレンド判定インターフェースを使用
    trend = detect_unified_trend(
        self.data.iloc[:idx + 1],  # ← idx日までのデータを使用
        price_column=self.price_column,
        strategy="contrarian_strategy",
        method="combined"  # 複合メソッドを使用
    )
    # 許可されたトレンド内にあるか確認
    if trend not in self.params["allowed_trends"]:
        return 0
```

**確定事項**:
1. ✅ トレンド判定: `self.data.iloc[:idx + 1]` - **idx日までのデータを使用**
2. ✅ idx日のデータを含むため、厳密にはルックアヘッドバイアスの可能性
3. ⚠️ ただし、トレンド判定は長期的な傾向を見るため、影響は小さい可能性

**矛盾の発見**:
- RSIは正しくshift(1)適用済み（前日RSIを使用）
- しかし、エントリー価格は当日終値を使用（ルックアヘッドバイアス）
- **判断は正しいが、エントリー価格が誤り**

---

### 結果4: 境界条件とループ範囲 ✅確定（問題あり）

#### 証拠: contrarian_strategy.py Line 253

**ファイル確認 - backtest()メソッド**:
```python
# Line 253
for idx in range(len(self.data)):  # ← 最終日を含む
```

**確定事項**:
1. ✅ ループ範囲: `range(len(self.data))` - **最終日を含む**
2. ✅ idx+1アクセスの有無: generate_entry_signal()では使用していない
3. ✅ 最終日のエントリー: **可能**（問題なし）

**Phase 1修正時の考慮事項**:
- エントリー価格を翌日始値（idx+1）に変更する場合、ループ範囲を`range(len(self.data) - 1)`に変更する必要がある
- 現状では最終日のエントリーが可能だが、idx+1アクセスに変更すると IndexError が発生

---

### 結果5: イグジット価格の確認 ✅確定（ルックアヘッドバイアス存在）

#### 証拠: contrarian_strategy.py Lines 209-240

**ファイル確認 - generate_exit_signal()メソッド**:
```python
# Lines 187-240（抜粋）
def generate_exit_signal(self, idx: int) -> int:
    """
    イグジットシグナルを生成する。
    """
    if idx < 1:
        return 0

    # ポジション状態管理を追加
    # 現在までのエントリー・エグジット数を計算
    current_entries = (self.data['Entry_Signal'].iloc[:idx+1] == 1).sum()
    current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())
    
    # アクティブなポジションがない場合はエグジット不可
    if current_entries <= current_exits:
        return 0

    # 最新のエントリー価格を取得
    entry_indices = self.data[self.data['Entry_Signal'] == 1].index
    if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
        return 0

    latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
    entry_price = self.entry_prices.get(latest_entry_idx)
    if entry_price is None:
        return 0

    current_price = self.data[self.price_column].iloc[idx]  # ← Line 213: 当日終値取得

    # RSIによるイグジット
    current_rsi = self.data['RSI'].iloc[idx]
    if current_rsi >= self.params["rsi_exit_level"]:
        return -1

    # トレーリングストップ
    if latest_entry_idx not in self.high_prices:
        self.high_prices[latest_entry_idx] = entry_price
    self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], current_price)  # ← Line 224: 当日価格で更新
    trailing_stop_price = self.high_prices[latest_entry_idx] * (1.0 - self.params["trailing_stop_pct"])
    if current_price <= trailing_stop_price:
        return -1

    # 利益確定
    if current_price >= entry_price * (1.0 + self.params["take_profit"]):  # ← Line 229: 当日終値で判定
        return -1

    # ストップロス
    if current_price <= entry_price * (1.0 - self.params["stop_loss"]):  # ← Line 233: 当日終値で判定
        return -1

    # 最大保有日数
    days_held = idx - latest_entry_idx
    if days_held >= self.params["max_hold_days"]:
        return -1

    return 0
```

**確定事項**:
1. ✅ Line 213: `current_price = self.data[self.price_column].iloc[idx]` - **当日終値取得**
2. ✅ Line 224: `self.high_prices[latest_entry_idx] = max(...)` - **当日価格で更新**
3. ✅ Line 229: 利益確定判定 - **当日終値で判定**
4. ✅ Line 233: ストップロス判定 - **当日終値で判定**
5. ✅ RSI: shift(1)適用済みのため、前日RSI使用（正しい）

**ルックアヘッドバイアス確定（イグジット）**:
- 当日終値を見てから、当日終値でイグジット判断
- トレーリングストップの高値更新に当日価格を使用
- リアルトレードでは不可能

**HIGH値の確認**:
- コード内で`self.data['High']`の使用を確認したが、イグジット判定では未使用
- トレーリングストップは`current_price`（終値）ベースで計算
- 当日高値（High）を使用していないため、その点では問題なし

---

## 原因分析

### 根本原因

**直接原因1: エントリー価格（generate_entry_signal()）**:
```python
# contrarian_strategy.py Line 147, 178, 182
current_price = self.data[self.price_column].iloc[idx]  # 当日終値
self.entry_prices[idx] = current_price  # 当日終値を記録
```

**直接原因2: イグジット判定（generate_exit_signal()）**:
```python
# contrarian_strategy.py Line 213, 224, 229, 233
current_price = self.data[self.price_column].iloc[idx]  # 当日終値
# 当日終値で利益確定・ストップロス・トレーリングストップ判定
```

**構造的問題**:
1. 独自backtest()メソッド実装により、BaseStrategyの修正（Line 285）が無効
2. エントリー価格をgenerate_entry_signal()内で記録（当日終値）
3. イグジット判定をgenerate_exit_signal()内で実行（当日終値）
4. ループ範囲が最終日を含む（idx+1アクセスに変更時にIndexError）

### 正しい実装（Phase 1修正案）

#### エントリー価格の修正

```python
# 現状（誤り）
def generate_entry_signal(self, idx: int) -> int:
    current_price = self.data[self.price_column].iloc[idx]  # idx日の終値
    self.entry_prices[idx] = current_price
    return 1

# 正しい実装
def generate_entry_signal(self, idx: int) -> int:
    # エントリー価格記録を削除（backtest()で翌日始値を記録）
    # current_price は判断のみに使用
    return 1  # シグナルのみ返す

# backtest()メソッドに追加
def backtest(self, trading_start_date=None, trading_end_date=None):
    for idx in range(len(self.data) - 1):  # 最終日を除外
        # ...
        if entry_signal == 1:
            self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
            # Phase 1修正: 翌日始値でエントリー価格を記録
            next_day_open = self.data['Open'].iloc[idx + 1]
            self.entry_prices[idx] = next_day_open
```

**理由**:
- idx日の市場終了後に判断
- 翌日（idx+1日）の市場開始時（始値）でエントリー
- これがリアルトレードの実態

#### イグジット価格の修正（Phase 2対応）

```python
# 現状（誤り）
current_price = self.data[self.price_column].iloc[idx]  # idx日の終値

# Phase 2修正案（EXIT_INVESTIGATION_REPORT.mdで詳細検討）
# 当日高値・安値を使用する場合の検討が必要
```

---

## 影響範囲

### 影響を受けるファイル

#### 確定（本調査で確認済み）

1. **`strategies/contrarian_strategy.py`** - 逆張り戦略
   - Lines 178, 182: エントリー価格記録（当日終値）
   - Line 213: イグジット判定用の価格取得（当日終値）
   - Lines 224, 229, 233: イグジット条件判定（当日終値）
   - 影響度: **最大**

### バックテスト結果への影響

**ContrararianStrategyのバックテスト結果**:
- 全て楽観的な結果となっている可能性が高い
- 特に、ギャップダウン検出時のエントリーで影響が大きい可能性
- トレーリングストップが当日価格で更新されるため、過剰な利益確定の可能性

**影響の深刻度**:
- リターン率: **過大評価**（有利なエントリー価格）
- シャープレシオ: **過大評価**
- 最大ドローダウン: **過小評価**
- 勝率: **過大評価**

---

## 改善提案

### Phase 1: エントリー価格修正（必須）

#### 修正箇所1: ループ範囲（backtest()メソッド）

**ファイル**: contrarian_strategy.py Line 253

**修正前**:
```python
for idx in range(len(self.data)):
```

**修正後**:
```python
# Phase 1修正: 最終日を除外してidx+1アクセスを安全に（ルックアヘッドバイアス修正）
# 理由: エントリー価格を翌日始値（idx+1）に変更するため、最終日でのIndexError回避
for idx in range(len(self.data) - 1):
```

#### 修正箇所2: エントリー価格記録（generate_entry_signal()メソッド）

**ファイル**: contrarian_strategy.py Lines 178, 182

**修正前**:
```python
# 条件1: RSI過売り + ギャップダウン
if rsi <= self.params["rsi_oversold"] and gap_down:
    self.entry_prices[idx] = current_price
    return 1
# 条件2: RSI過売り + ピンバー（RSI条件追加）
if rsi <= self.params["rsi_oversold"] and pin_bar:
    self.entry_prices[idx] = current_price
    return 1
```

**修正後**:
```python
# 条件1: RSI過売り + ギャップダウン
if rsi <= self.params["rsi_oversold"] and gap_down:
    # Phase 1修正: エントリー価格記録を削除（backtest()で翌日始値を記録するため）
    # self.entry_prices[idx] = current_price  # ← 削除
    return 1
# 条件2: RSI過売り + ピンバー（RSI条件追加）
if rsi <= self.params["rsi_oversold"] and pin_bar:
    # Phase 1修正: エントリー価格記録を削除（backtest()で翌日始値を記録するため）
    # self.entry_prices[idx] = current_price  # ← 削除
    return 1
```

#### 修正箇所3: エントリー価格記録（backtest()メソッド）

**ファイル**: contrarian_strategy.py Lines 268-271

**修正前**:
```python
# エントリーシグナル
if not self.data['Entry_Signal'].iloc[max(0, idx - 1):idx + 1].any():
    entry_signal = self.generate_entry_signal(idx)
    if entry_signal == 1:
        self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
```

**修正後**:
```python
# エントリーシグナル
if not self.data['Entry_Signal'].iloc[max(0, idx - 1):idx + 1].any():
    entry_signal = self.generate_entry_signal(idx)
    if entry_signal == 1:
        self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
        # Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
        # 理由: idx日の終値を見てからidx日の終値で買うことは不可能
        # リアルトレードでは翌日（idx+1日目）の始値でエントリー
        next_day_open = self.data['Open'].iloc[idx + 1]
        self.entry_prices[idx] = next_day_open
```

### Phase 2: スリッページ・取引コスト（推奨）

```python
# Phase 2修正案
slippage = 0.001  # 0.1%
next_day_open = self.data['Open'].iloc[idx + 1]
entry_price = next_day_open * (1 + slippage)
self.entry_prices[idx] = entry_price
```

### Phase 3: イグジット問題（EXIT_INVESTIGATION_REPORT.mdで対応）

- Line 213: イグジット判断用の価格
- Line 224: トレーリングストップ用の高値更新
- Lines 229, 233: 利益確定・ストップロス判定

---

## セルフチェック

### a) 見落としチェック ✅

**確認したファイル**:
- ✅ `strategies/contrarian_strategy.py` Lines 1-318 - 全行確認済み
- ✅ backtest()メソッド（Lines 243-279）- 詳細確認済み
- ✅ generate_entry_signal()メソッド（Lines 140-182）- 詳細確認済み
- ✅ generate_exit_signal()メソッド（Lines 187-240）- 詳細確認済み
- ✅ initialize_strategy()メソッド（Lines 60-90）- 詳細確認済み

**確認した変数・カラム名**:
- ✅ `self.entry_prices` - Lines 37, 178, 182, 209使用確認
- ✅ `self.price_column` - Line 31でデフォルト値"Adj Close"確認
- ✅ `current_price` - Lines 147, 213で当日終値取得確認
- ✅ `self.data['RSI']` - Line 66でshift(1)適用確認

**データの流れ**:
- ✅ yfinance → DataFrame → backtest() → generate_entry_signal() → self.entry_prices[idx] = current_price
- ✅ 当日終値（idx）を直接記録している証拠確認

### b) 思い込みチェック ✅

**前提の検証**:
- ❌ 「BaseStrategy.backtest()を使用しているはず」 → ✅ 独自実装と確認（Line 243）
- ❌ 「エントリー価格はbacktest()で記録するはず」 → ✅ generate_entry_signal()で記録と確認（Lines 178, 182）
- ❌ 「shift(1)があれば大丈夫なはず」 → ✅ RSIは正しいが、エントリー価格は誤りと判明

**実際に確認した事実**:
- ✅ Line 243: `def backtest(...):` - 独自実装の証拠
- ✅ Lines 178, 182: `self.entry_prices[idx] = current_price` - 当日終値記録の証拠
- ✅ Line 147: `current_price = self.data[self.price_column].iloc[idx]` - 当日終値取得の証拠
- ✅ Line 66: `self.data['RSI'] = calculate_rsi(...).shift(1)` - shift(1)適用の証拠

### c) 矛盾チェック ✅

**調査結果の整合性**:
- ✅ 独自backtest()実装 → BaseStrategyの修正が無効 → 整合
- ✅ RSIはshift(1)適用 → エントリー価格は当日終値 → 矛盾を説明済み（判断は正しいが価格が誤り）
- ✅ ループ範囲が最終日含む → idx+1アクセスに変更時にIndexError → 整合

**コードとコメントの整合性**:
- ✅ Line 66コメント: "ルックアヘッドバイアス修正: shift(1)を追加" → RSIのみ適用、エントリー価格は未修正
- ⚠️ 部分的な修正のみ実施されている（RSIのみ）

---

## 次のステップ

### 推奨する作業順序

1. **Phase 1実装**: エントリー価格を翌日始値に変更
   - ループ範囲を`range(len(self.data) - 1)`に変更
   - generate_entry_signal()でのentry_prices記録を削除
   - backtest()メソッドで翌日始値を記録
   - 境界条件チェック（idx+1がデータ範囲内）

2. **検証**: 修正後のバックテスト実行
   - ダミーデータでの構文チェック
   - 実データでのエントリー価格確認
   - 13桁精度の消失を確認
   - 翌日始値との一致を確認

3. **Phase 2実装**: スリッページ・取引コスト追加
   - パラメータ化（config.yaml等で管理）
   - デフォルト値: slippage=0.1%, commission=0.1%

4. **Phase 3実装**: イグジット問題の対応
   - EXIT_INVESTIGATION_REPORT.mdで詳細設計
   - イグジット価格の翌日始値使用（または当日高値・安値の適切な使用）

5. **ドキュメント更新**
   - copilot-instructions.mdの遵守状況を記録
   - 修正前後のバックテスト結果比較レポート作成

---

## 付録

### 証拠ファイル

1. **`strategies/contrarian_strategy.py`** - 調査対象ファイル
2. **`docs/Lookhead bias problem/INVESTIGATION_REPORT.md`** - 参照報告書

### 参考資料

- [copilot-instructions.md](.github/copilot-instructions.md) - ルックアヘッドバイアス禁止ルール
- [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) - BaseStrategy調査報告書

---

**報告書作成者**: GitHub Copilot  
**最終更新日**: 2025-12-22  
**バージョン**: 1.0

---

## 調査結論

### 確定事項

1. ✅ **contrarian_strategy.pyは独自backtest()メソッドを実装**
   - BaseStrategy.backtest()を使用していない
   - base_strategy.py Line 285の修正が適用されない

2. ✅ **エントリー価格のルックアヘッドバイアス存在**
   - Lines 178, 182: `self.entry_prices[idx] = current_price`
   - current_priceは当日終値（Adj Close）
   - リアルトレードでは不可能

3. ✅ **イグジット判定のルックアヘッドバイアス存在**
   - Line 213: `current_price = self.data[self.price_column].iloc[idx]`
   - 当日終値で利益確定・ストップロス判定
   - トレーリングストップも当日価格で更新

4. ✅ **RSIはshift(1)適用済み**
   - Line 66: `self.data['RSI'] = calculate_rsi(...).shift(1)`
   - 判断ロジックは正しい

5. ✅ **ループ範囲が最終日を含む**
   - Line 253: `for idx in range(len(self.data)):`
   - idx+1アクセスに変更時にIndexErrorのリスク

### 不明な点

1. ⏳ **実際のバックテスト結果との比較**
   - 実データでの検証は未実施
   - 13桁精度の有無は未確認
   - エントリー価格と市場データの実際の差分は未計測

### 次の作業

1. **Phase 1修正の実装** - 3箇所の修正
   - ループ範囲変更（Line 253）
   - entry_prices記録削除（Lines 178, 182）
   - backtest()で翌日始値記録（Lines 268-271付近）

2. **検証スクリプト作成** - ダミーデータでの構文チェック

3. **実データ検証** - 修正前後の比較

