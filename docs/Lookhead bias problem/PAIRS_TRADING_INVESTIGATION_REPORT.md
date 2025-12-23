# PairsTradingStrategy ルックアヘッドバイアス調査報告書

**作成日**: 2025-12-22  
**調査者**: GitHub Copilot  
**調査範囲**: strategies/pairs_trading_strategy.py  
**参照ドキュメント**: docs/Lookhead bias problem/INVESTIGATION_REPORT.md  

---

## 目次

1. [調査目的](#調査目的)
2. [確認項目チェックリスト](#確認項目チェックリスト)
3. [調査結果](#調査結果)
4. [原因分析](#原因分析)
5. [影響範囲](#影響範囲)
6. [修正提案](#修正提案)
7. [セルフチェック](#セルフチェック)

---

## 調査目的

strategies/pairs_trading_strategy.pyにおけるルックアヘッドバイアス（未来データ使用）問題の有無を調査し、問題箇所を特定する。

**ルックアヘッドバイアスとは:**
- 当日終値を見てから当日終値でエントリーする（リアルトレードでは不可能）
- インジケーターが当日データを使用する（shift(1)未適用）
- エントリー価格が翌日始値ではなく当日終値を使用

**調査の制約:**
- エグジット問題は別ファイル対応のため今回はスルー
- 修正は行わず、調査・問題特定のみ

---

## 確認項目チェックリスト

### 優先度A（最重要）

1. **[P-A1] エントリー価格の決定ロジック**
   - 対象: backtest()メソッド内のself.entry_prices[i]設定箇所
   - 確認内容: 当日終値を使用しているか、翌日始値を使用しているか
   - 重要度: 最高（ルックアヘッドバイアスの直接原因）

2. **[P-A2] ループ範囲の境界条件**
   - 対象: backtest()メソッドのfor i in range()
   - 確認内容: idx+1アクセスが発生する場合、最終日を除外しているか
   - 重要度: 高（IndexError防止）

3. **[P-A3] インジケーターのshift(1)適用状況**
   - 対象: initialize_strategy()メソッド内の全インジケーター計算
   - 確認内容: 前日データを使用しているか（shift(1)適用）
   - 重要度: 高（判断に使用するデータの正確性）

### 優先度B（重要）

4. **[P-B1] generate_entry_signal()の実装**
   - 対象: generate_entry_signal(idx)メソッド
   - 確認内容: idx日のデータのみを使用しているか、shift済みデータを使用しているか
   - 重要度: 中（エントリー判断の正確性）

5. **[P-B2] BaseStrategyとの関係**
   - 対象: class PairsTradingStrategy(BaseStrategy)
   - 確認内容: BaseStrategyのbacktest()を使用するか、独自実装か
   - 重要度: 中（修正範囲の特定）

---

## 調査結果

### [P-A1] エントリー価格の決定ロジック 問題あり

**ファイル**: strategies/pairs_trading_strategy.py  
**箇所**: Line 285

**コード:**
```python
# Line 282-287
entry_signal = self.generate_entry_signal(i)
if entry_signal == 1:
    result_data['Entry_Signal'].iloc[i] = 1
    position_size = 1.0
    self.entry_prices[i] = result_data[self.price_column].iloc[i]
    self.position_days[i] = 0
```

**確認事項:**
- result_data[self.price_column].iloc[i] を使用 → self.price_column = "Adj Close"（デフォルト）
- idx日目の終値（Adj Close）でエントリー価格を記録

**証拠:**
- Line 285: `self.entry_prices[i] = result_data[self.price_column].iloc[i]`
- Line 29: `price_column: str = "Adj Close"`（引数デフォルト値）
- Line 31: `self.price_column = price_column`

**結論:**
ルックアヘッドバイアス存在 - **当日終値でエントリー価格を決定**

**リアルトレードとの乖離:**
- idx日の市場終了後にシグナル判断
- しかしエントリー価格はidx日の終値を使用
- リアルトレードでは、idx日の終値を見てからidx日の終値で買うことは不可能
- 正しくは、idx+1日の始値でエントリー

---

### [P-A2] ループ範囲の境界条件 潜在的問題あり

**ファイル**: strategies/pairs_trading_strategy.py  
**箇所**: Line 271

**コード:**
```python
# Line 271
for i in range(len(result_data)):
```

**確認事項:**
- 現在は`len(result_data)`まで全日ループ
- エントリー価格を翌日始値（idx+1）に変更する場合、最終日でIndexErrorが発生

**現状の問題:**
- 当日終値を使用しているため、IndexErrorは発生しない
- しかし、修正後（翌日始値使用）にはIndexErrorが発生する

**必要な修正:**
```python
# 修正前
for i in range(len(result_data)):

# 修正後
for i in range(len(result_data) - 1):
```

**結論:**
Phase 1修正（翌日始値使用）に伴い、ループ範囲の変更が必要

---

### [P-A3] インジケーターのshift(1)適用状況 問題なし

**ファイル**: strategies/pairs_trading_strategy.py  
**箇所**: Lines 68-113（initialize_strategy()メソッド）

**確認したインジケーター:**

#### 1. SMA_Short（短期移動平均） - shift(1)未適用だが問題の可能性は低い
```python
# Line 69-72
self.data['SMA_Short'] = self.data[self.price_column].rolling(
    window=self.params["short_ma_period"]
).mean()
```
- 移動平均自体は過去データを使用
- ただし、idx日のSMA_Shortはidx日までのデータを含む
- 厳密には、idx-1日までのSMAを使用すべき（shift(1)適用）

#### 2. SMA_Long（長期移動平均） - shift(1)未適用だが問題の可能性は低い
```python
# Line 74-77
self.data['SMA_Long'] = self.data[self.price_column].rolling(
    window=self.params["long_ma_period"]
).mean()
```
- SMA_Shortと同様

#### 3. Spread（スプレッド） - shift(1)未適用
```python
# Line 79-80
self.data['Spread'] = self.data['SMA_Short'] - self.data['SMA_Long']
```
- SMA_Short、SMA_Longの差分
- 両方がshift(1)未適用のため、Spreadもshift(1)未適用

#### 4. Spread_MA（スプレッド移動平均） - shift(1)未適用
```python
# Line 82-85
self.data['Spread_MA'] = self.data['Spread'].rolling(
    window=self.params["spread_period"]
).mean()
```

#### 5. Spread_Std（スプレッド標準偏差） - shift(1)未適用
```python
# Line 87-90
self.data['Spread_Std'] = self.data['Spread'].rolling(
    window=self.params["spread_period"]
).std()
```

#### 6. Spread_ZScore（Z-Score） - shift(1)未適用
```python
# Line 92-95
self.data['Spread_ZScore'] = (
    (self.data['Spread'] - self.data['Spread_MA']) / self.data['Spread_Std']
)
```

#### 7. Volume_MA（ボリューム移動平均） - shift(1)未適用
```python
# Line 98-101
if self.params["volume_filter"]:
    self.data['Volume_MA'] = self.data['Volume'].rolling(
        window=self.params["spread_period"]
    ).mean()
```

#### 8. Volatility（ボラティリティ） - shift(1)未適用
```python
# Line 104-108
if self.params["volatility_filter"]:
    returns = self.data[self.price_column].pct_change()
    self.data['Volatility'] = returns.rolling(
        window=self.params["volatility_period"]
    ).std()
```

#### 9. MA_Correlation（移動平均間相関） - shift(1)未適用
```python
# Line 110-114
if len(self.data) >= self.params["cointegration_lookback"]:
    correlation_window = min(self.params["cointegration_lookback"], len(self.data))
    self.data['MA_Correlation'] = self.data['SMA_Short'].rolling(
        window=correlation_window
    ).corr(self.data['SMA_Long'])
```

**評価:**

**インジケーター種類の特性:**
- 移動平均（SMA）: idx日の値は、idx日を含む過去N日間の平均
- 相関（Correlation）: idx日の値は、idx日を含む過去N日間の相関
- これらは「idx日の終値が確定してから計算可能」

**ルックアヘッドバイアスの観点:**
- VWAP_Breakout.pyやMeanReversionStrategyは`.shift(1)`を明示的に適用
- PairsTradingStrategyは`.shift(1)`を適用していない
- しかし、移動平均自体は「過去データの集約」であり、厳密には問題ではない

**判断:**
- 厳密に言えば、idx日の終値を含むSMAを使用するのは軽度のルックアヘッドバイアス
- しかし、INVESTIGATION_REPORT.mdの主な問題（エントリー価格が当日終値）に比べれば影響は小さい
- **結論: 軽度の問題あり（Phase 2で対応推奨）**

**修正推奨:**
```python
# 修正例（Phase 2推奨）
self.data['SMA_Short'] = self.data[self.price_column].rolling(
    window=self.params["short_ma_period"]
).mean().shift(1)

self.data['SMA_Long'] = self.data[self.price_column].rolling(
    window=self.params["long_ma_period"]
).mean().shift(1)
```

---

### [P-B1] generate_entry_signal()の実装 問題なし（条件判断のみ）

**ファイル**: strategies/pairs_trading_strategy.py  
**箇所**: Lines 165-203

**コード:**
```python
def generate_entry_signal(self, idx: int) -> int:
    """エントリーシグナル生成"""
    if idx < max(self.params["spread_period"], self.params["long_ma_period"]):
        return 0
        
    spread_zscore = self.data['Spread_ZScore'].iloc[idx]
    
    if pd.isna(spread_zscore):
        return 0
        
    # 相関条件チェック
    if not self._check_correlation_condition(idx):
        return 0
        
    # ボリューム条件チェック
    if not self._check_volume_condition(idx):
        return 0
        
    # ボラティリティ条件チェック
    if not self._check_volatility_condition(idx):
        return 0
        
    # エントリー条件：スプレッドが異常に拡大した場合
    entry_threshold = self.params["entry_threshold"]
    
    # 正の異常値（短期MAが長期MAを大幅に上回る）
    # → 回帰を期待してショート的な動き（実際はロング）
    if spread_zscore >= entry_threshold:
        return 1  # ロングエントリー
        
    # 負の異常値（短期MAが長期MAを大幅に下回る）
    # → 回帰を期待してロング
    elif spread_zscore <= -entry_threshold:
        return 1  # ロングエントリー
        
    return 0
```

**確認事項:**
- idx日のSpread_ZScoreを使用してエントリー判断
- Spread_ZScoreはidx日までのデータで計算されたもの（shift(1)未適用）
- しかし、エントリー判断自体はシグナル生成のみ（価格決定は別箇所）

**結論:**
- generate_entry_signal()は条件判断のみを行い、価格決定は行わない
- 価格決定はbacktest()メソッドのLine 285で実施
- generate_entry_signal()自体に直接的な問題はなし

**ただし:**
- インジケーター（Spread_ZScore等）がshift(1)未適用の場合、軽度の問題あり
- Phase 2でインジケーターのshift(1)適用を推奨

---

### [P-B2] BaseStrategyとの関係 独自実装

**ファイル**: strategies/pairs_trading_strategy.py  
**箇所**: Lines 268-313

**確認事項:**
```python
# Line 23
class PairsTradingStrategy(BaseStrategy):

# Line 268
def backtest(self, trading_start_date=None, trading_end_date=None) -> pd.DataFrame:
```

**結論:**
- BaseStrategyを継承しているが、backtest()メソッドは独自実装
- BaseStrategy.backtest()を呼び出していない（super().backtest()なし）
- base_strategy.pyの修正（翌日始値使用）の影響を受けない

**修正範囲:**
- pairs_trading_strategy.py単独で修正が必要
- base_strategy.pyの修正パターンを参考にする

---

## 原因分析

### 根本原因

**直接原因**: strategies/pairs_trading_strategy.py Line 285
```python
self.entry_prices[i] = result_data[self.price_column].iloc[i]  # price_column = 'Adj Close'
```

**問題の構造:**
1. i日目に`generate_entry_signal(i)`でエントリー判断
2. 判断はi日目までのインジケーター（Spread_ZScore等）を使用
3. しかし、エントリー価格はi日目の終値（Adj Close）を使用
4. リアルトレードでは、i日目の終値を見てからi日目の終値で買うことは不可能

### 正しい実装

```python
# 現状（誤り）
self.entry_prices[i] = result_data[self.price_column].iloc[i]  # i日の終値

# 正しい実装（Phase 1）
entry_price = result_data['Open'].iloc[i + 1]  # i+1日の始値
self.entry_prices[i] = entry_price
```

**理由:**
- i日の市場終了後にシグナル判断
- 翌日（i+1日）の市場開始時（始値）でエントリー
- これがリアルトレードの実態

---

## 影響範囲

### ルックアヘッドバイアスの影響

**エントリー価格の精度:**
- 現状: 当日終値（Adj Close）でエントリー
- 修正後: 翌日始値（Open）でエントリー
- 価格差: INVESTIGATION_REPORT.mdの実データ検証では3-7%の乖離を確認

**バックテスト結果への影響:**
- リターン率: 過大評価（有利なエントリー価格）
- シャープレシオ: 過大評価（ボラティリティが正しく反映されない）
- 最大ドローダウン: 過小評価（不利な価格でのエントリーが反映されない）
- 勝率: 過大評価（有利なエントリー価格での取引）

### インジケーターshift(1)未適用の影響

**影響度:**
- エントリー価格問題に比べて影響は小さい
- 移動平均自体は過去データを使用しているため
- しかし、idx日の終値を含むMAを使用するのは厳密には問題

**推奨対応:**
- Phase 1: エントリー価格のみ修正（最優先）
- Phase 2: インジケーターのshift(1)適用（推奨）

---

## 修正提案

### Phase 1: 最小限の修正（必須）

#### 修正箇所1: ループ範囲の変更

**ファイル**: strategies/pairs_trading_strategy.py Line 271

**修正前:**
```python
for i in range(len(result_data)):
```

**修正後:**
```python
# Phase 1修正: 最終日を除外してi+1アクセスを安全に（ルックアヘッドバイアス修正）
# 理由: エントリー価格を翌日始値（i+1）に変更するため、最終日でのIndexError回避
for i in range(len(result_data) - 1):
```

---

#### 修正箇所2: エントリー価格を翌日始値に変更

**ファイル**: strategies/pairs_trading_strategy.py Lines 282-287

**修正前:**
```python
entry_signal = self.generate_entry_signal(i)
if entry_signal == 1:
    result_data['Entry_Signal'].iloc[i] = 1
    position_size = 1.0
    self.entry_prices[i] = result_data[self.price_column].iloc[i]
    self.position_days[i] = 0
```

**修正後:**
```python
entry_signal = self.generate_entry_signal(i)
if entry_signal == 1:
    result_data['Entry_Signal'].iloc[i] = 1
    position_size = 1.0
    # Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
    # 理由: i日の終値を見てからi日の終値で買うことは不可能
    # リアルトレードでは翌日（i+1日目）の始値でエントリー
    next_day_open = result_data['Open'].iloc[i + 1]
    self.entry_prices[i] = next_day_open
    self.position_days[i] = 0
```

---

### Phase 2: 現実的な制約の追加（推奨）

#### 修正箇所3: スリッページの考慮

**ファイル**: strategies/pairs_trading_strategy.py Lines 285付近

**修正案:**
```python
# Phase 2修正: スリッページ考慮（推奨0.1%）
slippage = self.params.get("slippage", 0.001)  # デフォルト0.1%
next_day_open = result_data['Open'].iloc[i + 1]
entry_price_with_slippage = next_day_open * (1 + slippage)
self.entry_prices[i] = entry_price_with_slippage
```

#### 修正箇所4: インジケーターのshift(1)適用

**ファイル**: strategies/pairs_trading_strategy.py Lines 69-114

**修正案:**
```python
# Phase 2修正: インジケーターのshift(1)適用
self.data['SMA_Short'] = self.data[self.price_column].rolling(
    window=self.params["short_ma_period"]
).mean().shift(1)

self.data['SMA_Long'] = self.data[self.price_column].rolling(
    window=self.params["long_ma_period"]
).mean().shift(1)

# Spread等も同様にshift(1)の影響を受ける
self.data['Spread'] = self.data['SMA_Short'] - self.data['SMA_Long']
# （以降のインジケーターはSpreadベースなので自動的にshift適用）
```

---

## セルフチェック

### a) 見落としチェック

**確認したファイル:**
- strategies/pairs_trading_strategy.py（Lines 1-410、全行確認済み）
- docs/Lookhead bias problem/INVESTIGATION_REPORT.md（参照完了）

**確認したカラム名・変数名:**
- `self.price_column`（デフォルト: "Adj Close"）
- `self.entry_prices[i]`（エントリー価格記録）
- `result_data[self.price_column].iloc[i]`（当日終値使用）
- `result_data['Open'].iloc[i + 1]`（翌日始値、修正提案）

**確認したメソッド:**
- `backtest()`（Lines 268-313）
- `initialize_strategy()`（Lines 68-119）
- `generate_entry_signal()`（Lines 165-203）
- `generate_exit_signal()`（Lines 205-256、エグジット問題は別ファイル対応）

**データの流れ:**
- yfinance → CSV → DataFrame → initialize_strategy() → backtest() → entry_prices[i]
- 追跡完了

**確認していないファイル（今回の対象外）:**
- indicators/basic_indicators.py（インジケーター計算、必要であれば追加調査）
- strategies/base_strategy.py（既にINVESTIGATION_REPORT.mdで調査済み）

---

### b) 思い込みチェック

**前提の検証:**
- 「エントリー価格は終値のはず」 → コードで確認（Line 285）
- 「BaseStrategyのbacktest()を使用しているはず」 → 独自実装と確認（Line 268）
- 「インジケーターにshift(1)があるはず」 → shift(1)未適用と確認（Lines 69-114）

**実際に確認した事実:**
- Line 285で`self.entry_prices[i] = result_data[self.price_column].iloc[i]`を確認
- Line 29で`price_column: str = "Adj Close"`を確認
- Line 268で独自backtest()実装を確認
- Lines 69-114でshift(1)未適用を確認

**推測と事実の区別:**
- 事実: エントリー価格は当日終値を使用（Line 285）
- 事実: インジケーターはshift(1)未適用（Lines 69-114）
- 事実: backtest()は独自実装（Line 268）
- 推測: INVESTIGATION_REPORT.mdと同様に3-7%の価格差が発生する可能性
- 推測: インジケーターのshift(1)未適用は軽度の問題

---

### c) 矛盾チェック

**調査結果の整合性:**
- エントリー価格が当日終値 → ルックアヘッドバイアス存在 → 一貫
- インジケーターのshift(1)未適用 → 軽度の問題 → 一貫
- backtest()独自実装 → base_strategy.py修正の影響なし → 一貫

**INVESTIGATION_REPORT.mdとの整合性:**
- INVESTIGATION_REPORT.md: base_strategy.py Line 242で当日終値使用
- 本報告書: pairs_trading_strategy.py Line 285で当日終値使用
- 両方とも同様のパターン → 一貫

**copilot-instructions.mdとの整合性:**
- ルックアヘッドバイアス禁止ルール: 翌日始値でエントリー必須
- 本報告書の修正提案: 翌日始値に変更
- 一貫

---

## 判明したこと（証拠付き）

### 1. ルックアヘッドバイアス存在（確定）

**証拠:**
- strategies/pairs_trading_strategy.py Line 285
- `self.entry_prices[i] = result_data[self.price_column].iloc[i]`
- `self.price_column = "Adj Close"`（デフォルト）

**結論:**
- エントリー価格が当日終値（Adj Close）を使用
- リアルトレードでは不可能な取引
- ルックアヘッドバイアス存在を確定

---

### 2. インジケーターのshift(1)未適用（軽度の問題）

**証拠:**
- strategies/pairs_trading_strategy.py Lines 69-114
- 全インジケーターでshift(1)未適用を確認

**結論:**
- 厳密には軽度のルックアヘッドバイアス
- 移動平均自体は過去データを使用しているため、影響は小さい
- Phase 2で対応推奨

---

### 3. backtest()は独自実装

**証拠:**
- strategies/pairs_trading_strategy.py Line 268
- `def backtest(self, trading_start_date=None, trading_end_date=None) -> pd.DataFrame:`
- super().backtest()の呼び出しなし

**結論:**
- BaseStrategy.backtest()を使用していない
- base_strategy.pyの修正の影響を受けない
- pairs_trading_strategy.py単独で修正が必要

---

## 不明な点

### 1. 実データでの価格差

**不明内容:**
- 実際のバックテスト結果で、エントリー価格と翌日始値の価格差がどの程度か

**推測:**
- INVESTIGATION_REPORT.mdの実データ検証では3-7%の乖離を確認
- PairsTradingStrategyでも同様の乖離が発生する可能性

**確認方法:**
- verify_entry_prices.py相当のスクリプトを作成し、実データで検証
- 今回の調査範囲外（修正後の検証フェーズで実施）

---

### 2. インジケーターshift(1)の影響度

**不明内容:**
- インジケーターのshift(1)未適用が、バックテスト結果にどの程度影響するか

**推測:**
- 移動平均自体は過去データを使用しているため、影響は小さい
- しかし、厳密にはidx日の終値を含むMAを使用するのは問題

**確認方法:**
- Phase 1修正後、Phase 2でshift(1)適用し、結果を比較
- 今回の調査範囲外（修正後の検証フェーズで実施）

---

## 原因の推定（可能性順）

### 第1位: copilot-instructions.mdルール策定前の実装（可能性: 高）

**根拠:**
- copilot-instructions.mdのルックアヘッドバイアス禁止ルールは2025-12-20以降必須
- pairs_trading_strategy.pyの作成日: 2025-07-22（Line 11）
- ルール策定前の実装

**結論:**
- ルール策定前の実装のため、ルックアヘッドバイアスが混入
- 最も可能性が高い

---

### 第2位: BaseStrategy.backtest()を使用しなかったため（可能性: 中）

**根拠:**
- base_strategy.pyは2025-12-21に修正済み（INVESTIGATION_REPORT.md記載）
- しかし、pairs_trading_strategy.pyは独自backtest()実装のため、修正の影響を受けない

**結論:**
- 独自実装のため、base_strategy.pyの修正が適用されない
- 中程度の可能性

---

### 第3位: インジケーター計算の簡略化（可能性: 低）

**根拠:**
- 移動平均は過去データを使用しているため、shift(1)の必要性が明確でない
- 実装者がshift(1)の重要性を認識していなかった可能性

**結論:**
- インジケーターのshift(1)未適用は、実装者の認識不足の可能性
- しかし、エントリー価格問題に比べて影響は小さい
- 低い可能性

---

## 次のステップ（修正フェーズ）

### 推奨する作業順序

1. **Phase 1実装**: pairs_trading_strategy.pyの修正
   - Line 271: ループ範囲変更（`range(len(result_data) - 1)`）
   - Lines 282-287: エントリー価格を翌日始値に変更
   - 検証スクリプト作成・実行

2. **検証**: 修正後のバックテスト実行
   - ダミーデータでの構文チェック
   - エントリー価格と翌日始値の差分確認
   - 最終日のエントリーシグナル確認

3. **Phase 2実装**: スリッページ・インジケーターshift(1)追加
   - スリッページ考慮（推奨0.1%）
   - インジケーターのshift(1)適用

4. **ドキュメント更新**
   - copilot-instructions.mdの遵守状況を記録
   - 修正前後のバックテスト結果比較レポート作成

---

## 結論

### ルックアヘッドバイアス存在を確定

**証拠:**
- strategies/pairs_trading_strategy.py Line 285
- エントリー価格が当日終値（Adj Close）を使用
- リアルトレードでは不可能な取引

**修正必須箇所:**
1. Line 271: ループ範囲変更
2. Lines 282-287: エントリー価格を翌日始値に変更

**修正推奨箇所:**
- Lines 69-114: インジケーターのshift(1)適用（Phase 2）
- スリッページ考慮（Phase 2）

**遵守事項:**
- copilot-instructions.mdのルックアヘッドバイアス禁止ルール準拠
- Phase 1修正完了後、Phase 2実装を推奨

---

**報告書作成者**: GitHub Copilot  
**作成日**: 2025-12-22  
**バージョン**: 1.0

---
