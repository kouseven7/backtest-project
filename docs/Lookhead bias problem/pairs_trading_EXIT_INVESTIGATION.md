# pairs_trading_strategy.py イグジット問題 調査報告書

**作成日**: 2025-12-23  
**調査者**: GitHub Copilot  
**調査対象**: strategies/pairs_trading_strategy.py (generate_exit_signal)  
**調査ステータス**: 調査中  
**関連ドキュメント**: 
- [EXIT_INVESTIGATION_REPORT.md](EXIT_INVESTIGATION_REPORT.md) - イグジット問題の一般調査
- [VWAP_EXIT_INVESTIGATION.md](VWAP_EXIT_INVESTIGATION.md) - VWAP Phase 1b修正完了記録
- [support_resistance_contrarian_EXIT_INVESTIGATION.md](support_resistance_contrarian_EXIT_INVESTIGATION.md) - support_resistance Phase 1b-3修正完了記録

---

## 目次

1. [調査目的](#調査目的)
2. [確認項目チェックリスト](#確認項目チェックリスト)
3. [調査結果](#調査結果)
4. [原因分析](#原因分析)
5. [影響範囲](#影響範囲)
6. [改善提案](#改善提案)
7. [セルフチェック](#セルフチェック)
8. [次のステップ](#次のステップ)

---

## 調査目的

pairs_trading_strategy.pyのPhase 1（エントリー価格修正）が完了した後、**イグジット価格**についても同様のルックアヘッドバイアス問題が存在するか調査する。

### 調査の背景

**Phase 1完了状況**:
- ✅ エントリー価格修正完了（Lines 298-311）
- ✅ 翌日始値 + スリッページ適用
- ✅ ループ範囲調整完了（`range(len(result_data) - 1)`）

**Phase 1b（イグジット）調査対象**:
- ❓ イグジット価格の取得方法
- ❓ ストップロス・利益確定・スプレッド回帰判定のルックアヘッドバイアス有無
- ❓ VWAP_Breakout.py / support_resistance_contrarian_strategy.py Phase 1b修正パターンの適用可否

---

## 確認項目チェックリスト

### Phase 1: 基本構造確認（優先度：最高）
- [ ] [C1] pairs_trading_strategy.py全体構造確認
- [ ] [C2] generate_exit_signal()メソッドの存在確認
- [ ] [C3] BaseStrategy継承の確認
- [ ] [C4] backtest()メソッドの実装確認（独自実装 or 継承）

### Phase 2: イグジット価格問題特定（優先度：高）
- [ ] [C5] current_price取得箇所の特定
- [ ] [C6] entry_price取得方法の確認
- [ ] [C7] 各イグジット条件の実装確認
- [ ] [C8] ストップロス判定の確認
- [ ] [C9] 利益確定判定の確認
- [ ] [C10] スプレッド回帰判定の確認（ペアトレーディング特有）

### Phase 3: ルックアヘッドバイアス確認（優先度：高）
- [ ] [C11] current_priceが当日終値を使用しているか確認
- [ ] [C12] spread_zscoreのshift(1)適用確認
- [ ] [C13] インジケーターのshift(1)適用確認

### Phase 4: 修正方針検討（優先度：中）
- [ ] [C14] EXIT_INVESTIGATION_REPORT.mdとの比較
- [ ] [C15] support_resistance_contrarian_strategy.py Phase 1b修正パターンの適用可否判断

---

## 調査結果

### [C1] 全体構造確認 ✅

**確認結果**:
- ファイル名: pairs_trading_strategy.py
- クラス名: PairsTradingStrategy
- 総行数: 442行
- BaseStrategy継承確認✅（Line 24）

**戦略の特徴**:
- ペアトレーディング戦略の簡略版
- 短期MA（5日）と長期MA（20日）の乖離を利用
- スプレッドZ-Scoreベースのエントリー/エグジット
- Phase 1修正完了（エントリー価格を翌日始値に変更、Lines 298-311）

**根拠**: strategies/pairs_trading_strategy.py Lines 1-30

---

### [C2] generate_exit_signal()メソッドの存在確認 ✅

**確認箇所**: Line 207

**実際のコード**:
```python
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
```

**特徴**:
- 引数: `idx`（現在のインデックス）、`position_size`（ポジションサイズ）
- 戻り値: 1（イグジット）、0（ホールド）

**根拠**: grep_search実行結果

---

### [C3] BaseStrategy継承の確認 ✅

**確認箇所**: Line 24

**実際のコード**:
```python
class PairsTradingStrategy(BaseStrategy):
```

**根拠**: strategies/pairs_trading_strategy.py Line 24

---

### [C4] backtest()メソッドの実装確認 ✅

**確認箇所**: Line 267

**実際のコード**:
```python
def backtest(self, trading_start_date=None, trading_end_date=None) -> pd.DataFrame:
```

**結果**: **独自実装あり**（BaseStrategy.backtest()をオーバーライド）

**Phase 1修正完了確認**:
- Line 278: ループ範囲 `range(len(result_data) - 1)`（i+1アクセス安全）
- Lines 298-311: エントリー価格を翌日始値 + スリッページ適用

**根拠**: strategies/pairs_trading_strategy.py Lines 267-330

---

### [C5] current_price取得箇所の特定 ✅

**確認箇所**: Line 213

**実際のコード**:
```python
# スカラー値として取得
current_price_val = self.data[self.price_column].iloc[idx]
if isinstance(current_price_val, pd.Series):
    current_price_val = current_price_val.values[0]
```

**使用している価格カラム**:
- `self.price_column` = "Adj Close"（デフォルト、Line 28で設定）
- Line 213でidx日目の`Adj Close`を取得

**ルックアヘッドバイアスの有無**:
- ✅ **ルックアヘッドバイアスあり** - idx日目の終値を使用
- [ ] **ルックアヘッドバイアスなし** - 翌日始値を使用
- [ ] **不明** - 追加調査が必要

**証拠**:
```python
# Line 213
current_price_val = self.data[self.price_column].iloc[idx]  # idx日目の終値（Adj Close）
```

**問題の構造**:
1. idx日目の`generate_exit_signal(idx)`でイグジット判断
2. `current_price_val = self.data[self.price_column].iloc[idx]` - idx日目の終値を取得
3. idx日目の終値でストップロス・利益確定・スプレッド回帰判定
4. **リアルトレードでは、idx日目の終値を見てからidx日目の終値で売ることは不可能**

**根拠**: strategies/pairs_trading_strategy.py Lines 213-215

---

### [C6] entry_price取得方法の確認 ✅

**確認箇所**: Lines 221-234

**実際のコード**:
```python
# エントリー価格を取得
entry_price = None
entry_idx = None

for i in range(max(0, idx - self.params["max_hold_days"]), idx):
    if i in self.entry_prices:
        entry_price_raw = self.entry_prices[i]
        # スカラー値として取得
        if isinstance(entry_price_raw, pd.Series):
            entry_price = entry_price_raw.values[0]
        elif isinstance(entry_price_raw, (np.ndarray, np.generic)):
            entry_price = float(entry_price_raw)
        else:
            entry_price = entry_price_raw
        entry_idx = i
        break
```

**特徴**:
- `self.entry_prices`辞書から取得
- エントリー価格は独自backtest()で記録（Lines 298-311）
- Phase 1修正完了: 翌日始値 + スリッページ適用
- Series型エラー防止処理あり（Lines 225-232）

**検証**:
```python
# Lines 298-311（Phase 1修正完了）
next_day_open_val = result_data['Open'].iloc[i + 1]
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
entry_price = next_day_open_val * (1 + slippage + transaction_cost)
self.entry_prices[i] = entry_price
```

**結論**: entry_priceは問題なし（Phase 1修正完了済み）

**根拠**: strategies/pairs_trading_strategy.py Lines 221-234, 298-311

---

### [C7] 各イグジット条件の実装確認 ✅

**確認箇所**: Lines 240-264

**イグジット条件一覧**:

| イグジット条件 | 実装箇所 | 使用価格 | 問題 |
|---------------|---------|---------|------|
| 最大保有日数 | Lines 240-242 | `hold_days >= max_hold_days` | ✅ 問題なし（日数のみ） |
| ストップロス | Lines 247-249 | `pnl_pct <= -stop_loss_pct`（current_price使用） | ❌ 当日終値使用 |
| 利益確定 | Lines 252-254 | `pnl_pct >= take_profit_pct`（current_price使用） | ❌ 当日終値使用 |
| スプレッド回帰 | Lines 257-262 | `abs(spread_zscore_val) <= exit_threshold` | ❌ 当日スプレッド使用 |

**根拠**: strategies/pairs_trading_strategy.py Lines 240-264

---

### [C8] ストップロス判定の確認 ✅

**確認箇所**: Lines 245, 247-249

**実際のコード**:
```python
# 損益計算
pnl_pct = (current_price_val - entry_price) / entry_price

# ストップロス
if pnl_pct <= -self.params["stop_loss_pct"]:
    return 1
```

**ルックアヘッドバイアスの問題**:
- `current_price_val` = idx日目の終値（Line 213）
- ストップロス価格 = `entry_price * (1 - 0.04)` = 4%下落
- **問題**: idx日目の終値を見てからストップロス判定

**リアルトレードの実態**:
```
entry_price = 1000円
stop_loss_price = 1000 * (1 - 0.04) = 960円

現在のロジック（誤り）:
- idx日目の終値 = 955円
- 955 <= 960 → ストップロス発動
- イグジット価格 = 955円（当日終値）

リアルトレードの実態:
- idx日目の値動き: Open=995 → High=1005 → Low=950 → Close=955
- Low=950 < 960 → ストップロス発動（当日中）
- イグジット価格 ≒ 960円（ストップロス価格 - スリッページ）

差額: 5円（0.5%）→ 現在のロジックはストップロスが厳しすぎる
```

**EXIT_INVESTIGATION_REPORT.mdとの比較**:
- VWAP_Breakout.pyと同じ問題構造
- 当日終値でストップロス判定
- リアルトレードでは当日安値で判定すべき

**根拠**: strategies/pairs_trading_strategy.py Lines 245, 247-249

---

### [C9] 利益確定判定の確認 ✅

**確認箇所**: Lines 245, 252-254

**実際のコード**:
```python
# 損益計算
pnl_pct = (current_price_val - entry_price) / entry_price

# 利益確定
if pnl_pct >= self.params["take_profit_pct"]:
    return 1
```

**ルックアヘッドバイアスの問題**:
- `current_price_val` = idx日目の終値（Line 213）
- 利益確定価格 = `entry_price * (1 + 0.06)` = 6%上昇
- **問題**: idx日目の終値を見てから利確判定

**リアルトレードの実態**:
```
entry_price = 1000円
take_profit_price = 1000 * (1 + 0.06) = 1060円

現在のロジック（誤り）:
- idx日目の終値 = 1065円
- 1065 >= 1060 → 利確発動
- イグジット価格 = 1065円（当日終値）

リアルトレードの実態:
- idx日目の値動き: Open=1055 → High=1070 → Low=1050 → Close=1065
- High=1070 > 1060 → 利確発動（当日中）
- イグジット価格 ≒ 1060円（利確価格 - スリッページ）

差額: 5円（0.5%）→ 現在のロジックは利確が甘すぎる（楽観的）
```

**EXIT_INVESTIGATION_REPORT.mdとの比較**:
- VWAP_Breakout.pyと同じ問題構造
- 当日終値で利確判定
- リアルトレードでは当日高値で判定すべき

**根拠**: strategies/pairs_trading_strategy.py Lines 245, 252-254

---

### [C10] スプレッド回帰判定の確認 ✅（ペアトレーディング特有）

**確認箇所**: Lines 217-219, 257-262

**実際のコード**:
```python
# Line 217-219
spread_zscore_val = self.data['Spread_ZScore'].iloc[idx]
if isinstance(spread_zscore_val, pd.Series):
    spread_zscore_val = spread_zscore_val.values[0]

# Lines 257-262
# スプレッド回帰チェック（メイン エグジット条件）
if not pd.isna(spread_zscore_val):
    exit_threshold = self.params["exit_threshold"]
    
    # スプレッドが正常範囲に戻った場合
    if abs(spread_zscore_val) <= exit_threshold:
        return 1  # 回帰完了でエグジット
```

**ルックアヘッドバイアスの問題**:
- `spread_zscore_val` = idx日目のSpread_ZScore
- Spread_ZScore = (Spread - Spread_MA) / Spread_Std
- **問題**: idx日目のスプレッドを見てから回帰判定

**Spread_ZScoreの計算確認が必要**:
- initialize_strategy()でSpread_ZScoreを計算（Lines 70-120付近）
- `.shift(1)`適用の有無を確認する必要あり

**ペアトレーディング特有の問題**:
- スプレッド回帰判定はペアトレーディングのメインイグジット条件
- リアルトレードでは当日スプレッドを見てから判断することは不可能
- **翌日のスプレッドで判定**、または**前日スプレッド（shift(1)）で判定**が必要

**根拠**: strategies/pairs_trading_strategy.py Lines 217-219, 257-262

---

### [C11] current_priceが当日終値を使用しているか確認 ✅

**確認箇所**: Line 213

**実際のコード**:
```python
current_price_val = self.data[self.price_column].iloc[idx]
```

**詳細**:
- `self.price_column` = "Adj Close"（Line 28）
- **idx日目の終値（Adj Close）を使用** → ❌ルックアヘッドバイアス確定

**根拠**: strategies/pairs_trading_strategy.py Line 213

---

### [C12] spread_zscoreのshift(1)適用確認 ✅

**確認箇所**: initialize_strategy()メソッド（Lines 67-120）

**実際のコード**:
```python
# Lines 80-81
# スプレッド（短期MA - 長期MA）の計算
self.data['Spread'] = self.data['SMA_Short'] - self.data['SMA_Long']

# Lines 83-94
# Z-Scoreの計算（標準化されたスプレッド）
self.data['Spread_ZScore'] = (
    (self.data['Spread'] - self.data['Spread_MA']) / self.data['Spread_Std']
)
```

**確認結果**: **Spread_ZScoreに`.shift(1)`が適用されていない** ❌

**詳細**:
- Spread_ZScoreは当日のスプレッドから計算
- `.shift(1)`適用なし → 当日データを使用
- **ルックアヘッドバイアスあり**

**問題の影響**:
1. エントリー判定（generate_entry_signal）で当日Spread_ZScoreを使用
2. イグジット判定（generate_exit_signal）で当日Spread_ZScoreを使用
3. **idx日目のSpread_ZScoreを見てからidx日目の取引を決定** → リアルトレードで不可能

**根拠**: strategies/pairs_trading_strategy.py Lines 70-94

---

### [C13] インジケーターのshift(1)適用確認 ✅

**確認箇所**: initialize_strategy()メソッド（Lines 67-120）

**確認結果**: **shift(1)適用なし** ❌

**インジケーター一覧**:

| インジケーター | shift(1)適用 | 使用箇所 | 問題 |
|---------------|-------------|---------|------|
| SMA_Short | ❌ | エントリー判定 | 当日MA使用 |
| SMA_Long | ❌ | エントリー判定 | 当日MA使用 |
| Spread | ❌ | Spread_ZScore計算 | 当日スプレッド使用 |
| Spread_MA | ❌ | Spread_ZScore計算 | 当日MA使用 |
| Spread_Std | ❌ | Spread_ZScore計算 | 当日Std使用 |
| Spread_ZScore | ❌ | エントリー/イグジット判定 | **当日ZScore使用** |
| Volume_MA | ❌ | エントリー判定 | 当日MA使用 |
| Volatility | ❌ | エントリー判定 | 当日Volatility使用 |
| MA_Correlation | ❌ | エントリー判定 | 当日相関使用 |

**重要**: **全インジケーターがshift(1)未適用** → **全てルックアヘッドバイアスあり**

**ルックアヘッドバイアス3原則違反**:
1. **前日データで判断**: ✅ 違反（当日データ使用）
2. **翌日始値でエントリー**: ✅ Phase 1修正完了（Lines 298-311）
3. **取引コスト考慮**: ✅ Phase 2修正完了（Lines 307-310）

**根拠**: strategies/pairs_trading_strategy.py Lines 70-120

---

### [C14] EXIT_INVESTIGATION_REPORT.mdとの比較 ✅

**共通点**:
1. `current_price_val = self.data[self.price_column].iloc[idx]` - 当日終値使用
2. ストップロス判定が当日終値ベース
3. 利益確定判定が当日終値ベース

**相違点**:
1. **スプレッド回帰判定あり**（pairs_trading_strategy.py特有）
2. **全インジケーターがshift(1)未適用**（pairs_trading_strategy.pyのみ）
3. **トレーリングストップなし**（pairs_trading_strategy.py）
4. **当日高値/安値未使用**（pairs_trading_strategy.py）

**問題構造**:
- VWAP_Breakout.py、support_resistance_contrarian_strategy.pyより**深刻**
- イグジット価格だけでなく、**エントリー判定もルックアヘッドバイアスあり**
- **全インジケーターの修正が必要**（Spread_ZScore、Volume_MA、Volatility、MA_Correlation）

**根拠**: EXIT_INVESTIGATION_REPORT.md比較 + 本調査結果

---

### [C15] 修正パターンの適用可否判断 ✅

**VWAP_Breakout.py / support_resistance_contrarian_strategy.py Phase 1b修正パターン**:
```python
# Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
current_price = self.data['Open'].iloc[idx + 1]
```

**適用可否判断**: **適用可能だが、追加修正が必要** ⚠️

**理由**:
1. 同じ問題構造（当日終値使用） → Phase 1b修正適用可能
2. ループ範囲は`range(len(result_data) - 1)`（Line 278）→ idx+1アクセス安全
3. 当日高値/安値を使用していないため、Phase 1b修正は単純
4. **ただし、インジケーター（Spread_ZScore等）のshift(1)適用も必要**

**修正優先順位**:
| 優先度 | 修正内容 | 箇所 | 理由 |
|--------|---------|------|------|
| 最高 | インジケーターshift(1)適用 | Lines 70-120（initialize_strategy） | エントリー/イグジット両方に影響 |
| 高 | イグジット価格を翌日始値に変更 | Line 213（generate_exit_signal） | イグジット価格ルックアヘッドバイアス |

**複雑度**: ★★★☆☆（中）
- Phase 1b修正（イグジット価格）: ★★☆☆☆（簡単）
- インジケーターshift(1)適用: ★★★☆☆（中、9個のインジケーター修正）

**根拠**: VWAP_EXIT_INVESTIGATION.md + support_resistance_contrarian_EXIT_INVESTIGATION.md + 本調査結果

---

## 原因分析

### 根本原因

**ルックアヘッドバイアス（イグジット編 + インジケーター編）の根本原因**:
1. `current_price_val = self.data[self.price_column].iloc[idx]` - **idx日目の終値を使用**
2. **全インジケーターがshift(1)未適用** - 当日データを使用
3. 全イグジット条件が`current_price_val`と`spread_zscore_val`ベースで判定
4. idx日目の終値とスプレッドを見てからidx日目の終値で売ることは不可能

### 影響するイグジット条件

| イグジット条件 | 影響度 | 理由 |
|---------------|--------|------|
| ストップロス | 高 | 当日終値でストップロス判定 |
| 利益確定 | 高 | 当日終値で利確判定 |
| スプレッド回帰 | **最高** | 当日Spread_ZScoreで回帰判定（ペアトレーディングのメイン条件） |

### 影響するエントリー条件（追加発見）

| エントリー条件 | 影響度 | 理由 |
|---------------|--------|------|
| Spread_ZScore | **最高** | 当日ZScoreでエントリー判定 |
| Volume_MA | 中 | 当日ボリューム平均使用 |
| Volatility | 中 | 当日ボラティリティ使用 |
| MA_Correlation | 低 | 当日相関使用 |

### EXIT_INVESTIGATION_REPORT.mdとの共通点

**共通の問題構造**:
1. `current_price`を当日終値から取得
2. イグジット判定が当日価格ベース
3. リアルトレードでは実現不可能

**pairs_trading_strategy.pyの深刻度**:
- VWAP_Breakout.pyより**深刻**
- support_resistance_contrarian_strategy.pyより**深刻**
- **全インジケーターがshift(1)未適用**（エントリー/イグジット両方に影響）
- 修正箇所: Line 213（イグジット価格）+ Lines 70-120（インジケーター9個）

---

## 影響範囲

### 修正対象

**直接修正（優先度：最高）**:
- Lines 70-94: インジケーターにshift(1)適用（9箇所）
  - Line 71-73: SMA_Short（`.shift(1)`追加）
  - Line 75-77: SMA_Long（`.shift(1)`追加）
  - Line 80: Spread（`.shift(1)`追加）
  - Line 83-86: Spread_MA（`.shift(1)`追加）
  - Line 88-90: Spread_Std（`.shift(1)`追加）
  - Line 93-95: Spread_ZScore（`.shift(1)`追加）
  - Line 98-100: Volume_MA（`.shift(1)`追加）
  - Line 104-106: Volatility（`.shift(1)`追加）
  - Line 111-114: MA_Correlation（`.shift(1)`追加）

**直接修正（優先度：高）**:
- Line 213: `current_price_val`定義を翌日始値に変更

**間接影響**:
- ストップロスロジック（Lines 247-249）
- 利益確定ロジック（Lines 252-254）
- スプレッド回帰ロジック（Lines 257-262）
- エントリーロジック（generate_entry_signal全体）

### 境界条件の安全性

**ループ範囲**: pairs_trading_strategy.py Line 278
```python
for i in range(len(result_data) - 1):
```

**安全性確認**:
- ループ範囲は`len(result_data) - 1`まで
- `idx + 1`でアクセスしても境界条件エラーは発生しない
- **結論**: `self.data['Open'].iloc[idx + 1]`は安全

---

## 改善提案

### Phase 0修正案（最優先）: インジケーターにshift(1)適用

**修正理由**: エントリー/イグジット両方に影響、最も深刻なルックアヘッドバイアス

**修正箇所: initialize_strategy()メソッド（Lines 70-120）**

**修正前**:
```python
def initialize_strategy(self):
    """戦略初期化処理"""
    super().initialize_strategy()
    
    # 短期・長期移動平均の計算
    self.data['SMA_Short'] = self.data[self.price_column].rolling(
        window=self.params["short_ma_period"]
    ).mean()
    
    self.data['SMA_Long'] = self.data[self.price_column].rolling(
        window=self.params["long_ma_period"]
    ).mean()
    
    # スプレッド（短期MA - 長期MA）の計算
    self.data['Spread'] = self.data['SMA_Short'] - self.data['SMA_Long']
    
    # スプレッドの移動平均と標準偏差
    self.data['Spread_MA'] = self.data['Spread'].rolling(
        window=self.params["spread_period"]
    ).mean()
    
    self.data['Spread_Std'] = self.data['Spread'].rolling(
        window=self.params["spread_period"]
    ).std()
    
    # Z-Scoreの計算（標準化されたスプレッド）
    self.data['Spread_ZScore'] = (
        (self.data['Spread'] - self.data['Spread_MA']) / self.data['Spread_Std']
    )
    
    # ボリュームフィルター
    if self.params["volume_filter"]:
        self.data['Volume_MA'] = self.data['Volume'].rolling(
            window=self.params["spread_period"]
        ).mean()
    
    # ボラティリティフィルター
    if self.params["volatility_filter"]:
        returns = self.data[self.price_column].pct_change()
        self.data['Volatility'] = returns.rolling(
            window=self.params["volatility_period"]
        ).std()
        
    # 移動平均間の相関（ローリング相関）
    if len(self.data) >= self.params["cointegration_lookback"]:
        correlation_window = min(self.params["cointegration_lookback"], len(self.data))
        self.data['MA_Correlation'] = self.data['SMA_Short'].rolling(
            window=correlation_window
        ).corr(self.data['SMA_Long'])
```

**修正後**:
```python
def initialize_strategy(self):
    """戦略初期化処理"""
    super().initialize_strategy()
    
    # ルックアヘッドバイアス修正: 短期・長期移動平均の計算（shift(1)適用）
    # 理由: i日の終値を見てからi日の取引を決定することは不可能
    # リアルトレードでは前日（i-1日目）までのデータで判断
    self.data['SMA_Short'] = self.data[self.price_column].rolling(
        window=self.params["short_ma_period"]
    ).mean().shift(1)
    
    self.data['SMA_Long'] = self.data[self.price_column].rolling(
        window=self.params["long_ma_period"]
    ).mean().shift(1)
    
    # スプレッド（短期MA - 長期MA）の計算
    self.data['Spread'] = self.data['SMA_Short'] - self.data['SMA_Long']
    
    # スプレッドの移動平均と標準偏差
    self.data['Spread_MA'] = self.data['Spread'].rolling(
        window=self.params["spread_period"]
    ).mean().shift(1)
    
    self.data['Spread_Std'] = self.data['Spread'].rolling(
        window=self.params["spread_period"]
    ).std().shift(1)
    
    # Z-Scoreの計算（標準化されたスプレッド）
    # 注意: Spreadは既にshift(1)適用済みのSMA_Short/Longから計算されるため、
    #       Spread自体はshift(1)不要。Spread_MA/Spread_Stdにshift(1)適用済み
    self.data['Spread_ZScore'] = (
        (self.data['Spread'] - self.data['Spread_MA']) / self.data['Spread_Std']
    )
    
    # ルックアヘッドバイアス修正: ボリュームフィルター（shift(1)適用）
    if self.params["volume_filter"]:
        self.data['Volume_MA'] = self.data['Volume'].rolling(
            window=self.params["spread_period"]
        ).mean().shift(1)
    
    # ルックアヘッドバイアス修正: ボラティリティフィルター（shift(1)適用）
    if self.params["volatility_filter"]:
        returns = self.data[self.price_column].pct_change()
        self.data['Volatility'] = returns.rolling(
            window=self.params["volatility_period"]
        ).std().shift(1)
        
    # ルックアヘッドバイアス修正: 移動平均間の相関（shift(1)適用）
    if len(self.data) >= self.params["cointegration_lookback"]:
        correlation_window = min(self.params["cointegration_lookback"], len(self.data))
        self.data['MA_Correlation'] = self.data['SMA_Short'].rolling(
            window=correlation_window
        ).corr(self.data['SMA_Long']).shift(1)
```

**変更内容**:
- Lines 71-73: `SMA_Short`に`.shift(1)`追加
- Lines 75-77: `SMA_Long`に`.shift(1)`追加
- Lines 83-86: `Spread_MA`に`.shift(1)`追加
- Lines 88-90: `Spread_Std`に`.shift(1)`追加
- Lines 98-100: `Volume_MA`に`.shift(1)`追加
- Lines 104-106: `Volatility`に`.shift(1)`追加
- Lines 111-114: `MA_Correlation`に`.shift(1)`追加
- コメント追加: Phase 0修正理由を各インジケーターに明記

---

### Phase 1b修正案: イグジット価格を翌日始値に変更

**修正箇所: current_price定義（Line 213）**

**修正前**:
```python
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
    """エグジットシグナル生成"""
    if position_size <= 0:
        return 0
        
    # スカラー値として取得
    current_price_val = self.data[self.price_column].iloc[idx]
    if isinstance(current_price_val, pd.Series):
        current_price_val = current_price_val.values[0]
```

**修正後**:
```python
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
    """エグジットシグナル生成"""
    if position_size <= 0:
        return 0
    
    # Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
    # 理由: idx日目の終値を見てからidx日目の終値で売ることは不可能
    # リアルトレードでは翌日（idx+1日目）の始値でイグジット
    current_price_val = self.data['Open'].iloc[idx + 1]
    if isinstance(current_price_val, pd.Series):
        current_price_val = current_price_val.values[0]
```

**変更内容**:
- `self.data[self.price_column].iloc[idx]` → `self.data['Open'].iloc[idx + 1]`
- 当日終値（`Adj Close`） → 翌日始値（`Open`）
- コメント追加: Phase 1b修正理由を3行で明記

---

### 修正の優先順位

| 修正箇所 | 優先度 | 理由 |
|---------|--------|------|
| Phase 0: インジケーターshift(1)適用 | **最高** | エントリー/イグジット両方に影響 |
| Phase 1b: current_price定義 | 高 | イグジット価格ルックアヘッドバイアス |

---

## セルフチェック

### a) 見落としチェック ✅
- [x] pairs_trading_strategy.py全体確認（Lines 1-442）
- [x] generate_exit_signal()の全ロジック確認（Lines 207-264）
- [x] backtest()メソッドの確認（Lines 267-330）
- [x] current_price取得箇所の確認（Line 213）
- [x] 各イグジット条件の確認（Lines 240-262）
- [x] Phase 1修正の確認（Lines 298-311）
- [x] インジケーターshift(1)確認（Lines 70-120）→**未適用発見**
- [x] ループ範囲の確認（Line 278）

### b) 思い込みチェック ✅
- ❌ 「Phase 1修正でイグジットも修正されているはず」 → ✅ 実際はLine 213未修正
- ❌ 「インジケーターはshift(1)適用済みのはず」 → ✅ 実際は全てshift(1)未適用
- ✅ Line 213で`current_price_val = self.data[self.price_column].iloc[idx]`を確認
- ✅ 当日終値を使用していることを確認
- ✅ 全インジケーターがshift(1)未適用であることを確認

### c) 矛盾チェック ✅
- Phase 1完了（エントリー価格） vs Phase 1b未完了（イグジット価格） → **整合** → 予想通り
- Phase 1完了 vs インジケーターshift(1)未適用 → **整合** → エントリー価格のみ修正、インジケーターは別途対応
- VWAP_Breakout.pyより深刻 vs 全インジケーターshift(1)未適用 → **整合** → 修正箇所多数

---

## 次のステップ

### 推奨する修正手順

1. **Phase 0修正実施（最優先）**
   - pairs_trading_strategy.py Lines 70-120: インジケーターにshift(1)適用（9箇所）
   - SMA_Short、SMA_Long、Spread_MA、Spread_Std、Volume_MA、Volatility、MA_Correlationに`.shift(1)`追加

2. **Phase 1b修正実施（次優先）**
   - pairs_trading_strategy.py Line 213: current_price定義を翌日始値に変更

3. **検証テスト作成・実行**
   - 修正後のバックテスト実行
   - イグジット価格が翌日始値に変更されていることを確認
   - インジケーターがshift(1)適用されていることを確認
   - 取引件数・損益の変化を確認

4. **ドキュメント更新**
   - 本調査報告書を更新（修正完了記録）
   - INVESTIGATION_REPORT.mdに参照追加（イグジット編）

---

**調査完了日**: 2025-12-23  
**調査者**: GitHub Copilot  
**ステータス**: Phase 0・Phase 1b修正完了✅  
**次のアクション**: 検証テスト作成・実行（Phase 0/Phase 1b修正の効果確認）

---

## Phase 0修正完了記録（2025-12-23）

### 修正内容

**修正箇所**: strategies/pairs_trading_strategy.py Lines 70-122（initialize_strategy()メソッド）

**修正内容**: 全7インジケーターに`.shift(1)`適用

| インジケーター | 修正箇所 | 修正内容 | 理由 |
|---------------|---------|---------|------|
| SMA_Short | Line 75 | `.mean().shift(1)` | 前日MA使用 |
| SMA_Long | Line 80 | `.mean().shift(1)` | 前日MA使用 |
| Spread_MA | Line 87 | `.mean().shift(1)` | 前日MA使用 |
| Spread_Std | Line 91 | `.std().shift(1)` | 前日Std使用 |
| Volume_MA | Line 105 | `.mean().shift(1)` | 前日MA使用 |
| Volatility | Line 113 | `.std().shift(1)` | 前日Volatility使用 |
| MA_Correlation | Line 122 | `.corr(...).shift(1)` | 前日相関使用 |

**修正理由コメント追加**: 6箇所（Lines 70-72, 85, 94-96, 101, 108, 118）

**検証**: 修正後のコード構文確認（Lines 67-125）→ エラーなし✅

**ルックアヘッドバイアス3原則対応**:
1. ✅ 前日データで判断（shift(1)適用）- **Phase 0完了**
2. ✅ 翌日始値でエントリー（Phase 1完了）
3. ✅ 取引コスト考慮（Phase 2完了）

---

## Phase 1b修正完了記録（2025-12-23）

### 修正内容

**修正箇所**: strategies/pairs_trading_strategy.py Line 219（generate_exit_signal()メソッド）

**修正前**:
```python
current_price_val = self.data[self.price_column].iloc[idx]  # idx日目の終値（Adj Close）
```

**修正後**:
```python
current_price_val = self.data['Open'].iloc[idx + 1]  # 翌日始値
```

**修正理由コメント追加**: 3行（Lines 216-218）
- "Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）"
- "理由: idx日目の終値を見てからidx日目の終値で売ることは不可能"
- "リアルトレードでは翌日（idx+1日目）の始値でイグジット"

**検証**: 修正後のコード構文確認（Lines 211-270）→ エラーなし✅

**ループ範囲安全確認**: `range(len(result_data) - 1)`（Line 282）→ idx+1アクセス安全✅

**影響するイグジット条件**:
- ストップロス判定（Lines 253-255）
- 利益確定判定（Lines 258-260）
- スプレッド回帰判定（Lines 263-268）

**ルックアヘッドバイアス解消**:
- エントリー価格: 翌日始値 + スリッページ（Phase 1完了）
- イグジット価格: 翌日始値（**Phase 1b完了**）

---

## 残課題（優先度順）

### 1. 検証テスト作成・実行（次のタスク）

**目的**: Phase 0/Phase 1b修正の効果確認

**テスト内容**:
- tests/temp/test_20251223_pairs_trading_exit_price_check.py作成
- Phase 0修正後のバックテスト実行
- インジケーターがshift(1)適用されていることを確認
- Phase 1b修正後のバックテスト実行
- イグジット価格が翌日始値に変更されていることを確認
- 取引件数・損益の変化を確認

### 2. ChainedAssignmentWarning対応（推奨）

**箇所**: strategies/pairs_trading_strategy.py Lines 300, 323, 326

**内容**: `.iloc[]`書き込みを`.loc[]`または`.at[]`に変更

**影響度**: 低（警告のみ、動作には影響しない）

**優先度**: 低（pandas 3.0対応として推奨）

---
