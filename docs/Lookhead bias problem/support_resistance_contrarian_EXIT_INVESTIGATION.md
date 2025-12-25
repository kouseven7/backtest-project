# support_resistance_contrarian_strategy.py イグジット問題 調査報告書

**作成日**: 2025-12-23  
**調査者**: GitHub Copilot  
**調査対象**: strategies/support_resistance_contrarian_strategy.py (generate_exit_signal)  
**調査ステータス**: ✅ **調査完了（Phase 1b修正提案準備完了）**  
**関連ドキュメント**: 
- [EXIT_INVESTIGATION_REPORT.md](EXIT_INVESTIGATION_REPORT.md) - イグジット問題の一般調査
- [VWAP_EXIT_INVESTIGATION.md](VWAP_EXIT_INVESTIGATION.md) - VWAP Phase 1b修正完了記録

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

support_resistance_contrarian_strategy.pyのPhase 1（エントリー価格修正）が完了した後、**イグジット価格**についても同様のルックアヘッドバイアス問題が存在するか調査する。

### 調査の背景

**Phase 1完了状況**:
- ✅ エントリー価格修正完了（Lines 339-350）
- ✅ 翌日始値 + スリッページ適用
- ✅ ループ範囲調整完了（`range(len(result_data) - 1)`）

**Phase 1b（イグジット）調査対象**:
- ❓ イグジット価格の取得方法
- ❓ ストップロス・利益確定・抵抗線判定のルックアヘッドバイアス有無
- ❓ VWAP_Breakout.py Phase 1b修正パターンの適用可否

---

## 確認項目チェックリスト

### Phase 1: 基本構造確認（優先度：最高）
- [C1] support_resistance_contrarian_strategy.py全体構造確認 ✅
- [C2] generate_exit_signal()メソッドの存在確認 ✅
- [C3] BaseStrategy継承の確認 ✅
- [C4] backtest()メソッドの実装確認（独自実装 or 継承） ✅

### Phase 2: イグジット価格問題特定（優先度：高）
- [C5] current_price取得箇所の特定 ✅
- [C6] entry_price取得方法の確認 ✅
- [C7] 各イグジット条件の実装確認 ✅
- [C8] ストップロス判定の確認 ✅
- [C9] 利益確定判定の確認 ✅
- [C10] サポート/レジスタンスラインのイグジット判定確認 ✅

### Phase 3: ルックアヘッドバイアス確認（優先度：高）
- [C11] current_priceが当日終値を使用しているか確認 ✅
- [C12] 当日高値/安値の使用確認 ✅
- [C13] インジケーターのshift(1)適用確認 ✅

### Phase 4: 修正方針検討（優先度：中）
- [C14] EXIT_INVESTIGATION_REPORT.mdとの比較 ✅
- [C15] VWAP_Breakout.py Phase 1b修正パターンの適用可否判断 ✅

---

## 調査結果

### [C1] 全体構造確認 ✅

**確認結果**:
- ファイル名: support_resistance_contrarian_strategy.py
- クラス名: SupportResistanceContrarianStrategy
- 総行数: 461行
- BaseStrategy継承確認✅（Line 23）

**根拠**: strategies/support_resistance_contrarian_strategy.py Lines 1-100

---

### [C2] generate_exit_signal()メソッドの存在確認 ✅

**確認箇所**: Line 259

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

**確認箇所**: Line 23

**実際のコード**:
```python
class SupportResistanceContrarianStrategy(BaseStrategy):
```

**根拠**: support_resistance_contrarian_strategy.py Line 23

---

### [C4] backtest()メソッドの実装確認 ✅

**確認箇所**: Line 307

**実際のコード**:
```python
def backtest(self, trading_start_date=None, trading_end_date=None) -> pd.DataFrame:
```

**結果**: **独自実装あり**（BaseStrategy.backtest()をオーバーライド）

**根拠**: grep_search実行結果

---

### [C5] current_price取得箇所の特定 ✅

**確認箇所**: Line 264

**実際のコード**:
```python
current_price = self.data[self.price_column].iloc[idx]
```

**使用している価格カラム**:
- `self.price_column` = "Adj Close"（デフォルト、Line 35で設定）
- Line 264でidx日目の`Adj Close`を取得

**ルックアヘッドバイアスの有無**:
- ✅ **ルックアヘッドバイアスあり** - idx日目の終値を使用
- [ ] **ルックアヘッドバイアスなし** - 翌日始値を使用
- [ ] **不明** - 追加調査が必要

**証拠**:
```python
# Line 264
current_price = self.data[self.price_column].iloc[idx]  # idx日目の終値（Adj Close）
```

**問題の構造**:
1. idx日目の`generate_exit_signal(idx)`でイグジット判断
2. `current_price = self.data[self.price_column].iloc[idx]` - idx日目の終値を取得
3. idx日目の終値でストップロス・利益確定・抵抗線判定
4. **リアルトレードでは、idx日目の終値を見てからidx日目の終値で売ることは不可能**

**根拠**: support_resistance_contrarian_strategy.py Line 264

---

### [C6] entry_price取得方法の確認 ✅

**確認箇所**: Lines 267-274

**実際のコード**:
```python
# エントリー価格取得
entry_price = self.entry_prices.get(idx, None)
if entry_price is None:
    # 過去のエントリーを逆算
    for i in range(max(0, idx - self.params["max_hold_days"]), idx):
        if i in self.entry_prices:
            entry_price = self.entry_prices[i]
            break
```

**特徴**:
- `self.entry_prices`辞書から取得
- エントリー価格は独自backtest()で記録（Lines 339-350）
- Phase 1修正完了: 翌日始値 + スリッページ適用

**検証**:
```python
# Lines 339-350（Phase 1修正完了）
next_day_open_val = result_data['Open'].iloc[i + 1]
slippage = self.params.get("slippage", 0.001)
transaction_cost = self.params.get("transaction_cost", 0.0)
entry_price = next_day_open_val * (1 + slippage + transaction_cost)
self.entry_prices[i] = entry_price
```

**結論**: entry_priceは問題なし（Phase 1修正完了済み）

**根拠**: support_resistance_contrarian_strategy.py Lines 267-274, 339-350

---

### [C7] 各イグジット条件の実装確認 ✅

**確認箇所**: Lines 287-304

**イグジット条件一覧**:

| イグジット条件 | 実装箇所 | 使用価格 | 問題 |
|---------------|---------|---------|------|
| ストップロス | Lines 289-290 | `pnl_pct <= -stop_loss_pct`（current_price使用） | ❌ 当日終値使用 |
| 利益確定 | Lines 293-294 | `pnl_pct >= take_profit_pct`（current_price使用） | ❌ 当日終値使用 |
| 抵抗線到達 | Lines 297-300 | `current_price >= resistance * 0.995` | ❌ 当日終値使用 |
| 最大保有日数 | Lines 302-303 | コメントのみ（未実装） | - |

**根拠**: support_resistance_contrarian_strategy.py Lines 287-304

---

### [C8] ストップロス判定の確認 ✅

**確認箇所**: Lines 286, 289-290

**実際のコード**:
```python
# 損益計算
pnl_pct = (current_price - entry_price) / entry_price

# ストップロス
if pnl_pct <= -self.params["stop_loss_pct"]:
    return 1
```

**ルックアヘッドバイアスの問題**:
- `current_price` = idx日目の終値（Line 264）
- ストップロス価格 = `entry_price * (1 - 0.02)` = 2%下落
- **問題**: idx日目の終値を見てからストップロス判定

**リアルトレードの実態**:
```
entry_price = 1000円
stop_loss_price = 1000 * (1 - 0.02) = 980円

現在のロジック（誤り）:
- idx日目の終値 = 975円
- 975 <= 980 → ストップロス発動
- イグジット価格 = 975円（当日終値）

リアルトレードの実態:
- idx日目の値動き: Open=995 → High=1005 → Low=970 → Close=975
- Low=970 < 980 → ストップロス発動（当日中）
- イグジット価格 ≒ 980円（ストップロス価格 - スリッページ）

差額: 5円（0.5%）→ 現在のロジックはストップロスが厳しすぎる
```

**EXIT_INVESTIGATION_REPORT.mdとの比較**:
- VWAP_Breakout.pyと同じ問題構造
- 当日終値でストップロス判定
- リアルトレードでは当日安値で判定すべき

**根拠**: support_resistance_contrarian_strategy.py Lines 286, 289-290

---

### [C9] 利益確定判定の確認 ✅

**確認箇所**: Lines 286, 293-294

**実際のコード**:
```python
# 損益計算
pnl_pct = (current_price - entry_price) / entry_price

# 利益確定
if pnl_pct >= self.params["take_profit_pct"]:
    return 1
```

**ルックアヘッドバイアスの問題**:
- `current_price` = idx日目の終値（Line 264）
- 利益確定価格 = `entry_price * (1 + 0.04)` = 4%上昇
- **問題**: idx日目の終値を見てから利確判定

**リアルトレードの実態**:
```
entry_price = 1000円
take_profit_price = 1000 * (1 + 0.04) = 1040円

現在のロジック（誤り）:
- idx日目の終値 = 1045円
- 1045 >= 1040 → 利確発動
- イグジット価格 = 1045円（当日終値）

リアルトレードの実態:
- idx日目の値動き: Open=1035 → High=1050 → Low=1030 → Close=1045
- High=1050 > 1040 → 利確発動（当日中）
- イグジット価格 ≒ 1040円（利確価格 - スリッページ）

差額: 5円（0.5%）→ 現在のロジックは利確が甘すぎる（楽観的）
```

**EXIT_INVESTIGATION_REPORT.mdとの比較**:
- VWAP_Breakout.pyと同じ問題構造
- 当日終値で利確判定
- リアルトレードでは当日高値で判定すべき

**根拠**: support_resistance_contrarian_strategy.py Lines 286, 293-294

---

### [C10] サポート/レジスタンスラインのイグジット判定確認 ✅

**確認箇所**: Lines 297-300

**実際のコード**:
```python
# 抵抗線到達での利益確定
for resistance in self.resistance_levels:
    if current_price >= resistance * (1 - self.params["proximity_threshold"]):
        if pnl_pct > 0:  # 利益が出ている場合のみ
            return 1
```

**ルックアヘッドバイアスの問題**:
- `current_price` = idx日目の終値（Line 264）
- 抵抗線判定 = `resistance * (1 - 0.005)` = 抵抗線の99.5%
- **問題**: idx日目の終値を見てから抵抗線判定

**特有の問題**:
- `resistance * 0.995` = 抵抗線の99.5%で判定
- リアルトレードでは当日高値が抵抗線に到達したら即座にイグジット
- 逆張り戦略の特性上、抵抗線到達での利確は重要

**リアルトレードの実態**:
```
resistance = 1050円
proximity_threshold = 0.005（0.5%）
resistance_target = 1050 * 0.995 = 1044.75円

現在のロジック（誤り）:
- idx日目の終値 = 1046円
- 1046 >= 1044.75 → 抵抗線到達、イグジット
- イグジット価格 = 1046円（当日終値）

リアルトレードの実態:
- idx日目の値動き: Open=1040 → High=1052 → Low=1038 → Close=1046
- High=1052 > 1044.75 → 抵抗線到達、イグジット
- イグジット価格 ≒ 1044.75円（抵抗線ターゲット - スリッページ）

差額: 1.25円（0.1%）→ 現在のロジックは抵抗線判定が甘すぎる
```

**根拠**: support_resistance_contrarian_strategy.py Lines 297-300

---

### [C11] current_priceが当日終値を使用しているか確認 ✅

**確認箇所**: Line 264

**実際のコード**:
```python
current_price = self.data[self.price_column].iloc[idx]
```

**詳細**:
- `self.price_column` = "Adj Close"（Line 35）
- **idx日目の終値（Adj Close）を使用** → ❌ルックアヘッドバイアス確定

**根拠**: support_resistance_contrarian_strategy.py Line 264

---

### [C12] 当日高値/安値の使用確認 ✅

**確認結果**: 当日高値/安値は**使用していない**

**詳細**:
- generate_exit_signal()では`current_price`（当日終値）のみ使用
- ストップロス・利益確定・抵抗線判定、全て当日終値ベース
- High/Lowカラムは使用されていない

**EXIT_INVESTIGATION_REPORT.mdとの比較**:
- VWAP_Breakout.pyはトレーリングストップで`self.data['High'].iloc[entry_idx:idx+1].max()`を使用
- support_resistance_contrarian_strategy.pyは当日高値/安値を使用していない
- **より単純な問題構造**

**根拠**: support_resistance_contrarian_strategy.py Lines 259-304

---

### [C13] インジケーターのshift(1)適用確認 ✅

**確認箇所**: Lines 78-85

**実際のコード**:
```python
# ルックアヘッドバイアス修正: RSI計算（確認シグナル用）
if self.params["rsi_confirmation"]:
    self.data['RSI'] = self._calculate_rsi().shift(1)

# ルックアヘッドバイアス修正: ボリューム移動平均
self.data['Volume_MA'] = self.data['Volume'].rolling(
    window=self.params["lookback_period"]
).mean().shift(1)
```

**確認結果**:
- RSIは`.shift(1)`適用済み ✅
- ボリューム移動平均も`.shift(1)`適用済み ✅
- **インジケーターは問題なし**

**根拠**: support_resistance_contrarian_strategy.py Lines 78-85

---

### [C14] EXIT_INVESTIGATION_REPORT.mdとの比較 ✅

**共通点**:
1. `current_price = self.data[self.price_column].iloc[idx]` - 当日終値使用
2. ストップロス判定が当日終値ベース
3. 利益確定判定が当日終値ベース
4. インジケーターは`.shift(1)`適用済み

**相違点**:
1. **トレーリングストップなし**（support_resistance_contrarian_strategy.py）
2. **当日高値/安値未使用**（support_resistance_contrarian_strategy.py）
3. **抵抗線判定あり**（support_resistance_contrarian_strategy.py独自）

**問題構造**:
- VWAP_Breakout.pyより**単純**
- 全イグジット条件が`current_price`（当日終値）のみに依存
- 修正は**1箇所**（Line 264のcurrent_price定義）のみで済む可能性

**根拠**: EXIT_INVESTIGATION_REPORT.md比較 + 本調査結果

---

### [C15] VWAP_Breakout.py Phase 1b修正パターンの適用可否判断 ✅

**VWAP_Breakout.py Phase 1b修正パターン**:
```python
# Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
# 理由: idx日目の終値を見てからidx日目の終値で売ることは不可能
# リアルトレードでは翌日（idx+1日目）の始値でイグジット
current_price = self.data['Open'].iloc[idx + 1]
```

**適用可否判断**: **適用可能** ✅

**理由**:
1. 同じ問題構造（当日終値使用）
2. ループ範囲は`range(len(result_data) - 1)`（Line 316）→ idx+1アクセス安全
3. 当日高値/安値を使用していないため、修正は単純
4. Phase 1修正（エントリー価格）は完了済み（Lines 339-350）

**修正箇所**:
- Line 264: `current_price = self.data[self.price_column].iloc[idx]` → `current_price = self.data['Open'].iloc[idx + 1]`

**影響範囲**:
- ストップロス判定（Lines 289-290）
- 利益確定判定（Lines 293-294）
- 抵抗線判定（Lines 297-300）

**根拠**: VWAP_EXIT_INVESTIGATION.md + 本調査結果

---

## 原因分析

### 根本原因

**ルックアヘッドバイアス（イグジット編）の根本原因**:
1. `current_price = self.data[self.price_column].iloc[idx]` - **idx日目の終値を使用**
2. 全イグジット条件が`current_price`ベースで判定
3. idx日目の終値を見てからidx日目の終値で売ることは不可能

### 影響するイグジット条件

| イグジット条件 | 影響度 | 理由 |
|---------------|--------|------|
| ストップロス | 高 | 当日終値でストップロス判定 |
| 利益確定 | 高 | 当日終値で利確判定 |
| 抵抗線到達 | 高 | 当日終値で抵抗線判定（逆張り戦略の重要判定） |

### EXIT_INVESTIGATION_REPORT.mdとの共通点

**共通の問題構造**:
1. `current_price`を当日終値から取得
2. イグジット判定が当日価格ベース
3. リアルトレードでは実現不可能

**VWAP_Breakout.pyより単純**:
- 当日高値/安値を使用していない
- トレーリングストップなし
- 修正箇所は1箇所のみ（Line 264）

---

## 影響範囲

### 修正対象

**直接修正**:
- support_resistance_contrarian_strategy.py Line 264: `current_price`定義

**間接影響**:
- ストップロスロジック（Lines 289-290）
- 利益確定ロジック（Lines 293-294）
- 抵抗線判定ロジック（Lines 297-300）

### 境界条件の安全性

**ループ範囲**: support_resistance_contrarian_strategy.py Line 316
```python
for i in range(len(result_data) - 1):
```

**安全性確認**:
- ループ範囲は`len(result_data) - 1`まで
- `idx + 1`でアクセスしても境界条件エラーは発生しない
- **結論**: `self.data['Open'].iloc[idx + 1]`は安全

---

## 改善提案

### Phase 1b修正案: イグジット価格を翌日始値に変更

**修正箇所: current_price定義（Line 264）**

**修正前**:
```python
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
    """エグジットシグナル生成"""
    if position_size <= 0:
        return 0
        
    current_price = self.data[self.price_column].iloc[idx]
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
    current_price = self.data['Open'].iloc[idx + 1]
```

**変更内容**:
- `self.data[self.price_column].iloc[idx]` → `self.data['Open'].iloc[idx + 1]`
- 当日終値（`Adj Close`） → 翌日始値（`Open`）
- コメント追加: Phase 1b修正理由を3行で明記

---

### 修正の優先順位

| 修正箇所 | 優先度 | 理由 |
|---------|--------|------|
| Line 264: current_price定義 | 最高 | 全イグジット条件に影響 |

---

## セルフチェック

### a) 見落としチェック ✅
- [x] support_resistance_contrarian_strategy.py全体確認（Lines 1-461）
- [x] generate_exit_signal()の全ロジック確認（Lines 259-304）
- [x] backtest()メソッドの確認（Lines 307-370）
- [x] current_price取得箇所の確認（Line 264）
- [x] 各イグジット条件の確認（Lines 289-300）
- [x] Phase 1修正の確認（Lines 339-350）
- [x] インジケーターshift(1)確認（Lines 78-85）
- [x] ループ範囲の確認（Line 316）

### b) 思い込みチェック ✅
- ❌ 「Phase 1修正でイグジットも修正されているはず」 → ✅ 実際はLine 264未修正
- ✅ Line 264で`current_price = self.data[self.price_column].iloc[idx]`を確認
- ✅ 当日終値を使用していることを確認
- ✅ VWAP_Breakout.py Phase 1b修正パターンが適用可能と確認

### c) 矛盾チェック ✅
- Phase 1完了（エントリー） vs Phase 1b未完了（イグジット） → **整合** → 予想通り
- ループ範囲修正済み vs イグジット価格未修正 → **整合** → エントリーのみ修正、イグジットは別途対応
- VWAP_Breakout.pyより単純 vs 修正箇所1箇所 → **整合** → 当日高値/安値未使用のため

---

## 次のステップ

### 推奨する修正手順

1. **Phase 1b修正実施**
   - support_resistance_contrarian_strategy.py Line 264: current_price定義を翌日始値に変更

2. **検証テスト作成・実行**
   - 修正後のバックテスト実行
   - イグジット価格が翌日始値に変更されていることを確認
   - 取引件数・損益の変化を確認

3. **ドキュメント更新**
   - 本調査報告書を更新（修正完了記録）
   - INVESTIGATION_REPORT.mdに参照追加（イグジット編）

---

**調査完了日**: 2025-12-23  
**調査者**: GitHub Copilot  
**ステータス**: 調査完了、修正提案準備完了  
**次のアクション**: Phase 1b修正実施（ユーザー承認待ち）
