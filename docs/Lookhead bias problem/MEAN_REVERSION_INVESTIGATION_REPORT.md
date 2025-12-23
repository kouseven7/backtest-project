# MeanReversionStrategy ルックアヘッドバイアス調査報告書

**作成日**: 2025-12-22  
**調査者**: GitHub Copilot  
**調査対象**: strategies/mean_reversion_strategy.py  
**参照ドキュメント**: docs/Lookhead bias problem/INVESTIGATION_REPORT.md  

---

## 目次

1. [確認項目チェックリスト](#確認項目チェックリスト)
2. [調査結果](#調査結果)
3. [原因分析](#原因分析)
4. [修正提案](#修正提案)
5. [セルフチェック](#セルフチェック)

---

## 確認項目チェックリスト

### 優先度: 高（エントリー価格のルックアヘッドバイアス）

- [ ] **項目1**: backtest()メソッドでのエントリー価格決定ロジックの確認
  - 対象: Line 276付近
  - 確認内容: `self.entry_prices[i] = result_data[self.price_column].iloc[i]`の使用状況
  - 期待値: 当日終値を使用している場合は問題あり

- [ ] **項目2**: generate_entry_signal()での判断タイミング
  - 対象: Lines 173-192
  - 確認内容: idx日目のデータでエントリー判断しているか
  - 期待値: 判断自体は問題なし（インジケーターがshift(1)済み）

- [ ] **項目3**: ループ範囲の確認
  - 対象: Line 256 `for i in range(len(result_data)):`
  - 確認内容: idx+1アクセスの安全性
  - 期待値: 最終日のエントリーを防ぐ必要あり

### 優先度: 中（インジケーターのshift(1)適用）

- [ ] **項目4**: SMA（移動平均）のshift(1)適用
  - 対象: Lines 70-72
  - 確認内容: `.shift(1)`の使用
  - 期待値: 適用済み

- [ ] **項目5**: ボリンジャーバンドのshift(1)適用
  - 対象: Lines 74-82
  - 確認内容: `.shift(1)`の使用
  - 期待値: 適用済み

- [ ] **項目6**: Z-scoreのshift(1)適用
  - 対象: Lines 84-92
  - 確認内容: `.shift(1)`の使用
  - 期待値: 適用済み

- [ ] **項目7**: RSIのshift(1)適用
  - 対象: Lines 94-96
  - 確認内容: `.shift(1)`の使用
  - 期待値: 適用済み

- [ ] **項目8**: ATRのshift(1)適用
  - 対象: Lines 99-101
  - 確認内容: `.shift(1)`の使用
  - 期待値: 適用済み

- [ ] **項目9**: Volume_MAのshift(1)適用
  - 対象: Lines 103-107
  - 確認内容: `.shift(1)`の使用
  - 期待値: 適用済み

### 優先度: 低（今回スルー）

- [ ] **項目10**: エグジット価格のルックアヘッドバイアス（別ファイルで対応予定）
  - 対象: generate_exit_signal()内のストップロス・利益確定ロジック
  - 今回はスルー

---

## 調査結果

### 結果1: エントリー価格のルックアヘッドバイアス ✅確認完了

#### 証拠: mean_reversion_strategy.py Line 276

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Line 276  

```python
# backtest()メソッド内（Lines 249-296）
for i in range(len(result_data)):
    # ... (取引期間フィルタリング省略) ...
    
    if position_size == 0:
        # エントリーチェック
        entry_signal = self.generate_entry_signal(i)
        if entry_signal == 1:
            result_data['Entry_Signal'].iloc[i] = 1
            position_size = 1.0
            self.entry_prices[i] = result_data[self.price_column].iloc[i]  # ← Line 276
            self.position_days[i] = 0
```

**確認事項:**
- `self.entry_prices[i] = result_data[self.price_column].iloc[i]`
- `self.price_column`はデフォルトで`"Adj Close"`（Line 28確認）
- **i日目の終値（Adj Close）でエントリー価格を記録している**

**問題点:**
- i日目の終値は、その日の市場終了後にしか確定しない
- リアルトレードでは、終値を見てから終値で買うことは不可能
- **ルックアヘッドバイアスが発生している ✅確定**

**根拠:**
- INVESTIGATION_REPORT.mdのbase_strategy.py Line 242と同様の問題
- ContrararianStrategy Phase 1修正（Lines 273-281）と同様の修正が必要

---

### 結果2: インジケーターのshift(1)適用 ✅確認完了

#### 証拠1: SMA（移動平均）

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Lines 70-72  

```python
# ルックアヘッドバイアス修正: 移動平均の計算
self.data['SMA'] = self.data[self.price_column].rolling(
    window=self.params["sma_period"]
).mean().shift(1)
```

**確認事項:**
- `.shift(1)`適用済み ✅正しい実装
- コメントに「ルックアヘッドバイアス修正」と明記

---

#### 証拠2: ボリンジャーバンド

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Lines 74-82  

```python
# ルックアヘッドバイアス修正: ボリンジャーバンドの計算
bb_sma = self.data[self.price_column].rolling(
    window=self.params["bb_period"]
).mean()
bb_std = self.data[self.price_column].rolling(
    window=self.params["bb_period"]
).std()

self.data['BB_Upper'] = (bb_sma + (bb_std * self.params["bb_std_dev"])).shift(1)
self.data['BB_Lower'] = (bb_sma - (bb_std * self.params["bb_std_dev"])).shift(1)
self.data['BB_Middle'] = bb_sma.shift(1)
```

**確認事項:**
- `.shift(1)`適用済み ✅正しい実装
- 3つのバンド全てにshift(1)適用

---

#### 証拠3: Z-score（統計的異常値検出）

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Lines 84-92  

```python
# ルックアヘッドバイアス修正: Z-score計算（統計的異常値検出）
z_sma = self.data[self.price_column].rolling(
    window=self.params["zscore_period"]
).mean()
z_std = self.data[self.price_column].rolling(
    window=self.params["zscore_period"]
).std()

self.data['Z_Score'] = ((self.data[self.price_column] - z_sma) / z_std).shift(1)
```

**確認事項:**
- `.shift(1)`適用済み ✅正しい実装

---

#### 証拠4: RSI（相対力指数）

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Lines 94-96  

```python
# ルックアヘッドバイアス修正: RSIフィルター（オプション）
if self.params["rsi_filter"]:
    self.data['RSI'] = self._calculate_rsi().shift(1)
```

**確認事項:**
- `.shift(1)`適用済み ✅正しい実装
- オプション機能（rsi_filter=True時のみ）

---

#### 証拠5: ATR（Average True Range）

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Lines 99-101  

```python
# ルックアヘッドバイアス修正: ATR（ボラティリティベースのストップロス）
if self.params["atr_filter"]:
    self.data['ATR'] = self._calculate_atr().shift(1)
```

**確認事項:**
- `.shift(1)`適用済み ✅正しい実装
- オプション機能（atr_filter=True時のみ）

---

#### 証拠6: Volume_MA（ボリューム移動平均）

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Lines 103-107  

```python
# ルックアヘッドバイアス修正: ボリューム移動平均
if self.params["volume_confirmation"]:
    self.data['Volume_MA'] = self.data['Volume'].rolling(
        window=self.params["sma_period"]
    ).mean().shift(1)
```

**確認事項:**
- `.shift(1)`適用済み ✅正しい実装
- オプション機能（volume_confirmation=True時のみ）

---

#### インジケーターのshift(1)適用状況まとめ

| インジケーター | shift(1)適用 | 行番号 | 備考 |
|--------------|-------------|--------|------|
| SMA | ✅適用済み | Line 70-72 | 必須 |
| BB_Upper | ✅適用済み | Line 80 | 必須 |
| BB_Lower | ✅適用済み | Line 81 | 必須 |
| BB_Middle | ✅適用済み | Line 82 | 必須 |
| Z_Score | ✅適用済み | Line 92 | 必須 |
| RSI | ✅適用済み | Line 96 | オプション |
| ATR | ✅適用済み | Line 101 | オプション |
| Volume_MA | ✅適用済み | Line 106 | オプション |

**結論:**
- 全インジケーターに`.shift(1)`が適用されている ✅正しい実装
- コメントに「ルックアヘッドバイアス修正」と明記されている
- インジケーター側の実装は問題なし

---

### 結果3: ループ範囲の確認 ✅確認完了

#### 証拠: backtest()メソッドのループ範囲

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Line 256  

```python
def backtest(self, trading_start_date=None, trading_end_date=None) -> pd.DataFrame:
    """バックテスト実行"""
    result_data = self.data.copy()
    result_data['Entry_Signal'] = 0
    result_data['Exit_Signal'] = 0
    result_data['Position'] = 0
    
    position_size = 0
    
    for i in range(len(result_data)):  # ← Line 256
        # ... (省略) ...
        if position_size == 0:
            entry_signal = self.generate_entry_signal(i)
            if entry_signal == 1:
                result_data['Entry_Signal'].iloc[i] = 1
                position_size = 1.0
                self.entry_prices[i] = result_data[self.price_column].iloc[i]  # ← 当日終値
```

**確認事項:**
- ループ範囲: `range(len(result_data))` - 最終日まで含む
- エントリー価格: `result_data[self.price_column].iloc[i]` - 当日終値使用

**問題点:**
- Phase 1修正後は`iloc[i + 1]`（翌日始値）にアクセスする必要がある
- 最終日（i = len(result_data) - 1）でエントリーするとIndexError発生
- **ループ範囲を`range(len(result_data) - 1)`に変更する必要あり**

**根拠:**
- ContrararianStrategy Phase 1修正（Lines 253-256）と同様の修正が必要
- 「最終日を除外してidx+1アクセスを安全に」コメント参照

---

### 結果4: generate_entry_signal()での判断タイミング ✅確認完了

#### 証拠: generate_entry_signal()メソッド

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Lines 173-192  

```python
def generate_entry_signal(self, idx: int) -> int:
    """エントリーシグナル生成"""
    if idx < max(self.params["sma_period"], self.params["zscore_period"]):
        return 0
        
    # 統計的異常値チェック
    if not self._is_statistical_anomaly(idx):
        return 0
        
    # ボリューム確認
    if not self._volume_confirmation_check(idx):
        return 0
        
    # RSIフィルター（オプション）
    if self.params["rsi_filter"] and 'RSI' in self.data.columns:
        rsi = self.data['RSI'].iloc[idx]
        if pd.notna(rsi) and rsi > self.params["rsi_oversold"]:
            return 0  # RSIが過売り状態でない場合はエントリーしない
            
    return 1  # ロングエントリー
```

**確認事項:**
- idx日目のインジケーター（全てshift(1)済み）でエントリー判断
- 判断自体は前日までのデータを使用している ✅正しい実装

**結論:**
- エントリー判断のタイミングは問題なし
- インジケーターがshift(1)済みのため、リアルトレードと同じ判断

---

## 原因分析

### 根本原因

**直接原因**: strategies/mean_reversion_strategy.py Line 276
```python
self.entry_prices[i] = result_data[self.price_column].iloc[i]  # price_column = 'Adj Close'
```

**問題の構造:**
1. i日目に`generate_entry_signal(i)`でエントリー判断（インジケーターはshift(1)済み）
2. しかし、エントリー価格はi日目の終値（Adj Close）を使用
3. リアルトレードでは、i日目の終値を見てからi日目の終値で買うことは不可能
4. **ルックアヘッドバイアスが発生している**

### 正しい実装

```python
# 現状（誤り）
self.entry_prices[i] = result_data[self.price_column].iloc[i]  # i日の終値

# 正しい実装（Phase 1修正）
next_day_open = result_data['Open'].iloc[i + 1]  # i+1日の始値
self.entry_prices[i] = next_day_open
```

**理由:**
- i日の市場終了後に判断（インジケーターはshift(1)済み）
- 翌日（i+1日）の市場開始時（始値）でエントリー
- これがリアルトレードの実態

### 他戦略との共通点

**類似の問題を持つ戦略:**
- ContrararianStrategy（Phase 1修正完了）
- ForceCloseStrategy（修正不要、他戦略に依存）
- VWAP_Breakout（Phase 1修正完了）
- Momentum_Investing（修正状況未確認）
- Breakout（修正状況未確認）
- GC Strategy（修正状況未確認）

**MeanReversionStrategyの特徴:**
- 独自backtest()メソッド実装（BaseStrategyを継承していない）
- ContrararianStrategyと同様の修正パターン
- インジケーターのshift(1)は既に適用済み（良い実装）

---

## 修正提案

### Phase 1: 最小限の修正（必須）

#### 修正箇所1: ループ範囲の変更

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Line 256  

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

**理由:**
- i+1で翌日始値にアクセスするため、最終日を除外
- IndexErrorの防止

---

#### 修正箇所2: エントリー価格の変更

**ファイル**: strategies/mean_reversion_strategy.py  
**行番号**: Line 276  

**修正前:**
```python
if entry_signal == 1:
    result_data['Entry_Signal'].iloc[i] = 1
    position_size = 1.0
    self.entry_prices[i] = result_data[self.price_column].iloc[i]
    self.position_days[i] = 0
```

**修正後:**
```python
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

**理由:**
- エントリー価格を当日終値（Adj Close）から翌日始値（Open）に変更
- リアルトレードとの整合性確保

---

### Phase 2: 現実的な制約の追加（推奨）

#### 修正箇所3: スリッページの考慮

**Phase 1修正完了後に実装:**

```python
# Phase 2修正: スリッページ考慮
slippage = 0.001  # 0.1%
next_day_open = result_data['Open'].iloc[i + 1]
entry_price_with_slippage = next_day_open * (1 + slippage)
self.entry_prices[i] = entry_price_with_slippage
```

**理由:**
- リアルトレードでは始値で必ずしも約定しない
- 0.1%のスリッページを加味するのが一般的

---

### Phase 1修正による影響

**予想される変化:**
1. **エントリー価格の精度変化**: 13桁精度 → 2桁精度（市場価格の精度）
2. **エントリー価格の値変化**: 当日終値 → 翌日始値（3-7%の差異）
3. **バックテスト結果への影響**:
   - リターン率: やや悪化（現実的な結果）
   - 最大ドローダウン: やや増加（不利なエントリー価格反映）
   - 勝率: やや低下（有利すぎるエントリー価格の補正）

**検証方法:**
- ContrararianStrategy Phase 1修正と同様の検証スクリプト作成
- ダミーデータで4-5件のエントリー発生
- エントリー価格と翌日始値の差分を確認（0.00円が期待値）

---

### copilot-instructions.md遵守状況

**ルックアヘッドバイアス禁止ルール（2025-12-20以降必須）:**

**Phase 1修正前:**
- [ ] エントリー価格は`data['Open'].iloc[idx + 1]` ← **未実装**
- [x] インジケーターに`.shift(1)`適用 ← **実装済み**
- [ ] スリッページ考慮（推奨0.1%） ← **未実装**

**Phase 1修正後:**
- [x] エントリー価格は`data['Open'].iloc[idx + 1]` ← **実装予定**
- [x] インジケーターに`.shift(1)`適用 ← **実装済み**
- [ ] スリッページ考慮（推奨0.1%） ← **Phase 2対応**

---

## セルフチェック

### a) 見落としチェック ✅

**確認したファイル:**
- ✅ `strategies/mean_reversion_strategy.py` Lines 1-407 - 全行確認済み
- ✅ `docs/Lookhead bias problem/INVESTIGATION_REPORT.md` - 参照済み
- ✅ `docs/Lookhead bias problem/CONTRARIAN_INVESTIGATION_REPORT.md` - 参照済み

**確認していないファイル:**
- なし（mean_reversion_strategy.pyは単独ファイル、他モジュールへの依存なし）

**カラム名・変数名の確認:**
- ✅ `self.price_column` = "Adj Close" (Line 28確認)
- ✅ `result_data[self.price_column].iloc[i]` (Line 276確認)
- ✅ `self.entry_prices[i]` (Line 276確認)
- ✅ `.shift(1)` (Lines 72, 80-82, 92, 96, 101, 106確認)

**データの流れ:**
- ✅ yfinance → DataFrame → initialize_strategy() → backtest() → entry_prices - 追跡済み

### b) 思い込みチェック ✅

**前提の検証:**
- ❌ 「独自backtest()だから問題ないはず」 → ✅ Line 276で当日終値使用を確認
- ❌ 「インジケーターにshift(1)があれば大丈夫なはず」 → ✅ エントリー価格は別問題と判明
- ❌ 「BaseStrategy継承だから基底クラスの問題のはず」 → ✅ 独自backtest()実装を確認

**実際に確認した事実:**
- ✅ Line 276で`self.entry_prices[i] = result_data[self.price_column].iloc[i]`を確認
- ✅ Line 28で`self.price_column = price_column`（デフォルト"Adj Close"）を確認
- ✅ Lines 70-107で全インジケーターに`.shift(1)`適用を確認
- ✅ Line 256で`for i in range(len(result_data)):`を確認

### c) 矛盾チェック ✅

**調査結果の整合性:**
- ✅ インジケーターはshift(1)済み（正しい実装）
- ✅ しかしエントリー価格は当日終値使用（誤った実装）
- ✅ ContrararianStrategyと同様の問題パターン
- ✅ INVESTIGATION_REPORT.mdの知見と一致

**ログ/エラーとの整合性:**
- 該当なし（調査段階のため実行ログなし）

---

## 次のステップ（推奨作業順序）

### 1. Phase 1修正実施 ✅推奨

**対象:**
- strategies/mean_reversion_strategy.py Line 256（ループ範囲）
- strategies/mean_reversion_strategy.py Line 276（エントリー価格）

**修正内容:**
- ループ範囲: `range(len(result_data))` → `range(len(result_data) - 1)`
- エントリー価格: `result_data[self.price_column].iloc[i]` → `result_data['Open'].iloc[i + 1]`

**参考:**
- ContrararianStrategy Phase 1修正（Lines 253-256, 273-281）
- INVESTIGATION_REPORT.md Phase 1実装結果

---

### 2. 検証スクリプト作成・実行 ✅推奨

**検証内容:**
- ダミーデータ（平均回帰パターン）生成
- MeanReversionStrategy Phase 1修正版でバックテスト実行
- エントリー価格と翌日始値の差分確認（期待値: 0.00円）
- 最終日のエントリーシグナル確認（期待値: 0）

**参考:**
- test_contrarian_syntax.py（ContrararianStrategy検証スクリプト）

---

### 3. Phase 2実装（スリッページ・取引コスト） ⏳延期

**Phase 1完了後に実施:**
- スリッページ: 0.1%
- 取引コスト: 0.1%（オプション）
- パラメータ化（config.yaml等で管理）

---

### 4. 統合テスト実行 ⏳延期

**Phase 1修正完了後:**
- DSSMS統合バックテスト実行
- 修正前後のバックテスト結果比較
- 取引件数・リターン率・最大ドローダウンの変化確認

---

## 付録

### 修正箇所一覧

| 箇所 | ファイル | 行番号 | 修正内容 | Phase |
|------|---------|--------|---------|-------|
| 1 | mean_reversion_strategy.py | Line 256 | ループ範囲変更 | Phase 1 |
| 2 | mean_reversion_strategy.py | Line 276 | エントリー価格を翌日始値に変更 | Phase 1 |
| 3 | mean_reversion_strategy.py | Line 276 | スリッページ追加 | Phase 2 |

### 参考資料

- [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) - ルックアヘッドバイアス問題の全体像
- [CONTRARIAN_INVESTIGATION_REPORT.md](CONTRARIAN_INVESTIGATION_REPORT.md) - ContrararianStrategy Phase 1修正事例
- [copilot-instructions.md](../../.github/copilot-instructions.md) - ルックアヘッドバイアス禁止ルール

---

## 結論

### 調査結果まとめ

**判明したこと（証拠付き）:**
1. ✅ **エントリー価格のルックアヘッドバイアス存在**
   - Line 276: `self.entry_prices[i] = result_data[self.price_column].iloc[i]`
   - 当日終値（Adj Close）でエントリー価格を記録
   - リアルトレードでは不可能な取引

2. ✅ **全インジケーターに`.shift(1)`適用済み**
   - SMA, ボリンジャーバンド, Z-score, RSI, ATR, Volume_MA
   - コメントに「ルックアヘッドバイアス修正」と明記
   - インジケーター側の実装は正しい

3. ✅ **ループ範囲の問題**
   - Line 256: `for i in range(len(result_data))`
   - 最終日まで含むため、Phase 1修正後にIndexError発生の可能性

4. ✅ **ContrararianStrategyと同様の問題パターン**
   - 独自backtest()メソッド実装
   - インジケーターはshift(1)済みだがエントリー価格は当日終値使用
   - 同様の修正方針が適用可能

**不明な点:**
- なし（調査範囲の全項目を確認済み）

**原因の推定:**
1. **確定**: Line 276でi日目の終値を使用している（証拠: コード確認）
2. **確定**: インジケーターはshift(1)済みで正しい実装（証拠: Lines 70-107）
3. **確定**: エントリー価格とインジケーター判断のタイミングが矛盾（証拠: 上記2点）

### 修正の必要性

**Phase 1修正（必須）:**
- ✅ 必要（ルックアヘッドバイアス解消のため）
- 対象: Line 256（ループ範囲）, Line 276（エントリー価格）
- 優先度: **最高**

**Phase 2修正（推奨）:**
- ⏳ 推奨（リアルトレードとの整合性向上のため）
- 対象: スリッページ・取引コスト追加
- 優先度: **中**

**Phase 3修正（別ファイル対応）:**
- ⏳ 今回スルー（別ファイルで対応予定）
- 対象: エグジット問題（EXIT_INVESTIGATION_REPORT.md）
- 優先度: **中**

---

**報告書作成者**: GitHub Copilot  
**最終更新日**: 2025-12-22  
**バージョン**: 1.0  
**調査状況**: ✅完了（修正提案まで含む）
