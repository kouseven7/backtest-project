# SupportResistanceContrarianStrategy ルックアヘッドバイアス調査報告

**調査日**: 2025-12-21  
**調査対象**: `strategies/support_resistance_contrarian_strategy.py`  
**調査目的**: ルックアヘッドバイアス（Look-Ahead Bias）の特定と修正方針の策定

---

## 1. 調査概要

### 1.1 戦略概要
- **戦略名**: Support/Resistance Contrarian Strategy
- **戦略タイプ**: 支持線・抵抗線ベース逆張り戦略
- **ファイル行数**: 433行
- **作成日**: 2025-07-22
- **BaseStrategy継承**: あり
- **独自backtest()実装**: あり（Line 296）

### 1.2 調査手法
1. ファイル全体の構造確認（Lines 1-100）
2. grep検索による問題箇所の特定（`backtest|initialize_strategy|entry_price|self.entry_prices`）
3. backtest()メソッドの詳細確認（Lines 296-340）
4. エントリー価格決定ロジックの分析（Line 323）
5. ループ範囲の確認（Line 305）

---

## 2. 調査結果

### 2.1 ルックアヘッドバイアスの存在（確定）

#### 問題箇所1: エントリー価格決定（Line 323）

**コード**:
```python
self.entry_prices[i] = result_data[self.price_column].iloc[i]
```

**証拠**:
- Line 29: `price_column: str = "Adj Close"` - デフォルト値
- Line 36: `self.price_column = price_column` - インスタンス変数に代入
- Line 323: `result_data[self.price_column].iloc[i]` - i日目の終値（Adj Close）使用

**問題の詳細**:
- i日目のエントリーシグナル発生時、i日目の終値（Adj Close）でエントリー価格を記録
- リアルトレードでは、終値時点ではすでに市場が閉まっており、その価格での取引は不可能
- 正しくは翌日（i+1日目）の始値でエントリーすべき

**影響範囲**:
- backtest()メソッド内の全エントリー処理（Line 323）
- エントリー価格が過度に有利な価格で記録される
- バックテスト結果が実際のトレードパフォーマンスを過大評価

#### 問題箇所2: ループ範囲（Line 305）

**コード**:
```python
for i in range(len(result_data)):
```

**問題の詳細**:
- 最終日（len(result_data) - 1）まで全日ループ
- 翌日始値（i+1）でエントリーする場合、最終日のシグナルでi+1日目のデータが存在せずIndexError
- 正しくは`range(len(result_data) - 1)`で最終日の前日までループすべき

**影響範囲**:
- Phase 1修正（翌日始値使用）実行時にIndexErrorが発生する可能性

---

### 2.2 良い実装（shift(1)適用済み）

#### initialize_strategy()メソッド（Lines 69-73）

**コード**:
```python
def initialize_strategy(self):
    """戦略初期化処理"""
    super().initialize_strategy()
    
    # ルックアヘッドバイアス修正: RSI計算（確認シグナル用）
    if self.params["rsi_confirmation"]:
        self.data['RSI'] = self._calculate_rsi().shift(1)
    
    # ルックアヘッドバイアス修正: ボリューム移動平均
    self.data['Volume_MA'] = self.data['Volume'].rolling(
        window=self.params["lookback_period"]
    ).mean().shift(1)
```

**良い点**:
- RSI、Volume_MAにshift(1)適用済み
- コメントで「ルックアヘッドバイアス修正」と明記
- 開発者のルックアヘッドバイアスへの意識が高い

**備考**:
エントリー価格決定のルックアヘッドバイアスは見落とされているが、インジケーター部分は正しく実装されている。

---

## 3. PairsTradingStrategyとの比較

### 3.1 共通パターン

| 項目 | PairsTradingStrategy | SupportResistanceContrarianStrategy |
|------|----------------------|-------------------------------------|
| BaseStrategy継承 | あり | あり |
| 独自backtest()実装 | あり（Line 185） | あり（Line 296） |
| エントリー価格問題 | Line 259（当日終値） | Line 323（当日終値） |
| ループ範囲問題 | Line 243（最終日含む） | Line 305（最終日含む） |
| インジケーターshift(1) | 未適用 | 適用済み（良い実装） |

### 3.2 PairsTradingStrategy Phase 1修正内容

**修正箇所1: ループ範囲（Line 243）**:
```python
# 修正前
for i in range(len(result_data)):

# 修正後
for i in range(len(result_data) - 1):
```

**修正箇所2: エントリー価格（Lines 259-268）**:
```python
# 修正前
entry_price = result_data['Adj Close'].iloc[i]

# 修正後
if i + 1 >= len(result_data):
    continue
entry_price = result_data['Open'].iloc[i + 1]
```

**検証結果**:
- エントリー2件、差分0.00円（完全一致）
- 最終日エントリーシグナル=0
- Phase 1修正成功

---

## 4. Phase 1修正提案

### 4.1 修正箇所

#### 修正1: ループ範囲（Line 305）

**修正前**:
```python
for i in range(len(result_data)):
```

**修正後**:
```python
for i in range(len(result_data) - 1):
```

**理由**:
- 翌日始値（i+1）でエントリーするため、最終日のシグナルは取引不可
- IndexError回避

#### 修正2: エントリー価格決定（Line 323）

**修正前**:
```python
self.entry_prices[i] = result_data[self.price_column].iloc[i]
```

**修正後**:
```python
# IndexError対策（念のため）
if i + 1 >= len(result_data):
    continue

# 翌日始値でエントリー
self.entry_prices[i] = result_data['Open'].iloc[i + 1]
```

**理由**:
- i日目のシグナル発生時、翌日（i+1日目）の始値でエントリー
- リアルトレードと整合性を保つ
- ルックアヘッドバイアス解消

---

## 5. 検証計画

### 5.1 検証スクリプト作成

**ファイル名**: `tests/temp/test_20241221_support_resistance_contrarian_syntax.py`

**検証内容**:
1. ダミーデータでバックテスト実行
2. エントリー価格と翌日始値の差分確認（許容誤差: 0.01円）
3. 最終日エントリーシグナル確認（0であるべき）
4. entry_pricesのキー（日付インデックス）がresult_data範囲内であることを確認

### 5.2 検証手順

1. **ダミーデータ生成**:
   - 60日分のOHLCVデータ
   - サイン波ベースのレンジ相場（95-105円）
   - エントリーシグナルが発生しやすいパラメータ設定

2. **バックテスト実行**:
   - SupportResistanceContrarianStrategy.backtest()呼び出し
   - entry_pricesとresult_data['Open']の照合

3. **差分確認**:
   - 各エントリー価格とresult_data['Open'].iloc[idx + 1]の差分計算
   - 許容誤差: 0.01円（浮動小数点誤差考慮）

4. **最終日確認**:
   - result_data['Entry_Signal'].iloc[-1] == 0 を確認

---

## 6. セルフチェック

### 6.1 調査完了項目

- [x] エントリー価格決定ロジックの特定（Line 323）
- [x] ループ範囲の確認（Line 305）
- [x] インジケーターshift(1)の確認（Lines 69-73、適用済み）
- [x] backtest()独自実装の確認（Line 296）
- [x] price_columnデフォルト値の確認（Line 29: "Adj Close"）
- [x] PairsTradingStrategyとの比較分析

### 6.2 証拠明示

- [x] Line 323: `self.entry_prices[i] = result_data[self.price_column].iloc[i]`
- [x] Line 29: `price_column: str = "Adj Close"`
- [x] Line 305: `for i in range(len(result_data))`
- [x] Lines 69-73: RSI、Volume_MAのshift(1)適用

### 6.3 修正提案の妥当性

- [x] PairsTradingStrategy Phase 1修正と同様のパターン
- [x] copilot-instructions.mdのルールに準拠
- [x] IndexError対策を含む
- [x] リアルトレードとの整合性確保

---

## 7. 次のステップ

### 7.1 Phase 1修正実行（優先度: HIGH）

1. **Line 305修正**: ループ範囲を`range(len(result_data) - 1)`に変更
2. **Line 323修正**: エントリー価格を`result_data['Open'].iloc[i + 1]`に変更
3. **multi_replace_string_in_file使用**: 2箇所を同時修正

### 7.2 検証実行（優先度: HIGH）

1. **検証スクリプト作成**: `tests/temp/test_20241221_support_resistance_contrarian_syntax.py`
2. **バックテスト実行**: ダミーデータで動作確認
3. **差分確認**: エントリー価格と翌日始値の一致確認
4. **最終日確認**: 最終日エントリーシグナル=0の確認

### 7.3 他戦略の調査（優先度: MEDIUM）

1. **momentum_investing.py**: Phase 1修正状況未確認
2. **breakout.py**: Phase 1修正状況未確認
3. **gc_strategy.py**: Phase 1修正状況未確認

---

## 8. 備考

### 8.1 開発者の意識

- initialize_strategy()でRSI、Volume_MAにshift(1)適用済み
- コメントで「ルックアヘッドバイアス修正」と明記
- エントリー価格決定のルックアヘッドバイアスは見落とし

### 8.2 イグジット問題

- Lines 263-268: generate_exit_signal()でentry_price使用
- 今回はイグジット問題は別ファイル対応予定（スルー）

---

## 9. まとめ

**調査結果**:
- SupportResistanceContrarianStrategyにルックアヘッドバイアスが存在（Line 323）
- エントリー価格が当日終値（Adj Close）で決定されている
- ループ範囲も最終日まで含む（Line 305）
- インジケーター（RSI、Volume_MA）はshift(1)適用済み（良い実装）

**Phase 1修正提案**:
- Line 305: ループ範囲を`range(len(result_data) - 1)`に変更
- Line 323: エントリー価格を`result_data['Open'].iloc[i + 1]`に変更

**次のステップ**:
- Phase 1修正実行 → 検証スクリプト作成・実行 → 他戦略調査

**調査完了**: 2025-12-21  
**調査担当**: Backtest Project Team
