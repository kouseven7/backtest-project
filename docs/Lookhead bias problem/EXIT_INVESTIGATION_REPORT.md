# ルックアヘッドバイアス問題 調査報告書（エグジット編）

**作成日**: 2025-12-21  
**調査期間**: 2025-12-21  
**調査者**: GitHub Copilot  
**調査範囲**: strategies/VWAP_Breakout.py (generate_exit_signal)  
**関連ドキュメント**: [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) (エントリー編)

---

## 目次

1. [調査目的](#調査目的)
2. [調査対象](#調査対象)
3. [調査方法](#調査方法)
4. [調査結果](#調査結果)
5. [原因分析](#原因分析)
6. [影響範囲](#影響範囲)
7. [改善目標](#改善目標)
8. [質問への回答](#質問への回答)
9. [セルフチェック](#セルフチェック)
10. [次のステップ](#次のステップ)

---

## 調査目的

エントリー編（[INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md)）で判明したルックアヘッドバイアス問題に続き、**イグジット価格**についても同様の問題が存在することが判明した。

本調査の目的は以下の通り：
1. イグジット価格のルックアヘッドバイアスを詳細に特定する
2. ストップロス・利益確定・トレーリングストップの問題箇所を明確化する
3. リアルトレードに近いイグジット実装のための改善方針を策定する
4. エントリー修正（Phase 1）との整合性を確保する

---

## 調査対象

### 対象ファイル

#### 主要対象
- **`strategies/VWAP_Breakout.py`**: VWAP Breakout戦略
  - Line 330-395: generate_exit_signal()メソッド
  - Line 344: current_price取得（当日終値）
  - Line 346: entry_price取得（既に問題のある価格）
  - Line 357: ストップロス判定
  - Line 361: 利益確定判定
  - Line 367: トレーリングストップ（当日高値使用）
  - Line 387-393: RSI/MACD反転

#### 関連ファイル
- **`strategies/base_strategy.py`**: 全戦略の基底クラス（エントリー修正対象）
- **[INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md)**: エントリー編の調査結果

### 対象外
- エントリー価格問題（Phase 1で修正済み）
- 他の戦略クラス（各戦略ごとに別途調査・修正予定）

---

## 調査方法

### 調査手順

#### 1. コードレビュー
- generate_exit_signal()メソッドの全ロジックを確認
- 各イグジット条件での価格取得方法を特定
- 当日高値/安値の使用箇所を確認

#### 2. リアルトレードとの比較
- 各イグジット条件でのリアルトレードの実態を確認
- 当日高値/安値の使用可否を判断
- イグジット価格の選択肢を整理

#### 3. エントリー修正との整合性確認
- Phase 1修正（エントリー価格）との整合性を確認
- 修正の優先順位を決定

---

## 調査結果

### 結果1: generate_exit_signal()の全ロジック ✅確定

#### 証拠: VWAP_Breakout.py Line 330-395

**コード構造**:
```python
def generate_exit_signal(self, idx: int, entry_idx: int = None) -> int:
    if idx < 1 or entry_idx is None:
        return 0
    
    # Line 344: 当日終値を取得
    current_price = self.data[self.price_column].iloc[idx]  # price_column = "Adj Close"
    
    # Line 346: エントリー価格を取得（既に問題のある価格）
    entry_price = self.data[self.price_column].iloc[entry_idx]
    
    vwap = self.data['VWAP'].iloc[idx]
```

**問題点**:
- `current_price` = 当日終値（`Adj Close`）
- `entry_price` = エントリー時の価格（既に当日終値として記録済み）
- 全イグジット判定が当日終値ベース

---

### 結果2: ストップロス判定の問題 ✅確定

#### 証拠: Line 357

```python
# ストップロス条件
if current_price <= entry_price * (1 - self.params["stop_loss"]):
    self.log_trade(f"VWAP Breakout イグジットシグナル: ストップロス 日付={self.data.index[idx]}, 価格={current_price}")
    return -1
```

**問題の構造**:
1. `current_price` = 当日終値
2. ストップロスは `entry_price * (1 - 0.03)` = 3%下落で発動
3. **問題**: 当日終値を見てからストップロス判定
4. **リアルトレード**: 当日の値動きを監視して、安値がストップロス価格に到達したら即座にイグジット

**具体例**:
```
entry_price = 1000円
stop_loss_price = 1000 * (1 - 0.03) = 970円

現在のロジック（誤り）:
- 当日終値 = 965円 → ストップロス発動、イグジット価格=965円

リアルトレードの実態:
- 当日の値動き: Open=995 → High=1005 → Low=960 → Close=965
- Low=960 < 970 → ストップロス発動、イグジット価格≒970円（スリッページ考慮）
```

**影響**:
- 現在のロジックは当日終値でイグジット（965円）
- リアルトレードではストップロス価格近辺でイグジット（970円）
- **5円の差（0.5%）** → ストップロスが厳しすぎる結果に

---

### 結果3: 利益確定判定の問題 ✅確定

#### 証拠: Line 361

```python
# 利益確定条件
if current_price >= entry_price * (1 + self.params["take_profit"]):
    self.log_trade(f"VWAP Breakout イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
    return -1
```

**問題の構造**:
1. `current_price` = 当日終値
2. 利益確定は `entry_price * (1 + 0.15)` = 15%上昇で発動
3. **問題**: 当日終値を見てから利確判定
4. **リアルトレード**: 当日の値動きを監視して、高値が利確価格に到達したら即座にイグジット

**具体例**:
```
entry_price = 1000円
take_profit_price = 1000 * (1 + 0.15) = 1150円

現在のロジック（誤り）:
- 当日終値 = 1155円 → 利確発動、イグジット価格=1155円

リアルトレードの実態:
- 当日の値動き: Open=1140 → High=1160 → Low=1135 → Close=1155
- High=1160 > 1150 → 利確発動、イグジット価格≒1150円（スリッページ考慮）
```

**影響**:
- 現在のロジックは当日終値でイグジット（1155円）
- リアルトレードでは利確価格近辺でイグジット（1150円）
- **5円の差（0.4%）** → 利確が甘すぎる結果に（楽観的）

---

### 結果4: トレーリングストップの問題 ✅確定

#### 証拠: Line 365-371

```python
# 高度なトレーリングストップ
profit_pct = (current_price - entry_price) / entry_price
if profit_pct >= self.params.get("trailing_start_threshold", 0):
    high_since_entry = self.data['High'].iloc[entry_idx:idx+1].max()  # Line 367
    trailing_stop = high_since_entry * (1 - self.params["trailing_stop"])
    if current_price <= trailing_stop:
        self.log_trade(f"VWAP Breakout イグジットシグナル: トレーリングストップ 日付={self.data.index[idx]}, 価格={current_price}")
        return -1
```

**問題の構造**:
1. **Line 367**: `high_since_entry = self.data['High'].iloc[entry_idx:idx+1].max()`
   - `idx+1`は当日を含む → **当日高値を使用**
2. `trailing_stop = high_since_entry * (1 - 0.05)` = 最高値から5%下落
3. `current_price <= trailing_stop`で判定（当日終値 vs トレーリングストップ）

**具体例**:
```
entry_idx=10, idx=15（5日間保有）
params["trailing_stop"] = 0.05（5%）

現在のロジック（誤り）:
- 前日までの最高値 = 980円
- 当日高値 = 1000円（新高値）
- 当日終値 = 960円
- high_since_entry = max(High[10:16]) = 1000円（当日高値含む）
- trailing_stop = 1000 * 0.95 = 950円
- 960 > 950 → イグジットしない

リアルトレードの実態:
- 前日までの最高値 = 980円
- trailing_stop = 980 * 0.95 = 931円
- 当日の値動き: Open=970 → High=1000 → Low=925 → Close=960
- Low=925 < 931 → トレーリングストップ発動、イグジット価格≒931円
```

**影響**:
- **当日高値を即座にトレーリングストップ計算に使用**
- リアルトレードでは、高値更新とストップ価格到達のタイミングが重要
- 現在のロジックは「当日終値で判定」しているため、日中の値動きを無視
- **楽観的な結果**（イグジットが遅れる可能性）

---

### 結果5: RSI/MACD反転の問題 ✅確定

#### 証拠: Line 387-393

```python
# RSIやMACDの反転
rsi = self.data['RSI'].iloc[idx]
macd = self.data['MACD'].iloc[idx]
signal_line = self.data['Signal_Line'].iloc[idx]
if rsi > 70 and rsi < self.data['RSI'].iloc[idx - 1]:
    self.log_trade(f"VWAP Breakout イグジットシグナル: RSI反転 日付={self.data.index[idx]}, 価格={current_price}")
    return -1
if macd < signal_line and self.data['MACD'].iloc[idx-1] >= self.data['Signal_Line'].iloc[idx-1]:
    self.log_trade(f"VWAP Breakout イグジットシグナル: MACD反転 日付={self.data.index[idx]}, 価格={current_price}")
    return -1
```

**問題の構造**:
- RSI, MACDは`.shift(1)`適用済み（Line 110-126で確認済み）
- `self.data['RSI'].iloc[idx]` = **前日RSI**（正しい）
- しかし、イグジット価格は`current_price`（当日終値）を使用

**問題点**:
- RSI/MACD反転判定自体は正しい（前日データ使用）
- しかし、イグジット価格が当日終値
- **リアルトレード**: 前日市場終了後に判断 → 翌日始値でイグジット

**具体例**:
```
idx=15の市場終了後:
- RSI[15] = 前日RSI = 72（shift(1)により）
- RSI[14] = 2日前RSI = 68
- 72 > 70 かつ 72 < 68 は偽 → 反転なし

実際には:
- RSI判定: 正しく前日データ使用
- イグジット価格: 当日終値（current_price）
- 期待: 翌日始値でイグジット
```

---

## 原因分析

### 根本原因

**直接原因**: [`strategies/VWAP_Breakout.py`](c:\Users\imega\Documents\my_backtest_project\strategies\VWAP_Breakout.py#L344-L346) Line 344, 346

```python
current_price = self.data[self.price_column].iloc[idx]  # 当日終値
entry_price = self.data[self.price_column].iloc[entry_idx]  # 既に問題のある価格
```

**問題の構造**:
1. 全イグジット判定が`current_price`（当日終値）ベース
2. ストップロス・利確は当日の高値/安値を見るべき
3. トレーリングストップは当日高値を使用すべきでない
4. RSI/MACD反転は翌日始値でイグジットすべき

### エントリー問題との関係

**エントリー問題** (Phase 1):
- エントリー価格 = 当日終値（誤り）
- 修正: 翌日始値 + スリッページ

**イグジット問題** (Phase 2):
- イグジット価格 = 当日終値（誤り）
- 修正: 条件により異なる（後述）

**相互影響**:
- エントリー価格が修正されても、イグジット価格が未修正では不整合
- エントリー修正により取引が減少する可能性（最終日エントリー不可）
- イグジット修正により損益が変化（ストップロス厳格化、利確緩和）

---

## 影響範囲

### 影響を受けるロジック

#### 確定（本調査で確認済み）
1. **ストップロス判定** (Line 357)
   - 影響度: **高**
   - 現状: 当日終値で判定 → 甘すぎる
   - 修正: 当日安値で判定 → 厳格化

2. **利益確定判定** (Line 361)
   - 影響度: **高**
   - 現状: 当日終値で判定 → 甘すぎる
   - 修正: 当日高値で判定 → 厳格化

3. **トレーリングストップ** (Line 367)
   - 影響度: **中**
   - 現状: 当日高値含む → 楽観的
   - 修正: 前日までの高値のみ → 保守的

4. **RSI/MACD反転** (Line 387-393)
   - 影響度: **中**
   - 現状: 当日終値でイグジット
   - 修正: 翌日始値でイグジット

#### 推定（未検証だが同様の問題を持つ可能性）
5. **VWAP下抜け** (Line 350)
   - 当日終値 vs VWAP（shift(1)適用済み）
   - 修正: 翌日始値でイグジット？

### バックテスト結果への影響

**イグジット修正の影響**:
- **ストップロス**: より早期に発動 → 損失限定強化（リターン低下の可能性）
- **利益確定**: より早期に発動 → 利益確定強化（リターン安定化）
- **トレーリングストップ**: より保守的 → 利益取りこぼし減少
- **RSI/MACD反転**: 1日遅延 → スリッページ増加

**エントリー修正との相乗効果**:
- エントリー価格上昇（翌日始値） + イグジット価格変化
- 総合的な影響は実測が必要

---

## 改善目標

### 目的
イグジット価格のルックアヘッドバイアスを修正し、リアルトレードに近いバックテストを実現する。

### 目標

#### Phase 2: イグジット価格修正（必須）

**2-1. ストップロス修正**
```python
# 現状（誤り）
if current_price <= entry_price * (1 - self.params["stop_loss"]):
    return -1

# 修正案
daily_low = self.data['Low'].iloc[idx]
stop_loss_price = entry_price * (1 - self.params["stop_loss"])
if daily_low <= stop_loss_price:
    # イグジット価格を記録（別途処理）
    self.exit_price = stop_loss_price * (1 - slippage)
    return -1
```

**2-2. 利益確定修正**
```python
# 現状（誤り）
if current_price >= entry_price * (1 + self.params["take_profit"]):
    return -1

# 修正案
daily_high = self.data['High'].iloc[idx]
take_profit_price = entry_price * (1 + self.params["take_profit"])
if daily_high >= take_profit_price:
    # イグジット価格を記録（別途処理）
    self.exit_price = take_profit_price * (1 - slippage)
    return -1
```

**2-3. トレーリングストップ修正**
```python
# 現状（誤り）
high_since_entry = self.data['High'].iloc[entry_idx:idx+1].max()  # 当日高値含む

# 修正案
high_since_entry = self.data['High'].iloc[entry_idx:idx].max()  # 当日高値除外
trailing_stop = high_since_entry * (1 - self.params["trailing_stop"])
daily_low = self.data['Low'].iloc[idx]
if daily_low <= trailing_stop:
    self.exit_price = trailing_stop * (1 - slippage)
    return -1
```

**2-4. RSI/MACD反転修正**
```python
# 現状: 当日終値でイグジット
if rsi > 70 and rsi < self.data['RSI'].iloc[idx - 1]:
    return -1

# 修正案: 翌日始値でイグジット（ループ範囲調整必要）
if rsi > 70 and rsi < self.data['RSI'].iloc[idx - 1]:
    self.exit_price = self.data['Open'].iloc[idx + 1] * (1 + slippage)
    return -1
```

**2-5. イグジット価格の管理**
```python
# backtest()メソッドでイグジット価格を管理
if exit_signal == -1:
    # generate_exit_signal()で設定されたexit_priceを使用
    if hasattr(self, 'exit_price') and self.exit_price is not None:
        actual_exit_price = self.exit_price
        self.exit_price = None  # リセット
    else:
        # フォールバック: 翌日始値
        actual_exit_price = self.data['Open'].iloc[idx + 1] * (1 + slippage)
    
    self.data.loc[self.data.index[idx], 'Exit_Price'] = actual_exit_price
```

#### Phase 3: スリッページ・取引コスト（推奨）

**3-1. スリッページパラメータ化**
```python
# パラメータ追加
default_params = {
    ...
    "slippage_entry": 0.001,  # 0.1% エントリー時スリッページ
    "slippage_exit": 0.001,   # 0.1% イグジット時スリッページ
    "commission_rate": 0.001, # 0.1% 手数料
}
```

### 成功基準

1. **コードレベル**
   - [ ] ストップロス判定に`self.data['Low'].iloc[idx]`を使用
   - [ ] 利益確定判定に`self.data['High'].iloc[idx]`を使用
   - [ ] トレーリングストップに前日までの高値のみ使用
   - [ ] RSI/MACD反転は翌日始値でイグジット
   - [ ] スリッページ考慮

2. **検証レベル**
   - [ ] 実データ検証でイグジット価格が適切な範囲内
   - [ ] ストップロス発動時の価格がストップロス価格近辺
   - [ ] 利確発動時の価格が利確価格近辺

3. **バックテスト結果**
   - [ ] 修正前後でバックテスト結果を比較
   - [ ] リターン率の変化を定量評価
   - [ ] 最大ドローダウンの変化を確認

### 遵守事項

- **[`.github/copilot-instructions.md`](.github/copilot-instructions.md)** 完全遵守
  - ルックアヘッドバイアス禁止ルール（2025-12-20以降必須）
  - 3原則: 前日判断・翌日始値・取引コスト
  - フォールバック禁止

---

## 質問への回答

### 質問1: イグジット問題の詳細解説

**Q: トレーリングストップの問題が理解できていないので詳しく解説してください**

**A: トレーリングストップの問題（詳細解説）**

**トレーリングストップとは**:
- エントリー後の最高値から一定割合（例: 5%）下落したらイグジット
- 利益を確保しつつ、さらなる上昇の可能性を残す手法
- 最高値が更新されると、ストップ価格も上昇（trailing = 追跡）

**現在のコードの問題**:

```python
# Line 367
high_since_entry = self.data['High'].iloc[entry_idx:idx+1].max()
```

**問題点**: `idx+1`により、`self.data['High'].iloc[idx]`（当日高値）が含まれる

**具体的なシナリオ**:

```
【設定】
- entry_price = 1000円（5日前にエントリー）
- trailing_stop = 0.05（5%）

【5日間の値動き】
日付      Open   High   Low    Close
Day 1    1000   1020   990    1010  → 最高値更新: 1020
Day 2    1010   1050   1005   1040  → 最高値更新: 1050
Day 3    1040   1060   1030   1055  → 最高値更新: 1060
Day 4    1055   1070   1045   1065  → 最高値更新: 1070
Day 5    1065   1100   1010   1020  ← 問題の日

【現在のロジック（誤り）】
idx=Day 5のとき:
1. high_since_entry = max(High[Day1:Day5+1]) 
   = max(1020, 1050, 1060, 1070, 1100) 
   = 1100（Day 5の高値を含む）

2. trailing_stop = 1100 * (1 - 0.05) = 1045円

3. current_price = Close[Day 5] = 1020円

4. 1020 <= 1045 → トレーリングストップ発動
   イグジット価格 = 1020円（当日終値）

【リアルトレードの実態】
Day 5の市場開始前:
1. 前日までの最高値 = 1070円（Day 4まで）

2. trailing_stop = 1070 * (1 - 0.05) = 1016.5円

3. Day 5の値動き:
   - Open = 1065円（ストップ未発動）
   - High = 1100円に上昇（最高値更新！トレーリングストップも1045円に上昇）
   - Low = 1010円に下落（1016.5円を下回る！）
   - この時点でトレーリングストップ発動
   - イグジット価格 ≒ 1016.5円（スリッページ考慮）

【問題の本質】
- 現在のロジック: Day 5の高値（1100円）を即座にトレーリングストップ計算に使用
- リアルトレード: Day 5の高値到達とLowでのストップ発動のタイミングが重要
- **日足データでは、High=1100円とLow=1010円の発生順序が不明**

【正しいアプローチ（保守的）】
```python
# 当日高値を除外（前日までの最高値のみ使用）
high_since_entry = self.data['High'].iloc[entry_idx:idx].max()  # idxを含まない
trailing_stop = high_since_entry * (1 - self.params["trailing_stop"])

# 当日安値でストップ判定
daily_low = self.data['Low'].iloc[idx]
if daily_low <= trailing_stop:
    self.exit_price = trailing_stop * (1 - slippage)
    return -1
```

この実装では:
- Day 5の開始時点での最高値（1070円）を使用
- trailing_stop = 1016.5円
- daily_low = 1010円 < 1016.5円 → イグジット発動
- イグジット価格 ≒ 1016.5円

**なぜ保守的なのか**:
- 実際には、High=1100円到達後にLow=1010円になった可能性もある
- その場合、trailing_stop = 1045円、イグジット価格 ≒ 1045円が正しい
- しかし、日足データではタイミング不明なので、前日最高値を使用（保守的）

### 質問2: EXIT_INVESTIGATION_REPORT.mdへの記録

**Q: 別途調査、修正していくのでEXIT_INVESTIGATION_REPORT.mdに記録してください**

**A: 本ドキュメント（EXIT_INVESTIGATION_REPORT.md）に全て記録しました**

記録内容:
- ✅ イグジット問題の詳細調査結果
- ✅ ストップロス・利確・トレーリングストップの問題点
- ✅ リアルトレードとの比較
- ✅ 修正方針（Phase 2）
- ✅ 質問への回答
- ✅ 次のステップ（詳細な修正設計は次のチャット）

### 質問3: 他の戦略クラスの扱い

**Q: VWAP_Breakout.pyの修正完了後、他の8戦略も同様に修正しますが、一つ一つの戦略ごとにdocs/Lookhead bias problemにファイルを作成して修正していくつもりです**

**A: 推奨するアプローチ**

**戦略ごとのドキュメント構成**:
```
docs/Lookhead bias problem/
├── INVESTIGATION_REPORT.md          （エントリー編、完了済み）
├── EXIT_INVESTIGATION_REPORT.md     （イグジット編、本ドキュメント）
├── VWAP_Breakout_IMPLEMENTATION.md  （Phase 1 + Phase 2実装記録）
├── Breakout_INVESTIGATION.md        （Breakout戦略の調査）
├── Breakout_IMPLEMENTATION.md       （Breakout戦略の実装記録）
├── contrarian_INVESTIGATION.md      （Contrarian戦略の調査）
├── contrarian_IMPLEMENTATION.md     （Contrarian戦略の実装記録）
├── ...（他の戦略も同様）
└── SUMMARY.md                       （全戦略の修正完了後のまとめ）
```

**各戦略のドキュメントに含めるべき内容**:
1. **INVESTIGATION.md**（調査編）:
   - 戦略固有のbacktest()実装確認
   - エントリー価格の取得方法
   - イグジット条件の確認
   - 問題箇所の特定

2. **IMPLEMENTATION.md**（実装編）:
   - Phase 1修正（エントリー価格）
   - Phase 2修正（イグジット価格）
   - 修正前後のバックテスト比較
   - 検証結果

**修正の優先順位**:
1. **VWAP_Breakout.py**（最優先、参考実装）
2. **Breakout.py**（類似戦略）
3. ✅ **contrarian_strategy.py**（Phase 1b修正完了 - 2025-12-23）
   - 詳細調査報告書: [contrarian_strategy_EXIT_INVESTIGATION.md](contrarian_strategy_EXIT_INVESTIGATION.md)
   - 修正内容: Line 224（current_price=翌日始値）、Line 238（高値更新=当日高値）
   - 検証完了: test_20251223_contrarian_exit_price_verification.py（全10項目成功）
4. **Momentum_Investing.py**
5. **Opening_Gap.py**
6. **mean_reversion_strategy.py**
7. **pairs_trading_strategy.py**
8. **support_resistance_contrarian_strategy.py**
9. **gc_strategy.py**

**メリット**:
- 各戦略の特性を詳細に記録
- 修正の進捗を明確化
- 後から見返しやすい
- 問題の再発防止

---

## セルフチェック

### a) 見落としチェック ✅

**実行した項目**:
- ✅ generate_exit_signal()全体の確認（Line 330-395）
- ✅ ストップロス判定の確認（Line 357）
- ✅ 利益確定判定の確認（Line 361）
- ✅ トレーリングストップの確認（Line 367）
- ✅ RSI/MACD反転の確認（Line 387-393）
- ✅ リアルトレードとの比較
- ✅ エントリー修正との整合性確認

**実行していない項目**:
- ⏳ VWAP下抜けの詳細確認（Line 350）
- ⏳ 部分利確ロジックの確認（Line 374-382）
- ⏳ 他の戦略クラスの調査（各戦略ごとに別途実施予定）

**データの流れ**:
- ✅ generate_exit_signal() → current_price使用 → 問題特定完了
- ✅ イグジット価格 → CSV出力（推測、未確認）

### b) 思い込みチェック ✅

**確認した事実**:
- ✅ Line 344: `current_price = self.data[self.price_column].iloc[idx]`（コード確認）
- ✅ Line 367: `self.data['High'].iloc[entry_idx:idx+1].max()`（コード確認、idx+1含む）
- ✅ インジケーターは`.shift(1)`適用済み（git diff確認済み）
- ✅ リアルトレードでの当日高値/安値の使用可否（一般的な理解）

**排除した思い込み**:
- ❌ 「イグジット価格は問題ない」 → 当日終値を使用している
- ❌ 「トレーリングストップは正しい」 → 当日高値を使用している
- ❌ 「高値/安値は使えない」 → 保守的に使用可能

---

## 次のステップ

### 1. 次のチャット: 詳細な修正設計

**Phase 1修正（エントリー価格）**:
- base_strategy.py Line 242の修正
- VWAP_Breakout.py backtest()のエントリー部分修正
- ループ範囲の調整（`range(len(self.data) - 1)`）

**Phase 2修正（イグジット価格）**:
- generate_exit_signal()の全ロジック修正
- ストップロス・利確・トレーリングストップ
- RSI/MACD反転
- イグジット価格管理の実装

**Phase 3修正（スリッページ・取引コスト）**:
- パラメータ化
- エントリー・イグジット両方に適用

### 2. 実装順序

1. **Phase 1実装**: VWAP_Breakout.pyのエントリー価格修正
2. **検証1**: エントリー価格のみ修正後のバックテスト実行
3. **Phase 2実装**: VWAP_Breakout.pyのイグジット価格修正
4. **検証2**: Phase 1 + Phase 2修正後のバックテスト実行
5. **Phase 3実装**: スリッページ・取引コスト追加
6. **最終検証**: 全修正後のバックテスト実行

### 3. 検証コマンド

```bash
# 2025-01-15 〜 2025-01-31で検証
python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-31
```

**検証項目**:
- エントリー価格が翌日始値近辺か
- イグジット価格が適切か（ストップロス価格、利確価格近辺）
- 取引件数の変化
- リターン率の変化

### 4. 他の戦略への展開

**VWAP_Breakout.py修正完了後**:
- 修正内容を参考に他の8戦略を順次修正
- 各戦略ごとに調査・実装ドキュメント作成
- `docs/Lookhead bias problem/` に蓄積

---

## 付録

### 証拠ファイル

1. **本ドキュメント**: `EXIT_INVESTIGATION_REPORT.md`
2. **エントリー編**: `INVESTIGATION_REPORT.md`
3. **修正対象**: `strategies/VWAP_Breakout.py`

### 参考資料

- [copilot-instructions.md](.github/copilot-instructions.md) - ルックアヘッドバイアス禁止ルール
- [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) - エントリー編調査結果

---

**報告書作成者**: GitHub Copilot  
**最終更新日**: 2025-12-21  
**バージョン**: 1.0  
**ステータス**: Phase 2設計待ち（次のチャットで詳細設計実施）

---

## 新たな問題・課題

### 発見した問題

**1. VWAP下抜けの判定（Line 350）**
- 現状: `current_price < vwap` → 当日終値 vs VWAP
- VWAPは`.shift(1)`適用済み → 前日VWAP
- 問題: 当日終値で判定
- 対応: 詳細調査が必要（Phase 2実装時に確認）

**2. 部分利確ロジック（Line 374-382）**
- 現状: `profit_pct >= partial_exit_threshold`で判定
- `profit_pct = (current_price - entry_price) / entry_price`
- 問題: 当日終値ベースの判定
- 対応: Phase 2実装時に修正方針を決定

**3. イグジット価格の管理方法**
- 現在、Exit_Priceカラムへの記録方法が不明
- backtest()メソッドでの処理が必要
- strategy_execution_manager.pyでの取得方法も要確認

**4. ループ範囲の調整**
- エントリー: `range(len(self.data) - 1)`に変更必要
- イグジット: RSI/MACD反転で翌日始値使用時も`idx + 1`アクセス
- 最終日の処理に注意が必要

### 課題

**1. 日足データの限界**
- 高値と安値の発生順序が不明
- ストップロスと利確が同日に発動する可能性
- → 保守的な仮定（ストップロス優先）が必要

**2. 検証の複雑性**
- Phase 1とPhase 2の相互影響
- 修正前後の比較が複雑
- → 段階的な検証が必要

**3. 他の戦略への展開**
- 各戦略の実装パターンが異なる
- 工数見積もり: 各戦略2-3時間 × 9戦略 = 18-27時間

---

**次のチャットで実施すること**:
1. Phase 1修正の詳細設計
2. Phase 2修正の詳細設計
3. 実装計画の策定
4. 検証方法の詳細化