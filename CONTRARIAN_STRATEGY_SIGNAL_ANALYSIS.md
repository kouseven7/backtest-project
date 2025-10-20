# contrarian_strategy.py シグナル表示詳細調査レポート

**調査日**: 2025年10月20日  
**調査対象**: `strategies/contrarian_strategy.py`  
**ファイルパス**: `c:\Users\imega\Documents\my_backtest_project\strategies\contrarian_strategy.py`

---

## 📊 調査結果サマリー

### ✅ **一貫性評価: 良好 (GOOD)**

`contrarian_strategy.py`は、プロジェクト全体の標準シグナル規約に**完全に準拠**しています。

---

## 🔍 詳細分析

### 1️⃣ エントリーシグナル (Entry_Signal)

#### **generate_entry_signal() メソッド**

**シグナル値の定義**:
```python
# Line 91-140
def generate_entry_signal(self, idx: int) -> int:
    """エントリーシグナルを生成する。"""
    
    # シグナルなし
    if idx < 5:
        return 0  # ← Line 96
    
    # トレンドフィルター不合格
    if trend not in self.params["allowed_trends"]:
        return 0  # ← Line 124
    
    # 従来のトレンド判定不合格
    if not range_market:
        return 0  # ← Line 130
    
    # ✅ エントリー条件1: RSI過売り + ギャップダウン
    if rsi <= self.params["rsi_oversold"] and gap_down:
        self.entry_prices[idx] = current_price
        return 1  # ← Line 135 エントリーシグナル
    
    # ✅ エントリー条件2: ピンバー形成
    if pin_bar:
        self.entry_prices[idx] = current_price
        return 1  # ← Line 138 エントリーシグナル
    
    # デフォルト: シグナルなし
    return 0  # ← Line 140
```

#### **シグナル値の使用箇所**

| 行番号 | コード | 意味 | 値 |
|--------|--------|------|-----|
| 96 | `return 0` | データ不足 | `0` |
| 124 | `return 0` | トレンド不合格 | `0` |
| 130 | `return 0` | レンジ相場でない | `0` |
| **135** | **`return 1`** | **エントリー（RSI+ギャップダウン）** | **`1`** ✅ |
| **138** | **`return 1`** | **エントリー（ピンバー）** | **`1`** ✅ |
| 140 | `return 0` | シグナルなし | `0` |

#### **DataFrame格納時の処理**

```python
# Line 209-211 (backtest()メソッド内)
entry_signal = self.generate_entry_signal(idx)
if entry_signal == 1:  # ← 1と比較
    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1  # ← 1を格納
```

**結論**: ✅ **完全に統一されている**
- 戻り値: `1` (エントリー), `0` (なし)
- DataFrame格納値: `1` または `0`
- 判定条件: `== 1`

---

### 2️⃣ イグジットシグナル (Exit_Signal)

#### **generate_exit_signal() メソッド**

**シグナル値の定義**:
```python
# Line 142-196
def generate_exit_signal(self, idx: int) -> int:
    """イグジットシグナルを生成する。"""
    
    # データ不足
    if idx < 1:
        return 0  # ← Line 147
    
    # ✅ ポジション状態管理
    current_entries = (self.data['Entry_Signal'].iloc[:idx+1] == 1).sum()  # ← Line 151
    current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())  # ← Line 152
    
    # アクティブなポジションがない
    if current_entries <= current_exits:
        return 0  # ← Line 156
    
    # エントリー情報取得失敗
    entry_indices = self.data[self.data['Entry_Signal'] == 1].index  # ← Line 159
    if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
        return 0  # ← Line 161
    
    # エントリー価格取得失敗
    if entry_price is None:
        return 0  # ← Line 166
    
    # ✅ イグジット条件1: RSI中立域到達
    if current_rsi >= self.params["rsi_exit_level"]:
        return -1  # ← Line 173 イグジットシグナル
    
    # ✅ イグジット条件2: トレーリングストップ
    if current_price <= trailing_stop_price:
        return -1  # ← Line 181 イグジットシグナル
    
    # ✅ イグジット条件3: 利益確定
    if current_price >= entry_price * (1.0 + self.params["take_profit"]):
        return -1  # ← Line 185 イグジットシグナル
    
    # ✅ イグジット条件4: ストップロス
    if current_price <= entry_price * (1.0 - self.params["stop_loss"]):
        return -1  # ← Line 189 イグジットシグナル
    
    # ✅ イグジット条件5: 最大保有日数
    if days_held >= self.params["max_hold_days"]:
        return -1  # ← Line 194 イグジットシグナル
    
    # デフォルト: シグナルなし
    return 0  # ← Line 196
```

#### **シグナル値の使用箇所**

| 行番号 | コード | 意味 | 値 |
|--------|--------|------|-----|
| 147 | `return 0` | データ不足 | `0` |
| 156 | `return 0` | ポジションなし | `0` |
| 161 | `return 0` | エントリー情報なし | `0` |
| 166 | `return 0` | エントリー価格なし | `0` |
| **173** | **`return -1`** | **イグジット（RSI中立域）** | **`-1`** ✅ |
| **181** | **`return -1`** | **イグジット（トレーリングストップ）** | **`-1`** ✅ |
| **185** | **`return -1`** | **イグジット（利益確定）** | **`-1`** ✅ |
| **189** | **`return -1`** | **イグジット（ストップロス）** | **`-1`** ✅ |
| **194** | **`return -1`** | **イグジット（最大保有日数）** | **`-1`** ✅ |
| 196 | `return 0` | シグナルなし | `0` |

#### **DataFrame格納時の処理**

```python
# Line 214-215 (backtest()メソッド内)
exit_signal = self.generate_exit_signal(idx)
if exit_signal == -1:  # ← -1と比較
    self.data.at[self.data.index[idx], 'Exit_Signal'] = -1  # ← -1を格納
```

**結論**: ✅ **完全に統一されている**
- 戻り値: `-1` (イグジット), `0` (なし)
- DataFrame格納値: `-1` または `0`
- 判定条件: `== -1`

---

### 3️⃣ ポジション管理機能

#### **✅ 優れた実装ポイント**

```python
# Line 151-156
# ✅ ポジション状態管理を追加
current_entries = (self.data['Entry_Signal'].iloc[:idx+1] == 1).sum()
current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())

# アクティブなポジションがない場合はエグジット不可
if current_entries <= current_exits:
    return 0
```

**特徴**:
1. エントリー・エグジット回数を常に追跡
2. アクティブなポジションがない場合は**イグジットシグナルを出さない**
3. 同日のエントリー/イグジット問題を回避

**評価**: ✅ **EXCELLENT** - 他の戦略の模範となる実装

---

### 4️⃣ backtest() メソッドの実装

```python
# Line 198-217
def backtest(self):
    """バックテストを実行する。"""
    self.data['Entry_Signal'] = 0
    self.data['Exit_Signal'] = 0

    for idx in range(len(self.data)):
        # エントリーシグナル
        if not self.data['Entry_Signal'].iloc[max(0, idx - 1):idx + 1].any():
            entry_signal = self.generate_entry_signal(idx)
            if entry_signal == 1:  # ← エントリーシグナル判定
                self.data.at[self.data.index[idx], 'Entry_Signal'] = 1

        # イグジットシグナル
        exit_signal = self.generate_exit_signal(idx)
        if exit_signal == -1:  # ← イグジットシグナル判定
            self.data.at[self.data.index[idx], 'Exit_Signal'] = -1

    return self.data
```

#### **一貫性チェック**

| 項目 | 実装 | 標準規約 | 評価 |
|------|------|---------|------|
| エントリーシグナル値 | `1` | `1` | ✅ |
| イグジットシグナル値 | `-1` | `-1` | ✅ |
| シグナルなし | `0` | `0` | ✅ |
| エントリー判定条件 | `== 1` | `== 1` | ✅ |
| イグジット判定条件 | `== -1` | `== -1` | ✅ |
| DataFrame格納値 | 戻り値と同じ | 戻り値と同じ | ✅ |

**結論**: ✅ **完全に標準に準拠**

---

## 📋 シグナル値一覧表

### **エントリーシグナル**

| シグナル値 | 意味 | 使用箇所 |
|-----------|------|---------|
| `1` | エントリー | `generate_entry_signal()` 戻り値, DataFrame格納値 |
| `0` | シグナルなし | デフォルト戻り値 |

### **イグジットシグナル**

| シグナル値 | 意味 | 使用箇所 |
|-----------|------|---------|
| `-1` | イグジット | `generate_exit_signal()` 戻り値, DataFrame格納値 |
| `0` | シグナルなし | デフォルト戻り値 |

### **判定条件**

| 判定内容 | コード | 行番号 |
|---------|--------|--------|
| エントリー確認 | `self.data['Entry_Signal'].iloc[:idx+1] == 1` | 151, 159, 209 |
| イグジット確認 | `self.data['Exit_Signal'].iloc[:idx+1] == -1` | 152 |
| エントリーシグナル判定 | `if entry_signal == 1:` | 210 |
| イグジットシグナル判定 | `if exit_signal == -1:` | 214 |

---

## 🎯 評価・推奨事項

### ✅ **優れている点**

1. **シグナル値の完全な統一**
   - エントリー: `1`
   - イグジット: `-1`
   - シグナルなし: `0`
   - 標準規約に完全準拠

2. **ポジション管理の実装**
   - エントリー/イグジット回数の追跡
   - アクティブポジションの確認
   - 同日エントリー/イグジット問題の回避

3. **判定ロジックの明確性**
   - `== 1` (エントリー)
   - `== -1` (イグジット)
   - 一貫した判定条件

4. **コードの可読性**
   - コメントが適切
   - 条件分岐が明確
   - 各イグジット条件にラベル付け

### 📝 **推奨事項**

#### **現状維持を推奨**
- 現在の実装は標準規約に完全準拠
- 修正の必要なし
- 他の戦略の参考実装として推奨

#### **ドキュメント化の提案**
```python
# 戦略クラスのdocstringに追加推奨
"""
Signal Convention:
    Entry Signal: 1 (Buy), 0 (No signal)
    Exit Signal: -1 (Sell), 0 (No signal)
    
Position Management:
    - Tracks active positions
    - Prevents duplicate entries
    - Ensures entry/exit balance
"""
```

---

## 🔄 他戦略との比較

### **標準準拠度の比較**

| 戦略名 | エントリーシグナル | イグジットシグナル | 一貫性 |
|--------|-------------------|-------------------|--------|
| **contrarian_strategy.py** | ✅ `1` | ✅ `-1` | ✅ **GOOD** |
| base_strategy.py | ✅ `1` | ✅ `-1` | ✅ GOOD |
| gc_strategy_signal.py | ✅ `1` | ✅ `-1` | ✅ GOOD |
| Momentum_Investing.py | ✅ `1` | ✅ `-1` | ✅ GOOD |
| mean_reversion_strategy.py | ✅ `1` | ❌ `1` | ❌ **BAD** |
| pairs_trading_strategy.py | ✅ `1` | ❌ `1` | ❌ **BAD** |

**結論**: `contrarian_strategy.py`は**標準グループ**に属し、修正不要

---

## 📊 戦略内部の一貫性

### **generate_entry_signal() 内の一貫性**

```python
✅ すべての return 文で統一
return 0  # シグナルなし
return 1  # エントリー
```

### **generate_exit_signal() 内の一貫性**

```python
✅ すべての return 文で統一
return 0   # シグナルなし
return -1  # イグジット
```

### **backtest() 内の一貫性**

```python
✅ 判定条件と格納値が統一
if entry_signal == 1:
    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1

if exit_signal == -1:
    self.data.at[self.data.index[idx], 'Exit_Signal'] = -1
```

**結論**: ✅ **戦略内部で完全に一貫している**

---

## 🏁 最終評価

### **総合評価: ⭐⭐⭐⭐⭐ (5/5)**

| 評価項目 | スコア | コメント |
|---------|--------|---------|
| シグナル値の統一 | ⭐⭐⭐⭐⭐ | 標準規約に完全準拠 |
| ポジション管理 | ⭐⭐⭐⭐⭐ | 優れた実装 |
| コードの可読性 | ⭐⭐⭐⭐⭐ | 非常に明確 |
| 一貫性 | ⭐⭐⭐⭐⭐ | 完璧に統一 |
| ドキュメント | ⭐⭐⭐⭐ | 良好（さらに改善可能） |

### **結論**

✅ **修正不要 - 標準実装として推奨**

`contrarian_strategy.py`は、プロジェクト全体のシグナル規約に完全に準拠しており、他の戦略開発時の**参考実装（ベストプラクティス）**として活用できます。

---

## 📎 添付情報

- **調査ファイル**: `contrarian_strategy.py`
- **総行数**: 256行
- **調査対象メソッド**:
  - `generate_entry_signal()` (Line 91-140)
  - `generate_exit_signal()` (Line 142-196)
  - `backtest()` (Line 198-217)
- **検出されたシグナル値**:
  - エントリー: `1` (2箇所)
  - イグジット: `-1` (5箇所)
  - シグナルなし: `0` (多数)

---

**調査完了**: 2025年10月20日  
**調査者**: GitHub Copilot AI Assistant  
**評価**: ✅ **EXCELLENT - 標準準拠**
