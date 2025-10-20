# エントリー・イグジットシグナル表示一貫性調査レポート

**調査日**: 2025年10月20日  
**調査対象**: 全戦略ファイルのシグナル表示形式  
**調査目的**: シグナル表示の不一致による後続モジュールへの影響確認

---

## 📊 調査結果サマリー

### ✅ 一貫性が保たれている項目
- **エントリーシグナルの戻り値**: 全戦略で `1` を使用（統一）
- **イグジットシグナルの戻り値**: 全戦略で `-1` を使用（統一）
- **シグナルなしの戻り値**: 全戦略で `0` を使用（統一）

### ⚠️ 不一致が見つかった項目

#### **重大な不一致 (CRITICAL)**

1. **イグジットシグナルのDataFrame格納値**
   - **問題の詳細**: 戦略によってDataFrameに格納する値が異なる
   - **影響度**: 🔴 **HIGH** - 後続モジュールでのシグナル判定に直接影響

2. **戦略内部でのシグナル判定値**
   - **問題の詳細**: `generate_exit_signal()`の戻り値と`backtest()`での判定値が戦略間で不統一

---

## 🔍 詳細調査結果

### 1️⃣ エントリーシグナル (Entry_Signal)

#### ✅ 統一されている点
```python
# 全戦略で共通
generate_entry_signal() の戻り値: 1 (エントリー), 0 (なし)
DataFrame への格納値: 1 または 0
```

#### 戦略別の実装パターン

| 戦略名 | `generate_entry_signal()`戻り値 | DataFrame格納値 | 判定コード | 一貫性 |
|--------|--------------------------------|----------------|-----------|--------|
| **base_strategy.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **Breakout.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **contrarian_strategy.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **enhanced_base_strategy.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **gc_strategy_signal.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **mean_reversion_strategy.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **Momentum_Investing.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **Opening_Gap.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **Opening_Gap_Enhanced.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **Opening_Gap_Fixed.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **VWAP_Bounce.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **VWAP_Breakout.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **pairs_trading_strategy.py** | `1` / `0` | `1` / `0` | `== 1` | ✅ |
| **strategy_manager.py** | - | `1` / `0` | `== 1` | ✅ |

**結論**: ✅ **エントリーシグナルは完全に統一されている**

---

### 2️⃣ イグジットシグナル (Exit_Signal)

#### ⚠️ 不一致が存在する点

##### 🔴 **問題1: DataFrame格納値の不一致**

| 戦略名 | `generate_exit_signal()`戻り値 | DataFrame格納値 | 判定コード | 一貫性 |
|--------|--------------------------------|----------------|-----------|--------|
| **base_strategy.py** | `-1` / `0` | `-1` / `0` | なし | ✅ |
| **Breakout.py** | `-1` / `0` | `-1` / `0` | `== -1` | ✅ |
| **contrarian_strategy.py** | `-1` / `0` | `-1` / `0` | `== -1` | ✅ |
| **enhanced_base_strategy.py** | `-1` / `0` | **`exit_signal`変数をそのまま格納** | `!= 0` | ⚠️ **注意** |
| **gc_strategy_signal.py** | `-1` / `0` | `-1` / `0` | `== -1` | ✅ |
| **mean_reversion_strategy.py** | **`1` / `0`** | **`1` / `0`** | **不明** | ❌ **不一致** |
| **Momentum_Investing.py** | `-1` / `0` | `-1` / `0` | `== -1` | ✅ |
| **Opening_Gap.py** | `-1` / `0` | `-1` / `0` | `== -1` | ✅ |
| **Opening_Gap_Enhanced.py** | `-1` / `0` | `-1` / `0` | なし | ✅ |
| **Opening_Gap_Fixed.py** | `-1` / `0` | `-1` / `0` | `== -1` | ✅ |
| **VWAP_Bounce.py** | `-1` / `0` | `-1` / `0` | `== -1` | ✅ |
| **VWAP_Breakout.py** | `-1` / `0` | `-1` / `0` | `== -1` | ✅ |
| **pairs_trading_strategy.py** | **`1` / `0`** | **`1` / `0`** | **不明** | ❌ **不一致** |
| **strategy_manager.py** | - | `-1` / `0` | `== -1` | ✅ |

##### 🔴 **問題2: enhanced_base_strategy.pyでの特殊な実装**

```python
# enhanced_base_strategy.py Line 103-104
if exit_signal != 0:  # 0以外のイグジットシグナル（-1など）
    result.at[result.index[idx], 'Exit_Signal'] = exit_signal
```

**問題点**:
- `exit_signal`変数の値をそのまま格納
- コメントでは「-1など」と記載されているが、実際には`generate_exit_signal()`の戻り値がそのまま格納される
- もし他の値（例: `-2`, `1`など）が返される可能性がある場合、予期しない動作を引き起こす

---

### 3️⃣ 重大な不一致の詳細

#### ❌ **mean_reversion_strategy.py**

**ファイル**: `c:\Users\imega\Documents\my_backtest_project\strategies\mean_reversion_strategy.py`

```python
# Line 215-223
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
    # ...
    if hold_days >= self.params["max_hold_days"]:
        return 1  # 最大保有日数到達 ← ⚠️ -1ではなく1を返す
        
    if pnl_pct <= -self.params["stop_loss_pct"]:
        return 1  # ← ⚠️ -1ではなく1を返す
        
    if pnl_pct >= self.params["take_profit_pct"]:
        return 1  # ← ⚠️ -1ではなく1を返す
```

**問題点**:
- イグジットシグナルとして`1`を返している
- 他の全戦略は`-1`を返している
- **エントリーシグナルと同じ値**を使用しているため、混同のリスクがある

**影響範囲**:
- `backtest()`メソッド内でのシグナル判定
- 後続のポジション管理モジュール
- シグナル集計・分析モジュール

---

#### ❌ **pairs_trading_strategy.py**

**ファイル**: `c:\Users\imega\Documents\my_backtest_project\strategies\pairs_trading_strategy.py`

```python
# Line 229-248
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
    # ...
    if hold_days >= self.params["max_hold_days"]:
        return 1  # 最大保有日数到達 ← ⚠️ -1ではなく1を返す
        
    if pnl_pct <= -self.params["stop_loss_pct"]:
        return 1  # ← ⚠️ -1ではなく1を返す
        
    if pnl_pct >= self.params["take_profit_pct"]:
        return 1  # ← ⚠️ -1ではなく1を返す
        
    if abs(spread_zscore) <= exit_threshold:
        return 1  # 回帰完了でエグジット ← ⚠️ -1ではなく1を返す
```

**問題点**:
- `mean_reversion_strategy.py`と同様にイグジットシグナルとして`1`を返している
- 他の戦略との一貫性がない

---

### 4️⃣ 戦略内部での不一致

#### ⚠️ **strategy_manager.py**

**ファイル**: `c:\Users\imega\Documents\my_backtest_project\strategies\strategy_manager.py`

```python
# Line 114-116 (イグジット判定)
if exit_signal == -1:
    # イグジットが出たら記録
    stock_data.at[current_date, 'Exit_Signal'] = -1

# Line 140 (カウント)
exit_count = (stock_data['Exit_Signal'] == -1).sum()
```

**問題点**:
- `strategy_manager.py`は全戦略のイグジットシグナルが`-1`であることを前提にしている
- `mean_reversion_strategy.py`や`pairs_trading_strategy.py`が`1`を返す場合、**シグナルが正しく検出されない**

---

## 🎯 影響分析

### 影響を受けるモジュール

#### 🔴 **直接的影響 (HIGH PRIORITY)**

1. **strategy_manager.py**
   - イグジットシグナルの判定: `== -1`を前提としている
   - `mean_reversion_strategy.py`と`pairs_trading_strategy.py`のシグナルを見逃す可能性

2. **ポジション管理モジュール**
   - イグジット判定が正しく機能しない
   - ポジションが解放されずに残る可能性

3. **パフォーマンス計算モジュール**
   - トレード回数のカウントが不正確になる
   - 損益計算に誤差が生じる

#### 🟡 **間接的影響 (MEDIUM PRIORITY)**

4. **レポート生成モジュール**
   - シグナル統計が不正確
   - エントリー/イグジット回数の不一致

5. **バックテスト結果の信頼性**
   - 不一致により結果の妥当性に疑問

---

## 📋 推奨される修正方針

### 🎯 **標準化案**

#### **Option 1: `-1` に統一（推奨）**

**理由**:
- 大多数の戦略が既に`-1`を使用している
- エントリーシグナル(`1`)との区別が明確
- 修正対象が2戦略のみで影響範囲が小さい

**修正対象**:
- `mean_reversion_strategy.py`
- `pairs_trading_strategy.py`

**修正内容**:
```python
# 修正前
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
    if condition:
        return 1  # ← これを修正

# 修正後
def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
    if condition:
        return -1  # ← -1に変更
```

#### **Option 2: `1` に統一（非推奨）**

**理由**:
- 修正対象が多数（10戦略以上）
- エントリーシグナルとの区別が曖昧
- 混同リスクが高い

---

### 🔧 **修正優先度**

| 優先度 | 対象ファイル | 修正内容 | 影響度 |
|--------|-------------|---------|--------|
| 🔴 **P0** | `mean_reversion_strategy.py` | `return 1` → `return -1` | HIGH |
| 🔴 **P0** | `pairs_trading_strategy.py` | `return 1` → `return -1` | HIGH |
| 🟡 **P1** | `enhanced_base_strategy.py` | 判定ロジックの明確化 | MEDIUM |

---

## 📝 追加確認事項

### 確認が必要な項目

1. **mean_reversion_strategy.pyのbacktest()メソッド**
   - イグジットシグナル`1`を前提とした実装があるか確認
   
2. **pairs_trading_strategy.pyのbacktest()メソッド**
   - 同様の実装確認

3. **テストコード**
   - 既存のテストが`1`を前提にしている可能性

---

## 🏁 結論

### ✅ **一貫性が保たれている部分**
- エントリーシグナル: 完全に統一（全戦略で`1`を使用）
- シグナルなし: 完全に統一（全戦略で`0`を使用）

### ❌ **不一致が存在する部分**
- イグジットシグナル: 
  - **10戦略が `-1` を使用**
  - **2戦略が `1` を使用** ← 🔴 要修正

### 🎯 **推奨アクション**

1. **即時対応が必要**:
   - `mean_reversion_strategy.py`のイグジットシグナルを`-1`に統一
   - `pairs_trading_strategy.py`のイグジットシグナルを`-1`に統一

2. **確認・テストが必要**:
   - 修正後の動作確認
   - 既存のテストコードの更新
   - `backtest()`メソッド内の判定ロジック確認

3. **ドキュメント化**:
   - シグナル規約をドキュメント化
   - 今後の新規戦略開発時の参照用

---

## 📊 シグナル規約（推奨）

### **標準シグナル値**

```python
# エントリーシグナル
ENTRY_SIGNAL = 1     # エントリー
NO_ENTRY = 0         # エントリーなし

# イグジットシグナル
EXIT_SIGNAL = -1     # イグジット（標準）
NO_EXIT = 0          # イグジットなし

# ⚠️ 非推奨（混同を避けるため）
# EXIT_SIGNAL = 1    # エントリーと同じ値なので非推奨
```

### **DataFrameカラム命名規則**

```python
# 推奨される列名
'Entry_Signal'   # エントリーシグナル (1 or 0)
'Exit_Signal'    # イグジットシグナル (-1 or 0)
'Position_Size'  # ポジションサイズ (0.0 ~ 1.0)
'Strategy'       # 戦略名（文字列）
```

---

**調査完了**: 2025年10月20日  
**調査者**: GitHub Copilot AI Assistant  
**レポート形式**: Markdown

---

## 📎 添付資料

- 調査対象ファイル一覧: 13戦略ファイル
- 検出された grep パターン: 3種類
- 確認した行数: 約5,000行
