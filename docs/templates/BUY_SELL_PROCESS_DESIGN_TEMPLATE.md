# BUY/SELL処理設計テンプレート

**目的**: BUY/SELL処理の実装漏れを防止する標準設計フォーマット  
**適用対象**: トレーディングシステム、バックテストエンジン全般  
**作成日**: 2026-02-10  
**参考**: KNOWN_ISSUES_AND_PREVENTION.md Issue #7

---

## 📋 **設計概要**

### **機能名**
[例: DSSMS統合バックテスト BUY/SELL処理]

### **目的**
[例: 複数銘柄保有対応のBUY/SELL処理実装]

### **責務**
- [ ] BUY注文実行
- [ ] SELL注文実行
- [ ] ポジション状態管理
- [ ] 資金残高管理
- [ ] 取引履歴記録

---

## 🎯 **BUY処理設計**

### **メソッド名**
[例: `_execute_multi_strategies_daily()` または `_execute_buy()`]

### **実装チェックリスト**

#### 必須項目（4項目全てクリア必須）
- [ ] **1. 資金残高更新**: `self.cash_balance -= trade_cost`
- [ ] **2. ポジション追加**: `self.positions[symbol] = {...}`
- [ ] **3. 取引履歴記録**: `execution_details.append(...)`
- [ ] **4. ログ出力**: `self.logger.info("[POSITION_ADD] ...")`

#### データ構造仕様
```python
# self.positions[symbol]のデータ構造
{
    'symbol': str,           # 銘柄コード（例: '9101.T'）
    'strategy': str,         # 戦略名（例: 'BreakoutStrategy'）
    'entry_price': float,    # エントリー価格
    'shares': int,           # 株数
    'entry_date': str,       # エントリー日（YYYY-MM-DD）
    'entry_idx': int         # データフレームインデックス（Noneも可）
}
```

#### execution_details記録形式
```python
execution_details.append({
    'date': str,             # 実行日（YYYY-MM-DD）
    'action': 'buy',         # 固定値
    'symbol': str,           # 銘柄コード
    'price': float,          # 約定価格
    'shares': int,           # 株数
    'strategy': str,         # 戦略名
    'cash_balance': float,   # 取引後の残高
    # その他必要な情報
})
```

#### ログ出力形式
```python
self.logger.info(
    f"[POSITION_ADD] ポジション追加: {symbol}, "
    f"価格={price:.2f}円, 株数={shares}株, 戦略={strategy_name}"
)
```

### **エラーハンドリング**
- [ ] 資金不足チェック（`if self.cash_balance < trade_cost:`）
- [ ] max_positionsチェック（`if len(self.positions) >= self.max_positions:`）
- [ ] 既存ポジション重複チェック（`if symbol in self.positions:`）

### **推奨実装パターン**

#### パターンA: インライン実装（小規模）
```python
if result['action'] == 'buy':
    trade_cost = result['price'] * result['shares']
    
    # 1. 資金残高更新
    self.cash_balance -= trade_cost
    
    # 2. ポジション追加
    self.positions[symbol] = {
        'symbol': symbol,
        'strategy': best_strategy_name,
        'entry_price': result['price'],
        'shares': result['shares'],
        'entry_date': adjusted_target_date,
        'entry_idx': None,
    }
    
    # 3. 取引履歴記録
    execution_details.append({
        'date': adjusted_target_date,
        'action': 'buy',
        'symbol': symbol,
        'price': result['price'],
        'shares': result['shares'],
        'strategy': best_strategy_name,
        'cash_balance': self.cash_balance,
    })
    
    # 4. ログ出力
    self.logger.info(
        f"[POSITION_ADD] ポジション追加: {symbol}, "
        f"価格={result['price']:.2f}円, 株数={result['shares']}株, "
        f"戦略={best_strategy_name}"
    )
```

#### パターンB: メソッド分離（大規模・再利用）
```python
def _execute_buy(self, symbol, price, shares, strategy_name, target_date):
    """BUY注文実行（cash_balance更新、positions追加、ログ記録を一括処理）
    
    Args:
        symbol: 銘柄コード
        price: 約定価格
        shares: 株数
        strategy_name: 戦略名
        target_date: 実行日（YYYY-MM-DD）
    
    Returns:
        float: 取引コスト
    
    Raises:
        ValueError: 資金不足、またはポジション上限超過時
    """
    trade_cost = price * shares
    
    # エラーチェック
    if self.cash_balance < trade_cost:
        raise ValueError(f"資金不足: 必要={trade_cost}, 残高={self.cash_balance}")
    if len(self.positions) >= self.max_positions:
        raise ValueError(f"ポジション上限超過: {len(self.positions)}/{self.max_positions}")
    if symbol in self.positions:
        self.logger.warning(f"既にポジション保有中: {symbol}")
        return 0.0
    
    # 1. 資金残高更新
    self.cash_balance -= trade_cost
    
    # 2. ポジション追加
    self.positions[symbol] = {
        'symbol': symbol,
        'strategy': strategy_name,
        'entry_price': price,
        'shares': shares,
        'entry_date': target_date,
        'entry_idx': None,
    }
    
    # 3. 取引履歴記録
    self.execution_details.append({
        'date': target_date,
        'action': 'buy',
        'symbol': symbol,
        'price': price,
        'shares': shares,
        'strategy': strategy_name,
        'cash_balance': self.cash_balance,
    })
    
    # 4. ログ出力
    self.logger.info(
        f"[POSITION_ADD] ポジション追加: {symbol}, "
        f"価格={price:.2f}円, 株数={shares}株, 戦略={strategy_name}"
    )
    
    return trade_cost
```

---

## 🎯 **SELL処理設計**

### **メソッド名**
[例: `_execute_multi_strategies_daily()` または `_execute_sell()`]

### **実装チェックリスト**

#### 必須項目（4項目全てクリア必須）
- [ ] **1. 資金残高更新**: `self.cash_balance += trade_profit`
- [ ] **2. ポジション削除**: `del self.positions[symbol]`（KeyErrorチェック実装）
- [ ] **3. 取引履歴記録**: `execution_details.append(...)`
- [ ] **4. ログ出力**: `self.logger.info("[POSITION_DELETE] ...")`

#### execution_details記録形式
```python
execution_details.append({
    'date': str,             # 実行日（YYYY-MM-DD）
    'action': 'sell',        # 固定値
    'symbol': str,           # 銘柄コード
    'price': float,          # 約定価格
    'shares': int,           # 株数
    'strategy': str,         # 戦略名
    'cash_balance': float,   # 取引後の残高
    'pnl': float,            # 損益（sell_price - entry_price）* shares
    # その他必要な情報
})
```

#### ログ出力形式
```python
self.logger.info(
    f"[POSITION_DELETE] ポジション削除: {symbol}, "
    f"PnL={pnl:+,.0f}円({return_pct:+.2%})"
)
```

### **エラーハンドリング**
- [ ] ポジション存在チェック（`if symbol in self.positions:`）
- [ ] KeyError防止処理（`if symbol not in self.positions: ... return`）
- [ ] ログ警告出力（`self.logger.warning(...)`）

### **推奨実装パターン**

#### パターンA: インライン実装（小規模）
```python
elif result['action'] == 'sell':
    trade_profit = result['price'] * result['shares']
    
    # 1. 資金残高更新
    self.cash_balance += trade_profit
    
    # 2. ポジション削除（KeyErrorチェック）
    if symbol in self.positions:
        position_entry_price = self.positions[symbol]['entry_price']
        position_shares = self.positions[symbol]['shares']
        pnl = (result['price'] - position_entry_price) * position_shares
        return_pct = (result['price'] - position_entry_price) / position_entry_price if position_entry_price > 0 else 0.0
        
        del self.positions[symbol]
        
        # 4. ログ出力
        self.logger.info(
            f"[POSITION_DELETE] ポジション削除: {symbol}, "
            f"PnL={pnl:+,.0f}円({return_pct:+.2%})"
        )
    else:
        self.logger.warning(
            f"[POSITION_DELETE] 警告: {symbol}がself.positionsに存在しません"
            f"（SELL実行されたがポジション未記録）"
        )
    
    # 3. 取引履歴記録
    execution_details.append({
        'date': adjusted_target_date,
        'action': 'sell',
        'symbol': symbol,
        'price': result['price'],
        'shares': result['shares'],
        'strategy': best_strategy_name,
        'cash_balance': self.cash_balance,
        'pnl': pnl if symbol in self.positions else 0.0,
    })
```

#### パターンB: メソッド分離（大規模・再利用）
```python
def _execute_sell(self, symbol, price, shares, target_date):
    """SELL注文実行（cash_balance更新、positions削除、ログ記録を一括処理）
    
    Args:
        symbol: 銘柄コード
        price: 約定価格
        shares: 株数
        target_date: 実行日（YYYY-MM-DD）
    
    Returns:
        float: 取引利益
    
    Raises:
        ValueError: ポジション未保有時
    """
    trade_profit = price * shares
    
    # エラーチェック
    if symbol not in self.positions:
        self.logger.warning(
            f"[POSITION_DELETE] 警告: {symbol}がself.positionsに存在しません"
        )
        # 資金残高のみ更新して終了
        self.cash_balance += trade_profit
        return trade_profit
    
    # ポジション情報取得
    position_entry_price = self.positions[symbol]['entry_price']
    position_shares = self.positions[symbol]['shares']
    strategy_name = self.positions[symbol].get('strategy', 'Unknown')
    pnl = (price - position_entry_price) * position_shares
    return_pct = (price - position_entry_price) / position_entry_price if position_entry_price > 0 else 0.0
    
    # 1. 資金残高更新
    self.cash_balance += trade_profit
    
    # 2. ポジション削除
    del self.positions[symbol]
    
    # 3. 取引履歴記録
    self.execution_details.append({
        'date': target_date,
        'action': 'sell',
        'symbol': symbol,
        'price': price,
        'shares': shares,
        'strategy': strategy_name,
        'cash_balance': self.cash_balance,
        'pnl': pnl,
    })
    
    # 4. ログ出力
    self.logger.info(
        f"[POSITION_DELETE] ポジション削除: {symbol}, "
        f"PnL={pnl:+,.0f}円({return_pct:+.2%})"
    )
    
    return trade_profit
```

---

## ✅ **実装時のレビューポイント**

### **コードレビューチェックリスト**
- [ ] BUY処理に4項目全て実装（残高、positions追加、履歴、ログ）
- [ ] SELL処理に4項目全て実装（残高、positions削除、履歴、ログ）
- [ ] KeyError防止の存在チェック（`if symbol in self.positions:`）
- [ ] ログ出力に統一フォーマット使用（`[POSITION_ADD]`, `[POSITION_DELETE]`）
- [ ] エラーハンドリング実装（資金不足、ポジション上限、未保有）
- [ ] ドキュメント文字列（docstring）記載

### **動作確認チェックリスト**
- [ ] BUY注文が正常に実行される
- [ ] SELL注文が正常に実行される
- [ ] `self.positions`が正確に更新される
- [ ] `self.cash_balance`が正確に更新される
- [ ] ログに`[POSITION_ADD]`と`[POSITION_DELETE]`が出力される
- [ ] all_transactions.csvにEXIT情報が記録される
- [ ] 強制決済が動作する（`[FINAL_CLOSE]`ログ確認）

---

## 🧪 **検証方法**

### **テスト1: ポジション追加確認**
```bash
# ログでPOSITION_ADD確認
grep "\[POSITION_ADD\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

### **テスト2: ポジション削除確認**
```bash
# ログでPOSITION_DELETE確認
grep "\[POSITION_DELETE\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

### **テスト3: 強制決済確認**
```bash
# ログでFINAL_CLOSE確認
grep "\[FINAL_CLOSE\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

### **テスト4: 取引履歴完全性確認**
```python
import pandas as pd

# all_transactions.csv読み込み
df = pd.read_csv("output/dssms_integration/dssms_*/all_transactions.csv")

# EXIT情報の完全性確認
assert df['exit_date'].notna().all(), "exit_date に空の行があります"
assert df['exit_price'].notna().all(), "exit_price に空の行があります"
assert df['pnl'].notna().all(), "pnl に空の行があります"

print(f"✅ all_transactions.csv検証成功: {len(df)}件の取引全てにEXIT情報あり")
```

### **テスト5: ポジション整合性確認**
```python
# バックテスト終了時にself.positionsが空であることを確認
assert len(self.positions) == 0, f"未決済ポジションが{len(self.positions)}件残っています"
print("✅ 全ポジション決済完了")
```

---

## 📚 **参考資料**

- [KNOWN_ISSUES_AND_PREVENTION.md](../KNOWN_ISSUES_AND_PREVENTION.md): Issue #7
- [DSSMS EXIT未記録問題調査報告](../DSSMS_EXIT_NOT_RECORDED_INVESTIGATION_20260210.md)
- [copilot-instructions.md](../../.github/copilot-instructions.md): リファクタリング・再実装時のGit履歴活用

---

## 📝 **使用方法**

1. **設計時**: このテンプレートをコピーし、プロジェクト固有の情報を記入
2. **実装時**: 実装チェックリストに従って実装
3. **レビュー時**: レビューポイントチェックリストで確認
4. **テスト時**: 検証方法に従ってテスト実行

---

**注意**: このテンプレートは、Issue #7（BUY/SELL後のself.positions管理漏れ）の教訓を基に作成されています。
新しい知見が得られた場合は、このテンプレートを更新してください。
