# ポジション管理設計テンプレート

**目的**: ポジション管理の実装漏れを防止する標準設計フォーマット  
**適用対象**: トレーディングシステム、バックテストエンジン全般  
**作成日**: 2026-02-10  
**参考**: KNOWN_ISSUES_AND_PREVENTION.md Issue #7

---

## 📋 **設計概要**

### **機能名**
[例: DSSMS複数銘柄ポジション管理]

### **目的**
[例: 最大2銘柄の同時保有、FIFO決済、強制決済対応]

### **管理対象**
- [ ] ポジション保有状態（positions辞書）
- [ ] 最大保有数制限（max_positions）
- [ ] ポジション追加・削除タイミング
- [ ] 強制決済処理

---

## 🎯 **設計の3要素**

Issue #7の教訓から、状態管理の設計には以下の3要素が必須：

### **1. 初期化（Initialization）**
- いつ、どこで、どのように初期化するか

### **2. 状態更新（State Update）**
- どこで、いつ、どのように更新するか
- **重要**: BUY/SELL実行時の更新処理を明記

### **3. 状態確認（State Verification）**
- どのように検証するか
- テスト方法、ログ確認方法

**実装例**: "BUY処理: self.cash_balance更新 + self.positions追加 + execution_details記録"

---

## 🏗️ **1. 初期化設計**

### **初期化場所**
[例: `DSSMSIntegratedMain.__init__()`]

### **初期化タイミング**
[例: クラスインスタンス生成時]

### **データ構造定義**

```python
# self.positions: 辞書型（キー: 銘柄コード、値: ポジション情報辞書）
self.positions = {}

# ポジション情報の構造
# {
#     'symbol': str,           # 銘柄コード（例: '9101.T'）
#     'strategy': str,         # 戦略名（例: 'BreakoutStrategy'）
#     'entry_price': float,    # エントリー価格
#     'shares': int,           # 株数
#     'entry_date': str,       # エントリー日（YYYY-MM-DD）
#     'entry_idx': int         # データフレームインデックス（Noneも可）
# }

# 最大保有数
self.max_positions = 2  # [プロジェクト固有の値を設定]
```

### **初期化コード例**
```python
def __init__(self, config):
    # ... 他の初期化処理 ...
    
    # ポジション管理初期化
    self.positions = {}  # 複数銘柄ポジション管理
    self.max_positions = config.get('max_positions', 2)  # デフォルト2銘柄
    
    # ログ出力
    self.logger.info(
        f"[INIT] ポジション管理初期化: max_positions={self.max_positions}"
    )
```

### **初期化チェックリスト**
- [ ] `self.positions = {}`初期化
- [ ] `self.max_positions`設定（デフォルト値あり）
- [ ] ログ出力（初期化完了メッセージ）
- [ ] ドキュメント記載（データ構造、max_positionsの意味）

---

## 🔄 **2. 状態更新設計**

### **A. ポジション追加（BUY実行時）**

#### **実行場所**
[例: `_execute_multi_strategies_daily()` BUY判定後]

#### **実行タイミング**
[例: BUY注文実行直後、self.cash_balance更新後]

#### **更新処理**
```python
# BUY実行後、必ず以下を実行
self.positions[symbol] = {
    'symbol': symbol,
    'strategy': best_strategy_name,
    'entry_price': result['price'],
    'shares': result['shares'],
    'entry_date': adjusted_target_date,
    'entry_idx': None,
}

# ログ出力必須
self.logger.info(
    f"[POSITION_ADD] ポジション追加: {symbol}, 価格={result['price']:.2f}円, "
    f"株数={result['shares']}株, 戦略={best_strategy_name}"
)
```

#### **チェックリスト**
- [ ] BUY実行直後に`self.positions[symbol]`追加
- [ ] 全フィールド（symbol, strategy, entry_price, shares, entry_date, entry_idx）設定
- [ ] ログ出力（`[POSITION_ADD]`）
- [ ] max_positionsチェック（追加前に確認）

### **B. ポジション削除（SELL実行時）**

#### **実行場所**
[例: `_execute_multi_strategies_daily()` SELL判定後]

#### **実行タイミング**
[例: SELL注文実行直後、self.cash_balance更新後]

#### **更新処理**
```python
# SELL実行後、必ず以下を実行
if symbol in self.positions:
    position_entry_price = self.positions[symbol]['entry_price']
    position_shares = self.positions[symbol]['shares']
    pnl = (result['price'] - position_entry_price) * position_shares
    return_pct = (result['price'] - position_entry_price) / position_entry_price if position_entry_price > 0 else 0.0
    
    # ポジション削除
    del self.positions[symbol]
    
    # ログ出力必須
    self.logger.info(
        f"[POSITION_DELETE] ポジション削除: {symbol}, PnL={pnl:+,.0f}円({return_pct:+.2%})"
    )
else:
    # 警告: ポジション未保有なのにSELL実行
    self.logger.warning(
        f"[POSITION_DELETE] 警告: {symbol}がself.positionsに存在しません"
        f"（SELL実行されたがポジション未記録）"
    )
```

#### **チェックリスト**
- [ ] SELL実行直後に`del self.positions[symbol]`
- [ ] KeyErrorチェック（`if symbol in self.positions:`）
- [ ] PnL計算（exit_price - entry_price）* shares
- [ ] ログ出力（`[POSITION_DELETE]`）
- [ ] ポジション未保有時の警告ログ

### **C. 強制決済（バックテスト終了時）**

#### **実行場所**
[例: `run_backtest()` の最後]

#### **実行タイミング**
[例: バックテスト期間終了後、最終日の取引実行後]

#### **更新処理**
```python
# バックテスト終了時、未決済ポジションを強制決済
if len(self.positions) > 0:
    self.logger.info(
        f"[FINAL_CLOSE] バックテスト終了: 未決済ポジション{len(self.positions)}件を強制決済"
    )
    
    for symbol, position_data in list(self.positions.items()):
        # 強制決済処理
        final_price = data.loc[data.index[-1], 'Adj Close']  # 最終日の終値
        shares = position_data['shares']
        entry_price = position_data['entry_price']
        
        # SELL処理実行
        trade_profit = final_price * shares
        self.cash_balance += trade_profit
        
        # PnL計算
        pnl = (final_price - entry_price) * shares
        
        # execution_details記録
        execution_details.append({
            'date': data.index[-1].strftime('%Y-%m-%d'),
            'action': 'sell',
            'symbol': symbol,
            'price': final_price,
            'shares': shares,
            'strategy': position_data['strategy'],
            'cash_balance': self.cash_balance,
            'pnl': pnl,
            'note': 'FINAL_CLOSE',
        })
        
        # ポジション削除
        del self.positions[symbol]
        
        # ログ出力
        self.logger.info(
            f"[FINAL_CLOSE] 強制決済: {symbol}, PnL={pnl:+,.0f}円"
        )
    
    # 全ポジション決済確認
    self.positions.clear()
    self.logger.info("[FINAL_CLOSE] 全ポジション決済完了")
```

#### **チェックリスト**
- [ ] `if len(self.positions) > 0:`チェック
- [ ] 全ポジションをループ処理（`for symbol, position_data in list(self.positions.items()):`）
- [ ] SELL処理実行（cash_balance更新、execution_details記録）
- [ ] ポジション削除（`del self.positions[symbol]`）
- [ ] ログ出力（`[FINAL_CLOSE]`）
- [ ] 最終確認（`self.positions.clear()`）

---

## ✅ **3. 状態確認設計（検証方法）**

### **A. ログ確認**

#### **ポジション追加確認**
```bash
grep "\[POSITION_ADD\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

期待結果:
```
[POSITION_ADD] ポジション追加: 9101.T, 価格=1234.56円, 株数=100株, 戦略=BreakoutStrategy
[POSITION_ADD] ポジション追加: 9104.T, 価格=2345.67円, 株数=50株, 戦略=MomentumStrategy
```

#### **ポジション削除確認**
```bash
grep "\[POSITION_DELETE\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

期待結果:
```
[POSITION_DELETE] ポジション削除: 9101.T, PnL=+12,345円(+10.00%)
[POSITION_DELETE] ポジション削除: 9104.T, PnL=-5,678円(-2.42%)
```

#### **強制決済確認**
```bash
grep "\[FINAL_CLOSE\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

期待結果:
```
[FINAL_CLOSE] バックテスト終了: 未決済ポジション2件を強制決済
[FINAL_CLOSE] 強制決済: 9101.T, PnL=+8,900円
[FINAL_CLOSE] 強制決済: 9104.T, PnL=-3,200円
[FINAL_CLOSE] 全ポジション決済完了
```

### **B. CSV確認**

#### **all_transactions.csv完全性確認**
```python
import pandas as pd

# CSV読み込み
df = pd.read_csv("output/dssms_integration/dssms_*/all_transactions.csv")

# カラム存在確認
required_columns = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'shares', 'pnl']
missing_columns = [col for col in required_columns if col not in df.columns]
assert len(missing_columns) == 0, f"カラム不足: {missing_columns}"

# EXIT情報の完全性確認
assert df['exit_date'].notna().all(), "exit_date に空の行があります"
assert df['exit_price'].notna().all(), "exit_price に空の行があります"
assert df['pnl'].notna().all(), "pnl に空の行があります"

# 取引数確認
trade_count = len(df)
entry_count = len(df[df['entry_date'].notna()])
exit_count = len(df[df['exit_date'].notna()])

assert entry_count == exit_count, f"エントリー数({entry_count})とエグジット数({exit_count})が一致しません"

print(f"✅ all_transactions.csv検証成功:")
print(f"   - 取引数: {trade_count}件")
print(f"   - 全取引にEXIT情報あり")
print(f"   - エントリー/エグジット一致")
```

### **C. 内部状態確認**

#### **バックテスト終了時の確認**
```python
# バックテスト終了時、self.positionsが空であることを確認
assert len(self.positions) == 0, (
    f"未決済ポジションが{len(self.positions)}件残っています: "
    f"{list(self.positions.keys())}"
)

print("✅ 全ポジション決済完了: len(self.positions) == 0")
```

#### **実行中のポジション数確認**
```python
# max_positions上限チェック
current_position_count = len(self.positions)
assert current_position_count <= self.max_positions, (
    f"ポジション数上限超過: {current_position_count}/{self.max_positions}"
)

print(f"✅ ポジション数正常: {current_position_count}/{self.max_positions}")
```

### **D. 自動テストスクリプト**

#### **verify_positions_integrity.py**
```python
"""
ポジション管理の整合性を検証する自動スクリプト

使用方法:
    python verify_positions_integrity.py output/dssms_integration/dssms_*/
"""

import sys
import pandas as pd
from pathlib import Path

def verify_positions_integrity(output_dir):
    """ポジション管理の整合性を検証"""
    output_path = Path(output_dir)
    
    # 1. all_transactions.csv検証
    csv_path = output_path / "all_transactions.csv"
    if not csv_path.exists():
        print(f"❌ ファイル未発見: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    
    # EXIT情報の完全性確認
    missing_exit_date = df['exit_date'].isna().sum()
    missing_exit_price = df['exit_price'].isna().sum()
    missing_pnl = df['pnl'].isna().sum()
    
    if missing_exit_date > 0 or missing_exit_price > 0 or missing_pnl > 0:
        print(f"❌ EXIT情報不足: exit_date={missing_exit_date}, exit_price={missing_exit_price}, pnl={missing_pnl}")
        return False
    
    print(f"✅ all_transactions.csv検証成功: {len(df)}件の取引全てにEXIT情報あり")
    
    # 2. ログ検証
    log_path = output_path / "dssms_execution_log.txt"
    if not log_path.exists():
        print(f"❌ ログファイル未発見: {log_path}")
        return False
    
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    position_add_count = log_content.count('[POSITION_ADD]')
    position_delete_count = log_content.count('[POSITION_DELETE]')
    final_close_count = log_content.count('[FINAL_CLOSE]')
    
    print(f"✅ ログ検証:")
    print(f"   - POSITION_ADD: {position_add_count}件")
    print(f"   - POSITION_DELETE: {position_delete_count}件")
    print(f"   - FINAL_CLOSE: {final_close_count}件")
    
    # 3. エントリー/エグジット数の一致確認
    if position_add_count != position_delete_count:
        print(f"⚠️  警告: POSITION_ADDとPOSITION_DELETEの回数が一致しません（FINAL_CLOSEで決済された可能性）")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python verify_positions_integrity.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    success = verify_positions_integrity(output_dir)
    sys.exit(0 if success else 1)
```

---

## 📊 **実装チェックリスト（総合）**

### **設計完了の定義**
- [ ] 初期化（Initialization）を設計
- [ ] 状態更新（State Update）を設計
  - [ ] BUY時のpositions追加処理を明記
  - [ ] SELL時のpositions削除処理を明記
  - [ ] 強制決済処理を明記
- [ ] 状態確認（State Verification）を設計
  - [ ] ログ確認方法を記載
  - [ ] CSV検証方法を記載
  - [ ] 自動テストスクリプトを作成

### **実装完了の定義**
- [ ] 初期化コード実装（`self.positions = {}`）
- [ ] BUY処理にpositions追加実装
- [ ] SELL処理にpositions削除実装（KeyErrorチェック）
- [ ] 強制決済処理実装
- [ ] ログ出力実装（`[POSITION_ADD]`, `[POSITION_DELETE]`, `[FINAL_CLOSE]`）
- [ ] ドキュメント文字列（docstring）記載

### **検証完了の定義**
- [ ] ログ確認（POSITION_ADD, POSITION_DELETE, FINAL_CLOSEが出力される）
- [ ] CSV確認（all_transactions.csvにEXIT情報が記録される）
- [ ] 内部状態確認（`len(self.positions) == 0`）
- [ ] 自動テストスクリプト成功
- [ ] 複数銘柄同時保有の動作確認（max_positions=2の場合）

---

## 🔗 **参考資料**

- [KNOWN_ISSUES_AND_PREVENTION.md](../KNOWN_ISSUES_AND_PREVENTION.md): Issue #7
- [BUY_SELL_PROCESS_DESIGN_TEMPLATE.md](./BUY_SELL_PROCESS_DESIGN_TEMPLATE.md): BUY/SELL処理の詳細設計
- [DSSMS EXIT未記録問題調査報告](../DSSMS_EXIT_NOT_RECORDED_INVESTIGATION_20260210.md)
- [copilot-instructions.md](../../.github/copilot-instructions.md): リファクタリング・再実装時のGit履歴活用

---

## 📝 **使用方法**

1. **設計時**: このテンプレートをコピーし、プロジェクト固有の情報を記入
2. **実装時**: 3要素（初期化、状態更新、状態確認）に従って実装
3. **レビュー時**: 実装チェックリストで確認
4. **テスト時**: 状態確認設計に従ってテスト実行

---

**重要**: Issue #7の根本原因は、「状態更新（BUY/SELL時のpositions更新処理）」が設計から漏れたことです。
このテンプレートでは、状態管理の3要素（初期化、状態更新、状態確認）を明示的に設計することで、同様の問題を防止します。
