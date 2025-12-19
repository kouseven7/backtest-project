# Phase 2-1: 詳細設計書

**作成日**: 2025-12-19  
**Phase**: Phase 2-1（銘柄切替処理実装 - 詳細設計）  
**Status**: 設計完了、実装準備整う

---

## 📋 概要

Phase 1（DSSMS設計違反コード削除）完了後、Phase 2（銘柄切替処理実装）の詳細設計を実施。
PaperBroker、main_new.py、ForceClose戦略の現状分析を行い、4つの設計項目を完了。

### 設計目標
1. close_all_positions()の実装仕様詳細化
2. ForceClose実装方式の決定（Option A/B/C比較）
3. 銘柄切替フローの詳細設計
4. execution_details生成方法の設計

---

## 🔍 Phase 1完了実績

### 削除実績
- **合計削除**: 約413行
- **Stage 1**: _calculate_position_update()削除（116行）
- **Stage 2**: 銘柄切替時取引実行削除（39行）
- **Stage 3-1~3**: switch_cost/ForceClose呼び出し/メソッド本体削除（244行）
- **Stage 4-1~3**: position_size管理削除（14箇所）

### 検証結果
- **総収益率**: 8.97%
- **execution_details**: 6件
- **エラー**: なし
- **状態**: 正常動作確認

---

## 📊 現状分析結果

### 1. PaperBroker現状

#### positions構造
```python
{
    symbol: {
        'quantity': int,
        'entry_price': float,
        'entry_time': datetime
    }
}
```

#### Order関連クラス
- **Order**: dataclass、uuid自動生成、strategy_name含む
- **OrderSide**: BUY/SELL
- **OrderStatus**: PENDING/FILLED/CANCELLED
- **OrderType**: MARKET/LIMIT

#### 既存決済機能
- **個別決済**: SELL注文実行時にポジション削減/クローズ
- **対象**: 個別銘柄のみ
- **欠損**: close_all_positions()メソッド未実装

#### 既存メソッド（21個確認）
- submit_order()
- get_portfolio_value()
- get_positions()
- get_current_price()
- _execute_market_order()
- 他16個...

### 2. main_new.py現状

#### MainSystemController構成
- **管理コンポーネント**: 6個
  - IntegratedExecutionManager
  - RiskEngine
  - SystemExecutionEngine
  - OptimalPortfolioEngine
  - DynamicRiskManager
  - ConfigProcessor

#### execute_comprehensive_backtest()
```python
def execute_comprehensive_backtest(
    self,
    ticker: str,
    stock_data: Optional[pd.DataFrame] = None,
    index_data: Optional[pd.DataFrame] = None,
    days_back: int = 365,
    backtest_start_date: Optional[datetime] = None,
    backtest_end_date: Optional[datetime] = None,
    warmup_days: int = 90
) -> Dict[str, Any]:
```

**欠損パラメータ**: `force_close_on_entry` なし

#### IntegratedExecutionManager現状
- **既存メソッド**: 10個
- **ForceClose機能**: なし
- **銘柄切替専用メソッド**: なし

### 3. execution_details生成確認

#### 生成元
- **メソッド**: StrategyExecutionManager._execute_trades()
- **Location**: strategy_execution_manager.py Line 389-550

#### strategy_name取得
```python
# Line 419
strategy_name = getattr(self, 'current_strategy_name', 'Unknown')
```

#### 生成フロー
```
1. signals → _generate_trade_orders()
2. trade_orders → Order生成（strategy_name含む）
3. Order実行 → execution_results
4. execution_results → execution_details
```

#### 重要な発見
- **PaperBrokerの責務**: 注文実行のみ
- **execution_details生成**: StrategyExecutionManager
- **ForceClose適用可能**: 同じ仕組みで実装可能

---

## 🎯 詳細設計

### 設計1: close_all_positions()実装仕様

#### メソッドシグネチャ
```python
def close_all_positions(
    self, 
    current_date: datetime,
    reason: str = "symbol_switch"
) -> List[Dict[str, Any]]:
    """
    全ポジション強制決済
    
    Args:
        current_date: 決済日時
        reason: 決済理由（"symbol_switch", "backtest_end"等）
    
    Returns:
        List[Dict]: 決済結果のリスト
            - symbol: str
            - action: str ("SELL")
            - quantity: int
            - price: float
            - pnl: float
            - entry_price: float
            - exit_price: float
            - entry_time: datetime
            - exit_time: datetime
            - reason: str
    """
```

#### 実装仕様

**1. ポジションコピー（ループ中変更回避）**
```python
positions_to_close = list(self.positions.items())
```

**2. 個別SELL注文実行**
```python
for symbol, position_data in positions_to_close:
    sell_order = Order(
        id=str(uuid.uuid4()),
        symbol=symbol,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=position_data['quantity'],
        status=OrderStatus.PENDING,
        created_at=current_date,
        strategy_name="ForceClose"  # 明示的に設定
    )
    
    success = self.submit_order(sell_order)
```

**3. エラー耐性**
- 個別銘柄の決済失敗時も継続
- エラー時は警告ログ出力
- 成功した決済結果のみ返却

**4. 詳細ログ出力**
```python
self.logger.info(f"[FORCE_CLOSE] Closing all positions: {len(positions_to_close)} positions")
self.logger.info(f"[FORCE_CLOSE] Reason: {reason}")
self.logger.info(f"[FORCE_CLOSE] Successfully closed {len(results)}/{len(positions_to_close)} positions")
```

**5. 決済結果返却**
- 決済成功した全銘柄の詳細情報
- PnL計算結果
- エントリー/イグジット価格・時刻

#### エラーハンドリング
```python
try:
    success = self.submit_order(sell_order)
    if success:
        results.append({...})
    else:
        self.logger.warning(f"Failed to close position for {symbol}")
except Exception as e:
    self.logger.error(f"Error closing position for {symbol}: {e}")
    continue  # 次のポジション処理へ
```

---

### 設計2: ForceClose実装方式決定

#### Option比較

**Option A: IntegratedExecutionManager拡張**
- **実装場所**: integrated_execution_manager.py
- **メリット**: 既存コンポーネント活用
- **デメリット**: 
  - IntegratedExecutionManagerの肥大化
  - 責務の境界が不明確
  - 保守性低下

**Option B: ForceCloseStrategy実装（推奨）**
- **実装場所**: `src/strategies/force_close_strategy.py`（新規）
- **メリット**:
  - Phase 1整合性: 他戦略と同じパターン
  - 単一責任原則: 戦略としての独立性
  - 既存パターン踏襲: Strategy基底クラス活用
  - 保守性: 独立したモジュール
  - dssms_cleanup_plan.md整合性: 「main_new.pyのForceClose戦略」明記
- **デメリット**: 新規ファイル作成（影響は小）

**Option C: PaperBrokerのみ**
- **実装場所**: paper_broker.py
- **メリット**: シンプル
- **デメリット**:
  - 責務分離違反: PaperBrokerは注文実行のみ
  - execution_details生成の不整合
  - strategy_name設定が不自然

#### 決定: Option B（ForceCloseStrategy）

**推奨理由**:
1. **Phase 1整合性**: 他戦略（Contrarian, GoldenCross等）と同じパターン
2. **単一責任原則**: 戦略としての独立性保持
3. **既存パターン踏襲**: Strategy基底クラス活用、保守性向上
4. **dssms_cleanup_plan.md整合性**: 「main_new.pyのForceClose戦略」明記
5. **アーキテクチャ**: StrategyExecutionManagerとの統合容易

#### ForceCloseStrategy実装概要

```python
# src/strategies/force_close_strategy.py

class ForceCloseStrategy:
    """
    強制決済戦略（銘柄切替時/バックテスト終了時）
    
    PaperBroker.close_all_positions()を呼び出し、
    全ポジションを決済するsignalsを生成。
    
    strategy_name: "ForceClose"
    """
    
    def __init__(self, broker, logger=None):
        self.broker = broker
        self.logger = logger or logging.getLogger(__name__)
        self.strategy_name = "ForceClose"
    
    def generate_signals(
        self, 
        current_date: datetime,
        reason: str = "symbol_switch"
    ) -> List[Dict[str, Any]]:
        """
        全ポジション決済signalsを生成
        
        Args:
            current_date: 決済日時
            reason: 決済理由
        
        Returns:
            List[Dict]: signals（StrategyExecutionManager互換）
        """
        # 1. PaperBroker.close_all_positions()呼び出し
        close_results = self.broker.close_all_positions(
            current_date=current_date,
            reason=reason
        )
        
        # 2. signals形式に変換
        signals = []
        for result in close_results:
            signals.append({
                'symbol': result['symbol'],
                'action': 'SELL',
                'quantity': result['quantity'],
                'price': result['exit_price'],
                'timestamp': current_date,
                'strategy': self.strategy_name,
                'reason': reason,
                'pnl': result['pnl'],
                'entry_price': result['entry_price']
            })
        
        return signals
```

---

### 設計3: 銘柄切替フロー（8ステップ）

#### フロー図

```
1. DSSMS: _get_optimal_symbol() 
   ↓ 新銘柄選択
   
2. DSSMS: _evaluate_and_execute_switch()
   ↓ 切替判断（should_switch=True）
   ↓ 切替通知のみ（取引実行しない）
   
3. DSSMS: _execute_multi_strategies(new_symbol, force_close_on_entry=True)
   ↓ main_new.pyに新銘柄渡す
   
4. main_new.py: execute_comprehensive_backtest()
   ↓ force_close_on_entry判定
   ↓ if True: _force_close_all_positions()呼び出し
   
5. main_new.py: _force_close_all_positions()
   ↓ IntegratedExecutionManager.execute_force_close()呼び出し
   
6. IntegratedExecutionManager: execute_force_close()
   ↓ ForceCloseStrategy.generate_signals()呼び出し
   
7. PaperBroker: close_all_positions()
   ↓ 全ポジション決済（個別SELL注文実行）
   ↓ 決済結果返却
   
8. main_new.py: 通常バックテスト実行
   ↓ 各戦略が新銘柄のエントリー判断
   ↓ エントリー条件満たせばBUY、満たさなければ見送り
```

#### 各ステップ詳細

**Step 1-2: DSSMS銘柄選択**
```python
# dssms_integrated_main.py
selected_symbol = self._get_optimal_symbol(target_date, stock_data, index_data)

if should_switch:
    # 切替通知のみ（取引実行しない）
    self.logger.info(f"[SYMBOL_SWITCH] {self.current_symbol} → {selected_symbol}")
    switch_result['switch_requested'] = True
    switch_result['from_symbol'] = self.current_symbol
    switch_result['to_symbol'] = selected_symbol
    switch_result['switch_date'] = target_date
```

**Step 3: DSSMS→main_new.py通知**
```python
# dssms_integrated_main.py
backtest_result = self._execute_multi_strategies(
    symbol=selected_symbol,
    target_date=target_date,
    stock_data=stock_data,
    index_data=index_data,
    force_close_on_entry=True  # 新規パラメータ
)
```

**Step 4: main_new.py銘柄切替判定**
```python
# main_new.py
def execute_comprehensive_backtest(
    self,
    ticker: str,
    # ... 既存パラメータ ...
    force_close_on_entry: bool = False  # 新規追加
) -> Dict[str, Any]:
    """
    force_close_on_entry: True時は既存ポジション強制決済
    """
    if force_close_on_entry:
        self.logger.info(f"[FORCE_CLOSE] Closing all positions before entry for {ticker}")
        self._force_close_all_positions()
    
    # 通常のバックテスト実行
    # ...
```

**Step 5-6: ForceClose実行**
```python
# main_new.py
def _force_close_all_positions(self):
    """IntegratedExecutionManager経由でForceClose実行"""
    self.integrated_execution_manager.execute_force_close(
        current_date=self.current_date,
        reason="symbol_switch"
    )

# integrated_execution_manager.py（新規メソッド）
def execute_force_close(
    self, 
    current_date: datetime,
    reason: str = "symbol_switch"
):
    """ForceCloseStrategy実行"""
    force_close_strategy = ForceCloseStrategy(
        broker=self.broker,
        logger=self.logger
    )
    
    signals = force_close_strategy.generate_signals(
        current_date=current_date,
        reason=reason
    )
    
    # StrategyExecutionManager経由でexecution_details生成
    self.strategy_execution_manager.execute_signals(signals)
```

**Step 7: PaperBroker決済実行**
```python
# paper_broker.py
def close_all_positions(
    self, 
    current_date: datetime,
    reason: str = "symbol_switch"
) -> List[Dict[str, Any]]:
    """全ポジション決済（設計1参照）"""
    # 実装済み（設計1参照）
```

**Step 8: 新銘柄エントリー判断**
```python
# main_new.py
# execute_comprehensive_backtest()内で通常のバックテスト実行
# 各戦略が独自にエントリー判断
# エントリー条件満たせばBUY、満たさなければ見送り
```

#### 新規パラメータ

**force_close_on_entry**
- **型**: bool
- **デフォルト**: False
- **用途**: 銘柄切替時にTrue指定
- **追加場所**:
  - execute_comprehensive_backtest()
  - _execute_multi_strategies()（DSSMS）

---

### 設計4: execution_details生成方法

#### strategy_name
```python
strategy_name = "ForceClose"
```

#### execution_details構造
```python
{
    'symbol': str,           # 銘柄コード
    'action': str,           # "SELL"固定
    'quantity': int,         # 決済数量
    'price': float,          # 決済価格
    'timestamp': datetime,   # 決済日時
    'strategy': str,         # "ForceClose"
    'reason': str,           # "symbol_switch"等
    'pnl': float,            # 損益
    'entry_price': float,    # エントリー価格
    'exit_price': float,     # イグジット価格
    'entry_time': datetime,  # エントリー日時
    'exit_time': datetime    # イグジット日時
}
```

#### 生成フロー

```
1. PaperBroker.close_all_positions()
   ↓ 決済実行、結果返却
   
2. ForceCloseStrategy.generate_signals()
   ↓ 決済結果 → signals変換
   ↓ strategy="ForceClose"設定
   
3. StrategyExecutionManager._execute_trades()
   ↓ signals → Order生成
   ↓ Order.strategy_name = "ForceClose"
   
4. StrategyExecutionManager._execute_trades()
   ↓ Order実行結果 → execution_details生成
   ↓ strategy_name自動取得（Line 419）
   
5. execute_comprehensive_backtest()
   ↓ execution_details統合
   ↓ レポート生成
```

#### StrategyExecutionManager連携

```python
# strategy_execution_manager.py Line 419
strategy_name = getattr(self, 'current_strategy_name', 'Unknown')

# ForceCloseStrategyの場合
self.current_strategy_name = "ForceClose"

# execution_details生成時に自動設定
execution_details.append({
    'strategy': strategy_name,  # "ForceClose"
    # ...
})
```

#### 既存execution_detailsとの統合

**通常戦略のexecution_details**:
```python
{
    'strategy': 'Contrarian',
    'action': 'BUY',
    # ...
}
```

**ForceCloseのexecution_details**:
```python
{
    'strategy': 'ForceClose',
    'action': 'SELL',
    'reason': 'symbol_switch',  # 追加フィールド
    # ...
}
```

**統合後のexecution_details**:
```python
[
    {'strategy': 'Contrarian', 'action': 'BUY', ...},
    {'strategy': 'GoldenCross', 'action': 'SELL', ...},
    {'strategy': 'ForceClose', 'action': 'SELL', 'reason': 'symbol_switch', ...},
    {'strategy': 'ForceClose', 'action': 'SELL', 'reason': 'symbol_switch', ...},
    # ...
]
```

---

## 📝 実装計画（main_new_switch_impl_plan.md参照）

### Priority 1: PaperBroker.close_all_positions()実装
- **タスク**: 全ポジション決済機能
- **影響範囲**: PaperBrokerのみ
- **リスク**: 低
- **実装内容**: 設計1参照

### Priority 2: main_new.py銘柄切替実装
- **タスク**: 
  - execute_comprehensive_backtestにforce_close_on_entryパラメータ追加
  - _force_close_all_positions()メソッド追加
  - IntegratedExecutionManager.execute_force_close()追加
- **影響範囲**: MainSystemController、IntegratedExecutionManager
- **リスク**: 中

### Priority 3: ForceCloseStrategy実装
- **タスク**: src/strategies/force_close_strategy.py新規作成
- **影響範囲**: 新規モジュール
- **リスク**: 中
- **実装内容**: 設計2参照

### Priority 4: DSSMS銘柄切替ロジック修正
- **タスク**:
  - _evaluate_and_execute_switch()修正（取引実行削除）
  - _execute_multi_strategies()にforce_close_on_entryパラメータ追加
- **影響範囲**: 広範
- **リスク**: 高

### Priority 5: execution_type='switch'削除
- **タスク**: execution_detailsの整理
- **影響範囲**: レポート生成
- **リスク**: 中

---

## 🧪 テスト計画

### Phase 2-2: 実装準備

#### ForceCloseStrategy実装仕様書
- クラス構造詳細
- メソッドシグネチャ
- エラーハンドリング
- ログ出力仕様

#### テストケース設計
```python
# test_force_close_strategy.py

def test_close_all_positions_success():
    """全ポジション決済成功テスト"""
    # Given: 3銘柄保有
    # When: close_all_positions()実行
    # Then: 3銘柄全て決済、results返却

def test_close_all_positions_partial_failure():
    """一部決済失敗テスト"""
    # Given: 3銘柄保有、1銘柄決済失敗
    # When: close_all_positions()実行
    # Then: 2銘柄決済成功、エラーログ出力

def test_force_close_strategy_signals():
    """ForceCloseStrategy signals生成テスト"""
    # Given: ForceCloseStrategy初期化
    # When: generate_signals()実行
    # Then: signals形式で返却

def test_execution_details_generation():
    """execution_details生成テスト"""
    # Given: ForceCloseStrategy実行
    # When: StrategyExecutionManager連携
    # Then: strategy="ForceClose"のexecution_details生成
```

#### バックテスト検証シナリオ
```python
# 1. 単一銘柄ForceClose
# - 銘柄: 7203.T
# - ポジション: 1件
# - 期待: 決済成功、execution_details 1件

# 2. 複数銘柄ForceClose
# - 銘柄: 7203.T, 6758.T, 9984.T
# - ポジション: 3件
# - 期待: 決済成功3件、execution_details 3件

# 3. 銘柄切替統合テスト
# - 銘柄: 7203.T → 6758.T
# - フロー: ForceClose → 新銘柄エントリー
# - 期待: 
#   - ForceClose execution_details生成
#   - 新銘柄エントリー判断成功
#   - 総収益率正常
```

---

## ✅ 設計完了チェックリスト

- [x] PaperBroker現状分析（21メソッド、positions構造）
- [x] main_new.py現状分析（6コンポーネント、既存パラメータ）
- [x] execution_details生成確認（StrategyExecutionManager）
- [x] close_all_positions()実装仕様（完全設計）
- [x] ForceClose実装方式決定（Option B推奨、理由明確）
- [x] 銘柄切替フロー設計（8ステップ定義）
- [x] execution_details生成設計（strategy_name="ForceClose"）
- [x] Option A/B/C比較（実装コスト、アーキテクチャ整合性、保守性）
- [x] 実装優先順位決定（Priority 1~5）
- [x] テストケース設計（3種類）
- [x] バックテスト検証シナリオ設計（3シナリオ）

---

## 🚀 次のステップ: Phase 2-2実装準備

### 実装準備タスク
1. **ForceCloseStrategy実装仕様書作成**
   - クラス構造詳細
   - メソッドシグネチャ
   - エラーハンドリング詳細
   - ログ出力仕様

2. **テストケース実装**
   - test_force_close_strategy.py作成
   - テストデータ準備
   - モックオブジェクト設計

3. **バックテスト検証方法確定**
   - 検証シナリオ詳細化
   - 期待結果定義
   - 検証スクリプト作成

4. **実装順序確定**
   - Priority 1: PaperBroker.close_all_positions()
   - Priority 2: main_new.py銘柄切替実装
   - Priority 3: ForceCloseStrategy実装
   - Priority 4: DSSMS修正
   - Priority 5: execution_type削除

---

## 📚 参考ドキュメント

- [main_new_switch_impl_plan.md](./main_new_switch_impl_plan.md): 実装計画詳細
- [dssms_cleanup_plan.md](./dssms_cleanup_plan.md): Phase 1実績記録
- strategy_execution_manager.py Line 389-550: execution_details生成実装
- paper_broker.py: PaperBroker実装
- main_new.py: MainSystemController実装

---

## 💡 重要な設計判断

### 1. Option B（ForceCloseStrategy）推奨理由
- **Phase 1整合性**: 他戦略と同じパターン
- **単一責任原則**: 戦略としての独立性
- **既存パターン踏襲**: Strategy基底クラス活用
- **保守性**: 独立したモジュール
- **dssms_cleanup_plan.md整合性**: 「main_new.pyのForceClose戦略」明記

### 2. close_all_positions()責務分離
- **PaperBroker**: 注文実行のみ
- **StrategyExecutionManager**: execution_details生成
- **ForceCloseStrategy**: signals生成

### 3. 銘柄切替通知方法
- **force_close_on_entryパラメータ**: 明示的な通知
- **DSSMS→main_new.py**: パラメータ渡し
- **取引実行**: DSSMS側では実行しない

### 4. strategy_name命名
- **"ForceClose"**: シンプルで明確
- **既存ログとの整合性**: Phase 1コメント言及
- **レポート生成**: execution_detailsで識別可能

---

**設計完了日**: 2025-12-19  
**設計者**: GitHub Copilot  
**承認**: Phase 2-1完了、Phase 2-2実装準備へ
