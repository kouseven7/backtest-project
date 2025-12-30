# backtest_daily実装ガイド: Phase 3設計ドキュメント

**作成日**: 2025年12月30日  
**目的**: Phase 3実装時の参考資料として設計問題の解決方針を提供  
**対象者**: 将来のbacktest_daily()実装者  

---

## 📋 概要

### Phase 3実装目標
DSSMS日次判断とマルチ戦略全期間一括判定の設計不一致を解決し、リアルトレード対応の日次バックテストシステムを構築する。

### 現在の問題（Phase 1時点）
```python
# 現在の問題: 設計不一致
DSSMS側: 毎日最適な1銘柄を選択 → 日次判断前提
マルチ戦略側: backtest(start, end) → 全期間一括判定

# 結果: リアルトレードと乖離したバックテスト
```

---

## 🎯 実装方針

### 新しい設計（Phase 3目標）
```python
# 目標: 日次対応の統一
class BaseStrategy:
    def backtest_daily(self, current_date, stock_data, existing_position=None):
        """日次バックテスト: その日のみを判定（リアルトレード対応）"""
        
    def backtest(self, trading_start_date, trading_end_date):
        """従来のバックテスト（後方互換性のため維持）"""
```

---

## 🔧 実装手順

### Step 1: BaseStrategyの拡張
```python
# strategies/base_strategy.py
class BaseStrategy:
    def backtest_daily(self, current_date: datetime, stock_data: pd.DataFrame, 
                      existing_position: Optional[Dict] = None) -> Dict[str, Any]:
        """
        日次バックテスト実行
        
        Parameters:
            current_date: 対象日（その日のみを判定）
            stock_data: 株価データ（ウォームアップ期間込み）
            existing_position: 既存ポジション情報
            
        Returns:
            {
                'action': 'entry'|'exit'|'hold',
                'price': float,
                'quantity': int,
                'reason': str,
                'updated_position': Dict
            }
        """
        # 実装内容:
        # 1. current_dateのindexを特定
        # 2. ウォームアップ期間を考慮したインジケーター計算
        # 3. 前日データのみを使用した判定（ルックアヘッドバイアス防止）
        # 4. 翌日始値エントリー価格の設定
        # 5. 既存ポジションとの整合性チェック
```

### Step 2: DSSMS統合の改善
```python
# src/dssms/dssms_integrated_main.py
def _execute_multi_strategies(self, target_date, symbol, stock_data):
    """改善版: 日次実行制御"""
    
    # 既存ポジション情報を取得
    existing_position = self.main_controller.paper_broker.get_position(symbol)
    
    # 日次バックテスト実行
    for strategy_class in strategy_classes:
        strategy = strategy_class(stock_data)
        result = strategy.backtest_daily(target_date, stock_data, existing_position)
        
        # アクション実行
        if result['action'] == 'entry':
            self._execute_entry(symbol, result)
        elif result['action'] == 'exit':
            self._execute_exit(symbol, result)
```

### Step 3: 銘柄切替対応
```python
def _handle_symbol_switch(self, old_symbol, new_symbol, target_date):
    """銘柄切替時のポジション処理"""
    
    # 旧銘柄のポジションを強制決済
    if self.main_controller.paper_broker.has_position(old_symbol):
        self._force_close_position(old_symbol, target_date)
    
    # 新銘柄でのバックテスト実行
    new_data = self._get_stock_data(new_symbol, target_date)
    self._execute_multi_strategies(target_date, new_symbol, new_data)
```

---

## ⚠️ 重要な注意事項

### ルックアヘッドバイアス防止（必須）
```python
# 禁止: 当日データで判定
signal = data['Close'].iloc[current_idx]

# 必須: 前日データで判定
signal = data['Close'].shift(1).iloc[current_idx]

# 禁止: 当日終値でエントリー
entry_price = data['Adj Close'].iloc[current_idx]

# 必須: 翌日始値でエントリー
entry_price = data['Open'].iloc[current_idx + 1] * (1 + slippage)
```

### 決定論保証
```python
# 問題: ランダム性による再現性の欠如
random_factor = np.random.random()

# 解決: 決定論的な計算のみ使用
deterministic_factor = some_calculation_based_on_data
```

### ポジション継続性
```python
# 重要: 銘柄切替時のポジション処理
def handle_position_continuity(self, old_symbol, new_symbol):
    # 1. 旧銘柄ポジションの適切な決済
    # 2. 新銘柄での新規ポジション検討
    # 3. 資金・リスク管理の継続
```

---

## 🧪 テスト方針

### 単体テスト
```python
def test_backtest_daily_lookback_bias():
    """ルックアヘッドバイアス防止テスト"""
    # 実装: 未来データ使用の検出

def test_backtest_daily_determinism():
    """決定論テスト"""
    # 実装: 同じ入力で同じ出力の保証

def test_symbol_switch_handling():
    """銘柄切替テスト"""
    # 実装: ポジション継続性の確認
```

### 統合テスト
```python
def test_dssms_integration():
    """DSSMS統合テスト"""
    # 実装: 日次判断とマルチ戦略の整合性確認

def test_real_trade_simulation():
    """リアルトレード模擬テスト"""
    # 実装: kabu STATION API呼び出し前の最終検証
```

---

## 📈 パフォーマンス考慮事項

### データ取得最適化
```python
# 効率的なデータ取得
def get_optimized_data(symbol, current_date, warmup_days=150):
    # キャッシュ活用
    # 必要最小限のデータ取得
    # メモリ使用量の最適化
```

### 計算負荷軽減
```python
# インジケーターの段階的計算
def incremental_indicator_calculation():
    # 前日計算結果の再利用
    # 当日分のみの追加計算
    # メモリ効率の向上
```

---

## 🔄 移行戦略

### Phase 3-A: 基盤実装（1週間）
1. BaseStrategy.backtest_daily()の基本実装
2. 1戦略での動作検証（VWAPBreakout推奨）
3. ルックアヘッドバイアステストの実装

### Phase 3-B: 全戦略展開（1週間）
1. 全戦略クラスでのbacktest_daily()実装
2. DSSMS統合の改善
3. 銘柄切替処理の実装

### Phase 3-C: 最終検証（2-3日）
1. リアルトレード模擬テスト
2. パフォーマンス最適化
3. ドキュメント更新

---

## 📚 参考資料

### 既存ドキュメント
- [DESIGN_DECISION_ANALYSIS_20251230.md](../Fund%20position%20reset%20issue/DESIGN_DECISION_ANALYSIS_20251230.md): 設計問題の詳細分析
- [copilot-instructions.md](../../.github/copilot-instructions.md): ルックアヘッドバイアス防止ルール（3原則）

### コード参照
- [dssms_integrated_main.py Line 1720-1740](../../src/dssms/dssms_integrated_main.py): Phase 1実装コメント
- [strategies/base_strategy.py](../../strategies/base_strategy.py): 現行backtest()実装

### 外部参考
- kabu STATION API仕様: リアルトレード制約の理解
- pandas.DataFrame.shift(1): ルックアヘッドバイアス防止の基本

---

**作成者**: GitHub Copilot  
**Phase**: Phase 2 ドキュメント化  
**次のPhase**: Phase 3 backtest_daily()実装  
**予想工数**: Phase 3-A(1週間) + Phase 3-B(1週間) + Phase 3-C(2-3日) = 2-3週間