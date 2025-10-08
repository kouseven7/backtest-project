# DSSMS統合システム クラス図・シーケンス図

**作成日**: 2025年9月25日  
**Phase**: Phase 2 - アーキテクチャ設計  
**ステータス**: [OK] **完了**

---

## 🏗️ **1. 統合システム クラス図**

### **全体アーキテクチャ図**
```
┌─────────────────────────────────────────────────────────────────┐
│                    DSSMS統合システム                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │             DSSMSIntegratedBacktester              │    │
│  │                (メインコントローラー)                  │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │ - dss_core: DSSBacktesterV3              │  │    │
│  │  │ - strategy_adapter: MultiStrategyAdapter │  │    │
│  │  │ - switch_manager: SymbolSwitchManager    │  │    │
│  │  │ - position_manager: PositionManager      │  │    │
│  │  │ - data_cache: DataCacheManager           │  │    │
│  │  │ - performance_tracker: PerformanceTracker│  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  │                                                    │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │ + run_dynamic_backtest()                    │  │    │
│  │  │ + _process_daily_trading()                  │  │    │
│  │  │ + _execute_symbol_switch()                  │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐│
│  │MultiStrategy    │  │SymbolSwitch     │  │PositionManager   ││
│  │Adapter          │  │Manager          │  │                  ││
│  │                 │  │                 │  │                  ││
│  │- risk_manager   │  │- switch_history │  │- positions       ││
│  │- param_manager  │  │- holding_start  │  │- cash_balance    ││
│  │- strategy_stats │  │- config         │  │- risk_limits     ││
│  │                 │  │                 │  │                  ││
│  │+ execute_       │  │+ evaluate_      │  │+ execute_        ││
│  │  strategies()   │  │  symbol_switch()│  │  symbol_switch() ││
│  │+ _fetch_symbol_ │  │+ record_switch_ │  │+ check_risk_     ││
│  │  data()         │  │  executed()     │  │  limits()        ││
│  └─────────────────┘  └─────────────────┘  └──────────────────┘│
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │DataCache        │  │Performance      │                      │
│  │Manager          │  │Tracker          │                      │
│  │                 │  │                 │                      │
│  │- stock_data_    │  │- execution_     │                      │
│  │  cache          │  │  times          │                      │
│  │- index_data_    │  │- memory_usage   │                      │
│  │  cache          │  │- success_rates  │                      │
│  │- cache_metadata │  │- switch_costs   │                      │
│  │                 │  │                 │                      │
│  │+ get_cached_    │  │+ record_daily_  │                      │
│  │  data()         │  │  performance()  │                      │
│  │+ store_cached_  │  │+ get_performance│                      │
│  │  data()         │  │  _summary()     │                      │
│  └─────────────────┘  └─────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **外部依存関係図**
```
┌─────────────────────────────────────────────────────────────────┐
│                        外部システム連携                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────┐    ┌─────────────────────────────────┐   │
│  │   DSS Core V3     │    │         既存マルチ戦略          │   │
│  │  (動的銘柄選択)   │    │        (main.pyシステム)        │   │
│  │                   │    │                                 │   │
│  │ ┌───────────────┐ │    │ ┌─────────────────────────────┐ │   │
│  │ │dssms_         │ │    │ │ VWAPBreakoutStrategy        │ │   │
│  │ │backtester_    │◄┼────┼─│ MomentumInvestingStrategy   │ │   │
│  │ │v3.py          │ │    │ │ BreakoutStrategy            │ │   │
│  │ └───────────────┘ │    │ │ VWAPBounceStrategy          │ │   │
│  │                   │    │ │ OpeningGapStrategy          │ │   │
│  │ + run_daily_      │    │ │ ContrarianStrategy          │ │   │
│  │   selection()     │    │ │ GCStrategy                  │ │   │
│  └───────────────────┘    │ └─────────────────────────────┘ │   │
│                           │                                 │   │
│                           │ + apply_strategies_with_        │   │
│                           │   optimized_params()            │   │
│                           └─────────────────────────────────┘   │
│                                                                 │
│  ┌───────────────────┐    ┌─────────────────────────────────┐   │
│  │   データ取得      │    │         設定管理システム        │   │
│  │   (yfinance)      │    │                                 │   │
│  │                   │    │                                 │   │
│  │ + download()      │    │ ┌─────────────────────────────┐ │   │
│  │ + Ticker()        │◄───┼─│ config/risk_management.py   │ │   │
│  └───────────────────┘    │ │ config/optimized_           │ │   │
│                           │ │        parameters.py        │ │   │
│                           │ │ config/multi_strategy_      │ │   │
│                           │ │        manager.py           │ │   │
│                           │ └─────────────────────────────┘ │   │
│                           └─────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 **2. 動的銘柄切替シーケンス図**

### **日次取引処理フロー**
```
参加者：
- Client: クライアント
- DSSMS: DSSMSIntegratedBacktester
- DSS: DSSBacktesterV3  
- Switch: SymbolSwitchManager
- Strategy: MultiStrategyAdapter
- Position: PositionManager
- Cache: DataCacheManager

Client          DSSMS           DSS             Switch          Strategy        Position        Cache
  |               |               |               |               |               |               |
  |─run_dynamic_  |               |               |               |               |               |
  |  backtest()──►|               |               |               |               |               |
  |               |               |               |               |               |               |
  |               |─_process_     |               |               |               |               |
  |               | daily_trading |               |               |               |               |
  |               | (target_date) |               |               |               |               |
  |               |               |               |               |               |               |
  |               |─run_daily_    |               |               |               |               |
  |               | selection()──►|               |               |               |               |
  |               |               |               |               |               |               |
  |               |◄─selected_    |               |               |               |               |
  |               |  symbol───────|               |               |               |               |
  |               |               |               |               |               |               |
  |               |─evaluate_     |               |               |               |               |
  |               | symbol_switch |               |               |               |               |
  |               | (from, to)───►|               |               |               |               |
  |               |               |               |               |               |               |
  |               |◄─switch_      |               |               |               |               |
  |               |  evaluation───|               |               |               |               |
  |               |               |               |               |               |               |
  |              [IF should_switch = True]        |               |               |               |
  |               |               |               |               |               |               |
  |               |─_execute_     |               |               |               |               |
  |               | symbol_switch |               |               |               |               |
  |               | (switch_result|               |               |               |               |
  |               |               |               |               |               |               |
  |               |                               |               |               |               |
  |               |─execute_symbol_switch()──────────────────────────────────────►|               |
  |               |                               |               |               |               |
  |               |◄─position_result──────────────────────────────────────────────|               |
  |               |                               |               |               |               |
  |               |─record_switch_executed()─────►|               |               |               |
  |               |                               |               |               |               |
  |               |               |               |               |               |               |
  |               |─execute_strategies(symbol, date)─────────────►|               |               |
  |               |               |               |               |               |               |
  |               |               |               |               |─get_cached_   |               |
  |               |               |               |               | data()───────►|               |
  |               |               |               |               |               |               |
  |               |               |               |               |◄─data or None─|               |
  |               |               |               |               |               |               |
  |              [IF cache miss]  |               |               |               |               |
  |               |               |               |               |─yfinance.     |               |
  |               |               |               |               | download()    |               |
  |               |               |               |               |               |               |
  |               |               |               |               |─store_cached_ |               |
  |               |               |               |               | data()───────►|               |
  |               |               |               |               |               |               |
  |               |               |               |               |─apply_        |               |
  |               |               |               |               | strategies_   |               |
  |               |               |               |               | for_single_   |               |
  |               |               |               |               | day()         |               |
  |               |               |               |               |               |               |
  |               |◄─strategy_result──────────────────────────────|               |               |
  |               |               |               |               |               |               |
  |               |─record_daily_performance()    |               |               |               |
  |               |               |               |               |               |               |
  |◄─daily_result─|               |               |               |               |               |
  | (日次結果)    |               |               |               |               |               |
  |               |               |               |               |               |               |
```

### **銘柄切替評価詳細フロー**
```
SymbolSwitchManager内部フロー：

Switch          History         Config
  |               |               |
  |─evaluate_     |               |
  | symbol_switch |               |
  | (from, to,    |               |
  |  target_date) |               |
  |               |               |
 [同一銘柄チェック]              |
  |               |               |
  | if from == to |               |
  | └─return False|               |
  |               |               |
 [最小保有期間チェック]          |
  |               |               |
  |─_check_min_   |               |
  | holding_period|               |
  | (target_date) |               |
  |               |               |
  |─_get_current_ |               |
  | holding_days()│◄─check────────|
  | (target_date) |  history      |
  |               |               |
  | if days <     │◄─get──────────|
  |   min_days    |  config       |
  | └─return False|               |
  |               |               |
 [月次制限チェック]              |
  |               |               |
  |─_get_monthly_ |               |
  | switch_count()│◄─count────────|
  | (target_date) |  this_month   |
  |               |               |
  | if count >=   │◄─get──────────|
  |   max_monthly |  config       |
  | └─return False|               |
  |               |               |
 [切替実行決定]                  |
  |               |               |
  | return True   |               |
  | (switch_      |               |
  |  evaluation)  |               |
  |               |               |
```

---

## [CHART] **3. データフロー図**

### **データ取得・キャッシュフロー**
```
外部データソース → DataCacheManager → 戦略実行
                                   ↓
                              キャッシュ管理
                                   ↓
                         ┌─────────────────────┐
                         │  メモリキャッシュ   │
                         │                     │
                         │ stock_data_cache    │
                         │ {symbol_date:       │
                         │  DataFrame}         │
                         │                     │
                         │ index_data_cache    │
                         │ {N225_date:         │
                         │  DataFrame}         │
                         │                     │
                         │ cache_metadata      │
                         │ {access_time,       │
                         │  access_count,      │
                         │  data_size}         │
                         └─────────────────────┘
                                   ↓
                         ┌─────────────────────┐
                         │   キャッシュ戦略    │
                         │                     │
                         │ ● LRU削除           │
                         │ ● 容量制限監視      │
                         │ ● 期限管理          │
                         │ ● アクセス統計      │
                         └─────────────────────┘
```

### **パフォーマンス監視データフロー**
```
日次処理結果 → PerformanceTracker → 監視レポート
                      ↓
              ┌─────────────────────┐
              │   パフォーマンス    │
              │     メトリクス      │
              │                     │
              │ ● 実行時間          │
              │ ● メモリ使用量      │
              │ ● 成功率            │
              │ ● 切替コスト        │
              │ ● エラー率          │
              └─────────────────────┘
                      ↓
              ┌─────────────────────┐
              │     閾値評価        │
              │                     │
              │ excellent: 全目標達成 │
              │ good: 成功率90%以上   │
              │ acceptable: 80%以上  │
              │ needs_improvement:   │
              │          80%未満    │
              └─────────────────────┘
```

---

## 🔗 **4. クラス依存関係図**

### **依存関係マトリックス**
```
                    │DSS │Multi│Symbol│Position│Data │Perf│
                    │Core│Strat│Switch│Manager │Cache│Track│
─────────────────────┼────┼─────┼──────┼────────┼─────┼────┤
DSSMSIntegrated     │ ●  │  ●  │  ●   │   ●    │  ●  │ ● │
Backtester          │    │     │      │        │     │   │
─────────────────────┼────┼─────┼──────┼────────┼─────┼────┤
MultiStrategy       │    │     │      │        │  ●  │   │
Adapter             │    │     │      │        │     │   │
─────────────────────┼────┼─────┼──────┼────────┼─────┼────┤
SymbolSwitch        │    │     │      │        │     │   │
Manager             │    │     │      │        │     │   │
─────────────────────┼────┼─────┼──────┼────────┼─────┼────┤
PositionManager     │    │     │      │        │     │   │
─────────────────────┼────┼─────┼──────┼────────┼─────┼────┤
DataCacheManager    │    │     │      │        │     │   │
─────────────────────┼────┼─────┼──────┼────────┼─────┼────┤
PerformanceTracker  │    │     │      │        │     │   │

● = 依存関係あり
```

### **レイヤー構造**
```
┌─────────────────────────────────────────────────────────────────┐
│                      アプリケーション層                         │
│                                                                 │
│            DSSMSIntegratedBacktester                           │
│                   (統合制御)                                   │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                        ビジネスロジック層                       │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐│
│  │MultiStrategy    │  │SymbolSwitch     │  │PositionManager   ││
│  │Adapter          │  │Manager          │  │                  ││
│  │(戦略実行制御)   │  │(切替判定・実行) │  │(ポジション管理)  ││
│  └─────────────────┘  └─────────────────┘  └──────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                       インフラストラクチャ層                   │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │DataCache        │  │Performance      │                      │
│  │Manager          │  │Tracker          │                      │
│  │(データ管理)     │  │(監視・計測)     │                      │
│  └─────────────────┘  └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                       外部システム層                           │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │DSS Core V3      │  │Multi-Strategy   │  │Configuration    │ │
│  │(動的銘柄選択)   │  │System (main.py) │  │Management       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## [TARGET] **5. インターフェース定義**

### **主要インターフェース一覧**

```python
# DSSMSIntegratedBacktester 公開インターフェース
class DSSMSIntegratedBacktester:
    def run_dynamic_backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]
    def get_current_status(self) -> Dict[str, Any]
    def get_performance_summary(self) -> Dict[str, Any]
    def export_results(self, output_path: str, format: str = 'excel') -> bool

# MultiStrategyAdapter 公開インターフェース  
class MultiStrategyAdapter:
    def execute_strategies(self, symbol: str, target_date: datetime, portfolio_value: float) -> Dict[str, Any]
    def get_strategy_stats(self) -> Dict[str, Any]
    def validate_symbol_data(self, symbol: str, target_date: datetime) -> bool

# SymbolSwitchManager 公開インターフェース
class SymbolSwitchManager:
    def evaluate_symbol_switch(self, from_symbol: str, to_symbol: str, target_date: datetime) -> Dict[str, Any]
    def record_switch_executed(self, switch_result: Dict[str, Any]) -> None
    def get_switch_statistics(self) -> Dict[str, Any]

# PositionManager 公開インターフェース
class PositionManager:
    def execute_symbol_switch(self, switch_result: Dict[str, Any], portfolio_value: float) -> Dict[str, Any]
    def check_risk_limits(self, symbol: str, position_size: float) -> bool
    def get_current_positions(self) -> Dict[str, Any]

# DataCacheManager 公開インターフェース
class DataCacheManager:
    def get_cached_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]
    def store_cached_data(self, symbol: str, start_date: datetime, end_date: datetime, stock_data: pd.DataFrame, index_data: pd.DataFrame) -> None
    def clear_cache(self, older_than_days: int = None) -> int

# PerformanceTracker 公開インターフェース
class PerformanceTracker:
    def record_daily_performance(self, daily_result: Dict[str, Any]) -> None
    def get_performance_summary(self) -> Dict[str, Any]
    def should_check_performance(self, current_date: datetime) -> bool
```

---

## [UP] **6. エラーハンドリング・例外フロー**

### **例外処理階層**
```
┌─────────────────────────────────────────────────────────────────┐
│                     例外処理レベル                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1: CRITICAL (システム停止)                              │
│  ├─ システム初期化失敗                                         │
│  ├─ メモリ不足                                                 │
│  └─ 設定ファイル読み込み失敗                                   │
│                                                                 │
│  Level 2: ERROR (処理継続・フォールバック)                     │
│  ├─ データ取得失敗 → キャッシュデータ使用                      │
│  ├─ 戦略実行失敗 → デフォルト戦略適用                          │
│  └─ 切替処理失敗 → 現銘柄継続                                  │
│                                                                 │
│  Level 3: WARNING (ログ記録・継続)                             │
│  ├─ キャッシュヒット率低下                                     │
│  ├─ パフォーマンス目標未達                                     │
│  └─ 切替制限超過                                               │
│                                                                 │
│  Level 4: INFO/DEBUG (状態記録)                               │
│  ├─ 銘柄切替実行                                              │
│  ├─ 戦略シグナル生成                                          │
│  └─ キャッシュ操作                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **エラー復旧フロー**
```
エラー発生
    ↓
エラーレベル判定
    ↓
┌───CRITICAL──→ システム停止・管理者通知
│
├───ERROR────→ フォールバック処理実行
│                ↓
│              処理継続可能性評価
│                ↓
│              継続 or 安全停止
│
├───WARNING──→ ログ記録・監視アラート
│                ↓
│              処理継続
│
└───INFO/DEBUG→ 状態ログ記録
                 ↓
               正常継続
```

---

**クラス図・シーケンス図作成完了**  
**Phase 2: アーキテクチャ設計 - 100%完了**

*作成日: 2025年9月25日*  
*設計完了時刻: Phase 2終了*