# DSSMS統合システム インターフェース仕様書

**作成日**: 2025年9月25日  
**Phase**: Phase 2 - アーキテクチャ設計  
**バージョン**: v1.0  
**ステータス**: ✅ **完了**

---

## 📋 **仕様書概要**

### **目的**
DSSMS統合システムの全クラス間インターフェースを詳細定義し、Phase 3実装時の統一的な開発基準を提供する。

### **対象クラス**
1. **DSSMSIntegratedBacktester** - メインコントローラー
2. **MultiStrategyAdapter** - 戦略実行アダプター  
3. **SymbolSwitchManager** - 銘柄切替管理
4. **PositionManager** - ポジション管理
5. **DataCacheManager** - データキャッシュ管理
6. **PerformanceTracker** - パフォーマンス監視

---

## 🎯 **1. DSSMSIntegratedBacktester**

### **クラス概要**
**役割**: DSSMS統合システムのメインコントローラー  
**責任**: 全コンポーネントの統合・制御、バックテスト実行管理  
**依存**: DSS Core V3, 全サブコンポーネント

### **1.1 コンストラクター**

```python
def __init__(self, config: Dict[str, Any]) -> None:
    """
    DSSMS統合バックテスターの初期化
    
    Parameters:
        config (Dict[str, Any]): 統合設定辞書
            Required keys:
                - 'initial_capital' (float): 初期資本金 (>= 100000)
                - 'backtest_period' (Dict): バックテスト期間設定
                    - 'start_date' (str): 開始日 'YYYY-MM-DD'
                    - 'end_date' (str): 終了日 'YYYY-MM-DD'
            Optional keys:
                - 'switch_cost_rate' (float): 銘柄切替コスト率 (default: 0.001)
                - 'performance_targets' (Dict): パフォーマンス目標値
                - 'enable_cache' (bool): キャッシュ有効化 (default: True)
                - 'log_level' (str): ログレベル (default: 'INFO')
    
    Raises:
        ValueError: 必須設定項目の不足・無効値
        ConfigError: 設定ファイル形式エラー
        SystemError: システム初期化失敗
    
    Example:
        config = {
            'initial_capital': 1000000,
            'backtest_period': {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            },
            'switch_cost_rate': 0.002
        }
        backtester = DSSMSIntegratedBacktester(config)
    """
```

### **1.2 メインメソッド**

```python
def run_dynamic_backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """
    動的銘柄選択バックテストの実行
    
    Parameters:
        start_date (datetime): バックテスト開始日
        end_date (datetime): バックテスト終了日
    
    Returns:
        Dict[str, Any]: バックテスト結果
            {
                'status': str,                    # 'success' | 'partial_success' | 'failed'
                'execution_summary': {
                    'total_days': int,            # 総実行日数
                    'success_days': int,          # 成功日数
                    'switch_count': int,          # 銘柄切替回数
                    'total_execution_time_ms': float
                },
                'performance_metrics': {
                    'final_portfolio_value': float,
                    'total_return': float,        # 総収益率
                    'sharpe_ratio': float,        # シャープレシオ
                    'max_drawdown': float,        # 最大ドローダウン
                    'win_rate': float,            # 勝率
                    'switch_cost_total': float    # 総切替コスト
                },
                'daily_results': List[Dict],      # 日次結果配列
                'switch_history': List[Dict],     # 切替履歴
                'error_log': List[Dict]           # エラーログ
            }
    
    Raises:
        ValueError: 無効な日付範囲
        DataError: 必要データの取得失敗
        SystemError: システム実行エラー
    
    Example:
        result = backtester.run_dynamic_backtest(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        print(f"Final Portfolio Value: {result['performance_metrics']['final_portfolio_value']}")
    """

def get_current_status(self) -> Dict[str, Any]:
    """
    現在のシステム状態を取得
    
    Returns:
        Dict[str, Any]: システム状態情報
            {
                'is_running': bool,               # 実行中フラグ
                'current_symbol': str,            # 現在の選択銘柄
                'portfolio_value': float,         # 現在のポートフォリオ価値
                'last_update': datetime,          # 最終更新時刻
                'performance_status': str,        # パフォーマンス状態
                'cache_status': {
                    'hit_rate': float,            # キャッシュヒット率
                    'used_memory_mb': float       # 使用メモリ量
                },
                'component_status': {
                    'dss_core': str,              # DSS Core状態
                    'strategy_adapter': str,      # 戦略アダプター状態
                    'switch_manager': str,        # 切替管理状態
                    'position_manager': str       # ポジション管理状態
                }
            }
    
    Example:
        status = backtester.get_current_status()
        if status['is_running']:
            print(f"Currently trading: {status['current_symbol']}")
    """

def export_results(self, output_path: str, format: str = 'excel') -> bool:
    """
    バックテスト結果をエクスポート
    
    Parameters:
        output_path (str): 出力ファイルパス
        format (str): 出力形式 ('excel' | 'csv' | 'json')
    
    Returns:
        bool: エクスポート成功フラグ
    
    Raises:
        ValueError: 無効な出力形式
        IOError: ファイル出力エラー
        DataError: 結果データ不整合
    
    Example:
        success = backtester.export_results(
            output_path='./results/dssms_backtest_2023.xlsx',
            format='excel'
        )
    """
```

### **1.3 内部メソッド（テスト用）**

```python
def _process_daily_trading(self, target_date: datetime) -> Dict[str, Any]:
    """
    日次取引処理（内部メソッド・テスト用公開）
    
    Parameters:
        target_date (datetime): 対象日付
    
    Returns:
        Dict[str, Any]: 日次処理結果
            {
                'date': datetime,
                'dss_result': Dict,               # DSS選択結果
                'switch_result': Dict,            # 切替評価結果
                'strategy_result': Dict,          # 戦略実行結果
                'portfolio_value': float,         # 更新後ポートフォリオ価値
                'execution_time_ms': float        # 実行時間
            }
    
    Note: 単体テスト・デバッグ用途での利用想定
    """

def _execute_symbol_switch(self, switch_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    銘柄切替実行（内部メソッド・テスト用公開）
    
    Parameters:
        switch_result (Dict[str, Any]): 切替評価結果
    
    Returns:
        Dict[str, Any]: 切替実行結果
    
    Note: 単体テスト・切替ロジック検証用途での利用想定
    """
```

---

## 🔄 **2. MultiStrategyAdapter**

### **クラス概要**
**役割**: main.pyの既存7戦略との連携アダプター  
**責任**: 動的銘柄での戦略実行、パラメータ管理、リスク制御  
**依存**: RiskManagement, OptimizedParameterManager, yfinance

### **2.1 コンストラクター**

```python
def __init__(self, config: Dict[str, Any]) -> None:
    """
    戦略アダプターの初期化
    
    Parameters:
        config (Dict[str, Any]): アダプター設定
            Required keys:
                - 'initial_capital' (float): 初期資本金
            Optional keys:
                - 'enable_risk_management' (bool): リスク管理有効化 (default: True)
                - 'default_position_size' (float): デフォルトポジションサイズ (default: 1.0)
                - 'strategy_timeout_sec' (int): 戦略実行タイムアウト (default: 30)
    
    Raises:
        ConfigError: 設定値エラー
        SystemError: 戦略システム初期化失敗
    """
```

### **2.2 メインメソッド**

```python
def execute_strategies(self, symbol: str, target_date: datetime, 
                      portfolio_value: float) -> Dict[str, Any]:
    """
    指定銘柄・日付での全7戦略実行
    
    Parameters:
        symbol (str): 対象銘柄コード (例: '7203')
        target_date (datetime): 対象日付
        portfolio_value (float): 現在のポートフォリオ価値
    
    Returns:
        Dict[str, Any]: 戦略実行結果
            {
                'status': str,                    # 'success' | 'no_signal' | 'data_fetch_failed' | 'error'
                'symbol': str,                    # 対象銘柄
                'date': datetime,                 # 対象日付
                'entry_signal': int,              # エントリーシグナル (0|1)
                'exit_signal': int,               # エグジットシグナル (0|1)
                'strategy': str,                  # 選択された戦略名
                'position_size': float,           # ポジションサイズ
                'risk_assessment': {
                    'risk_level': str,            # 'low' | 'medium' | 'high'
                    'position_limit_check': bool, # ポジション制限チェック結果
                    'drawdown_risk': float        # ドローダウンリスク
                },
                'execution_details': {
                    'strategies_evaluated': List[str],  # 評価された戦略リスト
                    'data_quality_score': float,        # データ品質スコア
                    'execution_time_ms': float           # 実行時間
                },
                'updated_portfolio_value': float  # 更新後ポートフォリオ価値
            }
    
    Raises:
        ValueError: 無効な銘柄コード・日付
        DataError: データ取得・処理エラー
        StrategyError: 戦略実行エラー
    
    Example:
        result = adapter.execute_strategies(
            symbol='7203',
            target_date=datetime(2023, 6, 15),
            portfolio_value=1000000
        )
        if result['status'] == 'success' and result['entry_signal']:
            print(f"Entry signal generated by {result['strategy']}")
    """

def get_strategy_stats(self) -> Dict[str, Any]:
    """
    戦略実行統計の取得
    
    Returns:
        Dict[str, Any]: 戦略統計情報
            {
                'total_executions': int,          # 総実行回数
                'success_rate': float,            # 成功率
                'strategy_performance': {
                    'VWAPBreakoutStrategy': {
                        'executions': int,        # 実行回数
                        'success_rate': float,    # 成功率
                        'avg_execution_time_ms': float
                    },
                    # ... 他の戦略
                },
                'data_quality_stats': {
                    'avg_quality_score': float,   # 平均データ品質スコア
                    'data_fetch_failures': int   # データ取得失敗回数
                },
                'risk_distribution': {
                    'low_risk': int,              # 低リスク取引回数
                    'medium_risk': int,           # 中リスク取引回数
                    'high_risk': int              # 高リスク取引回数
                }
            }
    """

def validate_symbol_data(self, symbol: str, target_date: datetime) -> bool:
    """
    指定銘柄のデータ有効性を検証
    
    Parameters:
        symbol (str): 銘柄コード
        target_date (datetime): 対象日付
    
    Returns:
        bool: データ有効性フラグ
    
    Example:
        if adapter.validate_symbol_data('7203', datetime(2023, 6, 15)):
            # データが有効な場合の処理
            pass
    """
```

### **2.3 内部メソッド**

```python
def _fetch_symbol_data(self, symbol: str, target_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    指定銘柄の市場データ取得
    
    Parameters:
        symbol (str): 銘柄コード
        target_date (datetime): 対象日付
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (株価データ, インデックスデータ)
    
    Raises:
        DataError: データ取得失敗
    """

def _load_optimized_parameters(self, symbol: str) -> Dict[str, Dict[str, Any]]:
    """
    指定銘柄の最適化パラメータ読み込み
    
    Parameters:
        symbol (str): 銘柄コード
    
    Returns:
        Dict[str, Dict[str, Any]]: 戦略別最適化パラメータ
    """

def _apply_strategies_for_single_day(self, stock_data: pd.DataFrame, 
                                    index_data: pd.DataFrame,
                                    optimized_params: Dict[str, Dict[str, Any]], 
                                    target_date: datetime,
                                    portfolio_value: float) -> Dict[str, Any]:
    """
    単日用戦略適用ロジック
    
    Parameters:
        stock_data (pd.DataFrame): 株価データ
        index_data (pd.DataFrame): インデックスデータ
        optimized_params (Dict): 最適化パラメータ
        target_date (datetime): 対象日付
        portfolio_value (float): ポートフォリオ価値
    
    Returns:
        Dict[str, Any]: 戦略実行結果
    """
```

---

## 🔀 **3. SymbolSwitchManager**

### **クラス概要**
**役割**: 動的銘柄切替の判定・管理  
**責任**: 切替必要性評価、切替制限管理、切替履歴記録  
**依存**: 設定管理システム

### **3.1 コンストラクター**

```python
def __init__(self, config: Dict[str, Any]) -> None:
    """
    銘柄切替管理の初期化
    
    Parameters:
        config (Dict[str, Any]): 切替管理設定
            Optional keys:
                - 'switch_cost_rate' (float): 切替コスト率 (default: 0.001)
                - 'min_holding_days' (int): 最小保有日数 (default: 1)
                - 'max_switches_per_month' (int): 月次切替制限 (default: 10)
                - 'enable_switch_optimization' (bool): 切替最適化有効化 (default: True)
    
    Raises:
        ConfigError: 設定値エラー
    """
```

### **3.2 メインメソッド**

```python
def evaluate_symbol_switch(self, from_symbol: str, to_symbol: str, 
                          target_date: datetime) -> Dict[str, Any]:
    """
    銘柄切替の必要性評価
    
    Parameters:
        from_symbol (str): 切替前銘柄 (None可)
        to_symbol (str): 切替後銘柄
        target_date (datetime): 対象日付
    
    Returns:
        Dict[str, Any]: 切替評価結果
            {
                'should_switch': bool,            # 切替実行フラグ
                'reason': str,                    # 切替理由
                    # 'initial_setup' | 'same_symbol' | 'dss_selection_changed' |
                    # 'min_holding_period_not_met' | 'monthly_switch_limit_exceeded'
                'from_symbol': str,               # 切替前銘柄
                'to_symbol': str,                 # 切替後銘柄
                'target_date': datetime,          # 対象日付
                'switch_cost_estimated': float,   # 推定切替コスト
                'evaluation_details': {
                    'days_held': int,             # 現在保有日数
                    'switches_this_month': int,   # 当月切替回数
                    'holding_period_ok': bool,    # 最小保有期間チェック
                    'monthly_limit_ok': bool      # 月次制限チェック
                }
            }
    
    Example:
        evaluation = switch_mgr.evaluate_symbol_switch(
            from_symbol='7203',
            to_symbol='9984',
            target_date=datetime(2023, 6, 15)
        )
        if evaluation['should_switch']:
            print(f"Switch approved: {evaluation['reason']}")
    """

def record_switch_executed(self, switch_result: Dict[str, Any]) -> None:
    """
    切替実行の記録
    
    Parameters:
        switch_result (Dict[str, Any]): 切替実行結果
            Required keys:
                - 'from_symbol' (str): 切替前銘柄
                - 'to_symbol' (str): 切替後銘柄
                - 'target_date' (datetime): 切替日付
                - 'switch_cost' (float): 実際の切替コスト
    
    Raises:
        ValueError: 必須キーの不足
        DataError: 履歴記録エラー
    
    Example:
        switch_result = {
            'from_symbol': '7203',
            'to_symbol': '9984',
            'target_date': datetime(2023, 6, 15),
            'switch_cost': 1000.0
        }
        switch_mgr.record_switch_executed(switch_result)
    """

def get_switch_statistics(self) -> Dict[str, Any]:
    """
    切替統計情報の取得
    
    Returns:
        Dict[str, Any]: 切替統計
            {
                'total_switches': int,            # 総切替回数
                'avg_holding_days': float,        # 平均保有日数
                'total_switch_cost': float,       # 総切替コスト
                'monthly_distribution': {
                    '2023-01': int,               # 月別切替回数
                    '2023-02': int,
                    # ...
                },
                'switch_reasons': {
                    'dss_selection_changed': int, # 理由別集計
                    'initial_setup': int,
                    # ...
                },
                'cost_efficiency': {
                    'avg_cost_per_switch': float, # 平均切替コスト
                    'cost_as_pct_portfolio': float # ポートフォリオ比コスト
                }
            }
    """
```

### **3.3 内部メソッド**

```python
def _check_min_holding_period(self, target_date: datetime) -> bool:
    """最小保有期間チェック"""

def _check_monthly_switch_limit(self, target_date: datetime) -> bool:
    """月次切替制限チェック"""

def _get_monthly_switch_count(self, target_date: datetime) -> int:
    """当月の切替回数取得"""

def _get_current_holding_days(self, target_date: datetime) -> int:
    """現在の保有日数取得"""
```

---

## 💼 **4. PositionManager**

### **クラス概要**
**役割**: ポジション管理・リスク制御  
**責任**: 銘柄切替時のポジション処理、リスク制限監視  
**依存**: RiskManagement

### **4.1 コンストラクター**

```python
def __init__(self, config: Dict[str, Any]) -> None:
    """
    ポジション管理の初期化
    
    Parameters:
        config (Dict[str, Any]): ポジション管理設定
            Optional keys:
                - 'max_positions_per_symbol' (int): 銘柄別最大ポジション数 (default: 5)
                - 'max_total_positions' (int): 総最大ポジション数 (default: 100)
                - 'enable_risk_limits' (bool): リスク制限有効化 (default: True)
                - 'max_portfolio_risk' (float): 最大ポートフォリオリスク (default: 0.10)
    
    Raises:
        ConfigError: 設定値エラー
    """
```

### **4.2 メインメソッド**

```python
def execute_symbol_switch(self, switch_result: Dict[str, Any], 
                         portfolio_value: float) -> Dict[str, Any]:
    """
    銘柄切替時のポジション処理実行
    
    Parameters:
        switch_result (Dict[str, Any]): 切替評価結果
        portfolio_value (float): 現在のポートフォリオ価値
    
    Returns:
        Dict[str, Any]: ポジション処理結果
            {
                'status': str,                    # 'success' | 'partial_success' | 'failed'
                'old_position_closed': {
                    'symbol': str,                # 決済銘柄
                    'quantity': float,            # 決済数量
                    'close_price': float,         # 決済価格
                    'realized_pnl': float         # 実現損益
                },
                'new_position_ready': {
                    'symbol': str,                # 新規銘柄
                    'max_quantity': float,        # 最大取引可能数量
                    'risk_allocation': float      # リスク配分
                },
                'risk_assessment': {
                    'portfolio_risk': float,      # ポートフォリオリスク
                    'position_concentration': float, # ポジション集中度
                    'risk_limit_status': str      # 'within_limits' | 'approaching_limits' | 'exceeds_limits'
                },
                'cash_impact': {
                    'cash_released': float,       # 解放キャッシュ
                    'available_cash': float       # 利用可能キャッシュ
                }
            }
    
    Raises:
        RiskError: リスク制限違反
        PositionError: ポジション処理エラー
    
    Example:
        position_result = pos_mgr.execute_symbol_switch(
            switch_result=switch_evaluation,
            portfolio_value=1000000
        )
        if position_result['status'] == 'success':
            print(f"Position switch completed: {position_result['old_position_closed']['realized_pnl']}")
    """

def check_risk_limits(self, symbol: str, position_size: float) -> bool:
    """
    リスク制限チェック
    
    Parameters:
        symbol (str): 銘柄コード
        position_size (float): ポジションサイズ
    
    Returns:
        bool: リスク制限内フラグ
    
    Example:
        if pos_mgr.check_risk_limits('7203', 1000000):
            # リスク制限内での取引実行
            pass
    """

def get_current_positions(self) -> Dict[str, Any]:
    """
    現在のポジション状況取得
    
    Returns:
        Dict[str, Any]: ポジション状況
            {
                'total_positions': int,           # 総ポジション数
                'total_market_value': float,      # 総時価
                'available_cash': float,          # 利用可能キャッシュ
                'portfolio_risk': float,          # ポートフォリオリスク
                'positions_by_symbol': {
                    '7203': {
                        'quantity': float,        # 保有数量
                        'avg_price': float,       # 平均取得価格
                        'current_price': float,   # 現在価格
                        'unrealized_pnl': float,  # 含み損益
                        'risk_contribution': float # リスク寄与度
                    },
                    # ... 他の銘柄
                },
                'risk_metrics': {
                    'var_1day': float,            # 1日VaR
                    'max_drawdown_risk': float,   # 最大ドローダウンリスク
                    'concentration_risk': float   # 集中リスク
                }
            }
    """
```

---

## 💾 **5. DataCacheManager**

### **クラス概要**
**役割**: 複数銘柄データの効率的キャッシュ管理  
**責任**: データキャッシュ戦略、メモリ管理、アクセス最適化  
**依存**: yfinance

### **5.1 コンストラクター**

```python
def __init__(self, config: Dict[str, Any]) -> None:
    """
    データキャッシュマネージャーの初期化
    
    Parameters:
        config (Dict[str, Any]): キャッシュ設定
            Optional keys:
                - 'cache_size_mb' (int): キャッシュサイズ制限MB (default: 100)
                - 'cache_retention_days' (int): キャッシュ保持日数 (default: 30)
                - 'enable_compression' (bool): データ圧縮有効化 (default: False)
                - 'cache_strategy' (str): キャッシュ戦略 'LRU'|'LFU' (default: 'LRU')
    
    Raises:
        ConfigError: 設定値エラー
    """
```

### **5.2 メインメソッド**

```python
def get_cached_data(self, symbol: str, start_date: datetime, 
                   end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    キャッシュからデータ取得
    
    Parameters:
        symbol (str): 銘柄コード
        start_date (datetime): 開始日
        end_date (datetime): 終了日
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (株価データ, インデックスデータ)
        キャッシュミスの場合は (None, None)
    
    Example:
        stock_data, index_data = cache_mgr.get_cached_data(
            symbol='7203',
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2023, 6, 30)
        )
        if stock_data is not None:
            print("Cache hit!")
    """

def store_cached_data(self, symbol: str, start_date: datetime, 
                     end_date: datetime, stock_data: pd.DataFrame, 
                     index_data: pd.DataFrame) -> None:
    """
    データをキャッシュに保存
    
    Parameters:
        symbol (str): 銘柄コード
        start_date (datetime): 開始日
        end_date (datetime): 終了日
        stock_data (pd.DataFrame): 株価データ
        index_data (pd.DataFrame): インデックスデータ
    
    Raises:
        CacheError: キャッシュ容量不足・保存エラー
    
    Example:
        cache_mgr.store_cached_data(
            symbol='7203',
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2023, 6, 30),
            stock_data=stock_df,
            index_data=index_df
        )
    """

def clear_cache(self, older_than_days: int = None) -> int:
    """
    キャッシュクリア
    
    Parameters:
        older_than_days (int, optional): 指定日数より古いデータを削除
    
    Returns:
        int: 削除されたキャッシュエントリ数
    
    Example:
        deleted_count = cache_mgr.clear_cache(older_than_days=7)
        print(f"Deleted {deleted_count} cache entries")
    """
```

### **5.3 統計・監視メソッド**

```python
def get_cache_statistics(self) -> Dict[str, Any]:
    """
    キャッシュ統計情報取得
    
    Returns:
        Dict[str, Any]: キャッシュ統計
            {
                'total_entries': int,             # 総エントリ数
                'total_size_mb': float,           # 総サイズMB
                'hit_rate': float,                # ヒット率
                'avg_access_time_ms': float,      # 平均アクセス時間
                'most_accessed_symbols': List[str], # 最頻アクセス銘柄
                'cache_efficiency': float         # キャッシュ効率
            }
    """
```

---

## 📊 **6. PerformanceTracker**

### **クラス概要**
**役割**: DSSMS統合システムのパフォーマンス監視  
**責任**: パフォーマンス計測、閾値監視、アラート生成  
**依存**: psutil

### **6.1 コンストラクター**

```python
def __init__(self) -> None:
    """
    パフォーマンストラッカーの初期化
    
    固定の目標値でシステムを監視:
        - max_daily_execution_time_ms: 1000ms
        - max_memory_usage_mb: 1024MB
        - min_success_rate: 95%
    """
```

### **6.2 メインメソッド**

```python
def record_daily_performance(self, daily_result: Dict[str, Any]) -> None:
    """
    日次パフォーマンス記録
    
    Parameters:
        daily_result (Dict[str, Any]): 日次処理結果
            Required keys:
                - 'execution_time_ms' (float): 実行時間
                - 'strategy_result' (Dict): 戦略実行結果
            Optional keys:
                - 'switch_result' (Dict): 切替結果
                - 'memory_usage_mb' (float): メモリ使用量
    
    Example:
        daily_result = {
            'execution_time_ms': 850.0,
            'strategy_result': {'status': 'success'},
            'switch_result': {'switch_cost_estimated': 0.001}
        }
        perf_tracker.record_daily_performance(daily_result)
    """

def get_performance_summary(self) -> Dict[str, Any]:
    """
    パフォーマンスサマリー取得
    
    Returns:
        Dict[str, Any]: パフォーマンス概要
            {
                'status': str,                    # 'no_data' | '有効データあり'
                'avg_execution_time_ms': float,   # 平均実行時間
                'max_execution_time_ms': float,   # 最大実行時間
                'avg_memory_usage_mb': float,     # 平均メモリ使用量
                'success_rate': float,            # 成功率
                'total_switch_cost': float,       # 総切替コスト
                'total_switches': int,            # 総切替回数
                'performance_status': str         # パフォーマンス評価
                    # 'excellent' | 'good' | 'acceptable' | 'needs_improvement'
            }
    
    Example:
        summary = perf_tracker.get_performance_summary()
        if summary['performance_status'] == 'needs_improvement':
            print("Performance optimization required")
    """

def should_check_performance(self, current_date: datetime) -> bool:
    """
    パフォーマンスチェックタイミング判定
    
    Parameters:
        current_date (datetime): 現在日付
    
    Returns:
        bool: チェック実行フラグ（週次・金曜日）
    
    Example:
        if perf_tracker.should_check_performance(datetime.now()):
            summary = perf_tracker.get_performance_summary()
            # パフォーマンス分析実行
    """
```

---

## 🚨 **7. エラーハンドリング仕様**

### **例外クラス階層**

```python
# カスタム例外定義
class DSSMSError(Exception):
    """DSSMS統合システム基底例外"""
    pass

class ConfigError(DSSMSError):
    """設定関連エラー"""
    pass

class DataError(DSSMSError):
    """データ関連エラー"""
    pass

class StrategyError(DSSMSError):
    """戦略実行エラー"""
    pass

class RiskError(DSSMSError):
    """リスク管理エラー"""
    pass

class PositionError(DSSMSError):
    """ポジション管理エラー"""
    pass

class CacheError(DSSMSError):
    """キャッシュ関連エラー"""
    pass

class SystemError(DSSMSError):
    """システムレベルエラー"""
    pass
```

### **エラーレベル定義**

| レベル | 対応 | 例 |
|--------|------|-----|
| **CRITICAL** | システム停止・管理者通知 | メモリ不足、設定ファイル破損 |
| **ERROR** | フォールバック・処理継続 | データ取得失敗、戦略実行失敗 |
| **WARNING** | ログ記録・継続 | パフォーマンス目標未達、切替制限超過 |
| **INFO/DEBUG** | 状態記録 | 銘柄切替実行、キャッシュヒット |

---

## 📋 **8. 型定義・定数**

### **共通型定義**

```python
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import pandas as pd

# 共通型エイリアス
ConfigDict = Dict[str, Any]
ResultDict = Dict[str, Any]
SymbolCode = str
DateTimeStamp = datetime
DataFrameTuple = Tuple[pd.DataFrame, pd.DataFrame]

# 戦略実行結果型
StrategyResult = Dict[str, Union[str, int, float, datetime, Dict[str, Any]]]

# 切替評価結果型
SwitchEvaluation = Dict[str, Union[bool, str, datetime, float, Dict[str, Any]]]

# パフォーマンス統計型
PerformanceMetrics = Dict[str, Union[int, float, str, Dict[str, Any]]]
```

### **定数定義**

```python
# システム定数
DEFAULT_INITIAL_CAPITAL = 1000000
DEFAULT_SWITCH_COST_RATE = 0.001
DEFAULT_MIN_HOLDING_DAYS = 1
DEFAULT_MAX_SWITCHES_PER_MONTH = 10

# パフォーマンス目標
PERFORMANCE_TARGETS = {
    'max_daily_execution_time_ms': 1000,
    'max_memory_usage_mb': 1024,
    'min_success_rate': 0.95
}

# 戦略リスト
AVAILABLE_STRATEGIES = [
    'VWAPBreakoutStrategy',
    'MomentumInvestingStrategy',
    'BreakoutStrategy',
    'VWAPBounceStrategy',
    'OpeningGapStrategy',
    'ContrarianStrategy',
    'GCStrategy'
]

# エラーメッセージ
ERROR_MESSAGES = {
    'invalid_config': "Invalid configuration provided",
    'data_fetch_failed': "Failed to fetch market data",
    'strategy_execution_failed': "Strategy execution failed",
    'risk_limit_exceeded': "Risk limit exceeded",
    'cache_capacity_exceeded': "Cache capacity exceeded"
}
```

---

## ✅ **インターフェース仕様書 - 完了**

### **Phase 2 成果物確認**

1. **✅ 6クラス全インターフェース定義完了**
   - メソッドシグネチャ
   - パラメータ・戻り値詳細仕様
   - 例外ハンドリング仕様

2. **✅ 型定義・定数整備完了**
   - 共通型エイリアス
   - システム定数
   - エラーメッセージ定義

3. **✅ エラーハンドリング階層確立**
   - カスタム例外クラス階層
   - エラーレベル分類
   - 処理方針明確化

**Phase 3: 実装・単体テスト** への準備完了

---

*インターフェース仕様書作成完了日: 2025年9月25日*  
*総仕様項目数: 25メソッド + 7例外クラス + 4型定義*  
*実装準備完了度: 100%*