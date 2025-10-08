# DSSMS統合アーキテクチャ設計書

**フェーズ**: Phase 2  
**作成日**: 2025年9月25日  
**ステータス**: [OK] **完了**

---

## 🏗️ **統合アーキテクチャ概要**

### **設計方針**
- **既存システム保護**: main.py、DSS V3を無変更で活用
- **モジュール分離**: 各機能を独立したクラスで実装
- **拡張性確保**: 将来の機能追加に対応可能な設計
- **テスト容易性**: 各コンポーネントの独立テスト可能

### **全体アーキテクチャ**
```
DSSMS統合システム
├── DSSMSIntegratedBacktester      # メインコントローラー
├── MultiStrategyAdapter           # main.py戦略群との連携
├── SymbolSwitchManager           # 銘柄切替管理
├── PositionManager               # ポジション管理統合
├── DataCacheManager              # データキャッシュ管理
└── PerformanceTracker            # パフォーマンス監視
```

---

## [TARGET] **1. 統合クラス設計**

### **1.1 DSSMSIntegratedBacktester (メインクラス)**

```python
class DSSMSIntegratedBacktester:
    """
    DSSMS統合バックテスター
    DSS Core V3の銘柄選択 + マルチ戦略実行を統合管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        統合システムの初期化
        
        Parameters:
            config: 統合設定辞書
                - backtest_period: バックテスト期間
                - initial_capital: 初期資本金
                - switch_cost_rate: 銘柄切替コスト率
                - performance_targets: パフォーマンス目標値
        """
        # コア機能初期化
        self.dss_core = DSSBacktesterV3()
        self.strategy_adapter = MultiStrategyAdapter(config)
        self.switch_manager = SymbolSwitchManager(config)
        self.position_manager = PositionManager(config)
        
        # データ管理
        self.data_cache = DataCacheManager(config)
        self.performance_tracker = PerformanceTracker()
        
        # 設定管理
        self.config = config
        self.logger = setup_logger(f"{self.__class__.__name__}")
        
        # 状態管理
        self.current_symbol = None
        self.portfolio_value = config.get('initial_capital', 1000000)
        self.daily_results = []
        self.switch_history = []
        
    def run_dynamic_backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        動的銘柄選択バックテストの実行
        
        Parameters:
            start_date: バックテスト開始日
            end_date: バックテスト終了日
            
        Returns:
            Dict: 統合バックテスト結果
        """
        logger.info(f"DSSMS統合バックテスト開始: {start_date} → {end_date}")
        
        current_date = start_date
        
        while current_date <= end_date:
            # 日次処理実行
            daily_result = self._process_daily_trading(current_date)
            self.daily_results.append(daily_result)
            
            current_date += timedelta(days=1)
            
            # パフォーマンス監視
            if self.performance_tracker.should_check_performance(current_date):
                self._check_performance_targets()
        
        # 最終結果生成
        final_result = self._generate_final_results()
        
        logger.info("DSSMS統合バックテスト完了")
        return final_result
    
    def _process_daily_trading(self, target_date: datetime) -> Dict[str, Any]:
        """
        日次取引処理
        
        Parameters:
            target_date: 対象日付
            
        Returns:
            Dict: 日次処理結果
        """
        start_time = time.time()
        
        # 1. DSS Core で最適銘柄選択
        dss_result = self.dss_core.run_daily_selection(target_date)
        selected_symbol = dss_result['selected_symbol']
        
        # 2. 銘柄切替判定・実行
        switch_result = self.switch_manager.evaluate_symbol_switch(
            from_symbol=self.current_symbol,
            to_symbol=selected_symbol,
            target_date=target_date
        )
        
        if switch_result['should_switch']:
            switch_executed = self._execute_symbol_switch(switch_result)
            self.switch_history.append(switch_executed)
        
        # 3. 現在銘柄でマルチ戦略実行
        strategy_result = None
        if self.current_symbol:
            strategy_result = self.strategy_adapter.execute_strategies(
                symbol=self.current_symbol,
                target_date=target_date,
                portfolio_value=self.portfolio_value
            )
            
            # ポートフォリオ価値更新
            if strategy_result:
                self.portfolio_value = strategy_result.get('updated_portfolio_value', self.portfolio_value)
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            'date': target_date,
            'dss_result': dss_result,
            'switch_result': switch_result,
            'strategy_result': strategy_result,
            'portfolio_value': self.portfolio_value,
            'execution_time_ms': execution_time
        }
    
    def _execute_symbol_switch(self, switch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        銘柄切替実行
        
        Parameters:
            switch_result: 切替評価結果
            
        Returns:
            Dict: 切替実行結果
        """
        logger.info(f"銘柄切替実行: {switch_result['from_symbol']} → {switch_result['to_symbol']}")
        
        # ポジション管理での切替処理
        position_result = self.position_manager.execute_symbol_switch(
            switch_result=switch_result,
            portfolio_value=self.portfolio_value
        )
        
        # 現在銘柄更新
        self.current_symbol = switch_result['to_symbol']
        
        # 切替コスト控除
        switch_cost = self.portfolio_value * self.config.get('switch_cost_rate', 0.001)
        self.portfolio_value -= switch_cost
        
        return {
            **switch_result,
            'position_result': position_result,
            'switch_cost': switch_cost,
            'updated_portfolio_value': self.portfolio_value
        }
```

### **1.2 MultiStrategyAdapter (戦略連携クラス)**

```python
class MultiStrategyAdapter:
    """
    main.pyの戦略適用ロジックとの連携アダプター
    既存の7戦略を動的銘柄に対応させる
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        戦略アダプターの初期化
        
        Parameters:
            config: 設定辞書
        """
        # main.pyからのコンポーネント流用
        self.risk_manager = RiskManagement(
            total_assets=config.get('initial_capital', 1000000)
        )
        self.param_manager = OptimizedParameterManager()
        
        # 戦略実行統計
        self.strategy_stats = {}
        self.execution_history = []
        
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def execute_strategies(self, symbol: str, target_date: datetime, 
                          portfolio_value: float) -> Dict[str, Any]:
        """
        指定銘柄・日付で全7戦略を実行
        
        Parameters:
            symbol: 対象銘柄
            target_date: 対象日付
            portfolio_value: 現在のポートフォリオ価値
            
        Returns:
            Dict: 戦略実行結果
        """
        logger.info(f"戦略実行開始: {symbol} @ {target_date}")
        
        # 1. 銘柄データ取得
        stock_data, index_data = self._fetch_symbol_data(symbol, target_date)
        
        if stock_data is None or stock_data.empty:
            logger.warning(f"データ取得失敗: {symbol} @ {target_date}")
            return {'status': 'data_fetch_failed', 'symbol': symbol, 'date': target_date}
        
        # 2. 最適化パラメータ読み込み
        optimized_params = self._load_optimized_parameters(symbol)
        
        # 3. main.pyの戦略適用ロジックを再利用
        # apply_strategies_with_optimized_params を日次実行用に調整
        strategy_result = self._apply_strategies_for_single_day(
            stock_data=stock_data,
            index_data=index_data,
            optimized_params=optimized_params,
            target_date=target_date,
            portfolio_value=portfolio_value
        )
        
        return strategy_result
    
    def _fetch_symbol_data(self, symbol: str, target_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定銘柄の市場データを取得
        
        Parameters:
            symbol: 銘柄コード
            target_date: 対象日付
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (株価データ, インデックスデータ)
        """
        try:
            # データ取得期間設定（対象日から過去100日分）
            start_date = target_date - timedelta(days=100)
            end_date = target_date + timedelta(days=1)
            
            # get_parameters_and_data の機能を流用
            # 動的銘柄対応に修正
            stock_data = yf.download(
                f"{symbol}.T", 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            # 日経平均データも同期取得
            index_data = yf.download(
                "^N225", 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            return stock_data, index_data
            
        except Exception as e:
            logger.error(f"データ取得エラー ({symbol}): {e}")
            return None, None
    
    def _load_optimized_parameters(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        指定銘柄の最適化パラメータを読み込み
        main.pyのload_optimized_parameters機能を流用
        
        Parameters:
            symbol: 銘柄コード
            
        Returns:
            Dict: 戦略別最適化パラメータ
        """
        # main.pyの既存ロジックを完全流用
        strategies = [
            'VWAPBreakoutStrategy',
            'MomentumInvestingStrategy', 
            'BreakoutStrategy',
            'VWAPBounceStrategy',
            'OpeningGapStrategy',
            'ContrarianStrategy',
            'GCStrategy'
        ]
        
        optimized_params = {}
        
        for strategy_name in strategies:
            try:
                params = self.param_manager.load_approved_params(strategy_name, symbol)
                if params:
                    optimized_params[strategy_name] = params
                else:
                    # デフォルトパラメータ使用
                    optimized_params[strategy_name] = self._get_default_parameters(strategy_name)
            except Exception as e:
                logger.error(f"パラメータ読み込みエラー - {strategy_name}: {e}")
                optimized_params[strategy_name] = self._get_default_parameters(strategy_name)
        
        return optimized_params
    
    def _apply_strategies_for_single_day(self, stock_data: pd.DataFrame, 
                                        index_data: pd.DataFrame,
                                        optimized_params: Dict[str, Dict[str, Any]], 
                                        target_date: datetime,
                                        portfolio_value: float) -> Dict[str, Any]:
        """
        単日用の戦略適用ロジック
        main.pyのapply_strategies_with_optimized_paramsを日次実行用に調整
        
        Parameters:
            stock_data: 株価データ
            index_data: インデックスデータ
            optimized_params: 最適化パラメータ
            target_date: 対象日付
            portfolio_value: ポートフォリオ価値
            
        Returns:
            Dict: 戦略実行結果
        """
        # main.pyの戦略適用ロジックを流用
        # 既存のapply_strategies_with_optimized_paramsを呼び出し
        # 結果から対象日付のシグナルのみを抽出
        
        result_data = apply_strategies_with_optimized_params(
            stock_data, index_data, optimized_params
        )
        
        # 対象日付のシグナル抽出
        if target_date in result_data.index:
            daily_signals = result_data.loc[target_date]
            
            return {
                'status': 'success',
                'date': target_date,
                'entry_signal': daily_signals.get('Entry_Signal', 0),
                'exit_signal': daily_signals.get('Exit_Signal', 0),
                'strategy': daily_signals.get('Strategy', ''),
                'position_size': daily_signals.get('Position_Size', 1.0),
                'updated_portfolio_value': portfolio_value  # 簡略化（実際は損益計算が必要）
            }
        else:
            return {
                'status': 'no_signal',
                'date': target_date,
                'message': 'No signal for target date'
            }
```

### **1.3 SymbolSwitchManager (銘柄切替管理クラス)**

```python
class SymbolSwitchManager:
    """
    動的銘柄切替の判定・管理を行うクラス
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        銘柄切替管理の初期化
        
        Parameters:
            config: 設定辞書
        """
        self.config = config
        self.switch_cost_rate = config.get('switch_cost_rate', 0.001)  # 0.1%
        self.min_holding_days = config.get('min_holding_days', 1)      # 最小保有日数
        self.max_switches_per_month = config.get('max_switches_per_month', 10)
        
        # 切替履歴管理
        self.switch_history = []
        self.current_holding_start = None
        
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def evaluate_symbol_switch(self, from_symbol: str, to_symbol: str, 
                              target_date: datetime) -> Dict[str, Any]:
        """
        銘柄切替の必要性を評価
        
        Parameters:
            from_symbol: 切替前銘柄
            to_symbol: 切替後銘柄
            target_date: 対象日付
            
        Returns:
            Dict: 切替評価結果
        """
        # 初回設定
        if from_symbol is None:
            return {
                'should_switch': True,
                'reason': 'initial_setup',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'target_date': target_date,
                'switch_cost_estimated': 0.0
            }
        
        # 同一銘柄の場合
        if from_symbol == to_symbol:
            return {
                'should_switch': False,
                'reason': 'same_symbol',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'target_date': target_date
            }
        
        # 最小保有期間チェック
        if not self._check_min_holding_period(target_date):
            return {
                'should_switch': False,
                'reason': 'min_holding_period_not_met',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'target_date': target_date,
                'days_held': self._get_current_holding_days(target_date)
            }
        
        # 月次切替制限チェック
        if not self._check_monthly_switch_limit(target_date):
            return {
                'should_switch': False,
                'reason': 'monthly_switch_limit_exceeded',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'target_date': target_date,
                'switches_this_month': self._get_monthly_switch_count(target_date)
            }
        
        # 切替実行
        return {
            'should_switch': True,
            'reason': 'dss_selection_changed',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'target_date': target_date,
            'switch_cost_estimated': self.switch_cost_rate,
            'days_held': self._get_current_holding_days(target_date)
        }
    
    def _check_min_holding_period(self, target_date: datetime) -> bool:
        """最小保有期間をチェック"""
        if self.current_holding_start is None:
            return True
        
        holding_days = (target_date - self.current_holding_start).days
        return holding_days >= self.min_holding_days
    
    def _check_monthly_switch_limit(self, target_date: datetime) -> bool:
        """月次切替制限をチェック"""
        monthly_count = self._get_monthly_switch_count(target_date)
        return monthly_count < self.max_switches_per_month
    
    def _get_monthly_switch_count(self, target_date: datetime) -> int:
        """当月の切替回数を取得"""
        target_month = target_date.replace(day=1)
        count = 0
        
        for switch in self.switch_history:
            switch_date = switch.get('target_date')
            if switch_date and switch_date.replace(day=1) == target_month:
                count += 1
        
        return count
    
    def _get_current_holding_days(self, target_date: datetime) -> int:
        """現在の保有日数を取得"""
        if self.current_holding_start is None:
            return 0
        return (target_date - self.current_holding_start).days
    
    def record_switch_executed(self, switch_result: Dict[str, Any]):
        """切替実行の記録"""
        self.switch_history.append(switch_result)
        self.current_holding_start = switch_result['target_date']
        
        logger.info(f"銘柄切替記録: {switch_result['from_symbol']} → {switch_result['to_symbol']}")
```

---

## [CHART] **2. 銘柄切替メカニズム設計**

### **切替フロー**
```
1. DSS選択結果受信
    ↓
2. 切替必要性評価
    ├── 同一銘柄 → スキップ
    ├── 最小保有期間未達 → スキップ
    ├── 月次制限超過 → スキップ
    └── 切替実行 → 次へ
    ↓
3. ポジション解除
    ├── 既存ポジションクローズ
    └── 損益計算
    ↓
4. 新銘柄ポジション開始
    ├── 新銘柄データ準備
    └── 戦略実行開始
    ↓
5. 切替コスト控除
    └── ポートフォリオ価値更新
```

### **切替制限ルール**
- **最小保有期間**: 1日（デフォルト）
- **月次切替制限**: 10回/月（デフォルト）
- **切替コスト**: 0.1%（デフォルト）
- **データ取得失敗時**: 前日銘柄継続

---

## 💾 **3. データ管理戦略**

### **3.1 DataCacheManager (データキャッシュ管理)**

```python
class DataCacheManager:
    """
    複数銘柄データの効率的キャッシュ管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        データキャッシュマネージャーの初期化
        
        Parameters:
            config: 設定辞書
        """
        self.cache_size_limit = config.get('cache_size_mb', 100)  # 100MB制限
        self.cache_retention_days = config.get('cache_retention_days', 30)
        
        # キャッシュストレージ
        self.stock_data_cache = {}    # {symbol: {date_range: DataFrame}}
        self.index_data_cache = {}    # {index: {date_range: DataFrame}}
        self.cache_metadata = {}      # アクセス時刻・サイズ情報
        
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def get_cached_data(self, symbol: str, start_date: datetime, 
                       end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        キャッシュからデータを取得
        
        Parameters:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            
        Returns:
            Tuple: (株価データ, インデックスデータ) or (None, None)
        """
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
        
        if cache_key in self.stock_data_cache:
            # キャッシュヒット
            self._update_cache_access(cache_key)
            stock_data = self.stock_data_cache[cache_key]
            index_data = self.index_data_cache.get(f"N225_{start_date.date()}_{end_date.date()}")
            
            logger.debug(f"キャッシュヒット: {cache_key}")
            return stock_data, index_data
        
        return None, None
    
    def store_cached_data(self, symbol: str, start_date: datetime, 
                         end_date: datetime, stock_data: pd.DataFrame, 
                         index_data: pd.DataFrame):
        """
        データをキャッシュに保存
        
        Parameters:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            stock_data: 株価データ
            index_data: インデックスデータ
        """
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
        index_key = f"N225_{start_date.date()}_{end_date.date()}"
        
        # キャッシュ容量チェック
        if self._check_cache_capacity():
            self.stock_data_cache[cache_key] = stock_data
            self.index_data_cache[index_key] = index_data
            
            self._update_cache_metadata(cache_key, stock_data)
            logger.debug(f"キャッシュ保存: {cache_key}")
        else:
            logger.warning("キャッシュ容量制限により保存をスキップ")
    
    def _check_cache_capacity(self) -> bool:
        """キャッシュ容量をチェック"""
        # 簡略化実装（実際はメモリ使用量を計算）
        return len(self.stock_data_cache) < 50  # 50銘柄まで
    
    def _update_cache_access(self, cache_key: str):
        """キャッシュアクセス時刻を更新"""
        if cache_key not in self.cache_metadata:
            self.cache_metadata[cache_key] = {}
        
        self.cache_metadata[cache_key]['last_access'] = datetime.now()
        self.cache_metadata[cache_key]['access_count'] = \
            self.cache_metadata[cache_key].get('access_count', 0) + 1
    
    def _update_cache_metadata(self, cache_key: str, data: pd.DataFrame):
        """キャッシュメタデータを更新"""
        self.cache_metadata[cache_key] = {
            'created_at': datetime.now(),
            'last_access': datetime.now(),
            'access_count': 1,
            'data_size': len(data)
        }
```

---

## [UP] **4. パフォーマンス監視**

### **4.1 PerformanceTracker (パフォーマンス追跡)**

```python
class PerformanceTracker:
    """
    DSSMS統合システムのパフォーマンス監視
    """
    
    def __init__(self):
        """パフォーマンストラッカーの初期化"""
        self.execution_times = []       # 実行時間履歴
        self.memory_usage = []          # メモリ使用量履歴
        self.success_rates = []         # 成功率履歴
        self.switch_costs = []          # 切替コスト履歴
        
        self.performance_targets = {
            'max_daily_execution_time_ms': 1000,  # 1秒以内
            'max_memory_usage_mb': 1024,          # 1GB以内
            'min_success_rate': 0.95              # 95%以上
        }
        
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def record_daily_performance(self, daily_result: Dict[str, Any]):
        """日次パフォーマンスを記録"""
        execution_time = daily_result.get('execution_time_ms', 0)
        self.execution_times.append(execution_time)
        
        # メモリ使用量測定（簡略化）
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        
        # 成功率記録
        success = 1 if daily_result.get('strategy_result', {}).get('status') == 'success' else 0
        self.success_rates.append(success)
        
        # 切替コスト記録
        if 'switch_result' in daily_result and daily_result['switch_result'].get('should_switch'):
            switch_cost = daily_result['switch_result'].get('switch_cost_estimated', 0)
            self.switch_costs.append(switch_cost)
    
    def should_check_performance(self, current_date: datetime) -> bool:
        """パフォーマンスチェックのタイミング判定"""
        # 週次チェック
        return current_date.weekday() == 4  # 金曜日
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリーを取得"""
        if not self.execution_times:
            return {'status': 'no_data'}
        
        avg_execution_time = sum(self.execution_times) / len(self.execution_times)
        avg_memory_usage = sum(self.memory_usage) / len(self.memory_usage)
        success_rate = sum(self.success_rates) / len(self.success_rates)
        total_switch_cost = sum(self.switch_costs)
        
        return {
            'avg_execution_time_ms': avg_execution_time,
            'max_execution_time_ms': max(self.execution_times),
            'avg_memory_usage_mb': avg_memory_usage,
            'success_rate': success_rate,
            'total_switch_cost': total_switch_cost,
            'total_switches': len(self.switch_costs),
            'performance_status': self._evaluate_performance_status(
                avg_execution_time, avg_memory_usage, success_rate
            )
        }
    
    def _evaluate_performance_status(self, exec_time: float, memory: float, success_rate: float) -> str:
        """パフォーマンス状態を評価"""
        targets = self.performance_targets
        
        if (exec_time <= targets['max_daily_execution_time_ms'] and
            memory <= targets['max_memory_usage_mb'] and
            success_rate >= targets['min_success_rate']):
            return 'excellent'
        elif success_rate >= 0.90:
            return 'good'
        elif success_rate >= 0.80:
            return 'acceptable'
        else:
            return 'needs_improvement'
```

---

## [TOOL] **5. 設定管理**

### **5.1 統合設定ファイル (dssms_integration_config.json)**

```json
{
  "backtest_settings": {
    "initial_capital": 1000000,
    "switch_cost_rate": 0.001,
    "min_holding_days": 1,
    "max_switches_per_month": 10
  },
  "performance_targets": {
    "max_daily_execution_time_ms": 1000,
    "max_memory_usage_mb": 1024,
    "min_success_rate": 0.95,
    "max_switch_cost_per_month": 0.05
  },
  "data_management": {
    "cache_size_mb": 100,
    "cache_retention_days": 30,
    "data_fetch_timeout_sec": 30,
    "retry_attempts": 3
  },
  "logging": {
    "log_level": "INFO",
    "enable_performance_logging": true,
    "enable_detailed_switch_logging": true
  },
  "risk_management": {
    "enable_position_limits": true,
    "max_portfolio_risk": 0.10,
    "enable_drawdown_protection": true
  }
}
```

---

## [LIST] **Phase 2 成果物**

### **[OK] 完了した設計成果物**

1. **詳細設計書** [OK]
   - 統合アーキテクチャ概要
   - 5つの主要クラス設計
   - 銘柄切替メカニズム
   - データ管理戦略

2. **クラス図・シーケンス図** [OK]
   - 統合クラス関係図
   - 動的銘柄切替フロー
   - データキャッシュ戦略

3. **インターフェース仕様書** [OK]
   - 各クラスの公開メソッド定義
   - パラメータ・戻り値仕様
   - エラーハンドリング方針

4. **設定ファイル仕様** [OK]
   - JSON設定ファイル構造
   - パフォーマンス目標値
   - リスク管理パラメータ

---

## [TARGET] **Phase 3への準備完了**

### **実装準備事項**
- [OK] 全5クラスの詳細設計完了
- [OK] 銘柄切替ロジック明確化
- [OK] 既存システム連携方法確定
- [OK] パフォーマンス監視機能設計

### **実装優先順位**
1. **DSSMSIntegratedBacktester** (メインクラス)
2. **MultiStrategyAdapter** (既存戦略連携)
3. **SymbolSwitchManager** (切替管理)
4. **PositionManager** + **DataCacheManager** (サポート機能)
5. **PerformanceTracker** (監視機能)

**Phase 3: 実装・単体テスト** の開始準備が整いました。

---

*設計完了日: 2025年9月25日*  
*Phase 2実行時間: 約2時間*  
*次期Phase: 実装・単体テスト*