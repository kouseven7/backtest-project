"""
戦略実行管理システム
既存の戦略システムとの統合 + シンプル実行
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from datetime import datetime, timedelta
import pandas as pd

# プロジェクトパス追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
# Phase 4.2-16: 日本株手数料計算・100株単位対応
from src.execution.commission_calculator import calculate_max_affordable_quantity, adjust_to_trading_unit
# Phase 4.2-32: OrderStatusインポート
from src.execution.order_manager import OrderStatus
# Phase 5-B-5: 損益推移記録
from main_system.performance.equity_curve_recorder import EquityCurveRecorder

class StrategyExecutionManager:
    """戦略実行管理メインクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("StrategyExecutionManager", log_file="logs/strategy_execution.log")
        
        # 実行モード設定
        self.mode = config.get('execution_mode', 'simple')  # simple / integrated
        
        # コンポーネント初期化（遅延読み込み）
        self.data_feed = None
        self.paper_broker = None
        self.trade_executor = None
        self.strategy_manager = None
        self.strategy_selector = None
        self.multi_strategy_manager = None
        
        # [Phase 5-B-4] リスク管理統合
        self.integrated_manager = None
        
        # [Phase 5-B-5] 損益推移記録
        self.equity_recorder: Optional[EquityCurveRecorder] = None
        
        # [修正案2] ForceClose実行フラグ（2025-12-08追加）
        # 目的: ForceClose実行中の通常SELL処理を抑制し、同日2件SELL問題を解消
        self.force_close_in_progress = False
        
        # 実行履歴
        self.execution_history: List[Dict[str, Any]] = []
        
        # 初期化を試行
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """コンポーネント初期化"""
        try:
            # データフィード初期化（Phase 4.2: yfinance統合）
            try:
                from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
                self.data_feed = YFinanceDataFeed()
                self.logger.info("Data feed: YFinanceDataFeed initialized successfully")
            except ImportError as e:
                self.logger.warning(f"YFinanceDataFeed import failed: {e}")
                self.data_feed = None
                self.logger.info("Data feed: Disabled (yfinance not available)")
            
            # ペーパーブローカー初期化
            # Phase 4.2-15: 初期資金設定の修正（broker設定とトップレベル設定の両方を確認）
            from src.execution.paper_broker import PaperBroker
            broker_config = self.config.get('broker', {})
            
            # Phase 4.2-15: 初期資金取得ロジック強化
            # 優先順位: broker.initial_cash > トップレベルinitial_cash > デフォルト1,000,000円
            initial_cash = broker_config.get('initial_cash') or self.config.get('initial_cash', 1000000)
            
            self.paper_broker = PaperBroker(
                initial_balance=initial_cash,  # Phase 4.2-15: 修正済み
                commission_per_trade=broker_config.get('commission_per_trade', 1.0),
                slippage_pct=broker_config.get('slippage_bps', 5) / 10000.0,
                backtest_mode=True  # Phase 4.2-11: バックテスト時は市場時間チェックをスキップ
            )
            
            self.logger.info(f"PaperBroker initialized: initial_balance={initial_cash}円")
            
            # 取引実行エンジン初期化
            from src.execution.trade_executor import TradeExecutor
            from src.execution.portfolio_tracker import PortfolioTracker
            
            portfolio_tracker = PortfolioTracker()
            self.trade_executor = TradeExecutor(portfolio_tracker, self.paper_broker)
            
            self.logger.info("基本コンポーネント初期化完了")
            
            # 統合モード用コンポーネント
            if self.mode == 'integrated':
                self._initialize_integrated_components()
                
        except ImportError as e:
            self.logger.warning(f"データフィード初期化失敗: {e}")
            self.data_feed = None
        except Exception as e:
            self.logger.error(f"データフィード初期化エラー: {e}")
            self.data_feed = None
    
    def _initialize_integrated_components(self) -> None:
        """統合モード用コンポーネント初期化"""
        try:
            from config.strategy_selector import StrategySelector
            from config.multi_strategy_manager import MultiStrategyManager
            
            self.strategy_selector = StrategySelector(self.config.get('strategy_selector', {}))
            self.multi_strategy_manager = MultiStrategyManager(self.config.get('multi_strategy', {}))
            
            self.logger.info("統合モード用コンポーネント初期化完了")
        except Exception as e:
            self.logger.warning(f"統合モード初期化失敗、シンプルモードに切替: {e}")
            self.mode = 'simple'
    
    def set_integrated_manager(self, integrated_manager):
        """
        IntegratedExecutionManagerを設定（Phase 5-B-4追加）
        
        Args:
            integrated_manager: IntegratedExecutionManagerインスタンス
        """
        self.integrated_manager = integrated_manager
        self.logger.info("[PHASE_5_B_4] IntegratedExecutionManager設定完了")
    
    def execute_strategy(self, strategy_name: str, symbols: Optional[List[str]] = None, 
                        stock_data: Optional[pd.DataFrame] = None,
                        index_data: Optional[pd.DataFrame] = None,
                        trading_start_date: Optional[pd.Timestamp] = None,
                        trading_end_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        単一戦略実行（シンプルモード）
        
        Args:
            strategy_name: 戦略名
            symbols: ティッカーシンボルリスト
            stock_data: 株価データ（Phase 4.2: yfinanceから取得済み）
            index_data: インデックスデータ（Phase 4.2: yfinanceから取得済み）
            trading_start_date: 取引開始日（ウォームアップ期間後）
            trading_end_date: 取引終了日
        
        Returns:
            実行結果辞書
        """
        try:
            self.logger.info(f"戦略実行開始: {strategy_name}")
            
            # デフォルトシンボル
            if symbols is None:
                symbols = self.config.get('default_symbols', ['AAPL', 'MSFT'])
            
            # データ取得（Phase 4.2: 引数で渡されたデータを優先）
            if stock_data is None or index_data is None:
                market_data = self._get_market_data(symbols)
                if market_data is None or market_data.empty:
                    return self._create_error_result("market_data_unavailable")
                # 簡易実装：market_dataをstock_dataとして使用
                stock_data = market_data
                index_data = market_data  # フォールバック
            
            # 戦略インスタンス取得（Phase 4.2: データを渡す + ticker対応）
            # Phase 5.3: マルチ戦略システム対応 - tickerパラメータ追加
            ticker = symbols[0] if symbols else None  # 最初のティッカーシンボルを取得
            strategy = self._get_strategy_instance(strategy_name, stock_data, index_data, ticker=ticker)
            if strategy is None:
                return self._create_error_result(f"strategy_not_found: {strategy_name}")
            
            # 戦略実行（ウォームアップ期間対応）
            signals = strategy.backtest(
                trading_start_date=trading_start_date,
                trading_end_date=trading_end_date
            )
            
            # Phase 4.2-5-3: 戦略のbacktest結果を保持（取引統合のため）
            self.current_strategy = strategy
            self.current_strategy_name = strategy_name
            self.current_backtest_signals = signals  # バックテストシグナルを保持
            
            # 取引実行
            execution_results = self._execute_trades(signals, symbols)
            
            # 結果記録
            result = {
                "success": True,
                "strategy": strategy_name,
                "strategy_instance": strategy,  # Phase 4.2-5-3: 戦略インスタンスを保持
                "backtest_signals": signals,     # Phase 4.2-5-3: バックテストシグナルを保持
                "symbols": symbols,
                "signals_generated": len(signals) if signals is not None else 0,
                "trades_executed": len(execution_results),
                "execution_details": execution_results,
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_history.append(result)
            self.logger.info(f"戦略実行完了: {strategy_name}")
            
            return result
            
        except Exception as e:
            error_result = self._create_error_result(f"execution_error: {str(e)}")
            self.logger.error(f"戦略実行エラー[{strategy_name}]: {e}")
            return error_result
    
    def execute_integrated_strategies(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """統合戦略実行（統合モード）"""
        try:
            if self.mode != 'integrated' or self.strategy_selector is None:
                return self._create_error_result("integrated_mode_not_available")
            
            self.logger.info("統合戦略実行開始")
            
            # デフォルトシンボル
            if symbols is None:
                symbols = self.config.get('default_symbols', ['AAPL', 'MSFT'])
            
            # データ取得
            market_data = self._get_market_data(symbols)
            if market_data is None or market_data.empty:
                return self._create_error_result("market_data_unavailable")
            
            # 戦略選択
            selected_strategies = self.strategy_selector.select_strategies(market_data)
            
            # 複数戦略実行
            strategy_results = {}
            total_trades = 0
            
            for strategy_name in selected_strategies:
                try:
                    result = self._execute_single_strategy_integrated(strategy_name, market_data, symbols)
                    strategy_results[strategy_name] = result
                    total_trades += len(result.get('trades', []))
                except Exception as e:
                    self.logger.error(f"統合戦略実行エラー[{strategy_name}]: {e}")
                    strategy_results[strategy_name] = {"error": str(e)}
            
            # ポートフォリオ重み調整
            portfolio_weights = self._calculate_portfolio_weights(strategy_results)
            
            # 統合結果
            result = {
                "success": True,
                "mode": "integrated",
                "symbols": symbols,
                "selected_strategies": selected_strategies,
                "strategy_results": strategy_results,
                "portfolio_weights": portfolio_weights,
                "total_trades": total_trades,
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_history.append(result)
            self.logger.info(f"統合戦略実行完了: {len(selected_strategies)}戦略")
            
            return result
            
        except Exception as e:
            error_result = self._create_error_result(f"integrated_execution_error: {str(e)}")
            self.logger.error(f"統合戦略実行エラー: {e}")
            return error_result
    
    def _get_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """リアルタイム市場データ取得（Phase 4.2: yfinance統合版）"""
        try:
            # 過去Nペリオドのデータ取得（戦略計算用）
            lookback_days = self.config.get('lookback_days', 365)
            
            if self.data_feed is not None:
                # YFinanceDataFeed使用（Phase 4.2実装）
                # 現在は単一銘柄のみ対応（複数銘柄は将来実装）
                ticker = symbols[0] if symbols else "AAPL"
                
                try:
                    data = self.data_feed.get_stock_data(
                        ticker=ticker,
                        days_back=lookback_days
                    )
                    
                    if data is not None and not data.empty:
                        self.logger.debug(f"市場データ取得成功: {len(data)}行, ticker={ticker}")
                        return data
                    else:
                        self.logger.error(f"yfinance returned empty data for {ticker}")
                        return pd.DataFrame()
                        
                except Exception as e:
                    self.logger.error(f"yfinance data retrieval error for {ticker}: {e}")
                    return pd.DataFrame()
            
            # データフィードが無効な場合はエラー
            self.logger.error("CRITICAL: Market data unavailable. Data feed is None. Cannot proceed with backtest.")
            return pd.DataFrame()  # 空のDataFrameを返してエラーとして扱う
                
        except Exception as e:
            self.logger.error(f"Market data retrieval error: {e}")
            return pd.DataFrame()
    
    def _get_strategy_instance(self, strategy_name: str, stock_data: pd.DataFrame, 
                              index_data: pd.DataFrame, ticker: Optional[str] = None):
        """
        戦略インスタンス取得（Phase 4.2: データ必須版 + Phase 5.3: ticker対応）
        
        Args:
            strategy_name: 戦略名
            stock_data: 株価データ
            index_data: インデックスデータ
            ticker: ティッカーシンボル（最適化パラメータ読み込み用）
        
        Returns:
            戦略インスタンス（失敗時はNone）
        """
        try:
            # Phase 5-A-12修正: 戦略名→ファイル名マッピング（実際のファイル名に合わせる）
            strategy_mappings = {
                # 既存形式（アンダースコア区切り）
                'VWAP_Breakout': 'strategies.VWAP_Breakout.VWAPBreakoutStrategy',
                'VWAP_Bounce': 'strategies.VWAP_Bounce.VWAPBounceStrategy',
                'GC_Strategy': 'strategies.gc_strategy_signal.GCStrategy',
                'Breakout': 'strategies.Breakout.BreakoutStrategy',
                'Opening_Gap': 'strategies.Opening_Gap.OpeningGapStrategy',
                
                # 新形式（クラス名そのまま）- Phase 5-A-12: 実ファイル名に合わせて修正
                'VWAPBreakoutStrategy': 'strategies.VWAP_Breakout.VWAPBreakoutStrategy',
                'VWAPBounceStrategy': 'strategies.VWAP_Bounce.VWAPBounceStrategy',
                'GCStrategy': 'strategies.gc_strategy_signal.GCStrategy',
                'BreakoutStrategy': 'strategies.Breakout.BreakoutStrategy',
                'OpeningGapStrategy': 'strategies.Opening_Gap.OpeningGapStrategy',
                'OpeningGapFixedStrategy': 'strategies.Opening_Gap_Fixed.OpeningGapFixedStrategy',
                'MomentumInvestingStrategy': 'strategies.Momentum_Investing.MomentumInvestingStrategy',  # Phase 5-A-12修正
                'ContrarianStrategy': 'strategies.contrarian_strategy.ContrarianStrategy',
            }
            
            module_path = strategy_mappings.get(strategy_name)
            if not module_path:
                self.logger.error(f"CRITICAL: Unknown strategy name: '{strategy_name}'. Available strategies: {list(strategy_mappings.keys())}")
                return None
            
            # モジュールとクラスを分離
            module_name, class_name = module_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            
            # 戦略インスタンス化（Phase 4.2: データを渡す）
            # copilot-instructions.md: ダミーデータ生成禁止、実データ必須
            self.logger.info(f"Creating strategy instance: {strategy_name} with real data")
            
            # Phase 5-A-12修正: 段階的インスタンス化（柔軟な引数対応 + Phase 5.3: ticker対応）
            try:
                # 最初の試行: data + index_data + ticker（フル引数）
                strategy_instance = strategy_class(data=stock_data, index_data=index_data, ticker=ticker)
                self.logger.info(f"Strategy instance created with data+index_data+ticker: {strategy_name}")
                return strategy_instance
            except TypeError as e1:
                # 2回目の試行: dataのみ + ticker
                try:
                    self.logger.warning(f"Strategy {strategy_name} does not accept index_data, trying with data+ticker only")
                    strategy_instance = strategy_class(data=stock_data, ticker=ticker)
                    self.logger.info(f"Strategy instance created with data+ticker only: {strategy_name}")
                    return strategy_instance
                except TypeError as e2:
                    # 3回目の試行: stock_data + ticker（パラメータ名が異なる可能性）
                    try:
                        self.logger.warning(f"Strategy {strategy_name} does not accept 'data', trying with stock_data+ticker parameter")
                        strategy_instance = strategy_class(stock_data=stock_data, ticker=ticker)
                        self.logger.info(f"Strategy instance created with stock_data+ticker: {strategy_name}")
                        return strategy_instance
                    except Exception as e3:
                        # 4回目の試行: dataのみ（ticker非対応の旧戦略）
                        try:
                            self.logger.warning(f"Strategy {strategy_name} does not accept ticker, trying data only (legacy)")
                            strategy_instance = strategy_class(data=stock_data)
                            self.logger.info(f"Strategy instance created with data only (legacy): {strategy_name}")
                            return strategy_instance
                        except Exception as e4:
                            self.logger.error(f"CRITICAL: Strategy '{strategy_name}' initialization failed with all attempts. Errors: {e1}, {e2}, {e3}, {e4}")
                            return None
            
        except Exception as e:
            self.logger.error(f"Strategy instance creation error [{strategy_name}]: {e}", exc_info=True)
            return None
    
    def _execute_trades(self, signals: pd.DataFrame, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        取引実行（Phase 4.2-5-2: TradeExecutor統合版）
        
        Args:
            signals: 戦略からのシグナルDataFrame
            symbols: ティッカーシンボルリスト
        
        Returns:
            実行結果リスト
        """
        try:
            execution_results = []
            
            if signals is None or signals.empty:
                self.logger.warning("No signals to execute")
                return execution_results
            
            # trade_executor必須チェック
            if not self.trade_executor:
                self.logger.error("CRITICAL: Trade executor not available. Cannot execute trades without violating copilot-instructions.md (no mock execution allowed).")
                return []
            
            # Phase 4.2-31-B-1: REJECTEDされたBUY注文をトラッキング
            # Phase 4.2-31-B-4-2: symbol単位→インデックス単位に変更
            # copilot-instructions.md準拠: 実データのみ、モック禁止
            rejected_buy_indices = set()  # REJECTEDされたBUY注文のorder_index集合
            
            # シグナルから取引指示を生成（Phase 5.3: strategy_name追加）
            strategy_name = getattr(self, 'current_strategy_name', 'Unknown')
            trade_orders = self._generate_trade_orders(signals, symbols, strategy_name)
            
            if not trade_orders:
                self.logger.info("No trade orders generated from signals")
                return execution_results
            
            # 各注文を実行（実際の実行のみ）
            for order_dict in trade_orders:
                try:
                    symbol = order_dict['symbol']
                    order_index = order_dict.get('order_index', -1)  # Phase 4.2-31-B-4-3
                    
                    # Phase 4.2-32-2: ループ開始ログ
                    self.logger.info(
                        f"[PHASE_4_2_32_2] 注文処理開始: order_index={order_index}, "
                        f"symbol={symbol}, action={order_dict['action']}, quantity={order_dict['quantity']}"
                    )
                    
                    # Phase 4.2-31-B-4-3: SELL注文の場合、対応するBUY注文（order_index - 1）がREJECTEDされていればスキップ
                    # copilot-instructions.md準拠: BUY/SELLペア不一致を防ぐ（実データのみ）
                    # 仮定: BUY注文の直後にSELL注文が生成される（order_index順）
                    if order_dict['action'] == 'SELL':
                        # [修正案2] ForceClose実行中は通常SELL処理をスキップ（2025-12-08追加）
                        if self.force_close_in_progress:
                            strategy_name = order_dict.get('strategy_name', 'UnknownStrategy')
                            if strategy_name == 'UnknownStrategy':
                                self.logger.warning(
                                    f"[FALLBACK] 戦略名が取得できませんでした（ForceClose抑制ログ）: order_dict={order_dict.keys()}, "
                                    f"デフォルト値='UnknownStrategy'"
                                )
                            self.logger.warning(
                                f"[FORCE_CLOSE_SUPPRESS] ForceClose実行中のため通常SELL処理をスキップ: "
                                f"{symbol}, order_index={order_index}, strategy={strategy_name}"
                            )
                            continue  # 通常SELL処理をスキップ
                        
                        # 対応するBUY注文のインデックスを特定（SELL注文の直前）
                        corresponding_buy_index = order_index - 1
                        
                        # Phase 4.2-32-2: SELL注文チェック詳細ログ
                        self.logger.info(
                            f"[PHASE_4_2_32_2] SELL注文チェック: order_index={order_index}, "
                            f"corresponding_buy_index={corresponding_buy_index}, "
                            f"rejected_buy_indices={rejected_buy_indices}"
                        )
                        
                        if corresponding_buy_index in rejected_buy_indices:
                            self.logger.warning(
                                f"[PHASE_4_2_31_B] SELL注文スキップ（対応BUY REJECTED）: {symbol}, "
                                f"order_index={order_index}, corresponding_buy_index={corresponding_buy_index}, "
                                f"quantity={order_dict['quantity']}, "
                                f"理由=対応するBUY注文（index={corresponding_buy_index}）がREJECTEDされているため"
                            )
                            continue  # SELL注文をスキップ（execution_resultsに追加しない）
                        
                        # Phase 4.2-32-2: SELL注文スキップされなかったログ
                        self.logger.info(
                            f"[PHASE_4_2_32_2] SELL注文スキップされず、処理継続: order_index={order_index}"
                        )
                    
                    # Phase 4.2-22: 注文実行前に保存された価格をPaperBrokerに再登録
                    # 理由: _generate_trade_orders()ループ終了後に最終価格（4968.46円）で上書きされるため
                    # copilot-instructions.md: 実データのみ使用（各注文の正しい価格を再登録）
                    execution_price = order_dict.get('execution_price')
                    if execution_price and execution_price > 0 and self.paper_broker:
                        self.paper_broker.update_price(symbol, execution_price)
                        self.logger.info(f"[PHASE_4_2_22] 注文実行用価格再登録: {symbol} = {execution_price:.2f}円 (action={order_dict['action']})")
                    
                    # Phase 4.2-29-2: SELL注文の場合は実際の保有数量と照合・調整
                    # BUG FIX: _generate_trade_orders()で生成された注文数量は「予定数量」であり、
                    #          実際の実行時には前のSELL注文で減少している可能性がある
                    # copilot-instructions.md準拠: 実データのみ使用、モック注文禁止
                    if order_dict['action'] == 'SELL' and self.paper_broker:
                        broker_positions = self.paper_broker.get_positions()
                        actual_quantity = 0
                        
                        if symbol in broker_positions:
                            position_data = broker_positions[symbol]
                            actual_quantity = position_data.get('quantity', 0)
                        
                        requested_quantity = order_dict['quantity']
                        
                        self.logger.info(
                            f"[PHASE_4_2_29] SELL注文数量チェック: {symbol}, "
                            f"予定数量={requested_quantity}株, 実際の保有={actual_quantity}株"
                        )
                        
                        # Phase 4.2-29-2: 実際の保有数量に調整
                        if actual_quantity <= 0:
                            self.logger.warning(
                                f"[PHASE_4_2_29] SELL注文スキップ（実行時）: {symbol}, "
                                f"理由=ポジション未保有（実際の保有=0株）"
                            )
                            # Phase 4.2-32-2: スキップ詳細ログ
                            self.logger.info(
                                f"[PHASE_4_2_32_2] SELL注文がPhase 4.2-29でスキップされました: "
                                f"order_index={order_index}, actual_quantity=0"
                            )
                            continue  # 注文スキップ
                        
                        if requested_quantity > actual_quantity:
                            self.logger.warning(
                                f"[PHASE_4_2_29] SELL注文数量調整: {symbol}, "
                                f"{requested_quantity}株 -> {actual_quantity}株（実際の保有に合わせる）"
                            )
                            order_dict['quantity'] = actual_quantity  # 数量を実際の保有に調整
                        
                        # Phase 4.2-32-2: Phase 4.2-29通過ログ
                        self.logger.info(
                            f"[PHASE_4_2_32_2] Phase 4.2-29通過: order_index={order_index}, "
                            f"最終数量={order_dict['quantity']}株"
                        )
                    
                    # 辞書からOrderオブジェクト生成
                    from src.execution.order_manager import Order, OrderType, OrderSide
                    
                    # Phase 4.2-32-2: Order生成前ログ
                    self.logger.info(
                        f"[PHASE_4_2_32_2] Order生成準備: order_index={order_index}, "
                        f"action={order_dict['action']}, quantity={order_dict['quantity']}"
                    )
                    
                    # OrderSide決定
                    side = OrderSide.BUY if order_dict['action'] == 'BUY' else OrderSide.SELL
                    
                    # Order生成
                    order = Order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=order_dict['quantity']
                    )
                    
                    # Phase 4.2-32-2: Order生成完了ログ
                    self.logger.info(
                        f"[PHASE_4_2_32_2] Order生成完了: order_index={order_index}, "
                        f"order.id={order.id}, order.side={order.side.value}"
                    )
                    
                    # [Phase 5-B-4] 取引実行前のリスクチェック
                    # 診断ログ
                    has_attr = hasattr(self, 'integrated_manager')
                    is_not_none = self.integrated_manager if has_attr else None
                    self.logger.info(
                        f"[PHASE_5_B_4_DEBUG] Risk check condition: "
                        f"hasattr={has_attr}, integrated_manager={is_not_none is not None}"
                    )
                    
                    if hasattr(self, 'integrated_manager') and self.integrated_manager:
                        self.logger.info(f"[PHASE_5_B_4_DEBUG] Entering risk check block for order {order_index}")
                        
                        order_dict_for_check = {
                            'symbol': symbol,
                            'action': order_dict['action'],
                            'quantity': order_dict['quantity']
                        }
                        
                        self.logger.info(
                            f"[PHASE_5_B_4_DEBUG] Calling check_trade_risk: "
                            f"symbol={symbol}, action={order_dict['action']}, quantity={order_dict['quantity']}"
                        )
                        
                        can_execute = self.integrated_manager.check_trade_risk(order_dict_for_check)
                        
                        self.logger.info(f"[PHASE_5_B_4_DEBUG] check_trade_risk returned: {can_execute}")
                        
                        if not can_execute:
                            self.logger.warning(
                                f"[RISK_BLOCKED] Trade execution blocked by risk check: {symbol} "
                                f"{order_dict['action']} {order_dict['quantity']}"
                            )
                            # BUY注文がブロックされた場合もトラッキング
                            if order_dict['action'] == 'BUY':
                                rejected_buy_indices.add(order_index)
                            continue  # 取引をスキップ
                    
                    # TradeExecutor.submit_order()呼び出し
                    self.logger.info(
                        f"[PHASE_4_2_32_2] TradeExecutor.submit_order()呼び出し: order_index={order_index}"
                    )
                    order_id = self.trade_executor.submit_order(order)
                    self.logger.info(
                        f"[PHASE_4_2_32_2] TradeExecutor.submit_order()完了: order_index={order_index}, "
                        f"order_id={order_id}, order.status={order.status.value}"
                    )
                    
                    # Phase 4.2-32: order_idだけでなくorder.statusもチェック（order_managerは常にorder.idを返すため）
                    # copilot-instructions.md準拠: 実データのみ（REJECTEDされた注文はexecution_resultsに追加しない）
                    if order_id and order.status == OrderStatus.FILLED:
                        execution_results.append({
                            "success": True,
                            "status": "executed",  # Phase 4.2-5-3: ステータス追加
                            "order_id": order_id,
                            "order": order,  # Phase 4.2-5-3: Orderオブジェクト追加
                            "symbol": order_dict['symbol'],
                            "action": order_dict['action'],
                            "quantity": order_dict['quantity'],
                            "timestamp": order_dict['timestamp'],
                            "executed_price": order.filled_price,  # Phase 4.2-5-3: 約定価格追加
                            "strategy_name": order_dict.get('strategy_name', 'UnknownStrategy'),  # Phase 5.3: 戦略名追加
                            "execution_type": "trade"  # Phase 2025-12-15: execution_typeフィールド追加（通常取引）
                        })
                        
                        strategy_name = order_dict.get('strategy_name', 'UnknownStrategy')
                        if strategy_name == 'UnknownStrategy':
                            self.logger.warning(
                                f"[FALLBACK] 戦略名が取得できませんでした（execution_details保存）: order_dict={order_dict.keys()}, "
                                f"デフォルト値='UnknownStrategy'"
                            )
                        self.logger.info(f"Trade executed successfully: {order_dict['symbol']} {order_dict['action']} {order_dict['quantity']} strategy={strategy_name}")
                        
                        # [Phase 5-B-5] 取引実行後のスナップショット記録（Q1: C案 - 取引時）
                        if self.equity_recorder and self.paper_broker:
                            try:
                                portfolio_value = self.paper_broker.get_total_equity()
                                cash_balance = self.paper_broker.get_account_balance()
                                position_value = self.paper_broker.get_position_value()
                                
                                # DrawdownControllerからリスク情報取得（Q2: 案1）
                                peak_value = portfolio_value  # デフォルト値
                                drawdown_pct = 0.0
                                risk_status = "NORMAL"
                                
                                if self.integrated_manager and hasattr(self.integrated_manager, 'risk_controller'):
                                    try:
                                        perf_tracker = self.integrated_manager.risk_controller.performance_tracker
                                        peak_value = perf_tracker.get('portfolio_peak', portfolio_value)
                                        
                                        # ドローダウン計算
                                        if peak_value > 0:
                                            drawdown_pct = (peak_value - portfolio_value) / peak_value
                                        
                                        # リスク状態取得
                                        risk_status = self.integrated_manager.risk_controller.get_current_severity()
                                    except Exception as e:
                                        self.logger.warning(f"[EQUITY_CURVE] Risk info retrieval failed: {e}")
                                
                                # 累積損益計算
                                initial_balance = self.paper_broker.initial_balance
                                cumulative_pnl = portfolio_value - initial_balance
                                
                                # 当日損益（前回スナップショットとの差分、簡易版）
                                daily_pnl = 0.0
                                if self.equity_recorder.get_snapshot_count() > 0:
                                    last_snapshot = self.equity_recorder.get_latest_snapshot()
                                    if last_snapshot:
                                        daily_pnl = portfolio_value - last_snapshot['portfolio_value']
                                
                                # スナップショット記録
                                # タイムスタンプ変換（文字列→datetime）
                                from datetime import datetime as dt
                                timestamp_dt = dt.fromisoformat(order_dict['timestamp']) if isinstance(order_dict['timestamp'], str) else order_dict['timestamp']
                                
                                self.equity_recorder.record_snapshot(
                                    date=timestamp_dt,
                                    portfolio_value=portfolio_value,
                                    cash_balance=cash_balance,
                                    position_value=position_value,
                                    peak_value=peak_value,
                                    drawdown_pct=drawdown_pct,
                                    cumulative_pnl=cumulative_pnl,
                                    daily_pnl=daily_pnl,
                                    total_trades=len(execution_results),
                                    active_positions=len(self.paper_broker.get_positions()),
                                    risk_status=risk_status,
                                    blocked_trades=0,  # 取引成功時は0
                                    risk_action="NONE",
                                    snapshot_type="TRADE"
                                )
                            except Exception as e:
                                self.logger.error(f"[EQUITY_CURVE] Snapshot recording failed: {e}")
                        
                    else:
                        # Phase 4.2-31-B-4-4: BUY注文がREJECTEDされた場合、order_indexをトラッキング
                        if order_dict['action'] == 'BUY':
                            rejected_buy_indices.add(order_index)
                            self.logger.warning(
                                f"[PHASE_4_2_31_B] BUY注文REJECTED（トラッキング追加）: {symbol}, "
                                f"order_index={order_index}, quantity={order_dict['quantity']}, "
                                f"rejected_buy_indices={rejected_buy_indices}"
                            )
                        
                        # Phase 4.2-27: 失敗原因の詳細ログ
                        self.logger.warning(
                            f"[PHASE_4_2_27] Trade execution failed: {order_dict['symbol']} {order_dict['action']} "
                            f"quantity={order_dict['quantity']}, execution_price={execution_price}, "
                            f"order_id={order_id} (None=失敗)"
                        )
                        
                        # 現在のポジション確認（SELL失敗の場合）
                        if order_dict['action'] == 'SELL' and self.paper_broker:
                            current_positions = self.paper_broker.get_positions()
                            has_position = symbol in current_positions and current_positions[symbol]['quantity'] > 0
                            self.logger.warning(
                                f"[PHASE_4_2_27] SELL失敗時のポジション確認: {symbol}, "
                                f"has_position={has_position}, "
                                f"positions={current_positions.get(symbol, 'NO_POSITION')}"
                            )
                        
                        execution_results.append({
                            "success": False,
                            "status": "failed",
                            "error": "Order submission failed",
                            "order": order_dict
                        })
                        self.logger.warning(f"Trade execution failed: {order_dict['symbol']}")
                        
                except Exception as e:
                    self.logger.error(f"Trade execution error: {e}", exc_info=True)
                    execution_results.append({"success": False, "status": "error", "error": str(e), "order": order_dict})
            
            # Phase 4.2-5-3: 実行結果をbacktest_signalsに統合
            # copilot-instructions.md: 実際の取引件数 > 0 を検証
            if hasattr(self, 'current_backtest_signals') and self.current_backtest_signals is not None:
                signals_df = self.current_backtest_signals
                
                # 実行された取引をSignalsDataFrameに追加するための新しいカラム
                if 'ExecutedTrade' not in signals_df.columns:
                    signals_df['ExecutedTrade'] = False
                if 'ExecutedPrice' not in signals_df.columns:
                    signals_df['ExecutedPrice'] = 0.0
                if 'ExecutedQuantity' not in signals_df.columns:
                    signals_df['ExecutedQuantity'] = 0.0
                    
                # 実行された取引を記録
                for result in execution_results:
                    if result.get('status') == 'executed':
                        try:
                            timestamp = result.get('timestamp')
                            symbol = result['symbol']
                            action = result['action']
                            quantity = result['quantity']
                            exec_price = result.get('executed_price', 0.0)
                            
                            # SignalsDataFrameの最終行に実行情報を追加
                            if len(signals_df) > 0:
                                last_idx = signals_df.index[-1]
                                signals_df.at[last_idx, 'ExecutedTrade'] = True
                                signals_df.at[last_idx, 'ExecutedPrice'] = exec_price
                                signals_df.at[last_idx, 'ExecutedQuantity'] = quantity
                                
                            # Phase 4.2-5-3: 実行された取引をbacktest_signalsに記録
                            symbol = result['symbol']
                            action = result['action']
                            quantity = result['quantity']
                            exec_price = result.get('executed_price', 0)
                            
                            self.logger.info(f"[OK] Trade integrated into backtest_signals: {symbol} {action} {quantity} @ {exec_price}")
                                    
                        except Exception as e:
                            self.logger.error(f"Trade integration error: {e}", exc_info=True)
                            
                # 更新されたsignals_dfを保存
                self.current_backtest_signals = signals_df
            else:
                self.logger.warning(f"No current_backtest_signals available for trade integration")
            
            # Phase 4.2-23: 未決済ポジション強制決済（Option A: PaperBroker直接呼び出し）
            # copilot-instructions.md準拠: バックテスト終了時に未決済ポジションがあれば強制決済
            # 理由: TradeExecutorのリスク管理チェックをバイパスする必要があるため
            if self.paper_broker:
                open_positions = self.paper_broker.get_positions()
                if open_positions:
                    # [修正案2] ForceClose開始フラグを設定（2025-12-08追加）
                    self.force_close_in_progress = True
                    self.logger.warning(f"[PHASE_4_2_23] 未決済ポジション検出: {len(open_positions)}件")
                    self.logger.info("[FORCE_CLOSE_START] ForceClose開始、通常SELL処理を抑制")
                    
                    for symbol, position in open_positions.items():
                        try:
                            quantity = position['quantity']
                            entry_price = position.get('entry_price', 0)
                            
                            # 最終価格を取得
                            final_price = self.paper_broker.get_current_price(symbol)
                            
                            # Option A実装: PaperBrokerを直接呼び出し（リスク管理バイパス）
                            from src.execution.order_manager import Order, OrderType, OrderSide
                            
                            force_close_order = Order(
                                symbol=symbol,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=quantity
                            )
                            
                            # PaperBroker.submit_order()を直接呼び出し
                            # TradeExecutorをバイパスするため、リスク管理チェックは実行されない
                            success = self.paper_broker.submit_order(force_close_order)
                            
                            if success:
                                # 約定価格を取得（PaperBrokerが設定）
                                executed_price = force_close_order.filled_price or final_price
                                
                                profit_pct = 0
                                if entry_price > 0:
                                    profit_pct = (executed_price - entry_price) / entry_price * 100
                                
                                # Phase 4.2-23: PortfolioTrackerへの手動記録
                                # Option A実装の重要ポイント: TradeExecutorをバイパスするため、
                                # PortfolioTrackerへの記録は手動で行う必要がある
                                if hasattr(self, 'portfolio_tracker') and self.portfolio_tracker:
                                    try:
                                        self.portfolio_tracker.execute_trade(
                                            symbol=symbol,
                                            quantity=-abs(quantity),  # SELLなので負の値
                                            price=executed_price,
                                            commission=force_close_order.commission or 0,
                                            slippage=force_close_order.slippage or 0,
                                            strategy_name="ForceClose",
                                            trade_id=force_close_order.id
                                        )
                                        self.logger.debug(f"[FORCE_CLOSE] PortfolioTrackerに記録完了: {symbol}")
                                    except Exception as pt_error:
                                        self.logger.warning(f"[FORCE_CLOSE] PortfolioTracker記録エラー ({symbol}): {pt_error}")
                                
                                # Phase 4.2-23修正: timestampをバックテスト期間の最終日に変更
                                # 根拠: 通常のBUY/SELL注文（Line 940, 997）との一貫性
                                # 修正前: pd.Timestamp.now()（実行時刻）
                                # 修正後: signals.index[-1]（バックテスト最終日）
                                backtest_end_timestamp = signals.index[-1].isoformat() if hasattr(signals.index[-1], 'isoformat') else str(signals.index[-1])
                                
                                execution_results.append({
                                    "success": True,
                                    "status": "force_closed",  # 強制決済フラグ
                                    "order_id": force_close_order.id,
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "quantity": quantity,
                                    "timestamp": backtest_end_timestamp,
                                    "executed_price": executed_price,
                                    "strategy_name": "ForceClose",
                                    "profit_pct": profit_pct,
                                    "execution_type": "force_close"  # Phase 2025-12-15: execution_typeフィールド追加（強制決済）
                                })
                                
                                self.logger.info(
                                    f"[FORCE_CLOSE] {symbol} 強制決済完了: "
                                    f"数量={quantity}株, エントリー={entry_price:.2f}円, "
                                    f"決済={executed_price:.2f}円, 損益={profit_pct:.2f}%"
                                )
                            else:
                                self.logger.error(f"[PHASE_4_2_23] 強制決済失敗 ({symbol}): PaperBroker.submit_order() returned False")
                                
                        except Exception as e:
                            self.logger.error(f"[PHASE_4_2_23] 強制決済エラー ({symbol}): {e}", exc_info=True)
                
                # [修正案2] ForceClose終了フラグをリセット（2025-12-08追加）
                self.force_close_in_progress = False
                self.logger.info("[FORCE_CLOSE_END] ForceClose完了、通常SELL処理を再開")
            
            self.logger.info(f"Trade execution completed: {len(execution_results)} orders processed")
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Trade execution processing error: {e}", exc_info=True)
            return []
    
    def _generate_trade_orders(self, signals: pd.DataFrame, symbols: List[str], strategy_name: str = 'Unknown') -> List[Dict[str, Any]]:
        """
        シグナルから取引注文生成（Phase 4.2-9: 全シグナル履歴対応版 + Phase 5.3: strategy_name追加）
        
        Args:
            signals: バックテスト結果DataFrame (Entry_Signal, Exit_Signal, Position含む)
            symbols: ティッカーシンボルリスト
            strategy_name: 戦略名（Phase 5.3追加: CSV出力のstrategy列に使用）
        
        Returns:
            取引オーダーリスト
        
        copilot-instructions.md準拠:
        - 実際のシグナルデータのみ使用（モック禁止）
        - 全シグナル履歴を確認（最新だけでなく）
        - 詳細ログ出力（デバッグ用）
        """
        try:
            orders = []
            
            # DataFrame検証
            if signals.empty:
                self.logger.warning("_generate_trade_orders: signals DataFrame is empty")
                return orders
            
            # デバッグログ: signals DataFrame の構造確認
            self.logger.info(f"_generate_trade_orders: signals shape={signals.shape}, columns={list(signals.columns)}")
            
            # 必須カラムの存在確認
            if 'Entry_Signal' not in signals.columns or 'Exit_Signal' not in signals.columns:
                self.logger.error(f"_generate_trade_orders: Required columns missing. Available: {list(signals.columns)}")
                return orders
            
            # シグナル件数カウント
            entry_count = (signals['Entry_Signal'] == 1).sum()
            exit_count = (signals['Exit_Signal'] == -1).sum()  # Exit_Signal は -1 が正しい
            self.logger.info(f"_generate_trade_orders: Entry_Signal==1: {entry_count} 件, Exit_Signal==-1: {exit_count} 件")
            
            # [DEBUG_E2] Entry_Signal=1の日付と詳細を出力（調査用）
            if entry_count > 0:
                entry_dates = signals[signals['Entry_Signal'] == 1].index.tolist()
                self.logger.info(f"[DEBUG_E2] Entry_Signal=1の日付: {entry_dates}")
                for entry_date in entry_dates:
                    entry_row = signals.loc[entry_date]
                    self.logger.info(
                        f"[DEBUG_E2] Entry_Signal詳細: date={entry_date}, "
                        f"Close={entry_row.get('Close', 'N/A')}, "
                        f"Entry_Price={entry_row.get('Entry_Price', 'N/A')}, "
                        f"Position={entry_row.get('Position', 'N/A')}"
                    )
            
            # [DEBUG_E3] signals DataFrameの日付範囲と件数を出力
            if len(signals) > 0:
                self.logger.info(
                    f"[DEBUG_E3] signals DataFrame範囲: {signals.index[0]} ~ {signals.index[-1]}, "
                    f"総件数={len(signals)}行, strategy_name={strategy_name}"
                )
            
            # Phase 4.2-9-2: ポジション追跡用辞書（銘柄ごとに最新のBUY数量を記録）
            # 理由: _get_current_position() はオーダー生成時点ではまだ0を返すため
            # SELLオーダーは直前のBUYオーダーと同じ数量を使用する必要がある
            position_tracker: dict[str, int] = {}
            
            # Phase 4.2-26: 注文生成カウンター（デバッグ用）
            buy_order_count = 0
            sell_order_count = 0
            
            # Phase 4.2-31-B-4-1: 注文インデックスカウンター（BUY/SELLペアトラッキング用）
            # copilot-instructions.md準拠: 実データのみ、推測なし
            order_index = 0
            
            # 全シグナル履歴をスキャン（最新だけでなく全期間）
            for idx, row in signals.iterrows():
                for symbol in symbols:
                    # エントリーシグナル検出
                    if row.get('Entry_Signal', 0) == 1:
                        # Phase 4.2-25: 既存ポジションチェック（重複BUY防止）
                        # copilot-instructions.md準拠: 実データのみ使用、モック注文禁止
                        if symbol in position_tracker and position_tracker[symbol] > 0:
                            self.logger.warning(
                                f"[PHASE_4_2_25] BUY注文スキップ: {symbol} @ {idx} "
                                f"(理由=既存ポジション保有中, 数量={position_tracker[symbol]}株)"
                            )
                            continue  # 既にポジションを持っている場合はスキップ
                        
                        self.logger.info(f"[PHASE_4_2_25] Entry_Signal==1検出: {symbol} @ {idx}, position_tracker={position_tracker.get(symbol, 0)}")
                        
                        # Phase 4.2-21 DEBUG: 価格カラム優先順位とデータ内容確認
                        entry_price_col = row.get('Entry_Price', None)
                        close_col = row.get('Close', None)
                        adj_close_col = row.get('Adj Close', None)
                        self.logger.info(f"[PRICE_SOURCE] Row {idx}: Entry_Price={entry_price_col} | Close={close_col} | Adj Close={adj_close_col}")
                        
                        entry_price = row.get('Entry_Price', row.get('Close', 0))
                        
                        # Phase 4.2-20: 価格登録を_calculate_position_size()より前に実行
                        # BUG FIX: PaperBrokerに確実に価格登録（デフォルト100円使用を防ぐ）
                        if entry_price and entry_price > 0:
                            if self.paper_broker:
                                self.paper_broker.update_price(symbol, entry_price)
                                self.logger.info(f"[PRICE_REGISTER] PaperBrokerへ登録: {symbol} = {entry_price:.2f}円 (Row {idx})")
                        
                        buy_quantity = self._calculate_position_size(symbol)
                        
                        # Phase 4.2-24: ゼロ数量チェック（資金不足・価格取得失敗時のスキップ）
                        # copilot-instructions.md準拠: 実データのみ使用、モック注文禁止
                        if buy_quantity <= 0:
                            self.logger.warning(
                                f"[PHASE_4_2_24] BUY注文スキップ: {symbol} @ {idx} "
                                f"(数量={buy_quantity}株, 理由=資金不足または価格取得失敗)"
                            )
                            continue  # 注文をスキップ（execution_detailsに追加しない）
                        
                        # ポジション追跡: BUY実行時に数量を記録
                        position_tracker[symbol] = buy_quantity
                        
                        # Phase 4.2-22: execution_price追加（注文実行時に使用）
                        # copilot-instructions.md: 実データ保持（ループ終了後の最終価格上書きを防ぐ）
                        orders.append({
                            "symbol": symbol,
                            "action": "BUY",
                            "quantity": buy_quantity,
                            "order_type": "MARKET",
                            "timestamp": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                            "entry_price": entry_price,
                            "execution_price": entry_price,  # Phase 4.2-22: 注文実行用価格
                            "signal_date": idx,
                            "strategy_name": strategy_name,  # Phase 5.3: 戦略名追加
                            "order_index": order_index  # Phase 4.2-31-B-4-1: 注文インデックス追加
                        })
                        buy_order_count += 1  # Phase 4.2-26: BUY注文カウント
                        order_index += 1  # Phase 4.2-31-B-4-1: インデックスインクリメント
                        # Phase 4.2-32: order_index詳細ログ
                        self.logger.info(f"[PHASE_4_2_32] BUY order generated: {symbol}, order_index={order_index-1}, quantity={buy_quantity}, price={entry_price}")
                        self.logger.debug(f"BUY order generated: {symbol} @ {idx}, quantity={buy_quantity}, price={entry_price}, strategy={strategy_name}, order_index={order_index-1}")
                    
                    # イグジットシグナル検出（Exit_Signal == -1 が正しい）
                    if row.get('Exit_Signal', 0) == -1:
                        self.logger.info(f"[PHASE_4_2_29] Exit_Signal==-1検出: {symbol} @ {idx}, position_tracker={position_tracker.get(symbol, 0)}")
                        
                        # Phase 4.2-29-2: position_trackerでポジション保有チェック
                        # 理由: _generate_trade_orders()は注文「生成」のみ担当
                        #       実際の実行後の数量調整は_execute_trades()で行う
                        # copilot-instructions.md準拠: 実データのみ使用、モック注文禁止
                        if symbol not in position_tracker or position_tracker[symbol] <= 0:
                            self.logger.warning(
                                f"[PHASE_4_2_29] SELL注文スキップ: {symbol} @ {idx} "
                                f"(理由=ポジション未保有, tracker={position_tracker.get(symbol, 'NOT_IN_DICT')})"
                            )
                            continue  # ポジションを持っていない場合はスキップ
                        
                        exit_price = row.get('Close', 0)
                        
                        # Phase 4.2-20: 価格登録（SELL注文用）
                        # BUG FIX: PaperBrokerに確実に価格登録（デフォルト100円使用を防ぐ）
                        if exit_price and exit_price > 0 and self.paper_broker:
                            self.paper_broker.update_price(symbol, exit_price)
                            self.logger.debug(f"[SELL] Price registered to PaperBroker: {symbol} = {exit_price}円")
                        
                        # Phase 4.2-29-2: position_trackerから予定数量を取得
                        # 注意: これは「注文生成時の予定数量」であり、
                        #       実際の実行時には_execute_trades()で実際の保有数量に調整される
                        sell_quantity = position_tracker.get(symbol, 0)
                        
                        # Phase 4.2-29-2: ゼロ数量チェック（念のため）
                        # copilot-instructions.md準拠: 実データのみ使用、モック注文禁止
                        if sell_quantity <= 0:
                            self.logger.warning(
                                f"[PHASE_4_2_29] SELL注文スキップ: {symbol} @ {idx} "
                                f"(数量={sell_quantity}株, 理由=position_tracker数量が0)"
                            )
                            continue  # 注文をスキップ（execution_detailsに追加しない）
                        
                        # Phase 4.2-22: execution_price追加（注文実行時に使用）
                        # copilot-instructions.md: 実データ保持（ループ終了後の最終価格上書きを防ぐ）
                        orders.append({
                            "symbol": symbol,
                            "action": "SELL",
                            "quantity": sell_quantity,
                            "order_type": "MARKET",
                            "timestamp": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                            "exit_price": exit_price,
                            "execution_price": exit_price,  # Phase 4.2-22: 注文実行用価格
                            "signal_date": idx,
                            "strategy_name": strategy_name,  # Phase 5.3: 戦略名追加
                            "order_index": order_index  # Phase 4.2-31-B-4-1: 注文インデックス追加
                        })
                        sell_order_count += 1  # Phase 4.2-26: SELL注文カウント
                        order_index += 1  # Phase 4.2-31-B-4-1: インデックスインクリメント
                        # Phase 4.2-32: order_index詳細ログ
                        self.logger.info(f"[PHASE_4_2_32] SELL order generated: {symbol}, order_index={order_index-1}, quantity={sell_quantity}, price={exit_price}")
                        self.logger.debug(f"[PHASE_4_2_29] SELL order generated: {symbol} @ {idx}, quantity={sell_quantity}, price={exit_price}, strategy={strategy_name}, order_index={order_index-1}")
                        
                        # Phase 4.2-29: ポジションクリア（SELL後はposition_trackerも0にする）
                        # 注意: これは次のBUY注文の重複防止用のみ
                        position_tracker[symbol] = 0
            
            # 最終ログ: 生成された取引オーダー数
            self.logger.info(f"_generate_trade_orders: Generated {len(orders)} trade orders from {len(signals)} signals")
            self.logger.info(f"[PHASE_4_2_26] Order generation summary: BUY={buy_order_count}, SELL={sell_order_count}, Total={buy_order_count + sell_order_count}")
            
            # copilot-instructions.md: 実際の取引件数 > 0 を検証
            if len(orders) == 0 and (entry_count > 0 or exit_count > 0):
                self.logger.warning(f"[WARNING] Signal detected but no orders generated! Entry={entry_count}, Exit={exit_count}")
            
            return orders
            
        except Exception as e:
            self.logger.error(f"取引注文生成エラー: {e}", exc_info=True)
            return []
    
    def _calculate_position_size(self, symbol: str) -> int:
        """
        ポジションサイズ計算（Phase 4.2-16: 日本株対応版）
        
        Phase 4.2-16の変更点:
        - 90%資金使用（余剰資金運用のため）
        - 三菱UFJ eスマート証券手数料体系対応
        - 100株単位（単元株制度）対応
        - 手数料・スリッページ込みで最大購入可能株数を計算
        
        copilot-instructions.md準拠:
        - デフォルト価格使用禁止（モック/ダミーデータ禁止）
        - 実際の利用可能資金とリアルタイム価格を使用
        - フォールバック禁止（資金不足時は0を返す）
        
        Args:
            symbol: ティッカーシンボル
        
        Returns:
            購入株数（整数、100株単位）
        """
        try:
            # 1. 総資産を取得（実データ）
            if not self.paper_broker:
                self.logger.error(f"[ERROR] PaperBrokerが初期化されていません: {symbol}")
                raise ValueError("PaperBroker not initialized")
            
            # Phase 4.2-16: total_equity = 現金 + ポジション評価額
            available_cash = self.paper_broker.get_account_balance()
            
            # ポジション評価額を取得
            position_value = 0.0
            if hasattr(self.paper_broker, 'get_positions'):
                positions = self.paper_broker.get_positions()
                for pos_symbol, pos_data in positions.items():
                    if hasattr(pos_data, 'quantity') and hasattr(pos_data, 'current_price'):
                        position_value += pos_data.quantity * pos_data.current_price
            
            total_equity = available_cash + position_value
            
            # Phase 4.2-16: 90%を使用（余剰資金運用）
            # Task 8検証完了により本番設定に復元（2025-12-08）
            available_funds = total_equity * 0.90
            
            self.logger.debug(
                f"資金状況: {symbol} - 総資産={total_equity:,.0f}円, "
                f"現金={available_cash:,.0f}円, ポジション={position_value:,.0f}円, "
                f"使用可能額={available_funds:,.0f}円 (90%)"
            )
            
            # 2. リアルタイム価格を取得（実データ）
            # Phase 4.2-20: 価格取得優先順位変更（PaperBrokerを優先）
            current_price = None
            
            # 2-1. PaperBrokerから価格取得を試行（優先）
            # 理由: _generate_trade_orders()でupdate_price()済みのため
            if self.paper_broker and hasattr(self.paper_broker, 'get_current_price'):
                try:
                    current_price = self.paper_broker.get_current_price(symbol)
                    if current_price and current_price > 0:
                        # Phase 4.2-21 DEBUG: 価格取得詳細ログ
                        self.logger.info(f"[CALC_POS_SIZE] PaperBroker価格取得: {symbol} = {current_price:.2f}円")
                except Exception as e:
                    self.logger.warning(f"[WARNING] PaperBrokerからの価格取得失敗: {symbol} - {e}")
            
            # 2-2. data_feedから価格取得を試行（フォールバック）
            if (current_price is None or current_price <= 0) and self.data_feed:
                try:
                    current_price = self.data_feed.get_current_price(symbol)
                    if current_price and current_price > 0:
                        self.logger.debug(f"[OK] DataFeedから価格取得: {symbol} = {current_price:.2f}円")
                except Exception as e:
                    self.logger.warning(f"[WARNING] DataFeedからの価格取得失敗: {symbol} - {e}")
            
            # 2-3. 価格取得失敗時はエラー（デフォルト価格使用禁止）
            if current_price is None or current_price <= 0:
                error_msg = f"[ERROR] 実際の価格取得失敗: {symbol} (copilot-instructions.md: デフォルト価格使用禁止)"
                self.logger.error(error_msg)
                return 0  # Phase 4.2-16: エラー時は0を返す（取引しない）
            
            # 3. Phase 4.2-16: 手数料込みで最大購入可能株数を計算
            try:
                quantity, contract_value, commission, total_cost = calculate_max_affordable_quantity(
                    available_funds=available_funds,
                    stock_price=current_price,
                    unit_size=100,  # 日本株は100株単位
                    include_slippage=True,
                    slippage_rate=0.0001  # 0.01% (引数名修正)
                )
                
                self.logger.debug(
                    f"手数料計算結果: {symbol} - "
                    f"株数={quantity}株, 約定代金={contract_value:,.0f}円, "
                    f"手数料={commission:.0f}円, 総コスト={total_cost:,.0f}円"
                )
                
            except Exception as e:
                self.logger.error(f"[ERROR] calculate_max_affordable_quantity失敗: {symbol} - {e}")
                return 0
            
            # 4. Phase 4.2-16: 100株単位に調整（念のため）
            final_quantity = adjust_to_trading_unit(quantity, unit_size=100)
            
            # 5. 最小株数チェック
            if final_quantity < 100:
                self.logger.warning(
                    f"[WARNING] 資金不足で100株未満: {symbol} (計算株数={final_quantity}株, "
                    f"必要資金={current_price * 100:,.0f}円, 利用可能額={available_funds:,.0f}円)"
                )
                return 0  # Phase 4.2-16: 100株未満なら取引しない（copilot-instructions.md準拠）
            
            # 6. 成功ログ出力
            remaining_funds = available_funds - total_cost
            self.logger.info(
                f"[OK] {symbol} ポジションサイズ: {final_quantity}株 @ {current_price:.2f}円 "
                f"(約定代金: {contract_value:,.0f}円, 手数料: {commission:.0f}円, "
                f"総コスト: {total_cost:,.0f}円, 残金: {remaining_funds:,.0f}円)"
            )
            
            return final_quantity
                
        except Exception as e:
            error_msg = f"[ERROR] ポジションサイズ計算エラー: {symbol} - {e}"
            self.logger.error(error_msg)
            # Phase 4.2-16: エラー時は0を返す（copilot-instructions.md: フォールバック禁止）
            return 0
    
    def _get_current_position(self, symbol: str) -> int:
        """現在のポジション取得"""
        try:
            if self.paper_broker:
                return self.paper_broker.get_position(symbol)
            return 0
        except Exception:
            return 0
    
    def _execute_single_strategy_integrated(self, strategy_name: str, market_data: pd.DataFrame, symbols: List[str]) -> Dict[str, Any]:
        """統合モード用単一戦略実行"""
        try:
            strategy = self._get_strategy_instance(strategy_name)
            if strategy is None:
                return {"error": f"strategy_not_found: {strategy_name}"}
            
            # 戦略実行（既存戦略はデータ引数なしで実行）
            signals = strategy.backtest()
            trades = self._execute_trades(signals, symbols)
            
            return {
                "strategy": strategy_name,
                "signals": len(signals) if signals is not None else 0,
                "trades": trades,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "strategy": strategy_name}
    
    def _calculate_portfolio_weights(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        """ポートフォリオ重み計算"""
        # 簡易実装：等重み
        successful_strategies = [name for name, result in strategy_results.items() 
                               if result.get("success", False)]
        
        if successful_strategies:
            weight = 1.0 / len(successful_strategies)
            return {name: weight for name in successful_strategies}
        
        return {}
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果生成"""
        result = {
            "success": False,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        self.execution_history.append(result)
        return result
    
    def cleanup(self) -> None:
        """終了処理"""
        try:
            if self.data_feed:
                if hasattr(self.data_feed, 'close'):
                    self.data_feed.close()
            
            if self.paper_broker:
                if hasattr(self.paper_broker, 'close'):
                    self.paper_broker.close()
            
            self.logger.info("戦略実行管理システム終了処理完了")
            
        except Exception as e:
            self.logger.error(f"終了処理エラー: {e}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """実行サマリー取得"""
        total_executions = len(self.execution_history)
        successful_executions = len([r for r in self.execution_history if r.get("success", False)])
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "last_execution": self.execution_history[-1] if self.execution_history else None
        }
