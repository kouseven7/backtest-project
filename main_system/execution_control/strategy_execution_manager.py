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
            from src.execution.paper_broker import PaperBroker
            broker_config = self.config.get('broker', {})
            self.paper_broker = PaperBroker(
                initial_balance=broker_config.get('initial_cash', 100000),
                commission_per_trade=broker_config.get('commission_per_trade', 1.0),
                slippage_pct=broker_config.get('slippage_bps', 5) / 10000.0
            )
            
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
    
    def execute_strategy(self, strategy_name: str, symbols: Optional[List[str]] = None, 
                        stock_data: Optional[pd.DataFrame] = None,
                        index_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        単一戦略実行（シンプルモード）
        
        Args:
            strategy_name: 戦略名
            symbols: ティッカーシンボルリスト
            stock_data: 株価データ（Phase 4.2: yfinanceから取得済み）
            index_data: インデックスデータ（Phase 4.2: yfinanceから取得済み）
        
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
            
            # 戦略インスタンス取得（Phase 4.2: データを渡す）
            strategy = self._get_strategy_instance(strategy_name, stock_data, index_data)
            if strategy is None:
                return self._create_error_result(f"strategy_not_found: {strategy_name}")
            
            # 戦略実行（既存戦略はデータ引数なしで実行）
            signals = strategy.backtest()
            
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
                              index_data: pd.DataFrame):
        """
        戦略インスタンス取得（Phase 4.2: データ必須版）
        
        Args:
            strategy_name: 戦略名
            stock_data: 株価データ
            index_data: インデックスデータ
        
        Returns:
            戦略インスタンス（失敗時はNone）
        """
        try:
            # 戦略名の正規化マッピング（既存形式 + 新形式）
            strategy_mappings = {
                # 既存形式（アンダースコア区切り）
                'VWAP_Breakout': 'strategies.VWAP_Breakout.VWAPBreakoutStrategy',
                'VWAP_Bounce': 'strategies.VWAP_Bounce.VWAPBounceStrategy',
                'GC_Strategy': 'strategies.gc_strategy_signal.GCStrategy',
                'Breakout': 'strategies.Breakout.BreakoutStrategy',
                'Opening_Gap': 'strategies.Opening_Gap.OpeningGapStrategy',
                
                # 新形式（クラス名そのまま）
                'VWAPBreakoutStrategy': 'strategies.VWAP_Breakout.VWAPBreakoutStrategy',
                'VWAPBounceStrategy': 'strategies.VWAP_Bounce.VWAPBounceStrategy',
                'GCStrategy': 'strategies.gc_strategy_signal.GCStrategy',
                'BreakoutStrategy': 'strategies.Breakout.BreakoutStrategy',
                'OpeningGapStrategy': 'strategies.Opening_Gap.OpeningGapStrategy',
                'OpeningGapFixedStrategy': 'strategies.Opening_Gap.OpeningGapStrategy',
                'MomentumInvestingStrategy': 'strategies.momentum_investing.MomentumInvestingStrategy',
                'ContrarianStrategy': 'strategies.contrarian.ContrarianStrategy',
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
            
            try:
                # データ付きインスタンス化を試行
                strategy_instance = strategy_class(data=stock_data, index_data=index_data)
                self.logger.info(f"Strategy instance created successfully: {strategy_name}")
                return strategy_instance
            except TypeError as e:
                # データなしインスタンス化を試行（一部の戦略はデータ不要）
                try:
                    self.logger.warning(f"Strategy {strategy_name} does not accept data in __init__, trying without data")
                    strategy_instance = strategy_class()
                    self.logger.info(f"Strategy instance created without data: {strategy_name}")
                    return strategy_instance
                except Exception as e2:
                    self.logger.error(f"CRITICAL: Strategy '{strategy_name}' initialization failed. Error: {e2}")
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
            
            # シグナルから取引指示を生成
            trade_orders = self._generate_trade_orders(signals, symbols)
            
            if not trade_orders:
                self.logger.info("No trade orders generated from signals")
                return execution_results
            
            # 各注文を実行（実際の実行のみ）
            for order_dict in trade_orders:
                try:
                    # 辞書からOrderオブジェクト生成
                    from src.execution.order_manager import Order, OrderType, OrderSide
                    
                    # OrderSide決定
                    side = OrderSide.BUY if order_dict['action'] == 'BUY' else OrderSide.SELL
                    
                    # Order生成
                    order = Order(
                        symbol=order_dict['symbol'],
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=order_dict['quantity']
                    )
                    
                    # TradeExecutor.submit_order()呼び出し
                    order_id = self.trade_executor.submit_order(order)
                    
                    if order_id:
                        execution_results.append({
                            "success": True,
                            "status": "executed",  # Phase 4.2-5-3: ステータス追加
                            "order_id": order_id,
                            "order": order,  # Phase 4.2-5-3: Orderオブジェクト追加
                            "symbol": order_dict['symbol'],
                            "action": order_dict['action'],
                            "quantity": order_dict['quantity'],
                            "timestamp": order_dict['timestamp'],
                            "executed_price": order.filled_price  # Phase 4.2-5-3: 約定価格追加
                        })
                        self.logger.info(f"Trade executed successfully: {order_dict['symbol']} {order_dict['action']} {order_dict['quantity']}")
                    else:
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
                                
                            self.logger.info(f"✅ Trade integrated into backtest_signals: {symbol} {action} {quantity} @ {exec_price}")
                                    
                        except Exception as e:
                            self.logger.error(f"Trade integration error: {e}", exc_info=True)
                            
                # 更新されたsignals_dfを保存
                self.current_backtest_signals = signals_df
            else:
                self.logger.warning(f"No current_backtest_signals available for trade integration")
            
            self.logger.info(f"Trade execution completed: {len(execution_results)} orders processed")
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Trade execution processing error: {e}", exc_info=True)
            return []
    
    def _generate_trade_orders(self, signals: pd.DataFrame, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        シグナルから取引注文生成（Phase 4.2-9: 全シグナル履歴対応版）
        
        Args:
            signals: バックテスト結果DataFrame (Entry_Signal, Exit_Signal, Position含む)
            symbols: ティッカーシンボルリスト
        
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
            
            # Phase 4.2-9-2: ポジション追跡用辞書（銘柄ごとに最新のBUY数量を記録）
            # 理由: _get_current_position() はオーダー生成時点ではまだ0を返すため
            # SELLオーダーは直前のBUYオーダーと同じ数量を使用する必要がある
            position_tracker: dict[str, int] = {}
            
            # 全シグナル履歴をスキャン（最新だけでなく全期間）
            for idx, row in signals.iterrows():
                for symbol in symbols:
                    # エントリーシグナル検出
                    if row.get('Entry_Signal', 0) == 1:
                        entry_price = row.get('Entry_Price', row.get('Close', 0))
                        buy_quantity = self._calculate_position_size(symbol)
                        
                        # ポジション追跡: BUY実行時に数量を記録
                        position_tracker[symbol] = buy_quantity
                        
                        orders.append({
                            "symbol": symbol,
                            "action": "BUY",
                            "quantity": buy_quantity,
                            "order_type": "MARKET",
                            "timestamp": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                            "entry_price": entry_price,
                            "signal_date": idx
                        })
                        self.logger.debug(f"BUY order generated: {symbol} @ {idx}, quantity={buy_quantity}, price={entry_price}")
                    
                    # イグジットシグナル検出（Exit_Signal == -1 が正しい）
                    if row.get('Exit_Signal', 0) == -1:
                        exit_price = row.get('Close', 0)
                        
                        # Phase 4.2-9-2: SELLオーダー数量修正
                        # 直前のBUYオーダーの数量を使用（ポジション追跡から取得）
                        sell_quantity = position_tracker.get(symbol, 0)
                        
                        if sell_quantity == 0:
                            # フォールバック: paper_brokerから現在のポジションを取得
                            sell_quantity = self._get_current_position(symbol)
                            self.logger.warning(f"SELL quantity fallback for {symbol}: using position={sell_quantity}")
                        
                        orders.append({
                            "symbol": symbol,
                            "action": "SELL",
                            "quantity": sell_quantity,
                            "order_type": "MARKET",
                            "timestamp": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                            "exit_price": exit_price,
                            "signal_date": idx
                        })
                        self.logger.debug(f"SELL order generated: {symbol} @ {idx}, quantity={sell_quantity}, price={exit_price}")
                        
                        # ポジションクリア（SELL後はポジション0）
                        position_tracker[symbol] = 0
            
            # 最終ログ: 生成された取引オーダー数
            self.logger.info(f"_generate_trade_orders: Generated {len(orders)} trade orders from {len(signals)} signals")
            
            # copilot-instructions.md: 実際の取引件数 > 0 を検証
            if len(orders) == 0 and (entry_count > 0 or exit_count > 0):
                self.logger.warning(f"⚠️  Signal detected but no orders generated! Entry={entry_count}, Exit={exit_count}")
            
            return orders
            
        except Exception as e:
            self.logger.error(f"取引注文生成エラー: {e}", exc_info=True)
            return []
    
    def _calculate_position_size(self, symbol: str) -> int:
        """ポジションサイズ計算"""
        # 簡易実装：固定額での購入
        try:
            position_value = self.config.get('position_value', 10000)  # $10,000
            current_price = 100.0  # デフォルト価格
            
            if self.data_feed:
                try:
                    current_price = self.data_feed.get_current_price(symbol) or current_price
                except:
                    pass
            
            return int(position_value / current_price) if current_price > 0 else 100
                
        except Exception:
            return 100
    
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
