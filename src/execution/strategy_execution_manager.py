"""
【非推奨 - DEPRECATED】
このファイルは旧バージョンです。新規開発・修正では使用しないでください。

⚠️ 警告: このモジュールは非推奨です
現在のメインシステムでは main_system/execution_control/strategy_execution_manager.py を使用してください。

このファイルは後方互換性のためにのみ保持されています。
- 旧スクリプト (performance_monitor*.py, paper_trade_runner.py等) がインポートエラーを起こさないための措置
- 新規機能追加・バグ修正は行われません
- 将来的には deprecated/ フォルダへ移動または削除予定

推奨される移行先:
    from main_system.execution_control.strategy_execution_manager import StrategyExecutionManager

Author: Backtest Project Team
Created: 2024-XX-XX (旧版)
Deprecated: 2025-10-20
Last Modified: 2025-10-20
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
# Phase 4.2-16: 手数料計算モジュールのインポート
from src.execution.commission_calculator import calculate_max_affordable_quantity, adjust_to_trading_unit

class StrategyExecutionManager:
    """
    戦略実行管理メインクラス
    
    ⚠️⚠️⚠️ 【非推奨 - DEPRECATED】 ⚠️⚠️⚠️
    このクラスは main_system/execution_control/strategy_execution_manager.py に移行されました。
    新規開発では main_system 版を使用してください。
    
    参照している旧スクリプトは deprecated/ フォルダに移動されています:
    - paper_trade_runner.py
    - performance_monitor*.py
    - demo_paper_trade_runner.py
    """
    
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
            # データフィード初期化（簡易版）
            self.data_feed = None  # シンプル実装のため無効化
            self.logger.info("データフィード: シンプルモード（サンプルデータ使用）")
            
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
    
    def execute_strategy(self, strategy_name: str, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """単一戦略実行（シンプルモード）"""
        try:
            self.logger.info(f"戦略実行開始: {strategy_name}")
            
            # デフォルトシンボル
            if symbols is None:
                symbols = self.config.get('default_symbols', ['AAPL', 'MSFT'])
            
            # データ取得
            market_data = self._get_market_data(symbols)
            if market_data is None or market_data.empty:
                return self._create_error_result("market_data_unavailable")
            
            # 戦略インスタンス取得
            strategy = self._get_strategy_instance(strategy_name)
            if strategy is None:
                return self._create_error_result(f"strategy_not_found: {strategy_name}")
            
            # 戦略実行（既存戦略はデータ引数なしで実行）
            signals = strategy.backtest()
            
            # 取引実行
            execution_results = self._execute_trades(signals, symbols)
            
            # 結果記録
            result = {
                "success": True,
                "strategy": strategy_name,
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
        """リアルタイム市場データ取得"""
        try:
            # 過去Nペリオドのデータ取得（戦略計算用）
            lookback_periods = self.config.get('lookback_periods', 100)
            
            if self.data_feed is not None:
                # 既存のリアルタイムフィードシステム使用
                data = self.data_feed.get_historical_data(
                    symbols=symbols, 
                    periods=lookback_periods
                )
                
                if data is not None and not data.empty:
                    self.logger.debug(f"市場データ取得成功: {len(data)}行, {len(symbols)}銘柄")
                    return data
            
            # フォールバック：簡易データ生成
            self.logger.warning("データフィード利用不可、サンプルデータを生成")
            return self._generate_sample_data(symbols, lookback_periods)
                
        except Exception as e:
            self.logger.error(f"市場データ取得エラー: {e}")
            return pd.DataFrame()
    
    def _generate_sample_data(self, symbols: List[str], periods: int) -> pd.DataFrame:
        """サンプルデータ生成（フォールバック用）"""
        import numpy as np
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=periods), 
                             end=datetime.now(), freq='D')
        
        # 基本的なサンプルデータ生成
        data_dict = {
            'Date': dates[:periods] if len(dates) >= periods else dates
        }
        
        # 各シンボルの価格データ生成
        for symbol in symbols:
            base_price = 100.0
            prices = [base_price]
            for i in range(1, len(data_dict['Date'])):
                change = np.random.normal(0, 0.02)  # 2%のボラティリティ
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 10.0))  # 最低価格制限
            
            # 戦略で必要なカラム名に合わせる
            data_dict['Adj Close'] = prices
            data_dict['High'] = [p * 1.02 for p in prices]  # High価格
            data_dict['Low'] = [p * 0.98 for p in prices]   # Low価格
            data_dict['Open'] = prices  # Open価格
            data_dict['Close'] = prices  # Close価格
            data_dict['Volume'] = [np.random.randint(1000000, 10000000) for _ in prices]
            break  # 最初のシンボルのみでサンプル作成
            
        return pd.DataFrame(data_dict)
    
    def _get_strategy_instance(self, strategy_name: str):
        """戦略インスタンス取得"""
        try:
            # 既知の戦略マッピング
            strategy_mappings = {
                'VWAP_Breakout': 'strategies.VWAP_Breakout.VWAPBreakoutStrategy',
                'VWAP_Bounce': 'strategies.VWAP_Bounce.VWAPBounceStrategy',
                'GC_Strategy': 'strategies.gc_strategy_signal.GCStrategy',
                'Breakout': 'strategies.Breakout.BreakoutStrategy',
                'Opening_Gap': 'strategies.Opening_Gap.OpeningGapStrategy'
            }
            
            module_path = strategy_mappings.get(strategy_name)
            if module_path:
                module_name, class_name = module_path.rsplit('.', 1)
                module = __import__(module_name, fromlist=[class_name])
                strategy_class = getattr(module, class_name)
                
                # 適切なサンプルデータで初期化
                sample_data = self._generate_sample_data(['AAPL'], 100)
                index_data = sample_data.copy()  # インデックスデータも同じにする
                return strategy_class(sample_data, index_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"戦略インスタンス取得エラー[{strategy_name}]: {e}")
            return None
    
    def _execute_trades(self, signals: pd.DataFrame, symbols: List[str]) -> List[Dict[str, Any]]:
        """取引実行"""
        try:
            execution_results = []
            
            if signals is None or signals.empty:
                return execution_results
            
            # シグナルから取引指示を生成
            trade_orders = self._generate_trade_orders(signals, symbols)
            
            # 各注文を実行
            for order in trade_orders:
                try:
                    if self.trade_executor:
                        result = self.trade_executor.execute_order(order)
                        execution_results.append(result)
                    else:
                        # モック実行
                        result = {
                            "order": order,
                            "status": "executed",
                            "timestamp": datetime.now().isoformat()
                        }
                        execution_results.append(result)
                except Exception as e:
                    self.logger.error(f"取引実行エラー: {e}")
                    execution_results.append({"error": str(e), "order": order})
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"取引実行処理エラー: {e}")
            return []
    
    def _generate_trade_orders(self, signals: pd.DataFrame, symbols: List[str]) -> List[Dict[str, Any]]:
        """シグナルから取引注文生成（Phase 4.2-20: 価格登録BUG FIX）"""
        try:
            orders = []
            
            # 最新のシグナルを確認
            if signals.empty:
                return orders
            
            latest_signals = signals.tail(1).iloc[0]
            
            # Phase 4.2-21 DEBUG: latest_signalsの構造確認
            self.logger.info(f"[SIGNALS_DEBUG] latest_signals type: {type(latest_signals)}")
            self.logger.info(f"[SIGNALS_DEBUG] latest_signals columns: {list(latest_signals.index)}")
            self.logger.info(f"[SIGNALS_DEBUG] latest_signals values sample: {dict(list(latest_signals.items())[:5])}")
            
            for symbol in symbols:
                # Phase 4.2-20: BUY/SELL実行前に、必ずPaperBrokerに正しい価格を登録
                # これにより_calculate_position_size()でデフォルト100円が使われるバグを防ぐ
                # Phase 4.2-21 BUG FIX: 'Close'だけでなく'Adj Close'もチェック
                current_price = None
                
                # 価格カラムの優先順位: Close > Adj Close > 他の価格カラム
                price_columns_to_check = ['Close', 'Adj Close', 'close', 'adj_close', 'price']
                for price_col in price_columns_to_check:
                    if price_col in latest_signals:
                        price_value = latest_signals[price_col]
                        if pd.notna(price_value) and price_value > 0:
                            current_price = float(price_value)
                            self.logger.debug(f"[PRICE_SOURCE] {symbol}: 使用カラム='{price_col}', 価格={current_price:.2f}円")
                            break
                
                if current_price is not None and self.paper_broker:
                    # Phase 4.2-21: symbol文字列検証デバッグログ
                    self.logger.info(f"[PRICE_REG_DEBUG] symbol='{symbol}' | len={len(symbol)} | repr={repr(symbol)} | type={type(symbol).__name__}")
                    self.logger.info(f"[PRICE_REG_DEBUG] price={current_price:.2f} | type={type(current_price).__name__}")
                    
                    self.paper_broker.update_price(symbol, current_price)
                    self.logger.debug(f"[PRICE_REG] PaperBrokerに価格登録: {symbol} = {current_price:.2f}円")
                    
                    # Phase 4.2-21: 登録直後の取得テスト
                    verify_price = self.paper_broker.get_current_price(symbol)
                    match_status = "OK" if abs(verify_price - current_price) < 0.01 else "NG"
                    self.logger.info(f"[PRICE_REG_VERIFY] 登録直後の取得: {verify_price:.2f}円 (expected: {current_price:.2f}円) [{match_status}]")
                else:
                    self.logger.warning(f"[PRICE_REG_WARN] {symbol}: 価格カラムが見つからない、または価格が無効です。available columns: {list(latest_signals.index)}")
                
                # エントリーシグナルチェック
                if hasattr(latest_signals, 'Entry_Signal') and latest_signals.Entry_Signal == 1:
                    orders.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "quantity": self._calculate_position_size(symbol),
                        "order_type": "MARKET",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # エグジットシグナルチェック
                elif hasattr(latest_signals, 'Exit_Signal') and latest_signals.Exit_Signal == 1:
                    orders.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": self._get_current_position(symbol),
                        "order_type": "MARKET",
                        "timestamp": datetime.now().isoformat()
                    })
            
            return orders
            
        except Exception as e:
            self.logger.error(f"取引注文生成エラー: {e}")
            return []
    
    def _calculate_position_size(self, symbol: str) -> int:
        """
        ポジションサイズ計算（Phase 4.2-16: 日本株対応版）
        
        手数料・単元株制度を考慮した正確な計算を実行します。
        利用可能資金の90%を使用し、100株単位で購入可能な最大株数を返します。
        
        Returns:
            int: 100株単位に調整された株数（購入不可の場合は0）
        """
        try:
            # 1. 利用可能資金の90%を使用（Phase 4.2-16要件）
            if not self.paper_broker:
                self.logger.error("❌ PaperBrokerが初期化されていません")
                return 0
            
            total_equity = self.paper_broker.account_balance
            available_funds = total_equity * 0.90  # 90%使用
            
            self.logger.debug(f"📊 資金確認: 総資産={total_equity:.0f}円, 利用可能(90%)={available_funds:.0f}円")
            
            # 2. 現在価格取得（実データのみ使用 - copilot-instructions.md準拠）
            current_price = None
            
            # DataFeedから取得試行
            if self.data_feed:
                try:
                    current_price = self.data_feed.get_current_price(symbol)
                    if current_price and current_price > 0:
                        self.logger.debug(f"💰 DataFeedから価格取得: {symbol} = {current_price:.2f}円")
                except Exception as e:
                    self.logger.warning(f"⚠️ DataFeedから価格取得失敗: {e}")
            
            # PaperBrokerから取得（DataFeedが失敗した場合）
            if current_price is None or current_price <= 0:
                if self.paper_broker:
                    current_price = self.paper_broker.get_current_price(symbol)
                    if current_price and current_price > 0:
                        self.logger.debug(f"💰 PaperBrokerから価格取得: {symbol} = {current_price:.2f}円")
            
            # 価格検証
            if current_price is None or current_price <= 0:
                self.logger.error(f"❌ 有効な価格が取得できません: {symbol} (price={current_price})")
                return 0
            
            # 3. 手数料を考慮した購入可能株数を計算
            # commission_calculator.pyの関数を使用
            quantity, contract_value, commission, total_cost = calculate_max_affordable_quantity(
                available_funds=available_funds,
                stock_price=current_price,
                unit_size=100,  # 日本株は100株単位
                include_slippage=True,
                slippage_rate=0.0001  # 0.01%
            )
            
            if quantity == 0:
                self.logger.warning(
                    f"⚠️ 資金不足: {symbol} 価格={current_price:.2f}円, "
                    f"利用可能={available_funds:.0f}円では100株も購入できません"
                )
                return 0
            
            # 4. 単元株数に調整（念のため再確認）
            final_quantity = adjust_to_trading_unit(quantity, unit_size=100, round_up=False)
            
            if final_quantity != quantity:
                self.logger.warning(
                    f"⚠️ 株数調整: {quantity}株 → {final_quantity}株（100株単位）"
                )
            
            # 5. 詳細ログ出力
            self.logger.info(
                f"✅ ポジションサイズ計算完了: {symbol} "
                f"{final_quantity}株 @ {current_price:.2f}円 "
                f"(約定代金: {contract_value:,.0f}円, 手数料: {commission:.0f}円, "
                f"総コスト: {total_cost:,.0f}円, 残金: {available_funds - total_cost:,.0f}円)"
            )
            
            return final_quantity
            
        except Exception as e:
            self.logger.error(f"❌ ポジションサイズ計算エラー: {e}", exc_info=True)
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
