"""
Trade Executor Module - 取引実行エンジン

このモジュールは、バックテスト、ペーパートレーディング、ライブトレーディングにおける
取引実行を管理する中核的なコンポーネントです。

主な機能:
- 成行注文、指値注文、逆指値注文の作成と実行
- リスク管理チェック(資金残高、ポジションサイズ制限など)
- ポジションサイジング調整
- 注文管理と約定処理
- ポートフォリオトラッキングとの統合
- 緊急停止とポジション一括決済

統合コンポーネント:
- OrderManager: 注文の管理と実行
- PortfolioTracker: ポートフォリオの追跡と記録
- BrokerInterface: ブローカーとの通信
- リスク管理システム(portfolio_risk_manager.py)との連携
- ポジションサイジングシステム(position_size_adjuster.py)との連携

使用例:
    executor = TradeExecutor(
        portfolio_tracker=portfolio_tracker,
        broker=broker,
        mode=ExecutionMode.BACKTEST
    )
    
    # 成行注文の作成と実行
    order = executor.create_market_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        strategy_name="MyStrategy"
    )
    order_id = executor.submit_order(order)
    
    # 緊急停止
    executor.emergency_stop()

セーフティ機能:
- 実行モード切り替え(バックテスト/ペーパー/ライブ)
- 取引実行の有効/無効切り替え
- リスクチェックの有効/無効切り替え
- 最大ポジションサイズ制限
- 銘柄別ポジション制限
- 緊急停止機能

Author: Backtest Project Team
Created: 2025-10-20
Last Modified: 2025-10-20
"""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging
import threading
from enum import Enum

# 他のモジュールからインポート(実際の実装で調整)
from .order_manager import Order, OrderType, OrderSide, OrderStatus, OrderManager, BrokerInterface
from .portfolio_tracker import PortfolioTracker
# Phase 4.2-16: 手数料計算モジュールのインポート
from .commission_calculator import calculate_japanese_stock_commission

class ExecutionMode(Enum):
    """実行モード"""
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"

class TradeExecutor:
    """取引実行エンジン"""
    
    def __init__(self, 
                 portfolio_tracker: PortfolioTracker,
                 broker: BrokerInterface,
                 mode: ExecutionMode = ExecutionMode.PAPER_TRADING):
        
        self.portfolio_tracker = portfolio_tracker
        self.broker = broker
        self.mode = mode
        self.order_manager = OrderManager(broker)
        self.logger = logging.getLogger(__name__)
        
        # Phase 4.2-11: バックテストモードの場合、brokerに伝播
        if mode == ExecutionMode.BACKTEST and hasattr(broker, 'backtest_mode'):
            broker.backtest_mode = True
            self.logger.info("Backtest mode enabled: market hours check disabled")
        
        # 実行設定
        self.execution_enabled = True
        self.risk_check_enabled = True
        self.position_limits: Dict[str, float] = {}  # シンボル別ポジション制限
        self.max_position_size_pct = 90.0  # Phase 4.2-16: 最大ポジションサイズ90%に変更
        self.max_daily_loss_pct = 5.0  # 最大日次損失（%）
        
        # 注文実行待ちキュー
        self.pending_orders: List[Order] = []
        self.execution_lock = threading.Lock()
        
        # リスク管理インテグレーション用（既存システムとの接続）
        self.risk_manager: Optional[Any] = None  # portfolio_risk_manager.pyと接続
        self.position_sizer: Optional[Any] = None  # position_size_adjuster.pyと接続
        
    def set_risk_manager(self, risk_manager: Any):
        """リスク管理システムを設定"""
        self.risk_manager = risk_manager
        
    def set_position_sizer(self, position_sizer: Any):
        """ポジションサイジングシステムを設定"""
        self.position_sizer = position_sizer
    
    def enable_execution(self, enabled: bool = True):
        """取引実行の有効/無効を設定"""
        self.execution_enabled = enabled
        self.logger.info(f"取引実行: {'有効' if enabled else '無効'}")
    
    def set_position_limit(self, symbol: str, limit: float):
        """銘柄別ポジション制限を設定"""
        self.position_limits[symbol] = limit
        
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float, 
                           strategy_name: str = "") -> Optional[Order]:
        """成行注文を作成"""
        try:
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                strategy_name=strategy_name
            )
            return order
        except Exception as e:
            self.logger.error(f"成行注文作成エラー: {e}")
            return None
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, 
                          price: float, strategy_name: str = "") -> Optional[Order]:
        """指値注文を作成"""
        try:
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                strategy_name=strategy_name
            )
            return order
        except Exception as e:
            self.logger.error(f"指値注文作成エラー: {e}")
            return None
    
    def create_stop_order(self, symbol: str, side: OrderSide, quantity: float, 
                         stop_price: float, strategy_name: str = "") -> Optional[Order]:
        """逆指値注文を作成"""
        try:
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP_LOSS,
                quantity=quantity,
                stop_price=stop_price,
                strategy_name=strategy_name
            )
            return order
        except Exception as e:
            self.logger.error(f"逆指値注文作成エラー: {e}")
            return None
    
    def submit_order(self, order: Order, 
                    callback: Optional[Callable[[Order], None]] = None) -> Optional[str]:
        """注文を提出"""
        # Phase 4.2-32-2: submit_order()開始ログ
        self.logger.info(
            f"[PHASE_4_2_32_2] submit_order() 開始: {order.symbol} {order.side.value}, quantity={order.quantity}"
        )
        
        if not self.execution_enabled:
            self.logger.warning("取引実行が無効化されています")
            return None
            
        try:
            with self.execution_lock:
                # Phase 4.2-32-2: リスク管理チェック前ログ
                self.logger.info(
                    f"[PHASE_4_2_32_2] リスク管理チェック前: risk_check_enabled={self.risk_check_enabled}"
                )
                
                # リスク管理チェック
                if self.risk_check_enabled and not self._risk_check(order):
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"リスク管理により注文拒否: {order.symbol}")
                    return None
                
                # ポジションサイジング調整
                if self.position_sizer:
                    adjusted_quantity = self._adjust_position_size(order)
                    if adjusted_quantity != order.quantity:
                        self.logger.info(f"ポジションサイズ調整: {order.quantity} -> {adjusted_quantity}")
                        order.quantity = adjusted_quantity
                
                # 注文管理システムに提出
                order_id = self.order_manager.submit_order(order, self._create_order_callback(callback))
                
                # Phase 4.2-30: order_id検証とorder.status確認
                self.logger.info(
                    f"[PHASE_4_2_30] 注文提出完了: {order.symbol} {order.side.value} {order.quantity}, "
                    f"order_id={order_id}, order.status={order.status.value}"
                )
                
                # Phase 4.2-30: REJECTEDの場合はNoneを返す
                if order.status == OrderStatus.REJECTED:
                    self.logger.warning(
                        f"[PHASE_4_2_30] 注文がREJECTED状態のためNoneを返却: {order.symbol} {order.side.value}"
                    )
                    return None
                
                return order_id
                
        except Exception as e:
            self.logger.error(f"注文提出エラー: {e}")
            return None
    
    def submit_batch_orders(self, orders: List[Order]) -> List[Optional[str]]:
        """複数注文を一括提出"""
        order_ids: List[Optional[str]] = []
        for order in orders:
            order_id = self.submit_order(order)
            order_ids.append(order_id)
        return order_ids
    
    def cancel_order(self, order_id: str) -> bool:
        """注文をキャンセル"""
        try:
            result = self.order_manager.cancel_order(order_id)
            if result:
                self.logger.info(f"注文キャンセル: {order_id}")
            return result
        except Exception as e:
            self.logger.error(f"注文キャンセルエラー: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """全注文またはシンボル別注文をキャンセル"""
        try:
            active_orders = self.order_manager.get_active_orders()
            cancelled_count = 0
            
            for order in active_orders:
                if symbol is None or order.symbol == symbol:
                    if self.cancel_order(order.id):
                        cancelled_count += 1
            
            self.logger.info(f"注文キャンセル完了: {cancelled_count}件")
            return cancelled_count
            
        except Exception as e:
            self.logger.error(f"全注文キャンセルエラー: {e}")
            return 0
    
    def _risk_check(self, order: Order) -> bool:
        """
        リスク管理チェック（Phase 4.2-16: 手数料・単元株対応版）
        """
        try:
            # Phase 4.2-32-2: _risk_check()開始ログ
            self.logger.info(
                f"[PHASE_4_2_32_2] _risk_check() 開始: {order.symbol} {order.side.value}, quantity={order.quantity}"
            )
            
            # 1. 現金残高チェック（手数料込み）
            if order.side == OrderSide.BUY:
                current_price = self.broker.get_current_price(order.symbol)
                if current_price <= 0:
                    self.logger.error(f"[ERROR] 無効な価格: {order.symbol}")
                    return False
                
                # 約定代金
                contract_value = order.quantity * current_price
                
                # 手数料計算
                commission = calculate_japanese_stock_commission(contract_value)
                
                # スリッページ（0.01%）
                slippage = contract_value * 0.0001
                
                # 必要資金 = 約定代金 + 手数料 + スリッページ
                required_cash = contract_value + commission + slippage
                
                available_cash = self.portfolio_tracker.get_cash_balance()
                
                if available_cash < required_cash:
                    self.logger.warning(
                        f"[WARNING] 資金不足: 必要 {required_cash:.2f}円 "
                        f"(約定代金 {contract_value:.2f}円 + 手数料 {commission:.0f}円 + スリッページ {slippage:.2f}円), "
                        f"利用可能 {available_cash:.2f}円"
                    )
                    return False
            
            # Phase 4.2-27: SELL注文のポジション数量チェック
            # Phase 4.2-28: PaperBrokerから直接ポジション取得（PortfolioTracker同期問題の解決）
            if order.side == OrderSide.SELL:
                # Phase 4.2-32-2: SELL注文チェック開始ログ
                self.logger.info(
                    f"[PHASE_4_2_32_2] _risk_check() SELL注文チェック開始: {order.symbol}, "
                    f"注文数量={order.quantity}"
                )
                
                # Brokerから現在のポジションを取得
                broker_positions = self.broker.get_positions()
                current_qty = 0
                
                # Phase 4.2-32-2: broker_positions詳細ログ
                self.logger.info(
                    f"[PHASE_4_2_32_2] broker_positions keys: {list(broker_positions.keys())}, "
                    f"has {order.symbol}: {order.symbol in broker_positions}"
                )
                
                if order.symbol in broker_positions:
                    position_data = broker_positions[order.symbol]
                    current_qty = position_data.get('quantity', 0)
                    
                    # Phase 4.2-32-2: position_data詳細ログ
                    self.logger.info(
                        f"[PHASE_4_2_32_2] position_data: {position_data}"
                    )
                
                self.logger.info(
                    f"[PHASE_4_2_28] SELL注文チェック: {order.symbol}, "
                    f"注文数量={order.quantity}, 保有数量={current_qty} (from PaperBroker)"
                )
                
                if current_qty <= 0:
                    self.logger.warning(
                        f"[PHASE_4_2_28] SELL注文拒否: {order.symbol} "
                        f"(理由=ポジション未保有, 保有数量={current_qty})"
                    )
                    # Phase 4.2-32-2: 拒否理由詳細ログ
                    self.logger.info(
                        f"[PHASE_4_2_32_2] SELL注文REJECTED: current_qty={current_qty} <= 0"
                    )
                    return False
                
                if order.quantity > current_qty:
                    self.logger.warning(
                        f"[PHASE_4_2_28] SELL注文拒否: {order.symbol} "
                        f"(理由=数量超過, 注文={order.quantity}, 保有={current_qty})"
                    )
                    # Phase 4.2-32-2: 拒否理由詳細ログ
                    self.logger.info(
                        f"[PHASE_4_2_32_2] SELL注文REJECTED: order.quantity={order.quantity} > current_qty={current_qty}"
                    )
                    return False
                
                # Phase 4.2-32-2: SELL注文チェック通過ログ
                self.logger.info(
                    f"[PHASE_4_2_32_2] _risk_check() SELL注文チェック通過: {order.symbol}"
                )
            
            # 2. ポジションサイズ制限チェック（Phase 4.2-27: SELL注文の数量反転対応）
            # Phase 4.2-28: PaperBrokerから直接ポジション取得
            broker_positions = self.broker.get_positions()
            current_qty = 0
            
            if order.symbol in broker_positions:
                position_data = broker_positions[order.symbol]
                current_qty = position_data.get('quantity', 0)
            
            if order.symbol in self.position_limits:
                # Phase 4.2-27: SELL注文の場合は減算
                if order.side == OrderSide.SELL:
                    new_qty = current_qty - order.quantity
                else:
                    new_qty = current_qty + order.quantity
                
                if abs(new_qty) > self.position_limits[order.symbol]:
                    self.logger.warning(
                        f"[PHASE_4_2_27] ポジション制限超過: {order.symbol}, "
                        f"現在={current_qty}, 注文後={new_qty}, 上限={self.position_limits[order.symbol]}"
                    )
                    return False
            
            # 3. 最大ポジションサイズチェック（正確な計算）
            # Phase 4.2-32-3: current_priceを事前取得（外部リスク管理システム連携でも使用）
            current_price = self.broker.get_current_price(order.symbol)
            
            # Phase 4.2-32-3: SELL注文はポジション減少のため、最大ポジションサイズチェックをスキップ
            if order.side == OrderSide.SELL:
                self.logger.info(
                    f"[PHASE_4_2_32_3] SELL注文のため、最大ポジションサイズチェックをスキップ: {order.symbol}"
                )
            else:
                # BUY注文のみチェック
                # Phase 4.2-32-2: total_equity取得前ログ
                self.logger.info(
                    f"[PHASE_4_2_32_2] 最大ポジションサイズチェック開始: {order.symbol} {order.side.value}"
                )
                
                total_equity = self.portfolio_tracker.get_total_equity()
                
                # Phase 4.2-32-2: total_equity取得後ログ
                self.logger.info(
                    f"[PHASE_4_2_32_2] total_equity={total_equity:.2f}円"
                )
                
                if total_equity <= 0:
                    self.logger.error(f"[ERROR] total_equity が無効: {total_equity}")
                    # Phase 4.2-32-2: 拒否理由詳細ログ
                    self.logger.info(
                        f"[PHASE_4_2_32_2] REJECTED: total_equity={total_equity} <= 0"
                    )
                    return False
                
                position_value = abs(order.quantity) * current_price
                position_pct = (position_value / total_equity) * 100
                
                # Phase 4.2-32-2: ポジションサイズ計算結果ログ
                self.logger.info(
                    f"[PHASE_4_2_32_2] ポジションサイズ計算: position_value={position_value:.2f}円, "
                    f"position_pct={position_pct:.1f}%, max={self.max_position_size_pct}%"
                )
                
                self.logger.debug(
                    f"[INFO] ポジションサイズチェック: {order.symbol} "
                    f"注文額={position_value:.2f}円, 総資産={total_equity:.2f}円, "
                    f"比率={position_pct:.1f}%, 上限={self.max_position_size_pct}%"
                )
                
                if position_pct > self.max_position_size_pct:
                    self.logger.warning(
                        f"[WARNING] 最大ポジションサイズ超過: {position_pct:.1f}% > {self.max_position_size_pct}%"
                    )
                    # Phase 4.2-32-2: 拒否理由詳細ログ
                    self.logger.info(
                        f"[PHASE_4_2_32_2] REJECTED: position_pct={position_pct:.1f}% > max={self.max_position_size_pct}%"
                    )
                    return False
                
                # Phase 4.2-32-2: ポジションサイズチェック通過ログ
                self.logger.info(
                    f"[PHASE_4_2_32_2] 最大ポジションサイズチェック通過: {order.symbol}"
                )
            
            # 4. 単元株チェック（日本株）
            if order.quantity % 100 != 0:
                self.logger.warning(f"[WARNING] 単元未満株: {order.quantity}株（100株単位必須）")
                return False
            
            # 5. 外部リスク管理システム連携
            if self.risk_manager:
                try:
                    # 既存のportfolio_risk_manager.pyと連携
                    risk_result = self.risk_manager.check_trade_risk(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        price=current_price
                    )
                    if not risk_result.get('approved', True):
                        self.logger.warning(f"外部リスク管理拒否: {risk_result.get('reason', 'Unknown')}")
                        return False
                except Exception as e:
                    self.logger.error(f"外部リスク管理チェックエラー: {e}")
                    # エラー時は保守的に拒否
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] リスク管理チェックエラー: {e}", exc_info=True)
            return False
    
    def _adjust_position_size(self, order: Order) -> float:
        """ポジションサイズを調整"""
        try:
            if not self.position_sizer:
                return order.quantity
            
            # 既存のposition_size_adjuster.pyと連携
            adjusted_size = self.position_sizer.calculate_position_size(
                symbol=order.symbol,
                signal_strength=1.0,  # デフォルト値
                current_price=self.broker.get_current_price(order.symbol),
                portfolio_value=self.portfolio_tracker.get_total_equity()
            )
            
            # 売り注文の場合は負の値にする
            if order.side == OrderSide.SELL:
                adjusted_size = -abs(adjusted_size)
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"ポジションサイズ調整エラー: {e}")
            return order.quantity
    
    def _create_order_callback(self, user_callback: Optional[Callable[[Order], None]]) -> Callable[[Order], None]:
        """注文コールバックを作成"""
        def order_callback(order: Order):
            try:
                # ポートフォリオ追跡への通知
                if order.status == OrderStatus.FILLED:
                    self._handle_filled_order(order)
                
                # ユーザーコールバック実行
                if user_callback:
                    user_callback(order)
                    
            except Exception as e:
                self.logger.error(f"注文コールバックエラー: {e}")
        
        return order_callback
    
    def _handle_filled_order(self, order: Order):
        """約定注文の処理"""
        try:
            # ポートフォリオトラッカーに取引を記録
            quantity = order.filled_quantity or order.quantity
            price = order.filled_price or order.price
            
            if order.side == OrderSide.SELL:
                quantity = -abs(quantity)
            
            self.portfolio_tracker.execute_trade(
                symbol=order.symbol,
                quantity=quantity,
                price=price,
                commission=order.commission,
                slippage=order.slippage,
                strategy_name=order.strategy_name,
                trade_id=order.id
            )
            
            self.logger.info(f"取引記録完了: {order.symbol} {quantity} @ {price}")
            
        except Exception as e:
            self.logger.error(f"約定処理エラー: {e}")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """実行状況を取得"""
        try:
            order_stats = self.order_manager.get_order_statistics()
            portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
            
            status = {
                'execution_enabled': self.execution_enabled,
                'mode': self.mode.value,
                'orders': order_stats,
                'portfolio': portfolio_summary,
                'risk_settings': {
                    'max_position_size_pct': self.max_position_size_pct,
                    'max_daily_loss_pct': self.max_daily_loss_pct,
                    'position_limits': self.position_limits
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"実行状況取得エラー: {e}")
            return {}
    
    def emergency_stop(self):
        """緊急停止"""
        try:
            self.logger.warning("緊急停止実行中...")
            
            # 取引実行を無効化
            self.execution_enabled = False
            
            # 全注文をキャンセル
            cancelled_count = self.cancel_all_orders()
            
            self.logger.warning(f"緊急停止完了: {cancelled_count}件の注文をキャンセル")
            
        except Exception as e:
            self.logger.error(f"緊急停止エラー: {e}")
    
    def close_all_positions(self, emergency: bool = False) -> List[str]:
        """全ポジションを決済"""
        try:
            positions = self.portfolio_tracker.get_all_positions()
            orders_submitted: List[str] = []
            
            for symbol, position in positions.items():
                if abs(position.quantity) > 1e-8:  # 微小ポジションは無視
                    # 反対売買で決済
                    side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                    quantity = abs(position.quantity)
                    
                    order = self.create_market_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        strategy_name="POSITION_CLOSE"
                    )
                    
                    if order:
                        if emergency:
                            # 緊急時はリスクチェックをスキップ
                            self.risk_check_enabled = False
                        
                        order_id = self.submit_order(order)
                        if order_id:
                            orders_submitted.append(order_id)
                        
                        if emergency:
                            # リスクチェックを復元
                            self.risk_check_enabled = True
            
            self.logger.info(f"ポジション決済注文提出: {len(orders_submitted)}件")
            return orders_submitted
            
        except Exception as e:
            self.logger.error(f"ポジション決済エラー: {e}")
            return []
