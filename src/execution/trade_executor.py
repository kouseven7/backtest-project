from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging
import threading
from enum import Enum

# 他のモジュールからインポート（実際の実装で調整）
from .order_manager import Order, OrderType, OrderSide, OrderStatus, OrderManager, BrokerInterface
from .portfolio_tracker import PortfolioTracker

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
        
        # 実行設定
        self.execution_enabled = True
        self.risk_check_enabled = True
        self.position_limits: Dict[str, float] = {}  # シンボル別ポジション制限
        self.max_position_size_pct = 20.0  # 最大ポジションサイズ（ポートフォリオの%）
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
        if not self.execution_enabled:
            self.logger.warning("取引実行が無効化されています")
            return None
            
        try:
            with self.execution_lock:
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
                
                self.logger.info(f"注文提出: {order.symbol} {order.side.value} {order.quantity}")
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
        """リスク管理チェック"""
        try:
            # 1. 現金残高チェック
            if order.side == OrderSide.BUY:
                current_price = self.broker.get_current_price(order.symbol)
                required_cash = order.quantity * current_price * 1.1  # 10%マージン
                if self.portfolio_tracker.get_cash_balance() < required_cash:
                    self.logger.warning(f"資金不足: 必要 {required_cash}, 利用可能 {self.portfolio_tracker.get_cash_balance()}")
                    return False
            
            # 2. ポジションサイズ制限チェック
            current_position = self.portfolio_tracker.get_position(order.symbol)
            current_qty = current_position.quantity if current_position else 0
            
            if order.symbol in self.position_limits:
                if abs(current_qty + order.quantity) > self.position_limits[order.symbol]:
                    self.logger.warning(f"ポジション制限超過: {order.symbol}")
                    return False
            
            # 3. 最大ポジションサイズチェック
            total_equity = self.portfolio_tracker.get_total_equity()
            current_price = self.broker.get_current_price(order.symbol)
            position_value = abs(order.quantity) * current_price
            position_pct = (position_value / total_equity) * 100
            
            if position_pct > self.max_position_size_pct:
                self.logger.warning(f"最大ポジションサイズ超過: {position_pct:.1f}% > {self.max_position_size_pct}%")
                return False
            
            # 4. 外部リスク管理システム連携
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
            self.logger.error(f"リスク管理チェックエラー: {e}")
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
