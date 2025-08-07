"""
注文管理システム
成行・指値・逆指値対応
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime
import uuid
import logging

# ブローカーインターフェイス（抽象基底クラス）
class BrokerInterface:
    """ブローカーインターフェイス - 実際のブローカーまたはペーパートレードで実装"""
    
    def execute_order(self, order: 'Order') -> bool:
        """オーダーを実行する"""
        raise NotImplementedError
    
    def submit_order(self, order: 'Order') -> bool:
        """オーダーを提出する"""
        raise NotImplementedError
    
    def cancel_order(self, order_id: str) -> bool:
        """オーダーをキャンセルする"""
        raise NotImplementedError
    
    def get_current_price(self, symbol: str) -> float:
        """現在価格を取得"""
        raise NotImplementedError
    
    def get_account_balance(self) -> float:
        """口座残高を取得"""
        raise NotImplementedError


class OrderType(Enum):
    """注文タイプ"""
    MARKET = "market"           # 成行
    LIMIT = "limit"             # 指値
    STOP_LOSS = "stop_loss"     # 逆指値（ストップロス）
    STOP_LIMIT = "stop_limit"   # ストップ指値


class OrderSide(Enum):
    """注文方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """注文状態"""
    PENDING = "pending"         # 待機中
    FILLED = "filled"           # 約定
    CANCELLED = "cancelled"     # 取消
    REJECTED = "rejected"       # 拒否
    PARTIAL = "partial"         # 部分約定


@dataclass
class Order:
    """注文クラス"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None       # 指値価格
    stop_price: Optional[float] = None  # ストップ価格
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    strategy_name: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        """辞書形式に変換"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price else '',
            'stop_price': str(self.stop_price) if self.stop_price else '',
            'status': self.status.value,
            'created_at': self.created_at.isoformat() if self.created_at else '',
            'filled_at': self.filled_at.isoformat() if self.filled_at else '',
            'filled_price': str(self.filled_price) if self.filled_price else '',
            'filled_quantity': str(self.filled_quantity) if self.filled_quantity else '',
            'commission': str(self.commission),
            'slippage': str(self.slippage),
            'strategy_name': self.strategy_name or ''
        }


class OrderManager:
    """注文管理"""
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.orders: Dict[str, Order] = {}  # 注文辞書
        self.order_callbacks: Dict[str, Callable[[Order], None]] = {}  # コールバック
        self.logger = logging.getLogger(__name__)
        
    def submit_order(self, order: Order, callback: Optional[Callable[[Order], None]] = None) -> str:
        """注文提出"""
        try:
            # 注文検証
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"注文検証失敗: {order.symbol} {order.side.value} {order.quantity}")
                return order.id
                
            # 注文登録
            self.orders[order.id] = order
            if callback:
                self.order_callbacks[order.id] = callback
                
            # ブローカーに送信
            if self.broker:
                self.broker.submit_order(order)
                
            self.logger.info(f"注文提出: {order.symbol} {order.side.value} {order.quantity}")
            return order.id
            
        except Exception as e:
            self.logger.error(f"注文提出エラー: {e}")
            order.status = OrderStatus.REJECTED
            return order.id
        
    def cancel_order(self, order_id: str) -> bool:
        """注文取消"""
        try:
            if order_id not in self.orders:
                self.logger.warning(f"注文ID不明: {order_id}")
                return False
                
            order = self.orders[order_id]
            if order.status == OrderStatus.FILLED:
                self.logger.warning(f"約定済み注文の取消失敗: {order_id}")
                return False
                
            order.status = OrderStatus.CANCELLED
            if self.broker:
                self.broker.cancel_order(order_id)
                
            self.logger.info(f"注文取消: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"注文取消エラー: {e}")
            return False
        
    def get_order(self, order_id: str) -> Optional[Order]:
        """注文取得"""
        return self.orders.get(order_id)
        
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """シンボル別注文取得"""
        return [order for order in self.orders.values() if order.symbol == symbol]
        
    def get_active_orders(self) -> List[Order]:
        """アクティブ注文取得"""
        return [order for order in self.orders.values() 
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]]
        
    def _validate_order(self, order: Order) -> bool:
        """注文検証"""
        try:
            # 基本検証
            if order.quantity <= 0:
                self.logger.warning(f"無効な数量: {order.quantity}")
                return False
                
            if not order.symbol:
                self.logger.warning("シンボルが空")
                return False
                
            # 指値注文の価格チェック
            if order.order_type == OrderType.LIMIT and order.price is None:
                self.logger.warning("指値注文に価格が設定されていません")
                return False
                
            # ストップ注文の価格チェック
            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT] and order.stop_price is None:
                self.logger.warning("ストップ注文にストップ価格が設定されていません")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"注文検証エラー: {e}")
            return False
        
    def register_callback(self, order_id: str, callback: Callable[[Order], None]):
        """コールバック登録"""
        self.order_callbacks[order_id] = callback
        
    def execute_callback(self, order_id: str, order: Order):
        """コールバック実行"""
        if order_id in self.order_callbacks:
            try:
                callback = self.order_callbacks[order_id]
                callback(order)
            except Exception as e:
                self.logger.error(f"コールバック実行エラー {order_id}: {e}")
            finally:
                # コールバック削除
                if order_id in self.order_callbacks:
                    del self.order_callbacks[order_id]
                    
    def get_order_statistics(self) -> Dict[str, Any]:
        """注文統計情報"""
        try:
            total_orders = len(self.orders)
            status_counts = {}
            
            for status in OrderStatus:
                count = sum(1 for order in self.orders.values() if order.status == status)
                status_counts[status.value] = count
                
            return {
                "total_orders": total_orders,
                "status_breakdown": status_counts,
                "active_orders": len(self.get_active_orders())
            }
            
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {}
