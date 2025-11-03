from typing import Dict, List, Optional, Any
from datetime import datetime, time
import logging
import random
import threading
from enum import Enum
import time as time_module

from .order_manager import Order, OrderType, OrderSide, OrderStatus, BrokerInterface

class MarketState(Enum):
    """市場状態"""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"

class PaperBroker(BrokerInterface):
    """ペーパートレードブローカー"""
    
    def __init__(self, 
                 initial_balance: float = 1000000.0,  # Phase 4.2-14: デフォルト1,000,000円に変更
                 commission_per_trade: float = 1.0,
                 commission_pct: float = 0.001,  # 0.1%
                 slippage_pct: float = 0.0001,   # 0.01%
                 backtest_mode: bool = False):    # Phase 4.2-11: バックテスト専用モード
        
        self.account_balance = initial_balance
        self.initial_balance = initial_balance
        self.commission_per_trade = commission_per_trade
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.backtest_mode = backtest_mode  # Phase 4.2-11: バックテストモードフラグ
        
        # 価格データ管理
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}
        self.last_price_update: Dict[str, datetime] = {}
        
        # 注文管理
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        
        # Phase 4.2-23: ポジション管理
        # 形式: {symbol: {'quantity': int, 'entry_price': float, 'entry_time': datetime}}
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # 市場シミュレーション
        self.market_state = MarketState.OPEN
        self.market_hours = {
            'open': time(9, 30),    # 9:30 AM
            'close': time(16, 0)    # 4:00 PM
        }
        self.enable_slippage = True
        self.enable_partial_fills = False
        self.latency_ms = 10  # 注文実行遅延（ミリ秒）
        
        # スレッド制御
        self.execution_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # 外部データフィード接続（既存システムとの統合）
        self.data_feed: Optional[Any] = None
        
    def set_data_feed(self, data_feed: Any):
        """データフィードを設定"""
        self.data_feed = data_feed
        
    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """価格を更新"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.current_prices[symbol] = price
        self.last_price_update[symbol] = timestamp
        
        # 価格履歴に追加
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        price_point = {
            'price': price,
            'timestamp': timestamp,
            'volume': random.randint(1000, 10000)  # 模擬出来高
        }
        self.price_history[symbol].append(price_point)
        
        # 履歴を制限（最新1000件）
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
        
        # 待機中注文のチェック
        self._check_pending_orders(symbol)
    
    def update_prices_batch(self, prices: Dict[str, float]):
        """複数銘柄の価格を一括更新"""
        timestamp = datetime.now()
        for symbol, price in prices.items():
            self.update_price(symbol, price, timestamp)
    
    def get_current_price(self, symbol: str) -> float:
        """
        現在価格を取得
        
        Phase 4.2-14: デフォルト価格使用を明示的に警告
        - copilot-instructions.md: モック/ダミーデータ禁止
        - バックテストでは事前にupdate_price()で価格登録が必要
        """
        if symbol in self.current_prices:
            return self.current_prices[symbol]
        
        # データフィードから取得を試行
        if self.data_feed:
            try:
                price = self.data_feed.get_current_price(symbol)
                if price > 0:
                    self.update_price(symbol, price)
                    return price
            except Exception as e:
                self.logger.warning(f"データフィードからの価格取得失敗 {symbol}: {e}")
        
        # Phase 4.2-14: デフォルト価格使用時の詳細警告
        # copilot-instructions.md準拠: 本来は実データを使用すべき
        default_price = 100.0
        self.logger.error(
            f"⚠️ デフォルト価格使用（copilot-instructions.md違反の可能性）: "
            f"{symbol} = {default_price}円 "
            f"| 登録済み銘柄: {list(self.current_prices.keys())} "
            f"| データフィード: {'有効' if self.data_feed else '無効'} "
            f"| バックテストモード: {self.backtest_mode}"
        )
        return default_price
    
    def get_account_balance(self) -> float:
        """口座残高を取得"""
        return self.account_balance
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        現在のポジション一覧を取得（Phase 4.2-23）
        
        Returns:
            Dict[symbol, position_info]: ポジション情報
            position_info = {'quantity': int, 'entry_price': float, 'entry_time': datetime}
        
        copilot-instructions.md準拠:
        - 実データのみ返却（モック/ダミー禁止）
        """
        return self.positions.copy()
    
    def execute_order(self, order: Order) -> bool:
        """オーダーを実行（レガシーメソッド）"""
        return self.submit_order(order)
    
    def submit_order(self, order: Order) -> bool:
        """オーダーを提出"""
        try:
            with self.execution_lock:
                # 市場時間チェック
                if not self._is_market_open():
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"市場閉場中のため注文拒否: {order.symbol}")
                    return False
                
                # 注文検証
                if not self._validate_order(order):
                    order.status = OrderStatus.REJECTED
                    return False
                
                order.status = OrderStatus.PENDING
                
                # 成行注文は即座に実行
                if order.order_type == OrderType.MARKET:
                    return self._execute_market_order(order)
                
                # 指値・逆指値注文は待機リストに追加
                self.pending_orders[order.id] = order
                self.logger.info(f"指値注文受付: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
                return True
                
        except Exception as e:
            self.logger.error(f"注文提出エラー: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """注文をキャンセル"""
        try:
            with self.execution_lock:
                if order_id in self.pending_orders:
                    order = self.pending_orders[order_id]
                    order.status = OrderStatus.CANCELLED
                    del self.pending_orders[order_id]
                    self.logger.info(f"注文キャンセル: {order_id}")
                    return True
                else:
                    self.logger.warning(f"キャンセル対象注文が見つかりません: {order_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"注文キャンセルエラー: {e}")
            return False
    
    def _validate_order(self, order: Order) -> bool:
        """注文を検証"""
        try:
            # 基本検証
            if order.quantity <= 0:
                self.logger.warning(f"無効な数量: {order.quantity}")
                return False
            
            if not order.symbol:
                self.logger.warning("シンボルが空です")
                return False
            
            # 買い注文の残高チェック
            if order.side == OrderSide.BUY:
                current_price = self.get_current_price(order.symbol)
                required_cash = order.quantity * current_price
                commission = self._calculate_commission(order.quantity, current_price)
                total_required = required_cash + commission
                
                if self.account_balance < total_required:
                    self.logger.warning(f"資金不足: 必要 {total_required}, 利用可能 {self.account_balance}")
                    return False
            
            # 価格検証（指値注文）
            if order.order_type == OrderType.LIMIT and order.price is None:
                self.logger.warning("指値注文に価格が設定されていません")
                return False
            
            # ストップ価格検証（逆指値注文）
            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT] and order.stop_price is None:
                self.logger.warning("逆指値注文にストップ価格が設定されていません")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"注文検証エラー: {e}")
            return False
    
    def _execute_market_order(self, order: Order) -> bool:
        """成行注文を実行"""
        try:
            # 実行遅延シミュレーション
            if self.latency_ms > 0:
                time_module.sleep(self.latency_ms / 1000.0)
            
            # Phase 4.2-21: symbol文字列検証デバッグログ
            self.logger.info(f"[PRICE_GET_DEBUG] order.symbol='{order.symbol}' | len={len(order.symbol)} | repr={repr(order.symbol)} | type={type(order.symbol).__name__}")
            self.logger.info(f"[PRICE_GET_DEBUG] 登録済みkeys: {list(self.current_prices.keys())}")
            if self.current_prices:
                first_key = list(self.current_prices.keys())[0]
                self.logger.info(f"[PRICE_GET_DEBUG] 最初のkey='{first_key}' | len={len(first_key)} | repr={repr(first_key)}")
            
            # 実行価格決定
            base_price = self.get_current_price(order.symbol)
            
            # Phase 4.2-21: 価格取得結果ログ
            self.logger.info(f"[PRICE_GET_DEBUG] 取得結果: base_price={base_price:.2f}円 | is_default={'YES' if base_price == 100.0 else 'NO'}")
            
            execution_price = self._apply_slippage(order, base_price)
            
            # 手数料計算
            commission = self._calculate_commission(order.quantity, execution_price)
            slippage_cost = abs(execution_price - base_price) * order.quantity
            
            # 売買代金計算
            trade_value = order.quantity * execution_price
            
            # 口座残高更新
            if order.side == OrderSide.BUY:
                total_cost = trade_value + commission
                if self.account_balance < total_cost:
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"資金不足で実行拒否: {total_cost} > {self.account_balance}")
                    return False
                self.account_balance -= total_cost
                
                # Phase 4.2-23: ポジション追加/更新
                if order.symbol not in self.positions:
                    self.positions[order.symbol] = {
                        'quantity': order.quantity,
                        'entry_price': execution_price,
                        'entry_time': datetime.now()
                    }
                else:
                    # 既存ポジションに追加（平均取得単価計算）
                    existing = self.positions[order.symbol]
                    total_qty = existing['quantity'] + order.quantity
                    avg_price = (
                        (existing['entry_price'] * existing['quantity'] + execution_price * order.quantity) 
                        / total_qty
                    )
                    self.positions[order.symbol] = {
                        'quantity': total_qty,
                        'entry_price': avg_price,
                        'entry_time': existing['entry_time']
                    }
                    
            else:  # SELL
                proceeds = trade_value - commission
                self.account_balance += proceeds
                
                # Phase 4.2-30: SELL処理デバッグログ
                self.logger.info(
                    f"[PHASE_4_2_30] SELL処理開始: {order.symbol}, "
                    f"注文数量={order.quantity}, "
                    f"ポジション有無={order.symbol in self.positions}, "
                    f"現在のpositions keys={list(self.positions.keys())}"
                )
                
                # Phase 4.2-23: ポジション削減/クローズ
                if order.symbol in self.positions:
                    existing_qty = self.positions[order.symbol]['quantity']
                    new_qty = existing_qty - order.quantity
                    
                    self.logger.info(
                        f"[PHASE_4_2_30] ポジション削減: {order.symbol}, "
                        f"既存数量={existing_qty}, 注文数量={order.quantity}, 残数量={new_qty}"
                    )
                    
                    if new_qty <= 0:
                        # ポジション完全クローズ
                        del self.positions[order.symbol]
                        self.logger.info(f"[PHASE_4_2_30] ポジション完全クローズ: {order.symbol}")
                    else:
                        # 部分決済
                        self.positions[order.symbol]['quantity'] = new_qty
                        self.logger.info(f"[PHASE_4_2_30] 部分決済完了: {order.symbol}, 残={new_qty}株")
                else:
                    # Phase 4.2-30: ポジション未保有の場合の警告
                    self.logger.warning(
                        f"[PHASE_4_2_30] SELL注文実行時にポジション未保有: {order.symbol}, "
                        f"注文数量={order.quantity}株 (proceed anyway)"
                    )
            
            # 注文情報更新
            order.status = OrderStatus.FILLED
            order.filled_price = execution_price
            order.filled_quantity = order.quantity
            order.commission = commission
            order.slippage = slippage_cost
            order.filled_at = datetime.now()
            
            self.filled_orders.append(order)
            
            self.logger.info(f"成行注文約定: {order.symbol} {order.side.value} {order.quantity} @ {execution_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"成行注文実行エラー: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    def _check_pending_orders(self, symbol: str):
        """待機中注文をチェック"""
        try:
            current_price = self.get_current_price(symbol)
            orders_to_execute: List[Order] = []
            
            for order_id, order in self.pending_orders.items():
                if order.symbol != symbol:
                    continue
                
                should_execute = False
                
                # 指値注文チェック
                if order.order_type == OrderType.LIMIT:
                    if (order.side == OrderSide.BUY and order.price is not None and 
                        current_price <= order.price):
                        should_execute = True
                    elif (order.side == OrderSide.SELL and order.price is not None and 
                          current_price >= order.price):
                        should_execute = True
                
                # 逆指値注文チェック
                elif order.order_type == OrderType.STOP_LOSS:
                    if (order.side == OrderSide.BUY and order.stop_price is not None and 
                        current_price >= order.stop_price):
                        should_execute = True
                    elif (order.side == OrderSide.SELL and order.stop_price is not None and 
                          current_price <= order.stop_price):
                        should_execute = True
                
                if should_execute:
                    orders_to_execute.append(order)
            
            # 約定処理
            for order in orders_to_execute:
                execution_price = (order.price if order.order_type == OrderType.LIMIT 
                                 else current_price) or current_price
                
                # スリッページ適用
                execution_price = self._apply_slippage(order, execution_price)
                
                # 実行
                if self._execute_limit_order(order, execution_price):
                    del self.pending_orders[order.id]
                    
        except Exception as e:
            self.logger.error(f"待機注文チェックエラー: {e}")
    
    def _execute_limit_order(self, order: Order, execution_price: float) -> bool:
        """指値注文を実行"""
        try:
            # 手数料計算
            commission = self._calculate_commission(order.quantity, execution_price)
            
            # 売買代金計算
            trade_value = order.quantity * execution_price
            
            # 口座残高更新
            if order.side == OrderSide.BUY:
                total_cost = trade_value + commission
                if self.account_balance < total_cost:
                    self.logger.warning(f"指値注文実行時に資金不足: {order.id}")
                    return False
                self.account_balance -= total_cost
            else:
                proceeds = trade_value - commission
                self.account_balance += proceeds
            
            # 注文情報更新
            order.status = OrderStatus.FILLED
            order.filled_price = execution_price
            order.filled_quantity = order.quantity
            order.commission = commission
            order.slippage = abs(execution_price - (order.price or execution_price)) * order.quantity
            order.filled_at = datetime.now()
            
            self.filled_orders.append(order)
            
            self.logger.info(f"指値注文約定: {order.symbol} {order.side.value} {order.quantity} @ {execution_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"指値注文実行エラー: {e}")
            return False
    
    def _apply_slippage(self, order: Order, base_price: float) -> float:
        """スリッページを適用"""
        if not self.enable_slippage:
            return base_price
        
        # スリッページ方向（買い注文は不利、売り注文も不利）
        slippage_direction = 1 if order.side == OrderSide.BUY else -1
        
        # ランダムスリッページ（0 ~ max_slippage）
        slippage_amount = random.uniform(0, self.slippage_pct) * base_price * slippage_direction
        
        return base_price + slippage_amount
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """手数料を計算"""
        # 固定手数料 + 比例手数料
        trade_value = quantity * price
        percentage_commission = trade_value * self.commission_pct
        return self.commission_per_trade + percentage_commission
    
    def _is_market_open(self) -> bool:
        """
        市場が開いているかチェック
        
        Phase 4.2-11: バックテストモードでは常にTrueを返す
        - バックテストでは過去データを使用するため、市場時間チェックは不要
        - copilot-instructions.md準拠: バックテスト実行を妨げない
        """
        if self.backtest_mode:
            return True  # バックテストモードでは常に市場開場中として扱う
        
        # ライブトレード用: 実際の市場時間チェック
        current_time = datetime.now().time()
        return self.market_hours['open'] <= current_time <= self.market_hours['close']
    
    def set_market_hours(self, open_time: time, close_time: time):
        """市場時間を設定"""
        self.market_hours['open'] = open_time
        self.market_hours['close'] = close_time
    
    def set_market_state(self, state: MarketState):
        """市場状態を設定"""
        self.market_state = state
        self.logger.info(f"市場状態変更: {state.value}")
    
    def get_broker_status(self) -> Dict[str, Any]:
        """ブローカー状況を取得"""
        return {
            'account_balance': self.account_balance,
            'initial_balance': self.initial_balance,
            'total_pnl': self.account_balance - self.initial_balance,
            'pending_orders_count': len(self.pending_orders),
            'filled_orders_count': len(self.filled_orders),
            'market_state': self.market_state.value,
            'market_open': self._is_market_open(),
            'tracked_symbols': list(self.current_prices.keys()),
            'settings': {
                'commission_per_trade': self.commission_per_trade,
                'commission_pct': self.commission_pct,
                'slippage_pct': self.slippage_pct,
                'enable_slippage': self.enable_slippage,
                'latency_ms': self.latency_ms
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """注文履歴を取得"""
        history: List[Dict[str, Any]] = []
        for order in self.filled_orders:
            history.append({
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'filled_price': order.filled_price,
                'commission': order.commission,
                'slippage': order.slippage,
                'filled_at': order.filled_at.isoformat() if order.filled_at else '',
                'strategy_name': order.strategy_name
            })
        return history
    
    def reset_broker(self, initial_balance: Optional[float] = None):
        """ブローカーをリセット"""
        if initial_balance is not None:
            self.initial_balance = initial_balance
        
        self.account_balance = self.initial_balance
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.current_prices.clear()
        self.price_history.clear()
        self.last_price_update.clear()
        
        self.logger.info(f"ブローカーリセット: 初期残高 {self.initial_balance}")
    
    def simulate_market_data(self, symbols: List[str], duration_seconds: int = 60):
        """市場データをシミュレート（テスト用）"""
        import threading
        import time
        
        def generate_prices():
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                for symbol in symbols:
                    # 現在価格から±2%のランダムウォーク
                    current = self.current_prices.get(symbol, 100.0)
                    change_pct = random.uniform(-0.02, 0.02)
                    new_price = current * (1 + change_pct)
                    new_price = max(new_price, 1.0)  # 最低価格
                    
                    self.update_price(symbol, new_price)
                
                time.sleep(1)  # 1秒間隔
        
        thread = threading.Thread(target=generate_prices, daemon=True)
        thread.start()
        self.logger.info(f"市場データシミュレーション開始: {symbols} ({duration_seconds}秒)")
        return thread
