from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from enum import Enum

# Position クラス
@dataclass
class Position:
    """ポジション情報"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_price(self, price: float):
        """現在価格を更新"""
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity
        self.last_updated = datetime.now()
    
    def add_trade(self, quantity: float, price: float, commission: float = 0.0, slippage: float = 0.0):
        """取引を追加してポジションを更新"""
        # 手数料・スリッページを累積
        self.total_commission += commission
        self.total_slippage += slippage
        
        if self.quantity == 0:
            # 新規ポジション
            self.quantity = quantity
            self.avg_price = price
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # ポジション増加
            total_cost = self.quantity * self.avg_price + quantity * price
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity if self.quantity != 0 else 0
        else:
            # ポジション減少または反転
            if abs(quantity) <= abs(self.quantity):
                # 部分決済
                self.realized_pnl += (price - self.avg_price) * abs(quantity) * (1 if self.quantity > 0 else -1)
                self.quantity += quantity
            else:
                # 完全決済 + 反転
                self.realized_pnl += (price - self.avg_price) * abs(self.quantity) * (1 if self.quantity > 0 else -1)
                remaining = quantity + (-self.quantity)  # 反転分
                self.quantity = remaining
                self.avg_price = price if remaining != 0 else 0
        
        self.update_price(price)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で返す"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'market_value': self.quantity * self.current_price,
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    total_return: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

class PortfolioTracker:
    """ポートフォリオ追跡クラス"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # パフォーマンス追跡
        self.peak_equity = initial_cash
        self.drawdown_history: List[float] = []
        self.daily_returns: List[float] = []
        
    def get_cash_balance(self) -> float:
        """現金残高を取得"""
        return self.cash
    
    def get_total_equity(self) -> float:
        """総資産を計算"""
        market_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        return self.cash + market_value
    
    def update_position_prices(self, prices: Dict[str, float]):
        """ポジションの現在価格を更新"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
    
    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     commission: float = 0.0, slippage: float = 0.0, 
                     strategy_name: str = "", trade_id: str = ""):
        """取引を実行"""
        try:
            # 手数料・スリッページを含む取引コスト
            trade_cost = abs(quantity) * price + commission + abs(quantity * slippage)
            
            if quantity > 0:  # 買い注文
                if self.cash < trade_cost:
                    self.logger.warning(f"資金不足: 必要額 {trade_cost}, 利用可能 {self.cash}")
                    return False
                self.cash -= trade_cost
            else:  # 売り注文
                self.cash += abs(quantity) * price - commission - abs(quantity * slippage)
            
            # ポジション更新
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol, quantity=0, avg_price=0)
            
            self.positions[symbol].add_trade(quantity, price, commission, slippage)
            
            # ポジションが0になった場合は削除
            if abs(self.positions[symbol].quantity) < 1e-8:
                del self.positions[symbol]
            
            # 取引履歴に記録
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'trade_id': trade_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'slippage': slippage,
                'strategy_name': strategy_name,
                'cash_after': self.cash,
                'total_equity': self.get_total_equity()
            }
            self.trade_history.append(trade_record)
            
            # エクイティカーブ更新
            self._update_equity_curve()
            
            self.logger.info(f"取引実行: {symbol} {quantity} @ {price}")
            return True
            
        except Exception as e:
            self.logger.error(f"取引実行エラー: {e}")
            return False
    
    def _update_equity_curve(self):
        """エクイティカーブを更新"""
        current_equity = self.get_total_equity()
        current_time = datetime.now()
        
        equity_point = {
            'timestamp': current_time.isoformat(),
            'equity': current_equity,
            'cash': self.cash,
            'positions_value': current_equity - self.cash,
            'total_return': (current_equity - self.initial_cash) / self.initial_cash * 100
        }
        self.equity_curve.append(equity_point)
        
        # ドローダウン計算
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
        self.drawdown_history.append(drawdown)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """特定銘柄のポジションを取得"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """全ポジションを取得"""
        return self.positions.copy()
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """パフォーマンス指標を計算"""
        try:
            metrics = PerformanceMetrics()
            
            if not self.trade_history:
                return metrics
            
            # 基本指標
            current_equity = self.get_total_equity()
            metrics.total_return = (current_equity - self.initial_cash) / self.initial_cash * 100
            metrics.total_pnl = current_equity - self.initial_cash
            
            # ポジション別PnL
            for pos in self.positions.values():
                metrics.realized_pnl += pos.realized_pnl
                metrics.unrealized_pnl += pos.unrealized_pnl
                metrics.total_commission += pos.total_commission
                metrics.total_slippage += pos.total_slippage
            
            # 取引統計
            winning_trades: List[float] = []
            losing_trades: List[float] = []
            
            for trade in self.trade_history:
                if 'pnl' in trade and trade['pnl'] != 0:
                    if trade['pnl'] > 0:
                        winning_trades.append(trade['pnl'])
                    else:
                        losing_trades.append(abs(trade['pnl']))
            
            metrics.total_trades = len(winning_trades) + len(losing_trades)
            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades * 100
                
                if winning_trades:
                    metrics.avg_win = sum(winning_trades) / len(winning_trades)
                    metrics.largest_win = max(winning_trades)
                
                if losing_trades:
                    metrics.avg_loss = sum(losing_trades) / len(losing_trades)
                    metrics.largest_loss = max(losing_trades)
                    
                    # プロフィットファクター
                    if sum(losing_trades) > 0:
                        metrics.profit_factor = sum(winning_trades) / sum(losing_trades)
            
            # ドローダウン
            if self.drawdown_history:
                metrics.max_drawdown = max(self.drawdown_history)
            
            # シャープレシオ（簡易版）
            if len(self.equity_curve) > 1:
                returns: List[float] = []
                for i in range(1, len(self.equity_curve)):
                    prev_equity = self.equity_curve[i-1]['equity']
                    curr_equity = self.equity_curve[i]['equity']
                    daily_return = (curr_equity - prev_equity) / prev_equity
                    returns.append(daily_return)
                
                if returns and len(returns) > 1:
                    import statistics
                    avg_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns)
                    if std_return > 0:
                        metrics.sharpe_ratio = avg_return / std_return * (252 ** 0.5)  # 年換算
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"パフォーマンス計算エラー: {e}")
            return PerformanceMetrics()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """ポートフォリオサマリーを取得"""
        try:
            metrics = self.calculate_performance_metrics()
            
            summary = {
                'cash': self.cash,
                'total_equity': self.get_total_equity(),
                'positions_count': len(self.positions),
                'total_trades': len(self.trade_history),
                'performance': {
                    'total_return_pct': metrics.total_return,
                    'total_pnl': metrics.total_pnl,
                    'realized_pnl': metrics.realized_pnl,
                    'unrealized_pnl': metrics.unrealized_pnl,
                    'max_drawdown_pct': metrics.max_drawdown,
                    'win_rate_pct': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'sharpe_ratio': metrics.sharpe_ratio
                },
                'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"サマリー生成エラー: {e}")
            return {}
    
    def reset_portfolio(self, initial_cash: float = None):
        """ポートフォリオをリセット"""
        if initial_cash is not None:
            self.initial_cash = initial_cash
            
        self.cash = self.initial_cash
        self.positions.clear()
        self.trade_history.clear()
        self.equity_curve.clear()
        self.peak_equity = self.initial_cash
        self.drawdown_history.clear()
        self.daily_returns.clear()
        
        self.logger.info(f"ポートフォリオリセット: 初期資金 {self.initial_cash}")
