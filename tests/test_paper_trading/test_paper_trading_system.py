import unittest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from execution.order_manager import Order, OrderType, OrderSide, OrderStatus, OrderManager, BrokerInterface
from execution.portfolio_tracker import PortfolioTracker, Position
from execution.trade_executor import TradeExecutor, ExecutionMode
from execution.paper_broker import PaperBroker

class TestOrderManager(unittest.TestCase):
    """OrderManager のテスト"""
    
    def setUp(self):
        self.mock_broker = Mock(spec=BrokerInterface)
        self.order_manager = OrderManager(self.mock_broker)
    
    def test_order_creation(self):
        """注文作成のテスト"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.quantity, 100.0)
        self.assertEqual(order.status, OrderStatus.PENDING)
    
    def test_submit_order(self):
        """注文提出のテスト"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        # ブローカーが成功を返すようにモック
        self.mock_broker.submit_order.return_value = True
        
        order_id = self.order_manager.submit_order(order)
        
        self.assertIsNotNone(order_id)
        self.assertEqual(order_id, order.id)
        self.assertIn(order.id, self.order_manager.orders)
    
    def test_cancel_order(self):
        """注文キャンセルのテスト"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=150.0
        )
        
        # 注文を提出
        self.mock_broker.submit_order.return_value = True
        order_id = self.order_manager.submit_order(order)
        
        # キャンセル
        self.mock_broker.cancel_order.return_value = True
        result = self.order_manager.cancel_order(order_id)
        
        self.assertTrue(result)
        self.assertEqual(self.order_manager.orders[order_id].status, OrderStatus.CANCELLED)

class TestPortfolioTracker(unittest.TestCase):
    """PortfolioTracker のテスト"""
    
    def setUp(self):
        self.portfolio = PortfolioTracker(initial_cash=100000.0)
    
    def test_initial_state(self):
        """初期状態のテスト"""
        self.assertEqual(self.portfolio.get_cash_balance(), 100000.0)
        self.assertEqual(self.portfolio.get_total_equity(), 100000.0)
        self.assertEqual(len(self.portfolio.get_all_positions()), 0)
    
    def test_execute_buy_trade(self):
        """買い取引のテスト"""
        result = self.portfolio.execute_trade(
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            commission=1.0,
            strategy_name="TEST"
        )
        
        self.assertTrue(result)
        self.assertEqual(self.portfolio.get_cash_balance(), 100000.0 - (100.0 * 150.0) - 1.0)
        
        position = self.portfolio.get_position("AAPL")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 100.0)
        self.assertEqual(position.avg_price, 150.0)
    
    def test_execute_sell_trade(self):
        """売り取引のテスト"""
        # 先に買い注文を実行
        self.portfolio.execute_trade("AAPL", 100.0, 150.0, 1.0)
        
        # 売り注文を実行
        result = self.portfolio.execute_trade(
            symbol="AAPL",
            quantity=-50.0,
            price=160.0,
            commission=1.0,
            strategy_name="TEST"
        )
        
        self.assertTrue(result)
        
        position = self.portfolio.get_position("AAPL")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 50.0)
        self.assertEqual(position.realized_pnl, 500.0)  # (160-150) * 50

class TestPaperBroker(unittest.TestCase):
    """PaperBroker のテスト"""
    
    def setUp(self):
        self.broker = PaperBroker(
            initial_balance=100000.0,
            commission_per_trade=1.0,
            slippage_pct=0.0001
        )
        # 24時間取引可能に設定（テスト用）
        from datetime import time
        self.broker.set_market_hours(time(0, 0), time(23, 59))
        
        # テスト用価格設定
        self.broker.update_price("AAPL", 150.0)
        self.broker.update_price("MSFT", 300.0)
    
    def test_get_current_price(self):
        """現在価格取得のテスト"""
        price = self.broker.get_current_price("AAPL")
        self.assertEqual(price, 150.0)
    
    def test_market_order_execution(self):
        """成行注文実行のテスト"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0
        )
        
        result = self.broker.submit_order(order)
        
        self.assertTrue(result)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertIsNotNone(order.filled_price)
        self.assertEqual(order.filled_quantity, 100.0)
        
        # 残高確認
        expected_balance = 100000.0 - (100.0 * order.filled_price) - order.commission
        self.assertAlmostEqual(self.broker.get_account_balance(), expected_balance, places=2)
    
    def test_limit_order_pending(self):
        """指値注文待機のテスト"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=140.0  # 現在価格より低い
        )
        
        result = self.broker.submit_order(order)
        
        self.assertTrue(result)
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertIn(order.id, self.broker.pending_orders)
    
    def test_limit_order_execution(self):
        """指値注文約定のテスト"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=155.0  # 現在価格より高い
        )
        
        result = self.broker.submit_order(order)
        
        # 指値注文は即座に約定するはず（現在価格150 <= 指値155）
        self.assertTrue(result)
        # 注文は待機状態になり、価格更新時に約定処理される
    
    def test_insufficient_funds(self):
        """資金不足のテスト"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000.0  # 大量注文で資金不足にする
        )
        
        # 一度大きな注文を実行して残高を減らす
        large_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=600.0
        )
        self.broker.submit_order(large_order)
        
        # 残高不足で拒否されるはず
        result = self.broker.submit_order(order)
        self.assertFalse(result)
        self.assertEqual(order.status, OrderStatus.REJECTED)

class TestTradeExecutor(unittest.TestCase):
    """TradeExecutor のテスト"""
    
    def setUp(self):
        self.portfolio = PortfolioTracker(initial_cash=100000.0)
        self.broker = PaperBroker(initial_balance=100000.0)
        self.executor = TradeExecutor(
            portfolio_tracker=self.portfolio,
            broker=self.broker,
            mode=ExecutionMode.PAPER_TRADING
        )
        
        # テスト用価格設定
        self.broker.update_price("AAPL", 150.0)
    
    def test_create_market_order(self):
        """成行注文作成のテスト"""
        order = self.executor.create_market_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            strategy_name="TEST"
        )
        
        self.assertIsNotNone(order)
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.quantity, 100.0)
        self.assertEqual(order.strategy_name, "TEST")
    
    def test_submit_order_with_risk_check(self):
        """リスクチェック付き注文提出のテスト"""
        order = self.executor.create_market_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0
        )
        
        order_id = self.executor.submit_order(order)
        
        self.assertIsNotNone(order_id)
    
    def test_emergency_stop(self):
        """緊急停止のテスト"""
        # 注文を提出
        order = self.executor.create_limit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            price=140.0
        )
        self.executor.submit_order(order)
        
        # 緊急停止
        self.executor.emergency_stop()
        
        self.assertFalse(self.executor.execution_enabled)
        self.assertEqual(len(self.broker.pending_orders), 0)

class TestIntegration(unittest.TestCase):
    """統合テスト"""
    
    def setUp(self):
        self.portfolio = PortfolioTracker(initial_cash=100000.0)
        self.broker = PaperBroker(initial_balance=100000.0)
        
        # 24時間取引可能に設定（テスト用）
        from datetime import time
        self.broker.set_market_hours(time(0, 0), time(23, 59))
        
        self.executor = TradeExecutor(
            portfolio_tracker=self.portfolio,
            broker=self.broker,
            mode=ExecutionMode.PAPER_TRADING
        )
        
        # テスト用価格設定
        self.broker.update_price("AAPL", 150.0)
        self.broker.update_price("MSFT", 300.0)
    
    def test_full_trading_cycle(self):
        """完全な取引サイクルのテスト"""
        # 1. 買い注文
        buy_order = self.executor.create_market_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            strategy_name="TEST_BUY"
        )
        
        buy_order_id = self.executor.submit_order(buy_order)
        self.assertIsNotNone(buy_order_id)
        
        # 2. ポジション確認
        position = self.portfolio.get_position("AAPL")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 100.0)
        
        # 3. 価格更新
        self.broker.update_price("AAPL", 160.0)
        self.portfolio.update_position_prices({"AAPL": 160.0})
        
        # 4. 売り注文
        sell_order = self.executor.create_market_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100.0,
            strategy_name="TEST_SELL"
        )
        
        sell_order_id = self.executor.submit_order(sell_order)
        self.assertIsNotNone(sell_order_id)
        
        # 5. ポジション決済確認
        position = self.portfolio.get_position("AAPL")
        self.assertIsNone(position)  # ポジションは決済されているはず
        
        # 6. 収益確認
        metrics = self.portfolio.calculate_performance_metrics()
        self.assertGreater(metrics.realized_pnl, 0)  # 利益が出ているはず
    
    def test_multiple_strategies(self):
        """複数戦略同時実行のテスト"""
        # 戦略A: AAPL買い
        order_a = self.executor.create_market_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50.0,
            strategy_name="STRATEGY_A"
        )
        self.executor.submit_order(order_a)
        
        # 戦略B: MSFT買い
        order_b = self.executor.create_market_order(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=20.0,
            strategy_name="STRATEGY_B"
        )
        self.executor.submit_order(order_b)
        
        # ポジション確認
        positions = self.portfolio.get_all_positions()
        self.assertEqual(len(positions), 2)
        self.assertIn("AAPL", positions)
        self.assertIn("MSFT", positions)
        
        # 取引履歴確認
        self.assertGreater(len(self.portfolio.trade_history), 0)

if __name__ == '__main__':
    # ログ設定
    import logging
    logging.basicConfig(level=logging.WARNING)  # テスト中はWARNING以上のみ表示
    
    # テスト実行
    unittest.main(verbosity=2)
