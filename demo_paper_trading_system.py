#!/usr/bin/env python3
"""
フェーズ3B: ペーパートレード環境デモスクリプト
Paper Trading Environment Demo Script

このスクリプトは、新しく実装されたペーパートレード環境の機能をデモンストレーションします。

実行例:
python demo_paper_trading_system.py
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
import logging

# パスを追加して src モジュールをインポート可能にする
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from execution.order_manager import Order, OrderType, OrderSide, OrderStatus, OrderManager
    from execution.portfolio_tracker import PortfolioTracker, Position
    from execution.trade_executor import TradeExecutor, ExecutionMode
    from execution.paper_broker import PaperBroker
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("src/execution/ モジュールが見つかりません。パスを確認してください。")
    sys.exit(1)

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('demo_paper_trading.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """設定ファイルを読み込み"""
    try:
        config_path = os.path.join('config', 'paper_trading', 'paper_trading_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("設定ファイルが見つかりません。デフォルト設定を使用します。")
        return {
            "broker": {
                "initial_balance": 100000.0,
                "commission_per_trade": 1.0,
                "commission_pct": 0.001,
                "slippage_pct": 0.0001
            },
            "portfolio": {
                "initial_cash": 100000.0
            }
        }

def demo_basic_functionality(executor, broker, portfolio, logger):
    """基本機能のデモ"""
    logger.info("=== 基本機能デモ開始 ===")
    
    # テスト用価格設定
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    test_prices = {
        "AAPL": 150.0,
        "MSFT": 300.0,
        "GOOGL": 2500.0,
        "TSLA": 200.0
    }
    
    logger.info("1. 市場価格設定")
    for symbol, price in test_prices.items():
        broker.update_price(symbol, price)
        logger.info(f"  {symbol}: ${price}")
    
    # 成行注文のテスト
    logger.info("\\n2. 成行注文実行テスト")
    
    # AAPL 買い注文
    buy_order = executor.create_market_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        strategy_name="DEMO_BUY"
    )
    
    if buy_order:
        order_id = executor.submit_order(buy_order)
        logger.info(f"  AAPL買い注文提出: ID={order_id}")
        time.sleep(0.1)  # 処理待ち
        
        # 約定確認
        filled_order = executor.order_manager.get_order(order_id)
        if filled_order and filled_order.status == OrderStatus.FILLED:
            logger.info(f"  ✅ 約定: {filled_order.filled_quantity} 株 @ ${filled_order.filled_price}")
        else:
            logger.warning(f"  ❌ 約定失敗: {filled_order.status if filled_order else 'None'}")
    
    # ポジション確認
    logger.info("\\n3. ポジション状況")
    positions = portfolio.get_all_positions()
    for symbol, position in positions.items():
        logger.info(f"  {symbol}: {position.quantity} 株, 平均価格: ${position.avg_price:.2f}")
    
    # 口座状況
    logger.info("\\n4. 口座状況")
    logger.info(f"  現金残高: ${portfolio.get_cash_balance():,.2f}")
    logger.info(f"  総資産: ${portfolio.get_total_equity():,.2f}")
    
    return True

def demo_limit_orders(executor, broker, portfolio, logger):
    """指値注文のデモ"""
    logger.info("\\n=== 指値注文デモ開始 ===")
    
    # 現在価格より低い指値買い注文（待機状態になる）
    logger.info("1. 指値買い注文（現在価格より低い価格）")
    
    current_price = broker.get_current_price("MSFT")
    limit_price = current_price - 10.0  # 現在価格より10ドル低い
    
    limit_order = executor.create_limit_order(
        symbol="MSFT",
        side=OrderSide.BUY,
        quantity=50.0,
        price=limit_price,
        strategy_name="DEMO_LIMIT"
    )
    
    if limit_order:
        order_id = executor.submit_order(limit_order)
        logger.info(f"  MSFT指値買い注文提出: {limit_price} <= {current_price} (待機中)")
        
        # 待機注文確認
        time.sleep(0.1)
        order_status = executor.order_manager.get_order(order_id)
        logger.info(f"  注文状況: {order_status.status if order_status else 'None'}")
        
        # 価格を下げて約定させる
        logger.info("\\n2. 価格変動による約定テスト")
        new_price = limit_price - 1.0
        broker.update_price("MSFT", new_price)
        logger.info(f"  価格更新: MSFT ${current_price} -> ${new_price}")
        
        time.sleep(0.2)  # 約定処理待ち
        
        # 約定確認
        final_order = executor.order_manager.get_order(order_id)
        if final_order and final_order.status == OrderStatus.FILLED:
            logger.info(f"  ✅ 約定: {final_order.filled_quantity} 株 @ ${final_order.filled_price}")
        else:
            logger.info(f"  待機中: {final_order.status if final_order else 'None'}")

def demo_risk_management(executor, broker, portfolio, logger):
    """リスク管理機能のデモ"""
    logger.info("\\n=== リスク管理デモ開始 ===")
    
    # 大量注文でリスク管理をテスト
    logger.info("1. 大量注文によるリスク管理テスト")
    
    large_order = executor.create_market_order(
        symbol="GOOGL",
        side=OrderSide.BUY,
        quantity=1000.0,  # 大量注文
        strategy_name="RISK_TEST"
    )
    
    if large_order:
        # 最大ポジションサイズ制限を設定
        executor.max_position_size_pct = 10.0  # 10%制限
        
        order_id = executor.submit_order(large_order)
        if order_id:
            logger.info("  大量注文が承認されました")
        else:
            logger.info("  ✅ リスク管理により大量注文が拒否されました")
    
    # 資金不足テスト
    logger.info("\\n2. 資金不足テスト")
    
    # 現在の残高を確認
    current_balance = portfolio.get_cash_balance()
    required_amount = current_balance + 10000.0  # 残高以上の金額
    
    expensive_order = executor.create_market_order(
        symbol="GOOGL",
        side=OrderSide.BUY,
        quantity=10.0,  # GOOGLは高額なので資金不足になる可能性
        strategy_name="INSUFFICIENT_FUNDS_TEST"
    )
    
    if expensive_order:
        order_id = executor.submit_order(expensive_order)
        if order_id:
            order_result = executor.order_manager.get_order(order_id)
            if order_result and order_result.status == OrderStatus.REJECTED:
                logger.info("  ✅ 資金不足により注文が拒否されました")
            else:
                logger.info("  注文が受け入れられました")

def demo_portfolio_analytics(executor, broker, portfolio, logger):
    """ポートフォリオ分析のデモ"""
    logger.info("\\n=== ポートフォリオ分析デモ開始 ===")
    
    # 価格を更新してPnLを発生させる
    logger.info("1. 価格変動によるPnL変化")
    
    price_changes = {
        "AAPL": 160.0,  # 10ドル上昇
        "MSFT": 290.0,  # 10ドル下落
    }
    
    for symbol, new_price in price_changes.items():
        old_price = broker.get_current_price(symbol)
        broker.update_price(symbol, new_price)
        change_pct = ((new_price - old_price) / old_price) * 100
        logger.info(f"  {symbol}: ${old_price} -> ${new_price} ({change_pct:+.1f}%)")
    
    # ポジション価格を更新
    portfolio.update_position_prices(price_changes)
    
    # パフォーマンス指標計算
    logger.info("\\n2. パフォーマンス指標")
    metrics = portfolio.calculate_performance_metrics()
    
    logger.info(f"  総リターン: {metrics.total_return:.2f}%")
    logger.info(f"  実現PnL: ${metrics.realized_pnl:.2f}")
    logger.info(f"  未実現PnL: ${metrics.unrealized_pnl:.2f}")
    logger.info(f"  総手数料: ${metrics.total_commission:.2f}")
    logger.info(f"  総取引数: {metrics.total_trades}")
    
    # ポートフォリオサマリー
    logger.info("\\n3. ポートフォリオサマリー")
    summary = portfolio.get_portfolio_summary()
    
    logger.info(f"  現金: ${summary['cash']:,.2f}")
    logger.info(f"  総資産: ${summary['total_equity']:,.2f}")
    logger.info(f"  ポジション数: {summary['positions_count']}")
    
    for symbol, pos_info in summary['positions'].items():
        market_value = pos_info['quantity'] * pos_info['current_price']
        logger.info(f"    {symbol}: {pos_info['quantity']} 株, 評価額: ${market_value:,.2f}")

def demo_emergency_functions(executor, broker, portfolio, logger):
    """緊急機能のデモ"""
    logger.info("\\n=== 緊急機能デモ開始 ===")
    
    # 緊急停止機能
    logger.info("1. 緊急停止機能テスト")
    
    # いくつかの指値注文を出して待機状態にする
    symbols = ["AAPL", "MSFT"]
    order_ids = []
    
    for symbol in symbols:
        current_price = broker.get_current_price(symbol)
        limit_order = executor.create_limit_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=10.0,
            price=current_price + 50.0,  # 高い価格で売り注文
            strategy_name="EMERGENCY_TEST"
        )
        
        if limit_order:
            order_id = executor.submit_order(limit_order)
            if order_id:
                order_ids.append(order_id)
    
    logger.info(f"  待機注文数: {len(order_ids)}")
    logger.info(f"  ブローカー待機注文: {len(broker.pending_orders)}")
    
    # 緊急停止実行
    logger.info("\\n2. 緊急停止実行")
    executor.emergency_stop()
    
    logger.info(f"  取引実行状態: {'無効' if not executor.execution_enabled else '有効'}")
    logger.info(f"  残存待機注文: {len(broker.pending_orders)}")
    
    # 全ポジション決済テスト
    logger.info("\\n3. 全ポジション決済テスト")
    
    # 取引を再開
    executor.enable_execution(True)
    
    positions_before = len(portfolio.get_all_positions())
    logger.info(f"  決済前ポジション数: {positions_before}")
    
    if positions_before > 0:
        close_orders = executor.close_all_positions(emergency=True)
        logger.info(f"  決済注文数: {len(close_orders)}")
        
        time.sleep(0.2)  # 決済処理待ち
        
        positions_after = len(portfolio.get_all_positions())
        logger.info(f"  決済後ポジション数: {positions_after}")

def main():
    """メイン実行関数"""
    print("フェーズ3B: ペーパートレード環境デモ")
    print("=" * 50)
    
    # ログ設定
    logger = setup_logging()
    logger.info("ペーパートレードデモ開始")
    
    try:
        # 設定読み込み
        config = load_config()
        logger.info("設定ファイル読み込み完了")
        
        # システム初期化
        logger.info("\\n=== システム初期化 ===")
        
        # ブローカー初期化
        broker = PaperBroker(
            initial_balance=config['broker']['initial_balance'],
            commission_per_trade=config['broker']['commission_per_trade'],
            commission_pct=config['broker']['commission_pct'],
            slippage_pct=config['broker']['slippage_pct']
        )
        logger.info(f"ペーパーブローカー初期化: 初期残高 ${broker.initial_balance:,.2f}")
        
        # ポートフォリオトラッカー初期化
        portfolio = PortfolioTracker(
            initial_cash=config['portfolio']['initial_cash']
        )
        logger.info(f"ポートフォリオトラッカー初期化: 初期資金 ${portfolio.initial_cash:,.2f}")
        
        # トレード実行エンジン初期化
        executor = TradeExecutor(
            portfolio_tracker=portfolio,
            broker=broker,
            mode=ExecutionMode.PAPER_TRADING
        )
        logger.info("トレード実行エンジン初期化完了")
        
        # デモ実行
        print("\\n" + "=" * 50)
        print("デモ実行開始")
        print("=" * 50)
        
        # 1. 基本機能デモ
        demo_basic_functionality(executor, broker, portfolio, logger)
        
        # 2. 指値注文デモ
        demo_limit_orders(executor, broker, portfolio, logger)
        
        # 3. リスク管理デモ
        demo_risk_management(executor, broker, portfolio, logger)
        
        # 4. ポートフォリオ分析デモ
        demo_portfolio_analytics(executor, broker, portfolio, logger)
        
        # 5. 緊急機能デモ
        demo_emergency_functions(executor, broker, portfolio, logger)
        
        # 最終状況レポート
        logger.info("\\n=== 最終状況レポート ===")
        
        broker_status = broker.get_broker_status()
        portfolio_summary = portfolio.get_portfolio_summary()
        execution_status = executor.get_execution_status()
        
        logger.info("ブローカー状況:")
        logger.info(f"  口座残高: ${broker_status['account_balance']:,.2f}")
        logger.info(f"  総PnL: ${broker_status['total_pnl']:,.2f}")
        logger.info(f"  約定注文数: {broker_status['filled_orders_count']}")
        
        logger.info("\\nポートフォリオ状況:")
        logger.info(f"  総資産: ${portfolio_summary['total_equity']:,.2f}")
        logger.info(f"  ポジション数: {portfolio_summary['positions_count']}")
        logger.info(f"  総取引数: {portfolio_summary['total_trades']}")
        
        print("\\n" + "=" * 50)
        print("デモ実行完了")
        print("=" * 50)
        print(f"ログファイル: demo_paper_trading.log")
        print(f"実行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info("ペーパートレードデモ完了")
        
    except Exception as e:
        logger.error(f"デモ実行中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
