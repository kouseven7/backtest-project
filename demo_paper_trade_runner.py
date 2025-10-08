"""
ペーパートレード実行システム デモ・テストスクリプト
フェーズ4A1 実装検証用
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

def demo_simple_mode():
    """シンプルモードデモ"""
    print("=== シンプルモードデモ実行 ===")
    
    try:
        # paper_trade_runner.pyをインポート
        from paper_trade_runner import PaperTradeRunner
        
        # 設定作成
        test_config = "config/paper_trading/runner_config.json"
        
        # ランナー初期化テスト
        runner = PaperTradeRunner(test_config)
        
        # 擬似args作成
        class MockArgs:
            mode = "simple"
            strategy = "VWAP_Breakout"
            interval = 15
            config = test_config
            dry_run = True
        
        args = MockArgs()
        
        # 初期化テスト
        if runner.initialize(args):
            print("[OK] 初期化成功")
            
            # 単一実行テスト（実際の実行ループは行わない）
            print("[CHART] コンポーネント状態確認...")
            
            if runner.scheduler:
                print(f"  - スケジューラー: {runner.scheduler.get_status()}")
            
            if runner.monitor:
                print(f"  - モニター: {runner.monitor.get_status()}")
            
            if runner.strategy_manager:
                print(f"  - 戦略管理: {runner.strategy_manager.get_execution_summary()}")
            
            print("[OK] シンプルモードデモ完了")
        else:
            print("[ERROR] 初期化失敗")
            
    except Exception as e:
        print(f"[ERROR] シンプルモードデモエラー: {e}")

def demo_configuration_validation():
    """設定ファイル検証デモ"""
    print("\n=== 設定ファイル検証 ===")
    
    config_files = [
        "config/paper_trading/runner_config.json",
        "config/paper_trading/paper_trading_config.json",
        "config/paper_trading/trading_rules.json",
        "config/paper_trading/market_hours.json"
    ]
    
    for config_file in config_files:
        try:
            if Path(config_file).exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"[OK] {config_file}: 有効")
            else:
                print(f"[WARNING]  {config_file}: ファイル不存在")
        except Exception as e:
            print(f"[ERROR] {config_file}: {e}")

def demo_component_integration():
    """コンポーネント統合テスト"""
    print("\n=== コンポーネント統合テスト ===")
    
    try:
        # 各コンポーネントの個別テスト
        from src.execution.paper_trade_scheduler import PaperTradeScheduler
        from src.execution.paper_trade_monitor import PaperTradeMonitor
        from src.execution.strategy_execution_manager import StrategyExecutionManager
        
        # スケジューラーテスト
        scheduler = PaperTradeScheduler({'default_interval_minutes': 15})
        print(f"[OK] スケジューラー初期化: {scheduler.get_status()}")
        
        # モニターテスト
        monitor = PaperTradeMonitor({'performance_window_hours': 24})
        print(f"[OK] モニター初期化: {monitor.get_status()}")
        
        # 戦略管理テスト
        strategy_manager = StrategyExecutionManager({'execution_mode': 'simple'})
        print(f"[OK] 戦略管理初期化: {strategy_manager.get_execution_summary()}")
        
        print("[OK] 全コンポーネント統合テスト完了")
        
    except Exception as e:
        print(f"[ERROR] コンポーネント統合テストエラー: {e}")

def demo_strategy_execution():
    """戦略実行デモテスト"""
    print("\n=== 戦略実行デモテスト ===")
    
    try:
        from src.execution.strategy_execution_manager import StrategyExecutionManager
        
        # 戦略実行管理初期化
        config = {
            'execution_mode': 'simple',
            'default_symbols': ['AAPL'],
            'lookback_periods': 50,
            'position_value': 5000
        }
        
        strategy_manager = StrategyExecutionManager(config)
        
        # シンプル戦略実行テスト
        result = strategy_manager.execute_strategy('VWAP_Breakout', ['AAPL'])
        
        if result.get('success', False):
            print(f"[OK] 戦略実行成功: {result['strategy']}")
            print(f"  - シグナル数: {result.get('signals_generated', 0)}")
            print(f"  - 取引数: {result.get('trades_executed', 0)}")
        else:
            print(f"[WARNING] 戦略実行失敗: {result.get('error', 'Unknown error')}")
        
        print("[OK] 戦略実行デモ完了")
        
    except Exception as e:
        print(f"[ERROR] 戦略実行デモエラー: {e}")

def main():
    """メインデモ実行"""
    print("[ROCKET] ペーパートレード実行システム デモ開始")
    print(f"実行時刻: {datetime.now()}")
    
    # ログディレクトリ作成
    Path("logs").mkdir(exist_ok=True)
    Path("logs/paper_trading").mkdir(exist_ok=True)
    
    # デモ実行
    demo_configuration_validation()
    demo_component_integration()
    demo_strategy_execution()
    demo_simple_mode()
    
    print("\n[SUCCESS] デモ完了")

if __name__ == "__main__":
    main()
