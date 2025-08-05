"""
Phase 2.A.2 拡張トレンド切替テスター デモスクリプト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.enhanced_trend_switching_tester import EnhancedTrendSwitchingTester
from config.logger_config import setup_logger

logger = setup_logger(__name__)

def run_demo():
    """デモ実行"""
    try:
        print("="*80)
        print("Phase 2.A.2 拡張トレンド切替テスター デモ")
        print("="*80)
        
        # 設定ファイルパス
        config_path = "src/analysis/trend_switching_config.json"
        
        # テスター初期化
        logger.info("Initializing Enhanced Trend Switching Tester...")
        tester = EnhancedTrendSwitchingTester(config_path)
        
        # 簡易テスト（単一銘柄）
        print("\n1. 単一銘柄テスト実行中...")
        single_result = tester.run_single_symbol_test(
            symbol="SPY",
            timeframe="1h", 
            days=30,
            num_scenarios=4
        )
        
        if 'error' not in single_result:
            summary = single_result.get('test_summary', {})
            print(f"   結果: 成功率 {summary.get('success_rate', 0):.1%}, 実行時間 {summary.get('total_execution_time', 0):.1f}秒")
        else:
            print(f"   エラー: {single_result.get('error')}")
        
        # 小規模バッチテスト
        print("\n2. 小規模バッチテスト実行中...")
        batch_result = tester.run_batch_tests(
            custom_symbols=["SPY", "QQQ"],
            custom_timeframes=["1h"],
            custom_date_ranges=[{"days": 30}]
        )
        
        if 'error' not in batch_result:
            exec_summary = batch_result.get('execution_summary', {})
            print(f"   結果: {exec_summary.get('total_jobs', 0)}ジョブ, 成功率 {exec_summary.get('success_rate', 0):.1%}")
            print(f"   実行時間: {exec_summary.get('total_execution_time', 0):.1f}秒")
        else:
            print(f"   エラー: {batch_result.get('error')}")
        
        # データソース切替テスト
        print("\n3. 合成データモードテスト...")
        tester.config.data_source = "synthetic"
        synthetic_result = tester.run_single_symbol_test(
            symbol="TEST",
            timeframe="1h",
            days=7,
            num_scenarios=2
        )
        
        if 'error' not in synthetic_result:
            summary = synthetic_result.get('test_summary', {})
            print(f"   結果: 成功率 {summary.get('success_rate', 0):.1%}")
        else:
            print(f"   エラー: {synthetic_result.get('error')}")
        
        print("\n" + "="*80)
        print("デモ完了")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nデモ実行エラー: {e}")
        return False

if __name__ == "__main__":
    success = run_demo()
    exit(0 if success else 1)
