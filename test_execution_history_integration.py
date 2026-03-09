"""ExecutionHistoryとMarketConditionMonitorの連携テスト"""
import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.dssms.execution_history import ExecutionHistory
from src.dssms.market_condition_monitor import MarketConditionMonitor

print("=" * 60)
print("ExecutionHistory & MarketConditionMonitor 連携テスト")
print("=" * 60)

# 1. MarketConditionMonitor初期化
print("\n[1] MarketConditionMonitor初期化...")
market_monitor = MarketConditionMonitor()
print("✓ MarketConditionMonitor初期化成功")

# 2. ExecutionHistory初期化（market_monitor連携あり）
print("\n[2] ExecutionHistory初期化（market_monitor連携あり）...")
execution_history = ExecutionHistory(market_monitor=market_monitor)
print(f"✓ ExecutionHistory初期化成功")
print(f"  - market_monitor属性: {hasattr(execution_history, 'market_monitor')}")
print(f"  - market_monitorがNoneでない: {execution_history.market_monitor is not None}")

# 3. _capture_market_snapshot()テスト
print("\n[3] _capture_market_snapshot()テスト...")
try:
    snapshot = execution_history._capture_market_snapshot()
    print(f"✓ スナップショット取得成功:")
    print(f"  - timestamp: {snapshot.get('timestamp', 'N/A')}")
    print(f"  - nikkei225_trend: {snapshot.get('nikkei225_trend', 'N/A')}")
    print(f"  - volatility_level: {snapshot.get('volatility_level', 'N/A')}")
    
    # nikkei225_trendが"unknown"以外かチェック
    if snapshot.get('nikkei225_trend') != 'unknown':
        print(f"\n✅ SUCCESS: nikkei225_trendが正常に取得されました（値: {snapshot.get('nikkei225_trend')}）")
    else:
        print(f"\n⚠ WARNING: nikkei225_trendが'unknown'のままです")
        print("  （これは市場データ取得に失敗した可能性があります）")
        
except Exception as e:
    print(f"✗ スナップショット取得失敗: {e}")
    import traceback
    traceback.print_exc()

# 4. ExecutionHistory初期化（market_monitor連携なし）
print("\n[4] ExecutionHistory初期化（market_monitor連携なし）...")
execution_history_no_monitor = ExecutionHistory()
print(f"✓ ExecutionHistory初期化成功")
print(f"  - market_monitorがNone: {execution_history_no_monitor.market_monitor is None}")

snapshot_no_monitor = execution_history_no_monitor._capture_market_snapshot()
print(f"  - nikkei225_trend（監視なし）: {snapshot_no_monitor.get('nikkei225_trend', 'N/A')}")

print("\n" + "=" * 60)
print("テスト完了")
print("=" * 60)
