"""
RuntimeError発生テスト
copilot-instructions.md準拠: モックフォールバック禁止の確認
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
from datetime import datetime

print("=" * 60)
print("RuntimeError発生テスト")
print("=" * 60)

# テスト1: 存在しない銘柄でのデータ取得
print("\nテスト1: 存在しない銘柄 (INVALID_SYMBOL)")
print("-" * 60)

try:
    bt = DSSMSIntegratedBacktester()
    
    # data_cacheがNoneの場合の対策
    # 直接yfinanceを使用してテスト
    import yfinance as yf
    from datetime import timedelta
    
    symbol = "INVALID_SYMBOL.T"
    end_date = datetime(2024, 1, 10)
    start_date = end_date - timedelta(days=60)
    
    ticker = yf.Ticker(symbol)
    stock_data = ticker.history(start=start_date, end=end_date + timedelta(days=1))
    
    if len(stock_data) == 0:
        print(f"結果: データなし (Empty DataFrame)")
        print(f"  Rows: {len(stock_data)}")
        print(f"  期待動作: RuntimeError発生")
        print(f"\n問題: yfinanceは存在しない銘柄でもエラーを発生させず、")
        print(f"      空のDataFrameを返すため、RuntimeErrorは")
        print(f"      DSSMS側のロジックで発生させる必要がある")
    else:
        print(f"結果: データ取得成功 (予期せぬ)")
        print(f"  Rows: {len(stock_data)}")
        
except RuntimeError as e:
    print(f"結果: RuntimeError発生 (期待通り)")
    print(f"  Error: {e}")
    print(f"\ncopilot-instructions.md準拠: OK")
except Exception as e:
    print(f"結果: 別の例外")
    print(f"  Exception type: {type(e).__name__}")
    print(f"  Error: {e}")

# テスト2: _legacy_random_selection削除確認
print("\n" + "=" * 60)
print("テスト2: _legacy_random_selection削除確認")
print("-" * 60)

try:
    bt = DSSMSIntegratedBacktester()
    
    # メソッドが存在しないことを確認
    if hasattr(bt, '_legacy_random_selection'):
        print(f"結果: FAILED")
        print(f"  Error: _legacy_random_selection()が still exists")
        print(f"\ncopilot-instructions.md違反: メソッドが削除されていない")
    else:
        print(f"結果: SUCCESS")
        print(f"  _legacy_random_selection()メソッドは削除済み")
        print(f"\ncopilot-instructions.md準拠: OK")
        
except Exception as e:
    print(f"結果: ERROR")
    print(f"  Exception: {e}")

# テスト3: _generate_mock_data削除確認
print("\n" + "=" * 60)
print("テスト3: _generate_mock_data削除確認")
print("-" * 60)

try:
    bt = DSSMSIntegratedBacktester()
    
    # メソッドが存在しないことを確認
    if hasattr(bt, '_generate_mock_data'):
        print(f"結果: FAILED")
        print(f"  Error: _generate_mock_data()が still exists")
        print(f"\ncopilot-instructions.md違反: メソッドが削除されていない")
    else:
        print(f"結果: SUCCESS")
        print(f"  _generate_mock_data()メソッドは削除済み")
        print(f"\ncopilot-instructions.md準拠: OK")
        
except Exception as e:
    print(f"結果: ERROR")
    print(f"  Exception: {e}")

print("\n" + "=" * 60)
print("RuntimeError発生テスト完了")
print("=" * 60)
