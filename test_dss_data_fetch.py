#!/usr/bin/env python3
"""
DSS V3 データ取得機能テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from src.dssms.dssms_backtester_v3 import DSSBacktesterV3

def test_data_fetch():
    """データ取得機能テスト"""
    print("=== DSS V3 データ取得テスト ===")
    
    try:
        # DSS V3 初期化
        dss = DSSBacktesterV3()
        
        # 小規模テスト（2銘柄のみ）
        test_symbols = ['7203', '9984']  # トヨタ、ソフトバンク
        test_date = datetime(2023, 1, 15)
        
        print(f"テスト対象: {test_symbols}")
        print(f"取得日: {test_date.strftime('%Y-%m-%d')}")
        
        # データ取得実行
        market_data = dss.fetch_market_data(test_symbols, test_date)
        
        # 結果確認
        print(f"\n=== 取得結果 ===")
        for symbol, data in market_data.items():
            print(f"{symbol}: {len(data)}日分, 最新価格: {data['Close'].iloc[-1]:.2f}")
            print(f"  期間: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
        
        print(f"\n[OK] データ取得テスト成功: {len(market_data)}/{len(test_symbols)}銘柄")
        return True
        
    except Exception as e:
        print(f"[ERROR] データ取得テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_fetch()