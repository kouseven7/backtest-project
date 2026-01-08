#!/usr/bin/env python3
"""
大量銘柄でのレート制限検証

50銘柄、100銘柄での連続テストでレート制限が実際に発生するかを検証
"""

import time
import json
import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime

def load_nikkei225_symbols() -> List[str]:
    """実際のDSSMS設定から225銘柄を読み込み"""
    try:
        with open('config/dssms/nikkei225_components.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('symbols', [])
    except Exception as e:
        print(f"設定ファイル読み込みエラー: {e}")
        return []

def test_bulk_rate_limiting(symbol_count: int) -> Dict[str, Any]:
    """指定数の銘柄でレート制限テスト"""
    symbols = load_nikkei225_symbols()[:symbol_count]
    
    print(f"\n{'='*60}")
    print(f"{symbol_count}銘柄連続取得テスト")
    print(f"{'='*60}")
    
    successful = 0
    errors = 0
    rate_limit_errors = 0
    error_details = []
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        print(f"進行: {i+1}/{symbol_count} - {symbol}", end='')
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get('currentPrice')
            if price:
                successful += 1
                print(f" ✓ {price}")
            else:
                errors += 1
                print(f" ✗ 価格取得失敗")
        except Exception as e:
            errors += 1
            error_msg = str(e)
            print(f" ✗ エラー: {error_msg}")
            
            # レート制限エラーの検出
            if any(keyword in error_msg.lower() for keyword in ['rate limit', 'too many requests', 'throttle']):
                rate_limit_errors += 1
                print(f"   🚨 レート制限エラー検出！")
                error_details.append(f"{symbol}: {error_msg}")
                # レート制限が検出されたら少し待つ
                print("   30秒待機中...")
                time.sleep(30)
    
    total_time = time.time() - start_time
    requests_per_minute = (symbol_count / total_time) * 60
    
    print(f"\n結果:")
    print(f"成功: {successful}/{symbol_count}")
    print(f"エラー: {errors}")
    print(f"レート制限エラー: {rate_limit_errors}")
    print(f"実行時間: {total_time:.1f}秒")
    print(f"リクエスト頻度: {requests_per_minute:.1f}回/分")
    
    if rate_limit_errors > 0:
        print(f"\n🚨 レート制限発生！")
        print("エラー詳細:")
        for error in error_details:
            print(f"  {error}")
        return False
    else:
        print(f"✅ レート制限なし")
        return True

def main():
    """段階的レート制限テスト"""
    print("段階的レート制限検証テスト")
    print(f"実行日時: {datetime.now()}")
    
    # 段階的にテスト
    test_counts = [25, 50, 75]
    
    for count in test_counts:
        success = test_bulk_rate_limiting(count)
        
        if not success:
            print(f"\n🚨 {count}銘柄でレート制限が発生しました。")
            print("これ以上の銘柄数での連続取得は困難です。")
            print("キャッシュシステムが必要と判断されます。")
            break
        else:
            print(f"✅ {count}銘柄では問題なし。次の段階に進みます。")
            time.sleep(5)  # 次のテストまで少し間隔を空ける
    
    print("\nテスト完了")

if __name__ == "__main__":
    main()