#!/usr/bin/env python3
"""
DSSMS実際レート制限検証テスト

DSSMSが225銘柄を処理する際の実際のAPI使用状況と
レート制限発生の実証的検証を実行。

エラーログではなく実際のAPIテストでレート制限の有無を確認。
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

def test_single_price_fetch(symbol: str) -> Dict[str, Any]:
    """単一銘柄のcurrentPrice取得テスト"""
    start_time = time.time()
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        current_price = info.get('currentPrice')
        
        elapsed = time.time() - start_time
        
        return {
            'symbol': symbol,
            'success': True,
            'price': current_price,
            'time_taken': elapsed,
            'error': None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'symbol': symbol,
            'success': False,
            'price': None,
            'time_taken': elapsed,
            'error': str(e)
        }

def test_bulk_price_fetch(symbols: List[str], delay: float = 0.0) -> Dict[str, Any]:
    """225銘柄一括価格取得テスト（DSSMS実際処理をシミュレート）"""
    print(f"225銘柄一括価格取得テスト開始 (遅延: {delay}秒)")
    print(f"対象銘柄数: {len(symbols)}")
    
    results = []
    total_start = time.time()
    successful_count = 0
    error_count = 0
    rate_limit_errors = 0
    
    for i, symbol in enumerate(symbols):
        print(f"進行: {i+1}/{len(symbols)} - {symbol}")
        
        result = test_single_price_fetch(symbol)
        results.append(result)
        
        if result['success']:
            successful_count += 1
        else:
            error_count += 1
            if 'rate limit' in str(result['error']).lower() or 'too many requests' in str(result['error']).lower():
                rate_limit_errors += 1
                print(f"⚠️ レート制限エラー検出: {symbol} - {result['error']}")
        
        # 遅延設定がある場合
        if delay > 0 and i < len(symbols) - 1:
            time.sleep(delay)
    
    total_time = time.time() - total_start
    
    return {
        'total_symbols': len(symbols),
        'successful_count': successful_count,
        'error_count': error_count,
        'rate_limit_errors': rate_limit_errors,
        'total_time': total_time,
        'avg_time_per_symbol': total_time / len(symbols),
        'requests_per_minute': (len(symbols) / total_time) * 60,
        'results': results
    }

def analyze_rate_limiting(test_result: Dict[str, Any]) -> None:
    """レート制限分析結果の表示"""
    print("\n" + "="*60)
    print("レート制限分析結果")
    print("="*60)
    
    print(f"総銘柄数: {test_result['total_symbols']}")
    print(f"成功: {test_result['successful_count']}")
    print(f"エラー: {test_result['error_count']}")
    print(f"レート制限エラー: {test_result['rate_limit_errors']}")
    print(f"実行時間: {test_result['total_time']:.2f}秒")
    print(f"1銘柄あたり平均時間: {test_result['avg_time_per_symbol']:.3f}秒")
    print(f"1分間のリクエスト数: {test_result['requests_per_minute']:.1f}回/分")
    
    # レート制限の判定
    if test_result['rate_limit_errors'] > 0:
        print(f"\n🚨 レート制限が発生しました！")
        print(f"   レート制限エラー数: {test_result['rate_limit_errors']}")
        print(f"   1分間リクエスト数: {test_result['requests_per_minute']:.1f}回")
        print(f"   → キャッシュシステムが必要です")
    else:
        print(f"\n✅ レート制限は発生していません")
        print(f"   1分間リクエスト数: {test_result['requests_per_minute']:.1f}回")
        print(f"   → 直接yfinance APIで動作可能")
    
    # エラー内容詳細表示
    if test_result['error_count'] > 0:
        print(f"\nエラー詳細:")
        for result in test_result['results']:
            if not result['success']:
                print(f"  {result['symbol']}: {result['error']}")

def main():
    """メイン実行"""
    print("DSSMS実際レート制限検証テスト")
    print(f"実行日時: {datetime.now()}")
    
    # 225銘柄読み込み
    symbols = load_nikkei225_symbols()
    if not symbols:
        print("❌ 銘柄リスト読み込み失敗")
        return
    
    print(f"読み込み銘柄数: {len(symbols)}")
    
    # Phase 1: 遅延なしテスト（DSSMS現在の動作をシミュレート）
    print("\n" + "="*60)
    print("Phase 1: 遅延なしテスト（DSSMS現在動作シミュレート）")
    print("="*60)
    
    # 最初の10銘柄でテスト（全225だと時間がかかりすぎるため）
    test_symbols = symbols[:10]
    print(f"テスト対象: 最初の{len(test_symbols)}銘柄")
    
    result_no_delay = test_bulk_price_fetch(test_symbols, delay=0.0)
    analyze_rate_limiting(result_no_delay)
    
    # Phase 2: 0.15秒遅延テスト（SmartCache設定シミュレート）
    if result_no_delay['rate_limit_errors'] > 0:
        print("\n" + "="*60)
        print("Phase 2: 0.15秒遅延テスト（SmartCache設定シミュレート）")
        print("="*60)
        
        result_with_delay = test_bulk_price_fetch(test_symbols, delay=0.15)
        analyze_rate_limiting(result_with_delay)

if __name__ == "__main__":
    main()