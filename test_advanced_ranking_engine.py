#!/usr/bin/env python3
"""
AdvancedRankingEngine 実装状況テスト
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# プロジェクトパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dssms.advanced_ranking_system.advanced_ranking_engine import (
    AdvancedRankingEngine, 
    AdvancedRankingConfig, 
    AnalysisMode, 
    OptimizationLevel
)


async def create_test_data():
    """テスト用データ作成"""
    # 100日分のテストデータ作成
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # ランダムな価格データ生成（現実的な値）
    np.random.seed(42)  # 再現性のため
    
    base_price = 1000
    price_changes = np.random.normal(0, 0.02, 100)  # 2%の日次変動
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # 最低価格1円
    
    volumes = np.random.lognormal(mean=10, sigma=0.5, size=100)
    
    # DataFrameを作成
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'Close': prices,
        'Volume': volumes.astype(int)
    })
    
    test_data.set_index('Date', inplace=True)
    return test_data


async def test_ranking_engine():
    """AdvancedRankingEngine テスト実行"""
    
    print("=== AdvancedRankingEngine 実装状況テスト ===\n")
    
    # 設定作成
    config = AdvancedRankingConfig(
        analysis_mode=AnalysisMode.ENHANCED,
        optimization_level=OptimizationLevel.ADVANCED,
        enable_parallel_processing=False,  # テスト用に無効化
        max_workers=1
    )
    
    # エンジン初期化
    engine = AdvancedRankingEngine(config)
    
    # テストデータ作成
    test_data = await create_test_data()
    print(f"テストデータ作成完了: {len(test_data)}日分")
    print(f"データ期間: {test_data.index[0]} - {test_data.index[-1]}")
    print(f"価格範囲: ¥{test_data['Close'].min():.2f} - ¥{test_data['Close'].max():.2f}\n")
    
    # テスト銘柄リスト
    test_symbols = ['TEST001', 'TEST002', 'TEST003']
    symbol_data = {symbol: test_data for symbol in test_symbols}
    
    print("1. 基本分析メソッドテスト:")
    try:
        basic_result = engine._perform_basic_analysis(test_data)
        print(f"   ✓ _perform_basic_analysis: {basic_result}")
    except Exception as e:
        print(f"   ✗ _perform_basic_analysis エラー: {e}")
    
    print("\n2. テクニカル分析メソッドテスト:")
    try:
        technical_result = engine._perform_technical_analysis(test_data)
        print(f"   ✓ _perform_technical_analysis: {technical_result}")
    except Exception as e:
        print(f"   ✗ _perform_technical_analysis エラー: {e}")
    
    print("\n3. 出来高分析メソッドテスト:")
    try:
        volume_result = engine._perform_volume_analysis(test_data)
        print(f"   ✓ _perform_volume_analysis: {volume_result}")
    except Exception as e:
        print(f"   ✗ _perform_volume_analysis エラー: {e}")
    
    print("\n4. ボラティリティ分析メソッドテスト:")
    try:
        volatility_result = engine._perform_volatility_analysis(test_data)
        print(f"   ✓ _perform_volatility_analysis: {volatility_result}")
    except Exception as e:
        print(f"   ✗ _perform_volatility_analysis エラー: {e}")
    
    print("\n5. モメンタム分析メソッドテスト:")
    try:
        momentum_result = engine._perform_momentum_analysis(test_data)
        print(f"   ✓ _perform_momentum_analysis: {momentum_result}")
    except Exception as e:
        print(f"   ✗ _perform_momentum_analysis エラー: {e}")
    
    print("\n6. 全体的なランキング実行テスト:")
    try:
        # 非同期ランキング実行
        ranking_results = await engine.analyze_symbols_advanced(test_symbols, symbol_data)
        
        print(f"   ✓ ランキング結果数: {len(ranking_results)}")
        
        for i, result in enumerate(ranking_results):
            print(f"   順位 {i+1}: {result.symbol}")
            print(f"      最終スコア: {result.final_score:.2f}")
            print(f"      信頼度: {result.confidence_level:.2f}")
            print(f"      市場状況: {result.market_condition}")
            
            # 多次元スコア表示
            print(f"      多次元スコア:")
            for dimension, score in result.multi_dimensional_scores.items():
                print(f"        {dimension}: {score:.3f}")
            print()
            
    except Exception as e:
        print(f"   ✗ ランキング実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("7. システム状態確認:")
    try:
        status = engine.get_system_status()
        print(f"   設定: {status['config']}")
        print(f"   キャッシュ状態: {status['cache_status']}")
        print(f"   統合状況: {status['integration']}")
    except Exception as e:
        print(f"   ✗ システム状態取得エラー: {e}")
    
    # クリーンアップ
    engine.cleanup()
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    asyncio.run(test_ranking_engine())