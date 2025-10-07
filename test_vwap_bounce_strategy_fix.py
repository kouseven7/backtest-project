#!/usr/bin/env python3
"""
VWAPBounceStrategy登録・動作検証テスト
2025年10月7日実行 - TODO #9 フォローアップ
"""

import sys
sys.path.append('.')

print('=== VWAPBounceStrategy登録・動作検証テスト ===')

try:
    from config.multi_strategy_manager import MultiStrategyManager
    
    # MultiStrategyManager初期化
    manager = MultiStrategyManager()
    print('✅ MultiStrategyManager: インポート成功')
    
    # 戦略レジストリシステム初期化
    init_success = manager.initialize_systems()
    print(f'✅ 戦略レジストリ初期化結果: {init_success}')
    
    # 戦略レジストリ確認
    available_strategies = manager.get_available_strategies()
    print(f'✅ 利用可能戦略数: {len(available_strategies)}')
    print(f'✅ 戦略リスト: {available_strategies}')
    
    # VWAPBounceStrategyが登録されているか確認
    vwap_bounce_registered = 'VWAPBounceStrategy' in available_strategies
    print(f'🎯 VWAPBounceStrategy登録状況: {"✅ 登録済み" if vwap_bounce_registered else "❌ 未登録"}')
    
    # 7/7戦略完全登録確認
    expected_strategies = [
        'VWAPBreakoutStrategy', 'MomentumInvestingStrategy', 'BreakoutStrategy',
        'OpeningGapStrategy', 'ContrarianStrategy', 'GCStrategy', 'VWAPBounceStrategy'
    ]
    
    registered_count = len([s for s in expected_strategies if s in available_strategies])
    registration_rate = (registered_count / len(expected_strategies)) * 100
    
    print(f'📊 戦略登録状況: {registered_count}/{len(expected_strategies)} ({registration_rate:.1f}%)')
    
    # 未登録戦略の確認
    unregistered = [s for s in expected_strategies if s not in available_strategies]
    if unregistered:
        print(f'⚠️ 未登録戦略: {unregistered}')
    
    if vwap_bounce_registered:
        print('\\n🧪 VWAPBounceStrategy動作確認テスト')
        
        # ダミーデータでテスト
        import pandas as pd
        import numpy as np
        test_data = pd.DataFrame({
            'Close': np.random.randn(10) + 100,
            'High': np.random.randn(10) + 101,
            'Low': np.random.randn(10) + 99,
            'Volume': np.random.randint(1000, 10000, 10),
            'Adj Close': np.random.randn(10) + 100
        })
        test_params = {'param1': 1.0, 'param2': 2.0}
        
        try:
            strategy_instance = manager.get_strategy_instance('VWAPBounceStrategy', test_data, test_params)
            has_backtest = hasattr(strategy_instance, 'backtest')
            print(f'✅ VWAPBounceStrategy インスタンス化成功: backtest()={has_backtest}')
            
            # backtest実行テスト
            if has_backtest:
                try:
                    # backtest実行テスト（エラーハンドリング付き）
                    result = strategy_instance.backtest()
                    print(f'✅ VWAPBounceStrategy backtest実行成功: {result.shape[0]}行')
                    
                    # シグナル列確認
                    required_columns = ['Entry_Signal', 'Exit_Signal']
                    missing_columns = [col for col in required_columns if col not in result.columns]
                    if not missing_columns:
                        print('✅ VWAPBounceStrategy: Entry_Signal/Exit_Signal列確認済み')
                    else:
                        print(f'⚠️ VWAPBounceStrategy: シグナル列不足 {missing_columns}')
                        
                except Exception as e:
                    print(f'⚠️ VWAPBounceStrategy backtest実行エラー: {e}')
                    
        except Exception as e:
            print(f'❌ VWAPBounceStrategy インスタンス化エラー: {e}')
    
    # 戦略レジストリステータス
    registry_status = manager.get_registry_status()
    print(f'\\n📋 レジストリステータス: 初期化={registry_status["is_initialized"]}, 戦略数={registry_status["total_strategies"]}')
    
    print(f'\\n🎯 検証結果サマリー:')
    print(f'  VWAPBounceStrategy登録: {"✅" if vwap_bounce_registered else "❌"}')
    print(f'  7/7戦略完全登録: {"✅" if registration_rate == 100.0 else "❌"}')
    print(f'  戦略レジストリ品質: {registration_rate:.1f}%')
    
    if registration_rate == 100.0:
        print('🎉 VWAPBounceStrategy修正成功！7/7戦略完全登録達成！')
    else:
        print('⚠️ まだ修正が必要な戦略があります')
        
except ImportError as e:
    print(f'❌ インポートエラー: {e}')
except Exception as e:
    print(f'❌ 実行エラー: {e}')
    import traceback
    traceback.print_exc()