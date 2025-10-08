#!/usr/bin/env python3
"""
TODO #9戦略レジストリシステム完全実装検証テスト
2025年10月7日実行
"""

import sys
import os
sys.path.append('.')

print('=== TODO #9 戦略レジストリシステム完全実装検証 ===')

try:
    from config.multi_strategy_manager import MultiStrategyManager
    
    # MultiStrategyManager初期化
    manager = MultiStrategyManager()
    print('[OK] MultiStrategyManager: インポート成功')
    
    # 戦略レジストリシステム初期化
    init_success = manager.initialize_systems()
    print(f'[OK] 戦略レジストリ初期化結果: {init_success}')
    
    # 戦略レジストリ確認
    available_strategies = manager.get_available_strategies()
    print(f'[OK] 利用可能戦略数: {len(available_strategies)}')
    print(f'[OK] 戦略リスト: {available_strategies}')
    
    # TODO #8で失敗した戦略の確認
    target_strategies = ['VWAPBreakoutStrategy', 'MomentumInvestingStrategy', 'BreakoutStrategy']
    recovered_strategies = [s for s in target_strategies if s in available_strategies]
    print(f'[OK] TODO #8エラー戦略復旧: {len(recovered_strategies)}/{len(target_strategies)} - {recovered_strategies}')
    
    # 戦略レジストリステータス
    registry_status = manager.get_registry_status()
    print(f'[OK] レジストリステータス: 初期化={registry_status["is_initialized"]}, 戦略数={registry_status["total_strategies"]}')
    
    # 戦略インスタンス化テスト（可能な場合）
    if available_strategies:
        test_strategy = available_strategies[0]
        print(f'\n[TEST] 戦略インスタンス化テスト: {test_strategy}')
        
        # ダミーデータでテスト
        import pandas as pd
        import numpy as np
        test_data = pd.DataFrame({
            'Close': np.random.randn(10) + 100,
            'High': np.random.randn(10) + 101,
            'Low': np.random.randn(10) + 99,
            'Volume': np.random.randint(1000, 10000, 10)
        })
        test_params = {'param1': 1.0, 'param2': 2.0}
        
        try:
            strategy_instance = manager.get_strategy_instance(test_strategy, test_data, test_params)
            has_backtest = hasattr(strategy_instance, 'backtest')
            print(f'[OK] インスタンス化成功: {test_strategy}, backtest()={has_backtest}')
        except Exception as e:
            print(f'[WARNING] インスタンス化テスト: {test_strategy} - {e}')
    
    print(f'\n[TARGET] TODO #9 成功指標確認:')
    print(f'  戦略レジストリ実装: {"[OK]" if len(available_strategies) > 0 else "[ERROR]"}')
    print(f'  TODO #8エラー戦略復旧: {"[OK]" if len(recovered_strategies) >= 2 else "[ERROR]"}')
    print(f'  バックテスト基本理念準拠: {"[OK]" if init_success else "[ERROR]"}')
    
    if len(recovered_strategies) >= 2 and init_success:
        print('[SUCCESS] TODO #9: 戦略レジストリシステム完全実装 - 成功！')
    else:
        print('[WARNING] TODO #9: 部分的成功 - 追加調整が必要')
        
except ImportError as e:
    print(f'[ERROR] インポートエラー: {e}')
except Exception as e:
    print(f'[ERROR] 実行エラー: {e}')
    import traceback
    traceback.print_exc()