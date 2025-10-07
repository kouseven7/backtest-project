import sys
sys.path.append('.')

print("=== Phase 4: 重み判断システム確認 ===")

# 統合システム利用可能性確認
try:
    from config.multi_strategy_manager import MultiStrategyManager
    manager = MultiStrategyManager()
    init_success = manager.initialize_systems()
    print(f'統合システム初期化: {init_success}')
    if init_success:
        print('統合システムが動作中 - 重み計算有効')
    else:
        print('統合システム初期化失敗 - 従来方式使用')
except ImportError as e:
    print(f'統合システム利用不可: {e}')
    print('従来のマルチ戦略システム使用')

# 戦略優先度確認
try:
    from main import load_optimized_parameters
    from data_fetcher import get_parameters_and_data
    ticker, _, _, _, _ = get_parameters_and_data()
    params = load_optimized_parameters(ticker)
    print(f'\n戦略優先度順: {list(params.keys())}')
    
    # パラメータ詳細確認
    print('\n=== 戦略パラメータ詳細 ===')
    for strategy_name, strategy_params in params.items():
        print(f'{strategy_name}: {len(strategy_params) if strategy_params else 0}個のパラメータ')
        if strategy_params:
            # 最初の数個のパラメータを表示
            param_keys = list(strategy_params.keys())[:3]
            for key in param_keys:
                print(f'  {key}: {strategy_params[key]}')
            if len(strategy_params) > 3:
                print(f'  ... 他{len(strategy_params)-3}個')
except Exception as e:
    print(f'パラメータ読み込みエラー: {e}')
    
# デフォルトパラメータ確認
print('\n=== デフォルトパラメータ確認 ===')
try:
    from main import get_default_parameters
    default_params = get_default_parameters()
    print(f'デフォルトパラメータ設定戦略数: {len(default_params)}')
    for strategy_name in list(default_params.keys())[:3]:
        print(f'{strategy_name}: {list(default_params[strategy_name].keys())[:3]}...')
except Exception as e:
    print(f'デフォルトパラメータ確認エラー: {e}')