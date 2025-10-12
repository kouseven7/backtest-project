#!/usr/bin/env python3
"""
戦略初期化詳細テストツール

main.pyと同じ方法で各戦略を初期化し、VWAPBreakoutの条件も詳細分析します。
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def get_default_parameters(strategy_name: str):
    """戦略のデフォルトパラメータを取得"""
    defaults = {
        'VWAPBreakoutStrategy': {
            'stop_loss': 0.03,
            'take_profit': 0.15,
            'sma_short': 10,
            'sma_long': 30,
            'volume_threshold': 1.2,
            'confirmation_bars': 1,
            'breakout_min_percent': 0.003,
            'trailing_stop': 0.05,
            'trailing_start_threshold': 0.03,
            'max_holding_period': 10,
        },
        'MomentumInvestingStrategy': {
            'sma_short': 20,
            'sma_long': 50,
            'rsi_period': 14,
            'take_profit': 0.12,
            'stop_loss': 0.06,
        },
        'BreakoutStrategy': {
            'volume_threshold': 1.2,
            'take_profit': 0.03,
            'look_back': 1,
            'trailing_stop': 0.02,
        },
        'VWAPBounceStrategy': {
            'vwap_lower_threshold': 0.99,
            'stop_loss': 0.02,
            'take_profit': 0.05,
        },
        'OpeningGapStrategy': {
            'gap_threshold': 0.02,
            'volume_threshold': 1.5,
            'confirmation_period': 3,
        },
        'ContrarianStrategy': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'gap_threshold': 0.05,
        },
        'GCStrategy': {
            'short_window': 5,
            'long_window': 25,
            'take_profit': 0.05,
            'stop_loss': 0.03,
        }
    }
    return defaults.get(strategy_name, {})

def test_main_py_style_initialization():
    """main.pyと同じスタイルで戦略初期化テスト"""
    print("🧪 **main.py式戦略初期化テスト**")
    print("=" * 60)
    
    # データ準備
    try:
        from data_fetcher import get_parameters_and_data
        from data_processor import preprocess_data
        
        ticker, start_date, end_date = "7203.T", "2024-01-01", "2024-12-31"
        result = get_parameters_and_data(ticker, start_date, end_date)
        ticker_ret, start_ret, end_ret, stock_data, index_data = result
        stock_data = preprocess_data(stock_data)
        print(f"✅ データ準備完了: {stock_data.shape}")
    except Exception as e:
        print(f"❌ データ準備エラー: {e}")
        return
    
    # 戦略リスト（main.pyと同じ順序）
    strategies = [
        ('VWAPBreakoutStrategy', 'strategies.VWAP_Breakout'),
        ('MomentumInvestingStrategy', 'strategies.Momentum_Investing'),
        ('BreakoutStrategy', 'strategies.Breakout'),
        ('VWAPBounceStrategy', 'strategies.VWAP_Bounce'),
        ('OpeningGapStrategy', 'strategies.Opening_Gap'),
        ('ContrarianStrategy', 'strategies.contrarian_strategy'),
        ('GCStrategy', 'strategies.gc_strategy_signal')
    ]
    
    successful_strategies = []
    failed_strategies = []
    
    for strategy_name, module_path in strategies:
        print(f"\n🎯 {strategy_name} main.py式初期化テスト")
        print("-" * 50)
        
        try:
            # インポート
            module = __import__(module_path, fromlist=[strategy_name])
            strategy_class = getattr(module, strategy_name)
            
            # パラメータ取得
            params = get_default_parameters(strategy_name)
            print(f"📋 使用パラメータ: {params}")
            
            # main.pyと同じ初期化ロジック
            strategy = None
            
            if strategy_name == 'VWAPBreakoutStrategy':
                strategy = strategy_class(
                    data=stock_data.copy(),
                    index_data=index_data,
                    params=params,
                    price_column="Adj Close"
                )
            elif strategy_name == 'OpeningGapStrategy':
                strategy = strategy_class(
                    data=stock_data.copy(),
                    params=params,
                    price_column="Adj Close",
                    dow_data=index_data
                )
            else:
                # その他の戦略は共通パラメータで初期化
                strategy = strategy_class(
                    data=stock_data.copy(),
                    params=params,
                    price_column="Adj Close"
                )
            
            print(f"✅ 初期化成功: {strategy_name}")
            
            # backtest実行テスト
            try:
                result = strategy.backtest()
                entries = (result['Entry_Signal'] == 1).sum()
                exits = (result['Exit_Signal'] == 1).sum()
                print(f"📊 backtest結果: エントリー={entries}, エグジット={exits}")
                
                successful_strategies.append({
                    'name': strategy_name,
                    'entries': entries,
                    'exits': exits,
                    'result_shape': result.shape
                })
                
                # VWAPBreakoutの場合は詳細分析
                if strategy_name == 'VWAPBreakoutStrategy':
                    analyze_vwap_detailed(result, strategy)
                    
            except Exception as e:
                print(f"❌ backtest実行エラー: {e}")
                failed_strategies.append({
                    'name': strategy_name,
                    'error_type': 'backtest_failed',
                    'error': str(e)
                })
            
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            failed_strategies.append({
                'name': strategy_name,
                'error_type': 'init_failed',
                'error': str(e)
            })
    
    # 結果サマリー
    print(f"\n" + "=" * 60)
    print("📋 **初期化テスト結果サマリー**")
    print("=" * 60)
    
    print(f"\n✅ 成功戦略 ({len(successful_strategies)}個):")
    for s in successful_strategies:
        print(f"  - {s['name']}: エントリー={s['entries']}, エグジット={s['exits']}")
    
    print(f"\n❌ 失敗戦略 ({len(failed_strategies)}個):")
    for s in failed_strategies:
        print(f"  - {s['name']}: {s['error_type']} - {s['error']}")

def analyze_vwap_detailed(result, strategy):
    """VWAPBreakout詳細分析"""
    print(f"\n📊 VWAPBreakout詳細条件分析:")
    print("-" * 40)
    
    total_rows = len(result)
    entries = (result['Entry_Signal'] == 1).sum()
    exits = (result['Exit_Signal'] == 1).sum()
    
    print(f"📈 基本統計:")
    print(f"  - 総データ行数: {total_rows}")
    print(f"  - エントリー回数: {entries}")
    print(f"  - エグジット回数: {exits}")
    print(f"  - エントリー率: {entries/total_rows*100:.2f}%")
    
    # エントリー条件分析
    if hasattr(strategy, 'params') and strategy.params:
        params = strategy.params
        print(f"\n🔧 現在のパラメータ:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
    
    # 実際のエントリー/エグジット日付確認
    entry_dates = result[result['Entry_Signal'] == 1].index
    exit_dates = result[result['Exit_Signal'] == 1].index
    
    print(f"\n📅 実際のシグナル日付:")
    print(f"  - エントリー日: {list(entry_dates)}")
    print(f"  - エグジット日: {list(exit_dates)}")
    
    # 条件緩和提案
    print(f"\n💡 条件緩和提案:")
    print(f"  - volume_threshold: 1.2 → 1.0 (出来高条件緩和)")
    print(f"  - sma_long: 30 → 20 (データ不足期間短縮)")
    print(f"  - breakout_min_percent: 0.003 → 0.001 (ブレイクアウト閾値緩和)")

def main():
    test_main_py_style_initialization()

if __name__ == "__main__":
    main()