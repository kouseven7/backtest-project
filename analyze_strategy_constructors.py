#!/usr/bin/env python3
"""
戦略コンストラクタ詳細調査ツール

各戦略のコンストラクタシグネチャを調べ、main.pyからの呼び出しとの不整合を特定します。
"""

import sys
import inspect
import traceback
from pathlib import Path

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def analyze_strategy_constructor(strategy_name, strategy_class):
    """戦略のコンストラクタを詳細分析"""
    print(f"\n🔍 {strategy_name} コンストラクタ分析:")
    print("-" * 50)
    
    try:
        # コンストラクタシグネチャを取得
        init_signature = inspect.signature(strategy_class.__init__)
        print(f"✅ __init__ シグネチャ: {init_signature}")
        
        # パラメータ詳細を表示
        print("📋 パラメータ詳細:")
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            default = param.default if param.default != inspect.Parameter.empty else "必須"
            print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'} (デフォルト: {default})")
        
        # index_dataパラメータの存在確認
        has_index_data = 'index_data' in init_signature.parameters
        print(f"🎯 index_data対応: {'✅ あり' if has_index_data else '❌ なし'}")
        
        return has_index_data, init_signature
        
    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        return False, None

def test_strategy_instantiation(strategy_name, strategy_class, stock_data, index_data):
    """戦略インスタンス化テスト"""
    print(f"\n🧪 {strategy_name} インスタンス化テスト:")
    print("-" * 40)
    
    # パターン1: index_dataありでテスト
    try:
        if strategy_name == 'VWAPBreakoutStrategy':
            # VWAPBreakoutは特別なパラメータを持つ
            strategy = strategy_class(
                stock_data, 
                index_data=index_data,
                stop_loss=0.03,
                take_profit=0.15,
                sma_short=10,
                sma_long=30
            )
        else:
            strategy = strategy_class(stock_data, index_data=index_data)
        print("✅ パターン1 (index_data付き): 成功")
        return True, "with_index_data"
    except Exception as e:
        print(f"❌ パターン1 (index_data付き): {str(e)}")
    
    # パターン2: index_dataなしでテスト
    try:
        if strategy_name == 'VWAPBreakoutStrategy':
            strategy = strategy_class(
                stock_data,
                stop_loss=0.03,
                take_profit=0.15,
                sma_short=10,
                sma_long=30
            )
        else:
            strategy = strategy_class(stock_data)
        print("✅ パターン2 (index_dataなし): 成功")
        return True, "without_index_data"
    except Exception as e:
        print(f"❌ パターン2 (index_dataなし): {str(e)}")
    
    # パターン3: 基本パラメータのみでテスト
    try:
        strategy = strategy_class(stock_data)
        print("✅ パターン3 (基本のみ): 成功")
        return True, "basic_only"
    except Exception as e:
        print(f"❌ パターン3 (基本のみ): {str(e)}")
    
    return False, "all_failed"

def analyze_vwap_breakout_conditions(strategy_result):
    """VWAPBreakoutの条件分析"""
    print(f"\n📊 VWAPBreakout条件分析:")
    print("-" * 40)
    
    if 'Entry_Signal' not in strategy_result.columns:
        print("❌ Entry_Signal列が存在しません")
        return
    
    # エントリーシグナル統計
    total_entries = (strategy_result['Entry_Signal'] == 1).sum()
    total_rows = len(strategy_result)
    entry_rate = (total_entries / total_rows * 100) if total_rows > 0 else 0
    
    print(f"📈 エントリー統計:")
    print(f"  - 総行数: {total_rows}")
    print(f"  - エントリー回数: {total_entries}")
    print(f"  - エントリー率: {entry_rate:.2f}%")
    
    # エグジットシグナル統計
    if 'Exit_Signal' in strategy_result.columns:
        total_exits = (strategy_result['Exit_Signal'] == 1).sum()
        exit_rate = (total_exits / total_rows * 100) if total_rows > 0 else 0
        print(f"📉 エグジット統計:")
        print(f"  - エグジット回数: {total_exits}")
        print(f"  - エグジット率: {exit_rate:.2f}%")
        print(f"  - エントリー/エグジット比: {total_exits/total_entries if total_entries > 0 else 'N/A'}")
    
    # 条件別分析（ログメッセージから推測）
    print(f"\n🔍 条件別分析（推測）:")
    print(f"  - データ不足期間: 最初の30行程度（MA長期=30のため）")
    print(f"  - 出来高条件: volume_threshold=1.2による制限")
    print(f"  - MA順序条件: current > sma_long による上昇トレンド制限")
    print(f"  - VWAPブレイク条件: current > vwap による制限")

def main():
    print("🔍 **戦略コンストラクタ詳細調査開始**")
    print("=" * 60)
    
    # データ準備
    try:
        from data_fetcher import get_parameters_and_data
        from data_processor import preprocess_data
        
        print("📊 テスト用データ準備中...")
        ticker, start_date, end_date = "7203.T", "2024-01-01", "2024-12-31"
        result = get_parameters_and_data(ticker, start_date, end_date)
        
        # get_parameters_and_dataの戻り値を確認（5個の要素がある）
        print(f"戻り値の長さ: {len(result)}")
        print(f"戻り値の各要素の型: {[type(x).__name__ for x in result]}")
        
        # 通常は ticker, start_date, end_date, stock_data, index_data の順
        if len(result) == 5:
            ticker_ret, start_ret, end_ret, stock_data, index_data = result
            print(f"展開結果: ticker={ticker_ret}, start={start_ret}, end={end_ret}")
        else:
            print(f"❌ 予期しない戻り値形式: {len(result)}個の要素")
            return
            
        # preprocess_dataは1つの引数のみを取る
        stock_data = preprocess_data(stock_data)
        print(f"✅ データ準備完了: {stock_data.shape}")
    except Exception as e:
        print(f"❌ データ準備エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 戦略リスト
    strategies = [
        ('VWAPBreakoutStrategy', 'strategies.VWAP_Breakout'),
        ('MomentumInvestingStrategy', 'strategies.Momentum_Investing'),
        ('BreakoutStrategy', 'strategies.Breakout'),
        ('VWAPBounceStrategy', 'strategies.VWAP_Bounce'),
        ('OpeningGapStrategy', 'strategies.Opening_Gap'),
        ('ContrarianStrategy', 'strategies.contrarian_strategy'),
        ('GCStrategy', 'strategies.gc_strategy_signal')
    ]
    
    analysis_results = {}
    
    for strategy_name, module_path in strategies:
        print(f"\n" + "="*60)
        print(f"🎯 {strategy_name} 詳細分析")
        print("="*60)
        
        try:
            # モジュールインポート
            module_parts = module_path.split('.')
            module = __import__(module_path, fromlist=[strategy_name])
            strategy_class = getattr(module, strategy_name)
            print(f"✅ インポート成功: {module_path}.{strategy_name}")
            
            # コンストラクタ分析
            has_index_data, signature = analyze_strategy_constructor(strategy_name, strategy_class)
            
            # インスタンス化テスト
            success, pattern = test_strategy_instantiation(strategy_name, strategy_class, stock_data, index_data)
            
            analysis_results[strategy_name] = {
                'has_index_data': has_index_data,
                'signature': str(signature) if signature else None,
                'instantiation_success': success,
                'working_pattern': pattern
            }
            
            # VWAPBreakoutの場合は条件分析も実行
            if strategy_name == 'VWAPBreakoutStrategy' and success:
                try:
                    if pattern == 'with_index_data':
                        strategy = strategy_class(stock_data, index_data=index_data, stop_loss=0.03, take_profit=0.15, sma_short=10, sma_long=30)
                    else:
                        strategy = strategy_class(stock_data, stop_loss=0.03, take_profit=0.15, sma_short=10, sma_long=30)
                    
                    result = strategy.backtest()
                    analyze_vwap_breakout_conditions(result)
                except Exception as e:
                    print(f"❌ VWAPBreakout条件分析エラー: {e}")
            
        except Exception as e:
            print(f"❌ {strategy_name} 分析エラー: {e}")
            traceback.print_exc()
            analysis_results[strategy_name] = {
                'error': str(e),
                'has_index_data': False,
                'instantiation_success': False
            }
    
    # 総合分析結果
    print(f"\n" + "="*60)
    print("📋 **総合分析結果**")
    print("="*60)
    
    index_data_compatible = []
    index_data_incompatible = []
    instantiation_failed = []
    
    for strategy_name, result in analysis_results.items():
        if result.get('error'):
            instantiation_failed.append(strategy_name)
        elif result.get('has_index_data', False):
            index_data_compatible.append(strategy_name)
        else:
            index_data_incompatible.append(strategy_name)
    
    print(f"\n✅ index_data対応戦略 ({len(index_data_compatible)}個):")
    for name in index_data_compatible:
        pattern = analysis_results[name].get('working_pattern', 'unknown')
        print(f"  - {name} (動作パターン: {pattern})")
    
    print(f"\n❌ index_data非対応戦略 ({len(index_data_incompatible)}個):")
    for name in index_data_incompatible:
        pattern = analysis_results[name].get('working_pattern', 'unknown')
        print(f"  - {name} (動作パターン: {pattern})")
    
    print(f"\n🚫 インスタンス化失敗戦略 ({len(instantiation_failed)}個):")
    for name in instantiation_failed:
        error = analysis_results[name].get('error', 'unknown')
        print(f"  - {name}: {error}")
    
    # 修正提案
    print(f"\n🔧 **修正提案**")
    print("="*40)
    
    if index_data_incompatible:
        print("1. index_data非対応戦略の修正:")
        print("   - コンストラクタにindex_dataパラメータを追加")
        print("   - またはmain.pyで条件分岐してindex_dataを渡さない")
    
    if instantiation_failed:
        print("2. インスタンス化失敗戦略の修正:")
        print("   - エラーメッセージを確認してコンストラクタを修正")
    
    print("3. VWAPBreakout条件緩和:")
    print("   - volume_threshold, breakout_threshold等のパラメータ調整")
    print("   - 上昇トレンド限定条件の緩和")

if __name__ == "__main__":
    main()