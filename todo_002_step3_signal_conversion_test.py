#!/usr/bin/env python3
"""
TODO-002 Step3: 個別戦略のシグナル変換問題検証
Step2で全戦略が-1を使用していることが判明。
TODO-001と同様の「Exit_Signal: -1 → 1 変換問題」が発生しているか検証する。
"""

import os
import sys
import pandas as pd
import numpy as np

print("=" * 80)
print("🔍 TODO-002 Step3: 個別戦略シグナル変換問題検証")
print("=" * 80)

def test_individual_strategy_signals(strategy_name, strategy_file):
    """個別戦略のシグナル生成・変換問題を検証"""
    print(f"\n📊 **{strategy_name}単体テスト**")
    
    if not os.path.exists(strategy_file):
        print(f"❌ ファイル存在せず: {strategy_file}")
        return None
    
    try:
        # ストラテジーファイルを動的インポート
        sys.path.append(os.path.dirname(strategy_file))
        module_name = os.path.basename(strategy_file).replace('.py', '')
        
        # テスト用の簡単なデータ作成
        test_data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 99, 104, 98, 105, 97, 106],
            'Volume': [1000, 1100, 950, 1200, 800, 1300, 700, 1400, 600, 1500],
            'High': [101, 103, 102, 104, 100, 105, 99, 106, 98, 107],
            'Low': [99, 101, 100, 102, 98, 103, 97, 104, 96, 105],
            'Open': [100, 102, 101, 103, 99, 104, 98, 105, 97, 106]
        })
        
        # Entry_Signal/Exit_Signal列を初期化
        test_data['Entry_Signal'] = 0
        test_data['Exit_Signal'] = 0
        
        # 戦略クラスの特定とインスタンス化
        strategy_classes = {
            'contrarian_strategy': 'ContrarianStrategy',
            'gc_strategy_signal': 'GCStrategy', 
            'VWAP_Breakout': 'VWAPBreakoutStrategy',
            'Momentum_Investing': 'MomentumInvestingStrategy'
        }
        
        if module_name not in strategy_classes:
            print(f"⚠️ 未対応戦略: {module_name}")
            return None
        
        # モジュール動的インポート
        spec = importlib.util.spec_from_file_location(module_name, strategy_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        strategy_class = getattr(module, strategy_classes[module_name])
        
        # 戦略インスタンス作成（デフォルトパラメータ）
        try:
            strategy = strategy_class()
            strategy.data = test_data.copy()
        except Exception as e:
            print(f"⚠️ 戦略初期化エラー: {e}")
            return None
        
        print(f"✅ 戦略初期化成功: {strategy_classes[module_name]}")
        
        # Step3-1: generate_exit_signal()の直接テスト
        print(f"\n🔧 **Step3-1: generate_exit_signal()直接テスト**")
        
        if hasattr(strategy, 'generate_exit_signal'):
            try:
                # 複数のインデックスでテスト
                test_indices = [2, 4, 6, 8]
                exit_signals = []
                
                for idx in test_indices:
                    try:
                        # メソッドの引数に応じて呼び出し
                        if module_name == 'VWAP_Breakout':
                            signal = strategy.generate_exit_signal(idx, idx-1)  # entry_idxも必要
                        else:
                            signal = strategy.generate_exit_signal(idx)
                        exit_signals.append(signal)
                        print(f"  idx={idx}: generate_exit_signal() → {signal}")
                    except Exception as e:
                        print(f"  idx={idx}: エラー → {e}")
                        exit_signals.append(None)
                
                # -1が正常に返されているか確認
                negative_ones = [s for s in exit_signals if s == -1]
                print(f"📊 -1返却回数: {len(negative_ones)}/{len(exit_signals)}")
                
            except Exception as e:
                print(f"❌ generate_exit_signal()テストエラー: {e}")
        else:
            print(f"❌ generate_exit_signal()メソッドなし")
        
        # Step3-2: backtest()実行とExit_Signal列確認
        print(f"\n🔧 **Step3-2: backtest()実行テスト**")
        
        try:
            # backtest実行前のExit_Signal状態
            pre_backtest = strategy.data['Exit_Signal'].copy()
            print(f"backtest前Exit_Signal: {pre_backtest.tolist()}")
            
            # backtest実行
            strategy.backtest()
            
            # backtest実行後のExit_Signal状態
            post_backtest = strategy.data['Exit_Signal'].copy()
            print(f"backtest後Exit_Signal: {post_backtest.tolist()}")
            
            # 変化の確認
            changes = post_backtest != pre_backtest
            changed_indices = post_backtest[changes].index.tolist()
            changed_values = post_backtest[changes].tolist()
            
            print(f"📊 Exit_Signal変更: {len(changed_indices)}箇所")
            for idx, val in zip(changed_indices, changed_values):
                print(f"  idx={idx}: 0 → {val}")
            
            # -1 vs 1の確認  
            negative_ones_count = (post_backtest == -1).sum()
            positive_ones_count = (post_backtest == 1).sum()
            
            print(f"🚨 **TODO-001類似問題チェック**:")
            print(f"  Exit_Signal=-1: {negative_ones_count}件")
            print(f"  Exit_Signal= 1: {positive_ones_count}件")
            
            if negative_ones_count > 0 and positive_ones_count == 0:
                print(f"✅ **正常**: 戦略は-1を設定、1への変換なし")
                conversion_issue = False
            elif negative_ones_count == 0 and positive_ones_count > 0:
                print(f"🚨 **異常**: 戦略は1を設定（設計と矛盾の可能性）")
                conversion_issue = True
            elif negative_ones_count > 0 and positive_ones_count > 0:
                print(f"⚠️ **混在**: -1と1が両方存在（処理ロジック要確認）")
                conversion_issue = True
            else:
                print(f"ℹ️ **エグジットなし**: Exit_Signal変更なし")
                conversion_issue = False
            
            return {
                'strategy_name': strategy_name,
                'initialization_success': True,
                'exit_signal_changes': len(changed_indices),
                'negative_ones': negative_ones_count,
                'positive_ones': positive_ones_count,
                'conversion_issue': conversion_issue
            }
            
        except Exception as e:
            print(f"❌ backtest()実行エラー: {e}")
            return {
                'strategy_name': strategy_name,
                'initialization_success': True,
                'backtest_error': str(e)
            }
        
    except Exception as e:
        print(f"❌ 戦略テスト全般エラー: {e}")
        return {
            'strategy_name': strategy_name,
            'initialization_success': False,
            'error': str(e)
        }

# 必要なモジュールのインポート
import importlib.util

# テスト対象戦略
test_strategies = [
    ('contrarian_strategy', 'strategies/contrarian_strategy.py'),
    ('gc_strategy_signal', 'strategies/gc_strategy_signal.py'),
    ('VWAP_Breakout', 'strategies/VWAP_Breakout.py'),
    ('Momentum_Investing', 'strategies/Momentum_Investing.py')
]

print("🔍 **個別戦略シグナル変換問題検証開始**")
test_results = []

for strategy_name, strategy_file in test_strategies:
    result = test_individual_strategy_signals(strategy_name, strategy_file)
    if result:
        test_results.append(result)

# 統合結果分析
print("\n" + "=" * 80)
print("📊 **TODO-002 Step3 統合結果**")
print("=" * 80)

if test_results:
    successful_tests = [r for r in test_results if r.get('initialization_success', False)]
    print(f"📋 成功テスト: {len(successful_tests)}/{len(test_results)}戦略")
    
    # TODO-001類似問題の検証結果
    conversion_issues = [r for r in successful_tests if r.get('conversion_issue', False)]
    
    print(f"\n🚨 **TODO-001類似問題検証結果**:")
    print(f"シグナル変換問題: {len(conversion_issues)}戦略で検出")
    
    for result in successful_tests:
        if 'negative_ones' in result and 'positive_ones' in result:
            strategy = result['strategy_name']
            neg = result['negative_ones']
            pos = result['positive_ones']
            issue = "🚨 変換問題" if result.get('conversion_issue', False) else "✅ 正常"
            print(f"  - {strategy}: Exit_Signal=-1({neg}件), =1({pos}件) {issue}")

    # 結論
    if len(conversion_issues) == 0:
        print(f"\n✅ **重要結論**: 個別戦略レベルでは-1→1変換問題は発生していない")
        print(f"   → TODO-001の変換問題はmain.py処理パイプラインで発生")
    else:
        print(f"\n🚨 **重要結論**: {len(conversion_issues)}戦略で変換問題を確認")
        print(f"   → TODO-001と同様の問題が複数戦略で発生")

else:
    print("❌ 全戦略でテスト失敗")

print(f"\n✅ TODO-002 Step3 個別戦略検証完了")
print(f"📋 次: TODO-002最終報告書の作成")