#!/usr/bin/env python3
"""
TODO-002 Step3 修正版: コンストラクタ引数対応
Step3でコンストラクタエラーが発生したため、適切なパラメータで戦略を初期化する
"""

import os
import sys
import pandas as pd
import numpy as np
import importlib.util

print("=" * 80)
print("🔍 TODO-002 Step3 修正版: シグナル変換問題検証")
print("=" * 80)

def test_strategy_with_proper_init(strategy_name, strategy_file):
    """適切なコンストラクタ引数で戦略をテスト"""
    print(f"\n📊 **{strategy_name}単体テスト（修正版）**")
    
    if not os.path.exists(strategy_file):
        print(f"❌ ファイル存在せず: {strategy_file}")
        return None
    
    try:
        # テスト用データ準備
        test_data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 99, 104, 98, 105, 97, 106, 108, 107, 109, 106, 110],
            'Volume': [1000, 1100, 950, 1200, 800, 1300, 700, 1400, 600, 1500, 1600, 1550, 1700, 1450, 1800],
            'High': [101, 103, 102, 104, 100, 105, 99, 106, 98, 107, 109, 108, 110, 107, 111],
            'Low': [99, 101, 100, 102, 98, 103, 97, 104, 96, 105, 107, 106, 108, 105, 109],
            'Open': [100, 102, 101, 103, 99, 104, 98, 105, 97, 106, 108, 107, 109, 106, 110]
        })
        
        # Index設定（日付ベース）
        test_data.index = pd.date_range('2024-01-01', periods=len(test_data), freq='D')
        
        # Entry_Signal/Exit_Signal列を初期化
        test_data['Entry_Signal'] = 0
        test_data['Exit_Signal'] = 0
        
        # モジュール動的インポート
        sys.path.append(os.path.dirname(strategy_file))
        module_name = os.path.basename(strategy_file).replace('.py', '')
        
        spec = importlib.util.spec_from_file_location(module_name, strategy_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 戦略クラスの取得と適切な初期化
        strategy = None
        
        if strategy_name == 'contrarian_strategy':
            strategy_class = getattr(module, 'ContrarianStrategy')
            strategy = strategy_class(test_data.copy())
            
        elif strategy_name == 'gc_strategy_signal':
            strategy_class = getattr(module, 'GCStrategy')
            strategy = strategy_class(test_data.copy())
            
        elif strategy_name == 'VWAP_Breakout':
            strategy_class = getattr(module, 'VWAPBreakoutStrategy')
            # VWAPBreakoutはindex_dataも必要
            index_data = test_data.copy()  # 簡易的にsame dataを使用
            strategy = strategy_class(test_data.copy(), index_data)
            
        elif strategy_name == 'Momentum_Investing':
            strategy_class = getattr(module, 'MomentumInvestingStrategy')
            strategy = strategy_class(test_data.copy())
        
        if strategy is None:
            print(f"❌ 戦略クラス初期化失敗")
            return None
        
        print(f"✅ 戦略初期化成功")
        
        # Step3-1: generate_exit_signal()直接テスト（エラーを恐れずテスト）
        print(f"\n🔧 **generate_exit_signal()直接テスト**")
        
        if hasattr(strategy, 'generate_exit_signal'):
            exit_signals = []
            test_indices = [3, 5, 7, 9, 11]  # 十分なデータがある範囲でテスト
            
            for idx in test_indices:
                try:
                    # VWAPBreakoutは特別な引数が必要
                    if strategy_name == 'VWAP_Breakout':
                        signal = strategy.generate_exit_signal(idx, idx-2)  # entry_idx
                    else:
                        signal = strategy.generate_exit_signal(idx)
                    
                    exit_signals.append(signal)
                    print(f"  idx={idx}: generate_exit_signal() → {signal}")
                    
                except Exception as e:
                    print(f"  idx={idx}: エラー → {str(e)[:50]}...")
                    exit_signals.append(None)
            
            # 有効なシグナルのみ分析
            valid_signals = [s for s in exit_signals if s is not None]
            negative_ones = [s for s in valid_signals if s == -1]
            positive_ones = [s for s in valid_signals if s == 1]
            
            print(f"📊 有効シグナル: {len(valid_signals)}件")
            print(f"📊 -1返却: {len(negative_ones)}件, +1返却: {len(positive_ones)}件")
        
        # Step3-2: 簡易backtest実行テスト
        print(f"\n🔧 **簡易backtest実行テスト**")
        
        try:
            # backtest前の状態記録
            pre_exit_signal = strategy.data['Exit_Signal'].copy()
            pre_exit_sum = (pre_exit_signal != 0).sum()
            
            print(f"backtest前Exit_Signal非零: {pre_exit_sum}件")
            
            # backtest実行
            strategy.backtest()
            
            # backtest後の状態確認
            post_exit_signal = strategy.data['Exit_Signal'].copy()
            post_exit_sum = (post_exit_signal != 0).sum()
            
            print(f"backtest後Exit_Signal非零: {post_exit_sum}件")
            
            # 具体的な値の分析
            negative_ones_final = (post_exit_signal == -1).sum()
            positive_ones_final = (post_exit_signal == 1).sum()
            
            print(f"🚨 **最終Exit_Signal値分析**:")
            print(f"  Exit_Signal = -1: {negative_ones_final}件")
            print(f"  Exit_Signal = +1: {positive_ones_final}件")
            
            # TODO-001問題判定
            if negative_ones_final > 0 and positive_ones_final == 0:
                conclusion = "✅ 正常（-1のみ使用）"
                conversion_issue = False
            elif negative_ones_final == 0 and positive_ones_final > 0:
                conclusion = "⚠️ 要注意（+1のみ使用、設計と矛盾可能性）"
                conversion_issue = True
            elif negative_ones_final > 0 and positive_ones_final > 0:
                conclusion = "🚨 異常（-1と+1が混在）"
                conversion_issue = True
            else:
                conclusion = "ℹ️ エグジットシグナルなし"
                conversion_issue = False
            
            print(f"  判定: {conclusion}")
            
            # 実際の値を確認（非零のもの）
            nonzero_exits = post_exit_signal[post_exit_signal != 0]
            if len(nonzero_exits) > 0:
                print(f"  非零Exit_Signal値: {nonzero_exits.tolist()}")
            
            return {
                'strategy_name': strategy_name,
                'success': True,
                'negative_ones': int(negative_ones_final),
                'positive_ones': int(positive_ones_final),
                'conversion_issue': conversion_issue,
                'conclusion': conclusion
            }
            
        except Exception as e:
            print(f"❌ backtest実行エラー: {e}")
            return {
                'strategy_name': strategy_name,
                'success': False,
                'error': str(e)
            }
        
    except Exception as e:
        print(f"❌ 全般エラー: {e}")
        return {
            'strategy_name': strategy_name,
            'success': False,
            'error': str(e)
        }

# テスト実行
test_strategies = [
    ('contrarian_strategy', 'strategies/contrarian_strategy.py'),
    ('gc_strategy_signal', 'strategies/gc_strategy_signal.py'),
    ('VWAP_Breakout', 'strategies/VWAP_Breakout.py'),
    ('Momentum_Investing', 'strategies/Momentum_Investing.py')
]

print("🔍 **個別戦略シグナル変換問題検証開始（修正版）**")
test_results = []

for strategy_name, strategy_file in test_strategies:
    result = test_strategy_with_proper_init(strategy_name, strategy_file)
    if result:
        test_results.append(result)

# 統合結果分析
print("\n" + "=" * 80)
print("📊 **TODO-002 Step3 修正版統合結果**")
print("=" * 80)

successful_tests = [r for r in test_results if r.get('success', False)]
print(f"📋 成功テスト: {len(successful_tests)}/{len(test_results)}戦略")

if successful_tests:
    print(f"\n🚨 **TODO-001類似問題検証結果**:")
    
    for result in successful_tests:
        strategy = result['strategy_name']
        neg = result.get('negative_ones', 0)
        pos = result.get('positive_ones', 0)
        issue = result.get('conversion_issue', False)
        conclusion = result.get('conclusion', 'N/A')
        
        status = "🚨 変換問題" if issue else "✅ 正常"
        print(f"  - {strategy}: Exit_Signal=-1({neg}件), =1({pos}件) {status}")
        print(f"    判定: {conclusion}")
    
    # 総合判定
    conversion_issues = [r for r in successful_tests if r.get('conversion_issue', False)]
    
    if len(conversion_issues) == 0:
        print(f"\n✅ **重要結論**: 個別戦略レベルでは-1→1変換問題は発生していない")
        print(f"   → TODO-001の変換問題はmain.py処理パイプラインで発生している")
        print(f"   → 6戦略全てで同様の現象が起きている可能性が高い")
    else:
        print(f"\n🚨 **重要結論**: {len(conversion_issues)}戦略で変換問題を確認")
        print(f"   → TODO-001と同様の問題が複数戦略で発生している")

else:
    print("❌ 全戦略でテスト失敗")

print(f"\n✅ TODO-002 Step3 修正版完了")
print(f"📋 TODO-002最終報告書作成準備完了")