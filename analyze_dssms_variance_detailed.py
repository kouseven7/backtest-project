#!/usr/bin/env python3
"""
DSSMS結果差異詳細分析ツール
同一条件での実行間差異の原因を特定
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback
from pathlib import Path

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def analyze_dssms_variance():
    """DSSMS結果差異の詳細分析"""
    
    print("🔍 DSSMS結果差異詳細分析開始")
    print("=" * 60)
    
    # 複数回実行して差異を分析
    results = []
    
    try:
        from src.dssms.dssms_backtester import DSSMSBacktester
        
        # 固定テスト条件
        symbols = ['6758', '7203', '9984']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        print(f"📅 分析期間: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        print(f"📊 対象銘柄: {symbols}")
        
        # 5回連続実行
        for i in range(5):
            print(f"\n🚀 実行 {i+1}/5...")
            
            backtester = DSSMSBacktester(config={
                'deterministic_mode': True,
                'random_seed': 42,
                'initial_capital': 1000000
            })
            
            result = backtester.simulate_dynamic_selection(
                start_date=start_date,
                end_date=end_date,
                symbol_universe=symbols
            )
            
            # 結果記録
            results.append({
                'run': i + 1,
                'final_value': result.get('final_portfolio_value', 0),
                'total_return': result.get('total_return', 0),
                'switches': result.get('total_switches', 0),
                'switch_cost': result.get('total_switch_cost', 0),
                'volatility': result.get('volatility', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'vs_7203': result.get('comparison_results', {}).get('vs_7203', 0),
                'vs_9984': result.get('comparison_results', {}).get('vs_9984', 0),
                'vs_6758': result.get('comparison_results', {}).get('vs_6758', 0)
            })
            
            print(f"   最終価値: {result.get('final_portfolio_value', 0):,.0f}円")
            print(f"   総リターン: {result.get('total_return', 0):.4f} ({result.get('total_return', 0)*100:.2f}%)")
            print(f"   スイッチ回数: {result.get('total_switches', 0)}回")
        
        # 差異分析
        print(f"\n📊 結果差異分析:")
        print("=" * 40)
        
        df = pd.DataFrame(results)
        
        # 統計計算
        for col in ['final_value', 'total_return', 'switches', 'switch_cost', 'volatility', 'max_drawdown', 'sharpe_ratio', 'vs_7203', 'vs_9984', 'vs_6758']:
            if col in df.columns:
                values = df[col]
                print(f"\n{col}:")
                print(f"  平均: {values.mean():.6f}")
                print(f"  標準偏差: {values.std():.6f}")
                print(f"  最小値: {values.min():.6f}")
                print(f"  最大値: {values.max():.6f}")
                print(f"  差異範囲: {values.max() - values.min():.6f}")
                print(f"  変動係数: {(values.std() / values.mean() * 100) if values.mean() != 0 else 0:.3f}%")
        
        # 詳細結果表示
        print(f"\n📋 実行別詳細結果:")
        print("-" * 80)
        for i, result in enumerate(results):
            print(f"実行{i+1}: 最終価値={result['final_value']:,.0f}円, " +
                  f"リターン={result['total_return']:.4f}, " +
                  f"vs_7203={result['vs_7203']:.2f}%, " +
                  f"vs_9984={result['vs_9984']:.2f}%, " +
                  f"vs_6758={result['vs_6758']:.2f}%")
        
        # 潜在的原因分析
        print(f"\n🔍 潜在的原因分析:")
        print("=" * 40)
        
        # 1. データ取得タイミング
        print("1. データ取得タイミング差異:")
        print("   - yfinanceからのデータ取得時刻によるスナップショット差異")
        print("   - 市場データの更新タイミングによる価格差")
        
        # 2. 浮動小数点計算
        print("\n2. 浮動小数点計算精度:")
        print("   - 累積計算での丸め誤差")
        print("   - プラットフォーム依存の数値計算差異")
        
        # 3. 外部依存システム
        print("\n3. 外部システム依存:")
        print("   - ネットワーク遅延によるデータ取得順序差異")
        print("   - システムリソース状況による処理順序変動")
        
        # 4. 未制御ランダム要素
        print("\n4. 未制御ランダム要素:")
        print("   - ハッシュ関数の実装差異")
        print("   - 日時計算での微小な差異")
        
        # 差異の深刻度評価
        max_return_diff = df['total_return'].max() - df['total_return'].min()
        max_final_value_diff = df['final_value'].max() - df['final_value'].min()
        
        print(f"\n⚠️ 差異深刻度評価:")
        print(f"   総リターン差異: {max_return_diff:.6f} ({max_return_diff*100:.4f}%)")
        print(f"   最終価値差異: {max_final_value_diff:,.0f}円")
        
        if max_return_diff > 0.001:  # 0.1%以上
            print("   🚨 重大: 実取引に影響する可能性があります")
        elif max_return_diff > 0.0001:  # 0.01%以上
            print("   ⚠️ 中程度: 長期的に影響する可能性があります")
        else:
            print("   ✅ 軽微: 許容範囲内の差異です")
        
        # 推奨対策
        print(f"\n💡 推奨対策:")
        print("1. データキャッシュシステム導入")
        print("2. 固定タイムスタンプでのデータ取得")
        print("3. 高精度数値計算ライブラリ使用")
        print("4. 完全オフライン実行環境構築")
        print("5. 結果検証機能強化")
        
        return {
            'success': True,
            'results': results,
            'variance_analysis': df.describe().to_dict(),
            'max_return_diff': max_return_diff,
            'max_final_value_diff': max_final_value_diff
        }
        
    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")
        print(f"📋 詳細:")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    analysis_results = analyze_dssms_variance()
    
    if analysis_results['success']:
        print(f"\n🎉 DSSMS差異分析完了: 成功")
    else:
        print(f"\n💥 DSSMS差異分析完了: 失敗")
        sys.exit(1)
