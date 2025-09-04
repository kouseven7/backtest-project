"""
Phase 2.3 実装テスト
データ抽出エンハンサーの動作確認

Purpose: 新しいデータ抽出機能の動作テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Phase 2.3の新機能をテスト
from output.data_extraction_enhancer import MainDataExtractor, extract_and_analyze_main_data

def create_test_data():
    """テスト用のデータ生成"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    # 基本的な株価データ
    np.random.seed(42)
    prices = 1000 + np.cumsum(np.random.randn(len(dates)) * 10)
    
    data = {
        'Close': prices,
        'Entry_Signal': [0] * len(dates),
        'Exit_Signal': [0] * len(dates),
        'Strategy': ['TestStrategy'] * len(dates)
    }
    
    # いくつかのエントリー・エグジットシグナルを設定
    data['Entry_Signal'][5] = 1   # 1回目のエントリー
    data['Exit_Signal'][10] = 1   # 1回目のエグジット
    data['Entry_Signal'][15] = 1  # 2回目のエントリー
    data['Exit_Signal'][20] = 1   # 2回目のエグジット
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_data_extraction():
    """データ抽出のテスト"""
    print("=== Phase 2.3 データ抽出エンハンサー テスト ===")
    
    # テストデータ作成
    test_data = create_test_data()
    print(f"テストデータ: {len(test_data)}行, 期間: {test_data.index[0]} - {test_data.index[-1]}")
    
    # エンハンサーでデータ抽出
    result = extract_and_analyze_main_data(test_data, "TEST")
    
    print(f"\n📊 解析結果:")
    print(f"  銘柄: {result['ticker']}")
    print(f"  データ品質: {result['data_quality']}")
    print(f"  取引数: {len(result['trades'])}件")
    
    if result['trades']:
        print(f"  取引詳細:")
        for i, trade in enumerate(result['trades']):
            print(f"    取引{i+1}: エントリー{trade['entry_price']:.2f}円 → エグジット{trade['exit_price']:.2f}円")
            print(f"            損益: {trade['pnl']:.2f}円 ({trade['return_pct']*100:.2f}%)")
    
    print(f"\n💰 パフォーマンス:")
    perf = result['performance']
    print(f"  最終ポートフォリオ価値: {perf['final_portfolio_value']:,.0f}円")
    print(f"  総損益: {perf['total_pnl']:,.0f}円")
    print(f"  総リターン: {perf['total_return']*100:.2f}%")
    print(f"  勝率: {perf['win_rate']*100:.1f}%")
    
    # 重要: ゼロ値チェック
    zero_issues = []
    if perf['final_portfolio_value'] == 0:
        zero_issues.append('final_portfolio_value')
    if perf['total_pnl'] == 0 and result['trades']:
        zero_issues.append('total_pnl')
    
    if zero_issues:
        print(f"\n⚠️  ゼロ値問題検出: {zero_issues}")
        print("   → これはExcel出力で修正すべき問題です")
    else:
        print(f"\n✅ ゼロ値問題なし - 正常なデータ抽出")
    
    return result

def test_empty_data():
    """空データのテスト"""
    print(f"\n=== 空データテスト ===")
    
    empty_df = pd.DataFrame()
    result = extract_and_analyze_main_data(empty_df, "EMPTY")
    
    print(f"空データ結果: 取引数={len(result['trades'])}, データ品質={result['data_quality']}")
    
    return result

if __name__ == "__main__":
    print("Phase 2.3 データ抽出エンハンサー テスト実行")
    print("=" * 50)
    
    try:
        # 通常データテスト
        result1 = test_data_extraction()
        
        # 空データテスト
        result2 = test_empty_data()
        
        print(f"\n🎯 テスト完了")
        print(f"   通常データ: 最終価値={result1['performance']['final_portfolio_value']:,.0f}円")
        print(f"   空データ: 最終価値={result2['performance']['final_portfolio_value']:,.0f}円")
        
        # 成功判定
        if result1['performance']['final_portfolio_value'] > 0:
            print(f"✅ Phase 2.3 データ抽出エンハンサー: 正常動作確認")
        else:
            print(f"⚠️  Phase 2.3 データ抽出エンハンサー: 要調整")
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
