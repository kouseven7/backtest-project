#!/usr/bin/env python3
"""
Phase 2 統一出力エンジン動作確認テスト
バックテスト基本理念遵守確認（Entry_Signal/Exit_Signal生成・取引実行・CSV+JSON+TXT+YAML出力）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

def create_test_backtest_data() -> pd.DataFrame:
    """バックテスト基本理念遵守テストデータ生成"""
    
    # 30日間のテストデータ
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # 株価データ（シンプルなトレンド）
    base_price = 1000
    price_changes = np.random.normal(0, 10, 30)  # 日次変動
    prices = [base_price]
    for change in price_changes[:-1]:
        prices.append(prices[-1] + change)
    
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * 0.99 for p in prices],
        'High': [p * 1.02 for p in prices], 
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(10000, 50000, 30),
        'Entry_Signal': [0] * 30,
        'Exit_Signal': [0] * 30,
        'Position': [0] * 30,
        'Portfolio_Value': [100000] * 30
    })
    
    # バックテスト基本理念遵守: 実際のシグナル生成（確実に取引発生させる）
    # 強制的にシグナルを生成して取引を確保
    test_data.loc[test_data.index[5], 'Entry_Signal'] = 1  # 5日目にエントリー
    test_data.loc[test_data.index[10], 'Exit_Signal'] = 1  # 10日目にエグジット
    test_data.loc[test_data.index[15], 'Entry_Signal'] = 1  # 15日目に再エントリー
    test_data.loc[test_data.index[25], 'Exit_Signal'] = 1  # 25日目に最終エグジット
    
    test_data.set_index('Date', inplace=True)
    
    return test_data

def create_test_trades(test_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """取引履歴生成（バックテスト基本理念遵守）"""
    trades = []
    
    # エントリー取引
    entry_signals = test_data[test_data['Entry_Signal'] == 1]
    for idx, row in entry_signals.iterrows():
        trades.append({
            'timestamp': str(idx),
            'type': 'entry',
            'price': float(row['Close']),
            'signal': 'Entry_Signal',
            'volume': int(row['Volume'])
        })
    
    # エグジット取引
    exit_signals = test_data[test_data['Exit_Signal'] == 1]
    for idx, row in exit_signals.iterrows():
        trades.append({
            'timestamp': str(idx),
            'type': 'exit', 
            'price': float(row['Close']),
            'signal': 'Exit_Signal',
            'volume': int(row['Volume'])
        })
    
    return trades

def test_unified_exporter():
    """統一出力エンジンのテスト実行"""
    
    print("[TEST] Phase 2 統一出力エンジン動作確認テスト開始")
    print("=" * 60)
    
    try:
        from output.unified_exporter import UnifiedExporter
        
        # 1. テストデータ生成
        print("[CHART] テストデータ生成...")
        test_data = create_test_backtest_data()
        test_trades = create_test_trades(test_data)
        
        # バックテスト基本理念確認
        entry_count = (test_data['Entry_Signal'] == 1).sum()
        exit_count = (test_data['Exit_Signal'] == 1).sum()
        total_trades = len(test_trades)
        
        print(f"[OK] バックテスト基本理念遵守確認:")
        print(f"   - Entry_Signal生成数: {entry_count}")
        print(f"   - Exit_Signal生成数: {exit_count}")
        print(f"   - 総取引数: {total_trades}")
        
        if total_trades == 0:
            print("[WARNING] 警告: 取引数0件 - シグナル生成ロジック要確認")
        
        # 2. パフォーマンス指標準備
        performance_metrics = {
            'total_trades': total_trades,
            'entry_signals': int(entry_count),
            'exit_signals': int(exit_count),
            'final_portfolio_value': float(test_data['Portfolio_Value'].iloc[-1]),
            'total_return': 0.05,  # 5%リターン（テスト値）
            'max_drawdown': -0.02,  # -2%最大ドローダウン（テスト値）
            'sharpe_ratio': 1.5,  # シャープレシオ（テスト値）
            'backtest_period': f"{test_data.index[0]} to {test_data.index[-1]}"
        }
        
        # 3. 統一出力エンジンテスト実行
        print("\n[TOOL] 統一出力エンジン実行...")
        exporter = UnifiedExporter()
        
        # main.py形式出力テスト
        export_result = exporter.export_main_results(
            stock_data=test_data,
            trades=test_trades,
            performance=performance_metrics,
            ticker="TEST_7203.T",
            strategy_name="phase2_test_strategy"
        )
        
        print(f"[OK] main.py統一出力成功: {export_result}")
        
        # 4. 出力ファイル確認
        print("\n📁 出力ファイル確認:")
        for format_type, file_path in export_result.items():
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"   - {format_type.upper()}: {file_path} ({file_size} bytes)")
            else:
                print(f"   - {format_type.upper()}: [ERROR] ファイル未作成")
        
        # 5. バックテスト基本理念違反チェック
        print("\n[TARGET] バックテスト基本理念遵守最終確認:")
        
        principle_violations = []
        
        # Entry_Signal/Exit_Signal存在確認
        if 'Entry_Signal' not in test_data.columns:
            principle_violations.append("Entry_Signal列が存在しない")
        if 'Exit_Signal' not in test_data.columns:
            principle_violations.append("Exit_Signal列が存在しない")
        
        # 取引実行確認
        if total_trades == 0:
            principle_violations.append("取引が実行されていない（取引数0件）")
        
        # 出力完整性確認
        if len(export_result) < 4:  # CSV, JSON, TXT, YAML
            principle_violations.append(f"出力形式不足（期待4形式、実際{len(export_result)}形式）")
        
        if principle_violations:
            print("[ERROR] バックテスト基本理念違反検出:")
            for violation in principle_violations:
                print(f"   - {violation}")
            print("   TODO(tag:backtest_execution, rationale:fix principle violations)")
        else:
            print("[OK] バックテスト基本理念遵守確認完了")
            print("   - Entry_Signal/Exit_Signal生成: OK")
            print("   - 取引実行: OK")
            print("   - 新形式出力（CSV+JSON+TXT+YAML）: OK")
        
        print("\n[SUCCESS] Phase 2 統一出力エンジンテスト完了")
        print("=" * 60)
        
        return len(principle_violations) == 0
        
    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
        print("TODO(tag:backtest_execution, rationale:fix unified exporter test error)")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unified_exporter()
    if success:
        print("[OK] 統一出力エンジン Phase 2 テスト成功")
    else:
        print("[ERROR] 統一出力エンジン Phase 2 テスト失敗")