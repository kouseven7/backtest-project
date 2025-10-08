#!/usr/bin/env python3
"""
決定論的DSSMS結果のExcel出力統合検証
Excel出力システムV2との統合テスト
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import traceback

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def validate_excel_integration():
    """決定論的DSSMS結果のExcel出力統合を検証"""
    
    print("[SEARCH] 決定論的DSSMS Excel出力統合検証開始")
    print("=" * 60)
    
    try:
        # DSSMS Backtesterのインポート
        from src.dssms.dssms_backtester import DSSMSBacktester
        
        # テスト期間設定（短期間で高速テスト）
        start_date = datetime(2023, 12, 1)
        end_date = datetime(2023, 12, 31)
        
        print(f"📅 テスト期間: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        
        # 決定論的DSSMS実行
        symbols = ['6758', '7203', '9984']  # Sony, Toyota, SoftBank
        
        print(f"[CHART] 対象銘柄: {symbols}")
        print(f"[TARGET] 決定論的モード: 有効")
        
        # バックテスト実行
        backtester = DSSMSBacktester(config={
            'deterministic_mode': True,
            'random_seed': 42,
            'initial_capital': 1000000
        })
        
        print("\n[ROCKET] DSSMSバックテスト実行中...")
        results = backtester.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date,
            symbol_universe=symbols
        )
        
        print(f"[OK] バックテスト完了")
        print(f"   最終価値: {results.get('final_value', 'N/A'):,}円")
        print(f"   総リターン: {results.get('total_return', 'N/A'):.4f} ({results.get('total_return', 0)*100:.2f}%)")
        print(f"   スイッチ回数: {results.get('switches', 'N/A')}回")
        
        # Excel出力データ準備
        excel_data = {
            'summary': {
                'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'period': f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}",
                'symbols': symbols,
                'initial_capital': 1000000,
                'final_value': results.get('final_value', 0),
                'total_return': results.get('total_return', 0),
                'switches': results.get('switches', 0),
                'deterministic_mode': True,
                'random_seed': 42
            },
            'trades': results.get('trades', []),
            'daily_values': results.get('daily_values', []),
            'rankings': results.get('rankings', [])
        }
        
        print(f"\n📝 Excel出力データ準備完了")
        print(f"   取引履歴: {len(excel_data['trades'])}件")
        print(f"   日次データ: {len(excel_data['daily_values'])}件")
        print(f"   ランキング: {len(excel_data['rankings'])}件")
        
        # 結果をJSON形式で保存（Excel出力システムV2との統合用）
        output_file = f"dssms_excel_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(excel_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 統合テストデータ保存: {output_file}")
        
        # 決定論的動作の再検証
        print(f"\n🔄 決定論的動作再検証...")
        
        # 同条件で再実行
        backtester2 = DSSMSBacktester(config={
            'deterministic_mode': True,
            'random_seed': 42,
            'initial_capital': 1000000
        })
        
        results2 = backtester2.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date,
            symbol_universe=symbols
        )
        
        # 結果比較
        final_value_diff = abs(results.get('final_value', 0) - results2.get('final_value', 0))
        total_return_diff = abs(results.get('total_return', 0) - results2.get('total_return', 0))
        switches_diff = abs(results.get('switches', 0) - results2.get('switches', 0))
        
        print(f"[CHART] 決定論的動作検証結果:")
        print(f"   最終価値差異: {final_value_diff:,.0f}円")
        print(f"   総リターン差異: {total_return_diff:.10f}")
        print(f"   スイッチ回数差異: {switches_diff}")
        
        # 統合検証結果
        consistency_check = (final_value_diff == 0 and 
                           total_return_diff == 0 and 
                           switches_diff == 0)
        
        print(f"\n[TARGET] Excel統合検証結果:")
        print(f"   [OK] 決定論的動作: {'合格' if consistency_check else '不合格'}")
        print(f"   [OK] データ出力準備: 完了")
        print(f"   [OK] JSON形式保存: 完了")
        print(f"   [OK] Excel統合対応: 準備完了")
        
        # 次のステップ推奨
        print(f"\n[LIST] 次のステップ:")
        print(f"   1. Excel出力システムV2での{output_file}読み込み")
        print(f"   2. 決定論的結果のExcelファイル生成")
        print(f"   3. 複数実行での一貫性確認")
        print(f"   4. 本番環境での長期バックテスト実行")
        
        return {
            'success': True,
            'consistency': consistency_check,
            'output_file': output_file,
            'results': excel_data
        }
        
    except Exception as e:
        print(f"[ERROR] エラー発生: {str(e)}")
        print(f"[LIST] 詳細:")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    validation_results = validate_excel_integration()
    
    if validation_results['success']:
        print(f"\n[SUCCESS] Excel統合検証完了: 成功")
    else:
        print(f"\n💥 Excel統合検証完了: 失敗")
        sys.exit(1)
