#!/usr/bin/env python3
"""
TODO #4: OpeningGapStrategy大量エグジット調査実行テスト
"""

import sys
sys.path.append('.')

print('=== TODO #4 OpeningGapStrategy大量エグジット調査 ===')

try:
    from analysis.opening_gap_exit_analysis import investigate_opening_gap_exit_anomaly
    
    print('✅ 調査モジュールインポート成功')
    
    # 調査実行
    judgment = investigate_opening_gap_exit_anomaly()
    
    if judgment:
        print('\n✅ 調査完了')
        is_abnormal = judgment.get('is_abnormal', False)
        severity = judgment.get('severity', 'UNKNOWN')
        print(f'判定結果: {"異常" if is_abnormal else "正常"}')
        print(f'重要度: {severity}')
        
        if is_abnormal:
            print('⚠️ 異常が検出されました - 修正が必要です')
            
            issues = judgment.get('issues_found', [])
            if issues:
                print('\n🔍 検出された問題:')
                for issue in issues:
                    print(f'  - {issue}')
            
            recommendations = judgment.get('recommendations', [])
            if recommendations:
                print('\n💡 推奨対応:')
                for rec in recommendations:
                    print(f'  - {rec}')
        else:
            print('✅ 戦略動作は正常範囲内です')
    else:
        print('❌ 調査が完了しませんでした')
        
except ImportError as e:
    print(f'❌ インポートエラー: {e}')
except Exception as e:
    print(f'❌ 調査エラー: {e}')
    import traceback
    traceback.print_exc()