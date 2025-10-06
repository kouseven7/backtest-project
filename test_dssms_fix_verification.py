#!/usr/bin/env python3
"""
DSSMSReportGenerator インデント修正後の検証テスト
Created: 2025-10-06 by Agent
Purpose: TODO-REPORT-001 修正後検証
"""

import sys
sys.path.insert(0, '.')

from src.dssms.dssms_report_generator import DSSMSReportGenerator

def test_indentation_fix():
    """インデント修正後の検証"""
    print('🔍 修正後DSSMSReportGenerator検証')
    print('=' * 50)
    
    try:
        generator = DSSMSReportGenerator()
        
        print('🔍 対象メソッド存在確認:')
        target_methods = [
            '_analyze_concentration_risk', 
            '_analyze_strategy_combinations', 
            '_calculate_advanced_performance_metrics'
        ]
        
        all_exists = True
        for method in target_methods:
            exists = hasattr(generator, method)
            status = '✅ 存在' if exists else '❌ 不存在'
            print(f'  {method}: {status}')
            if not exists:
                all_exists = False
        
        method_count = len([m for m in dir(generator) if not m.startswith('__')])
        print(f'\n📋 修正後メソッド総数: {method_count}')
        
        if all_exists:
            print('\n🎉 インデント修正成功！全メソッドがアクセス可能になりました')
            
            # 簡単な実行テスト
            print('\n🧪 メソッド呼び出しテスト:')
            try:
                # 空データでテスト（エラーハンドリング確認）
                result = generator._analyze_concentration_risk([])
                print('  _analyze_concentration_risk: ✅ 実行可能')
                
                result = generator._analyze_strategy_combinations({})
                print('  _analyze_strategy_combinations: ✅ 実行可能')
                
                result = generator._calculate_advanced_performance_metrics({})
                print('  _calculate_advanced_performance_metrics: ✅ 実行可能')
                
                print('\n🎊 TODO-REPORT-001 Stage 2完了！')
                return True
                
            except Exception as e:
                print(f'  メソッド実行エラー: {e}')
                return False
        else:
            print('\n❌ まだ問題が残っています')
            return False
            
    except Exception as e:
        print(f'❌ インポートまたは初期化エラー: {e}')
        return False

if __name__ == "__main__":
    success = test_indentation_fix()
    exit(0 if success else 1)