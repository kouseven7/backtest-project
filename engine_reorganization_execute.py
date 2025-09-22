# 段階的エンジン整理実行（実行版）
import sys
sys.path.append('src/dssms')
from engine_audit_manager import EngineAuditManager
import json

def run_engine_reorganization_execute():
    """エンジン整理を実際に実行"""
    print('=== 段階的エンジン整理実行（実際のファイル移動） ===')
    
    # エンジン監査マネージャ初期化
    audit_manager = EngineAuditManager()
    
    # 評価済みデータ再利用
    audit_manager.audit_all_engines()
    classification = audit_manager.classify_engines()
    
    print(f'採用エンジン: {len(classification["adopted"])}個')
    print(f'アーカイブ対象: {len(classification["archived"])}個')
    print(f'削除対象: {len(classification["deprecated"])}個')
    
    # 実際の整理実行
    print('')
    print('=== 実際の整理実行中 ===')
    actual_result = audit_manager.execute_reorganization(classification, dry_run=False)
    print(f'実行結果: {len(actual_result["actions"])}アクション完了, {len(actual_result["errors"])}エラー')
    
    for action in actual_result['actions']:
        print(f'  完了: {action["type"]} - {action["source"]} -> {action["destination"]}')
    
    if actual_result['errors']:
        print('\nエラー:')
        for error in actual_result['errors']:
            print(f'  ERROR: {error}')
    
    # 整理後の統計
    print('')
    print('=== 整理完了統計 ===')
    reorganization_stats = audit_manager.get_reorganization_statistics()
    print(f'整理率: {reorganization_stats["reorganization_percentage"]:.1f}%')
    print(f'採用エンジン: {reorganization_stats["adopted_count"]}個')
    print(f'アーカイブエンジン: {reorganization_stats["archived_count"]}個')
    print(f'削除エンジン: {reorganization_stats["deprecated_count"]}個')
    
    print('')
    print('=== Problem 13 エンジン競合解決完了 ===')
    
    return actual_result, reorganization_stats

if __name__ == "__main__":
    result, stats = run_engine_reorganization_execute()