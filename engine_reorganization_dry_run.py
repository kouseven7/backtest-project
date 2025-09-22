# 段階的エンジン整理実行
import sys
sys.path.append('src/dssms')
from engine_audit_manager import EngineAuditManager
import json

def run_engine_reorganization_dry_run():
    """エンジン整理のDry Runを実行"""
    print('=== 段階的エンジン整理実行 ===')
    
    # エンジン監査マネージャ初期化
    audit_manager = EngineAuditManager()
    
    # 評価済みデータ再利用
    audit_manager.audit_all_engines()
    classification = audit_manager.classify_engines()
    
    print(f'採用エンジン: {len(classification["adopted"])}個')
    print(f'アーカイブ対象: {len(classification["archived"])}個')
    print(f'削除対象: {len(classification["deprecated"])}個')
    
    # Dry Run実行
    print('')
    print('=== Dry Run実行 ===')
    dry_run_result = audit_manager.execute_reorganization(classification, dry_run=True)
    print(f'Dry Run結果: {len(dry_run_result["actions"])}アクション, {len(dry_run_result["errors"])}エラー')
    
    for action in dry_run_result['actions']:
        print(f'  {action["type"]}: {action["source"]}')
    
    if dry_run_result['errors']:
        print('\nエラー:')
        for error in dry_run_result['errors']:
            print(f'  ERROR: {error}')
    
    print('')
    print('=== 実際の整理実行準備完了 ===')
    
    return dry_run_result

if __name__ == "__main__":
    result = run_engine_reorganization_dry_run()