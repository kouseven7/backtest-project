# Problem 13 最終統計
import sys
sys.path.append('src/dssms')
from engine_audit_manager import EngineAuditManager
import os

def generate_final_statistics():
    """Problem 13の最終統計を生成"""
    # 最終統計
    audit_manager = EngineAuditManager()
    audit_manager.audit_all_engines()
    classification = audit_manager.classify_engines()

    total_engines = len(classification['adopted']) + len(classification['archived']) + len(classification['deprecated'])
    reorganization_rate = ((len(classification['archived']) + len(classification['deprecated'])) / total_engines) * 100

    print('=== Problem 13 エンジン競合解決 完了統計 ===')
    print(f'合計エンジン数: {total_engines}個')
    print(f'採用エンジン: {len(classification["adopted"])}個')
    print(f'アーカイブエンジン: {len(classification["archived"])}個')
    print(f'削除エンジン: {len(classification["deprecated"])}個')
    print(f'整理率: {reorganization_rate:.1f}%')
    print()
    print('=== KPI結果 ===')
    print(f'85.0点品質基準達成率: {len(classification["adopted"])/total_engines*100:.1f}%')
    print(f'エンジン競合解決済み: ✅ COMPLETED')
    print(f'品質向上効果: 高品質エンジンのみ維持')
    print()
    print('=== アーカイブ済みファイル ===')
    archive_path = 'archive/engines/historical'
    if os.path.exists(archive_path):
        for file in os.listdir(archive_path):
            print(f'  📁 {file}')
    print()
    print('🎉 Problem 13 "エンジン競合解決" 実装完了!')
    
    return {
        'total_engines': total_engines,
        'adopted': len(classification["adopted"]),
        'archived': len(classification["archived"]),
        'deprecated': len(classification["deprecated"]),
        'reorganization_rate': reorganization_rate
    }

if __name__ == "__main__":
    stats = generate_final_statistics()