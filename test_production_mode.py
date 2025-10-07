#!/usr/bin/env python3
"""
Production mode対応テストスクリプト
"""

import sys
import os
import json
from pathlib import Path

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

from config.multi_strategy_manager import MultiStrategyManager

def test_production_readiness():
    """Production準備状況のテスト"""
    print("=== Production Mode対応テスト開始 ===")
    
    try:
        # 1. Development modeでテスト
        print("\n1. Development modeテスト:")
        config_path = "config/main_integration_config.json"
        manager = MultiStrategyManager(config_path=config_path)
        
        # システム初期化
        init_result = manager.initialize_system()
        print(f"   初期化結果: {init_result}")
        
        # Production準備状況確認
        readiness = manager.get_production_readiness_status()
        print(f"   システムモード: {readiness['system_mode']}")
        print(f"   Production準備度: {readiness['overall_ready']}")
        print(f"   フォールバックポリシー利用可: {readiness['fallback_policy_available']}")
        print(f"   フォールバック使用量: {readiness['fallback_usage_statistics']['total_failures']}")
        
        # 2. Production mode設定変更テスト
        print("\n2. Production mode設定変更テスト:")
        
        # 設定ファイル一時変更
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        original_mode = config.get('system_mode', 'development')
        config['system_mode'] = 'production'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Production modeでマネージャー作成
        prod_manager = MultiStrategyManager(config_path=config_path)
        prod_init = prod_manager.initialize_system()
        
        print(f"   Production mode初期化: {prod_init}")
        
        # Production mode設定確認
        if hasattr(prod_manager, 'system_mode') and prod_manager.system_mode:
            print(f"   システムモード: {prod_manager.system_mode.value}")
        if hasattr(prod_manager, 'production_constraints'):
            print(f"   フォールバック禁止: {prod_manager.production_constraints.get('fallback_forbidden', False)}")
            print(f"   即座停止設定: {prod_manager.production_constraints.get('immediate_failure_on_error', False)}")
        
        # 設定を元に戻す
        config['system_mode'] = original_mode
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("\n3. フォールバック統計確認:")
        if hasattr(manager, 'fallback_policy') and manager.fallback_policy:
            stats = manager.fallback_policy.get_usage_statistics()
            print(f"   総失敗数: {stats['total_failures']}")
            print(f"   フォールバック成功数: {stats['successful_fallbacks']}")
            print(f"   フォールバック使用率: {stats['fallback_usage_rate']:.1%}")
        
        print("\n=== Production Mode対応テスト完了 ===")
        print("✅ 全テスト成功")
        
        return True
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_production_readiness()
    sys.exit(0 if success else 1)