#!/usr/bin/env python3
"""
パラメータ管理の直接テスト（parameter_reviewer.pyをインポートしない）
"""

import json
import os
import sys
from datetime import datetime

# プロジェクトルートの設定
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.optimized_parameters import OptimizedParameterManager

def test_parameter_manager():
    """パラメータマネージャーのテスト"""
    print("🧪 パラメータマネージャーのテストを開始...")
    
    try:
        manager = OptimizedParameterManager()
        print("✅ OptimizedParameterManagerの初期化成功")
        
        # 全ファイルを確認
        print("\n📂 利用可能なファイル一覧:")
        all_configs = manager.list_available_configs()
        print(f"  総ファイル数: {len(all_configs)}")
        
        for i, config in enumerate(all_configs):
            print(f"  {i+1}. {config['filename']}")
            print(f"     戦略: {config.get('strategy', 'N/A')}")
            print(f"     ティッカー: {config.get('ticker', 'N/A')}")
            print(f"     ステータス: {config.get('status', 'N/A')}")
            print(f"     作成日: {config.get('created_at', 'N/A')}")
            print()
        
        # pending_reviewのファイルを確認
        print("📋 レビュー待ちファイル:")
        pending_configs = manager.list_available_configs(status="pending_review")
        print(f"  レビュー待ちファイル数: {len(pending_configs)}")
        
        for config in pending_configs:
            print(f"  • {config['filename']}")
            print(f"    戦略: {config.get('strategy', 'N/A')}")
            print(f"    パラメータ数: {len(config.get('parameters', {}))}")
            
            # パラメータ詳細
            params = config.get('parameters', {})
            if params:
                print(f"    パラメータ:")
                for key, value in params.items():
                    print(f"      - {key}: {value}")
            print()
        
        # 戦略名別検索テスト
        print("🔍 戦略名別検索テスト:")
        
        # MomentumInvestingStrategyで検索
        momentum_configs = manager.list_available_configs(
            strategy_name="MomentumInvestingStrategy",
            status="pending_review"
        )
        print(f"  MomentumInvestingStrategy: {len(momentum_configs)}件")
        
        # momentumで検索
        momentum_short_configs = manager.list_available_configs(
            strategy_name="momentum",
            status="pending_review"  
        )
        print(f"  momentum: {len(momentum_short_configs)}件")
        
        # Momentumで検索
        momentum_capital_configs = manager.list_available_configs(
            strategy_name="Momentum",
            status="pending_review"
        )
        print(f"  Momentum: {len(momentum_capital_configs)}件")
        
        return len(pending_configs) > 0
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def simulate_review_process():
    """レビュープロセスのシミュレーション"""
    print("\n🔄 レビュープロセスのシミュレーション...")
    
    try:
        manager = OptimizedParameterManager()
        
        # レビュー待ちファイルを取得
        pending_configs = manager.list_available_configs(status="pending_review")
        
        if not pending_configs:
            print("❌ レビュー待ちファイルがありません")
            return False
        
        first_config = pending_configs[0]
        print(f"📄 テスト対象: {first_config['filename']}")
        
        # 承認をシミュレート
        first_config['status'] = 'approved'
        first_config['approval_info'] = {
            'approved_by': 'test_reviewer',
            'approved_at': datetime.now().isoformat(),
            'rejection_reason': None
        }
        
        # ファイルに保存
        filepath = first_config['filepath']
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                k: v for k, v in first_config.items() 
                if k not in ['filename', 'filepath']
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {first_config['filename']} を承認済みに変更しました")
        
        # 再度確認
        updated_configs = manager.list_available_configs(status="pending_review")
        print(f"📊 残りのレビュー待ちファイル: {len(updated_configs)}件")
        
        return True
        
    except Exception as e:
        print(f"❌ シミュレーションでエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("📋 パラメータ管理システムテスト")
    print("="*60)
    
    # 基本テست
    has_pending_files = test_parameter_manager()
    
    if has_pending_files:
        print("\n" + "="*60)
        # レビュープロセスのシミュレーション
        simulate_review_process()
    
    print("\n✅ テスト完了")
