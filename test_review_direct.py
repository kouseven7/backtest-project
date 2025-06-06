#!/usr/bin/env python3
"""
レビューシステムの直接テスト
"""

import os
import sys

# プロジェクトルートの設定
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from tools.parameter_reviewer import ParameterReviewer

def test_review_direct():
    """レビューシステムを直接テスト"""
    print("🧪 レビューシステムの直接テストを開始...")
    
    try:
        reviewer = ParameterReviewer()
        
        # MomentumInvestingStrategy戦略のレビュー対象ファイルをチェック
        print("\n1. MomentumInvestingStrategy で検索...")
        configs = reviewer.parameter_manager.list_available_configs(
            strategy_name="MomentumInvestingStrategy",
            status="pending_review"
        )
        print(f"  見つかったファイル数: {len(configs)}")
        
        # momentum戦略のレビュー対象ファイルをチェック
        print("\n2. momentum で検索...")
        configs2 = reviewer.parameter_manager.list_available_configs(
            strategy_name="momentum",
            status="pending_review"
        )
        print(f"  見つかったファイル数: {len(configs2)}")
        
        # 戦略名なしで検索
        print("\n3. 戦略名なしで検索...")
        configs3 = reviewer.parameter_manager.list_available_configs(
            status="pending_review"
        )
        print(f"  見つかったファイル数: {len(configs3)}")
        
        # 全ファイルを確認
        print("\n4. 全ファイルを確認...")
        all_configs = reviewer.parameter_manager.list_available_configs()
        print(f"  全ファイル数: {len(all_configs)}")
        
        for config in all_configs:
            print(f"    ファイル: {config['filename']}")
            print(f"    戦略: {config.get('strategy', 'N/A')}")
            print(f"    ステータス: {config.get('status', 'N/A')}")
            print()
        
        # 使用可能な場合はレビューをテスト
        if configs:
            print("\n5. 実際のレビューをテスト...")
            print("最初のレビュー対象ファイル:")
            first_config = configs[0]
            print(f"  ファイル名: {first_config['filename']}")
            print(f"  戦略: {first_config.get('strategy', 'N/A')}")
            print(f"  パラメータ: {first_config.get('parameters', {})}")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_review_direct()
