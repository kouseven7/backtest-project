#!/usr/bin/env python3
"""
レビューシステムのテスト
"""

import os
import sys

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from tools.parameter_reviewer import ParameterReviewer

def test_review_system():
    print("🧪 レビューシステムのテストを開始...")
    
    try:
        # レビューアインスタンスを作成
        reviewer = ParameterReviewer()
        print("✅ ParameterReviewerの初期化成功")
        
        # レビュー対象ファイルの確認
        available_configs = reviewer.parameter_manager.list_available_configs(
            strategy_name="momentum",
            status="pending_review"
        )
        
        print(f"📋 レビュー対象ファイル数: {len(available_configs)}")
        
        if available_configs:
            for i, config in enumerate(available_configs):
                print(f"  {i+1}. {config['filename']}")
                print(f"     - 戦略: {config.get('strategy', 'N/A')}")
                print(f"     - ティッカー: {config.get('ticker', 'N/A')}")
                print(f"     - 作成日: {config.get('created_at', 'N/A')}")
                print(f"     - ステータス: {config.get('status', 'N/A')}")
        else:
            print("❌ レビュー待ちのファイルがありません")
            
        # 第一ファイルの詳細確認（もしあれば）
        if available_configs:
            first_config = available_configs[0]
            print(f"\n📊 詳細情報（{first_config['filename']}）:")
            
            # パラメータ情報
            params = first_config.get('parameters', {})
            print(f"  📈 パラメータ:")
            for key, value in params.items():
                print(f"    - {key}: {value}")
            
            # パフォーマンス指標
            metrics = first_config.get('performance_metrics', {})
            print(f"  📊 パフォーマンス:")
            for key, value in metrics.items():
                print(f"    - {key}: {value}")
                
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_review_system()
