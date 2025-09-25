#!/usr/bin/env python3
"""
HierarchicalRankingSystem テストスクリプト
"""

import sys
sys.path.append('src')

try:
    from dssms.hierarchical_ranking_system import HierarchicalRankingSystem
    print("=== HierarchicalRankingSystem テスト ===")
    print("✓ インポート成功")
    
    # インスタンス化テスト（設定辞書付き）
    test_config = {
        'ranking_system': {
            'scoring_weights': {
                "fundamental": 0.40,
                "technical": 0.30,
                "volume": 0.20,
                "volatility": 0.10
            }
        }
    }
    ranking_system = HierarchicalRankingSystem(test_config)
    print("✓ インスタンス化成功")
    
    # 利用可能メソッド確認
    public_methods = [m for m in dir(ranking_system) if not m.startswith("_")]
    print(f"利用可能メソッド: {public_methods}")
    
    print("✓ HierarchicalRankingSystem 基本テスト完了")
    
except ImportError as e:
    print(f"✗ インポートエラー: {e}")
except Exception as e:
    print(f"✗ エラー: {e}")