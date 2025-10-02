#!/usr/bin/env python3
"""
DSSMS統合システムの実在課題調査テスト
"""

import sys
import os
from datetime import datetime

# プロジェクトパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_hierarchical_ranking_integration():
    """HierarchicalRankingSystem統合問題調査"""
    print("=== HierarchicalRankingSystem統合調査 ===\n")
    
    print("1. HierarchicalRankingSystemの初期化テスト:")
    try:
        from dssms.hierarchical_ranking_system import HierarchicalRankingSystem
        
        # 引数なしで初期化を試行（AdvancedRankingEngineで行われているもの）
        try:
            system = HierarchicalRankingSystem()
            print("   ✗ 問題：引数なし初期化が成功してしまいました（設計上エラーになるべき）")
        except Exception as e:
            print(f"   ✓ 予想通りエラー: {e}")
            print("   → AdvancedRankingEngineの統合コードに問題があります")
        
        # 正しい初期化方法
        config = {"ranking_system": {"scoring_weights": {"fundamental": 0.4, "technical": 0.3, "volume": 0.2, "volatility": 0.1}}}
        system = HierarchicalRankingSystem(config)
        print("   ✓ 正しい設定での初期化成功")
        
        # 主要メソッドの動作確認
        print("\n2. 主要メソッド動作確認:")
        try:
            result = system.get_top_candidate(available_funds=1000000)
            print(f"   ✓ get_top_candidate: {result}")
        except Exception as e:
            print(f"   ✗ get_top_candidate エラー: {e}")
            
        try:
            backups = system.get_backup_candidates(n=3)
            print(f"   ✓ get_backup_candidates: {backups}")
        except Exception as e:
            print(f"   ✗ get_backup_candidates エラー: {e}")
            
    except ImportError as e:
        print(f"   ✗ インポートエラー: {e}")

def test_random_selection_issue():
    """ランダム選択問題調査"""
    print("\n=== エントリーポイント ランダム選択調査 ===\n")
    
    try:
        from dssms.dssms_integrated_main import DSSMSIntegratedSystem
        
        print("1. DSSMSIntegratedSystemのランダム選択確認:")
        
        # ソースコード内のランダム選択を確認
        import inspect
        source = inspect.getsource(DSSMSIntegratedSystem)
        
        if "random.choice" in source:
            print("   ✓ 問題確認：random.choice がソースコード内に存在")
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if "random.choice" in line:
                    print(f"   → 行 {i+1}: {line.strip()}")
        else:
            print("   ✗ ランダム選択は見つかりませんでした")
            
    except ImportError as e:
        print(f"   ✗ インポートエラー: {e}")

def test_fallback_score_calculation():
    """フォールバックスコア計算問題調査"""
    print("\n=== フォールバックスコア計算調査 ===\n")
    
    try:
        from dssms.dssms_backtester import DSSMSBacktester
        
        print("1. フォールバックスコア計算メソッド確認:")
        
        # メソッドの存在確認
        if hasattr(DSSMSBacktester, '_calculate_market_based_fallback_score'):
            print("   ✓ _calculate_market_based_fallback_score メソッド存在")
            
            # メソッドのソースコード確認
            import inspect
            method_source = inspect.getsource(DSSMSBacktester._calculate_market_based_fallback_score)
            
            if "0.3 + random.random() * 0.4" in method_source:
                print("   ✓ 問題確認：限定的なランダムスコア生成(0.3-0.7範囲)を発見")
            elif "random" in method_source:
                print("   ✓ 問題確認：ランダム要素を含むスコア計算を発見")
            else:
                print("   ? ランダム要素は見つかりませんでした")
                
        else:
            print("   ✗ フォールバックスコア計算メソッドが見つかりません")
            
    except ImportError as e:
        print(f"   ✗ インポートエラー: {e}")

def test_integration_bridge():
    """統合ブリッジの問題調査"""
    print("\n=== 統合システム強化調査 ===\n")
    
    try:
        from dssms.advanced_ranking_system.integration_bridge import IntegrationBridge
        
        print("1. IntegrationBridge 実装確認:")
        
        # 初期化テスト
        try:
            bridge = IntegrationBridge()
            print("   ✓ IntegrationBridge 初期化成功")
            
            # 統合状況確認
            if hasattr(bridge, 'get_integration_status'):
                status = bridge.get_integration_status()
                print(f"   統合状況: {status}")
            else:
                print("   ✗ 統合状況確認メソッドが見つかりません")
                
        except Exception as e:
            print(f"   ✗ IntegrationBridge 初期化エラー: {e}")
            
    except ImportError as e:
        print(f"   ✗ IntegrationBridge インポートエラー: {e}")

def main():
    """メイン調査実行"""
    print("DSSMS 残存課題 実在性調査")
    print("=" * 50)
    print(f"調査実行日時: {datetime.now()}")
    print()
    
    test_hierarchical_ranking_integration()
    test_random_selection_issue()
    test_fallback_score_calculation()
    test_integration_bridge()
    
    print("\n" + "=" * 50)
    print("調査完了")

if __name__ == "__main__":
    main()