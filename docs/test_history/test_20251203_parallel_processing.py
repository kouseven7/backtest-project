"""
並列処理実装のテスト (Phase 1)

Phase 1: 並列処理の実装テスト
- パフォーマンステスト: 実行時間が11秒 → 2-3秒に短縮
- 正確性テスト: 優先度分類結果が正しい
- エラーハンドリングテスト: 個別銘柄の失敗時の挙動

一時テスト: 成功後削除
削除基準:
- [ ] 全アサーション成功
- [ ] 実データ検証完了（モック/ダミー不使用）
- [ ] フォールバックなし動作確認
- [ ] docs/test_history/ に記録済み

Author: Backtest Project Team
Created: 2025-12-03
Last Modified: 2025-12-03
"""
import pytest
import time
import sys
import os
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem


class TestPhase1ParallelProcessing:
    """Phase 1: 並列処理実装のテスト"""
    
    @staticmethod
    def _create_test_config():
        """テスト用設定作成"""
        return {
            'ranking_system': {
                'scoring_weights': {
                    "fundamental": 0.40,
                    "technical": 0.30,
                    "volume": 0.20,
                    "volatility": 0.10
                }
            }
        }
    
    def test_parallel_processing_performance(self):
        """並列処理のパフォーマンステスト"""
        # Setup
        config = self._create_test_config()
        system = HierarchicalRankingSystem(config)
        test_symbols = ['7203.T', '6758.T', '9984.T']  # 3銘柄でテスト
        
        # Execute
        start_time = time.time()
        priority_groups = system.categorize_by_perfect_order_priority(test_symbols)
        execution_time = time.time() - start_time
        
        # Assert
        assert execution_time < 10.0, f"実行時間が長すぎる: {execution_time:.2f}秒 (3銘柄で10秒以内が期待)"
        assert len(priority_groups) == 3, "優先度グループが3つ必要"
        assert sum(len(g) for g in priority_groups.values()) == len(test_symbols), "全銘柄が分類される必要"
        
        print(f"✅ パフォーマンステスト成功: {execution_time:.2f}秒")
        print(f"   優先度グループ: {priority_groups}")
    
    def test_parallel_processing_correctness(self):
        """並列処理の正確性テスト"""
        # Setup
        config = self._create_test_config()
        system = HierarchicalRankingSystem(config)
        test_symbols = ['7203.T', '6758.T']
        
        # Execute
        priority_groups = system.categorize_by_perfect_order_priority(test_symbols)
        
        # Assert
        assert isinstance(priority_groups, dict), "辞書型で返される"
        assert all(isinstance(v, list) for v in priority_groups.values()), "各値はリスト"
        assert all(k in [1, 2, 3] for k in priority_groups.keys()), "優先度は1, 2, 3のみ"
        
        # 全銘柄が分類されている
        total_classified = sum(len(g) for g in priority_groups.values())
        assert total_classified == len(test_symbols), f"全銘柄が分類される必要: {total_classified}/{len(test_symbols)}"
        
        print(f"✅ 正確性テスト成功: {priority_groups}")
    
    def test_parallel_processing_error_handling(self):
        """並列処理のエラーハンドリングテスト"""
        # Setup
        config = self._create_test_config()
        system = HierarchicalRankingSystem(config)
        test_symbols = ['INVALID.T', '7203.T']  # 無効銘柄を含む
        
        # Execute
        priority_groups = system.categorize_by_perfect_order_priority(test_symbols)
        
        # Assert
        # 無効銘柄は優先度3に分類される
        assert 'INVALID.T' in priority_groups[3], "無効銘柄は優先度3に分類"
        
        # 全銘柄が分類されている
        total = sum(len(g) for g in priority_groups.values())
        assert total == len(test_symbols), "全銘柄が分類される"
        
        print(f"✅ エラーハンドリングテスト成功: {priority_groups}")
    
    def test_parallel_processing_data_integrity(self):
        """並列処理のデータ整合性テスト"""
        # Setup
        config = self._create_test_config()
        system = HierarchicalRankingSystem(config)
        test_symbols = ['7203.T', '6758.T', '9984.T']
        
        # Execute 2回実行して結果が一致することを確認
        priority_groups_1 = system.categorize_by_perfect_order_priority(test_symbols)
        priority_groups_2 = system.categorize_by_perfect_order_priority(test_symbols)
        
        # Assert
        # 同じ銘柄が同じ優先度に分類される
        for level in [1, 2, 3]:
            symbols_1 = set(priority_groups_1[level])
            symbols_2 = set(priority_groups_2[level])
            assert symbols_1 == symbols_2, f"優先度レベル{level}の結果が一致しない"
        
        print(f"✅ データ整合性テスト成功")


def test_real_data_performance_50_symbols():
    """実データパフォーマンステスト: 50銘柄"""
    # Setup
    config = {
        'ranking_system': {
            'scoring_weights': {
                "fundamental": 0.40,
                "technical": 0.30,
                "volume": 0.20,
                "volatility": 0.10
            }
        }
    }
    system = HierarchicalRankingSystem(config)
    
    # 日経225の主要50銘柄（実在する銘柄）
    test_symbols = [
        '7203.T', '6758.T', '9984.T', '6861.T', '8306.T',  # 5銘柄
        '9432.T', '8035.T', '6098.T', '6501.T', '8316.T',  # 10銘柄
        '8058.T', '4063.T', '6954.T', '4568.T', '4502.T',  # 15銘柄
        '4503.T', '9433.T', '2914.T', '4507.T', '6367.T',  # 20銘柄
        '6981.T', '8031.T', '7751.T', '4452.T', '6702.T',  # 25銘柄
        '5108.T', '4188.T', '8001.T', '8002.T', '8015.T',  # 30銘柄
        '7733.T', '9301.T', '4543.T', '7201.T', '5020.T',  # 35銘柄
        '7267.T', '6902.T', '6971.T', '7832.T', '9020.T',  # 40銘柄
        '5401.T', '6178.T', '8801.T', '6305.T', '3382.T',  # 45銘柄
        '4324.T', '6273.T', '9005.T', '8697.T', '9007.T'   # 50銘柄
    ]
    
    # Execute
    print(f"\n50銘柄パフォーマンステスト開始...")
    start_time = time.time()
    priority_groups = system.categorize_by_perfect_order_priority(test_symbols)
    execution_time = time.time() - start_time
    
    # Assert
    # 50銘柄で30秒以内（並列処理で大幅短縮を期待）
    assert execution_time < 30.0, f"実行時間が長すぎる: {execution_time:.2f}秒 (30秒以内が期待)"
    
    # 結果サマリー
    total_classified = sum(len(g) for g in priority_groups.values())
    success_rate = (total_classified / len(test_symbols)) * 100
    
    print(f"\n✅ 50銘柄パフォーマンステスト成功")
    print(f"   実行時間: {execution_time:.2f}秒")
    print(f"   分類成功: {total_classified}/{len(test_symbols)}銘柄 ({success_rate:.1f}%)")
    print(f"   優先度1: {len(priority_groups[1])}銘柄")
    print(f"   優先度2: {len(priority_groups[2])}銘柄")
    print(f"   優先度3: {len(priority_groups[3])}銘柄")
    
    # 高速化率の計算（ベースライン11秒と比較）
    baseline_time = 11.0
    speedup = baseline_time / execution_time if execution_time > 0 else 0
    print(f"   高速化率: {speedup:.1f}倍 (ベースライン11秒との比較)")
    
    # 成功率が90%以上を期待
    assert success_rate >= 90.0, f"成功率が低い: {success_rate:.1f}% (90%以上が期待)"


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v", "-s"])
