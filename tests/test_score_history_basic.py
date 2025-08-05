"""
スコア履歴保存システム 簡単テスト
基本機能の動作確認用スクリプト
"""

import sys
import os
from datetime import datetime

# モジュールパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """インポートテスト"""
    print("=== インポートテスト ===")
    
    try:
        from config.score_history_manager import ScoreHistoryManager, ScoreHistoryConfig, ScoreHistoryEntry
        print("✅ score_history_manager のインポート成功")
    except ImportError as e:
        print(f"❌ score_history_manager のインポート失敗: {e}")
        return False
    
    try:
        from config.strategy_scoring_model import StrategyScore, StrategyScoreCalculator
        print("✅ strategy_scoring_model のインポート成功")
    except ImportError as e:
        print(f"❌ strategy_scoring_model のインポート失敗: {e}")
        return False
    
    return True

def test_basic_functionality():
    """基本機能テスト"""
    print("\n=== 基本機能テスト ===")
    
    try:
        from config.score_history_manager import ScoreHistoryManager, ScoreHistoryConfig
        from config.strategy_scoring_model import StrategyScore
        
        # 設定作成
        config = ScoreHistoryConfig(
            storage_directory="test_score_history",
            max_entries_per_file=10,
            cache_size=5
        )
        print("✅ 設定作成成功")
        
        # マネージャー初期化
        manager = ScoreHistoryManager(config=config)
        print("✅ ScoreHistoryManager 初期化成功")
        
        # ダミースコア作成
        dummy_score = StrategyScore(
            strategy_name="test_strategy",
            ticker="TEST",
            total_score=0.75,
            component_scores={
                "performance": 0.8,
                "stability": 0.7,
                "risk_adjusted": 0.75,
                "reliability": 0.8
            },
            trend_fitness=0.7,
            confidence=0.85,
            metadata={"test": True},
            calculated_at=datetime.now()
        )
        print("✅ ダミーStrategyScore作成成功")
        
        # スコア保存
        entry_id = manager.save_score(
            strategy_score=dummy_score,
            trigger_event="test",
            event_metadata={"test_run": True}
        )
        print(f"✅ スコア保存成功: {entry_id}")
        
        # 履歴取得
        history = manager.get_score_history(limit=5)
        print(f"✅ 履歴取得成功: {len(history)}件")
        
        # 統計取得
        stats = manager.get_score_statistics(days=1)
        print(f"✅ 統計取得成功: {stats.get('count', 0)}件のデータ")
        
        # キャッシュ情報
        cache_info = manager.get_cache_info()
        print(f"✅ キャッシュ情報取得成功: {cache_info.get('cached_entries', 0)}件キャッシュ")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本機能テスト失敗: {e}")
        return False

def main():
    """メインテスト関数"""
    print("スコア履歴保存システム (2-3-1) 簡単テスト")
    print("=" * 50)
    
    # インポートテスト
    if not test_imports():
        print("\n❌ インポートテストに失敗しました")
        return 1
    
    # 基本機能テスト
    if not test_basic_functionality():
        print("\n❌ 基本機能テストに失敗しました")
        return 1
    
    print("\n🎉 全テスト成功!")
    print("スコア履歴保存システム (2-3-1) の基本実装が完了しました")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
