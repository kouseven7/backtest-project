"""
スコア履歴保存システム デモンストレーション (2-3-1)
既存のStrategy Scoringシステムと統合したスコア履歴管理のデモ

実行例:
python demo_score_history_system.py

作成者: GitHub Copilot
作成日: 2024年
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 既存モジュールをインポート
try:
    from config.score_history_manager import ScoreHistoryManager, ScoreHistoryConfig
    from config.strategy_scoring_model import StrategyScoreCalculator, StrategyScore
    from config.strategy_characteristics_data_loader import StrategyCharacteristicsDataLoader
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なモジュールが見つかりません。")
    sys.exit(1)

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScoreHistoryDemo:
    """スコア履歴システムのデモクラス"""
    
    def __init__(self):
        """初期化"""
        self.base_dir = Path(os.getcwd())
        
        # スコア履歴設定
        self.history_config = ScoreHistoryConfig(
            storage_directory="demo_score_history",
            max_entries_per_file=100,
            cache_size=50,
            max_history_days=90,
            auto_cleanup_enabled=True
        )
        
        # スコア履歴マネージャーを初期化
        self.history_manager = ScoreHistoryManager(
            config=self.history_config,
            base_dir=str(self.base_dir)
        )
        
        # スコア計算機を初期化
        self.score_calculator = StrategyScoreCalculator()
        
        logger.info("ScoreHistoryDemo initialized")
    
    def setup_demo_data(self):
        """デモ用のスコアデータを生成"""
        print("\n=== デモデータの生成 ===")
        
        # デモ戦略とティッカーのリスト
        strategies = [
            "momentum_strategy",
            "mean_reversion_strategy", 
            "breakout_strategy",
            "grid_trading_strategy"
        ]
        
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        # 過去30日分のデータを生成
        demo_scores = []
        base_date = datetime.now() - timedelta(days=30)
        
        for day in range(30):
            current_date = base_date + timedelta(days=day)
            
            for strategy in strategies:
                for ticker in tickers:
                    # デモ用スコアデータを作成
                    demo_score = self._create_demo_score(
                        strategy, ticker, current_date, day
                    )
                    
                    if demo_score:
                        # スコア履歴に保存
                        entry_id = self.history_manager.save_score(
                            strategy_score=demo_score,
                            trigger_event="demo_data_generation",
                            event_metadata={
                                "demo_day": day,
                                "data_source": "synthetic",
                                "batch_id": f"demo_batch_{datetime.now().strftime('%Y%m%d')}"
                            }
                        )
                        demo_scores.append((entry_id, demo_score))
        
        print(f"[OK] {len(demo_scores)}件のデモスコアを生成・保存しました")
        return demo_scores
    
    def _create_demo_score(self, strategy_name: str, ticker: str, 
                          calculated_at: datetime, day_offset: int) -> StrategyScore:
        """デモ用のStrategyScoreを作成"""
        try:
            import random
            import math
            
            # 日数に基づいてトレンドを作成
            trend_factor = math.sin(day_offset * 0.2) * 0.3 + 0.7  # 0.4 - 1.0
            noise = random.uniform(-0.1, 0.1)
            
            # 戦略別の基本性能
            strategy_base_scores = {
                "momentum_strategy": 0.75,
                "mean_reversion_strategy": 0.65,
                "breakout_strategy": 0.70,
                "grid_trading_strategy": 0.60
            }
            
            # ティッカー別の調整
            ticker_multipliers = {
                "AAPL": 1.1,
                "GOOGL": 1.05,
                "MSFT": 1.0,
                "TSLA": 0.95
            }
            
            base_score = strategy_base_scores.get(strategy_name, 0.65)
            ticker_mult = ticker_multipliers.get(ticker, 1.0)
            
            # コンポーネントスコアを計算
            performance = max(0.0, min(1.0, base_score * ticker_mult * trend_factor + noise))
            stability = max(0.0, min(1.0, 0.8 - abs(noise) * 2))
            risk_adjusted = max(0.0, min(1.0, performance * 0.9 + random.uniform(-0.05, 0.05)))
            reliability = max(0.0, min(1.0, 0.85 + random.uniform(-0.1, 0.1)))
            
            component_scores = {
                "performance": performance,
                "stability": stability,
                "risk_adjusted": risk_adjusted,
                "reliability": reliability
            }
            
            # トレンド適合度
            trend_fitness = max(0.0, min(1.0, trend_factor + random.uniform(-0.1, 0.1)))
            
            # 総合スコア
            total_score = (
                performance * 0.35 +
                stability * 0.25 +
                risk_adjusted * 0.20 +
                reliability * 0.05 +
                trend_fitness * 0.15
            )
            
            # 信頼度
            confidence = max(0.0, min(1.0, 0.8 + random.uniform(-0.2, 0.2)))
            
            # StrategyScoreオブジェクトを作成
            return StrategyScore(
                strategy_name=strategy_name,
                ticker=ticker,
                total_score=total_score,
                component_scores=component_scores,
                trend_fitness=trend_fitness,
                confidence=confidence,
                metadata={
                    "demo_generated": True,
                    "day_offset": day_offset,
                    "trend_factor": trend_factor,
                    "data_source": "synthetic"
                },
                calculated_at=calculated_at
            )
            
        except Exception as e:
            logger.error(f"Error creating demo score for {strategy_name}_{ticker}: {e}")
            return None
    
    def demonstrate_basic_operations(self):
        """基本操作のデモンストレーション"""
        print("\n=== 基本操作のデモンストレーション ===")
        
        # 1. 全履歴の取得
        print("\n1. 全スコア履歴の取得（最新10件）")
        all_history = self.history_manager.get_score_history(limit=10)
        for i, entry in enumerate(all_history[:5], 1):
            score = entry.strategy_score
            print(f"  {i}. {score.strategy_name} - {score.ticker}: {score.total_score:.3f} "
                  f"({score.calculated_at.strftime('%Y-%m-%d %H:%M')})")
        
        # 2. 戦略別フィルタリング
        print("\n2. 特定戦略のスコア履歴（momentum_strategy、最新5件）")
        momentum_history = self.history_manager.get_score_history(
            strategy_name="momentum_strategy",
            limit=5
        )
        for i, entry in enumerate(momentum_history, 1):
            score = entry.strategy_score
            print(f"  {i}. {score.ticker}: {score.total_score:.3f}")
        
        # 3. ティッカー別フィルタリング
        print("\n3. 特定ティッカーのスコア履歴（AAPL、最新5件）")
        aapl_history = self.history_manager.get_score_history(
            ticker="AAPL",
            limit=5
        )
        for i, entry in enumerate(aapl_history, 1):
            score = entry.strategy_score
            print(f"  {i}. {score.strategy_name}: {score.total_score:.3f}")
        
        # 4. 日付範囲フィルタリング
        print("\n4. 過去7日間のスコア履歴")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        recent_history = self.history_manager.get_score_history(
            date_range=(start_date, end_date),
            limit=10
        )
        print(f"  過去7日間で{len(recent_history)}件のスコアが記録されています")
        
        # 5. スコア範囲フィルタリング
        print("\n5. 高スコア履歴（0.7以上）")
        high_score_history = self.history_manager.get_score_history(
            score_range=(0.7, 1.0),
            limit=10
        )
        for i, entry in enumerate(high_score_history[:5], 1):
            score = entry.strategy_score
            print(f"  {i}. {score.strategy_name} - {score.ticker}: {score.total_score:.3f}")
    
    def demonstrate_statistics(self):
        """統計機能のデモンストレーション"""
        print("\n=== 統計機能のデモンストレーション ===")
        
        # 1. 全体統計
        print("\n1. 全体統計（過去30日）")
        overall_stats = self.history_manager.get_score_statistics(days=30)
        if 'score_stats' in overall_stats:
            stats = overall_stats['score_stats']
            print(f"  データ件数: {overall_stats['count']}")
            print(f"  平均スコア: {stats['mean']:.3f}")
            print(f"  最高スコア: {stats['max']:.3f}")
            print(f"  最低スコア: {stats['min']:.3f}")
            print(f"  標準偏差: {stats['std']:.3f}")
            print(f"  トレンド: {overall_stats['score_trend']}")
        
        # 2. 戦略別統計
        print("\n2. 戦略別統計")
        strategies = ["momentum_strategy", "mean_reversion_strategy", "breakout_strategy"]
        for strategy in strategies:
            stats = self.history_manager.get_score_statistics(
                strategy_name=strategy,
                days=30
            )
            if 'score_stats' in stats:
                mean_score = stats['score_stats']['mean']
                count = stats['count']
                print(f"  {strategy}: 平均{mean_score:.3f} ({count}件)")
        
        # 3. ティッカー別統計
        print("\n3. ティッカー別統計")
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        for ticker in tickers:
            stats = self.history_manager.get_score_statistics(
                ticker=ticker,
                days=30
            )
            if 'score_stats' in stats:
                mean_score = stats['score_stats']['mean']
                count = stats['count']
                print(f"  {ticker}: 平均{mean_score:.3f} ({count}件)")
    
    def demonstrate_real_time_scoring(self):
        """リアルタイムスコアリングのデモ"""
        print("\n=== リアルタイムスコアリングのデモ ===")
        
        # イベントリスナーを設定
        def score_change_listener(event_data):
            print(f"  [CHART] 新しいスコアが保存されました: "
                  f"{event_data['strategy_name']} - {event_data['ticker']} "
                  f"スコア: {event_data['score']:.3f}")
        
        if self.history_manager.event_manager:
            self.history_manager.event_manager.add_listener(
                'score_saved', score_change_listener
            )
        
        # リアルタイムスコア生成をシミュレート
        print("\n新しいスコアを3秒間隔で5回生成します...")
        
        strategies = ["momentum_strategy", "breakout_strategy"]
        tickers = ["AAPL", "MSFT"]
        
        for i in range(5):
            print(f"\n--- スコア生成 {i+1}/5 ---")
            
            for strategy in strategies:
                for ticker in tickers:
                    # 新しいスコアを生成
                    current_score = self._create_demo_score(
                        strategy, ticker, datetime.now(), i
                    )
                    
                    if current_score:
                        # 履歴に保存（イベントが発火される）
                        self.history_manager.save_score(
                            strategy_score=current_score,
                            trigger_event="real_time_update",
                            event_metadata={
                                "simulation_round": i + 1,
                                "auto_generated": True
                            }
                        )
            
            if i < 4:  # 最後のループではスリープしない
                time.sleep(3)
    
    def demonstrate_cache_and_performance(self):
        """キャッシュとパフォーマンスのデモ"""
        print("\n=== キャッシュとパフォーマンス情報 ===")
        
        # キャッシュ情報を表示
        cache_info = self.history_manager.get_cache_info()
        print(f"キャッシュ済みエントリ数: {cache_info['cached_entries']}")
        print(f"最近のエントリ数: {cache_info['recent_entries']}")
        print(f"キャッシュ制限: {cache_info['cache_limit']}")
        print(f"ストレージディレクトリ: {cache_info['storage_directory']}")
        print(f"インデックス有効: {cache_info['index_enabled']}")
        print(f"遅延ローディング: {cache_info['lazy_loading']}")
        
        # パフォーマンステスト
        print("\n--- 検索パフォーマンステスト ---")
        
        # 大量検索のテスト
        start_time = time.time()
        large_result = self.history_manager.get_score_history(limit=100)
        search_time = time.time() - start_time
        print(f"大量検索（100件制限）: {len(large_result)}件を{search_time:.3f}秒で取得")
        
        # フィルタ検索のテスト
        start_time = time.time()
        filtered_result = self.history_manager.get_score_history(
            strategy_name="momentum_strategy",
            score_range=(0.6, 1.0)
        )
        filter_time = time.time() - start_time
        print(f"フィルタ検索: {len(filtered_result)}件を{filter_time:.3f}秒で取得")
    
    def cleanup_demo_data(self):
        """デモデータのクリーンアップ"""
        print("\n=== デモデータのクリーンアップ ===")
        
        try:
            # デモディレクトリを削除
            demo_dir = self.base_dir / self.history_config.storage_directory
            if demo_dir.exists():
                import shutil
                shutil.rmtree(demo_dir)
                print(f"[OK] デモディレクトリを削除しました: {demo_dir}")
            else:
                print("削除するデモディレクトリが見つかりません")
                
        except Exception as e:
            print(f"[ERROR] クリーンアップエラー: {e}")
    
    def run_full_demo(self):
        """完全なデモを実行"""
        print("[ROCKET] スコア履歴保存システム (2-3-1) デモンストレーション開始")
        print("=" * 60)
        
        try:
            # 1. デモデータ生成
            self.setup_demo_data()
            
            # 2. 基本操作
            self.demonstrate_basic_operations()
            
            # 3. 統計機能
            self.demonstrate_statistics()
            
            # 4. リアルタイムスコアリング
            self.demonstrate_real_time_scoring()
            
            # 5. キャッシュとパフォーマンス
            self.demonstrate_cache_and_performance()
            
            print("\n[SUCCESS] デモンストレーション完了")
            print("=" * 60)
            
            # クリーンアップの確認
            response = input("\nデモデータを削除しますか？ (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                self.cleanup_demo_data()
            else:
                print(f"デモデータは残されました: {self.base_dir / self.history_config.storage_directory}")
            
        except KeyboardInterrupt:
            print("\n\n[WARNING]  デモが中断されました")
        except Exception as e:
            print(f"\n[ERROR] デモ実行エラー: {e}")
            logger.error(f"Demo execution error: {e}", exc_info=True)

def main():
    """メイン実行関数"""
    print("スコア履歴保存システム (2-3-1) デモンストレーション")
    print("既存のStrategy Scoringシステムとの統合確認")
    
    try:
        demo = ScoreHistoryDemo()
        demo.run_full_demo()
        
    except Exception as e:
        print(f"[ERROR] 初期化エラー: {e}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
