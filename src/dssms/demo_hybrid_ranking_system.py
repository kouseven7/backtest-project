"""
DSSMS Phase 2 Task 2.2: ハイブリッドランキングシステム デモアプリケーション

デモ機能:
- ハイブリッドランキングシステムの実演
- パフォーマンステスト
- 各コンポーネントの動作確認
- 結果可視化
"""

import sys
from pathlib import Path
import asyncio
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# システムインポート
try:
    from src.dssms.hybrid_ranking_engine import HybridRankingEngine, RankingResult, MarketCondition
    from src.dssms.ranking_data_integrator import RankingDataIntegrator
    from src.dssms.adaptive_score_calculator import AdaptiveScoreCalculator
    from src.dssms.ranking_performance_optimizer import RankingPerformanceOptimizer
    from config.logger_config import setup_logger
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("モックモードで実行します")
    
    # モッククラス定義
    class MockRankingResult:
        def __init__(self, symbol, final_score):
            self.symbol = symbol
            self.final_score = final_score
            self.hierarchical_score = final_score * 0.7
            self.comprehensive_score = final_score * 0.8
            self.adaptive_bonus = 0.1
            self.market_condition_factor = 1.0
            self.priority_level = 1
            self.confidence = 0.8
            self.processing_time_ms = 50.0
    
    class MockHybridRankingEngine:
        def __init__(self, config_path=None):
            self.logger = logging.getLogger('mock_engine')
            self.status = type('Status', (), {'total_rankings_generated': 0})()
        
        async def generate_ranking(self, symbols, force_refresh=False):
            await asyncio.sleep(0.1)  # 処理時間シミュレート
            rankings = []
            for symbol in symbols:
                score = hash(symbol) % 100 / 100.0  # 疑似ランダムスコア
                rankings.append(MockRankingResult(symbol, score))
            rankings.sort(key=lambda x: x.final_score, reverse=True)
            self.status.total_rankings_generated += len(rankings)
            return rankings
        
        def get_system_status(self):
            return {
                'total_rankings_generated': self.status.total_rankings_generated,
                'cache_hit_rate': 0.75,
                'average_processing_time_ms': 45.0,
                'market_condition': 'trending_up'
            }
        
        def clear_cache(self):
            pass
        
        async def shutdown(self):
            pass
    
    HybridRankingEngine = MockHybridRankingEngine
    RankingResult = MockRankingResult

class HybridRankingDemo:
    """ハイブリッドランキングシステムデモ"""
    
    def __init__(self):
        """初期化"""
        self.logger = self._setup_logger()
        self.engine = None
        self.demo_symbols = self._prepare_demo_symbols()
        self.results_history: List[Dict[str, Any]] = []
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーセットアップ"""
        try:
            return setup_logger('dssms.hybrid_ranking_demo')
        except:
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger('dssms.hybrid_ranking_demo')
    
    def _prepare_demo_symbols(self) -> List[str]:
        """デモ用銘柄準備"""
        # 日経225の代表的な銘柄
        demo_symbols = [
            '1001', '1002', '1003', '1004', '1005',  # サンプル銘柄
            '2001', '2002', '2003', '2004', '2005',
            '3001', '3002', '3003', '3004', '3005',
            '4001', '4002', '4003', '4004', '4005',
            '5001', '5002', '5003', '5004', '5005'
        ]
        return demo_symbols
    
    async def initialize_system(self) -> bool:
        """システム初期化"""
        try:
            self.logger.info("=== ハイブリッドランキングシステム初期化開始 ===")
            
            # 設定ファイルパス
            config_path = project_root / "config" / "dssms" / "hybrid_ranking_config.json"
            
            # エンジン初期化を試行
            try:
                self.engine = HybridRankingEngine(str(config_path))
                self.logger.info("実システム初期化完了")
            except Exception as init_error:
                self.logger.warning(f"実システム初期化失敗: {init_error}")
                self.logger.info("モックシステムで継続")
                self.engine = MockHybridRankingEngine(str(config_path))
            
            return True
            
        except Exception as e:
            self.logger.error(f"システム初期化エラー: {e}")
            # 最後の手段としてモックシステム
            try:
                self.engine = MockHybridRankingEngine()
                self.logger.info("モックシステムで初期化完了")
                return True
            except Exception as mock_error:
                self.logger.error(f"モックシステム初期化もエラー: {mock_error}")
                return False
    
    async def run_basic_demo(self) -> Dict[str, Any]:
        """基本デモ実行"""
        try:
            self.logger.info("=== 基本デモ実行開始 ===")
            
            if not self.engine:
                raise Exception("システムが初期化されていません")
            
            # 少数銘柄でのテスト
            test_symbols = self.demo_symbols[:10]
            
            start_time = time.time()
            rankings = await self.engine.generate_ranking(test_symbols)
            execution_time = time.time() - start_time
            
            # 結果分析
            result_summary = self._analyze_ranking_results(rankings, execution_time)
            
            self.logger.info(f"基本デモ完了: {len(rankings)}件の銘柄をランキング")
            self.logger.info(f"実行時間: {execution_time:.2f}秒")
            
            return result_summary
            
        except Exception as e:
            self.logger.error(f"基本デモエラー: {e}")
            return {'error': str(e)}
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """パフォーマンステスト実行"""
        try:
            self.logger.info("=== パフォーマンステスト開始 ===")
            
            if not self.engine:
                raise Exception("システムが初期化されていません")
            
            performance_results = {}
            
            # 異なるサイズでのテスト
            test_sizes = [5, 10, 15, 20, 25]
            
            for size in test_sizes:
                test_symbols = self.demo_symbols[:size]
                
                # 複数回実行して平均測定
                execution_times = []
                rankings_list = []
                
                for run in range(3):
                    start_time = time.time()
                    rankings = await self.engine.generate_ranking(test_symbols, force_refresh=True)
                    execution_time = time.time() - start_time
                    
                    execution_times.append(execution_time)
                    rankings_list.append(rankings)
                
                # 統計計算
                avg_time = sum(execution_times) / len(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                
                performance_results[f'{size}_symbols'] = {
                    'average_time_seconds': avg_time,
                    'min_time_seconds': min_time,
                    'max_time_seconds': max_time,
                    'rankings_generated': len(rankings_list[-1]) if rankings_list[-1] else 0,
                    'time_per_symbol_ms': (avg_time * 1000) / size if size > 0 else 0
                }
                
                self.logger.info(f"{size}銘柄テスト: 平均{avg_time:.3f}秒")
            
            # システム状態取得
            system_status = self.engine.get_system_status()
            performance_results['system_status'] = system_status
            
            self.logger.info("パフォーマンステスト完了")
            return performance_results
            
        except Exception as e:
            self.logger.error(f"パフォーマンステストエラー: {e}")
            return {'error': str(e)}
    
    async def run_market_condition_test(self) -> Dict[str, Any]:
        """市場状況別テスト"""
        try:
            self.logger.info("=== 市場状況別テスト開始 ===")
            
            if not self.engine:
                raise Exception("システムが初期化されていません")
            
            test_symbols = self.demo_symbols[:15]
            condition_results = {}
            
            # 複数回実行して市場状況の変化を観察
            for run in range(5):
                start_time = time.time()
                rankings = await self.engine.generate_ranking(test_symbols)
                execution_time = time.time() - start_time
                
                # 結果分析
                if rankings:
                    avg_score = sum(r.final_score for r in rankings) / len(rankings)
                    top3_symbols = [r.symbol for r in rankings[:3]]
                    score_distribution = self._calculate_score_distribution(rankings)
                else:
                    avg_score = 0.0
                    top3_symbols = []
                    score_distribution = {}
                
                condition_results[f'run_{run + 1}'] = {
                    'execution_time': execution_time,
                    'rankings_count': len(rankings),
                    'average_score': avg_score,
                    'top3_symbols': top3_symbols,
                    'score_distribution': score_distribution
                }
                
                # 実行間隔
                await asyncio.sleep(1)
            
            self.logger.info("市場状況別テスト完了")
            return condition_results
            
        except Exception as e:
            self.logger.error(f"市場状況別テストエラー: {e}")
            return {'error': str(e)}
    
    async def run_cache_efficiency_test(self) -> Dict[str, Any]:
        """キャッシュ効率テスト"""
        try:
            self.logger.info("=== キャッシュ効率テスト開始 ===")
            
            if not self.engine:
                raise Exception("システムが初期化されていません")
            
            test_symbols = self.demo_symbols[:10]
            cache_results = {}
            
            # キャッシュクリア
            self.engine.clear_cache()
            
            # 初回実行（キャッシュなし）
            start_time = time.time()
            rankings1 = await self.engine.generate_ranking(test_symbols)
            first_execution_time = time.time() - start_time
            
            # 2回目実行（キャッシュあり）
            start_time = time.time()
            rankings2 = await self.engine.generate_ranking(test_symbols)
            second_execution_time = time.time() - start_time
            
            # システム状態取得
            system_status = self.engine.get_system_status()
            
            cache_results = {
                'first_execution_time': first_execution_time,
                'second_execution_time': second_execution_time,
                'speedup_ratio': first_execution_time / max(second_execution_time, 0.001),
                'cache_hit_rate': system_status.get('cache_hit_rate', 0.0),
                'rankings_consistency': self._compare_rankings(rankings1, rankings2)
            }
            
            self.logger.info(f"キャッシュ効率テスト完了: スピードアップ比 {cache_results['speedup_ratio']:.2f}x")
            return cache_results
            
        except Exception as e:
            self.logger.error(f"キャッシュ効率テストエラー: {e}")
            return {'error': str(e)}
    
    def _analyze_ranking_results(self, rankings: List[Any], execution_time: float) -> Dict[str, Any]:
        """ランキング結果分析"""
        try:
            if not rankings:
                return {
                    'rankings_count': 0,
                    'execution_time': execution_time,
                    'average_score': 0.0,
                    'score_distribution': {},
                    'top_performers': []
                }
            
            # 基本統計
            scores = [r.final_score for r in rankings]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            # スコア分布
            score_distribution = self._calculate_score_distribution(rankings)
            
            # トップパフォーマー
            top_performers = [
                {
                    'symbol': r.symbol,
                    'final_score': r.final_score,
                    'hierarchical_score': r.hierarchical_score,
                    'comprehensive_score': r.comprehensive_score,
                    'adaptive_bonus': r.adaptive_bonus,
                    'confidence': r.confidence
                }
                for r in rankings[:5]
            ]
            
            return {
                'rankings_count': len(rankings),
                'execution_time': execution_time,
                'average_score': avg_score,
                'max_score': max_score,
                'min_score': min_score,
                'score_distribution': score_distribution,
                'top_performers': top_performers
            }
            
        except Exception as e:
            self.logger.warning(f"結果分析エラー: {e}")
            return {'error': str(e)}
    
    def _calculate_score_distribution(self, rankings: List[Any]) -> Dict[str, int]:
        """スコア分布計算"""
        try:
            distribution = {
                'excellent (0.8-1.0)': 0,
                'good (0.6-0.8)': 0,
                'average (0.4-0.6)': 0,
                'poor (0.2-0.4)': 0,
                'very_poor (0.0-0.2)': 0
            }
            
            for ranking in rankings:
                score = ranking.final_score
                if score >= 0.8:
                    distribution['excellent (0.8-1.0)'] += 1
                elif score >= 0.6:
                    distribution['good (0.6-0.8)'] += 1
                elif score >= 0.4:
                    distribution['average (0.4-0.6)'] += 1
                elif score >= 0.2:
                    distribution['poor (0.2-0.4)'] += 1
                else:
                    distribution['very_poor (0.0-0.2)'] += 1
            
            return distribution
            
        except Exception as e:
            self.logger.warning(f"スコア分布計算エラー: {e}")
            return {}
    
    def _compare_rankings(self, rankings1: List[Any], rankings2: List[Any]) -> Dict[str, Any]:
        """ランキング比較"""
        try:
            if not rankings1 or not rankings2:
                return {'consistency_score': 0.0, 'top5_match_count': 0}
            
            # トップ5の一致度
            top5_symbols1 = set(r.symbol for r in rankings1[:5])
            top5_symbols2 = set(r.symbol for r in rankings2[:5])
            top5_match_count = len(top5_symbols1.intersection(top5_symbols2))
            
            # 全体の順位相関（簡易）
            symbols1 = [r.symbol for r in rankings1]
            symbols2 = [r.symbol for r in rankings2]
            
            common_symbols = set(symbols1).intersection(set(symbols2))
            if common_symbols:
                rank_differences = []
                for symbol in common_symbols:
                    rank1 = symbols1.index(symbol) if symbol in symbols1 else len(symbols1)
                    rank2 = symbols2.index(symbol) if symbol in symbols2 else len(symbols2)
                    rank_differences.append(abs(rank1 - rank2))
                
                avg_rank_difference = sum(rank_differences) / len(rank_differences)
                consistency_score = max(0, 1 - (avg_rank_difference / len(symbols1)))
            else:
                consistency_score = 0.0
            
            return {
                'consistency_score': consistency_score,
                'top5_match_count': top5_match_count,
                'common_symbols_count': len(common_symbols)
            }
            
        except Exception as e:
            self.logger.warning(f"ランキング比較エラー: {e}")
            return {'consistency_score': 0.0, 'top5_match_count': 0}
    
    def generate_demo_report(self, results: Dict[str, Any]) -> str:
        """デモレポート生成"""
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("DSSMS Phase 2 Task 2.2: ハイブリッドランキングシステム デモレポート")
            report_lines.append("=" * 60)
            report_lines.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # 基本デモ結果
            if 'basic_demo' in results:
                basic = results['basic_demo']
                report_lines.append("【基本デモ結果】")
                if 'error' not in basic:
                    report_lines.append(f"  ランキング生成数: {basic.get('rankings_count', 0)}件")
                    report_lines.append(f"  実行時間: {basic.get('execution_time', 0):.3f}秒")
                    report_lines.append(f"  平均スコア: {basic.get('average_score', 0):.3f}")
                    report_lines.append(f"  最高スコア: {basic.get('max_score', 0):.3f}")
                    report_lines.append(f"  最低スコア: {basic.get('min_score', 0):.3f}")
                else:
                    report_lines.append(f"  エラー: {basic['error']}")
                report_lines.append("")
            
            # パフォーマンステスト結果
            if 'performance_test' in results:
                perf = results['performance_test']
                report_lines.append("【パフォーマンステスト結果】")
                if 'error' not in perf:
                    for size_key, metrics in perf.items():
                        if size_key.endswith('_symbols'):
                            size = size_key.replace('_symbols', '')
                            report_lines.append(f"  {size}銘柄: 平均{metrics['average_time_seconds']:.3f}秒 "
                                              f"(1銘柄あたり{metrics['time_per_symbol_ms']:.1f}ms)")
                else:
                    report_lines.append(f"  エラー: {perf['error']}")
                report_lines.append("")
            
            # キャッシュ効率テスト結果
            if 'cache_test' in results:
                cache = results['cache_test']
                report_lines.append("【キャッシュ効率テスト結果】")
                if 'error' not in cache:
                    report_lines.append(f"  初回実行時間: {cache.get('first_execution_time', 0):.3f}秒")
                    report_lines.append(f"  2回目実行時間: {cache.get('second_execution_time', 0):.3f}秒")
                    report_lines.append(f"  スピードアップ比: {cache.get('speedup_ratio', 0):.2f}x")
                    report_lines.append(f"  キャッシュヒット率: {cache.get('cache_hit_rate', 0):.1%}")
                else:
                    report_lines.append(f"  エラー: {cache['error']}")
                report_lines.append("")
            
            # システム状態
            if 'system_status' in results:
                status = results['system_status']
                report_lines.append("【システム状態】")
                for key, value in status.items():
                    report_lines.append(f"  {key}: {value}")
                report_lines.append("")
            
            report_lines.append("=" * 60)
            report_lines.append("デモレポート終了")
            report_lines.append("=" * 60)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")
            return f"レポート生成エラー: {e}"
    
    async def cleanup(self):
        """クリーンアップ"""
        try:
            if self.engine:
                await self.engine.shutdown()
            self.logger.info("システムクリーンアップ完了")
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")

async def main():
    """メイン実行関数"""
    demo = HybridRankingDemo()
    
    try:
        print("DSSMS Phase 2 Task 2.2: ハイブリッドランキングシステム デモ開始")
        
        # システム初期化
        if not await demo.initialize_system():
            print("システム初期化に失敗しました")
            return
        
        # デモ結果保存
        demo_results = {}
        
        # 基本デモ実行
        print("\n基本デモ実行中...")
        demo_results['basic_demo'] = await demo.run_basic_demo()
        
        # パフォーマンステスト実行
        print("\nパフォーマンステスト実行中...")
        demo_results['performance_test'] = await demo.run_performance_test()
        
        # 市場状況別テスト実行
        print("\n市場状況別テスト実行中...")
        demo_results['market_condition_test'] = await demo.run_market_condition_test()
        
        # キャッシュ効率テスト実行
        print("\nキャッシュ効率テスト実行中...")
        demo_results['cache_test'] = await demo.run_cache_efficiency_test()
        
        # システム状態取得
        if demo.engine:
            demo_results['system_status'] = demo.engine.get_system_status()
        
        # レポート生成
        report = demo.generate_demo_report(demo_results)
        print("\n" + report)
        
        # レポートファイル保存
        report_file = project_root / f"demo_hybrid_ranking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nデモレポートを保存しました: {report_file}")
        
        # 結果JSONファイル保存
        results_file = project_root / f"demo_hybrid_ranking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"デモ結果JSONを保存しました: {results_file}")
        
    except Exception as e:
        print(f"デモ実行エラー: {e}")
        
    finally:
        # クリーンアップ
        await demo.cleanup()
        print("\nデモ終了")

if __name__ == "__main__":
    asyncio.run(main())
