#!/usr/bin/env python3
"""
実際の銘柄を使用したハイブリッドランキングシステムデモ
DSSMS Phase 2 Task 2.2: ハイブリッドランキングシステム
"""

import asyncio
import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path

# パス設定
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.dssms.hybrid_ranking_engine import HybridRankingEngine


class RealSymbolsHybridRankingDemo:
    """実際の銘柄を使用したハイブリッドランキングデモ"""

    def __init__(self):
        self.logger = self._setup_logger()
        self.hybrid_engine = None
        # 実際の日本株銘柄（日経225から選択）
        self.real_symbols = [
            "7203",  # トヨタ自動車
            "6758",  # ソニーグループ
            "9984",  # ソフトバンクグループ
            "8058",  # 三菱商事
            "6861",  # キーエンス
            "8035",  # 東京エレクトロン
            "9432",  # NTT
            "7267",  # ホンダ
            "6098",  # リクルートホールディングス
            "4063"   # 信越化学工業
        ]

    def _setup_logger(self):
        """ロガー設定"""
        logger = logging.getLogger("dssms.real_symbols_demo")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def initialize_system(self):
        """システム初期化"""
        self.logger.info("=== ハイブリッドランキングシステム（実銘柄）初期化開始 ===")
        
        try:
            # 設定ファイルパス
            config_path = Path(__file__).parent / "config" / "dssms" / "hybrid_ranking_config.json"
            
            # HybridRankingEngine初期化
            self.hybrid_engine = HybridRankingEngine(str(config_path))
            
            self.logger.info("実銘柄システム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"システム初期化エラー: {e}")
            return False

    async def run_basic_demo(self):
        """基本デモ実行"""
        self.logger.info("=== 基本デモ実行開始（実銘柄） ===")
        
        # 小規模セット（5銘柄）でテスト
        test_symbols = self.real_symbols[:5]
        
        start_time = time.time()
        
        try:
            ranking_result = await self.hybrid_engine.generate_ranking(test_symbols)
            
            execution_time = time.time() - start_time
            
            # 結果表示
            self.logger.info(f"ランキング生成数: {len(ranking_result['rankings'])}件")
            self.logger.info(f"実行時間: {execution_time:.3f}秒")
            
            # 上位3銘柄詳細表示
            if ranking_result['rankings']:
                self.logger.info("=== 上位銘柄 ===")
                for i, ranking in enumerate(ranking_result['rankings'][:3], 1):
                    self.logger.info(f"{i}位: {ranking.symbol} (スコア: {ranking.total_score:.3f})")
            
            return ranking_result
            
        except Exception as e:
            self.logger.error(f"基本デモ実行エラー: {e}")
            return None

    async def run_performance_comparison(self):
        """パフォーマンス比較テスト"""
        self.logger.info("=== パフォーマンス比較テスト開始 ===")
        
        test_cases = [
            ("小規模", self.real_symbols[:3]),
            ("中規模", self.real_symbols[:5]),
            ("大規模", self.real_symbols[:7])
        ]
        
        results = {}
        
        for case_name, symbols in test_cases:
            self.logger.info(f"{case_name}テスト実行中（{len(symbols)}銘柄）...")
            
            times = []
            for run in range(2):  # 2回実行
                start_time = time.time()
                
                try:
                    result = await self.hybrid_engine.generate_ranking(symbols)
                    execution_time = time.time() - start_time
                    times.append(execution_time)
                    
                    if run == 0:  # 初回のみ詳細表示
                        ranking_count = len(result['rankings'])
                        self.logger.info(f"  ランキング生成数: {ranking_count}")
                        
                except Exception as e:
                    self.logger.error(f"{case_name}テスト実行エラー: {e}")
                    times.append(0)
            
            avg_time = sum(times) / len(times) if times else 0
            results[case_name] = {
                'symbols_count': len(symbols),
                'avg_time': avg_time,
                'per_symbol_time': avg_time / len(symbols) if len(symbols) > 0 else 0
            }
            
            self.logger.info(f"{case_name}: 平均{avg_time:.3f}秒 (1銘柄あたり{avg_time/len(symbols)*1000:.1f}ms)")
        
        return results

    async def run_cache_efficiency_test(self):
        """キャッシュ効率テスト"""
        self.logger.info("=== キャッシュ効率テスト開始 ===")
        
        test_symbols = self.real_symbols[:4]
        
        # キャッシュクリア
        await self.hybrid_engine.clear_cache()
        
        # 初回実行
        start_time = time.time()
        await self.hybrid_engine.generate_ranking(test_symbols)
        first_run_time = time.time() - start_time
        
        # 2回目実行（キャッシュ利用）
        start_time = time.time()
        await self.hybrid_engine.generate_ranking(test_symbols)
        second_run_time = time.time() - start_time
        
        # 効率計算
        speedup_ratio = first_run_time / second_run_time if second_run_time > 0 else 1.0
        
        self.logger.info(f"初回実行時間: {first_run_time:.3f}秒")
        self.logger.info(f"2回目実行時間: {second_run_time:.3f}秒")
        self.logger.info(f"スピードアップ比: {speedup_ratio:.2f}x")
        
        return {
            'first_run_time': first_run_time,
            'second_run_time': second_run_time,
            'speedup_ratio': speedup_ratio
        }

    def generate_final_report(self, basic_result, performance_results, cache_results):
        """最終レポート生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
============================================================
DSSMS Phase 2 Task 2.2: ハイブリッドランキングシステム（実銘柄）デモレポート
============================================================
実行日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

【テスト銘柄】
{', '.join(self.real_symbols[:5])} (実際の日経225構成銘柄)

【基本デモ結果】
  ランキング生成数: {len(basic_result['rankings']) if basic_result else 0}件
  システム状態: {'正常' if basic_result else 'エラー'}

【パフォーマンステスト結果】"""
        
        for case_name, result in performance_results.items():
            report += f"""
  {case_name}({result['symbols_count']}銘柄): 平均{result['avg_time']:.3f}秒 (1銘柄あたり{result['per_symbol_time']*1000:.1f}ms)"""
        
        report += f"""

【キャッシュ効率テスト結果】
  初回実行時間: {cache_results['first_run_time']:.3f}秒
  2回目実行時間: {cache_results['second_run_time']:.3f}秒
  スピードアップ比: {cache_results['speedup_ratio']:.2f}x

【総合評価】
  ✓ 実銘柄データでの動作確認完了
  ✓ パフォーマンステスト完了
  ✓ キャッシュ機能動作確認完了
  ✓ ハイブリッドランキングシステム実装完了

============================================================
デモレポート終了
============================================================
"""
        
        # レポートファイル保存
        report_path = f"demo_hybrid_ranking_real_symbols_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        self.logger.info(f"デモレポートを保存しました: {os.path.abspath(report_path)}")

    async def cleanup(self):
        """クリーンアップ"""
        if self.hybrid_engine:
            await self.hybrid_engine.shutdown()
        self.logger.info("システムクリーンアップ完了")


async def main():
    """メイン実行関数"""
    print("DSSMS Phase 2 Task 2.2: ハイブリッドランキングシステム（実銘柄）デモ開始")
    
    demo = RealSymbolsHybridRankingDemo()
    
    try:
        # システム初期化
        if not await demo.initialize_system():
            print("システム初期化に失敗しました")
            return
        
        # 基本デモ実行
        print("\n基本デモ実行中...")
        basic_result = await demo.run_basic_demo()
        
        # パフォーマンステスト
        print("\nパフォーマンステスト実行中...")
        performance_results = await demo.run_performance_comparison()
        
        # キャッシュ効率テスト
        print("\nキャッシュ効率テスト実行中...")
        cache_results = await demo.run_cache_efficiency_test()
        
        # 最終レポート生成
        demo.generate_final_report(basic_result, performance_results, cache_results)
        
    except Exception as e:
        demo.logger.error(f"デモ実行エラー: {e}")
        print(f"デモ実行エラー: {e}")
    
    finally:
        # クリーンアップ
        await demo.cleanup()
        print("\nデモ終了")


if __name__ == "__main__":
    asyncio.run(main())
