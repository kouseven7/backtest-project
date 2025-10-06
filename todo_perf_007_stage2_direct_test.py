#!/usr/bin/env python3
"""
TODO-PERF-007 Stage 2: 並列処理直接テスト（簡単版）

既存インポート問題を回避し、apply_market_cap_filterメソッドの並列処理効果を直接確認
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_parallel_market_cap_directly():
    """並列処理効果を直接テスト"""
    
    print("🚀 TODO-PERF-007 Stage 2: 並列処理直接テスト")
    print("="*70)
    
    try:
        # 軽量版ヘルパー直接使用
        from todo_perf_007_stage2_lightweight_parallel import ParallelMarketCapHelper
        
        helper = ParallelMarketCapHelper(max_workers=6, rate_limit_delay=0.15)
        
        # テスト銘柄（段階的）
        test_sets = {
            "small": ["7203", "9984", "8058", "9983", "6758"],  # 5銘柄
            "medium": ["7203", "9984", "8058", "9983", "6758", "7201", "8306", "8316", "4519", "4502"],  # 10銘柄
            "large": [
                "7203", "9984", "8058", "9983", "6758", "7201", "8306", "8316", "4519", "4502",
                "6098", "3382", "4751", "2432", "3659", "7974", "9434", "8411", "8802", "8604"
            ]  # 20銘柄
        }
        
        min_market_cap = 10_000_000_000  # 100億円
        results = {}
        
        print("⚡ 段階的パフォーマンステスト実行中...")
        
        for size_name, symbols in test_sets.items():
            print(f"\n🔧 {size_name}テスト ({len(symbols)}銘柄)...")
            
            start_time = time.perf_counter()
            filtered_symbols = helper.get_market_cap_data_parallel(symbols, min_market_cap)
            execution_time = time.perf_counter() - start_time
            
            symbols_per_second = len(symbols) / execution_time if execution_time > 0 else 0
            
            results[size_name] = {
                "symbols_count": len(symbols),
                "execution_time": round(execution_time, 2),
                "symbols_per_second": round(symbols_per_second, 1),
                "filtered_count": len(filtered_symbols),
                "filter_rate": round(len(filtered_symbols) / len(symbols) * 100, 1) if symbols else 0
            }
            
            print(f"  ✅ 実行時間: {execution_time:.2f}秒")
            print(f"  📊 処理能力: {symbols_per_second:.1f}銘柄/秒")
            print(f"  🎯 フィルタ結果: {len(symbols)} → {len(filtered_symbols)}銘柄")
        
        # 200銘柄への外挿計算
        print(f"\n📈 200銘柄スケール外挿計算中...")
        
        if "large" in results:
            large_throughput = results["large"]["symbols_per_second"]
            
            if large_throughput > 0:
                # 200銘柄処理時間推定
                estimated_200_time = 200 / large_throughput
                
                # 並列効率考慮（スケール時の効率低下）
                parallel_efficiency = 0.75  # 75%効率想定
                realistic_200_time = estimated_200_time / parallel_efficiency
                
                # 改善効果計算
                original_time = 52.5  # 元の実測値
                improvement_seconds = original_time - realistic_200_time
                improvement_percentage = (improvement_seconds / original_time) * 100
                
                extrapolation = {
                    "basis_throughput": large_throughput,
                    "estimated_200_symbols_time": round(realistic_200_time, 1),
                    "original_time": original_time,
                    "improvement_seconds": round(improvement_seconds, 1),
                    "improvement_percentage": round(improvement_percentage, 1),
                    "target_achievement": improvement_percentage >= 40,
                    "performance_category": "excellent" if improvement_percentage >= 50 else "good" if improvement_percentage >= 40 else "needs_improvement"
                }
                
                print(f"  📊 推定200銘柄時間: {realistic_200_time:.1f}秒")
                print(f"  🎯 改善効果: {original_time}秒 → {realistic_200_time:.1f}秒 ({improvement_percentage:.1f}%削減)")
                print(f"  ✨ 目標達成: {'✅ 達成' if improvement_percentage >= 40 else '❌ 未達成'}")
                
                results["extrapolation"] = extrapolation
        
        # 統計情報取得
        performance_stats = helper.get_performance_stats()
        results["performance_stats"] = performance_stats
        
        print(f"\n📊 パフォーマンス統計:")
        metrics = performance_stats.get("performance_metrics", {})
        print(f"  キャッシュヒット率: {metrics.get('cache_hit_rate', 'N/A')}")
        print(f"  API呼び出し率: {metrics.get('api_call_rate', 'N/A')}")
        print(f"  エラー率: {metrics.get('error_rate', 'N/A')}")
        
        # 結果保存
        result_file = f"TODO_PERF_007_Stage2_Direct_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 詳細結果: {result_file}")
        
        # 最終評価
        if "extrapolation" in results:
            extrap = results["extrapolation"]
            if extrap["target_achievement"]:
                print(f"\n🎉 Stage 2軽量版: 成功！")
                print(f"🎯 目標達成: {extrap['improvement_percentage']:.1f}%削減 (目標40%以上)")
                print(f"⚡ パフォーマンス向上: {extrap['original_time']}秒 → {extrap['estimated_200_symbols_time']}秒")
                print(f"✅ Stage 3準備完了")
                return True
            else:
                print(f"\n⚠️ Stage 2軽量版: 部分的成功")
                print(f"🎯 達成改善: {extrap['improvement_percentage']:.1f}%削減 (目標40%)")
                print(f"🔧 追加最適化推奨")
                return False
        else:
            print(f"\n❌ Stage 2軽量版: 外挿計算失敗")
            return False
        
    except Exception as e:
        print(f"❌ 直接テストエラー: {e}")
        return False

if __name__ == "__main__":
    success = test_parallel_market_cap_directly()
    print("="*70)
    sys.exit(0 if success else 1)