"""
DSSMS Phase 2 階層的ランキングシステム デモンストレーション
Task 2.1: 階層的銘柄ランキングシステムの実動作確認
"""

import sys
import os
from datetime import datetime
import json
import traceback

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# DSSMSモジュールのインポート（エラーハンドリング付き）
try:
    from src.dssms.hierarchical_ranking_system import (
        DSSMSRankingIntegrator, 
        HierarchicalRankingSystem
    )
    print("✓ DSSMS階層的ランキングシステムモジュール読み込み成功")
except ImportError as e:
    print(f"✗ DSSMSモジュール読み込みエラー: {e}")
    print("Phase 1コンポーネントが正しく実装されているか確認してください")
    sys.exit(1)

from typing import Dict, Any

def create_demo_config():
    """デモ用設定ファイル作成"""
    demo_config: Dict[str, Any] = {
        "ranking_system": {
            "scoring_weights": {
                "fundamental": 0.40,
                "technical": 0.30,
                "volume": 0.20,
                "volatility": 0.10
            },
            "priority_classification": {
                "adaptive_priority_levels": 3,
                "perfect_order_weight": 0.35,
                "level_1_threshold": 0.8,
                "level_2_threshold": 0.6
            },
            "affordability_penalty": {
                "high_price_threshold": 5000,
                "penalty_rate": 0.15,
                "max_investment_ratio": 0.8
            },
            "technical_indicators": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            },
            "cache_settings": {
                "cache_duration_minutes": 30,
                "max_cache_size": 1000
            }
        }
    }
    
    # config/dssms/ディレクトリ作成
    config_dir = "config/dssms"
    os.makedirs(config_dir, exist_ok=True)
    
    # 設定ファイル書き込み
    config_path = os.path.join(config_dir, "ranking_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(demo_config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ デモ用設定ファイル作成: {config_path}")
    return config_path

def demonstrate_priority_classification():
    """優先度分類のデモンストレーション"""
    print("\n" + "="*60)
    print("DSSMS Phase 2 Task 2.1: 階層的銘柄ランキングシステム")
    print("="*60)
    
    try:
        # 設定ファイル作成
        config_path = create_demo_config()
        
        # ランキング統合システム初期化
        print("\n1. ランキング統合システム初期化...")
        integrator = DSSMSRankingIntegrator(config_path)
        print("✓ DSSMSRankingIntegrator初期化完了")
        
        # 利用可能資金設定
        available_funds = 10_000_000  # 1000万円
        print(f"✓ 利用可能資金設定: {available_funds:,}円")
        
        # 階層的ランキング実行
        print("\n2. 階層的ランキングプロセス実行...")
        
        try:
            result = integrator.execute_full_ranking_process(available_funds)
            print("✓ ランキングプロセス実行完了")
            
            # 結果表示
            print("\n3. ランキング結果:")
            print("-" * 40)
            print(f"主候補銘柄: {result.primary_candidate}")
            print(f"選択理由: {result.selection_reason}")
            print(f"資金利用率: {result.available_fund_ratio:.1%}")
            print(f"評価対象銘柄数: {result.total_candidates_evaluated}銘柄")
            
            print(f"\nバックアップ候補: {', '.join(result.backup_candidates)}")
            
            print("\n優先度分布:")
            for level, count in result.priority_distribution.items():
                level_name = {
                    1: "レベル1 (全軸パーフェクトオーダー)",
                    2: "レベル2 (月週軸パーフェクトオーダー)", 
                    3: "レベル3 (その他)"
                }.get(level, f"レベル{level}")
                print(f"  {level_name}: {count}銘柄")
                
        except Exception as e:
            print(f"✗ ランキングプロセスエラー: {e}")
            print("フォールバックモードでデモ続行...")
            
            # フォールバック: モック結果表示
            print("\n3. フォールバック結果 (モックデータ):")
            print("-" * 40)
            print("主候補銘柄: 7203 (トヨタ自動車)")
            print("選択理由: 全時間軸パーフェクトオーダー銘柄を選択")
            print("資金利用率: 28.5%")
            print("評価対象銘柄数: 5銘柄")
            print("バックアップ候補: 6758, 9984, 8035")
            print("\n優先度分布:")
            print("  レベル1 (全軸パーフェクトオーダー): 1銘柄")
            print("  レベル2 (月週軸パーフェクトオーダー): 1銘柄")
            print("  レベル3 (その他): 3銘柄")
        
        # サマリー取得テスト
        print("\n4. ランキングサマリー生成...")
        try:
            summary = integrator.get_ranking_summary(available_funds)
            print("✓ サマリー生成完了")
            print(f"実行タイムスタンプ: {summary.get('execution_timestamp', 'N/A')}")
            print(f"システムステータス: {summary.get('system_status', 'N/A')}")
        except Exception as e:
            print(f"✗ サマリー生成エラー: {e}")
            
    except Exception as e:
        print(f"✗ デモ実行エラー: {e}")
        print("\nエラー詳細:")
        traceback.print_exc()

def test_individual_components():
    """個別コンポーネントのテスト"""
    print("\n" + "="*50)
    print("個別コンポーネント動作確認")
    print("="*50)
    
    try:
        # 設定読み込みテスト
        config_path = "config/dssms/ranking_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("✓ 設定ファイル読み込み成功")
            
            # 設定内容確認
            scoring_weights = config.get('ranking_system', {}).get('scoring_weights', {})
            print(f"  スコア重み - ファンダメンタル: {scoring_weights.get('fundamental', 0):.0%}")
            print(f"  スコア重み - テクニカル: {scoring_weights.get('technical', 0):.0%}")
            print(f"  スコア重み - 出来高: {scoring_weights.get('volume', 0):.0%}")
            print(f"  スコア重み - ボラティリティ: {scoring_weights.get('volatility', 0):.0%}")
        else:
            print("✗ 設定ファイルが見つかりません")
        
        # クラス初期化テスト
        print("\n階層的ランキングシステム初期化テスト:")
        try:
            # configが定義されているか確認
            if 'config' not in locals():
                config: Dict[str, Any] = {"ranking_system": {"scoring_weights": {}}}
            
            ranking_system = HierarchicalRankingSystem(config)
            print("✓ HierarchicalRankingSystem初期化成功")
            
            # メソッド存在確認
            required_methods = [
                'categorize_by_perfect_order_priority',
                'rank_within_priority_group', 
                'get_top_candidate',
                'get_backup_candidates',
                'get_selection_result'
            ]
            
            for method_name in required_methods:
                if hasattr(ranking_system, method_name):
                    print(f"✓ メソッド {method_name} 存在確認")
                else:
                    print(f"✗ メソッド {method_name} が見つかりません")
                    
        except Exception as e:
            print(f"✗ HierarchicalRankingSystem初期化エラー: {e}")
    
    except Exception as e:
        print(f"✗ コンポーネントテストエラー: {e}")

def performance_benchmark():
    """パフォーマンス測定"""
    print("\n" + "="*50)
    print("パフォーマンス測定")
    print("="*50)
    
    try:
        import time
        
        # 測定開始
        start_time = time.time()
        
        # ランキングプロセス実行
        config_path = "config/dssms/ranking_config.json"
        integrator = DSSMSRankingIntegrator(config_path)
        
        process_start = time.time()
        try:
            _ = integrator.execute_full_ranking_process(10_000_000)
            process_time = time.time() - process_start
            print(f"✓ ランキングプロセス実行時間: {process_time:.2f}秒")
        except Exception as e:
            process_time = time.time() - process_start
            print(f"✗ ランキングプロセスエラー (実行時間: {process_time:.2f}秒): {e}")
        
        total_time = time.time() - start_time
        print(f"✓ 総実行時間: {total_time:.2f}秒")
        
        # メモリ使用量確認（概算）
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"✓ メモリ使用量: {memory_info.rss / 1024 / 1024:.1f} MB")
        except ImportError:
            print("! psutilがインストールされていないため、メモリ使用量は測定できません")
            
    except Exception as e:
        print(f"✗ パフォーマンス測定エラー: {e}")

def main():
    """メインデモ実行"""
    print("DSSMS Phase 2 Task 2.1 階層的ランキングシステム デモ")
    print(f"実行開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 優先度分類デモンストレーション
    demonstrate_priority_classification()
    
    # 2. 個別コンポーネントテスト
    test_individual_components()
    
    # 3. パフォーマンス測定
    performance_benchmark()
    
    print("\n" + "="*60)
    print("Phase 2 Task 2.1 デモ完了")
    print("="*60)
    print("\n次のステップ:")
    print("1. Phase 1とPhase 2の統合テスト")
    print("2. 実際の株価データでの検証")
    print("3. パフォーマンス最適化")
    print("4. リアルタイム銘柄選択システムとの連携")

if __name__ == "__main__":
    main()
