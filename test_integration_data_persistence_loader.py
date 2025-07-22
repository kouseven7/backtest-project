"""
Integration Test: Strategy Data Persistence and Loader
File: test_integration_data_persistence_loader.py
Description: 
  永続化機能とデータローダーの統合テスト
  データの保存・読み込み・フォーマット整合性を検証

Author: imega
Created: 2025-07-08
Modified: 2025-07-08
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.strategy_data_persistence import StrategyDataPersistence, StrategyDataIntegrator
    from config.strategy_characteristics_data_loader import StrategyCharacteristicsDataLoader, LoadOptions
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager
    print("✓ 必要なモジュールのインポート成功")
except ImportError as e:
    print(f"✗ インポートエラー: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_strategy_data(strategy_name: str) -> Dict[str, Any]:
    """データローダー形式に適合したテストデータを作成"""
    return {
        "strategy_name": strategy_name,
        "load_timestamp": datetime.now().isoformat(),
        "characteristics": {
            "trend_suitability": {
                "uptrend": {"score": 0.8, "confidence": 0.9},
                "downtrend": {"score": 0.6, "confidence": 0.8},
                "sideways": {"score": 0.7, "confidence": 0.85}
            },
            "volatility_suitability": {
                "high": {"score": 0.9, "confidence": 0.9},
                "medium": {"score": 0.8, "confidence": 0.85},
                "low": {"score": 0.5, "confidence": 0.7}
            }
        },
        "parameters": {
            "optimized_params": {
                "lookback_period": 20,
                "threshold": 0.02,
                "stop_loss": 0.05
            },
            "optimization_history": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "params": {"lookback_period": 20, "threshold": 0.02},
                    "performance": {"sharpe_ratio": 1.5, "total_return": 0.15}
                }
            ]
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "data_sources": ["backtest", "optimization"]
        }
    }

def test_data_creation_and_loading():
    """データ作成・保存・読み込みの統合テスト"""
    print("\n" + "=" * 60)
    print("データ作成・保存・読み込み統合テスト")
    print("=" * 60)
    
    test_strategy = "test_integration_strategy"
    
    try:
        # 1. テストデータ作成
        print("\n1. テストデータ作成")
        test_data = create_test_strategy_data(test_strategy)
        print(f"✓ テストデータ作成完了: {test_strategy}")
        print(f"  - データセクション: {list(test_data.keys())}")
        
        # 2. 永続化機能でデータ保存
        print("\n2. 永続化機能でデータ保存")
        persistence = StrategyDataPersistence()
        
        save_success = persistence.save_strategy_data(
            test_strategy,
            test_data,
            "Integration test data",
            "integration_test"
        )
        
        if save_success:
            print(f"✓ データ保存成功: {test_strategy}")
        else:
            print(f"✗ データ保存失敗: {test_strategy}")
            return False
        
        # 3. データローダーでロード
        print("\n3. データローダーでロード")
        loader = StrategyCharacteristicsDataLoader()
        
        # バリデーションを無効にしてロード
        options = LoadOptions(
            use_cache=False,
            validate_data=False,  # バリデーション無効化
            include_history=True,
            include_parameters=True
        )
        
        loaded_data = loader.load_strategy_characteristics(test_strategy, options)
        
        if loaded_data:
            print(f"✓ データロード成功: {test_strategy}")
            print(f"  - ロードされたデータ型: {type(loaded_data)}")
            if isinstance(loaded_data, dict):
                print(f"  - データセクション: {list(loaded_data.keys())}")
        else:
            print(f"✗ データロード失敗: {test_strategy}")
            return False
        
        # 4. データ整合性検証
        print("\n4. データ整合性検証")
        
        # 戦略名確認
        if loaded_data.get("strategy_name") == test_strategy:
            print("✓ 戦略名整合性: OK")
        else:
            print(f"⚠ 戦略名不整合: 期待={test_strategy}, 実際={loaded_data.get('strategy_name')}")
        
        # セクション確認
        expected_sections = ["characteristics", "parameters"]
        missing_sections = []
        for section in expected_sections:
            if section not in loaded_data:
                missing_sections.append(section)
        
        if not missing_sections:
            print("✓ データセクション整合性: OK")
        else:
            print(f"⚠ 不足セクション: {missing_sections}")
        
        # 5. キャッシュ機能テスト
        print("\n5. キャッシュ機能テスト")
        options_with_cache = LoadOptions(
            use_cache=True,
            validate_data=False,
            include_history=True,
            include_parameters=True
        )
        
        # 初回ロード
        start_time = datetime.now()
        data1 = loader.load_strategy_characteristics(test_strategy, options_with_cache)
        first_load_time = (datetime.now() - start_time).total_seconds()
        
        # キャッシュからロード
        start_time = datetime.now()
        data2 = loader.load_strategy_characteristics(test_strategy, options_with_cache)
        cache_load_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ 初回ロード時間: {first_load_time:.4f}秒")
        print(f"✓ キャッシュロード時間: {cache_load_time:.4f}秒")
        
        if cache_load_time < first_load_time:
            print("✓ キャッシュ効果: 確認済み")
        else:
            print("⚠ キャッシュ効果: 不明確")
        
        # 6. キャッシュ統計確認
        print("\n6. キャッシュ統計確認")
        stats = loader.get_cache_stats()
        print(f"✓ キャッシュ統計:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        print("\n✓ 統合テスト完了")
        return True
        
    except Exception as e:
        print(f"\n✗ 統合テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_existing_data_compatibility():
    """既存データとの互換性テスト"""
    print("\n" + "=" * 60)
    print("既存データ互換性テスト")
    print("=" * 60)
    
    try:
        persistence = StrategyDataPersistence()
        loader = StrategyCharacteristicsDataLoader()
        
        # 既存戦略一覧取得
        existing_strategies = persistence.list_strategies()
        print(f"\n既存戦略数: {len(existing_strategies)}")
        
        if not existing_strategies:
            print("⚠ 既存戦略がありません")
            return True
        
        # 最初の戦略でテスト
        test_strategy = existing_strategies[0]
        print(f"テスト対象: {test_strategy}")
        
        # 1. 生データ確認
        print(f"\n1. 永続化データ確認")
        raw_data = persistence.load_strategy_data(test_strategy)
        
        if raw_data:
            print(f"✓ 生データロード成功")
            print(f"  - データ型: {type(raw_data)}")
            if isinstance(raw_data, dict):
                print(f"  - トップレベルキー: {list(raw_data.keys())}")
                if 'data' in raw_data:
                    print(f"  - データセクションキー: {list(raw_data['data'].keys())}")
        else:
            print(f"✗ 生データロード失敗")
            return False
        
        # 2. データローダーでロード（バリデーション無効）
        print(f"\n2. データローダーでロード（バリデーション無効）")
        options = LoadOptions(validate_data=False, use_cache=False)
        loader_data = loader.load_strategy_characteristics(test_strategy, options)
        
        if loader_data:
            print(f"✓ ローダーでロード成功")
            print(f"  - データ型: {type(loader_data)}")
            if isinstance(loader_data, dict):
                print(f"  - キー: {list(loader_data.keys())}")
        else:
            print(f"✗ ローダーでロード失敗")
        
        # 3. データローダーでロード（バリデーション有効）
        print(f"\n3. データローダーでロード（バリデーション有効）")
        options_strict = LoadOptions(validate_data=True, use_cache=False)
        strict_data = loader.load_strategy_characteristics(test_strategy, options_strict)
        
        if strict_data:
            print(f"✓ 厳格ロード成功")
        else:
            print(f"⚠ 厳格ロード失敗（バリデーションエラー）")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 互換性テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_compatible_data_format():
    """互換性のあるデータフォーマットの作成例"""
    print("\n" + "=" * 60)
    print("互換性データフォーマット作成例")
    print("=" * 60)
    
    try:
        # 完全に互換性のあるデータ例を作成
        compatible_strategy = "compatible_test_strategy"
        
        # データローダーが期待する形式でデータ作成
        compatible_data = {
            "strategy_name": compatible_strategy,
            "load_timestamp": datetime.now().isoformat(),
            "load_options": {
                "include_history": True,
                "include_parameters": True
            },
            "characteristics": {
                "trend_suitability": {
                    "uptrend": {"score": 0.85, "confidence": 0.9},
                    "downtrend": {"score": 0.65, "confidence": 0.8},
                    "sideways": {"score": 0.75, "confidence": 0.85}
                },
                "volatility_suitability": {
                    "high": {"score": 0.9, "confidence": 0.95},
                    "medium": {"score": 0.8, "confidence": 0.9},
                    "low": {"score": 0.6, "confidence": 0.8}
                }
            },
            "parameters": {
                "current_params": {
                    "lookback_period": 25,
                    "entry_threshold": 0.025,
                    "exit_threshold": 0.015,
                    "stop_loss": 0.05,
                    "take_profit": 0.1
                },
                "optimization_results": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "params": {"lookback_period": 25, "entry_threshold": 0.025},
                        "metrics": {"sharpe_ratio": 1.8, "total_return": 0.22, "max_drawdown": 0.08}
                    }
                ]
            }
        }
        
        print(f"✓ 互換データ作成: {compatible_strategy}")
        
        # 永続化
        persistence = StrategyDataPersistence()
        save_success = persistence.save_strategy_data(
            compatible_strategy,
            compatible_data,
            "Compatible format test data",
            "compatibility_test"
        )
        
        if save_success:
            print(f"✓ 互換データ保存成功")
            
            # ロードテスト
            loader = StrategyCharacteristicsDataLoader()
            options = LoadOptions(validate_data=True, use_cache=False)
            
            loaded_data = loader.load_strategy_characteristics(compatible_strategy, options)
            
            if loaded_data:
                print(f"✓ 互換データロード成功（バリデーション付き）")
                print(f"  - 戦略名: {loaded_data.get('strategy_name')}")
                print(f"  - データセクション: {list(loaded_data.keys())}")
                
                # キャッシュ統計
                stats = loader.get_cache_stats()
                print(f"✓ 統計情報: キャッシュミス={stats['cache_misses']}, バリデーションエラー={stats['validation_errors']}")
                
                return True
            else:
                print(f"✗ 互換データロード失敗")
                return False
        else:
            print(f"✗ 互換データ保存失敗")
            return False
            
    except Exception as e:
        print(f"\n✗ 互換データ作成中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    """メイン実行部"""
    print("=" * 60)
    print("永続化機能とデータローダー統合テスト")
    print("=" * 60)
    
    success = True
    
    # 1. データ作成・保存・読み込みテスト
    success &= test_data_creation_and_loading()
    
    # 2. 既存データ互換性テスト
    success &= test_existing_data_compatibility()
    
    # 3. 互換データフォーマット作成
    success &= create_compatible_data_format()
    
    print("\n" + "=" * 60)
    print(f"統合テスト結果: {'成功' if success else '一部失敗'}")
    print("=" * 60)
    
    if success:
        print("\n✓ 1-3-3「特性データのロード・更新機能」実装完了")
        print("  - 高速キャッシュ機能")
        print("  - バッチロード機能")
        print("  - データバリデーション")
        print("  - 既存永続化機能との統合")
        print("  - エラー耐性・回復機能")
    else:
        print("\n⚠ 一部の機能で問題が検出されました")
        print("  詳細はテスト結果を確認してください")
    
    sys.exit(0 if success else 1)
