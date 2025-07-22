"""
Simple Demo: Strategy Characteristics Data Loader
File: simple_demo_data_loader.py
Description: 
  strategy_characteristics_data_loader.pyの簡単なデモ
  実際のデータでロード・キャッシュ機能をテスト

Author: imega
Created: 2025-07-08
Modified: 2025-07-08
"""

import os
import sys
import logging
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.strategy_characteristics_data_loader import (
        StrategyCharacteristicsDataLoader,
        LoadOptions,
        UpdateOptions,
        create_data_loader,
        create_load_options
    )
    print("✓ データローダーモジュールのインポート成功")
except ImportError as e:
    print(f"✗ インポートエラー: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_functionality():
    """基本機能のテスト"""
    print("\n" + "=" * 60)
    print("戦略特性データローダー 基本機能テスト")
    print("=" * 60)
    
    try:
        # データローダー初期化
        print("\n1. データローダー初期化")
        loader = create_data_loader()
        print("✓ データローダー初期化完了")
        
        # 基本情報表示
        print(f"  - ベースパス: {loader.base_path}")
        print(f"  - キャッシュディレクトリ: {loader.cache_dir}")
        print(f"  - インデックスディレクトリ: {loader.index_dir}")
        
        # 利用可能な戦略確認
        print("\n2. 利用可能な戦略確認")
        try:
            strategies = loader.persistence_manager.list_strategies()
            print(f"✓ 利用可能な戦略: {len(strategies)} 個")
            
            if strategies:
                print("  戦略一覧:")
                for i, strategy in enumerate(strategies[:5]):  # 最初の5つだけ表示
                    print(f"    {i+1}. {strategy}")
                if len(strategies) > 5:
                    print(f"    ... 他 {len(strategies)-5} 個")
            else:
                print("  ⚠ 利用可能な戦略がありません")
                return False
            
        except Exception as e:
            print(f"✗ 戦略リスト取得エラー: {e}")
            return False
        
        # 戦略インデックス情報
        print("\n3. 戦略インデックス情報")
        try:
            loader.load_strategy_index()
            index_count = len(loader.strategy_index)
            print(f"✓ インデックス作成完了: {index_count} エントリー")
            
            if loader.strategy_index:
                sample_strategy = list(loader.strategy_index.keys())[0]
                sample_data = loader.strategy_index[sample_strategy]
                print(f"  サンプル ({sample_strategy}):")
                for key, value in sample_data.items():
                    print(f"    - {key}: {value}")
            
        except Exception as e:
            print(f"✗ インデックス作成エラー: {e}")
        
        # 単一戦略ロードテスト
        print("\n4. 単一戦略ロードテスト")
        try:
            test_strategy = strategies[0]
            options = create_load_options(
                use_cache=True,
                include_history=True,
                include_parameters=True,
                validate_data=True
            )
            
            print(f"  テスト戦略: {test_strategy}")
            print(f"  ロードオプション: キャッシュ={options.use_cache}, "
                  f"履歴={options.include_history}, パラメータ={options.include_parameters}")
            
            # ロード実行
            start_time = datetime.now()
            data = loader.load_strategy_characteristics(test_strategy, options)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if data:
                print(f"✓ ロード成功 (時間: {elapsed:.4f}秒)")
                print(f"  - データタイプ: {type(data)}")
                if isinstance(data, dict):
                    print(f"  - データセクション: {list(data.keys())}")
                    if 'strategy_name' in data:
                        print(f"  - 戦略名: {data['strategy_name']}")
                    if 'load_timestamp' in data:
                        print(f"  - ロード日時: {data['load_timestamp']}")
            else:
                print(f"⚠ ロード結果なし")
                
        except Exception as e:
            print(f"✗ 単一戦略ロードエラー: {e}")
            import traceback
            traceback.print_exc()
        
        # キャッシュ機能テスト
        print("\n5. キャッシュ機能テスト")
        try:
            if strategies:
                test_strategy = strategies[0]
                options = create_load_options(use_cache=True)
                
                # 2回目ロード（キャッシュヒット期待）
                start_time = datetime.now()
                data2 = loader.load_strategy_characteristics(test_strategy, options)
                elapsed2 = (datetime.now() - start_time).total_seconds()
                
                print(f"✓ 2回目ロード (時間: {elapsed2:.4f}秒)")
                
                # キャッシュ統計表示
                stats = loader.get_cache_stats()
                print(f"✓ キャッシュ統計:")
                for key, value in stats.items():
                    print(f"  - {key}: {value}")
                
        except Exception as e:
            print(f"✗ キャッシュ機能テストエラー: {e}")
        
        # バッチロードテスト
        print("\n6. バッチロードテスト")
        try:
            batch_strategies = strategies[:min(3, len(strategies))]
            options = create_load_options(use_cache=True, include_parameters=False)
            
            print(f"  対象戦略: {batch_strategies}")
            
            start_time = datetime.now()
            batch_data = loader.load_multiple_strategies(batch_strategies, options)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if batch_data:
                loaded_count = len([v for v in batch_data.values() if v is not None])
                print(f"✓ バッチロード完了: {loaded_count}/{len(batch_strategies)} 戦略 "
                      f"(時間: {elapsed:.4f}秒)")
                
                for strategy, data in batch_data.items():
                    status = "OK" if data else "データなし"
                    print(f"  - {strategy}: {status}")
            else:
                print(f"⚠ バッチロード結果なし")
                
        except Exception as e:
            print(f"✗ バッチロードエラー: {e}")
        
        # 最終統計表示
        print("\n7. 最終統計情報")
        try:
            final_stats = loader.get_cache_stats()
            print(f"✓ 最終キャッシュ統計:")
            for key, value in final_stats.items():
                print(f"  - {key}: {value}")
            
        except Exception as e:
            print(f"✗ 統計取得エラー: {e}")
        
        print("\n✓ 全テスト完了")
        return True
        
    except Exception as e:
        print(f"\n✗ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_functionality():
    """検索機能のテスト"""
    print("\n" + "=" * 60)
    print("検索機能テスト")
    print("=" * 60)
    
    try:
        loader = create_data_loader()
        
        # 戦略検索テスト
        print("\n1. 戦略検索テスト")
        try:
            # 基本検索
            search_results = loader.search_strategies({
                "has_characteristics": True
            })
            
            print(f"✓ 基本検索結果: {len(search_results)} 戦略")
            for strategy in search_results[:3]:
                print(f"  - {strategy}")
            
        except Exception as e:
            print(f"✗ 検索機能エラー: {e}")
            
    except Exception as e:
        print(f"✗ 検索機能テスト中にエラー: {e}")

def run_performance_test():
    """パフォーマンステスト"""
    print("\n" + "=" * 60)
    print("パフォーマンステスト")
    print("=" * 60)
    
    try:
        loader = create_data_loader(cache_size=100)
        strategies = loader.persistence_manager.list_strategies()
        
        if not strategies:
            print("⚠ テスト用戦略がありません")
            return
        
        test_strategies = strategies[:min(5, len(strategies))]
        
        print(f"\n対象戦略: {len(test_strategies)} 個")
        
        # キャッシュなしテスト
        print("\n1. キャッシュなしテスト")
        options_no_cache = create_load_options(use_cache=False)
        
        start_time = datetime.now()
        for strategy in test_strategies:
            data = loader.load_strategy_characteristics(strategy, options_no_cache)
        elapsed_no_cache = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ キャッシュなし時間: {elapsed_no_cache:.4f}秒")
        
        # キャッシュありテスト
        print("\n2. キャッシュありテスト")
        options_with_cache = create_load_options(use_cache=True)
        
        start_time = datetime.now()
        for strategy in test_strategies:
            data = loader.load_strategy_characteristics(strategy, options_with_cache)
        elapsed_with_cache = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ キャッシュあり時間: {elapsed_with_cache:.4f}秒")
        
        # パフォーマンス比較
        if elapsed_with_cache > 0:
            speedup = elapsed_no_cache / elapsed_with_cache
            print(f"✓ 速度向上: {speedup:.1f}倍")
        
        # キャッシュ統計
        stats = loader.get_cache_stats()
        print(f"✓ 最終キャッシュ統計: {stats}")
        
    except Exception as e:
        print(f"✗ パフォーマンステストエラー: {e}")

if __name__ == "__main__":
    """メイン実行部"""
    import argparse
    
    parser = argparse.ArgumentParser(description="戦略特性データローダーデモ")
    parser.add_argument("--basic", action="store_true", help="基本機能テスト")
    parser.add_argument("--search", action="store_true", help="検索機能テスト")
    parser.add_argument("--performance", action="store_true", help="パフォーマンステスト")
    parser.add_argument("--all", action="store_true", help="全テスト実行")
    
    args = parser.parse_args()
    
    success = True
    
    if args.all or args.basic or (not any([args.basic, args.search, args.performance])):
        success &= test_basic_functionality()
    
    if args.all or args.search:
        test_search_functionality()
    
    if args.all or args.performance:
        run_performance_test()
    
    print("\n" + "=" * 60)
    print(f"デモ実行結果: {'成功' if success else '一部失敗'}")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
