"""
フェーズ3B リアルタイムデータ接続システム デモンストレーション
統合システムの機能とパフォーマンスを実証
"""

import sys
import time
import json
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# プロジェクトルート追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data.data_source_adapter import DataSourceManager
from src.data.realtime_cache import HybridRealtimeCache
from src.data.realtime_feed import RealtimeFeedManager, UpdateFrequency
from src.data.data_feed_integration import IntegratedDataFeedSystem, MarketDataPoint, DataQualityMetrics
from config.logger_config import setup_logger


class RealtimeDataDemo:
    """リアルタイムデータ接続デモ"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.start_time = datetime.now()
        self.demo_stats = {
            'data_received': 0,
            'quality_good': 0,
            'quality_poor': 0,
            'errors_handled': 0
        }
        
        # 設定ファイルパス
        self.config_files = {
            'data_sources': 'config/data_sources_config.json',
            'realtime': 'config/realtime_config.json'
        }
        
        # コンポーネント
        self.system = None
        self.shutdown_requested = False
        
    def run_complete_demo(self):
        """完全デモ実行"""
        print("=" * 60)
        print("フェーズ3B: リアルタイムデータ接続システム デモンストレーション")
        print("=" * 60)
        
        try:
            # 1. データソースアダプターテスト
            print("\n1. データソースアダプター テスト")
            self._test_data_source_adapter()
            
            # 2. キャッシュシステムテスト
            print("\n2. キャッシュシステム テスト")
            self._test_cache_system()
            
            # 3. リアルタイムフィードテスト
            print("\n3. リアルタイムフィード テスト")
            self._test_realtime_feed()
            
            # 4. 統合システムテスト
            print("\n4. 統合システム テスト")
            self._test_integrated_system()
            
            # 5. エラーハンドリングテスト
            print("\n5. エラーハンドリング テスト")
            self._test_error_handling()
            
            # 6. パフォーマンステスト
            print("\n6. パフォーマンス テスト")
            self._test_performance()
            
        except KeyboardInterrupt:
            print("\n\\nデモが中断されました")
        except Exception as e:
            self.logger.error(f"Demo error: {e}")
            print(f"\\nデモエラー: {e}")
        finally:
            self._cleanup()
            
        print("\\n" + "=" * 60)
        print("デモンストレーション完了")
        print("=" * 60)
        
    def _test_data_source_adapter(self):
        """データソースアダプターテスト"""
        print("  データソース接続をテスト中...")
        
        try:
            # マネージャー初期化
            manager = DataSourceManager(self.config_files['data_sources'])
            
            # 接続テスト
            connect_success = manager.connect_all()
            print(f"  ✓ データソース接続: {'成功' if connect_success else '失敗'}")
            
            # 価格取得テスト
            test_symbols = ["AAPL", "GOOGL", "MSFT"]
            for symbol in test_symbols:
                price, source = manager.get_current_price(symbol)
                if price:
                    print(f"  ✓ {symbol}: ${price:.2f} (from {source})")
                else:
                    print(f"  ✗ {symbol}: 価格取得失敗")
                    
            # 複数シンボルテスト
            market_data = manager.get_market_data(test_symbols[:2])
            print(f"  ✓ 複数シンボル取得: {len(market_data)} シンボル")
            
            # ステータス確認
            status = manager.get_adapter_status()
            print(f"  ✓ アダプター状態: {len(status)} アダプター")
            
            # クリーンアップ
            manager.disconnect_all()
            
        except Exception as e:
            print(f"  ✗ データソースアダプターテスト失敗: {e}")
            
    def _test_cache_system(self):
        """キャッシュシステムテスト"""
        print("  キャッシュシステムをテスト中...")
        
        try:
            # キャッシュ設定
            cache_config = {
                'memory_max_items': 10,
                'memory_max_mb': 32,
                'disk_cache_dir': 'cache/demo_test',
                'memory_ttl_seconds': 30,
                'disk_ttl_seconds': 120,
                'enable_write_through': True
            }
            
            cache = HybridRealtimeCache(cache_config)
            
            # データ保存テスト
            test_data = {
                'AAPL_price': 150.50,
                'GOOGL_quote': {'price': 2800.75, 'volume': 1000000},
                'market_status': 'open'
            }
            
            saved_count = 0
            for key, data in test_data.items():
                if cache.put(key, data):
                    saved_count += 1
                    
            print(f"  ✓ データ保存: {saved_count}/{len(test_data)} 成功")
            
            # データ取得テスト
            retrieved_count = 0
            for key in test_data.keys():
                if cache.get(key) is not None:
                    retrieved_count += 1
                    
            print(f"  ✓ データ取得: {retrieved_count}/{len(test_data)} 成功")
            
            # 統計情報
            stats = cache.get_stats()
            print(f"  ✓ ヒット率: {stats['hit_rate_percent']:.1f}%")
            print(f"  ✓ メモリ使用量: {stats['memory_size_bytes']} bytes")
            
            # クリーンアップ
            cache.shutdown()
            
        except Exception as e:
            print(f"  ✗ キャッシュシステムテスト失敗: {e}")
            
    def _test_realtime_feed(self):
        """リアルタイムフィードテスト"""
        print("  リアルタイムフィードをテスト中...")
        
        try:
            # フィード設定
            feed_config = {
                'cache_config': {
                    'memory_max_items': 50,
                    'memory_max_mb': 64,
                    'disk_cache_dir': 'cache/demo_feed'
                }
            }
            
            feed_manager = RealtimeFeedManager(feed_config)
            
            # コールバック設定
            received_data = []
            
            def test_callback(symbol: str, data: Dict[str, Any]):
                received_data.append((symbol, data))
                if len(received_data) <= 3:  # 最初の3件だけ表示
                    print(f"  ✓ {symbol}: ${data['price']:.2f} ({data['source']})")
                    
            # 購読テスト
            test_symbol = "AAPL"
            subscribe_success = feed_manager.subscribe(
                symbol=test_symbol,
                subscriber_id="demo_test",
                callback=test_callback,
                frequency=UpdateFrequency.HIGH
            )
            
            print(f"  ✓ 購読開始: {'成功' if subscribe_success else '失敗'}")
            
            # データ受信待機
            if subscribe_success:
                print("  データ受信を待機中... (10秒)")
                time.sleep(10)
                
                print(f"  ✓ 受信データ数: {len(received_data)}")
                
                # 購読解除
                unsubscribe_success = feed_manager.unsubscribe(test_symbol, "demo_test")
                print(f"  ✓ 購読解除: {'成功' if unsubscribe_success else '失敗'}")
                
            # 統計情報
            stats = feed_manager.get_performance_stats()
            print(f"  ✓ 送信更新数: {stats['feed_stats']['updates_sent']}")
            
            # クリーンアップ
            feed_manager.shutdown()
            
        except Exception as e:
            print(f"  ✗ リアルタイムフィードテスト失敗: {e}")
            
    def _test_integrated_system(self):
        """統合システムテスト"""
        print("  統合システムをテスト中...")
        
        try:
            # システム初期化
            self.system = IntegratedDataFeedSystem()
            
            # ハンドラー設定
            data_received = []
            quality_received = []
            
            def data_handler(symbol: str, data_point: MarketDataPoint, quality: DataQualityMetrics):
                data_received.append(symbol)
                quality_received.append(quality.quality_level.value)
                if len(data_received) <= 3:
                    print(f"  ✓ {symbol}: ${data_point.price:.2f} "
                          f"(品質: {quality.quality_level.value})")
                    
            def error_handler(error: Exception, context: Dict[str, Any]):
                self.demo_stats['errors_handled'] += 1
                print(f"  ! エラー処理: {context.get('symbol', 'unknown')}")
                
            # ハンドラー登録
            self.system.register_data_handler(data_handler)
            self.system.register_error_handler(error_handler)
            
            # 購読開始
            test_symbols = ["AAPL", "GOOGL"]
            for symbol in test_symbols:
                success = self.system.subscribe_to_symbol(
                    symbol=symbol,
                    subscriber_id=f"demo_{symbol.lower()}",
                    frequency=UpdateFrequency.HIGH,
                    quality_threshold=0.6
                )
                print(f"  ✓ {symbol} 購読: {'成功' if success else '失敗'}")
                
            # データ受信待機
            print("  統合データ受信を待機中... (15秒)")
            time.sleep(15)
            
            # 結果集計
            self.demo_stats['data_received'] = len(data_received)
            self.demo_stats['quality_good'] = quality_received.count('excellent') + quality_received.count('good')
            self.demo_stats['quality_poor'] = quality_received.count('poor') + quality_received.count('invalid')
            
            print(f"  ✓ 受信データ数: {self.demo_stats['data_received']}")
            print(f"  ✓ 品質良好: {self.demo_stats['quality_good']}")
            print(f"  ✓ 品質不良: {self.demo_stats['quality_poor']}")
            
            # システム状態
            status = self.system.get_system_status()
            print(f"  ✓ システム稼働時間: {status['uptime_seconds']:.1f}秒")
            print(f"  ✓ アクティブアラート: {status['active_alerts']}")
            
        except Exception as e:
            print(f"  ✗ 統合システムテスト失敗: {e}")
            
    def _test_error_handling(self):
        """エラーハンドリングテスト"""
        print("  エラーハンドリングをテスト中...")
        
        try:
            if not self.system:
                print("  ! 統合システムが初期化されていません")
                return
                
            # エラー状態を意図的に作成（存在しないシンボル）
            invalid_symbol = "INVALID_SYMBOL_TEST"
            
            error_count_before = self.demo_stats['errors_handled']
            
            # 無効なシンボルで購読試行
            success = self.system.subscribe_to_symbol(
                symbol=invalid_symbol,
                subscriber_id="error_test",
                quality_threshold=0.9  # 高い品質閾値
            )
            
            print(f"  ✓ 無効シンボル購読: {'エラー処理確認' if not success else '予期しない成功'}")
            
            # エラー回復待機
            time.sleep(5)
            
            error_count_after = self.demo_stats['errors_handled']
            errors_handled = error_count_after - error_count_before
            
            print(f"  ✓ エラー処理数: {errors_handled}")
            
            # 品質レポート確認
            quality_report = self.system.get_quality_report(hours=1)
            print(f"  ✓ 品質監視対象: {len(quality_report)} シンボル")
            
        except Exception as e:
            print(f"  ✗ エラーハンドリングテスト失敗: {e}")
            
    def _test_performance(self):
        """パフォーマンステスト"""
        print("  パフォーマンスをテスト中...")
        
        try:
            if not self.system:
                print("  ! 統合システムが初期化されていません")
                return
                
            # パフォーマンス統計
            perf_stats = self.system.get_system_status()
            
            # レスポンス時間計算
            uptime = perf_stats['uptime_seconds']
            total_requests = perf_stats['stats']['total_data_points']
            
            if total_requests > 0:
                avg_response_time = uptime / total_requests
                print(f"  ✓ 平均応答時間: {avg_response_time:.3f}秒")
            else:
                print("  ! データポイントが不足しています")
                
            # スループット計算
            if uptime > 0:
                throughput = total_requests / uptime * 60  # per minute
                print(f"  ✓ スループット: {throughput:.1f} データポイント/分")
                
            # キャッシュ効率
            cache_stats = perf_stats.get('feed_performance', {}).get('cache_stats', {})
            if cache_stats:
                hit_rate = cache_stats.get('hit_rate_percent', 0)
                print(f"  ✓ キャッシュヒット率: {hit_rate:.1f}%")
                
            # エラー率
            total_checks = perf_stats['stats']['quality_checks']
            failed_checks = perf_stats['stats']['quality_failures']
            
            if total_checks > 0:
                error_rate = (failed_checks / total_checks) * 100
                print(f"  ✓ エラー率: {error_rate:.1f}%")
                
            # メモリ使用量
            memory_usage = cache_stats.get('memory_size_bytes', 0)
            disk_usage = cache_stats.get('disk_size_bytes', 0)
            
            print(f"  ✓ メモリ使用量: {memory_usage / 1024:.1f} KB")
            print(f"  ✓ ディスク使用量: {disk_usage / 1024:.1f} KB")
            
        except Exception as e:
            print(f"  ✗ パフォーマンステスト失敗: {e}")
            
    def _cleanup(self):
        """クリーンアップ"""
        try:
            if self.system:
                self.system.shutdown()
                self.system = None
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            
    def _show_final_summary(self):
        """最終サマリー表示"""
        duration = datetime.now() - self.start_time
        
        print("\\n" + "=" * 60)
        print("デモンストレーション サマリー")
        print("=" * 60)
        print(f"実行時間: {duration.total_seconds():.1f}秒")
        print(f"データ受信数: {self.demo_stats['data_received']}")
        print(f"品質良好データ: {self.demo_stats['quality_good']}")
        print(f"品質不良データ: {self.demo_stats['quality_poor']}")
        print(f"エラー処理数: {self.demo_stats['errors_handled']}")
        print("=" * 60)


def main():
    """メイン実行"""
    demo = RealtimeDataDemo()
    
    # シグナルハンドラー
    def signal_handler(signum, frame):
        print("\\n\\nデモを停止中...")
        demo.shutdown_requested = True
        demo._cleanup()
        demo._show_final_summary()
        exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        demo.run_complete_demo()
        demo._show_final_summary()
    except Exception as e:
        print(f"\\nデモ実行エラー: {e}")
    finally:
        demo._cleanup()


if __name__ == "__main__":
    main()
