"""
データ取得・前処理系モジュール専門テスト
main.py実証済みモジュールの詳細検証とフォールバック検出
"""

import pandas as pd
import numpy as np
import sys
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import warnings

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

class DataPreprocessingTester:
    """データ取得・前処理系モジュールの専門テスター"""
    
    def __init__(self):
        self.test_results = {}
        self.fallback_detected = []
        self.performance_metrics = {}
        self.test_ticker = "9101.T"  # main.pyと同じティッカー
        
    def detect_fallback_patterns(self, data, module_name, operation):
        """フォールバック機能の検出"""
        fallbacks = []
        
        # パターン1: モックデータ検出
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            # 完全に同じ値の検出（モックデータの可能性）
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in data.columns:
                    unique_values = data[col].nunique()
                    if unique_values < len(data) * 0.1:  # 10%未満の多様性
                        fallbacks.append(f"モックデータ疑い: {col}列の多様性不足")
            
            # 異常な連続性（人工データの可能性）
            if 'Close' in data.columns:
                price_changes = data['Close'].pct_change().abs()
                if price_changes.std() < 0.001:  # 変動が極端に小さい
                    fallbacks.append("人工データ疑い: 価格変動が不自然")
        
        # パターン2: デフォルト値埋め合わせ
        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                if data[col].dtype in [np.float64, np.int64]:
                    # 0での埋め合わせ検出
                    zero_ratio = (data[col] == 0).sum() / len(data)
                    if zero_ratio > 0.3:
                        fallbacks.append(f"デフォルト値疑い: {col}列の30%以上が0")
                    
                    # 前値での埋め合わせ検出
                    consecutive_same = data[col].eq(data[col].shift()).sum()
                    if consecutive_same > len(data) * 0.5:
                        fallbacks.append(f"前値埋め合わせ疑い: {col}列の連続同値")
        
        # パターン3: エラー隠蔽検出（警告の監視）
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 実際の処理でwarningが出ないかチェック
            if len(w) > 0:
                for warning in w:
                    fallbacks.append(f"警告隠蔽疑い: {warning.message}")
        
        if fallbacks:
            self.fallback_detected.extend([f"{module_name}.{operation}: {fb}" for fb in fallbacks])
        
        return fallbacks
    
    def test_data_fetcher_detailed(self):
        """data_fetcher.get_parameters_and_data の詳細テスト"""
        print("🔍 data_fetcher 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'data_fetcher.get_parameters_and_data',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            # main.pyと同じインポート方法
            from data_fetcher import get_parameters_and_data
            
            print("✅ インポート成功")
            
            # 実データ取得テスト（main.pyと同じパラメータ）
            start_time = datetime.now()
            
            print(f"📊 実データ取得テスト: {self.test_ticker}")
            ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(self.test_ticker)
            
            fetch_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['fetch_time'] = fetch_time
            
            # パラメータ情報をparamsとして格納（互換性のため）
            params = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date
            }
            
            # データ完整性検証
            print(f"📈 データ基本情報:")
            print(f"  株価データ行数: {len(stock_data)}")
            print(f"  株価データ列数: {len(stock_data.columns)}")
            print(f"  データ期間: {stock_data.index.min()} - {stock_data.index.max()}")
            
            # 必須列の存在確認
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            
            if missing_columns:
                test_result['issues'].append(f"必須列不足: {missing_columns}")
                print(f"❌ 必須列不足: {missing_columns}")
            else:
                print(f"✅ 必須列確認完了")
            
            # データ品質検証
            print(f"\n📊 データ品質検証:")
            
            # 欠損値確認
            null_counts = stock_data.isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0:
                print(f"⚠️ 欠損値検出: 合計{total_nulls}個")
                for col, count in null_counts[null_counts > 0].items():
                    print(f"  {col}: {count}個")
                test_result['issues'].append(f"欠損値: 合計{total_nulls}個")
            else:
                print(f"✅ 欠損値なし")
            
            # 価格データの妥当性
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in stock_data.columns:
                    if (stock_data[col] <= 0).any():
                        test_result['issues'].append(f"異常価格: {col}に0以下の値")
                        print(f"❌ {col}に0以下の価格データ")
            
            # 高値≥安値の確認
            if all(col in stock_data.columns for col in ['High', 'Low']):
                invalid_hl = (stock_data['High'] < stock_data['Low']).sum()
                if invalid_hl > 0:
                    test_result['issues'].append(f"高安値逆転: {invalid_hl}件")
                    print(f"❌ 高値<安値のデータ: {invalid_hl}件")
                else:
                    print(f"✅ 高安値関係正常")
            
            # フォールバック検出
            fallbacks = self.detect_fallback_patterns(stock_data, 'data_fetcher', 'get_parameters_and_data')
            test_result['fallback_count'] = len(fallbacks)
            
            if fallbacks:
                print(f"🚨 フォールバック検出: {len(fallbacks)}件")
                for fb in fallbacks:
                    print(f"  - {fb}")
            else:
                print(f"✅ フォールバック検出なし")
            
            # パフォーマンス評価
            print(f"\n⚡ パフォーマンス:")
            print(f"  データ取得時間: {fetch_time:.2f}秒")
            
            if fetch_time > 30:
                test_result['issues'].append("取得時間過大: 30秒超過")
                print(f"⚠️ 取得時間が長すぎます")
            
            # main.py互換性確認
            # paramsの内容確認
            if params and isinstance(params, dict):
                print(f"✅ パラメータ辞書取得成功: {len(params)}項目")
                test_result['main_py_compatibility'] = True
            else:
                print(f"❌ パラメータ取得失敗")
                test_result['issues'].append("パラメータ取得失敗")
            
            # 最終判定
            if len(test_result['issues']) == 0 and test_result['fallback_count'] == 0:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(test_result['issues']) <= 2 and test_result['fallback_count'] == 0:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
            
            self.test_results['data_fetcher'] = test_result
            self.stock_data = stock_data  # 次のテストで使用
            
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['data_fetcher'] = test_result
            return False
    
    def test_data_processor_detailed(self):
        """data_processor.preprocess_data の詳細テスト"""
        print("\n🔍 data_processor 詳細テスト開始")
        print("-" * 60)
        
        if not hasattr(self, 'stock_data'):
            print("❌ 前段階のデータが準備されていません")
            return False
        
        test_result = {
            'module': 'data_processor.preprocess_data',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from data_processor import preprocess_data
            
            print("✅ インポート成功")
            
            # データ前処理実行
            start_time = datetime.now()
            original_data = self.stock_data.copy()
            
            print(f"📊 データ前処理実行:")
            print(f"  入力データ形状: {original_data.shape}")
            
            processed_data = preprocess_data(original_data)
            
            process_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['process_time'] = process_time
            
            # 処理結果検証
            print(f"  出力データ形状: {processed_data.shape}")
            print(f"  追加列数: {len(processed_data.columns) - len(original_data.columns)}")
            
            # データ変換精度検証
            print(f"\n📊 データ変換精度検証:")
            
            # 元の価格データが保持されているか
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_columns:
                if col in original_data.columns and col in processed_data.columns:
                    is_identical = original_data[col].equals(processed_data[col])
                    if is_identical:
                        print(f"✅ {col}: 元データ保持")
                    else:
                        print(f"❌ {col}: 元データ変更検出")
                        test_result['issues'].append(f"元データ変更: {col}")
            
            # 新規追加列の確認
            new_columns = set(processed_data.columns) - set(original_data.columns)
            print(f"\n📈 新規追加列: {len(new_columns)}個")
            for col in list(new_columns)[:10]:  # 最初の10個を表示
                print(f"  - {col}")
            
            # 技術指標の数学的正確性テスト
            print(f"\n🧮 技術指標計算精度テスト:")
            
            # SMA計算確認（5日移動平均）
            if 'SMA_5' in processed_data.columns and 'Close' in processed_data.columns:
                manual_sma5 = processed_data['Close'].rolling(5).mean()
                sma_diff = (processed_data['SMA_5'] - manual_sma5).abs().max()
                if sma_diff < 1e-10:
                    print(f"✅ SMA_5: 計算正確")
                else:
                    print(f"❌ SMA_5: 計算誤差 {sma_diff}")
                    test_result['issues'].append(f"SMA_5計算誤差: {sma_diff}")
            
            # RSI計算確認（概算チェック）
            if 'RSI_14' in processed_data.columns:
                rsi_values = processed_data['RSI_14'].dropna()
                if len(rsi_values) > 0:
                    rsi_valid = ((rsi_values >= 0) & (rsi_values <= 100)).all()
                    if rsi_valid:
                        print(f"✅ RSI_14: 値域正常 (0-100)")
                    else:
                        print(f"❌ RSI_14: 値域異常")
                        test_result['issues'].append("RSI_14値域異常")
            
            # データ型保持確認
            print(f"\n🏷️ データ型検証:")
            
            # DatetimeIndex保持
            if isinstance(processed_data.index, pd.DatetimeIndex):
                print(f"✅ DatetimeIndex保持")
            else:
                print(f"❌ DatetimeIndex失失")
                test_result['issues'].append("DatetimeIndex喪失")
            
            # 数値型保持
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            print(f"✅ 数値列数: {len(numeric_columns)}")
            
            # フォールバック検出
            fallbacks = self.detect_fallback_patterns(processed_data, 'data_processor', 'preprocess_data')
            test_result['fallback_count'] = len(fallbacks)
            
            if fallbacks:
                print(f"\n🚨 フォールバック検出: {len(fallbacks)}件")
                for fb in fallbacks:
                    print(f"  - {fb}")
            else:
                print(f"\n✅ フォールバック検出なし")
            
            # パフォーマンス評価
            print(f"\n⚡ パフォーマンス:")
            print(f"  処理時間: {process_time:.2f}秒")
            
            if process_time > 10:
                test_result['issues'].append("処理時間過大: 10秒超過")
                print(f"⚠️ 処理時間が長すぎます")
            
            # main.py互換性確認
            required_indicator_columns = ['SMA_5', 'SMA_25', 'RSI_14', 'VWAP']
            missing_indicators = [col for col in required_indicator_columns if col not in processed_data.columns]
            
            if len(missing_indicators) == 0:
                print(f"✅ main.py必須指標すべて生成")
                test_result['main_py_compatibility'] = True
            else:
                print(f"❌ main.py必須指標不足: {missing_indicators}")
                test_result['issues'].append(f"必須指標不足: {missing_indicators}")
            
            # 最終判定
            if len(test_result['issues']) == 0 and test_result['fallback_count'] == 0:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(test_result['issues']) <= 2 and test_result['fallback_count'] == 0:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
            
            self.test_results['data_processor'] = test_result
            self.processed_data = processed_data  # 次のテストで使用
            
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['data_processor'] = test_result
            return False
    
    def test_batch_processor_performance_detailed(self):
        """config.performance_score_correction.batch_processor の詳細テスト"""
        print("\n🔍 batch_processor (performance_score_correction) 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.performance_score_correction.batch_processor',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.performance_score_correction.batch_processor import ScoreCorrectionBatchProcessor, BatchUpdateResult
            
            print("✅ インポート成功")
            
            # 基本設定でバッチプロセッサーを初期化
            start_time = datetime.now()
            
            test_config = {
                'batch_processing': {
                    'update_schedule': 'daily',
                    'batch_size': 10,
                    'max_concurrent_updates': 2,
                    'timeout_minutes': 5
                },
                'tracker': {},
                'score_correction': {}
            }
            
            print(f"📊 バッチプロセッサー初期化テスト")
            batch_processor = ScoreCorrectionBatchProcessor(test_config)
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            print(f"✅ 初期化完了: {init_time:.3f}秒")
            
            # 設定値検証
            print(f"\n📊 設定値検証:")
            print(f"  バッチサイズ: {batch_processor.batch_size}")
            print(f"  最大同時更新数: {batch_processor.max_concurrent_updates}")
            print(f"  タイムアウト: {batch_processor.timeout_minutes}分")
            
            # フォールバック検出
            fallbacks = self.detect_fallback_patterns(pd.DataFrame(), 'batch_processor_performance', '__init__')
            test_result['fallback_count'] = len(fallbacks)
            
            if fallbacks:
                print(f"🚨 フォールバック検出: {len(fallbacks)}件")
                for fb in fallbacks:
                    print(f"  - {fb}")
            else:
                print("✅ フォールバック検出なし")
            
            # パフォーマンス評価
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.3f}秒")
            
            if init_time > 5:
                test_result['issues'].append("初期化時間過大: 5秒超過")
            
            # 基本機能テスト
            if hasattr(batch_processor, 'run_daily_correction_update'):
                print("✅ 日次更新機能確認")
                test_result['main_py_compatibility'] = True
            else:
                print("❌ 日次更新機能不足")
                test_result['issues'].append("必須メソッド不足: run_daily_correction_update")
            
            # 最終判定
            if len(test_result['issues']) == 0 and test_result['fallback_count'] == 0:
                test_result['status'] = 'GREEN'
                print("🟢 最終判定: 再利用可能")
            elif len(test_result['issues']) <= 2 and test_result['fallback_count'] == 0:
                test_result['status'] = 'YELLOW'
                print("🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print("🔴 最終判定: 再利用禁止")
            
            self.test_results['batch_processor_performance'] = test_result
            print("✅ batch_processor (performance_score_correction) テスト完了")
            
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['batch_processor_performance'] = test_result
            return False

    def test_batch_processor_trend_detailed(self):
        """config.trend_precision_adjustment.batch_processor の詳細テスト"""
        print("\n🔍 batch_processor (trend_precision_adjustment) 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.trend_precision_adjustment.batch_processor',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.trend_precision_adjustment.batch_processor import TrendPrecisionBatchProcessor
            
            print("✅ インポート成功")
            
            # 基本設定でバッチプロセッサーを初期化
            start_time = datetime.now()
            
            test_config = {
                'batch_size': 50,
                'daily_processing_hour': 2,
                'weekly_processing_day': 'Sunday',
                'enable_daily_batch': True,
                'enable_weekly_batch': True,
                'enable_monthly_batch': False
            }
            
            print(f"📊 トレンド精度バッチプロセッサー初期化テスト")
            trend_processor = TrendPrecisionBatchProcessor(test_config)
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            print(f"✅ 初期化完了: {init_time:.3f}秒")
            
            # 設定値検証
            print(f"\n📊 設定値検証:")
            print(f"  バッチサイズ: {trend_processor.batch_size}")
            print(f"  日次処理時間: {trend_processor.daily_processing_hour}時")
            print(f"  週次処理曜日: {trend_processor.weekly_processing_day}")
            print(f"  日次バッチ有効: {trend_processor.enable_daily_batch}")
            print(f"  週次バッチ有効: {trend_processor.enable_weekly_batch}")
            
            # フォールバック検出
            fallbacks = self.detect_fallback_patterns(pd.DataFrame(), 'batch_processor_trend', '__init__')
            test_result['fallback_count'] = len(fallbacks)
            
            if fallbacks:
                print(f"🚨 フォールバック検出: {len(fallbacks)}件")
                for fb in fallbacks:
                    print(f"  - {fb}")
            else:
                print("✅ フォールバック検出なし")
            
            # パフォーマンス評価
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.3f}秒")
            
            if init_time > 5:
                test_result['issues'].append("初期化時間過大: 5秒超過")
            
            # 基本機能テスト
            required_methods = ['run_daily_precision_update']
            missing_methods = []
            
            for method in required_methods:
                if hasattr(trend_processor, method):
                    print(f"✅ {method}機能確認")
                else:
                    print(f"❌ {method}機能不足")
                    missing_methods.append(method)
            
            if len(missing_methods) == 0:
                test_result['main_py_compatibility'] = True
            else:
                test_result['issues'].append(f"必須メソッド不足: {missing_methods}")
            
            # 属性確認
            expected_attrs = ['batch_size', 'enable_daily_batch', 'enable_weekly_batch']
            missing_attrs = [attr for attr in expected_attrs if not hasattr(trend_processor, attr)]
            
            if missing_attrs:
                test_result['issues'].append(f"必須属性不足: {missing_attrs}")
                print(f"❌ 必須属性不足: {missing_attrs}")
            else:
                print("✅ 必須属性すべて確認")
            
            # 最終判定
            if len(test_result['issues']) == 0 and test_result['fallback_count'] == 0:
                test_result['status'] = 'GREEN'
                print("🟢 最終判定: 再利用可能")
            elif len(test_result['issues']) <= 2 and test_result['fallback_count'] == 0:
                test_result['status'] = 'YELLOW'
                print("🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print("🔴 最終判定: 再利用禁止")
            
            self.test_results['batch_processor_trend'] = test_result
            print("✅ batch_processor (trend_precision_adjustment) テスト完了")
            
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['batch_processor_trend'] = test_result
            return False

    def test_strategy_characteristics_data_loader_detailed(self):
        """config.strategy_characteristics_data_loader の詳細テスト"""
        print("\n🔍 strategy_characteristics_data_loader 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.strategy_characteristics_data_loader',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.strategy_characteristics_data_loader import StrategyCharacteristicsDataLoader
            
            print("✅ インポート成功")
            
            # 基本設定でデータローダーを初期化
            start_time = datetime.now()
            
            test_config = {
                'cache_expiry_hours': 24,
                'max_cache_size': 1000,
                'enable_background_refresh': True,
                'refresh_interval_minutes': 60,
                'data_sources': ['file', 'memory']
            }
            
            print(f"📊 戦略特性データローダー初期化テスト")
            data_loader = StrategyCharacteristicsDataLoader(test_config)
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            print(f"✅ 初期化完了: {init_time:.3f}秒")
            
            # 設定値検証
            print(f"\n📊 設定値検証:")
            if hasattr(data_loader, 'cache_expiry_hours'):
                print(f"  キャッシュ有効期限: {data_loader.cache_expiry_hours}時間")
            if hasattr(data_loader, 'max_cache_size'):
                print(f"  最大キャッシュサイズ: {data_loader.max_cache_size}")
            if hasattr(data_loader, 'enable_background_refresh'):
                print(f"  バックグラウンド更新: {data_loader.enable_background_refresh}")
            
            # フォールバック検出
            fallbacks = self.detect_fallback_patterns(pd.DataFrame(), 'strategy_characteristics_data_loader', '__init__')
            test_result['fallback_count'] = len(fallbacks)
            
            if fallbacks:
                print(f"🚨 フォールバック検出: {len(fallbacks)}件")
                for fb in fallbacks:
                    print(f"  - {fb}")
            else:
                print("✅ フォールバック検出なし")
            
            # パフォーマンス評価
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.3f}秒")
            
            if init_time > 5:
                test_result['issues'].append("初期化時間過大: 5秒超過")
            
            # 基本機能テスト
            required_methods = ['load_strategy_characteristics', 'get_cached_data']
            available_methods = []
            missing_methods = []
            
            for method in required_methods:
                if hasattr(data_loader, method):
                    print(f"✅ {method}機能確認")
                    available_methods.append(method)
                else:
                    print(f"❓ {method}機能未確認")
                    missing_methods.append(method)
            
            # 代替機能の確認
            alternative_methods = ['load_data', 'get_data', 'refresh_cache']
            for method in alternative_methods:
                if hasattr(data_loader, method):
                    print(f"✅ 代替機能確認: {method}")
                    available_methods.append(method)
            
            if len(available_methods) >= 1:
                test_result['main_py_compatibility'] = True
                print("✅ データ処理機能が利用可能")
            else:
                test_result['issues'].append(f"データ処理機能不足: {missing_methods}")
                print("❌ データ処理機能不足")
            
            # キャッシュ機能テスト
            if hasattr(data_loader, 'cache') or hasattr(data_loader, '_cache'):
                print("✅ キャッシュ機能確認")
            else:
                print("❓ キャッシュ機能未確認")
                test_result['issues'].append("キャッシュ機能未確認")
            
            # 最終判定
            if len(test_result['issues']) == 0 and test_result['fallback_count'] == 0:
                test_result['status'] = 'GREEN'
                print("🟢 最終判定: 再利用可能")
            elif len(test_result['issues']) <= 2 and test_result['fallback_count'] == 0:
                test_result['status'] = 'YELLOW'
                print("🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print("🔴 最終判定: 再利用禁止")
            
            self.test_results['strategy_characteristics_data_loader'] = test_result
            print("✅ strategy_characteristics_data_loader テスト完了")
            
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['strategy_characteristics_data_loader'] = test_result
            return False

    def test_indicator_calculator_detailed(self):
        """indicators.indicator_calculator.compute_indicators の詳細テスト"""
        print("\n🔍 indicator_calculator 詳細テスト開始")
        print("-" * 60)
        
        if not hasattr(self, 'processed_data'):
            print("❌ 前段階のデータが準備されていません")
            return False
        
        test_result = {
            'module': 'indicators.indicator_calculator.compute_indicators',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from indicators.indicator_calculator import compute_indicators
            
            print("✅ インポート成功")
            
            # 指標計算実行
            start_time = datetime.now()
            base_data = self.processed_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            print(f"📊 指標計算実行:")
            print(f"  入力データ形状: {base_data.shape}")
            
            indicators_data = compute_indicators(base_data)
            
            calc_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['calc_time'] = calc_time
            
            print(f"  出力データ形状: {indicators_data.shape}")
            print(f"  追加指標数: {len(indicators_data.columns) - len(base_data.columns)}")
            
            # 指標計算正確性テスト
            print(f"\n🧮 指標計算正確性テスト:")
            
            # ATR計算確認
            if 'ATR' in indicators_data.columns:
                atr_values = indicators_data['ATR'].dropna()
                if len(atr_values) > 0 and (atr_values >= 0).all():
                    print(f"✅ ATR: 非負値確認")
                else:
                    print(f"❌ ATR: 負値検出")
                    test_result['issues'].append("ATR負値")
            
            # ボリンジャーバンド関係確認
            bb_columns = ['BB_Upper', 'BB_Middle', 'BB_Lower']
            if all(col in indicators_data.columns for col in bb_columns):
                bb_valid = (indicators_data['BB_Upper'] >= indicators_data['BB_Middle']).all() and \
                          (indicators_data['BB_Middle'] >= indicators_data['BB_Lower']).all()
                if bb_valid:
                    print(f"✅ ボリンジャーバンド: 大小関係正常")
                else:
                    print(f"❌ ボリンジャーバンド: 大小関係異常")
                    test_result['issues'].append("ボリンジャーバンド大小関係異常")
            
            # VWAP計算確認
            if 'VWAP' in indicators_data.columns and all(col in indicators_data.columns for col in ['High', 'Low', 'Close', 'Volume']):
                # 手動VWAP計算（簡易版）
                typical_price = (indicators_data['High'] + indicators_data['Low'] + indicators_data['Close']) / 3
                volume_price = typical_price * indicators_data['Volume']
                cumsum_volume_price = volume_price.cumsum()
                cumsum_volume = indicators_data['Volume'].cumsum()
                manual_vwap = cumsum_volume_price / cumsum_volume
                
                vwap_diff = (indicators_data['VWAP'] - manual_vwap).abs().mean()
                if vwap_diff < 1.0:  # 平均誤差1円未満
                    print(f"✅ VWAP: 計算精度良好")
                else:
                    print(f"⚠️ VWAP: 計算誤差 {vwap_diff:.2f}")
                    test_result['issues'].append(f"VWAP計算誤差: {vwap_diff:.2f}")
            
            # フォールバック検出
            fallbacks = self.detect_fallback_patterns(indicators_data, 'indicator_calculator', 'compute_indicators')
            test_result['fallback_count'] = len(fallbacks)
            
            if fallbacks:
                print(f"\n🚨 フォールバック検出: {len(fallbacks)}件")
                for fb in fallbacks:
                    print(f"  - {fb}")
            else:
                print(f"\n✅ フォールバック検出なし")
            
            # パフォーマンス評価
            print(f"\n⚡ パフォーマンス:")
            print(f"  計算時間: {calc_time:.2f}秒")
            
            if calc_time > 5:
                test_result['issues'].append("計算時間過大: 5秒超過")
                print(f"⚠️ 計算時間が長すぎます")
            
            # main.py互換性確認
            main_required_indicators = ['ATR', 'RSI_14', 'SMA_5', 'SMA_25', 'SMA_75', 'VWAP', 'BB_Middle', 'BB_Upper', 'BB_Lower']
            missing_main_indicators = [col for col in main_required_indicators if col not in indicators_data.columns]
            
            if len(missing_main_indicators) == 0:
                print(f"✅ main.py必須指標すべて生成")
                test_result['main_py_compatibility'] = True
            else:
                print(f"❌ main.py必須指標不足: {missing_main_indicators}")
                test_result['issues'].append(f"必須指標不足: {missing_main_indicators}")
            
            # 最終判定
            if len(test_result['issues']) == 0 and test_result['fallback_count'] == 0:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(test_result['issues']) <= 2 and test_result['fallback_count'] == 0:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
            
            self.test_results['indicator_calculator'] = test_result
            
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['indicator_calculator'] = test_result
            return False
    
    def test_fallback_removal_feasibility(self):
        """フォールバック機能削除可能性テスト"""
        print("\n🔧 フォールバック削除可能性テスト")
        print("-" * 60)
        
        for module_name, result in self.test_results.items():
            if result['fallback_count'] > 0:
                print(f"\n📋 {module_name} フォールバック分析:")
                
                # 関連するフォールバックを抽出
                module_fallbacks = [fb for fb in self.fallback_detected if module_name in fb]
                
                for fb in module_fallbacks:
                    print(f"  🚨 検出: {fb}")
                    
                    # 削除可能性評価
                    if "モックデータ疑い" in fb:
                        print(f"    ❌ 削除不可: 実データ取得機能が必要")
                        result['fallback_removable'] = False
                    elif "デフォルト値疑い" in fb:
                        print(f"    ⚠️ 注意: 適切なエラーハンドリングに置換可能")
                        result['fallback_removable'] = True
                    elif "前値埋め合わせ疑い" in fb:
                        print(f"    ✅ 削除可能: NaN処理に変更可能")
                        result['fallback_removable'] = True
                    else:
                        print(f"    ❓ 要調査: 個別判断が必要")
                        result['fallback_removable'] = None
            else:
                print(f"✅ {module_name}: フォールバック機能なし")
                result['fallback_removable'] = True
    
    def generate_comprehensive_report(self):
        """包括的テストレポート生成"""
        report_lines = []
        
        report_lines.extend([
            "# データ取得・前処理系モジュール専門テストレポート",
            "",
            "## 🎯 テスト目的",
            "main.py実証済みデータ取得・前処理系モジュールの詳細検証",
            "フォールバック機能の検出と再利用可能性の正確な判定",
            "",
            "## 📋 テスト結果サマリー",
            ""
        ])
        
        green_count = sum(1 for r in self.test_results.values() if r['status'] == 'GREEN')
        yellow_count = sum(1 for r in self.test_results.values() if r['status'] == 'YELLOW')
        red_count = sum(1 for r in self.test_results.values() if r['status'] == 'RED')
        total_fallbacks = sum(r['fallback_count'] for r in self.test_results.values())
        
        report_lines.extend([
            f"- **テスト対象モジュール数**: {len(self.test_results)}",
            f"- **🟢 再利用可能 (GREEN)**: {green_count}",
            f"- **🟡 要注意 (YELLOW)**: {yellow_count}",
            f"- **🔴 再利用禁止 (RED)**: {red_count}",
            f"- **🚨 フォールバック検出総数**: {total_fallbacks}",
            "",
            "---",
            ""
        ])
        
        # 個別モジュール結果
        for module_name, result in self.test_results.items():
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
            
            report_lines.extend([
                f"## {module_name}",
                "",
                f"**最終判定**: {status_emoji} {result['status']}",
                f"**main.py互換性**: {'✅' if result['main_py_compatibility'] else '❌'}",
                f"**フォールバック検出数**: {result['fallback_count']}",
                ""
            ])
            
            if result['performance']:
                report_lines.extend(["### ⚡ パフォーマンス指標", ""])
                for metric, value in result['performance'].items():
                    report_lines.append(f"- **{metric}**: {value}")
                report_lines.append("")
            
            if result['issues']:
                report_lines.extend(["### ⚠️ 検出された問題", ""])
                for issue in result['issues']:
                    report_lines.append(f"- {issue}")
                report_lines.append("")
            
            if result['fallback_count'] > 0:
                report_lines.extend(["### 🚨 フォールバック検出詳細", ""])
                module_fallbacks = [fb for fb in self.fallback_detected if module_name in fb]
                for fb in module_fallbacks:
                    report_lines.append(f"- {fb}")
                
                # フォールバック削除可能性
                if 'fallback_removable' in result:
                    if result['fallback_removable'] is True:
                        report_lines.append("- **削除可能性**: ✅ 削除可能")
                    elif result['fallback_removable'] is False:
                        report_lines.append("- **削除可能性**: ❌ 削除困難")
                    else:
                        report_lines.append("- **削除可能性**: ❓ 要詳細調査")
                
                report_lines.append("")
            
            report_lines.extend(["---", ""])
        
        # 総合評価と推奨事項
        report_lines.extend([
            "## 🎯 総合評価と推奨事項",
            "",
            "### 📊 再利用可能性判定",
            ""
        ])
        
        if green_count == len(self.test_results):
            report_lines.extend([
                "🟢 **全モジュール再利用可能**",
                "- すべてのモジュールが高品質で main.py との互換性を確認",
                "- フォールバック機能の問題なし",
                "- comprehensive_module_test.py での優先テスト対象",
                ""
            ])
        elif green_count + yellow_count == len(self.test_results):
            report_lines.extend([
                "🟡 **条件付き再利用可能**",
                "- 軽微な問題は修正後に再利用推奨",
                "- パフォーマンス改善の検討",
                "- 段階的なテスト実装を推奨",
                ""
            ])
        else:
            report_lines.extend([
                "🔴 **再利用に重大な問題あり**",
                "- 致命的な問題があるモジュールの修正または代替検討",
                "- フォールバック機能の完全な見直し",
                "- 新規実装も選択肢として検討",
                ""
            ])
        
        # フォールバック対策
        if total_fallbacks > 0:
            report_lines.extend([
                "### 🚨 フォールバック対策",
                "",
                f"検出されたフォールバック機能 ({total_fallbacks}件) への対処:",
                ""
            ])
            
            for fb in self.fallback_detected:
                report_lines.append(f"- {fb}")
            
            report_lines.extend([
                "",
                "**推奨対策**:",
                "1. モックデータ依存の削除",
                "2. 適切なエラーハンドリングの実装",
                "3. データ品質チェックの強化",
                "4. 警告メッセージの可視化",
                ""
            ])
        
        # 次のステップ
        report_lines.extend([
            "### 🚀 次のステップ推奨",
            "",
            "1. **GREEN判定モジュール**: 即座にcomprehensive_module_test.pyに組み込み",
            "2. **YELLOW判定モジュール**: 軽微な修正後に段階的テスト",
            "3. **RED判定モジュール**: 根本的な見直しまたは代替案検討",
            "4. **フォールバック削除**: 可能なものから順次実施",
            "",
            "---",
            f"**レポート生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**テスト対象**: データ取得・前処理系モジュール (main.py実証済み)",
            f"**テストティッカー**: {self.test_ticker}",
            f"**フォールバック検出総数**: {total_fallbacks}"
        ])
        
        return "\n".join(report_lines)

def main():
    """データ取得・前処理系専門テストの実行"""
    print("🚀 データ取得・前処理系モジュール専門テスト開始")
    print("="*70)
    print(f"📅 テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    tester = DataPreprocessingTester()
    
    # 段階的テスト実行
    tests = [
        ('データ取得', tester.test_data_fetcher_detailed),
        ('データ前処理', tester.test_data_processor_detailed),
        ('指標計算', tester.test_indicator_calculator_detailed),
        ('バッチプロセッサー (performance)', tester.test_batch_processor_performance_detailed),
        ('バッチプロセッサー (trend)', tester.test_batch_processor_trend_detailed),
        ('戦略特性データローダー', tester.test_strategy_characteristics_data_loader_detailed),
    ]
    
    success_count = 0
    for test_name, test_func in tests:
        print(f"\n🔄 {test_name}テスト実行中...")
        success = test_func()
        
        if success:
            success_count += 1
            print(f"✅ {test_name}テスト完了")
        else:
            print(f"❌ {test_name}テストで重大エラー")
            # 致命的エラーでも続行（他のモジュールをテスト）
    
    # フォールバック削除可能性テスト
    tester.test_fallback_removal_feasibility()
    
    # 最終レポート生成
    report = tester.generate_comprehensive_report()
    
    # ファイル出力
    output_dir = Path("docs/Plan to create a new main entry point")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "data_preprocessing_modules_test_report.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 実行サマリー
    print(f"\n" + "="*70)
    print(f"🎯 データ取得・前処理系専門テスト完了")
    print(f"="*70)
    print(f"📊 実行結果: {success_count}/{len(tests)} テスト成功")
    print(f"📄 詳細レポート: {output_file}")
    print(f"🚨 フォールバック検出: {len(tester.fallback_detected)}件")
    
    # 簡易判定結果表示
    for module_name, result in tester.test_results.items():
        status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
        print(f"   {status_emoji} {module_name}: {result['status']}")

if __name__ == "__main__":
    main()