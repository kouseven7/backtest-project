#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正しい段階的調査 - main.pyの動作フローを段階別に検証
どこまで正常に動いているかを特定する
"""
import sys
import os
import pandas as pd
from datetime import datetime

class ProperFlowAnalysis:
    """正しい動作フロー分析"""
    
    def __init__(self):
        self.results = {}
        self.current_step = 0
        
    def step_by_step_analysis(self):
        """段階的分析実行"""
        
        print("🔍 **正しい段階的調査開始**")
        print("main.pyの動作フローを順番に検証")
        print("=" * 50)
        
        # Step 1: データ取得の確認
        self.check_data_fetching()
        
        # Step 2: データ前処理の確認  
        self.check_data_preprocessing()
        
        # Step 3: 戦略実行の確認
        self.check_strategy_execution()
        
        # Step 4: シグナル生成の確認
        self.check_signal_generation()
        
        # Step 5: 出力処理の確認
        self.check_output_processing()
        
        # 最終分析
        self.analyze_flow_breakdown()
    
    def check_data_fetching(self):
        """Step 1: yfinanceデータ取得の確認"""
        
        self.current_step = 1
        print(f"\n📊 **Step {self.current_step}: yfinanceデータ取得確認**")
        print("-" * 30)
        
        try:
            # data_fetcher.pyの動作確認
            sys.path.insert(0, os.getcwd())
            
            # data_fetcherのインポート確認
            try:
                import data_fetcher
                print("✅ data_fetcher.py インポート成功")
                
                # get_parameters_and_data関数の存在確認
                if hasattr(data_fetcher, 'get_parameters_and_data'):
                    print("✅ get_parameters_and_data関数 存在確認")
                    
                    # 実際にデータ取得実行
                    ticker, start_date, end_date, stock_data, index_data = data_fetcher.get_parameters_and_data()
                    
                    print(f"✅ 取得成功:")
                    print(f"   ティッカー: {ticker}")
                    print(f"   期間: {start_date} ~ {end_date}")
                    print(f"   株価データ形状: {stock_data.shape if stock_data is not None else 'None'}")
                    print(f"   インデックスデータ形状: {index_data.shape if index_data is not None else 'None'}")
                    
                    if stock_data is not None and len(stock_data) > 0:
                        print(f"   株価データ列: {list(stock_data.columns)}")
                        print(f"   データサンプル:")
                        print(f"     最初の行: {stock_data.iloc[0].to_dict()}")
                        print(f"     最後の行: {stock_data.iloc[-1].to_dict()}")
                        
                        self.results['data_fetching'] = {
                            'status': 'SUCCESS',
                            'ticker': ticker,
                            'data_shape': stock_data.shape,
                            'columns': list(stock_data.columns),
                            'date_range': (start_date, end_date)
                        }
                    else:
                        print("❌ 株価データが空またはNone")
                        self.results['data_fetching'] = {'status': 'FAILED', 'reason': 'Empty stock data'}
                        
                else:
                    print("❌ get_parameters_and_data関数が存在しない")
                    self.results['data_fetching'] = {'status': 'FAILED', 'reason': 'Function not found'}
                    
            except ImportError as e:
                print(f"❌ data_fetcher.py インポート失敗: {e}")
                self.results['data_fetching'] = {'status': 'FAILED', 'reason': f'Import error: {e}'}
                
        except Exception as e:
            print(f"❌ データ取得確認エラー: {e}")
            self.results['data_fetching'] = {'status': 'FAILED', 'reason': str(e)}
    
    def check_data_preprocessing(self):
        """Step 2: データ前処理の確認"""
        
        self.current_step = 2
        print(f"\n🔧 **Step {self.current_step}: データ前処理確認**")
        print("-" * 30)
        
        if self.results.get('data_fetching', {}).get('status') != 'SUCCESS':
            print("❌ データ取得が失敗しているため前処理をスキップ")
            self.results['data_preprocessing'] = {'status': 'SKIPPED', 'reason': 'Data fetching failed'}
            return
        
        try:
            # data_processor.pyの確認
            try:
                import data_processor
                print("✅ data_processor.py インポート成功")
                
                if hasattr(data_processor, 'preprocess_data'):
                    print("✅ preprocess_data関数 存在確認")
                    
                    # サンプルデータで前処理テスト
                    import data_fetcher
                    _, _, _, stock_data, _ = data_fetcher.get_parameters_and_data()
                    
                    if stock_data is not None:
                        processed_data = data_processor.preprocess_data(stock_data)
                        
                        print(f"✅ 前処理成功:")
                        print(f"   処理前形状: {stock_data.shape}")
                        print(f"   処理後形状: {processed_data.shape}")
                        print(f"   処理後列: {list(processed_data.columns)}")
                        
                        # 必要な列の確認
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        missing_cols = [col for col in required_cols if col not in processed_data.columns]
                        
                        if not missing_cols:
                            print("✅ 必要な基本列がすべて存在")
                        else:
                            print(f"⚠️ 不足列: {missing_cols}")
                        
                        self.results['data_preprocessing'] = {
                            'status': 'SUCCESS',
                            'processed_shape': processed_data.shape,
                            'columns': list(processed_data.columns),
                            'missing_required_cols': missing_cols
                        }
                    else:
                        print("❌ 前処理用データがNone")
                        self.results['data_preprocessing'] = {'status': 'FAILED', 'reason': 'No data to process'}
                        
                else:
                    print("❌ preprocess_data関数が存在しない")
                    self.results['data_preprocessing'] = {'status': 'FAILED', 'reason': 'Function not found'}
                    
            except ImportError as e:
                print(f"❌ data_processor.py インポート失敗: {e}")
                self.results['data_preprocessing'] = {'status': 'FAILED', 'reason': f'Import error: {e}'}
                
        except Exception as e:
            print(f"❌ データ前処理確認エラー: {e}")
            self.results['data_preprocessing'] = {'status': 'FAILED', 'reason': str(e)}
    
    def check_strategy_execution(self):
        """Step 3: 戦略実行の確認"""
        
        self.current_step = 3
        print(f"\n⚡ **Step {self.current_step}: 戦略実行確認**")
        print("-" * 30)
        
        if self.results.get('data_preprocessing', {}).get('status') != 'SUCCESS':
            print("❌ データ前処理が失敗しているため戦略実行をスキップ")
            self.results['strategy_execution'] = {'status': 'SKIPPED', 'reason': 'Data preprocessing failed'}
            return
        
        try:
            # 各戦略の確認
            strategies_to_test = [
                ('strategies.Opening_Gap', 'OpeningGapStrategy'),
                ('strategies.gc_strategy_signal', 'GCStrategy'),
                ('strategies.VWAP_Breakout', 'VWAPBreakoutStrategy')
            ]
            
            strategy_results = {}
            
            for module_name, class_name in strategies_to_test:
                try:
                    print(f"\n🎯 {class_name}の確認:")
                    
                    # モジュールインポート
                    module = __import__(module_name, fromlist=[class_name])
                    strategy_class = getattr(module, class_name)
                    print(f"  ✅ {module_name} インポート成功")
                    
                    # テストデータで戦略初期化
                    import data_fetcher
                    import data_processor
                    
                    _, _, _, stock_data, index_data = data_fetcher.get_parameters_and_data()
                    processed_data = data_processor.preprocess_data(stock_data)
                    
                    # 戦略インスタンス作成
                    strategy = strategy_class(
                        data=processed_data,
                        index_data=index_data,
                        params={'period': 20},  # 基本パラメータ
                        price_column="Close"
                    )
                    print(f"  ✅ {class_name} インスタンス作成成功")
                    
                    # backtest()メソッドの確認
                    if hasattr(strategy, 'backtest'):
                        print(f"  ✅ backtest()メソッド 存在確認")
                        
                        # 実際にbacktest実行
                        result = strategy.backtest()
                        
                        if result is not None:
                            print(f"  ✅ backtest()実行成功")
                            print(f"     結果形状: {result.shape}")
                            print(f"     結果列: {list(result.columns)}")
                            
                            strategy_results[class_name] = {
                                'status': 'SUCCESS',
                                'result_shape': result.shape,
                                'columns': list(result.columns)
                            }
                        else:
                            print(f"  ❌ backtest()結果がNone")
                            strategy_results[class_name] = {'status': 'FAILED', 'reason': 'backtest returned None'}
                    else:
                        print(f"  ❌ backtest()メソッドが存在しない")
                        strategy_results[class_name] = {'status': 'FAILED', 'reason': 'backtest method not found'}
                        
                except Exception as e:
                    print(f"  ❌ {class_name}エラー: {e}")
                    strategy_results[class_name] = {'status': 'FAILED', 'reason': str(e)}
            
            self.results['strategy_execution'] = strategy_results
            
        except Exception as e:
            print(f"❌ 戦略実行確認エラー: {e}")
            self.results['strategy_execution'] = {'status': 'FAILED', 'reason': str(e)}
    
    def check_signal_generation(self):
        """Step 4: シグナル生成の確認"""
        
        self.current_step = 4
        print(f"\n📈 **Step {self.current_step}: シグナル生成確認**")
        print("-" * 30)
        
        strategy_results = self.results.get('strategy_execution', {})
        if not strategy_results or all(result.get('status') != 'SUCCESS' for result in strategy_results.values()):
            print("❌ 戦略実行が失敗しているためシグナル生成をスキップ")
            self.results['signal_generation'] = {'status': 'SKIPPED', 'reason': 'Strategy execution failed'}
            return
        
        print("🔍 各戦略のシグナル生成詳細確認:")
        
        signal_results = {}
        
        for strategy_name, result in strategy_results.items():
            if result.get('status') == 'SUCCESS':
                print(f"\n📊 {strategy_name}のシグナル確認:")
                
                columns = result.get('columns', [])
                
                # Entry_Signal/Exit_Signalの確認
                has_entry = 'Entry_Signal' in columns
                has_exit = 'Exit_Signal' in columns
                
                print(f"  Entry_Signal列: {'✅' if has_entry else '❌'}")
                print(f"  Exit_Signal列: {'✅' if has_exit else '❌'}")
                
                if has_entry or has_exit:
                    signal_results[strategy_name] = {
                        'status': 'SUCCESS',
                        'has_entry_signal': has_entry,
                        'has_exit_signal': has_exit,
                        'total_columns': len(columns)
                    }
                else:
                    signal_results[strategy_name] = {
                        'status': 'FAILED',
                        'reason': 'No signal columns found',
                        'available_columns': columns
                    }
        
        self.results['signal_generation'] = signal_results
    
    def check_output_processing(self):
        """Step 5: 出力処理の確認"""
        
        self.current_step = 5
        print(f"\n💾 **Step {self.current_step}: 出力処理確認**")
        print("-" * 30)
        
        try:
            # unified_exporterの確認
            sys.path.insert(0, os.path.join(os.getcwd(), 'output'))
            
            try:
                import unified_exporter
                print("✅ unified_exporter インポート成功")
                
                # 主要クラス・関数の確認
                if hasattr(unified_exporter, 'UnifiedExporter'):
                    print("✅ UnifiedExporter クラス存在確認")
                    self.results['output_processing'] = {
                        'status': 'SUCCESS',
                        'unified_exporter_available': True
                    }
                else:
                    print("❌ UnifiedExporter クラスが存在しない")
                    self.results['output_processing'] = {
                        'status': 'FAILED',
                        'reason': 'UnifiedExporter class not found'
                    }
                    
            except ImportError as e:
                print(f"❌ unified_exporter インポート失敗: {e}")
                self.results['output_processing'] = {
                    'status': 'FAILED',
                    'reason': f'Import error: {e}'
                }
                
        except Exception as e:
            print(f"❌ 出力処理確認エラー: {e}")
            self.results['output_processing'] = {'status': 'FAILED', 'reason': str(e)}
    
    def analyze_flow_breakdown(self):
        """最終分析：どこで処理が破綻しているかを特定"""
        
        print(f"\n🎯 **最終分析：処理破綻ポイント特定**")
        print("=" * 50)
        
        steps = [
            ('data_fetching', 'データ取得'),
            ('data_preprocessing', 'データ前処理'),
            ('strategy_execution', '戦略実行'),
            ('signal_generation', 'シグナル生成'),
            ('output_processing', '出力処理')
        ]
        
        working_until = 0
        broken_at = None
        
        for i, (step_key, step_name) in enumerate(steps):
            result = self.results.get(step_key, {})
            status = result.get('status', 'UNKNOWN')
            
            if status == 'SUCCESS':
                print(f"✅ Step {i+1} ({step_name}): 正常")
                working_until = i + 1
            elif status == 'SKIPPED':
                print(f"⏭️ Step {i+1} ({step_name}): スキップ")
                if broken_at is None:
                    broken_at = i + 1
                break
            else:
                print(f"❌ Step {i+1} ({step_name}): 異常")
                if broken_at is None:
                    broken_at = i + 1
                break
        
        print(f"\n🔍 **診断結果**:")
        print(f"   正常動作: Step 1 ～ Step {working_until}")
        
        if broken_at:
            print(f"   破綻ポイント: Step {broken_at}")
            broken_step_key, broken_step_name = steps[broken_at - 1]
            reason = self.results.get(broken_step_key, {}).get('reason', 'Unknown')
            print(f"   破綻理由: {reason}")
        else:
            print(f"   すべてのステップが正常")
        
        return working_until, broken_at

if __name__ == "__main__":
    analyzer = ProperFlowAnalysis()
    analyzer.step_by_step_analysis()
