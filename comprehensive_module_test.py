"""
包括的モジュールテスト - main_v2統合前の徹底検証
各モジュールの動作を完全に理解してから統合する
"""

import pandas as pd
import sys
import os
from datetime import datetime
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

class ModuleTester:
    """各モジュールの詳細テストクラス"""
    
    def __init__(self):
        self.test_results = {}
        self.test_data = None
        
    def test_data_fetcher_detailed(self):
        """データ取得機能の詳細テスト"""
        print("🔍 データ取得機能の詳細テスト")
        print("-" * 40)
        
        try:
            from data_fetcher import get_parameters_and_data
            # Excelファイルから設定を取得（引数なしでExcel自動読み込み）
            ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
            print(f"    📊 取得データ: {ticker} ({start_date} ~ {end_date})")
            
            # 基本検証
            assert len(stock_data) > 0, "データが空"
            assert 'Close' in stock_data.columns, "Close列がない"
            assert 'Volume' in stock_data.columns, "Volume列がない"
            
            # 詳細検証
            print(f"✅ データ行数: {len(stock_data)}")
            print(f"✅ データ期間: {stock_data.index.min()} - {stock_data.index.max()}")
            print(f"✅ 列数: {len(stock_data.columns)}")
            print(f"✅ 欠損値: {stock_data.isnull().sum().sum()}")
            
            # データの連続性確認
            date_gaps = pd.date_range(stock_data.index.min(), stock_data.index.max(), freq='D')
            missing_dates = len(date_gaps) - len(stock_data)
            print(f"✅ 日付の連続性: {missing_dates}日のギャップ（土日祝除く）")
            
            self.test_data = stock_data
            self.test_results['data_fetcher'] = True
            return True
            
        except Exception as e:
            print(f"❌ データ取得エラー: {e}")
            self.test_results['data_fetcher'] = False
            return False
    
    def test_data_processor_detailed(self):
        """データ前処理の詳細テスト"""
        print("\n🔍 データ前処理の詳細テスト")
        print("-" * 40)
        
        if self.test_data is None:
            print("❌ テストデータが準備されていません")
            return False
            
        try:
            from data_processor import preprocess_data
            processed_data = preprocess_data(self.test_data.copy())
            
            # 基本検証
            assert len(processed_data) > 0, "処理後データが空"
            assert len(processed_data.columns) >= len(self.test_data.columns), "列が減少"
            
            # 新しい列の確認
            original_cols = set(self.test_data.columns)
            new_cols = set(processed_data.columns) - original_cols
            
            print(f"✅ 処理後行数: {len(processed_data)}")
            print(f"✅ 追加された列: {len(new_cols)}")
            print(f"✅ 新列一覧: {list(new_cols)[:5]}...")  # 最初の5つだけ表示
            
            # データの整合性確認
            price_consistency = (processed_data['Close'] == self.test_data['Close']).all()
            print(f"✅ 価格データ整合性: {price_consistency}")
            
            self.processed_data = processed_data
            self.test_results['data_processor'] = True
            return True
            
        except Exception as e:
            print(f"❌ データ前処理エラー: {e}")
            self.test_results['data_processor'] = False
            return False
    
    def test_single_strategy_detailed(self, strategy_name="VWAPBreakoutStrategy"):
        """単一戦略の詳細テスト"""
        print(f"\n🔍 {strategy_name}の詳細テスト")
        print("-" * 40)
        
        if not hasattr(self, 'processed_data'):
            print("❌ 前処理データが準備されていません")
            return False
            
        try:
            # 戦略クラスのインポート
            if strategy_name == "VWAPBreakoutStrategy":
                from strategies.VWAP_Breakout import VWAPBreakoutStrategy
                strategy = VWAPBreakoutStrategy()
            else:
                print(f"❌ 未対応の戦略: {strategy_name}")
                return False
            
            # バックテスト実行
            print("📊 バックテスト実行中...")
            result = strategy.backtest(self.processed_data.copy())
            
            # 必須列の存在確認
            required_columns = ['Entry_Signal', 'Exit_Signal']
            for col in required_columns:
                assert col in result.columns, f"{col}列が存在しない"
                print(f"✅ {col}列: 存在")
            
            # シグナル統計
            entry_count = result['Entry_Signal'].sum()
            exit_count = result['Exit_Signal'].sum()
            
            print(f"✅ エントリーシグナル数: {entry_count}")
            print(f"✅ エグジットシグナル数: {exit_count}")
            
            # 重要: 実際のシグナル日付を確認
            entry_dates = result[result['Entry_Signal'] == 1].index
            exit_dates = result[result['Exit_Signal'] == 1].index
            
            print(f"✅ エントリー日付範囲: {entry_dates.min()} - {entry_dates.max()}")
            print(f"✅ エグジット日付範囲: {exit_dates.min()} - {exit_dates.max()}")
            
            # 同一日エントリー/エグジット問題の検出
            same_day_signals = 0
            for date in result.index:
                if result.loc[date, 'Entry_Signal'] == 1 and result.loc[date, 'Exit_Signal'] == 1:
                    same_day_signals += 1
            
            print(f"⚠️ 同一日Entry/Exit: {same_day_signals}")
            
            if same_day_signals > 0:
                print(f"🚨 警告: 同一日Entry/Exit問題が検出されました")
                print(f"   この戦略は既知の問題を含んでいる可能性があります")
                
            # シグナルの論理的整合性確認
            total_signals = entry_count + exit_count
            signal_ratio = exit_count / entry_count if entry_count > 0 else 0
            
            print(f"✅ Exit/Entry比率: {signal_ratio:.2f}")
            
            if signal_ratio < 0.1:
                print(f"⚠️ 警告: エグジットシグナルが極端に少ない")
            elif signal_ratio > 2.0:
                print(f"⚠️ 警告: エグジットシグナルが過多")
            
            self.strategy_result = result
            self.test_results[f'strategy_{strategy_name}'] = {
                'success': True,
                'entry_count': entry_count,
                'exit_count': exit_count,
                'same_day_signals': same_day_signals,
                'signal_ratio': signal_ratio
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 戦略テストエラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
            self.test_results[f'strategy_{strategy_name}'] = {'success': False, 'error': str(e)}
            return False
    
    def test_output_generation(self):
        """出力生成のテスト"""
        print(f"\n🔍 出力生成テスト")
        print("-" * 40)
        
        if not hasattr(self, 'strategy_result'):
            print("❌ 戦略結果が準備されていません")
            return False
            
        try:
            # シンプルなCSV出力テスト
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"test_output_{timestamp}.csv"
            
            self.strategy_result.to_csv(output_file)
            
            # ファイル存在確認
            assert os.path.exists(output_file), "出力ファイルが作成されていない"
            
            # ファイル内容確認
            saved_data = pd.read_csv(output_file, index_col=0)
            assert len(saved_data) == len(self.strategy_result), "データ行数が一致しない"
            
            print(f"✅ CSV出力成功: {output_file}")
            print(f"✅ 出力行数: {len(saved_data)}")
            
            # 必須列の保存確認
            for col in ['Entry_Signal', 'Exit_Signal']:
                assert col in saved_data.columns, f"{col}列が保存されていない"
                saved_sum = saved_data[col].sum()
                original_sum = self.strategy_result[col].sum()
                assert saved_sum == original_sum, f"{col}の値が変更されている"
                print(f"✅ {col}保存整合性: OK")
            
            # テストファイル削除
            os.remove(output_file)
            
            self.test_results['output_generation'] = True
            return True
            
        except Exception as e:
            print(f"❌ 出力生成エラー: {e}")
            self.test_results['output_generation'] = False
            return False
    
    def generate_comprehensive_report(self):
        """包括的テストレポート生成"""
        print(f"\n" + "="*60)
        print(f"📋 包括的テスト結果レポート")
        print(f"="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for v in self.test_results.values() if v is True or (isinstance(v, dict) and v.get('success', False)))
        
        print(f"総テスト数: {total_tests}")
        print(f"成功: {passed_tests}")
        print(f"失敗: {total_tests - passed_tests}")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n詳細結果:")
        for test_name, result in self.test_results.items():
            if result is True:
                print(f"  ✅ {test_name}: 成功")
            elif isinstance(result, dict) and result.get('success', False):
                print(f"  ✅ {test_name}: 成功")
                if 'entry_count' in result:
                    print(f"     - エントリー: {result['entry_count']}")
                    print(f"     - エグジット: {result['exit_count']}")
                    print(f"     - 同一日問題: {result['same_day_signals']}")
            else:
                print(f"  ❌ {test_name}: 失敗")
                if isinstance(result, dict) and 'error' in result:
                    print(f"     エラー: {result['error']}")
        
        print(f"\n🎯 main_v2統合準備状況:")
        if passed_tests == total_tests:
            print(f"  ✅ 全テスト成功 - 統合準備完了")
        elif passed_tests >= total_tests * 0.8:
            print(f"  ⚠️ 一部問題あり - 要注意統合")
        else:
            print(f"  ❌ 重大な問題 - 統合前に修正必要")

def main():
    """包括的モジュールテストの実行"""
    print("🚀 包括的モジュールテスト開始")
    print("="*60)
    
    tester = ModuleTester()
    
    # 段階的テスト実行
    tests = [
        ('データ取得', tester.test_data_fetcher_detailed),
        ('データ前処理', tester.test_data_processor_detailed),
        ('戦略実行', lambda: tester.test_single_strategy_detailed("VWAPBreakoutStrategy")),
        ('出力生成', tester.test_output_generation)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔄 {test_name}テスト実行中...")
        success = test_func()
        
        if not success:
            print(f"❌ {test_name}テストで致命的エラー - 以降のテスト中止")
            break
        else:
            print(f"✅ {test_name}テスト完了")
    
    # 最終レポート
    tester.generate_comprehensive_report()

if __name__ == "__main__":
    main()