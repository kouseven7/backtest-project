"""
main_v2.py - クリーンスタート バックテストシステム

Phase 1: 基礎実装 (VWAPBreakoutStrategy単体)
Step 1-1: VWAPBreakoutStrategy + 必須設定モジュール統合 ✅
Step 1-2: データ処理実装 (data_fetcher + data_processor統合)
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# プロジェクトルートをPATHに追加
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # main_v2.pyの親ディレクトリ
sys.path.insert(0, PROJECT_ROOT)

print(f"📁 プロジェクトルート: {PROJECT_ROOT}")
print(f"📁 現在のディレクトリ: {CURRENT_DIR}")

# main.py実証済みモジュールのインポート
try:
    # Step 1-1 モジュール
    from config.logger_config import setup_logger
    from strategies.VWAP_Breakout import VWAPBreakoutStrategy
    
    # Step 1-2 データ処理モジュール
    from data_fetcher import get_parameters_and_data
    from data_processor import preprocess_data
    
    print("✅ 必須モジュールインポート成功 (Step 1-1 + 1-2)")
except ImportError as e:
    print(f"❌ モジュールインポートエラー: {e}")
    print(f"Python PATH: {sys.path[:3]}")
    raise

def setup_main_v2_logger():
    """main_v2専用ログ設定"""
    log_file_path = os.path.join(CURRENT_DIR, "logs", "main_v2.log")
    logger = setup_logger(
        name="main_v2", 
        level=logging.INFO,
        log_file=log_file_path
    )
    logger.info("=== main_v2.py Phase 1 Step 1-1 開始 ===")
    return logger

def test_vwap_strategy_import():
    """VWAPBreakoutStrategy インポートテスト (Step 1-1)"""
    try:
        # ダミーデータでクラス初期化テスト
        dummy_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Adj Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        dummy_data.set_index('Date', inplace=True)
        
        # 市場データも同様
        index_data = dummy_data.copy()
        
        # VWAPBreakoutStrategy初期化テスト
        strategy = VWAPBreakoutStrategy(
            data=dummy_data,
            index_data=index_data,
            params={'stop_loss': 0.03, 'take_profit': 0.15}
        )
        
        print("✅ VWAPBreakoutStrategy 初期化成功")
        return True, strategy
        
    except Exception as e:
        print(f"❌ VWAPBreakoutStrategy テストエラー: {e}")
        return False, None

def test_data_fetcher():
    """data_fetcher モジュールテスト (Step 1-2)"""
    try:
        print("📡 データ取得テスト開始...")
        
        # get_parameters_and_data実行テスト
        result = get_parameters_and_data()
        
        if result and len(result) == 5:
            # 正しい戻り値の順番: ticker, start_date, end_date, stock_data, index_data
            ticker, start_date, end_date, data, index_data = result
            
            print(f"  🏷️ 銘柄: {ticker}")
            print(f"  📅 期間: {start_date} ~ {end_date}")
            print(f"  📊 メインデータ形状: {data.shape if data is not None else 'None'}")
            print(f"  📈 インデックスデータ形状: {index_data.shape if index_data is not None else 'None'}")
            
            if data is not None and not data.empty:
                print(f"  📅 データ期間: {data.index[0]} ~ {data.index[-1]}")
                
                # パラメータ辞書を作成
                params = {
                    'ticker': ticker,
                    'start_date': start_date,
                    'end_date': end_date
                }
                
                return True, (data, index_data, params)
            else:
                print("  ⚠️ 空のデータが返されました")
                return False, None
        else:
            print(f"  ❌ データ取得結果が不正です (期待値: 5, 実際: {len(result) if result else 'None'})")
            return False, None
            
    except Exception as e:
        print(f"❌ data_fetcher テストエラー: {e}")
        import traceback
        print(f"詳細エラー: {traceback.format_exc()}")
        return False, None

def test_data_processor(raw_data):
    """data_processor モジュールテスト (Step 1-2)"""
    try:
        print("🔧 データ前処理テスト開始...")
        
        if raw_data is None or raw_data.empty:
            print("  ❌ 処理対象データがありません")
            return False, None
            
        # preprocess_data実行
        processed_data = preprocess_data(raw_data)
        
        if processed_data is not None and not processed_data.empty:
            print(f"  📊 処理後データ形状: {processed_data.shape}")
            print(f"  🏷️ カラム数: {len(processed_data.columns)}")
            
            # 追加された計算カラムの確認
            added_columns = []
            if 'ATR' in processed_data.columns:
                added_columns.append('ATR')
            if 'Returns' in processed_data.columns:
                added_columns.append('Returns')
            if 'Volatility' in processed_data.columns:
                added_columns.append('Volatility')
                
            print(f"  ➕ 追加カラム: {added_columns}")
            return True, processed_data
        else:
            print("  ❌ データ前処理が失敗しました")
            return False, None
            
    except Exception as e:
        print(f"❌ data_processor テストエラー: {e}")
        return False, None

def test_integrated_data_pipeline():
    """統合データパイプラインテスト (Step 1-2)"""
    try:
        print("\n🔄 統合データパイプラインテスト開始")
        
        # 1. データ取得
        fetch_success, fetch_result = test_data_fetcher()
        if not fetch_success:
            return False, None
            
        data, index_data, params = fetch_result
        
        # 2. データ前処理
        process_success, processed_data = test_data_processor(data)
        if not process_success:
            return False, None
            
        # 3. インデックスデータも処理（必要に応じて）
        processed_index = None
        if index_data is not None and not index_data.empty:
            _, processed_index = test_data_processor(index_data)
            
        print("✅ 統合データパイプライン成功")
        return True, (processed_data, processed_index, params)
        
    except Exception as e:
        print(f"❌ 統合データパイプラインエラー: {e}")
        return False, None

def test_vwap_strategy_with_real_data(processed_data, processed_index, params):
    """実際のデータでVWAPBreakoutStrategy実行テスト (Step 1-3)"""
    try:
        print("🎯 実データVWAPBreakoutStrategy実行テスト開始...")
        
        if processed_data is None or processed_data.empty:
            print("  ❌ 処理済みデータがありません")
            return False, None
            
        # VWAPBreakoutStrategy初期化（実データ使用）
        strategy_params = {
            'stop_loss': 0.03,
            'take_profit': 0.15,
            'sma_short': 10,
            'sma_long': 30,
            'volume_threshold': 1.2,
            'confirmation_bars': 1
        }
        
        print(f"  📊 データ形状: {processed_data.shape}")
        print(f"  🏷️ 銘柄: {params.get('ticker', 'Unknown')}")
        print(f"  📅 期間: {params.get('start_date', 'Unknown')} ~ {params.get('end_date', 'Unknown')}")
        
        # VWAPBreakoutStrategy初期化
        strategy = VWAPBreakoutStrategy(
            data=processed_data,
            index_data=processed_index if processed_index is not None else processed_data.copy(),
            params=strategy_params
        )
        
        print("  ✅ VWAPBreakoutStrategy初期化成功")
        return True, strategy
        
    except Exception as e:
        print(f"❌ VWAPBreakoutStrategy実データテストエラー: {e}")
        import traceback
        print(f"詳細エラー: {traceback.format_exc()}")
        return False, None

def test_backtest_execution(strategy, params):
    """バックテスト実行とシグナル生成確認 (Step 1-3)"""
    try:
        print("🚀 バックテスト実行テスト開始...")
        
        if strategy is None:
            print("  ❌ 戦略オブジェクトがありません")
            return False, None
            
        # バックテスト実行（必須！）
        print("  ⚡ strategy.backtest() 実行中...")
        backtest_result = strategy.backtest()
        
        if backtest_result is None:
            print("  ❌ バックテスト結果がNoneです")
            return False, None
            
        print(f"  📊 バックテスト結果形状: {backtest_result.shape}")
        print(f"  🏷️ カラム一覧: {list(backtest_result.columns)}")
        
        # Entry_Signal/Exit_Signal列の確認（必須！）
        signal_columns_found = []
        if 'Entry_Signal' in backtest_result.columns:
            signal_columns_found.append('Entry_Signal')
            entry_signals = backtest_result['Entry_Signal'].sum()
            print(f"  📈 Entry_Signal合計: {entry_signals}")
        
        if 'Exit_Signal' in backtest_result.columns:
            signal_columns_found.append('Exit_Signal')
            exit_signals = backtest_result['Exit_Signal'].sum()
            print(f"  📉 Exit_Signal合計: {exit_signals}")
            
        if not signal_columns_found:
            print("  ❌ Entry_Signal/Exit_Signal列が見つかりません")
            return False, None
            
        print(f"  ✅ シグナル列確認: {signal_columns_found}")
        
        # 実際のトレード数確認（必須！）
        if 'Entry_Signal' in backtest_result.columns:
            actual_trades = backtest_result['Entry_Signal'].sum()
            if actual_trades > 0:
                print(f"  ✅ 実際のトレード数: {actual_trades}回")
            else:
                print("  ⚠️ トレード数が0です - パラメータ調整が必要かもしれません")
        
        return True, backtest_result
        
    except Exception as e:
        print(f"❌ バックテスト実行エラー: {e}")
        import traceback
        print(f"詳細エラー: {traceback.format_exc()}")
        return False, None

def test_strategy_execution_integration():
    """戦略実行統合テスト (Step 1-3)"""
    try:
        print("\n🎯 戦略実行統合テスト開始")
        
        # 1. データパイプライン実行
        pipeline_success, pipeline_result = test_integrated_data_pipeline()
        if not pipeline_success:
            return False, None
            
        processed_data, processed_index, params = pipeline_result
        
        # 2. 実データで戦略初期化
        strategy_success, strategy = test_vwap_strategy_with_real_data(
            processed_data, processed_index, params
        )
        if not strategy_success:
            return False, None
            
        # 3. バックテスト実行とシグナル確認
        backtest_success, backtest_result = test_backtest_execution(strategy, params)
        if not backtest_success:
            return False, None
            
        print("✅ 戦略実行統合成功")
        return True, (strategy, backtest_result, params)
        
    except Exception as e:
        print(f"❌ 戦略実行統合エラー: {e}")
        return False, None

def run_step_1_1():
    """Step 1-1: VWAPBreakoutStrategy + 必須設定モジュール統合"""
    print("� Step 1-1: VWAPBreakoutStrategy + 必須設定モジュール統合")
    
    try:
        # Step 1-1-1: ログ設定テスト
        print("  📝 ログ設定テスト")
        logger = setup_main_v2_logger()
        logger.info("ログ設定完了")
        print("  ✅ ログ設定成功")
        
        # Step 1-1-2: VWAPBreakoutStrategy統合テスト
        print("  🎯 VWAPBreakoutStrategy統合テスト")
        success, _ = test_vwap_strategy_import()
        
        if success:
            logger.info("VWAPBreakoutStrategy統合成功")
            print("  ✅ VWAPBreakoutStrategy統合成功")
            return True, logger
        else:
            logger.error("VWAPBreakoutStrategy統合失敗")
            print("  ❌ VWAPBreakoutStrategy統合失敗")
            return False, logger
            
    except Exception as e:
        print(f"  ❌ Step 1-1 エラー: {e}")
        return False, None

def run_step_1_2(logger):
    """Step 1-2: データ処理実装 (data_fetcher + data_processor統合)"""
    print("\n� Step 1-2: データ処理実装")
    
    try:
        # Step 1-2-1: 統合データパイプラインテスト
        print("  🔄 統合データパイプラインテスト")
        pipeline_success, pipeline_result = test_integrated_data_pipeline()
        
        if pipeline_success:
            logger.info("データパイプライン統合成功")
            print("  ✅ データパイプライン統合成功")
            
            # パイプライン結果の詳細表示
            if pipeline_result:
                processed_data, processed_index, params = pipeline_result
                print(f"  📊 処理済みデータ: {processed_data.shape if processed_data is not None else 'None'}")
                print(f"  📈 処理済みインデックス: {processed_index.shape if processed_index is not None else 'None'}")
                print(f"  ⚙️ パラメータ: {len(params) if params else 0}件")
                
            return True, pipeline_result
        else:
            logger.error("データパイプライン統合失敗")
            print("  ❌ データパイプライン統合失敗")
            return False, None
            
    except Exception as e:
        print(f"  ❌ Step 1-2 エラー: {e}")
        logger.error(f"Step 1-2 エラー: {e}")
        return False, None

def run_step_1_3(logger):
    """Step 1-3: 戦略実行統合 (VWAPBreakoutStrategy + 実バックテスト)"""
    print("\n📋 Step 1-3: 戦略実行統合")
    
    try:
        # Step 1-3-1: 戦略実行統合テスト
        print("  🎯 戦略実行統合テスト")
        execution_success, execution_result = test_strategy_execution_integration()
        
        if execution_success:
            logger.info("戦略実行統合成功")
            print("  ✅ 戦略実行統合成功")
            
            # 実行結果の詳細表示
            if execution_result:
                strategy, backtest_result, params = execution_result
                print(f"  📊 バックテスト結果形状: {backtest_result.shape if backtest_result is not None else 'None'}")
                
                # シグナル列の確認
                if backtest_result is not None:
                    signal_info = []
                    if 'Entry_Signal' in backtest_result.columns:
                        entry_count = backtest_result['Entry_Signal'].sum()
                        signal_info.append(f"Entry: {entry_count}")
                    if 'Exit_Signal' in backtest_result.columns:
                        exit_count = backtest_result['Exit_Signal'].sum()
                        signal_info.append(f"Exit: {exit_count}")
                    print(f"  🎯 シグナル生成: {', '.join(signal_info)}")
                
            return True, execution_result
        else:
            logger.error("戦略実行統合失敗")
            print("  ❌ 戦略実行統合失敗")
            return False, None
            
    except Exception as e:
        print(f"  ❌ Step 1-3 エラー: {e}")
        logger.error(f"Step 1-3 エラー: {e}")
        return False, None

def main():
    """
    main_v2.py Phase 1 Step 1-1 & 1-2 & 1-3 実行
    
    Step 1-1: VWAPBreakoutStrategy + 必須設定モジュール統合 ✅
    Step 1-2: データ処理実装 (data_fetcher + data_processor統合) ✅
    Step 1-3: 戦略実行統合 (実バックテスト + シグナル生成確認) 🆕
    
    必須チェック:
    - モジュールインポート確認
    - logger_config動作確認  
    - VWAPBreakoutStrategy初期化確認
    - data_fetcher動作確認
    - data_processor動作確認
    - 統合データパイプライン確認
    - 実バックテスト実行確認
    - Entry_Signal/Exit_Signal生成確認
    """
    print("🚀 main_v2.py Phase 1 Step 1-1 & 1-2 & 1-3 開始")
    print("完全統合バックテストシステム + VWAPBreakoutStrategy実証")
    
    try:
        # Step 1-1 実行
        step_1_1_success, logger = run_step_1_1()
        if not step_1_1_success:
            print("❌ Phase 1 失敗: Step 1-1で中断")
            return False
            
        # Step 1-2 実行
        step_1_2_success, pipeline_result = run_step_1_2(logger)
        if not step_1_2_success:
            print("❌ Phase 1 失敗: Step 1-2で中断")
            return False
            
        # Step 1-3 実行
        step_1_3_success, execution_result = run_step_1_3(logger)
        if not step_1_3_success:
            print("❌ Phase 1 失敗: Step 1-3で中断")
            return False
        
        # Phase 1 Step 1-1 & 1-2 & 1-3 完了報告
        print("\n🎉 Phase 1 Step 1-1 & 1-2 & 1-3 完了!")
        print("📊 完了項目:")
        print("  ✅ Step 1-1: VWAPBreakoutStrategy + logger統合")
        print("  ✅ Step 1-2: データ処理パイプライン統合")
        print("  ✅ Step 1-3: 戦略実行統合 + 実バックテスト")
        print("  ✅ data_fetcher.get_parameters_and_data 動作確認")
        print("  ✅ data_processor.preprocess_data 動作確認")
        print("  ✅ 統合データパイプライン動作確認")
        print("  ✅ VWAPBreakoutStrategy.backtest() 実行確認")
        print("  ✅ Entry_Signal/Exit_Signal生成確認")
        
        logger.info("Phase 1 Step 1-1 & 1-2 & 1-3 完了 - VWAPBreakoutStrategy実証成功")
        print("\n🚀 Phase 1 完了! VWAPBreakoutStrategy単体バックテストシステム構築成功")
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 実行エラー: {e}")
        return False

if __name__ == "__main__":
    main()