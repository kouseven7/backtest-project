#!/usr/bin/env python3
"""
DSSMSバックテスターのExcel出力機能テスト

改善されたExcel出力機能を包括的にテストし、
取引履歴とパフォーマンス指標が正しく出力されることを確認する。
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
from src.dssms.dssms_backtester import DSSMSBacktester, DSSMSPerformanceMetrics

def setup_test_logger():
    """テスト用ロガーを設定"""
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: logger = setup_logger("DSSMS_Excel_Test", log_file="test_dssms_excel_output.log")
    logger.setLevel(logging.INFO)
    return logger

def create_test_config():
    """テスト用設定を作成"""
    return {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'initial_capital': 1000000,  # 100万円
        'commission_rate': 0.001,    # 0.1%手数料
        'switch_cost': 5000,         # 切り替えコスト
        'min_holding_period_hours': 4,  # 最低保有時間
        'volatility_threshold': 0.3,
        'momentum_threshold': 0.1,
        'correlation_threshold': 0.7,
        'output_excel': True,
        'excel_detailed': True
    }

def create_mock_market_data(symbols, start_date, end_date):
    """テスト用の市場データを作成"""
    logger = logging.getLogger("DSSMS_Excel_Test")
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # 営業日のみ（土日を除く）
    business_days = date_range[date_range.weekday < 5]
    
    market_data = {}
    
    for symbol in symbols:
        # シンボルごとに異なる価格トレンドを生成
        np.random.seed(hash(symbol) % 2**32)  # 再現可能な乱数
        
        # 基準価格
        base_prices = {
            'AAPL': 150,
            'GOOGL': 2500,
            'MSFT': 300,
            'TSLA': 800
        }
        
        base_price = base_prices.get(symbol, 100)
        prices = []
        current_price = base_price
        
        for i, date in enumerate(business_days):
            # トレンドと変動を追加
            trend = 0.001 * (i % 50 - 25)  # 周期的なトレンド
            volatility = np.random.normal(0, 0.02)  # 日次変動
            
            current_price *= (1 + trend + volatility)
            prices.append(current_price)
        
        # データフレーム作成
        df = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
            'Close': prices,
            'Adj Close': prices,
            'Volume': [np.random.randint(1000000, 5000000) for _ in prices]
        }, index=business_days)
        
        market_data[symbol] = df
        logger.info(f"{symbol}のテストデータ作成完了: {len(df)}日分")
    
    return market_data

def run_dssms_excel_test():
    """DSSMSバックテスターExcel出力テストを実行"""
    logger = setup_test_logger()
    logger.info("DSSMSバックテスターExcel出力テスト開始")
    
    try:
        import numpy as np
        
        # テスト設定
        config = create_test_config()
        
        # テスト期間（1年間）
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        
        logger.info(f"テスト期間: {start_date} - {end_date}")
        
        # モックデータ作成
        market_data = create_mock_market_data(
            config['symbols'], 
            start_date, 
            end_date
        )
        
        # DSSMSバックテスター初期化
        backtester = DSSMSBacktester(config)
        
        # バックテスト実行
        logger.info("DSSMSバックテスト実行中...")
        results = backtester.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date,
            symbol_universe=config['symbols']
        )
        
        # 結果確認
        if results and isinstance(results, dict):
            logger.info("=== バックテスト結果 ===")
            logger.info(f"実行成功: {results.get('success', 'Unknown')}")
            logger.info(f"最終価値: {results.get('final_value', 0):,.0f}円")
            logger.info(f"総収益率: {results.get('total_return', 0):.2%}")
            logger.info(f"取引日数: {results.get('trading_days', 0)}")
            logger.info(f"銘柄切り替え回数: {results.get('switch_count', 0)}")
            
            # パフォーマンス指標を簡易的に作成
            performance_metrics = DSSMSPerformanceMetrics(
                total_return=results.get('total_return', 0),
                volatility=0.2,  # 仮の値
                max_drawdown=0.1,  # 仮の値
                sharpe_ratio=1.0,  # 仮の値
                sortino_ratio=1.0,  # 仮の値
                symbol_switches_count=results.get('switch_count', 0),
                average_holding_period_hours=24,  # 仮の値
                switch_success_rate=0.6,  # 仮の値
                switch_costs_total=results.get('switch_count', 0) * 1000,  # 仮の値
                dynamic_selection_efficiency=0.8  # 仮の値
            )
            
            # Excel出力テスト
            logger.info("=== Excel出力テスト開始 ===")
            
            # 出力ディレクトリ作成
            output_dir = "test_output/dssms_excel_test"
            os.makedirs(output_dir, exist_ok=True)
            
            # Excel出力実行
            excel_path = backtester.export_results_to_excel(
                simulation_result=results,
                performance_metrics=performance_metrics,
                comparison_result={},
                output_dir=output_dir
            )
            
            if excel_path and os.path.exists(excel_path):
                logger.info(f"[SUCCESS] Excel出力成功: {excel_path}")
                
                # Excelファイルの内容確認
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: verify_excel_output(excel_path, logger)
                
                return True
            else:
                logger.error("[FAILED] Excel出力失敗")
                return False
        else:
            logger.error("[FAILED] バックテスト結果が取得できませんでした")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def verify_excel_output(excel_path, logger):
    """Excel出力内容を検証"""
    try:
        # Excelファイルを読み込んで内容確認
        excel_file = pd.ExcelFile(excel_path)
        sheets = excel_file.sheet_names
        
        logger.info(f"=== Excel内容検証 ===")
        logger.info(f"ファイルパス: {excel_path}")
        logger.info(f"シート数: {len(sheets)}")
        logger.info(f"シート名: {sheets}")
        
        # 各シートの内容確認
        for sheet_name in sheets:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                logger.info(f"  {sheet_name}: {len(df)}行 x {len(df.columns)}列")
                
                if sheet_name == '取引履歴' and not df.empty:
                    logger.info(f"    取引回数: {len(df)}回")
                    logger.info(f"    列: {list(df.columns)}")
                    
                    # 取引結果の統計
                    if '取引結果' in df.columns:
                        total_pnl = df['取引結果'].sum()
                        winning_trades = len(df[df['取引結果'] > 0])
                        losing_trades = len(df[df['取引結果'] < 0])
                        win_rate = winning_trades / len(df) * 100 if len(df) > 0 else 0
                        
                        logger.info(f"    総損益: {total_pnl:,.0f}円")
                        logger.info(f"    勝ち取引: {winning_trades}回")
                        logger.info(f"    負け取引: {losing_trades}回")
                        logger.info(f"    勝率: {win_rate:.1f}%")
                
                elif sheet_name == 'パフォーマンス指標' and not df.empty:
                    logger.info(f"    指標数: {len(df)}個")
                    
            except Exception as sheet_error:
                logger.warning(f"  {sheet_name}シート読み込みエラー: {sheet_error}")
        
        # ファイルサイズ確認
        file_size = os.path.getsize(excel_path)
        logger.info(f"ファイルサイズ: {file_size:,} bytes")
        
        # 検証結果の評価
        if len(sheets) >= 3:  # 最低限のシート数
            logger.info("[SUCCESS] Excel出力検証成功")
        else:
            logger.warning("[WARNING] 期待されるシートが不足している可能性があります")
            
    except Exception as e:
        logger.error(f"[ERROR] Excel検証エラー: {e}")

if __name__ == "__main__":
    print("DSSMSバックテスターExcel出力機能テスト")
    print("=" * 50)
    
    success = run_dssms_excel_test()
    
    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] テスト完了: 成功")
        print("Excel出力機能が正常に動作しています")
    else:
        print("[FAILED] テスト完了: 失敗")
        print("Excel出力機能に問題があります")
    
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: print("\n詳細なログは test_dssms_excel_output.log をご確認ください")
