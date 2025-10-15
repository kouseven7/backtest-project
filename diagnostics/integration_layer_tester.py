"""
同日Entry/Exit問題診断 - 統合レイヤーテスター
main.pyのマルチ戦略統合処理で同日Entry/Exit問題が発生するかを検証
"""
import sys
import os
import pandas as pd
import json
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger

logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\diagnostics\integration_diagnosis.log")

def test_main_integration_layer():
    """main.pyの統合レイヤーで同日Entry/Exit問題を診断"""
    
    logger.info("=== main.py統合レイヤー診断開始 ===")
    
    try:
        # main.pyから統合関数をインポート
        from main import apply_strategies_with_optimized_params, load_optimized_parameters
        from data_fetcher import get_parameters_and_data
        from data_processor import preprocess_data
        from indicators.indicator_calculator import compute_indicators
        
        # データ準備
        ticker = "^N225"
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(ticker)
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        
        logger.info(f"統合レイヤー診断用データ: {len(stock_data)}行, 期間: {stock_data.index[0]} - {stock_data.index[-1]}")
        
        # 最適化パラメータ取得
        optimized_params = load_optimized_parameters(ticker)
        
        # main.pyの統合処理実行（ここで複数戦略が統合される）
        logger.info("main.py apply_strategies_with_optimized_params 実行開始")
        integrated_result = apply_strategies_with_optimized_params(stock_data, index_data, optimized_params)
        
        if integrated_result is None:
            logger.error("統合処理結果がNone")
            return None
            
        # 統合後の同日Entry/Exit検出
        same_day_entries_exits = []
        
        if 'Entry_Signal' in integrated_result.columns and 'Exit_Signal' in integrated_result.columns:
            for date in integrated_result.index:
                entry_val = integrated_result.loc[date, 'Entry_Signal']
                exit_val = integrated_result.loc[date, 'Exit_Signal']
                
                if entry_val == 1 and exit_val == 1:
                    same_day_info = {
                        'date': date.strftime('%Y-%m-%d'),
                        'entry_signal': int(entry_val),
                        'exit_signal': int(exit_val),
                        'active_strategy': integrated_result.loc[date, 'Active_Strategy'] if 'Active_Strategy' in integrated_result.columns else 'Unknown'
                    }
                    same_day_entries_exits.append(same_day_info)
                    
                    # 要求仕様のログ出力
                    logger.warning(f"main.py統合処理で同日Entry/Exit検出: {date.strftime('%Y-%m-%d')} Entry={entry_val} Exit={exit_val}")
        
        # 統計情報
        entry_signals = integrated_result[integrated_result['Entry_Signal'] == 1] if 'Entry_Signal' in integrated_result.columns else pd.DataFrame()
        exit_signals = integrated_result[integrated_result['Exit_Signal'] == 1] if 'Exit_Signal' in integrated_result.columns else pd.DataFrame()
        
        # 正常パターンのログ出力
        if not same_day_entries_exits:
            if len(entry_signals) > 0 and len(exit_signals) > 0:
                first_entry = entry_signals.index[0].strftime('%Y-%m-%d')
                first_exit = exit_signals.index[0].strftime('%Y-%m-%d')
                logger.info(f"main.py統合処理 エントリーシグナル {first_entry} エグジットシグナル {first_exit} 正常")
            elif len(entry_signals) > 0:
                first_entry = entry_signals.index[0].strftime('%Y-%m-%d')
                logger.info(f"main.py統合処理 エントリーシグナル {first_entry} エグジットシグナルなし 正常")
            elif len(exit_signals) > 0:
                first_exit = exit_signals.index[0].strftime('%Y-%m-%d')
                logger.info(f"main.py統合処理 エントリーシグナルなし エグジットシグナル {first_exit} 正常")
            else:
                logger.info(f"main.py統合処理 エントリー・エグジットシグナルなし 正常")
        
        # 結果構造
        diagnosis_result = {
            'layer': 'main_integration',
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'has_same_day_problem': len(same_day_entries_exits) > 0,
            'same_day_entries_exits': same_day_entries_exits,
            'signal_summary': {
                'total_entries': len(entry_signals),
                'total_exits': len(exit_signals),
                'entry_dates': entry_signals.index.strftime('%Y-%m-%d').tolist() if len(entry_signals) > 0 else [],
                'exit_dates': exit_signals.index.strftime('%Y-%m-%d').tolist() if len(exit_signals) > 0 else []
            },
            'columns_present': list(integrated_result.columns),
            'data_shape': integrated_result.shape
        }
        
        logger.info(f"main.py統合処理診断完了: {'同日Entry/Exit問題検出' if diagnosis_result['has_same_day_problem'] else '同日Entry/Exit問題なし'}")
        
        return diagnosis_result
        
    except Exception as e:
        logger.error(f"main.py統合レイヤー診断エラー: {e}")
        return {
            'layer': 'main_integration',
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e)
        }

def run_integration_layer_diagnostics():
    """統合レイヤー診断実行"""
    
    print("=== 同日Entry/Exit問題診断 - 統合レイヤー ===")
    
    # 統合レイヤー診断
    integration_result = test_main_integration_layer()
    
    # 結果保存
    diagnosis_summary = {
        'diagnosis_timestamp': datetime.now().isoformat(),
        'layer': 'integration',
        'description': 'main.py multi-strategy integration layer diagnosis',
        'integration_result': integration_result
    }
    
    # JSON保存
    output_file = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\integration_layer_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diagnosis_summary, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"統合レイヤー診断結果保存: {output_file}")
    
    if integration_result and integration_result.get('status') == 'completed':
        problem_detected = integration_result.get('has_same_day_problem', False)
        problem_count = len(integration_result.get('same_day_entries_exits', []))
        
        print(f"\n統合レイヤー診断完了:")
        print(f"- 同日Entry/Exit問題: {'検出' if problem_detected else '検出されず'}")
        if problem_detected:
            print(f"- 問題検出件数: {problem_count}")
        print(f"- 結果ファイル: diagnostics/integration_layer_results.json")
    else:
        print(f"\n統合レイヤー診断失敗")
        if integration_result:
            print(f"- エラー: {integration_result.get('error', 'Unknown error')}")
    
    return diagnosis_summary

if __name__ == "__main__":
    results = run_integration_layer_diagnostics()