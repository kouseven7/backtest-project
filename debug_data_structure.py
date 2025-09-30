#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSSMSデータ構造調査スクリプト
Excel出力で0%となる原因調査
"""

import json
import sys
from datetime import datetime, timedelta

sys.path.append('.')
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

def debug_data_structure():
    """バックテスト結果のデータ構造を詳細調査"""
    print('=== DSSMS結果構造テスト ===')
    
    # テストラン実行
    backtester = DSSMSIntegratedBacktester()
    start_date = datetime(2023, 1, 2)
    end_date = datetime(2023, 1, 5)  # 短期間でテスト
    
    result = backtester.run_dynamic_backtest(start_date, end_date)
    
    print('\n=== 結果のキー構造 ===')
    for key in result.keys():
        print(f'{key}: {type(result[key])}')
    
    print('\n=== portfolio_performance構造 ===')
    portfolio_perf = result.get('portfolio_performance', {})
    for key, value in portfolio_perf.items():
        print(f'{key}: {value} ({type(value)})')
    
    print('\n=== switch_statistics構造 ===')
    switch_stats = result.get('switch_statistics', {})
    for key, value in switch_stats.items():
        print(f'{key}: {value} ({type(value)})')
    
    print('\n=== execution_metadata構造 ===')
    exec_meta = result.get('execution_metadata', {})
    for key, value in exec_meta.items():
        print(f'{key}: {value} ({type(value)})')
    
    print('\n=== 重要値確認 ===')
    print(f"初期資本: {portfolio_perf.get('initial_capital', 'N/A'):,}円")
    print(f"最終資本: {portfolio_perf.get('final_capital', 'N/A'):,}円")
    print(f"総収益: {portfolio_perf.get('total_return', 'N/A'):,}円")
    print(f"総収益率: {portfolio_perf.get('total_return_rate', 'N/A'):.4%}")
    print(f"切替回数: {switch_stats.get('total_switches', 'N/A')}回")
    print(f"成功率: {switch_stats.get('success_rate', 'N/A'):.2%}")
    
    print('\n=== JSON出力（調査用） ===')
    debug_output = {
        'portfolio_performance': portfolio_perf,
        'switch_statistics': switch_stats,
        'execution_metadata': exec_meta
    }
    
    with open('debug_data_structure.json', 'w', encoding='utf-8') as f:
        json.dump(debug_output, f, indent=2, ensure_ascii=False, default=str)
    
    print("詳細データ構造をdebug_data_structure.jsonに出力しました")
    
    return result

if __name__ == "__main__":
    result = debug_data_structure()