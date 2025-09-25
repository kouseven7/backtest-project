#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
決定論的計算除去テスト
"""

from src.dssms.dssms_backtester import DSSMSBacktester
import pandas as pd
from datetime import datetime

def main():
    """短期間テスト実行"""
    print("決定論的計算除去テスト開始")
    
    # バックテスター初期化
    backtester = DSSMSBacktester()
    
    # 短期間（10日間）でテスト
    date_range = pd.date_range('2023-01-01', '2023-01-10')
    initial_capital = 100000
    
    print(f"期間: {date_range[0]} - {date_range[-1]} ({len(date_range)}日)")
    
    # シミュレーション実行（引数: start_date, end_date, symbol_universe, strategies=None）
    symbol_universe = ['7203', '9984', '6758', '7741', '4063']  # テスト用銘柄
    start_date = date_range[0].to_pydatetime()
    end_date = date_range[-1].to_pydatetime()
    
    # 初期資本はsimulate_dynamic_selectionでは設定できないため、別途設定
    backtester.initial_capital = initial_capital
    
    result = backtester.simulate_dynamic_selection(start_date, end_date, symbol_universe)
    
    # 結果出力
    switches = result.get("switches", [])
    portfolio_values = result.get("portfolio_values", [])
    
    print(f"切替回数: {len(switches)}")
    print(f"初期資本: {initial_capital:,}円")
    print(f"最終価値: {portfolio_values[-1]:,.0f}円" if portfolio_values else "データなし")
    
    if switches:
        print("\n切替履歴:")
        for i, switch in enumerate(switches[:3]):  # 最初の3件
            print(f"  {i+1}: {switch}")
    
    print("テスト完了")

if __name__ == "__main__":
    main()