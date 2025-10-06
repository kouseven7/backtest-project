#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
取引履歴データ検証スクリプト
エントリー価格、エグジット価格、損益計算、保有期間の妥当性を検証
"""

# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def analyze_trading_history():
    """取引履歴の詳細分析"""
    
    # 最新のExcelファイルを特定
    excel_dir = Path("output/dssms_integration")
    excel_files = list(excel_dir.glob("backtest_results_*.xlsx"))
    
    if not excel_files:
        print("❌ Excelファイルが見つかりません")
        return
    
    latest_file = sorted(excel_files, key=lambda x: x.name)[-1]
    print(f"📊 分析対象: {latest_file}")
    
    try:
        workbook = openpyxl.load_workbook(latest_file)
        
        if "取引履歴" not in workbook.sheetnames:
            print("❌ 取引履歴シートが見つかりません")
            return
        
        ws = workbook["取引履歴"]
        print(f"\n=== 取引履歴データ構造分析 ===")
        
        # ヘッダー行を確認
        headers = []
        for col in range(1, 15):  # A-N列まで確認
            header = ws.cell(row=1, column=col).value
            if header:
                headers.append((col, header))
        
        print(f"📋 列構造:")
        for col_idx, header in headers:
            print(f"  {chr(64+col_idx)}列: {header}")
        
        # データを取得（最初の20件）
        trades = []
        max_row = min(21, ws.max_row)  # 最初の20取引を分析
        
        for row in range(2, max_row + 1):
            trade_data = {}
            for col_idx, header in headers:
                value = ws.cell(row=row, column=col_idx).value
                trade_data[header] = value
            trades.append(trade_data)
        
        print(f"\n=== 取引データ異常検出 ===")
        print(f"分析対象: {len(trades)}件の取引")
        
        # 1. 価格データの妥当性チェック
        print(f"\n🔍 1. 価格データ分析:")
        entry_prices = set()
        exit_prices = set()
        
        for trade in trades:
            if trade.get('エントリー価格'):
                entry_prices.add(trade['エントリー価格'])
            if trade.get('エグジット価格'):
                exit_prices.add(trade['エグジット価格'])
        
        print(f"  エントリー価格の種類数: {len(entry_prices)}")
        print(f"  エントリー価格一覧: {sorted(entry_prices) if entry_prices else 'なし'}")
        print(f"  エグジット価格の種類数: {len(exit_prices)}")
        print(f"  エグジット価格一覧: {sorted(exit_prices) if exit_prices else 'なし'}")
        
        if len(entry_prices) == 1 and len(exit_prices) == 1:
            print("  ⚠️ 警告: 全取引で同一価格 - モックデータの可能性")
        
        # 2. 損益計算の妥当性チェック
        print(f"\n🔍 2. 損益計算分析:")
        calculation_errors = []
        
        for i, trade in enumerate(trades):
            if (trade.get('売買') == '売り' and 
                trade.get('エントリー価格') and 
                trade.get('エグジット価格') and 
                trade.get('数量') and
                trade.get('損益')):
                
                entry_price = float(trade['エントリー価格'])
                exit_price = float(trade['エグジット価格'])
                quantity = int(trade['数量'])
                actual_profit = float(trade['損益'])
                
                # 理論上の損益計算（簡単な計算）
                theoretical_profit = (exit_price - entry_price) * quantity
                
                if abs(actual_profit - theoretical_profit) > 0.01:  # 1円以上の差
                    calculation_errors.append({
                        'row': i + 2,
                        'date': trade.get('日付'),
                        'symbol': trade.get('銘柄コード'),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'theoretical_profit': theoretical_profit,
                        'actual_profit': actual_profit,
                        'difference': actual_profit - theoretical_profit
                    })
        
        print(f"  損益計算エラー件数: {len(calculation_errors)}")
        if calculation_errors:
            print(f"  最初の5件のエラー:")
            for error in calculation_errors[:5]:
                print(f"    {error['date']} {error['symbol']}: ")
                print(f"      理論値={error['theoretical_profit']:.2f}, 実際={error['actual_profit']:.2f}, 差={error['difference']:.2f}")
        
        # 3. 保有期間の妥当性チェック
        print(f"\n🔍 3. 保有期間分析:")
        holding_periods = set()
        
        for trade in trades:
            if trade.get('保有期間'):
                holding_periods.add(trade['保有期間'])
        
        print(f"  保有期間の種類数: {len(holding_periods)}")
        print(f"  保有期間一覧: {sorted(holding_periods) if holding_periods else 'なし'}")
        
        if len(holding_periods) == 1:
            print("  ⚠️ 警告: 全取引で同一保有期間 - 計算ロジックに問題の可能性")
        
        # 4. 銘柄切替パターンの分析
        print(f"\n🔍 4. 銘柄切替パターン分析:")
        symbols_used = set()
        switch_pattern = []
        
        for trade in trades:
            symbol = trade.get('銘柄コード')
            if symbol:
                symbols_used.add(symbol)
                if trade.get('売買') == '買い':
                    switch_pattern.append(f"買い:{symbol}")
                elif trade.get('売買') == '売り':
                    switch_pattern.append(f"売り:{symbol}")
        
        print(f"  使用銘柄数: {len(symbols_used)}")
        print(f"  使用銘柄: {sorted(symbols_used)}")
        print(f"  切替パターン（最初の10件）: {switch_pattern[:10]}")
        
        # 5. 累積損益の整合性チェック
        print(f"\n🔍 5. 累積損益整合性分析:")
        cumulative_errors = []
        expected_cumulative = 0
        
        for i, trade in enumerate(trades):
            if trade.get('損益') is not None:
                profit = float(trade['損益'])
                expected_cumulative += profit
                
                actual_cumulative = trade.get('累積損益')
                if actual_cumulative is not None:
                    actual_cumulative = float(actual_cumulative)
                    
                    if abs(expected_cumulative - actual_cumulative) > 0.01:
                        cumulative_errors.append({
                            'row': i + 2,
                            'date': trade.get('日付'),
                            'expected': expected_cumulative,
                            'actual': actual_cumulative,
                            'difference': actual_cumulative - expected_cumulative
                        })
        
        print(f"  累積損益エラー件数: {len(cumulative_errors)}")
        if cumulative_errors:
            print(f"  最初の3件のエラー:")
            for error in cumulative_errors[:3]:
                print(f"    {error['date']}: 期待値={error['expected']:.2f}, 実際={error['actual']:.2f}")
        
        # 6. データソース調査の提案
        print(f"\n=== データソース調査提案 ===")
        print("📋 以下の調査を推奨:")
        print("  1. DSSMSバックテスターの価格データ取得ロジック確認")
        print("  2. ポジション計算ロジック（_calculate_position_update）の検証")
        print("  3. 実際の株価データ vs モックデータの使用状況確認")
        print("  4. Excel出力時のデータ変換処理の確認")
        
        workbook.close()
        
    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        import traceback
        traceback.print_exc()

def check_dssms_data_sources():
    """DSSMSシステムのデータソースを確認"""
    print(f"\n=== DSSMSデータソース確認 ===")
    
    # データソースファイルの存在確認
    files_to_check = [
        "src/dssms/dssms_integrated_main.py",
        "src/dssms/dssms_excel_exporter.py", 
        "dssms_backtester_v3.py",
        "data_fetcher.py"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {file_path}: 存在")
        else:
            print(f"❌ {file_path}: 不存在")
    
    # モックデータ使用の可能性チェック
    try:
        # data_fetcherの利用可能性確認
        try:
            import yfinance
            print("✅ yfinance: 利用可能 - 実際の株価データ取得可能")
        except ImportError:
            print("❌ yfinance: 不利用可能 - モックデータ使用の可能性")
        
        # DSSコアの利用可能性確認
        sys.path.append(".")
        try:
            from dssms_backtester_v3 import DSSBacktesterV3
            print("✅ DSS Core V3: 利用可能")
        except ImportError:
            print("❌ DSS Core V3: 不利用可能 - モック処理の可能性")
            
    except Exception as e:
        print(f"⚠️ 依存関係確認エラー: {e}")

def suggest_investigation_steps():
    """調査ステップの提案"""
    print(f"\n=== 推奨調査ステップ ===")
    print("🔍 1. 価格データ取得ロジックの確認:")
    print("   - dssms_integrated_main.py の _get_symbol_data メソッド")
    print("   - _generate_mock_data の使用状況")
    print("   - yfinanceの実際の利用状況")
    
    print("🔍 2. 損益計算ロジックの確認:")
    print("   - _calculate_position_update メソッド")
    print("   - _close_position と _open_position の実装")
    print("   - 手数料やスプレッドの考慮")
    
    print("🔍 3. Excel出力データ変換の確認:")
    print("   - dssms_excel_exporter.py の _create_trades_sheet")
    print("   - データの数値変換処理")
    print("   - 日付・時間計算の処理")
    
    print("🔍 4. デバッグ用ログ出力の追加:")
    print("   - 各取引での実際の価格データ")
    print("   - 損益計算の中間過程")
    print("   - データソースの識別（実データ vs モック）")

if __name__ == "__main__":
    print("取引履歴データ検証スクリプト")
    print("=" * 60)
    
    analyze_trading_history()
    check_dssms_data_sources()
    suggest_investigation_steps()
    
    print(f"\n✅ 分析完了")