#!/usr/bin/env python3
"""
修正後DSSMSExcel詳細確認ツール
作成日: 2025-09-08

目的:
- 戦略別統計シートの内容を詳細確認
- 保有期間データの詳細分析
- 完全修正の確認
"""

import pandas as pd
import json
from pathlib import Path
# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl
# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
from src.utils.openpyxl_lazy_wrapper import load_workbook

def analyze_fixed_dssms_excel():
    """修正後のDSSMSExcelファイルを詳細確認"""
    print("=" * 80)
    print("修正後DSSMSExcel詳細確認")
    print("=" * 80)
    
    # 最新ファイルを特定
    results_dir = Path("backtest_results/dssms_results")
    excel_files = list(results_dir.glob("dssms_unified_backtest_*.xlsx"))
    
    if not excel_files:
        print("[ERROR] DSSMSバックテストExcelファイルが見つかりません")
        return None
    
    # 最新ファイルを取得
    latest_file = sorted(excel_files, key=lambda x: x.name)[-1]
    print(f"📄 分析対象ファイル: {latest_file}")
    
    try:
        # Excelファイルを読み込み
        workbook = load_workbook(latest_file)
        print(f"[LIST] 利用可能シート: {workbook.sheetnames}")
        
        # 1. 戦略別統計シートの詳細確認
        if "戦略別統計" in workbook.sheetnames:
            print("\n[SEARCH] 戦略別統計シート詳細分析:")
            strategy_sheet = workbook["戦略別統計"]
            
            # シートの内容を読み取り
            strategy_data = []
            for row in strategy_sheet.iter_rows(values_only=True):
                if any(cell is not None for cell in row):
                    strategy_data.append(row)
            
            print(f"   [CHART] 戦略別統計シート行数: {len(strategy_data)}")
            
            if strategy_data:
                # ヘッダー行を表示
                print(f"   📝 ヘッダー: {strategy_data[0]}")
                
                # 戦略データを表示
                for i, row in enumerate(strategy_data[1:8]):  # 戦略データ行のみ（最大7戦略）
                    if row and row[0]:
                        print(f"   [UP] 戦略{i+1}: {row}")
                
                # 戦略名の確認
                strategy_names = [row[0] for row in strategy_data[1:] if row and row[0]]
                print(f"   🏷️  戦略名リスト: {strategy_names}")
                
                if len(strategy_names) == 7:
                    print("   [OK] 7つの個別戦略が正常に表示されています")
                elif "DSSMSStrategy" in strategy_names and len(strategy_names) == 1:
                    print("   [ERROR] 依然として単一戦略のみ表示されています")
                else:
                    print(f"   [WARNING]  戦略数が想定外: {len(strategy_names)}個")
        
        # 2. 取引履歴シートの保有期間詳細分析
        if "取引履歴" in workbook.sheetnames:
            print("\n[SEARCH] 取引履歴シート保有期間詳細分析:")
            trade_sheet = workbook["取引履歴"]
            
            # DataFrameとして読み込み
            trade_df = pd.read_excel(latest_file, sheet_name="取引履歴")
            
            print(f"   [CHART] 取引履歴行数: {len(trade_df)}")
            
            if "保有期間" in trade_df.columns:
                holding_periods = trade_df["保有期間"].dropna()
                unique_periods = holding_periods.unique()
                
                print(f"   ⏱️  ユニーク保有期間数: {len(unique_periods)}")
                print(f"   [LIST] 保有期間の種類: {list(unique_periods[:10])}")  # 最初の10種類
                
                # 統計分析
                period_values = []
                for period in holding_periods:
                    if period and isinstance(period, str) and "時間" in period:
                        try:
                            value = float(period.replace("時間", ""))
                            period_values.append(value)
                        except:
                            pass
                
                if period_values:
                    print(f"   [UP] 保有期間統計:")
                    print(f"      平均: {pd.Series(period_values).mean():.2f}時間")
                    print(f"      最小: {min(period_values):.2f}時間")
                    print(f"      最大: {max(period_values):.2f}時間")
                    print(f"      標準偏差: {pd.Series(period_values).std():.2f}時間")
                    
                    if pd.Series(period_values).std() < 0.1:
                        print("   [ERROR] 保有期間が固定値（24時間固定問題継続）")
                    else:
                        print("   [OK] 保有期間に多様性があります")
        
        # 3. 対応JSONファイルの戦略統計詳細確認
        json_file = latest_file.with_suffix('').name.replace('dssms_unified_backtest_', 'dssms_unified_data_') + '.json'
        json_path = results_dir / json_file
        
        if json_path.exists():
            print("\n[SEARCH] JSON戦略統計詳細分析:")
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if "strategy_statistics" in json_data:
                strategy_stats = json_data["strategy_statistics"]
                print(f"   [CHART] 戦略統計キー数: {len(strategy_stats)}")
                
                for strategy_name, stats in strategy_stats.items():
                    print(f"   [UP] {strategy_name}:")
                    print(f"      取引数: {stats.get('trade_count', 0)}")
                    print(f"      勝率: {stats.get('win_rate', 0):.2%}")
                    print(f"      総損益: {stats.get('total_pnl', 0):.2f}")
        
        # 4. 結果サマリー
        print("\n[LIST] 修正結果サマリー:")
        issues_resolved = []
        issues_remaining = []
        
        # JSON戦略統計の確認
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            strategy_stats = json_data.get("strategy_statistics", {})
            
            if len(strategy_stats) == 7:
                issues_resolved.append("JSON戦略統計: 7つの個別戦略正常生成")
            elif len(strategy_stats) == 1:
                issues_remaining.append("JSON戦略統計: 依然として単一戦略のみ")
        
        # Excel戦略統計の確認（シート名の問題）
        if "戦略別統計" in workbook.sheetnames:
            issues_resolved.append("Excel戦略統計シート: 存在（シート名は「戦略別統計」）")
        else:
            issues_remaining.append("Excel戦略統計シート: 見つからない")
        
        print("   [OK] 解決済み問題:")
        for issue in issues_resolved:
            print(f"      - {issue}")
        
        if issues_remaining:
            print("   [ERROR] 残存問題:")
            for issue in issues_remaining:
                print(f"      - {issue}")
        
        if not issues_remaining:
            print("   [SUCCESS] すべての問題が解決されました！")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 分析中にエラーが発生: {e}")
        return False

def main():
    """メイン実行関数"""
    analyze_fixed_dssms_excel()

if __name__ == "__main__":
    main()
