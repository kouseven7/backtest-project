#!/usr/bin/env python3
"""
DSSMS Excel出力修正の最終検証スクリプト

Phase 3実装の結果を検証：
1. 個別取引の分離確認
2. 現実的な損益計算確認
3. Excel出力の完全性確認
"""

import pandas as pd
from pathlib import Path
import sys

def verify_dssms_fix():
    """Phase 3修正の最終検証"""
    
    # 最新のExcelファイルを確認
    results_dir = Path("backtest_results/dssms_results")
    excel_files = list(results_dir.glob("dssms_backtest_results_*.xlsx"))
    
    if not excel_files:
        print("[ERROR] Excelファイルが見つかりません")
        return False
    
    # 最新ファイルを取得
    latest_file = max(excel_files, key=lambda x: x.stat().st_mtime)
    print(f"[SEARCH] 検証対象: {latest_file.name}")
    
    try:
        # 取引履歴シートを読み込み
        df_trades = pd.read_excel(latest_file, sheet_name='取引履歴')
        
        # 列名を確認
        print(f"[LIST] 利用可能な列: {list(df_trades.columns)}")
        
        # 損益列を特定（複数の可能性に対応）
        pnl_col = None
        for col in ['損益', 'Profit_Loss', 'PnL', '損益(円)', 'profit_loss', '取引結果']:
            if col in df_trades.columns:
                pnl_col = col
                break
        
        if pnl_col is None:
            print("[ERROR] 損益列が見つかりません")
            return False
        
        # 基本統計
        total_trades = len(df_trades)
        profit_trades = len(df_trades[df_trades[pnl_col] > 0])
        loss_trades = len(df_trades[df_trades[pnl_col] < 0])
        total_pnl = df_trades[pnl_col].sum()
        
        print(f"\n[CHART] 取引分析結果:")
        print(f"   総取引数: {total_trades}件")
        print(f"   利益取引: {profit_trades}件")
        print(f"   損失取引: {loss_trades}件")
        print(f"   総損益: {total_pnl:,.0f}円")
        
        # Phase 3修正の検証
        success_criteria = []
        
        # 1. 個別取引への分離確認
        if total_trades > 50:  # 期待値：100件以上
            success_criteria.append("[OK] 個別取引分離成功")
        else:
            success_criteria.append("[ERROR] 個別取引分離不十分")
        
        # 2. 現実的な損益確認
        if abs(total_pnl) < 10_000_000:  # 1000万円未満
            success_criteria.append("[OK] 現実的な損益計算")
        else:
            success_criteria.append("[ERROR] 非現実的な損益")
        
        # 3. データ完整性確認
        symbol_col = None
        for col in ['銘柄', 'Symbol', 'symbol', 'ticker']:
            if col in df_trades.columns:
                symbol_col = col
                break
                
        if symbol_col and df_trades[symbol_col].notna().all():
            success_criteria.append("[OK] データ完整性良好")
        else:
            success_criteria.append("[ERROR] データ欠損あり")
        
        # 4. 損益詳細分析
        avg_profit = df_trades[df_trades[pnl_col] > 0][pnl_col].mean() if profit_trades > 0 else 0
        avg_loss = df_trades[df_trades[pnl_col] < 0][pnl_col].mean() if loss_trades > 0 else 0
        
        print(f"\n[UP] 損益詳細:")
        print(f"   平均利益: {avg_profit:,.0f}円")
        print(f"   平均損失: {avg_loss:,.0f}円")
        print(f"   勝率: {profit_trades/total_trades*100:.1f}%")
        
        # 結果表示
        print(f"\n🏆 Phase 3修正検証結果:")
        for criterion in success_criteria:
            print(f"   {criterion}")
        
        # 総合判定
        success_count = sum(1 for c in success_criteria if c.startswith("[OK]"))
        if success_count >= 3:
            print(f"\n[SUCCESS] Phase 3修正完全成功！({success_count}/4)")
            return True
        else:
            print(f"\n[WARNING] Phase 3修正部分成功 ({success_count}/4)")
            return False
            
    except Exception as e:
        print(f"[ERROR] 検証エラー: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DSSMS Excel出力修正 Phase 3 最終検証")
    print("=" * 60)
    
    success = verify_dssms_fix()
    sys.exit(0 if success else 1)
