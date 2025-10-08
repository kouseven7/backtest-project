"""
main.py経由でのExcel出力テスト

trade_simulation.pyの修正がmain.py → excel_result_exporter.py経由で
正しくExcelに出力されるかをテストします。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# プロジェクトパスを追加
sys.path.append(os.path.dirname(__file__))

def create_test_excel_data():
    """Excel出力テスト用のサンプルデータ作成"""
    from trade_simulation import simulate_trades
    
    # 20日間のテストデータ作成（より多くの取引が発生するように）
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    
    # より多くの取引シグナルを生成
    test_data = pd.DataFrame({
        'Adj Close': [100, 102, 105, 103, 107, 110, 108, 112, 115, 118, 
                     120, 119, 122, 125, 123, 128, 130, 127, 132, 135],
        'Entry_Signal': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                        1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        'Exit_Signal': [0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
                       0, 0, -1, 0, 0, -1, 0, 0, -1, 0],
        'Strategy': ['VWAPBreakoutStrategy'] * 20,
        'Position_Size': [1.0] * 20,
        'Partial_Exit': [0.0] * 20
    }, index=dates)
    
    return test_data

# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def test_excel_output():
    """Excel出力のテスト"""
    print("=== Excel出力テスト ===")
    
    try:
        # テストデータで取引シミュレーション実行
        test_data = create_test_excel_data()
        from trade_simulation import simulate_trades
        
        print("1. 取引シミュレーション実行...")
        result = simulate_trades(test_data, "TEST_EXCEL")
        print("   [OK] 取引シミュレーション成功")
        
        # Excel出力テスト
        print("2. Excel出力テスト...")
        from output.excel_result_exporter import save_backtest_results
        
        output_dir = "test_backtest_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_backtest_results_{timestamp}"
        
        # Excel出力実行
        filepath = save_backtest_results(result, output_dir, filename)
        print(f"   [OK] Excel出力成功: {filepath}")
        
        # 出力されたファイルの確認
        if os.path.exists(filepath):
            print(f"   [OK] ファイルが正常に作成されました: {filepath}")
            
            # ファイルサイズをチェック
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"   ファイルサイズ: {file_size:.1f} KB")
            
            # Excelファイルを読み込んで内容を確認
            try:
                excel_file = pd.ExcelFile(filepath)
                sheet_names = excel_file.sheet_names
                print(f"   Excel シート一覧: {sheet_names}")
                
                # 各シートの内容を確認
                for sheet_name in sheet_names:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    print(f"   シート '{sheet_name}': {len(df)} 行, {len(df.columns)} 列")
                    
                    # 特定のシートの内容詳細チェック
                    if sheet_name == "取引履歴" and len(df) > 0:
                        print(f"      取引履歴の列: {list(df.columns)}")
                        print(f"      取引数: {len(df)} 件")
                        
                        # リスク状態列がないことを確認
                        if 'リスク状態' not in df.columns:
                            print("      [OK] リスク状態列が正しく削除されています")
                        else:
                            print("      [ERROR] リスク状態列が残っています")
                            
                        # 取引量(株)列があることを確認
                        if '取引量(株)' in df.columns:
                            print("      [OK] 取引量(株)列が追加されています")
                            if len(df) > 0:
                                sample_shares = df['取引量(株)'].iloc[0]
                                print(f"      株数の例: {sample_shares}")
                        else:
                            print("      [ERROR] 取引量(株)列がありません")
                    
                    elif sheet_name == "パフォーマンス指標":
                        # 高度なパフォーマンス指標があることを確認
                        if len(df) > 0 and '指標' in df.columns:
                            metrics_list = df['指標'].tolist()
                            advanced_metrics = ['シャープレシオ', 'ソルティノレシオ', '期待値']
                            
                            for metric in advanced_metrics:
                                if metric in metrics_list:
                                    print(f"      [OK] {metric}が含まれています")
                                else:
                                    print(f"      [ERROR] {metric}が含まれていません")
                    
                    elif sheet_name == "損益推移":
                        # 累積損益計算の確認
                        if len(df) > 0 and '累積損益' in df.columns:
                            print("      [OK] 累積損益が正しく計算されています")
                            final_pnl = df['累積損益'].iloc[-1]
                            print(f"      最終累積損益: {final_pnl:.2f}円")
                        else:
                            print("      [ERROR] 累積損益計算に問題があります")
                
                print("   [OK] Excel内容確認完了")
                
            except Exception as e:
                print(f"   [ERROR] Excelファイル読み込みエラー: {e}")
        else:
            print(f"   [ERROR] ファイルが作成されませんでした")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Excel出力テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_integration():
    """main.pyとの統合テスト（簡易版）"""
    print("\n=== main.py統合テスト ===")
    
    try:
        # simulation_handlerを通じたテスト
        from output.simulation_handler import simulate_and_save
        
        test_data = create_test_excel_data()
        
        print("1. simulation_handler経由でのテスト...")
        
        # simulate_and_save関数のテスト
        output_path = simulate_and_save(test_data, "TEST_MAIN")
        
        if output_path and os.path.exists(output_path):
            print(f"   [OK] main.py統合テスト成功: {output_path}")
            
            # ファイルサイズをチェック
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"   ファイルサイズ: {file_size:.1f} KB")
            
            return True
        else:
            print("   [ERROR] ファイル作成に失敗しました")
            return False
        
    except Exception as e:
        print(f"[ERROR] main.py統合テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("[SEARCH] trade_simulation.py修正のExcel出力テスト開始")
    
    # Excel出力テスト
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: excel_success = test_excel_output()
    
    # main.py統合テスト
    main_success = test_main_integration()
    
    print("\n=== テスト結果まとめ ===")
    if excel_success:
        print("[OK] Excel出力テスト: 成功")
    else:
        print("[ERROR] Excel出力テスト: 失敗")
        
    if main_success:
        print("[OK] main.py統合テスト: 成功")
    else:
        print("[ERROR] main.py統合テスト: 失敗")
    
    if excel_success and main_success:
        print("\n[SUCCESS] すべてのテストが成功しました！")
        print("trade_simulation.pyの修正がExcel出力まで正しく反映されています。")
    else:
        print("\n[WARNING] 一部のテストで問題が見つかりました。")
