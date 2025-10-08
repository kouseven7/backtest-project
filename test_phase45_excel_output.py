#!/usr/bin/env python3
"""
Phase 4.5.4: Excel出力統合テスト
DSSMSExcelExporter の型安全性修正後の動作確認
"""
import sys
import os
import tempfile
import logging
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dssms.dssms_excel_exporter import DSSMSExcelExporter

def test_phase45_type_safety():
    """Phase 4.5型安全性修正のテスト"""
    print("=== Phase 4.5.4 Excel出力統合テスト開始 ===")
    
    # ログ設定
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Phase 4.5.1: dict型configでの初期化テスト（元のエラー原因）
    print("\n1. dict型config初期化テスト")
    dict_config = {
        'initial_capital': 1500000,
        'other_settings': 'value'
    }
    
    try:
        exporter = DSSMSExcelExporter(config=dict_config, logger=logger)
        print(f"[OK] dict型config初期化成功: initial_capital = {exporter.initial_capital:,.0f}円")
    except Exception as e:
        print(f"[ERROR] dict型config初期化失敗: {e}")
        return False
    
    # Phase 4.5.2: 問題のあるresult データでのテスト
    print("\n2. 型安全なf-string書式テスト")
    problematic_result = {
        'execution_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'backtest_period': "2024-01-01 to 2024-12-31",
        'final_portfolio_value': {'nested': 'dict'},  # 問題データ
        'total_return': "0.15",  # 文字列
        'annualized_return': 0.12,  # 正常
        'max_drawdown': [0.05, 0.08],  # リスト
        'sharpe_ratio': None,  # None
        'switch_count': 150,
        'switch_success_rate': {'rate': 0.75},  # 問題データ
        'avg_holding_period_hours': 48.5,
        'total_switch_cost': "25000",  # 文字列
    }
    
    # 一時ファイル作成
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Excel出力実行
        success = exporter.export_dssms_results(problematic_result, temp_path)
        
        if success and os.path.exists(temp_path):
            print("[OK] 問題データでのExcel出力成功")
            print(f"   出力ファイル: {temp_path}")
            
            # ファイルサイズ確認
            file_size = os.path.getsize(temp_path)
            print(f"   ファイルサイズ: {file_size:,} bytes")
            
            if file_size > 1000:  # 最低限のサイズチェック
                print("[OK] Excel出力内容確認OK")
            else:
                print("[WARNING] Excel出力ファイルが小さすぎる可能性")
        else:
            print("[ERROR] Excel出力失敗")
            return False
    
    except Exception as e:
        print(f"[ERROR] Excel出力でエラー: {e}")
        return False
    finally:
        # クリーンアップ
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"   テンポラリファイル削除: {temp_path}")
            except:
                pass
    
    # Phase 4.5.3: ヘルパー関数の直接テスト
    print("\n3. ヘルパー関数テスト")
    
    test_cases = [
        ("正常な数値", 123.45, 123.45),
        ("文字列数値", "456.78", 456.78),
        ("dict型", {'value': 100}, 0.0),
        ("None", None, 0.0),
        ("リスト", [1, 2, 3], 0.0),
    ]
    
    all_passed = True
    for desc, input_val, expected in test_cases:
        try:
            result = exporter._ensure_numeric(input_val)
            if abs(result - expected) < 0.001:
                print(f"[OK] {desc}: {input_val} → {result}")
            else:
                print(f"[ERROR] {desc}: {input_val} → {result} (期待値: {expected})")
                all_passed = False
        except Exception as e:
            print(f"[ERROR] {desc}でエラー: {e}")
            all_passed = False
    
    print("\n=== Phase 4.5.4 テスト結果 ===")
    if all_passed:
        print("[OK] 全テスト通過 - Phase 4.5 修正は成功です！")
        return True
    else:
        print("[ERROR] 一部テスト失敗 - 追加修正が必要です")
        return False

if __name__ == "__main__":
    success = test_phase45_type_safety()
    sys.exit(0 if success else 1)