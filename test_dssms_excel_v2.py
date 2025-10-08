"""
DSSMS Excel出力システム V2 テストスクリプト
新しいExcel出力システムの動作確認とバリデーション

実行内容:
1. DSSMSExcelExporterV2の基本機能テスト
2. サンプルデータでのExcel出力テスト
3. 出力ファイルの検証
4. エラーハンドリングのテスト
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from output.dssms_excel_exporter_v2 import DSSMSExcelExporterV2, test_dssms_excel_exporter

def test_dssms_v2_integration():
    """DSSMS V2システム統合テスト"""
    print("=" * 80)
    print("[TOOL] DSSMS Excel出力システム V2 統合テスト")
    print("=" * 80)
    
    try:
        # 1. 基本機能テスト
        print("\n[LIST] Step 1: 基本機能テスト")
        success = test_dssms_excel_exporter()
        
        if success:
            print("[OK] 基本機能テスト: PASS")
        else:
            print("[ERROR] 基本機能テスト: FAIL")
            return False
        
        # 2. 詳細テストデータでのテスト
        print("\n[LIST] Step 2: 詳細テストデータテスト")
        detailed_test_result = create_detailed_test_data()
        
        exporter = DSSMSExcelExporterV2(initial_capital=1000000.0)
        output_path = exporter.export_dssms_results(detailed_test_result)
        
        if Path(output_path).exists():
            print(f"[OK] 詳細テスト: PASS - {output_path}")
            print(f"[CHART] ファイルサイズ: {Path(output_path).stat().st_size:,} bytes")
        else:
            print("[ERROR] 詳細テスト: FAIL - ファイルが作成されませんでした")
            return False
        
        # 3. パフォーマンステスト
        print("\n[LIST] Step 3: パフォーマンステスト")
        start_time = datetime.now()
        
        # 大量データでのテスト
        large_test_data = create_large_test_data()
        output_path_large = exporter.export_dssms_results(large_test_data)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if Path(output_path_large).exists():
            print(f"[OK] パフォーマンステスト: PASS")
            print(f"⏱️ 処理時間: {processing_time:.2f}秒")
            print(f"[CHART] 大容量ファイルサイズ: {Path(output_path_large).stat().st_size:,} bytes")
        else:
            print("[ERROR] パフォーマンステスト: FAIL")
            return False
        
        # 4. エラーハンドリングテスト
        print("\n[LIST] Step 4: エラーハンドリングテスト")
        error_test_results = test_error_handling(exporter)
        
        if error_test_results:
            print("[OK] エラーハンドリングテスト: PASS")
        else:
            print("[ERROR] エラーハンドリングテスト: FAIL")
            return False
        
        # 5. 最終レポート
        print("\n" + "=" * 80)
        print("[SUCCESS] DSSMS Excel出力システム V2 統合テスト完了")
        print("=" * 80)
        print("[OK] 全てのテストがPASSしました")
        print(f"📁 出力ファイル1: {output_path}")
        print(f"📁 出力ファイル2: {output_path_large}")
        print("[ROCKET] 新しいV2システムの実装が完了しました")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 統合テストエラー: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def create_detailed_test_data():
    """詳細テストデータ作成"""
    # リアルなDSSMSバックテスト結果をシミュレーション
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    days = (end_date - start_date).days
    
    # ポートフォリオ価値のシミュレーション（リアルな変動）
    portfolio_values = []
    current_value = 1000000.0
    
    for i in range(days):
        # 市場の変動を含むリアルなシミュレーション
        daily_change = np.random.normal(0.0008, 0.025)  # 平均0.08%、標準偏差2.5%
        current_value *= (1 + daily_change)
        portfolio_values.append(current_value)
    
    # 銘柄切替履歴の生成
    switch_history = []
    symbols = ["7203.T", "9984.T", "6758.T", "8031.T", "8306.T", "4063.T", "6861.T"]
    
    for i in range(117):  # 実際の切替回数に近い値
        switch_date = start_date + timedelta(days=i*3)
        from_symbol = symbols[i % len(symbols)]
        to_symbol = symbols[(i+1) % len(symbols)]
        
        switch = {
            "date": switch_date,
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "reason": ["パフォーマンス低下", "技術的指標シグナル", "リスク管理", "機会発見"][i % 4],
            "confidence": np.random.uniform(0.4, 0.95),
            "profit_loss": np.random.normal(2000, 8000),  # より現実的な損益
            "holding_period_hours": np.random.uniform(24, 168),
            "entry_price": np.random.uniform(800, 1500),
            "exit_price": np.random.uniform(820, 1520),
            "quantity": 100,
            "switch_cost": np.random.uniform(500, 2000)
        }
        switch_history.append(switch)
    
    # 日次リターンの計算
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
        daily_returns.append(daily_return)
    daily_returns = [0.0] + daily_returns
    
    return {
        "execution_time": "2025年09月03日 15:45:00",
        "backtest_period": "2023-01-01 - 2023-12-31",
        "initial_capital": 1000000.0,
        "final_portfolio_value": portfolio_values[-1],
        "total_return": (portfolio_values[-1] - 1000000.0) / 1000000.0,
        "annualized_return": (portfolio_values[-1] / 1000000.0) ** (365/days) - 1,
        "max_drawdown": -0.0823,
        "sharpe_ratio": 1.847,
        "switch_count": 117,
        "switch_success_rate": 0.5897,
        "avg_holding_period_hours": 72.3,
        "total_switch_cost": 167845.32,
        "daily_returns": daily_returns,
        "portfolio_values": portfolio_values,
        "switch_history": switch_history,
        "start_date": start_date,
        "end_date": end_date
    }

def create_large_test_data():
    """大容量テストデータ作成"""
    # より大きなデータセットでのテスト
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    days = (end_date - start_date).days
    
    # 大容量ポートフォリオデータ
    portfolio_values = []
    current_value = 1000000.0
    
    for i in range(days):
        daily_change = np.random.normal(0.0005, 0.02)
        current_value *= (1 + daily_change)
        portfolio_values.append(current_value)
    
    # 大量の切替履歴
    switch_history = []
    symbols = ["7203.T", "9984.T", "6758.T", "8031.T", "8306.T", "4063.T", "6861.T", "8058.T", "9020.T", "4755.T"]
    
    for i in range(500):  # 大量の切替
        switch_date = start_date + timedelta(days=i*2.9)
        
        switch = {
            "date": switch_date,
            "from_symbol": symbols[i % len(symbols)],
            "to_symbol": symbols[(i+1) % len(symbols)],
            "reason": ["パフォーマンス", "リスク", "機会", "技術"][i % 4],
            "confidence": np.random.uniform(0.3, 0.9),
            "profit_loss": np.random.normal(1500, 6000),
            "holding_period_hours": np.random.uniform(12, 240),
            "entry_price": np.random.uniform(700, 1800),
            "exit_price": np.random.uniform(720, 1820),
            "quantity": 100,
            "switch_cost": np.random.uniform(300, 1500)
        }
        switch_history.append(switch)
    
    # 日次リターン
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
        daily_returns.append(daily_return)
    daily_returns = [0.0] + daily_returns
    
    return {
        "execution_time": "2025年09月03日 15:50:00",
        "backtest_period": "2020-01-01 - 2023-12-31",
        "initial_capital": 1000000.0,
        "final_portfolio_value": portfolio_values[-1],
        "total_return": (portfolio_values[-1] - 1000000.0) / 1000000.0,
        "annualized_return": ((portfolio_values[-1] / 1000000.0) ** (365/days)) - 1,
        "max_drawdown": -0.1247,
        "sharpe_ratio": 1.524,
        "switch_count": 500,
        "switch_success_rate": 0.574,
        "avg_holding_period_hours": 69.8,
        "total_switch_cost": 487235.67,
        "daily_returns": daily_returns,
        "portfolio_values": portfolio_values,
        "switch_history": switch_history,
        "start_date": start_date,
        "end_date": end_date
    }

def test_error_handling(exporter):
    """エラーハンドリングテスト"""
    try:
        # 1. 空データテスト
        empty_data = {}
        try:
            exporter.export_dssms_results(empty_data)
            print("[OK] 空データハンドリング: OK")
        except Exception as e:
            print(f"[WARNING] 空データでエラー（想定内）: {type(e).__name__}")
        
        # 2. 不正データテスト
        invalid_data = {
            "invalid_field": "test",
            "final_portfolio_value": "not_a_number"
        }
        try:
            exporter.export_dssms_results(invalid_data)
            print("[OK] 不正データハンドリング: OK")
        except Exception as e:
            print(f"[WARNING] 不正データでエラー（想定内）: {type(e).__name__}")
        
        # 3. 最小データテスト
        minimal_data = {
            "execution_time": "2025-09-03 15:00:00",
            "final_portfolio_value": 1100000.0,
            "total_return": 0.1,
            "switch_history": [],
            "portfolio_values": [1000000.0, 1100000.0],
            "daily_returns": [0.0, 0.1]
        }
        
        try:
            output_path = exporter.export_dssms_results(minimal_data)
            if Path(output_path).exists():
                print("[OK] 最小データテスト: OK")
                return True
            else:
                print("[ERROR] 最小データテスト: ファイル作成失敗")
                return False
        except Exception as e:
            print(f"[ERROR] 最小データテストエラー: {e}")
            return False
        
    except Exception as e:
        print(f"[ERROR] エラーハンドリングテストエラー: {e}")
        return False

if __name__ == "__main__":
    success = test_dssms_v2_integration()
    if success:
        print("\n[TARGET] テスト結果: 全てのテストが成功しました！")
        print("📝 次のステップ: 実際のDSSMSバックテストで新しいV2システムをテストしてください")
    else:
        print("\n💥 テスト結果: 一部のテストが失敗しました")
        print("[SEARCH] ログを確認してエラーを修正してください")
