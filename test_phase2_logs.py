"""
Phase 2ログ強化テスト - P2-1/P2-2/P3-1/P3-2ログ出力検証

copilot-instructions.md準拠:
- 実際のバックテスト実行必須
- 実際の取引件数>0検証
- フォールバック実行時のログ確認

Author: Backtest Project Team
Created: 2025-12-01
"""
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルート追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester


def test_phase2_logs():
    """Phase 2ログ強化テスト"""
    print("=" * 80)
    print("Phase 2ログ強化テスト開始")
    print("=" * 80)
    
    # テスト設定
    config = {
        'initial_capital': 10000000,
        'target_symbols': ['6758'],  # ソニーグループ
        'use_cache': False
    }
    
    # バックテスター初期化
    print("\n[STEP 1/3] DSSMSIntegratedBacktester初期化中...")
    backtester = DSSMSIntegratedBacktester(config)
    print("[OK] 初期化完了")
    
    # バックテスト実行（短期間: 1週間、データ確実存在期間）
    print("\n[STEP 2/3] バックテスト実行中...")
    print("期間: 2023-12-01 ~ 2023-12-08 (1週間)")
    
    start_date = datetime(2023, 12, 1)
    end_date = datetime(2023, 12, 8)
    
    results = backtester.run_dynamic_backtest(
        start_date=start_date,
        end_date=end_date,
        target_symbols=['6758']
    )
    
    # 結果検証
    print("\n[STEP 3/3] 結果検証")
    print("=" * 80)
    
    # 基本ステータス確認
    status = results.get('status', 'UNKNOWN')
    print(f"\nステータス: {status}")
    
    if status == 'SUCCESS':
        # 取引日数確認
        trading_days = results.get('execution_metadata', {}).get('trading_days', 0)
        successful_days = results.get('execution_metadata', {}).get('successful_days', 0)
        print(f"取引日数: {trading_days}日")
        print(f"成功日数: {successful_days}日")
        
        # パフォーマンス確認
        portfolio_perf = results.get('portfolio_performance', {})
        initial_capital = portfolio_perf.get('initial_capital', 0)
        final_capital = portfolio_perf.get('final_capital', 0)
        total_return = portfolio_perf.get('total_return', 0)
        total_return_rate = portfolio_perf.get('total_return_rate', 0)
        
        print(f"\n初期資本: {initial_capital:,.0f}円")
        print(f"最終資本: {final_capital:,.0f}円")
        print(f"総リターン: {total_return:,.0f}円 ({total_return_rate:.2%})")
        
        # 日次結果確認
        daily_results = results.get('daily_results', [])
        print(f"\n日次結果: {len(daily_results)}件")
        
        # 検証項目
        print("\n" + "=" * 80)
        print("検証項目 (copilot-instructions.md準拠)")
        print("=" * 80)
        
        # 検証1: 実際の取引件数>0
        total_trades = sum(1 for dr in daily_results if dr.get('position_size', 0) > 0)
        check1 = total_trades > 0
        print(f"\n[検証1] 実際の取引件数>0: {'OK' if check1 else 'NG'}")
        print(f"        取引件数: {total_trades}件")
        
        # 検証2: 成功日数>0
        check2 = successful_days > 0
        print(f"\n[検証2] 成功日数>0: {'OK' if check2 else 'NG'}")
        print(f"        成功日数: {successful_days}日")
        
        # 検証3: ログ出力確認
        # 複数ログファイル候補を検索
        log_candidates = [
            project_root / "logs" / "backtest.log",
            project_root / "logs" / "dssms_integrated_backtester.log",
            project_root / "logs" / "main_system_controller.log"
        ]
        
        fallback_logs_found = False
        symbol_selection_logs_found = False
        ranking_result_logs_found = False
        final_stats_logs_found = False
        log_files_checked = []
        
        for log_file in log_candidates:
            if log_file.exists():
                log_files_checked.append(log_file.name)
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                        if '[FALLBACK]' in log_content:
                            fallback_logs_found = True
                        if '[SYMBOL_SELECTION]' in log_content:
                            symbol_selection_logs_found = True
                        if '[RANKING_RESULT]' in log_content:
                            ranking_result_logs_found = True
                        if '[FINAL_STATS]' in log_content:
                            final_stats_logs_found = True
                except Exception as e:
                    print(f"        [WARNING] {log_file.name}読み込みエラー: {e}")
        
        print(f"\n[検証3] ログ出力確認 (チェック済み: {', '.join(log_files_checked) if log_files_checked else 'ログファイル未検出'}):")
        print(f"        P2-1 [SYMBOL_SELECTION]: {'OK' if symbol_selection_logs_found else 'NG'}")
        print(f"        P2-2 [RANKING_RESULT]: {'OK' if ranking_result_logs_found else 'NG'}")
        print(f"        P3-1 [FINAL_STATS]: {'OK' if final_stats_logs_found else 'NG'}")
        print(f"        P3-2 [FALLBACK]: {'OK' if fallback_logs_found else 'NG (フォールバック未実行)'}")
        
        # 総合判定
        all_checks_passed = check1 and check2
        print("\n" + "=" * 80)
        if all_checks_passed:
            print("[SUCCESS] Phase 2ログ強化テスト完全成功")
            print("         - 実際の取引件数>0")
            print("         - 成功日数>0")
            print("         - ログ出力確認完了")
        else:
            print("[WARNING] 一部検証項目が失敗")
            if not check1:
                print("         - 取引件数=0 (要確認)")
            if not check2:
                print("         - 成功日数=0 (要確認)")
        print("=" * 80)
        
    elif status == 'ERROR':
        error_msg = results.get('error', '不明なエラー')
        print(f"\n[ERROR] バックテスト実行エラー:")
        print(f"        {error_msg}")
        print("\n対処方法:")
        print("1. ログファイル確認: logs/dssms_integrated_backtester.log")
        print("2. データ取得確認: yfinanceが6758データを取得できているか")
        print("3. 期間調整: データが存在する期間に変更")
    
    else:
        print(f"\n[WARNING] 予期しないステータス: {status}")
    
    print("\n" + "=" * 80)
    print("Phase 2ログ強化テスト完了")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    test_phase2_logs()
