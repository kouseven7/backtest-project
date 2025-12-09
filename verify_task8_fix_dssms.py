"""
Task 8修正案2の検証スクリプト（DSSMS統合システム経由）

DSSMS統合システムを経由してmain_new.pyを呼び出し、
修正案2（ForceCloseフラグ導入）が正しく機能することを検証します。

検証項目:
1. 2023-01-13のSELL件数: 2件 → 1件（ForceCloseのみ）
2. 8306銘柄のBUY/SELL一致: BUY=3, SELL=3
3. holding_period_days: 全て正の値
4. ログ出力: [FORCE_CLOSE_START], [FORCE_CLOSE_SUPPRESS], [FORCE_CLOSE_END]

Author: Backtest Project Team
Created: 2025-12-08
"""

from datetime import datetime
import sys
sys.path.insert(0, 'src')

from dssms.dssms_integrated_main import DSSMSIntegratedBacktester

def main():
    """Task 8修正案2の検証バックテスト実行"""
    
    print("\n" + "=" * 80)
    print("Task 8修正案2 検証バックテスト（DSSMS統合システム経由）")
    print("銘柄: 8306")
    print("期間: 2023-01-01 ~ 2023-01-31")
    print("=" * 80 + "\n")
    
    # DSSMS統合システム設定
    config = {
        'initial_capital': 1000000,
        'switch_cost_rate': 0.001,
        'min_holding_days': 1,
        'data_cache_days': 365
    }
    
    # システム初期化
    print("[INFO] DSSMS統合バックテスター初期化中...")
    backtester = DSSMSIntegratedBacktester(config)
    
    # バックテスト実行
    print(f"[INFO] バックテスト実行: 8306")
    print(f"       開始日: 2023-01-01")
    print(f"       終了日: 2023-01-31")
    print("-" * 80 + "\n")
    
    results = backtester.run_dynamic_backtest(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        target_symbols=['8306']
    )
    
    # 結果出力
    print("\n" + "=" * 80)
    print("バックテスト完了")
    print("=" * 80)
    
    if results:
        print(f"\n[SUCCESS] バックテスト成功")
        print(f"出力フォルダ: {backtester.output_dir}")
        
        # 簡易サマリー
        print(f"\n【バックテスト結果】")
        print(f"  最終資本: {results.get('final_capital', 0):,.0f}円")
        print(f"  総収益率: {results.get('total_return_pct', 0):.2f}%")
        print(f"  取引件数: {results.get('total_trades', 0)}")
        
    else:
        print(f"\n[ERROR] バックテスト失敗")
    
    print("\n" + "=" * 80)
    print("検証バックテスト完了")
    print("次のステップ:")
    print(f"1. 出力フォルダ確認: {backtester.output_dir}")
    print("2. execution_results確認（2023-01-13のSELL件数）")
    print("3. ログ確認: logs/dssms_integrated_backtest.log")
    print("4. main_new.pyログ確認: logs/main_system_controller.log")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    main()
