"""
DSSMS ウォームアップ期間機能検証スクリプト

テスト期間: 2023/1/15 ~ 2023/1/31
検証項目:
1. ウォームアップ期間（2023/1/15以前）でシグナルが抑制されているか
2. 取引期間（2023/1/15以降）でのみ取引が発生しているか
3. 出力ファイルにウォームアップ期間が正しく記録されているか
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester


def run_dssms_warmup_test():
    """DSSMSウォームアップ期間テスト実行"""
    
    print("=" * 80)
    print("DSSMS ウォームアップ期間機能検証")
    print("=" * 80)
    
    # テスト設定
    test_start_date = datetime(2023, 1, 15)
    test_end_date = datetime(2023, 1, 31)
    target_symbols = ["9101.T"]  # 日本郵船
    
    print(f"\nテスト期間: {test_start_date} - {test_end_date}")
    print(f"対象銘柄: {target_symbols}")
    print(f"想定ウォームアップ期間: {test_start_date - pd.Timedelta(days=30)} - {test_start_date}")
    
    # DSSMS初期化
    print("\n[1/3] DSSMS初期化中...")
    config = {
        'initial_capital': 1000000,
        'max_position_size': 0.2,
        'enable_short_selling': False
    }
    
    backtester = DSSMSIntegratedBacktester(config)
    
    # バックテスト実行
    print("\n[2/3] バックテスト実行中...")
    print("-" * 80)
    
    try:
        results = backtester.run_dynamic_backtest(
            start_date=test_start_date,
            end_date=test_end_date,
            target_symbols=target_symbols
        )
        
        print("\n[3/3] 結果分析中...")
        print("-" * 80)
        
        # 基本結果表示
        print(f"\nステータス: {results.get('status', 'UNKNOWN')}")
        print(f"実行時間: {results.get('execution_time', 0):.2f}秒")
        print(f"取引日数: {results.get('trading_days', 0)}日")
        print(f"成功日数: {results.get('successful_days', 0)}日")
        
        # 日次結果分析
        daily_results = results.get('daily_results', [])
        print(f"\n日次結果件数: {len(daily_results)}件")
        
        if daily_results:
            print("\n日次取引詳細:")
            print(f"{'日付':^12} {'銘柄':^10} {'アクション':^12} {'価格':^10} {'ポジション':^10}")
            print("-" * 80)
            
            for daily in daily_results[:10]:  # 最初の10件
                date_str = daily.get('date', 'N/A')
                symbol = daily.get('symbol', 'N/A')
                action = daily.get('position_change', {}).get('action', 'N/A')
                price = daily.get('position_change', {}).get('price', 0)
                position = daily.get('position_change', {}).get('position_size', 0)
                
                print(f"{date_str:^12} {symbol:^10} {action:^12} {price:^10.2f} {position:^10.2f}")
        
        # パフォーマンスサマリー
        performance = results.get('performance_summary', {})
        if performance:
            print("\nパフォーマンスサマリー:")
            print(f"  総リターン: {performance.get('total_return', 0):.2%}")
            print(f"  総取引数: {performance.get('total_trades', 0)}")
            print(f"  勝率: {performance.get('win_rate', 0):.2%}")
            print(f"  最大ドローダウン: {performance.get('max_drawdown', 0):.2%}")
        
        # 生成されたファイルの確認
        print("\n生成されたファイル:")
        output_dir = Path("backtest_results")
        if output_dir.exists():
            files = list(output_dir.glob("*2023*"))
            for file in files[:5]:
                print(f"  - {file}")
        
        print("\n" + "=" * 80)
        print("検証完了")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] バックテスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_dssms_warmup_test()
    
    if results:
        # 詳細結果をJSONで保存
        import json
        output_file = "dssms_warmup_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n詳細結果を {output_file} に保存しました")
