"""
Task 8修正案2の検証スクリプト

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
from main_new import MainSystemController

def main():
    """Task 8修正案2の検証バックテスト実行"""
    
    print("\n" + "=" * 80)
    print("Task 8修正案2 検証バックテスト")
    print("銘柄: 8306.T")
    print("期間: 2023-01-01 ~ 2023-01-31")
    print("=" * 80 + "\n")
    
    # システム設定
    config = {
        'execution': {
            'execution_mode': 'simple',
            'broker': {
                'initial_cash': 1000000,
                'commission_per_trade': 1.0
            }
        },
        'risk_management': {
            'use_enhanced_risk': False,
            'max_drawdown_threshold': 0.15
        },
        'performance': {
            'use_aggregator': False
        }
    }
    
    # システム初期化
    print("[INFO] MainSystemController初期化中...")
    system = MainSystemController(config)
    
    # バックテスト実行
    ticker = "8306.T"
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    # データ取得期間を計算（ウォームアップ期間90日 + バックテスト期間31日 + マージン30日）
    # 2023-01-01から90日前 = 2022-10-03
    # 安全のため、2022-01-01から取得（約760日分）
    days_back = 760
    
    print(f"[INFO] バックテスト実行: {ticker}")
    print(f"       開始日: {start_date.strftime('%Y-%m-%d')}")
    print(f"       終了日: {end_date.strftime('%Y-%m-%d')}")
    print(f"       ウォームアップ期間: 90日")
    print(f"       データ取得期間: {days_back}日")
    print("-" * 80 + "\n")
    
    results = system.execute_comprehensive_backtest(
        ticker=ticker,
        stock_data=None,  # yfinanceから取得
        index_data=None,
        days_back=days_back,  # データ取得期間を指定
        backtest_start_date=start_date,
        backtest_end_date=end_date,
        warmup_days=90
    )
    
    # 結果出力
    print("\n" + "=" * 80)
    print("バックテスト完了")
    print("=" * 80)
    
    if results['status'] == 'SUCCESS':
        print(f"\n[SUCCESS] ステータス: 成功")
        print(f"銘柄: {results['ticker']}")
        print(f"実行時間: {results['execution_timestamp']}")
        
        # パフォーマンス結果
        performance = results.get('performance_results', {})
        summary = performance.get('summary_statistics', {})
        
        if summary:
            print(f"\n【パフォーマンスサマリー】")
            print(f"  総リターン: {summary.get('total_return', 0):.2%}")
            print(f"  総取引数: {summary.get('total_trades', 0)}")
            print(f"  勝率: {summary.get('win_rate', 0):.2%}")
        
        # レポート出力先
        report = results.get('report_results', {})
        if report.get('output_directory'):
            print(f"\nレポート出力: {report['output_directory']}")
            
    elif results['status'] == 'EXECUTION_DENIED':
        print(f"\n[WARNING] ステータス: 実行拒否")
        print(f"理由: {results.get('message', 'リスク評価により実行拒否')}")
        
    else:
        print(f"\n[ERROR] ステータス: エラー")
        print(f"エラー: {results.get('error', '不明なエラー')}")
    
    print("\n" + "=" * 80)
    print("検証バックテスト完了")
    print("次のステップ:")
    print("1. 出力ファイルを確認（execution_results, trades.csv）")
    print("2. ログファイルを確認（logs/main_system_controller.log）")
    print("3. 成功基準を検証")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    main()
