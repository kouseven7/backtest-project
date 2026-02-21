"""
max_positions制約検証スクリプト（デバッグ用）

2026-02-15作成
- 5日間の短期バックテスト実行
- 標準出力に全ログを出力してmax_positions動作確認
"""
import logging
import sys
from datetime import datetime
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

# ログ設定: 標準出力にDEBUGレベルで出力
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    print("=" * 80)
    print("max_positions制約検証バックテスト開始")
    print("=" * 80)
    print(f"期間: 2024-01-17 ~ 2024-01-26 (10日間)")
    print(f"設定: max_positions=2")
    print("=" * 80)
    print()
    
    # 設定辞書作成
    config = {
        'initial_capital': 1000000,
        'target_symbols': ['9101.T', '9104.T', '9107.T', '5802.T', '8802.T', '6301.T', '6703.T'],  # 7銘柄
        'dssms_backtest_start_date': '2024-01-17',
        'dssms_backtest_end_date': '2024-01-26'  # 10日間
    }
    
    # DSSMSインスタンス生成
    backtester = DSSMSIntegratedBacktester(config=config)
    
    # max_positions設定確認
    print(f"\n初期化完了:")
    print(f"  max_positions: {backtester.max_positions}")
    print(f"  positions: {backtester.positions}")
    print(f"  initial_capital: {backtester.initial_capital:,.0f}円")
    print()
    
    # バックテスト実行
    print("\nバックテスト実行中...")
    print("-" * 80)
    
    from datetime import datetime
    start_date = datetime.strptime('2024-01-17', '%Y-%m-%d')
    end_date = datetime.strptime('2024-01-26', '%Y-%m-%d')
    target_symbols = config['target_symbols']
    
    results = backtester.run_dynamic_backtest(start_date, end_date, target_symbols)
    
    print("-" * 80)
    print("\nバックテスト完了")
    print("=" * 80)
    print(f"総取引数: {results.get('total_trades', 0)}")
    print(f"総損益: {results.get('total_pnl', 0):,.0f}円")
    print(f"最終保有: {len(backtester.positions)}")
    print(f"保有銘柄: {list(backtester.positions.keys())}")
    print("=" * 80)
    
    # all_transactions.csv確認
    import pandas as pd
    output_dir = results.get('output_dir', '')
    if output_dir:
        csv_path = f"{output_dir}/all_transactions.csv"
        try:
            df = pd.read_csv(csv_path)
            print(f"\nall_transactions.csv読み取り成功: {len(df)}行")
            print(df[['symbol', 'entry_date', 'exit_date', 'strategy', 'pnl']].to_string())
        except Exception as e:
            print(f"CSV読み取りエラー: {e}")

if __name__ == "__main__":
    main()
