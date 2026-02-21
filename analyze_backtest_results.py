"""
バックテスト結果包括分析スクリプト

6ヶ月・1年バックテストディレクトリを分析し、以下を検証:
1. max_positions制約遵守
2. ウォームアップ期間エントリー
3. 取引の完全性
4. FIFO決済の動作
5. パフォーマンスサマリー
6. 銘柄切替パターン

Author: Backtest Project Team
Created: 2026-02-15
"""
import pandas as pd
from collections import defaultdict
from pathlib import Path

def analyze_max_positions(output_dir):
    """max_positions制約遵守を検証"""
    
    # all_transactions.csv読み込み
    csv_path = Path(output_dir) / "all_transactions.csv"
    df = pd.read_csv(csv_path)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    
    # 各日付の保有銘柄を追跡（修正版: エグジット日を含まないように変更）
    holdings_by_date = defaultdict(set)
    
    for _, row in df.iterrows():
        current = row['entry_date']
        # エグジット日の前日まで保有とする（エグジット日当日は保有していない）
        end_date = row['exit_date'] - pd.Timedelta(days=1)
        while current <= end_date:
            holdings_by_date[current].add(row['symbol'])
            current += pd.Timedelta(days=1)
    
    # 保有数の分布を集計
    position_counts = defaultdict(int)
    for date, symbols in holdings_by_date.items():
        count = len(symbols)
        position_counts[count] += 1
    
    # 最大保有数
    max_holdings = max(len(symbols) for symbols in holdings_by_date.values()) if holdings_by_date else 0
    
    # 結果出力
    print("=" * 80)
    print("max_positions制約遵守検証")
    print("=" * 80)
    print(f"バックテスト期間: {df['entry_date'].min().date()} ~ {df['exit_date'].max().date()}")
    print(f"総取引数: {len(df)}件")
    print(f"\n保有銘柄数の分布:")
    for count in sorted(position_counts.keys()):
        print(f"  {count}銘柄: {position_counts[count]}日")
    print(f"\n最大同時保有数: {max_holdings}銘柄")
    
    # 制約違反チェック
    violations = {date: list(symbols) for date, symbols in holdings_by_date.items() 
                  if len(symbols) > 2}
    
    if violations:
        print(f"\n[NG] 制約違反検出: {len(violations)}日")
        print("違反詳細（最初の10日）:")
        for i, (date, symbols) in enumerate(sorted(violations.items())[:10]):
            print(f"  {date.date()}: {len(symbols)}銘柄 {symbols}")
        return False
    else:
        print("\n[OK] max_positions=2 完全遵守")
        return True

def check_warmup_entries(output_dir, trading_start_date="2024-01-01"):
    """ウォームアップ期間のエントリーチェック"""
    
    csv_path = Path(output_dir) / "all_transactions.csv"
    df = pd.read_csv(csv_path)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    
    trading_start = pd.Timestamp(trading_start_date)
    warmup_entries = df[df['entry_date'] < trading_start]
    
    print("=" * 80)
    print("ウォームアップ期間エントリー検証")
    print("=" * 80)
    print(f"trading_start_date: {trading_start_date}")
    print(f"ウォームアップ期間エントリー数: {len(warmup_entries)}件")
    
    if len(warmup_entries) > 0:
        print("\n[NG] ウォームアップ期間エントリー検出:")
        print(warmup_entries[['symbol', 'entry_date', 'strategy_name']])
        return False
    else:
        print("\n[OK] ウォームアップ期間エントリー: 0件")
        return True

def check_transactions_completeness(output_dir):
    """all_transactions.csvの完全性確認"""
    
    csv_path = Path(output_dir) / "all_transactions.csv"
    df = pd.read_csv(csv_path)
    
    # 空欄チェック
    empty_exit_date = df['exit_date'].isna().sum()
    empty_exit_price = df['exit_price'].isna().sum()
    zero_exit_price = (df['exit_price'] == 0.0).sum()
    empty_pnl = df['pnl'].isna().sum()
    
    print("=" * 80)
    print("all_transactions.csv完全性検証")
    print("=" * 80)
    print(f"総取引数: {len(df)}件")
    print(f"\n未決済チェック:")
    print(f"  exit_date空欄: {empty_exit_date}件")
    print(f"  exit_price空欄: {empty_exit_price}件")
    print(f"  exit_price=0.0: {zero_exit_price}件")
    print(f"  pnl空欄: {empty_pnl}件")
    
    if empty_exit_date > 0 or empty_exit_price > 0 or zero_exit_price > 0 or empty_pnl > 0:
        print("\n[NG] 未決済取引が存在します")
        # 未決済取引の詳細
        incomplete = df[(df['exit_date'].isna()) | 
                       (df['exit_price'].isna()) | 
                       (df['exit_price'] == 0.0) | 
                       (df['pnl'].isna())]
        print(incomplete[['symbol', 'entry_date', 'exit_date', 'exit_price', 'pnl']])
        return False
    else:
        print("\n[OK] 全取引が正常に決済されています")
        return True

def analyze_performance(output_dir):
    """パフォーマンス分析"""
    
    csv_path = Path(output_dir) / "all_transactions.csv"
    df = pd.read_csv(csv_path)
    
    # 基本統計
    total_trades = len(df)
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    max_profit = df['pnl'].max()
    max_loss = df['pnl'].min()
    
    # 保有期間
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df['holding_days'] = (df['exit_date'] - df['entry_date']).dt.days
    avg_holding_days = df['holding_days'].mean()
    
    print("=" * 80)
    print("パフォーマンスサマリー")
    print("=" * 80)
    print(f"バックテスト期間: {df['entry_date'].min().date()} ~ {df['exit_date'].max().date()}")
    print(f"\n取引統計:")
    print(f"  総取引数: {total_trades}件")
    print(f"  勝ちトレード: {winning_trades}件")
    print(f"  負けトレード: {losing_trades}件")
    print(f"  勝率: {win_rate:.1f}%")
    print(f"\n損益統計:")
    print(f"  総損益: {total_pnl:,.0f}円")
    print(f"  平均損益: {avg_pnl:,.0f}円")
    print(f"  最大利益: {max_profit:,.0f}円")
    print(f"  最大損失: {max_loss:,.0f}円")
    print(f"\n保有期間:")
    print(f"  平均保有日数: {avg_holding_days:.1f}日")
    
    # 戦略別集計
    if 'strategy_name' in df.columns:
        print(f"\n戦略別統計:")
        strategy_stats = df.groupby('strategy_name').agg({
            'pnl': ['count', 'sum', 'mean'],
        }).round(0)
        strategy_stats.columns = ['取引数', '総損益', '平均損益']
        print(strategy_stats)

def analyze_symbol_switching(output_dir):
    """銘柄切替パターン分析"""
    
    csv_path = Path(output_dir) / "all_transactions.csv"
    df = pd.read_csv(csv_path)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df = df.sort_values('entry_date')
    
    # ユニーク銘柄数
    unique_symbols = df['symbol'].nunique()
    
    # 銘柄別取引回数
    symbol_counts = df['symbol'].value_counts()
    
    # 銘柄別平均損益
    symbol_pnl = df.groupby('symbol')['pnl'].agg(['count', 'sum', 'mean']).round(0)
    symbol_pnl.columns = ['取引数', '総損益', '平均損益']
    symbol_pnl = symbol_pnl.sort_values('総損益', ascending=False)
    
    print("=" * 80)
    print("銘柄切替パターン分析")
    print("=" * 80)
    print(f"取引した銘柄数: {unique_symbols}銘柄")
    print(f"\n銘柄別取引回数（上位10銘柄）:")
    print(symbol_counts.head(10))
    print(f"\n銘柄別損益（上位10銘柄）:")
    print(symbol_pnl.head(10))
    
    # 切替頻度
    switches = 0
    prev_symbol = None
    for symbol in df['symbol']:
        if prev_symbol and symbol != prev_symbol:
            switches += 1
        prev_symbol = symbol
    
    print(f"\n銘柄切替回数: {switches}回")
    print(f"取引あたり切替頻度: {switches / len(df) * 100:.1f}%")

def analyze_fifo_operations(output_dir):
    """FIFO決済の動作確認"""
    
    log_path = Path(output_dir) / "dssms_execution_log.txt"
    
    if not log_path.exists():
        print("=" * 80)
        print("FIFO決済動作確認")
        print("=" * 80)
        print("[WARNING] ログファイルが見つかりません")
        return
    
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # FIFO関連ログをカウント
    case4_count = log_content.count('[SWITCH_DEBUG] ケース4実行: FIFO決済')
    force_close_count = log_content.count('[FORCE_CLOSE]')
    fifo_success = log_content.count('FIFO決済成功')
    fifo_fallback = log_content.count('[FORCE_CLOSE_FALLBACK]')
    
    print("=" * 80)
    print("FIFO決済動作確認")
    print("=" * 80)
    print(f"ケース4発動回数: {case4_count}回")
    print(f"[FORCE_CLOSE]ログ: {force_close_count}件")
    print(f"FIFO決済成功: {fifo_success}回")
    print(f"フォールバック作動: {fifo_fallback}回")
    
    if case4_count > 0:
        success_rate = (fifo_success / case4_count * 100) if case4_count > 0 else 0
        print(f"成功率: {success_rate:.1f}%")

def comprehensive_analysis(output_dir, period_name):
    """包括的バックテスト分析"""
    
    print("\n" + "=" * 100)
    print(f"{period_name} バックテスト包括分析")
    print("=" * 100 + "\n")
    
    # 必須分析1-4を実行
    results = {
        'max_positions': analyze_max_positions(output_dir),
        'warmup_entries': check_warmup_entries(output_dir),
        'completeness': check_transactions_completeness(output_dir),
    }
    
    print("\n")
    
    # FIFO決済分析
    analyze_fifo_operations(output_dir)
    
    print("\n")
    
    # 推奨分析5-6を実行
    analyze_performance(output_dir)
    
    print("\n")
    
    analyze_symbol_switching(output_dir)
    
    # 総合判定
    print("\n" + "=" * 80)
    print("総合判定")
    print("=" * 80)
    all_passed = all(results.values())
    if all_passed:
        print("[OK] 全ての必須検証に合格")
    else:
        print("[NG] 一部の検証で問題が検出されました")
        for check, passed in results.items():
            status = "[OK]" if passed else "[NG]"
            print(f"  {status} {check}")
    
    return all_passed

# メイン実行
if __name__ == "__main__":
    # 6ヶ月バックテスト分析
    print("\n" + "#" * 100)
    print("# 6ヶ月バックテスト（2024年上半期）分析開始")
    print("#" * 100)
    output_dir_6m = "output/dssms_integration/dssms_20260215_103605"
    result_6m = comprehensive_analysis(output_dir_6m, "6ヶ月（2024年上半期）")

    # 1年バックテスト分析
    print("\n\n" + "#" * 100)
    print("# 1年バックテスト（2024年通年）分析開始")
    print("#" * 100)
    output_dir_1y = "output/dssms_integration/dssms_20260215_105052"
    result_1y = comprehensive_analysis(output_dir_1y, "1年（2024年通年）")

    # 最終サマリー
    print("\n\n" + "=" * 100)
    print("最終サマリー")
    print("=" * 100)
    print(f"6ヶ月バックテスト: {'[OK] 合格' if result_6m else '[NG] 問題あり'}")
    print(f"1年バックテスト: {'[OK] 合格' if result_1y else '[NG] 問題あり'}")

    if result_6m and result_1y:
        print("\n[SUCCESS] 両方のバックテストが全ての検証に合格しました")
        print("\n次のステップ:")
        print("- Sprint 2完了報告の作成")
        print("- 複数銘柄保有機能の正式リリース")
    else:
        print("\n[WARNING] 問題が検出されました。修正が必要です。")
