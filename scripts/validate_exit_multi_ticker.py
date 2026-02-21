"""
エグジット戦略複数銘柄検証スクリプト（Phase 5実装）

日経225構成銘柄10銘柄でTrendFollowing(1/30/0.4)の汎化性能を検証。

主な機能:
- 複数銘柄一括バックテスト
- TrendFollowingExit(1/30/0.4)固定
- GCエントリー固定
- 業種分散10銘柄（製造業、金融、通信、小売等）
- 統計分析（平均PF、標準偏差、最小/最大PF）
- 汎化性能評価（平均PF > 30.0目標）

統合コンポーネント:
- GCStrategyWithExit
- TrendFollowingExit(1/30/0.4)
- data_fetcher経由でデータ取得
- CSV+JSON統一出力

セーフティ機能/注意事項:
- 期間固定（2023-01-01 ~ 2024-12-31）
- エントリー・エグジット固定（過学習回避）
- ルックアヘッドバイアス防止（copilot-instructions.md準拠）
- データ取得失敗時はスキップ（フォールバック禁止）
- 最低取引数10以上の銘柄のみ統計に含める

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

# プロジェクトルートをパス追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.gc_strategy_with_exit import GCStrategyWithExit
from strategies.exit_strategies.trend_following_exit import TrendFollowingExit
from data_fetcher import get_parameters_and_data


# PF上限制約（カーブフィッティング対策）
PF_CRITICAL_THRESHOLD = 100.0  # PF > 100は失格
PF_WARNING_THRESHOLD = 50.0    # PF > 50は警告


def validate_pf_threshold(pf_value: float, ticker: str) -> dict:
    """
    PF異常値チェック（カーブフィッティング対策）
    
    Args:
        pf_value: Profit Factor値
        ticker: 銘柄コード
    
    Returns:
        {'status': 'OK'|'WARNING'|'FAIL', 'message': str}
    """
    if pf_value > PF_CRITICAL_THRESHOLD:
        return {
            'status': 'FAIL',
            'message': f'{ticker}: PF={pf_value:.2f} exceeds critical threshold ({PF_CRITICAL_THRESHOLD}). Likely overfitting.'
        }
    elif pf_value > PF_WARNING_THRESHOLD:
        return {
            'status': 'WARNING',
            'message': f'{ticker}: PF={pf_value:.2f} exceeds warning threshold ({PF_WARNING_THRESHOLD}). Review required.'
        }
    else:
        return {
            'status': 'OK',
            'message': f'{ticker}: PF={pf_value:.2f} within acceptable range.'
        }


# 日経225構成銘柄10銘柄リスト（業種分散）
TICKERS = [
    # 製造業
    ("7203.T", "トヨタ自動車", "製造業（自動車）"),
    ("6758.T", "ソニーグループ", "製造業（電機）"),
    ("6501.T", "日立製作所", "製造業（電機）"),
    
    # 金融
    ("8306.T", "三菱UFJフィナンシャル・グループ", "金融（銀行）"),
    ("8316.T", "三井住友フィナンシャルグループ", "金融（銀行）"),
    
    # 通信
    ("9432.T", "日本電信電話", "通信（固定通信）"),
    ("9984.T", "ソフトバンクグループ", "通信（総合通信）"),
    
    # 小売・サービス
    ("9983.T", "ファーストリテイリング", "小売（衣料品）"),
    ("9433.T", "KDDI", "通信（移動体通信）"),
    
    # エネルギー
    ("5020.T", "ENEOSホールディングス", "エネルギー（石油）")
]


def calculate_performance_metrics(results_df: pd.DataFrame) -> dict:
    """
    パフォーマンス指標計算（validate_exit_strategy.pyから移植）
    
    Args:
        results_df: バックテスト結果DataFrame（Profit_Loss列必須）
    
    Returns:
        パフォーマンス指標辞書
    """
    if results_df is None or len(results_df) == 0:
        return {
            'total_trades': 0,
            'total_return': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'sharpe_ratio_annualized': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # Profit_Loss列確認
    if 'Profit_Loss' not in results_df.columns:
        print(f"[ERROR] Profit_Loss列が見つかりません")
        return {
            'total_trades': 0,
            'total_return': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'sharpe_ratio_annualized': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # 取引フィルタ（Entry/Exit両方ある行のみ）
    # BaseStrategy.backtest()の列構造: Date, Adj Close, Profit_Loss, Entry_Price, Exit_Price, Trade_ID
    # Signal列は存在しないため、Trade_ID列で取引を識別
    trades = results_df[results_df['Trade_ID'].notna()].copy()
    
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'total_return': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'sharpe_ratio_annualized': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # Profit_Loss取得（Trade_IDごとに1つのProfit_Lossがある）
    returns = trades['Profit_Loss'].dropna()
    
    if len(returns) == 0:
        return {
            'total_trades': 0,
            'total_return': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'sharpe_ratio_annualized': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # 基本統計
    total_trades = len(returns)
    total_return = returns.sum()
    
    # Profit Factor
    total_profit = returns[returns > 0].sum()
    total_loss = abs(returns[returns < 0].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    
    # Win Rate
    wins = len(returns[returns > 0])
    losses = len(returns[returns < 0])
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    
    # Sharpe Ratio（単純版）
    sharpe_ratio = 0.0
    if returns.std() > 0 and total_trades > 1:
        sharpe_ratio = returns.mean() / returns.std()
    
    # Sharpe Ratio年率換算（Phase 4実装）
    sharpe_ratio_annualized = 0.0
    if returns.std() > 0 and total_trades > 1:
        trading_days = len(results_df)
        trades_per_year = 250 / (trading_days / total_trades) if trading_days > 0 else 1
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(trades_per_year)
    
    # Max Drawdown（Phase 4実装）
    cumulative_returns = returns.cumsum()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()
    
    # Max Drawdown%（初期資金100万円基準）
    initial_capital = 1_000_000
    max_drawdown_pct = (max_drawdown / initial_capital) * 100
    
    return {
        'total_trades': total_trades,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'sharpe_ratio_annualized': sharpe_ratio_annualized,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct
    }


def validate_exit_on_ticker(
    ticker: str,
    ticker_name: str,
    industry: str,
    start_date: str,
    end_date: str
) -> dict:
    """
    単一銘柄でエグジット戦略検証
    
    Args:
        ticker: ティッカーシンボル（例: "7203.T"）
        ticker_name: 銘柄名（例: "トヨタ自動車"）
        industry: 業種（例: "製造業（自動車）"）
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
    
    Returns:
        検証結果辞書
    """
    print(f"\n{'='*80}")
    print(f"[TICKER] {ticker} - {ticker_name} ({industry})")
    print(f"{'='*80}")
    
    try:
        # データ取得
        print(f"[STEP 1/3] データ取得中...")
        _, _, _, stock_data, index_data = get_parameters_and_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            warmup_days=150  # Option A-2暦日拡大方式
        )
        
        if stock_data is None or len(stock_data) == 0:
            print(f"[ERROR] データ取得失敗: {ticker}")
            return None
        
        print(f"[OK] データ取得完了: {len(stock_data)} rows")
        
        # エグジット戦略作成
        print(f"[STEP 2/3] バックテスト実行中...")
        exit_strategy = TrendFollowingExit(
            min_hold_days=1,
            max_hold_days=30,
            confidence_threshold=0.4
        )
        
        # GC戦略作成
        strategy = GCStrategyWithExit(
            data=stock_data,
            exit_strategy=exit_strategy,
            params={
                'short_window': 5,
                'long_window': 25
            },
            ticker=ticker
        )
        
        # バックテスト実行
        results_df = strategy.backtest()
        
        if results_df is None or len(results_df) == 0:
            print(f"[ERROR] バックテスト失敗: {ticker}")
            return None
        
        print(f"[OK] バックテスト完了")
        
        # パフォーマンス指標計算
        print(f"[STEP 3/3] パフォーマンス指標計算中...")
        metrics = calculate_performance_metrics(results_df)
        
        # PF制約チェック
        pf_check = validate_pf_threshold(metrics['profit_factor'], ticker)
        pf_status = pf_check['status']
        
        print(f"[RESULTS]")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Sharpe Ratio (年率): {metrics['sharpe_ratio_annualized']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  PF Validation: [{pf_status}] {pf_check['message']}")
        
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'industry': industry,
            'metrics': metrics,
            'status': 'SUCCESS',
            'pf_status': pf_status  # 'OK', 'WARNING', 'FAIL'
        }
        
    except Exception as e:
        print(f"[ERROR] 検証失敗: {ticker} - {e}")
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'industry': industry,
            'metrics': None,
            'status': 'ERROR',
            'error': str(e)
        }


def run_multi_ticker_validation():
    """
    複数銘柄一括検証実行
    
    Returns:
        全銘柄検証結果リスト
    """
    print("\n" + "="*80)
    print("Phase 5: 他銘柄検証 - TrendFollowingExit(1/30/0.4)汎化性能確認")
    print("="*80)
    print(f"対象銘柄数: {len(TICKERS)}")
    print(f"検証期間: 2023-01-01 ~ 2024-12-31")
    print(f"エントリー戦略: GCStrategy（固定）")
    print(f"エグジット戦略: TrendFollowingExit(1/30/0.4)（固定）")
    print("="*80)
    
    # 期間固定
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    # 全銘柄検証
    all_results = []
    
    for ticker, ticker_name, industry in TICKERS:
        result = validate_exit_on_ticker(
            ticker=ticker,
            ticker_name=ticker_name,
            industry=industry,
            start_date=start_date,
            end_date=end_date
        )
        
        if result is not None:
            all_results.append(result)
    
    return all_results


def analyze_and_output_results(all_results: list):
    """
    結果分析・出力
    
    Args:
        all_results: 全銘柄検証結果リスト
    """
    print("\n" + "="*80)
    print("統計分析")
    print("="*80)
    
    # 成功銘柄フィルタ
    success_results = [
        r for r in all_results 
        if r['status'] == 'SUCCESS' and r['metrics'] is not None
    ]
    
    print(f"成功銘柄数: {len(success_results)} / {len(all_results)}")
    
    if len(success_results) == 0:
        print("[ERROR] 有効な結果なし")
        return
    
    # 統計的有意性フィルタ（取引数 >= 10）
    valid_results = [
        r for r in success_results
        if r['metrics']['total_trades'] >= 10
    ]
    
    print(f"統計的有意性あり銘柄数: {len(valid_results)} / {len(success_results)}")
    print(f"（取引数 >= 10の銘柄）")
    
    if len(valid_results) == 0:
        print("[WARNING] 統計的有意性のある銘柄なし")
        # 全成功銘柄で統計計算（警告付き）
        valid_results = success_results
        print("[INFO] 全成功銘柄で統計計算（取引数不足の銘柄含む）")
    
    # PF統計
    pf_values = [r['metrics']['profit_factor'] for r in valid_results]
    avg_pf = np.mean(pf_values)
    std_pf = np.std(pf_values)
    min_pf = np.min(pf_values)
    max_pf = np.max(pf_values)
    
    print(f"\n[Profit Factor統計]")
    print(f"  平均PF: {avg_pf:.2f}")
    print(f"  標準偏差: {std_pf:.2f}")
    print(f"  最小PF: {min_pf:.2f}")
    print(f"  最大PF: {max_pf:.2f}")
    print(f"  目標（平均PF > 30.0）: {'OK 達成' if avg_pf > 30.0 else 'NG 未達成'}")
    
    # Win Rate統計
    wr_values = [r['metrics']['win_rate'] for r in valid_results]
    avg_wr = np.mean(wr_values)
    
    print(f"\n[Win Rate統計]")
    print(f"  平均Win Rate: {avg_wr:.1%}")
    
    # Sharpe Ratio年率統計
    sharpe_values = [r['metrics']['sharpe_ratio_annualized'] for r in valid_results]
    avg_sharpe = np.mean(sharpe_values)
    
    print(f"\n[Sharpe Ratio（年率）統計]")
    print(f"  平均Sharpe Ratio: {avg_sharpe:.2f}")
    
    # 取引数統計
    trades_values = [r['metrics']['total_trades'] for r in valid_results]
    avg_trades = np.mean(trades_values)
    
    print(f"\n[取引数統計]")
    print(f"  平均取引数: {avg_trades:.1f}")
    
    # PF制約統計（Phase 6対策）
    pf_ok_count = sum(1 for r in success_results if r.get('pf_status') == 'OK')
    pf_warning_count = sum(1 for r in success_results if r.get('pf_status') == 'WARNING')
    pf_fail_count = sum(1 for r in success_results if r.get('pf_status') == 'FAIL')
    
    print(f"\n[PF制約統計（カーブフィッティング対策）]")
    print(f"  OK (PF <= 50): {pf_ok_count} 銘柄")
    print(f"  WARNING (50 < PF <= 100): {pf_warning_count} 銘柄")
    print(f"  FAIL (PF > 100): {pf_fail_count} 銘柄")
    print(f"  Phase 6推奨銘柄: OK判定の{min(3, pf_ok_count)}銘柄")
    
    # CSV出力
    output_dir = Path("output/exit_strategy_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"multi_ticker_validation_{timestamp}.csv"
    
    # DataFrame作成
    df_rows = []
    for r in success_results:
        m = r['metrics']
        df_rows.append({
            'ticker': r['ticker'],
            'ticker_name': r['ticker_name'],
            'industry': r['industry'],
            'total_trades': int(m['total_trades']),
            'total_return': float(m['total_return']),
            'profit_factor': float(m['profit_factor']),
            'win_rate': float(m['win_rate']),
            'sharpe_ratio_annualized': float(m['sharpe_ratio_annualized']),
            'max_drawdown_pct': float(m['max_drawdown_pct']),
            'statistical_significance': 'OK' if m['total_trades'] >= 10 else 'NG',
            'pf_status': r.get('pf_status', 'UNKNOWN')  # Phase 6対策追加
        })
    
    df_output = pd.DataFrame(df_rows)
    df_output.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[OUTPUT] CSV出力完了: {csv_path}")
    
    # JSON出力（統計サマリー）
    json_path = output_dir / f"multi_ticker_summary_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'total_tickers': len(all_results),
        'success_tickers': len(success_results),
        'valid_tickers': len(valid_results),
        'statistics': {
            'avg_pf': float(avg_pf),
            'std_pf': float(std_pf),
            'min_pf': float(min_pf),
            'max_pf': float(max_pf),
            'avg_win_rate': float(avg_wr),
            'avg_sharpe_ratio': float(avg_sharpe),
            'avg_trades': float(avg_trades)
        },
        'target_achievement': {
            'avg_pf_target': 30.0,
            'achieved': bool(avg_pf > 30.0)
        },
        'pf_constraints': {  # Phase 6対策追加
            'ok_count': pf_ok_count,
            'warning_count': pf_warning_count,
            'fail_count': pf_fail_count,
            'critical_threshold': PF_CRITICAL_THRESHOLD,
            'warning_threshold': PF_WARNING_THRESHOLD
        },
        'tickers': df_rows
    }
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OUTPUT] JSON出力完了: {json_path}")
    except Exception as e:
        print(f"[WARNING] JSON出力スキップ: {e}")
    
    # 詳細結果表示
    print(f"\n{'='*80}")
    print("銘柄別詳細結果")
    print(f"{'='*80}")
    
    # PFでソート
    sorted_results = sorted(
        success_results,
        key=lambda x: x['metrics']['profit_factor'],
        reverse=True
    )
    
    for i, r in enumerate(sorted_results, 1):
        m = r['metrics']
        sig = 'OK' if m['total_trades'] >= 10 else 'NG'
        pf_status = r.get('pf_status', 'UNKNOWN')
        print(f"{i}. {r['ticker']} - {r['ticker_name']} ({r['industry']})")
        print(f"   PF={m['profit_factor']:.2f}, WR={m['win_rate']:.1%}, "
              f"Trades={m['total_trades']}, Sharpe(annualized)={m['sharpe_ratio_annualized']:.2f}, "
              f"Significance={sig}, PF_Status={pf_status}")
    
    # Phase 6推奨銘柄（PF <= 50、Win Rate >= 60%）
    print(f"\n{'='*80}")
    print("Phase 6推奨銘柄（PF制約適用）")
    print(f"{'='*80}")
    
    phase6_candidates = [
        r for r in success_results
        if r.get('pf_status') == 'OK' and r['metrics']['win_rate'] >= 0.60
    ]
    
    # PFでソート
    phase6_candidates = sorted(
        phase6_candidates,
        key=lambda x: x['metrics']['profit_factor'],
        reverse=True
    )
    
    if len(phase6_candidates) >= 3:
        print(f"推奨3銘柄（PF <= 50、Win Rate >= 60%）:")
        for i, r in enumerate(phase6_candidates[:3], 1):
            m = r['metrics']
            print(f"{i}. {r['ticker']} - {r['ticker_name']} ({r['industry']})")
            print(f"   PF={m['profit_factor']:.2f}, WR={m['win_rate']:.1%}, "
                  f"Trades={m['total_trades']}, Sharpe(annualized)={m['sharpe_ratio_annualized']:.2f}")
    else:
        print(f"[WARNING] PF <= 50かつWin Rate >= 60%の銘柄が{len(phase6_candidates)}銘柄のみ")
        print(f"代替案: 全OK判定銘柄から選定")
        ok_results = [r for r in success_results if r.get('pf_status') == 'OK']
        ok_results = sorted(ok_results, key=lambda x: x['metrics']['profit_factor'], reverse=True)
        for i, r in enumerate(ok_results[:3], 1):
            m = r['metrics']
            print(f"{i}. {r['ticker']} - {r['ticker_name']} ({r['industry']})")
            print(f"   PF={m['profit_factor']:.2f}, WR={m['win_rate']:.1%}, "
                  f"Trades={m['total_trades']}, Sharpe(annualized)={m['sharpe_ratio_annualized']:.2f}")


def main():
    """メインエントリーポイント"""
    print("\n" + "="*80)
    print("Phase 5: 他銘柄検証スクリプト")
    print("="*80 + "\n")
    
    # 検証実行
    all_results = run_multi_ticker_validation()
    
    # 結果分析・出力
    analyze_and_output_results(all_results)
    
    print("\n" + "="*80)
    print("Phase 5検証完了")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
