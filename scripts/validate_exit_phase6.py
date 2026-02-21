"""
Phase 6: ペーパートレード準備検証スクリプト（Out-of-Sample検証 - Option A拡張版）

推奨3銘柄（9983.T、6501.T、6758.T）でTrendFollowing(1/30/0.4)の汎化性能を検証。

主な機能:
- Out-of-Sample検証（2025年1-12月、Phase 5と重複なし、季節性・市場環境変化評価）
- TrendFollowingExit(1/30/0.4)固定（パラメータ最適化なし）
- GCエントリー固定
- 成功基準8項目チェック（平均PF>2.0、Win Rate>60%、取引数>30、PF標準偏差<平均50%、Max DD<15%、Sharpe>2.0、PF最大/最小比<3倍、全銘柄PF<50）
- MainSystemController統合準備（Phase 7リアルタイムペーパートレード移行）

統合コンポーネント:
- GCStrategyWithExit
- TrendFollowingExit(1/30/0.4)
- data_fetcher経由でデータ取得
- CSV+JSON統一出力

セーフティ機能/注意事項:
- 期間固定（2025-01-01 ~ 2025-03-31、3ヶ月）
- エントリー・エグジット固定（過学習回避）
- ルックアヘッドバイアス防止（copilot-instructions.md準拠）
- データ取得失敗時はスキップ（フォールバック禁止）
- PF上限制約（PF > 100は失格、PF > 50は警告）
- カーブフィッティング対策（Phase 5問題点への対応）

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


# Phase 6設定（Option A: 検証期間延長 2025年1-12月）
PHASE6_START_DATE = "2025-01-01"  # Out-of-Sample開始日
PHASE6_END_DATE = "2025-12-31"    # Out-of-Sample終了日（12ヶ月、季節性・市場環境変化評価）
WARMUP_DAYS = 150  # ウォームアップ期間（Option A-2暦日拡大方式）

# PF上限制約（カーブフィッティング対策）
PF_CRITICAL_THRESHOLD = 100.0  # PF > 100は失格
PF_WARNING_THRESHOLD = 50.0    # PF > 50は警告

# Phase 6成功基準
PHASE6_SUCCESS_CRITERIA = {
    'avg_pf': 2.0,              # 平均PF > 2.0
    'avg_win_rate': 0.60,       # 平均Win Rate > 60%
    'min_trades_per_ticker': 30, # 取引数/銘柄 > 30
    'pf_std_ratio': 0.50,       # PF標準偏差 < 平均の50%
    'max_drawdown': 0.15,       # Max Drawdown < 15%
    'sharpe_ratio': 2.0,        # Sharpe Ratio年率 > 2.0
    'pf_max_min_ratio': 3.0,    # PF最大/最小比 < 3倍
    'pf_upper_limit': 50.0      # 全銘柄PF < 50
}

# Phase 6推奨3銘柄（PF制約適用済み）
PHASE6_TICKERS = [
    ("9983.T", "ファーストリテイリング", "小売（衣料品）"),  # Phase 5 PF=5.72
    ("6501.T", "日立製作所", "製造業（電機）"),           # Phase 5 PF=3.67
    ("6758.T", "ソニーグループ", "製造業（電機）")        # Phase 5 PF=2.18
]


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


def calculate_performance_metrics(results_df: pd.DataFrame) -> dict:
    """
    パフォーマンス指標計算
    
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
    
    # 取引フィルタ（Trade_ID > 0で実トレード行を識別）
    # 理由: BaseStrategy.backtest()がTrade_ID列を0で初期化し、エグジット行のみ1,2,3...を付与
    trades = results_df[results_df['Trade_ID'] > 0].copy()
    
    # デバッグ出力
    print(f"[DEBUG] calculate_performance_metrics: total_rows={len(results_df)}, trades_filtered={len(trades)}")
    
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
    
    # Profit_Loss統計
    total_return = trades['Profit_Loss'].sum()
    total_trades = len(trades)
    
    # Profit Factor計算
    gains = trades[trades['Profit_Loss'] > 0]['Profit_Loss'].sum()
    losses = abs(trades[trades['Profit_Loss'] < 0]['Profit_Loss'].sum())
    profit_factor = gains / losses if losses != 0 else 0.0
    
    # Win Rate計算
    wins = len(trades[trades['Profit_Loss'] > 0])
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    # Sharpe Ratio計算（日次リターンベース）
    returns = trades['Profit_Loss'].values
    if len(returns) > 1:
        sharpe_ratio = np.mean(returns) / np.std(returns, ddof=1) if np.std(returns, ddof=1) != 0 else 0.0
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)  # 年率換算
    else:
        sharpe_ratio = 0.0
        sharpe_ratio_annualized = 0.0
    
    # Max Drawdown計算（累積P&L曲線から）
    cumulative_pl = trades['Profit_Loss'].cumsum()
    running_max = cumulative_pl.cummax()
    drawdown = cumulative_pl - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    max_drawdown_pct = (max_drawdown / abs(running_max.max())) * 100 if running_max.max() != 0 else 0.0
    
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


def validate_exit_on_ticker(ticker: str, ticker_name: str, industry: str) -> dict:
    """
    単一銘柄でエグジット戦略検証
    
    Args:
        ticker: ティッカーシンボル
        ticker_name: 銘柄名
        industry: 業種
    
    Returns:
        検証結果辞書
    """
    print(f"\n{'='*80}")
    print(f"[TICKER] {ticker} - {ticker_name} ({industry})")
    print(f"{'='*80}")
    
    try:
        # データ取得（ウォームアップ期間込み）
        print(f"[INFO] データ取得中: {ticker}")
        print(f"       検証期間: {PHASE6_START_DATE} ~ {PHASE6_END_DATE}")
        print(f"       ウォームアップ期間: {WARMUP_DAYS}日")
        
        ticker_data, start_date, end_date, stock_data, index_data = get_parameters_and_data(
            ticker=ticker,
            start_date=PHASE6_START_DATE,
            end_date=PHASE6_END_DATE,
            warmup_days=WARMUP_DAYS
        )
        
        if stock_data is None or len(stock_data) == 0:
            print(f"[WARNING] データ取得失敗: {ticker}")
            return {
                'ticker': ticker,
                'ticker_name': ticker_name,
                'industry': industry,
                'status': 'DATA_ERROR',
                'message': 'データ取得失敗'
            }
        
        print(f"[INFO] データ取得完了: {len(stock_data)}行")
        print(f"       データ期間: {stock_data.index[0].date()} ~ {stock_data.index[-1].date()}")
        
        # TrendFollowingExit(1/30/0.4)初期化
        exit_strategy = TrendFollowingExit(
            min_hold_days=1,
            max_hold_days=30,
            confidence_threshold=0.4
        )
        print(f"[INFO] TrendFollowingExit初期化完了: (1/30/0.4)")
        
        # GCStrategyWithExit初期化
        strategy = GCStrategyWithExit(
            data=stock_data,
            exit_strategy=exit_strategy,
            params={
                'short_window': 5,
                'long_window': 25
            }
        )
        print(f"[INFO] GCStrategyWithExit初期化完了")
        
        # バックテスト実行
        print(f"[INFO] バックテスト実行中...")
        # 日付文字列をpd.Timestampに変換（BaseStrategy.backtest()の要求仕様）
        start_ts = pd.Timestamp(PHASE6_START_DATE)
        end_ts = pd.Timestamp(PHASE6_END_DATE)
        results_df = strategy.backtest(
            trading_start_date=start_ts,
            trading_end_date=end_ts
        )
        
        if results_df is None or len(results_df) == 0:
            print(f"[WARNING] バックテスト結果が空: {ticker}")
            return {
                'ticker': ticker,
                'ticker_name': ticker_name,
                'industry': industry,
                'status': 'NO_RESULTS',
                'message': 'バックテスト結果が空'
            }
        
        # パフォーマンス指標計算
        metrics = calculate_performance_metrics(results_df)
        
        # PF制約チェック
        pf_check = validate_pf_threshold(metrics['profit_factor'], ticker)
        pf_status = pf_check['status']
        
        # 結果出力
        print(f"\n[RESULTS]")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Sharpe Ratio（年率）: {metrics['sharpe_ratio_annualized']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  PF Validation: [{pf_status}] {pf_check['message']}")
        
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'industry': industry,
            'metrics': metrics,
            'status': 'SUCCESS',
            'pf_status': pf_status
        }
        
    except Exception as e:
        print(f"[ERROR] 検証エラー: {ticker}")
        print(f"        {type(e).__name__}: {e}")
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'industry': industry,
            'status': 'ERROR',
            'message': str(e)
        }


def run_phase6_validation() -> list:
    """
    Phase 6推奨3銘柄で一括検証実行
    
    Returns:
        全銘柄の検証結果リスト
    """
    results = []
    
    for ticker, ticker_name, industry in PHASE6_TICKERS:
        result = validate_exit_on_ticker(ticker, ticker_name, industry)
        results.append(result)
    
    return results


def check_success_criteria(success_results: list) -> dict:
    """
    Phase 6成功基準チェック
    
    Args:
        success_results: 成功した銘柄の検証結果リスト
    
    Returns:
        成功基準チェック結果辞書
    """
    if len(success_results) == 0:
        return {
            'overall_status': 'FAIL',
            'message': '全銘柄で検証失敗',
            'criteria_results': {}
        }
    
    # 統計計算
    pf_values = [r['metrics']['profit_factor'] for r in success_results]
    win_rates = [r['metrics']['win_rate'] for r in success_results]
    trades_counts = [r['metrics']['total_trades'] for r in success_results]
    sharpe_ratios = [r['metrics']['sharpe_ratio_annualized'] for r in success_results]
    max_drawdowns = [r['metrics']['max_drawdown_pct'] for r in success_results]
    
    avg_pf = np.mean(pf_values)
    pf_std = np.std(pf_values, ddof=1)
    avg_win_rate = np.mean(win_rates)
    avg_trades = np.mean(trades_counts)
    avg_sharpe = np.mean(sharpe_ratios)
    avg_max_dd = np.mean(max_drawdowns)
    pf_max = np.max(pf_values)
    pf_min = np.min(pf_values)
    pf_max_min_ratio = pf_max / pf_min if pf_min > 0 else float('inf')
    
    # 成功基準チェック
    criteria_results = {
        'avg_pf': {
            'value': avg_pf,
            'threshold': PHASE6_SUCCESS_CRITERIA['avg_pf'],
            'pass': avg_pf > PHASE6_SUCCESS_CRITERIA['avg_pf']
        },
        'avg_win_rate': {
            'value': avg_win_rate,
            'threshold': PHASE6_SUCCESS_CRITERIA['avg_win_rate'],
            'pass': avg_win_rate > PHASE6_SUCCESS_CRITERIA['avg_win_rate']
        },
        'min_trades_per_ticker': {
            'value': avg_trades,
            'threshold': PHASE6_SUCCESS_CRITERIA['min_trades_per_ticker'],
            'pass': all(t > PHASE6_SUCCESS_CRITERIA['min_trades_per_ticker'] for t in trades_counts)
        },
        'pf_std_ratio': {
            'value': pf_std / avg_pf if avg_pf > 0 else 0,
            'threshold': PHASE6_SUCCESS_CRITERIA['pf_std_ratio'],
            'pass': (pf_std / avg_pf) < PHASE6_SUCCESS_CRITERIA['pf_std_ratio'] if avg_pf > 0 else False
        },
        'max_drawdown': {
            'value': avg_max_dd / 100.0,  # %を小数に変換
            'threshold': PHASE6_SUCCESS_CRITERIA['max_drawdown'],
            'pass': avg_max_dd < (PHASE6_SUCCESS_CRITERIA['max_drawdown'] * 100)
        },
        'sharpe_ratio': {
            'value': avg_sharpe,
            'threshold': PHASE6_SUCCESS_CRITERIA['sharpe_ratio'],
            'pass': avg_sharpe > PHASE6_SUCCESS_CRITERIA['sharpe_ratio']
        },
        'pf_max_min_ratio': {
            'value': pf_max_min_ratio,
            'threshold': PHASE6_SUCCESS_CRITERIA['pf_max_min_ratio'],
            'pass': pf_max_min_ratio < PHASE6_SUCCESS_CRITERIA['pf_max_min_ratio']
        },
        'pf_upper_limit': {
            'value': pf_max,
            'threshold': PHASE6_SUCCESS_CRITERIA['pf_upper_limit'],
            'pass': all(pf < PHASE6_SUCCESS_CRITERIA['pf_upper_limit'] for pf in pf_values)
        }
    }
    
    # 全基準PASS判定
    all_pass = all(c['pass'] for c in criteria_results.values())
    overall_status = 'PASS' if all_pass else 'FAIL'
    
    return {
        'overall_status': overall_status,
        'criteria_results': criteria_results,
        'statistics': {
            'avg_pf': avg_pf,
            'pf_std': pf_std,
            'avg_win_rate': avg_win_rate,
            'avg_trades': avg_trades,
            'avg_sharpe': avg_sharpe,
            'avg_max_dd': avg_max_dd,
            'pf_max': pf_max,
            'pf_min': pf_min,
            'pf_max_min_ratio': pf_max_min_ratio
        }
    }


def analyze_and_output_results(all_results: list):
    """
    Phase 6検証結果の分析と出力
    
    Args:
        all_results: 全銘柄の検証結果リスト
    """
    print(f"\n{'='*80}")
    print("Phase 6検証結果分析")
    print(f"{'='*80}")
    
    # 成功・失敗集計
    success_results = [r for r in all_results if r.get('status') == 'SUCCESS']
    error_results = [r for r in all_results if r.get('status') != 'SUCCESS']
    
    print(f"\n[SUMMARY]")
    print(f"  総検証銘柄数: {len(all_results)}")
    print(f"  成功: {len(success_results)} 銘柄")
    print(f"  失敗: {len(error_results)} 銘柄")
    
    if len(error_results) > 0:
        print(f"\n[失敗銘柄]")
        for r in error_results:
            print(f"  {r['ticker']} - {r['ticker_name']}: {r.get('message', 'Unknown error')}")
    
    if len(success_results) == 0:
        print(f"\n[ERROR] 成功した銘柄が0件のため、統計分析をスキップします。")
        return
    
    # 成功基準チェック
    criteria_check = check_success_criteria(success_results)
    
    print(f"\n{'='*80}")
    print("Phase 6成功基準チェック")
    print(f"{'='*80}")
    print(f"\n総合判定: {criteria_check['overall_status']}")
    
    print(f"\n基準別結果:")
    for criterion, result in criteria_check['criteria_results'].items():
        status = 'PASS' if result['pass'] else 'FAIL'
        if criterion == 'avg_win_rate':
            print(f"  [{status}] {criterion}: {result['value']:.1%} (threshold: {result['threshold']:.1%})")
        elif criterion in ['pf_std_ratio', 'max_drawdown']:
            print(f"  [{status}] {criterion}: {result['value']:.2%} (threshold: {result['threshold']:.2%})")
        else:
            print(f"  [{status}] {criterion}: {result['value']:.2f} (threshold: {result['threshold']:.2f})")
    
    stats = criteria_check['statistics']
    print(f"\n統計サマリー:")
    print(f"  平均PF: {stats['avg_pf']:.2f}")
    print(f"  PF標準偏差: {stats['pf_std']:.2f} ({(stats['pf_std']/stats['avg_pf']*100):.1f}% of mean)")
    print(f"  平均Win Rate: {stats['avg_win_rate']:.1%}")
    print(f"  平均取引数: {stats['avg_trades']:.1f}")
    print(f"  平均Sharpe（年率）: {stats['avg_sharpe']:.2f}")
    print(f"  平均Max Drawdown: {stats['avg_max_dd']:.2f}%")
    print(f"  PF最大/最小比: {stats['pf_max_min_ratio']:.2f}")
    
    # PF制約統計
    pf_ok_count = sum(1 for r in success_results if r.get('pf_status') == 'OK')
    pf_warning_count = sum(1 for r in success_results if r.get('pf_status') == 'WARNING')
    pf_fail_count = sum(1 for r in success_results if r.get('pf_status') == 'FAIL')
    
    print(f"\n[PF制約統計（カーブフィッティング対策）]")
    print(f"  OK (PF <= 50): {pf_ok_count} 銘柄")
    print(f"  WARNING (50 < PF <= 100): {pf_warning_count} 銘柄")
    print(f"  FAIL (PF > 100): {pf_fail_count} 銘柄")
    
    # CSV出力
    output_dir = Path("output/exit_strategy_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"phase6_validation_{timestamp}.csv"
    
    df_rows = []
    for r in success_results:
        m = r['metrics']
        df_rows.append({
            'ticker': r['ticker'],
            'ticker_name': r['ticker_name'],
            'industry': r['industry'],
            'total_trades': m['total_trades'],
            'profit_factor': m['profit_factor'],
            'win_rate': m['win_rate'],
            'sharpe_ratio_annualized': m['sharpe_ratio_annualized'],
            'max_drawdown_pct': m['max_drawdown_pct'],
            'pf_status': r.get('pf_status', 'UNKNOWN')
        })
    
    df = pd.DataFrame(df_rows)
    df = df.sort_values('profit_factor', ascending=False)
    
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[OUTPUT] CSV出力完了: {csv_path}")
    except Exception as e:
        print(f"[WARNING] CSV出力スキップ: {e}")
    
    # JSON出力
    json_path = output_dir / f"phase6_summary_{timestamp}.json"
    
    summary = {
        'phase': 'Phase 6',
        'validation_period': f'{PHASE6_START_DATE} ~ {PHASE6_END_DATE}',
        'exit_strategy': 'TrendFollowingExit(1/30/0.4)',
        'entry_strategy': 'GCStrategy',
        'timestamp': timestamp,
        'overall_status': criteria_check['overall_status'],
        'success_criteria': {
            k: {
                'value': v['value'],
                'threshold': v['threshold'],
                'pass': v['pass']
            }
            for k, v in criteria_check['criteria_results'].items()
        },
        'statistics': stats,
        'pf_constraints': {
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
    
    sorted_results = sorted(success_results, key=lambda x: x['metrics']['profit_factor'], reverse=True)
    
    for i, r in enumerate(sorted_results, 1):
        m = r['metrics']
        pf_status = r.get('pf_status', 'UNKNOWN')
        print(f"{i}. {r['ticker']} - {r['ticker_name']} ({r['industry']})")
        print(f"   PF={m['profit_factor']:.2f}, WR={m['win_rate']:.1%}, "
              f"Trades={m['total_trades']}, Sharpe(annualized)={m['sharpe_ratio_annualized']:.2f}, "
              f"PF_Status={pf_status}")
    
    # Phase 5比較
    print(f"\n{'='*80}")
    print("Phase 5 vs Phase 6 比較")
    print(f"{'='*80}")
    print(f"\nPhase 5（2023-2024、In-Sample）:")
    print(f"  推奨3銘柄: 9983.T (PF=5.72), 6501.T (PF=3.67), 6758.T (PF=2.18)")
    print(f"  平均PF: 3.86")
    print(f"\nPhase 6（2025年1-12月、Out-of-Sample - Option A拡張版）:")
    print(f"  平均PF: {stats['avg_pf']:.2f}")
    print(f"  PF変化率: {((stats['avg_pf']/3.86 - 1) * 100):.1f}%")
    print(f"  平均取引数: {stats['avg_trades']:.1f}（目標30以上）")
    
    if stats['avg_pf'] < 3.86 * 0.5:
        print(f"\n[WARNING] Out-of-Sample性能が50%以上低下")
        print(f"          パラメータ最適化バイアスの可能性")
    elif stats['avg_pf'] > 3.86 * 1.5:
        print(f"\n[INFO] Out-of-Sample性能が50%以上向上")
        print(f"       2025年市場環境に適合している可能性")
    else:
        print(f"\n[INFO] Out-of-Sample性能がIn-Sampleと近似")
        print(f"       汎化性能良好")


def main():
    """メインエントリーポイント"""
    print("\n" + "="*80)
    print("Phase 6: ペーパートレード準備検証（Out-of-Sample - Option A拡張版）")
    print("="*80 + "\n")
    
    print(f"検証期間: {PHASE6_START_DATE} ~ {PHASE6_END_DATE}（12ヶ月）")
    print(f"推奨3銘柄: {', '.join([t[0] for t in PHASE6_TICKERS])}")
    print(f"エグジット戦略: TrendFollowingExit(1/30/0.4)固定")
    print(f"ウォームアップ期間: {WARMUP_DAYS}日")
    print(f"目的: 季節性・市場環境変化の影響評価、統計的有意性向上\n")
    
    # 検証実行
    all_results = run_phase6_validation()
    
    # 結果分析・出力
    analyze_and_output_results(all_results)
    
    print("\n" + "="*80)
    print("Phase 6検証完了")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
