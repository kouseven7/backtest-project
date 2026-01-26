"""
Priority 1: TrendFollowing 3パラメータセット並行検証スクリプト

推奨3銘柄（9983.T、6501.T、6758.T）でTrendFollowingの3パラメータセット
(1/30/0.4、3/60/0.5、5/90/0.6)を2025年Out-of-Sample期間で並行検証。

主な機能:
- Out-of-Sample検証（2025年1-12月、Phase 5と重複なし）
- TrendFollowing 3パラメータセット並行検証（パラメータ最適化バイアス排除）
- GCエントリー固定
- パラメータセット間性能比較（平均PF、Win Rate、取引数、Sharpe年率、Max DD）
- 2025年市場環境に最適なパラメータ特定

統合コンポーネント:
- GCStrategyWithExit
- TrendFollowingExit（3パラメータセット）
- data_fetcher経由でデータ取得
- CSV+JSON統一出力

セーフティ機能/注意事項:
- 期間固定（2025-01-01 ~ 2025-12-31、12ヶ月）
- エントリー固定（GCStrategy、過学習回避）
- ルックアヘッドバイアス防止（copilot-instructions.md準拠）
- データ取得失敗時はスキップ（フォールバック禁止）
- PF上限制約（PF > 50は警告）

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


# Priority 1設定
PRIORITY1_START_DATE = "2025-01-01"  # Out-of-Sample開始日
PRIORITY1_END_DATE = "2025-12-31"    # Out-of-Sample終了日（12ヶ月）
WARMUP_DAYS = 150  # ウォームアップ期間（Option A-2暦日拡大方式）

# TrendFollowing 3パラメータセット
TRENDFOLLOW_PARAM_SETS = [
    {
        'name': 'Aggressive_1_30_0.4',
        'min_hold_days': 1,
        'max_hold_days': 30,
        'confidence_threshold': 0.4,
        'description': '攻撃的（Phase 5最優秀、2023-2024最適化）'
    },
    {
        'name': 'Moderate_3_60_0.5',
        'min_hold_days': 3,
        'max_hold_days': 60,
        'confidence_threshold': 0.5,
        'description': '中間（Phase 3検証済み）'
    },
    {
        'name': 'Conservative_5_90_0.6',
        'min_hold_days': 5,
        'max_hold_days': 90,
        'confidence_threshold': 0.6,
        'description': '保守的（Phase 3検証済み）'
    }
]

# Priority 1対象3銘柄（Phase 6推奨銘柄）
PRIORITY1_TICKERS = [
    ("9983.T", "ファーストリテイリング", "小売（衣料品）"),
    ("6501.T", "日立製作所", "製造業（電機）"),
    ("6758.T", "ソニーグループ", "製造業（電機）")
]

# Priority 1成功基準（Phase 6継承）
PRIORITY1_SUCCESS_CRITERIA = {
    'avg_pf': 2.0,              # 平均PF > 2.0
    'avg_win_rate': 0.60,       # 平均Win Rate > 60%
    'min_trades_per_ticker': 30, # 取引数/銘柄 > 30
    'pf_std_ratio': 0.50,       # PF標準偏差 < 平均の50%
    'max_drawdown': 0.15,       # Max Drawdown < 15%
    'sharpe_ratio': 2.0,        # Sharpe Ratio年率 > 2.0
}

# PF上限制約
PF_WARNING_THRESHOLD = 50.0


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
            'sharpe_ratio_annualized': 0.0,
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
            'sharpe_ratio_annualized': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # 取引フィルタ（Trade_ID > 0で実トレード行を識別）
    trades = results_df[results_df['Trade_ID'] > 0].copy()
    
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'total_return': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio_annualized': 0.0,
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
    
    # Sharpe Ratio年率換算
    returns = trades['Profit_Loss'].values
    if len(returns) > 1:
        sharpe_ratio = np.mean(returns) / np.std(returns, ddof=1) if np.std(returns, ddof=1) != 0 else 0.0
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)
    else:
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
        'sharpe_ratio_annualized': sharpe_ratio_annualized,
        'max_drawdown_pct': max_drawdown_pct
    }


def validate_exit_on_ticker_param(ticker: str, ticker_name: str, industry: str, 
                                   param_set: dict, stock_data: pd.DataFrame) -> dict:
    """
    単一銘柄・単一パラメータセットでエグジット戦略検証
    
    Args:
        ticker: ティッカーシンボル
        ticker_name: 銘柄名
        industry: 業種
        param_set: パラメータセット辞書
        stock_data: 株価データ（ウォームアップ期間込み）
    
    Returns:
        検証結果辞書
    """
    try:
        # TrendFollowingExit初期化
        exit_strategy = TrendFollowingExit(
            min_hold_days=param_set['min_hold_days'],
            max_hold_days=param_set['max_hold_days'],
            confidence_threshold=param_set['confidence_threshold']
        )
        
        # GCStrategyWithExit初期化
        strategy = GCStrategyWithExit(
            data=stock_data,
            exit_strategy=exit_strategy,
            ticker=ticker
        )
        
        # バックテスト実行（2025年期間のみ）
        trading_start_date = pd.Timestamp(PRIORITY1_START_DATE)
        trading_end_date = pd.Timestamp(PRIORITY1_END_DATE)
        
        results_df = strategy.backtest(
            trading_start_date=trading_start_date,
            trading_end_date=trading_end_date
        )
        
        # パフォーマンス指標計算
        metrics = calculate_performance_metrics(results_df)
        
        # PF警告チェック
        pf_warning = ""
        if metrics['profit_factor'] > PF_WARNING_THRESHOLD:
            pf_warning = f" [WARNING: PF > {PF_WARNING_THRESHOLD}]"
        
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'industry': industry,
            'param_name': param_set['name'],
            'param_description': param_set['description'],
            'status': 'SUCCESS',
            'total_trades': metrics['total_trades'],
            'profit_factor': metrics['profit_factor'],
            'win_rate': metrics['win_rate'],
            'sharpe_ratio_annualized': metrics['sharpe_ratio_annualized'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'pf_warning': pf_warning
        }
        
    except Exception as e:
        print(f"[ERROR] {ticker} - {param_set['name']}: {str(e)}")
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'industry': industry,
            'param_name': param_set['name'],
            'param_description': param_set['description'],
            'status': 'ERROR',
            'message': str(e)
        }


def check_success_criteria(results: list) -> dict:
    """
    Priority 1成功基準チェック（パラメータセットごと）
    
    Args:
        results: 全検証結果リスト
    
    Returns:
        成功基準判定結果
    """
    # パラメータセットごとに集計
    param_stats = {}
    
    for param_set in TRENDFOLLOW_PARAM_SETS:
        param_name = param_set['name']
        param_results = [r for r in results if r.get('param_name') == param_name and r['status'] == 'SUCCESS']
        
        if len(param_results) == 0:
            param_stats[param_name] = {
                'status': 'NO_DATA',
                'avg_pf': 0.0,
                'avg_win_rate': 0.0,
                'avg_trades': 0.0,
                'avg_sharpe': 0.0,
                'avg_max_dd': 0.0
            }
            continue
        
        # 平均値計算
        avg_pf = np.mean([r['profit_factor'] for r in param_results])
        avg_win_rate = np.mean([r['win_rate'] for r in param_results])
        avg_trades = np.mean([r['total_trades'] for r in param_results])
        avg_sharpe = np.mean([r['sharpe_ratio_annualized'] for r in param_results])
        avg_max_dd = np.mean([r['max_drawdown_pct'] for r in param_results])
        
        # PF標準偏差比
        pf_values = [r['profit_factor'] for r in param_results]
        pf_std = np.std(pf_values, ddof=1) if len(pf_values) > 1 else 0.0
        pf_std_ratio = (pf_std / avg_pf) if avg_pf > 0 else 0.0
        
        # 成功基準チェック
        criteria_checks = {
            'avg_pf': avg_pf > PRIORITY1_SUCCESS_CRITERIA['avg_pf'],
            'avg_win_rate': avg_win_rate > PRIORITY1_SUCCESS_CRITERIA['avg_win_rate'],
            'min_trades': avg_trades > PRIORITY1_SUCCESS_CRITERIA['min_trades_per_ticker'],
            'pf_std_ratio': pf_std_ratio < PRIORITY1_SUCCESS_CRITERIA['pf_std_ratio'],
            'max_drawdown': (avg_max_dd / 100) < PRIORITY1_SUCCESS_CRITERIA['max_drawdown'],
            'sharpe_ratio': avg_sharpe > PRIORITY1_SUCCESS_CRITERIA['sharpe_ratio']
        }
        
        overall_pass = all(criteria_checks.values())
        
        param_stats[param_name] = {
            'status': 'PASS' if overall_pass else 'FAIL',
            'avg_pf': avg_pf,
            'avg_win_rate': avg_win_rate * 100,  # パーセント表示
            'avg_trades': avg_trades,
            'avg_sharpe': avg_sharpe,
            'avg_max_dd': avg_max_dd,
            'pf_std_ratio': pf_std_ratio * 100,  # パーセント表示
            'criteria_checks': criteria_checks,
            'pass_count': sum(criteria_checks.values()),
            'total_criteria': len(criteria_checks)
        }
    
    return param_stats


def main():
    """Priority 1メイン実行"""
    
    print("\n" + "="*80)
    print("Priority 1: TrendFollowing 3パラメータセット並行検証")
    print("="*80 + "\n")
    
    print(f"検証期間: {PRIORITY1_START_DATE} ~ {PRIORITY1_END_DATE}（12ヶ月）")
    print(f"対象3銘柄: {', '.join([t[0] for t in PRIORITY1_TICKERS])}")
    print(f"ウォームアップ期間: {WARMUP_DAYS}日")
    print(f"目的: パラメータ最適化バイアス排除、2025年最適パラメータ特定\n")
    
    print("TrendFollowing 3パラメータセット:")
    for param_set in TRENDFOLLOW_PARAM_SETS:
        print(f"  - {param_set['name']}: ({param_set['min_hold_days']}/{param_set['max_hold_days']}/{param_set['confidence_threshold']}) - {param_set['description']}")
    print()
    
    # 全検証結果
    all_results = []
    
    # 銘柄ごとにデータ取得・全パラメータセット検証
    for ticker, ticker_name, industry in PRIORITY1_TICKERS:
        print(f"\n{'='*80}")
        print(f"[TICKER] {ticker} - {ticker_name} ({industry})")
        print(f"{'='*80}")
        
        try:
            # データ取得（1回のみ、全パラメータセットで共用）
            print(f"[INFO] データ取得中...")
            ticker_data, start_date, end_date, stock_data, index_data = get_parameters_and_data(
                ticker=ticker,
                start_date=PRIORITY1_START_DATE,
                end_date=PRIORITY1_END_DATE,
                warmup_days=WARMUP_DAYS
            )
            
            if stock_data is None or len(stock_data) == 0:
                print(f"[WARNING] データ取得失敗: {ticker}")
                continue
            
            print(f"[INFO] データ取得完了: {len(stock_data)}行")
            print(f"       データ期間: {stock_data.index[0].date()} ~ {stock_data.index[-1].date()}")
            
            # 3パラメータセット並行検証
            for param_set in TRENDFOLLOW_PARAM_SETS:
                print(f"\n[PARAM] {param_set['name']} - {param_set['description']}")
                
                result = validate_exit_on_ticker_param(
                    ticker, ticker_name, industry, param_set, stock_data
                )
                
                if result['status'] == 'SUCCESS':
                    print(f"[RESULTS]")
                    print(f"  Total Trades: {result['total_trades']}")
                    print(f"  Profit Factor: {result['profit_factor']:.2f}{result['pf_warning']}")
                    print(f"  Win Rate: {result['win_rate']*100:.1f}%")
                    print(f"  Sharpe Ratio (年率): {result['sharpe_ratio_annualized']:.2f}")
                    print(f"  Max Drawdown: {result['max_drawdown_pct']:.2f}%")
                else:
                    print(f"[ERROR] {result.get('message', 'Unknown error')}")
                
                all_results.append(result)
        
        except Exception as e:
            print(f"[ERROR] {ticker}データ取得失敗: {str(e)}")
            continue
    
    # 成功基準チェック（パラメータセットごと）
    print(f"\n{'='*80}")
    print("Priority 1成功基準チェック（パラメータセットごと）")
    print(f"{'='*80}")
    
    param_stats = check_success_criteria(all_results)
    
    # 最優秀パラメータセット特定
    best_param = None
    best_pf = 0.0
    
    for param_name, stats in param_stats.items():
        if stats['status'] == 'NO_DATA':
            print(f"\n[PARAM] {param_name}")
            print(f"  Status: NO_DATA (検証結果なし)")
            continue
        
        print(f"\n[PARAM] {param_name}")
        print(f"  総合判定: {stats['status']}")
        print(f"  平均PF: {stats['avg_pf']:.2f} (目標: {PRIORITY1_SUCCESS_CRITERIA['avg_pf']:.2f})")
        print(f"  平均Win Rate: {stats['avg_win_rate']:.1f}% (目標: {PRIORITY1_SUCCESS_CRITERIA['avg_win_rate']*100:.1f}%)")
        print(f"  平均取引数: {stats['avg_trades']:.1f} (目標: {PRIORITY1_SUCCESS_CRITERIA['min_trades_per_ticker']:.1f})")
        print(f"  PF標準偏差比: {stats['pf_std_ratio']:.1f}% (目標: {PRIORITY1_SUCCESS_CRITERIA['pf_std_ratio']*100:.1f}%)")
        print(f"  平均Sharpe年率: {stats['avg_sharpe']:.2f} (目標: {PRIORITY1_SUCCESS_CRITERIA['sharpe_ratio']:.2f})")
        print(f"  平均Max DD: {stats['avg_max_dd']:.2f}% (目標: {PRIORITY1_SUCCESS_CRITERIA['max_drawdown']*100:.2f}%)")
        print(f"  成功基準達成: {stats['pass_count']}/{stats['total_criteria']}項目")
        
        # 最優秀パラメータ更新
        if stats['avg_pf'] > best_pf:
            best_pf = stats['avg_pf']
            best_param = param_name
    
    # Phase 6比較
    print(f"\n{'='*80}")
    print("Phase 6 vs Priority 1 比較")
    print(f"{'='*80}")
    
    print(f"\nPhase 6（TrendFollowing 1/30/0.4固定）:")
    print(f"  平均PF: 1.09")
    print(f"  平均Win Rate: 22.4%")
    print(f"  平均取引数: 34.0")
    
    if best_param and param_stats[best_param]['status'] != 'NO_DATA':
        best_stats = param_stats[best_param]
        pf_change = ((best_stats['avg_pf'] / 1.09 - 1) * 100) if 1.09 > 0 else 0.0
        
        print(f"\nPriority 1最優秀パラメータ（{best_param}）:")
        print(f"  平均PF: {best_stats['avg_pf']:.2f}")
        print(f"  PF変化率: {pf_change:+.1f}%")
        print(f"  平均Win Rate: {best_stats['avg_win_rate']:.1f}%")
        print(f"  平均取引数: {best_stats['avg_trades']:.1f}")
        
        if best_stats['avg_pf'] > 1.09:
            print(f"\n[SUCCESS] Priority 1で性能改善達成")
        else:
            print(f"\n[WARNING] Priority 1でも性能改善せず")
    
    # CSV出力
    output_dir = project_root / "output" / "exit_strategy_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"priority1_validation_{timestamp}.csv"
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] CSV出力完了: {csv_path}")
    
    # JSON出力（パラメータセット統計）
    json_path = output_dir / f"priority1_summary_{timestamp}.json"
    
    summary = {
        'validation_date': datetime.now().isoformat(),
        'period': f"{PRIORITY1_START_DATE} ~ {PRIORITY1_END_DATE}",
        'tickers': [t[0] for t in PRIORITY1_TICKERS],
        'param_sets': TRENDFOLLOW_PARAM_SETS,
        'param_stats': param_stats,
        'best_param': best_param,
        'phase6_comparison': {
            'phase6_avg_pf': 1.09,
            'priority1_best_pf': param_stats[best_param]['avg_pf'] if best_param and param_stats[best_param]['status'] != 'NO_DATA' else 0.0
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] JSON出力完了: {json_path}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
