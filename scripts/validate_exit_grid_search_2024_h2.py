"""
Option A Step 2: 2024年7-12月Out-of-Sample検証スクリプト

2024年7-12月データでStep 1最優秀パラメータ（1_30_0.3）の汎化性能検証。
In-Sample（Step 1）で最適化されたパラメータが、Out-of-Sampleでも性能維持できるか確認。

主な機能:
- Out-of-Sample検証（2024年7-12月、6ヶ月）
- パラメータ固定（Step 1最優秀: min_hold=1, max_hold=30, confidence=0.3）
- GCエントリー固定
- 推奨3銘柄（9983.T、6501.T、6758.T）検証
- Step 1比較分析（PF低下率、Win Rate維持、取引数変化）
- CSV+JSON統一出力（grid_search_2024_h2_*.csv、grid_search_summary_2024_h2_*.json）

統合コンポーネント:
- GCStrategyWithExit
- TrendFollowingExit（固定: 1/30/0.3）
- data_fetcher経由でデータ取得

セーフティ機能/注意事項:
- 期間固定（2024-07-01 ~ 2024-12-31、6ヶ月）
- エントリー固定（GCStrategy、過学習回避）
- ルックアヘッドバイアス防止（copilot-instructions.md準拠）
- データ取得失敗時はスキップ（フォールバック禁止）
- PF低下率閾値（Step 1: 54.00 → Step 2: > 27.00で成功）

Option A Step 3:
- Step 3: 2025年1-3月真のOut-of-Sample検証（validate_exit_grid_search_2025_q1.py）

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


# Option A Step 2設定
GRID_SEARCH_START_DATE = "2024-07-01"  # Out-of-Sample開始日
GRID_SEARCH_END_DATE = "2024-12-31"    # Out-of-Sample終了日（6ヶ月）
WARMUP_DAYS = 150  # ウォームアップ期間（Option A-2暦日拡大方式）

# Step 1最優秀パラメータ固定
BEST_PARAM_STEP1 = {
    'min_hold_days': 1,
    'max_hold_days': 30,
    'confidence_threshold': 0.3,
    'param_name': '1_30_0.3',
    'step1_avg_pf': 54.00,  # Step 1結果（参照値）
    'step1_avg_win_rate': 0.667,  # Step 1結果（参照値）
    'step1_avg_trades': 7.3  # Step 1結果（参照値）
}

# Option A対象3銘柄（Phase 6推奨銘柄）
GRID_SEARCH_TICKERS = [
    ("9983.T", "ファーストリテイリング", "小売（衣料品）"),
    ("6501.T", "日立製作所", "製造業（電機）"),
    ("6758.T", "ソニーグループ", "製造業（電機）")
]

# Option A Step 2成功基準
GRID_SEARCH_SUCCESS_CRITERIA = {
    'avg_pf': 2.0,              # 平均PF > 2.0維持
    'avg_win_rate': 0.40,       # 平均Win Rate > 40%維持
    'min_trades_per_ticker': 10, # 取引数/銘柄 > 10（6ヶ月想定緩和）
    'pf_degradation_max': 0.50, # PF低下率 < 50%（Step 1: 54.00 → Step 2: > 27.00）
    'max_drawdown': 0.15,       # Max Drawdown < 15%
    'sharpe_ratio': 1.0,        # Sharpe Ratio年率 > 1.0（Step 2緩和基準）
}

# PF上限制約
PF_WARNING_THRESHOLD = 50.0
PF_DISQUALIFICATION_THRESHOLD = 100.0


def calculate_performance_metrics(results_df: pd.DataFrame) -> dict:
    """
    パフォーマンス指標計算
    
    Args:
        results_df: バックテスト結果DataFrame
    
    Returns:
        指標dict（PF、Win Rate、取引数、Sharpe年率、Max DD）
    """
    if results_df.empty or 'Profit_Loss' not in results_df.columns:
        return {
            'total_trades': 0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio_annual': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # 取引フィルタリング
    trades = results_df[results_df['Profit_Loss'] != 0].copy()
    total_trades = len(trades)
    
    if total_trades == 0:
        return {
            'total_trades': 0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio_annual': 0.0,
            'max_drawdown_pct': 0.0
        }
    
    # Profit Factor計算
    profits = trades[trades['Profit_Loss'] > 0]['Profit_Loss']
    losses = trades[trades['Profit_Loss'] < 0]['Profit_Loss'].abs()
    total_profit = profits.sum() if not profits.empty else 0.0
    total_loss = losses.sum() if not losses.empty else 0.0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    
    # Win Rate計算
    wins = len(profits)
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    # Sharpe Ratio年率換算
    trade_returns = trades['Profit_Loss'] / 1000000  # 初期資金100万円想定
    mean_return = trade_returns.mean()
    std_return = trade_returns.std()
    sharpe_ratio_annual = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
    
    # Max Drawdown計算
    cumulative_pl = trades['Profit_Loss'].cumsum()
    running_max = cumulative_pl.expanding().max()
    drawdown = cumulative_pl - running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = abs(max_drawdown / 1000000) if max_drawdown < 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'sharpe_ratio_annual': sharpe_ratio_annual,
        'max_drawdown_pct': max_drawdown_pct
    }


def run_single_backtest(
    ticker: str,
    ticker_name: str,
    param: dict
) -> dict:
    """
    単一銘柄・単一パラメータでバックテスト実行
    
    Args:
        ticker: ティッカーシンボル
        ticker_name: 銘柄名
        param: パラメータdict（min_hold_days、max_hold_days、confidence_threshold）
    
    Returns:
        結果dict（指標含む）
    """
    try:
        # データ取得
        _, _, _, stock_data, index_data = get_parameters_and_data(
            ticker=ticker,
            start_date=GRID_SEARCH_START_DATE,
            end_date=GRID_SEARCH_END_DATE,
            warmup_days=WARMUP_DAYS
        )
        
        # TrendFollowingExit生成
        exit_strategy = TrendFollowingExit(
            min_hold_days=param['min_hold_days'],
            max_hold_days=param['max_hold_days'],
            confidence_threshold=param['confidence_threshold']
        )
        
        # GCStrategyWithExit生成（正しい引数: data=stock_dataのみ）
        strategy = GCStrategyWithExit(
            data=stock_data,
            exit_strategy=exit_strategy,
            ticker=ticker
        )
        
        # バックテスト実行
        results_df = strategy.backtest(
            trading_start_date=pd.Timestamp(GRID_SEARCH_START_DATE),
            trading_end_date=pd.Timestamp(GRID_SEARCH_END_DATE)
        )
        
        # パフォーマンス指標計算
        metrics = calculate_performance_metrics(results_df)
        
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'param_name': param['param_name'],
            'min_hold_days': param['min_hold_days'],
            'max_hold_days': param['max_hold_days'],
            'confidence_threshold': param['confidence_threshold'],
            'total_trades': metrics['total_trades'],
            'profit_factor': metrics['profit_factor'],
            'win_rate': metrics['win_rate'],
            'sharpe_ratio_annual': metrics['sharpe_ratio_annual'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        print(f"  [ERROR] {ticker} バックテスト失敗: {e}")
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'param_name': param['param_name'],
            'status': 'FAILED',
            'error': str(e)
        }


def check_success_criteria(
    param_results: list,
    param_name: str
) -> dict:
    """
    成功基準チェック
    
    Args:
        param_results: パラメータ組み合わせの全銘柄結果リスト
        param_name: パラメータ名（例: 1_30_0.3）
    
    Returns:
        成功基準判定結果
    """
    success_results = [r for r in param_results if r['status'] == 'SUCCESS']
    
    if not success_results:
        return {
            'param_name': param_name,
            'avg_pf': 0.0,
            'avg_win_rate': 0.0,
            'avg_trades': 0.0,
            'avg_sharpe_annual': 0.0,
            'avg_max_dd': 0.0,
            'pf_std': 0.0,
            'pf_degradation': 100.0,  # Step 1比較（100%低下 = 全滅）
            'criteria_pass': {
                'avg_pf': False,
                'avg_win_rate': False,
                'min_trades': False,
                'pf_degradation': False,
                'max_dd': False,
                'sharpe': False
            },
            'pass_count': 0,
            'overall_status': 'FAIL'
        }
    
    # 平均指標計算（numpy型→Python型変換）
    pf_values = [r['profit_factor'] for r in success_results if r['profit_factor'] > 0]
    avg_pf = float(np.mean(pf_values)) if pf_values else 0.0
    avg_win_rate = float(np.mean([r['win_rate'] for r in success_results]))
    avg_trades = float(np.mean([r['total_trades'] for r in success_results]))
    avg_sharpe_annual = float(np.mean([r['sharpe_ratio_annual'] for r in success_results]))
    avg_max_dd = float(np.mean([r['max_drawdown_pct'] for r in success_results]))
    pf_std = float(np.std(pf_values)) if len(pf_values) > 1 else 0.0
    
    # Step 1比較（PF低下率）
    pf_degradation = (BEST_PARAM_STEP1['step1_avg_pf'] - avg_pf) / BEST_PARAM_STEP1['step1_avg_pf'] if avg_pf > 0 else 1.0
    
    # 成功基準チェック
    criteria_pass = {
        'avg_pf': avg_pf > GRID_SEARCH_SUCCESS_CRITERIA['avg_pf'],
        'avg_win_rate': avg_win_rate > GRID_SEARCH_SUCCESS_CRITERIA['avg_win_rate'],
        'min_trades': avg_trades > GRID_SEARCH_SUCCESS_CRITERIA['min_trades_per_ticker'],
        'pf_degradation': pf_degradation < GRID_SEARCH_SUCCESS_CRITERIA['pf_degradation_max'],  # PF低下率 < 50%
        'max_dd': avg_max_dd < GRID_SEARCH_SUCCESS_CRITERIA['max_drawdown'],
        'sharpe': avg_sharpe_annual > GRID_SEARCH_SUCCESS_CRITERIA['sharpe_ratio']
    }
    
    # 総合判定（6基準中4基準以上PASS）
    pass_count_int = int(sum(criteria_pass.values()))
    overall_status = 'PASS' if pass_count_int >= 4 else 'FAIL'
    
    return {
        'param_name': param_name,
        'avg_pf': avg_pf,
        'avg_win_rate': avg_win_rate,
        'avg_trades': avg_trades,
        'avg_sharpe_annual': avg_sharpe_annual,
        'avg_max_dd': avg_max_dd,
        'pf_std': pf_std,
        'pf_degradation': pf_degradation,  # Python int型
        'criteria_pass': criteria_pass,
        'pass_count': pass_count_int,  # Python int型
        'overall_status': overall_status
    }


def main():
    """メインエントリーポイント"""
    
    print("\n" + "=" * 80)
    print("Option A Step 2: 2024年7-12月Out-of-Sample検証（TrendFollowing固定パラメータ）")
    print("=" * 80)
    print(f"検証期間: {GRID_SEARCH_START_DATE} ~ {GRID_SEARCH_END_DATE}（6ヶ月、Out-of-Sample）")
    print(f"対象銘柄: {len(GRID_SEARCH_TICKERS)}銘柄（Phase 6推奨3銘柄）")
    print(f"固定パラメータ: {BEST_PARAM_STEP1['param_name']} (min_hold={BEST_PARAM_STEP1['min_hold_days']}, max_hold={BEST_PARAM_STEP1['max_hold_days']}, confidence={BEST_PARAM_STEP1['confidence_threshold']})")
    print(f"Step 1参照値: 平均PF={BEST_PARAM_STEP1['step1_avg_pf']:.2f}, Win Rate={BEST_PARAM_STEP1['step1_avg_win_rate']*100:.1f}%, 取引数={BEST_PARAM_STEP1['step1_avg_trades']:.1f}")
    print("=" * 80 + "\n")
    
    # 全検証結果格納
    all_results = []
    param_results = []
    
    # パラメータ固定（Step 1最優秀）
    param = BEST_PARAM_STEP1.copy()
    
    print(f"[PARAM] {param['param_name']}")
    print(f"  min_hold={param['min_hold_days']}, max_hold={param['max_hold_days']}, confidence={param['confidence_threshold']}\n")
    
    # 各銘柄でバックテスト
    for ticker, ticker_name, sector in GRID_SEARCH_TICKERS:
        print(f"[TICKER] {ticker} - {ticker_name}")
        print(f"  [PARAM] {param['param_name']}")
        
        result = run_single_backtest(ticker, ticker_name, param)
        
        if result['status'] == 'SUCCESS':
            print(f"    Total Trades: {result['total_trades']}")
            print(f"    Profit Factor: {result['profit_factor']:.2f}")
            print(f"    Win Rate: {result['win_rate']*100:.1f}%")
            print(f"    Sharpe Ratio (Annual): {result['sharpe_ratio_annual']:.2f}")
            print(f"    Max Drawdown: {result['max_drawdown_pct']*100:.2f}%")
            
            # PF警告
            if result['profit_factor'] > PF_DISQUALIFICATION_THRESHOLD:
                print(f"    [DISQUALIFIED] PF > {PF_DISQUALIFICATION_THRESHOLD}（過学習確定）")
            elif result['profit_factor'] > PF_WARNING_THRESHOLD:
                print(f"    [WARNING] PF > {PF_WARNING_THRESHOLD}（過学習注意）")
            
            all_results.append(result)
            param_results.append(result)
        else:
            print(f"    [FAILED] {result.get('error', 'Unknown error')}")
        
        print()
    
    # 成功基準チェック
    if param_results:
        summary = check_success_criteria(param_results, param['param_name'])
        
        print(f"\n[PARAM SUMMARY] {param['param_name']}")
        print(f"  平均PF: {summary['avg_pf']:.2f} ({'PASS' if summary['criteria_pass']['avg_pf'] else 'FAIL'}) [Step 1: {BEST_PARAM_STEP1['step1_avg_pf']:.2f}]")
        print(f"  平均Win Rate: {summary['avg_win_rate']*100:.1f}% ({'PASS' if summary['criteria_pass']['avg_win_rate'] else 'FAIL'}) [Step 1: {BEST_PARAM_STEP1['step1_avg_win_rate']*100:.1f}%]")
        print(f"  平均取引数: {summary['avg_trades']:.1f} ({'PASS' if summary['criteria_pass']['min_trades'] else 'FAIL'}) [Step 1: {BEST_PARAM_STEP1['step1_avg_trades']:.1f}]")
        print(f"  PF低下率: {summary['pf_degradation']*100:.1f}% ({'PASS' if summary['criteria_pass']['pf_degradation'] else 'FAIL'}) [目標: < 50%]")
        print(f"  平均Sharpe年率: {summary['avg_sharpe_annual']:.2f} ({'PASS' if summary['criteria_pass']['sharpe'] else 'FAIL'})")
        print(f"  平均Max DD: {summary['avg_max_dd']*100:.2f}% ({'PASS' if summary['criteria_pass']['max_dd'] else 'FAIL'})")
        print(f"  総合判定: {summary['overall_status']} ({summary['pass_count']}/6基準PASS)")
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("Option A Step 2 Out-of-Sample検証結果サマリー")
    print("=" * 80)
    
    if param_results:
        print(f"\n[パラメータ] {param['param_name']}")
        print(f"  平均PF: {summary['avg_pf']:.2f} (Step 1比較: {(1-summary['pf_degradation'])*100:+.1f}%)")
        print(f"  平均Win Rate: {summary['avg_win_rate']*100:.1f}%")
        print(f"  平均取引数: {summary['avg_trades']:.1f}")
        print(f"  平均Sharpe年率: {summary['avg_sharpe_annual']:.2f}")
        print(f"  平均Max DD: {summary['avg_max_dd']*100:.2f}%")
        print(f"  総合判定: {summary['overall_status']}")
        
        # Step 3移行判断
        if summary['overall_status'] == 'PASS':
            print(f"\n[SUCCESS] Step 2成功！Step 3（2025年1-3月真のOut-of-Sample）へ移行推奨")
        else:
            print(f"\n[FAIL] Step 2失敗。PASS判定基準未達、パラメータ再検討必要")
            print(f"  推奨対応: Step 1のPASS判定4個から次候補選択（1_15_0.5等）")
    else:
        print("\n[ERROR] 全銘柄でバックテスト失敗、データ取得・設定確認必要")
    
    # CSV出力
    output_dir = project_root / "output" / "grid_search_2024"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = output_dir / f"grid_search_2024_h2_{timestamp}.csv"
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] CSV出力完了: {csv_filename}")
    
    # JSON出力（サマリー）
    json_filename = output_dir / f"grid_search_summary_2024_h2_{timestamp}.json"
    
    summary_data = {
        'validation_date': datetime.now().isoformat(),
        'step': 'Option A Step 2',
        'period': f"{GRID_SEARCH_START_DATE} ~ {GRID_SEARCH_END_DATE}",
        'period_type': 'Out-of-Sample',
        'fixed_param': {
            'param_name': param['param_name'],
            'min_hold_days': param['min_hold_days'],
            'max_hold_days': param['max_hold_days'],
            'confidence_threshold': param['confidence_threshold']
        },
        'step1_reference': {
            'avg_pf': BEST_PARAM_STEP1['step1_avg_pf'],
            'avg_win_rate': BEST_PARAM_STEP1['step1_avg_win_rate'],
            'avg_trades': BEST_PARAM_STEP1['step1_avg_trades']
        },
        'step2_results': {
            **summary,
            'criteria_pass': {k: int(v) for k, v in summary['criteria_pass'].items()}  # bool→int変換
        },
        'success_criteria': GRID_SEARCH_SUCCESS_CRITERIA,
        'ticker_results': all_results
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] JSON出力完了: {json_filename}")
    
    print("\n" + "=" * 80)
    print("Option A Step 2完了")
    if summary['overall_status'] == 'PASS':
        print("次のステップ: validate_exit_grid_search_2025_q1.py（2025年1-3月真のOut-of-Sample検証）")
    else:
        print("次のステップ: Step 1 PASS判定4個から次候補選択、Step 2再実行")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
