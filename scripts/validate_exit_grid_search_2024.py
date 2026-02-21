"""
Option A Step 1: 2024年1-6月グリッドサーチスクリプト

2024年1-6月データで125組み合わせのTrendFollowingパラメータグリッドサーチを実行。
2023-2024過学習パラメータを完全排除し、2024年市場環境に適合する最優秀パラメータを特定。

主な機能:
- In-Sample検証（2024年1-6月、6ヶ月）
- TrendFollowing 125組み合わせグリッドサーチ（5×5×5）
- GCエントリー固定
- 推奨3銘柄（9983.T、6501.T、6758.T）並行検証
- 最優秀パラメータ特定（平均PF、Win Rate、取引数、Sharpe年率、Max DD）
- CSV+JSON統一出力（grid_search_2024_h1_*.csv、grid_search_summary_2024_h1_*.json）

統合コンポーネント:
- GCStrategyWithExit
- TrendFollowingExit（125組み合わせ）
- data_fetcher経由でデータ取得

セーフティ機能/注意事項:
- 期間固定（2024-01-01 ~ 2024-06-30、6ヶ月）
- エントリー固定（GCStrategy、過学習回避）
- ルックアヘッドバイアス防止（copilot-instructions.md準拠）
- データ取得失敗時はスキップ（フォールバック禁止）
- PF上限制約（PF > 50は警告、PF > 100は失格）

Option A Step 2/3:
- Step 2: 2024年7-12月Out-of-Sample検証（validate_exit_grid_search_2024_h2.py）
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
from itertools import product

# プロジェクトルートをパス追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.gc_strategy_with_exit import GCStrategyWithExit
from strategies.exit_strategies.trend_following_exit import TrendFollowingExit
from data_fetcher import get_parameters_and_data


# Option A Step 1設定
GRID_SEARCH_START_DATE = "2024-01-01"  # In-Sample開始日
GRID_SEARCH_END_DATE = "2024-06-30"    # In-Sample終了日（6ヶ月）
WARMUP_DAYS = 150  # ウォームアップ期間（Option A-2暦日拡大方式）

# パラメータグリッド（125組み合わせ）
PARAM_GRID_2024 = {
    'min_hold_days': [1, 3, 5, 7, 10],
    'max_hold_days': [15, 30, 45, 60, 90],
    'confidence_threshold': [0.3, 0.4, 0.5, 0.6, 0.7]
}

# Option A対象3銘柄（Phase 6推奨銘柄）
GRID_SEARCH_TICKERS = [
    ("9983.T", "ファーストリテイリング", "小売（衣料品）"),
    ("6501.T", "日立製作所", "製造業（電機）"),
    ("6758.T", "ソニーグループ", "製造業（電機）")
]

# Option A Step 1成功基準
GRID_SEARCH_SUCCESS_CRITERIA = {
    'avg_pf': 2.0,              # 平均PF > 2.0
    'avg_win_rate': 0.40,       # 平均Win Rate > 40%（現実的基準）
    'min_trades_per_ticker': 15, # 取引数/銘柄 > 15（6ヶ月想定）
    'pf_std_ratio': 0.50,       # PF標準偏差 < 平均の50%
    'max_drawdown': 0.15,       # Max Drawdown < 15%
    'sharpe_ratio': 2.0,        # Sharpe Ratio年率 > 2.0
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
            'max_drawdown_pct': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0
        }
    
    # 取引数（Profit_Loss != 0の行のみカウント、0は取引なし）
    trades = results_df[results_df['Profit_Loss'] != 0].copy()
    total_trades = len(trades)
    
    if total_trades == 0:
        return {
            'total_trades': 0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio_annual': 0.0,
            'max_drawdown_pct': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0
        }
    
    # PF・Win Rate
    profits = trades[trades['Profit_Loss'] > 0]['Profit_Loss']
    losses = trades[trades['Profit_Loss'] < 0]['Profit_Loss'].abs()
    
    total_profit = profits.sum() if len(profits) > 0 else 0.0
    total_loss = losses.sum() if len(losses) > 0 else 0.0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
    win_rate = len(profits) / total_trades if total_trades > 0 else 0.0
    
    # Sharpe Ratio年率換算
    if len(trades) >= 2:
        returns = trades['Profit_Loss'].values
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return > 0:
            sharpe_ratio_per_trade = mean_return / std_return
            # 年率換算（6ヶ月想定、取引数ベース）
            trades_per_year = total_trades * 2  # 6ヶ月→12ヶ月換算
            sharpe_ratio_annual = sharpe_ratio_per_trade * np.sqrt(trades_per_year)
        else:
            sharpe_ratio_annual = 0.0
    else:
        sharpe_ratio_annual = 0.0
    
    # Max Drawdown
    cumulative_pl = trades['Profit_Loss'].cumsum()
    running_max = cumulative_pl.cummax()
    drawdown = running_max - cumulative_pl
    max_drawdown = drawdown.max()
    
    # 初期資金仮定（100万円）
    initial_capital = 1000000
    max_drawdown_pct = max_drawdown / initial_capital if initial_capital > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'sharpe_ratio_annual': sharpe_ratio_annual,
        'max_drawdown_pct': max_drawdown_pct,
        'total_profit': total_profit,
        'total_loss': total_loss
    }


def validate_grid_search_on_ticker_param(
    ticker: str,
    company_name: str,
    sector: str,
    param_combo: dict,
    param_name: str
) -> dict:
    """
    単一銘柄×単一パラメータ組み合わせでグリッドサーチ検証
    
    Args:
        ticker: ティッカーシンボル
        company_name: 会社名
        sector: 業種
        param_combo: パラメータ組み合わせ（min_hold_days、max_hold_days、confidence_threshold）
        param_name: パラメータ組み合わせ名（例：1_15_0.3）
    
    Returns:
        検証結果dict
    """
    print(f"\n[TICKER] {ticker} - {company_name}")
    print(f"  [PARAM] {param_name}")
    print(f"    min_hold={param_combo['min_hold_days']}, max_hold={param_combo['max_hold_days']}, confidence={param_combo['confidence_threshold']}")
    
    try:
        # データ取得
        _, _, _, stock_data, index_data = get_parameters_and_data(
            ticker=ticker,
            start_date=GRID_SEARCH_START_DATE,
            end_date=GRID_SEARCH_END_DATE,
            warmup_days=WARMUP_DAYS
        )
        
        if stock_data is None or stock_data.empty:
            print(f"    [SKIP] データ取得失敗: {ticker}")
            return None
        
        # TrendFollowingExitインスタンス生成
        exit_strategy = TrendFollowingExit(
            min_hold_days=param_combo['min_hold_days'],
            max_hold_days=param_combo['max_hold_days'],
            confidence_threshold=param_combo['confidence_threshold']
        )
        
        # GCStrategyWithExit統合（正しい引数: data=stock_dataのみ）
        strategy = GCStrategyWithExit(
            data=stock_data,
            exit_strategy=exit_strategy,
            ticker=ticker
        )
        
        # バックテスト実行（2024年1-6月期間）
        results_df = strategy.backtest(
            trading_start_date=pd.Timestamp(GRID_SEARCH_START_DATE),
            trading_end_date=pd.Timestamp(GRID_SEARCH_END_DATE)
        )
        
        # パフォーマンス指標計算
        metrics = calculate_performance_metrics(results_df)
        
        # 結果出力
        print(f"    Total Trades: {metrics['total_trades']}")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"    Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"    Sharpe Ratio (Annual): {metrics['sharpe_ratio_annual']:.2f}")
        print(f"    Max Drawdown: {metrics['max_drawdown_pct']*100:.2f}%")
        
        # PF上限制約チェック
        if metrics['profit_factor'] > PF_DISQUALIFICATION_THRESHOLD:
            print(f"    [DISQUALIFICATION] PF > {PF_DISQUALIFICATION_THRESHOLD}（過学習疑い、失格）")
        elif metrics['profit_factor'] > PF_WARNING_THRESHOLD:
            print(f"    [WARNING] PF > {PF_WARNING_THRESHOLD}（過学習注意）")
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'param_name': param_name,
            'min_hold_days': param_combo['min_hold_days'],
            'max_hold_days': param_combo['max_hold_days'],
            'confidence_threshold': param_combo['confidence_threshold'],
            'total_trades': metrics['total_trades'],
            'profit_factor': metrics['profit_factor'],
            'win_rate': metrics['win_rate'],
            'sharpe_ratio_annual': metrics['sharpe_ratio_annual'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'total_profit': metrics['total_profit'],
            'total_loss': metrics['total_loss']
        }
        
    except Exception as e:
        print(f"    [ERROR] 検証失敗: {str(e)}")
        return None


def check_success_criteria(param_results: list, param_name: str) -> dict:
    """
    パラメータ組み合わせごとの成功基準チェック
    
    Args:
        param_results: パラメータ組み合わせの検証結果リスト
        param_name: パラメータ組み合わせ名
    
    Returns:
        成功基準チェック結果dict
    """
    if not param_results:
        return {
            'param_name': param_name,
            'avg_pf': 0.0,
            'avg_win_rate': 0.0,
            'avg_trades': 0.0,
            'avg_sharpe_annual': 0.0,
            'avg_max_dd': 0.0,
            'criteria_pass': {},
            'overall_status': 'FAIL'
        }
    
    # 平均値計算（numpy型→Python型変換でJSON出力対応）
    pf_values = [r['profit_factor'] for r in param_results if r['profit_factor'] != float('inf')]
    avg_pf = float(np.mean(pf_values)) if pf_values else 0.0
    avg_win_rate = float(np.mean([r['win_rate'] for r in param_results]))
    avg_trades = float(np.mean([r['total_trades'] for r in param_results]))
    avg_sharpe_annual = float(np.mean([r['sharpe_ratio_annual'] for r in param_results]))
    avg_max_dd = float(np.mean([r['max_drawdown_pct'] for r in param_results]))
    
    # 成功基準チェック
    criteria_pass = {
        'avg_pf': avg_pf >= GRID_SEARCH_SUCCESS_CRITERIA['avg_pf'],
        'avg_win_rate': avg_win_rate >= GRID_SEARCH_SUCCESS_CRITERIA['avg_win_rate'],
        'min_trades': all(r['total_trades'] >= GRID_SEARCH_SUCCESS_CRITERIA['min_trades_per_ticker'] for r in param_results),
        'pf_std': np.std(pf_values) < avg_pf * GRID_SEARCH_SUCCESS_CRITERIA['pf_std_ratio'] if pf_values else False,
        'max_dd': avg_max_dd < GRID_SEARCH_SUCCESS_CRITERIA['max_drawdown'],
        'sharpe': avg_sharpe_annual >= GRID_SEARCH_SUCCESS_CRITERIA['sharpe_ratio']
    }
    
    # 総合判定（numpy型→Python型変換でJSON出力対応）
    pass_count_int = int(sum(criteria_pass.values()))
    overall_status = 'PASS' if pass_count_int >= 4 else 'FAIL'
    
    return {
        'param_name': param_name,
        'avg_pf': avg_pf,
        'avg_win_rate': avg_win_rate,
        'avg_trades': avg_trades,
        'avg_sharpe_annual': avg_sharpe_annual,
        'avg_max_dd': avg_max_dd,
        'criteria_pass': criteria_pass,
        'pass_count': pass_count_int,  # Python int型
        'overall_status': overall_status
    }


def main():
    """メインエントリーポイント - Option A Step 1グリッドサーチ"""
    
    print("\n" + "=" * 80)
    print("Option A Step 1: 2024年1-6月グリッドサーチ（TrendFollowing 125組み合わせ）")
    print("=" * 80)
    print(f"検証期間: {GRID_SEARCH_START_DATE} ~ {GRID_SEARCH_END_DATE}（6ヶ月、In-Sample）")
    print(f"対象銘柄: {len(GRID_SEARCH_TICKERS)}銘柄（Phase 6推奨3銘柄）")
    print(f"パラメータ空間: {len(PARAM_GRID_2024['min_hold_days'])}×{len(PARAM_GRID_2024['max_hold_days'])}×{len(PARAM_GRID_2024['confidence_threshold'])} = 125組み合わせ")
    print(f"総検証数: 125組み合わせ × 3銘柄 = 375検証")
    print("=" * 80 + "\n")
    
    # パラメータ組み合わせ生成
    param_combinations = []
    for min_hold, max_hold, confidence in product(
        PARAM_GRID_2024['min_hold_days'],
        PARAM_GRID_2024['max_hold_days'],
        PARAM_GRID_2024['confidence_threshold']
    ):
        # min_hold < max_holdのみ有効
        if min_hold < max_hold:
            param_name = f"{min_hold}_{max_hold}_{confidence}"
            param_combinations.append({
                'name': param_name,
                'min_hold_days': min_hold,
                'max_hold_days': max_hold,
                'confidence_threshold': confidence
            })
    
    print(f"[INFO] 有効パラメータ組み合わせ: {len(param_combinations)}個（min_hold < max_hold制約適用）\n")
    
    # Quick版: 最初の10パラメータのみ実行（動作確認用）
    # Full版にするには下の行をコメントアウト
    # param_combinations = param_combinations[:10]  # ← Full版：コメントアウト
    # print(f"[QUICK MODE] 最初の10パラメータのみ実行（動作確認）\n")  # ← Full版：コメントアウト
    print(f"[FULL MODE] 全{len(param_combinations)}パラメータ実行（Option A Step 1）\n")
    
    # 全検証結果格納
    all_results = []
    param_summary = []
    
    # パラメータ組み合わせごとに検証
    for idx, param_combo in enumerate(param_combinations, 1):
        print(f"\n{'=' * 80}")
        print(f"[PARAM {idx}/{len(param_combinations)}] {param_combo['name']}")
        print(f"  min_hold={param_combo['min_hold_days']}, max_hold={param_combo['max_hold_days']}, confidence={param_combo['confidence_threshold']}")
        print("=" * 80)
        
        param_results = []
        
        # 3銘柄で検証
        for ticker, company_name, sector in GRID_SEARCH_TICKERS:
            result = validate_grid_search_on_ticker_param(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                param_combo=param_combo,
                param_name=param_combo['name']
            )
            
            if result is not None:
                all_results.append(result)
                param_results.append(result)
        
        # パラメータ組み合わせごとの成功基準チェック
        if param_results:
            summary = check_success_criteria(param_results, param_combo['name'])
            param_summary.append(summary)
            
            print(f"\n  [PARAM SUMMARY] {param_combo['name']}")
            print(f"    平均PF: {summary['avg_pf']:.2f} ({'PASS' if summary['criteria_pass']['avg_pf'] else 'FAIL'})")
            print(f"    平均Win Rate: {summary['avg_win_rate']*100:.1f}% ({'PASS' if summary['criteria_pass']['avg_win_rate'] else 'FAIL'})")
            print(f"    平均取引数: {summary['avg_trades']:.1f} ({'PASS' if summary['criteria_pass']['min_trades'] else 'FAIL'})")
            print(f"    平均Sharpe年率: {summary['avg_sharpe_annual']:.2f} ({'PASS' if summary['criteria_pass']['sharpe'] else 'FAIL'})")
            print(f"    平均Max DD: {summary['avg_max_dd']*100:.2f}% ({'PASS' if summary['criteria_pass']['max_dd'] else 'FAIL'})")
            print(f"    総合判定: {summary['overall_status']} ({summary['pass_count']}/6基準PASS)")
    
    # 最優秀パラメータ特定
    print("\n" + "=" * 80)
    print("Option A Step 1 グリッドサーチ結果サマリー")
    print("=" * 80)
    
    if param_summary:
        # PFでソート
        param_summary_sorted = sorted(param_summary, key=lambda x: x['avg_pf'], reverse=True)
        
        print(f"\n[TOP 10パラメータ組み合わせ（PF順）]")
        for rank, summary in enumerate(param_summary_sorted[:10], 1):
            print(f"  {rank}. {summary['param_name']}")
            print(f"      平均PF={summary['avg_pf']:.2f}, Win Rate={summary['avg_win_rate']*100:.1f}%, 取引数={summary['avg_trades']:.1f}, 判定={summary['overall_status']}")
        
        # 最優秀パラメータ
        best_param = param_summary_sorted[0]
        print(f"\n[最優秀パラメータ] {best_param['param_name']}")
        print(f"  平均PF: {best_param['avg_pf']:.2f}")
        print(f"  平均Win Rate: {best_param['avg_win_rate']*100:.1f}%")
        print(f"  平均取引数: {best_param['avg_trades']:.1f}")
        print(f"  平均Sharpe年率: {best_param['avg_sharpe_annual']:.2f}")
        print(f"  平均Max DD: {best_param['avg_max_dd']*100:.2f}%")
        print(f"  総合判定: {best_param['overall_status']} ({best_param['pass_count']}/6基準PASS)")
        
        # PASS判定のパラメータ数
        pass_params = [s for s in param_summary if s['overall_status'] == 'PASS']
        print(f"\n[PASS判定パラメータ数] {len(pass_params)}/{len(param_summary)}個")
        
        if not pass_params:
            print(f"  [WARNING] PASS判定パラメータなし、成功基準緩和検討")
    
    # CSV出力
    output_dir = project_root / "output" / "grid_search_2024"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = output_dir / f"grid_search_2024_h1_{timestamp}.csv"
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] CSV出力完了: {csv_filename}")
    
    # JSON出力（サマリー）
    json_filename = output_dir / f"grid_search_summary_2024_h1_{timestamp}.json"
    
    # PASS判定パラメータリスト作成（JSON出力用）
    pass_params = [s for s in param_summary if s['overall_status'] == 'PASS']
    
    summary_data = {
        'validation_date': datetime.now().isoformat(),
        'period': f"{GRID_SEARCH_START_DATE} ~ {GRID_SEARCH_END_DATE}",
        'total_param_combinations': len(param_combinations),
        'total_param_tested': len(param_summary),
        'total_validations': len(all_results),
        'pass_count': len(pass_params),
        'pass_rate': float(len(pass_params) / len(param_summary)) if param_summary else 0.0,
        'success_criteria': GRID_SEARCH_SUCCESS_CRITERIA,
        'param_summary': [
            {
                **s,
                'criteria_pass': {k: int(v) for k, v in s['criteria_pass'].items()}  # bool -> int変換
            }
            for s in param_summary
        ],
        'pass_params': [
            {
                **p,
                'criteria_pass': {k: int(v) for k, v in p['criteria_pass'].items()}
            }
            for p in pass_params
        ],
        'best_param': {
            **param_summary_sorted[0],
            'criteria_pass': {k: int(v) for k, v in param_summary_sorted[0]['criteria_pass'].items()}
        } if param_summary_sorted else None
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] JSON出力完了: {json_filename}")
    
    print("\n" + "=" * 80)
    print("Option A Step 1完了")
    print("次のステップ: validate_exit_grid_search_2024_h2.py（2024年7-12月Out-of-Sample検証）")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
