"""
Option A Step 3: 2025年1-3月真のOut-of-Sample検証スクリプト

2025年1-3月データでStep 1最優秀パラメータ（1_30_0.3）の真の汎化性能検証。
In-Sample（Step 1: 2024年1-6月）で最適化されたパラメータが、
完全未見データ（Step 3: 2025年1-3月）でも性能維持できるか最終確認。
これによりリアルトレード移行判断を行う。

主な機能:
- 真のOut-of-Sample検証（2025年1-3月、3ヶ月）
- パラメータ固定（Step 1最優秀: min_hold=1, max_hold=30, confidence=0.3）
- GCエントリー固定
- 推奨3銘柄（9983.T、6501.T、6758.T）検証
- Step 1/Step 2比較分析（PF推移、Win Rate推移、取引数変化）
- Phase 6既存結果（Priority 1: PF 0.96-1.09）との比較
- CSV+JSON統一出力（grid_search_2025_q1_*.csv、grid_search_summary_2025_q1_*.json）

統合コンポーネント:
- GCStrategyWithExit
- TrendFollowingExit（固定: 1/30/0.3）
- data_fetcher経由でデータ取得

セーフティ機能/注意事項:
- 期間固定（2025-01-01 ~ 2025-03-31、3ヶ月）
- エントリー固定（GCStrategy、過学習回避）
- ルックアヘッドバイアス防止（copilot-instructions.md準拠）
- データ取得失敗時はスキップ（フォールバック禁止）
- 成功基準緩和（3ヶ月想定: 取引数 > 5、PF > 2.0維持）

リアルトレード移行判断:
- Step 3成功（PF > 2.0維持）→ Phase 6実施、リアルトレード準備開始
- Step 3失敗（PF < 2.0）→ 次候補選択（1_15_0.5等）、Option A再実行

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


# Option A Step 3設定
GRID_SEARCH_START_DATE = "2025-01-01"  # 真のOut-of-Sample開始日
GRID_SEARCH_END_DATE = "2025-03-31"    # 真のOut-of-Sample終了日（3ヶ月）
WARMUP_DAYS = 150  # ウォームアップ期間（Option A-2暦日拡大方式）

# Step 1最優秀パラメータ固定
BEST_PARAM_STEP1 = {
    'min_hold_days': 1,
    'max_hold_days': 30,
    'confidence_threshold': 0.3,
    'param_name': '1_30_0.3',
    'step1_avg_pf': 54.00,  # Step 1結果（参照値、2024年1-6月）
    'step1_avg_win_rate': 0.667,  # Step 1結果（参照値）
    'step1_avg_trades': 7.3  # Step 1結果（参照値）
}

# Step 2参照値
BEST_PARAM_STEP2 = {
    'step2_avg_pf': 2.43,  # Step 2結果（参照値、2024年7-12月）
    'step2_avg_win_rate': 0.564,  # Step 2結果（参照値）
    'step2_avg_trades': 8.0,  # Step 2結果（参照値）
    'step2_avg_sharpe': 3.41,  # Step 2結果（参照値）
    'step2_avg_max_dd': 0.0006  # Step 2結果（参照値、0.06%）
}

# Phase 6既存結果（Priority 1、2025年1-3月）
PHASE6_PRIORITY1_RESULTS = {
    'aggressive_pf': 1.09,  # Aggressive(1/30/0.4) ← 2023-2024過学習
    'moderate_pf': 1.08,    # Moderate(3/60/0.5) ← 2023-2024過学習
    'conservative_pf': 0.96 # Conservative(5/90/0.6) ← 2023-2024過学習
}

# Option A対象3銘柄（Phase 6推奨銘柄）
GRID_SEARCH_TICKERS = [
    ("9983.T", "ファーストリテイリング", "小売（衣料品）"),
    ("6501.T", "日立製作所", "製造業（電機）"),
    ("6758.T", "ソニーグループ", "製造業（電機）")
]

# Option A Step 3成功基準（3ヶ月想定緩和）
GRID_SEARCH_SUCCESS_CRITERIA = {
    'avg_pf': 2.0,              # 平均PF > 2.0維持（必須、Step 2と同じ）
    'avg_win_rate': 0.40,       # 平均Win Rate > 40%維持
    'min_trades_per_ticker': 5, # 取引数/銘柄 > 5（3ヶ月想定緩和、Step 2: 10から引き下げ）
    'max_drawdown': 0.15,       # Max Drawdown < 15%
    'sharpe_ratio': 1.0,        # Sharpe Ratio年率 > 1.0
    'phase6_priority1_beat': True  # Phase 6 Priority 1（PF 0.96-1.09）を大幅上回る
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
        print(f"  [ERROR] {ticker} ({ticker_name}): {str(e)}")
        return {
            'ticker': ticker,
            'ticker_name': ticker_name,
            'param_name': param['param_name'],
            'min_hold_days': param['min_hold_days'],
            'max_hold_days': param['max_hold_days'],
            'confidence_threshold': param['confidence_threshold'],
            'total_trades': 0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio_annual': 0.0,
            'max_drawdown_pct': 0.0,
            'status': 'FAILED',
            'error': str(e)
        }


def check_success_criteria(results: list) -> dict:
    """
    成功基準チェック
    
    Args:
        results: 全銘柄の結果リスト
    
    Returns:
        成功基準チェック結果dict
    """
    success_results = [r for r in results if r['status'] == 'SUCCESS']
    
    if not success_results:
        return {
            'overall_status': 'FAIL',
            'reason': '全銘柄でバックテスト失敗',
            'avg_pf': 0.0,
            'avg_win_rate': 0.0,
            'avg_trades': 0.0,
            'avg_sharpe_annual': 0.0,
            'avg_max_dd': 0.0,
            'criteria_pass': {},
            'phase6_comparison': {}
        }
    
    # 平均値計算
    avg_pf = np.mean([r['profit_factor'] for r in success_results])
    avg_win_rate = np.mean([r['win_rate'] for r in success_results])
    avg_trades = np.mean([r['total_trades'] for r in success_results])
    avg_sharpe_annual = np.mean([r['sharpe_ratio_annual'] for r in success_results])
    avg_max_dd = np.mean([r['max_drawdown_pct'] for r in success_results])
    
    # Step 1/Step 2比較（PF推移）
    step1_pf = BEST_PARAM_STEP1['step1_avg_pf']
    step2_pf = BEST_PARAM_STEP2['step2_avg_pf']
    step1_to_step3_degradation = (step1_pf - avg_pf) / step1_pf if avg_pf > 0 else 1.0
    step2_to_step3_degradation = (step2_pf - avg_pf) / step2_pf if avg_pf > 0 else 1.0
    
    # Phase 6 Priority 1比較
    phase6_best_pf = max(PHASE6_PRIORITY1_RESULTS.values())  # 1.09
    phase6_comparison = {
        'phase6_best_pf': phase6_best_pf,
        'step3_avg_pf': avg_pf,
        'improvement': ((avg_pf - phase6_best_pf) / phase6_best_pf * 100) if phase6_best_pf > 0 else 0.0,
        'beats_phase6': avg_pf > phase6_best_pf
    }
    
    # 成功基準チェック
    criteria_pass = {
        'avg_pf': avg_pf > GRID_SEARCH_SUCCESS_CRITERIA['avg_pf'],
        'avg_win_rate': avg_win_rate > GRID_SEARCH_SUCCESS_CRITERIA['avg_win_rate'],
        'min_trades': avg_trades > GRID_SEARCH_SUCCESS_CRITERIA['min_trades_per_ticker'],
        'max_dd': avg_max_dd < GRID_SEARCH_SUCCESS_CRITERIA['max_drawdown'],
        'sharpe': avg_sharpe_annual > GRID_SEARCH_SUCCESS_CRITERIA['sharpe_ratio'],
        'phase6_beat': phase6_comparison['beats_phase6']
    }
    
    # 総合判定（6基準中4つ以上でPASS、avg_pf必須）
    pass_count = sum(criteria_pass.values())
    overall_status = 'PASS' if pass_count >= 4 and criteria_pass['avg_pf'] else 'FAIL'
    
    return {
        'overall_status': overall_status,
        'pass_count': pass_count,
        'avg_pf': avg_pf,
        'avg_win_rate': avg_win_rate,
        'avg_trades': avg_trades,
        'avg_sharpe_annual': avg_sharpe_annual,
        'avg_max_dd': avg_max_dd,
        'step1_to_step3_degradation': step1_to_step3_degradation,
        'step2_to_step3_degradation': step2_to_step3_degradation,
        'criteria_pass': criteria_pass,
        'phase6_comparison': phase6_comparison
    }


def save_results(results: list, summary: dict, output_dir: Path):
    """
    結果保存（CSV + JSON）
    
    Args:
        results: 全銘柄の結果リスト
        summary: サマリーdict
        output_dir: 出力ディレクトリ
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV出力
    results_df = pd.DataFrame(results)
    csv_path = output_dir / f"grid_search_2025_q1_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] CSV出力: {csv_path}")
    
    # JSON出力（pandas/numpy型をJSON serializable型に変換）
    # 全データをDataFrame経由で標準型に変換
    summary_clean = {k: (v.item() if hasattr(v, 'item') else v) for k, v in summary.items()}
    
    # criteria_pass、phase6_comparisonの再帰的変換
    if 'criteria_pass' in summary_clean:
        summary_clean['criteria_pass'] = {
            k: (v.item() if hasattr(v, 'item') else v) 
            for k, v in summary_clean['criteria_pass'].items()
        }
    if 'phase6_comparison' in summary_clean:
        summary_clean['phase6_comparison'] = {
            k: (v.item() if hasattr(v, 'item') else v) 
            for k, v in summary_clean['phase6_comparison'].items()
        }
    
    results_clean = []
    for r in results:
        r_clean = {}
        for k, v in r.items():
            if hasattr(v, 'item'):  # numpy scalar
                r_clean[k] = v.item()
            elif isinstance(v, (np.integer, np.floating)):
                r_clean[k] = float(v)
            else:
                r_clean[k] = v
        results_clean.append(r_clean)
    
    json_data = {
        'execution_date': timestamp,
        'period': f"{GRID_SEARCH_START_DATE} ~ {GRID_SEARCH_END_DATE}",
        'parameter': BEST_PARAM_STEP1['param_name'],
        'tickers': [t[0] for t in GRID_SEARCH_TICKERS],
        'summary': summary_clean,
        'detailed_results': results_clean
    }
    json_path = output_dir / f"grid_search_summary_2025_q1_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON出力: {json_path}")


def main():
    """メインエントリーポイント"""
    print("\n" + "=" * 80)
    print("Option A Step 3: 2025年1-3月真のOut-of-Sample検証")
    print("=" * 80)
    print(f"期間: {GRID_SEARCH_START_DATE} ~ {GRID_SEARCH_END_DATE}")
    print(f"パラメータ固定: {BEST_PARAM_STEP1['param_name']}")
    print(f"  min_hold_days={BEST_PARAM_STEP1['min_hold_days']}")
    print(f"  max_hold_days={BEST_PARAM_STEP1['max_hold_days']}")
    print(f"  confidence_threshold={BEST_PARAM_STEP1['confidence_threshold']}")
    print(f"対象銘柄: {len(GRID_SEARCH_TICKERS)}銘柄")
    print("=" * 80 + "\n")
    
    # 出力ディレクトリ作成
    output_dir = project_root / "output" / "exit_strategy_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # バックテスト実行
    results = []
    for ticker, ticker_name, sector in GRID_SEARCH_TICKERS:
        print(f"\n[銘柄] {ticker} ({ticker_name}, {sector})")
        result = run_single_backtest(
            ticker=ticker,
            ticker_name=ticker_name,
            param=BEST_PARAM_STEP1
        )
        results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"  取引数: {result['total_trades']}")
            print(f"  PF: {result['profit_factor']:.2f}")
            print(f"  Win Rate: {result['win_rate']*100:.1f}%")
            print(f"  Sharpe年率: {result['sharpe_ratio_annual']:.2f}")
            print(f"  Max DD: {result['max_drawdown_pct']*100:.2f}%")
        else:
            print(f"  [FAILED] {result.get('error', 'Unknown error')}")
    
    # 成功基準チェック
    print("\n" + "=" * 80)
    print("成功基準チェック")
    print("=" * 80)
    
    summary = check_success_criteria(results)
    
    # Step 1/Step 2比較出力
    print(f"\n[Step 1 → Step 2 → Step 3 推移]")
    print(f"  Step 1（2024年1-6月）: 平均PF={BEST_PARAM_STEP1['step1_avg_pf']:.2f}, Win Rate={BEST_PARAM_STEP1['step1_avg_win_rate']*100:.1f}%, 取引数={BEST_PARAM_STEP1['step1_avg_trades']:.1f}")
    print(f"  Step 2（2024年7-12月）: 平均PF={BEST_PARAM_STEP2['step2_avg_pf']:.2f}, Win Rate={BEST_PARAM_STEP2['step2_avg_win_rate']*100:.1f}%, 取引数={BEST_PARAM_STEP2['step2_avg_trades']:.1f}")
    print(f"  Step 3（2025年1-3月）: 平均PF={summary['avg_pf']:.2f}, Win Rate={summary['avg_win_rate']*100:.1f}%, 取引数={summary['avg_trades']:.1f}")
    print(f"\n[PF推移分析]")
    print(f"  Step 1 → Step 3: {summary['step1_to_step3_degradation']*100:.1f}%低下")
    print(f"  Step 2 → Step 3: {summary['step2_to_step3_degradation']*100:.1f}%低下")
    
    # Phase 6比較出力
    print(f"\n[Phase 6 Priority 1比較（2025年1-3月同期間）]")
    print(f"  Phase 6最優秀: PF={summary['phase6_comparison']['phase6_best_pf']:.2f}（2023-2024過学習パラメータ）")
    print(f"  Option A Step 3: PF={summary['phase6_comparison']['step3_avg_pf']:.2f}（2024年最適化パラメータ）")
    print(f"  改善率: {summary['phase6_comparison']['improvement']:+.1f}%")
    
    # 成功基準詳細
    print(f"\n[パラメータ] {BEST_PARAM_STEP1['param_name']}")
    print(f"  平均PF: {summary['avg_pf']:.2f} ({'PASS' if summary['criteria_pass']['avg_pf'] else 'FAIL'}) [目標: > {GRID_SEARCH_SUCCESS_CRITERIA['avg_pf']:.1f}]")
    print(f"  平均Win Rate: {summary['avg_win_rate']*100:.1f}% ({'PASS' if summary['criteria_pass']['avg_win_rate'] else 'FAIL'}) [目標: > {GRID_SEARCH_SUCCESS_CRITERIA['avg_win_rate']*100:.0f}%]")
    print(f"  平均取引数: {summary['avg_trades']:.1f} ({'PASS' if summary['criteria_pass']['min_trades'] else 'FAIL'}) [目標: > {GRID_SEARCH_SUCCESS_CRITERIA['min_trades_per_ticker']}]")
    print(f"  平均Sharpe年率: {summary['avg_sharpe_annual']:.2f} ({'PASS' if summary['criteria_pass']['sharpe'] else 'FAIL'}) [目標: > {GRID_SEARCH_SUCCESS_CRITERIA['sharpe_ratio']:.1f}]")
    print(f"  平均Max DD: {summary['avg_max_dd']*100:.2f}% ({'PASS' if summary['criteria_pass']['max_dd'] else 'FAIL'}) [目標: < {GRID_SEARCH_SUCCESS_CRITERIA['max_drawdown']*100:.0f}%]")
    print(f"  Phase 6超え: {'PASS' if summary['criteria_pass']['phase6_beat'] else 'FAIL'} [改善率: {summary['phase6_comparison']['improvement']:+.1f}%]")
    
    # 総合判定
    print(f"\n[総合判定] {summary['overall_status']} ({summary['pass_count']}/6基準PASS)")
    
    if summary['overall_status'] == 'PASS':
        print("\n" + "=" * 80)
        print("[SUCCESS] Step 3成功！リアルトレード移行準備推奨")
        print("=" * 80)
        print("[次のステップ]")
        print("  1. Phase 6実施: PaperBroker統合テスト（MainSystemController経由）")
        print("  2. kabu STATION API統合準備（一旦保留）")
        print("  3. 実資金投入準備（リスク管理、初期資金設定）")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("[FAIL] Step 3失敗、次候補選択推奨")
        print("=" * 80)
        print("[次のステップ]")
        print("  1. Step 1 PASS判定4個から次候補選択（1_15_0.5、1_15_0.3、3_15_0.5、5_15_0.5）")
        print("  2. Step 2/Step 3再実行（次候補で検証）")
        print("  3. Option B検討（ウォークフォワード分析、パラメータ安定性重視）")
        print("=" * 80)
    
    # 結果保存
    save_results(results, summary, output_dir)
    
    print("\n" + "=" * 80)
    print("検証完了")
    print("=" * 80 + "\n")
    
    return summary['overall_status']


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status == 'PASS' else 1)
