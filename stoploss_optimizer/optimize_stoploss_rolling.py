"""
optimize_stoploss_rolling.py
Phase 2: Rolling Window 検証によるストップロス閾値の時系列安定性検証

使い方:
    # Phase 1 の推奨候補を自動読み込みして実行
    python optimize_stoploss_rolling.py

    # 候補を手動指定して実行
    python optimize_stoploss_rolling.py --candidates -0.05 -0.06

出力:
    results/phase2_rolling/candidate_-X.XX/window1.json ...
    results/phase2_rolling/rolling_analysis_report.md
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# プロジェクトルートをパスに追加
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[1]
OPTIMIZER_DIR = _THIS_FILE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stoploss_optimizer.utils.backtest_runner import run_backtest
from stoploss_optimizer.utils.result_analyzer import collect_results, analyze_rolling_results

# ========================================================
# 設定
# ========================================================

# Rolling Window の定義（訓練: 使用しない / テスト: 評価対象）
# 6ヶ月単位で2022年〜2024年をスライド
WINDOWS = [
    {'name': 'window1', 'test_start': '2022-01-01', 'test_end': '2022-06-30'},
    {'name': 'window2', 'test_start': '2022-07-01', 'test_end': '2022-12-31'},
    {'name': 'window3', 'test_start': '2023-01-01', 'test_end': '2023-06-30'},
    {'name': 'window4', 'test_start': '2023-07-01', 'test_end': '2023-12-31'},
    {'name': 'window5', 'test_start': '2024-01-01', 'test_end': '2024-06-30'},
    {'name': 'window6', 'test_start': '2024-07-01', 'test_end': '2024-12-31'},
]

RESULTS_DIR = OPTIMIZER_DIR / "results" / "phase2_rolling"
PROGRESS_FILE = RESULTS_DIR / "progress.json"

# ========================================================
# ログ設定（後でファイルを追加）
# ========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ========================================================
# 候補の読み込み
# ========================================================

def load_phase1_candidates() -> list[float]:
    """Phase 1 の分析結果から推奨候補を読み込む"""
    phase1_csv = OPTIMIZER_DIR / "results" / "phase1_k_fold" / "k_fold_analysis.csv"

    if not phase1_csv.exists():
        logger.warning(f"Phase 1 結果が見つかりません: {phase1_csv}")
        logger.warning("デフォルト候補 [-0.05, -0.06] を使用します")
        return [-0.05, -0.06]

    df = pd.read_csv(phase1_csv)
    if df.empty:
        return [-0.05, -0.06]

    # reliability_score 上位2件
    top2 = df.sort_values('reliability_score', ascending=False).head(2)
    candidates = top2['threshold'].tolist()
    logger.info(f"Phase 1 から候補を読み込み: {[f'{c*100:.0f}%' for c in candidates]}")
    return candidates


# ========================================================
# 進捗管理
# ========================================================

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def window_key(threshold: float, window: dict) -> str:
    return f"{threshold:.2f}_{window['name']}"


# ========================================================
# 単一 Window の実行
# ========================================================

def run_single_window(threshold: float, window: dict) -> dict:
    logger.info(
        f"Window実行: threshold={threshold*100:.0f}%, "
        f"{window['name']}({window['test_start']}〜{window['test_end']})"
    )

    bt_result = run_backtest(
        start_date=window['test_start'],
        end_date=window['test_end'],
        threshold=threshold
    )

    if not bt_result['success']:
        return {
            'success': False,
            'threshold': threshold,
            'window': window['name'],
            'test_period': f"{window['test_start']}〜{window['test_end']}",
            'error': bt_result['error'],
            'duration_seconds': bt_result.get('duration_seconds', 0)
        }

    metrics = collect_results(bt_result['output_dir'])

    result = {
        'success': metrics['success'],
        'threshold': threshold,
        'window': window['name'],
        'test_period': f"{window['test_start']}〜{window['test_end']}",
        'output_dir': str(bt_result['output_dir']),
        'duration_seconds': bt_result['duration_seconds'],
        **{k: v for k, v in metrics.items() if k != 'success'},
        'error': metrics.get('error')
    }

    if metrics['success']:
        logger.info(
            f"  → PF={metrics['pf']:.3f}, 損益={metrics['pnl']:+,.0f}円, "
            f"勝率={metrics['win_rate']*100:.1f}%"
        )

    return result


# ========================================================
# Phase 2 メイン処理
# ========================================================

def run_phase2(candidates: list[float]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ログファイル追加
    file_handler = logging.FileHandler(
        RESULTS_DIR / "phase2_rolling.log", encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    progress = load_progress()
    total = len(candidates) * len(WINDOWS)
    done = sum(1 for k in progress if progress[k].get('done'))

    logger.info(f"Phase 2 開始: {total}実験中 {done}件済み")
    logger.info(f"候補: {[f'{c*100:.0f}%' for c in candidates]}")
    logger.info(f"Window数: {len(WINDOWS)}期間")

    all_results = {c: [] for c in candidates}
    phase2_start = time.time()

    for threshold in candidates:
        threshold_dir = RESULTS_DIR / f"candidate_{threshold:.2f}"
        threshold_dir.mkdir(exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"候補 {threshold*100:.0f}% の Rolling Window 実験")
        logger.info(f"{'='*60}")

        for window in WINDOWS:
            key = window_key(threshold, window)

            if key in progress and progress[key].get('done'):
                logger.info(f"スキップ（済み）: {key}")
                result_file = threshold_dir / f"{window['name']}.json"
                if result_file.exists():
                    with open(result_file, "r", encoding="utf-8") as f:
                        all_results[threshold].append(json.load(f))
                continue

            result = run_single_window(threshold, window)

            # 保存
            result_file = threshold_dir / f"{window['name']}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            all_results[threshold].append(result)

            progress[key] = {'done': True, 'timestamp': datetime.now().isoformat()}
            save_progress(progress)

            elapsed = time.time() - phase2_start
            completed = sum(len(v) for v in all_results.values())
            if completed > 0:
                avg = elapsed / completed
                remaining = (total - completed) * avg
                logger.info(
                    f"進捗: {completed}/{total}, 残り約{remaining/60:.0f}分"
                )

    # 分析とレポート生成
    rolling_df = analyze_rolling_results(all_results)
    if not rolling_df.empty:
        rolling_df.to_csv(
            RESULTS_DIR / "rolling_analysis.csv",
            index=False, encoding="utf-8-sig"
        )

    _generate_phase2_report(rolling_df, all_results, RESULTS_DIR)

    elapsed = time.time() - phase2_start
    logger.info(f"\nPhase 2 完了! 所要時間: {elapsed/60:.1f}分")

    return rolling_df, all_results


# ========================================================
# レポート生成
# ========================================================

def _generate_phase2_report(rolling_df: pd.DataFrame, all_results: dict, output_dir: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    window_labels = [f"{w['test_start'][:7]}〜{w['test_end'][:7]}" for w in WINDOWS]

    lines = [
        f"# Phase 2: Rolling Window 分析レポート",
        f"",
        f"**生成日時**: {now}  ",
        f"**Window定義**: 6ヶ月単位, {len(WINDOWS)}期間  ",
        f"",
        f"---",
        f"",
        f"## サマリー（安定性順）",
        f"",
    ]

    if not rolling_df.empty:
        lines.append("| ランク | 閾値 | 平均PF | 標準偏差 | 最小PF | 変動幅 | PF>1.0率 | トレンド傾き |")
        lines.append("|--------|------|--------|----------|--------|--------|----------|--------------|")
        for _, row in rolling_df.iterrows():
            lines.append(
                f"| {row['rank']} | {row['threshold_pct']} | {row['mean_pf']:.3f} | "
                f"{row['std_pf']:.3f} | {row['min_pf']:.3f} | "
                f"{row['variation']:.3f} | {row['ratio_above_1']*100:.0f}% | "
                f"{row['trend_slope']:+.4f} |"
            )
        lines.append("")

        # 最優秀候補
        best = rolling_df.iloc[0]
        lines += [
            "---",
            "",
            f"## 最終推奨候補: -{abs(best['threshold'])*100:.0f}%",
            "",
            f"- 平均PF: {best['mean_pf']:.3f}",
            f"- 時系列変動: {best['variation']:.3f}（小さいほど安定）",
            f"- 全期間でPF>1.0: {best['ratio_above_1']*100:.0f}%",
            f"- トレンド傾き: {best['trend_slope']:+.4f}（0に近いほど時代遅れにならない）",
            "",
        ]

        # Window別詳細
        lines += [
            "---",
            "",
            "## Window別の詳細",
            "",
        ]
        for threshold, windows in all_results.items():
            lines.append(f"### -{abs(threshold)*100:.0f}% のWindow別結果")
            lines.append("")
            lines.append("| Window | 期間 | PF | 損益 | 勝率 | 取引数 |")
            lines.append("|--------|------|----|------|------|--------|")
            for r in windows:
                if r.get('success'):
                    lines.append(
                        f"| {r['window']} | {r['test_period']} | "
                        f"{r.get('pf', 'N/A'):.3f} | "
                        f"{r.get('pnl', 0):+,.0f}円 | "
                        f"{r.get('win_rate', 0)*100:.1f}% | "
                        f"{r.get('trades', 0)} |"
                    )
                else:
                    lines.append(f"| {r.get('window', '?')} | {r.get('test_period', '?')} | エラー | - | - | - |")
            lines.append("")

    lines += [
        "---",
        "",
        "## 次のステップ",
        "",
        "```powershell",
        "# 最終レポート生成",
        "python generate_final_report.py",
        "```",
    ]

    report_path = output_dir / "rolling_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Phase 2 レポート保存: {report_path}")


# ========================================================
# エントリーポイント
# ========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Rolling Window 検証")
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=float,
        default=None,
        help="テストする閾値（例: --candidates -0.05 -0.06）"
    )
    args = parser.parse_args()

    candidates = args.candidates if args.candidates else load_phase1_candidates()

    print("=" * 70)
    print("Phase 2: Rolling Window ストップロス安定性検証")
    print("=" * 70)
    print(f"候補: {[f'-{abs(c)*100:.0f}%' for c in candidates]}")
    print(f"Window数: {len(WINDOWS)}期間")
    print(f"実験数: {len(candidates)} × {len(WINDOWS)} = {len(candidates)*len(WINDOWS)} 実験")
    print(f"推定時間: 約{len(candidates)*len(WINDOWS)*17/60:.1f}時間")
    print(f"出力先: {RESULTS_DIR}")
    print("=" * 70)
    print()

    rolling_df, all_results = run_phase2(candidates)

    if not rolling_df.empty:
        print("\n" + "=" * 70)
        print("Phase 2 結果（安定性順）:")
        print("=" * 70)
        display_cols = ['rank', 'threshold_pct', 'mean_pf', 'std_pf',
                        'min_pf', 'variation', 'ratio_above_1']
        print(rolling_df[display_cols].to_string(index=False))
        print()
        best = rolling_df.iloc[0]
        print(f"最終推奨: -{abs(best['threshold'])*100:.0f}%")
        print()
        print("次のコマンドで最終レポートを生成:")
        print("  python generate_final_report.py")
