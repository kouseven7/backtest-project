"""
optimize_stoploss_kfold.py
Phase 1: K-Fold Cross Validation によるストップロス閾値最適化

使い方:
    python optimize_stoploss_kfold.py

出力:
    results/phase1_k_fold/summary.csv
    results/phase1_k_fold/details/threshold_-X.XX/...
    results/phase1_k_fold/k_fold_analysis_report.md

中断・再開:
    results/phase1_k_fold/progress.json に進捗を保存
    再実行すると済みの実験をスキップする
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# プロジェクトルート（このスクリプトの2階層上）をパスに追加
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[1]  # my_backtest_project/
OPTIMIZER_DIR = _THIS_FILE.parent     # stoploss_optimizer/
sys.path.insert(0, str(PROJECT_ROOT))

from stoploss_optimizer.utils.backtest_runner import run_backtest, get_current_stop_loss
from stoploss_optimizer.utils.result_analyzer import (
    collect_results,
    analyze_k_fold_results,
    get_top_candidates
)


# ========================================================
# 設定
# ========================================================

# テスト対象の閾値（マイナス値）
THRESHOLDS = [-0.03, -0.04, -0.05, -0.06, -0.07]

# K-Fold の Fold 定義（年単位）
FOLDS = {
    'fold1': ('2022-01-01', '2022-12-31'),
    'fold2': ('2023-01-01', '2023-12-31'),
    'fold3': ('2024-01-01', '2024-12-31'),
}

# 出力ディレクトリ
RESULTS_DIR = OPTIMIZER_DIR / "results" / "phase1_k_fold"
PROGRESS_FILE = RESULTS_DIR / "progress.json"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            RESULTS_DIR / "phase1_kfold.log" if RESULTS_DIR.exists()
            else Path("phase1_kfold.log"),
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger(__name__)


# ========================================================
# 実験の組み合わせ生成
# ========================================================

def generate_experiments() -> list[dict]:
    """6通りの訓練/テスト組み合わせを生成"""
    experiments = []
    fold_keys = list(FOLDS.keys())

    for train_key in fold_keys:
        for test_key in fold_keys:
            if train_key != test_key:
                experiments.append({
                    'train_fold': train_key,
                    'test_fold': test_key,
                    'train_start': FOLDS[train_key][0],
                    'train_end': FOLDS[train_key][1],
                    'test_start': FOLDS[test_key][0],
                    'test_end': FOLDS[test_key][1],
                })

    return experiments


# ========================================================
# 進捗管理（中断・再開対応）
# ========================================================

def load_progress() -> dict:
    """保存済みの進捗を読み込む"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    """進捗を保存する"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def experiment_key(threshold: float, exp: dict) -> str:
    """実験を一意に識別するキー"""
    return f"{threshold:.2f}_{exp['train_fold']}_{exp['test_fold']}"


# ========================================================
# 単一実験の実行
# ========================================================

def run_single_experiment(threshold: float, exp: dict) -> dict:
    """
    1つの実験（訓練fold / テストfold / 閾値）を実行する。

    NOTE: K-Fold では「訓練データ」でパラメータを選ぶという考え方だが、
    今回のバックテストシステムは期間を指定して実行するのみ。
    そのため、テスト期間でのバックテスト結果をそのまま評価に使う。
    （訓練データは「このパラメータが通用する根拠の期間」という位置づけ）
    """
    logger.info(
        f"実験開始: threshold={threshold*100:.0f}%, "
        f"train={exp['train_fold']}({exp['train_start']}〜{exp['train_end']}), "
        f"test={exp['test_fold']}({exp['test_start']}〜{exp['test_end']})"
    )

    # テスト期間でバックテスト実行
    bt_result = run_backtest(
        start_date=exp['test_start'],
        end_date=exp['test_end'],
        threshold=threshold
    )

    if not bt_result['success']:
        logger.error(f"バックテスト失敗: {bt_result['error']}")
        return {
            'success': False,
            'threshold': threshold,
            'train_fold': exp['train_fold'],
            'test_fold': exp['test_fold'],
            'test_period': f"{exp['test_start']}〜{exp['test_end']}",
            'error': bt_result['error'],
            'duration_seconds': bt_result.get('duration_seconds', 0)
        }

    # 結果収集
    metrics = collect_results(bt_result['output_dir'])

    result = {
        'success': metrics['success'],
        'threshold': threshold,
        'train_fold': exp['train_fold'],
        'test_fold': exp['test_fold'],
        'test_period': f"{exp['test_start']}〜{exp['test_end']}",
        'output_dir': str(bt_result['output_dir']),
        'duration_seconds': bt_result['duration_seconds'],
        **{k: v for k, v in metrics.items() if k != 'success'},
        'error': metrics.get('error')
    }

    if metrics['success']:
        logger.info(
            f"  → PF={metrics['pf']:.3f}, 勝率={metrics['win_rate']*100:.1f}%, "
            f"損益={metrics['pnl']:+,.0f}円, 取引={metrics['trades']}件"
        )

    return result


# ========================================================
# Phase 1 メイン処理
# ========================================================

def run_phase1():
    """Phase 1: K-Fold Cross Validation を実行する"""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    details_dir = RESULTS_DIR / "details"
    details_dir.mkdir(exist_ok=True)

    # ログファイルを再設定（ディレクトリ確定後）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS_DIR / "phase1_kfold.log", encoding="utf-8")
        ],
        force=True
    )

    experiments = generate_experiments()
    progress = load_progress()

    total = len(THRESHOLDS) * len(experiments)
    done = sum(1 for k in progress if progress[k].get('done'))
    logger.info(f"Phase 1 開始: {total}実験中 {done}件済み")
    logger.info(f"閾値: {[f'{t*100:.0f}%' for t in THRESHOLDS]}")
    logger.info(f"実験パターン: {len(experiments)}通り × {len(THRESHOLDS)}閾値")

    # 全結果を格納
    all_results = {}  # {threshold: [result, ...]}
    for t in THRESHOLDS:
        all_results[t] = []

    phase1_start = time.time()

    for threshold in THRESHOLDS:
        threshold_dir = details_dir / f"threshold_{threshold:.2f}"
        threshold_dir.mkdir(exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"閾値 {threshold*100:.0f}% の実験開始")
        logger.info(f"{'='*60}")

        for exp in experiments:
            key = experiment_key(threshold, exp)

            # 済みの実験はスキップ
            if key in progress and progress[key].get('done'):
                logger.info(f"スキップ（済み）: {key}")
                # 保存済み結果を読み込む
                result_file = threshold_dir / f"{exp['train_fold']}_train_{exp['test_fold']}_test.json"
                if result_file.exists():
                    with open(result_file, "r", encoding="utf-8") as f:
                        all_results[threshold].append(json.load(f))
                continue

            # 実験実行
            result = run_single_experiment(threshold, exp)

            # 結果を保存
            result_file = threshold_dir / f"{exp['train_fold']}_train_{exp['test_fold']}_test.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            all_results[threshold].append(result)

            # 進捗を更新
            progress[key] = {'done': True, 'timestamp': datetime.now().isoformat()}
            save_progress(progress)

            # 経過時間の目安を表示
            elapsed = time.time() - phase1_start
            completed = sum(len(v) for v in all_results.values())
            if completed > 0:
                avg_per_exp = elapsed / completed
                remaining = (total - completed) * avg_per_exp
                logger.info(
                    f"進捗: {completed}/{total} ({completed/total*100:.0f}%), "
                    f"残り約{remaining/60:.0f}分"
                )

    # 全閾値の結果をCSVに保存
    logger.info("\nPhase 1 全実験完了。分析中...")
    _save_summary_csv(all_results, RESULTS_DIR)

    # K-Fold分析
    k_fold_df = analyze_k_fold_results(all_results)
    if not k_fold_df.empty:
        k_fold_df.to_csv(RESULTS_DIR / "k_fold_analysis.csv", index=False, encoding="utf-8-sig")
        candidates = get_top_candidates(k_fold_df, n=2)
        logger.info(f"\nPhase 2 推奨候補: {[f'{c*100:.0f}%' for c in candidates]}")

    # レポート生成
    _generate_phase1_report(k_fold_df, all_results, RESULTS_DIR)

    total_elapsed = time.time() - phase1_start
    logger.info(f"\nPhase 1 完了! 所要時間: {total_elapsed/60:.1f}分")
    logger.info(f"結果: {RESULTS_DIR}")

    return k_fold_df, all_results


# ========================================================
# CSV・レポート生成
# ========================================================

def _save_summary_csv(all_results: dict, output_dir: Path):
    """全実験結果をフラットなCSVに保存"""
    rows = []
    for threshold, experiments in all_results.items():
        for r in experiments:
            rows.append({
                'threshold': threshold,
                'threshold_pct': f"{abs(threshold)*100:.0f}%",
                'train_fold': r.get('train_fold', ''),
                'test_fold': r.get('test_fold', ''),
                'test_period': r.get('test_period', ''),
                'pf': r.get('pf'),
                'pnl': r.get('pnl'),
                'win_rate': r.get('win_rate'),
                'trades': r.get('trades'),
                'max_loss': r.get('max_loss'),
                'rr_ratio': r.get('rr_ratio'),
                'duration_sec': r.get('duration_seconds', 0),
                'success': r.get('success', False),
                'error': r.get('error', '')
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
        logger.info(f"サマリーCSV保存: {output_dir / 'summary.csv'}")


def _generate_phase1_report(k_fold_df: pd.DataFrame, all_results: dict, output_dir: Path):
    """Phase 1 分析レポートを Markdown で生成"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Phase 1: K-Fold Cross Validation 分析レポート",
        f"",
        f"**生成日時**: {now}  ",
        f"**Fold定義**: fold1=2022年, fold2=2023年, fold3=2024年  ",
        f"**テスト閾値**: {[f'{abs(t)*100:.0f}%' for t in THRESHOLDS]}  ",
        f"",
        f"---",
        f"",
        f"## サマリー（信頼性スコア順）",
        f"",
    ]

    if not k_fold_df.empty:
        # テーブル
        lines.append("| ランク | 閾値 | 平均PF | 標準偏差 | 最小PF | 最大PF | 信頼性スコア | 勝率 | 平均取引数 |")
        lines.append("|--------|------|--------|----------|--------|--------|--------------|------|------------|")
        for _, row in k_fold_df.iterrows():
            lines.append(
                f"| {row['rank']} | {row['threshold_pct']} | {row['mean_pf']:.3f} | "
                f"{row['std_pf']:.3f} | {row['min_pf']:.3f} | {row['max_pf']:.3f} | "
                f"**{row['reliability_score']:.3f}** | "
                f"{row['mean_win_rate']*100:.1f}% | {row['mean_trades']:.0f} |"
            )
        lines.append("")

        # 上位2候補
        candidates = get_top_candidates(k_fold_df, n=2)
        lines += [
            "---",
            "",
            "## Phase 2 推奨候補",
            "",
        ]
        for i, c in enumerate(candidates, 1):
            row = k_fold_df[k_fold_df['threshold'] == c].iloc[0]
            lines.append(
                f"### 候補{i}: -{abs(c)*100:.0f}% (信頼性スコア: {row['reliability_score']:.3f})"
            )
            lines.append(f"- 平均PF: {row['mean_pf']:.3f}")
            lines.append(f"- 最小PF: {row['min_pf']:.3f}")
            lines.append(f"- 標準偏差: {row['std_pf']:.3f}")
            lines.append("")

        # 各閾値の詳細
        lines += [
            "---",
            "",
            "## 各閾値の詳細",
            "",
        ]
        for threshold in THRESHOLDS:
            t_pct = f"{abs(threshold)*100:.0f}%"
            experiments = all_results.get(threshold, [])
            lines.append(f"### -{t_pct} の全実験結果")
            lines.append("")
            lines.append("| 訓練 | テスト | PF | 損益 | 勝率 | 取引数 |")
            lines.append("|------|--------|----|------|------|--------|")
            for r in experiments:
                if r.get('success'):
                    lines.append(
                        f"| {r['train_fold']} | {r['test_fold']} | "
                        f"{r.get('pf', 'N/A'):.3f} | "
                        f"{r.get('pnl', 0):+,.0f}円 | "
                        f"{r.get('win_rate', 0)*100:.1f}% | "
                        f"{r.get('trades', 0)} |"
                    )
                else:
                    lines.append(
                        f"| {r.get('train_fold', '?')} | {r.get('test_fold', '?')} | "
                        f"エラー | - | - | - |"
                    )
            lines.append("")

    else:
        lines.append("分析結果が得られませんでした。")

    lines += [
        "---",
        "",
        "## 次のステップ",
        "",
        "```powershell",
        "# Phase 2 実行（Phase 1の推奨候補を使用）",
        "python optimize_stoploss_rolling.py",
        "```",
    ]

    report_path = output_dir / "k_fold_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Phase 1 レポート保存: {report_path}")


# ========================================================
# エントリーポイント
# ========================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 1: K-Fold Cross Validation ストップロス最適化")
    print("=" * 70)
    print(f"閾値: {[f'{t*100:.0f}%' for t in THRESHOLDS]}")
    print(f"実験数: {len(THRESHOLDS)} × 6パターン = {len(THRESHOLDS)*6} 実験")
    print(f"推定時間: 約{len(THRESHOLDS)*6*17/60:.1f}時間 (1実験17分として)")
    print(f"出力先: {RESULTS_DIR}")
    print("=" * 70)
    print("中断後に再実行するとスキップして再開します。")
    print()

    k_fold_df, all_results = run_phase1()

    if not k_fold_df.empty:
        print("\n" + "=" * 70)
        print("Phase 1 結果サマリー（信頼性スコア順）:")
        print("=" * 70)
        display_cols = ['rank', 'threshold_pct', 'mean_pf', 'std_pf',
                        'min_pf', 'reliability_score']
        print(k_fold_df[display_cols].to_string(index=False))
        print()
        candidates = get_top_candidates(k_fold_df, n=2)
        print(f"Phase 2 推奨候補: {[f'-{abs(c)*100:.0f}%' for c in candidates]}")
        print()
        print("次のコマンドでPhase 2を実行:")
        print(f"  python optimize_stoploss_rolling.py --candidates "
              f"{' '.join(str(c) for c in candidates)}")
