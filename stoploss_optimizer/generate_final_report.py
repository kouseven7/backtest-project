"""
generate_final_report.py
Phase 1 + Phase 2 の結果を統合して最終推奨レポートを生成する

使い方:
    python generate_final_report.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[1]
OPTIMIZER_DIR = _THIS_FILE.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = OPTIMIZER_DIR / "results"
PHASE1_DIR = RESULTS_DIR / "phase1_k_fold"
PHASE2_DIR = RESULTS_DIR / "phase2_rolling"
FINAL_REPORT = RESULTS_DIR / "final_recommendation.md"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================
# データ読み込み
# ========================================================

def load_phase1() -> pd.DataFrame:
    csv = PHASE1_DIR / "k_fold_analysis.csv"
    if not csv.exists():
        logger.warning("Phase 1 データなし")
        return pd.DataFrame()
    return pd.read_csv(csv)


def load_phase2() -> pd.DataFrame:
    csv = PHASE2_DIR / "rolling_analysis.csv"
    if not csv.exists():
        logger.warning("Phase 2 データなし")
        return pd.DataFrame()
    return pd.read_csv(csv)


def load_all_results() -> tuple[dict, dict]:
    """各フェーズの全実験結果を読み込む"""
    phase1_results = {}
    phase2_results = {}

    # Phase 1
    details_dir = PHASE1_DIR / "details"
    if details_dir.exists():
        for threshold_dir in details_dir.iterdir():
            if threshold_dir.is_dir():
                try:
                    t = float(threshold_dir.name.replace("threshold_", ""))
                    phase1_results[t] = []
                    for jf in sorted(threshold_dir.glob("*.json")):
                        with open(jf) as f:
                            phase1_results[t].append(json.load(f))
                except (ValueError, json.JSONDecodeError):
                    pass

    # Phase 2
    if PHASE2_DIR.exists():
        for cand_dir in PHASE2_DIR.iterdir():
            if cand_dir.is_dir() and cand_dir.name.startswith("candidate_"):
                try:
                    t = float(cand_dir.name.replace("candidate_", ""))
                    phase2_results[t] = []
                    for jf in sorted(cand_dir.glob("window*.json")):
                        with open(jf) as f:
                            phase2_results[t].append(json.load(f))
                except (ValueError, json.JSONDecodeError):
                    pass

    return phase1_results, phase2_results


# ========================================================
# 最終推奨の決定ロジック
# ========================================================

def determine_final_recommendation(
    p1_df: pd.DataFrame, p2_df: pd.DataFrame
) -> dict:
    """
    Phase 1 + Phase 2 の結果を統合して最終推奨閾値を決定する。

    判断基準（優先順）:
    1. Phase 2 でPF>1.0率が100%（全Windowで黒字）
    2. Phase 2 の変動幅が最小（市場環境変化に強い）
    3. Phase 1 の信頼性スコアが高い（オーバーフィッティング耐性）
    4. Phase 2 の平均PFが高い
    """

    # Phase 2 のデータがある場合
    if not p2_df.empty:
        # 全Window黒字を優先
        perfect = p2_df[p2_df['ratio_above_1'] == 1.0]
        if not perfect.empty:
            # その中で変動幅最小
            best = perfect.sort_values('variation').iloc[0]
        else:
            # 変動幅最小
            best = p2_df.sort_values('variation').iloc[0]

        return {
            'threshold': best['threshold'],
            'basis': 'phase2',
            'phase2_mean_pf': best['mean_pf'],
            'phase2_min_pf': best['min_pf'],
            'phase2_variation': best['variation'],
            'phase2_ratio_above_1': best['ratio_above_1'],
        }

    # Phase 1 のみの場合
    if not p1_df.empty:
        best = p1_df.sort_values('reliability_score', ascending=False).iloc[0]
        return {
            'threshold': best['threshold'],
            'basis': 'phase1_only',
            'phase1_reliability': best['reliability_score'],
            'phase1_mean_pf': best['mean_pf'],
        }

    return {'threshold': -0.05, 'basis': 'default'}


# ========================================================
# 95%信頼区間の計算
# ========================================================

def calc_confidence_interval(pf_list: list, confidence: float = 0.95) -> tuple:
    """正規分布を仮定した信頼区間を計算"""
    if len(pf_list) < 2:
        return (None, None)

    from scipy import stats
    mean = np.mean(pf_list)
    sem = stats.sem(pf_list)
    ci = stats.t.interval(confidence, len(pf_list) - 1, loc=mean, scale=sem)
    return (round(ci[0], 3), round(ci[1], 3))


# ========================================================
# レポート生成
# ========================================================

def generate_final_report(
    p1_df: pd.DataFrame,
    p2_df: pd.DataFrame,
    p1_results: dict,
    p2_results: dict
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rec = determine_final_recommendation(p1_df, p2_df)
    threshold = rec['threshold']
    threshold_pct = f"{abs(threshold)*100:.0f}%"

    # 信頼区間計算（Phase 2 の PF リストから）
    pf_list_p2 = []
    if threshold in p2_results:
        pf_list_p2 = [r['pf'] for r in p2_results[threshold]
                      if r.get('success') and r.get('pf') is not None]

    try:
        ci = calc_confidence_interval(pf_list_p2)
    except ImportError:
        ci = (None, None)  # scipy なしの場合

    pf_list_p1 = []
    if threshold in p1_results:
        pf_list_p1 = [r['pf'] for r in p1_results[threshold]
                      if r.get('success') and r.get('pf') is not None]

    all_pf = pf_list_p1 + pf_list_p2

    lines = [
        f"# ストップロス最適化 最終推奨レポート",
        f"",
        f"**生成日時**: {now}  ",
        f"**使用データ**: 2022年〜2024年 (日経225銘柄)  ",
        f"",
        f"---",
        f"",
        f"## 推奨パラメータ",
        f"",
        f"### ✅ ストップロス閾値: **-{threshold_pct}**",
        f"",
    ]

    # 根拠セクション
    lines += [
        "---",
        "",
        "## 根拠",
        "",
    ]

    if not p1_df.empty:
        row = p1_df[p1_df['threshold'] == threshold]
        lines += [
            "### Phase 1: K-Fold Cross Validation (2022〜2024年、6実験)",
            "",
        ]
        if not row.empty:
            row = row.iloc[0]
            lines += [
                f"| 指標 | 値 |",
                f"|------|-----|",
                f"| 平均PF | {row['mean_pf']:.3f} |",
                f"| 標準偏差 | {row['std_pf']:.3f} |",
                f"| 最小PF | {row['min_pf']:.3f} |",
                f"| 最大PF | {row['max_pf']:.3f} |",
                f"| 信頼性スコア (平均PF-標準偏差) | **{row['reliability_score']:.3f}** |",
                f"| 全閾値中のランク | {int(row['rank'])}位 |",
                "",
            ]

    if not p2_df.empty:
        row = p2_df[p2_df['threshold'] == threshold]
        lines += [
            "### Phase 2: Rolling Window 検証 (6ヶ月×6期間)",
            "",
        ]
        if not row.empty:
            row = row.iloc[0]
            pf_seq = row.get('pf_by_window', pf_list_p2)
            lines += [
                f"| 指標 | 値 |",
                f"|------|-----|",
                f"| 平均PF | {row['mean_pf']:.3f} |",
                f"| 時系列変動幅 | {row['variation']:.3f}（小さいほど安定） |",
                f"| 全期間でPF>1.0 | {row['ratio_above_1']*100:.0f}% |",
                f"| トレンド傾き | {row['trend_slope']:+.4f}（0に近い = 時代遅れリスク低） |",
                f"| 最小PF | {row['min_pf']:.3f} |",
                "",
            ]

            if isinstance(pf_seq, list):
                lines += ["**期間別PF推移:**", ""]
                for i, (w, pf) in enumerate(zip(
                    ["2022H1", "2022H2", "2023H1", "2023H2", "2024H1", "2024H2"],
                    pf_seq
                ), 1):
                    bar = "█" * min(int(pf * 5), 30)
                    lines.append(f"- {w}: {pf:.3f} {bar}")
                lines.append("")

    # 期待パフォーマンス
    lines += [
        "---",
        "",
        "## 期待されるパフォーマンス",
        "",
    ]

    if all_pf:
        mean_pf = np.mean(all_pf)
        min_pf = np.min(all_pf)
        lines += [
            f"| 指標 | 値 |",
            f"|------|-----|",
            f"| 予測PF (全実験の平均) | {mean_pf:.3f} |",
            f"| 最悪ケースPF | {min_pf:.3f} |",
        ]
        if ci[0] is not None:
            lines.append(f"| 95%信頼区間 | [{ci[0]:.3f}, {ci[1]:.3f}] |")
        lines.append("")
    else:
        lines += [
            "（実験データが不足しています）",
            "",
        ]

    # リスク評価
    overfitting_risk = "低"
    if not p1_df.empty and not p1_df[p1_df['threshold'] == threshold].empty:
        std = p1_df[p1_df['threshold'] == threshold].iloc[0]['std_pf']
        if std > 1.0:
            overfitting_risk = "中〜高"
        elif std > 0.5:
            overfitting_risk = "中"

    env_resilience = "不明"
    if not p2_df.empty and not p2_df[p2_df['threshold'] == threshold].empty:
        ratio = p2_df[p2_df['threshold'] == threshold].iloc[0]['ratio_above_1']
        variation = p2_df[p2_df['threshold'] == threshold].iloc[0]['variation']
        if ratio == 1.0 and variation < 1.0:
            env_resilience = "高"
        elif ratio >= 0.8:
            env_resilience = "中〜高"
        else:
            env_resilience = "中"

    lines += [
        "---",
        "",
        "## リスク評価",
        "",
        f"| リスク項目 | 評価 | 根拠 |",
        f"|------------|------|------|",
        f"| オーバーフィッティングリスク | {overfitting_risk} | K-Fold標準偏差 |",
        f"| 市場環境変化への耐性 | {env_resilience} | Rolling Window全期間PF>1.0率 |",
        f"| 推奨採用 | {'✅' if overfitting_risk in ['低', '中'] and env_resilience in ['高', '中〜高'] else '⚠️ 要検討'} | 上記評価の総合 |",
        "",
    ]

    # 全閾値比較
    if not p1_df.empty:
        lines += [
            "---",
            "",
            "## 全閾値の比較（Phase 1）",
            "",
            "| 閾値 | 平均PF | 標準偏差 | 信頼性スコア | 最小PF |",
            "|------|--------|----------|--------------|--------|",
        ]
        for _, row in p1_df.iterrows():
            marker = " ← **推奨**" if row['threshold'] == threshold else ""
            lines.append(
                f"| -{abs(row['threshold'])*100:.0f}% | {row['mean_pf']:.3f} | "
                f"{row['std_pf']:.3f} | {row['reliability_score']:.3f} | "
                f"{row['min_pf']:.3f} |{marker}"
            )
        lines.append("")

    # 実装手順
    lines += [
        "---",
        "",
        "## 実装手順",
        "",
        f"推奨パラメータ (-{threshold_pct}) をシステムに適用:",
        "",
        "```python",
        f"# gc_strategy_signal.py Line 57 を変更",
        f"default_params = {{",
        f'    "stop_loss": {abs(threshold):.2f},  # {threshold_pct} ← 最適化後の値',
        f"    # ... 他のパラメータは変更なし",
        f"}}",
        "```",
        "",
        "```powershell",
        f"# 適用後の検証バックテスト (2024年通年)",
        f"python run_dssms_with_detailed_logs.py --start-date 2024-01-01 --end-date 2024-12-31",
        "```",
        "",
        "```powershell",
        "# Git commit",
        f'git add strategies/gc_strategy_signal.py',
        f'git commit -m "feat: ストップロス閾値を-{threshold_pct}に最適化',
        f'',
        f'Phase 1 (K-Fold) + Phase 2 (Rolling Window) による検証済み',
        f'信頼性スコア: {p1_df[p1_df["threshold"]==threshold].iloc[0]["reliability_score"]:.3f}' if not p1_df.empty and not p1_df[p1_df['threshold']==threshold].empty else '',
        f'"',
        "```",
    ]

    report_content = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"最終レポート保存: {FINAL_REPORT}")
    return rec


# ========================================================
# エントリーポイント
# ========================================================

if __name__ == "__main__":
    print("=" * 70)
    print("最終レポート生成")
    print("=" * 70)

    p1_df = load_phase1()
    p2_df = load_phase2()
    p1_results, p2_results = load_all_results()

    if p1_df.empty and p2_df.empty:
        print("警告: Phase 1 / Phase 2 のデータが見つかりません。")
        print("先に optimize_stoploss_kfold.py を実行してください。")
        sys.exit(1)

    rec = generate_final_report(p1_df, p2_df, p1_results, p2_results)

    print(f"\n推奨ストップロス閾値: -{abs(rec['threshold'])*100:.0f}%")
    print(f"根拠: {rec['basis']}")
    print(f"\n最終レポート: {FINAL_REPORT}")
