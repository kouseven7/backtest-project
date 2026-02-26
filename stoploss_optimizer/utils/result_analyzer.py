"""
result_analyzer.py
バックテスト結果（all_transactions.csv）を読み込み、
パフォーマンス指標を計算するユーティリティ
"""

import logging
import math
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ========================================================
# 結果の読み込みと計算
# ========================================================

def collect_results(output_dir: Path) -> dict:
    """
    all_transactions.csv から主要指標を計算して返す。

    Returns:
        dict: {
            'success': bool,
            'pnl': float,           # 総損益（円）
            'pf': float,            # プロフィットファクター
            'win_rate': float,      # 勝率（0〜1）
            'max_loss': float,      # 最大損失（円、負値）
            'trades': int,          # 総取引数
            'avg_pnl': float,       # 平均損益（円）
            'gross_profit': float,  # 総利益（円）
            'gross_loss': float,    # 総損失（円、負値）
            'rr_ratio': float,      # リスクリワード比
            'error': str or None
        }
    """
    csv_path = output_dir / "all_transactions.csv"

    if not csv_path.exists():
        return _error_result(f"ファイルが見つかりません: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return _error_result(f"CSV読み込みエラー: {e}")

    if df.empty:
        return _error_result("取引データが空です")

    # pnl カラムの確認
    if 'pnl' not in df.columns:
        return _error_result(f"pnlカラムが見つかりません。カラム: {list(df.columns)}")

    # NaN除去
    pnl_series = df['pnl'].dropna()
    if len(pnl_series) == 0:
        return _error_result("有効な pnl データがありません")

    total_pnl = pnl_series.sum()
    trades = len(pnl_series)
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]

    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = losses.sum() if len(losses) > 0 else 0.0
    win_rate = len(wins) / trades if trades > 0 else 0.0
    max_loss = pnl_series.min()

    # プロフィットファクター（損失がない場合は無限大→999で代替）
    if abs(gross_loss) < 0.01:
        pf = 999.0
    else:
        pf = gross_profit / abs(gross_loss)

    # リスクリワード比（勝ちトレードの平均 / 負けトレードの平均絶対値）
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    avg_pnl = total_pnl / trades if trades > 0 else 0.0

    return {
        'success': True,
        'pnl': round(total_pnl, 0),
        'pf': round(pf, 3),
        'win_rate': round(win_rate, 3),
        'max_loss': round(max_loss, 0),
        'trades': trades,
        'avg_pnl': round(avg_pnl, 0),
        'gross_profit': round(gross_profit, 0),
        'gross_loss': round(gross_loss, 0),
        'rr_ratio': round(rr_ratio, 3),
        'error': None
    }


def _error_result(msg: str) -> dict:
    logger.warning(f"結果取得失敗: {msg}")
    return {
        'success': False,
        'pnl': None,
        'pf': None,
        'win_rate': None,
        'max_loss': None,
        'trades': None,
        'avg_pnl': None,
        'gross_profit': None,
        'gross_loss': None,
        'rr_ratio': None,
        'error': msg
    }


# ========================================================
# K-Fold 分析
# ========================================================

def analyze_k_fold_results(results: dict) -> pd.DataFrame:
    """
    K-Fold実験結果を分析してDataFrameを返す。

    Args:
        results: {threshold: [{実験結果dict}, ...], ...}

    Returns:
        DataFrame: 閾値ごとの統計サマリー
    """
    rows = []

    for threshold, experiments in results.items():
        pf_list = [r['pf'] for r in experiments if r.get('success') and r.get('pf') is not None]

        if not pf_list:
            continue

        mean_pf = np.mean(pf_list)
        std_pf = np.std(pf_list)
        min_pf = np.min(pf_list)
        max_pf = np.max(pf_list)
        reliability = mean_pf - std_pf  # 信頼性スコア

        # 平均勝率・平均取引数
        wr_list = [r['win_rate'] for r in experiments if r.get('success') and r.get('win_rate') is not None]
        trades_list = [r['trades'] for r in experiments if r.get('success') and r.get('trades') is not None]
        pnl_list = [r['pnl'] for r in experiments if r.get('success') and r.get('pnl') is not None]

        rows.append({
            'threshold': threshold,
            'threshold_pct': f"{abs(threshold)*100:.0f}%",
            'n_experiments': len(pf_list),
            'mean_pf': round(mean_pf, 3),
            'std_pf': round(std_pf, 3),
            'min_pf': round(min_pf, 3),
            'max_pf': round(max_pf, 3),
            'reliability_score': round(reliability, 3),
            'mean_win_rate': round(np.mean(wr_list), 3) if wr_list else None,
            'mean_trades': round(np.mean(trades_list), 1) if trades_list else None,
            'mean_pnl': round(np.mean(pnl_list), 0) if pnl_list else None,
            'pf_values': pf_list  # 詳細確認用
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values('reliability_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df


def get_top_candidates(k_fold_df: pd.DataFrame, n: int = 2) -> list[float]:
    """K-Fold分析結果から上位N候補の閾値リストを返す"""
    if k_fold_df.empty:
        return []

    # 最小PF ≥ 1.0 を満たすものを優先（最悪ケースでも利益）
    valid = k_fold_df[k_fold_df['min_pf'] >= 1.0]
    if len(valid) >= n:
        return valid.head(n)['threshold'].tolist()

    # 条件を満たすものが少ない場合は信頼性スコア上位を返す
    return k_fold_df.head(n)['threshold'].tolist()


# ========================================================
# Rolling Window 分析
# ========================================================

def analyze_rolling_results(results: dict) -> pd.DataFrame:
    """
    Rolling Window実験結果を分析してDataFrameを返す。

    Args:
        results: {threshold: [{実験結果dict}, ...], ...}

    Returns:
        DataFrame: 候補ごとの時系列安定性サマリー
    """
    rows = []

    for threshold, windows in results.items():
        pf_list = [r['pf'] for r in windows if r.get('success') and r.get('pf') is not None]

        if not pf_list:
            continue

        mean_pf = np.mean(pf_list)
        std_pf = np.std(pf_list)
        min_pf = np.min(pf_list)
        variation = max(pf_list) - min(pf_list)  # 時系列変動幅

        # 時系列トレンド（線形回帰の傾き）
        if len(pf_list) >= 2:
            x = np.arange(len(pf_list))
            slope = np.polyfit(x, pf_list, 1)[0]
        else:
            slope = 0.0

        # 全期間黒字の割合（PF > 1.0）
        above_1 = sum(1 for pf in pf_list if pf > 1.0) / len(pf_list)

        rows.append({
            'threshold': threshold,
            'threshold_pct': f"{abs(threshold)*100:.0f}%",
            'n_windows': len(pf_list),
            'mean_pf': round(mean_pf, 3),
            'std_pf': round(std_pf, 3),
            'min_pf': round(min_pf, 3),
            'variation': round(variation, 3),
            'trend_slope': round(slope, 4),
            'ratio_above_1': round(above_1, 3),
            'pf_by_window': pf_list
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 変動幅が小さく、全期間でPF > 1.0 が多い順にソート
    df = df.sort_values(['ratio_above_1', 'variation'],
                        ascending=[False, True]).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df
