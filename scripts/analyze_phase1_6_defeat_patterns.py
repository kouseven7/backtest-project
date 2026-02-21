"""
Phase 1.6大敗パターン分析スクリプト

武田薬品工業（4502.T）の取引履歴から大敗パターンを分析し、
エグジット戦略改善のための知見を抽出する。

主な機能:
- 分析1: 大損失トレード（profit_loss_pct < -10%）の特徴抽出
- 分析2: 時系列での勝敗パターン（月別・四半期別集計）
- 分析3: エントリー品質分析（SMA乖離・トレンド強度）
- 分析4: ボラティリティ影響分析（ATR vs 成績）
- 分析5: 急騰→急落パターン検出（max_profit_pct > 15% → 損失）
- 図表自動生成（PNG形式、日本語対応）
- Markdown結果レポート自動作成

統合コンポーネント:
- pandas: データ分析・集計
- matplotlib/seaborn: グラフ描画
- pathlib: ファイル管理

セーフティ機能/注意事項:
- CSVファイル存在確認必須
- 図表保存ディレクトリ自動作成
- 日本語フォント設定（Windows環境対応）
- 数値丸め（小数点2桁統一）

Author: Backtest Project Team
Created: 2026-01-26
Last Modified: 2026-01-26
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

# プロジェクトルート追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 日本語フォント設定（Windows）
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# Seaborn設定
sns.set_style("whitegrid")
sns.set_palette("husl")

# 定数
CSV_PATH = Path('results/phase1.6_trades_20260126_124409a.csv')
TICKER = '4502.T'
FIGURES_DIR = Path('docs/exit_strategy/figures')
RESULT_MD_PATH = Path('docs/exit_strategy/PHASE1_6_DEFEAT_PATTERNS_RESULT.md')

# 大損失閾値
BIG_LOSS_THRESHOLD = -10.0  # %

# 急騰→急落パターン閾値
PUMP_THRESHOLD_MAX_PROFIT = 15.0  # %
PUMP_THRESHOLD_HOLDING_DAYS = 30


# ==================== ユーティリティ関数 ====================

def load_data() -> pd.DataFrame:
    """
    CSVデータを読み込み、武田薬品のみ抽出
    
    Returns:
        武田薬品の取引履歴DataFrame
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {CSV_PATH}")
    
    # ヘッダーなしCSVを読み込み、カラム名を手動設定
    column_names = [
        'trade_id', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 
        'profit_loss', 'exit_reason', 'holding_days', 'profit_loss_pct', 
        'r_multiple', 'entry_gap_pct', 'max_profit_pct', 'entry_atr_pct', 
        'sma_distance_pct', 'entry_trend_strength', 'entry_volume', 
        'exit_volume', 'volume_ratio', 'exit_signal_pct', 'sma_value', 
        'trend_value', 'above_trend', 'unused1', 'entry_price_base', 
        'unused2', 'unused3', 'ticker', 'stop_loss_pct', 'trailing_stop_pct'
    ]
    
    df = pd.read_csv(CSV_PATH, header=None, names=column_names)
    
    # ティッカーフィルター
    takeda_df = df[df['ticker'] == TICKER].copy()
    
    if len(takeda_df) == 0:
        raise ValueError(f"ティッカー {TICKER} のデータが見つかりません")
    
    # 日付変換
    takeda_df['entry_date'] = pd.to_datetime(takeda_df['entry_date'])
    takeda_df['exit_date'] = pd.to_datetime(takeda_df['exit_date'])
    
    print(f"データ読み込み完了: {len(takeda_df)}件の取引")
    print(f"期間: {takeda_df['entry_date'].min().date()} ～ {takeda_df['exit_date'].max().date()}")
    
    return takeda_df


def save_figure(fig, filename: str):
    """
    図表をPNG形式で保存
    
    Args:
        fig: matplotlibのfigureオブジェクト
        filename: ファイル名（拡張子なし）
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / f"{filename}.png"
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"図表保存: {filepath}")


def calculate_pf(df: pd.DataFrame) -> float:
    """
    プロフィットファクター計算
    
    Args:
        df: 取引履歴DataFrame
    
    Returns:
        プロフィットファクター
    """
    winning = df[df['profit_loss_pct'] > 0]
    losing = df[df['profit_loss_pct'] <= 0]
    
    total_profit = winning['profit_loss_pct'].sum()
    total_loss = abs(losing['profit_loss_pct'].sum())
    
    if total_loss > 0:
        return total_profit / total_loss
    elif total_profit > 0:
        return float('inf')
    else:
        return 0.0


# ==================== 分析1: 大損失トレード特徴抽出 ====================

def analyze_big_loss_trades(df: pd.DataFrame) -> Dict:
    """
    分析1: 大損失トレード（profit_loss_pct < -10%）の特徴抽出
    
    Args:
        df: 取引履歴DataFrame
    
    Returns:
        分析結果辞書
    """
    print("\n" + "=" * 80)
    print("分析1: 大損失トレード特徴抽出")
    print("=" * 80)
    
    # 大損失トレード抽出
    big_loss = df[df['profit_loss_pct'] < BIG_LOSS_THRESHOLD].copy()
    normal = df[df['profit_loss_pct'] >= BIG_LOSS_THRESHOLD].copy()
    
    print(f"大損失トレード: {len(big_loss)}件（{len(big_loss)/len(df)*100:.1f}%）")
    print(f"通常トレード: {len(normal)}件（{len(normal)/len(df)*100:.1f}%）")
    
    # 統計計算
    analysis_cols = ['entry_gap_pct', 'entry_atr_pct', 'holding_days', 
                     'max_profit_pct', 'entry_trend_strength', 'sma_distance_pct']
    
    big_loss_stats = big_loss[analysis_cols].describe().T
    normal_stats = normal[analysis_cols].describe().T
    
    # exit_reason内訳
    big_loss_exit_reasons = big_loss['exit_reason'].value_counts()
    normal_exit_reasons = normal['exit_reason'].value_counts()
    
    # 比較表作成
    comparison = pd.DataFrame({
        '大損失_平均': big_loss[analysis_cols].mean(),
        '大損失_中央値': big_loss[analysis_cols].median(),
        '通常_平均': normal[analysis_cols].mean(),
        '通常_中央値': normal[analysis_cols].median(),
        '差分（大損失-通常）': big_loss[analysis_cols].mean() - normal[analysis_cols].mean()
    })
    
    print("\n【特徴比較】")
    print(comparison.round(2))
    
    # 図1: exit_reason円グラフ
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].pie(big_loss_exit_reasons.values, labels=big_loss_exit_reasons.index, 
                autopct='%1.1f%%', startangle=90)
    axes[0].set_title(f'大損失トレード exit_reason内訳（n={len(big_loss)}）')
    
    axes[1].pie(normal_exit_reasons.values, labels=normal_exit_reasons.index, 
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'通常トレード exit_reason内訳（n={len(normal)}）')
    
    save_figure(fig, 'analysis1_exit_reason_pie')
    
    # 図2: entry_atr_pctヒストグラム
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(big_loss['entry_atr_pct'], bins=20, alpha=0.6, label='大損失', color='red')
    ax.hist(normal['entry_atr_pct'], bins=20, alpha=0.6, label='通常', color='blue')
    ax.set_xlabel('Entry ATR (%)')
    ax.set_ylabel('取引数')
    ax.set_title('Entry ATR分布比較')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'analysis1_entry_atr_histogram')
    
    # 図3: entry_gap_pctヒストグラム
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(big_loss['entry_gap_pct'], bins=20, alpha=0.6, label='大損失', color='red')
    ax.hist(normal['entry_gap_pct'], bins=20, alpha=0.6, label='通常', color='blue')
    ax.set_xlabel('Entry Gap (%)')
    ax.set_ylabel('取引数')
    ax.set_title('Entry Gap分布比較')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'analysis1_entry_gap_histogram')
    
    return {
        'big_loss_count': len(big_loss),
        'big_loss_ratio': len(big_loss) / len(df),
        'comparison': comparison,
        'big_loss_exit_reasons': big_loss_exit_reasons,
        'normal_exit_reasons': normal_exit_reasons
    }


# ==================== 分析2: 時系列での勝敗パターン ====================

def analyze_time_series_patterns(df: pd.DataFrame) -> Dict:
    """
    分析2: 時系列での勝敗パターン（月別・四半期別）
    
    Args:
        df: 取引履歴DataFrame
    
    Returns:
        分析結果辞書
    """
    print("\n" + "=" * 80)
    print("分析2: 時系列での勝敗パターン")
    print("=" * 80)
    
    # 月別集計
    df['year_month'] = df['entry_date'].dt.to_period('M')
    
    monthly = df.groupby('year_month').agg({
        'profit_loss_pct': ['count', 'mean', 'min', 'sum']
    }).round(2)
    
    monthly.columns = ['総取引数', '平均損益率', '最大損失', '合計損益']
    
    # 勝率計算
    monthly['勝ち取引数'] = df.groupby('year_month').apply(
        lambda x: (x['profit_loss_pct'] > 0).sum()
    )
    monthly['勝率'] = monthly['勝ち取引数'] / monthly['総取引数']
    
    # 2023年8-10月をハイライト
    bad_months = monthly[(monthly.index >= '2023-08') & (monthly.index <= '2023-10')]
    
    print("\n【月別集計（全期間）】")
    print(monthly)
    
    print("\n【2023年8-10月（ハイライト）】")
    print(bad_months)
    
    # 特に成績が悪い期間（平均損益率 < -5%）
    very_bad = monthly[monthly['平均損益率'] < -5.0]
    print(f"\n【平均損益率 < -5%の期間】: {len(very_bad)}期間")
    if not very_bad.empty:
        print(very_bad)
    
    # 図1: 月別平均損益率の折れ線グラフ
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_values = range(len(monthly))
    ax.plot(x_values, monthly['平均損益率'].values, marker='o', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='危険水準 -5%')
    
    # 2023年8-10月をハイライト
    highlight_indices = [i for i, idx in enumerate(monthly.index) 
                        if '2023-08' <= str(idx) <= '2023-10']
    if highlight_indices:
        ax.axvspan(min(highlight_indices), max(highlight_indices), 
                  alpha=0.2, color='red', label='2023年8-10月')
    
    ax.set_xlabel('期間')
    ax.set_ylabel('平均損益率 (%)')
    ax.set_title('月別平均損益率推移')
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(idx) for idx in monthly.index], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'analysis2_monthly_profit_loss')
    
    # 図2: 月別勝率の折れ線グラフ
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(x_values, monthly['勝率'].values, marker='o', linewidth=2, color='green')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50%')
    
    # 2023年8-10月をハイライト
    if highlight_indices:
        ax.axvspan(min(highlight_indices), max(highlight_indices), 
                  alpha=0.2, color='red', label='2023年8-10月')
    
    ax.set_xlabel('期間')
    ax.set_ylabel('勝率')
    ax.set_title('月別勝率推移')
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(idx) for idx in monthly.index], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'analysis2_monthly_win_rate')
    
    return {
        'monthly': monthly,
        'bad_months_2023': bad_months,
        'very_bad_periods': very_bad
    }


# ==================== 分析3: エントリー品質分析 ====================

def analyze_entry_quality(df: pd.DataFrame) -> Dict:
    """
    分析3: エントリー品質分析（SMA乖離・トレンド強度）
    
    Args:
        df: 取引履歴DataFrame
    
    Returns:
        分析結果辞書
    """
    print("\n" + "=" * 80)
    print("分析3: エントリー品質分析")
    print("=" * 80)
    
    # 3-1: SMA乖離範囲分類
    def classify_sma_distance(val):
        if val < 5:
            return '0-5%'
        elif val < 10:
            return '5-10%'
        elif val < 15:
            return '10-15%'
        elif val < 20:
            return '15-20%'
        else:
            return '20%+'
    
    df['sma_range'] = df['sma_distance_pct'].apply(classify_sma_distance)
    
    # SMA範囲別集計
    sma_analysis = df.groupby('sma_range').agg({
        'profit_loss_pct': ['count', 'mean'],
        'r_multiple': 'mean'
    }).round(2)
    
    sma_analysis.columns = ['取引数', '平均損益率', '平均R倍']
    sma_analysis['勝率'] = df.groupby('sma_range').apply(
        lambda x: (x['profit_loss_pct'] > 0).mean()
    ).round(3)
    
    # 範囲順にソート
    order = ['0-5%', '5-10%', '10-15%', '15-20%', '20%+']
    sma_analysis = sma_analysis.reindex([o for o in order if o in sma_analysis.index])
    
    print("\n【SMA乖離範囲別成績】")
    print(sma_analysis)
    
    # 3-2: トレンド強度3分割
    quantiles = df['entry_trend_strength'].quantile([0.33, 0.66])
    
    def classify_trend_strength(val):
        if val < quantiles[0.33]:
            return '低'
        elif val < quantiles[0.66]:
            return '中'
        else:
            return '高'
    
    df['trend_strength_level'] = df['entry_trend_strength'].apply(classify_trend_strength)
    
    # トレンド強度別集計
    trend_analysis = df.groupby('trend_strength_level').agg({
        'profit_loss_pct': ['count', 'mean'],
        'r_multiple': 'mean'
    }).round(2)
    
    trend_analysis.columns = ['取引数', '平均損益率', '平均R倍']
    trend_analysis['勝率'] = df.groupby('trend_strength_level').apply(
        lambda x: (x['profit_loss_pct'] > 0).mean()
    ).round(3)
    
    # 順序指定
    trend_analysis = trend_analysis.reindex(['低', '中', '高'])
    
    print("\n【トレンド強度別成績】")
    print(trend_analysis)
    
    # 図1: SMA乖離範囲別平均損益率（棒グラフ）
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = range(len(sma_analysis))
    colors = ['green' if v > 0 else 'red' for v in sma_analysis['平均損益率'].values]
    
    ax.bar(x_pos, sma_analysis['平均損益率'].values, color=colors, alpha=0.7)
    ax.set_xlabel('SMA乖離範囲')
    ax.set_ylabel('平均損益率 (%)')
    ax.set_title('SMA乖離範囲別 平均損益率')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sma_analysis.index)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 取引数を棒の上に表示
    for i, (v, count) in enumerate(zip(sma_analysis['平均損益率'].values, 
                                        sma_analysis['取引数'].values)):
        ax.text(i, v, f'n={count}', ha='center', va='bottom' if v > 0 else 'top')
    
    save_figure(fig, 'analysis3_sma_distance_bar')
    
    # 図2: トレンド強度別成績比較（棒グラフ）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = range(len(trend_analysis))
    colors = ['green' if v > 0 else 'red' for v in trend_analysis['平均損益率'].values]
    
    ax.bar(x_pos, trend_analysis['平均損益率'].values, color=colors, alpha=0.7)
    ax.set_xlabel('トレンド強度')
    ax.set_ylabel('平均損益率 (%)')
    ax.set_title('トレンド強度別 平均損益率')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(trend_analysis.index)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 取引数を棒の上に表示
    for i, (v, count) in enumerate(zip(trend_analysis['平均損益率'].values, 
                                        trend_analysis['取引数'].values)):
        ax.text(i, v, f'n={count}', ha='center', va='bottom' if v > 0 else 'top')
    
    save_figure(fig, 'analysis3_trend_strength_bar')
    
    return {
        'sma_analysis': sma_analysis,
        'trend_analysis': trend_analysis
    }


# ==================== 分析4: ボラティリティ影響 ====================

def analyze_volatility_impact(df: pd.DataFrame) -> Dict:
    """
    分析4: ボラティリティ影響分析（ATR vs 成績）
    
    Args:
        df: 取引履歴DataFrame
    
    Returns:
        分析結果辞書
    """
    print("\n" + "=" * 80)
    print("分析4: ボラティリティ影響分析")
    print("=" * 80)
    
    # ATR基本統計
    atr_stats = df['entry_atr_pct'].describe()
    print("\n【Entry ATR基本統計】")
    print(atr_stats.round(2))
    
    # ATR範囲分類
    def classify_atr(val):
        if val < 2:
            return '低 (0-2%)'
        elif val < 3:
            return '中 (2-3%)'
        elif val < 4:
            return '高 (3-4%)'
        else:
            return '超高 (4%+)'
    
    df['atr_range'] = df['entry_atr_pct'].apply(classify_atr)
    
    # ATR範囲別集計
    atr_analysis = df.groupby('atr_range').agg({
        'profit_loss_pct': ['count', 'mean']
    }).round(2)
    
    atr_analysis.columns = ['取引数', '平均損益率']
    atr_analysis['勝率'] = df.groupby('atr_range').apply(
        lambda x: (x['profit_loss_pct'] > 0).mean()
    ).round(3)
    
    # 順序指定
    order = ['低 (0-2%)', '中 (2-3%)', '高 (3-4%)', '超高 (4%+)']
    atr_analysis = atr_analysis.reindex([o for o in order if o in atr_analysis.index])
    
    print("\n【ATR範囲別成績】")
    print(atr_analysis)
    
    # 高ATRトレード抽出（entry_atr_pct > 3.5%）
    high_atr = df[df['entry_atr_pct'] > 3.5].sort_values(
        'entry_atr_pct', ascending=False
    ).head(10)
    
    print(f"\n【高ATRトレード（entry_atr_pct > 3.5%）】: {len(df[df['entry_atr_pct'] > 3.5])}件")
    if not high_atr.empty:
        print("\nTOP 10:")
        print(high_atr[['entry_date', 'entry_atr_pct', 'profit_loss_pct', 
                        'holding_days', 'exit_reason']].to_string())
    
    # 図1: ATR vs 損益率の散布図
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if p > 0 else 'red' for p in df['profit_loss_pct'].values]
    ax.scatter(df['entry_atr_pct'], df['profit_loss_pct'], 
              c=colors, alpha=0.5, s=50)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=-10, color='red', linestyle='--', alpha=0.5, label='大損失閾値 -10%')
    ax.axvline(x=3.5, color='orange', linestyle='--', alpha=0.5, label='高ATR閾値 3.5%')
    
    ax.set_xlabel('Entry ATR (%)')
    ax.set_ylabel('損益率 (%)')
    ax.set_title('Entry ATR vs 損益率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'analysis4_atr_vs_profit')
    
    return {
        'atr_stats': atr_stats,
        'atr_analysis': atr_analysis,
        'high_atr_trades': high_atr
    }


# ==================== 分析5: 急騰→急落パターン検出 ====================

def detect_pump_and_dump_pattern(df: pd.DataFrame) -> Dict:
    """
    分析5: 急騰→急落パターン検出
    
    検出条件:
    - max_profit_pct > 15%
    - profit_loss_pct < 0%
    - holding_days < 30
    
    Args:
        df: 取引履歴DataFrame
    
    Returns:
        分析結果辞書
    """
    print("\n" + "=" * 80)
    print("分析5: 急騰→急落パターン検出")
    print("=" * 80)
    
    # パターン検出
    pump_and_dump = df[
        (df['max_profit_pct'] > PUMP_THRESHOLD_MAX_PROFIT) &
        (df['profit_loss_pct'] < 0) &
        (df['holding_days'] < PUMP_THRESHOLD_HOLDING_DAYS)
    ].copy()
    
    pattern_count = len(pump_and_dump)
    total_trades = len(df)
    pattern_ratio = pattern_count / total_trades
    
    print(f"該当パターン: {pattern_count}件（{pattern_ratio*100:.1f}%）")
    
    if pattern_count == 0:
        print("該当するパターンが見つかりませんでした")
        return {
            'pattern_count': 0,
            'pattern_ratio': 0,
            'pump_and_dump': pd.DataFrame()
        }
    
    # 統計計算
    total_loss = pump_and_dump['profit_loss_pct'].sum()
    avg_max_profit = pump_and_dump['max_profit_pct'].mean()
    avg_final_loss = pump_and_dump['profit_loss_pct'].mean()
    
    print(f"合計損失: {total_loss:.2f}%")
    print(f"平均最大到達点: {avg_max_profit:.2f}%")
    print(f"平均最終損失: {avg_final_loss:.2f}%")
    
    # trailing_stop_pct分布
    trailing_dist = pump_and_dump['trailing_stop_pct'].value_counts().sort_index()
    print("\n【トレーリング設定分布】")
    print(trailing_dist)
    
    # exit_reason内訳
    exit_reason_dist = pump_and_dump['exit_reason'].value_counts()
    print("\n【exit_reason内訳】")
    print(exit_reason_dist)
    
    # 全体成績への影響度
    pf_with = calculate_pf(df)
    pf_without = calculate_pf(df[~df.index.isin(pump_and_dump.index)])
    
    print(f"\n【全体成績への影響】")
    print(f"PF（パターン含む）: {pf_with:.2f}")
    print(f"PF（パターン除外）: {pf_without:.2f}")
    print(f"PF改善: {pf_without - pf_with:.2f} ({(pf_without/pf_with - 1)*100:.1f}%)")
    
    # 該当取引リスト
    print("\n【該当取引一覧】")
    display_cols = ['entry_date', 'exit_date', 'max_profit_pct', 'profit_loss_pct', 
                   'holding_days', 'trailing_stop_pct', 'exit_reason']
    print(pump_and_dump[display_cols].to_string())
    
    # 図1: 時系列プロット
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.scatter(pump_and_dump['entry_date'], pump_and_dump['profit_loss_pct'], 
              color='red', s=100, alpha=0.6, label=f'急騰→急落（n={pattern_count}）')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('エントリー日')
    ax.set_ylabel('最終損益率 (%)')
    ax.set_title('急騰→急落パターンの発生時期')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'analysis5_pump_dump_timeline')
    
    # 図2: max_profit_pct vs profit_loss_pctの散布図
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(pump_and_dump['max_profit_pct'], pump_and_dump['profit_loss_pct'], 
              color='red', s=100, alpha=0.6)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=PUMP_THRESHOLD_MAX_PROFIT, color='orange', linestyle='--', 
              alpha=0.5, label=f'閾値 {PUMP_THRESHOLD_MAX_PROFIT}%')
    
    ax.set_xlabel('最大到達点 (%)')
    ax.set_ylabel('最終損益率 (%)')
    ax.set_title('急騰→急落パターン: 最大到達点 vs 最終損益')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'analysis5_max_profit_vs_final')
    
    return {
        'pattern_count': pattern_count,
        'pattern_ratio': pattern_ratio,
        'total_loss': total_loss,
        'avg_max_profit': avg_max_profit,
        'avg_final_loss': avg_final_loss,
        'trailing_dist': trailing_dist,
        'exit_reason_dist': exit_reason_dist,
        'pf_with': pf_with,
        'pf_without': pf_without,
        'pump_and_dump': pump_and_dump
    }


# ==================== Markdown結果レポート作成 ====================

def create_result_report(df: pd.DataFrame, results: Dict):
    """
    分析結果をMarkdownレポートとして保存
    
    Args:
        df: 取引履歴DataFrame
        results: 各分析の結果辞書
    """
    print("\n" + "=" * 80)
    print("結果レポート作成")
    print("=" * 80)
    
    RESULT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(RESULT_MD_PATH, 'w', encoding='utf-8') as f:
        f.write("# Phase 1.6大敗パターン分析結果\n\n")
        f.write(f"**分析日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**対象銘柄**: 武田薬品工業（{TICKER}）  \n")
        f.write(f"**総取引数**: {len(df)}件  \n")
        f.write(f"**期間**: {df['entry_date'].min().date()} ～ {df['exit_date'].max().date()}  \n\n")
        
        f.write("---\n\n")
        
        # 分析1
        f.write("## 分析1: 大損失トレード特徴抽出\n\n")
        r1 = results['analysis1']
        f.write(f"**大損失トレード**: {r1['big_loss_count']}件（{r1['big_loss_ratio']*100:.1f}%）\n\n")
        f.write("### 特徴比較\n\n")
        f.write(r1['comparison'].to_markdown())
        f.write("\n\n")
        
        f.write("### exit_reason内訳\n\n")
        f.write("**大損失トレード**:\n")
        for reason, count in r1['big_loss_exit_reasons'].items():
            f.write(f"- {reason}: {count}件 ({count/r1['big_loss_count']*100:.1f}%)\n")
        f.write("\n")
        
        f.write("**図表**:\n")
        f.write("- [exit_reason円グラフ](figures/analysis1_exit_reason_pie.png)\n")
        f.write("- [Entry ATRヒストグラム](figures/analysis1_entry_atr_histogram.png)\n")
        f.write("- [Entry Gapヒストグラム](figures/analysis1_entry_gap_histogram.png)\n\n")
        
        # 分析2
        f.write("## 分析2: 時系列での勝敗パターン\n\n")
        r2 = results['analysis2']
        f.write("### 月別集計\n\n")
        f.write(r2['monthly'].to_markdown())
        f.write("\n\n")
        
        f.write("### 2023年8-10月（ハイライト）\n\n")
        f.write(r2['bad_months_2023'].to_markdown())
        f.write("\n\n")
        
        if not r2['very_bad_periods'].empty:
            f.write("### 平均損益率 < -5%の期間\n\n")
            f.write(r2['very_bad_periods'].to_markdown())
            f.write("\n\n")
        
        f.write("**図表**:\n")
        f.write("- [月別平均損益率推移](figures/analysis2_monthly_profit_loss.png)\n")
        f.write("- [月別勝率推移](figures/analysis2_monthly_win_rate.png)\n\n")
        
        # 分析3
        f.write("## 分析3: エントリー品質分析\n\n")
        r3 = results['analysis3']
        f.write("### SMA乖離範囲別成績\n\n")
        f.write(r3['sma_analysis'].to_markdown())
        f.write("\n\n")
        
        f.write("### トレンド強度別成績\n\n")
        f.write(r3['trend_analysis'].to_markdown())
        f.write("\n\n")
        
        f.write("**図表**:\n")
        f.write("- [SMA乖離範囲別平均損益率](figures/analysis3_sma_distance_bar.png)\n")
        f.write("- [トレンド強度別平均損益率](figures/analysis3_trend_strength_bar.png)\n\n")
        
        # 分析4
        f.write("## 分析4: ボラティリティ影響分析\n\n")
        r4 = results['analysis4']
        f.write("### Entry ATR基本統計\n\n")
        f.write("| 統計量 | 値 |\n")
        f.write("|--------|----|\n")
        for stat, value in r4['atr_stats'].items():
            f.write(f"| {stat} | {value:.2f}% |\n")
        f.write("\n")
        
        f.write("### ATR範囲別成績\n\n")
        f.write(r4['atr_analysis'].to_markdown())
        f.write("\n\n")
        
        high_atr_count = len(df[df['entry_atr_pct'] > 3.5])
        f.write(f"### 高ATRトレード（entry_atr_pct > 3.5%）: {high_atr_count}件\n\n")
        
        f.write("**図表**:\n")
        f.write("- [ATR vs 損益率散布図](figures/analysis4_atr_vs_profit.png)\n\n")
        
        # 分析5
        f.write("## 分析5: 急騰→急落パターン検出\n\n")
        r5 = results['analysis5']
        
        if r5['pattern_count'] > 0:
            f.write(f"**該当パターン**: {r5['pattern_count']}件（{r5['pattern_ratio']*100:.1f}%）\n\n")
            f.write(f"- 合計損失: {r5['total_loss']:.2f}%\n")
            f.write(f"- 平均最大到達点: {r5['avg_max_profit']:.2f}%\n")
            f.write(f"- 平均最終損失: {r5['avg_final_loss']:.2f}%\n\n")
            
            f.write("### トレーリング設定分布\n\n")
            for trail, count in r5['trailing_dist'].items():
                f.write(f"- {trail}: {count}件\n")
            f.write("\n")
            
            f.write("### exit_reason内訳\n\n")
            for reason, count in r5['exit_reason_dist'].items():
                f.write(f"- {reason}: {count}件\n")
            f.write("\n")
            
            f.write("### 全体成績への影響\n\n")
            f.write(f"- PF（パターン含む）: {r5['pf_with']:.2f}\n")
            f.write(f"- PF（パターン除外）: {r5['pf_without']:.2f}\n")
            f.write(f"- PF改善: {r5['pf_without'] - r5['pf_with']:.2f} ")
            f.write(f"({(r5['pf_without']/r5['pf_with'] - 1)*100:.1f}%)\n\n")
            
            f.write("**図表**:\n")
            f.write("- [急騰→急落パターン発生時期](figures/analysis5_pump_dump_timeline.png)\n")
            f.write("- [最大到達点 vs 最終損益](figures/analysis5_max_profit_vs_final.png)\n\n")
        else:
            f.write("該当するパターンは見つかりませんでした。\n\n")
        
        # 発見事項と改善提案
        f.write("---\n\n")
        f.write("## 主要発見事項\n\n")
        f.write("### 発見1: [自動記入不可 - 手動で追記してください]\n\n")
        f.write("### 発見2: [自動記入不可 - 手動で追記してください]\n\n")
        f.write("### 発見3: [自動記入不可 - 手動で追記してください]\n\n")
        
        f.write("## 改善提案\n\n")
        f.write("### 提案1: [自動記入不可 - 手動で追記してください]\n\n")
        f.write("### 提案2: [自動記入不可 - 手動で追記してください]\n\n")
        f.write("### 提案3: [自動記入不可 - 手動で追記してください]\n\n")
        
        f.write("---\n\n")
        f.write("**作成者**: analyze_phase1_6_defeat_patterns.py  \n")
        f.write(f"**最終更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    
    print(f"結果レポート保存: {RESULT_MD_PATH}")


# ==================== メイン実行 ====================

def main():
    """メインエントリーポイント"""
    print("=" * 80)
    print("Phase 1.6大敗パターン分析開始")
    print("=" * 80)
    
    try:
        # データ読み込み
        df = load_data()
        
        # 分析実行
        results = {}
        
        results['analysis1'] = analyze_big_loss_trades(df)
        results['analysis2'] = analyze_time_series_patterns(df)
        results['analysis3'] = analyze_entry_quality(df)
        results['analysis4'] = analyze_volatility_impact(df)
        results['analysis5'] = detect_pump_and_dump_pattern(df)
        
        # 結果レポート作成
        create_result_report(df, results)
        
        print("\n" + "=" * 80)
        print("全分析完了")
        print("=" * 80)
        print(f"\n結果ファイル:")
        print(f"- Markdown: {RESULT_MD_PATH}")
        print(f"- 図表: {FIGURES_DIR}")
        
    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
