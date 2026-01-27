"""
Phase 1.9-A MA乖離フィルター検証スクリプト

SMA乖離<5%エントリー禁止フィルターの効果を9銘柄+武田薬品で検証。

主な機能:
- 10銘柄（9銘柄+武田薬品）それぞれでMA乖離率を算出
- SMA乖離<5%フィルター適用前後のPF・勝率・取引数を比較
- 改善銘柄数カウント（PF改善率>10%）
- 普遍性スコア算出（改善銘柄数 / 10）
- SMA乖離範囲別成績分析（0-5%, 5-10%, 10%+）
- 5種類以上の比較図表生成
- 統合Markdownレポート生成

統合コンポーネント:
- results/phase1.6_trades_20260126_200241.csv: Phase 1.6グリッドサーチ結果（9銘柄）
- results/phase1.6_trades_20260126_124409a.csv: 武田薬品取引履歴
- docs/exit_strategy/PHASE1_9_FILTERS_DESIGN.md: 設計書

セーフティ機能/注意事項:
- ゼロ除算保護（PF計算、割合計算等）
- 銘柄データ存在確認（最低100件の取引が必要）
- 日本語フォント設定（MS Gothic使用）
- SMA乖離<0のケースも考慮（絶対値処理）

Author: Backtest Project Team
Created: 2026-01-26
Last Modified: 2026-01-26
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

# 検証対象銘柄（9銘柄+武田薬品）
VALIDATION_TICKERS = [
    "4063.T",  # 信越化学工業
    "6501.T",  # 日立製作所
    "6758.T",  # ソニーグループ
    "6861.T",  # キーエンス
    "7203.T",  # トヨタ自動車
    "8001.T",  # 伊藤忠商事
    "8306.T",  # 三菱UFJ FG
    "9983.T",  # ファーストリテイリング
    "9984.T",  # ソフトバンクグループ
    "4502.T"   # 武田薬品工業（参考銘柄）
]

# MA乖離フィルター閾値
SMA_DISTANCE_THRESHOLD = 5.0  # 5.0%


def load_and_preprocess_data():
    """
    Step 1: データ読み込み・前処理
    
    Returns:
        dict: {ticker: DataFrame} 銘柄別データ
    """
    print("=" * 80)
    print("Step 1: データ読み込み・前処理")
    print("=" * 80)
    
    # 9銘柄データ
    csv_path_9 = Path('results/phase1.6_trades_20260126_200241.csv')
    # 武田薬品データ
    csv_path_takeda = Path('results/phase1.6_trades_20260126_124409a.csv')
    
    ticker_data = {}
    
    # 9銘柄読み込み（ヘッダーあり）
    if csv_path_9.exists():
        df_9 = pd.read_csv(csv_path_9)
        print(f"\n9銘柄データ読み込み: {len(df_9)}件")
        
        for ticker in VALIDATION_TICKERS[:-1]:  # 武田以外
            ticker_df = df_9[df_9['ticker'] == ticker].copy()
            if len(ticker_df) >= 100:
                ticker_data[ticker] = ticker_df
                print(f"  {ticker}: {len(ticker_df)}件")
            else:
                print(f"  {ticker}: データ不足（{len(ticker_df)}件）スキップ")
    else:
        print(f"\n警告: {csv_path_9} が見つかりません")
    
    # 武田薬品読み込み（ヘッダーなし）
    if csv_path_takeda.exists():
        column_names = [
            'trade_id', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 
            'profit_loss', 'exit_reason', 'holding_days', 'profit_loss_pct', 
            'r_multiple', 'entry_gap_pct', 'max_profit_pct', 'entry_atr_pct', 
            'sma_distance_pct', 'entry_trend_strength', 'entry_volume', 
            'exit_volume', 'volume_ratio', 'exit_signal_pct', 'sma_value', 
            'trend_value', 'above_trend', 'unused1', 'entry_price_base', 
            'unused2', 'unused3', 'ticker', 'stop_loss_pct', 'trailing_stop_pct'
        ]
        df_takeda = pd.read_csv(csv_path_takeda, header=None, names=column_names)
        takeda_df = df_takeda[df_takeda['ticker'] == '4502.T'].copy()
        
        if len(takeda_df) >= 100:
            ticker_data['4502.T'] = takeda_df
            print(f"\n武田薬品データ読み込み: {len(takeda_df)}件")
        else:
            print(f"\n警告: 武田薬品データ不足（{len(takeda_df)}件）")
    else:
        print(f"\n警告: {csv_path_takeda} が見つかりません")
    
    print(f"\n検証対象銘柄数: {len(ticker_data)}銘柄\n")
    return ticker_data


def calculate_sma_statistics(ticker_df):
    """
    SMA乖離率の基本統計を算出
    
    Args:
        ticker_df: 銘柄のDataFrame
        
    Returns:
        dict: 統計情報
    """
    # SMA乖離率の絶対値を計算（既存カラムがあればそのまま使用）
    if 'sma_distance_pct' not in ticker_df.columns:
        print("  警告: sma_distance_pct カラムが存在しません")
        return None
    
    sma_dist = ticker_df['sma_distance_pct'].abs()
    
    # 範囲別カウント
    range_0_5 = len(ticker_df[sma_dist < 5.0])
    range_5_10 = len(ticker_df[(sma_dist >= 5.0) & (sma_dist < 10.0)])
    range_10_plus = len(ticker_df[sma_dist >= 10.0])
    
    return {
        'mean': sma_dist.mean(),
        'median': sma_dist.median(),
        'std': sma_dist.std(),
        'min': sma_dist.min(),
        'max': sma_dist.max(),
        'q25': sma_dist.quantile(0.25),
        'q50': sma_dist.quantile(0.50),
        'q75': sma_dist.quantile(0.75),
        'range_0_5': range_0_5,
        'range_5_10': range_5_10,
        'range_10_plus': range_10_plus
    }


def calculate_performance_metrics(trades_df):
    """
    パフォーマンス指標計算
    
    Args:
        trades_df: 取引履歴DataFrame
    
    Returns:
        dict: パフォーマンス指標
    """
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'pf': 0.0,
            'avg_profit_pct': 0.0,
            'avg_r_multiple': 0.0
        }
    
    winning = trades_df[trades_df['profit_loss_pct'] > 0]
    losing = trades_df[trades_df['profit_loss_pct'] <= 0]
    
    total_profit = winning['profit_loss_pct'].sum() if len(winning) > 0 else 0.0
    total_loss = abs(losing['profit_loss_pct'].sum()) if len(losing) > 0 else 0.0
    
    pf = total_profit / total_loss if total_loss > 0 else (999.0 if total_profit > 0 else 0.0)
    
    return {
        'total_trades': len(trades_df),
        'winning_trades': len(winning),
        'win_rate': len(winning) / len(trades_df),
        'total_profit': total_profit,
        'total_loss': total_loss,
        'pf': pf,
        'avg_profit_pct': trades_df['profit_loss_pct'].mean(),
        'avg_r_multiple': trades_df['r_multiple'].mean()
    }


def apply_sma_filter(ticker_df, threshold=SMA_DISTANCE_THRESHOLD):
    """
    MA乖離フィルター適用
    
    Args:
        ticker_df: 銘柄のDataFrame
        threshold: SMA乖離閾値（デフォルト5.0%）
    
    Returns:
        DataFrame: フィルター適用後のデータ
    """
    sma_dist_abs = ticker_df['sma_distance_pct'].abs()
    filtered_df = ticker_df[sma_dist_abs < threshold].copy()
    return filtered_df


def save_figure(fig, filename, figures_dir):
    """
    図表を保存
    
    Args:
        fig: matplotlib figure
        filename: ファイル名
        figures_dir: 保存先ディレクトリ
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    filepath = figures_dir / filename
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"  図表保存: {filepath.name}")


def main():
    print("=" * 80)
    print("Phase 1.9-A MA乖離フィルター検証開始")
    print("=" * 80)
    print(f"フィルター閾値: SMA乖離 < {SMA_DISTANCE_THRESHOLD}%\n")
    
    # データ読み込み
    ticker_data = load_and_preprocess_data()
    
    if len(ticker_data) == 0:
        print("エラー: 検証対象データが見つかりません")
        return
    
    # 結果格納
    results = []
    sma_range_results = []
    
    # 銘柄別検証
    print("=" * 80)
    print("Step 2: 銘柄別フィルター効果検証")
    print("=" * 80)
    
    for ticker in VALIDATION_TICKERS:
        if ticker not in ticker_data:
            continue
        
        print(f"\n【{ticker}】")
        ticker_df = ticker_data[ticker]
        
        # SMA統計
        sma_stats = calculate_sma_statistics(ticker_df)
        if sma_stats is None:
            continue
        
        print(f"  SMA乖離率: 平均={sma_stats['mean']:.2f}%, 中央値={sma_stats['median']:.2f}%")
        print(f"  範囲別: 0-5%={sma_stats['range_0_5']}件, 5-10%={sma_stats['range_5_10']}件, 10%+={sma_stats['range_10_plus']}件")
        
        # フィルター適用前
        metrics_before = calculate_performance_metrics(ticker_df)
        
        # フィルター適用後（SMA乖離<5%のみ）
        filtered_df = apply_sma_filter(ticker_df, SMA_DISTANCE_THRESHOLD)
        metrics_after = calculate_performance_metrics(filtered_df)
        
        # 除外された取引（SMA乖離>=5%）
        sma_dist_abs = ticker_df['sma_distance_pct'].abs()
        excluded_df = ticker_df[sma_dist_abs >= SMA_DISTANCE_THRESHOLD].copy()
        metrics_excluded = calculate_performance_metrics(excluded_df)
        
        print(f"\n  フィルター前: PF={metrics_before['pf']:.2f}, 勝率={metrics_before['win_rate']*100:.1f}%, 取引数={metrics_before['total_trades']}")
        print(f"  フィルター後: PF={metrics_after['pf']:.2f}, 勝率={metrics_after['win_rate']*100:.1f}%, 取引数={metrics_after['total_trades']}")
        print(f"  除外取引: PF={metrics_excluded['pf']:.2f}, 勝率={metrics_excluded['win_rate']*100:.1f}%, 取引数={metrics_excluded['total_trades']}")
        
        # 改善率計算
        pf_improvement = 0.0
        if metrics_before['pf'] > 0:
            pf_improvement = ((metrics_after['pf'] - metrics_before['pf']) / metrics_before['pf']) * 100
        
        wr_improvement = (metrics_after['win_rate'] - metrics_before['win_rate']) * 100
        trade_reduction = ((metrics_before['total_trades'] - metrics_after['total_trades']) / metrics_before['total_trades'] * 100) if metrics_before['total_trades'] > 0 else 0.0
        
        print(f"  PF改善率: {pf_improvement:+.1f}%")
        print(f"  勝率改善: {wr_improvement:+.1f}pt")
        print(f"  取引削減率: {trade_reduction:.1f}%")
        
        # 結果記録
        results.append({
            'ticker': ticker,
            'before_pf': metrics_before['pf'],
            'after_pf': metrics_after['pf'],
            'pf_improvement': pf_improvement,
            'before_wr': metrics_before['win_rate'] * 100,
            'after_wr': metrics_after['win_rate'] * 100,
            'wr_improvement': wr_improvement,
            'before_trades': metrics_before['total_trades'],
            'after_trades': metrics_after['total_trades'],
            'trade_reduction': trade_reduction,
            'excluded_pf': metrics_excluded['pf'],
            'excluded_wr': metrics_excluded['win_rate'] * 100
        })
        
        # SMA範囲別成績
        ranges = {
            '0-5%': (0.0, 5.0),
            '5-10%': (5.0, 10.0),
            '10%+': (10.0, 999.0)
        }
        
        for range_name, (low, high) in ranges.items():
            range_df = ticker_df[(sma_dist_abs >= low) & (sma_dist_abs < high)]
            range_metrics = calculate_performance_metrics(range_df)
            
            sma_range_results.append({
                'ticker': ticker,
                'range': range_name,
                'trades': range_metrics['total_trades'],
                'pf': range_metrics['pf'],
                'win_rate': range_metrics['win_rate'] * 100,
                'avg_profit_pct': range_metrics['avg_profit_pct']
            })
    
    # 普遍性スコア算出
    print("\n" + "=" * 80)
    print("Step 3: 普遍性スコア算出")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    improved_count = len(results_df[results_df['pf_improvement'] > 10])
    total_count = len(results_df)
    universality_score = improved_count / total_count if total_count > 0 else 0.0
    
    print(f"\n改善銘柄数: {improved_count}/{total_count}銘柄（PF改善率>10%）")
    print(f"普遍性スコア: {universality_score:.2f}\n")
    
    # 改善銘柄リスト
    if improved_count > 0:
        print("改善銘柄:")
        improved_df = results_df[results_df['pf_improvement'] > 10].sort_values('pf_improvement', ascending=False)
        for _, row in improved_df.iterrows():
            print(f"  {row['ticker']}: PF改善率{row['pf_improvement']:+.1f}%")
    
    # 可視化
    print("\n" + "=" * 80)
    print("Step 4: 可視化")
    print("=" * 80)
    
    figures_dir = Path('docs/exit_strategy/figures')
    
    # 図表1: PF改善率比較棒グラフ
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    tickers = results_df['ticker'].tolist()
    pf_improvements = results_df['pf_improvement'].tolist()
    colors = ['green' if x > 10 else 'red' for x in pf_improvements]
    
    bars = ax1.bar(tickers, pf_improvements, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(10, color='orange', linestyle='--', linewidth=2, label='改善閾値 (+10%)')
    ax1.set_xlabel('Ticker')
    ax1.set_ylabel('PF改善率 (%)')
    ax1.set_title('MA乖離フィルター効果（PF改善率比較）')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure(fig1, 'phase1_9a_pf_improvement_comparison.png', figures_dir)
    
    # 図表2: フィルター前後PF比較
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tickers))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, results_df['before_pf'], width, label='フィルター前', color='skyblue', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, results_df['after_pf'], width, label='フィルター後', color='orange', alpha=0.7, edgecolor='black')
    
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1, label='損益分岐点 (PF=1.0)')
    ax2.set_xlabel('Ticker')
    ax2.set_ylabel('Profit Factor (PF)')
    ax2.set_title('MA乖離フィルター前後のPF比較')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_figure(fig2, 'phase1_9a_pf_before_after_comparison.png', figures_dir)
    
    # 図表3: 勝率改善比較
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    wr_improvements = results_df['wr_improvement'].tolist()
    colors_wr = ['green' if x > 0 else 'red' for x in wr_improvements]
    
    bars = ax3.bar(tickers, wr_improvements, color=colors_wr, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Ticker')
    ax3.set_ylabel('勝率改善 (pt)')
    ax3.set_title('MA乖離フィルター効果（勝率改善比較）')
    ax3.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure(fig3, 'phase1_9a_winrate_improvement_comparison.png', figures_dir)
    
    # 図表4: SMA範囲別成績ヒートマップ
    sma_range_df = pd.DataFrame(sma_range_results)
    pivot_pf = sma_range_df.pivot(index='ticker', columns='range', values='pf')
    
    fig4, ax4 = plt.subplots(figsize=(8, 10))
    im = ax4.imshow(pivot_pf.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
    ax4.set_xticks(np.arange(len(pivot_pf.columns)))
    ax4.set_yticks(np.arange(len(pivot_pf.index)))
    ax4.set_xticklabels(pivot_pf.columns)
    ax4.set_yticklabels(pivot_pf.index)
    ax4.set_title('SMA乖離範囲別PFヒートマップ')
    ax4.set_xlabel('SMA乖離範囲')
    ax4.set_ylabel('Ticker')
    
    for i in range(len(pivot_pf.index)):
        for j in range(len(pivot_pf.columns)):
            val = pivot_pf.values[i, j]
            if not np.isnan(val):
                ax4.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=9)
    
    plt.colorbar(im, ax=ax4, label='Profit Factor')
    plt.tight_layout()
    save_figure(fig4, 'phase1_9a_sma_range_heatmap.png', figures_dir)
    
    # 図表5: 取引数削減率
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    trade_reductions = results_df['trade_reduction'].tolist()
    
    bars = ax5.bar(tickers, trade_reductions, color='purple', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Ticker')
    ax5.set_ylabel('取引数削減率 (%)')
    ax5.set_title('MA乖離フィルターによる取引数削減')
    ax5.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure(fig5, 'phase1_9a_trade_reduction.png', figures_dir)
    
    # 結果レポート作成
    print("\n" + "=" * 80)
    print("Step 5: 結果レポート作成")
    print("=" * 80)
    
    result_md = []
    result_md.append("# Phase 1.9-A MA乖離フィルター検証結果\n")
    result_md.append(f"**作成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    result_md.append(f"**検証銘柄数**: {total_count}銘柄  ")
    result_md.append(f"**フィルター条件**: SMA乖離 < {SMA_DISTANCE_THRESHOLD}%  ")
    result_md.append(f"**参照元**: [PHASE1_9_FILTERS_DESIGN.md](PHASE1_9_FILTERS_DESIGN.md)\n")
    result_md.append("---\n\n")
    
    result_md.append("## 1. 銘柄別フィルター効果サマリー\n\n")
    result_md.append("| Ticker | Before PF | After PF | PF Improvement (%) | Before WR (%) | After WR (%) | WR Improvement (pt) | Trades Before | Trades After | Trade Reduction (%) |\n")
    result_md.append("|--------|-----------|----------|-------------------|---------------|--------------|---------------------|---------------|--------------|--------------------|\n")
    
    for _, row in results_df.iterrows():
        result_md.append(f"| {row['ticker']} | {row['before_pf']:.2f} | {row['after_pf']:.2f} | {row['pf_improvement']:+.1f} | {row['before_wr']:.1f} | {row['after_wr']:.1f} | {row['wr_improvement']:+.1f} | {row['before_trades']} | {row['after_trades']} | {row['trade_reduction']:.1f} |\n")
    
    result_md.append("\n**Figure**: [PF Improvement Comparison](figures/phase1_9a_pf_improvement_comparison.png)\n\n")
    
    result_md.append("## 2. 普遍性評価\n\n")
    result_md.append(f"- **普遍性スコア**: {universality_score:.2f} ({improved_count}/{total_count}銘柄で改善)\n")
    result_md.append(f"- **改善基準**: PF改善率 > 10%\n\n")
    
    if improved_count > 0:
        result_md.append("### 改善銘柄リスト\n\n")
        for _, row in improved_df.iterrows():
            result_md.append(f"- **{row['ticker']}**: PF改善率{row['pf_improvement']:+.1f}%\n")
        result_md.append("\n")
    
    result_md.append("**Figure**: [PF Before/After Comparison](figures/phase1_9a_pf_before_after_comparison.png)\n\n")
    
    result_md.append("## 3. SMA乖離範囲別成績\n\n")
    result_md.append("| Ticker | 0-5% PF | 5-10% PF | 10%+ PF | 0-5% WR (%) | 5-10% WR (%) | 10%+ WR (%) |\n")
    result_md.append("|--------|---------|----------|---------|-------------|--------------|-------------|\n")
    
    for ticker in tickers:
        ticker_ranges = sma_range_df[sma_range_df['ticker'] == ticker]
        range_0_5 = ticker_ranges[ticker_ranges['range'] == '0-5%'].iloc[0] if len(ticker_ranges[ticker_ranges['range'] == '0-5%']) > 0 else None
        range_5_10 = ticker_ranges[ticker_ranges['range'] == '5-10%'].iloc[0] if len(ticker_ranges[ticker_ranges['range'] == '5-10%']) > 0 else None
        range_10_plus = ticker_ranges[ticker_ranges['range'] == '10%+'].iloc[0] if len(ticker_ranges[ticker_ranges['range'] == '10%+']) > 0 else None
        
        pf_0_5 = range_0_5['pf'] if range_0_5 is not None else 0.0
        pf_5_10 = range_5_10['pf'] if range_5_10 is not None else 0.0
        pf_10_plus = range_10_plus['pf'] if range_10_plus is not None else 0.0
        
        wr_0_5 = range_0_5['win_rate'] if range_0_5 is not None else 0.0
        wr_5_10 = range_5_10['win_rate'] if range_5_10 is not None else 0.0
        wr_10_plus = range_10_plus['win_rate'] if range_10_plus is not None else 0.0
        
        result_md.append(f"| {ticker} | {pf_0_5:.2f} | {pf_5_10:.2f} | {pf_10_plus:.2f} | {wr_0_5:.1f} | {wr_5_10:.1f} | {wr_10_plus:.1f} |\n")
    
    result_md.append("\n**Figure**: [SMA Range Heatmap](figures/phase1_9a_sma_range_heatmap.png)\n\n")
    
    result_md.append("## 4. 結論と次フェーズ提案\n\n")
    result_md.append("### 主要発見事項\n\n")
    
    if universality_score >= 0.50:
        result_md.append(f"- MA乖離フィルターは**普遍的に有効**（普遍性スコア{universality_score:.2f}）\n")
    elif universality_score >= 0.30:
        result_md.append(f"- MA乖離フィルターは**部分的に有効**（普遍性スコア{universality_score:.2f}）\n")
    else:
        result_md.append(f"- MA乖離フィルターは**効果が限定的**（普遍性スコア{universality_score:.2f}）\n")
    
    result_md.append(f"- 取引数の平均削減率: {results_df['trade_reduction'].mean():.1f}%\n")
    result_md.append(f"- PF平均改善率: {results_df['pf_improvement'].mean():+.1f}%\n\n")
    
    result_md.append("### Phase 1.9-B提案\n\n")
    result_md.append("次フェーズでは、以下を実施：\n\n")
    result_md.append("1. **ATR範囲フィルター検証**: 低ATR禁止の効果を確認\n")
    result_md.append("2. **ATR利確ルール検証**: 利確ライン = entry_price + (entry_atr × 10)の効果を確認\n")
    result_md.append("3. **複合フィルター準備**: MA乖離 + ATRのAND/OR組み合わせ検討\n\n")
    
    result_md.append("---\n\n")
    result_md.append("**作成者**: validate_phase1_9_sma_filter.py  \n")
    result_md.append(f"**最終更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # レポート保存
    report_path = Path('docs/exit_strategy/PHASE1_9A_SMA_FILTER_RESULT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(result_md)
    
    print(f"\nレポート保存: {report_path}")
    print("\n" + "=" * 80)
    print("Phase 1.9-A MA乖離フィルター検証完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
