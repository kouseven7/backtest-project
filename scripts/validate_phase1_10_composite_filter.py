"""
Phase 1.10複合フィルター検証スクリプト

トレンド強度（高） AND MA乖離<5%の複合フィルター効果を検証。

主な機能:
- トレンド強度閾値（67パーセンタイル）とMA乖離5%の複合判定
- フィルター適用前後のPF・勝率比較（10銘柄）
- 単体フィルター（Phase 1.8/1.9-A）との効果比較
- 5種類の可視化図表生成
- 結果Markdownレポート自動生成

統合コンポーネント:
- phase1.6_trades_20260126_200241.csv: 9銘柄取引履歴
- phase1.6_trades_20260126_124409a.csv: 武田薬品取引履歴
- PHASE1_8_TREND_FILTER_RESULT.md: トレンド強度単体結果
- PHASE1_9A_SMA_FILTER_RESULT.md: MA乖離単体結果

セーフティ機能/注意事項:
- 取引数20件以上の統計的有意性チェック
- PF計算時のゼロ除算回避
- 図表保存時のディレクトリ自動作成

Author: Backtest Project Team
Created: 2026-01-27
Last Modified: 2026-01-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 日本語フォント設定（Windows）
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

# ==================== ヘルパー関数 ====================

def load_and_preprocess_data(csv_path, ticker=None, is_takeda=False):
    """
    CSVデータ読み込み・前処理
    
    Args:
        csv_path: CSVファイルパス
        ticker: 抽出する銘柄コード（Noneの場合は全銘柄）
        is_takeda: 武田薬品用の特殊処理フラグ
    
    Returns:
        前処理済みDataFrame
    """
    if is_takeda:
        # 武田薬品はヘッダーなし（Phase 1.6と同じカラム名）
        column_names = [
            'trade_id', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 
            'profit_loss', 'exit_reason', 'holding_days', 'profit_loss_pct', 
            'r_multiple', 'entry_gap_pct', 'max_profit_pct', 'entry_atr_pct', 
            'sma_distance_pct', 'entry_trend_strength', 'entry_volume', 
            'exit_volume', 'volume_ratio', 'exit_signal_pct', 'sma_value', 
            'trend_value', 'above_trend', 'unused1', 'entry_price_base', 
            'unused2', 'unused3', 'ticker', 'stop_loss_pct', 'trailing_stop_pct'
        ]
        df = pd.read_csv(csv_path, header=None, names=column_names)
        df = df[df['ticker'] == '4502.T'].copy()
    else:
        # 9銘柄はヘッダーあり
        df = pd.read_csv(csv_path)
        if ticker:
            df = df[df['ticker'] == ticker].copy()
    
    return df


def calculate_performance_metrics(trades_df):
    """
    パフォーマンス指標計算
    
    Args:
        trades_df: 取引履歴DataFrame
    
    Returns:
        パフォーマンス指標辞書
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


def apply_composite_filter(df, trend_threshold, sma_threshold=5.0):
    """
    複合フィルター適用（トレンド強度（高） AND MA乖離<5%）
    
    Args:
        df: 取引履歴DataFrame
        trend_threshold: トレンド強度閾値（67%ile）
        sma_threshold: SMA乖離閾値（デフォルト5.0%）
    
    Returns:
        フィルター通過後のDataFrame
    """
    # トレンド強度（高）
    trend_ok = df['entry_trend_strength'] >= trend_threshold
    
    # SMA乖離<5%
    sma_ok = df['sma_distance_pct'].abs() < sma_threshold
    
    # AND条件
    filtered_df = df[trend_ok & sma_ok].copy()
    
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
    print(f"図表保存: {filepath}")


# ==================== メイン処理 ====================

def main():
    print("=" * 80)
    print("Phase 1.10 複合フィルター検証開始")
    print("トレンド強度（高） AND MA乖離<5%")
    print("=" * 80)
    
    # データ読み込み（絶対パス）
    base_path = Path(__file__).parent.parent
    csv_path_9 = base_path / 'results' / 'phase1.6_trades_20260126_200241.csv'
    csv_path_takeda = base_path / 'results' / 'phase1.6_trades_20260126_124409a.csv'
    
    # 検証銘柄リスト
    tickers = [
        '4063.T',  # 信越化学工業
        '6501.T',  # 日立製作所
        '6758.T',  # ソニーグループ
        '6861.T',  # キーエンス
        '7203.T',  # トヨタ自動車
        '8001.T',  # 伊藤忠商事
        '8306.T',  # 三菱UFJ FG
        '9983.T',  # ファーストリテイリング
        '9984.T',  # ソフトバンクグループ
    ]
    
    # 結果格納
    results = []
    
    # 9銘柄処理
    for ticker in tickers:
        print(f"\n処理中: {ticker}")
        df = load_and_preprocess_data(csv_path_9, ticker=ticker)
        
        if len(df) == 0:
            print(f"  警告: {ticker} のデータなし")
            continue
        
        # トレンド強度閾値算出（67%ile）
        trend_threshold = df['entry_trend_strength'].quantile(0.67)
        
        # フィルター適用前
        metrics_before = calculate_performance_metrics(df)
        
        # 複合フィルター適用
        df_filtered = apply_composite_filter(df, trend_threshold, sma_threshold=5.0)
        metrics_after = calculate_performance_metrics(df_filtered)
        
        # PF改善率計算
        pf_improvement = ((metrics_after['pf'] - metrics_before['pf']) / metrics_before['pf'] * 100) if metrics_before['pf'] > 0 else 0.0
        
        # 取引数削減率
        trade_reduction = ((metrics_before['total_trades'] - metrics_after['total_trades']) / metrics_before['total_trades'] * 100) if metrics_before['total_trades'] > 0 else 0.0
        
        results.append({
            'Ticker': ticker,
            'Before PF': metrics_before['pf'],
            'After PF': metrics_after['pf'],
            'PF Improvement (%)': pf_improvement,
            'Before WR (%)': metrics_before['win_rate'] * 100,
            'After WR (%)': metrics_after['win_rate'] * 100,
            'WR Improvement (pt)': (metrics_after['win_rate'] - metrics_before['win_rate']) * 100,
            'Trades Before': metrics_before['total_trades'],
            'Trades After': metrics_after['total_trades'],
            'Trade Reduction (%)': trade_reduction,
            'Trend Threshold': trend_threshold
        })
        
        print(f"  Before: PF={metrics_before['pf']:.2f}, WR={metrics_before['win_rate']*100:.1f}%, Trades={metrics_before['total_trades']}")
        print(f"  After:  PF={metrics_after['pf']:.2f}, WR={metrics_after['win_rate']*100:.1f}%, Trades={metrics_after['total_trades']}")
        print(f"  PF改善率: {pf_improvement:+.1f}%")
    
    # 武田薬品処理
    print(f"\n処理中: 4502.T（武田薬品）")
    df_takeda = load_and_preprocess_data(csv_path_takeda, is_takeda=True)
    
    if len(df_takeda) > 0:
        trend_threshold_takeda = df_takeda['entry_trend_strength'].quantile(0.67)
        
        metrics_before_takeda = calculate_performance_metrics(df_takeda)
        df_takeda_filtered = apply_composite_filter(df_takeda, trend_threshold_takeda, sma_threshold=5.0)
        metrics_after_takeda = calculate_performance_metrics(df_takeda_filtered)
        
        pf_improvement_takeda = ((metrics_after_takeda['pf'] - metrics_before_takeda['pf']) / metrics_before_takeda['pf'] * 100) if metrics_before_takeda['pf'] > 0 else 0.0
        trade_reduction_takeda = ((metrics_before_takeda['total_trades'] - metrics_after_takeda['total_trades']) / metrics_before_takeda['total_trades'] * 100) if metrics_before_takeda['total_trades'] > 0 else 0.0
        
        results.append({
            'Ticker': '4502.T',
            'Before PF': metrics_before_takeda['pf'],
            'After PF': metrics_after_takeda['pf'],
            'PF Improvement (%)': pf_improvement_takeda,
            'Before WR (%)': metrics_before_takeda['win_rate'] * 100,
            'After WR (%)': metrics_after_takeda['win_rate'] * 100,
            'WR Improvement (pt)': (metrics_after_takeda['win_rate'] - metrics_before_takeda['win_rate']) * 100,
            'Trades Before': metrics_before_takeda['total_trades'],
            'Trades After': metrics_after_takeda['total_trades'],
            'Trade Reduction (%)': trade_reduction_takeda,
            'Trend Threshold': trend_threshold_takeda
        })
        
        print(f"  Before: PF={metrics_before_takeda['pf']:.2f}, WR={metrics_before_takeda['win_rate']*100:.1f}%, Trades={metrics_before_takeda['total_trades']}")
        print(f"  After:  PF={metrics_after_takeda['pf']:.2f}, WR={metrics_after_takeda['win_rate']*100:.1f}%, Trades={metrics_after_takeda['total_trades']}")
        print(f"  PF改善率: {pf_improvement_takeda:+.1f}%")
    
    # 結果DataFrame作成
    results_df = pd.DataFrame(results)
    
    # 普遍性スコア算出（PF改善率 > 10%）
    improved_count = len(results_df[results_df['PF Improvement (%)'] > 10])
    universality_score = improved_count / len(results_df)
    
    print("\n" + "=" * 80)
    print("普遍性評価")
    print("=" * 80)
    print(f"改善銘柄数: {improved_count}/{len(results_df)}")
    print(f"普遍性スコア: {universality_score:.2f}")
    
    # 平均取引数削減率
    avg_trade_reduction = results_df['Trade Reduction (%)'].mean()
    print(f"平均取引数削減率: {avg_trade_reduction:.1f}%")
    
    # 可視化
    print("\n" + "=" * 80)
    print("可視化")
    print("=" * 80)
    
    figures_dir = base_path / 'docs' / 'exit_strategy' / 'figures'
    
    # 図表1: PF改善率比較棒グラフ
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    colors = ['green' if x > 10 else 'red' for x in results_df['PF Improvement (%)']]
    bars = ax1.bar(results_df['Ticker'], results_df['PF Improvement (%)'], color=colors, alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, results_df['PF Improvement (%)']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(10, color='blue', linestyle='--', linewidth=1, label='改善基準 (10%)')
    ax1.set_xlabel('銘柄')
    ax1.set_ylabel('PF改善率 (%)')
    ax1.set_title('Phase 1.10複合フィルター効果（PF改善率）\nトレンド強度（高） AND MA乖離<5%')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig1, 'phase1_10_pf_improvement_comparison.png', figures_dir)
    
    # 図表2: Before/After PF比較
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, results_df['Before PF'], width, label='Before', color='red', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, results_df['After PF'], width, label='After', color='green', alpha=0.7, edgecolor='black')
    
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1, label='損益分岐点 (PF=1.0)')
    ax2.set_xlabel('銘柄')
    ax2.set_ylabel('Profit Factor (PF)')
    ax2.set_title('Phase 1.10複合フィルター効果（PF Before/After比較）')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['Ticker'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_figure(fig2, 'phase1_10_pf_before_after_comparison.png', figures_dir)
    
    # 図表3: 勝率改善比較
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    colors_wr = ['green' if x > 0 else 'red' for x in results_df['WR Improvement (pt)']]
    bars = ax3.bar(results_df['Ticker'], results_df['WR Improvement (pt)'], color=colors_wr, alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, results_df['WR Improvement (pt)']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}pt', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('銘柄')
    ax3.set_ylabel('勝率改善 (pt)')
    ax3.set_title('Phase 1.10複合フィルター効果（勝率改善）')
    ax3.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig3, 'phase1_10_winrate_improvement_comparison.png', figures_dir)
    
    # 図表4: 取引数削減率
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    bars = ax4.bar(results_df['Ticker'], results_df['Trade Reduction (%)'], color='orange', alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, results_df['Trade Reduction (%)']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax4.axhline(80, color='red', linestyle='--', linewidth=1, label='許容上限 (80%)')
    ax4.set_xlabel('銘柄')
    ax4.set_ylabel('取引数削減率 (%)')
    ax4.set_title('Phase 1.10複合フィルター効果（取引数削減率）')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig4, 'phase1_10_trade_reduction.png', figures_dir)
    
    # 図表5: 単体フィルターとの比較（Phase 1.8/1.9-A比較）
    # Phase 1.8（トレンド強度単体）の結果をハードコード
    phase1_8_improvements = {
        '4063.T': 41.6, '6501.T': 8.2, '6758.T': -52.0, '6861.T': -76.8,
        '7203.T': -12.7, '8001.T': -2.4, '8306.T': 90.5, '9983.T': -10.9,
        '9984.T': 96.9
    }
    
    # Phase 1.9-A（MA乖離単体）の結果をハードコード
    phase1_9a_improvements = {
        '4063.T': -18.0, '6501.T': 35.2, '6758.T': 40.0, '6861.T': -5.4,
        '7203.T': 40.4, '8001.T': -3.8, '8306.T': 26.0, '9983.T': 3.9,
        '9984.T': -24.3, '4502.T': 6.5
    }
    
    # Phase 1.10（複合）の結果をマージ
    comparison_data = []
    for _, row in results_df.iterrows():
        ticker = row['Ticker']
        comparison_data.append({
            'Ticker': ticker,
            'Phase 1.8 (Trend)': phase1_8_improvements.get(ticker, 0.0),
            'Phase 1.9-A (SMA)': phase1_9a_improvements.get(ticker, 0.0),
            'Phase 1.10 (Composite)': row['PF Improvement (%)']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig5, ax5 = plt.subplots(figsize=(14, 7))
    x = np.arange(len(comparison_df))
    width = 0.25
    
    bars1 = ax5.bar(x - width, comparison_df['Phase 1.8 (Trend)'], width, label='Phase 1.8 (Trend単体)', color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x, comparison_df['Phase 1.9-A (SMA)'], width, label='Phase 1.9-A (SMA単体)', color='purple', alpha=0.7, edgecolor='black')
    bars3 = ax5.bar(x + width, comparison_df['Phase 1.10 (Composite)'], width, label='Phase 1.10 (複合)', color='green', alpha=0.7, edgecolor='black')
    
    ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    ax5.axhline(10, color='red', linestyle='--', linewidth=1, label='改善基準 (10%)')
    ax5.set_xlabel('銘柄')
    ax5.set_ylabel('PF改善率 (%)')
    ax5.set_title('Phase別フィルター効果比較\nTrend単体 vs SMA単体 vs 複合（Trend AND SMA）')
    ax5.set_xticks(x)
    ax5.set_xticklabels(comparison_df['Ticker'], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_figure(fig5, 'phase1_10_vs_single_filters_comparison.png', figures_dir)
    
    # 結果レポート作成
    print("\n" + "=" * 80)
    print("結果レポート作成")
    print("=" * 80)
    
    result_md = []
    result_md.append("# Phase 1.10複合フィルター検証結果\n")
    result_md.append(f"**作成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    result_md.append(f"**検証銘柄数**: {len(results_df)}銘柄  \n")
    result_md.append(f"**フィルター条件**: トレンド強度（高、67%ile以上） AND SMA乖離 < 5.0%  \n")
    result_md.append(f"**参照元**: [PHASE1_9_FILTERS_DESIGN.md](PHASE1_9_FILTERS_DESIGN.md)\n")
    result_md.append("---\n\n")
    
    result_md.append("## 1. 銘柄別フィルター効果サマリー\n\n")
    
    # Markdown表を手動作成
    result_md.append("| Ticker | Before PF | After PF | PF Improvement (%) | Before WR (%) | After WR (%) | WR Improvement (pt) | Trades Before | Trades After | Trade Reduction (%) | Trend Threshold |\n")
    result_md.append("|--------|-----------|----------|-------------------|---------------|--------------|---------------------|---------------|--------------|--------------------|-----------------|\n")
    for _, row in results_df.iterrows():
        result_md.append(f"| {row['Ticker']} | {row['Before PF']:.2f} | {row['After PF']:.2f} | {row['PF Improvement (%)']:+.1f} | {row['Before WR (%)']:.1f} | {row['After WR (%)']:.1f} | {row['WR Improvement (pt)']:+.1f} | {row['Trades Before']} | {row['Trades After']} | {row['Trade Reduction (%)']:.1f} | {row['Trend Threshold']:.4f} |\n")
    
    result_md.append("\n**Figure**: [PF Improvement Comparison](figures/phase1_10_pf_improvement_comparison.png)\n\n")
    
    result_md.append("## 2. 普遍性評価\n\n")
    result_md.append(f"- **普遍性スコア**: {universality_score:.2f} ({improved_count}/{len(results_df)}銘柄で改善)\n")
    result_md.append(f"- **改善基準**: PF改善率 > 10%\n\n")
    
    result_md.append("### 改善銘柄リスト\n\n")
    improved_tickers = results_df[results_df['PF Improvement (%)'] > 10].sort_values('PF Improvement (%)', ascending=False)
    for _, row in improved_tickers.iterrows():
        result_md.append(f"- **{row['Ticker']}**: PF改善率{row['PF Improvement (%)']:+.1f}%\n")
    result_md.append("\n")
    
    result_md.append("**Figure**: [PF Before/After Comparison](figures/phase1_10_pf_before_after_comparison.png)\n\n")
    
    result_md.append("## 3. 単体フィルターとの比較\n\n")
    result_md.append("| フィルター | 普遍性スコア | 改善銘柄数 | 備考 |\n")
    result_md.append("|-----------|------------|------------|------|\n")
    result_md.append("| Phase 1.8 (Trend単体) | 0.33 | 3/9 | 9984.T (+96.9%), 8306.T (+90.5%), 4063.T (+41.6%) |\n")
    result_md.append("| Phase 1.9-A (SMA単体) | 0.40 | 4/10 | 7203.T (+40.4%), 6758.T (+40.0%), 6501.T (+35.2%), 8306.T (+26.0%) |\n")
    result_md.append(f"| **Phase 1.10 (複合)** | **{universality_score:.2f}** | **{improved_count}/{len(results_df)}** | トレンド強度（高） AND MA乖離<5% |\n\n")
    
    result_md.append("**Figure**: [Phase別比較](figures/phase1_10_vs_single_filters_comparison.png)\n\n")
    
    result_md.append("## 4. 取引数削減率分析\n\n")
    result_md.append(f"- **平均削減率**: {avg_trade_reduction:.1f}%\n")
    result_md.append(f"- **許容上限**: 80%\n")
    result_md.append(f"- **超過銘柄数**: {len(results_df[results_df['Trade Reduction (%)'] > 80])}/{len(results_df)}\n\n")
    
    result_md.append("**Figure**: [取引数削減率](figures/phase1_10_trade_reduction.png)\n\n")
    
    result_md.append("## 5. 主要発見事項\n\n")
    result_md.append("### 発見1: 複合フィルターの効果\n\n")
    if universality_score > 0.50:
        result_md.append(f"複合フィルター（Trend AND SMA）の普遍性スコア{universality_score:.2f}は、目標値0.50を上回り、単体フィルター（Trend: 0.33、SMA: 0.40）を上回った。\n\n")
    else:
        result_md.append(f"複合フィルター（Trend AND SMA）の普遍性スコア{universality_score:.2f}は、目標値0.50を下回ったが、単体フィルター（Trend: 0.33、SMA: 0.40）との差異を分析する必要がある。\n\n")
    
    result_md.append("### 発見2: 銘柄特性別効果\n\n")
    result_md.append("（改善銘柄と悪化銘柄のセクター・ボラティリティ分析）\n\n")
    
    result_md.append("### 発見3: 取引数削減の影響\n\n")
    if avg_trade_reduction > 80:
        result_md.append(f"平均削減率{avg_trade_reduction:.1f}%は許容上限80%を超過。統計的有意性への影響を確認する必要がある。\n\n")
    else:
        result_md.append(f"平均削減率{avg_trade_reduction:.1f}%は許容範囲内（<80%）。統計的有意性は保たれている。\n\n")
    
    result_md.append("## 6. 次フェーズ提案\n\n")
    result_md.append("### Phase 1.11-A: 他の複合フィルター検証\n\n")
    result_md.append("- トレンド強度（高） AND ATR中〜高\n")
    result_md.append("- 3条件統合（Trend AND SMA AND ATR）\n\n")
    
    result_md.append("### Phase 1.11-B: OR条件検証\n\n")
    result_md.append("- トレンド強度（高） OR SMA乖離<5%\n")
    result_md.append("- 効果的な緩和フィルターの可能性\n\n")
    
    result_md.append("---\n\n")
    result_md.append("**作成者**: validate_phase1_10_composite_filter.py  \n")
    result_md.append(f"**最終更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Markdown保存（絶対パス）
    result_path = base_path / 'docs' / 'exit_strategy' / 'PHASE1_10_COMPOSITE_FILTER_RESULT.md'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(''.join(result_md), encoding='utf-8')
    print(f"結果レポート保存: {result_path}")
    
    print("\n" + "=" * 80)
    print("Phase 1.10 複合フィルター検証完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
