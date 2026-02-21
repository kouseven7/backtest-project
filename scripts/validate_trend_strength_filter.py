"""
トレンド強度フィルター検証スクリプト

Phase 1.6大敗パターン分析結果に基づき、トレンド強度フィルターの効果を検証。

主な機能:
- トレンド強度閾値（67パーセンタイル）算出
- フィルター適用前後のPF・勝率比較
- 急騰→急落パターンへの影響分析
- 3種類の可視化図表生成
- 結果Markdownレポート自動生成

統合コンポーネント:
- phase1.6_trades_20260126_124409a.csv: 武田薬品取引履歴
- PHASE1_6_DEFEAT_PATTERNS_RESULT.md: 分析結果参照元

セーフティ機能/注意事項:
- 取引数20件以上の統計的有意性チェック
- PF計算時のゼロ除算回避
- 図表保存時のディレクトリ自動作成

Author: Backtest Project Team
Created: 2026-01-26
Last Modified: 2026-01-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# 日本語フォント設定（Windows）
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

# ==================== ヘルパー関数 ====================

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


def detect_pump_dump(trades_df):
    """
    急騰→急落パターン検出
    
    Args:
        trades_df: 取引履歴DataFrame
    
    Returns:
        該当取引のDataFrame
    """
    pattern = trades_df[
        (trades_df['max_profit_pct'] > 15) &
        (trades_df['profit_loss_pct'] < 0) &
        (trades_df['holding_days'] < 30)
    ]
    return pattern


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
    print("トレンド強度フィルター検証開始")
    print("=" * 80)
    
    # データ読み込み
    csv_path = Path('results/phase1.6_trades_20260126_124409a.csv')
    if not csv_path.exists():
        print(f"エラー: {csv_path} が見つかりません")
        return
    
    # ヘッダーなしCSVを読み込み、カラム名を手動設定（analyze_phase1_6_defeat_patterns.pyと同じ）
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
    
    # 武田薬品のみ抽出
    takeda_df = df[df['ticker'] == '4502.T'].copy()
    print(f"武田薬品取引数: {len(takeda_df)}件\n")
    
    # Step 1: トレンド強度閾値の算出
    print("=" * 80)
    print("Step 1: トレンド強度閾値の算出")
    print("=" * 80)
    
    threshold_high = takeda_df['entry_trend_strength'].quantile(0.67)
    threshold_mid = takeda_df['entry_trend_strength'].quantile(0.33)
    
    print(f"トレンド強度閾値:")
    print(f"  低/中境界（33%ile）: {threshold_mid:.4f}")
    print(f"  中/高境界（67%ile）: {threshold_high:.4f}")
    
    # 分類確認
    high_count = len(takeda_df[takeda_df['entry_trend_strength'] >= threshold_high])
    mid_count = len(takeda_df[(takeda_df['entry_trend_strength'] >= threshold_mid) & 
                              (takeda_df['entry_trend_strength'] < threshold_high)])
    low_count = len(takeda_df[takeda_df['entry_trend_strength'] < threshold_mid])
    
    print(f"\n分類確認:")
    print(f"  高: {high_count}件")
    print(f"  中: {mid_count}件")
    print(f"  低: {low_count}件\n")
    
    # Step 2: フィルター適用と成績比較
    print("=" * 80)
    print("Step 2: フィルター適用と成績比較")
    print("=" * 80)
    
    # フィルター適用前
    metrics_before = calculate_performance_metrics(takeda_df)
    
    # フィルター適用後（高トレンドのみ）
    takeda_high_trend = takeda_df[takeda_df['entry_trend_strength'] >= threshold_high].copy()
    metrics_after = calculate_performance_metrics(takeda_high_trend)
    
    print("\n=== フィルター適用前 ===")
    print(f"総取引数: {metrics_before['total_trades']}件")
    print(f"勝ち取引数: {metrics_before['winning_trades']}件")
    print(f"勝率: {metrics_before['win_rate']*100:.2f}%")
    print(f"PF: {metrics_before['pf']:.2f}")
    print(f"平均損益率: {metrics_before['avg_profit_pct']:.2f}%")
    print(f"平均R倍: {metrics_before['avg_r_multiple']:.2f}")
    
    print("\n=== フィルター適用後（高トレンドのみ） ===")
    print(f"総取引数: {metrics_after['total_trades']}件")
    print(f"勝ち取引数: {metrics_after['winning_trades']}件")
    print(f"勝率: {metrics_after['win_rate']*100:.2f}%")
    print(f"PF: {metrics_after['pf']:.2f}")
    print(f"平均損益率: {metrics_after['avg_profit_pct']:.2f}%")
    print(f"平均R倍: {metrics_after['avg_r_multiple']:.2f}")
    
    # 改善率計算
    if metrics_before['pf'] > 0:
        pf_improvement = (metrics_after['pf'] - metrics_before['pf']) / metrics_before['pf'] * 100
        print(f"\nPF改善率: {pf_improvement:+.1f}%")
        win_rate_improvement = (metrics_after['win_rate'] - metrics_before['win_rate']) * 100
        print(f"勝率改善: {win_rate_improvement:+.1f}ポイント\n")
    
    # Step 3: エグジット理由別分析
    print("=" * 80)
    print("Step 3: エグジット理由別分析")
    print("=" * 80)
    
    print("\n【フィルター前】")
    exit_reason_before = takeda_df['exit_reason'].value_counts()
    print(exit_reason_before)
    
    print("\n【フィルター後（高トレンドのみ）】")
    exit_reason_after = takeda_high_trend['exit_reason'].value_counts()
    print(exit_reason_after)
    print()
    
    # Step 4: 急騰→急落パターンへの影響
    print("=" * 80)
    print("Step 4: 急騰→急落パターンへの影響")
    print("=" * 80)
    
    pump_dump_before = detect_pump_dump(takeda_df)
    pump_dump_after = detect_pump_dump(takeda_high_trend)
    
    print(f"\nフィルター前: {len(pump_dump_before)}件 ({len(pump_dump_before)/len(takeda_df)*100:.1f}%)" if len(takeda_df) > 0 else "フィルター前: 0件")
    print(f"フィルター後: {len(pump_dump_after)}件 ({len(pump_dump_after)/len(takeda_high_trend)*100:.1f}%)" if len(takeda_high_trend) > 0 else "フィルター後: 0件")
    reduction = len(pump_dump_before) - len(pump_dump_after)
    reduction_rate = (reduction / len(pump_dump_before) * 100) if len(pump_dump_before) > 0 else 0.0
    print(f"削減: {reduction}件 ({reduction_rate:.1f}%削減)\n")
    
    # Step 5: 可視化
    print("=" * 80)
    print("Step 5: 可視化")
    print("=" * 80)
    
    figures_dir = Path('docs/exit_strategy/figures')
    
    # 図表1: トレンド強度分布ヒストグラム
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(takeda_df['entry_trend_strength'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(threshold_high, color='red', linestyle='--', linewidth=2, label=f'高閾値 ({threshold_high:.4f})')
    ax1.axvline(threshold_mid, color='orange', linestyle='--', linewidth=2, label=f'中閾値 ({threshold_mid:.4f})')
    ax1.set_xlabel('Entry Trend Strength')
    ax1.set_ylabel('取引数')
    ax1.set_title('トレンド強度分布と閾値')
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig1, 'trend_strength_distribution.png', figures_dir)
    
    # 図表2: フィルター前後のPF比較棒グラフ
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    categories = ['フィルター前', 'フィルター後\n(高トレンドのみ)']
    pf_values = [metrics_before['pf'], metrics_after['pf']]
    bars = ax2.bar(categories, pf_values, color=['red', 'green'], alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, pf_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Profit Factor (PF)')
    ax2.set_title('トレンド強度フィルター効果（PF比較）')
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1, label='損益分岐点 (PF=1.0)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_figure(fig2, 'trend_filter_pf_comparison.png', figures_dir)
    
    # 図表3: 勝率比較棒グラフ
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    win_rates = [metrics_before['win_rate']*100, metrics_after['win_rate']*100]
    bars = ax3.bar(categories, win_rates, color=['orange', 'skyblue'], alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, win_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax3.set_ylabel('勝率 (%)')
    ax3.set_title('トレンド強度フィルター効果（勝率比較）')
    ax3.set_ylim(0, 80)
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_figure(fig3, 'trend_filter_winrate_comparison.png', figures_dir)
    
    # Step 6: 結果レポート作成
    print("\n" + "=" * 80)
    print("Step 6: 結果レポート作成")
    print("=" * 80)
    
    result_md = []
    result_md.append("# Phase 1.6トレンド強度フィルター検証結果\n")
    result_md.append(f"**作成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    result_md.append(f"**対象銘柄**: 武田薬品工業（4502.T）  ")
    result_md.append(f"**参照元**: [PHASE1_6_DEFEAT_PATTERNS_RESULT.md](PHASE1_6_DEFEAT_PATTERNS_RESULT.md)\n")
    result_md.append("---\n\n")
    
    result_md.append("## 1. 検証目的\n\n")
    result_md.append("トレンド強度「高」のみでエントリーするフィルターを適用し、PF改善効果を検証。\n\n")
    
    result_md.append("## 2. トレンド強度閾値\n\n")
    result_md.append(f"- **低/中境界（33%ile）**: {threshold_mid:.4f}\n")
    result_md.append(f"- **中/高境界（67%ile）**: {threshold_high:.4f}\n\n")
    result_md.append("### 分類確認\n\n")
    result_md.append(f"- 高: {high_count}件\n")
    result_md.append(f"- 中: {mid_count}件\n")
    result_md.append(f"- 低: {low_count}件\n\n")
    
    result_md.append("## 3. フィルター適用前後の比較\n\n")
    result_md.append("### フィルター適用前（全取引）\n\n")
    result_md.append(f"- 総取引数: {metrics_before['total_trades']}件\n")
    result_md.append(f"- 勝ち取引数: {metrics_before['winning_trades']}件\n")
    result_md.append(f"- 勝率: {metrics_before['win_rate']*100:.2f}%\n")
    result_md.append(f"- PF: {metrics_before['pf']:.2f}\n")
    result_md.append(f"- 平均損益率: {metrics_before['avg_profit_pct']:.2f}%\n")
    result_md.append(f"- 平均R倍: {metrics_before['avg_r_multiple']:.2f}\n\n")
    
    result_md.append("### フィルター適用後（高トレンドのみ）\n\n")
    result_md.append(f"- 総取引数: {metrics_after['total_trades']}件\n")
    result_md.append(f"- 勝ち取引数: {metrics_after['winning_trades']}件\n")
    result_md.append(f"- 勝率: {metrics_after['win_rate']*100:.2f}%\n")
    result_md.append(f"- PF: {metrics_after['pf']:.2f}\n")
    result_md.append(f"- 平均損益率: {metrics_after['avg_profit_pct']:.2f}%\n")
    result_md.append(f"- 平均R倍: {metrics_after['avg_r_multiple']:.2f}\n\n")
    
    result_md.append("### 改善効果\n\n")
    if metrics_before['pf'] > 0:
        result_md.append(f"- **PF改善率**: {pf_improvement:+.1f}%\n")
        result_md.append(f"- **勝率改善**: {win_rate_improvement:+.1f}ポイント\n\n")
    
    result_md.append("**図表**:\n")
    result_md.append("- [トレンド強度分布](figures/trend_strength_distribution.png)\n")
    result_md.append("- [PF比較](figures/trend_filter_pf_comparison.png)\n")
    result_md.append("- [勝率比較](figures/trend_filter_winrate_comparison.png)\n\n")
    
    result_md.append("## 4. エグジット理由分析\n\n")
    result_md.append("### フィルター前\n\n")
    for reason, count in exit_reason_before.items():
        result_md.append(f"- {reason}: {count}件\n")
    result_md.append("\n### フィルター後（高トレンドのみ）\n\n")
    for reason, count in exit_reason_after.items():
        result_md.append(f"- {reason}: {count}件\n")
    result_md.append("\n")
    
    result_md.append("## 5. 急騰→急落パターンへの影響\n\n")
    result_md.append(f"- フィルター前: {len(pump_dump_before)}件 ({len(pump_dump_before)/len(takeda_df)*100:.1f}%)\n")
    result_md.append(f"- フィルター後: {len(pump_dump_after)}件 ({len(pump_dump_after)/len(takeda_high_trend)*100:.1f}%)\n")
    result_md.append(f"- **削減**: {reduction}件 ({reduction_rate:.1f}%削減)\n\n")
    
    result_md.append("## 6. 結論\n\n")
    
    # 成功判定
    success_criteria = []
    if metrics_after['total_trades'] >= 20:
        success_criteria.append("✅ 取引数20件以上確保")
    else:
        success_criteria.append("❌ 取引数不足")
    
    if metrics_after['pf'] > 1.0:
        success_criteria.append("✅ PF > 1.0達成")
    else:
        success_criteria.append("❌ PF < 1.0")
    
    if metrics_after['win_rate'] > 0.5:
        success_criteria.append("✅ 勝率50%超達成")
    else:
        success_criteria.append("❌ 勝率50%未満")
    
    if metrics_before['pf'] > 0 and pf_improvement >= 30:
        success_criteria.append("✅ PF改善率30%以上達成")
    else:
        success_criteria.append("❌ PF改善率30%未満")
    
    for criterion in success_criteria:
        result_md.append(f"{criterion}\n")
    
    result_md.append("\n### 総合評価\n\n")
    if all('✅' in c for c in success_criteria):
        result_md.append("**結論**: トレンド強度フィルターは有効。Phase 1.7（他9銘柄検証）へ進む。\n")
    else:
        result_md.append("**結論**: 一部基準未達。追加分析が必要。\n")
    
    result_md.append("\n---\n\n")
    result_md.append("**作成者**: validate_trend_strength_filter.py  \n")
    result_md.append(f"**最終更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Markdown保存
    result_path = Path('docs/exit_strategy/PHASE1_6_TREND_FILTER_RESULT.md')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(''.join(result_md))
    
    print(f"\n結果レポート保存: {result_path}")
    
    print("\n" + "=" * 80)
    print("全分析完了")
    print("=" * 80)
    print(f"\n結果ファイル:")
    print(f"- Markdown: {result_path}")
    print(f"- 図表: {figures_dir}/*.png")


if __name__ == "__main__":
    main()
