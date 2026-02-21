"""
Phase 1.8トレンド強度フィルターマルチティッカー検証スクリプト

Phase 1.6で武田薬品（4502.T）において驚異的な効果を示したトレンド強度フィルターを、
他の9銘柄にも適用し、普遍性を検証する。

主な機能:
- 9銘柄それぞれでトレンド強度閾値（67パーセンタイル）を算出
- 銘柄別にフィルター適用前後のPF・勝率・取引数を比較
- 改善銘柄数カウント（PF改善率>10%）
- 普遍性スコア再計算（改善銘柄数 / 9）
- 急騰→急落パターン削減効果を銘柄別に評価
- 5種類以上の比較図表生成
- 統合Markdownレポート生成

統合コンポーネント:
- results/phase1.6_trades_20260126_200241.csv: Phase 1.6グリッドサーチ結果（9銘柄）
- docs/exit_strategy/PHASE1_8_TREND_FILTER_MULTI_TICKER_DESIGN.md: 設計書
- docs/exit_strategy/figures/: 図表出力先

セーフティ機能/注意事項:
- CSVヘッダーあり（自動読み込み）
- ゼロ除算保護（PF計算、割合計算等）
- 銘柄データ存在確認（最低100件の取引が必要）
- 日本語フォント設定（MS Gothic使用）

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

# 検証対象銘柄（武田薬品除く9銘柄）
VALIDATION_TICKERS = [
    "7203.T",  # トヨタ自動車
    "9984.T",  # ソフトバンクグループ
    "8306.T",  # 三菱UFJ FG
    "6758.T",  # ソニーグループ
    "9983.T",  # ファーストリテイリング
    "6501.T",  # 日立製作所
    "8001.T",  # 伊藤忠商事
    "4063.T",  # 信越化学工業
    "6861.T"   # キーエンス
]


def load_and_preprocess_data():
    """
    Step 1: データ読み込み・前処理
    
    Returns:
        dict: {ticker: DataFrame} 銘柄別データ
    """
    csv_path = Path('results/phase1.6_trades_20260126_200241.csv')
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
    
    # データ読み込み（ヘッダーあり）
    df = pd.read_csv(csv_path)
    
    print(f"総取引数: {len(df)}件")
    
    # 銘柄別にデータ分割
    ticker_data = {}
    for ticker in VALIDATION_TICKERS:
        ticker_df = df[df['ticker'] == ticker].copy()
        if len(ticker_df) >= 100:
            ticker_data[ticker] = ticker_df
            print(f"{ticker}: {len(ticker_df)}件")
        else:
            print(f"{ticker}: データ不足（{len(ticker_df)}件）スキップ")
    
    return ticker_data


def calculate_trend_thresholds(ticker_df):
    """
    銘柄別にトレンド強度閾値を算出
    
    Args:
        ticker_df: 銘柄のDataFrame
        
    Returns:
        dict: 閾値情報
    """
    threshold_high = ticker_df['entry_trend_strength'].quantile(0.67)
    threshold_mid = ticker_df['entry_trend_strength'].quantile(0.33)
    
    high_count = len(ticker_df[ticker_df['entry_trend_strength'] >= threshold_high])
    mid_count = len(ticker_df[(ticker_df['entry_trend_strength'] >= threshold_mid) & 
                              (ticker_df['entry_trend_strength'] < threshold_high)])
    low_count = len(ticker_df[ticker_df['entry_trend_strength'] < threshold_mid])
    
    return {
        'threshold_high': threshold_high,
        'threshold_mid': threshold_mid,
        'high_count': high_count,
        'mid_count': mid_count,
        'low_count': low_count
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


def detect_pump_dump(trades_df):
    """
    急騰→急落パターン検出
    
    Args:
        trades_df: 取引履歴DataFrame
    
    Returns:
        DataFrame: 該当取引
    """
    pattern = trades_df[
        (trades_df['max_profit_pct'] > 15) &
        (trades_df['profit_loss_pct'] < 0) &
        (trades_df['holding_days'] < 30)
    ]
    return pattern


def apply_trend_filter(ticker_data):
    """
    Step 2-4: トレンド強度フィルター適用と成績比較
    
    Args:
        ticker_data: 銘柄別データ辞書
        
    Returns:
        tuple: (results, thresholds, pump_dump_results)
    """
    print("\n" + "=" * 80)
    print("Step 2-4: トレンド強度フィルター適用と成績比較")
    print("=" * 80)
    
    results = {}
    thresholds = {}
    pump_dump_results = {}
    
    for ticker, ticker_df in ticker_data.items():
        print(f"\n=== {ticker} ===")
        
        # Step 2: トレンド強度閾値算出
        thresholds[ticker] = calculate_trend_thresholds(ticker_df)
        threshold_high = thresholds[ticker]['threshold_high']
        
        print(f"高閾値（67%ile）: {threshold_high:.4f}")
        print(f"高: {thresholds[ticker]['high_count']}件")
        print(f"中: {thresholds[ticker]['mid_count']}件")
        print(f"低: {thresholds[ticker]['low_count']}件")
        
        # Step 3: フィルター適用前後を比較
        metrics_before = calculate_performance_metrics(ticker_df)
        high_trend_df = ticker_df[ticker_df['entry_trend_strength'] >= threshold_high].copy()
        metrics_after = calculate_performance_metrics(high_trend_df)
        
        # 改善率計算
        pf_improvement = 0.0
        if metrics_before['pf'] > 0:
            pf_improvement = (metrics_after['pf'] - metrics_before['pf']) / metrics_before['pf'] * 100
        
        win_rate_improvement = (metrics_after['win_rate'] - metrics_before['win_rate']) * 100
        
        results[ticker] = {
            'before': metrics_before,
            'after': metrics_after,
            'pf_improvement': pf_improvement,
            'win_rate_improvement': win_rate_improvement,
            'is_improved': pf_improvement > 10.0  # 改善閾値10%
        }
        
        print(f"\nフィルター前: PF={metrics_before['pf']:.2f}, 勝率={metrics_before['win_rate']*100:.1f}%, 取引数={metrics_before['total_trades']}件")
        print(f"フィルター後: PF={metrics_after['pf']:.2f}, 勝率={metrics_after['win_rate']*100:.1f}%, 取引数={metrics_after['total_trades']}件")
        print(f"PF改善率: {pf_improvement:+.1f}%")
        print(f"勝率改善: {win_rate_improvement:+.1f}ポイント")
        
        # Step 4: 急騰→急落パターン削減効果
        pump_dump_before = detect_pump_dump(ticker_df)
        pump_dump_after = detect_pump_dump(high_trend_df)
        
        reduction = len(pump_dump_before) - len(pump_dump_after)
        reduction_rate = (reduction / len(pump_dump_before) * 100) if len(pump_dump_before) > 0 else 0.0
        
        pump_dump_results[ticker] = {
            'before_count': len(pump_dump_before),
            'after_count': len(pump_dump_after),
            'reduction': reduction,
            'reduction_rate': reduction_rate
        }
        
        print(f"\n急騰→急落パターン:")
        print(f"  フィルター前: {len(pump_dump_before)}件")
        print(f"  フィルター後: {len(pump_dump_after)}件")
        print(f"  削減: {reduction}件 ({reduction_rate:.1f}%)")
    
    return results, thresholds, pump_dump_results


def evaluate_universality(results):
    """
    Step 5: 普遍性スコア再計算
    
    Args:
        results: フィルター適用結果辞書
        
    Returns:
        tuple: (universality_score, improved_count, improved_tickers)
    """
    print("\n" + "=" * 80)
    print("Step 5: 普遍性スコア再計算")
    print("=" * 80)
    
    improved_count = sum(1 for r in results.values() if r['is_improved'])
    universality_score = improved_count / len(results)
    improved_tickers = [ticker for ticker, r in results.items() if r['is_improved']]
    
    print(f"\n改善銘柄数: {improved_count} / {len(results)}銘柄")
    print(f"普遍性スコア: {universality_score:.2f}")
    print(f"武田薬品比較: Phase 1.7では0.00（0/9銘柄）")
    
    if improved_tickers:
        print(f"\n改善銘柄:")
        for ticker in improved_tickers:
            print(f"  - {ticker}: PF改善率{results[ticker]['pf_improvement']:+.1f}%")
    else:
        print("\n改善銘柄: なし")
    
    return universality_score, improved_count, improved_tickers


def generate_all_figures(results, thresholds, pump_dump_results):
    """
    Step 6: 可視化（5種類以上）
    
    Args:
        results: フィルター適用結果
        thresholds: トレンド強度閾値
        pump_dump_results: 急騰→急落パターン削減効果
    """
    print("\n" + "=" * 80)
    print("Step 6: 可視化生成")
    print("=" * 80)
    
    figures_dir = Path('docs/exit_strategy/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = sorted(results.keys())
    
    # 図表1: 銘柄別PF改善率比較棒グラフ
    print("\n図表1: 銘柄別PF改善率比較棒グラフ")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    pf_improvements = [results[t]['pf_improvement'] for t in tickers]
    colors = ['green' if x > 10 else 'red' for x in pf_improvements]
    bars = ax1.bar(range(len(tickers)), pf_improvements, color=colors, alpha=0.7, edgecolor='black')
    
    # 値ラベル追加
    for i, (bar, val) in enumerate(zip(bars, pf_improvements)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 5 if height > 0 else height - 5,
                f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    ax1.set_xticks(range(len(tickers)))
    ax1.set_xticklabels(tickers, rotation=45, ha='right')
    ax1.set_ylabel('PF Improvement (%)')
    ax1.set_title('Ticker PF Improvement by Trend Filter')
    ax1.axhline(10, color='black', linestyle='--', label='Improvement Threshold (10%)')
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig1.savefig(figures_dir / 'phase1_8_pf_improvement_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  保存: phase1_8_pf_improvement_comparison.png")
    
    # 図表2: フィルター前後のPF比較（ドットプロット）
    print("\n図表2: フィルター前後のPF比較")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for i, ticker in enumerate(tickers):
        pf_before = results[ticker]['before']['pf']
        pf_after = results[ticker]['after']['pf']
        ax2.plot([i, i], [pf_before, pf_after], 'o-', color='blue', markersize=8, linewidth=2)
        ax2.scatter(i, pf_before, color='red', s=100, zorder=3, label='Before' if i == 0 else '')
        ax2.scatter(i, pf_after, color='green', s=100, zorder=3, label='After' if i == 0 else '')
    
    ax2.set_xticks(range(len(tickers)))
    ax2.set_xticklabels(tickers, rotation=45, ha='right')
    ax2.set_ylabel('Profit Factor (PF)')
    ax2.set_title('PF Before/After Trend Filter')
    ax2.axhline(1.0, color='black', linestyle='--', label='Break-even (PF=1.0)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig(figures_dir / 'phase1_8_pf_before_after_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  保存: phase1_8_pf_before_after_comparison.png")
    
    # 図表3: 勝率改善ヒートマップ
    print("\n図表3: 勝率改善ヒートマップ")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    win_rate_matrix = []
    for ticker in tickers:
        win_rate_matrix.append([
            results[ticker]['before']['win_rate'] * 100,
            results[ticker]['after']['win_rate'] * 100,
            results[ticker]['win_rate_improvement']
        ])
    win_rate_matrix = np.array(win_rate_matrix)
    im = ax3.imshow(win_rate_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['Before (%)', 'After (%)', 'Improvement (pt)'])
    ax3.set_yticks(range(len(tickers)))
    ax3.set_yticklabels(tickers)
    ax3.set_title('Win Rate Improvement Heatmap')
    for i in range(len(tickers)):
        for j in range(3):
            text = ax3.text(j, i, f'{win_rate_matrix[i, j]:.1f}', ha='center', va='center', color='black', fontsize=9)
    plt.colorbar(im, ax=ax3)
    plt.tight_layout()
    fig3.savefig(figures_dir / 'phase1_8_winrate_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  保存: phase1_8_winrate_heatmap.png")
    
    # 図表4: 急騰→急落パターン削減効果
    print("\n図表4: 急騰→急落パターン削減効果")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    reduction_rates = [pump_dump_results[t]['reduction_rate'] for t in tickers]
    bars = ax4.bar(range(len(tickers)), reduction_rates, color='skyblue', alpha=0.7, edgecolor='black')
    
    # 値ラベル追加
    for bar, val in zip(bars, reduction_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax4.set_xticks(range(len(tickers)))
    ax4.set_xticklabels(tickers, rotation=45, ha='right')
    ax4.set_ylabel('Reduction Rate (%)')
    ax4.set_title('Pump & Dump Pattern Reduction by Trend Filter')
    ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig4.savefig(figures_dir / 'phase1_8_pump_dump_reduction.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"  保存: phase1_8_pump_dump_reduction.png")
    
    # 図表5: 取引数変化（フィルター前後）
    print("\n図表5: 取引数変化（フィルター前後）")
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    trades_before = [results[t]['before']['total_trades'] for t in tickers]
    trades_after = [results[t]['after']['total_trades'] for t in tickers]
    x = np.arange(len(tickers))
    width = 0.35
    bars1 = ax5.bar(x - width/2, trades_before, width, label='Before', color='orange', alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x + width/2, trades_after, width, label='After (High Trend)', color='green', alpha=0.7, edgecolor='black')
    ax5.set_xticks(x)
    ax5.set_xticklabels(tickers, rotation=45, ha='right')
    ax5.set_ylabel('Number of Trades')
    ax5.set_title('Trade Count Before/After Trend Filter')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig5.savefig(figures_dir / 'phase1_8_trade_count_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"  保存: phase1_8_trade_count_comparison.png")
    
    print(f"\n全5図表保存完了: {figures_dir}")


def create_integrated_report(results, thresholds, pump_dump_results, universality_score, improved_count, improved_tickers):
    """
    Step 7: 統合レポート生成
    
    Args:
        results: フィルター適用結果
        thresholds: トレンド強度閾値
        pump_dump_results: 急騰→急落パターン削減効果
        universality_score: 普遍性スコア
        improved_count: 改善銘柄数
        improved_tickers: 改善銘柄リスト
    """
    print("\n" + "=" * 80)
    print("Step 7: 統合レポート生成")
    print("=" * 80)
    
    report_path = Path('docs/exit_strategy/PHASE1_8_TREND_FILTER_RESULT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 1.8トレンド強度フィルターマルチティッカー検証結果\n\n")
        f.write(f"**作成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**検証銘柄数**: {len(results)}銘柄\n")
        f.write(f"**参照元**: \n")
        f.write(f"- [PHASE1_6_TREND_FILTER_RESULT.md](PHASE1_6_TREND_FILTER_RESULT.md)\n")
        f.write(f"- [PHASE1_8_TREND_FILTER_MULTI_TICKER_DESIGN.md](PHASE1_8_TREND_FILTER_MULTI_TICKER_DESIGN.md)\n\n")
        f.write("---\n\n")
        
        # セクション1: 銘柄別フィルター効果サマリー
        f.write("## 1. 銘柄別フィルター効果サマリー\n\n")
        f.write("| Ticker | Before PF | After PF | PF Improvement (%) | Before WR (%) | After WR (%) | WR Improvement (pt) | Trades Before | Trades After |\n")
        f.write("|--------|-----------|----------|-------------------|---------------|--------------|---------------------|---------------|-------------|\n")
        for ticker in sorted(results.keys()):
            r = results[ticker]
            f.write(f"| {ticker} | {r['before']['pf']:.2f} | {r['after']['pf']:.2f} | {r['pf_improvement']:+.1f} | {r['before']['win_rate']*100:.1f} | {r['after']['win_rate']*100:.1f} | {r['win_rate_improvement']:+.1f} | {r['before']['total_trades']} | {r['after']['total_trades']} |\n")
        f.write("\n**Figure**: [PF Improvement Comparison](figures/phase1_8_pf_improvement_comparison.png)\n\n")
        
        # セクション2: 普遍性評価
        f.write("## 2. 普遍性評価\n\n")
        f.write(f"- **普遍性スコア**: {universality_score:.2f} ({improved_count}/{len(results)}銘柄で改善)\n")
        f.write(f"- **Phase 1.7比較**: 0.00（0/9銘柄）→ {universality_score:.2f}（{improved_count}/9銘柄）\n")
        f.write(f"- **武田薬品比較**: PF改善率+2537.5%（Phase 1.6）\n\n")
        
        f.write(f"### 改善銘柄リスト\n\n")
        if improved_tickers:
            for ticker in improved_tickers:
                f.write(f"- **{ticker}**: PF改善率{results[ticker]['pf_improvement']:+.1f}%\n")
        else:
            f.write("- なし\n")
        f.write("\n**Figure**: [PF Before/After Comparison](figures/phase1_8_pf_before_after_comparison.png)\n\n")
        
        # セクション3: 急騰→急落パターン削減効果
        f.write("## 3. 急騰→急落パターン削減効果\n\n")
        f.write("| Ticker | Before Count | After Count | Reduction | Reduction Rate (%) |\n")
        f.write("|--------|--------------|-------------|-----------|-------------------|\n")
        for ticker in sorted(pump_dump_results.keys()):
            pdr = pump_dump_results[ticker]
            f.write(f"| {ticker} | {pdr['before_count']} | {pdr['after_count']} | {pdr['reduction']} | {pdr['reduction_rate']:.1f} |\n")
        f.write("\n**Figure**: [Pump & Dump Reduction](figures/phase1_8_pump_dump_reduction.png)\n\n")
        
        # セクション4: トレンド強度閾値サマリー
        f.write("## 4. 銘柄別トレンド強度閾値\n\n")
        f.write("| Ticker | High Threshold (67%ile) | High Count | Mid Count | Low Count |\n")
        f.write("|--------|------------------------|------------|-----------|----------|\n")
        for ticker in sorted(thresholds.keys()):
            t = thresholds[ticker]
            f.write(f"| {ticker} | {t['threshold_high']:.4f} | {t['high_count']} | {t['mid_count']} | {t['low_count']} |\n")
        f.write("\n")
        
        # セクション5: 結論と次フェーズ提案
        f.write("## 5. 結論と次フェーズ提案\n\n")
        f.write("### 主要発見事項\n\n")
        if universality_score >= 0.5:
            f.write(f"- トレンド強度フィルターは**{improved_count}/9銘柄**で有効（普遍性スコア{universality_score:.2f}）\n")
            f.write(f"- 武田薬品の驚異的効果（PF改善+2537.5%）は再現できないが、部分的効果を確認\n")
            f.write(f"- Phase 1.9で複合フィルター実装を推奨\n")
        elif universality_score >= 0.3:
            f.write(f"- トレンド強度フィルターは**部分的に有効**（{improved_count}/9銘柄で改善）\n")
            f.write(f"- 銘柄特性によって効果に大きな差がある\n")
            f.write(f"- Phase 1.9で銘柄特性別最適化を検討\n")
        else:
            f.write(f"- トレンド強度フィルターは**普遍性が低い**（{improved_count}/9銘柄のみ改善）\n")
            f.write(f"- 武田薬品での効果は銘柄特異的である可能性が高い\n")
            f.write(f"- Phase 1.9でトレンド強度以外のフィルター（SMA乖離、ATR等）を優先\n")
        
        f.write("\n### Phase 1.9提案\n\n")
        f.write("次フェーズでは、以下のいずれかを実施：\n\n")
        f.write("1. **銘柄特性別フィルター最適化**: セクター・ボラティリティ・流動性別にフィルター閾値を調整\n")
        f.write("2. **複合フィルター実装**: トレンド強度 + SMA乖離 + ATRの3条件統合\n")
        f.write("3. **機械学習モデル**: ランダムフォレストで最適エントリー条件を学習\n\n")
        
        f.write("---\n\n")
        f.write("**作成者**: validate_phase1_8_trend_filter_multi_ticker.py\n")
        f.write(f"**最終更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nレポート生成完了: {report_path}")


def main():
    """メイン処理"""
    print("=" * 80)
    print("Phase 1.8トレンド強度フィルターマルチティッカー検証開始")
    print("=" * 80)
    
    # Step 1: データ読み込み
    ticker_data = load_and_preprocess_data()
    
    if len(ticker_data) == 0:
        print("\nエラー: 有効な銘柄データがありません")
        return
    
    # Step 2-4: トレンド強度フィルター適用
    results, thresholds, pump_dump_results = apply_trend_filter(ticker_data)
    
    # Step 5: 普遍性スコア再計算
    universality_score, improved_count, improved_tickers = evaluate_universality(results)
    
    # Step 6: 可視化
    generate_all_figures(results, thresholds, pump_dump_results)
    
    # Step 7: 統合レポート生成
    create_integrated_report(results, thresholds, pump_dump_results, universality_score, improved_count, improved_tickers)
    
    print("\n" + "=" * 80)
    print("Phase 1.8検証完了")
    print("=" * 80)
    print(f"\n最終結果:")
    print(f"  普遍性スコア: {universality_score:.2f}")
    print(f"  改善銘柄数: {improved_count}/{len(results)}銘柄")
    if improved_tickers:
        print(f"  改善銘柄: {', '.join(improved_tickers)}")
    else:
        print(f"  改善銘柄: なし")


if __name__ == "__main__":
    main()
