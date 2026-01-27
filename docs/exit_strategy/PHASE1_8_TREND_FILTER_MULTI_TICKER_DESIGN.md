# Phase 1.8トレンド強度フィルターマルチティッカー検証設計書

**作成日**: 2026-01-26  
**目的**: トレンド強度フィルターを9銘柄に適用し、武田薬品以外での効果を検証  
**参照元**: 
- [PHASE1_6_TREND_FILTER_DESIGN.md](PHASE1_6_TREND_FILTER_DESIGN.md)
- [PHASE1_6_TREND_FILTER_RESULT.md](PHASE1_6_TREND_FILTER_RESULT.md)
- [PHASE1_7_MULTI_TICKER_RESULT.md](PHASE1_7_MULTI_TICKER_RESULT.md)

---

## 1. 目的とゴール

### 主目的
Phase 1.6で武田薬品（4502.T）において驚異的な効果を示したトレンド強度フィルターが、他の9銘柄でも有効かを検証する。

### 仮説
- トレンド強度フィルターは武田薬品で**PF改善率+2537.5%**、**勝率49.0ポイント向上**を達成
- しかし、Phase 1.7では発見2「トレンド強度の決定的影響」の普遍性スコアは**0.00（0/9銘柄）**
- 大敗パターン自体が武田薬品独特の可能性があるが、他銘柄でも部分的な効果が期待できる

### 成功条件
- [ ] 9銘柄それぞれでトレンド強度閾値（67パーセンタイル）を算出
- [ ] 銘柄別にフィルター適用前後のPF・勝率・取引数を比較
- [ ] 改善銘柄数をカウント（PF改善率>10%）
- [ ] 普遍性スコア再計算（改善銘柄数 / 9）
- [ ] 急騰→急落パターン削減効果を銘柄別に評価
- [ ] 結果をMarkdownレポートと図表で記録

---

## 2. 対象銘柄

Phase 1.7で検証済みの9銘柄（武田薬品除く）:

1. 7203.T - トヨタ自動車
2. 9984.T - ソフトバンクグループ
3. 8306.T - 三菱UFJ FG
4. 6758.T - ソニーグループ
5. 9983.T - ファーストリテイリング
6. 6501.T - 日立製作所
7. 8001.T - 伊藤忠商事
8. 4063.T - 信越化学工業
9. 6861.T - キーエンス

---

## 3. Phase 1.7基本統計（再掲）

| Ticker | Trades | Win Rate | PF | Avg Profit % | Avg R-Multiple |
|--------|--------|----------|----|--------------|----------------|
| 7203.T | 1257 | 29.8% | 0.97 | -0.07% | -0.02 |
| 9984.T | 1377 | 32.9% | 1.35 | 1.00% | 0.22 |
| 8306.T | 1077 | 36.8% | 1.89 | 1.58% | 0.36 |
| 6758.T | 1161 | 34.6% | 1.00 | -0.01% | -0.00 |
| 9983.T | 1020 | 36.5% | 1.45 | 1.02% | 0.23 |
| 6501.T | 1170 | 38.7% | 1.46 | 1.02% | 0.23 |
| 8001.T | 1089 | 39.4% | 1.41 | 0.82% | 0.19 |
| 4063.T | 1017 | 40.1% | 1.24 | 0.52% | 0.12 |
| 6861.T | 1005 | 39.7% | 0.95 | -0.11% | -0.03 |

---

## 4. 実装設計

### Step 1: データ読み込みと銘柄別分割

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

# 検証対象銘柄
VALIDATION_TICKERS = [
    "7203.T", "9984.T", "8306.T", "6758.T", "9983.T",
    "6501.T", "8001.T", "4063.T", "6861.T"
]

# CSV列名定義（29列）
COLUMN_NAMES = [
    'trade_id', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
    'profit_loss', 'exit_reason', 'holding_days', 'profit_loss_pct',
    'r_multiple', 'entry_gap_pct', 'max_profit_pct', 'entry_atr_pct',
    'sma_distance_pct', 'entry_trend_strength', 'entry_volume', 'exit_volume',
    'volume_ratio', 'exit_signal_pct', 'sma_value', 'trend_value',
    'above_trend', 'unused1', 'entry_price_base', 'unused2', 'unused3',
    'ticker', 'stop_loss_pct', 'trailing_stop_pct'
]

# データ読み込み
csv_path = Path('results/phase1.6_trades_20260126_200241.csv')
df = pd.read_csv(csv_path)

# 銘柄別にデータ分割
ticker_data = {}
for ticker in VALIDATION_TICKERS:
    ticker_df = df[df['ticker'] == ticker].copy()
    if len(ticker_df) >= 100:  # 最低100件の取引が必要
        ticker_data[ticker] = ticker_df
        print(f"{ticker}: {len(ticker_df)}件")
    else:
        print(f"{ticker}: データ不足（{len(ticker_df)}件）スキップ")
```

### Step 2: 銘柄別トレンド強度閾値算出

```python
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

# 銘柄別閾値算出
thresholds = {}
for ticker, ticker_df in ticker_data.items():
    thresholds[ticker] = calculate_trend_thresholds(ticker_df)
    print(f"\n{ticker}:")
    print(f"  高閾値（67%ile）: {thresholds[ticker]['threshold_high']:.4f}")
    print(f"  中閾値（33%ile）: {thresholds[ticker]['threshold_mid']:.4f}")
    print(f"  高: {thresholds[ticker]['high_count']}件")
    print(f"  中: {thresholds[ticker]['mid_count']}件")
    print(f"  低: {thresholds[ticker]['low_count']}件")
```

### Step 3: フィルター適用と成績比較

```python
def calculate_performance_metrics(trades_df):
    """パフォーマンス指標計算"""
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

# 銘柄別にフィルター適用前後を比較
results = {}
for ticker, ticker_df in ticker_data.items():
    threshold_high = thresholds[ticker]['threshold_high']
    
    # フィルター前
    metrics_before = calculate_performance_metrics(ticker_df)
    
    # フィルター後（高トレンドのみ）
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
    
    print(f"\n=== {ticker} ===")
    print(f"フィルター前: PF={metrics_before['pf']:.2f}, 勝率={metrics_before['win_rate']*100:.1f}%, 取引数={metrics_before['total_trades']}件")
    print(f"フィルター後: PF={metrics_after['pf']:.2f}, 勝率={metrics_after['win_rate']*100:.1f}%, 取引数={metrics_after['total_trades']}件")
    print(f"PF改善率: {pf_improvement:+.1f}%")
    print(f"勝率改善: {win_rate_improvement:+.1f}ポイント")
```

### Step 4: 急騰→急落パターン削減効果

```python
def detect_pump_dump(trades_df):
    """急騰→急落パターン検出"""
    pattern = trades_df[
        (trades_df['max_profit_pct'] > 15) &
        (trades_df['profit_loss_pct'] < 0) &
        (trades_df['holding_days'] < 30)
    ]
    return pattern

# 銘柄別に急騰→急落パターン削減効果を評価
pump_dump_results = {}
for ticker, ticker_df in ticker_data.items():
    threshold_high = thresholds[ticker]['threshold_high']
    high_trend_df = ticker_df[ticker_df['entry_trend_strength'] >= threshold_high].copy()
    
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
    
    print(f"\n{ticker}:")
    print(f"  フィルター前: {len(pump_dump_before)}件")
    print(f"  フィルター後: {len(pump_dump_after)}件")
    print(f"  削減: {reduction}件 ({reduction_rate:.1f}%)")
```

### Step 5: 普遍性スコア再計算

```python
# 改善銘柄数カウント
improved_count = sum(1 for r in results.values() if r['is_improved'])
universality_score = improved_count / len(results)

print(f"\n=== 普遍性評価 ===")
print(f"改善銘柄数: {improved_count} / {len(results)}銘柄")
print(f"普遍性スコア: {universality_score:.2f}")
print(f"武田薬品比較: Phase 1.7では0.00（0/9銘柄）")

# 改善銘柄リスト
improved_tickers = [ticker for ticker, r in results.items() if r['is_improved']]
print(f"\n改善銘柄: {', '.join(improved_tickers) if improved_tickers else 'なし'}")
```

### Step 6: 可視化（5種類以上）

```python
figures_dir = Path('docs/exit_strategy/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

# 図表1: 銘柄別PF改善率比較棒グラフ
fig1, ax1 = plt.subplots(figsize=(12, 6))
tickers = list(results.keys())
pf_improvements = [results[t]['pf_improvement'] for t in tickers]
colors = ['green' if x > 10 else 'red' for x in pf_improvements]
bars = ax1.bar(range(len(tickers)), pf_improvements, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(tickers)))
ax1.set_xticklabels(tickers, rotation=45, ha='right')
ax1.set_ylabel('PF Improvement (%)')
ax1.set_title('Ticker PF Improvement by Trend Filter')
ax1.axhline(10, color='black', linestyle='--', label='Improvement Threshold (10%)')
ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig1.savefig(figures_dir / 'phase1_8_pf_improvement_comparison.png', dpi=150)

# 図表2: フィルター前後のPF比較（ドットプロット）
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
fig2.savefig(figures_dir / 'phase1_8_pf_before_after_comparison.png', dpi=150)

# 図表3: 勝率改善ヒートマップ
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
fig3.savefig(figures_dir / 'phase1_8_winrate_heatmap.png', dpi=150)

# 図表4: 急騰→急落パターン削減効果
fig4, ax4 = plt.subplots(figsize=(12, 6))
reduction_rates = [pump_dump_results[t]['reduction_rate'] for t in tickers]
bars = ax4.bar(range(len(tickers)), reduction_rates, color='skyblue', alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(tickers)))
ax4.set_xticklabels(tickers, rotation=45, ha='right')
ax4.set_ylabel('Reduction Rate (%)')
ax4.set_title('Pump & Dump Pattern Reduction by Trend Filter')
ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig4.savefig(figures_dir / 'phase1_8_pump_dump_reduction.png', dpi=150)

# 図表5: 取引数変化（フィルター前後）
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
fig5.savefig(figures_dir / 'phase1_8_trade_count_comparison.png', dpi=150)

print(f"\n図表保存完了: {figures_dir}")
```

### Step 7: 結果レポート作成

```python
def create_integrated_report(results, thresholds, pump_dump_results, universality_score):
    """統合レポート生成"""
    report_path = Path('docs/exit_strategy/PHASE1_8_TREND_FILTER_RESULT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 1.8トレンド強度フィルターマルチティッカー検証結果\n\n")
        f.write(f"**作成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**検証銘柄数**: {len(results)}銘柄\n")
        f.write(f"**参照元**: [PHASE1_6_TREND_FILTER_RESULT.md](PHASE1_6_TREND_FILTER_RESULT.md)\n\n")
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
        improved_count = sum(1 for r in results.values() if r['is_improved'])
        f.write("## 2. 普遍性評価\n\n")
        f.write(f"- **普遍性スコア**: {universality_score:.2f} ({improved_count}/{len(results)}銘柄で改善)\n")
        f.write(f"- **Phase 1.7比較**: 0.00（0/9銘柄）→ {universality_score:.2f}（{improved_count}/9銘柄）\n\n")
        
        improved_tickers = [ticker for ticker, r in results.items() if r['is_improved']]
        f.write(f"### 改善銘柄リスト\n\n")
        if improved_tickers:
            for ticker in improved_tickers:
                f.write(f"- {ticker}: PF改善率{results[ticker]['pf_improvement']:+.1f}%\n")
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
        
        # セクション4: 結論と次フェーズ提案
        f.write("## 4. 結論と次フェーズ提案\n\n")
        f.write("### 主要発見事項\n\n")
        if universality_score >= 0.5:
            f.write(f"- トレンド強度フィルターは**{improved_count}/9銘柄**で有効（普遍性スコア{universality_score:.2f}）\n")
            f.write(f"- 武田薬品の驚異的効果（PF改善+2537.5%）は再現できないが、部分的効果を確認\n")
        elif universality_score >= 0.3:
            f.write(f"- トレンド強度フィルターは**部分的に有効**（{improved_count}/9銘柄で改善）\n")
            f.write(f"- 銘柄特性によって効果に大きな差がある\n")
        else:
            f.write(f"- トレンド強度フィルターは**普遍性が低い**（{improved_count}/9銘柄のみ改善）\n")
            f.write(f"- 武田薬品での効果は銘柄特異的である可能性が高い\n")
        
        f.write("\n### Phase 1.9提案\n\n")
        f.write("次フェーズでは、以下のいずれかを実施：\n\n")
        f.write("1. **銘柄特性別フィルター最適化**: セクター・ボラティリティ・流動性別にフィルター閾値を調整\n")
        f.write("2. **複合フィルター実装**: トレンド強度 + SMA乖離 + ATRの3条件統合\n")
        f.write("3. **機械学習モデル**: ランダムフォレストで最適エントリー条件を学習\n\n")
        
        f.write("---\n\n")
        f.write("**作成者**: validate_phase1_8_trend_filter_multi_ticker.py\n")
        f.write(f"**最終更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nレポート生成完了: {report_path}")
```

---

## 5. 検証手順

1. **スクリプト作成**: `scripts/validate_phase1_8_trend_filter_multi_ticker.py`
2. **実行**: `python scripts/validate_phase1_8_trend_filter_multi_ticker.py`
3. **出力確認**:
   - `docs/exit_strategy/PHASE1_8_TREND_FILTER_RESULT.md`
   - `docs/exit_strategy/figures/phase1_8_*.png`（5種類）
4. **評価基準**:
   - 普遍性スコア ≥ 0.3: 部分的に有効
   - 普遍性スコア ≥ 0.5: 高い普遍性
   - 普遍性スコア < 0.3: 銘柄特異的

---

## 6. 期待される結果

### 楽観シナリオ（普遍性スコア ≥ 0.5）
- 5銘柄以上でPF改善率>10%
- 武田薬品と同様の急騰→急落パターン削減効果
- Phase 1.9で複合フィルター実装へ進む

### 現実的シナリオ（普遍性スコア 0.3-0.5）
- 3-4銘柄でPF改善率>10%
- 部分的な効果確認、銘柄特性による差が大きい
- Phase 1.9で銘柄特性別最適化を検討

### 悲観シナリオ（普遍性スコア < 0.3）
- 2銘柄以下でPF改善率>10%
- 武田薬品の効果は銘柄特異的
- Phase 1.9でトレンド強度以外のフィルター（SMA乖離、ATR等）を優先

---

**作成者**: GitHub Copilot  
**最終更新**: 2026-01-26
