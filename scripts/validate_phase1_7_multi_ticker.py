"""
Phase 1.7マルチティッカー大敗パターン検証スクリプト

Phase 1.6で武田薬品（4502.T）から発見した5つの大敗パターンが、
他の9銘柄でも普遍的に適用されるか検証する。

主な機能:
- 銘柄別基本統計集計（取引数、勝率、PF、平均損益率等）
- 分析1～5のマルチティッカー実行（大損失トレード、時系列、エントリー品質、ATR、急騰急落）
- 銘柄間比較・普遍性評価（5発見事項のスコア算出）
- 統合レポート生成（PHASE1_7_MULTI_TICKER_RESULT.md）
- 6種類以上の比較図表生成

統合コンポーネント:
- results/phase1.6_trades_20260126_124409a.csv: Phase 1.6グリッドサーチ結果（10銘柄×平均840件）
- docs/exit_strategy/PHASE1_7_MULTI_TICKER_DEFEAT_PATTERNS_DESIGN.md: 設計書
- docs/exit_strategy/figures/: 図表出力先

セーフティ機能/注意事項:
- CSVヘッダーなし（29列の列名を手動設定）
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
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import Counter
from math import pi

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

# 検証対象銘柄
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


def load_and_preprocess_data():
    """
    Step 1: データ読み込み・前処理
    
    Returns:
        dict: {ticker: DataFrame} 銘柄別データ
    """
    # Phase 1.7: 新しいグリッドサーチ結果（9銘柄、武田除外）
    csv_path = Path('results/phase1.6_trades_20260126_200241.csv')
    
    if not csv_path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {csv_path}")
    
    # データ読み込み（Phase 1.7: ヘッダーあり）
    df = pd.read_csv(csv_path)
    
    print(f"総取引数: {len(df)}件")
    
    # 銘柄別にデータ分割（Phase 1.7: 武田除外、9銘柄のみ）
    ticker_data = {}
    all_tickers = VALIDATION_TICKERS
    
    for ticker in all_tickers:
        ticker_df = df[df['ticker'] == ticker].copy()
        
        if len(ticker_df) == 0:
            print(f"  [警告] {ticker}: データなし")
            continue
        
        if len(ticker_df) < 100:
            print(f"  [警告] {ticker}: {len(ticker_df)}件（100件未満）")
        else:
            print(f"  {ticker}: {len(ticker_df)}件")
        
        # 日付変換
        ticker_df['entry_date'] = pd.to_datetime(ticker_df['entry_date'])
        ticker_df['exit_date'] = pd.to_datetime(ticker_df['exit_date'])
        
        ticker_data[ticker] = ticker_df
    
    return ticker_data


def calculate_basic_stats(ticker_df):
    """
    銘柄別基本統計を計算
    
    Args:
        ticker_df: 銘柄のDataFrame
        
    Returns:
        dict: 基本統計情報
    """
    total_trades = len(ticker_df)
    
    if total_trades == 0:
        return None
    
    winning_trades = len(ticker_df[ticker_df['profit_loss_pct'] > 0])
    win_rate = winning_trades / total_trades
    
    avg_profit_pct = ticker_df['profit_loss_pct'].mean()
    
    # PF計算
    total_profit = ticker_df[ticker_df['profit_loss_pct'] > 0]['profit_loss_pct'].sum()
    total_loss = abs(ticker_df[ticker_df['profit_loss_pct'] < 0]['profit_loss_pct'].sum())
    pf = total_profit / total_loss if total_loss > 0 else 0
    
    avg_r_multiple = ticker_df['r_multiple'].mean()
    avg_holding_days = ticker_df['holding_days'].mean()
    
    exit_reason_dist = ticker_df['exit_reason'].value_counts().to_dict()
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'avg_profit_pct': avg_profit_pct,
        'pf': pf,
        'avg_r_multiple': avg_r_multiple,
        'avg_holding_days': avg_holding_days,
        'exit_reason_dist': exit_reason_dist
    }


def calculate_basic_stats_all_tickers(ticker_data):
    """
    Step 2: 全銘柄の基本統計集計
    
    Args:
        ticker_data: 銘柄別データ辞書
        
    Returns:
        dict: {ticker: stats} 銘柄別基本統計
    """
    basic_stats = {}
    
    for ticker, ticker_df in ticker_data.items():
        stats = calculate_basic_stats(ticker_df)
        if stats is not None:
            basic_stats[ticker] = stats
            print(f"{ticker}: 取引数={stats['total_trades']}, 勝率={stats['win_rate']*100:.1f}%, PF={stats['pf']:.2f}")
    
    return basic_stats


def run_analysis1(ticker_data):
    """
    分析1: 大損失トレードの特徴抽出
    
    Args:
        ticker_data: 銘柄別データ辞書
        
    Returns:
        dict: {ticker: analysis_result}
    """
    print("\n  [分析1] 大損失トレード（profit_loss_pct < -10%）の特徴抽出")
    
    big_loss_analysis = {}
    
    for ticker, ticker_df in ticker_data.items():
        big_loss = ticker_df[ticker_df['profit_loss_pct'] < -10]
        normal = ticker_df[ticker_df['profit_loss_pct'] >= -10]
        
        big_loss_ratio = len(big_loss) / len(ticker_df) if len(ticker_df) > 0 else 0
        
        if len(big_loss) > 0:
            big_loss_features = big_loss[['entry_gap_pct', 'entry_atr_pct', 
                                          'holding_days', 'max_profit_pct', 
                                          'entry_trend_strength', 'sma_distance_pct']].mean()
            exit_reason_dist = big_loss['exit_reason'].value_counts().to_dict()
        else:
            big_loss_features = None
            exit_reason_dist = {}
        
        big_loss_analysis[ticker] = {
            'ticker': ticker,
            'big_loss_count': len(big_loss),
            'big_loss_ratio': big_loss_ratio,
            'big_loss_features': big_loss_features,
            'exit_reason_dist': exit_reason_dist
        }
        
        print(f"    {ticker}: {len(big_loss)}件（{big_loss_ratio*100:.1f}%）")
    
    return big_loss_analysis


def run_analysis2(ticker_data):
    """
    分析2: 時系列での勝敗パターン
    
    Args:
        ticker_data: 銘柄別データ辞書
        
    Returns:
        dict: {ticker: time_series_result}
    """
    print("\n  [分析2] 月別成績推移と低迷期間の特定")
    
    time_series_analysis = {}
    
    for ticker, ticker_df in ticker_data.items():
        ticker_df['year_month'] = ticker_df['entry_date'].dt.to_period('M')
        
        monthly = ticker_df.groupby('year_month').agg({
            'profit_loss_pct': ['count', 'mean', 'min']
        })
        monthly.columns = ['total_trades', 'avg_profit_pct', 'max_loss']
        monthly['win_rate'] = ticker_df.groupby('year_month').apply(
            lambda x: (x['profit_loss_pct'] > 0).mean()
        )
        
        # 低迷期間（avg_profit_pct < -5%）
        poor_periods = monthly[monthly['avg_profit_pct'] < -5].index.tolist()
        
        time_series_analysis[ticker] = {
            'ticker': ticker,
            'monthly_stats': monthly,
            'poor_periods': poor_periods
        }
        
        if len(poor_periods) > 0:
            print(f"    {ticker}: 低迷期間 {len(poor_periods)}ヶ月")
    
    return time_series_analysis


def run_analysis3(ticker_data):
    """
    分析3: エントリー品質の分析
    
    Args:
        ticker_data: 銘柄別データ辞書
        
    Returns:
        dict: {ticker: entry_quality_result}
    """
    print("\n  [分析3] エントリー品質分析（SMA乖離・トレンド強度）")
    
    entry_quality_analysis = {}
    
    for ticker, ticker_df in ticker_data.items():
        # SMA乖離範囲分類
        def classify_sma_distance(val):
            if val < 5: return '0-5%'
            elif val < 10: return '5-10%'
            elif val < 15: return '10-15%'
            elif val < 20: return '15-20%'
            else: return '20%+'
        
        ticker_df['sma_range'] = ticker_df['sma_distance_pct'].apply(classify_sma_distance)
        
        sma_analysis = ticker_df.groupby('sma_range').agg({
            'profit_loss_pct': ['count', 'mean'],
            'r_multiple': 'mean'
        })
        sma_analysis.columns = ['count', 'avg_profit_pct', 'avg_r_multiple']
        sma_analysis['win_rate'] = ticker_df.groupby('sma_range').apply(
            lambda x: (x['profit_loss_pct'] > 0).mean()
        )
        
        # トレンド強度分類（33%ile, 67%ile）
        threshold_high = ticker_df['entry_trend_strength'].quantile(0.67)
        threshold_mid = ticker_df['entry_trend_strength'].quantile(0.33)
        
        def classify_trend_strength(val):
            if val < threshold_mid: return '低'
            elif val < threshold_high: return '中'
            else: return '高'
        
        ticker_df['trend_strength_level'] = ticker_df['entry_trend_strength'].apply(classify_trend_strength)
        
        trend_analysis = ticker_df.groupby('trend_strength_level').agg({
            'profit_loss_pct': ['count', 'mean'],
            'r_multiple': 'mean'
        })
        trend_analysis.columns = ['count', 'avg_profit_pct', 'avg_r_multiple']
        trend_analysis['win_rate'] = ticker_df.groupby('trend_strength_level').apply(
            lambda x: (x['profit_loss_pct'] > 0).mean()
        )
        
        entry_quality_analysis[ticker] = {
            'ticker': ticker,
            'sma_analysis': sma_analysis,
            'trend_analysis': trend_analysis,
            'trend_thresholds': {'mid': threshold_mid, 'high': threshold_high}
        }
        
        # 高トレンド勝率表示
        if '高' in trend_analysis.index:
            high_wr = trend_analysis.loc['高', 'win_rate']
            print(f"    {ticker}: 高トレンド勝率={high_wr*100:.1f}%")
    
    return entry_quality_analysis


def run_analysis4(ticker_data):
    """
    分析4: ボラティリティの影響
    
    Args:
        ticker_data: 銘柄別データ辞書
        
    Returns:
        dict: {ticker: volatility_result}
    """
    print("\n  [分析4] ATR範囲別成績分析")
    
    volatility_analysis = {}
    
    for ticker, ticker_df in ticker_data.items():
        # ATR基本統計
        atr_stats = ticker_df['entry_atr_pct'].describe()
        
        # ATR範囲分類
        def classify_atr(val):
            if val < 2: return '低 (0-2%)'
            elif val < 3: return '中 (2-3%)'
            elif val < 4: return '高 (3-4%)'
            else: return '超高 (4%+)'
        
        ticker_df['atr_range'] = ticker_df['entry_atr_pct'].apply(classify_atr)
        
        atr_analysis = ticker_df.groupby('atr_range').agg({
            'profit_loss_pct': ['count', 'mean']
        })
        atr_analysis.columns = ['count', 'avg_profit_pct']
        atr_analysis['win_rate'] = ticker_df.groupby('atr_range').apply(
            lambda x: (x['profit_loss_pct'] > 0).mean()
        )
        
        volatility_analysis[ticker] = {
            'ticker': ticker,
            'atr_stats': atr_stats,
            'atr_analysis': atr_analysis
        }
        
        # 超高ATR勝率表示（あれば）
        if '超高 (4%+)' in atr_analysis.index:
            super_high_wr = atr_analysis.loc['超高 (4%+)', 'win_rate']
            print(f"    {ticker}: 超高ATR勝率={super_high_wr*100:.1f}%")
    
    return volatility_analysis


def calculate_pf(trades_df):
    """PF計算"""
    if len(trades_df) == 0:
        return 0
    total_profit = trades_df[trades_df['profit_loss_pct'] > 0]['profit_loss_pct'].sum()
    total_loss = abs(trades_df[trades_df['profit_loss_pct'] < 0]['profit_loss_pct'].sum())
    return total_profit / total_loss if total_loss > 0 else 0


def run_analysis5(ticker_data):
    """
    分析5: 急騰→急落パターンの検出
    
    Args:
        ticker_data: 銘柄別データ辞書
        
    Returns:
        dict: {ticker: pump_dump_result}
    """
    print("\n  [分析5] 急騰→急落パターン検出")
    
    pump_dump_analysis = {}
    
    for ticker, ticker_df in ticker_data.items():
        # パターン検出（max_profit_pct > 15% & profit_loss_pct < 0 & holding_days < 30）
        pattern = ticker_df[
            (ticker_df['max_profit_pct'] > 15) &
            (ticker_df['profit_loss_pct'] < 0) &
            (ticker_df['holding_days'] < 30)
        ]
        
        pattern_count = len(pattern)
        pattern_ratio = pattern_count / len(ticker_df) if len(ticker_df) > 0 else 0
        
        if pattern_count > 0:
            total_loss = pattern['profit_loss'].sum()
            avg_max_profit = pattern['max_profit_pct'].mean()
            avg_final_loss = pattern['profit_loss_pct'].mean()
            exit_reason_dist = pattern['exit_reason'].value_counts().to_dict()
        else:
            total_loss = 0
            avg_max_profit = 0
            avg_final_loss = 0
            exit_reason_dist = {}
        
        # PF影響度計算
        pf_with_pattern = calculate_pf(ticker_df)
        pf_without_pattern = calculate_pf(ticker_df[~ticker_df.index.isin(pattern.index)])
        pf_improvement = ((pf_without_pattern - pf_with_pattern) / pf_with_pattern * 100) if pf_with_pattern > 0 else 0
        
        pump_dump_analysis[ticker] = {
            'ticker': ticker,
            'pattern_count': pattern_count,
            'pattern_ratio': pattern_ratio,
            'total_loss': total_loss,
            'avg_max_profit': avg_max_profit,
            'avg_final_loss': avg_final_loss,
            'exit_reason_dist': exit_reason_dist,
            'pf_with_pattern': pf_with_pattern,
            'pf_without_pattern': pf_without_pattern,
            'pf_improvement': pf_improvement
        }
        
        print(f"    {ticker}: {pattern_count}件（{pattern_ratio*100:.1f}%）、PF改善={pf_improvement:.1f}%")
    
    return pump_dump_analysis


def evaluate_universality(pump_dump_analysis, entry_quality_analysis, volatility_analysis, time_series_analysis):
    """
    Step 4: 銘柄間比較・普遍性評価
    
    Args:
        pump_dump_analysis: 急騰→急落分析結果
        entry_quality_analysis: エントリー品質分析結果
        volatility_analysis: ボラティリティ分析結果
        time_series_analysis: 時系列分析結果
        
    Returns:
        tuple: (universality_scores, results_matrix)
    """
    print("\n  [普遍性評価] 5つの発見事項のスコア算出")
    
    universality_scores = {
        '発見1_急騰急落': 0,
        '発見2_トレンド強度': 0,
        '発見3_ATRパラドックス': 0,
        '発見4_SMA乖離5%以上': 0,
        '発見5_極端下落相場': 0
    }
    
    results_matrix = []
    
    for ticker in VALIDATION_TICKERS:
        ticker_results = {'ticker': ticker}
        
        # 発見1: 急騰→急落（パターン発生率 > 30% かつ PF改善率 > 50%）
        if ticker in pump_dump_analysis:
            pump_dump = pump_dump_analysis[ticker]
            if pump_dump['pattern_ratio'] > 0.3 and pump_dump['pf_improvement'] > 50:
                universality_scores['発見1_急騰急落'] += 1
                ticker_results['発見1'] = 'OK'
            else:
                ticker_results['発見1'] = 'NG'
        else:
            ticker_results['発見1'] = 'N/A'
        
        # 発見2: トレンド強度（高トレンド勝率 > 60% かつ 中/低トレンド勝率 < 30%）
        if ticker in entry_quality_analysis:
            trend = entry_quality_analysis[ticker]['trend_analysis']
            high_win_rate = trend.loc['高', 'win_rate'] if '高' in trend.index else 0
            mid_win_rate = trend.loc['中', 'win_rate'] if '中' in trend.index else 0
            low_win_rate = trend.loc['低', 'win_rate'] if '低' in trend.index else 0
            mid_low_win_rate = (mid_win_rate + low_win_rate) / 2 if mid_win_rate > 0 or low_win_rate > 0 else 0
            
            if high_win_rate > 0.6 and mid_low_win_rate < 0.3:
                universality_scores['発見2_トレンド強度'] += 1
                ticker_results['発見2'] = 'OK'
            else:
                ticker_results['発見2'] = 'NG'
        else:
            ticker_results['発見2'] = 'N/A'
        
        # 発見3: ATRパラドックス（超高ATR勝率 > 中央値+20pt かつ 低ATR勝率 < 中央値）
        if ticker in volatility_analysis:
            atr = volatility_analysis[ticker]['atr_analysis']
            if '超高 (4%+)' in atr.index and '低 (0-2%)' in atr.index:
                super_high_wr = atr.loc['超高 (4%+)', 'win_rate']
                low_wr = atr.loc['低 (0-2%)', 'win_rate']
                median_wr = atr['win_rate'].median()
                
                if super_high_wr > median_wr + 0.2 and low_wr < median_wr:
                    universality_scores['発見3_ATRパラドックス'] += 1
                    ticker_results['発見3'] = 'OK'
                else:
                    ticker_results['発見3'] = 'NG'
            else:
                ticker_results['発見3'] = 'N/A'
        else:
            ticker_results['発見3'] = 'N/A'
        
        # 発見4: SMA乖離5%以上（該当範囲の勝率 < 30%）
        if ticker in entry_quality_analysis:
            sma = entry_quality_analysis[ticker]['sma_analysis']
            sma_5plus_ranges = ['5-10%', '10-15%', '15-20%', '20%+']
            sma_5plus_records = sma[sma.index.isin(sma_5plus_ranges)]
            
            if len(sma_5plus_records) > 0:
                sma_5plus_wr = sma_5plus_records['win_rate'].mean()
                if sma_5plus_wr < 0.3:
                    universality_scores['発見4_SMA乖離5%以上'] += 1
                    ticker_results['発見4'] = 'OK'
                else:
                    ticker_results['発見4'] = 'NG'
            else:
                ticker_results['発見4'] = 'N/A'
        else:
            ticker_results['発見4'] = 'N/A'
        
        # 発見5: 極端下落相場（2021年4月 or 8月に平均損益率 < -3%）
        if ticker in time_series_analysis:
            monthly_stats = time_series_analysis[ticker]['monthly_stats']
            target_periods = ['2021-04', '2021-08']
            
            found_poor_period = False
            for period_str in target_periods:
                matching_periods = [p for p in monthly_stats.index if str(p) == period_str]
                if matching_periods:
                    for period in matching_periods:
                        if monthly_stats.loc[period, 'avg_profit_pct'] < -3:
                            found_poor_period = True
                            break
            
            if found_poor_period:
                universality_scores['発見5_極端下落相場'] += 1
                ticker_results['発見5'] = 'OK'
            else:
                ticker_results['発見5'] = 'NG'
        else:
            ticker_results['発見5'] = 'N/A'
        
        results_matrix.append(ticker_results)
    
    # 普遍性スコア正規化（0-1）
    universality_scores_normalized = {k: v / 9 for k, v in universality_scores.items()}
    
    print("\n  [普遍性スコア（0-1）]")
    for key, score in universality_scores_normalized.items():
        print(f"    {key}: {score:.2f} ({int(score*9)}/9銘柄)")
    
    return universality_scores_normalized, results_matrix


def save_figure(fig, filename, figures_dir):
    """図表保存"""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / filename
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    図表保存: {filename}")
    plt.close(fig)


def generate_all_figures(basic_stats, entry_quality_analysis, pump_dump_analysis, 
                        volatility_analysis, time_series_analysis, universality_scores):
    """
    Step 5: 可視化生成
    
    Args:
        basic_stats: 基本統計
        entry_quality_analysis: エントリー品質分析
        pump_dump_analysis: 急騰→急落分析
        volatility_analysis: ボラティリティ分析
        time_series_analysis: 時系列分析
        universality_scores: 普遍性スコア
    """
    figures_dir = Path('docs/exit_strategy/figures')
    
    print("\n  [図表生成]")
    
    # Phase 1.7: 9銘柄のみ（武田除外）
    all_tickers = VALIDATION_TICKERS
    tickers = [t for t in all_tickers if t in basic_stats]
    
    # 図表1: 銘柄別基本統計比較（勝率・PF）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    win_rates = [basic_stats[t]['win_rate']*100 for t in tickers]
    axes[0].bar(range(len(tickers)), win_rates, color='skyblue', edgecolor='black')
    axes[0].set_xticks(range(len(tickers)))
    axes[0].set_xticklabels(tickers, rotation=45, ha='right')
    axes[0].set_ylabel('Winrate (%)')
    axes[0].set_title('Ticker Winrate Comparison')
    axes[0].axhline(50, color='red', linestyle='--', label='50% line')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    pf_values = [basic_stats[t]['pf'] for t in tickers]
    axes[1].bar(range(len(tickers)), pf_values, color='green', edgecolor='black')
    axes[1].set_xticks(range(len(tickers)))
    axes[1].set_xticklabels(tickers, rotation=45, ha='right')
    axes[1].set_ylabel('Profit Factor (PF)')
    axes[1].set_title('Ticker PF Comparison')
    axes[1].axhline(1.0, color='red', linestyle='--', label='Break-even (PF=1.0)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'phase1_7_basic_stats_comparison.png', figures_dir)
    
    # 図表2: トレンド強度別勝率ヒートマップ
    trend_wr_matrix = []
    for ticker in tickers:
        if ticker in entry_quality_analysis:
            trend = entry_quality_analysis[ticker]['trend_analysis']
            trend_wr_matrix.append([
                trend.loc['低', 'win_rate'] if '低' in trend.index else 0,
                trend.loc['中', 'win_rate'] if '中' in trend.index else 0,
                trend.loc['高', 'win_rate'] if '高' in trend.index else 0
            ])
        else:
            trend_wr_matrix.append([0, 0, 0])
    
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(trend_wr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=['Low', 'Mid', 'High'], 
                yticklabels=tickers,
                cbar_kws={'label': 'Win Rate'},
                vmin=0, vmax=1)
    ax.set_title('Trend Strength Win Rate Heatmap by Ticker')
    ax.set_xlabel('Trend Strength')
    ax.set_ylabel('Ticker')
    plt.tight_layout()
    save_figure(fig, 'phase1_7_trend_strength_heatmap.png', figures_dir)
    
    # 図表3: 急騰→急落パターン発生率とPF影響
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    pattern_ratios = [pump_dump_analysis[t]['pattern_ratio']*100 if t in pump_dump_analysis else 0 for t in tickers]
    axes[0].bar(range(len(tickers)), pattern_ratios, color='orange', edgecolor='black')
    axes[0].set_xticks(range(len(tickers)))
    axes[0].set_xticklabels(tickers, rotation=45, ha='right')
    axes[0].set_ylabel('Pattern Rate (%)')
    axes[0].set_title('Pump & Dump Pattern Rate by Ticker')
    axes[0].axhline(56.1, color='red', linestyle='--', label='Takeda (56.1%)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    pf_improvements = [pump_dump_analysis[t]['pf_improvement'] if t in pump_dump_analysis else 0 for t in tickers]
    axes[1].bar(range(len(tickers)), pf_improvements, color='purple', edgecolor='black')
    axes[1].set_xticks(range(len(tickers)))
    axes[1].set_xticklabels(tickers, rotation=45, ha='right')
    axes[1].set_ylabel('PF Improvement (%)')
    axes[1].set_title('PF Improvement after Excluding Pattern')
    axes[1].axhline(295.4, color='red', linestyle='--', label='Takeda (295.4%)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'phase1_7_pump_dump_comparison.png', figures_dir)
    
    # 図表4: ATR範囲別勝率ヒートマップ
    atr_wr_matrix = []
    atr_ranges = ['低 (0-2%)', '中 (2-3%)', '高 (3-4%)', '超高 (4%+)']
    for ticker in tickers:
        if ticker in volatility_analysis:
            atr = volatility_analysis[ticker]['atr_analysis']
            atr_wr_matrix.append([
                atr.loc[ar, 'win_rate'] if ar in atr.index else 0 for ar in atr_ranges
            ])
        else:
            atr_wr_matrix.append([0, 0, 0, 0])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(atr_wr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=['Low', 'Mid', 'High', 'Super-High'],
                yticklabels=tickers,
                cbar_kws={'label': 'Win Rate'},
                vmin=0, vmax=1)
    ax.set_title('ATR Range Win Rate Heatmap by Ticker')
    ax.set_xlabel('ATR Range')
    ax.set_ylabel('Ticker')
    plt.tight_layout()
    save_figure(fig, 'phase1_7_atr_heatmap.png', figures_dir)
    
    # 図表5: 普遍性スコアレーダーチャート
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = list(universality_scores.keys())
    values = list(universality_scores.values())
    values += values[:1]  # 閉じるため
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label='Universality Score (9 tickers avg)')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.set_title('5 Discovery Universality Score', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    save_figure(fig, 'phase1_7_universality_radar.png', figures_dir)
    
    # 図表6: 月別平均損益率推移（10銘柄重ね合わせ）
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for ticker in tickers:
        if ticker in time_series_analysis:
            monthly_stats = time_series_analysis[ticker]['monthly_stats']
            ax.plot(monthly_stats.index.astype(str), monthly_stats['avg_profit_pct'], 
                    marker='o', label=ticker, alpha=0.7)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(-5, color='red', linestyle='--', linewidth=1, label='Alert Line (-5%)')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Avg Profit/Loss (%)')
    ax.set_title('Monthly Avg Profit/Loss Timeline (10 tickers)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, 'phase1_7_monthly_profit_timeline.png', figures_dir)


def create_integrated_report(basic_stats, big_loss_analysis, time_series_analysis, 
                             entry_quality_analysis, volatility_analysis, 
                             pump_dump_analysis, universality_scores, results_matrix):
    """
    Step 6: 統合レポート作成
    
    Args:
        各種分析結果とスコア
    """
    report_path = Path('docs/exit_strategy/PHASE1_7_MULTI_TICKER_RESULT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 1.7マルチティッカー大敗パターン検証結果\n\n")
        f.write(f"**作成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**検証銘柄数**: 10銘柄（武田薬品 + 9銘柄）\n")
        f.write(f"**参照元**: [PHASE1_7_MULTI_TICKER_DEFEAT_PATTERNS_DESIGN.md](PHASE1_7_MULTI_TICKER_DEFEAT_PATTERNS_DESIGN.md)\n\n")
        f.write("---\n\n")
        
        # 銘柄別基本統計
        f.write("## 1. 銘柄別基本統計\n\n")
        f.write("| Ticker | Trades | Win Rate | PF | Avg Profit % | Avg R-Multiple | Avg Holding Days |\n")
        f.write("|--------|--------|----------|----|--------------|-----------------|-----------------|\n")
        
        # Phase 1.7: 9銘柄のみ（武田除外）
        all_tickers = VALIDATION_TICKERS
        for ticker in all_tickers:
            if ticker in basic_stats:
                stats = basic_stats[ticker]
                f.write(f"| {ticker} | {stats['total_trades']} | {stats['win_rate']*100:.1f}% | "
                       f"{stats['pf']:.2f} | {stats['avg_profit_pct']:.2f}% | "
                       f"{stats['avg_r_multiple']:.2f} | {stats['avg_holding_days']:.1f} |\n")
        
        f.write("\n**Figure**: [Basic Stats Comparison](figures/phase1_7_basic_stats_comparison.png)\n\n")
        
        # 普遍性評価
        f.write("## 2. 普遍性評価結果\n\n")
        f.write("### 普遍性スコア（0-1）\n\n")
        f.write("| Discovery | Score | Confirmed Tickers |\n")
        f.write("|-----------|-------|-------------------|\n")
        
        for key, score in universality_scores.items():
            confirmed_count = int(score * 9)
            f.write(f"| {key} | {score:.2f} | {confirmed_count}/9 |\n")
        
        f.write("\n**Figure**: [Universality Radar](figures/phase1_7_universality_radar.png)\n\n")
        
        # 銘柄別再現性マトリックス
        f.write("### 銘柄別再現性マトリックス\n\n")
        f.write("| Ticker | Discovery 1 | Discovery 2 | Discovery 3 | Discovery 4 | Discovery 5 |\n")
        f.write("|--------|-------------|-------------|-------------|-------------|-------------|\n")
        
        for result in results_matrix:
            f.write(f"| {result['ticker']} | {result.get('発見1', 'N/A')} | "
                   f"{result.get('発見2', 'N/A')} | {result.get('発見3', 'N/A')} | "
                   f"{result.get('発見4', 'N/A')} | {result.get('発見5', 'N/A')} |\n")
        
        # トレンド強度分析サマリー
        f.write("\n## 3. トレンド強度分析サマリー\n\n")
        f.write("| Ticker | High Trend WR | Mid Trend WR | Low Trend WR | High Threshold |\n")
        f.write("|--------|---------------|--------------|--------------|----------------|\n")
        
        for ticker in all_tickers:
            if ticker in entry_quality_analysis:
                trend = entry_quality_analysis[ticker]['trend_analysis']
                thresholds = entry_quality_analysis[ticker]['trend_thresholds']
                
                high_wr = trend.loc['高', 'win_rate']*100 if '高' in trend.index else 0
                mid_wr = trend.loc['中', 'win_rate']*100 if '中' in trend.index else 0
                low_wr = trend.loc['低', 'win_rate']*100 if '低' in trend.index else 0
                
                f.write(f"| {ticker} | {high_wr:.1f}% | {mid_wr:.1f}% | {low_wr:.1f}% | {thresholds['high']:.4f} |\n")
        
        f.write("\n**Figure**: [Trend Strength Heatmap](figures/phase1_7_trend_strength_heatmap.png)\n\n")
        
        # 急騰→急落パターンサマリー
        f.write("## 4. 急騰→急落パターンサマリー\n\n")
        f.write("| Ticker | Pattern Count | Pattern Rate | PF with | PF without | Improvement |\n")
        f.write("|--------|---------------|--------------|---------|------------|-------------|\n")
        
        for ticker in all_tickers:
            if ticker in pump_dump_analysis:
                pd = pump_dump_analysis[ticker]
                f.write(f"| {ticker} | {pd['pattern_count']} | {pd['pattern_ratio']*100:.1f}% | "
                       f"{pd['pf_with_pattern']:.2f} | {pd['pf_without_pattern']:.2f} | "
                       f"{pd['pf_improvement']:.1f}% |\n")
        
        f.write("\n**Figure**: [Pump & Dump Comparison](figures/phase1_7_pump_dump_comparison.png)\n\n")
        
        # ATR分析サマリー
        f.write("## 5. ATR分析サマリー\n\n")
        f.write("| Ticker | Low ATR WR | Mid ATR WR | High ATR WR | Super-High ATR WR |\n")
        f.write("|--------|------------|------------|-------------|-------------------|\n")
        
        for ticker in all_tickers:
            if ticker in volatility_analysis:
                atr = volatility_analysis[ticker]['atr_analysis']
                
                low_wr = atr.loc['低 (0-2%)', 'win_rate']*100 if '低 (0-2%)' in atr.index else 0
                mid_wr = atr.loc['中 (2-3%)', 'win_rate']*100 if '中 (2-3%)' in atr.index else 0
                high_wr = atr.loc['高 (3-4%)', 'win_rate']*100 if '高 (3-4%)' in atr.index else 0
                super_high_wr = atr.loc['超高 (4%+)', 'win_rate']*100 if '超高 (4%+)' in atr.index else 0
                
                f.write(f"| {ticker} | {low_wr:.1f}% | {mid_wr:.1f}% | {high_wr:.1f}% | {super_high_wr:.1f}% |\n")
        
        f.write("\n**Figure**: [ATR Heatmap](figures/phase1_7_atr_heatmap.png)\n\n")
        
        # 主要発見事項
        f.write("## 6. 主要発見事項\n\n")
        
        f.write("### 発見1: 急騰→急落パターンの普遍性\n\n")
        discovery1_score = universality_scores['発見1_急騰急落']
        f.write(f"- **普遍性スコア**: {discovery1_score:.2f} ({int(discovery1_score*9)}/9銘柄で確認)\n")
        f.write(f"- **武田薬品との比較**: 56.1%の発生率、295.4%のPF改善余地\n")
        f.write(f"- **結論**: {'普遍的' if discovery1_score >= 0.6 else '部分的'}\n\n")
        
        f.write("### 発見2: トレンド強度の決定的影響\n\n")
        discovery2_score = universality_scores['発見2_トレンド強度']
        f.write(f"- **普遍性スコア**: {discovery2_score:.2f} ({int(discovery2_score*9)}/9銘柄で確認)\n")
        f.write(f"- **武田薬品との比較**: 高トレンド71.1%勝率 vs 中/低0%勝率\n")
        f.write(f"- **結論**: {'普遍的' if discovery2_score >= 0.6 else '部分的'}\n\n")
        
        f.write("### 発見3: ATRパラドックス\n\n")
        discovery3_score = universality_scores['発見3_ATRパラドックス']
        f.write(f"- **普遍性スコア**: {discovery3_score:.2f} ({int(discovery3_score*9)}/9銘柄で確認)\n")
        f.write(f"- **武田薬品との比較**: 超高ATR100%勝率 vs 低ATR22.1%勝率\n")
        f.write(f"- **結論**: {'普遍的' if discovery3_score >= 0.6 else '銘柄特異的'}\n\n")
        
        f.write("### 発見4: SMA乖離5%以上でのエントリー失敗\n\n")
        discovery4_score = universality_scores['発見4_SMA乖離5%以上']
        f.write(f"- **普遍性スコア**: {discovery4_score:.2f} ({int(discovery4_score*9)}/9銘柄で確認)\n")
        f.write(f"- **武田薬品との比較**: 33件のエントリーが全敗\n")
        f.write(f"- **結論**: {'普遍的' if discovery4_score >= 0.6 else '部分的'}\n\n")
        
        f.write("### 発見5: 極端下落相場の共通性\n\n")
        discovery5_score = universality_scores['発見5_極端下落相場']
        f.write(f"- **普遍性スコア**: {discovery5_score:.2f} ({int(discovery5_score*9)}/9銘柄で確認)\n")
        f.write(f"- **武田薬品との比較**: 2021年4月、8月の平均損益率-5%以上\n")
        f.write(f"- **結論**: {'市場全体影響' if discovery5_score >= 0.6 else 'セクター依存'}\n\n")
        
        # 次フェーズへの提案
        f.write("## 7. 次フェーズへの提案\n\n")
        f.write("### Phase 1.8: フィルター統合実装\n\n")
        f.write("**実装優先度**（普遍性スコア順）:\n\n")
        
        sorted_scores = sorted(universality_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (key, score) in enumerate(sorted_scores, 1):
            f.write(f"{i}. **{key}** (スコア: {score:.2f})\n")
            
            if 'トレンド強度' in key and score >= 0.6:
                f.write("   - 実装: entry_trend_strength >= 閾値（67%ile）のみエントリー許可\n")
                f.write("   - 統合先: validate_exit_simple_v2.py GCStrategy\n")
                f.write("   - 期待効果: 勝率25% → 60%以上、PF 0.5 → 2.0以上\n\n")
            
            elif 'SMA乖離' in key and score >= 0.6:
                f.write("   - 実装: sma_distance_pct < 5%のみエントリー許可\n")
                f.write("   - 統合先: validate_exit_simple_v2.py GCStrategy\n")
                f.write("   - 期待効果: 追いかけエントリー失敗の排除\n\n")
            
            elif '急騰急落' in key and score >= 0.6:
                f.write("   - 実装: 利確ライン強化（entry_price + entry_atr × 10）\n")
                f.write("   - 統合先: validate_exit_simple_v2.py GCStrategy exit条件\n")
                f.write("   - 期待効果: パターン発生率56% → 15%以下\n\n")
        
        f.write("---\n\n")
        f.write(f"**作成者**: validate_phase1_7_multi_ticker.py\n")
        f.write(f"**最終更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n  統合レポート作成完了: {report_path}")


def main():
    """
    Phase 1.7メイン実行関数
    """
    print("=" * 80)
    print("Phase 1.7: Multi-Ticker Defeat Pattern Validation")
    print("=" * 80)
    
    try:
        # Step 1: データ読み込み・前処理
        print("\nStep 1: Data Loading & Preprocessing")
        ticker_data = load_and_preprocess_data()
        
        if len(ticker_data) == 0:
            print("[ERROR] No ticker data loaded")
            return
        
        # Step 2: 銘柄別基本統計集計
        print("\nStep 2: Basic Statistics by Ticker")
        basic_stats = calculate_basic_stats_all_tickers(ticker_data)
        
        # Step 3: 分析1～5のマルチティッカー実行
        print("\nStep 3: Analysis 1-5 Execution")
        big_loss_analysis = run_analysis1(ticker_data)
        time_series_analysis = run_analysis2(ticker_data)
        entry_quality_analysis = run_analysis3(ticker_data)
        volatility_analysis = run_analysis4(ticker_data)
        pump_dump_analysis = run_analysis5(ticker_data)
        
        # Step 4: 銘柄間比較・普遍性評価
        print("\nStep 4: Cross-Ticker Comparison & Universality Evaluation")
        universality_scores, results_matrix = evaluate_universality(
            pump_dump_analysis, entry_quality_analysis, 
            volatility_analysis, time_series_analysis
        )
        
        # Step 5: 可視化
        print("\nStep 5: Figure Generation")
        generate_all_figures(basic_stats, entry_quality_analysis, 
                           pump_dump_analysis, volatility_analysis,
                           time_series_analysis, universality_scores)
        
        # Step 6: 統合レポート作成
        print("\nStep 6: Integrated Report Creation")
        create_integrated_report(basic_stats, big_loss_analysis, 
                                time_series_analysis, entry_quality_analysis,
                                volatility_analysis, pump_dump_analysis,
                                universality_scores, results_matrix)
        
        print("\n" + "=" * 80)
        print("Phase 1.7 Validation Complete")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
