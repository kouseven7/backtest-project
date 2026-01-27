# Phase 1.7マルチティッカー大敗パターン検証設計書

**作成日**: 2026-01-26  
**目的**: Phase 1.6で武田薬品（4502.T）で発見した大敗パターンが、他の9銘柄でも普遍的に適用されるか検証  
**参照元**: 
- [PHASE1_6_DEFEAT_PATTERNS_RESULT.md](PHASE1_6_DEFEAT_PATTERNS_RESULT.md)
- [PHASE1_6_DEFEAT_PATTERNS_ANALYSIS_DESIGN.md](PHASE1_6_DEFEAT_PATTERNS_ANALYSIS_DESIGN.md)
- [PHASE1_6_TREND_FILTER_DESIGN.md](PHASE1_6_TREND_FILTER_DESIGN.md)

---

## 1. 目的とゴール

### 主目的
Phase 1.6で武田薬品（4502.T）から発見した5つの大敗パターンが、他の9銘柄でも共通して観測されるか検証し、パターンの普遍性を確認する。

### 検証対象銘柄（9銘柄）
```python
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
```

### 武田薬品（Phase 1.6）で発見した主要パターン

| パターン名 | 発見内容 | 影響度 |
|-----------|---------|--------|
| **発見1: 急騰→急落** | 471件（56.1%）が最大利益65.6%到達後に平均-3.02%の損失に転落 | PF影響: 2.09→0.53（295.4%悪化） |
| **発見2: トレンド強度** | 高: 勝率71.1%、中/低: 勝率0% | 決定的な勝敗要因 |
| **発見3: ATRパラドックス** | 超高ATR（4%+）: 勝率100%、低ATR（0-2%）: 勝率22.1% | 逆相関 |
| **発見4: SMA乖離5%以上** | 33件のエントリーが全敗 | 追いかけエントリー失敗 |
| **発見5: 極端下落相場** | 2021年4月、8月の平均損益率-5%以上 | 期間集中リスク |

### 成功条件
- [ ] 9銘柄すべてで分析1～5を完了
- [ ] 各銘柄の基本統計（取引数、勝率、PF）を集計
- [ ] 5つの発見事項が各銘柄で再現されるか確認
- [ ] 銘柄間の共通点・相違点を抽出（最低3項目）
- [ ] 普遍性スコア（9銘柄中何銘柄でパターン確認）を算出
- [ ] 結果をMarkdownレポートに記録（PHASE1_7_MULTI_TICKER_RESULT.md）
- [ ] 銘柄別比較図表を5種類以上作成

---

## 2. データ構造確認

### CSVファイル構造
```
ファイル: results/phase1.6_trades_20260126_124409a.csv
総取引数: 約8400件（10銘柄 × 平均840件）

カラム一覧（29列）:
- trade_id: 取引ID
- entry_date: エントリー日
- entry_price: エントリー価格
- exit_date: エグジット日
- exit_price: エグジット価格
- profit_loss: 損益額
- exit_reason: エグジット理由（stop_loss/trailing_stop/dead_cross/force_close）
- holding_days: 保有日数
- profit_loss_pct: 損益率（%）
- r_multiple: リスクリワードレシオ
- entry_gap_pct: エントリーギャップ率（%）
- max_profit_pct: 最大利益率（%）
- entry_atr_pct: エントリー時ATR（%）
- sma_distance_pct: SMA乖離率（%）
- entry_trend_strength: エントリー時トレンド強度
- entry_volume: エントリー時出来高
- exit_volume: エグジット時出来高
- volume_ratio: 出来高比率
- exit_signal_pct: エグジットシグナル率
- sma_value: SMA値
- trend_value: トレンド値
- above_trend: トレンド以上フラグ
- unused1: 未使用1
- entry_price_base: エントリー価格ベース
- unused2: 未使用2
- unused3: 未使用3
- ticker: ティッカー（4502.T, 7203.T等）
- stop_loss_pct: 損切設定（0.03/0.05/0.07）
- trailing_stop_pct: トレーリング設定（0.05～0.30）
```

### 銘柄別フィルター
```python
# 各銘柄ごとにデータ抽出
df = pd.read_csv('results/phase1.6_trades_20260126_124409a.csv', header=None)
df.columns = column_names  # 29列の列名設定

ticker_data = {}
for ticker in VALIDATION_TICKERS:
    ticker_data[ticker] = df[df['ticker'] == ticker].copy()
    print(f"{ticker}: {len(ticker_data[ticker])}件")
```

---

## 3. 分析設計（マルチティッカー対応）

### 全体フロー
```
Step 1: データ読み込み・前処理
  ↓
Step 2: 銘柄別基本統計集計
  ↓
Step 3: 分析1～5を各銘柄で実行
  ↓
Step 4: 銘柄間比較・普遍性評価
  ↓
Step 5: 統合レポート作成
```

---

### Step 2: 銘柄別基本統計集計

**目的**: 各銘柄の取引特性を把握し、武田薬品との比較ベースラインを作成

**集計項目**:
- 総取引数
- 勝ち取引数
- 勝率（%）
- 平均損益率（%）
- PF（Profit Factor）
- 平均R倍率
- 平均保有日数
- exit_reason分布（stop_loss/trailing_stop/dead_cross/force_close件数）

**出力物**:
- 銘柄別基本統計比較表（Markdownテーブル）
- 勝率比較棒グラフ（10銘柄横並び）
- PF比較棒グラフ（10銘柄横並び）
- exit_reason分布積み上げ棒グラフ（10銘柄）

**Python実装概要**:
```python
def calculate_basic_stats(ticker_df):
    """
    銘柄別基本統計を計算
    """
    total_trades = len(ticker_df)
    winning_trades = len(ticker_df[ticker_df['profit_loss_pct'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_profit_pct = ticker_df['profit_loss_pct'].mean()
    
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

# 全銘柄の基本統計を集計
basic_stats = {}
for ticker in ['4502.T'] + VALIDATION_TICKERS:
    basic_stats[ticker] = calculate_basic_stats(ticker_data[ticker])
```

---

### Step 3: 分析1～5のマルチティッカー実行

#### 分析1: 大損失トレードの特徴抽出（各銘柄）

**目的**: profit_loss_pct < -10%の大損失トレードが各銘柄で何件あり、どのような特徴を持つか

**手順（各銘柄）**:
1. 大損失トレード（profit_loss_pct < -10%）を抽出
2. 通常トレード（profit_loss_pct >= -10%）と比較
3. 銘柄間で大損失比率を比較

**分析項目**:
- 大損失取引件数と割合
- 平均値比較: entry_gap_pct, entry_atr_pct, holding_days, max_profit_pct, entry_trend_strength, sma_distance_pct
- exit_reason内訳

**出力物**:
- 銘柄別大損失比率比較表
- 大損失トレード特徴比較ヒートマップ（10銘柄 × 6指標）
- exit_reason内訳比較（10銘柄の円グラフ or 積み上げ棒グラフ）

**Python実装概要**:
```python
def analyze_big_loss_per_ticker(ticker_df, ticker_name):
    """
    銘柄ごとの大損失トレード分析
    """
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
    
    return {
        'ticker': ticker_name,
        'big_loss_count': len(big_loss),
        'big_loss_ratio': big_loss_ratio,
        'big_loss_features': big_loss_features,
        'exit_reason_dist': exit_reason_dist
    }

# 全銘柄で実行
big_loss_analysis = {}
for ticker in ['4502.T'] + VALIDATION_TICKERS:
    big_loss_analysis[ticker] = analyze_big_loss_per_ticker(ticker_data[ticker], ticker)
```

---

#### 分析2: 時系列での勝敗パターン（各銘柄）

**目的**: 各銘柄で月別・四半期別の成績変動を可視化し、共通の低迷期間を特定

**手順（各銘柄）**:
1. entry_dateをdatetime型に変換
2. 月別・四半期別に集計
3. 「平均profit_loss_pct < -5%」の期間をリストアップ
4. 10銘柄で共通する低迷期間を特定

**集計項目**:
- 総取引数
- 勝率
- 平均profit_loss_pct
- 最大損失

**出力物**:
- 銘柄別月別成績テーブル（各銘柄）
- 10銘柄の月別平均profit_loss_pct折れ線グラフ（重ね合わせ）
- 低迷期間共通性分析表（2021年4月、8月等が何銘柄で再現）

**Python実装概要**:
```python
def analyze_time_series_per_ticker(ticker_df, ticker_name):
    """
    銘柄ごとの時系列分析
    """
    ticker_df['entry_date'] = pd.to_datetime(ticker_df['entry_date'])
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
    
    return {
        'ticker': ticker_name,
        'monthly_stats': monthly,
        'poor_periods': poor_periods
    }

# 全銘柄で実行
time_series_analysis = {}
for ticker in ['4502.T'] + VALIDATION_TICKERS:
    time_series_analysis[ticker] = analyze_time_series_per_ticker(ticker_data[ticker], ticker)

# 共通低迷期間の特定
from collections import Counter
all_poor_periods = []
for analysis in time_series_analysis.values():
    all_poor_periods.extend(analysis['poor_periods'])

common_poor_periods = Counter(all_poor_periods)
# 5銘柄以上で共通する期間を抽出
common_periods = {period: count for period, count in common_poor_periods.items() if count >= 5}
```

---

#### 分析3: エントリー品質の分析（各銘柄）

**目的**: SMA乖離・トレンド強度と成績の相関を各銘柄で検証

**分析3-1: SMA乖離範囲別成績**

sma_distance_pct範囲:
- 0-5%（GC直後）
- 5-10%
- 10-15%
- 15-20%
- 20%以上（追いかけエントリー）

**分析3-2: トレンド強度別成績**

entry_trend_strength分類（各銘柄の33%ile, 67%ile）:
- 低: 0-33パーセンタイル
- 中: 33-66パーセンタイル
- 高: 66-100パーセンタイル

**出力物**:
- 銘柄別SMA乖離範囲成績比較表
- SMA乖離5%以上エントリーの勝率比較（10銘柄棒グラフ）
- トレンド強度別勝率比較ヒートマップ（10銘柄 × 3強度レベル）
- 武田薬品での発見「高トレンド: 71.1%勝率」が他銘柄で再現されるか検証

**Python実装概要**:
```python
def analyze_entry_quality_per_ticker(ticker_df, ticker_name):
    """
    銘柄ごとのエントリー品質分析
    """
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
    sma_analysis['win_rate'] = ticker_df.groupby('sma_range').apply(
        lambda x: (x['profit_loss_pct'] > 0).mean()
    )
    
    # トレンド強度分類
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
    trend_analysis['win_rate'] = ticker_df.groupby('trend_strength_level').apply(
        lambda x: (x['profit_loss_pct'] > 0).mean()
    )
    
    return {
        'ticker': ticker_name,
        'sma_analysis': sma_analysis,
        'trend_analysis': trend_analysis,
        'trend_thresholds': {'mid': threshold_mid, 'high': threshold_high}
    }

# 全銘柄で実行
entry_quality_analysis = {}
for ticker in ['4502.T'] + VALIDATION_TICKERS:
    entry_quality_analysis[ticker] = analyze_entry_quality_per_ticker(ticker_data[ticker], ticker)
```

---

#### 分析4: ボラティリティの影響（各銘柄）

**目的**: entry_atr_pctとprofit_loss_pctの関係を各銘柄で検証

**ATR範囲分類**:
- 低: 0-2%
- 中: 2-3%
- 高: 3-4%
- 超高: 4%以上

**出力物**:
- 銘柄別ATR基本統計比較表
- ATR範囲別勝率比較ヒートマップ（10銘柄 × 4 ATR範囲）
- 武田薬品での「ATRパラドックス」（超高ATR勝率100%）が他銘柄で再現されるか検証
- 散布図（entry_atr_pct vs profit_loss_pct）10銘柄サブプロット

**Python実装概要**:
```python
def analyze_volatility_per_ticker(ticker_df, ticker_name):
    """
    銘柄ごとのボラティリティ分析
    """
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
    atr_analysis['win_rate'] = ticker_df.groupby('atr_range').apply(
        lambda x: (x['profit_loss_pct'] > 0).mean()
    )
    
    return {
        'ticker': ticker_name,
        'atr_stats': atr_stats,
        'atr_analysis': atr_analysis
    }

# 全銘柄で実行
volatility_analysis = {}
for ticker in ['4502.T'] + VALIDATION_TICKERS:
    volatility_analysis[ticker] = analyze_volatility_per_ticker(ticker_data[ticker], ticker)
```

---

#### 分析5: 急騰→急落パターンの検出（各銘柄）

**目的**: 「最大利益大→最終損失」パターンが各銘柄で何件発生し、PFへの影響度を定量化

**検出条件**:
1. max_profit_pct > 15%（一度大きく上昇）
2. profit_loss_pct < 0%（最終的に損失）
3. holding_days < 30日（短期間で転落）

**分析項目**:
- 該当取引件数と割合
- 合計損失額
- 平均max_profit_pct
- 平均profit_loss_pct
- PF影響度（パターン除外時vs含む時）

**出力物**:
- 銘柄別急騰→急落パターン発生率比較表
- 武田薬品での「56.1%該当、PF 2.09→0.53（295.4%悪化）」が他銘柄で再現されるか検証
- 銘柄別PF影響度比較棒グラフ（パターン除外時の改善率）
- 時系列プロット（10銘柄のパターン発生時期重ね合わせ）

**Python実装概要**:
```python
def detect_pump_dump_per_ticker(ticker_df, ticker_name):
    """
    銘柄ごとの急騰→急落パターン検出
    """
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
    
    return {
        'ticker': ticker_name,
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

def calculate_pf(trades_df):
    """PF計算"""
    total_profit = trades_df[trades_df['profit_loss_pct'] > 0]['profit_loss_pct'].sum()
    total_loss = abs(trades_df[trades_df['profit_loss_pct'] < 0]['profit_loss_pct'].sum())
    return total_profit / total_loss if total_loss > 0 else 0

# 全銘柄で実行
pump_dump_analysis = {}
for ticker in ['4502.T'] + VALIDATION_TICKERS:
    pump_dump_analysis[ticker] = detect_pump_dump_per_ticker(ticker_data[ticker], ticker)
```

---

### Step 4: 銘柄間比較・普遍性評価

**目的**: 5つの発見事項が何銘柄で再現されるか定量化

#### 普遍性判定基準

| 発見事項 | 再現条件 | 普遍性スコア算出 |
|---------|---------|--------------|
| **発見1: 急騰→急落** | パターン発生率 > 30% かつ PF改善率 > 50% | 条件満たす銘柄数 / 9 |
| **発見2: トレンド強度** | 高トレンド勝率 > 60% かつ 中/低トレンド勝率 < 30% | 条件満たす銘柄数 / 9 |
| **発見3: ATRパラドックス** | 超高ATR勝率 > 中央値+20pt かつ 低ATR勝率 < 中央値 | 条件満たす銘柄数 / 9 |
| **発見4: SMA乖離5%以上** | sma_distance_pct >= 5%の勝率 < 30% | 条件満たす銘柄数 / 9 |
| **発見5: 極端下落相場** | 2021年4月 or 8月の平均損益率 < -3% | 条件満たす銘柄数 / 9 |

**出力物**:
- 普遍性スコア集計表（5発見 × 10銘柄のチェックマーク表）
- 普遍性スコアレーダーチャート（5発見の9銘柄平均）
- 銘柄特性分類（パターン類似グループ）

**Python実装概要**:
```python
def evaluate_universality():
    """
    5つの発見事項の普遍性を評価
    """
    universality_scores = {
        '発見1_急騰急落': 0,
        '発見2_トレンド強度': 0,
        '発見3_ATRパラドックス': 0,
        '発見4_SMA乖離5%以上': 0,
        '発見5_極端下落相場': 0
    }
    
    results_matrix = []  # 銘柄 × 発見事項のマトリックス
    
    for ticker in VALIDATION_TICKERS:
        ticker_results = {'ticker': ticker}
        
        # 発見1: 急騰→急落
        pump_dump = pump_dump_analysis[ticker]
        if pump_dump['pattern_ratio'] > 0.3 and pump_dump['pf_improvement'] > 50:
            universality_scores['発見1_急騰急落'] += 1
            ticker_results['発見1'] = '✓'
        else:
            ticker_results['発見1'] = '✗'
        
        # 発見2: トレンド強度
        trend = entry_quality_analysis[ticker]['trend_analysis']
        high_win_rate = trend.loc['高', 'win_rate'] if '高' in trend.index else 0
        mid_low_win_rate = (trend.loc['中', 'win_rate'] + trend.loc['低', 'win_rate']) / 2 if '中' in trend.index and '低' in trend.index else 0
        if high_win_rate > 0.6 and mid_low_win_rate < 0.3:
            universality_scores['発見2_トレンド強度'] += 1
            ticker_results['発見2'] = '✓'
        else:
            ticker_results['発見2'] = '✗'
        
        # 発見3: ATRパラドックス
        atr = volatility_analysis[ticker]['atr_analysis']
        if '超高 (4%+)' in atr.index and '低 (0-2%)' in atr.index:
            super_high_wr = atr.loc['超高 (4%+)', 'win_rate']
            low_wr = atr.loc['低 (0-2%)', 'win_rate']
            median_wr = atr['win_rate'].median()
            if super_high_wr > median_wr + 0.2 and low_wr < median_wr:
                universality_scores['発見3_ATRパラドックス'] += 1
                ticker_results['発見3'] = '✓'
            else:
                ticker_results['発見3'] = '✗'
        else:
            ticker_results['発見3'] = '✗'
        
        # 発見4: SMA乖離5%以上
        sma = entry_quality_analysis[ticker]['sma_analysis']
        sma_5plus_ranges = ['5-10%', '10-15%', '15-20%', '20%+']
        sma_5plus_wr = sma[sma.index.isin(sma_5plus_ranges)]['win_rate'].mean() if any(r in sma.index for r in sma_5plus_ranges) else 1.0
        if sma_5plus_wr < 0.3:
            universality_scores['発見4_SMA乖離5%以上'] += 1
            ticker_results['発見4'] = '✓'
        else:
            ticker_results['発見4'] = '✗'
        
        # 発見5: 極端下落相場
        poor_periods = time_series_analysis[ticker]['poor_periods']
        target_periods = ['2021-04', '2021-08']
        if any(str(period) in target_periods for period in poor_periods):
            universality_scores['発見5_極端下落相場'] += 1
            ticker_results['発見5'] = '✓'
        else:
            ticker_results['発見5'] = '✗'
        
        results_matrix.append(ticker_results)
    
    # 普遍性スコア正規化（0-1）
    universality_scores_normalized = {k: v / 9 for k, v in universality_scores.items()}
    
    return universality_scores_normalized, results_matrix
```

---

## 4. 可視化設計（10銘柄比較）

### 図表1: 銘柄別基本統計比較（勝率・PF）
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 勝率比較
tickers = ['4502.T'] + VALIDATION_TICKERS
win_rates = [basic_stats[t]['win_rate']*100 for t in tickers]
axes[0].bar(range(len(tickers)), win_rates, color='skyblue', edgecolor='black')
axes[0].set_xticks(range(len(tickers)))
axes[0].set_xticklabels(tickers, rotation=45, ha='right')
axes[0].set_ylabel('勝率 (%)')
axes[0].set_title('銘柄別勝率比較')
axes[0].axhline(50, color='red', linestyle='--', label='50%ライン')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# PF比較
pf_values = [basic_stats[t]['pf'] for t in tickers]
axes[1].bar(range(len(tickers)), pf_values, color='green', edgecolor='black')
axes[1].set_xticks(range(len(tickers)))
axes[1].set_xticklabels(tickers, rotation=45, ha='right')
axes[1].set_ylabel('Profit Factor (PF)')
axes[1].set_title('銘柄別PF比較')
axes[1].axhline(1.0, color='red', linestyle='--', label='損益分岐点 (PF=1.0)')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/phase1_7_basic_stats_comparison.png', dpi=150)
```

### 図表2: トレンド強度別勝率ヒートマップ
```python
# 10銘柄 × 3強度レベル（低/中/高）のヒートマップ
trend_wr_matrix = []
for ticker in ['4502.T'] + VALIDATION_TICKERS:
    trend = entry_quality_analysis[ticker]['trend_analysis']
    trend_wr_matrix.append([
        trend.loc['低', 'win_rate'] if '低' in trend.index else 0,
        trend.loc['中', 'win_rate'] if '中' in trend.index else 0,
        trend.loc['高', 'win_rate'] if '高' in trend.index else 0
    ])

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(trend_wr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            xticklabels=['低', '中', '高'], 
            yticklabels=['4502.T'] + VALIDATION_TICKERS,
            cbar_kws={'label': '勝率'},
            vmin=0, vmax=1)
ax.set_title('銘柄別トレンド強度別勝率ヒートマップ')
ax.set_xlabel('トレンド強度')
ax.set_ylabel('銘柄')
plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/phase1_7_trend_strength_heatmap.png', dpi=150)
```

### 図表3: 急騰→急落パターン発生率とPF影響
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# パターン発生率
pattern_ratios = [pump_dump_analysis[t]['pattern_ratio']*100 for t in tickers]
axes[0].bar(range(len(tickers)), pattern_ratios, color='orange', edgecolor='black')
axes[0].set_xticks(range(len(tickers)))
axes[0].set_xticklabels(tickers, rotation=45, ha='right')
axes[0].set_ylabel('パターン発生率 (%)')
axes[0].set_title('急騰→急落パターン発生率（銘柄別）')
axes[0].axhline(56.1, color='red', linestyle='--', label='武田薬品（56.1%）')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# PF改善率（パターン除外時）
pf_improvements = [pump_dump_analysis[t]['pf_improvement'] for t in tickers]
axes[1].bar(range(len(tickers)), pf_improvements, color='purple', edgecolor='black')
axes[1].set_xticks(range(len(tickers)))
axes[1].set_xticklabels(tickers, rotation=45, ha='right')
axes[1].set_ylabel('PF改善率 (%)')
axes[1].set_title('急騰→急落パターン除外時のPF改善率')
axes[1].axhline(295.4, color='red', linestyle='--', label='武田薬品（295.4%）')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/phase1_7_pump_dump_comparison.png', dpi=150)
```

### 図表4: ATR範囲別勝率ヒートマップ
```python
# 10銘柄 × 4 ATR範囲のヒートマップ
atr_wr_matrix = []
atr_ranges = ['低 (0-2%)', '中 (2-3%)', '高 (3-4%)', '超高 (4%+)']
for ticker in ['4502.T'] + VALIDATION_TICKERS:
    atr = volatility_analysis[ticker]['atr_analysis']
    atr_wr_matrix.append([
        atr.loc[ar, 'win_rate'] if ar in atr.index else 0 for ar in atr_ranges
    ])

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(atr_wr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=['低', '中', '高', '超高'],
            yticklabels=['4502.T'] + VALIDATION_TICKERS,
            cbar_kws={'label': '勝率'},
            vmin=0, vmax=1)
ax.set_title('銘柄別ATR範囲別勝率ヒートマップ')
ax.set_xlabel('ATR範囲')
ax.set_ylabel('銘柄')
plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/phase1_7_atr_heatmap.png', dpi=150)
```

### 図表5: 普遍性スコアレーダーチャート
```python
from math import pi

universality_scores_norm, _ = evaluate_universality()

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

categories = list(universality_scores_norm.keys())
values = list(universality_scores_norm.values())
values += values[:1]  # 閉じるため

angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2, label='普遍性スコア（9銘柄平均）')
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax.set_title('5つの発見事項の普遍性スコア', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/phase1_7_universality_radar.png', dpi=150)
```

### 図表6: 月別平均損益率推移（10銘柄重ね合わせ）
```python
fig, ax = plt.subplots(figsize=(16, 8))

for ticker in ['4502.T'] + VALIDATION_TICKERS:
    monthly_stats = time_series_analysis[ticker]['monthly_stats']
    ax.plot(monthly_stats.index.astype(str), monthly_stats['avg_profit_pct'], 
            marker='o', label=ticker, alpha=0.7)

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axhline(-5, color='red', linestyle='--', linewidth=1, label='警戒ライン (-5%)')
ax.set_xlabel('年月')
ax.set_ylabel('平均損益率 (%)')
ax.set_title('月別平均損益率推移（10銘柄比較）')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('docs/exit_strategy/figures/phase1_7_monthly_profit_timeline.png', dpi=150)
```

---

## 5. 実装フロー

### メインスクリプト構造
```python
"""
Phase 1.7マルチティッカー大敗パターン検証スクリプト

Phase 1.6で武田薬品（4502.T）から発見した5つの大敗パターンが、
他の9銘柄でも普遍的に適用されるか検証する。

主な機能:
- 銘柄別基本統計集計
- 分析1～5のマルチティッカー実行
- 銘柄間比較・普遍性評価
- 統合レポート生成

Author: Backtest Project Team
Created: 2026-01-26
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
    "7203.T", "9984.T", "8306.T", "6758.T", "9983.T",
    "6501.T", "8001.T", "4063.T", "6861.T"
]

def main():
    print("=" * 80)
    print("Phase 1.7: マルチティッカー大敗パターン検証")
    print("=" * 80)
    
    # Step 1: データ読み込み・前処理
    print("\nStep 1: データ読み込み・前処理")
    ticker_data = load_and_preprocess_data()
    
    # Step 2: 銘柄別基本統計集計
    print("\nStep 2: 銘柄別基本統計集計")
    basic_stats = calculate_basic_stats_all_tickers(ticker_data)
    
    # Step 3: 分析1～5のマルチティッカー実行
    print("\nStep 3: 分析1～5の実行")
    print("  分析1: 大損失トレードの特徴抽出")
    big_loss_analysis = run_analysis1(ticker_data)
    
    print("  分析2: 時系列での勝敗パターン")
    time_series_analysis = run_analysis2(ticker_data)
    
    print("  分析3: エントリー品質の分析")
    entry_quality_analysis = run_analysis3(ticker_data)
    
    print("  分析4: ボラティリティの影響")
    volatility_analysis = run_analysis4(ticker_data)
    
    print("  分析5: 急騰→急落パターンの検出")
    pump_dump_analysis = run_analysis5(ticker_data)
    
    # Step 4: 銘柄間比較・普遍性評価
    print("\nStep 4: 銘柄間比較・普遍性評価")
    universality_scores, results_matrix = evaluate_universality(
        pump_dump_analysis, entry_quality_analysis, 
        volatility_analysis, time_series_analysis
    )
    
    # Step 5: 可視化
    print("\nStep 5: 可視化生成")
    generate_all_figures(basic_stats, entry_quality_analysis, 
                         pump_dump_analysis, volatility_analysis,
                         time_series_analysis, universality_scores)
    
    # Step 6: 統合レポート作成
    print("\nStep 6: 統合レポート作成")
    create_integrated_report(basic_stats, big_loss_analysis, 
                            time_series_analysis, entry_quality_analysis,
                            volatility_analysis, pump_dump_analysis,
                            universality_scores, results_matrix)
    
    print("\n" + "=" * 80)
    print("Phase 1.7検証完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

---

## 6. 成功条件チェックリスト

### 実行確認
- [ ] 9銘柄すべてでデータ読み込み成功（最低100件以上の取引データ）
- [ ] 分析1完了: 銘柄別大損失トレード特徴抽出
- [ ] 分析2完了: 銘柄別時系列パターン可視化
- [ ] 分析3完了: 銘柄別エントリー品質分析（SMA乖離・トレンド強度）
- [ ] 分析4完了: 銘柄別ボラティリティ影響分析
- [ ] 分析5完了: 銘柄別急騰→急落パターン検出

### 比較分析
- [ ] 銘柄別基本統計比較表作成（勝率、PF、平均損益率等）
- [ ] 5つの発見事項の普遍性スコア算出（各0-1の範囲）
- [ ] 銘柄間共通点を最低3項目抽出
- [ ] 銘柄間相違点を最低3項目抽出

### 可視化
- [ ] 図表6種類以上作成
  - [ ] 銘柄別基本統計比較（勝率・PF）
  - [ ] トレンド強度別勝率ヒートマップ
  - [ ] 急騰→急落パターン発生率とPF影響
  - [ ] ATR範囲別勝率ヒートマップ
  - [ ] 普遍性スコアレーダーチャート
  - [ ] 月別平均損益率推移（10銘柄重ね合わせ）

### レポート
- [ ] 統合レポート（PHASE1_7_MULTI_TICKER_RESULT.md）作成
- [ ] 各銘柄の分析結果記載
- [ ] 普遍性評価結果記載
- [ ] 次フェーズへの提案記載（Phase 1.8: フィルター統合実装）

---

## 7. 期待される発見

### 仮説1: トレンド強度は普遍的
- 9銘柄中7銘柄以上で「高トレンド勝率 > 60%、中/低トレンド勝率 < 30%」が再現される
- 普遍性スコア: 0.7以上（推定）

### 仮説2: 急騰→急落パターンは業種依存
- ボラティリティの高い業種（ソフトバンク、ファーストリテイリング）で発生率高
- 安定業種（MUFG、伊藤忠）で発生率低
- 普遍性スコア: 0.5-0.7（推定）

### 仮説3: ATRパラドックスは一部銘柄のみ
- 武田薬品の「超高ATR勝率100%」は特異ケースの可能性
- 9銘柄中3-4銘柄でのみ再現
- 普遍性スコア: 0.3-0.5（推定）

### 仮説4: SMA乖離5%以上は普遍的
- 9銘柄中8銘柄以上で「sma_distance_pct >= 5%の勝率 < 30%」が再現される
- 追いかけエントリー失敗は業種横断的
- 普遍性スコア: 0.8以上（推定）

### 仮説5: 極端下落相場は市場全体影響
- 2021年4月、8月の低迷は9銘柄中6銘柄以上で確認
- セクターローテーションによる一部例外あり
- 普遍性スコア: 0.6-0.8（推定）

---

## 8. 次フェーズへの展開

### Phase 1.8: フィルター統合実装

**実装内容**:
1. **トレンド強度フィルター**: entry_trend_strength >= 閾値（67%ile）のみエントリー許可
2. **SMA乖離フィルター**: sma_distance_pct < 5%のみエントリー許可
3. **ATR上限フィルター**: entry_atr_pct < 4%のエントリーを除外（超高ATR環境回避）

**検証方法**:
- validate_exit_simple_v2.pyのGCStrategyに統合
- 10銘柄で再度バックテスト実行
- PF改善効果を定量化

**期待効果**:
- 全体PF: 0.5-0.8 → 1.5-2.5（推定）
- 勝率: 25% → 60%以上（推定）
- 急騰→急落パターン削減: 56% → 15%以下（推定）

---

**作成者**: Phase 1.7設計チーム  
**最終更新**: 2026-01-26
