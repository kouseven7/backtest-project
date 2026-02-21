# Phase 1.9単体フィルター設計書

**作成日**: 2026-01-26  
**目的**: トレンド強度フィルター（Phase 1.8）の普遍性スコア0.33を受け、他の単体フィルターの効果を検証し、将来的な複合フィルター（AND/OR組み合わせ）の基盤を構築する  
**参照元**:
- [PHASE1_6_DEFEAT_PATTERNS_RESULT.md](PHASE1_6_DEFEAT_PATTERNS_RESULT.md)
- [PHASE1_8_TREND_FILTER_RESULT.md](PHASE1_8_TREND_FILTER_RESULT.md)
- [PHASE1_8_EXECUTION_SUMMARY.md](PHASE1_8_EXECUTION_SUMMARY.md)

---

## ゴール

1. **MA乖離<5%エントリー禁止フィルター**の設計図を完成させる
2. **ATR適正範囲（中〜高）フィルター**の設計図を完成させる
3. **ATRベースの利確**（利確ライン = entry_price + (entry_atr × 10)）の設計図を完成させる
4. 9銘柄＋武田薬品（参考銘柄）でフィルタースクリプトを作成する
5. 単体フィルターだが将来的な複合フィルター（AND/OR）も視野に入れた設計とする

---

## 背景と課題

### Phase 1.8の知見

- **トレンド強度フィルター単体**: 普遍性スコア0.33（3/9銘柄で改善）
- **改善銘柄**: 9984.T（+96.9%）、8306.T（+90.5%）、4063.T（+41.6%）
- **悪化銘柄**: 6861.T（-76.8%）、6758.T（-52.0%）など6銘柄

### Phase 1.6の発見事項（武田薬品）

- **発見1**: 急騰→急落パターン（56.1%の取引）→ 他銘柄では稀（武田特有）
- **発見2**: トレンド強度の決定的影響 → Phase 1.8で部分的に確認
- **発見3**: ATRパラドックス（超高ATR: 勝率100%、低ATR: 勝率22.1%）
- **発見4**: **SMA乖離5%以上でのエントリー全敗**（33件）→ 普遍性スコア0.33
- **発見5**: 極端な下落相場

### Phase 1.9の方針

1. **単体フィルターの効果を個別検証**: トレンド強度以外のフィルターの普遍性を確認
2. **将来の複合フィルター準備**: AND条件（厳格）/OR条件（緩和）の組み合わせ基盤構築
3. **武田薬品を参考銘柄として追加**: 急騰→急落パターン銘柄での知見獲得

---

## 設計方針

### 共通原則

1. **決定論的判定**: 同じデータで再実行すると同じ結果を返す
2. **前日データで判断**: ルックアヘッドバイアス禁止（`.shift(1)`適用）
3. **翌日始値でエントリー**: `data['Open'].iloc[idx + 1]`
4. **閾値の自動算出**: パーセンタイル方式（銘柄ごとに最適化）
5. **モジュール設計**: 各フィルターは独立して実装し、将来のAND/OR組み合わせに対応

### 複合フィルター対応設計

```python
# 将来的な複合フィルター実装イメージ
class FilterEngine:
    def __init__(self):
        self.filters = []
    
    def add_filter(self, filter_obj):
        self.filters.append(filter_obj)
    
    def evaluate_and(self, trade_data):
        """全フィルターが通過（AND条件）"""
        return all(f.check(trade_data) for f in self.filters)
    
    def evaluate_or(self, trade_data):
        """いずれかのフィルターが通過（OR条件）"""
        return any(f.check(trade_data) for f in self.filters)
```

---

## フィルター1: MA乖離<5%エントリー禁止フィルター

### 目的

**発見4**（SMA乖離5%以上での全敗）の逆をとり、SMA乖離<5%の範囲内でのみエントリーを許可する。

### ロジック詳細

#### 1. SMA乖離率の算出

```python
# 前日データで判断（ルックアヘッドバイアス回避）
sma_value = data['SMA_25'].iloc[idx - 1]
close_price = data['Adj Close'].iloc[idx - 1]

sma_distance_pct = abs((close_price - sma_value) / sma_value * 100)
```

#### 2. フィルター判定

```python
def check_sma_filter(idx, data, threshold=5.0):
    """
    MA乖離フィルター判定
    
    Args:
        idx: 現在の判定位置
        data: 株価データ
        threshold: 乖離率閾値（デフォルト5.0%）
    
    Returns:
        bool: True=エントリー許可、False=エントリー禁止
    """
    if idx < 1:
        return False
    
    sma_value = data['SMA_25'].iloc[idx - 1]
    close_price = data['Adj Close'].iloc[idx - 1]
    
    sma_distance_pct = abs((close_price - sma_value) / sma_value * 100)
    
    # SMA乖離<5%の場合のみエントリー許可
    return sma_distance_pct < threshold
```

#### 3. 閾値の調整可能性

- **デフォルト**: 5.0%（Phase 1.6の発見4に基づく）
- **銘柄別最適化**: 将来的にパーセンタイル方式で閾値を調整可能
- **複合フィルター時**: AND条件では5.0%のまま、OR条件では緩和（7.0%等）を検討

### 検証項目

| 項目 | 内容 |
|------|------|
| フィルター適用前 | 全取引のPF・勝率 |
| フィルター適用後 | SMA乖離<5%の取引のみのPF・勝率 |
| 除外された取引 | SMA乖離>=5%の取引のPF・勝率 |
| PF改善率 | (After PF - Before PF) / Before PF × 100 |
| 取引数削減率 | (Before Trades - After Trades) / Before Trades × 100 |

### 期待される効果

- **仮説**: SMA乖離が大きい時のエントリーは逆張り的で、トレンドフォローに不利
- **検証対象**: 9銘柄+武田薬品での普遍性スコア算出
- **成功基準**: 普遍性スコア > 0.50（5/10銘柄で改善）

---

## フィルター2: ATR適正範囲（中〜高）フィルター

### 目的

**発見3**（ATRパラドックス）に基づき、低ATR（0-2%）でのエントリーを禁止し、中〜超高ATR範囲でのみエントリーを許可する。

### ロジック詳細

#### 1. ATR範囲の算出

```python
# ATR範囲定義（Phase 1.6基準）
ATR_RANGES = {
    'low': (0.0, 2.0),      # 低: 0-2%
    'mid': (2.0, 3.0),      # 中: 2-3%
    'high': (3.0, 4.0),     # 高: 3-4%
    'super_high': (4.0, 100.0)  # 超高: 4%+
}
```

#### 2. フィルター判定

```python
def check_atr_filter(idx, data, allow_ranges=['mid', 'high', 'super_high']):
    """
    ATR適正範囲フィルター判定
    
    Args:
        idx: 現在の判定位置
        data: 株価データ
        allow_ranges: 許可するATR範囲リスト
    
    Returns:
        bool: True=エントリー許可、False=エントリー禁止
    """
    if idx < 1:
        return False
    
    entry_atr_pct = data['entry_atr_pct'].iloc[idx - 1]
    
    for range_name in allow_ranges:
        low, high = ATR_RANGES[range_name]
        if low <= entry_atr_pct < high:
            return True
    
    return False
```

#### 3. 銘柄別ATR閾値の動的算出

```python
def calculate_atr_thresholds(trades_df):
    """
    銘柄ごとのATR閾値を算出（パーセンタイル方式）
    
    Args:
        trades_df: 取引履歴DataFrame
    
    Returns:
        dict: 閾値辞書
    """
    thresholds = {
        'low_mid_boundary': trades_df['entry_atr_pct'].quantile(0.33),
        'mid_high_boundary': trades_df['entry_atr_pct'].quantile(0.67),
        'high_super_boundary': trades_df['entry_atr_pct'].quantile(0.90)
    }
    return thresholds
```

### 検証項目

| 項目 | 内容 |
|------|------|
| ATR範囲別成績 | 低/中/高/超高それぞれのPF・勝率 |
| フィルター適用前 | 全取引のPF・勝率 |
| フィルター適用後 | 中〜超高ATRの取引のみのPF・勝率 |
| 除外された取引 | 低ATRの取引のPF・勝率 |
| ATR閾値 | 銘柄ごとの33/67/90パーセンタイル値 |

### 期待される効果

- **仮説**: 低ATR時は値動きが小さく、ストップロスに引っかかりやすい
- **検証対象**: 武田薬品では超高ATR勝率100%、低ATR勝率22.1% → 他銘柄でも同様か？
- **成功基準**: 普遍性スコア > 0.30（3/10銘柄で改善）

---

## フィルター3: ATRベースの利確

### 目的

**提案2**（ATRベースの利確）を実装し、固定パーセントではなく、ボラティリティに応じた動的な利確ラインを設定する。

### ロジック詳細

#### 1. 利確ラインの算出

```python
def calculate_atr_profit_target(entry_price, entry_atr_pct, multiplier=10.0):
    """
    ATRベースの利確ライン算出
    
    Args:
        entry_price: エントリー価格
        entry_atr_pct: エントリー時のATR（パーセント）
        multiplier: ATR倍率（デフォルト10.0）
    
    Returns:
        float: 利確価格
    """
    profit_target_price = entry_price * (1 + entry_atr_pct / 100 * multiplier)
    return profit_target_price
```

#### 2. バックテスト統合

```python
def backtest_with_atr_profit_taking(data, entry_signals, multiplier=10.0):
    """
    ATR利確ルール適用のバックテスト
    
    Args:
        data: 株価データ
        entry_signals: エントリーシグナルリスト
        multiplier: ATR倍率
    
    Returns:
        DataFrame: 取引履歴
    """
    trades = []
    
    for entry_idx in entry_signals:
        entry_price = data['Open'].iloc[entry_idx + 1]
        entry_atr_pct = data['entry_atr_pct'].iloc[entry_idx]
        
        profit_target = calculate_atr_profit_target(entry_price, entry_atr_pct, multiplier)
        
        # 利確判定ループ
        for idx in range(entry_idx + 1, len(data)):
            current_high = data['High'].iloc[idx]
            
            if current_high >= profit_target:
                # 利確
                exit_price = profit_target
                exit_reason = 'atr_profit_target'
                # ... 取引記録
                break
            
            # 他のエグジット条件も判定（ストップロス等）
    
    return pd.DataFrame(trades)
```

#### 3. 倍率の最適化

- **デフォルト**: 10.0倍（Phase 1.6の提案2）
- **検証範囲**: 5.0倍、7.5倍、10.0倍、12.5倍、15.0倍
- **銘柄別最適化**: ボラティリティの高い銘柄では倍率を下げる可能性

### 検証項目

| 項目 | 内容 |
|------|------|
| 倍率別成績 | 5.0/7.5/10.0/12.5/15.0倍それぞれのPF・勝率 |
| 利確達成率 | 利確ラインに到達した取引の割合 |
| 平均利益率 | 利確時の平均損益率 |
| 保有日数 | 利確時の平均保有日数 vs 他のエグジット理由 |

### 期待される効果

- **仮説**: 固定パーセント利確よりも、ボラティリティに応じた動的利確が効率的
- **検証対象**: 武田薬品の急騰→急落パターン（最大到達点65.6%）をより効率的に刈り取れるか
- **成功基準**: 平均R倍 > 5.0、利確達成率 > 30%

---

## 複合フィルター設計（Phase 1.10以降）

### AND条件（厳格モード）

```python
# 例: トレンド強度（高） AND SMA乖離<5% AND ATR中〜高
def evaluate_and_filters(idx, data, thresholds):
    trend_ok = check_trend_filter(idx, data, thresholds['trend_high'])
    sma_ok = check_sma_filter(idx, data, threshold=5.0)
    atr_ok = check_atr_filter(idx, data, allow_ranges=['mid', 'high', 'super_high'])
    
    return trend_ok and sma_ok and atr_ok
```

**期待**: 取引数大幅減、PF大幅向上（超厳選エントリー）

### OR条件（緩和モード）

```python
# 例: トレンド強度（高） OR SMA乖離<5% OR ATR中〜高
def evaluate_or_filters(idx, data, thresholds):
    trend_ok = check_trend_filter(idx, data, thresholds['trend_high'])
    sma_ok = check_sma_filter(idx, data, threshold=5.0)
    atr_ok = check_atr_filter(idx, data, allow_ranges=['mid', 'high', 'super_high'])
    
    return trend_ok or sma_ok or atr_ok
```

**期待**: 取引数微減、PF微向上（ノイズ除去）

---

## 検証銘柄リスト

### 9銘柄（Phase 1.7/1.8基準）

1. **4063.T** - 信越化学工業
2. **6501.T** - 日立製作所
3. **6758.T** - ソニーグループ
4. **6861.T** - キーエンス
5. **7203.T** - トヨタ自動車
6. **8001.T** - 伊藤忠商事
7. **8306.T** - 三菱UFJ FG
8. **9983.T** - ファーストリテイリング
9. **9984.T** - ソフトバンクグループ

### 参考銘柄（Phase 1.9以降）

10. **4502.T** - 武田薬品工業

**理由**: 急騰→急落パターン（56.1%）が特徴的で、ATRパラドックス（超高ATR勝率100%）を示した銘柄。Phase 1.8では武田特有の現象と判明したが、今後の複合フィルター検討時に有用な知見を提供。

---

## 実装スケジュール

| Phase | タスク | ファイル | 期待成果 |
|-------|--------|---------|---------|
| Phase 1.9-A | MA乖離フィルター実装 | `scripts/validate_phase1_9_sma_filter.py` | 普遍性スコア算出 |
| Phase 1.9-B | ATR範囲フィルター実装 | `scripts/validate_phase1_9_atr_filter.py` | ATR範囲別成績比較 |
| Phase 1.9-C | ATR利確ルール実装 | `scripts/validate_phase1_9_atr_profit_taking.py` | 倍率別最適化 |
| Phase 1.9統合 | 3フィルター結果統合 | `docs/exit_strategy/PHASE1_9_FILTERS_RESULT.md` | 各フィルターの普遍性評価 |
| Phase 1.10 | 複合フィルター実装 | `scripts/validate_phase1_10_composite_filters.py` | AND/OR条件評価 |

---

## 成功基準

### 個別フィルター

| フィルター | 普遍性スコア目標 | PF改善率目標 | 取引数削減率許容 |
|------------|-----------------|-------------|-----------------|
| MA乖離<5% | > 0.50 | > 20% | < 70% |
| ATR中〜高 | > 0.30 | > 15% | < 50% |
| ATR利確 | > 0.40 | > 25% | N/A |

### 複合フィルター（Phase 1.10）

- **AND条件**: 普遍性スコア > 0.60、PF改善率 > 50%、取引数削減率 < 80%
- **OR条件**: 普遍性スコア > 0.40、PF改善率 > 10%、取引数削減率 < 30%

---

## リスクと制約

### リスク1: 過剰フィッティング

- **対策**: 9銘柄+武田薬品の計10銘柄で検証し、普遍性を確認
- **モニタリング**: 銘柄特性（セクター・ボラティリティ）別に効果を分析

### リスク2: 取引数の過剰削減

- **対策**: AND条件では取引数削減率<80%を許容範囲とする
- **モニタリング**: 取引数が20件未満の場合は統計的有意性を警告

### リスク3: 複数フィルターの干渉

- **対策**: Phase 1.9では単体効果を先に検証
- **Phase 1.10**: 複合フィルター時に相関分析を実施

---

## 次フェーズへの展望

### Phase 1.10: 複合フィルター実装

- トレンド強度 + MA乖離 + ATRの3条件AND
- トレンド強度 OR MA乖離 OR ATRの3条件OR
- 2条件の部分組み合わせ（3C2 = 3パターン）

### Phase 1.11: 機械学習モデル検討

- ランダムフォレストでフィルター条件を最適化
- 特徴量: トレンド強度、MA乖離、ATR、ボリューム比率、時系列情報

### Phase 1.12: リアルトレード準備

- 最適フィルター条件をBaseStrategy統合
- kabu STATION API連携テスト

---

**作成者**: Phase 1.9設計チーム  
**最終更新**: 2026-01-26
