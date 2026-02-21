# Phase 1.6 取引履歴拡張カラム設計書

**作成日**: 2026年1月26日  
**対象**: Phase 1.6トレーリング拡張グリッドサーチ（10銘柄×18パラメータ）  
**目的**: 大敗銘柄（武田薬品工業）の中から大敗パターンを特定し、将来的に回避可能にする

---

## 📋 目的

### 最終ゴール
**「大敗パターン発生時にGC戦略を禁止」または「大敗パターンから早期撤退」することでPF低下を回避し、PF 2.0に近づける**

### 分析方針
1. **パターン特定**: 大敗銘柄（武田薬品）の取引データから共通パターンを発見
2. **回避条件設計**: パターン検出ロジックを構築（例：ボラティリティ閾値、ギャップ率制限）
3. **実装**: 条件に該当する場合はエントリー禁止またはストップ強化

---

## 📊 既存カラム（Phase 1.6現状）

| カラム名 | 型 | 説明 | サンプル値 |
|---------|-----|------|-----------|
| trade_id | int | 取引ID | 1 |
| entry_date | date | エントリー日 | 2020-01-21 |
| entry_price | float | エントリー価格 | 1571.57 |
| exit_date | date | エグジット日 | 2020-01-31 |
| exit_price | float | エグジット価格 | 1509.89 |
| profit_loss | float | 損益（絶対値） | -61.68 |
| exit_reason | str | エグジット理由 | stop_loss |
| ticker | str | ティッカー | 7203.T |
| stop_loss_pct | float | 損切り% | 0.03 |
| trailing_stop_pct | float | トレーリング% | 0.05 |
| take_profit_pct | float/None | 利確% | None |

**総行数**: 11,013取引（2020-01-01 ~ 2025-12-31）

---

## ✅ 追加カラム設計（優先度別）

### 🔴 最優先（必須）- Phase 1実装対象

| カラム名 | 型 | 計算式 | 目的 | データソース |
|---------|-----|--------|------|-------------|
| **holding_days** | int | `(exit_date - entry_date).days` | 損切り早すぎ検証、保有期間比較 | 既存date列から計算 |
| **profit_loss_pct** | float | `(exit_price - entry_price) / entry_price * 100` | ペイオフレシオ計算、Rマルチプル算出 | 既存price列から計算 |
| **max_profit_pct** | float | `max((High - entry_price) / entry_price * 100) during hold` | トレーリング適切性判断（超重要） | stock_data['High']をループ |
| **max_loss_pct** | float | `max((entry_price - Low) / entry_price * 100) during hold` | 損切り設定の適切性確認 | stock_data['Low']をループ |
| **entry_atr** | float | `ATR(14) at entry_date` | ATRベース損切り移行検討 | stock_data計算（talib.ATR） |
| **entry_atr_pct** | float | `entry_atr / entry_price * 100` | ボラティリティ正規化（銘柄間比較） | entry_atr / entry_price |
| **entry_gap_pct** | float | `(entry_price - prev_close) / prev_close * 100` | ギャップダウン多発銘柄検出 | stock_data['Close'].shift(1) |

**実装優先順位**: 上記7項目を**Phase 1**で実装（1-2時間）

---

### 🟡 優先度高（強く推奨）- Phase 2実装対象

| カラム名 | 型 | 計算式 | 目的 | データソース |
|---------|-----|--------|------|-------------|
| **r_multiple** | float | `profit_loss_pct / stop_loss_pct` | リスク対リターン評価（1R = 損切り幅） | profit_loss_pct / stop_loss_pct |
| **entry_volume** | int | `Volume at entry_date` | 流動性確認、異常値検出 | stock_data['Volume'] |
| **avg_volume_20d** | float | `mean(Volume[-20:]) at entry_date` | 出来高平均との比較 | stock_data['Volume'].rolling(20).mean() |
| **volume_ratio** | float | `entry_volume / avg_volume_20d` | 出来高急増/急減検出（1.5倍以上など） | entry_volume / avg_volume_20d |
| **exit_gap_pct** | float | `(exit_price - prev_close) / prev_close * 100` | エグジット時のギャップ検証 | stock_data['Close'].shift(1) |
| **highest_price_during_hold** | float | `max(High) during hold` | トレーリング発動価格計算用 | stock_data['High'].max() |

**実装優先順位**: Phase 2（追加1-2時間）

---

### 🟢 優先度中（あると便利）- Phase 3実装対象

| カラム名 | 型 | 計算式 | 目的 | データソース |
|---------|-----|--------|------|-------------|
| **exit_atr** | float | `ATR(14) at exit_date` | エグジット時のボラティリティ変化 | stock_data計算（talib.ATR） |
| **max_gap_during_hold** | float | `max(abs(gap_pct)) during hold` | 保有中の最大ギャップ検出 | stock_data['Open']とshift(1) |
| **trailing_activated** | bool | `True if max_profit_pct >= trailing_stop_pct` | トレーリング発動有無 | max_profit_pct >= trailing_stop_pct |
| **trailing_trigger_price** | float | `entry_price * (1 + trailing_stop_pct)` | トレーリング発動価格 | entry_price * (1 + trailing_stop_pct) |
| **entry_trend_strength** | float | `ADX(14) at entry_date` | トレンド強度（追いかけエントリー評価） | talib.ADX（要計算） |
| **sma_distance_pct** | float | `(entry_price - SMA(20)) / SMA(20) * 100` | 移動平均線との乖離率 | stock_data['Close'].rolling(20).mean() |

**実装優先順位**: Phase 3（追加1-2時間、talib依存）

---

## 🔧 実装方式

### 実装アプローチ
**Option B: validate_exit_simple_v2.py修正方式（採用済み）** ✅
- メリット: 次回Phase 1.6実行時から自動反映、銘柄・パラメータ変更に対応
- デメリット: なし（過去データには適用されない→都度再実行すれば解決）
- 実装: `calculate_performance_metrics()`内で追加計算（2026-01-26完了）

**Option A: 既存CSV再計算方式（不採用）**
- メリット: 11,013行を一括処理、既存CSVを拡張
- デメリット: 将来の銘柄・パラメータ変更に未対応（手動再実行必要）
- 実装: `scripts/enhance_phase1_6_trades.py` 新規作成

**決定理由**: Option B採用により、戦略パラメータや銘柄リストを変更しても自動で拡張カラムが追加される

---

## 📝 実装仕様（Phase 1: 必須7項目）

### 入力ファイル
- `results/phase1.6_trades_20260126_100846.csv` (11,013行)
- `results/phase1.6_simple_20260123_224706.csv` (180行、パラメータ参照用)

### 出力ファイル
- `results/phase1.6_trades_enhanced_YYYYMMDD_HHMMSS.csv` (11,013行、18カラム = 既存11 + 追加7)

### 処理フロー
```python
for each row in phase1.6_trades.csv:
    # 1. 既存データ読み込み
    ticker, entry_date, exit_date, entry_price, exit_price = row[...]
    
    # 2. stock_dataロード（data_fetcher経由）
    stock_data = load_cached_data(ticker, start=entry_date-150days, end=exit_date)
    
    # 3. 保有期間データ抽出
    hold_data = stock_data.loc[entry_date:exit_date]
    
    # 4. 必須7項目計算
    holding_days = (exit_date - entry_date).days
    profit_loss_pct = (exit_price - entry_price) / entry_price * 100
    max_profit_pct = (hold_data['High'].max() - entry_price) / entry_price * 100
    max_loss_pct = (entry_price - hold_data['Low'].min()) / entry_price * 100
    
    # ATR計算（14日）
    atr_data = talib.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
    entry_atr = atr_data.loc[entry_date]
    entry_atr_pct = entry_atr / entry_price * 100
    
    # ギャップ計算
    prev_close = stock_data['Close'].shift(1).loc[entry_date]
    entry_gap_pct = (entry_price - prev_close) / prev_close * 100
    
    # 5. 既存データ + 新カラムでCSV書き込み
    enhanced_row = row + [holding_days, profit_loss_pct, ...]
```

---

## 🚨 実装上の注意点

### データ整合性
1. **entry_date境界問題**: `stock_data.loc[entry_date]`が存在しない場合（土日祝）
   - 対策: `stock_data.asof(entry_date)` または `ffill()`で補完

2. **ATR未定義問題**: エントリー日までのデータが14日未満
   - 対策: warmup_days=150を確保（Phase 1.6では既に実施済み）

3. **ギャップ計算のNaN**: 初日エントリーで`prev_close`が存在しない
   - 対策: `entry_gap_pct = 0.0`でフォールバック

### パフォーマンス
- 11,013行 × データロード → キャッシュ活用必須
- 銘柄ごとにグループ化してstock_data読み込み回数を削減
  ```python
  for ticker in tickers:
      ticker_trades = trades[trades['ticker'] == ticker]
      stock_data = load_once(ticker, min_date, max_date)
      for trade in ticker_trades:
          # 計算実行
  ```

---

## 📊 分析活用例（Phase 1実装後）

### 1. トレーリング適切性検証
```python
# 「+30%まで行ったのに10%トレーリングで+15%決済」を検出
df[(df['max_profit_pct'] > 30) & (df['trailing_stop_pct'] == 0.10) & (df['profit_loss_pct'] < 20)]
```

### 2. 損切り早すぎ検証
```python
# 損切り後に反転した可能性（max_loss < stop_loss）
df[(df['exit_reason'] == 'stop_loss') & (df['max_loss_pct'] < df['stop_loss_pct'] * 100)]
```

### 3. ギャップダウン多発銘柄
```python
# entry_gap_pct < -2%の頻度が高い銘柄
df[df['entry_gap_pct'] < -2.0].groupby('ticker').size().sort_values(ascending=False)
```

### 4. ボラティリティ別成績
```python
# ATR%が高い銘柄（>3%）でのPF
high_vol = df[df['entry_atr_pct'] > 3.0]
high_vol.groupby('ticker')['profit_loss_pct'].mean()
```

---

## ✅ Phase実装チェックリスト

### Phase 1（必須7項目）- ✅ 完了: 2026年1月26日
- [x] `scripts/validate_exit_simple_v2.py`修正（Option B採用）
- [x] holding_days実装
- [x] profit_loss_pct実装
- [x] max_profit_pct実装（保有期間High最大値）
- [x] max_loss_pct実装（保有期間Low最小値）
- [x] entry_atr実装（talib.ATR利用、未インストール時は簡易版）
- [x] entry_atr_pct実装
- [x] entry_gap_pct実装
- [x] 11,013行全件処理完了
- [x] 出力CSV検証（NaN/Inf確認→正常）
- [x] サンプル分析実行準備完了

**実装詳細**:
- `validate_exit_simple_v2.py` Line 47-56: talib インポート追加
- `validate_exit_simple_v2.py` Line 258-376: `calculate_performance_metrics()`内で7項目計算
- 出力ファイル: `results/phase1.6_trades_20260126_120501.csv` (11,013行、18カラム)
- 平均保有期間: 18.5日（想定範囲内）

### Phase 2（優先度高6項目）- ✅ 完了: 2026年1月26日
- [x] r_multiple実装（Line 355-359）
- [x] entry_volume実装（Line 361-376）
- [x] avg_volume_20d実装（Line 361-376）
- [x] volume_ratio実装（Line 361-376）
- [x] exit_gap_pct実装（Line 378-386）
- [x] highest_price_during_hold実装（Line 388-395）

**実装詳細**:
- `validate_exit_simple_v2.py` Line 355-395: Phase 2の6項目計算
- `calculate_performance_metrics()` 関数シグネチャ修正: exit_paramsパラメータ追加
- r_multiple計算: `profit_loss_pct / (stop_loss_pct * 100)`
- Volume系3項目: entry_volume, avg_volume_20d（20日移動平均）, volume_ratio
- exit_gap_pct: エグジット時のギャップ率（前日終値との乖離）
- highest_price_during_hold: 保有期間中の最高値（トレーリング発動価格計算用）

**検証結果**:
```
テスト対象: 7203.T（トヨタ）、69取引
総カラム数: 23（Phase 1: 18 + Phase 2: 6 - ticker等重複）
Phase 2カラム全て生成確認:
  [OK] r_multiple: 69/69件（-1.31 ~ 1.67等、正常計算）
  [OK] entry_volume: 69/69件（25,988,000 ~ 55,771,500）
  [OK] avg_volume_20d: 69/69件（19,349,600 ~ 59,282,375）
  [OK] volume_ratio: 69/69件（0.85 ~ 1.34）
  [OK] exit_gap_pct: 69/69件（-1.11% ~ 7.41%）
  [OK] highest_price_during_hold: 69/69件（1,326.80 ~ 1,578.80）
```

### Phase 3（優先度中6項目）- ✅ 完了: 2026年1月26日
- [x] exit_atr実装（Line 407-428）
- [x] max_gap_during_hold実装（Line 430-440）
- [x] trailing_activated実装（Line 442-448）
- [x] trailing_trigger_price実装（Line 450-455）
- [x] entry_trend_strength実装（ADX）（Line 457-470）
- [x] sma_distance_pct実装（Line 472-481）

**実装詳細**:
- `validate_exit_simple_v2.py` Line 407-481: Phase 3の6項目計算
- exit_atr: エグジット時のATR（talib.ATR または簡易版）
- max_gap_during_hold: 保有中のギャップ最大値（Open - prev_close）
- trailing_activated: max_profit_pct >= trailing_stop_pct の判定
- trailing_trigger_price: entry_price * (1 + trailing_stop_pct)
- entry_trend_strength: エントリー時のADX（talib.ADX、talib未利用時はNone）
- sma_distance_pct: エントリー価格とSMA(20)の乖離率

**検証結果**:
```
テスト対象: 7203.T（トヨタ）、69取引
総カラム数: 29（Phase 1: 7 + Phase 2: 6 + Phase 3: 6 + メタ: 10）
Phase 3カラム全て生成確認:
  [OK] exit_atr: 69/69件（17.41 ~ 61.63）
  [OK] max_gap_during_hold: 69/69件（0.59% ~ 1.00%）
  [OK] trailing_activated: 69/69件（False等のブール値）
  [OK] trailing_trigger_price: 69/69件（1393.99 ~ 1728.73）
  [OK] entry_trend_strength: 69/69件（14.32 ~ 23.78、ADX値）
  [OK] sma_distance_pct: 69/69件（-2.92% ~ 1.81%）
```

---

## 📁 成果物

### 生成ファイル（Phase 3完了時）
1. ✅ `docs/exit_strategy/PHASE1_6_ENHANCED_COLUMNS_DESIGN.md`（本ファイル、設計書）
2. ✅ `scripts/validate_exit_simple_v2.py`（修正完了: Line 134-481）
   - Phase 1: Line 258-354（7項目）
   - Phase 2: Line 355-405（6項目）
   - Phase 3: Line 407-481（6項目）
3. ✅ `test_phase2_columns.py`（検証用テストスクリプト、Phase 2+3対応）

**Phase 1検証結果**:
```
総カラム数: 18
  - 既存11カラム: trade_id, entry_date, entry_price, exit_date, exit_price, 
                   profit_loss, exit_reason, ticker, stop_loss_pct, 
                   trailing_stop_pct, take_profit_pct
  - 新規7カラム: holding_days, profit_loss_pct, max_profit_pct, max_loss_pct, 
                 entry_atr, entry_atr_pct, entry_gap_pct

サンプル統計:
  - 平均保有期間: 18.5日
  - entry_atr_pct範囲: 0.5% ~ 8.0%（銘柄・時期により変動）
  - max_profit_pct範囲: -5% ~ +30%（含み益の最大値）
```

**Phase 2検証結果**:
```
総カラム数: 23（Phase 1 + Phase 2の6カラム）
  - Phase 2追加6カラム: r_multiple, entry_volume, avg_volume_20d, 
                        volume_ratio, exit_gap_pct, highest_price_during_hold

サンプル統計（7203.T、69取引）:
  - r_multiple範囲: -1.31 ~ 1.67（リスクリターン比）
  - volume_ratio範囲: 0.85 ~ 1.34（平均出来高の85%~134%）
  - exit_gap_pct範囲: -1.11% ~ 7.41%（エグジット時ギャップ）
```

**Phase 3検証結果**:
```
総カラム数: 29（Phase 1 + Phase 2 + Phase 3の6カラム）
  - Phase 3追加6カラム: exit_atr, max_gap_during_hold, trailing_activated,
                        trailing_trigger_price, entry_trend_strength, sma_distance_pct

サンプル統計（7203.T、69取引）:
  - exit_atr範囲: 17.41 ~ 61.63（エグジット時のボラティリティ）
  - max_gap_during_hold範囲: 0.59% ~ 1.00%（保有中最大ギャップ）
  - trailing_activated: 全てFalse（トレーリング未発動）
  - trailing_trigger_price範囲: 1393.99 ~ 1728.73（発動価格）
  - entry_trend_strength範囲: 14.32 ~ 23.78（ADX値）
  - sma_distance_pct範囲: -2.92% ~ 1.81%（SMA乖離率）
```

### 分析レポート（Phase 1完了後）
1. `docs/exit_strategy/PHASE1_6_TRAILING_APPROPRIATENESS_ANALYSIS.md`
   - トレーリング適切性分析（max_profit_pct活用）
2. `docs/exit_strategy/PHASE1_6_STOP_LOSS_QUALITY_ANALYSIS.md`
   - 損切り早すぎ検証（max_loss_pct活用）
3. `docs/exit_strategy/PHASE1_6_GAP_PATTERN_ANALYSIS.md`
   - ギャップダウン多発パターン（entry_gap_pct活用）

---

## 🔗 関連ドキュメント

- [PHASE1_6_DEFEAT_PATTERNS.md](./PHASE1_6_DEFEAT_PATTERNS.md) - 大敗パターン基礎分析（武田薬品100%失敗）
- [PAYOFF_RATIO_EXIT_ANALYSIS_PROJECT.md](./PAYOFF_RATIO_EXIT_ANALYSIS_PROJECT.md) - ペイオフレシオプロジェクト
- [EXIT_STRATEGY_REDESIGN_V2.md](./EXIT_STRATEGY_REDESIGN_V2.md) - エグジット戦略再設計（損切3-7%推奨）
- [validate_exit_simple_v2.py](../../scripts/validate_exit_simple_v2.py) - Phase 1.6実行スクリプト

---

**作成者**: Backtest Project Team  
**最終更新**: 2026年1月26日  
**ステータス**: Phase 3実装完了（全19項目完了）、大敗パターン分析準備完了
