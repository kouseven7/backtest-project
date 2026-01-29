# Phase 1.13フィルター実装設計書

**作成日**: 2026-01-27 19:00:00  
**目的**: トレンド強度 AND/OR SMA乖離フィルターの詳細設計と実装方法を定義  
**参照元**: 
- [PHASE1_13_IMPLEMENTATION_INVESTIGATION_REPORT.md](PHASE1_13_IMPLEMENTATION_INVESTIGATION_REPORT.md)
- [PHASE1_10_COMPOSITE_FILTER_RESULT.md](PHASE1_10_COMPOSITE_FILTER_RESULT.md)
- [PHASE1_11B_OR_FILTER_RESULT.md](PHASE1_11B_OR_FILTER_RESULT.md)

---

## エグゼクティブサマリー

### 設計概要

| フィルター | 普遍性スコア | 取引削減率 | 実装優先度 | 統計的有意性 |
|-----------|-------------|-----------|-----------|-------------|
| **AND条件** | 0.60 (6/10) | 81.8% | Priority 3 | 懸念あり（過剰削減） |
| **OR条件** | 0.40 (4/10) | 13.8% | Priority 2 | 良好（適度な削減） |
| **フィルターなし** | - | 0% | Priority 1 | 基準値 |

**推奨実装順序**:
1. **Priority 1**: 損切・トレーリングパラメータ変更（5分、PF1.15達成）
2. **Priority 2**: OR条件フィルター実装（1時間、普遍性0.40）
3. **Priority 3**: AND条件フィルター実装（1時間、普遍性0.60、要検証）

---

## 📐 設計仕様

### 1. フィルター条件定義

#### 1.1 トレンド強度フィルター（Trend Strength Filter）

**目的**: 強いトレンド中のみエントリーし、レンジ相場や弱いトレンドを回避

**計算方法**:
```python
# 1. ADX（Average Directional Index）を計算
def calculate_trend_strength(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    トレンド強度を計算（ADX使用）
    
    Args:
        data: 株価データ（High, Low, Close必須）
        period: ADX計算期間（デフォルト14日）
    
    Returns:
        pd.Series: トレンド強度（0-100、高いほど強いトレンド）
    """
    # True Range計算
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift(1))
    low_close = np.abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement計算
    plus_dm = data['High'] - data['High'].shift(1)
    minus_dm = data['Low'].shift(1) - data['Low']
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0
    
    # Smoothed True Range and Directional Indicators
    atr = true_range.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # ADX計算
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

# 2. 銘柄別67%ile閾値を計算
def get_trend_strength_threshold(data: pd.DataFrame, percentile: float = 67.0) -> float:
    """
    銘柄別トレンド強度閾値を計算
    
    Args:
        data: 株価データ（ADXカラム含む）
        percentile: パーセンタイル（デフォルト67%）
    
    Returns:
        float: トレンド強度閾値
    """
    adx_values = data['ADX'].dropna()
    threshold = np.percentile(adx_values, percentile)
    return threshold

# 3. フィルター判定
def check_trend_strength_filter(idx: int, data: pd.DataFrame, threshold: float) -> bool:
    """
    トレンド強度フィルター判定
    
    Args:
        idx: 現在のインデックス
        data: 株価データ（ADXカラム含む）
        threshold: 閾値（67%ile値）
    
    Returns:
        bool: True=フィルター通過、False=フィルター不通過
    """
    current_adx = data['ADX'].iloc[idx]
    return current_adx >= threshold
```

**Phase 1.8検証結果より**:
- 銘柄別閾値例: 9984.T（ソフトバンクG）29.19、8306.T（三菱UFJ）22.01、7203.T（トヨタ）21.32
- 閾値は銘柄特性（ボラティリティ、流動性）により大きく異なる

#### 1.2 SMA乖離フィルター（SMA Divergence Filter）

**目的**: 移動平均線から大きく乖離したエントリーを回避（急騰後の暴落リスク削減）

**計算方法**:
```python
# 1. SMA乖離率を計算
def calculate_sma_divergence(data: pd.DataFrame, sma_period: int = 25, price_column: str = 'Adj Close') -> pd.Series:
    """
    SMA乖離率を計算
    
    Args:
        data: 株価データ
        sma_period: SMA期間（デフォルト25日、GC戦略の長期MA）
        price_column: 価格カラム
    
    Returns:
        pd.Series: SMA乖離率（%、正=価格がSMA上、負=価格がSMA下）
    """
    sma = data[price_column].rolling(window=sma_period).mean()
    divergence = ((data[price_column] - sma) / sma) * 100
    return divergence

# 2. フィルター判定
def check_sma_divergence_filter(idx: int, data: pd.DataFrame, threshold: float = 5.0) -> bool:
    """
    SMA乖離フィルター判定
    
    Args:
        idx: 現在のインデックス
        data: 株価データ（SMA_Divergenceカラム含む）
        threshold: 閾値（デフォルト5.0%）
    
    Returns:
        bool: True=フィルター通過、False=フィルター不通過
    """
    current_divergence = abs(data['SMA_Divergence'].iloc[idx])
    return current_divergence < threshold
```

**Phase 1.6発見4の検証結果より**:
- SMA乖離5%以上でのエントリーは大幅パフォーマンス低下
- 7203.T（トヨタ）: 0-5%でPF=1.36、5-10%でPF=0.34、10%+でPF=0.00

---

### 2. AND/ORフィルター設計

#### 2.1 AND条件フィルター（厳格フィルター）

**条件**: トレンド強度（高） **AND** SMA乖離 < 5.0%

**実装**:
```python
def check_and_filter(idx: int, data: pd.DataFrame, trend_threshold: float, sma_threshold: float = 5.0) -> bool:
    """
    AND条件フィルター判定（厳格）
    
    Args:
        idx: 現在のインデックス
        data: 株価データ（ADX, SMA_Divergenceカラム含む）
        trend_threshold: トレンド強度閾値（67%ile値）
        sma_threshold: SMA乖離閾値（デフォルト5.0%）
    
    Returns:
        bool: True=両条件満たす、False=どちらか満たさない
    """
    trend_ok = check_trend_strength_filter(idx, data, trend_threshold)
    sma_ok = check_sma_divergence_filter(idx, data, sma_threshold)
    
    return trend_ok and sma_ok  # 両方True必須
```

**特性**:
- 普遍性スコア: 0.60 (6/10銘柄で改善)
- 取引数削減率: 81.8%（平均）
- メリット: 高精度エントリー、大幅PF改善（最大+2535.4%）
- デメリット: 取引数過剰削減、統計的有意性への懸念

**推奨使用シーン**:
- ボラティリティが高い銘柄（9984.T、8306.T等）
- 急騰→急落パターンが多い銘柄（4502.T武田薬品）
- 長期投資（年10-20取引でも許容）

#### 2.2 OR条件フィルター（緩和フィルター）

**条件**: トレンド強度（高） **OR** SMA乖離 < 5.0%

**実装**:
```python
def check_or_filter(idx: int, data: pd.DataFrame, trend_threshold: float, sma_threshold: float = 5.0) -> bool:
    """
    OR条件フィルター判定（緩和）
    
    Args:
        idx: 現在のインデックス
        data: 株価データ（ADX, SMA_Divergenceカラム含む）
        trend_threshold: トレンド強度閾値（67%ile値）
        sma_threshold: SMA乖離閾値（デフォルト5.0%）
    
    Returns:
        bool: True=どちらか満たす、False=両方満たさない
    """
    trend_ok = check_trend_strength_filter(idx, data, trend_threshold)
    sma_ok = check_sma_divergence_filter(idx, data, sma_threshold)
    
    return trend_ok or sma_ok  # どちらかTrue
```

**特性**:
- 普遍性スコア: 0.40 (4/10銘柄で改善)
- 取引数削減率: 13.8%（平均）
- メリット: 取引数維持、統計的有意性確保
- デメリット: PF改善幅控えめ（最大+18.1%）

**推奨使用シーン**:
- 安定銘柄（7203.T、6758.T等）
- 取引数確保が重要な場合
- 短中期投資（年50取引以上を維持）

---

### 3. GCStrategy実装設計

#### 3.1 パラメータ定義

```python
# GCStrategy.__init__()のdefault_params
default_params = {
    # 既存パラメータ
    "short_window": 5,
    "long_window": 25,
    "take_profit": None,          # Phase 1推奨: 利確なし
    "stop_loss": 0.03,            # Phase 1推奨: 損切3%
    "trailing_stop_pct": 0.10,    # Phase 1推奨: トレーリング10%（5%→10%変更）
    "max_hold_days": 300,
    "exit_on_death_cross": True,
    
    # Phase 1.13新規追加: フィルター制御
    "use_entry_filter": False,        # フィルター有効化（デフォルト無効）
    "filter_mode": "or",              # フィルターモード（"and" or "or"）
    
    # Phase 1.13新規追加: トレンド強度フィルター
    "trend_strength_enabled": True,   # トレンド強度フィルター有効化
    "trend_strength_period": 14,      # ADX計算期間
    "trend_strength_percentile": 67,  # 閾値パーセンタイル
    
    # Phase 1.13新規追加: SMA乖離フィルター
    "sma_divergence_enabled": True,   # SMA乖離フィルター有効化
    "sma_divergence_threshold": 5.0,  # SMA乖離閾値（%）
    "sma_divergence_period": 25,      # SMA期間（long_windowと同じ）
    
    # 将来実装: 自動切り替え機能
    "auto_switch_enabled": False,     # 自動切り替え有効化（将来実装）
    "min_trades_threshold": 120,      # 最小取引数閾値（6年で120件=年20件）
    "fallback_to_or": True,           # AND→OR自動切り替え
    "fallback_to_none": True,         # OR→フィルターなし自動切り替え
}
```

#### 3.2 initialize_strategy()の拡張

```python
def initialize_strategy(self):
    """
    戦略の初期化処理（Phase 1.13拡張）
    """
    super().initialize_strategy()
    
    # 辞書を初期化
    self.entry_prices = {}
    self.high_prices = {}
    
    # 戦略パラメータの読み込み
    self.short_window = int(self.params.get("short_window", 5))
    self.long_window = int(self.params.get("long_window", 25))
    
    # 移動平均線の計算（既存）
    if f"SMA_{self.short_window}" not in self.data.columns:
        self.data[f"SMA_{self.short_window}"] = self.data[self.price_column].rolling(window=self.short_window).mean().shift(1)
    if f"SMA_{self.long_window}" not in self.data.columns:
        self.data[f"SMA_{self.long_window}"] = self.data[self.price_column].rolling(window=self.long_window).mean().shift(1)
    
    # Phase 1.13追加: フィルター用インジケーター計算
    if self.params.get("use_entry_filter", False):
        self._initialize_filters()
    
    # GC_Signalの計算（既存）
    self.data['GC_Signal'] = np.where(
        (self.data[f'SMA_{self.short_window}'] > self.data[f'SMA_{self.long_window}']) & 
        (self.data[f'SMA_{self.short_window}'].shift(1) <= self.data[f'SMA_{self.long_window}'].shift(1)),
        1, 0
    )

def _initialize_filters(self):
    """
    Phase 1.13新規追加: フィルター用インジケーター初期化
    """
    # トレンド強度（ADX）計算
    if self.params.get("trend_strength_enabled", True):
        if 'ADX' not in self.data.columns:
            period = self.params.get("trend_strength_period", 14)
            self.data['ADX'] = self._calculate_adx(period)
        
        # 銘柄別閾値計算
        percentile = self.params.get("trend_strength_percentile", 67)
        adx_values = self.data['ADX'].dropna()
        self.trend_strength_threshold = np.percentile(adx_values, percentile)
        
        self.logger.info(
            f"[FILTER_INIT] Trend Strength: ADX period={period}, "
            f"threshold={self.trend_strength_threshold:.2f} ({percentile}%ile)"
        )
    
    # SMA乖離率計算
    if self.params.get("sma_divergence_enabled", True):
        if 'SMA_Divergence' not in self.data.columns:
            sma_period = self.params.get("sma_divergence_period", self.long_window)
            sma = self.data[self.price_column].rolling(window=sma_period).mean()
            self.data['SMA_Divergence'] = ((self.data[self.price_column] - sma) / sma) * 100
        
        threshold = self.params.get("sma_divergence_threshold", 5.0)
        self.logger.info(
            f"[FILTER_INIT] SMA Divergence: period={sma_period}, threshold={threshold:.1f}%"
        )

def _calculate_adx(self, period: int = 14) -> pd.Series:
    """
    ADX（Average Directional Index）計算
    
    Args:
        period: 計算期間
    
    Returns:
        pd.Series: ADX値（0-100）
    """
    # True Range計算
    high_low = self.data['High'] - self.data['Low']
    high_close = np.abs(self.data['High'] - self.data['Close'].shift(1))
    low_close = np.abs(self.data['Low'] - self.data['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement計算
    plus_dm = self.data['High'] - self.data['High'].shift(1)
    minus_dm = self.data['Low'].shift(1) - self.data['Low']
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0
    
    # ATR and Directional Indicators
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx
```

#### 3.3 generate_entry_signal()の拡張

```python
def generate_entry_signal(self, idx: int) -> int:
    """
    Phase 1.13拡張: AND/ORフィルター統合
    """
    if idx < self.long_window:
        return 0
    
    # トレンドフィルター（既存、統一トレンド判定）
    use_trend_filter = self.params.get("trend_filter_enabled", False)
    if use_trend_filter:
        trend = detect_unified_trend(
            self.data.iloc[:idx + 1], 
            self.price_column, 
            strategy="Golden_Cross",
            method="combined"
        )
        allowed_trends = self.params.get("allowed_trends", ["uptrend"])
        if trend not in allowed_trends:
            return 0
    
    # ゴールデンクロス判定（既存）
    short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx]
    long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx]
    
    if pd.isna(short_sma) or pd.isna(long_sma):
        return 0
    
    prev_short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx-1]
    prev_long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx-1]
    
    golden_cross = short_sma > long_sma and prev_short_sma <= prev_long_sma
    uptrend_continuation = (
        short_sma > long_sma and 
        short_sma > prev_short_sma and
        long_sma > prev_long_sma
    )
    
    entry_signal = golden_cross or uptrend_continuation
    
    if not entry_signal:
        return 0
    
    # Phase 1.13新規追加: AND/ORフィルター適用
    if self.params.get("use_entry_filter", False):
        filter_passed = self._check_entry_filter(idx)
        if not filter_passed:
            self.logger.debug(
                f"[FILTER_REJECT] idx={idx}, date={self.data.index[idx]}, "
                f"GC signal but filter rejected"
            )
            return 0
    
    return 1

def _check_entry_filter(self, idx: int) -> bool:
    """
    Phase 1.13新規追加: エントリーフィルター判定
    
    Args:
        idx: 現在のインデックス
    
    Returns:
        bool: True=フィルター通過、False=フィルター不通過
    """
    # トレンド強度チェック
    trend_ok = True
    if self.params.get("trend_strength_enabled", True):
        current_adx = self.data['ADX'].iloc[idx]
        if pd.isna(current_adx):
            trend_ok = False
        else:
            trend_ok = current_adx >= self.trend_strength_threshold
    
    # SMA乖離チェック
    sma_ok = True
    if self.params.get("sma_divergence_enabled", True):
        current_divergence = abs(self.data['SMA_Divergence'].iloc[idx])
        if pd.isna(current_divergence):
            sma_ok = False
        else:
            threshold = self.params.get("sma_divergence_threshold", 5.0)
            sma_ok = current_divergence < threshold
    
    # AND/OR判定
    filter_mode = self.params.get("filter_mode", "or")
    
    if filter_mode == "and":
        result = trend_ok and sma_ok
        self.logger.debug(
            f"[FILTER_AND] idx={idx}, trend_ok={trend_ok}, sma_ok={sma_ok}, result={result}"
        )
    elif filter_mode == "or":
        result = trend_ok or sma_ok
        self.logger.debug(
            f"[FILTER_OR] idx={idx}, trend_ok={trend_ok}, sma_ok={sma_ok}, result={result}"
        )
    else:
        # 無効なモード
        self.logger.warning(f"[FILTER_ERROR] Invalid filter_mode={filter_mode}, using OR")
        result = trend_ok or sma_ok
    
    return result
```

---

### 4. 将来実装: 自動切り替え機能設計

#### 4.1 設計目的

**問題**: AND条件で取引数が過剰削減（81.8%）され、統計的有意性が低下

**解決策**: 取引数が閾値以下の場合、自動的にOR条件→フィルターなしへ切り替え

#### 4.2 切り替えロジック

```python
def _auto_adjust_filter_mode(self, results_df: pd.DataFrame) -> str:
    """
    将来実装: 取引数に基づくフィルターモード自動調整
    
    Args:
        results_df: バックテスト結果（Entry_Signalカラム含む）
    
    Returns:
        str: 推奨フィルターモード（"and", "or", "none"）
    """
    if not self.params.get("auto_switch_enabled", False):
        return self.params.get("filter_mode", "or")
    
    # 現在の取引数を計算
    current_trades = results_df['Entry_Signal'].sum()
    min_trades = self.params.get("min_trades_threshold", 120)
    
    self.logger.info(
        f"[AUTO_SWITCH] Current trades: {current_trades}, "
        f"Min threshold: {min_trades}"
    )
    
    # 閾値判定
    if current_trades >= min_trades:
        # 十分な取引数 → 現在のモード維持
        return self.params.get("filter_mode", "or")
    
    # 取引数不足 → 自動切り替え
    current_mode = self.params.get("filter_mode", "or")
    
    if current_mode == "and":
        # AND → OR切り替え
        if self.params.get("fallback_to_or", True):
            self.logger.warning(
                f"[AUTO_SWITCH] Insufficient trades ({current_trades} < {min_trades}), "
                f"switching from AND to OR"
            )
            return "or"
    
    elif current_mode == "or":
        # OR → フィルターなし切り替え
        if self.params.get("fallback_to_none", True):
            self.logger.warning(
                f"[AUTO_SWITCH] Insufficient trades ({current_trades} < {min_trades}), "
                f"disabling filter"
            )
            return "none"
    
    # フォールバック: 現在のモード維持
    return current_mode

def backtest(self, trading_start_date=None, trading_end_date=None):
    """
    バックテスト実行（将来実装: 自動切り替え統合）
    """
    # 第1回実行: 現在の設定でバックテスト
    results_df = super().backtest(trading_start_date, trading_end_date)
    
    # 自動切り替え判定
    if self.params.get("auto_switch_enabled", False):
        recommended_mode = self._auto_adjust_filter_mode(results_df)
        current_mode = self.params.get("filter_mode", "or")
        
        if recommended_mode != current_mode:
            # モード変更が推奨される場合、再実行
            self.logger.info(
                f"[AUTO_SWITCH] Re-running backtest with mode={recommended_mode}"
            )
            self.params["filter_mode"] = recommended_mode
            
            # フィルターなしの場合は無効化
            if recommended_mode == "none":
                self.params["use_entry_filter"] = False
            
            # 再実行
            results_df = super().backtest(trading_start_date, trading_end_date)
    
    return results_df
```

#### 4.3 切り替え閾値設定

| 期間 | 最小取引数 | 計算根拠 |
|------|-----------|---------|
| 6年 | 120件 | 年20件（月1-2件） |
| 5年 | 100件 | 年20件 |
| 3年 | 60件 | 年20件 |
| 1年 | 20件 | 月1-2件 |

**推奨**: 6年で120件（年20件、月1-2件）を最小閾値とする

#### 4.4 切り替えフロー

```
エントリーシグナル生成
    ↓
AND条件フィルター適用
    ↓
取引数カウント
    ↓
取引数 >= 120件？
    ├─ YES → AND条件採用
    └─ NO  → OR条件へ切り替え
             ↓
             取引数カウント
             ↓
             取引数 >= 120件？
                 ├─ YES → OR条件採用
                 └─ NO  → フィルターなし
```

---

## 📊 実装優先順位

### Priority 1: 損切・トレーリングパラメータ変更（即座実施推奨）

**工数**: 5分  
**リスク**: 最小  
**期待効果**: PF1.15達成（検証済み）

**変更箇所**: [gc_strategy_signal.py](c:\Users\imega\Documents\my_backtest_project\strategies\gc_strategy_signal.py) Line 51-62

```python
# 変更前
default_params = {
    "take_profit": 0.15,          # 利益確定（15%）
    "trailing_stop_pct": 0.05,    # トレーリングストップ（5%）
}

# 変更後（TASK 5-B推奨）
default_params = {
    "take_profit": None,          # 利益確定なし（トレンドフォロー維持）
    "trailing_stop_pct": 0.10,    # トレーリングストップ（10%）
}
```

### Priority 2: OR条件フィルター実装

**工数**: 1-2時間  
**リスク**: 低  
**期待効果**: 普遍性0.40、取引削減13.8%、統計的有意性良好

**実装手順**:
1. `initialize_strategy()`に`_initialize_filters()`追加
2. `_calculate_adx()`メソッド実装
3. `generate_entry_signal()`に`_check_entry_filter()`統合
4. `_check_entry_filter()`でOR条件判定実装
5. デフォルトパラメータ追加（`use_entry_filter=False`, `filter_mode="or"`）

**検証方法**:
```python
# テストスクリプト例
strategy = GCStrategy(
    data=stock_data,
    params={
        "use_entry_filter": True,
        "filter_mode": "or",
        "trend_strength_enabled": True,
        "sma_divergence_enabled": True,
    }
)
results = strategy.backtest()
print(f"Total trades: {results['Entry_Signal'].sum()}")
```

### Priority 3: AND条件フィルター実装

**工数**: 30分（OR実装後）  
**リスク**: 中（取引数過剰削減の可能性）  
**期待効果**: 普遍性0.60、取引削減81.8%

**実装手順**:
1. `_check_entry_filter()`のAND分岐を有効化
2. `filter_mode="and"`パラメータ設定

**検証必須項目**:
- [ ] 取引数が120件以上維持されるか
- [ ] 統計的有意性（サンプルサイズ）が十分か
- [ ] PF改善がOR条件を上回るか

### Priority 4: 自動切り替え機能実装（将来）

**工数**: 2-3時間  
**リスク**: 中（複雑度増加）  
**期待効果**: 取引数最適化、汎用性向上

**実装タイミング**: Priority 2/3完了後、AND条件で取引数不足が発生した場合

---

## 🧪 検証計画

### 検証フェーズ1: パラメータ変更（Priority 1）

**目的**: TASK 5-B推奨パラメータでPF1.15達成を確認

**手順**:
1. `gc_strategy_signal.py`のdefault_params変更
2. 単一銘柄（4502.T武田薬品）でバックテスト実行
3. PF=1.15、ペイオフレシオ=2.15を確認

**成功基準**:
- [ ] PF >= 1.15
- [ ] ペイオフレシオ >= 2.10
- [ ] 取引数 > 0

### 検証フェーズ2: OR条件フィルター（Priority 2）

**目的**: OR条件で普遍性0.40、取引削減13.8%を再現

**手順**:
1. 10銘柄（Phase 1.11-Bと同じ）でバックテスト実行
2. フィルターなし vs OR条件でPF比較
3. 取引数削減率を計算

**成功基準**:
- [ ] 普遍性スコア >= 0.40（4/10銘柄で改善）
- [ ] 平均取引削減率 <= 20%（統計的有意性維持）
- [ ] 改善銘柄でPF >= 1.10

### 検証フェーズ3: AND条件フィルター（Priority 3）

**目的**: AND条件で普遍性0.60を再現、取引数不足を検証

**手順**:
1. 10銘柄でバックテスト実行
2. フィルターなし vs AND条件でPF比較
3. 取引数が120件未満の銘柄を特定

**成功基準**:
- [ ] 普遍性スコア >= 0.60（6/10銘柄で改善）
- [ ] 取引数 >= 120件の銘柄数 >= 5/10
- [ ] 改善銘柄でPF >= 1.50

### 検証フェーズ4: 自動切り替え（将来）

**目的**: 取引数不足時の自動切り替えを検証

**手順**:
1. 取引数が少ない銘柄（6861.T等）で実行
2. AND → OR → なし の切り替えを確認
3. 最終取引数が120件以上になることを確認

**成功基準**:
- [ ] 取引数 >= 120件達成
- [ ] 切り替えログが出力される
- [ ] PFが極端に悪化しない（< 0.8）

---

## 📂 関連ファイル

### 実装対象ファイル
- [strategies/gc_strategy_signal.py](c:\Users\imega\Documents\my_backtest_project\strategies\gc_strategy_signal.py): GCStrategy本体

### 検証用スクリプト（新規作成予定）
- `validate_phase1_13_priority1.py`: Priority 1検証（パラメータ変更）
- `validate_phase1_13_priority2.py`: Priority 2検証（OR条件）
- `validate_phase1_13_priority3.py`: Priority 3検証（AND条件）

### ドキュメント
- [PHASE1_13_IMPLEMENTATION_INVESTIGATION_REPORT.md](PHASE1_13_IMPLEMENTATION_INVESTIGATION_REPORT.md): 調査報告書
- [PHASE1_10_COMPOSITE_FILTER_RESULT.md](PHASE1_10_COMPOSITE_FILTER_RESULT.md): AND条件検証結果
- [PHASE1_11B_OR_FILTER_RESULT.md](PHASE1_11B_OR_FILTER_RESULT.md): OR条件検証結果
- [TASK_5B_COMPLETION_REPORT.md](TASK_5B_COMPLETION_REPORT.md): 損切・トレーリング推奨値

---

## ✅ 完了条件

### 設計完了チェックリスト
- [x] AND条件フィルターの設計完了
- [x] OR条件フィルターの設計完了
- [x] AND/OR切り替え機能の設計完了
- [x] 将来の自動切り替え機能設計完了
- [x] GCStrategyへの実装方法記述完了

### 実装完了チェックリスト（今後）
- [ ] Priority 1実装完了（パラメータ変更）
- [ ] Priority 2実装完了（OR条件フィルター）
- [ ] Priority 3実装完了（AND条件フィルター）
- [ ] 検証フェーズ1完了（PF1.15達成確認）
- [ ] 検証フェーズ2完了（普遍性0.40確認）
- [ ] 検証フェーズ3完了（普遍性0.60確認）

---

**作成者**: Backtest Project Team  
**作成日**: 2026-01-27 19:00:00  
**ステータス**: 設計完了・実装待ち  
**次のアクション**: Priority 1実装開始（ユーザー承認後）
