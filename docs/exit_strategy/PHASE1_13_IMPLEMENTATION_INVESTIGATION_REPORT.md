# Phase 1.13実装場所調査報告書

**作成日**: 2026-01-27 18:45:00  
**目的**: トレンドフィルター（AND/OR条件）と損切・トレーリングストップパラメータの最適実装場所を調査  
**ステータス**: 調査完了・実装推奨提示  

---

## エグゼクティブサマリー

### 推奨実装場所

| 機能 | 推奨実装場所 | 理由 |
|------|------------|------|
| **トレンドフィルター（AND/OR）** | **個別戦略（GCStrategy）** | エントリーロジックの一部、戦略固有 |
| **損切（3%）** | **個別戦略（GCStrategy）** | 既存実装あり、Phase 1検証完了 |
| **トレーリングストップ（10%）** | **個別戦略（GCStrategy）** | 既存実装あり、Phase 1検証完了 |
| **利確（なし）** | **個別戦略（GCStrategy）** | 既存実装あり、Phase 1検証完了 |

**結論**: **全てGCStrategyに直接デフォルト値として実装推奨**

---

## 📋 Cycle記録

### Cycle 1: ドキュメント確認・コード構造調査

**問題**: トレンドフィルターと損切パラメータの最適実装場所が不明

**仮説**: BaseStrategyに実装すべき機能と個別戦略に実装すべき機能を明確に分離する必要がある

**調査内容**:
1. Phase 1.8～11Bドキュメント確認（検証結果把握）
2. TASK_5B_COMPLETION_REPORT.md確認（推奨パラメータ確認）
3. BaseStrategy/GCStrategyソースコード確認
4. EXIT_STRATEGY_SEPARATION_DESIGN.md確認（設計思想理解）

**検証**: ✅ 調査完了

**副作用**: なし

**次**: 調査結果を報告書にまとめる

---

## 🔍 調査結果詳細

### 1. Phase 1検証結果サマリー

#### Phase 1.11-B（OR条件フィルター）
- **普遍性スコア**: 0.40 (4/10銘柄で改善)
- **取引数削減率**: 13.8%（目標30%以下達成）
- **フィルター条件**: トレンド強度（高、67%ile以上） OR SMA乖離 < 5.0%
- **改善銘柄**: 8306.T (+18.1%), 9984.T (+17.4%), 7203.T (+16.0%), 6758.T (+13.4%)

#### Phase 1.10（AND条件フィルター）
- **普遍性スコア**: 0.60 (6/10銘柄で改善)
- **取引数削減率**: 81.8%（目標80%超過、統計的有意性への影響あり）
- **フィルター条件**: トレンド強度（高、67%ile以上） AND SMA乖離 < 5.0%
- **改善銘柄**: 4502.T (+2535.4%), 8306.T (+183.9%), 6501.T (+144.2%), 他3銘柄

#### TASK 5-B推奨パラメータ
- **損切**: 3%（3%/5%/7%は同等、3%が最も保守的）
- **トレーリング**: 10%（ペイオフレシオ2.15、PF1.15、発動率4.7%）
- **利確**: なし（トレンドフォロー維持）
- **期待PF**: 1.15（目標PF2未達、エントリー改善必要）

---

### 2. 既存コード構造分析

#### BaseStrategy（基底クラス）

**役割**:
- 全戦略共通の基本機能提供
- `generate_entry_signal(idx)`: エントリーシグナル生成（抽象メソッド）
- `generate_exit_signal(idx, entry_idx)`: エグジットシグナル生成（抽象メソッド）
- `backtest()`: バックテスト実行フレームワーク
- `backtest_daily()`: 日次バックテスト実行（Phase 3対応）

**特徴**:
- 抽象メソッドのみ定義、具体的な売買ロジックなし
- パラメータ管理（`self.params`）
- ログ管理（`self.logger`）

**トレンドフィルター実装状況**:
- **実装なし**（各戦略で独自実装）

**損切・トレーリング実装状況**:
- **実装なし**（各戦略で独自実装）

#### GCStrategy（ゴールデンクロス戦略）

**役割**:
- 短期MA・長期MAのゴールデンクロスでエントリー
- デッドクロス・トレーリングストップ・損切でエグジット

**トレンドフィルター実装**:
```python
# Line 137-151
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
        return 0  # トレンド不適合
```

**損切・トレーリング実装**:
```python
# Line 51-62（デフォルトパラメータ）
default_params = {
    "short_window": 5,
    "long_window": 25,
    "take_profit": 0.15,          # 利益確定（15%）
    "stop_loss": 0.03,            # ストップロス（3%）← TASK 5-B推奨
    "trailing_stop_pct": 0.05,    # トレーリングストップ（5%）← 10%に変更必要
    "max_hold_days": 300,
    "exit_on_death_cross": True,
    "trend_filter_enabled": False,
    "allowed_trends": ["uptrend"]
}
```

**エグジット判定実装**:
```python
# Line 185-280（generate_exit_signal）
# 1. デッドクロスでイグジット
# 2. トレーリングストップ
# 3. 利益確定
# 4. 損切り
# 5. 最大保有期間
```

---

### 3. エグジット戦略分離設計（EXIT_STRATEGY_SEPARATION_DESIGN.md）

#### 設計目的
- エントリー・エグジット分離で独立検証可能に
- エグジット単体で差し替えて比較
- 過学習回避（シンプルなルールベース）

#### 設計構造
```
BaseExitStrategy (ABC)
├── TrailingStopExit
├── TakeProfitExit
├── FixedStopLossExit
└── TrendFollowingExit
```

#### 実装状況
- `BaseExitStrategy`: 実装済み（`strategies/exit_strategies/base_exit_strategy.py`）
- `TrailingStopExit`: 実装済み（`strategies/exit_strategies/trailing_stop_exit.py`）
- `GCStrategyWithExit`: 実装済み（`strategies/gc_strategy_with_exit.py`）

#### 検証状況
- Phase 1.8～1.11Bの検証は**全てGCStrategyの既存実装**を使用
- BaseExitStrategy系は**未検証**（Phase 1では使用せず）

---

## 🎯 実装場所の推奨理由

### 推奨1: トレンドフィルターは**GCStrategy**に実装

#### 理由

1. **エントリーロジックの一部**
   - トレンドフィルターは「いつエントリーするか」の判定条件
   - `generate_entry_signal()`内で実装済み
   - エグジットロジックとは独立

2. **戦略固有の判断基準**
   - GC戦略: uptrend時のみエントリー
   - Contrarian戦略: range-bound時にエントリー
   - Mean Reversion戦略: downtrend時にエントリー（反転狙い）
   - **戦略ごとに異なるトレンド条件**が必要 → BaseStrategyでの共通実装は不適切

3. **既存実装との整合性**
   - GCStrategyは既に`trend_filter_enabled`パラメータを持つ
   - `detect_unified_trend()`を使用した実装が完了
   - 新規実装ではなく、既存パラメータのデフォルト値変更のみで対応可能

4. **Phase 1検証結果の反映**
   - Phase 1.11-B検証: OR条件フィルターが有効（普遍性0.40）
   - Phase 1.10検証: AND条件フィルターが最も有効（普遍性0.60）
   - これらは全てGCStrategyの`generate_entry_signal()`実装を使用

#### 実装方法

**Phase 1.11-B推奨（OR条件）**:
```python
# GCStrategy.__init__()のdefault_params
default_params = {
    "trend_filter_enabled": True,  # デフォルトで有効化
    "allowed_trends": ["uptrend"],
    "sma_divergence_threshold": 5.0,  # 新規追加: SMA乖離閾値
    "use_or_filter": True,  # 新規追加: OR条件フィルター有効化
    # ... 他のパラメータ
}
```

**Phase 1.10推奨（AND条件）**:
```python
default_params = {
    "trend_filter_enabled": True,
    "allowed_trends": ["uptrend"],
    "sma_divergence_threshold": 5.0,
    "use_and_filter": True,  # 新規追加: AND条件フィルター有効化
    # ... 他のパラメータ
}
```

**`generate_entry_signal()`の修正例**:
```python
def generate_entry_signal(self, idx: int) -> int:
    # ... 既存のゴールデンクロス判定 ...
    
    # Phase 1.13実装: OR/ANDフィルター
    if self.params.get("use_or_filter", False):
        # OR条件: トレンド強度（高） OR SMA乖離 < 5.0%
        trend_ok = self._check_trend_strength(idx)  # 67%ile以上
        sma_ok = self._check_sma_divergence(idx)    # < 5.0%
        if not (trend_ok or sma_ok):
            return 0
    
    elif self.params.get("use_and_filter", False):
        # AND条件: トレンド強度（高） AND SMA乖離 < 5.0%
        trend_ok = self._check_trend_strength(idx)
        sma_ok = self._check_sma_divergence(idx)
        if not (trend_ok and sma_ok):
            return 0
    
    # ... 既存のエントリー判定 ...
    return entry_signal
```

#### BaseStrategyに実装**しない**理由

1. **全戦略で異なるトレンド条件**
   - GC: uptrend → ロング
   - Contrarian: range-bound → ロング（反転狙い）
   - Mean Reversion: downtrend → ロング（底打ち狙い）

2. **BaseStrategyは抽象フレームワーク**
   - 具体的な売買ロジックは持たない設計思想
   - `generate_entry_signal()`は抽象メソッド

3. **コードの可読性・保守性**
   - 各戦略が独自にトレンドフィルターを持つ方が理解しやすい
   - BaseStrategyに実装すると、全戦略に影響（意図しない副作用）

---

### 推奨2: 損切・トレーリングは**GCStrategy**に実装

#### 理由

1. **既に実装済み**
   - GCStrategyの`generate_exit_signal()`に完全実装
   - デフォルトパラメータで管理（`default_params`）

2. **Phase 1検証完了**
   - TASK 5-B完了報告: 損切3%、トレーリング10%が最適
   - 180レコード、11,013件エグジットで検証済み
   - ペイオフレシオ2.15、PF1.15達成

3. **戦略固有のパラメータ**
   - GC戦略: トレンドフォロー型 → トレーリング重視
   - Breakout戦略: ボラティリティ重視 → ATR基準損切
   - Mean Reversion戦略: 統計的基準 → Z-Score損切
   - **戦略ごとに異なるエグジット戦略**

4. **BaseExitStrategy系は未使用**
   - Phase 1.8～1.11Bの検証は全てGCStrategy既存実装
   - BaseExitStrategy系は将来の拡張用（Phase 2以降）
   - 現時点で移行する必要性なし

#### 実装方法

**TASK 5-B推奨パラメータの反映**:
```python
# GCStrategy.__init__()のdefault_params
default_params = {
    "short_window": 5,
    "long_window": 25,
    "take_profit": None,          # 利確なし（Phase 1推奨）← 変更必要
    "stop_loss": 0.03,            # 損切3%（Phase 1推奨）← 既に実装済み
    "trailing_stop_pct": 0.10,    # トレーリング10%（Phase 1推奨）← 5%→10%に変更
    "max_hold_days": 300,
    "exit_on_death_cross": True,
}
```

**変更箇所**:
1. `take_profit`: `0.15` → `None`（利確なし）
2. `trailing_stop_pct`: `0.05` → `0.10`（5% → 10%）

#### BaseExitStrategy系に実装**しない**理由

1. **既存実装が十分に機能**
   - GCStrategy.generate_exit_signal()で全てのエグジット判定を実装
   - Phase 1検証で最適パラメータ発見済み

2. **移行コストが高い**
   - BaseExitStrategy系への移行は大規模リファクタリング
   - Phase 1検証結果の再現性検証が必要
   - リアルトレード準備が遅れる（本来の目的に反する）

3. **BaseExitStrategy系の設計目的**
   - エグジット戦略の**独立検証**が目的
   - 複数のエグジット戦略を**差し替えて比較**
   - Phase 1ではエントリー＋エグジットの総合検証を実施済み
   - Phase 2でエグジット単体の最適化を実施する場合に使用

4. **将来拡張への影響なし**
   - GCStrategyにデフォルトパラメータ実装
   - 将来BaseExitStrategy系に移行する場合は、GCStrategyWithExit使用
   - 既存GCStrategyは影響を受けない

---

## 📊 実装推奨まとめ

### 即座実施: GCStrategyのデフォルトパラメータ変更

#### 変更箇所: `strategies/gc_strategy_signal.py` Line 51-62

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

#### 追加実装: OR/ANDフィルター

**Phase 1.11-B推奨（OR条件）**:
```python
# 新規パラメータ追加
default_params = {
    # ... 既存パラメータ ...
    "use_or_filter": False,         # OR条件フィルター（デフォルト無効）
    "trend_strength_threshold": 67, # トレンド強度閾値（67%ile）
    "sma_divergence_threshold": 5.0,# SMA乖離閾値（5.0%）
}
```

**Phase 1.10推奨（AND条件）**:
```python
default_params = {
    # ... 既存パラメータ ...
    "use_and_filter": False,        # AND条件フィルター（デフォルト無効）
    "trend_strength_threshold": 67,
    "sma_divergence_threshold": 5.0,
}
```

#### 実装優先順位

| 優先度 | 実装内容 | 工数 | 期待効果 |
|--------|---------|------|---------|
| **Priority 1** | 損切・トレーリングパラメータ変更 | 5分 | PF1.15達成（検証済み） |
| **Priority 2** | OR条件フィルター実装 | 1時間 | 普遍性0.40、取引数削減13.8% |
| **Priority 3** | AND条件フィルター実装 | 1時間 | 普遍性0.60、取引数削減81.8% |

---

### 将来検討: BaseExitStrategy系への移行（Phase 2以降）

#### 移行条件
- Phase 1でエントリー改善の限界に達した場合
- エグジット単体の最適化を実施する場合
- 複数のエグジット戦略を比較検証する必要がある場合

#### 移行方法
1. `GCStrategyWithExit`を使用
2. 既存GCStrategyはそのまま維持
3. BaseExitStrategy系でエグジット戦略を差し替え
4. 最適なエグジット戦略を発見後、GCStrategyにフィードバック

---

## 🔍 他の戦略への影響分析

### Breakout戦略
- トレンドフィルター: GC戦略と異なる可能性（独自実装推奨）
- 損切・トレーリング: GC戦略とは異なるパラメータ必要（ボラティリティ重視）

### Momentum戦略
- トレンドフィルター: uptrendのみ（GC戦略と同様）
- 損切・トレーリング: GC戦略とは異なるパラメータ必要（RSI基準）

### Mean Reversion戦略
- トレンドフィルター: range-bound/downtrend推奨（GC戦略と逆）
- 損切・トレーリング: Z-Score基準（GC戦略とは全く異なる）

**結論**: **各戦略で独自実装が必要** → BaseStrategyでの共通実装は不適切

---

## ✅ 完了条件達成状況

### ゴールの成功条件

- [x] 関連ドキュメント（PHASE1_8～11B、TASK_5B）の検証結果を確認
- [x] トレンドフィルター実装場所（BaseStrategy vs 個別戦略）の推奨を提示
- [x] 損切・トレーリングストップ実装場所（BaseStrategy vs GC戦略）の推奨を提示
- [x] 実装上のメリット・デメリットを明確化
- [x] 実行は行わず、調査結果報告のみ

---

## 🚀 次のアクション（ユーザー判断待ち）

### Option 1: Priority 1のみ実施（推奨）
- GCStrategyのデフォルトパラメータ変更のみ
- 工数: 5分
- リスク: 最小
- 効果: PF1.15達成（検証済み）

### Option 2: Priority 1 + 2実施
- デフォルトパラメータ変更 + OR条件フィルター実装
- 工数: 1時間
- リスク: 低
- 効果: PF1.15 + 普遍性0.40

### Option 3: Priority 1 + 3実施
- デフォルトパラメータ変更 + AND条件フィルター実装
- 工数: 1時間
- リスク: 中（取引数81.8%削減、統計的有意性への影響）
- 効果: PF1.15 + 普遍性0.60

### Option 4: BaseExitStrategy系への移行（非推奨）
- エグジット戦略分離設計の完全実装
- 工数: 1週間
- リスク: 高（既存実装の再現性検証必要）
- 効果: エグジット単体最適化の準備（Phase 2）

---

**作成者**: Backtest Project Team  
**調査完了日**: 2026-01-27 18:45:00  
**ステータス**: 調査完了・実装推奨提示  
**次のアクション**: ユーザー判断待ち（Option 1～4選択）
