# 戦略選択ロジック分析レポート
**Task 1: なぜGC戦略のみ選択され、他戦略が選択されないのか**

**作成日**: 2026-01-15  
**最終更新**: 2026-01-15 23:15

---

## 1. 分析目的
main_new.py実行時に「GC戦略のみが選択され、他の6戦略が全く選択されない」原因を解明する。

---

## 2. 調査スコープ
- DynamicStrategySelector実装 (main_system/strategy_selection/)
- EnhancedStrategyScoreCalculator実装
- StrategyCharacteristicsDataLoader実装
- 戦略メタデータ (logs/strategy_characteristics/metadata/)
- 市場レジーム判定 (MarketAnalyzer)

---

## 3. 調査結果サマリー

### 3.1 根本原因（確定）
**市場レジーム="strong_uptrend"により、トップ2戦略のみ選択される設計**

[dynamic_strategy_selector.py Line 352-365](c:\Users\imega\Documents\my_backtest_project\main_system\strategy_selection\dynamic_strategy_selector.py#L352-L365):
```python
if 'strong' in market_regime.lower():
    # 強いトレンド: トップ2戦略
    selected = [s[0] for s in sorted_strategies[:2] if s[1] >= self.min_confidence_threshold]
elif 'sideways' in market_regime.lower() or 'volatile' in market_regime.lower():
    # レンジ・高ボラ: トップ3戦略（分散）
    selected = [s[0] for s in sorted_strategies[:3] if s[1] >= self.min_confidence_threshold]
else:
    # 通常トレンド: トップ2-3戦略
    selected = [s[0] for s in sorted_strategies[:2] if s[1] >= self.min_confidence_threshold]
```

### 3.2 実際のスコアランキング（2026-01-15実行結果）
```
1. GCStrategy:                 0.572 → 選択
2. VWAPBreakoutStrategy:       0.547 → 選択
3. BreakoutStrategy:           0.512 → トップ2制限により除外
4. VWAPBounceStrategy:         0.505 → トップ2制限により除外
5. MomentumInvestingStrategy:  0.484 → トップ2制限により除外
6. ContrarianStrategy:         0.419 → トップ2制限により除外
7. OpeningGapStrategy:         0.314 → 閾値0.35未満により除外
```

**検証ポイント**: 
- 全7戦略のスコア計算: 成功（OpeningGap除く全て0.35以上）
- メタデータ存在確認: 成功（logs/strategy_characteristics/metadata/に全7戦略のJSON実在）
- 市場レジーム判定: `strong_uptrend` → トップ2選択ロジック適用

---

## 4. 詳細調査記録

### 4.1 Cycle 1: DynamicStrategySelector構造分析（完了）
**実施日**: 2026-01-15 22:45  
**目的**: 戦略選択フローの全体構造を把握

**主要発見**:
1. `main_system/strategy_selection/dynamic_strategy_selector.py`が実装コア
2. 選択フロー:
   ```
   select_optimal_strategies()
     → _calculate_all_strategy_scores()
         → EnhancedStrategyScoreCalculator.calculate_enhanced_strategy_score()
     → _select_strategies_by_regime()  ← ★ここで市場レジームによる絞り込み
     → _calculate_strategy_weights()
   ```
3. `min_confidence_threshold = 0.35` (デフォルト)
4. `selection_mode = StrategySelectionMode.MARKET_ADAPTIVE` (デフォルト)

**仮説設定**:
- 仮説1: スコア計算器の問題
  - 1A: 他戦略でメタデータ欠如 → **却下** (全戦略メタデータ実在)
  - 1B: Phase 5-A-11でフォールバック禁止→Noneリターン → **却下** (スコア計算成功)
  - 1C: デフォルトスコア0.0が返される → **却下** (実スコア0.314~0.572)

- 仮説2: 閾値フィルターの問題 (min_confidence_threshold = 0.35)
  - **部分的に正しい**: OpeningGapStrategy (0.314) のみ除外
  - 他6戦略は全て0.35以上なので、閾値では説明不可

- 仮説3: 市場レジーム別選択ロジック
  - **正解**: `strong_uptrend`レジームでトップ2戦略のみ選択

### 4.2 Cycle 2: EnhancedStrategyScoreCalculator実装確認（完了）
**実施日**: 2026-01-15 23:00  
**目的**: スコア計算ロジックとメタデータ依存度を確認

**実施内容**:
1. `main_system/strategy_selection/enhanced_strategy_scoring_model.py` 読解
2. `config/strategy_scoring_model.py` 読解
3. `config/strategy_characteristics_data_loader.py` 読解
4. メタデータファイル存在確認
5. 実バックテスト実行（DEBUG_STRATEGY_SCORING=1）
6. スコア計算ログ抽出

**主要発見**:
1. **メタデータ実在確認** (logs/strategy_characteristics/metadata/):
   ```
   BreakoutStrategy_characteristics.json       (5525 bytes, 2025-11-12 11:15:45)
   ContrarianStrategy_characteristics.json     (xxxx bytes, 2025-07-22 16:59:33)
   GCStrategy_characteristics.json             (5525 bytes, 2025-11-12 11:08:10)
   MomentumInvestingStrategy_characteristics.json (xxxx bytes, 2025-07-22 16:59:33)
   OpeningGapStrategy_characteristics.json     (xxxx bytes, 2025-07-22 16:59:33)
   VWAPBounceStrategy_characteristics.json     (xxxx bytes, 2025-07-22 16:59:33)
   VWAPBreakoutStrategy_characteristics.json   (xxxx bytes, 2025-11-12 11:32:01)
   ```

2. **スコア計算成功確認** (temp_cycle2_scoring_debug.log):
   ```
   DEBUG:config.strategy_scoring_model:Calculated score for VWAPBounceStrategy_UNKNOWN: 0.505
   DEBUG:config.strategy_scoring_model:Calculated score for VWAPBreakoutStrategy_UNKNOWN: 0.547
   DEBUG:config.strategy_scoring_model:Calculated score for MomentumInvestingStrategy_UNKNOWN: 0.484
   DEBUG:config.strategy_scoring_model:Calculated score for ContrarianStrategy_UNKNOWN: 0.419
   DEBUG:config.strategy_scoring_model:Calculated score for GCStrategy_UNKNOWN: 0.572
   DEBUG:config.strategy_scoring_model:Calculated score for BreakoutStrategy_UNKNOWN: 0.512
   DEBUG:config.strategy_scoring_model:Calculated score for OpeningGapStrategy_UNKNOWN: 0.314
   ```

3. **市場レジーム確認**:
   ```
   INFO - Market analysis completed - Regime: strong_uptrend, Confidence: 1.00
   ```

4. **Phase 5-A-11修正箇所確認**:
   - [enhanced_strategy_scoring_model.py Line 207-209](c:\Users\imega\Documents\my_backtest_project\main_system\strategy_selection\enhanced_strategy_scoring_model.py#L207-L209):
     ```python
     if not base_score:
         logger.error(f"Base score calculation failed for {strategy_name}, returning None")
         return None
     ```
   - フォールバック禁止により、メタデータ欠如時はNone返却→0.0スコア変換
   - **しかし実際にはメタデータ存在するため、この経路は通らない**

**結論**:
- 仮説1A~1C: **全て却下** - スコア計算は正常動作
- 仮説2: **部分的に正しい** - OpeningGap (0.314) のみ閾値で除外
- 仮説3: **正解** - `strong_uptrend`レジームでトップ2選択ロジック適用

---

## 5. 確定した選択フロー

```
1. MarketAnalyzer実行
   └─> market_regime = "strong_uptrend"

2. スコア計算 (EnhancedStrategyScoreCalculator)
   ├─> GCStrategy: 0.572
   ├─> VWAPBreakoutStrategy: 0.547
   ├─> BreakoutStrategy: 0.512
   ├─> VWAPBounceStrategy: 0.505
   ├─> MomentumInvestingStrategy: 0.484
   ├─> ContrarianStrategy: 0.419
   └─> OpeningGapStrategy: 0.314

3. 閾値フィルター (min_confidence_threshold = 0.35)
   └─> OpeningGapStrategy除外（0.314 < 0.35）

4. 市場レジーム別選択ロジック
   └─> "strong" in "strong_uptrend"
       → トップ2戦略選択 (Line 355)
       → [GCStrategy, VWAPBreakoutStrategy]

5. 重み計算
   ├─> GCStrategy: 0.572 / (0.572 + 0.547) = 0.511
   └─> VWAPBreakoutStrategy: 0.547 / 1.119 = 0.489
```

---

## 6. 未解決の疑問

### 6.1 調査完了（Cycle 2で解決）
- ~~各戦略のスコア計算式は？~~ → メタデータJSONのtrend/volatility適性から計算
- ~~メタデータ依存度は？~~ → 必須依存、`logs/strategy_characteristics/metadata/`から読み込み
- ~~デフォルトスコアは0.0？~~ → Phase 5-A-11でNone→0.0変換、但し実際にはメタデータ存在するため未発動
- ~~他戦略でメタデータ欠如？~~ → **否**、全7戦略のメタデータ実在

### 6.2 新たな疑問
1. なぜGCStrategyのスコアが最高なのか？
   - メタデータ内容の差異（trend_adaptability等）が原因か？
   - 調査要否: 低（根本原因はレジーム別選択ロジックのため）

2. strong_uptrendレジーム判定は妥当か？
   - MarketAnalyzerの判定ロジックは正しいか？
   - 調査要否: 中（ユーザーの「取引回数減少」との関連性あり）

3. トップ2選択ロジックは意図的か？
   - 設計意図: 強いトレンドでは集中投資？
   - 調査要否: 低（設計思想の問題、動作は正常）

---

## 7. 推奨アクション

### 7.1 即座に対応可能
**Option A: 市場レジーム判定を手動オーバーライド**
- 現状: `strong_uptrend` → トップ2選択
- 変更案: `market_regime = "normal_uptrend"` → トップ2~3選択
- 影響: 選択戦略数が増加する可能性
- リスク: MarketAnalyzerの判定精度が低下する可能性

**Option B: selection_modeをTOP_Nに変更**
- 現状: `selection_mode = StrategySelectionMode.MARKET_ADAPTIVE`
- 変更案: `selection_mode = StrategySelectionMode.TOP_N` → 常時トップ3選択
- 影響: 市場レジームに関係なくトップ3戦略選択
- リスク: 強いトレンド時に分散投資となり、リターン低下の可能性

**Option C: min_confidence_thresholdを調整**
- 現状: `min_confidence_threshold = 0.35`
- 変更案: `min_confidence_threshold = 0.30` → OpeningGapも選択対象
- 影響: 低スコア戦略も選択される
- リスク: 低品質戦略の混入リスク

### 7.2 中期対応
**Option D: 市場レジーム別選択ロジックの見直し**
- 現状ロジック:
  ```python
  if 'strong' in market_regime.lower():
      selected = sorted_strategies[:2]  # トップ2
  ```
- 変更案:
  ```python
  if 'strong' in market_regime.lower():
      selected = sorted_strategies[:3]  # トップ3に拡大
  ```
- 影響: 強いトレンド時も3戦略選択
- リスク: 設計意図との乖離

### 7.3 長期対応
**Option E: 動的戦略数調整**
- スコア差分に基づいて選択数を動的に決定
- 例: スコア1位と2位の差が0.05未満なら3戦略選択
- 影響: より柔軟な戦略選択
- リスク: 実装コスト大

---

## 8. 検証計画（未実施）

### Phase 1: 単一戦略個別バックテスト
**目的**: 各戦略の実パフォーマンス確認  
**対象**: 全7戦略（GC、VWAPBreakout、Breakout、VWAPBounce、Momentum、Contrarian、OpeningGap）  
**期間**: 2024-08-19 ~ 2025-01-30 (110日)  
**指標**: 取引回数、総リターン、シャープレシオ、最大DD

### Phase 2: 市場レジーム判定検証
**目的**: MarketAnalyzerの判定精度確認  
**方法**: 手動チャート分析との比較  
**期間**: 直近3ヶ月

### Phase 3: 選択ロジック改善案テスト
**目的**: Option B~Eの効果測定  
**方法**: 過去データで各Optionのバックテスト比較

---

## 9. 参考情報

### 9.1 関連ファイル
- [main_system/strategy_selection/dynamic_strategy_selector.py](c:\Users\imega\Documents\my_backtest_project\main_system\strategy_selection\dynamic_strategy_selector.py)
- [main_system/strategy_selection/enhanced_strategy_scoring_model.py](c:\Users\imega\Documents\my_backtest_project\main_system\strategy_selection\enhanced_strategy_scoring_model.py)
- [config/strategy_scoring_model.py](c:\Users\imega\Documents\my_backtest_project\config\strategy_scoring_model.py)
- [config/strategy_characteristics_data_loader.py](c:\Users\imega\Documents\my_backtest_project\config\strategy_characteristics_data_loader.py)
- [logs/strategy_characteristics/metadata/*.json](c:\Users\imega\Documents\my_backtest_project\logs\strategy_characteristics\metadata)

### 9.2 重要定数
```python
# DynamicStrategySelector
min_confidence_threshold = 0.35  # スコア閾値
selection_mode = StrategySelectionMode.MARKET_ADAPTIVE  # 選択モード
available_strategies = [
    'VWAPBreakoutStrategy',
    'VWAPBounceStrategy',
    'GCStrategy',
    'BreakoutStrategy',
    'MomentumInvestingStrategy',
    'ContrarianStrategy',
    'OpeningGapStrategy'
]
```

---

## 10. 変更履歴
- 2026-01-15 23:15: Cycle 2調査結果追加、仮説3を正解と確定
- 2026-01-15 22:50: Cycle 1調査結果追加、仮説1~3設定
- 2026-01-15 22:30: 初版作成
