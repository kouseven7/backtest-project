# 戦略選択ロジック分析レポート（Phase 1: Task A-1）

**作成日**: 2026-01-15  
**目的**: なぜGC戦略のみ選択されるかを解明し、他戦略が選択されない原因を特定する  
**関連**: [Task1_Trade_Count_Reduction_Investigation.md](Task1_Trade_Count_Reduction_Investigation.md)

---

## 調査サイクル記録

### Cycle 1: 戦略選択関連ファイルの特定
- **問題**: GC戦略のみ選択され、他戦略（VWAP、Breakout、Momentum、Contrarian）が選択されない
- **仮説**: DynamicStrategySelector のスコアリングロジックまたは閾値設定に問題
- **実施**: DynamicStrategySelector関連ファイルの特定とコード読解
- **検証**: ✅ 戦略選択システムの構造を確認
- **副作用**: なし
- **次**: スコアリングロジックの詳細分析（Cycle 2）

---

## 1. 戦略選択システムの構造

### 1.1 主要コンポーネント

| コンポーネント | ファイル | 役割 |
|--------------|---------|------|
| **DynamicStrategySelector** | [main_system/strategy_selection/dynamic_strategy_selector.py](../../main_system/strategy_selection/dynamic_strategy_selector.py) | 市場分析結果に基づく動的戦略選択 |
| **EnhancedStrategyScoreCalculator** | main_system/strategy_selection/enhanced_strategy_scoring_model.py | 戦略スコア計算器 |
| **StrategyCharacteristicsManager** | main_system/strategy_selection/strategy_characteristics_manager.py | 戦略特性管理 |
| **StrategySelector** | main_system/strategy_selection/strategy_selector.py | 基底戦略選択器 |

### 1.2 統合先

- **main_new.py**: Line 52でインポート、Line 93で初期化
- **dssms_integrated_main.py**: Line 168-180でDSSMS統合

---

## 2. 戦略選択ロジックの詳細

### 2.1 利用可能戦略リスト

```python
# dynamic_strategy_selector.py Line 145-155
self.available_strategies = [
    'VWAPBreakoutStrategy',
    'MomentumInvestingStrategy',
    'BreakoutStrategy',
    'VWAPBounceStrategy',
    'ContrarianStrategy',
    'GCStrategy',
]
```

**注意**: OpeningGapFixedStrategy、OpeningGapStrategyは除外済み（メタデータ欠如・性能不良）

### 2.2 戦略選択の流れ

```
1. _calculate_all_strategy_scores()
   ↓ EnhancedStrategyScoreCalculatorでスコア計算
   ↓ 全戦略について calculate_enhanced_strategy_score()
   
2. _select_strategies_by_regime()
   ↓ スコアでソート
   ↓ 市場レジーム別選択ロジック適用
   ↓ min_confidence_threshold (0.35) でフィルタ
   
3. _calculate_strategy_weights()
   ↓ スコア比例で重み配分
   
4. _calculate_confidence()
   ↓ 選択信頼度計算
```

### 2.3 スコア閾値（重要発見）

#### min_confidence_threshold の変遷

```python
# Line 88-89
min_confidence_threshold: float = 0.35  
# Phase 5-A-11暫定: 0.45→0.35（スコア変換調査は別タスク）
```

**変更履歴**:
- 初期値: 0.5 (50%)
- Phase 5-A: 0.5 → 0.45 (45%)
- **Phase 5-A-11**: 0.45 → 0.35 (35%) ← **現在値**

**意味**: スコアが0.35未満の戦略は選択されない

### 2.4 市場レジーム別選択ロジック

```python
# Line 343-368
if self.selection_mode == StrategySelectionMode.MARKET_ADAPTIVE:
    if 'strong' in market_regime.lower():
        # 強いトレンド: トップ2戦略
        selected = [s[0] for s in sorted_strategies[:2] 
                   if s[1] >= self.min_confidence_threshold]
    elif 'sideways' in market_regime.lower() or 'volatile' in market_regime.lower():
        # レンジ・高ボラ: トップ3戦略（分散）
        selected = [s[0] for s in sorted_strategies[:3] 
                   if s[1] >= self.min_confidence_threshold]
    else:
        # 通常トレンド: トップ2-3戦略
        selected = [s[0] for s in sorted_strategies[:2] 
                   if s[1] >= self.min_confidence_threshold]
```

**重要**: `if s[1] >= self.min_confidence_threshold` で全てフィルタリング

### 2.5 フォールバック禁止ポリシー

```python
# Line 371-383
if not selected:
    max_score = sorted_strategies[0][1] if sorted_strategies else 0.0
    raise ValueError(
        f"No strategies passed confidence threshold. "
        f"Market regime: {market_regime}, "
        f"Min threshold: {self.min_confidence_threshold}, "
        f"Max score: {max_score:.3f}. "
        f"Fallback selection is prohibited by copilot-instructions.md."
    )
```

**copilot-instructions.md準拠**: エラー隠蔽型フォールバック（デフォルト戦略選択）は実装されていない

---

## 3. スコア計算ロジックの解析

### 3.1 スコア計算の呼び出し

```python
# Line 277-305
def _calculate_all_strategy_scores(
    self,
    market_analysis: Dict[str, Any],
    stock_data: pd.DataFrame
) -> Dict[str, float]:
    
    strategy_scores = {}
    for strategy_name in self.available_strategies:
        try:
            score_result = self.score_calculator.calculate_enhanced_strategy_score(
                strategy_name=strategy_name,
                ticker=ticker,
                market_data=stock_data,
                use_trend_validation=True,
                integration_method="adaptive"
            )
            
            if hasattr(score_result, 'total_score'):
                strategy_scores[strategy_name] = score_result.total_score
            else:
                strategy_scores[strategy_name] = 0.0
```

**重要**: EnhancedStrategyScoreCalculator の `calculate_enhanced_strategy_score()` が実際のスコアを計算

### 3.2 スコアが0.0になるケース

1. **メタデータ欠如**: 戦略特性データが存在しない
2. **計算例外**: スコア計算中にエラー発生
3. **total_score属性なし**: 返却オブジェクトの構造不正

```python
# Line 298-311
except Exception as e:
    self.logger.warning(f"Score calculation failed for {strategy_name}: {e}")
    failed_strategies.append(strategy_name)
    strategy_scores[strategy_name] = 0.0
```

---

## 4. 仮説: GC戦略のみ選択される理由

### 仮説1: スコア計算器の問題

#### A. GC戦略のみ正常にスコア計算
- 他戦略でメタデータ欠如
- 他戦略で計算例外発生
- GCStrategyのスコアのみ0.35以上

**検証方法**: EnhancedStrategyScoreCalculator のログを確認

#### B. スコアリング重み設定のバイアス
- GC戦略に有利な重み付け
- 他戦略に不利な指標選択

**検証方法**: EnhancedScoreWeights の設定を確認

### 仮説2: 市場レジーム判定の問題

#### A. 市場レジームが常に同じ
- 特定レジームでGC戦略が常に優位
- 他戦略が選択される市場条件が発生しない

**検証方法**: MarketAnalyzerの市場レジーム判定ログを確認

#### B. トレンド検出の偏り
- 全期間で「上昇トレンド」判定
- GC戦略（ゴールデンクロス）が上昇トレンドで有利

**検証方法**: 市場分析結果の統計を取得

### 仮説3: 日次対応変更の影響

#### A. 日次バックテストでの戦略特性計算エラー
- 期間バックテストから日次に変更した際、他戦略のメタデータが適用されない
- 日次データ不足でスコア計算失敗

**検証方法**: 日次対応変更前後のgit diffを確認

#### B. 日次API呼び出しでのキャッシュ問題
- GC戦略のみキャッシュヒット
- 他戦略は毎回計算エラー

**検証方法**: キャッシュロジックを確認

---

## 5. 次のステップ（Cycle 2以降）

### Cycle 2: EnhancedStrategyScoreCalculator の調査
```
調査対象:
□ calculate_enhanced_strategy_score() の実装詳細
□ 戦略メタデータの読み込みロジック
□ 各戦略のスコア計算成否ログ
□ Phase 5-A-11でのスコア変換調査の内容
```

### Cycle 3: 実際のスコアログ取得
```
方法:
□ ローカル実行でDEBUGログ有効化
□ 各戦略のスコア計算過程を記録
□ min_confidence_threshold (0.35) を通過するスコアはGCのみか確認
□ 他戦略のスコアが低い原因（計算失敗 or 実際に低評価）を特定
```

### Cycle 4: 戦略特性メタデータの確認
```
調査対象:
□ config/strategy_characteristics/*.json の存在確認
□ 各戦略のメタデータ完全性チェック
□ OpeningGap系戦略が除外された経緯から学ぶ
```

### Cycle 5: 市場レジーム判定の統計
```
取得データ:
□ Task 1前後の市場レジーム推移
□ 'strong uptrend' / 'sideways' / 'volatile' の出現頻度
□ レジーム判定とGC選択の相関
```

---

## 6. わからないこと（正直な記載）

### 6.1 スコア計算の詳細

- **EnhancedStrategyScoreCalculator**の実装未確認
  - 各戦略のスコア計算式は？
  - メタデータ依存度は？
  - デフォルトスコアは0.0？

- **Phase 5-A-11の「スコア変換調査は別タスク」**の意味
  - min_confidence_threshold を0.35に下げた理由
  - スコア変換問題とは何か？

### 6.2 実行ログの不在

- **Task 1後のログファイルが見つからない**
  - dssms_execution_log.txt は文字化け
  - dssms_comprehensive_report.json は JSON parse エラー
  - 実際のスコアログはどこに？

- **戦略選択の詳細ログ**
  - 各戦略のスコアは記録されているか？
  - どの市場レジームが判定されたか？
  - 選択ロジックのどこでGC以外が落ちたか？

### 6.3 日次対応変更の影響範囲

- **どのファイルが日次対応で変更されたか？**
  - dssms_integrated_main.py の変更点
  - MarketAnalyzer の日次対応
  - DynamicStrategySelector への影響

- **日次対応前は他戦略も選択されていたか？**
  - 履歴データの確認が必要

---

## 7. 暫定結論（Cycle 1時点）

### 7.1 システム構造の理解

✅ **確認できたこと**:
- DynamicStrategySelector が戦略選択の中核
- min_confidence_threshold = 0.35 がフィルタ閾値
- 6戦略が利用可能（GC含む）
- スコアベース選択（トップ2-3）
- フォールバック禁止ポリシー

### 7.2 問題の焦点

⚠️ **疑わしいポイント**:
1. **スコア計算器**: EnhancedStrategyScoreCalculator で他戦略のスコアが0.35未満
2. **メタデータ**: 他戦略の戦略特性データが欠如または不正
3. **市場レジーム**: 特定レジームでGCが常に優位

### 7.3 次の優先アクション

**最優先**: EnhancedStrategyScoreCalculator のコード解析（Cycle 2）  
**理由**: スコアが0.35未満なら選択されない構造が確定しているため、スコア計算ロジックの解明が不可欠

---

## 8. Claudeへの報告（Cycle 1完了）

### 実施内容
- ✅ DynamicStrategySelector関連ファイルを特定
- ✅ 戦略選択ロジックのコード解析完了
- ✅ min_confidence_threshold = 0.35 が重要閾値と判明
- ✅ 6戦略が利用可能リストに含まれることを確認
- ✅ フォールバック禁止ポリシーを確認

### 発見された重要事実
1. **閾値制約**: スコア < 0.35 の戦略は選択されない（厳格）
2. **トップN選択**: スコア上位2-3戦略のみ選択（市場レジーム依存）
3. **エラー時0.0スコア**: 計算失敗した戦略はスコア0.0（選択不可）

### 次のCycleで必要なこと
- EnhancedStrategyScoreCalculator のコード解析
- 各戦略のメタデータ確認
- 実際のスコアログ取得（ローカル実行）

---

**作成者**: GitHub Copilot  
**調査日時**: 2026-01-15  
**ステータス**: Cycle 1完了、Cycle 2準備中  
**関連ファイル**:
- [dynamic_strategy_selector.py](../../main_system/strategy_selection/dynamic_strategy_selector.py)
- [Task1_Trade_Count_Reduction_Investigation.md](Task1_Trade_Count_Reduction_Investigation.md)
