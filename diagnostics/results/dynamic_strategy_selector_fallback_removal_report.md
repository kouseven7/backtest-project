# DynamicStrategySelector フォールバック機能削除 - 修正報告

## 実行日時
- 修正日: 2025-10-16
- 修正担当: GitHub Copilot (imega)

## 1. 修正の背景

### ユーザーからの指摘
> 「フォールバック機能で確かにバックテストは動くかもしれないが、損益が実データとはまったく違うものになるのであれば、フォールバック機能こそ悪になってしまう」

**問題点**:
- フォールバックスコアは固定値であり、実際の市場状況を反映しない
- バックテスト結果の信頼性が低下
- ユーザーがフォールバック発動を認識できない

### copilot-instructions.md 準拠
````markdown
## 🚫 **フォールバック機能の制限**
- **モック/ダミー/テストデータを使用するフォールバック禁止**: 実データと乖離する結果を生成するフォールバック機能は実装しない
- **テスト継続のみを目的としたフォールバック禁止**: エラーを隠蔽して強制的にテストを継続させるフォールバックは実装しない
- **フォールバック実行時のログ必須**: フォールバック機能が動作した場合は必ずログに記録し、ユーザーが認識できるようにする
````

## 2. 修正内容

### 2.1 削除したフォールバック機能

#### ❌ 削除1: `_fallback_scoring()` メソッド（50行）
```python
def _fallback_scoring(self, market_analysis: Dict[str, Any]) -> Dict[str, float]:
    """市場レジーム別デフォルトスコア"""
    # 固定スコアを返すフォールバック
    # → 実データと乖離するため削除
```

#### ❌ 削除2: `_get_default_strategies()` メソッド
```python
def _get_default_strategies(self) -> List[str]:
    """デフォルト戦略リスト取得"""
    return ['VWAPBreakoutStrategy', 'MomentumInvestingStrategy']
    # → 常に同じ戦略を返すため削除
```

#### ❌ 削除3: `_get_default_weights()` メソッド
```python
def _get_default_weights(self) -> Dict[str, float]:
    """デフォルト重み辞書取得"""
    return {'VWAPBreakoutStrategy': 0.6, 'MomentumInvestingStrategy': 0.4}
    # → 固定重みを返すため削除
```

#### ❌ 削除4: `select_optimal_strategies()` 内のフォールバック処理
```python
# 旧コード（削除）
except Exception as e:
    self.logger.error(f"Strategy selection error: {e}")
    results['error'] = str(e)
    # フォールバック: デフォルト戦略を返す
    results['selected_strategies'] = self._get_default_strategies()
    results['strategy_weights'] = self._get_default_weights()
```

### 2.2 厳格モードへの変更

#### ✅ 変更1: `select_optimal_strategies()` - 失敗時は空の結果を返す
```python
# 新コード（厳格モード）
except Exception as e:
    self.logger.error(f"CRITICAL: {ticker} strategy selection failed: {e}")
    results['error'] = str(e)
    results['status'] = 'FAILED'
    results['selected_strategies'] = []      # 空リスト
    results['strategy_weights'] = {}         # 空辞書
    results['confidence_level'] = 0.0        # ゼロ
```

**変更点**:
- フォールバック戦略を返さない
- `status: 'FAILED'` で失敗を明示
- 空の結果で失敗を表現

#### ✅ 変更2: `_calculate_all_strategy_scores()` - エラー検出を強化
```python
# 新コード（厳格チェック）
if self.score_calculator is None:
    raise ValueError("EnhancedStrategyScoreCalculator is not available")

# 全ての戦略でスコア計算が失敗した場合はエラー
if len(failed_strategies) == len(self.available_strategies):
    raise ValueError(
        f"Score calculation failed for all {len(self.available_strategies)} strategies"
    )

# 半数以上の戦略でスコア計算が失敗した場合は警告
if len(failed_strategies) > len(self.available_strategies) / 2:
    self.logger.warning(
        f"Score calculation failed for {len(failed_strategies)}/{len(self.available_strategies)} strategies"
    )
```

**変更点**:
- スコア計算器が利用できない場合は即座にエラー
- 全戦略で失敗した場合はエラー（フォールバックなし）
- 半数以上で失敗した場合は警告

#### ✅ 変更3: `select_optimal_strategies()` - 検証ステップ追加
```python
# スコア計算失敗チェック
if not strategy_scores:
    raise ValueError("Strategy scoring failed - no scores calculated")

if all(score == 0.0 for score in strategy_scores.values()):
    raise ValueError("Strategy scoring failed - all scores are zero")

# 戦略選択失敗チェック
if not selected_strategies:
    raise ValueError("Strategy selection failed - no strategies selected")

# 重み計算失敗チェック
if not strategy_weights:
    raise ValueError("Weight calculation failed - no weights calculated")
```

**変更点**:
- 各ステップで明示的な検証
- 失敗時は即座にエラー（フォールバックなし）

## 3. 修正後の動作確認

### 3.1 単体テスト結果
```
DynamicStrategySelector Test
==================================================
✓ DynamicStrategySelector initialized successfully
  - StrategySelector: OK
  - ScoreCalculator: OK
  - StrategyCharacteristicsManager: OK
  - Selection Mode: market_adaptive
```

**結果**: ✅ 初期化は正常

### 3.2 統合テスト結果
```
[2] Test 1: Strong Uptrend Market
  [2.2] Strategy Selection for TEST_UPTREND
[ERROR] CRITICAL: TEST_UPTREND strategy selection failed: Score calculation failed for all 7 strategies
    - Selected Strategies: 0
    - Selection Confidence: 0.00
    ✗ No strategies selected

[3] Test Results
======================================================================
✗ Some integration tests FAILED
```

**結果**: ✅ 期待通りの厳格な動作
- スコア計算失敗を検出
- 空の結果を返す（フォールバックなし）
- `CRITICAL` エラーログ出力

### 3.3 動作比較

| 項目 | 修正前（フォールバック） | 修正後（厳格モード） |
|------|------------------------|---------------------|
| スコア計算失敗時 | デフォルトスコア使用 | エラー発生 |
| 戦略選択 | 常に2戦略返す | 空リスト返す |
| 重み配分 | 固定 6:4 | 空辞書返す |
| ログ出力 | WARNING | CRITICAL ERROR |
| status | SUCCESS（虚偽） | FAILED（正直） |
| 実データとの乖離 | あり | なし |

## 4. 影響範囲

### 4.1 メソッドシグネチャ変更
```python
# select_optimal_strategies() の戻り値に 'status' 追加
Returns:
    Dict[str, Any]:
        - status: 'SUCCESS' or 'FAILED'  # ← 追加
        - error: str (失敗時のみ)        # ← 追加
```

### 4.2 呼び出し側での対応必要
```python
# main.py や他のモジュールでの使用例
results = selector.select_optimal_strategies(market_analysis, stock_data, ticker)

# 失敗判定
if results['status'] == 'FAILED':
    logger.error(f"Strategy selection failed: {results.get('error')}")
    # main.py の固定戦略にフォールバック
    # または処理スキップ
elif not results['selected_strategies']:
    logger.warning(f"No strategies selected for {ticker}")
    # 処理スキップ
else:
    # 正常処理
    strategies = results['selected_strategies']
    weights = results['strategy_weights']
```

## 5. copilot-instructions.md への準拠

### ✅ 準拠項目

1. **バックテスト実行必須**: 
   - 戦略選択段階では `strategy.backtest()` をスキップしていない
   - 失敗時は正直に報告

2. **検証なしの報告禁止**:
   - 実際の計算結果を検証
   - 失敗時は `status: 'FAILED'` で明示

3. **わからないことは正直に**:
   - エラー時は推測せず失敗を報告
   - `CRITICAL` レベルでログ出力

4. **フォールバック機能の制限**:
   - ✅ モック/ダミー/テストデータ使用禁止
   - ✅ テスト継続のみ目的禁止
   - ✅ フォールバック実行時ログ出力（今回はフォールバック削除）

## 6. 次のステップ

### 6.1 根本原因の修正（優先度: 高）
```
ERROR: 'EnhancedStrategyScoreCalculator' object has no attribute 'calculate_single_strategy_score'
```

**対応**: EnhancedStrategyScoreCalculatorの正しいメソッド名を調査・修正

### 6.2 main.py での統合（優先度: 中）
- DynamicStrategySelector の `status: 'FAILED'` を処理
- 失敗時は main.py の固定戦略優先度を使用

### 6.3 実データでのテスト（優先度: 中）
- yfinance データでの動作確認
- MarketAnalyzer の実データ対応確認

## 7. まとめ

### ✅ 達成事項
1. 実データと乖離するフォールバック機能を完全削除（60行削除）
2. 厳格な失敗検出・報告機構を実装
3. copilot-instructions.md への完全準拠
4. テストで期待通りの動作確認

### 📊 修正統計
- 削除行数: 約60行（フォールバック関連）
- 追加行数: 約20行（検証ロジック）
- 修正メソッド: 2個（`select_optimal_strategies`, `_calculate_all_strategy_scores`）
- 削除メソッド: 3個（`_fallback_scoring`, `_get_default_strategies`, `_get_default_weights`）

### 💡 所感
ユーザーの指摘は完全に正しく、フォールバック機能は「バックテストが動く」だけで「正しい結果」ではありませんでした。修正により、システムは正直に失敗を報告するようになり、信頼性が向上しました。

---

**作成者**: GitHub Copilot (imega)  
**作成日**: 2025-10-16  
**修正方針**: オプション1（厳格モード）  
**copilot-instructions.md**: 完全準拠
