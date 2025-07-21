# 5-1-3「リスク調整後リターンの最適化」システム

## 概要
このシステムは、複数の戦略を持つポートフォリオにおいて、リスク調整後リターンを最大化する最適な重み配分を決定する包括的な最適化エンジンです。

## 主要機能

### 1. 複合目的関数最適化
- Sharpe Ratio、Sortino Ratio、Calmar Ratio等の複数指標を組み合わせ
- 動的重み調整による市場環境適応
- 信頼度スコアリングによる結果品質評価

### 2. 包括的制約管理
- 重み制約（最小・最大、単一戦略制限）
- リスク制約（ボラティリティ、ドローダウン、VaR）
- 分散投資制約（相関、集中度、回転率）
- 適応的制約調整（市場環境に応じた制約パラメータ自動調整）

### 3. マルチアルゴリズム最適化
- Differential Evolution（進化的アルゴリズム）
- Scipy Minimize（勾配法系）
- Gradient Descent（カスタム実装）
- アルゴリズム自動選択とフォールバック

### 4. 高度なパフォーマンス評価
- 6カテゴリ、30種類以上の指標
- リスク調整リターン指標群
- 分散投資指標群
- 安定性・効率性指標群

## システム構成

```
analysis/risk_adjusted_optimization/
├── __init__.py                    # パッケージ初期化
├── objective_function_builder.py # 目的関数構築システム
├── constraint_manager.py         # 制約管理システム
├── optimization_algorithms.py    # 最適化アルゴリズム集
├── performance_evaluator.py      # パフォーマンス評価システム
├── risk_return_optimizer.py      # メイン最適化エンジン
├── portfolio_optimizer.py        # 高度ポートフォリオ最適化
└── optimization_validator.py     # 結果検証システム

config/risk_optimization/
└── default_config.json          # システム設定ファイル
```

## 使用方法

### 基本的な最適化
```python
from analysis.risk_adjusted_optimization import RiskAdjustedOptimizationEngine, OptimizationContext

# データ準備
strategy_returns = pd.DataFrame({...})  # 戦略リターンデータ
current_weights = {'strategy1': 0.4, 'strategy2': 0.6}

# コンテキスト作成
context = OptimizationContext(
    strategy_returns=strategy_returns,
    current_weights=current_weights,
    market_volatility=0.20,
    market_regime="normal"
)

# 最適化実行
engine = RiskAdjustedOptimizationEngine()
result = engine.optimize_portfolio_allocation(context)

print(f"最適化成功: {result.optimization_success}")
print(f"最適重み: {result.optimal_weights}")
print(f"信頼度: {result.confidence_level:.3f}")
```

### 高度な包括最適化
```python
from analysis.risk_adjusted_optimization import AdvancedPortfolioOptimizer, PortfolioOptimizationProfile

# オプティマイザー初期化
optimizer = AdvancedPortfolioOptimizer()

# リスクプロファイル設定
profile = optimizer.create_optimization_profile(
    profile_name="moderate_growth",
    risk_tolerance="moderate",
    return_target=0.08,
    max_drawdown_tolerance=0.15
)

# マルチ期間分析設定
multi_period_request = MultiPeriodOptimizationRequest(
    optimization_horizons=[63, 126, 252],
    confidence_threshold=0.6
)

# 包括最適化実行
result = optimizer.optimize_portfolio_comprehensive(
    context, profile, multi_period_request
)

print(f"総合信頼度: {result.confidence_assessment['overall_confidence']:.3f}")
print(f"代替配分数: {len(result.alternative_allocations)}")
print(f"実行戦略: {result.execution_plan['execution_strategy']}")
```

### 結果検証
```python
from analysis.risk_adjusted_optimization import OptimizationValidator

# 検証システム初期化
validator = OptimizationValidator()

# 包括的検証実行
validation_report = validator.validate_optimization_result(result, context)

print(f"検証成功: {validation_report.validation_success}")
print(f"総合スコア: {validation_report.overall_score:.3f}")
print(f"重要な問題数: {len(validation_report.critical_failures)}")
print(f"改善提案数: {len(validation_report.improvement_suggestions)}")
```

## 設定カスタマイズ

### リスクプロファイル
システムでは3つの標準リスクプロファイルを提供：
- **Conservative**: 低リスク、安定性重視
- **Moderate**: バランス型、中程度リスク
- **Aggressive**: 高リターン追求、高リスク許容

### 最適化パラメータ
```json
{
    "objective_weights": {
        "sharpe": 0.4,
        "sortino": 0.3,
        "calmar": 0.2,
        "drawdown": 0.1
    },
    "optimization": {
        "method": "differential_evolution",
        "max_iterations": 1000,
        "population_size": 50,
        "tolerance": 1e-6
    }
}
```

## 出力結果

### 最適化結果（RiskAdjustedOptimizationResult）
- **optimal_weights**: 最適重み配分
- **performance_report**: 詳細パフォーマンス分析
- **constraint_result**: 制約充足状況
- **confidence_level**: 結果の信頼度（0-1）
- **recommendations**: 実行推奨事項

### 包括最適化結果（PortfolioOptimizationResult）
- **alternative_allocations**: 代替配分オプション
- **multi_period_analysis**: マルチ期間分析結果  
- **execution_plan**: 段階的実行プラン
- **confidence_assessment**: 多角的信頼度評価

### 検証レポート（ValidationReport）
- **validation_success**: 検証通過判定
- **category_scores**: カテゴリ別品質スコア
- **critical_failures**: 重要な問題リスト
- **improvement_suggestions**: 改善提案

## 技術的特徴

### 1. 独立設計
既存の`portfolio_risk_manager.py`との競合を回避する完全独立アーキテクチャ

### 2. フォールバック機能
- 依存関係不具合時の代替実装
- アルゴリズム失敗時の自動切替
- データ不足時の品質低下対応

### 3. 適応性
- 市場環境変化への自動調整
- データ品質に応じた信頼度調整
- ユーザー設定の柔軟な変更対応

### 4. 拡張性
- 新しい目的関数の追加容易
- カスタム制約の実装サポート
- 評価指標の拡張可能

## エラーハンドリング

システム全体で包括的なエラーハンドリングを実装：
- データ検証とクレンジング
- 数値計算例外の捕捉
- 最適化失敗時のフォールバック
- ログ記録による問題追跡

## パフォーマンス考慮事項

- 大規模ポートフォリオでの計算効率化
- キャッシュ機能による重複計算回避
- メモリ使用量の最適化
- 並列処理の活用可能性

## 今後の拡張予定

1. **機械学習統合**: 予測モデルとの連携
2. **リアルタイム最適化**: ストリーミングデータ対応
3. **分散処理**: 大規模計算の並列化
4. **API化**: REST API経由での最適化サービス

## 注意事項

- 最適化結果は過去データに基づく分析であり、将来の保証はありません
- 制約設定とリスク許容度の適切な設定が重要です
- 定期的なパラメータ見直しと結果検証を推奨します

## ライセンス
このシステムは内部利用のために開発されました。

## 問い合わせ
技術的な質問や改善提案は、開発チームまでお問い合わせください。
