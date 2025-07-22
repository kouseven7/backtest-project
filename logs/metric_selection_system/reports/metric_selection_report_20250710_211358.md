# 重要指標選定システム実行レポート

**実行日時**: 2025-07-10 21:13:58  
**分析成功**: ✅ 成功  
**信頼度レベル**: LOW  

## 実行概要

### 分析結果サマリー

- **分析対象指標数**: 17
- **データサンプル数**: 9
- **分析戦略数**: 3
- **分析手法**: correlation, regression, feature_selection
- **推奨指標数**: 6

## 推奨指標ランキング

| 順位 | 指標名 | 重要度スコア | 信頼度 | 手法数 |
|------|--------|-------------|--------|--------|
| 1 | expectancy | 0.996 | medium | 1 |
| 2 | win_rate | 0.989 | medium | 1 |
| 3 | consistency_ratio | 0.966 | medium | 1 |
| 4 | max_drawdown | 0.962 | medium | 1 |
| 5 | profit_factor | 0.484 | medium | 1 |
| 6 | avg_holding_period | 0.430 | medium | 1 |

## 重み最適化結果

- **最適化手法**: balanced_approach
- **改善スコア**: 0.039

### 重みの変化

| カテゴリ | 元の重み | 最適化後 | 変化量 |
|----------|----------|----------|--------|
| performance | 0.350 | 0.305 | -0.045 |
| stability | 0.250 | 0.340 | +0.090 |
| risk_adjusted | 0.200 | 0.193 | -0.007 |
| trend_adaptation | 0.150 | 0.103 | -0.047 |
| reliability | 0.050 | 0.059 | +0.009 |

## パフォーマンス影響評価

- **weight_improvement_score**: 0.039
- **avg_weight_change**: 0.039
- **max_weight_change**: 0.090
- **high_confidence_ratio**: 0.000
- **avg_importance_score**: 0.330
- **data_completeness**: 0.180
- **strategy_diversity**: 0.600

## 推奨事項

- ❌ 分析結果の信頼性が低いです
- ❌ データ品質の改善が必要です

---
*レポート生成日時: 2025-07-10 21:13:58*
