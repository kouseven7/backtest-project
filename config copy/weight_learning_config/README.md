# 5-2-3 最適な重み付け比率の学習アルゴリズム 設定ファイル

## システム概要
このディレクトリには、ベイジアン最適化による階層的重み学習システムの設定ファイルが含まれています。

## 設定ファイルの説明

### weight_learning_config.json
メイン設定ファイル。以下のセクションを含みます：

- **bayesian_optimization**: ベイジアン最適化のパラメータ
  - kernel_type: カーネルタイプ ("rbf", "matern")
  - acquisition_function: 取得関数 ("EI", "PI", "UCB")
  - max_iterations: 最大反復回数

- **performance_evaluation**: パフォーマンス評価の設定
  - target_return: 目標リターン（年率）
  - max_acceptable_drawdown: 許容最大ドローダウン
  - metric_weights: 指標の重み付け

- **adaptive_learning**: 適応的学習の設定
  - micro_adjustment_threshold: マイクロ調整の閾値（±2%）
  - standard_optimization_threshold: 標準最適化の閾値（±5%）
  - major_rebalancing_threshold: 主要リバランシングの閾値（±20%）

- **weight_constraints**: 重み制約の設定
  - strategy_weights: ストラテジー重みの制約
  - portfolio_weights: ポートフォリオ重みの制約
  - meta_parameters: メタパラメータの制約

- **integration**: システム統合の設定
  - performance_correction_enabled: 5-2-1システムとの連携
  - trend_precision_enabled: 5-2-2システムとの連携

- **meta_parameters**: メタパラメータコントローラーの設定
  - update_frequency_hours: 更新頻度
  - adaptation_threshold: 適応閾値

- **optimization_history**: 履歴管理の設定
  - max_history_days: 履歴保持期間
  - export_formats: エクスポート形式

## 設定のカスタマイズ

### 1. リスク許容度の調整
```json
"performance_evaluation": {
  "target_return": 0.08,  // より保守的な目標
  "max_acceptable_drawdown": 0.15  // より厳格なドローダウン制限
}
```

### 2. 学習頻度の調整
```json
"adaptive_learning": {
  "micro_adjustment_threshold": 0.01,  // より頻繁な微調整
  "min_days_between_major": 14  // より頻繁な主要リバランシング
}
```

### 3. 制約の調整
```json
"weight_constraints": {
  "portfolio_weights": {
    "max_single_asset": 0.25,  // より分散されたポートフォリオ
    "sector_concentration": 0.4
  }
}
```

### 4. 最適化パラメータの調整
```json
"bayesian_optimization": {
  "max_iterations": 100,  // より多くの反復
  "acquisition_function": "UCB",  // より探索的な取得関数
  "kappa": 5.0  // UCBの探索パラメータ
}
```

## 設定の検証

設定ファイルを変更した場合は、以下を確認してください：

1. **JSON形式の妥当性**: 構文エラーがないか確認
2. **数値の範囲**: min/max制約が論理的か確認
3. **重みの合計**: 指標重みが合計1.0になるか確認
4. **依存関係**: 有効化されたシステムが存在するか確認

## デフォルト値の説明

- **期待リターン**: 年率10%（市場平均程度）
- **最大ドローダウン**: 20%（リスク許容度の標準的な値）
- **学習率**: 1.0（ニュートラルな学習速度）
- **ボラティリティスケーリング**: 1.0（デフォルトスケーリング）

## 注意事項

1. 設定変更は次回システム起動時に反映されます
2. 過度に厳格な制約は最適化の収束を妨げる可能性があります
3. 市場条件の変化に応じて定期的な見直しが推奨されます
4. 本番環境での使用前に十分なバックテストを実施してください
