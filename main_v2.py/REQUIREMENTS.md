# main_v2.py 要件定義

## 🎯 コア原則
1. **クリーンスタート**: main.pyの問題を継承しない独立設計
2. **段階的開発**: Phase 1→2→3で確実に構築  
3. **実証必須**: 各段階で必ず`strategy.backtest()`実行確認
4. **モジュール再利用**: 調査済みモジュールを積極活用

## 📋 機能要件

### Phase 1: 基礎実装
- VWAPBreakoutStrategy単体実装
- Entry_Signal/Exit_Signal列生成
- CSV+JSON+TXT出力
- 実トレード件数 > 0検証

### Phase 2: 機能拡張  
- main.py実証済み7戦略追加
- 段階的バックテスト実行
- 戦略間連携機能

### Phase 3: 完成
- 未使用モジュール統合
- 高度な分析機能
- 本格運用対応

## 🚨 品質要件

### 必須チェック
- `strategy.backtest()`実行成功
- Entry_Signal/Exit_Signal列存在
- トレード件数 > 0
- profit=0時の原因調査
- Windows Unicode対応

### 禁止事項
- モック/テストデータ残存
- DSSMS関連モジュール使用
- Excel出力実装
- 未検証コード追加

## 🔄 モジュール再利用

### 再利用フロー
1. モジュール選定 → テスト実行 → 結果判定
2. 実用可 → 実装採用
3. 不可+修正可 → 修正 → 再テスト → 採用
4. 不可+修正困難 → 破棄

### 参照ドキュメント
- 実証済み: main_py_modules_investigation.md (高優先度20個)
- 未使用候補: reusable_modules_investigation_no_dssms.md (高優先度18個)