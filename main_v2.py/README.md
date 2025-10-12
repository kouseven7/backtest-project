# main_v2.py プロジェクト

## 🎯 プロジェクト概要
main.pyの問題を継承しない独立設計による新しいバックテストシステム

## 📁 プロジェクト構造
```
main_v2.py/
├── main_v2.py          # メインエントリーポイント
├── config/             # 設定管理
├── strategies/         # 戦略実装  
├── output/             # 結果出力
├── data/               # データ処理
└── README.md           # このファイル
```

## 🚀 開発計画

### Phase 1: 基礎実装
- **対象**: VWAPBreakoutStrategy 単体のみ
- **必須**: Entry_Signal/Exit_Signal列生成確認
- **出力**: CSV+JSON+TXT形式（Excel禁止）
- **検証**: 実際のトレード回数 > 0確認

### Phase 2: 機能拡張
- **戦略追加**: main.py実証済み7戦略を順次追加
- **確認**: 各戦略追加後に実バックテスト実行

### Phase 3: 完成
- **モジュール活用**: 未使用モジュール統合
- **最適化**: パフォーマンス向上
- **完成**: 本格運用

## 🔄 モジュール再利用戦略

### 実証済みモジュール (高優先度20個)
- data_fetcher.get_parameters_and_data
- data_processor.preprocess_data
- config.logger_config.setup_logger
- config.risk_management.RiskManagement
- strategies.VWAP_Breakout.VWAPBreakoutStrategy
- その他15個

### 未使用候補モジュール (高優先度18個)
- config.logger_config
- config.risk_management
- config.optimized_parameters
- その他15個

## 🚨 開発規則

### 必須チェック項目
- ✅ `strategy.backtest()`の実行確認
- ✅ Entry_Signal/Exit_Signal列の存在確認
- ✅ 実際のトレード件数 > 0の確認
- ✅ profit=0の場合は原因調査必須
- ✅ Unicodeエラー対策（Windows対応）

### 禁止事項
- ❌ モックデータ・テストデータの残存
- ❌ DSSMS関連モジュールの使用
- ❌ Excel出力の実装
- ❌ 未検証でのコード追加

## 📝 開発ログ

### 2025-10-13 基礎構造作成
- main_v2.py メインファイル作成
- config/ 設定管理フォルダ作成
- strategies/ 戦略フォルダ作成
- output/ 出力フォルダ作成
- data/ データ処理フォルダ作成

### 次のステップ
- Phase 1: VWAPBreakoutStrategy統合テスト
- モジュール再利用テスト実行
- バックテスト動作確認