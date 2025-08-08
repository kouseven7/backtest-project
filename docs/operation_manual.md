# マルチ戦略バックテストシステム運用マニュアル
## フェーズ4B: 本番環境移行準備

### 1. システム概要

本システムは、複数の投資戦略を統合してバックテスト・最適化・リスク管理を行うPythonベースの分析基盤です。

#### 1.1 主要コンポーネント
- **戦略実行エンジン**: `main.py` - 複数戦略の統合実行
- **リスク管理モジュール**: `config/risk_management.py` - ポートフォリオリスク制御
- **パラメータ管理**: `config/optimized_parameters.py` - 承認済みパラメータ管理
- **監視システム**: `src/monitoring/` - リアルタイム監視・アラート
- **エラーハンドリング**: `src/error_handling/` - 包括的例外処理

#### 1.2 システム構成
```
my_backtest_project/
├── main.py                    # メインエントリーポイント
├── strategies/                # 戦略実装
├── config/                    # 設定・パラメータ管理
├── src/                       # コアモジュール
│   ├── monitoring/            # 監視・ダッシュボード
│   ├── error_handling/        # エラー処理
│   └── execution/             # 実行エンジン
├── output/                    # 結果出力
├── logs/                      # ログファイル
└── docs/                      # ドキュメント
```

### 2. 日常運用手順

#### 2.1 システム起動前チェック

1. **環境確認**
   ```powershell
   # Python環境確認
   python --version
   
   # 必要パッケージ確認
   pip list | grep -E "(pandas|numpy|yfinance|scipy|openpyxl)"
   ```

2. **設定ファイル確認**
   ```powershell
   # 設定ファイルの存在確認
   ls config/backtest_config.xlsm ; ls config/*.json
   
   # ログディレクトリ確認
   ls logs/
   ```

3. **データソース接続確認**
   ```powershell
   # データ接続テスト
   python -c "import yfinance as yf; print(yf.Ticker('AAPL').info.get('symbol', 'Error'))"
   ```

#### 2.2 標準実行手順

1. **基本バックテスト実行**
   ```powershell
   # 基本実行
   python main.py
   
   # 特定銘柄での実行
   python main.py --ticker AAPL
   
   # デバッグモードでの実行
   python main.py --debug
   ```

2. **監視ダッシュボード起動**
   ```powershell
   # ダッシュボード起動
   python src/monitoring/dashboard.py
   
   # ブラウザで http://localhost:5000 にアクセス
   ```

3. **パフォーマンス監視**
   ```powershell
   # パフォーマンス監視開始
   python demo_performance_monitor.py
   ```

#### 2.3 結果確認手順

1. **ログファイル確認**
   ```powershell
   # メインログ確認
   Get-Content logs/main.log -Tail 50
   
   # エラーログ確認
   Get-Content logs/errors.log -Tail 20
   
   # 戦略別ログ確認
   ls logs/strategy_*.log
   ```

2. **出力ファイル確認**
   ```powershell
   # Excel出力確認
   ls output/*.xlsx
   
   # レポート確認
   ls reports/*.txt
   ```

3. **監視メトリクス確認**
   ```powershell
   # メトリクス収集状況確認
   python -c "from src.monitoring.metrics_collector import MetricsCollector; mc = MetricsCollector(); print(mc.get_current_metrics())"
   ```

### 3. 設定管理

#### 3.1 戦略パラメータ管理

1. **承認済みパラメータ確認**
   ```powershell
   # パラメータ一覧表示
   python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); print(opm.list_approved_params())"
   ```

2. **新規パラメータ承認プロセス**
   - パラメータ最適化実行
   - バックテスト結果検証
   - リスク評価実施
   - 承認申請・承認
   - 本番環境反映

3. **パラメータファイル管理**
   ```powershell
   # パラメータファイルバックアップ
   Copy-Item config/optimized_params/ config/optimized_params_backup_$(Get-Date -Format "yyyyMMdd") -Recurse
   ```

#### 3.2 リスク管理設定

1. **リスク制限確認**
   ```powershell
   # 現在のリスク設定確認
   python -c "from config.risk_management import RiskManagement; rm = RiskManagement(); print(rm.get_current_limits())"
   ```

2. **ポジションサイズ調整**
   ```powershell
   # ポジションサイズ計算確認
   python config/position_sizing/position_size_adjuster.py
   ```

#### 3.3 監視設定

1. **アラート設定確認**
   ```powershell
   # アラート設定表示
   python -c "from src.monitoring.alert_manager import AlertManager; am = AlertManager(); print(am.get_alert_rules())"
   ```

2. **メトリクス収集設定**
   ```powershell
   # メトリクス設定確認
   Get-Content config/monitoring_config.json
   ```

### 4. トラブルシューティング

#### 4.1 一般的な問題と解決策

1. **データ取得エラー**
   ```
   問題: yfinanceでのデータ取得失敗
   原因: ネットワーク接続問題、API制限
   解決: 
   - ネットワーク接続確認
   - プロキシ設定確認
   - レート制限待機
   ```

2. **メモリ不足エラー**
   ```
   問題: 大量データ処理時のメモリ不足
   原因: データサイズ過大、メモリリーク
   解決:
   - データ分割処理
   - ガベージコレクション実行
   - メモリ使用量監視
   ```

3. **パラメータ読み込みエラー**
   ```
   問題: 承認済みパラメータ読み込み失敗
   原因: ファイル破損、権限問題
   解決:
   - バックアップからの復旧
   - デフォルトパラメータ使用
   - ファイル権限確認
   ```

#### 4.2 診断コマンド

1. **システム診断**
   ```powershell
   # 総合診断実行
   python comprehensive_trend_switching_test_suite.py
   
   # コンポーネント診断
   python config/basic_system_test.py
   ```

2. **ログ分析**
   ```powershell
   # エラーログ検索
   Select-String -Path "logs/*.log" -Pattern "ERROR|CRITICAL" | Select-Object -Last 20
   
   # 警告ログ検索
   Select-String -Path "logs/*.log" -Pattern "WARNING" | Select-Object -Last 10
   ```

3. **パフォーマンス診断**
   ```powershell
   # パフォーマンステスト実行
   python benchmark_validator.py
   ```

### 5. バックアップ・復旧

#### 5.1 バックアップ手順

1. **設定ファイルバックアップ**
   ```powershell
   # 日次バックアップスクリプト
   $backupDate = Get-Date -Format "yyyyMMdd"
   New-Item -ItemType Directory -Path "backup/$backupDate" -Force
   Copy-Item config/ "backup/$backupDate/config/" -Recurse
   Copy-Item logs/ "backup/$backupDate/logs/" -Recurse
   ```

2. **パラメータファイルバックアップ**
   ```powershell
   # パラメータ専用バックアップ
   Copy-Item config/optimized_params/ "backup/params_$(Get-Date -Format "yyyyMMddHHmm")/" -Recurse
   ```

#### 5.2 復旧手順

1. **設定復旧**
   ```powershell
   # 設定ファイル復旧
   $restoreDate = "20250101"  # 復旧対象日付
   Copy-Item "backup/$restoreDate/config/" config/ -Recurse -Force
   ```

2. **パラメータ復旧**
   ```powershell
   # パラメータファイル復旧
   Copy-Item "backup/params_202501011200/" config/optimized_params/ -Recurse -Force
   ```

### 6. 監視・アラート

#### 6.1 監視項目

1. **システム監視**
   - CPU使用率
   - メモリ使用率
   - ディスク使用量
   - ネットワーク接続状態

2. **アプリケーション監視**
   - 戦略実行状態
   - データ取得状況
   - エラー発生率
   - 処理時間

3. **ビジネス監視**
   - ポジション状況
   - リスク指標
   - パフォーマンス指標
   - 約定状況

#### 6.2 アラート対応

1. **緊急アラート（Critical）**
   - システム停止
   - データ取得完全停止
   - 重大なエラー

   **対応手順:**
   ```powershell
   # 即座にシステム状態確認
   python src/monitoring/dashboard.py --check-critical
   
   # エラーログ確認
   Get-Content logs/errors.log -Tail 50
   
   # 必要に応じてシステム再起動
   ```

2. **警告アラート（Warning）**
   - データ取得遅延
   - パフォーマンス低下
   - リスク制限接近

   **対応手順:**
   ```powershell
   # 詳細状況確認
   python demo_performance_monitor.py --analyze
   
   # 必要に応じて設定調整
   ```

### 7. パフォーマンス最適化

#### 7.1 定期メンテナンス

1. **週次メンテナンス**
   ```powershell
   # ログローテーション
   $oldLogs = Get-ChildItem logs/*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)}
   $oldLogs | Move-Item -Destination "logs/archive/"
   
   # キャッシュクリア
   Remove-Item config/data_cache/* -Recurse -Force
   ```

2. **月次メンテナンス**
   ```powershell
   # 包括的なシステムチェック
   python comprehensive_matrix_test.py
   
   # パラメータ最適化レビュー
   python check_param_combinations.py
   ```

#### 7.2 パフォーマンス監視

1. **実行時間監視**
   ```powershell
   # 実行時間分析
   python -c "from src.monitoring.metrics_collector import MetricsCollector; mc = MetricsCollector(); print(mc.analyze_execution_times())"
   ```

2. **リソース使用量監視**
   ```powershell
   # リソース使用量確認
   python demo_performance_monitor.py --resource-usage
   ```

### 8. セキュリティ

#### 8.1 アクセス制御

1. **ファイル権限確認**
   ```powershell
   # 重要ファイルの権限確認
   Get-Acl config/optimized_parameters.py
   Get-Acl config/risk_management.py
   ```

2. **ログアクセス制御**
   ```powershell
   # ログファイル権限設定
   icacls logs/ /grant:r "Administrators:F" /t
   ```

#### 8.2 データ保護

1. **機密データ暗号化**
   ```powershell
   # パラメータファイル暗号化
   # 注意: 実装に応じてカスタマイズが必要
   ```

2. **バックアップ暗号化**
   ```powershell
   # バックアップファイル暗号化
   # 注意: 組織のセキュリティポリシーに従って実装
   ```

### 9. 運用チェックリスト

#### 9.1 日次チェックリスト

- [ ] システム起動状態確認
- [ ] データ取得状況確認
- [ ] エラーログ確認
- [ ] アラート状況確認
- [ ] パフォーマンス指標確認
- [ ] ディスク使用量確認

#### 9.2 週次チェックリスト

- [ ] バックアップ実行状況確認
- [ ] ログローテーション実行
- [ ] パラメータ最適化結果レビュー
- [ ] システムパフォーマンス分析
- [ ] セキュリティ状況確認

#### 9.3 月次チェックリスト

- [ ] 包括的システムテスト実行
- [ ] パラメータ最適化全体レビュー
- [ ] リスク管理設定見直し
- [ ] 監視設定最適化
- [ ] ドキュメント更新

### 10. 連絡先・エスカレーション

#### 10.1 緊急時連絡先

- **システム管理者**: [連絡先を記入]
- **開発チーム**: [連絡先を記入]
- **運用チーム**: [連絡先を記入]

#### 10.2 エスカレーション手順

1. **レベル1**: 運用担当者による初期対応
2. **レベル2**: システム管理者によるシステム調査
3. **レベル3**: 開発チームによる詳細調査・修正

---

**注意事項:**
- 本マニュアルは定期的に更新してください
- システム変更時は関連セクションを必ず更新してください
- 緊急時は安全第一で対応してください
- 不明な点は必ずエスカレーションしてください

**最終更新**: 2025年1月
**バージョン**: 1.0
