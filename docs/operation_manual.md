# マルチ戦略バックテストシステム運用マニュアル
## フェーズ4A: リアルタイム実行環境統合版

### 1. システム概要

本システムは、複数の投資戦略を統合してバックテスト・リアルタイム実行・最適化・リスク管理を行うPythonベースの統合トレーディングプラットフォームです。

#### 1.1 主要コンポーネント
- **戦略実行エンジン**: `main.py` + `src/execution/` - バックテスト・リアルタイム統合実行
- **リアルタイムデータシステム**: `src/data/` - 複数データソース統合フィード
- **ペーパートレード環境**: `src/execution/paper_broker.py` - 仮想取引実行
- **リスク管理モジュール**: `config/risk_management.py` - ポートフォリオリスク制御
- **パラメータ管理**: `config/optimized_parameters.py` - 承認済みパラメータ管理
- **統合監視システム**: `src/monitoring/` + `src/execution/paper_trade_moni#### 7.2 定期メンテナンス

1. **週次メンテナンス**
   ```powershell
   # ログローテーション
   $oldLogs = Get-ChildItem logs/*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)}
   $oldLogs | Move-Item -Destination "logs/archive/"
   
   # ペーパートレードログローテーション
   $oldTradeLogs = Get-ChildItem logs/paper_trading/*.json | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)}
   $oldTradeLogs | Move-Item -Destination "logs/paper_trading/archive/"
   
   # キャッシュクリア・最適化
   python -c "from src.data.realtime_cache import RealtimeCache; cache = RealtimeCache(); cache.optimize_cache()"
   
   # データ品質履歴クリーンアップ
   Remove-Item logs/data_quality_*.json | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-14)}
   ```

2. **月次メンテナンス**
   ```powershell
   # 包括的なシステムチェック
   python comprehensive_matrix_test.py
   
   # リアルタイムシステム総合チェック
   python demo_realtime_data_system.py --full-test
   
   # ペーパートレード統合チェック
   python demo_paper_trade_runner.py --comprehensive
   
   # パラメータ最適化レビュー
   python check_param_combinations.py
   ```ラート
- **エラーハンドリング**: `src/error_handling/` - 包括的例外処理・復旧

#### 1.2 システム構成（更新版）
```
my_backtest_project/
├── main.py                         # メインエントリーポイント
├── src/                            # コアモジュール（統合版）
│   ├── data/                       # リアルタイムデータ取得・統合
│   │   ├── data_source_adapter.py  # データソースアダプター
│   │   ├── realtime_cache.py       # ハイブリッドキャッシュ
│   │   ├── realtime_feed.py        # リアルタイムフィード
│   │   └── data_feed_integration.py # 統合データフィードシステム
│   ├── execution/                  # 実行エンジン（拡張版）
│   │   ├── paper_broker.py         # ペーパートレードブローカー
│   │   ├── paper_trade_monitor.py  # ペーパートレード監視
│   │   ├── paper_trade_scheduler.py # スケジューラー
│   │   ├── strategy_execution_manager.py # 戦略実行管理
│   │   └── portfolio_tracker.py    # ポートフォリオ追跡
│   ├── monitoring/                 # 監視・ダッシュボード
│   ├── error_handling/             # エラー処理・復旧
│   └── analysis/                   # 分析・レポート
├── strategies/                     # 戦略実装
├── config/                         # 設定・パラメータ管理
├── output/                         # 結果出力
├── logs/                          # ログファイル
├── examples/                      # デモスクリプト
└── docs/                          # ドキュメント
```

### 2. 日常運用手順

#### 2.1 システム起動前チェック

1. **環境確認**
   ```powershell
   # Python環境確認
   python --version
   
   # 必要パッケージ確認（リアルタイムデータ拡張）
   pip list | grep -E "(pandas|numpy|yfinance|scipy|openpyxl|aiohttp|requests)"
   ```

2. **設定ファイル確認**
   ```powershell
   # 基本設定ファイルの存在確認
   ls config/backtest_config.xlsm ; ls config/*.json
   
   # リアルタイムデータ設定確認
   ls config/data_sources_config.json ; ls config/realtime_config.json
   
   # ログディレクトリ確認
   ls logs/ ; ls logs/paper_trading/
   ```

3. **データソース接続確認**
   ```powershell
   # 統合データフィード接続テスト
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print('データシステム初期化完了')"
   
   # 従来のデータ接続テスト
   python -c "import yfinance as yf; print(yf.Ticker('AAPL').info.get('symbol', 'Error'))"
   ```

#### 2.2 標準実行手順

1. **基本バックテスト実行**
   ```powershell
   # 基本実行（バックテストモード）
   python main.py
   
   # 特定銘柄での実行
   python main.py --ticker AAPL
   
   # デバッグモードでの実行
   python main.py --debug
   ```

2. **ペーパートレード実行**
   ```powershell
   # ペーパートレード環境デモ
   python demo_paper_trade_runner.py
   
   # ペーパートレード監視システム
   python examples/demo_paper_trading_system.py
   ```

3. **リアルタイムデータシステム起動**
   ```powershell
   # リアルタイムデータシステムデモ
   python demo_realtime_data_system.py
   
   # 統合データフィード起動
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); system.start_all_feeds()"
   ```

4. **監視ダッシュボード起動**
   ```powershell
   # 統合監視ダッシュボード起動
   python src/monitoring/dashboard.py
   
   # ペーパートレード専用監視
   python performance_monitor.py
   
   # ブラウザで http://localhost:5000 にアクセス
   ```

#### 2.3 結果確認手順

1. **ログファイル確認**
   ```powershell
   # メインログ確認
   Get-Content logs/main.log -Tail 50
   
   # ペーパートレード専用ログ
   Get-Content logs/paper_trading/*.log -Tail 20
   
   # リアルタイムデータログ
   Get-Content logs/realtime_data.log -Tail 30
   
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
   
   # ペーパートレード実行結果
   ls logs/paper_trading/executions_*.json
   ```

3. **監視メトリクス確認**
   ```powershell
   # 統合メトリクス収集状況確認
   python -c "from src.monitoring.metrics_collector import MetricsCollector; mc = MetricsCollector(); print(mc.get_current_metrics())"
   
   # リアルタイムデータシステム状態確認
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print(system.get_system_status())"
   
   # ペーパートレード監視状態確認
   python -c "from src.execution.paper_trade_monitor import PaperTradeMonitor; monitor = PaperTradeMonitor({}); print(monitor.get_status())"
   ```

4. **データ品質確認**
   ```powershell
   # データ品質レポート確認
   ls logs/data_quality_*.json
   
   # キャッシュ状態確認
   python -c "from src.data.realtime_cache import RealtimeCache; cache = RealtimeCache(); print(cache.get_cache_stats())"
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

#### 3.2 リアルタイムデータ設定管理

1. **データソース設定確認**
   ```powershell
   # データソース設定表示
   Get-Content config/data_sources_config.json
   
   # リアルタイム設定表示
   Get-Content config/realtime_config.json
   ```

2. **データ品質設定**
   ```powershell
   # データ品質閾値確認
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print(system.get_quality_config())"
   ```

3. **キャッシュ設定最適化**
   ```powershell
   # キャッシュ設定確認・調整
   python -c "from src.data.realtime_cache import RealtimeCache; cache = RealtimeCache(); print(cache.get_configuration())"
   ```

#### 3.3 ペーパートレード設定

1. **ペーパートレード環境設定**
   ```powershell
   # ペーパートレード設定確認
   python -c "from src.execution.paper_broker import PaperBroker; broker = PaperBroker(); print(broker.get_account_info())"
   ```

2. **実行スケジュール設定**
   ```powershell
   # スケジューラー設定確認
   python -c "from src.execution.paper_trade_scheduler import PaperTradeScheduler; scheduler = PaperTradeScheduler({}); print(scheduler.get_status())"
   ```

#### 3.4 リスク管理設定

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

3. **リアルタイムリスク監視**
   ```powershell
   # リアルタイムリスク状況確認
   python -c "from src.execution.portfolio_tracker import PortfolioTracker; tracker = PortfolioTracker(); print(tracker.get_risk_metrics())"
   ```

#### 3.5 統合監視設定

1. **アラート設定確認**
   ```powershell
   # アラート設定表示
   python -c "from src.monitoring.alert_manager import AlertManager; am = AlertManager(); print(am.get_alert_rules())"
   
   # ペーパートレードアラート確認
   python -c "from src.execution.paper_trade_monitor import PaperTradeMonitor; monitor = PaperTradeMonitor({}); print('アラート設定確認完了')"
   ```

2. **メトリクス収集設定**
   ```powershell
   # メトリクス設定確認
   Get-Content config/monitoring_config.json
   
   # リアルタイムデータメトリクス
   ls logs/metrics_*.json
   ```

3. **データフィード監視設定**
   ```powershell
   # データフィード状態監視
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print(system.get_subscription_status())"
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
   - Alpha Vantageへのフェイルオーバー確認
   ```

2. **リアルタイムデータシステムエラー**
   ```
   問題: リアルタイムフィード停止
   原因: データソース接続問題、キャッシュ満杯
   解決:
   - データソース状態確認: python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print(system.get_system_status())"
   - キャッシュクリア: python -c "from src.data.realtime_cache import RealtimeCache; cache = RealtimeCache(); cache.clear_cache()"
   - フェイルオーバー機能確認
   ```

3. **ペーパートレード実行エラー**
   ```
   問題: ペーパートレード実行失敗
   原因: ポートフォリオ状態異常、注文処理エラー
   解決:
   - ポートフォリオ状態確認: python -c "from src.execution.portfolio_tracker import PortfolioTracker; tracker = PortfolioTracker(); print(tracker.get_positions())"
   - 注文履歴確認: Get-Content logs/paper_trading/orders_*.json
   - ブローカー状態リセット
   ```

4. **メモリ不足エラー**
   ```
   問題: 大量データ処理時のメモリ不足
   原因: データサイズ過大、メモリリーク、キャッシュ肥大化
   解決:
   - データ分割処理
   - ガベージコレクション実行
   - キャッシュサイズ制限調整
   - L2キャッシュクリーンアップ
   ```

5. **パラメータ読み込みエラー**
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
   
   # リアルタイムシステム診断
   python demo_realtime_data_system.py
   
   # ペーパートレード統合診断
   python demo_paper_trade_runner.py
   ```

2. **ログ分析**
   ```powershell
   # エラーログ検索
   Select-String -Path "logs/*.log" -Pattern "ERROR|CRITICAL" | Select-Object -Last 20
   
   # ペーパートレードエラー検索
   Select-String -Path "logs/paper_trading/*.log" -Pattern "ERROR|FAILED" | Select-Object -Last 10
   
   # リアルタイムデータエラー検索
   Select-String -Path "logs/realtime_data.log" -Pattern "ERROR|WARNING" | Select-Object -Last 15
   
   # 警告ログ検索
   Select-String -Path "logs/*.log" -Pattern "WARNING" | Select-Object -Last 10
   ```

3. **パフォーマンス診断**
   ```powershell
   # パフォーマンステスト実行
   python benchmark_validator.py
   
   # リアルタイムデータパフォーマンス
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print(system.get_performance_metrics())"
   
   # ペーパートレードパフォーマンス
   python performance_monitor.py --analyze
   ```

4. **データ品質診断**
   ```powershell
   # データ品質レポート生成
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print(system.generate_quality_report())"
   
   # キャッシュ効率分析
   python -c "from src.data.realtime_cache import RealtimeCache; cache = RealtimeCache(); print(cache.analyze_performance())"
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
   Copy-Item src/config/ "backup/$backupDate/src_config/" -Recurse
   ```

2. **パラメータファイルバックアップ**
   ```powershell
   # パラメータ専用バックアップ
   Copy-Item config/optimized_params/ "backup/params_$(Get-Date -Format "yyyyMMddHHmm")/" -Recurse
   ```

3. **リアルタイムデータキャッシュバックアップ**
   ```powershell
   # キャッシュデータバックアップ
   Copy-Item logs/cache/ "backup/cache_$(Get-Date -Format "yyyyMMddHHmm")/" -Recurse
   
   # データ品質履歴バックアップ
   Copy-Item logs/data_quality_*.json "backup/data_quality/" -Force
   ```

4. **ペーパートレード履歴バックアップ**
   ```powershell
   # ペーパートレード実行履歴
   Copy-Item logs/paper_trading/ "backup/paper_trading_$(Get-Date -Format "yyyyMMddHHmm")/" -Recurse
   ```

#### 5.2 復旧手順

1. **設定復旧**
   ```powershell
   # 設定ファイル復旧
   $restoreDate = "20250801"  # 復旧対象日付
   Copy-Item "backup/$restoreDate/config/" config/ -Recurse -Force
   Copy-Item "backup/$restoreDate/src_config/" src/config/ -Recurse -Force
   ```

2. **パラメータ復旧**
   ```powershell
   # パラメータファイル復旧
   Copy-Item "backup/params_202508011200/" config/optimized_params/ -Recurse -Force
   ```

3. **システム状態復旧**
   ```powershell
   # リアルタイムデータシステム再初期化
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); system.reset_system()"
   
   # ペーパートレード環境リセット
   python -c "from src.execution.paper_broker import PaperBroker; broker = PaperBroker(); broker.reset_account()"
   ```

### 6. 監視・アラート

#### 6.1 監視項目

1. **システム監視**
   - CPU使用率
   - メモリ使用率
   - ディスク使用量
   - ネットワーク接続状態
   - リアルタイムデータフィード状態

2. **アプリケーション監視**
   - 戦略実行状態
   - データ取得状況（複数ソース）
   - エラー発生率
   - 処理時間
   - キャッシュ効率
   - データ品質スコア

3. **ビジネス監視**
   - ポジション状況（ペーパートレード）
   - リスク指標
   - パフォーマンス指標
   - 仮想約定状況
   - 戦略切替頻度
   - バックテスト vs リアルタイム乖離

4. **データ品質監視**
   - データ完全性スコア
   - データ精度評価
   - データ適時性チェック
   - 異常値検出状況

#### 6.2 アラート対応

1. **緊急アラート（Critical）**
   - システム停止
   - データ取得完全停止
   - ペーパートレード実行エラー
   - 重大なエラー
   - データ品質スコア閾値以下

   **対応手順:**
   ```powershell
   # 即座にシステム状態確認
   python src/monitoring/dashboard.py --check-critical
   
   # エラーログ確認
   Get-Content logs/errors.log -Tail 50
   Get-Content logs/paper_trading/errors.log -Tail 20
   
   # リアルタイムデータシステム状態確認
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print(system.get_system_status())"
   
   # 必要に応じてシステム再起動
   ```

2. **警告アラート（Warning）**
   - データ取得遅延
   - パフォーマンス低下
   - リスク制限接近
   - キャッシュ効率低下
   - データ品質軽微低下
   - ペーパートレード実行遅延

   **対応手順:**
   ```powershell
   # 詳細状況確認
   python demo_performance_monitor.py --analyze
   
   # リアルタイムデータ状況確認
   python demo_realtime_data_system.py --status-check
   
   # データ品質レポート確認
   python -c "from src.data.data_feed_integration import IntegratedDataFeedSystem; system = IntegratedDataFeedSystem(); print(system.generate_quality_report())"
   
   # 必要に応じて設定調整
   ```

3. **情報アラート（Info）**
   - 戦略切替発生
   - 新規データソース接続
   - キャッシュ更新完了
   - パフォーマンス改善検出

   **対応手順:**
   ```powershell
   # 情報確認のみ
   python performance_monitor.py --summary
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
- [ ] データ取得状況確認（Yahoo Finance + Alpha Vantage）
- [ ] リアルタイムデータフィード状態確認
- [ ] ペーパートレード実行状況確認
- [ ] エラーログ確認
- [ ] アラート状況確認
- [ ] パフォーマンス指標確認
- [ ] データ品質スコア確認
- [ ] キャッシュ効率確認
- [ ] ディスク使用量確認

#### 9.2 週次チェックリスト

- [ ] バックアップ実行状況確認
- [ ] ログローテーション実行
- [ ] ペーパートレード履歴アーカイブ
- [ ] パラメータ最適化結果レビュー
- [ ] システムパフォーマンス分析
- [ ] データ品質トレンド分析
- [ ] キャッシュ最適化実行
- [ ] セキュリティ状況確認

#### 9.3 月次チェックリスト

- [ ] 包括的システムテスト実行
- [ ] リアルタイムシステム総合評価
- [ ] ペーパートレード vs バックテスト比較分析
- [ ] パラメータ最適化全体レビュー
- [ ] リスク管理設定見直し
- [ ] 監視設定最適化
- [ ] データソース性能評価
- [ ] 戦略パフォーマンス総合評価
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

## 📝 アップデート履歴

### Version 2.0 (2025年8月13日)
- **フェーズ3B: リアルタイムデータ接続システム**の追加
- **ペーパートレード環境**の運用手順追加
- **統合監視システム**の操作方法追加
- **データ品質管理**の監視項目追加
- **新しいディレクトリ構造**に対応
- **トラブルシューティング**の拡充
- **診断コマンド**の追加
- **バックアップ手順**の強化

### Version 1.0 (2025年8月)
- 初版作成
- 基本的なバックテストシステム運用手順

---

**最終更新**: 2025年8月13日  
**バージョン**: 2.0  
**更新者**: AI Assistant (GitHub Copilot)
