# インシデント対応計画
## フェーズ4B: 本番環境移行準備

### 1. インシデント対応概要

本文書は、マルチ戦略バックテストシステムで発生する可能性のあるインシデントに対する対応手順を定義します。

#### 1.1 インシデント分類

**重要度レベル**
- **Critical (緊急)**: システム完全停止、データ損失、セキュリティ侵害
- **High (高)**: 主要機能停止、パフォーマンス重大劣化
- **Medium (中)**: 部分機能停止、軽微なパフォーマンス低下
- **Low (低)**: 軽微な問題、将来的なリスク

#### 1.2 対応時間目標

| 重要度 | 初期対応時間 | 復旧目標時間 |
|--------|-------------|-------------|
| Critical | 15分以内 | 1時間以内 |
| High | 30分以内 | 4時間以内 |
| Medium | 1時間以内 | 8時間以内 |
| Low | 4時間以内 | 24時間以内 |

### 2. インシデント検知

#### 2.1 自動検知システム

1. **監視ダッシュボード**
   ```powershell
   # ダッシュボード監視状態確認
   python src/monitoring/dashboard.py --status
   ```

2. **アラートマネージャー**
   ```powershell
   # アクティブアラート確認
   python -c "from src.monitoring.alert_manager import AlertManager; am = AlertManager(); print(am.get_active_alerts())"
   ```

3. **メトリクス異常検知**
   ```powershell
   # メトリクス異常確認
   python src/monitoring/metrics_collector.py --check-anomalies
   ```

#### 2.2 手動検知方法

1. **ログモニタリング**
   ```powershell
   # エラーログ監視
   Get-Content logs/errors.log -Wait
   
   # 重要ログ検索
   Select-String -Path "logs/*.log" -Pattern "CRITICAL|ERROR" | Select-Object -Last 10
   ```

2. **システム状態確認**
   ```powershell
   # プロセス確認
   Get-Process | Where-Object {$_.ProcessName -like "*python*"}
   
   # ポート使用状況確認
   netstat -an | findstr :5000
   ```

### 3. 初期対応手順

#### 3.1 インシデント発生時の即座対応

1. **状況確認 (2分以内)**
   ```powershell
   # システム稼働状況確認
   python -c "import sys; print('Python OK'); import pandas; print('Pandas OK'); import yfinance; print('YFinance OK')"
   
   # メインプロセス確認
   Get-Process | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*main.py*"}
   ```

2. **緊急ログ収集 (5分以内)**
   ```powershell
   # 緊急ログ作成
   $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
   New-Item -ItemType Directory -Path "incident_logs/$timestamp" -Force
   
   # 重要ログコピー
   Copy-Item logs/main.log "incident_logs/$timestamp/"
   Copy-Item logs/errors.log "incident_logs/$timestamp/"
   Get-Content logs/main.log -Tail 100 > "incident_logs/$timestamp/main_recent.log"
   ```

3. **アラート状況確認 (5分以内)**
   ```powershell
   # アクティブアラート確認
   python -c "from src.monitoring.alert_manager import AlertManager; am = AlertManager(); alerts = am.get_active_alerts(); print(f'Active alerts: {len(alerts)}'); [print(f'- {alert}') for alert in alerts[:5]]"
   ```

#### 3.2 影響範囲評価

1. **システム機能確認**
   ```powershell
   # 基本機能テスト
   python config/basic_system_test.py
   
   # データ取得テスト
   python -c "import yfinance as yf; ticker = yf.Ticker('AAPL'); print('Data fetch:', 'OK' if ticker.history(period='1d').empty == False else 'FAILED')"
   ```

2. **戦略実行状況確認**
   ```powershell
   # 戦略実行テスト
   python -c "from strategies.VWAP_Breakout import VWAPBreakoutStrategy; strategy = VWAPBreakoutStrategy(); print('Strategy load: OK')"
   ```

### 4. Critical レベルインシデント対応

#### 4.1 システム完全停止

**症状:**
- main.pyが実行できない
- Pythonプロセスがクラッシュ
- 重要なモジュールが読み込めない

**対応手順:**

1. **即座の安全確保 (5分以内)**
   ```powershell
   # プロセス強制終了
   Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force
   
   # 一時ファイル削除
   Remove-Item temp/* -Force -ErrorAction SilentlyContinue
   ```

2. **システム診断 (10分以内)**
   ```powershell
   # Python環境確認
   python --version
   pip list | findstr "pandas numpy yfinance"
   
   # ファイル整合性確認
   python -m py_compile main.py
   python -m py_compile config/optimized_parameters.py
   ```

3. **復旧実行 (30分以内)**
   ```powershell
   # 設定ファイル復旧
   Copy-Item "backup/$(Get-Date -Format 'yyyyMMdd')/config/" config/ -Recurse -Force
   
   # システム再起動
   python main.py --safe-mode
   ```

#### 4.2 データ損失・破損

**症状:**
- パラメータファイルが読み込めない
- 設定ファイルが破損
- ログファイルが異常

**対応手順:**

1. **データ損失範囲確認 (5分以内)**
   ```powershell
   # ファイル整合性確認
   Test-Path config/optimized_parameters.py
   Test-Path config/backtest_config.xlsm
   
   # ファイルサイズ確認
   Get-ChildItem config/*.py | Select-Object Name, Length, LastWriteTime
   ```

2. **バックアップからの復旧 (15分以内)**
   ```powershell
   # 最新バックアップ確認
   Get-ChildItem backup/ | Sort-Object LastWriteTime -Descending | Select-Object -First 3
   
   # バックアップから復旧
   $latestBackup = (Get-ChildItem backup/ | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
   Copy-Item "backup/$latestBackup/config/" config/ -Recurse -Force
   ```

3. **データ整合性検証 (20分以内)**
   ```powershell
   # パラメータ読み込みテスト
   python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); print('Parameter load: OK')"
   
   # 設定ファイル読み込みテスト
   python config/basic_system_test.py
   ```

#### 4.3 セキュリティインシデント

**症状:**
- 不正アクセスの疑い
- ファイル改ざんの疑い
- 外部からの異常アクセス

**対応手順:**

1. **即座の隔離 (2分以内)**
   ```powershell
   # ネットワーク接続遮断
   # 注意: 組織のセキュリティポリシーに従って実行
   
   # プロセス停止
   Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force
   ```

2. **証跡保全 (10分以内)**
   ```powershell
   # ログ緊急保全
   $securityIncidentTime = Get-Date -Format "yyyyMMdd_HHmmss"
   New-Item -ItemType Directory -Path "security_incident/$securityIncidentTime" -Force
   Copy-Item logs/ "security_incident/$securityIncidentTime/logs/" -Recurse
   
   # ファイル変更履歴確認
   Get-ChildItem config/ -Recurse | Select-Object FullName, LastWriteTime | Sort-Object LastWriteTime -Descending
   ```

3. **セキュリティチーム連絡**
   - 即座にセキュリティ責任者に連絡
   - インシデント詳細を報告
   - 指示に従って追加対応実施

### 5. High レベルインシデント対応

#### 5.1 主要機能停止

**症状:**
- 特定戦略が実行できない
- データ取得が断続的に失敗
- 監視ダッシュボードが応答しない

**対応手順:**

1. **機能別診断 (10分以内)**
   ```powershell
   # 戦略別テスト
   python -c "from strategies.VWAP_Breakout import VWAPBreakoutStrategy; s = VWAPBreakoutStrategy(); print('VWAP Strategy: OK')"
   python -c "from strategies.momentum_investing import MomentumInvestingStrategy; s = MomentumInvestingStrategy(); print('Momentum Strategy: OK')"
   
   # データソーステスト
   python -c "import yfinance as yf; print('YFinance status:', 'OK' if yf.Ticker('AAPL').info else 'FAILED')"
   ```

2. **代替手段による継続 (20分以内)**
   ```powershell
   # セーフモードでの実行
   python main.py --safe-mode --single-strategy VWAPBreakoutStrategy
   
   # 最小限機能での実行
   python main.py --minimal-mode
   ```

3. **根本原因調査 (30分以内)**
   ```powershell
   # 詳細ログ分析
   Select-String -Path "logs/*.log" -Pattern "ERROR.*$(Get-Date -Format 'yyyy-MM-dd')" | Select-Object -Last 20
   
   # 依存関係確認
   python -m pip check
   ```

#### 5.2 パフォーマンス重大劣化

**症状:**
- 実行時間が通常の3倍以上
- メモリ使用量が異常増加
- CPU使用率が持続的に高い

**対応手順:**

1. **リソース使用状況確認 (5分以内)**
   ```powershell
   # プロセス監視
   Get-Process | Where-Object {$_.ProcessName -eq "python"} | Select-Object Id, ProcessName, CPU, WorkingSet
   
   # メモリ使用量確認
   python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%'); print(f'CPU: {psutil.cpu_percent()}%')"
   ```

2. **パフォーマンス調整 (15分以内)**
   ```powershell
   # 最適化モードでの実行
   python main.py --performance-mode
   
   # 並列処理制限
   python main.py --max-workers 2
   ```

3. **リソース解放 (20分以内)**
   ```powershell
   # ガベージコレクション強制実行
   python -c "import gc; gc.collect(); print('GC completed')"
   
   # 一時ファイル削除
   Remove-Item temp/* -Force -ErrorAction SilentlyContinue
   Remove-Item config/data_cache/* -Force -ErrorAction SilentlyContinue
   ```

### 6. Medium レベルインシデント対応

#### 6.1 部分機能停止

**症状:**
- 特定の戦略でエラーが発生
- 一部データ取得が失敗
- 非重要な機能が動作しない

**対応手順:**

1. **問題範囲特定 (15分以内)**
   ```powershell
   # 戦略別診断
   python comprehensive_trend_switching_test_suite.py --specific-strategy
   
   # 機能別テスト
   python config/basic_system_test.py --partial
   ```

2. **ワークアラウンド実装 (30分以内)**
   ```powershell
   # 問題戦略を除外して実行
   python main.py --exclude-strategy MomentumInvestingStrategy
   
   # デフォルトパラメータで実行
   python main.py --use-default-params
   ```

3. **修正実装 (1時間以内)**
   ```powershell
   # パラメータリセット
   python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); opm.reset_to_defaults('MomentumInvestingStrategy')"
   
   # 設定ファイル再生成
   python config/strategy_params_regenerate.py
   ```

#### 6.2 軽微なパフォーマンス低下

**症状:**
- 実行時間が通常の1.5-2倍
- 軽微なメモリリーク
- 応答時間の増加

**対応手順:**

1. **パフォーマンス分析 (20分以内)**
   ```powershell
   # パフォーマンスプロファイリング
   python demo_performance_monitor.py --detailed-analysis
   
   # メモリ使用パターン分析
   python -c "from src.monitoring.metrics_collector import MetricsCollector; mc = MetricsCollector(); print(mc.analyze_memory_usage())"
   ```

2. **最適化実装 (45分以内)**
   ```powershell
   # キャッシュ最適化
   python config/cache_manager.py --optimize
   
   # データ処理最適化
   python main.py --optimize-data-processing
   ```

### 7. Low レベルインシデント対応

#### 7.1 軽微な問題

**症状:**
- 警告レベルのログメッセージ
- 軽微な設定ミス
- 非重要データの欠損

**対応手順:**

1. **問題記録 (30分以内)**
   ```powershell
   # ログエントリ作成
   Add-Content logs/maintenance.log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Low priority issue detected"
   
   # 問題詳細記録
   python -c "from config.logger_config import setup_logger; logger = setup_logger('maintenance'); logger.info('Low priority maintenance required')"
   ```

2. **定期メンテナンス時対応**
   - 週次メンテナンス時に対応
   - 影響評価実施
   - 必要に応じて修正

### 8. 復旧後の手順

#### 8.1 システム検証

1. **機能確認テスト (30分以内)**
   ```powershell
   # 包括的システムテスト
   python comprehensive_matrix_test.py
   
   # 戦略実行テスト
   python main.py --test-mode
   ```

2. **パフォーマンス確認 (20分以内)**
   ```powershell
   # ベンチマークテスト
   python benchmark_validator.py
   
   # リソース使用量確認
   python demo_performance_monitor.py --benchmark
   ```

3. **データ整合性確認 (15分以内)**
   ```powershell
   # パラメータ整合性確認
   python check_param_combinations.py
   
   # 設定ファイル検証
   python config/basic_system_test.py --validation
   ```

#### 8.2 監視強化

1. **アラート設定調整**
   ```powershell
   # アラート感度調整
   python src/monitoring/alert_manager.py --adjust-sensitivity
   
   # 追加監視項目設定
   python src/monitoring/metrics_collector.py --add-monitoring
   ```

2. **ログ監視強化**
   ```powershell
   # ログ監視間隔短縮
   python -c "from src.monitoring.dashboard import Dashboard; d = Dashboard(); d.increase_monitoring_frequency()"
   ```

### 9. インシデント報告

#### 9.1 報告書作成

**Critical/High レベル**
- インシデント発生時刻
- 検知方法
- 影響範囲
- 対応実施時刻
- 復旧時刻
- 根本原因
- 再発防止策

**Medium/Low レベル**
- 問題概要
- 対応内容
- 今後の対応計画

#### 9.2 報告テンプレート

```powershell
# インシデント報告書生成
$incidentTime = Get-Date -Format "yyyyMMdd_HHmmss"
$reportPath = "incident_reports/incident_$incidentTime.md"

@"
# インシデント報告書

## 基本情報
- **発生時刻**: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
- **検知時刻**: 
- **重要度**: 
- **対応者**: 

## 問題概要


## 影響範囲


## 対応内容


## 根本原因


## 再発防止策


## 添付ログ
- logs/main.log
- logs/errors.log
- incident_logs/$incidentTime/
"@ | Out-File $reportPath
```

### 10. 予防策・改善

#### 10.1 定期メンテナンス

1. **週次予防メンテナンス**
   ```powershell
   # システム健全性チェック
   python comprehensive_trend_switching_test_suite.py --health-check
   
   # ログローテーション
   python config/log_rotation.py
   
   # キャッシュクリーンアップ
   python config/cache_manager.py --cleanup
   ```

2. **月次総合メンテナンス**
   ```powershell
   # 包括的システム診断
   python comprehensive_matrix_test.py --full-diagnostic
   
   # パラメータ最適化レビュー
   python check_param_combinations.py --review
   
   # セキュリティ監査
   python security_audit.py
   ```

#### 10.2 システム改善

1. **監視強化**
   - 新しいメトリクスの追加
   - アラート精度の向上
   - 予測分析の導入

2. **自動復旧機能**
   - 自動フェイルオーバー
   - 自動バックアップ復旧
   - 自動パフォーマンス調整

3. **ドキュメント更新**
   - インシデント事例の蓄積
   - 対応手順の改良
   - トレーニング資料の更新

### 11. 緊急連絡先

#### 11.1 エスカレーション順序

1. **Level 1**: 運用担当者
   - 連絡先: [運用担当者連絡先]
   - 対応時間: 24時間

2. **Level 2**: システム管理者
   - 連絡先: [システム管理者連絡先]
   - 対応時間: 24時間 (Critical時)、営業時間 (その他)

3. **Level 3**: 開発チーム
   - 連絡先: [開発チーム連絡先]
   - 対応時間: 営業時間 (緊急時は24時間)

#### 11.2 外部ベンダー

- **クラウドプロバイダー**: [連絡先]
- **データプロバイダー**: [連絡先]
- **システム保守**: [連絡先]

### 12. チェックリスト

#### 12.1 インシデント対応チェックリスト

**初期対応**
- [ ] インシデント検知時刻記録
- [ ] 重要度レベル判定
- [ ] 初期ログ収集
- [ ] 影響範囲評価
- [ ] 関係者通知

**対応実施**
- [ ] 対応手順書に従った実施
- [ ] 対応実施時刻記録
- [ ] 中間報告実施
- [ ] 復旧確認
- [ ] 二次被害防止確認

**事後対応**
- [ ] システム検証実施
- [ ] インシデント報告書作成
- [ ] 根本原因分析実施
- [ ] 再発防止策策定
- [ ] 改善計画作成

---

**重要な注意事項:**
- インシデント対応時は冷静に対応してください
- 不明な点は必ずエスカレーションしてください
- 独断での重大な変更は避けてください
- すべての対応を記録してください

**最終更新**: 2025年1月
**バージョン**: 1.0
