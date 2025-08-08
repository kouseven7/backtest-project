# フェーズ3B リアルタイムデータ監視ダッシュボード

このモジュールは、フェーズ3Bリアルタイムデータシステムの包括的な監視ダッシュボードを提供します。Webベースのインターフェースで、データ品質、システム性能、ネットワーク状態、アラートをリアルタイムで監視できます。

## 🚀 主要機能

### 📊 リアルタイム監視
- **データ品質監視**: 4軸品質評価（完全性、精度、適時性、一貫性）
- **システム性能監視**: CPU、メモリ、応答時間、エラー率
- **ネットワーク監視**: データソース別の接続状態、遅延、成功率
- **キャッシュ監視**: ヒット率、サイズ、使用状況

### 🎛️ Webダッシュボード
- **リアルタイム更新**: WebSocketによる自動更新
- **インタラクティブチャート**: Plotlyによる高度な可視化
- **レスポンシブデザイン**: モバイル対応のBootstrapベース
- **マルチクライアント対応**: 複数ユーザーの同時接続

### 🚨 アラート管理
- **ルールベースアラート**: カスタマイズ可能なアラート条件
- **マルチチャネル通知**: メール、Slack、Webhook対応
- **エスカレーション機能**: 未解決アラートの自動エスカレーション
- **アラート抑制**: 重複アラートの防止

### 📈 メトリクス収集・分析
- **自動メトリクス収集**: システム全体の包括的メトリクス
- **統計分析**: 平均、中央値、パーセンタイル計算
- **トレンド分析**: 時系列データの傾向分析
- **エクスポート機能**: JSON形式での履歴データ出力

## 📁 ファイル構成

```
src/monitoring/
├── __init__.py                 # モジュール初期化
├── dashboard.py               # メインダッシュボード
├── metrics_collector.py       # メトリクス収集
├── alert_manager.py           # アラート管理
├── test_dashboard.py          # テストスイート
├── demo_dashboard.py          # デモンストレーション
├── install_dependencies.py    # 依存関係インストール
├── README.md                  # このファイル
├── templates/                 # HTMLテンプレート
│   └── dashboard.html         # ダッシュボードUI
└── static/                    # 静的ファイル
    ├── css/                   # スタイルシート
    └── js/                    # JavaScript
```

## 🛠️ セットアップ

### 1. 依存関係インストール

```powershell
# 自動インストール
python src\monitoring\install_dependencies.py

# 手動インストール
pip install fastapi uvicorn jinja2 plotly numpy psutil requests websockets
```

### 2. 設定ファイル確認

`config/realtime_config.json` の設定を確認してください：

```json
{
  "cache": {
    "memory_max_items": 500,
    "memory_max_mb": 512
  },
  "data_quality": {
    "max_price_change_percent": 15.0,
    "enable_quality_alerts": true
  },
  "error_handling": {
    "auto_correction": true,
    "quality_threshold": 0.7
  }
}
```

## 🧪 テスト実行

### 基本テスト

```powershell
# テストスイート実行
python src\monitoring\test_dashboard.py
```

期待される出力：
```
フェーズ3B 監視ダッシュボード テストスイート
============================================================
[INFO] Testing component initialization...
[INFO] Test 'component_initialization': PASSED
[INFO] Testing metrics collection...
[INFO] Test 'metrics_collection': PASSED
...
============================================================
DASHBOARD TEST REPORT
============================================================
Total Tests: 6
Passed: 6
Failed: 0
Success Rate: 100.0%
============================================================
```

### デモンストレーション

```powershell
# デモ実行
python src\monitoring\demo_dashboard.py
```

期待される出力：
```
============================================================
フェーズ3B リアルタイムデータ監視ダッシュボード デモ
============================================================
ダッシュボードURL: http://localhost:8080
デモデータが自動生成され、リアルタイムで更新されます
Ctrl+C で停止
============================================================

📊 デモ状態 [14:30:25]
   収集メトリクス: 1250
   アクティブアラート: 2
   総アラート数: 8
   平均品質スコア: 0.82
   平均応答時間: 125.3ms
```

## 💻 ダッシュボード使用方法

### 1. ダッシュボード起動

```powershell
# デモデータ付きで起動
python src\monitoring\demo_dashboard.py

# または本番データで起動
python -c "
from src.monitoring.dashboard import create_dashboard
from src.data.data_feed_integration import IntegratedDataFeedSystem

data_feed = IntegratedDataFeedSystem()
dashboard = create_dashboard(data_feed)
dashboard.start()
"
```

### 2. Webアクセス

ブラウザで `http://localhost:8080` にアクセス

### 3. ダッシュボード画面

#### システム概要パネル
- データポイント/秒
- アクティブシンボル数
- 平均品質スコア
- キャッシュヒット率
- CPU/メモリ使用率

#### メインチャート
- **データ品質推移**: 時系列品質スコア
- **システムパフォーマンス**: 応答時間、エラー率

#### アラート・ログパネル
- 最新アラート一覧
- データソース状態
- 自動更新状態インジケーター

## 🔧 カスタマイズ

### アラートルール追加

```python
from src.monitoring.alert_manager import AlertRule, AlertLevel, AlertCategory

# カスタムアラートルール
custom_rule = AlertRule(
    rule_id="custom_quality_rule",
    name="カスタム品質アラート",
    category=AlertCategory.DATA_QUALITY,
    level=AlertLevel.WARNING,
    condition="overall_score < 0.8 and completeness_score < 0.9",
    threshold=0.8,
    time_window_minutes=10,
    description="品質スコアとデータ完全性の複合条件"
)

alert_manager.add_alert_rule(custom_rule)
```

### 通知チャネル設定

```python
from src.monitoring.alert_manager import NotificationChannel, AlertLevel

# Slack通知
slack_channel = NotificationChannel(
    channel_id="slack_alerts",
    name="Slack通知",
    type="slack",
    config={
        "webhook_url": "https://hooks.slack.com/services/...",
    },
    alert_levels=[AlertLevel.ERROR, AlertLevel.CRITICAL]
)

alert_manager.notification_manager.add_channel(slack_channel)

# メール通知
email_channel = NotificationChannel(
    channel_id="email_admin",
    name="管理者メール",
    type="email", 
    config={
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your_email@gmail.com",
        "password": "your_password",
        "from_email": "alerts@yourcompany.com",
        "to_email": "admin@yourcompany.com",
        "use_tls": True
    }
)

alert_manager.notification_manager.add_channel(email_channel)
```

### ダッシュボード設定カスタマイズ

```python
from src.monitoring.dashboard import DashboardConfig

config = DashboardConfig(
    host="0.0.0.0",  # 外部アクセス許可
    port=8080,
    auto_refresh_interval=5,  # 5秒間隔更新
    max_history_points=2000,  # 履歴データ数
    chart_update_interval=3,  # チャート更新間隔
    chart_colors={
        'excellent': '#28a745',
        'good': '#17a2b8',
        'fair': '#ffc107',
        'poor': '#fd7e14',
        'invalid': '#dc3545'
    }
)
```

## 🔍 トラブルシューティング

### よくある問題

#### 1. ダッシュボードが起動しない
```powershell
# ポート使用状況確認
netstat -an | findstr :8080

# 別ポートで起動
python -c "
config = DashboardConfig(port=8081)
dashboard = create_dashboard(data_feed, config)
dashboard.start()
"
```

#### 2. WebSocketで接続エラー
- ファイアウォール設定を確認
- ブラウザのWebSocket対応を確認
- プロキシ設定を確認

#### 3. メトリクスが表示されない
```python
# データフィード接続確認
data_feed = IntegratedDataFeedSystem()
print("データソース:", data_feed.data_manager.adapters.keys())

# メトリクス手動生成
metrics_collector.record_performance_metrics("test", 100, True)
summary = metrics_collector.get_all_metrics_summary()
print("メトリクスサマリー:", summary)
```

#### 4. アラートが発生しない
```python
# アラートルール確認
for rule_id, rule in alert_manager.alert_rules.items():
    print(f"{rule_id}: {rule.enabled} - {rule.condition}")

# 手動アラート評価
test_metrics = {'quality': {'overall_score': 0.5}}
alert_manager.evaluate_metrics(test_metrics)
print("アクティブアラート:", alert_manager.get_active_alerts())
```

## 📊 メトリクス仕様

### データ品質メトリクス
- **完全性スコア**: 必須フィールドの充足率
- **精度スコア**: データの正確性、異常値検出
- **適時性スコア**: データの鮮度、遅延評価
- **一貫性スコア**: データ間の整合性

### システム性能メトリクス
- **応答時間**: 平均、中央値、95/99パーセンタイル
- **エラー率**: 失敗リクエスト比率
- **スループット**: 秒間処理数
- **リソース使用率**: CPU、メモリ使用量

### ネットワークメトリクス
- **レスポンス時間**: データソース別応答時間
- **成功率**: 接続成功率
- **タイムアウト率**: タイムアウト発生率
- **エラー率**: ネットワークエラー率

## 🔒 セキュリティ注意事項

1. **本番環境での設定**
   - HTTPS使用を推奨
   - 認証機能の追加を検討
   - ファイアウォールでのアクセス制限

2. **機密情報の取り扱い**
   - 設定ファイルでの認証情報の暗号化
   - ログでの機密情報マスキング
   - 通信の暗号化

3. **アクセス制御**
   - IPアドレス制限
   - ユーザー認証
   - 操作ログの記録

## 📚 関連ドキュメント

- [フェーズ3B実装レポート](../../PHASE_3B_REALTIME_DATA_IMPLEMENTATION_REPORT.md)
- [データフィード統合](../data/data_feed_integration.py)
- [エラーハンドリング](../error_handling/)
- [設定管理](../../config/)

## 🆘 サポート

問題が発生した場合：

1. ログファイル確認: `logs/` ディレクトリ
2. テストスイート実行: `python src/monitoring/test_dashboard.py`
3. デモモードでの動作確認: `python src/monitoring/demo_dashboard.py`
4. 設定ファイル確認: `config/realtime_config.json`

---

**フェーズ3B リアルタイムデータ監視ダッシュボード v1.0.0**  
© 2024 My Backtest Project
