# DSSMS 初心者向け操作マニュアル
## Dynamic Stock Selection & Management System

---

## 目次
1. [DSSMSとは](#dssmsとは)
2. [前提知識とPowerShell基本操作](#前提知識とpowershell基本操作)
3. [基本操作](#基本操作)
4. [バックテストの実行方法](#バックテストの実行方法)
5. [設定の変更方法](#設定の変更方法)
6. [コマンド一覧](#コマンド一覧)
7. [実行例とサンプル](#実行例とサンプル)
8. [定期運用コマンド](#定期運用コマンド)
9. [トラブルシューティング](#トラブルシューティング)

---

## DSSMSとは

DSSMS（Dynamic Stock Selection & Management System）は、日経225銘柄を対象とした動的株式選定・管理システムです。複数の投資戦略を組み合わせて、効率的なバックテストと最適化を行うことができます。

### 主な機能
- **日経225銘柄スクリーニング**: ファンダメンタル分析による銘柄選定
- **パーフェクトオーダー検出**: トレンド判定による最適な売買タイミング
- **階層ランキングシステム**: 複数指標による銘柄評価
- **包括的スコアリング**: 総合的な投資判断
- **市場状況監視**: リアルタイム市場分析
- **インテリジェント切り替え**: 市場状況に応じた戦略変更
- **自動スケジューリング**: 定期実行とレポート生成

### システム構成
```
src/dssms/                    # DSSMSメインシステム
├── nikkei225_screener.py     # 日経225銘柄スクリーニング
├── fundamental_analyzer.py   # ファンダメンタル分析
├── perfect_order_detector.py # パーフェクトオーダー検出
├── hierarchical_ranking_system.py # 階層ランキング
├── comprehensive_scoring_engine.py # 包括的スコアリング
├── market_condition_monitor.py # 市場状況監視
├── intelligent_switch_manager.py # インテリジェント切り替え
├── dssms_scheduler.py        # スケジューラー
├── dssms_analyzer.py         # 分析システム
├── dssms_backtester.py       # バックテスター
└── dssms_data_manager.py     # データ管理

config/dssms/                 # 設定ファイル
├── dssms_config.json         # メイン設定
├── ranking_config.json       # ランキング設定
├── scoring_engine_config.json # スコアリング設定
├── market_monitoring_config.json # 市場監視設定
├── intelligent_switch_config.json # 切り替え設定
└── scheduler_config.json     # スケジューラー設定
```

### [WARNING] 重要な注意事項
- **UnifiedTrendDetector警告**: システム起動時に循環インポート警告が表示されますが、基本機能は正常に動作します
- **メソッド名**: 一部のメソッド名が従来バージョンから変更されています（例：`get_nikkei225_symbols()` → `fetch_nikkei225_symbols()`）
- **データ取得**: 初回実行時にネットワーク接続によりデータ取得に時間がかかる場合があります
- **yfinance警告**: データ取得時に設定変更の警告が表示されますが、正常にデータは取得されます

---

## 前提知識とPowerShell基本操作

### PowerShellの基本概念
PowerShellは、Windows標準のコマンドラインインターフェースです。DSSMSの操作には以下の基本知識が必要です。

### 基本的なPowerShellコマンド

#### ディレクトリ操作
```powershell
# 現在のディレクトリを確認
pwd

# ディレクトリの移動
cd C:\Users\imega\Documents\my_backtest_project

# ディレクトリの内容確認
dir
# または
Get-ChildItem

# 特定のファイルを検索
dir *dssms*
# または
Get-ChildItem -Name "*dssms*"
```
**期待される結果**: 現在位置の確認、指定フォルダへの移動、ファイル一覧の表示

#### Pythonコマンドの実行
```powershell
# Python環境の確認
python --version
# 期待される結果: Python 3.x.x のバージョン情報

# 仮想環境の有効化（必要に応じて）
.\.venv\Scripts\Activate.ps1
# 期待される結果: プロンプトの前に (.venv) が表示される

# Pythonスクリプトの実行
python script_name.py
# 期待される結果: スクリプトの実行結果が表示される

# 複数コマンドの連結（PowerShell方式）
python -c "print('Hello')"; echo "Done"
# 期待される結果: Hello と Done が順番に表示される
```

#### ファイル操作
```powershell
# ファイルの内容確認
Get-Content config\dssms\dssms_config.json

# ファイルの編集（メモ帳で開く）
notepad config\dssms\dssms_config.json

# ファイルの存在確認
Test-Path src\dssms\dssms_analyzer.py
# 期待される結果: True（存在する場合）
```

### PowerShellでのPythonコード実行

#### 単行コードの実行
```powershell
python -c "import pandas; print('pandas導入済み')"
# 期待される結果: "pandas導入済み" と表示される

python -c "print('Hello, DSSMS!')"
# 期待される結果: "Hello, DSSMS!" と表示される
```

#### 複数行コードの実行
```powershell
python -c "
import sys
print(f'Python version: {sys.version}')
print('DSSMSシステムチェック開始')
"
# 期待される結果: Pythonバージョンとメッセージが表示される
```

#### 文字エスケープの注意点
```powershell
# 正しい書き方（ダブルクォート内でのエスケープ）
python -c "print('Hello\nWorld')"

# JSON形式の文字列を扱う場合
python -c "import json; print(json.dumps({'key': 'value'}))"
```

## 6. DSSMSバックテスト期間の変更方法

### 基本的な期間変更手順

#### 設定ファイルによる期間変更
DSSMSでは設定ファイルを使用してバックテスト期間を管理できます。

```powershell
# 1. 設定ファイルを作成
python -c "
import json
from datetime import datetime

config = {
    'default_backtest_period': {
        'start_date': '2023-06-01',
        'end_date': '2023-12-31',
        'description': 'デフォルトバックテスト期間'
    },
    'test_periods': {
        '3months': {'start': '2023-10-01', 'end': '2023-12-31'},
        '6months': {'start': '2023-07-01', 'end': '2023-12-31'},
        '1year': {'start': '2023-01-01', 'end': '2023-12-31'}
    },
    'rebalance_settings': {
        'frequency': 'weekly',
        'switch_cost_rate': 0.001
    }
}

with open('config/dssms/backtester_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('設定ファイルが作成されました: config/dssms/backtester_config.json')
"
```

#### 直接的な期間指定によるバックテスト実行

##### 短期間（3ヶ月）でのテスト
```powershell
python -c "
from src.dssms.dssms_backtester import DSSMSBacktester
from datetime import datetime

print('=== 3ヶ月期間バックテスト ===')

backtester = DSSMSBacktester()
symbol_universe = ['7203', '6758', '9984', '8306', '9432']

start_date = datetime(2023, 10, 1)  # 2023年10月1日
end_date = datetime(2023, 12, 31)   # 2023年12月31日

print(f'期間: {start_date.strftime(\"%Y-%m-%d\")} ～ {end_date.strftime(\"%Y-%m-%d\")}')

results = backtester.simulate_dynamic_selection(
    start_date=start_date,
    end_date=end_date,
    symbol_universe=symbol_universe
)

print(f'バックテスト完了: 期間 {(end_date - start_date).days}日間')
"
```

##### 中期間（6ヶ月）でのテスト
```powershell
python -c "
from src.dssms.dssms_backtester import DSSMSBacktester
from datetime import datetime

print('=== 6ヶ月期間バックテスト ===')

backtester = DSSMSBacktester()
symbol_universe = ['7203', '6758', '9984', '8306', '9432']

start_date = datetime(2023, 7, 1)   # 2023年7月1日
end_date = datetime(2023, 12, 31)   # 2023年12月31日

print(f'期間: {start_date.strftime(\"%Y-%m-%d\")} ～ {end_date.strftime(\"%Y-%m-%d\")}')

results = backtester.simulate_dynamic_selection(
    start_date=start_date,
    end_date=end_date,
    symbol_universe=symbol_universe
)

print(f'バックテスト完了: 期間 {(end_date - start_date).days}日間')
"
```

##### 長期間（1年）でのテスト
```powershell
python -c "
from src.dssms.dssms_backtester import DSSMSBacktester
from datetime import datetime

print('=== 1年期間バックテスト ===')

backtester = DSSMSBacktester()
symbol_universe = ['7203', '6758', '9984', '8306', '9432']

start_date = datetime(2023, 1, 1)   # 2023年1月1日
end_date = datetime(2023, 12, 31)   # 2023年12月31日

print(f'期間: {start_date.strftime(\"%Y-%m-%d\")} ～ {end_date.strftime(\"%Y-%m-%d\")}')

results = backtester.simulate_dynamic_selection(
    start_date=start_date,
    end_date=end_date,
    symbol_universe=symbol_universe
)

print(f'バックテスト完了: 期間 {(end_date - start_date).days}日間')
"
```

### 複数期間での比較テスト

#### 複数期間を一括実行
```powershell
python -c "
from src.dssms.dssms_backtester import DSSMSBacktester
from datetime import datetime

print('=== 複数期間比較テスト ===')

# テスト期間の定義
test_periods = [
    ('短期（3ヶ月）', '2023-10-01', '2023-12-31'),
    ('中期（6ヶ月）', '2023-07-01', '2023-12-31'),
    ('長期（1年）', '2023-01-01', '2023-12-31')
]

backtester = DSSMSBacktester()
symbol_universe = ['7203', '6758', '9984']  # 高速テスト用に3銘柄

for period_name, start_date_str, end_date_str in test_periods:
    print(f'\\n[CHART] {period_name} ({start_date_str} ～ {end_date_str})')
    
    try:
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        results = backtester.simulate_dynamic_selection(
            start_date=start_dt,
            end_date=end_dt,
            symbol_universe=symbol_universe
        )
        
        print(f'   [OK] テスト完了 - 期間: {(end_dt - start_dt).days}日間')
        
    except Exception as e:
        print(f'   [ERROR] エラー: {str(e)[:50]}...')

print('\\n[TARGET] 複数期間比較テスト完了')
"
```

### カスタム期間の設定

#### 任意の期間を指定する方法
```powershell
# 例: 2024年の第1四半期をテスト
python -c "
from src.dssms.dssms_backtester import DSSMSBacktester
from datetime import datetime

print('=== カスタム期間バックテスト（2024 Q1） ===')

backtester = DSSMSBacktester()
symbol_universe = ['7203', '6758', '9984', '8306', '9432']

# カスタム期間設定
start_date = datetime(2024, 1, 1)   # 開始日
end_date = datetime(2024, 3, 31)    # 終了日

print(f'カスタム期間: {start_date.strftime(\"%Y-%m-%d\")} ～ {end_date.strftime(\"%Y-%m-%d\")}')

results = backtester.simulate_dynamic_selection(
    start_date=start_date,
    end_date=end_date,
    symbol_universe=symbol_universe
)

print(f'バックテスト完了: 期間 {(end_date - start_date).days}日間')
"
```

### 期間変更時の注意点

1. **データ可用性の確認**
   - 指定期間のデータが取得可能か事前に確認してください
   - 休日や市場休場日は自動的に除外されます

2. **計算時間の考慮**
   - 長期間（1年以上）の場合、計算時間が長くなります
   - テスト時は短期間または少数銘柄で動作確認することを推奨します

3. **メモリ使用量**
   - 大量の銘柄×長期間の組み合わせはメモリを多く消費します
   - システムリソースを確認してから実行してください

4. **結果の解釈**
   - 異なる期間の結果を比較する際は、市場環境の違いを考慮してください
   - 短期間の結果は偶然性の影響を受けやすい場合があります

---

## 基本操作

### プロジェクトディレクトリの確認
```powershell
# DSSMSプロジェクトディレクトリに移動
cd C:\Users\imega\Documents\my_backtest_project

# プロジェクト構造の確認
dir src\dssms
# 期待される結果: DSSMSの主要Pythonファイルが一覧表示される

dir config\dssms
# 期待される結果: DSSMS設定ファイル（*.json）が一覧表示される
```

### Python環境の確認
```powershell
# Python環境の基本確認
python --version
# 期待される結果: Python 3.x.x（3.8以上推奨）

# 必要なライブラリの確認
python -c "
try:
    import pandas, numpy, yfinance, scipy
    print('✓ 必要ライブラリ導入済み')
except ImportError as e:
    print(f'✗ ライブラリ不足: {e}')
"
# 期待される結果: "✓ 必要ライブラリ導入済み" または不足ライブラリの表示
```

### システムの基本動作確認
```powershell
# DSSMSモジュールの読み込みテスト
python -c "
try:
    import sys
    sys.path.append('src')
    from dssms import nikkei225_screener
    print('✓ DSSMSモジュール正常読み込み')
except Exception as e:
    print(f'✗ モジュール読み込みエラー: {e}')
"
# 期待される結果: "✓ DSSMSモジュール正常読み込み" または具体的なエラー内容
```

---

## バックテストの実行方法

### 1. Phase1テスト（銘柄スクリーニング）

```powershell
# Phase1デモの実行
python demo_dssms_phase1.py
# 期待される結果: 
# - 日経225銘柄のスクリーニング結果
# - 候補銘柄数の表示
# - 上位銘柄の詳細情報

# Phase1テストの実行
python test_dssms_phase1.py
# 期待される結果: スクリーニングシステムのテスト結果とパス/フェイル
```

### 2. Phase2テスト（ランキングと評価）

```powershell
# Phase2デモの実行
python demo_dssms_phase2.py
# 期待される結果:
# - 銘柄ランキング結果
# - スコアリング詳細
# - 最終的な投資推奨リスト

# Phase2テストの実行
python test_dssms_phase2.py
# 期待される結果: ランキングシステムのテスト結果とパス/フェイル
```

### 3. 市場監視システムの実行

```powershell
# 市場状況監視デモ
python demo_market_monitoring_system.py
# 期待される結果:
# - 現在の市場状況（上昇/下降/横ばい）
# - ボラティリティレベル
# - 市場健全性指標

# 市場監視システムのテスト
python -c "
from src.dssms.market_condition_monitor import MarketConditionMonitor
monitor = MarketConditionMonitor()
status = monitor.get_current_market_condition()
print(f'市場状況: {status}')
"
# 期待される結果: 現在の市場分析結果の辞書形式データ
```

### 4. スケジューラーによる自動実行

```powershell
# スケジューラーのテスト
python test_dssms_scheduler.py
# 期待される結果: スケジューラーの動作確認とテスト結果

# スケジューラーの手動実行
python -c "
from src.dssms.dssms_scheduler import DSSMSScheduler
scheduler = DSSMSScheduler()
info = scheduler.get_schedule_info()
print('現在のスケジュール:', info)
"
# 期待される結果: 設定されているスケジュールタスクの一覧
```

### 5. 包括的バックテストの実行

**[WARNING] 実行前の重要な注意事項:**
- UnifiedTrendDetector関連の警告メッセージが表示されますが、システムは正常動作します
- データ取得に時間がかかる場合があります（初回実行時は特に）
- yfinanceライブラリの警告メッセージは通常の動作です

```powershell
# DSSMSバックテスターの直接実行
python "src\dssms\dssms_backtester.py"
# 期待される結果: 
# - システム初期化メッセージ（警告含む）
# - デフォルト設定でのバックテスト結果
# - Excel出力ファイルの生成
# - 詳細レポートファイルの生成

# カスタム期間でのバックテスト
python -c "
from src.dssms.dssms_backtester import DSSMSBacktester
from datetime import datetime, timedelta

# 過去6ヶ月のバックテスト
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

backtester = DSSMSBacktester()
print(f'バックテスト期間: {start_date} ～ {end_date}')
try:
    results = backtester.run_backtest(start_date, end_date)
    print('バックテスト結果:', results)
except Exception as e:
    print(f'実行エラー: {e}')
"
# 期待される結果: 
# - 期間の表示
# - リターン、シャープレシオ、最大ドローダウンなどの結果
# - エラーの場合は具体的なエラー内容
```

### 6. 詳細分析の実行

```powershell
# DSSMSアナライザーによる総合分析
python -c "
from src.dssms.dssms_analyzer import DSSMSAnalyzer
analyzer = DSSMSAnalyzer()
report = analyzer.generate_performance_report()
print('パフォーマンスレポート:')
for key, value in report.items():
    print(f'  {key}: {value}')
"
# 期待される結果: 
# - 各種パフォーマンス指標
# - 分析結果の詳細
# - 推奨事項
```

---

## 設定の変更方法

### 1. メイン設定の変更

#### スクリーニング条件の確認と変更
```powershell
# 現在の設定確認
python -c "
import json
with open('config/dssms/dssms_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
print('=== 現在のスクリーニング設定 ===')
print(f'時価総額閾値: {config.get(\"market_cap_threshold\", \"N/A\"):,}円')
print(f'出来高閾値: {config.get(\"volume_threshold\", \"N/A\"):,}株')
print(f'流動性閾値: {config.get(\"liquidity_threshold\", \"N/A\")}')
"
# 期待される結果: 現在の設定値が整理されて表示される

# 時価総額条件の変更（例：1兆円以上）
python -c "
import json
with open('config/dssms/dssms_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

old_threshold = config.get('market_cap_threshold', 0)
config['market_cap_threshold'] = 1000000000000  # 1兆円

with open('config/dssms/dssms_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f'時価総額閾値変更: {old_threshold:,}円 → {config[\"market_cap_threshold\"]:,}円')
"
# 期待される結果: 変更前後の値が表示され、設定ファイルが更新される

# 変更後のテスト実行
python -c "
from src.dssms.nikkei225_screener import Nikkei225Screener
screener = Nikkei225Screener()
candidates = screener.get_qualified_symbols()
print(f'新しい条件での候補銘柄数: {len(candidates)}銘柄')
if candidates:
    print(f'候補例: {candidates[:5]}')
"
# 期待される結果: 新しい条件での候補銘柄数と例が表示される
```

### 2. ランキング重みの調整

```powershell
# 現在のランキング重み確認
python -c "
import json
with open('config/dssms/ranking_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
print('=== 現在のランキング重み ===')
weights = config.get('weights', {})
for key, value in weights.items():
    print(f'{key}: {value}')
total = sum(weights.values())
print(f'合計: {total}')
"
# 期待される結果: 各指標の重みと合計値（通常は1.0）

# テクニカル重視への変更
python -c "
import json
with open('config/dssms/ranking_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

print('=== ランキング重み変更（テクニカル重視） ===')
print('変更前:', config.get('weights', {}))

# テクニカル重視の設定
config['weights'] = {
    'technical_weight': 0.5,      # テクニカル50%
    'fundamental_weight': 0.25,   # ファンダメンタル25%
    'momentum_weight': 0.15,      # モメンタム15%
    'risk_weight': 0.10          # リスク10%
}

with open('config/dssms/ranking_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('変更後:', config['weights'])
"
# 期待される結果: 変更前後の重み設定が表示される

# 変更後のランキングテスト
python test_dssms_phase2.py
# 期待される結果: 新しい重み設定でのランキングテスト結果
```

### 3. 市場監視パラメータの調整

```powershell
# 市場監視設定の確認
python -c "
import json
with open('config/dssms/market_monitoring_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
print('=== 現在の市場監視設定 ===')
for key, value in config.items():
    print(f'{key}: {value}')
"
# 期待される結果: 市場監視の各種閾値設定が表示される

# より敏感な監視設定への変更
python -c "
import json
with open('config/dssms/market_monitoring_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

print('=== 市場監視設定変更（高感度） ===')
old_settings = config.copy()

# より敏感な設定
config['volatility_warning_level'] = 0.12     # 12%に下げる
config['trend_change_sensitivity'] = 0.03     # 3%に下げる
config['market_stress_threshold'] = 0.15      # 15%に下げる

with open('config/dssms/market_monitoring_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('ボラティリティ警告レベル:', old_settings.get('volatility_warning_level'), '→', config['volatility_warning_level'])
print('トレンド変化感度:', old_settings.get('trend_change_sensitivity'), '→', config['trend_change_sensitivity'])
print('市場ストレス閾値:', old_settings.get('market_stress_threshold'), '→', config['market_stress_threshold'])
"
# 期待される結果: 各設定項目の変更前後の値が表示される
```

### 4. 自動実行スケジュールの設定

```powershell
# 現在のスケジュール確認
python -c "
from src.dssms.dssms_scheduler import DSSMSScheduler
scheduler = DSSMSScheduler()
current_schedule = scheduler.get_schedule_info()
print('=== 現在のスケジュール ===')
if current_schedule:
    for task in current_schedule:
        print(f'- {task}')
else:
    print('スケジュールなし')
"
# 期待される結果: 現在設定されているスケジュールタスクの一覧

# 新しいスケジュール設定
python -c "
from src.dssms.dssms_scheduler import DSSMSScheduler

scheduler = DSSMSScheduler()
print('=== 自動実行スケジュール設定 ===')

# 既存スケジュールのクリア
scheduler.clear_all_schedules()

# 新しいスケジュール追加
scheduler.add_daily_task('morning_screening', '08:30', weekdays_only=True)
scheduler.add_daily_task('market_check', '15:00', weekdays_only=True)
scheduler.add_weekly_task('weekly_analysis', 'friday', '17:30')
scheduler.add_monthly_task('monthly_report', 1, '09:00')  # 毎月1日

# スケジュール保存
scheduler.save_schedule()

print('新しいスケジュールを設定:')
print('- 平日08:30: モーニングスクリーニング')
print('- 平日15:00: マーケットチェック')
print('- 毎週金曜17:30: 週次分析')
print('- 毎月1日09:00: 月次レポート')
"
# 期待される結果: 新しいスケジュールの設定完了メッセージ
```

### 5. スコアリングエンジンの設定

```powershell
# スコアリング設定の確認
python -c "
import json
with open('config/dssms/scoring_engine_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
print('=== 現在のスコアリング設定 ===')
for category, settings in config.items():
    print(f'{category}:')
    if isinstance(settings, dict):
        for key, value in settings.items():
            print(f'  {key}: {value}')
    else:
        print(f'  {settings}')
"
# 期待される結果: スコアリングエンジンの詳細設定が表示される

# スコアリング設定の調整
python -c "
import json
with open('config/dssms/scoring_engine_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

print('=== スコアリング設定調整 ===')

# より保守的なスコアリングに変更
if 'risk_factors' in config:
    config['risk_factors']['volatility_penalty'] = 0.3  # ボラティリティペナルティ増加
    config['risk_factors']['drawdown_penalty'] = 0.25   # ドローダウンペナルティ増加

if 'performance_weights' in config:
    config['performance_weights']['return_weight'] = 0.4    # リターン重み
    config['performance_weights']['stability_weight'] = 0.6  # 安定性重み重視

with open('config/dssms/scoring_engine_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('スコアリング設定を保守的に調整しました')
print('- ボラティリティペナルティ強化')
print('- 安定性重視の重み配分')
"
# 期待される結果: 設定変更の完了と調整内容の説明
```

---

## コマンド一覧

### 基本確認コマンド

```powershell
# 1. システム全体の健全性チェック
python -c "
from src.dssms.dssms_analyzer import DSSMSAnalyzer
analyzer = DSSMSAnalyzer()
health = analyzer.check_system_health()
print('=== システム健全性チェック ===')
for component, status in health.items():
    status_mark = '✓' if status else '✗'
    print(f'{status_mark} {component}: {\"正常\" if status else \"異常\"}')
"
# 期待される結果: 各システムコンポーネントの動作状況（✓/✗マーク付き）

# 2. 日経225銘柄リストの確認
python -c "
from src.dssms.nikkei225_screener import Nikkei225Screener
screener = Nikkei225Screener()
symbols = screener.fetch_nikkei225_symbols()
print(f'=== 日経225銘柄情報 ===')
print(f'総銘柄数: {len(symbols)}')
print(f'先頭10銘柄: {symbols[:10]}')
print(f'末尾5銘柄: {symbols[-5:]}')
"
# 期待される結果: 日経225の総銘柄数（224銘柄）と具体的な銘柄コード例

# 3. データ接続状況の確認
python -c "
from src.dssms.dssms_data_manager import DSSMSDataManager
import yfinance as yf

print('=== データ接続確認 ===')
# yfinance接続テスト
try:
    test_ticker = yf.Ticker('7203.T')  # トヨタ自動車
    info = test_ticker.info
    company_name = info.get('longName', 'Unknown')
    print(f'✓ yfinance接続正常: {company_name}')
except Exception as e:
    print(f'✗ yfinance接続エラー: {str(e)[:50]}...')

# DSSMSデータマネージャーテスト
try:
    dm = DSSMSDataManager()
    print('✓ データマネージャー初期化成功')
except Exception as e:
    print(f'✗ データマネージャーエラー: {str(e)[:50]}...')
"
# 期待される結果: yfinance接続確認（"Toyota Motor Corporation"）とデータマネージャー初期化成功
```

### データ取得・管理コマンド

```powershell
# 1. 特定銘柄のデータ取得
python -c "
from src.dssms.dssms_data_manager import DSSMSDataManager
from datetime import datetime, timedelta

dm = DSSMSDataManager()
symbol = '7203'  # トヨタ自動車
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print(f'=== {symbol} データ取得 ===')
data = dm.fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
if data is not None and len(data) > 0:
    print(f'取得データ: {len(data)}日分')
    start_str = data.index[0].strftime('%Y-%m-%d')
    end_str = data.index[-1].strftime('%Y-%m-%d')
    latest_price = float(data['Close'].iloc[-1])
    print(f'期間: {start_str} ～ {end_str}')
    print(f'最新価格: {latest_price:.2f}円')
else:
    print('データ取得失敗')
"
# 期待される結果: データ取得期間、データ数、最新価格の表示

# 2. 複数銘柄の一括データ取得（簡単なテスト版）
python -c "
import yfinance as yf

symbols = ['7203.T', '6758.T', '9984.T']  # 主要銘柄
print(f'=== {len(symbols)}銘柄 一括データ取得テスト ===')

results = {}
for symbol in symbols:
    try:
        data = yf.download(symbol, period='5d', progress=False)
        if data is not None and len(data) > 0:
            results[symbol] = data
            latest_price = float(data['Close'].iloc[-1])
            print(f'✓ {symbol}: {latest_price:.2f}円')
        else:
            print(f'✗ {symbol}: データなし')
    except Exception as e:
        print(f'✗ {symbol}: エラー - {str(e)[:30]}...')

print(f'\\n成功: {len(results)}/{len(symbols)} 銘柄')
"
# 期待される結果: 各銘柄の価格と取得状況、成功率（例：3/3銘柄成功）
```

### 分析・評価コマンド

```powershell
# 1. ファンダメンタル分析の実行
python -c "
from src.dssms.fundamental_analyzer import FundamentalAnalyzer

analyzer = FundamentalAnalyzer()
symbol = '7203'  # トヨタ自動車
print(f'=== {symbol} ファンダメンタル分析 ===')

try:
    result = analyzer.analyze_symbol(symbol)
    print(f'財務健全性スコア: {result.get(\"health_score\", \"N/A\")}/100')
    print(f'成長性評価: {result.get(\"growth_score\", \"N/A\")}/100')
    print(f'収益性評価: {result.get(\"profitability_score\", \"N/A\")}/100')
    print(f'総合評価: {result.get(\"overall_rating\", \"N/A\")}')
except Exception as e:
    print(f'分析エラー: {e}')
"
# 期待される結果: 各種ファンダメンタル指標のスコア（0-100点）と総合評価

# 2. テクニカル分析（パーフェクトオーダー）
python -c "
from src.dssms.perfect_order_detector import PerfectOrderDetector

detector = PerfectOrderDetector()
symbol = '7203'
print(f'=== {symbol} テクニカル分析 ===')

try:
    result = detector.detect_perfect_order(symbol)
    print(f'パーフェクトオーダー: {\"成立\" if result.get(\"is_perfect_order\", False) else \"非成立\"}')
    print(f'トレンド方向: {result.get(\"trend_direction\", \"N/A\")}')
    print(f'トレンド強度: {result.get(\"trend_strength\", \"N/A\")}/10')
    print(f'信頼度: {result.get(\"confidence\", \"N/A\")}%')
except Exception as e:
    print(f'分析エラー: {e}')
"
# 期待される結果: パーフェクトオーダーの成立状況、トレンド情報、信頼度

# 3. 包括的スコアリング
python -c "
from src.dssms.comprehensive_scoring_engine import ComprehensiveScoringEngine

engine = ComprehensiveScoringEngine()
symbol = '7203'
print(f'=== {symbol} 包括的スコアリング ===')

try:
    score = engine.calculate_comprehensive_score(symbol)
    print(f'総合スコア: {score.get(\"total_score\", \"N/A\")}/100')
    print(f'テクニカルスコア: {score.get(\"technical_score\", \"N/A\")}/100')
    print(f'ファンダメンタルスコア: {score.get(\"fundamental_score\", \"N/A\")}/100')
    print(f'リスクスコア: {score.get(\"risk_score\", \"N/A\")}/100')
    print(f'投資推奨: {score.get(\"recommendation\", \"N/A\")}')
except Exception as e:
    print(f'スコアリングエラー: {e}')
"
# 期待される結果: 各カテゴリーのスコアと最終的な投資推奨度
```

### 市場監視・分析コマンド

```powershell
# 1. 市場状況の総合確認
python -c "
from src.dssms.market_condition_monitor import MarketConditionMonitor
from src.dssms.market_health_indicators import MarketHealthIndicators

print('=== 市場状況総合確認 ===')

# 市場状況監視
monitor = MarketConditionMonitor()
try:
    condition = monitor.get_current_market_condition()
    print(f'市場状況: {condition.get(\"condition\", \"不明\")}')
    print(f'トレンド: {condition.get(\"trend\", \"不明\")}')
    print(f'ボラティリティ: {condition.get(\"volatility\", \"不明\")}')
except Exception as e:
    print(f'監視エラー: {e}')

# 市場健全性指標
try:
    health = MarketHealthIndicators()
    metrics = health.calculate_market_health()
    print(f'\\n健全性スコア: {metrics.get(\"health_score\", \"N/A\")}/100')
    print(f'リスクレベル: {metrics.get(\"risk_level\", \"N/A\")}')
    print(f'投資スタンス: {metrics.get(\"investment_stance\", \"N/A\")}')
except Exception as e:
    print(f'健全性計算エラー: {e}')
"
# 期待される結果: 現在の市場状況、健全性スコア、推奨投資スタンス

# 2. 市場時間の確認
python -c "
from datetime import datetime

print('=== 市場時間情報 ===')

current_time = datetime.now()
print(f'現在時刻: {current_time.strftime(\"%Y-%m-%d %H:%M:%S\")}')

# 基本的な市場時間判定（簡易版）
hour = current_time.hour
weekday = current_time.weekday()  # 0=月曜, 6=日曜

is_market_hours = (9 <= hour < 15) and (weekday < 5)  # 平日9:00-15:00
market_status = '開場時間内' if is_market_hours else '閉場時間'
print(f'市場状況: {market_status}')

# 取引日判定（平日かどうか）
is_trading_day = weekday < 5  # 土日以外
trading_status = 'はい' if is_trading_day else 'いいえ'
print(f'取引日: {trading_status}')
"
# 期待される結果: 現在時刻、市場の開場状況、取引日判定
```

### ランキング・選定コマンド

```powershell
# 1. 階層ランキングシステムの実行
python -c "
from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem

ranking = HierarchicalRankingSystem()
print('=== 階層ランキングシステム ===')

try:
    # トップ10銘柄取得
    top_stocks = ranking.get_top_ranked_stocks(10)
    print(f'トップ10銘柄ランキング:')
    for i, stock_info in enumerate(top_stocks, 1):
        symbol = stock_info.get('symbol', 'N/A')
        score = stock_info.get('score', 'N/A')
        rank_reason = stock_info.get('rank_reason', 'N/A')
        print(f'{i:2d}. {symbol}: スコア{score} ({rank_reason})')
except Exception as e:
    print(f'ランキングエラー: {e}')
"
# 期待される結果: 上位10銘柄のランキング、スコア、選定理由

# 2. スクリーニング結果の詳細確認
python -c "
from src.dssms.nikkei225_screener import Nikkei225Screener

screener = Nikkei225Screener()
print('=== スクリーニング詳細結果 ===')

try:
    # 条件別カウント
    all_symbols = screener.fetch_nikkei225_symbols()  # 修正: get_nikkei225_symbols → fetch_nikkei225_symbols
    filtered = screener.get_filtered_symbols()
    
    print(f'全日経225銘柄: {len(all_symbols)}')
    print(f'条件適合銘柄: {len(filtered)}')
    print(f'適合率: {len(filtered)/len(all_symbols)*100:.1f}%')
    
    # 上位5銘柄の詳細
    print(f'\\n適合銘柄トップ5:')
    for i, symbol in enumerate(filtered[:5], 1):
        print(f'{i}. {symbol}')
        
except Exception as e:
    print(f'スクリーニングエラー: {e}')
"
# 期待される結果: スクリーニング統計情報と上位銘柄の詳細
```

---

## 実行例とサンプル

### サンプル実行1: 日次スクリーニングの完全実行

**目的**: 毎日の投資候補銘柄を効率的に抽出し、詳細分析まで実行

```powershell
# 日次スクリーニング完全版
python -c "
from src.dssms.nikkei225_screener import Nikkei225Screener
from src.dssms.fundamental_analyzer import FundamentalAnalyzer
from src.dssms.perfect_order_detector import PerfectOrderDetector
from datetime import date

print('=' * 50)
print(f'   {date.today()} 日次スクリーニング')
print('=' * 50)

# Step 1: 基本スクリーニング
screener = Nikkei225Screener()
print('\\n1. 基本スクリーニング実行中...')
candidates = screener.get_filtered_symbols()
print(f'   候補銘柄数: {len(candidates)}')

if len(candidates) == 0:
    print('   候補銘柄なし。条件を緩和してください。')
    exit()

# Step 2: 上位10銘柄の詳細分析
print('\\n2. 上位銘柄詳細分析:')
fundamental = FundamentalAnalyzer()
technical = PerfectOrderDetector()

top_candidates = candidates[:10]
final_recommendations = []

for i, symbol in enumerate(top_candidates, 1):
    print(f'\\n   {i:2d}. {symbol} 分析中...')
    
    # ファンダメンタル分析
    try:
        fund_result = fundamental.analyze_symbol(symbol)
        fund_score = fund_result.get('health_score', 0)
    except:
        fund_score = 0
    
    # テクニカル分析
    try:
        tech_result = technical.detect_perfect_order(symbol)
        tech_score = tech_result.get('trend_strength', 0) * 10
        is_perfect = tech_result.get('is_perfect_order', False)
    except:
        tech_score = 0
        is_perfect = False
    
    # 総合スコア計算
    total_score = (fund_score + tech_score) / 2
    
    print(f'      ファンダメンタル: {fund_score:.1f}')
    print(f'      テクニカル: {tech_score:.1f}')
    print(f'      パーフェクトオーダー: {\"✓\" if is_perfect else \"✗\"}')
    print(f'      総合スコア: {total_score:.1f}')
    
    if total_score >= 60:  # 60点以上を推奨
        final_recommendations.append({
            'symbol': symbol,
            'score': total_score,
            'perfect_order': is_perfect
        })

# Step 3: 最終推奨銘柄
print('\\n3. 最終推奨銘柄:')
if final_recommendations:
    final_recommendations.sort(key=lambda x: x['score'], reverse=True)
    for i, rec in enumerate(final_recommendations, 1):
        po_mark = '⭐' if rec['perfect_order'] else ''
        print(f'   {i}. {rec[\"symbol\"]}: {rec[\"score\"]:.1f}点 {po_mark}')
else:
    print('   推奨銘柄なし')

print('\\n=' * 50)
print('   日次スクリーニング完了')
print('=' * 50)
"
```
**期待される結果**: 
- 候補銘柄数の表示
- 上位10銘柄の詳細分析結果
- 最終推奨銘柄リスト（スコア順）
- パーフェクトオーダー成立銘柄には⭐マーク

### サンプル実行2: 特定銘柄の総合評価

**目的**: 投資を検討している銘柄の詳細な総合評価

```powershell
# 特定銘柄の総合評価（例：トヨタ自動車）
python -c "
from src.dssms.fundamental_analyzer import FundamentalAnalyzer
from src.dssms.perfect_order_detector import PerfectOrderDetector
from src.dssms.comprehensive_scoring_engine import ComprehensiveScoringEngine
from src.dssms.dssms_data_manager import DSSMSDataManager
from datetime import datetime, timedelta

symbol = '7203'  # トヨタ自動車
company_name = 'トヨタ自動車'

print('=' * 60)
print(f'   {symbol} ({company_name}) 総合評価レポート')
print('=' * 60)

# 基本情報取得
# yfinanceを直接使用してデータ取得
try:
    import yfinance as yf
    ticker = yf.Ticker(f'{symbol}.T')
    recent_data = ticker.history(period='1mo')
    current_price = float(recent_data['Close'].iloc[-1])
    price_change = ((current_price - float(recent_data['Close'].iloc[0])) / float(recent_data['Close'].iloc[0])) * 100
    print(f'\\n[CHART] 基本情報:')
    print(f'   現在価格: {current_price:,.0f}円')
    print(f'   30日変化率: {price_change:+.2f}%')
except Exception as e:
    print(f'\\n[CHART] 基本情報: データ取得エラー - {e}')

# 1. ファンダメンタル分析
print(f'\\n[MONEY] ファンダメンタル分析:')
fundamental = FundamentalAnalyzer()
try:
    fund_result = fundamental.analyze_symbol(symbol)
    print(f'   財務健全性: {fund_result.get(\"health_score\", \"N/A\")}/100')
    print(f'   成長性評価: {fund_result.get(\"growth_score\", \"N/A\")}/100')
    print(f'   収益性評価: {fund_result.get(\"profitability_score\", \"N/A\")}/100')
    print(f'   配当利回り: {fund_result.get(\"dividend_yield\", \"N/A\")}%')
    
    # 総合判定
    avg_score = (fund_result.get('health_score', 0) + 
                fund_result.get('growth_score', 0) + 
                fund_result.get('profitability_score', 0)) / 3
    if avg_score >= 80:
        fund_rating = '優秀 ⭐⭐⭐'
    elif avg_score >= 60:
        fund_rating = '良好 ⭐⭐'
    elif avg_score >= 40:
        fund_rating = '普通 ⭐'
    else:
        fund_rating = '要注意 [WARNING]'
    print(f'   総合評価: {fund_rating}')
    
except Exception as e:
    print(f'   分析エラー: {e}')

# 2. テクニカル分析
print(f'\\n[UP] テクニカル分析:')
technical = PerfectOrderDetector()
try:
    tech_result = technical.detect_perfect_order(symbol)
    is_perfect = tech_result.get('is_perfect_order', False)
    trend_direction = tech_result.get('trend_direction', 'N/A')
    trend_strength = tech_result.get('trend_strength', 0)
    confidence = tech_result.get('confidence', 0)
    
    print(f'   パーフェクトオーダー: {\"成立 [OK]\" if is_perfect else \"非成立 [ERROR]\"}')
    print(f'   トレンド方向: {trend_direction}')
    print(f'   トレンド強度: {trend_strength}/10')
    print(f'   信頼度: {confidence}%')
    
    # テクニカル判定
    if is_perfect and confidence >= 70:
        tech_rating = '強力 ⭐⭐⭐'
    elif trend_strength >= 6:
        tech_rating = '良好 ⭐⭐'
    elif trend_strength >= 4:
        tech_rating = '普通 ⭐'
    else:
        tech_rating = '弱い [WARNING]'
    print(f'   総合評価: {tech_rating}')
    
except Exception as e:
    print(f'   分析エラー: {e}')

# 3. 包括的スコアリング
print(f'\\n[TARGET] 包括的スコアリング:')
scoring = ComprehensiveScoringEngine()
try:
    score_result = scoring.calculate_comprehensive_score(symbol)
    total_score = score_result.get('total_score', 0)
    technical_score = score_result.get('technical_score', 0)
    fundamental_score = score_result.get('fundamental_score', 0)
    risk_score = score_result.get('risk_score', 0)
    recommendation = score_result.get('recommendation', 'N/A')
    
    print(f'   総合スコア: {total_score}/100')
    print(f'   ├─ テクニカル: {technical_score}/100')
    print(f'   ├─ ファンダメンタル: {fundamental_score}/100')
    print(f'   └─ リスク評価: {risk_score}/100')
    print(f'\\n   投資推奨度: {recommendation}')
    
    # 最終判定
    if total_score >= 80:
        final_rating = '強く推奨 [ROCKET]'
    elif total_score >= 60:
        final_rating = '推奨 👍'
    elif total_score >= 40:
        final_rating = '中立 😐'
    else:
        final_rating = '非推奨 👎'
    
    print(f'   最終判定: {final_rating}')
    
except Exception as e:
    print(f'   スコアリングエラー: {e}')

print('\\n=' * 60)
print('   総合評価レポート完了')
print('=' * 60)
"
```
**期待される結果**:
- 現在価格と30日間の変化率
- ファンダメンタル各指標のスコアと総合評価
- テクニカル分析結果と強度評価
- 包括的スコアリングと最終的な投資推奨度
- 視覚的に分かりやすい評価マーク（⭐、[OK]、[ERROR]等）

### サンプル実行3: 市場状況に応じた戦略切り替え

**目的**: 現在の市場状況を分析し、最適な投資戦略を提案

```powershell
# 市場状況対応戦略システム
python -c "
from src.dssms.market_condition_monitor import MarketConditionMonitor
from src.dssms.market_health_indicators import MarketHealthIndicators
from src.dssms.intelligent_switch_manager import IntelligentSwitchManager
from datetime import datetime

print('=' * 55)
print(f'   市場状況対応戦略システム')
print(f'   実行時刻: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print('=' * 55)

# 1. 現在の市場状況分析
print('\\n🌍 市場状況分析:')
monitor = MarketConditionMonitor()
try:
    market_condition = monitor.get_current_market_condition()
    condition = market_condition.get('condition', '不明')
    trend = market_condition.get('trend', '不明')
    volatility = market_condition.get('volatility', '不明')
    
    print(f'   市場状況: {condition}')
    print(f'   トレンド: {trend}')
    print(f'   ボラティリティ: {volatility}')
    
    # 市場状況の視覚化
    if condition == '上昇':
        condition_icon = '[UP]'
    elif condition == '下降':
        condition_icon = '[DOWN]'
    else:
        condition_icon = '[CHART]'
    
    print(f'   状況表示: {condition_icon} {condition}市場')
    
except Exception as e:
    print(f'   分析エラー: {e}')

# 2. 市場健全性評価
print('\\n💚 市場健全性評価:')
try:
    health = MarketHealthIndicators()
    health_metrics = health.calculate_market_health()
    health_score = health_metrics.get('health_score', 0)
    risk_level = health_metrics.get('risk_level', '不明')
    investment_stance = health_metrics.get('investment_stance', '不明')
    
    print(f'   健全性スコア: {health_score}/100')
    print(f'   リスクレベル: {risk_level}')
    print(f'   推奨投資スタンス: {investment_stance}')
    
    # リスクレベルの視覚化
    if risk_level == '低':
        risk_icon = '🟢'
    elif risk_level == '中':
        risk_icon = '🟡'
    elif risk_level == '高':
        risk_icon = '🔴'
    else:
        risk_icon = '⚪'
    
    print(f'   リスク表示: {risk_icon} {risk_level}リスク')
    
except Exception as e:
    print(f'   評価エラー: {e}')

# 3. インテリジェント戦略切り替え
print('\\n🧠 戦略切り替え判定:')
try:
    switch_manager = IntelligentSwitchManager()
    switch_analysis = switch_manager.analyze_switch_conditions()
    
    current_strategy = switch_analysis.get('current_strategy', '不明')
    recommended_strategy = switch_analysis.get('recommended_strategy', '不明')
    switch_reason = switch_analysis.get('switch_reason', '不明')
    confidence = switch_analysis.get('confidence', 0)
    
    print(f'   現在の戦略: {current_strategy}')
    print(f'   推奨戦略: {recommended_strategy}')
    print(f'   切り替え理由: {switch_reason}')
    print(f'   信頼度: {confidence}%')
    
    # 戦略切り替えの必要性判定
    if current_strategy != recommended_strategy:
        if confidence >= 70:
            switch_recommendation = '即座に切り替え推奨 [ALERT]'
        elif confidence >= 50:
            switch_recommendation = '切り替え検討 [WARNING]'
        else:
            switch_recommendation = '現状維持 [OK]'
    else:
        switch_recommendation = '現状維持 [OK]'
    
    print(f'   判定: {switch_recommendation}')
    
except Exception as e:
    print(f'   切り替え分析エラー: {e}')

# 4. 推奨アクション
print('\\n[TARGET] 推奨アクション:')
try:
    # 市場状況とリスクレベルに基づく推奨
    if health_score >= 70 and volatility == '低':
        print('   [OK] 積極的な投資が可能です')
        print('   [OK] 成長株への投資を検討してください')
        print('   [OK] レバレッジの活用も検討可能です')
    elif health_score >= 50:
        print('   [WARNING]  慎重な投資が推奨されます')
        print('   [WARNING]  安定株中心のポートフォリオを維持してください')
        print('   [WARNING]  リスク管理を強化してください')
    else:
        print('   [ALERT] 防御的なポジションが必要です')
        print('   [ALERT] 現金比率を高めることを検討してください')
        print('   [ALERT] 損切りルールを厳格に適用してください')
        
    # 具体的な銘柄選定指針
    if condition == '上昇' and risk_level == '低':
        print('\\n   銘柄選定指針:')
        print('   [UP] 成長性重視（テック株、新興企業）')
        print('   [UP] モメンタム戦略の活用')
    elif condition == '下降' or risk_level == '高':
        print('\\n   銘柄選定指針:')
        print('   🛡️  ディフェンシブ株（公益、生活必需品）')
        print('   🛡️  高配当株への投資')
    else:
        print('\\n   銘柄選定指針:')
        print('   ⚖️  バランス型ポートフォリオ')
        print('   ⚖️  セクター分散の重視')
        
except Exception as e:
    print(f'   推奨生成エラー: {e}')

print('\\n=' * 55)
print('   市場状況対応戦略システム完了')
print('=' * 55)
"
```
**期待される結果**:
- 現在の市場状況の詳細分析
- 市場健全性スコアとリスクレベル
- 現在戦略vs推奨戦略の比較
- 具体的な投資アクションプラン
- 視覚的なアイコンによる状況表示

---

## 定期運用コマンド

### 日次ルーチン

**目的**: 毎日実行する基本的な市場分析とスクリーニング

```powershell
# 日次市場分析ルーチン（推奨実行時間：市場開始前8:30）
python -c "
print('🌅 === 日次市場分析ルーチン開始 ===')
from datetime import datetime
start_time = datetime.now()
print(f'開始時刻: {start_time.strftime(\"%Y-%m-%d %H:%M:%S\")}')

# 1. システム健全性チェック
print('\\n1️⃣ システム健全性チェック...')
try:
    from src.dssms.dssms_analyzer import DSSMSAnalyzer
    analyzer = DSSMSAnalyzer()
    health = analyzer.check_system_health()
    
    all_ok = all(health.values())
    print(f'   システム状況: {\"[OK] 正常\" if all_ok else \"[WARNING] 警告あり\"}')
    
    if not all_ok:
        print('   警告システム:')
        for component, status in health.items():
            if not status:
                print(f'     [ERROR] {component}')
except Exception as e:
    print(f'   [ERROR] システムチェックエラー: {e}')

# 2. 市場状況確認
print('\\n2️⃣ 市場状況分析...')
try:
    from src.dssms.market_condition_monitor import MarketConditionMonitor
    monitor = MarketConditionMonitor()
    condition = monitor.get_current_market_condition()
    
    market_status = condition.get('condition', '不明')
    print(f'   市場状況: {market_status}')
    print(f'   トレンド: {condition.get(\"trend\", \"不明\")}')
    print(f'   ボラティリティ: {condition.get(\"volatility\", \"不明\")}')
except Exception as e:
    print(f'   [ERROR] 市場分析エラー: {e}')

# 3. 日次スクリーニング
print('\\n3️⃣ 日次スクリーニング実行...')
try:
    from src.dssms.nikkei225_screener import Nikkei225Screener
    screener = Nikkei225Screener()
    candidates = screener.get_qualified_symbols()
    
    print(f'   候補銘柄数: {len(candidates)}')
    if len(candidates) > 0:
        print(f'   トップ5: {candidates[:5]}')
    else:
        print('   [WARNING] 候補銘柄なし - 条件を見直してください')
except Exception as e:
    print(f'   [ERROR] スクリーニングエラー: {e}')

# 4. アラート確認
print('\\n4️⃣ アラート確認...')
try:
    from src.dssms.emergency_detector import EmergencyDetector
    detector = EmergencyDetector()
    alerts = detector.check_daily_alerts()
    
    if alerts:
        print(f'   [ALERT] {len(alerts)}件のアラートがあります:')
        for alert in alerts[:3]:  # 最大3件表示
            print(f'     - {alert}')
    else:
        print('   [OK] アラートなし')
except Exception as e:
    print(f'   [ERROR] アラート確認エラー: {e}')

# 5. 実行時間計測
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()
print(f'\\n⏱️ 実行時間: {duration:.1f}秒')
print('🌅 === 日次ルーチン完了 ===')
"
```
**期待される結果**: システム状況、市場分析、候補銘柄、アラート状況を約30秒以内で確認

### 週次ルーチン

**目的**: 週末に実行するパフォーマンス分析と設定見直し

```powershell
# 週次パフォーマンス分析（推奨実行時間：金曜日夕方）
python -c "
print('[CHART] === 週次パフォーマンス分析開始 ===')
from datetime import datetime, timedelta
from src.dssms.dssms_analyzer import DSSMSAnalyzer
from src.dssms.execution_history import ExecutionHistory

analysis_date = datetime.now()
print(f'分析日: {analysis_date.strftime(\"%Y-%m-%d %H:%M\")}')

# 1. 週次実行統計
print('\\n[UP] 週次実行統計:')
try:
    history = ExecutionHistory()
    week_data = history.get_weekly_summary()
    
    print(f'   実行回数: {week_data.get(\"total_runs\", 0)}回')
    print(f'   成功率: {week_data.get(\"success_rate\", 0):.1f}%')
    print(f'   平均実行時間: {week_data.get(\"avg_duration\", 0):.1f}秒')
    print(f'   エラー回数: {week_data.get(\"error_count\", 0)}回')
    
except Exception as e:
    print(f'   [ERROR] 統計取得エラー: {e}')

# 2. パフォーマンス評価
print('\\n[MONEY] パフォーマンス評価:')
try:
    analyzer = DSSMSAnalyzer()
    performance = analyzer.calculate_weekly_performance()
    
    print(f'   週次リターン: {performance.get(\"weekly_return\", 0):+.2f}%')
    print(f'   月次リターン: {performance.get(\"monthly_return\", 0):+.2f}%')
    print(f'   勝率: {performance.get(\"win_rate\", 0):.1f}%')
    print(f'   シャープレシオ: {performance.get(\"sharpe_ratio\", 0):.2f}')
    
    # パフォーマンス評価
    weekly_return = performance.get('weekly_return', 0)
    if weekly_return > 2:
        perf_rating = '優秀 🏆'
    elif weekly_return > 0:
        perf_rating = '良好 👍'
    elif weekly_return > -2:
        perf_rating = '普通 😐'
    else:
        perf_rating = '要改善 [DOWN]'
    
    print(f'   総合評価: {perf_rating}')
    
except Exception as e:
    print(f'   [ERROR] パフォーマンス計算エラー: {e}')

# 3. 設定最適化提案
print('\\n⚙️ 設定最適化提案:')
try:
    optimization = analyzer.suggest_parameter_optimization()
    
    if optimization.get('suggestions'):
        print('   推奨調整:')
        for suggestion in optimization['suggestions'][:3]:
            print(f'     [IDEA] {suggestion}')
    else:
        print('   [OK] 現在の設定で良好です')
        
    # 来週の推奨戦略
    next_week_strategy = optimization.get('next_week_strategy', '継続')
    print(f'   来週の戦略: {next_week_strategy}')
    
except Exception as e:
    print(f'   [ERROR] 最適化提案エラー: {e}')

# 4. 市場見通し
print('\\n🔮 来週の市場見通し:')
try:
    from src.dssms.market_condition_monitor import MarketConditionMonitor
    monitor = MarketConditionMonitor()
    forecast = monitor.get_weekly_forecast()
    
    print(f'   予想トレンド: {forecast.get(\"trend\", \"不明\")}')
    print(f'   予想ボラティリティ: {forecast.get(\"volatility\", \"不明\")}')
    print(f'   注意事項: {forecast.get(\"warnings\", \"特になし\")}')
    
except Exception as e:
    print(f'   [ERROR] 市場予測エラー: {e}')

print('\\n[CHART] === 週次分析完了 ===')
"
```
**期待される結果**: 週次実行統計、パフォーマンス評価、設定最適化提案、来週の見通し

### 月次ルーチン

**目的**: 月末に実行する包括的なシステム評価と戦略見直し

```powershell
# 月次総合レビュー（推奨実行時間：月末最終営業日）
python -c "
print('🗓️ === 月次総合レビュー開始 ===')
from datetime import datetime
from src.dssms.dssms_analyzer import DSSMSAnalyzer

review_date = datetime.now()
print(f'レビュー日: {review_date.strftime(\"%Y年%m月%d日\")}')

# 1. 月次パフォーマンスサマリー
print('\\n[CHART] 月次パフォーマンスサマリー:')
try:
    analyzer = DSSMSAnalyzer()
    monthly_perf = analyzer.calculate_monthly_performance()
    
    print(f'   月次リターン: {monthly_perf.get(\"monthly_return\", 0):+.2f}%')
    print(f'   年初来リターン: {monthly_perf.get(\"ytd_return\", 0):+.2f}%')
    print(f'   最大ドローダウン: {monthly_perf.get(\"max_drawdown\", 0):.2f}%')
    print(f'   ボラティリティ: {monthly_perf.get(\"volatility\", 0):.2f}%')
    print(f'   シャープレシオ: {monthly_perf.get(\"sharpe_ratio\", 0):.2f}')
    
    # ベンチマーク比較
    benchmark_return = monthly_perf.get('benchmark_return', 0)
    excess_return = monthly_perf.get('monthly_return', 0) - benchmark_return
    print(f'   ベンチマーク比較: {excess_return:+.2f}% (vs TOPIX)')
    
except Exception as e:
    print(f'   [ERROR] パフォーマンス計算エラー: {e}')

# 2. 戦略別パフォーマンス
print('\\n[TARGET] 戦略別パフォーマンス:')
try:
    strategy_performance = analyzer.analyze_strategy_performance()
    
    for strategy, performance in strategy_performance.items():
        return_rate = performance.get('return', 0)
        usage_rate = performance.get('usage_rate', 0)
        print(f'   {strategy}:')
        print(f'     リターン: {return_rate:+.2f}%')
        print(f'     使用率: {usage_rate:.1f}%')
        
except Exception as e:
    print(f'   [ERROR] 戦略分析エラー: {e}')

# 3. システム信頼性評価
print('\\n[TOOL] システム信頼性評価:')
try:
    reliability = analyzer.calculate_system_reliability()
    
    print(f'   稼働率: {reliability.get(\"uptime\", 0):.1f}%')
    print(f'   エラー率: {reliability.get(\"error_rate\", 0):.1f}%')
    print(f'   平均応答時間: {reliability.get(\"avg_response_time\", 0):.1f}秒')
    
    # 信頼性評価
    uptime = reliability.get('uptime', 0)
    if uptime > 99:
        reliability_rating = '優秀 ⭐⭐⭐'
    elif uptime > 95:
        reliability_rating = '良好 ⭐⭐'
    elif uptime > 90:
        reliability_rating = '普通 ⭐'
    else:
        reliability_rating = '要改善 [WARNING]'
    
    print(f'   信頼性評価: {reliability_rating}')
    
except Exception as e:
    print(f'   [ERROR] 信頼性評価エラー: {e}')

# 4. 来月の戦略提案
print('\\n[ROCKET] 来月の戦略提案:')
try:
    next_month_strategy = analyzer.suggest_next_month_strategy()
    
    print(f'   推奨戦略: {next_month_strategy.get(\"strategy\", \"継続\")}')
    print(f'   重点セクター: {next_month_strategy.get(\"focus_sector\", \"バランス型\")}')
    print(f'   リスクレベル: {next_month_strategy.get(\"risk_level\", \"中\")}')
    
    # 具体的な調整提案
    adjustments = next_month_strategy.get('adjustments', [])
    if adjustments:
        print('   推奨調整:')
        for adj in adjustments:
            print(f'     [TOOL] {adj}')
    
except Exception as e:
    print(f'   [ERROR] 戦略提案エラー: {e}')

# 5. レポート生成
print('\\n📄 月次レポート生成:')
try:
    report_path = analyzer.generate_monthly_report()
    print(f'   [OK] レポート生成完了: {report_path}')
    print(f'   [CHART] グラフ、詳細データを含む包括的レポート')
    
except Exception as e:
    print(f'   [ERROR] レポート生成エラー: {e}')

print('\\n🗓️ === 月次総合レビュー完了 ===')
"
```
**期待される結果**: 月次パフォーマンス詳細、戦略別評価、システム信頼性、来月戦略、レポート生成

---

## トラブルシューティング

### 基本的なエラー診断

```powershell
# システム全体診断（エラー発生時に最初に実行）
python -c "
print('[SEARCH] === システム全体診断開始 ===')
import sys
import os
from datetime import datetime

print(f'診断時刻: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')

# 1. Python環境確認
print('\\n🐍 Python環境確認:')
print(f'   Pythonバージョン: {sys.version.split()[0]}')
print(f'   実行パス: {sys.executable}')
print(f'   現在ディレクトリ: {os.getcwd()}')

# 2. 必要ライブラリ確認
print('\\n📚 ライブラリ確認:')
required_libs = ['pandas', 'numpy', 'yfinance', 'scipy', 'matplotlib']
missing_libs = []

for lib in required_libs:
    try:
        __import__(lib)
        print(f'   [OK] {lib}: 利用可能')
    except ImportError:
        print(f'   [ERROR] {lib}: 未インストール')
        missing_libs.append(lib)

if missing_libs:
    print(f'\\n[WARNING] 不足ライブラリ: {missing_libs}')
    print('   インストールコマンド:')
    for lib in missing_libs:
        print(f'     pip install {lib}')

# 3. DSSMSモジュール確認
print('\\n[TARGET] DSSMSモジュール確認:')
dssms_modules = [
    'src.dssms.nikkei225_screener',
    'src.dssms.dssms_analyzer',
    'src.dssms.market_condition_monitor',
    'src.dssms.dssms_backtester'
]

for module in dssms_modules:
    try:
        __import__(module)
        print(f'   [OK] {module}: 読み込み成功')
    except ImportError as e:
        print(f'   [ERROR] {module}: 読み込み失敗 - {str(e)[:50]}...')
    except Exception as e:
        print(f'   [WARNING] {module}: エラー - {str(e)[:50]}...')

# 4. 設定ファイル確認
print('\\n⚙️ 設定ファイル確認:')
config_files = [
    'config/dssms/dssms_config.json',
    'config/dssms/ranking_config.json',
    'config/dssms/market_monitoring_config.json'
]

for config_file in config_files:
    if os.path.exists(config_file):
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                json.load(f)
            print(f'   [OK] {config_file}: 正常')
        except json.JSONDecodeError:
            print(f'   [ERROR] {config_file}: JSON形式エラー')
        except Exception as e:
            print(f'   [WARNING] {config_file}: 読み込みエラー')
    else:
        print(f'   [ERROR] {config_file}: ファイル不存在')

# 5. ネットワーク接続確認
print('\\n🌐 ネットワーク接続確認:')
try:
    import yfinance as yf
    test_ticker = yf.Ticker('7203.T')
    data = test_ticker.history(period='1d')
    if len(data) > 0:
        print('   [OK] yfinanceデータ取得: 成功')
    else:
        print('   [WARNING] yfinanceデータ取得: データなし')
except Exception as e:
    print(f'   [ERROR] yfinanceデータ取得: エラー - {str(e)[:50]}...')

print('\\n[SEARCH] === システム診断完了 ===')
"
```
**期待される結果**: 各コンポーネントの動作状況、不足ライブラリ、設定ファイル状況、ネットワーク接続状況

### データ取得エラーの対処

```powershell
# データ取得問題の詳細診断
python -c "
print('[CHART] === データ取得問題診断 ===')

# 1. yfinance接続詳細テスト
print('\\n🔗 yfinance接続詳細テスト:')
import yfinance as yf
test_symbols = ['7203.T', '6758.T', '9984.T']

for symbol in test_symbols:
    try:
        ticker = yf.Ticker(symbol)
        
        # 基本情報取得テスト
        info = ticker.info
        print(f'   {symbol} 基本情報: [OK] {info.get(\"longName\", \"名前不明\")}')
        
        # 履歴データ取得テスト
        hist = ticker.history(period='5d')
        if len(hist) > 0:
            print(f'   {symbol} 履歴データ: [OK] {len(hist)}日分')
        else:
            print(f'   {symbol} 履歴データ: [WARNING] データなし')
            
    except Exception as e:
        print(f'   {symbol} エラー: [ERROR] {str(e)[:50]}...')

# 2. ネットワーク遅延テスト
print('\\n⏱️ ネットワーク遅延テスト:')
import time
try:
    start_time = time.time()
    test_data = yf.download('7203.T', period='1mo', progress=False)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f'   データ取得時間: {duration:.2f}秒')
    
    if duration > 10:
        print('   [WARNING] 接続が遅い可能性があります')
    elif duration > 5:
        print('   [WARNING] やや遅いです')
    else:
        print('   [OK] 正常な速度です')
        
except Exception as e:
    print(f'   [ERROR] 遅延テストエラー: {e}')

# 3. データ品質確認
print('\\n[TARGET] データ品質確認:')
try:
    symbol = '7203.T'
    data = yf.download(symbol, period='1mo', progress=False)
    
    if len(data) > 0:
        # 欠損データチェック
        missing_data = data.isnull().sum().sum()
        print(f'   欠損データ: {missing_data}件')
        
        # 価格範囲チェック
        price_range = data['Close'].max() - data['Close'].min()
        avg_price = data['Close'].mean()
        volatility = (price_range / avg_price) * 100
        print(f'   価格ボラティリティ: {volatility:.2f}%')
        
        # 出来高チェック
        avg_volume = data['Volume'].mean()
        print(f'   平均出来高: {avg_volume:,.0f}株')
        
        if missing_data == 0 and avg_volume > 0:
            print('   [OK] データ品質: 良好')
        else:
            print('   [WARNING] データ品質: 要注意')
    else:
        print('   [ERROR] データ取得失敗')
        
except Exception as e:
    print(f'   [ERROR] 品質確認エラー: {e}')

print('\\n[CHART] === データ診断完了 ===')
"
```

### 設定ファイル修復

```powershell
# 設定ファイル自動修復
python -c "
print('[TOOL] === 設定ファイル修復開始 ===')
import json
import os

# デフォルト設定の定義
default_configs = {
    'config/dssms/dssms_config.json': {
        'market_cap_threshold': 100000000000,
        'volume_threshold': 1000000,
        'liquidity_threshold': 0.5,
        'financial_health_threshold': 0.6
    },
    'config/dssms/ranking_config.json': {
        'weights': {
            'technical_weight': 0.3,
            'fundamental_weight': 0.3,
            'momentum_weight': 0.2,
            'risk_weight': 0.2
        }
    },
    'config/dssms/market_monitoring_config.json': {
        'volatility_warning_level': 0.2,
        'trend_change_sensitivity': 0.1,
        'market_stress_threshold': 0.25
    }
}

# 設定ファイルの確認と修復
for config_path, default_config in default_configs.items():
    print(f'\\n[SEARCH] {config_path} 確認中...')
    
    # ディレクトリ作成
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    if os.path.exists(config_path):
        try:
            # 既存ファイルの検証
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            
            # 不足キーの補完
            updated = False
            for key, value in default_config.items():
                if key not in existing_config:
                    existing_config[key] = value
                    updated = True
                    print(f'   [OK] 不足キー追加: {key}')
            
            if updated:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_config, f, indent=2, ensure_ascii=False)
                print(f'   [TOOL] 設定ファイル更新完了')
            else:
                print(f'   [OK] 設定ファイル正常')
                
        except json.JSONDecodeError:
            print(f'   [ERROR] JSON形式エラー - デフォルト設定で置換')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f'   [ERROR] 読み込みエラー: {e}')
    else:
        # ファイルが存在しない場合は新規作成
        print(f'   📝 新規作成中...')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print(f'   [OK] デフォルト設定ファイル作成完了')

print('\\n[TOOL] === 設定ファイル修復完了 ===')
"
```

### パフォーマンス最適化

```powershell
# システムパフォーマンス最適化
python -c "
print('⚡ === パフォーマンス最適化開始 ===')
import time
import gc
import os

# 1. メモリ使用量確認
print('\\n🧠 メモリ使用量確認:')
try:
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f'   現在のメモリ使用量: {memory_mb:.1f} MB')
    
    if memory_mb > 500:
        print('   [WARNING] メモリ使用量が多いです')
        print('   [IDEA] ガベージコレクション実行中...')
        gc.collect()
        print('   [OK] メモリクリーンアップ完了')
    else:
        print('   [OK] メモリ使用量正常')
        
except ImportError:
    print('   [WARNING] psutilがインストールされていません')
    print('   [IDEA] インストール: pip install psutil')

# 2. データ取得速度最適化
print('\\n[ROCKET] データ取得速度最適化:')
try:
    # キャッシュディレクトリ作成
    cache_dir = 'cache/data'
    os.makedirs(cache_dir, exist_ok=True)
    print(f'   [OK] キャッシュディレクトリ: {cache_dir}')
    
    # 並列処理設定の最適化
    import concurrent.futures
    import multiprocessing
    
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(cpu_count, 4)  # 最大4並列
    print(f'   💻 CPU数: {cpu_count}, 最適ワーカー数: {optimal_workers}')
    
except Exception as e:
    print(f'   [ERROR] 最適化エラー: {e}')

# 3. 設定最適化提案
print('\\n⚙️ 設定最適化提案:')
optimization_tips = [
    '🔄 データ取得頻度を市場時間に合わせて調整',
    '[CHART] 不要な指標計算を無効化してパフォーマンス向上',
    '💾 結果キャッシュを活用して重複計算を回避',
    '⏰ バックテスト期間を短縮して高速化',
    '[TARGET] 銘柄数を制限して処理時間短縮'
]

for tip in optimization_tips:
    print(f'   {tip}')

# 4. 実行時間ベンチマーク
print('\\n⏱️ 実行時間ベンチマーク:')
try:
    from src.dssms.nikkei225_screener import Nikkei225Screener
    
    start_time = time.time()
    screener = Nikkei225Screener()
    symbols = screener.get_qualified_symbols()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f'   スクリーニング実行時間: {duration:.2f}秒')
    
    if duration > 30:
        print('   [WARNING] 実行が遅いです - 最適化が必要')
    elif duration > 10:
        print('   [WARNING] やや遅いです - 改善の余地あり')
    else:
        print('   [OK] 実行速度良好')
        
except Exception as e:
    print(f'   [ERROR] ベンチマークエラー: {e}')

print('\\n⚡ === パフォーマンス最適化完了 ===')
"
```
**期待される結果**: メモリ使用量、最適化提案、実行時間ベンチマーク、パフォーマンス改善案

---

## 付録：よく使用するコマンド集

### クイックチェックコマンド
```powershell
# 3分で完了する基本チェック
python -c "from src.dssms.dssms_analyzer import DSSMSAnalyzer; print('システム状況:', DSSMSAnalyzer().quick_health_check())"

# 今日の推奨銘柄TOP3
python -c "from src.dssms.nikkei225_screener import Nikkei225Screener; print('推奨TOP3:', Nikkei225Screener().get_qualified_symbols()[:3])"

# 現在の市場状況
python -c "from src.dssms.market_condition_monitor import MarketConditionMonitor; print('市場状況:', MarketConditionMonitor().get_current_market_condition())"
```

### 緊急対応コマンド
```powershell
# システム緊急診断
python -c "
print('[ALERT] === システム緊急診断 ===')
try:
    import yfinance as yf
    print('[OK] yfinance: 正常')
except:
    print('[ERROR] yfinance: エラー')

try:
    import pandas as pd
    print('[OK] pandas: 正常')
except:
    print('[ERROR] pandas: エラー')

try:
    from src.dssms.nikkei225_screener import Nikkei225Screener
    print('[OK] DSSMS: 正常')
except Exception as e:
    print(f'[ERROR] DSSMS: {e}')
"

# 設定ファイル緊急修復
python -c "
import json
import os

print('[TOOL] === 設定ファイル緊急修復 ===')

# 基本設定の修復
basic_config = {
    'market_cap_threshold': 100000000000,
    'volume_threshold': 1000000,
    'liquidity_threshold': 0.5
}

config_path = 'config/dssms/dssms_config.json'
if os.path.exists(config_path):
    print('[OK] 設定ファイル存在確認')
else:
    print('[WARNING] 設定ファイル作成中...')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(basic_config, f, indent=2, ensure_ascii=False)
    print('[OK] 設定ファイル修復完了')
"

# データキャッシュクリア
python -c "
import os
import shutil

print('🗑️ === データキャッシュクリア ===')
cache_dirs = ['cache', '__pycache__', '.pytest_cache']

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f'[OK] {cache_dir} クリア完了')
    else:
        print(f'ℹ️ {cache_dir} 存在しません')

print('[OK] キャッシュクリア完了')
"
```

このマニュアルにより、DSSMS初心者でもシステムを効果的に活用できるようになります。
