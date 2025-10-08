# DSSMS実行環境 銘柄最適化 ロードマップ計画
**Dynamic Stock Selection Management System - Implementation Roadmap**

## [LIST] プロジェクト概要

**目的**: DSSMSバックテストシステムの決定論的動作確保と実取引対応準備  
**期間**: 4週間  
**主要目標**: 
- 実行間差異の80-90%削減
- 完全オフライン実行環境構築
- 銘柄切替ルール最適化
- 実取引準備完了

---

## 🗓️ Week 1: オフラインシステム構築

### Week 1.1: 事前データ取得・キャッシュシステム構築
**期間**: 1-2日  
**担当**: データインフラ  

#### タスク詳細
- [ ] **T1.1.1**: データキャッシュシステム設計・実装
  - 全銘柄・全期間データの一括取得機能
  - 標準化フォーマットでの保存システム
  - データ整合性検証・補完機能
  
- [ ] **T1.1.2**: 高速アクセスインデックス作成
  - 銘柄別・日付別インデックス構築
  - 効率的なデータ読み込み最適化
  
- [ ] **T1.1.3**: データ鮮度管理システム
  - 自動更新スケジューラー
  - データ陳腐化検知機能

**成果物**: `src/dssms/offline_data_manager.py`, `config/dssms/data_cache_config.json`

### Week 1.2: 外部アクセス制御システム実装
**期間**: 1-2日  
**担当**: システム制御  

#### タスク詳細
- [ ] **T1.2.1**: オフライン実行モード実装
  - 実行時外部API呼び出し禁止機能
  - キャッシュデータ専用アクセス制御
  
- [ ] **T1.2.2**: ネットワーク依存性排除
  - yfinance等外部依存の完全制御
  - フォールバック機能の実装

**成果物**: `src/dssms/offline_execution_controller.py`

### Week 1.3: 決定論的モード強化
**期間**: 1日  
**担当**: コア機能  

#### タスク詳細
- [ ] **T1.3.1**: 既存決定論的モードの検証・強化
  - ランダムシード固定の完全実装
  - 設定ファイルベースの制御強化

**成果物**: 強化された決定論的モード

---

## 🗓️ Week 2: 切替ルール最適化

### Week 2.1: 既存最適化プログラム調査
**期間**: 1日  
**担当**: システム分析  

#### タスク詳細
- [ ] **T2.1.1**: プロジェクト内最適化モジュール検索
  - `analysis/`配下の最適化機能調査
  - `config/optimized_parameters.py`の活用可能性
  - `config/weight_learning_config/`の応用検討
  
- [ ] **T2.1.2**: 既存パラメータ最適化システムの評価
  - `bayesian-optimization`等の利用状況確認
  - DSSMS切替ルールへの適用可能性

**成果物**: 既存最適化システム調査レポート

### Week 2.2: 切替閾値最適化システム実装
**期間**: 2日  
**担当**: アルゴリズム最適化  

#### タスク詳細
- [ ] **T2.2.1**: 最小切替スコア差設定機能
  - 設定可能な閾値システム実装
  - 動的閾値調整機能
  
- [ ] **T2.2.2**: 同点処理ルール実装
  - 決定論的タイブレーク機能
  - 銘柄コード順等の固定優先順位

**成果物**: `src/dssms/switch_threshold_optimizer.py`

### Week 2.3: 保有期間最適化・スコア平滑化
**期間**: 2日  
**担当**: 戦略最適化  

#### タスク詳細
- [ ] **T2.3.1**: 動的最小保有期間実装
  - 市場ボラティリティ連動保有期間
  - 適応的パラメータ調整機能
  
- [ ] **T2.3.2**: スコア平滑化システム
  - 移動平均・EMA適用機能
  - 急激なスコア変動の抑制

**成果物**: `src/dssms/holding_period_optimizer.py`, `src/dssms/score_smoothing_engine.py`

---

## 🗓️ Week 3: 統合テスト・検証

### Week 3.1: システム統合テスト
**期間**: 2日  
**担当**: 品質保証  

#### タスク詳細
- [ ] **T3.1.1**: オフライン+最適化システム統合
  - 全コンポーネントの連携テスト
  - 設定ファイル整合性確認
  
- [ ] **T3.1.2**: 複数期間での一貫性検証
  - 短期・中期・長期期間での差異測定
  - 10銘柄環境での安定性確認

**成果物**: 統合テスト結果レポート

### Week 3.2: パフォーマンス検証
**期間**: 2日  
**担当**: 性能評価  

#### タスク詳細
- [ ] **T3.2.1**: 差異削減効果測定
  - Before/After比較分析
  - 目標80-90%削減の達成確認
  
- [ ] **T3.2.2**: 実行速度・安定性評価
  - システム負荷テスト
  - メモリ使用量最適化

**成果物**: パフォーマンス検証レポート

### Week 3.3: アウトオブサンプル検証
**期間**: 1日  
**担当**: 検証・妥当性  

#### タスク詳細
- [ ] **T3.3.1**: 未使用期間データでの検証
  - 過学習リスクの評価
  - ロバストネステスト実施

**成果物**: アウトオブサンプル検証結果

---

## 🗓️ Week 4: 実取引準備・段階検証

### Week 4.1: 実取引ギャップ分析
**期間**: 2日  
**担当**: 実装準備  

#### タスク詳細
- [ ] **T4.1.1**: 実行コスト精密モデリング
  - 取引手数料・スプレッド考慮
  - スリッページ・遅延シミュレーション強化
  
- [ ] **T4.1.2**: リアルタイムデータ連携準備
  - オフライン⇔オンライン切替機能
  - 段階的実取引移行システム

**成果物**: 実取引準備システム

### Week 4.2: 致命的欠陥リスク回避システム
**期間**: 1日  
**担当**: リスク管理  

#### タスク詳細
- [ ] **T4.2.1**: リスク監視システム実装
  - 過学習検知機能
  - 市場適応性監視
  - 実行現実ギャップ警告

**成果物**: `src/dssms/risk_monitoring_system.py`

### Week 4.3: 段階的実取引検証準備
**期間**: 2日  
**担当**: 運用準備  

#### タスク詳細
- [ ] **T4.3.1**: 少額実取引テスト環境構築
  - デモ取引システム連携
  - 実績追跡・比較システム
  
- [ ] **T4.3.2**: 運用監視ダッシュボード
  - リアルタイム性能監視
  - 差異・異常検知アラート

**成果物**: 実取引検証環境

---

## [CHART] 成功指標・検証基準

### 定量的目標
- **差異削減率**: > 80%（目標: 90%）
- **実行一貫性**: > 95%
- **パフォーマンス維持**: > 90%（最適化前比）
- **実行速度向上**: 10-15%削減
- **実取引ギャップ**: < 5%

### 定性的目標
- 完全オフライン実行の実現
- 設定ベースでの柔軟な制御
- 実取引適用準備完了
- 長期運用安定性確保

---

## 🛠️ 実装環境・ツール

### 開発環境
- **言語**: Python 3.8+
- **主要ライブラリ**: pandas, numpy, yfinance, scipy, openpyxl
- **設定管理**: JSON形式設定ファイル
- **ログ管理**: `config/logger_config.py`

### ファイル構造
```
src/dssms/
├── offline_data_manager.py          # データキャッシュシステム
├── offline_execution_controller.py   # オフライン実行制御
├── switch_threshold_optimizer.py     # 切替閾値最適化
├── holding_period_optimizer.py       # 保有期間最適化
├── score_smoothing_engine.py         # スコア平滑化
└── risk_monitoring_system.py         # リスク監視

config/dssms/
├── data_cache_config.json            # データキャッシュ設定
├── offline_execution_config.json     # オフライン実行設定
├── switch_optimization_config.json   # 切替最適化設定
└── risk_monitoring_config.json       # リスク監視設定
```

---

## [ROCKET] 実装開始準備

### 即座開始可能タスク
1. **T1.1.1**: データキャッシュシステム設計・実装
2. **T2.1.1**: 既存最適化プログラム調査
3. **T1.3.1**: 決定論的モード強化

### 依存関係
- Week 2の最適化はWeek 1のオフライン化完了後
- Week 3の統合テストはWeek 1-2完了後
- Week 4の実取引準備はWeek 1-3検証完了後

---

## 📝 進捗管理・報告

### 週次チェックポイント
- **Week 1 end**: オフライン実行環境動作確認
- **Week 2 end**: 切替ルール最適化効果測定
- **Week 3 end**: 統合システム安定性確認
- **Week 4 end**: 実取引準備完了判定

### 成果物管理
- 各週終了時に実装コード・設定ファイル・テストレポート提出
- 問題発生時の早期エスカレーション体制
- コードレビュー・品質チェック実施

---

**最終目標**: 信頼性の高い決定論的DSSMSシステムによる実取引成功準備完了

---

# [ALERT] 緊急問題調査・解決ロードマップ
**Main.py実行エラー & DSSMS出力システム不整合問題**

## [CHART] 問題現状

### 🔴 主要問題
1. **main.py実行エラー**: `ModuleNotFoundError: No module named 'output.simple_excel_exporter'`
2. **DSSMS出力不整合**: Excelレポートとテキストレポートのデータ乖離
3. **Excel出力異常**: サマリーシートの値が0または空白
4. **取引履歴不整合**: 取引回数・損益計算の相違

### [UP] 影響範囲
- **即座の影響**: main.pyによる統合バックテスト実行不可
- **中期的影響**: DSSMSシステムの信頼性低下
- **長期的影響**: 実取引準備の大幅遅延

---

## [SEARCH] Phase 1: 問題把握・原因特定

### Phase 1.1: モジュール構造調査
**目的**: インポートエラーの根本原因特定  
**期間**: 即時実行  

#### Task 1.1.1: 出力関連モジュール存在確認
```powershell
# 実行コマンド
python -c "
import os
import glob

print('=== 出力関連ファイル構造調査 ===')
# outputディレクトリの全ファイル確認
output_files = glob.glob('output/*.py')
print('📁 output/内のPythonファイル:')
for f in output_files:
    print(f'  {f}')

# simple_excel_exporterの存在確認
simple_excel_files = glob.glob('**/simple_excel_exporter.py', recursive=True)
print('\\n[SEARCH] simple_excel_exporter.pyの場所:')
for f in simple_excel_files:
    print(f'  {f}')

# __init__.pyの存在確認
init_files = glob.glob('output/__init__.py')
print('\\n[LIST] output/__init__.py:')
print(f'  存在: {\"あり\" if init_files else \"なし\"}')
"
```
**期待結果**: 欠損ファイルの特定、モジュール構造の把握

#### Task 1.1.2: main.pyインポート依存関係調査
```powershell
# main.pyの詳細インポート解析
python -c "
import ast
import os

print('=== main.py インポート依存関係調査 ===')

try:
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f'import {alias.name}')
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'from {module} import {alias.name}')
    
    print('[LIST] main.pyの全インポート文:')
    for imp in imports:
        print(f'  {imp}')
        
    # 問題のあるインポートの特定
    problem_imports = [imp for imp in imports if 'output' in imp and 'simple_excel_exporter' in imp]
    print('\\n[ERROR] 問題のあるインポート:')
    for imp in problem_imports:
        print(f'  {imp}')
        
except Exception as e:
    print(f'[ERROR] エラー: {e}')
"
```

### Phase 1.2: インポートパス詳細調査
**目的**: Pythonモジュール解決メカニズムの問題特定  

#### Task 1.2.1: システムパス・環境調査
```powershell
python -c "
import sys
import os

print('=== インポートパス調査 ===')
print('現在のワーキングディレクトリ:', os.getcwd())
print('\\nPythonパス:')
for i, path in enumerate(sys.path):
    print(f'  {i}: {path}')

print('\\n=== モジュール検索テスト ===')
try:
    import output
    print('[OK] outputモジュール: インポート成功')
    print(f'   パス: {output.__file__}')
    print(f'   内容: {dir(output)}')
except ImportError as e:
    print(f'[ERROR] outputモジュール: {e}')

try:
    from output import simple_excel_exporter
    print('[OK] simple_excel_exporter: インポート成功')
except ImportError as e:
    print(f'[ERROR] simple_excel_exporter: {e}')
    
try:
    import output.simple_excel_exporter
    print('[OK] output.simple_excel_exporter (直接): インポート成功')
except ImportError as e:
    print(f'[ERROR] output.simple_excel_exporter (直接): {e}')
"
```

### Phase 1.3: DSSMS出力システム調査
**目的**: 出力システムの内部構造と動作フロー把握  

#### Task 1.3.1: DSSMSバックテスター出力機能調査
```powershell
python -c "
import inspect
import sys
sys.path.append('src')

print('=== DSSMS出力システム調査 ===')
try:
    from dssms.dssms_backtester import DSSMSBacktester
    
    # クラスのメソッド一覧
    methods = [method for method in dir(DSSMSBacktester) if not method.startswith('_')]
    print('[TOOL] DSSMSBacktesterの公開メソッド:')
    for method in methods:
        print(f'  - {method}')
    
    # 出力関連メソッドの特定
    output_methods = [m for m in methods if any(keyword in m.lower() for keyword in ['save', 'export', 'output', 'write', 'generate'])]
    print('\\n📤 出力関連メソッド:')
    for method in output_methods:
        print(f'  - {method}')
        
    # 実際のインスタンス作成テスト
    print('\\n[TEST] インスタンス作成テスト:')
    instance = DSSMSBacktester()
    print('[OK] DSSMSBacktesterインスタンス作成成功')
    
except ImportError as e:
    print(f'[ERROR] DSSMSBacktesterインポートエラー: {e}')
except Exception as e:
    print(f'[ERROR] インスタンス作成エラー: {e}')
"
```

#### Task 1.3.2: 既存出力ファイル分析
```powershell
python -c "
import glob
import os
from datetime import datetime

print('=== 既存出力ファイル分析 ===')

# Excel出力ファイル
excel_files = glob.glob('**/*.xlsx', recursive=True)
excel_files.sort(key=os.path.getmtime, reverse=True)
print('[CHART] 最新のExcelファイル (上位5件):')
for i, f in enumerate(excel_files[:5]):
    mtime = datetime.fromtimestamp(os.path.getmtime(f))
    size = os.path.getsize(f)
    print(f'  {i+1}. {f}')
    print(f'     更新: {mtime.strftime(\"%Y-%m-%d %H:%M:%S\")} | サイズ: {size:,} bytes')

# レポートファイル
report_files = glob.glob('**/*report*.txt', recursive=True)
report_files.sort(key=os.path.getmtime, reverse=True)
print('\\n[LIST] 最新のレポートファイル (上位5件):')
for i, f in enumerate(report_files[:5]):
    mtime = datetime.fromtimestamp(os.path.getmtime(f))
    size = os.path.getsize(f)
    print(f'  {i+1}. {f}')
    print(f'     更新: {mtime.strftime(\"%Y-%m-%d %H:%M:%S\")} | サイズ: {size:,} bytes')

# DSSMSファイルのフィルタリング
dssms_files = [f for f in excel_files + report_files if 'dssms' in f.lower()]
print(f'\\n[TARGET] DSSMS関連ファイル数: {len(dssms_files)}')
"
```

### Phase 1.4: データ不整合詳細調査
**目的**: Excel出力とテキストレポートの乖離原因特定  

#### Task 1.4.1: 最新出力ファイル内容比較
```powershell
python -c "
import pandas as pd
import glob
import os
import json

print('=== データ不整合調査 ===')

# 最新のExcelファイルを特定
excel_files = glob.glob('**/*dssms*.xlsx', recursive=True)
if excel_files:
    latest_excel = max(excel_files, key=os.path.getmtime)
    print(f'[CHART] 最新Excelファイル: {latest_excel}')
    
    try:
        # Excelファイルのシート一覧
        xls = pd.ExcelFile(latest_excel)
        print(f'   シート一覧: {xls.sheet_names}')
        
        # 各シートの基本情報
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(latest_excel, sheet_name=sheet)
                print(f'\\n[LIST] [{sheet}] シート:')
                print(f'   形状: {df.shape}')
                print(f'   列名: {list(df.columns)[:10]}')  # 最初の10列
                if not df.empty:
                    print(f'   先頭3行:')
                    print(df.head(3).to_string())
            except Exception as e:
                print(f'   [ERROR] {sheet}シート読み込みエラー: {e}')
                
    except Exception as e:
        print(f'   [ERROR] Excel読み込みエラー: {e}')
else:
    print('[ERROR] DSSMSのExcelファイルが見つかりません')

# 最新のレポートファイル内容確認
report_files = glob.glob('**/*dssms*report*.txt', recursive=True)
if report_files:
    latest_report = max(report_files, key=os.path.getmtime)
    print(f'\\n[LIST] 最新レポートファイル: {latest_report}')
    
    try:
        with open(latest_report, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\\n')
            
        # 重要な数値を抽出
        key_values = {}
        for line in lines:
            if '総リターン:' in line:
                key_values['総リターン'] = line.strip()
            elif '最終ポートフォリオ価値:' in line:
                key_values['最終価値'] = line.strip()
            elif '銘柄切替回数:' in line:
                key_values['切替回数'] = line.strip()
        
        print('   [CHART] 重要指標:')
        for key, value in key_values.items():
            print(f'     {key}: {value}')
            
    except Exception as e:
        print(f'   [ERROR] レポート読み込みエラー: {e}')
else:
    print('[ERROR] DSSMSのレポートファイルが見つかりません')
"
```

### Phase 1.5: main.py実行環境調査
**目的**: main.py実行時の詳細エラー情報取得  

#### Task 1.5.1: main.py段階的実行テスト
```powershell
# 段階的インポートテスト
python -c "
print('=== main.py 段階的実行テスト ===')

import sys
import traceback

# Step 1: 基本インポート
try:
    print('Step 1: 基本ライブラリインポート')
    import pandas as pd
    import numpy as np
    import logging
    print('[OK] 基本ライブラリ: OK')
except Exception as e:
    print(f'[ERROR] 基本ライブラリ: {e}')

# Step 2: プロジェクト設定インポート
try:
    print('\\nStep 2: プロジェクト設定インポート')
    from config.logger_config import setup_logger
    from config.optimized_parameters import get_optimized_parameters
    print('[OK] プロジェクト設定: OK')
except Exception as e:
    print(f'[ERROR] プロジェクト設定: {e}')
    traceback.print_exc()

# Step 3: 統合システムインポート
try:
    print('\\nStep 3: 統合システムインポート')
    from config.multi_strategy_manager import MultiStrategyManager
    print('[OK] 統合システム: OK')
except Exception as e:
    print(f'[ERROR] 統合システム: {e}')

# Step 4: 問題のある出力モジュールインポート
try:
    print('\\nStep 4: 出力モジュールインポート')
    from output.simple_simulation_handler import simulate_and_save
    print('[OK] 出力モジュール: OK')
except Exception as e:
    print(f'[ERROR] 出力モジュール: {e}')
    traceback.print_exc()
"
```

---

## 🛠️ Phase 2: 解決策設計・実装

### Phase 2.1: 緊急修復 - モジュール構造修正
**目的**: main.py実行エラーの即座解決  
**優先度**: 🔴 最高  

#### Task 2.1.1: 欠損ファイル作成
- **simple_excel_exporter.py**: 基本Excel出力機能
- **__init__.py修正**: 適切なモジュール公開
- **インポートパス修正**: main.pyの依存関係正規化

#### Task 2.1.2: 暫定出力システム構築
- **統合出力ハンドラー**: 一時的な出力統合機能
- **エラーハンドリング強化**: 堅牢な例外処理
- **フォールバック機能**: 出力失敗時の代替処理

### Phase 2.2: DSSMS出力システム再構築
**目的**: データ整合性の確保と信頼性向上  
**優先度**: 🟡 高  

#### Task 2.2.1: 統一データ収集システム
- **単一データソース**: 一元化されたデータ管理
- **計算ロジック統一**: Excel・テキスト出力の同一計算基盤
- **検証システム**: 出力データの自動整合性チェック

#### Task 2.2.2: 多形式出力エンジン
- **テンプレートシステム**: 柔軟な出力フォーマット対応
- **動的レポート生成**: 設定ベースのレポート構成
- **品質保証**: 出力前データ検証・承認フロー

### Phase 2.3: 長期安定化
**目的**: 将来的な拡張性と保守性確保  
**優先度**: 🟢 中  

#### Task 2.3.1: アーキテクチャ最適化
- **依存関係整理**: クリーンなモジュール構造
- **設定外部化**: 柔軟な設定管理システム
- **テスト自動化**: 継続的品質保証

---

## 📅 実行スケジュール

### [ALERT] 即時実行 (Phase 1)
**今日実行**: 全調査タスクの順次実行  
**所要時間**: 2-3時間  
**成果物**: 問題調査レポート・根本原因特定

### ⚡ 緊急修復 (Phase 2.1)
**明日実行**: モジュール構造修正・main.py実行復旧  
**所要時間**: 4-6時間  
**成果物**: 動作するmain.py・基本出力機能

### [TOOL] システム再構築 (Phase 2.2-2.3)
**今週内**: 出力システム全面改修・品質向上  
**所要時間**: 2-3日  
**成果物**: 信頼性の高い統合出力システム

---

## [OK] 成功基準

### Phase 1 完了基準
- [ ] 全問題の根本原因特定完了
- [ ] 修復計画の詳細設計完了
- [ ] 影響範囲・リスク評価完了

### Phase 2.1 完了基準
- [ ] main.py正常実行確認
- [ ] 基本的な出力機能復旧
- [ ] エラーハンドリング強化

### Phase 2.2-2.3 完了基準
- [ ] Excel・テキストレポートの完全整合性
- [ ] 10回連続実行での同一結果保証
- [ ] 包括的テストスイートによる品質確認

---

**注意**: この緊急対応ロードマップは既存の長期計画と並行実行し、システムの安定性を最優先として進行する。