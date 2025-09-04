# 🚨 DSSMS出力システム問題調査・解決計画
**Dynamic Stock Selection & Management System - Output Problem Analysis & Resolution**

## 📊 問題現状サマリー

### 🔴 緊急度: 最高
**実行日**: 2025年9月4日  
**状況**: main.py実行不可、DSSMS出力システム全面的不整合

### 主要問題一覧
1. **main.py実行エラー**: `ModuleNotFoundError: No module named 'output.simple_excel_exporter'`
2. **DSSMS出力データ不整合**: Excel vs テキストレポートの値乖離
3. **Excel出力異常**: サマリーシートの重要値が0または空白
4. **取引履歴データ異常**: 取引回数・損益計算の相違

### 📈 影響度分析
- **即座の影響**: 統合バックテストシステム完全停止
- **短期影響**: DSSMSシステム信頼性失墜
- **長期影響**: 実取引準備プロジェクト大幅遅延

---

## 🔍 Phase 1: 問題把握・原因特定フェーズ

### Phase 1.1: モジュール構造・依存関係調査
**目的**: main.py実行エラーの根本原因特定  
**優先度**: 🔴 最高  
**所要時間**: 30分  

#### Task 1.1.1: 出力関連モジュール存在確認
**実行コマンド**:
```powershell
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
print('\n🔍 simple_excel_exporter.pyの場所:')
for f in simple_excel_files:
    print(f'  {f}')

# __init__.pyの存在確認
init_files = glob.glob('output/__init__.py')
print('\n📋 output/__init__.py:')
print(f'  存在: {\"あり\" if init_files else \"なし\"}')

# outputディレクトリ内の全ファイル詳細
print('\n📂 outputディレクトリの詳細:')
if os.path.exists('output'):
    for root, dirs, files in os.walk('output'):
        level = root.replace('output', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
else:
    print('  ❌ outputディレクトリが存在しません')
"
```

**期待結果**: 
- 欠損ファイルの特定
- ディレクトリ構造の把握
- `simple_excel_exporter.py`の場所確認

#### Task 1.1.2: main.pyインポート依存関係詳細調査
**実行コマンド**:
```powershell
python -c "
import ast
import os

print('=== main.py インポート依存関係調査 ===')

try:
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    imports = []
    import_lines = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_stmt = f'import {alias.name}'
                imports.append(import_stmt)
                import_lines[import_stmt] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                import_stmt = f'from {module} import {alias.name}'
                imports.append(import_stmt)
                import_lines[import_stmt] = node.lineno
    
    print('📋 main.pyの全インポート文:')
    for imp in sorted(imports):
        line_no = import_lines[imp]
        print(f'  行{line_no:3d}: {imp}')
        
    # 問題のあるインポートの特定
    problem_imports = [imp for imp in imports if 'output' in imp and 'simple' in imp]
    print('\n❌ 問題のあるインポート:')
    for imp in problem_imports:
        line_no = import_lines[imp]
        print(f'  行{line_no:3d}: {imp}')
        
    # output関連の全インポート
    output_imports = [imp for imp in imports if 'output' in imp]
    print('\n📤 output関連の全インポート:')
    for imp in output_imports:
        line_no = import_lines[imp]
        print(f'  行{line_no:3d}: {imp}')
        
except Exception as e:
    print(f'❌ エラー: {e}')
    import traceback
    traceback.print_exc()
"
```

**期待結果**: 
- 問題インポート文の特定
- 行番号付きでの詳細把握
- 修正対象の明確化

#### Task 1.1.3: モジュール解決パス調査
**実行コマンド**:
```powershell
python -c "
import sys
import os

print('=== Python モジュール解決パス調査 ===')
print('現在のワーキングディレクトリ:', os.getcwd())
print('\nPython実行パス:', sys.executable)
print('\nPythonバージョン:', sys.version)

print('\n📁 sys.path (モジュール検索パス):')
for i, path in enumerate(sys.path):
    exists = '✅' if os.path.exists(path) else '❌'
    print(f'  {i:2d}: {exists} {path}')

print('\n🧪 outputモジュール段階的インポートテスト:')

# Step 1: outputディレクトリの存在確認
if os.path.exists('output'):
    print('✅ Step 1: outputディレクトリ存在確認 - OK')
    
    # Step 2: __init__.pyの確認
    if os.path.exists('output/__init__.py'):
        print('✅ Step 2: output/__init__.py存在確認 - OK')
    else:
        print('❌ Step 2: output/__init__.py存在確認 - NG')
        
    # Step 3: outputモジュールインポートテスト
    try:
        import output
        print('✅ Step 3: outputモジュールインポート - OK')
        print(f'   パス: {output.__file__ if hasattr(output, \"__file__\") else \"不明\"}')
        print(f'   属性: {[attr for attr in dir(output) if not attr.startswith(\"_\")]}')
    except ImportError as e:
        print(f'❌ Step 3: outputモジュールインポート - NG: {e}')
        
    # Step 4: simple_excel_exporterファイル存在確認
    simple_excel_path = 'output/simple_excel_exporter.py'
    if os.path.exists(simple_excel_path):
        print(f'✅ Step 4: {simple_excel_path}存在確認 - OK')
        
        # Step 5: simple_excel_exporterインポートテスト
        try:
            from output import simple_excel_exporter
            print('✅ Step 5: simple_excel_exporterインポート - OK')
        except ImportError as e:
            print(f'❌ Step 5: simple_excel_exporterインポート - NG: {e}')
    else:
        print(f'❌ Step 4: {simple_excel_path}存在確認 - NG')
        
else:
    print('❌ Step 1: outputディレクトリ存在確認 - NG')
"
```

### Phase 1.2: DSSMS出力システム内部構造調査
**目的**: 出力データ不整合の根本原因特定  
**優先度**: 🟡 高  
**所要時間**: 45分  

#### Task 1.2.1: DSSMSバックテスター出力機能詳細調査
**実行コマンド**:
```powershell
python -c "
import inspect
import sys
import os
sys.path.append('src')

print('=== DSSMS出力システム詳細調査 ===')

try:
    from dssms.dssms_backtester import DSSMSBacktester
    
    print('✅ DSSMSBacktesterインポート成功')
    
    # クラス詳細情報
    print('\n🔧 クラス基本情報:')
    print(f'   モジュール: {DSSMSBacktester.__module__}')
    print(f'   ファイル: {inspect.getfile(DSSMSBacktester)}')
    
    # 全メソッド一覧
    methods = [method for method in dir(DSSMSBacktester) if not method.startswith('_')]
    print(f'\n📋 公開メソッド ({len(methods)}個):')
    for method in sorted(methods):
        method_obj = getattr(DSSMSBacktester, method)
        if callable(method_obj):
            print(f'  📝 {method}()')
        else:
            print(f'  📊 {method} (属性)')
    
    # 出力関連メソッドの詳細分析
    output_keywords = ['save', 'export', 'output', 'write', 'generate', 'report', 'excel', 'file']
    output_methods = []
    for method in methods:
        if any(keyword in method.lower() for keyword in output_keywords):
            output_methods.append(method)
    
    print(f'\n📤 出力関連メソッド ({len(output_methods)}個):')
    for method in output_methods:
        method_obj = getattr(DSSMSBacktester, method)
        if callable(method_obj):
            try:
                sig = inspect.signature(method_obj)
                print(f'  🔧 {method}{sig}')
            except:
                print(f'  🔧 {method}() - シグネチャ取得失敗')
        
    # インスタンス作成テスト
    print('\n🧪 インスタンス作成テスト:')
    try:
        instance = DSSMSBacktester()
        print('✅ DSSMSBacktesterインスタンス作成成功')
        
        # 主要属性の確認
        key_attrs = ['config', 'data', 'results', 'portfolio']
        print('   主要属性:')
        for attr in key_attrs:
            if hasattr(instance, attr):
                print(f'     ✅ {attr}: 存在')
            else:
                print(f'     ❌ {attr}: 不存在')
                
    except Exception as e:
        print(f'❌ インスタンス作成エラー: {e}')
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f'❌ DSSMSBacktesterインポートエラー: {e}')
    import traceback
    traceback.print_exc()
"
```

#### Task 1.2.2: 既存出力ファイル詳細分析
**実行コマンド**:
```powershell
python -c "
import glob
import os
import pandas as pd
from datetime import datetime

print('=== 既存出力ファイル詳細分析 ===')

# 全出力ファイルの収集
all_files = {
    'excel': glob.glob('**/*.xlsx', recursive=True),
    'text': glob.glob('**/*.txt', recursive=True),
    'json': glob.glob('**/*.json', recursive=True),
    'csv': glob.glob('**/*.csv', recursive=True)
}

# ファイル種別ごとの統計
print('📊 出力ファイル統計:')
for file_type, files in all_files.items():
    count = len(files)
    if count > 0:
        latest = max(files, key=os.path.getmtime)
        latest_time = datetime.fromtimestamp(os.path.getmtime(latest))
        print(f'  {file_type.upper():5s}: {count:3d}個 (最新: {latest_time.strftime(\"%m/%d %H:%M\")})')
    else:
        print(f'  {file_type.upper():5s}: {count:3d}個')

# DSSMS関連ファイルの特定
dssms_files = []
for file_type, files in all_files.items():
    for file in files:
        if 'dssms' in file.lower():
            dssms_files.append((file_type, file))

print(f'\n🎯 DSSMS関連ファイル ({len(dssms_files)}個):')
dssms_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)

for i, (file_type, file_path) in enumerate(dssms_files[:10]):  # 最新10件
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    size = os.path.getsize(file_path)
    print(f'  {i+1:2d}. [{file_type.upper()}] {file_path}')
    print(f'      更新: {mtime.strftime(\"%Y-%m-%d %H:%M:%S\")} | サイズ: {size:,} bytes')

# 最新のExcelファイル詳細分析
excel_dssms = [f for _, f in dssms_files if f.endswith('.xlsx')]
if excel_dssms:
    latest_excel = excel_dssms[0]  # 既にソート済み
    print(f'\n📊 最新Excel詳細分析: {latest_excel}')
    
    try:
        xls = pd.ExcelFile(latest_excel)
        print(f'   シート数: {len(xls.sheet_names)}')
        print(f'   シート名: {xls.sheet_names}')
        
        # 各シートの基本情報
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(latest_excel, sheet_name=sheet_name)
                print(f'   📋 [{sheet_name}]: {df.shape[0]}行 × {df.shape[1]}列')
                
                # 重要データの存在確認
                if df.shape[0] > 0:
                    non_null_cols = df.count().sum()
                    total_cells = df.shape[0] * df.shape[1]
                    fill_rate = (non_null_cols / total_cells * 100) if total_cells > 0 else 0
                    print(f'        データ充填率: {fill_rate:.1f}%')
                else:
                    print(f'        ❌ データなし')
                    
            except Exception as e:
                print(f'   ❌ [{sheet_name}]: 読み込みエラー - {e}')
                
    except Exception as e:
        print(f'   ❌ Excel読み込みエラー: {e}')

# 最新のテキストレポート詳細分析
text_dssms = [f for _, f in dssms_files if f.endswith('.txt') and 'report' in f]
if text_dssms:
    latest_text = text_dssms[0]
    print(f'\n📋 最新テキストレポート詳細分析: {latest_text}')
    
    try:
        with open(latest_text, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f'   総行数: {len(lines)}')
        
        # 重要な数値を抽出
        key_values = {}
        for i, line in enumerate(lines):
            line = line.strip()
            if '総リターン:' in line:
                key_values['総リターン'] = (i+1, line)
            elif '最終ポートフォリオ価値:' in line:
                key_values['最終価値'] = (i+1, line)
            elif '銘柄切替回数:' in line:
                key_values['切替回数'] = (i+1, line)
            elif 'シャープレシオ:' in line:
                key_values['シャープレシオ'] = (i+1, line)
        
        print('   📊 抽出された重要指標:')
        for key, (line_no, value) in key_values.items():
            print(f'     行{line_no:3d}: {value}')
            
    except Exception as e:
        print(f'   ❌ テキスト読み込みエラー: {e}')
"
```

### Phase 1.3: データ不整合詳細分析
**目的**: Excel vs テキストレポートの具体的な相違点特定  
**優先度**: 🟡 高  
**所要時間**: 45分  

#### Task 1.3.1: 同期間データの直接比較
**実行コマンド**:
```powershell
python -c "
import pandas as pd
import glob
import os
import re
from datetime import datetime

print('=== データ不整合詳細分析 ===')

# 最新のDSSMS出力ファイルペアを特定
excel_files = [f for f in glob.glob('**/*.xlsx', recursive=True) if 'dssms' in f.lower()]
text_files = [f for f in glob.glob('**/*.txt', recursive=True) if 'dssms' in f.lower() and 'report' in f.lower()]

if not excel_files or not text_files:
    print('❌ 比較対象ファイルが不足しています')
    print(f'   Excel: {len(excel_files)}個, Text: {len(text_files)}個')
    exit()

# 最新ファイルペアの選択
latest_excel = max(excel_files, key=os.path.getmtime)
latest_text = max(text_files, key=os.path.getmtime)

excel_time = datetime.fromtimestamp(os.path.getmtime(latest_excel))
text_time = datetime.fromtimestamp(os.path.getmtime(latest_text))

print(f'📊 比較対象ファイル:')
print(f'   Excel: {latest_excel} ({excel_time.strftime(\"%Y-%m-%d %H:%M:%S\")})')
print(f'   Text:  {latest_text} ({text_time.strftime(\"%Y-%m-%d %H:%M:%S\")})')

# 時間差チェック
time_diff = abs((excel_time - text_time).total_seconds())
if time_diff > 300:  # 5分以上の差
    print(f'   ⚠️  生成時間差: {time_diff:.0f}秒 (同期性に問題の可能性)')
else:
    print(f'   ✅ 生成時間差: {time_diff:.0f}秒 (同期性OK)')

print('\n' + '='*60)

# Excelデータの抽出
print('📊 Excelデータ抽出:')
excel_data = {}

try:
    # サマリーシートからデータ抽出
    df_summary = pd.read_excel(latest_excel, sheet_name=0)  # 最初のシート
    print(f'   ✅ サマリーシート読み込み: {df_summary.shape}')
    
    # 重要な値を抽出（様々な形式に対応）
    for idx, row in df_summary.iterrows():
        for col_idx, col in enumerate(df_summary.columns):
            cell_value = row.iloc[col_idx] if col_idx < len(row) else None
            
            if pd.notna(cell_value):
                cell_str = str(cell_value)
                
                # パターンマッチングで重要データを抽出
                if '最終ポートフォリオ価値' in cell_str or '最終価値' in cell_str:
                    # 次の列または次の行から値を取得
                    if col_idx + 1 < len(row):
                        excel_data['最終価値'] = row.iloc[col_idx + 1]
                elif '総リターン' in cell_str:
                    if col_idx + 1 < len(row):
                        excel_data['総リターン'] = row.iloc[col_idx + 1]
                elif '銘柄切替回数' in cell_str:
                    if col_idx + 1 < len(row):
                        excel_data['切替回数'] = row.iloc[col_idx + 1]
    
    print(f'   📋 抽出されたExcelデータ: {len(excel_data)}項目')
    for key, value in excel_data.items():
        print(f'     {key}: {value}')
        
except Exception as e:
    print(f'   ❌ Excel読み込みエラー: {e}')

print('\n' + '-'*40)

# テキストデータの抽出
print('📋 テキストデータ抽出:')
text_data = {}

try:
    with open(latest_text, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f'   ✅ テキストファイル読み込み: {len(content)}文字')
    
    # 正規表現で数値を抽出
    patterns = {
        '最終価値': r'最終ポートフォリオ価値[：:]\\s*([\\d,]+(?:\\.\\d+)?)円?',
        '総リターン': r'総リターン[：:]\\s*([\\d\\.\\-]+)%',
        '切替回数': r'銘柄切替回数[：:]\\s*(\\d+)回',
        'シャープレシオ': r'シャープレシオ[：:]\\s*([\\d\\.\\-]+)',
        '最大ドローダウン': r'最大ドローダウン[：:]\\s*([\\d\\.\\-]+)%'
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, content)
        if matches:
            text_data[key] = matches[0]
            print(f'     {key}: {matches[0]}')
        else:
            print(f'     {key}: 見つからず')
            
except Exception as e:
    print(f'   ❌ テキスト読み込みエラー: {e}')

print('\n' + '='*60)

# データ比較分析
print('🔍 データ比較分析:')

if excel_data and text_data:
    common_keys = set(excel_data.keys()) & set(text_data.keys())
    
    if common_keys:
        print(f'   📊 比較可能項目: {len(common_keys)}個')
        
        for key in common_keys:
            excel_val = excel_data[key]
            text_val = text_data[key]
            
            # 数値変換を試行
            try:
                excel_num = float(str(excel_val).replace(',', '').replace('%', '').replace('円', ''))
                text_num = float(str(text_val).replace(',', '').replace('%', '').replace('円', ''))
                
                diff = abs(excel_num - text_num)
                diff_pct = (diff / max(abs(excel_num), abs(text_num)) * 100) if max(abs(excel_num), abs(text_num)) > 0 else 0
                
                status = '✅ 一致' if diff < 0.01 else f'❌ 差異 ({diff_pct:.1f}%)'
                
                print(f'   📋 {key}:')
                print(f'     Excel: {excel_val}')
                print(f'     Text:  {text_val}')
                print(f'     判定:  {status}')
                
            except:
                # 文字列比較
                status = '✅ 一致' if str(excel_val) == str(text_val) else '❌ 不一致'
                print(f'   📋 {key}:')
                print(f'     Excel: {excel_val}')
                print(f'     Text:  {text_val}')
                print(f'     判定:  {status}')
    else:
        print('   ❌ 比較可能な共通項目がありません')
        print(f'     Excel項目: {list(excel_data.keys())}')
        print(f'     Text項目:  {list(text_data.keys())}')
else:
    print('   ❌ データ抽出に失敗しました')
"
```

### Phase 1.4: main.py実行環境詳細調査
**目的**: main.py実行時の段階的エラー分析  
**優先度**: 🔴 最高  
**所要時間**: 30分  

#### Task 1.4.1: main.py段階的実行テスト
**実行コマンド**:
```powershell
python -c "
print('=== main.py 段階的実行テスト ===')

import sys
import traceback
import os

# 実行環境確認
print('🖥️  実行環境:')
print(f'   Python: {sys.version}')
print(f'   実行パス: {sys.executable}')
print(f'   作業ディレクトリ: {os.getcwd()}')
print(f'   main.py存在: {\"✅\" if os.path.exists(\"main.py\") else \"❌\"}')

test_results = {}

# Step 1: 基本ライブラリインポート
print('\n📦 Step 1: 基本ライブラリインポート')
try:
    import pandas as pd
    import numpy as np
    import logging
    test_results['基本ライブラリ'] = '✅ 成功'
    print('✅ 基本ライブラリ: OK')
except Exception as e:
    test_results['基本ライブラリ'] = f'❌ 失敗: {e}'
    print(f'❌ 基本ライブラリ: {e}')

# Step 2: プロジェクト設定インポート
print('\n⚙️  Step 2: プロジェクト設定インポート')
try:
    from config.logger_config import setup_logger
    from config.optimized_parameters import get_optimized_parameters
    test_results['プロジェクト設定'] = '✅ 成功'
    print('✅ プロジェクト設定: OK')
except Exception as e:
    test_results['プロジェクト設定'] = f'❌ 失敗: {e}'
    print(f'❌ プロジェクト設定: {e}')
    print('詳細エラー:')
    traceback.print_exc()

# Step 3: 統合システムインポート
print('\n🔗 Step 3: 統合システムインポート')
try:
    from config.multi_strategy_manager import MultiStrategyManager
    test_results['統合システム'] = '✅ 成功'
    print('✅ 統合システム: OK')
except Exception as e:
    test_results['統合システム'] = f'❌ 失敗: {e}'
    print(f'❌ 統合システム: {e}')

# Step 4: データ処理モジュールインポート
print('\n💾 Step 4: データ処理モジュールインポート')
try:
    import data_fetcher
    import data_processor
    test_results['データ処理'] = '✅ 成功'
    print('✅ データ処理: OK')
except Exception as e:
    test_results['データ処理'] = f'❌ 失敗: {e}'
    print(f'❌ データ処理: {e}')

# Step 5: 戦略モジュールインポート
print('\n🎯 Step 5: 戦略モジュールインポート')
try:
    from strategies.vwap_breakout_strategy import VWAPBreakoutStrategy
    test_results['戦略モジュール'] = '✅ 成功'
    print('✅ 戦略モジュール: OK')
except Exception as e:
    test_results['戦略モジュール'] = f'❌ 失敗: {e}'
    print(f'❌ 戦略モジュール: {e}')

# Step 6: 問題の出力モジュールインポート
print('\n📤 Step 6: 出力モジュールインポート (問題箇所)')
try:
    from output.simple_simulation_handler import simulate_and_save
    test_results['出力モジュール'] = '✅ 成功'
    print('✅ 出力モジュール: OK')
except Exception as e:
    test_results['出力モジュール'] = f'❌ 失敗: {e}'
    print(f'❌ 出力モジュール: {e}')
    print('詳細エラー:')
    traceback.print_exc()

# 結果サマリー
print('\n' + '='*50)
print('📊 段階的テスト結果サマリー:')
for step, result in test_results.items():
    print(f'   {step}: {result}')

# 成功率計算
success_count = sum(1 for result in test_results.values() if '✅' in result)
total_count = len(test_results)
success_rate = (success_count / total_count * 100) if total_count > 0 else 0

print(f'\n📈 成功率: {success_count}/{total_count} ({success_rate:.1f}%)')

if success_rate < 100:
    print('\\n🔧 次のアクション:')
    for step, result in test_results.items():
        if '❌' in result:
            print(f'   - {step}の修復が必要')
"
```

#### Task 1.4.2: 出力モジュール詳細エラー分析
**実行コマンド**:
```powershell
python -c "
print('=== 出力モジュール詳細エラー分析 ===')

import os
import sys
import traceback

# output/simple_simulation_handler.pyの詳細分析
handler_path = 'output/simple_simulation_handler.py'
print(f'🔍 ターゲットファイル: {handler_path}')

if os.path.exists(handler_path):
    print('✅ ファイル存在確認: OK')
    
    try:
        with open(handler_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\\n')
        print(f'   📄 総行数: {len(lines)}')
        
        # インポート文の分析
        import_lines = []
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                import_lines.append((i+1, line.strip()))
        
        print(f'   📦 インポート文: {len(import_lines)}個')
        
        # 問題のあるインポート行を特定
        problem_line = None
        for line_no, import_stmt in import_lines:
            print(f'     行{line_no:2d}: {import_stmt}')
            if 'simple_excel_exporter' in import_stmt:
                problem_line = (line_no, import_stmt)
                print(f'          ↑ ❌ 問題のあるインポート')
        
        if problem_line:
            line_no, stmt = problem_line
            print(f'\\n🎯 問題箇所詳細:')
            print(f'   ファイル: {handler_path}')
            print(f'   行番号: {line_no}')
            print(f'   内容: {stmt}')
            
            # 前後の行も表示
            start = max(0, line_no - 3)
            end = min(len(lines), line_no + 2)
            print(f'\\n📋 コンテキスト (行{start+1}-{end}):')
            for i in range(start, end):
                marker = '>>> ' if i == line_no - 1 else '    '
                print(f'   {marker}{i+1:3d}: {lines[i]}')
        
        # 実際のインポートテスト
        print(f'\\n🧪 インポートテスト:')
        try:
            import output.simple_simulation_handler
            print('✅ simple_simulation_handlerインポート: 成功')
        except Exception as e:
            print(f'❌ simple_simulation_handlerインポート: {e}')
            print('詳細エラー:')
            traceback.print_exc()
            
    except Exception as e:
        print(f'❌ ファイル読み込みエラー: {e}')
else:
    print('❌ ファイル存在確認: NG')

# simple_excel_exporter.pyの存在確認と分析
exporter_path = 'output/simple_excel_exporter.py'
print(f'\\n🔍 依存ファイル: {exporter_path}')

if os.path.exists(exporter_path):
    print('✅ 依存ファイル存在: OK')
    
    try:
        # ファイル基本情報
        stat = os.stat(exporter_path)
        print(f'   📊 ファイルサイズ: {stat.st_size} bytes')
        
        with open(exporter_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\\n')
        print(f'   📄 行数: {len(lines)}')
        
        # 主要な関数・クラスの確認
        functions = []
        classes = []
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                func_name = line.split('(')[0].replace('def ', '')
                functions.append(func_name)
            elif line.startswith('class '):
                class_name = line.split('(')[0].split(':')[0].replace('class ', '')
                classes.append(class_name)
        
        print(f'   🔧 関数: {functions}')
        print(f'   🏛️  クラス: {classes}')
        
        # save_backtest_results_simple関数の存在確認
        if 'save_backtest_results_simple' in content:
            print('   ✅ save_backtest_results_simple関数: 存在')
        else:
            print('   ❌ save_backtest_results_simple関数: 不存在')
            
    except Exception as e:
        print(f'❌ 依存ファイル読み込みエラー: {e}')
else:
    print('❌ 依存ファイル存在: NG')
    print('   🔧 これが主要な問題原因です！')
"
```

---

## 📋 Phase 1 実行チェックリスト

### ✅ 実行前準備
- [ ] PowerShellで作業ディレクトリが `C:\Users\imega\Documents\my_backtest_project` であることを確認
- [ ] Python仮想環境が有効化されていることを確認
- [ ] このドキュメントを参照用に開いている

### 📝 実行手順
1. **Task 1.1.1** → **Task 1.1.2** → **Task 1.1.3** の順で実行
2. **Task 1.2.1** → **Task 1.2.2** の順で実行  
3. **Task 1.3.1** を実行
4. **Task 1.4.1** → **Task 1.4.2** の順で実行

### 📊 各Task実行後の記録項目
- **実行日時**
- **コマンド実行結果** (成功/失敗)
- **発見された問題点**
- **重要な出力内容**
- **次のTaskへの引き継ぎ事項**

---

## 🎯 Phase 1 完了後の期待成果

### 📋 問題特定レポート
1. **根本原因の明確化**
   - main.py実行エラーの具体的原因
   - 欠損ファイル・モジュールの特定
   - データ不整合の具体的箇所

2. **影響範囲の把握**
   - 機能停止している範囲
   - 正常動作している範囲
   - 修復優先度の決定

3. **修復計画の設計材料**
   - 必要な作業内容
   - 作業順序
   - 所要時間見積もり

### 🚀 Phase 2 (解決実装) への準備完了
Phase 1の調査結果に基づいて、具体的な修復作業を開始できる状態を目指します。
把握した問題と解決策の案をC:\Users\imega\Documents\my_backtest_project\docs\dssms\Output problem solving roadmap.mdにまとめていきます
最も重視することは解決のためのphaseとタスク化です、文面はおもにそこに重点をおき記入していきます

---

**📞 実行時サポート**: 各Taskで予期しないエラーが発生した場合は、エラーメッセージ全文と実行コマンドを報告してください。即座に代替手順を提供します。
