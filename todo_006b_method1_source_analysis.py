#!/usr/bin/env python3
"""
TODO-006-B 手法1: unified_exporterソースコード読み取り
目的: unified_exporterのペアリングロジックを解析して124取引生成メカニズムを特定
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from pathlib import Path

def analyze_unified_exporter_source():
    """
    unified_exporterのソースコード解析
    """
    print("=" * 60)
    print("🔍 TODO-006-B 手法1: unified_exporterソースコード解析")
    print("=" * 60)
    
    # unified_exporterの場所を特定
    possible_paths = [
        Path(r"C:\Users\imega\Documents\my_backtest_project\output\unified_exporter.py"),
        Path(r"C:\Users\imega\Documents\my_backtest_project\src\output\unified_exporter.py"),
        Path(r"C:\Users\imega\Documents\my_backtest_project\output"),
    ]
    
    unified_exporter_path = None
    for path in possible_paths:
        if path.exists():
            if path.is_file():
                unified_exporter_path = path
                break
            elif path.is_dir():
                # ディレクトリ内のunified_exporter.pyを探す
                for py_file in path.glob("*unified_exporter*.py"):
                    unified_exporter_path = py_file
                    break
    
    if not unified_exporter_path:
        print(f"❌ unified_exporterファイルが見つかりません")
        return None
    
    print(f"📁 unified_exporterファイル: {unified_exporter_path}")
    
    try:
        with open(unified_exporter_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return analyze_pairing_logic(content)
        
    except Exception as e:
        print(f"❌ unified_exporter読み込みエラー: {e}")
        return None

def analyze_pairing_logic(content):
    """
    ペアリングロジックの詳細分析
    """
    print(f"\n📊 ペアリングロジック分析:")
    
    # 重要な関数/メソッドを特定
    key_functions = [
        'create_trade_pairs',
        'pair_entries_exits', 
        'process_signals',
        'generate_trades',
        'export_trades',
        '_pair_',
        'pairing',
        'entry',
        'exit'
    ]
    
    found_functions = []
    for function in key_functions:
        if function in content:
            found_functions.append(function)
    
    print(f"  📍 発見された関数/キーワード: {len(found_functions)}個")
    for func in found_functions:
        print(f"    ✅ {func}")
    
    # ペアリング処理の特定
    pairing_patterns = [
        'Entry_Signal',
        'Exit_Signal', 
        'entry.*exit',
        'exit.*entry',
        'pairs',
        'unmatched',
        'trade_type',
        'timestamp'
    ]
    
    print(f"\n🔍 ペアリングパターン検索:")
    found_patterns = {}
    for pattern in pairing_patterns:
        if pattern in content:
            found_patterns[pattern] = True
            print(f"    ✅ {pattern}")
        else:
            found_patterns[pattern] = False
            print(f"    ❌ {pattern}")
    
    return {
        'found_functions': found_functions,
        'found_patterns': found_patterns,
        'content_length': len(content)
    }

def search_unified_exporter_files():
    """
    unified_exporter関連ファイルの包括的検索
    """
    print(f"\n" + "=" * 60)
    print("🔍 unified_exporter関連ファイル包括検索")
    print("=" * 60)
    
    project_root = Path(r"C:\Users\imega\Documents\my_backtest_project")
    
    # 全プロジェクトからunified_exporter関連ファイルを検索
    unified_files = []
    for pattern in ["*unified_exporter*", "*exporter*", "*export*"]:
        for ext in [".py", ".json", ".txt"]:
            files = list(project_root.rglob(f"{pattern}{ext}"))
            unified_files.extend(files)
    
    # 重複除去
    unified_files = list(set(unified_files))
    
    print(f"📁 発見されたファイル: {len(unified_files)}個")
    for file in unified_files:
        relative_path = file.relative_to(project_root)
        print(f"  📄 {relative_path}")
    
    return unified_files

def analyze_trade_generation_mechanism():
    """
    取引生成メカニズムの推測分析
    """
    print(f"\n" + "=" * 60)
    print("⚙️ 取引生成メカニズム推測分析")
    print("=" * 60)
    
    # 既知の事実
    known_facts = {
        'input_signals': {
            'Entry_Signal_count': 62,
            'Exit_Signal_count': 62,
            'simultaneous_rows': 62
        },
        'output_trades': {
            'total_trades': 124,
            'entry_trades': 62,
            'exit_trades': 62,
            'pairs': 61,
            'unmatched': 1
        }
    }
    
    print(f"📊 既知の事実:")
    print(f"  入力シグナル:")
    for key, value in known_facts['input_signals'].items():
        print(f"    - {key}: {value}")
    
    print(f"  出力取引:")
    for key, value in known_facts['output_trades'].items():
        print(f"    - {key}: {value}")
    
    # メカニズム仮説
    mechanisms = [
        {
            'name': 'メカニズムA: 同一行二重処理',
            'description': 'unified_exporterが同一行のEntry_Signal=1とExit_Signal=1を別々の取引として処理',
            'process': [
                '1. 行15: Entry_Signal=1 → Entry取引生成',
                '2. 行15: Exit_Signal=1 → Exit取引生成',
                '3. 結果: 同一行から2つの取引'
            ]
        },
        {
            'name': 'メカニズムB: インデックスベース処理',
            'description': 'unified_exporterがインデックスを基準にシグナルを処理し、同一インデックスを重複処理',
            'process': [
                '1. Entry_Signalのインデックス[15,22,24...]を処理',
                '2. Exit_Signalのインデックス[15,22,24...]を処理',
                '3. 結果: 同じインデックスから重複取引'
            ]
        },
        {
            'name': 'メカニズムC: データフレーム重複読み取り',
            'description': 'unified_exporterが同じDataFrameを複数回処理してシグナルを重複抽出',
            'process': [
                '1. DataFrameからEntry_Signal=1を抽出',
                '2. 同じDataFrameからExit_Signal=1を抽出',
                '3. 結果: 同じ行データの重複処理'
            ]
        }
    ]
    
    print(f"\n💡 取引生成メカニズム仮説:")
    for i, mechanism in enumerate(mechanisms, 1):
        print(f"\n{i}. {mechanism['name']}")
        print(f"   説明: {mechanism['description']}")
        print(f"   プロセス:")
        for step in mechanism['process']:
            print(f"     {step}")
    
    return mechanisms

def main():
    print("🔍 TODO-006-B 手法1: unified_exporterソースコード解析 開始")
    
    # Step 1: unified_exporterソースコード分析
    source_analysis = analyze_unified_exporter_source()
    
    # Step 2: 関連ファイル検索
    related_files = search_unified_exporter_files()
    
    # Step 3: 取引生成メカニズム分析
    mechanisms = analyze_trade_generation_mechanism()
    
    # 結論
    print(f"\n" + "=" * 60)
    print("🎯 手法1結論") 
    print("=" * 60)
    
    if source_analysis:
        print(f"✅ unified_exporterソースコード解析完了:")
        print(f"  - 発見された関数: {len(source_analysis.get('found_functions', []))}")
        print(f"  - ペアリングパターン検出: {sum(source_analysis.get('found_patterns', {}).values())}")
        
        print(f"\n🔍 最有力メカニズム:")
        print(f"  • メカニズムA: 同一行二重処理")
        print(f"    根拠: 62行で同時Entry/Exit → 124取引の数学的一致")
    else:
        print(f"❌ unified_exporterソースコード取得失敗")
        print(f"  - 手法2で実行時トレースによる補完が必要")
    
    print(f"\n📝 次の手法での検証項目:")
    print(f"  - unified_exporterの実際の処理フロー")
    print(f"  - シグナル読み取りと取引生成のタイミング")
    print(f"  - ペアリング処理の具体的アルゴリズム")
    
    return {
        'source_analysis': source_analysis,
        'related_files': related_files,
        'mechanisms': mechanisms
    }

if __name__ == "__main__":
    results = main()