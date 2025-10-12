"""
main.py使用中モジュール調査 - comprehensive_module_test.py用選定
既存main.pyで実際に使用されているモジュールの再利用可能性を評価
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from datetime import datetime

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def analyze_main_py_imports():
    """main.pyの実際のimport文を詳細解析"""
    main_path = "main.py"
    imports_info = {
        'from_imports': [],
        'direct_imports': [],
        'failed_imports': []
    }
    
    try:
        # BOM文字を含む可能性があるため、複数のエンコーディングを試行
        encodings = ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']
        content = None
        
        for encoding in encodings:
            try:
                with open(main_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    print(f"✅ エンコーディング {encoding} で読み込み成功")
                    break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise Exception("全てのエンコーディングで読み込みに失敗")
        
        # BOM文字を除去
        if content.startswith('\ufeff'):
            content = content[1:]
            print("✅ BOM文字を除去しました")
            
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports_info['direct_imports'].append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports_info['from_imports'].append({
                            'module': node.module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
        
        print(f"✅ main.py解析完了: {len(imports_info['from_imports'])}個のfromインポート, {len(imports_info['direct_imports'])}個の直接インポート")
        return imports_info
        
    except Exception as e:
        print(f"❌ main.py解析エラー: {e}")
        return imports_info

def categorize_main_imports(imports_info):
    """main.pyのインポートをカテゴリ別に分類"""
    categories = {
        "データ取得・前処理系": [],
        "設定・ログ系": [],
        "個別戦略クラス": [],
        "出力系": [],
        "指標計算系": [],
        "その他": [],
        "破棄予定ファイル": []
    }
    
    # from import の処理
    for imp in imports_info['from_imports']:
        module_path = imp['module']
        imported_name = imp['name']
        
        # モジュールパス解析
        module_info = {
            'name': imported_name,
            'module_path': module_path,
            'import_type': 'from_import',
            'line': imp['line'],
            'alias': imp['alias'],
            'reuse_potential': 'unknown',
            'function_desc': f"from {module_path} import {imported_name}"
        }
        
        # カテゴリ分類
        if any(keyword in module_path.lower() for keyword in ['data_fetch', 'data_processor', 'preprocess']):
            category = "データ取得・前処理系"
            module_info['reuse_potential'] = 'high'
        elif any(keyword in module_path.lower() for keyword in ['config', 'logger', 'risk_management', 'optimized_parameters']):
            category = "設定・ログ系"
            module_info['reuse_potential'] = 'high'
        elif 'strategies' in module_path.lower():
            category = "個別戦略クラス"
            module_info['reuse_potential'] = 'high'
        elif any(keyword in module_path.lower() for keyword in ['output', 'exporter', 'reporter']):
            category = "出力系"
            module_info['reuse_potential'] = 'medium'
        elif any(keyword in module_path.lower() for keyword in ['indicators', 'unified_trend']):
            category = "指標計算系"
            module_info['reuse_potential'] = 'high'
        elif any(keyword in module_path.lower() for keyword in ['dssms', 'archive', 'src']):
            category = "破棄予定ファイル"
            module_info['reuse_potential'] = 'none'
        else:
            category = "その他"
            module_info['reuse_potential'] = 'medium'
        
        categories[category].append(module_info)
    
    # direct import の処理
    for imp in imports_info['direct_imports']:
        module_name = imp['module']
        
        module_info = {
            'name': module_name,
            'module_path': module_name,
            'import_type': 'direct_import',
            'line': imp['line'],
            'alias': imp['alias'],
            'reuse_potential': 'low',  # 標準ライブラリは再利用対象外
            'function_desc': f"import {module_name}"
        }
        
        # 標準ライブラリとサードパーティを区別
        if module_name in ['sys', 'os', 'logging', 'datetime', 'typing']:
            category = "その他"
            module_info['function_desc'] = f"標準ライブラリ: {module_name}"
        elif module_name in ['pandas', 'numpy']:
            category = "その他"
            module_info['function_desc'] = f"データ処理ライブラリ: {module_name}"
        else:
            category = "その他"
        
        categories[category].append(module_info)
    
    return categories

def analyze_module_details(module_path):
    """モジュールの詳細情報を取得"""
    try:
        # 相対パスを絶対パスに変換
        if not module_path.startswith('/') and not module_path[1:3] == ':\\':
            # プロジェクトルートからの相対パス
            full_path = Path(".") / module_path.replace('.', '/') + '.py'
        else:
            full_path = Path(module_path)
        
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # ドックストリング抽出
            try:
                tree = ast.parse(content)
                docstring = ast.get_docstring(tree)
                if docstring:
                    first_line = docstring.split('\n')[0].strip()
                    return first_line[:100] + "..." if len(first_line) > 100 else first_line
            except:
                pass
                
            # クラス・関数名から推測
            try:
                tree = ast.parse(content)
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')]
                
                if classes:
                    return f"クラス: {', '.join(classes[:2])}"
                elif functions:
                    return f"関数: {', '.join(functions[:2])}"
            except:
                pass
        
        return "詳細情報取得不可"
        
    except Exception as e:
        return f"解析エラー: {str(e)[:50]}"

def generate_main_imports_report(categories):
    """main.py使用モジュール調査レポート生成"""
    report_lines = []
    
    report_lines.extend([
        "# main.py使用中モジュール調査レポート",
        "",
        "## 🎯 調査目的",
        "main.pyで実際に使用されているモジュールの再利用可能性評価",
        "comprehensive_module_test.py でテストすべき既存使用モジュールの選定",
        "",
        "## 📋 調査結果サマリー",
        ""
    ])
    
    total_modules = sum(len(modules) for modules in categories.values())
    high_potential = sum(len([m for m in modules if m['reuse_potential'] == 'high']) 
                        for modules in categories.values())
    medium_potential = sum(len([m for m in modules if m['reuse_potential'] == 'medium']) 
                         for modules in categories.values())
    
    report_lines.extend([
        f"- **総使用モジュール数**: {total_modules}",
        f"- **高優先度再利用候補**: {high_potential}",
        f"- **中優先度再利用候補**: {medium_potential}",
        f"- **再利用推奨total**: {high_potential + medium_potential}",
        "",
        "---",
        ""
    ])
    
    # カテゴリ別詳細
    for category_name, modules in categories.items():
        if not modules:
            continue
            
        report_lines.extend([
            f"## {category_name}",
            ""
        ])
        
        high_count = len([m for m in modules if m['reuse_potential'] == 'high'])
        medium_count = len([m for m in modules if m['reuse_potential'] == 'medium'])
        low_count = len([m for m in modules if m['reuse_potential'] == 'low'])
        none_count = len([m for m in modules if m['reuse_potential'] == 'none'])
        
        report_lines.extend([
            f"**概要**: {len(modules)}個のモジュール使用中",
            f"- 高優先度再利用候補: {high_count}個",
            f"- 中優先度再利用候補: {medium_count}個", 
            f"- 低優先度: {low_count}個",
            f"- 再利用禁止: {none_count}個",
            ""
        ])
        
        # 優先度順でソート
        priority_order = {'high': 1, 'medium': 2, 'low': 3, 'none': 4, 'unknown': 5}
        sorted_modules = sorted(modules, 
                              key=lambda x: (priority_order.get(x['reuse_potential'], 6), x['name']))
        
        for module in sorted_modules:
            if module['reuse_potential'] == 'high':
                reuse = "🚀 高優先度"
            elif module['reuse_potential'] == 'medium':  
                reuse = "⚡ 中優先度"
            elif module['reuse_potential'] == 'low':
                reuse = "⚠️ 低優先度"
            elif module['reuse_potential'] == 'none':
                reuse = "🚫 再利用禁止"
            else:
                reuse = "❓ 要調査"
            
            # 詳細情報取得
            if module['import_type'] == 'from_import':
                detailed_desc = analyze_module_details(module['module_path'])
            else:
                detailed_desc = module['function_desc']
            
            report_lines.extend([
                f"### {module['name']}",
                f"- **モジュールパス**: `{module['module_path']}`",
                f"- **インポート方法**: {module['function_desc']}",
                f"- **機能概要**: {detailed_desc}",
                f"- **再利用可能性**: {reuse}",
                f"- **main.py行番号**: {module['line']}",
                ""
            ])
        
        report_lines.append("---")
        report_lines.append("")
    
    # comprehensive_module_test.py推奨テスト対象
    report_lines.extend([
        "## 🎯 comprehensive_module_test.py 推奨テスト対象",
        "",
        "### 🔥 Phase 0: 高優先度テスト（main.py実証済み）",
        ""
    ])
    
    high_priority = []
    medium_priority = []
    
    for category_name, modules in categories.items():
        if category_name == "破棄予定ファイル":
            continue
            
        for module in modules:
            if module['reuse_potential'] == 'high':
                high_priority.append((category_name, module))
            elif module['reuse_potential'] == 'medium':
                medium_priority.append((category_name, module))
    
    if high_priority:
        for category, module in high_priority:
            report_lines.append(f"- **{module['name']}** ({category}): main.pyで実証済み")
    else:
        report_lines.append("- （高優先度候補なし）")
    
    report_lines.extend([
        "",
        "### ⚡ Phase 1: 中優先度テスト（要注意）",
        ""
    ])
    
    if medium_priority:
        for category, module in medium_priority:
            report_lines.append(f"- **{module['name']}** ({category}): 限定テスト推奨")
    else:
        report_lines.append("- （中優先度候補なし）")
    
    # テスト戦略
    report_lines.extend([
        "",
        "## 📝 main.py実証モジュール活用戦略",
        "",
        "### 🎯 main.pyで実証済みの利点",
        "1. **動作保証**: 実際のバックテスト環境で動作確認済み",
        "2. **互換性**: 既存システムとの完全互換性",
        "3. **パラメータ**: 実用的な設定値が既知",
        "4. **エラーパターン**: 既知の問題と対処法が明確",
        "",
        "### 🚀 Phase 0実行順序（main.py実証順）",
        "1. **設定・ログ系**: logger_config, risk_management等",
        "2. **データ取得・前処理系**: data_fetcher, data_processor等", 
        "3. **指標計算系**: indicator_calculator, unified_trend_detector等",
        "4. **個別戦略クラス**: main.pyで使用中の7戦略",
        "",
        "### ✅ 成功基準（main.py準拠）",
        "- ✅ main.pyと同じパラメータで動作",
        "- ✅ 同じシグナル生成パターン", 
        "- ✅ エラーハンドリングの再現",
        "- ✅ パフォーマンス指標の一致",
        "",
        "### 🚨 main.py依存問題の回避",
        "- **circular import回避**: 段階的インポートテスト",
        "- **設定依存の分離**: 独立した設定での動作確認",
        "- **データ依存の最小化**: テストデータでの検証",
        "",
        "---",
        f"**レポート生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**調査対象**: main.py使用中モジュール（実証済み）",
        f"**目的**: comprehensive_module_test.py用の確実な再利用候補選定"
    ])
    
    return "\n".join(report_lines)

def main():
    """main.py使用モジュール調査実行"""
    print("🔍 main.py使用中モジュール調査開始")
    print("="*60)
    
    # main.pyのimport解析
    imports_info = analyze_main_py_imports()
    
    if not imports_info['from_imports'] and not imports_info['direct_imports']:
        print("❌ インポート情報の取得に失敗しました")
        return
    
    # カテゴリ別分類
    categories = categorize_main_imports(imports_info)
    
    # レポート生成
    report = generate_main_imports_report(categories)
    
    # ファイル出力
    output_dir = Path("docs/Plan to create a new main entry point")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "main_py_modules_investigation.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 調査完了")
    print(f"📄 レポート出力: {output_file}")
    print(f"📊 調査結果: {sum(len(modules) for modules in categories.values())}個のモジュール分析完了")
    print(f"🎯 main.py実証済みモジュールの再利用可能性を評価しました")

if __name__ == "__main__":
    main()