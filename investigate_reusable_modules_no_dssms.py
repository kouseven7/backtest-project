"""
main.py未使用だが再利用可能なモジュール調査（DSSMS完全除外版）
comprehensive_module_test.py用のモジュール選定
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

class DSSMSExcluder:
    """DSSMS関連ファイルを確実に除外するクラス"""
    
    def __init__(self):
        # DSSMS除外パターン（大文字小文字問わず）
        self.exclude_patterns = [
            # ディレクトリパターン
            "dssms", "DSSMS", "Dssms",
            # ファイル名パターン  
            "dssms_", "DSSMS_", "Dssms_",
            "_dssms", "_DSSMS", "_Dssms",
            # 機能パターン
            "dynamic_stock", "stock_selection", "selection_system",
            "multi_stock", "stock_selector", "selector_system",
            # アーカイブ・テストパターン
            "__pycache__", ".git", "logs", ".pytest",
            "test_", "debug_", "temp_", "backup_", "old_"
        ]
    
    def is_dssms_related(self, file_path):
        """ファイルがDSSMS関連かどうかを判定"""
        path_str = str(file_path).lower()
        
        # パスにDSSMS関連パターンが含まれているかチェック
        for pattern in self.exclude_patterns:
            if pattern.lower() in path_str:
                return True
        
        # ファイル内容もチェック（念のため）
        if file_path.suffix == '.py':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                # DSSMS関連のキーワードをチェック
                dssms_keywords = [
                    "dssms", "dynamic stock selection", 
                    "multi stock system", "stock selector"
                ]
                
                for keyword in dssms_keywords:
                    if keyword in content:
                        return True
                        
            except Exception:
                # ファイル読み込みエラーは無視
                pass
        
        return False
    
    def filter_modules(self, modules):
        """モジュールリストからDSSMS関連を除外"""
        filtered = []
        excluded_count = 0
        
        for module in modules:
            if not self.is_dssms_related(Path(module['path'])):
                filtered.append(module)
            else:
                excluded_count += 1
        
        print(f"🚫 DSSMS関連除外: {excluded_count}個のファイル")
        return filtered

def analyze_main_imports():
    """main.pyのimport文を解析"""
    main_path = "main.py"
    used_modules = set()
    
    try:
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    used_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    used_modules.add(node.module)
                    
        print(f"✅ main.pyで使用中のモジュール数: {len(used_modules)}")
        return used_modules
        
    except Exception as e:
        print(f"❌ main.py解析エラー: {e}")
        return set()

def scan_project_modules():
    """プロジェクト内の全モジュールをスキャン（DSSMS除外）"""
    project_root = Path(".")
    modules = []
    
    # スキャン対象ディレクトリ（DSSMS関連は除外）  
    scan_dirs = [
        "strategies", "indicators", "output", "config", 
        "data_processor", "data_fetcher", "archive/engines",
        "src/strategies", "src/indicators", "src/config"
    ]
    
    # DSSMS除外器を初期化
    excluder = DSSMSExcluder()
    
    for scan_dir in scan_dirs:
        dir_path = project_root / scan_dir
        if dir_path.exists():
            # DSSMS関連ディレクトリをスキップ
            if excluder.is_dssms_related(dir_path):
                print(f"🚫 ディレクトリ除外: {scan_dir}")
                continue
                
            for py_file in dir_path.rglob("*.py"):
                # DSSMS関連ファイルをスキップ
                if excluder.is_dssms_related(py_file):
                    continue
                    
                modules.append({
                    'name': py_file.stem,
                    'path': str(py_file),
                    'relative_path': str(py_file.relative_to(project_root)),
                    'module_path': str(py_file.relative_to(project_root)).replace('/', '.').replace('\\', '.').replace('.py', '')
                })
    
    # 追加のDSSMS除外フィルタ
    modules = excluder.filter_modules(modules)
    
    print(f"✅ スキャン完了: {len(modules)}個のモジュール発見（DSSMS除外後）")
    return modules

def analyze_module_function(file_path):
    """モジュールの機能を解析"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # ドックストリングから機能を抽出
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
        
        if docstring:
            # 最初の行を機能概要として使用
            first_line = docstring.split('\n')[0].strip()
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            return first_line
        
        # クラス名や関数名から推測
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                functions.append(node.name)
        
        if classes:
            return f"クラス: {', '.join(classes[:2])}"
        elif functions:
            return f"関数: {', '.join(functions[:2])}"
        else:
            return "機能不明"
            
    except Exception as e:
        return f"解析エラー: {str(e)[:30]}"

def categorize_modules(modules, used_modules):
    """モジュールをカテゴリ別に分類"""
    categories = {
        "データ取得・前処理系": [],
        "設定・ログ系": [],
        "個別戦略クラス": [],
        "出力系": [],
        "指標計算系": [],
        "その他": [],
        "破棄予定ファイル": []
    }
    
    for module in modules:
        name = module['name'].lower()
        path = module['path'].lower()
        module_path = module['module_path']
        
        # 使用済みかチェック
        is_used = any(used_mod in module_path or module['name'] in used_mod 
                     for used_mod in used_modules)
        
        # 機能解析
        function_desc = analyze_module_function(module['path'])
        
        # カテゴリ分類
        if any(keyword in name for keyword in ['data_fetch', 'fetcher', 'loader', 'get_data']):
            category = "データ取得・前処理系"
        elif any(keyword in name for keyword in ['processor', 'preprocess', 'clean', 'transform']):
            category = "データ取得・前処理系"
        elif any(keyword in name for keyword in ['config', 'setting', 'param', 'logger', 'management']):
            category = "設定・ログ系"
        elif 'strategy' in path and 'strategies' in path:
            category = "個別戦略クラス"
        elif any(keyword in name for keyword in ['output', 'export', 'report', 'writer', 'saver']):
            category = "出力系"
        elif any(keyword in name for keyword in ['indicator', 'technical', 'signal', 'calc']):
            category = "指標計算系"
        elif any(keyword in name for keyword in ['old', 'deprecated', 'temp', 'backup']):
            category = "破棄予定ファイル"
        elif 'archive' in path or 'test' in path:
            category = "破棄予定ファイル"
        else:
            category = "その他"
        
        # 再利用可能性評価
        reuse_potential = 'low'
        if not is_used and category not in ["破棄予定ファイル"]:
            if category in ["データ取得・前処理系", "設定・ログ系"]:
                reuse_potential = 'high'
            elif category in ["個別戦略クラス", "指標計算系"]:
                reuse_potential = 'medium'
            else:
                reuse_potential = 'low'
        
        module_info = {
            **module,
            'function_desc': function_desc,
            'is_used': is_used,
            'reuse_potential': reuse_potential,
            'category': category
        }
        
        categories[category].append(module_info)
    
    return categories

def generate_report(categories):
    """調査レポートを生成"""
    report_lines = []
    
    report_lines.extend([
        "# main.py未使用モジュール調査レポート（DSSMS除外版）",
        "",
        "## 🎯 調査目的",
        "comprehensive_module_test.py でテストすべき再利用可能なモジュールの選定",
        "**DSSMS関連ファイルは完全除外済み**",
        "",
        "## 📋 調査結果サマリー",
        ""
    ])
    
    total_modules = sum(len(modules) for modules in categories.values())
    unused_modules = sum(len([m for m in modules if not m['is_used']]) 
                        for modules in categories.values())
    high_potential = sum(len([m for m in modules if m['reuse_potential'] == 'high']) 
                        for modules in categories.values())
    medium_potential = sum(len([m for m in modules if m['reuse_potential'] == 'medium']) 
                         for modules in categories.values())
    
    report_lines.extend([
        f"- **総モジュール数**: {total_modules}（DSSMS除外後）",
        f"- **未使用モジュール数**: {unused_modules}",
        f"- **高優先度再利用候補**: {high_potential}",
        f"- **中優先度再利用候補**: {medium_potential}",
        f"- **total再利用候補**: {high_potential + medium_potential}",
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
        
        unused_count = len([m for m in modules if not m['is_used']])
        high_count = len([m for m in modules if m['reuse_potential'] == 'high'])
        medium_count = len([m for m in modules if m['reuse_potential'] == 'medium'])
        
        report_lines.extend([
            f"**概要**: {len(modules)}個（未使用: {unused_count}個、高優先度: {high_count}個、中優先度: {medium_count}個）",
            ""
        ])
        
        # 優先度順でソート
        priority_order = {'high': 1, 'medium': 2, 'low': 3}
        sorted_modules = sorted(modules, 
                              key=lambda x: (x['is_used'], priority_order.get(x['reuse_potential'], 4), x['name']))
        
        for module in sorted_modules:
            status = "✅ 使用中" if module['is_used'] else "🔍 未使用"
            
            if module['reuse_potential'] == 'high':
                reuse = "🚀 高優先度"
            elif module['reuse_potential'] == 'medium':  
                reuse = "⚡ 中優先度"
            else:
                reuse = "⚠️ 低優先度"
            
            report_lines.extend([
                f"### {module['name']}",
                f"- **パス**: `{module['relative_path']}`",
                f"- **機能**: {module['function_desc']}",
                f"- **使用状況**: {status}",
                f"- **再利用可能性**: {reuse}",
                ""
            ])
        
        report_lines.append("---")
        report_lines.append("")
    
    # 推奨テスト対象
    report_lines.extend([
        "## 🎯 comprehensive_module_test.py 推奨テスト対象",
        "",
        "### 🔥 Phase 0: 高優先度テスト（安全性高・効果大）",
        ""
    ])
    
    high_priority = []
    medium_priority = []
    
    for category_name, modules in categories.items():
        if category_name == "破棄予定ファイル":
            continue
            
        for module in modules:
            if not module['is_used']:
                if module['reuse_potential'] == 'high':
                    high_priority.append((category_name, module))
                elif module['reuse_potential'] == 'medium':
                    medium_priority.append((category_name, module))
    
    if high_priority:
        for category, module in high_priority:
            report_lines.append(f"- **{module['name']}** ({category}): {module['function_desc']}")
    else:
        report_lines.append("- （高優先度候補なし）")
    
    report_lines.extend([
        "",
        "### ⚡ Phase 1: 中優先度テスト（要注意テスト）",
        ""
    ])
    
    if medium_priority:
        for category, module in medium_priority:
            report_lines.append(f"- **{module['name']}** ({category}): {module['function_desc']}")
    else:
        report_lines.append("- （中優先度候補なし）")
    
    # テスト戦略を追加
    report_lines.extend([
        "",
        "## 📝 推奨テスト戦略",
        "",
        "### Phase 0テスト順序",
        "1. **データ取得・前処理系**（高優先度）",
        "2. **設定・ログ系**（高優先度）", 
        "3. **個別戦略クラス**（中優先度から1つ選択）",
        "",
        "### 成功基準",
        "- ✅ インポートエラーなし",
        "- ✅ 基本メソッド動作確認",
        "- ✅ Entry_Signal/Exit_Signal生成確認（戦略のみ）",
        "- ✅ 同一日Entry/Exit問題なし",
        "",
        "### 失敗時の対応",
        "- 3回修正試行後も失敗 → 破棄",
        "- 複雑すぎて理解困難 → 破棄",
        "- DSSMS関連発見 → 即座に破棄",
        "",
        "---",
        f"**レポート生成日時**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**調査対象**: main.py未使用だが再利用可能なモジュール（DSSMS完全除外）",
        f"**除外対象**: DSSMS関連、テスト用、アーカイブ、廃止予定ファイル"
    ])
    
    return "\n".join(report_lines)

def main():
    """メイン調査処理"""
    print("🔍 main.py未使用モジュール調査開始（DSSMS除外版）")
    print("="*60)
    
    # main.pyのimport解析
    used_modules = analyze_main_imports()
    
    # プロジェクト全モジュールスキャン（DSSMS除外）
    all_modules = scan_project_modules()
    
    # カテゴリ別分類
    categories = categorize_modules(all_modules, used_modules)
    
    # レポート生成
    report = generate_report(categories)
    
    # ファイル出力
    output_dir = Path("docs/Plan to create a new main entry point")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "reusable_modules_investigation_no_dssms.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 調査完了")
    print(f"📄 レポート出力: {output_file}")
    print(f"📊 調査結果: {sum(len(modules) for modules in categories.values())}個のモジュール分析完了")
    print(f"🚫 DSSMS関連ファイルは完全除外されました")

if __name__ == "__main__":
    main()