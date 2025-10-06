#!/usr/bin/env python3
"""
TODO-PERF-001: Phase 2 Stage 3 構文エラー緊急修正

Stage 3最適化で発生した構文エラーを修正
"""

import ast
import sys
from pathlib import Path

def fix_dssms_report_generator_syntax():
    """dssms_report_generator.py 構文エラー修正"""
    print("🔧 dssms_report_generator.py 構文エラー修正中...")
    
    report_gen_path = Path("src/dssms/dssms_report_generator.py")
    if not report_gen_path.exists():
        print("  ❌ ファイルが存在しません")
        return False
    
    try:
        with open(report_gen_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 重複したコード挿入を修正
        lines = content.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            # 重複したクラス定義を検出
            if 'TODO-PERF-001:' in line and i > 0 and 'TODO-PERF-001:' in lines[i-1]:
                continue  # 重複行をスキップ
            
            # 不正な関数定義修正
            if line.strip().startswith('def ') and '\\1' in line:
                continue  # 正規表現の置換エラーをスキップ
            
            # デコレーター修正
            if '@_report_optimizer.cached_computation' in line:
                line = '    # @_report_optimizer.cached_computation  # TODO-PERF-001: デコレーター修正'
            
            cleaned_lines.append(line)
        
        # 基本的な構文チェック用に簡略化
        simplified_content = '\n'.join(cleaned_lines)
        
        # functools インポートが不足している場合に追加
        if 'import functools' not in simplified_content and '@functools' in simplified_content:
            import_section = []
            code_section = []
            in_imports = True
            
            for line in cleaned_lines:
                if line.strip() and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                    in_imports = False
                
                if in_imports:
                    import_section.append(line)
                else:
                    code_section.append(line)
            
            # functools インポートを追加
            import_section.append('import functools  # TODO-PERF-001: 最適化用インポート')
            simplified_content = '\n'.join(import_section + code_section)
        
        with open(report_gen_path, 'w', encoding='utf-8') as f:
            f.write(simplified_content)
        
        print("  ✅ dssms_report_generator.py 修正完了")
        return True
        
    except Exception as e:
        print(f"  ❌ 修正エラー: {e}")
        return False

def fix_hierarchical_ranking_syntax():
    """hierarchical_ranking_system.py 構文エラー修正"""
    print("🔧 hierarchical_ranking_system.py 構文エラー修正中...")
    
    ranking_path = Path("src/dssms/hierarchical_ranking_system.py")
    if not ranking_path.exists():
        print("  ❌ ファイルが存在しません")
        return False
    
    try:
        with open(ranking_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本的な構文問題を修正
        fixed_content = content
        
        # 重複したパフォーマンス監視コードを統合
        if content.count('performance_monitor') > 2:
            # 最初の定義以外を削除
            lines = fixed_content.split('\n')
            cleaned_lines = []
            monitor_defined = False
            
            for line in lines:
                if 'def performance_monitor' in line:
                    if not monitor_defined:
                        cleaned_lines.append(line)
                        monitor_defined = True
                    else:
                        continue  # 重複定義をスキップ
                else:
                    cleaned_lines.append(line)
            
            fixed_content = '\n'.join(cleaned_lines)
        
        with open(ranking_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("  ✅ hierarchical_ranking_system.py 修正完了")
        return True
        
    except Exception as e:
        print(f"  ❌ 修正エラー: {e}")
        return False

def validate_syntax():
    """構文検証"""
    print("🔍 構文検証中...")
    
    files_to_check = [
        "src/dssms/dssms_report_generator.py",
        "src/dssms/hierarchical_ranking_system.py"
    ]
    
    all_valid = True
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ast.parse(content)
                print(f"  ✅ {file_path}: 構文OK")
                
            except SyntaxError as e:
                print(f"  ❌ {file_path}: 構文エラー - {e}")
                print(f"    行 {e.lineno}: {e.text}")
                all_valid = False
            except Exception as e:
                print(f"  ⚠️ {file_path}: 検証エラー - {e}")
                all_valid = False
        else:
            print(f"  ❌ {file_path}: ファイル未存在")
            all_valid = False
    
    return all_valid

def main():
    """メイン実行"""
    print("🚀 TODO-PERF-001 Phase 2 Stage 3 構文エラー緊急修正開始")
    print("=" * 60)
    
    success = True
    
    # 1. dssms_report_generator.py修正
    if not fix_dssms_report_generator_syntax():
        success = False
    
    # 2. hierarchical_ranking_system.py修正
    if not fix_hierarchical_ranking_syntax():
        success = False
    
    # 3. 構文検証
    print("\n🔍 修正結果検証")
    if validate_syntax():
        print("✅ 全ファイル構文OK")
    else:
        print("⚠️ 一部ファイルに構文問題あり")
        success = False
    
    if success:
        print("\n🎉 構文エラー修正完了 - Stage 4進行可能")
    else:
        print("\n⚠️ 構文エラー修正に問題あり - さらなる修正が必要")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)