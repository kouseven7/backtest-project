#!/usr/bin/env python3
"""
TODO-003 Phase 1 調査手法1: main.pyソースコード静的解析
Exit_Signalを処理する全箇所の特定
- abs()関数、絶対値変換、符号反転処理の検索
- DataFrame操作でのExit_Signal列変換処理確認
"""

import os
import re
import ast

print("=" * 80)
print("🔍 TODO-003 Phase 1-1: main.pyソースコード静的解析")
print("=" * 80)

def analyze_main_py_exit_signal_processing():
    """main.pyでのExit_Signal処理箇所を静的解析"""
    
    main_py_path = "main.py"
    if not os.path.exists(main_py_path):
        print(f"❌ main.pyが見つかりません: {main_py_path}")
        return None
    
    try:
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ main.py読み込みエラー: {e}")
        return None
    
    print(f"✅ main.py読み込み完了 ({len(content)}文字)")
    
    # 1. Exit_Signal関連処理の検索
    print(f"\n🔍 **Exit_Signal関連処理の検索**")
    
    exit_signal_patterns = [
        r'Exit_Signal.*=.*',
        r'.*Exit_Signal.*',
        r'exit_signal.*=.*',
        r'.*exit_signal.*'
    ]
    
    exit_signal_matches = []
    for pattern in exit_signal_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        exit_signal_matches.extend(matches)
    
    # 重複除去と行番号取得
    lines = content.split('\n')
    exit_signal_lines = []
    
    for i, line in enumerate(lines, 1):
        if 'exit_signal' in line.lower():
            exit_signal_lines.append({
                'line_num': i,
                'content': line.strip(),
                'full_line': line
            })
    
    print(f"Exit_Signal関連行数: {len(exit_signal_lines)}行")
    for item in exit_signal_lines[:10]:  # 最初の10行表示
        print(f"  Line {item['line_num']}: {item['content']}")
    
    if len(exit_signal_lines) > 10:
        print(f"  ... (残り{len(exit_signal_lines)-10}行)")
    
    # 2. abs()関数、絶対値変換、符号反転処理の検索
    print(f"\n🔍 **abs()関数と符号変換処理の検索**")
    
    conversion_patterns = [
        r'abs\s*\(',
        r'\.abs\s*\(',
        r'numpy\.abs',
        r'np\.abs',
        r'\*\s*-1',
        r'\*\s*\(-1\)',
        r'.*=-.*\*.*',
        r'.*=.*abs.*'
    ]
    
    conversion_lines = []
    for i, line in enumerate(lines, 1):
        for pattern in conversion_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                conversion_lines.append({
                    'line_num': i,
                    'content': line.strip(),
                    'pattern': pattern
                })
                break
    
    print(f"符号変換関連行数: {len(conversion_lines)}行")
    for item in conversion_lines:
        print(f"  Line {item['line_num']}: {item['content']}")
        print(f"    パターン: {item['pattern']}")
    
    # 3. DataFrame操作でのExit_Signal列変換処理
    print(f"\n🔍 **DataFrame操作でのExit_Signal変換処理**")
    
    dataframe_patterns = [
        r'\.loc\[.*Exit_Signal.*\]',
        r'\.at\[.*Exit_Signal.*\]',
        r'\.iloc\[.*Exit_Signal.*\]',
        r'\[.*Exit_Signal.*\].*=',
        r'Exit_Signal.*\.map',
        r'Exit_Signal.*\.apply',
        r'Exit_Signal.*\.transform'
    ]
    
    dataframe_lines = []
    for i, line in enumerate(lines, 1):
        for pattern in dataframe_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                dataframe_lines.append({
                    'line_num': i,
                    'content': line.strip(),
                    'pattern': pattern
                })
                break
    
    print(f"DataFrame操作関連行数: {len(dataframe_lines)}行")
    for item in dataframe_lines:
        print(f"  Line {item['line_num']}: {item['content']}")
        print(f"    パターン: {item['pattern']}")
    
    # 4. 疑わしい変換処理の特定
    print(f"\n🚨 **疑わしい変換処理の特定**")
    
    suspicious_lines = []
    
    # Exit_Signalと符号変換が同じ行にある場合
    for item in exit_signal_lines:
        line_content = item['full_line'].lower()
        if ('abs(' in line_content or 
            '*-1' in line_content or 
            '* -1' in line_content or 
            '*(-1)' in line_content):
            suspicious_lines.append({
                'line_num': item['line_num'],
                'content': item['content'],
                'reason': 'Exit_Signalと符号変換が同じ行'
            })
    
    # Exit_Signalの値を1に設定している箇所
    for item in exit_signal_lines:
        line_content = item['full_line'].lower()
        if ('= 1' in line_content or 
            '=1' in line_content):
            suspicious_lines.append({
                'line_num': item['line_num'],
                'content': item['content'],
                'reason': 'Exit_Signalを1に設定'
            })
    
    print(f"疑わしい処理: {len(suspicious_lines)}行")
    for item in suspicious_lines:
        print(f"  🚨 Line {item['line_num']}: {item['content']}")
        print(f"     理由: {item['reason']}")
    
    # 5. 関数定義での Exit_Signal 処理
    print(f"\n🔍 **関数定義でのExit_Signal処理**")
    
    try:
        tree = ast.parse(content)
        functions_with_exit_signal = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_start = node.lineno
                func_end = getattr(node, 'end_lineno', func_start + 10)
                
                # 関数内でExit_Signalが使用されているかチェック
                func_lines = lines[func_start-1:func_end]
                for line in func_lines:
                    if 'exit_signal' in line.lower():
                        functions_with_exit_signal.append({
                            'function': node.name,
                            'line_start': func_start,
                            'line_end': func_end
                        })
                        break
        
        print(f"Exit_Signal処理関数: {len(functions_with_exit_signal)}個")
        for func in functions_with_exit_signal:
            print(f"  - {func['function']}() (Lines {func['line_start']}-{func['line_end']})")
    
    except Exception as e:
        print(f"⚠️ AST解析エラー: {e}")
    
    # 結果サマリー
    print(f"\n" + "=" * 80)
    print(f"📊 **静的解析結果サマリー**")
    print(f"=" * 80)
    
    result = {
        'exit_signal_lines_count': len(exit_signal_lines),
        'conversion_lines_count': len(conversion_lines),
        'dataframe_lines_count': len(dataframe_lines),
        'suspicious_lines_count': len(suspicious_lines),
        'exit_signal_lines': exit_signal_lines,
        'conversion_lines': conversion_lines,
        'dataframe_lines': dataframe_lines,
        'suspicious_lines': suspicious_lines
    }
    
    print(f"Exit_Signal関連処理: {result['exit_signal_lines_count']}行")
    print(f"符号変換処理: {result['conversion_lines_count']}行")
    print(f"DataFrame操作: {result['dataframe_lines_count']}行")
    print(f"疑わしい処理: {result['suspicious_lines_count']}行")
    
    if result['suspicious_lines_count'] > 0:
        print(f"\n⚠️ **最も疑わしい処理**:")
        for item in result['suspicious_lines'][:3]:
            print(f"  Line {item['line_num']}: {item['content']}")
    
    return result

# 実行
print("🔍 main.pyソースコード静的解析を開始")
analysis_result = analyze_main_py_exit_signal_processing()

if analysis_result:
    print(f"\n✅ TODO-003 Phase 1-1 静的解析完了")
    print(f"📋 次: Phase 1-2 実行時トレーシング調査")
else:
    print(f"\n❌ TODO-003 Phase 1-1 静的解析失敗")

print("=" * 80)