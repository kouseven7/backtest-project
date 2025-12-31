import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def analyze_syntax_errors():
    """構文エラー分析"""
    with open('src/dssms/dssms_integrated_main.py', 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    print("=== 構文エラー分析結果 ===")
    
    # 正確な文字数カウント
    patterns = {
        '全角括弧開': '（',
        '全角括弧閉': '）', 
        '句読点_読点': '、',
        '句読点_句点': '。',
        '矢印文字': '→'
    }
    
    for name, char in patterns.items():
        count = content.count(char)
        print(f'{name} ({repr(char)}): {count}箇所')
    
    # 先頭ゼロ数字の確認
    import re
    zero_matches = re.findall(r'TODO-PERF-0+\d+', content)
    print(f'先頭ゼロ数字: {len(zero_matches)}箇所')
    
    # 最初の構文エラー箇所
    print("\n=== 最初の構文エラー箇所 ===")
    for i, line in enumerate(lines[:70], 1):
        if '（' in line or '）' in line:
            print(f'Line {i}: {line.strip()}')
            if i == 63:  # 最初の構文エラー行
                break
    
    print(f"\n総行数: {len(lines)}")
    print(f"ファイルサイズ: {len(content)}文字")

if __name__ == "__main__":
    analyze_syntax_errors()