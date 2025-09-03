import sys
import os
sys.path.append('src')

def check_dssms_calculation_methods():
    """DSSMS計算ロジックの確認"""
    print("=" * 80)
    print("DSSMS計算ロジック確認")
    print("=" * 80)
    
    # vs_戦略比較の計算ロジックを確認
    backtester_file = "src/dssms/dssms_backtester.py"
    
    if not os.path.exists(backtester_file):
        print(f"❌ {backtester_file} が見つかりません")
        return
    
    with open(backtester_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # vs_戦略計算の箇所を検索
    print("🔍 vs_戦略計算ロジックの検索:")
    
    vs_calculation_lines = []
    for i, line in enumerate(lines):
        if any(pattern in line.lower() for pattern in ['vs_', 'compare_with', 'benchmark', 'static_strategy']):
            vs_calculation_lines.append(f"{i+1:4d}: {line.strip()}")
    
    print(f"   発見された関連行数: {len(vs_calculation_lines)}")
    
    if vs_calculation_lines:
        print("\n📋 vs_戦略計算関連コード:")
        for line in vs_calculation_lines[:15]:  # 最初の15行を表示
            print(f"   {line}")
    
    # ランダム要素の使用確認
    print("\n🎲 ランダム要素の確認:")
    random_lines = []
    for i, line in enumerate(lines):
        if any(pattern in line.lower() for pattern in ['random', 'shuffle', 'choice', 'seed', 'np.random']):
            random_lines.append(f"{i+1:4d}: {line.strip()}")
    
    if random_lines:
        print("   ⚠️ ランダム要素発見:")
        for line in random_lines[:10]:
            print(f"   {line}")
    else:
        print("   ✅ ランダム要素なし")
    
    # 日時・時刻関連の処理確認
    print("\n📅 日時処理の確認:")
    datetime_lines = []
    for i, line in enumerate(lines):
        if any(pattern in line.lower() for pattern in ['datetime', 'timestamp', 'now()', 'today()']):
            datetime_lines.append(f"{i+1:4d}: {line.strip()}")
    
    if datetime_lines:
        print("   日時処理関連行:")
        for line in datetime_lines[:10]:
            print(f"   {line}")
    
    # データ取得の一貫性確認
    print("\n📊 データ取得の確認:")
    data_fetch_lines = []
    for i, line in enumerate(lines):
        if any(pattern in line.lower() for pattern in ['yfinance', 'download', 'fetch_stock_data', '.get_data']):
            data_fetch_lines.append(f"{i+1:4d}: {line.strip()}")
    
    if data_fetch_lines:
        print("   データ取得関連行:")
        for line in data_fetch_lines[:10]:
            print(f"   {line}")

def check_excel_output_issue():
    """Excel出力の問題確認"""
    print("\n" + "=" * 80)
    print("Excel出力問題確認")
    print("=" * 80)
    
    # 最新のExcelファイルが存在するか確認
    excel_files = []
    results_dir = "backtest_results/dssms_results"
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.xlsx') and 'v2' in file:
                file_path = os.path.join(results_dir, file)
                size = os.path.getsize(file_path)
                mtime = os.path.getmtime(file_path)
                excel_files.append({
                    'file': file,
                    'size': size,
                    'mtime': mtime
                })
    
    if excel_files:
        # 最新ファイルを確認
        latest_file = max(excel_files, key=lambda x: x['mtime'])
        print(f"📄 最新Excelファイル: {latest_file['file']}")
        print(f"   サイズ: {latest_file['size']:,} bytes")
        
        if latest_file['size'] < 10000:  # 10KB未満は異常に小さい
            print("   ⚠️ ファイルサイズが異常に小さい（データが正しく出力されていない可能性）")
        else:
            print("   ✅ ファイルサイズは正常")
    else:
        print("❌ v2 Excelファイルが見つかりません")

def find_error_source():
    """エラーの原因を特定"""
    print("\n" + "=" * 80)
    print("エラー原因特定")
    print("=" * 80)
    
    # _prepare_dssms_results_for_excel_v2メソッドの詳細確認
    backtester_file = "src/dssms/dssms_backtester.py"
    
    if not os.path.exists(backtester_file):
        print(f"❌ {backtester_file} が見つかりません")
        return
    
    with open(backtester_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # エラーメッセージに関連する箇所を検索
    print("🚨 エラー関連コードの検索:")
    print("   エラー: 'unsupported operand type(s) for -: 'dict' and 'int'")
    
    error_lines = []
    for i, line in enumerate(lines):
        # 辞書と整数の減算が発生しそうな箇所
        if any(pattern in line for pattern in [' - ', '- initial_capital', '- int(', 'dict -']):
            error_lines.append(f"{i+1:4d}: {line.strip()}")
    
    print(f"\n   潜在的エラー箇所: {len(error_lines)}行")
    for line in error_lines[:15]:
        print(f"   {line}")
    
    # _prepare_dssms_results_for_excel_v2メソッドを詳細確認
    method_start = None
    for i, line in enumerate(lines):
        if 'def _prepare_dssms_results_for_excel_v2' in line:
            method_start = i
            break
    
    if method_start:
        print(f"\n📍 _prepare_dssms_results_for_excel_v2 メソッド詳細:")
        print(f"   開始行: {method_start + 1}")
        
        # メソッドの内容を詳細表示
        method_lines = []
        indent_level = None
        
        for i in range(method_start, min(method_start + 100, len(lines))):
            line = lines[i]
            if i == method_start:
                indent_level = len(line) - len(line.lstrip())
                method_lines.append(f"{i+1:4d}: {line}")
            elif line.strip() == "":
                method_lines.append(f"{i+1:4d}: {line}")
            elif len(line) - len(line.lstrip()) > indent_level:
                method_lines.append(f"{i+1:4d}: {line}")
            else:
                break
        
        print("\n📋 メソッド全体:")
        for line in method_lines:
            print(f"   {line}")
            
        # 特にエラーが発生しやすい計算部分を強調
        print("\n⚠️ エラー発生可能性の高い行:")
        for line in method_lines:
            if any(pattern in line for pattern in [' - ', 'portfolio_value', 'initial_capital', 'dict']):
                print(f"   >>> {line}")
    
    else:
        print("   ❌ _prepare_dssms_results_for_excel_v2 メソッドが見つかりません")

if __name__ == "__main__":
    check_dssms_calculation_methods()
    check_excel_output_issue()
    find_error_source()
