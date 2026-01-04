"""
Priority C3-1調査スクリプト - 統合実行時メソッド特定

統合実行時に実際に呼び出される_get_optimal_symbol()メソッドの場所・内容を特定し、
完全版メソッド（Line 1554-1621）との差異を明確化するための詳細調査スクリプト

主な調査対象:
- 統合実行時の実際のメソッド呼び出し先
- メソッド定義場所・内容の特定
- クラス継承・オーバーライド構造の解析
- メソッド解決順序（MRO）の確認

統合コンポーネント:
- src.dssms.dssms_integrated_main: DSSMSIntegratedBacktester
- inspect: メソッド情報取得
- logging: 調査ログ出力

セーフティ機能/注意事項:
- 実行時の動的メソッド解決を正確に追跡
- 複数のメソッド定義がある場合の優先順位確認
- 統合実行環境と同一条件でのメソッド特定保証

Author: Backtest Project Team
Created: 2026-01-04
Last Modified: 2026-01-04
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import inspect
import logging
from datetime import datetime
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

# ログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_method_resolution():
    """Priority C3-1: 統合実行時メソッド特定調査"""
    print("=== Priority C3-1調査：統合実行時メソッド特定 ===")
    
    target_date = datetime(2025, 1, 15)
    
    try:
        # Step 1: 統合実行版バックテスターインスタンス作成
        print(f"\n[C3-1-STEP1] 統合実行版DSSMSIntegratedBacktesterインスタンス化")
        backtester = DSSMSIntegratedBacktester()
        print(f"✅ インスタンス化成功: {type(backtester).__name__}")
        
        # Step 2: メソッド解決分析
        print(f"\n[C3-1-STEP2] メソッド解決分析")
        
        # 2-1: 実際のメソッドオブジェクト取得
        actual_method = backtester._get_optimal_symbol
        print(f"[C3-1-STEP2-1] 実際のメソッドオブジェクト:")
        print(f"  - メソッド: {actual_method}")
        print(f"  - メソッド型: {type(actual_method)}")
        
        # 2-2: メソッド定義場所特定
        try:
            method_func = actual_method.__func__
            source_file = inspect.getfile(method_func)
            source_lines = inspect.getsourcelines(method_func)
            line_number = source_lines[1]
            print(f"[C3-1-STEP2-2] メソッド定義場所:")
            print(f"  - ファイル: {source_file}")
            print(f"  - 開始行: {line_number}")
            print(f"  - 行数: {len(source_lines[0])}")
            
            # 実際のソースコード表示（最初の10行）
            print(f"[C3-1-STEP2-2] 実際のソースコード（最初の10行）:")
            for i, line in enumerate(source_lines[0][:10], start=line_number):
                print(f"    {i}: {line.rstrip()}")
                
        except Exception as e:
            print(f"[C3-1-STEP2-2] メソッド定義場所取得エラー: {e}")
        
        # 2-3: クラス継承構造（MRO）確認
        print(f"[C3-1-STEP2-3] クラス継承構造（MRO）:")
        mro = type(backtester).__mro__
        for i, cls in enumerate(mro):
            print(f"  {i+1}. {cls.__name__} ({cls.__module__})")
            
            # 各クラスで_get_optimal_symbolメソッドが定義されているか確認
            if hasattr(cls, '_get_optimal_symbol'):
                method_in_class = getattr(cls, '_get_optimal_symbol', None)
                if method_in_class and hasattr(method_in_class, '__func__'):
                    try:
                        class_source_file = inspect.getfile(method_in_class.__func__)
                        class_line_number = inspect.getsourcelines(method_in_class.__func__)[1]
                        print(f"     → _get_optimal_symbol定義あり: {class_source_file}:{class_line_number}")
                    except:
                        print(f"     → _get_optimal_symbol定義あり（場所取得失敗）")
                else:
                    print(f"     → _get_optimal_symbol定義なし")
        
        # Step 3: 完全版メソッド（Line 1554-1621）との比較
        print(f"\n[C3-1-STEP3] 完全版メソッドとの比較")
        
        # 3-1: 期待される完全版メソッドの場所確認
        expected_file = "src\\dssms\\dssms_integrated_main.py"
        expected_line = 1554
        
        try:
            if hasattr(actual_method, '__func__'):
                actual_file = inspect.getfile(actual_method.__func__)
                actual_line = inspect.getsourcelines(actual_method.__func__)[1]
                
                print(f"[C3-1-STEP3-1] メソッド場所比較:")
                print(f"  期待（完全版）: ...{expected_file}:{expected_line}")
                print(f"  実際: {actual_file}:{actual_line}")
                
                # ファイル名での判定（パス区切り文字対応）
                actual_file_normalized = actual_file.replace('/', '\\').replace('\\\\', '\\')
                is_same_file = expected_file in actual_file_normalized or actual_file_normalized.endswith(expected_file)
                is_same_line = actual_line == expected_line
                
                print(f"[C3-1-STEP3-1] 比較結果:")
                print(f"  - 同一ファイル: {is_same_file}")
                print(f"  - 同一行番号: {is_same_line}")
                print(f"  - 完全版メソッド判定: {'✅ YES' if (is_same_file and is_same_line) else '❌ NO'}")
                
                if not (is_same_file and is_same_line):
                    print(f"🚨 [CRITICAL] 統合実行時に異なるメソッドが呼び出されている！")
                    print(f"  - 期待: 完全版メソッド（DSS Core V3機能付き）")
                    print(f"  - 実際: 別メソッド（機能制限版の可能性）")
                
        except Exception as e:
            print(f"[C3-1-STEP3-1] 比較エラー: {e}")
        
        # Step 4: メソッド実行テスト
        print(f"\n[C3-1-STEP4] メソッド実行テスト")
        
        try:
            print(f"[C3-1-STEP4] 統合実行時メソッドの実行テスト:")
            print(f"  - 呼び出し: backtester._get_optimal_symbol({target_date}, None)")
            
            # 実際の統合実行時と同じ方式で呼び出し
            result = backtester._get_optimal_symbol(target_date, None)
            
            print(f"[C3-1-STEP4] 実行結果:")
            print(f"  - 戻り値: {result}")
            print(f"  - 戻り値型: {type(result)}")
            print(f"  - 成功判定: {'✅ SUCCESS' if result else '❌ FAILED (None returned)'}")
            
        except Exception as e:
            print(f"[C3-1-STEP4] メソッド実行エラー: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'method_object': actual_method,
            'method_location': f"{actual_file}:{actual_line}" if 'actual_file' in locals() else 'unknown',
            'is_complete_version': (is_same_file and is_same_line) if 'is_same_file' in locals() else False,
            'execution_result': result if 'result' in locals() else None,
            'mro': [cls.__name__ for cls in mro]
        }
        
    except Exception as e:
        print(f"❌ Priority C3-1調査実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Priority C3-1調査スクリプト実行開始")
    result = analyze_method_resolution()
    
    print(f"\n=== Priority C3-1調査結果サマリー ===")
    if result:
        print(f"メソッド場所: {result.get('method_location', 'unknown')}")
        print(f"完全版判定: {result.get('is_complete_version', False)}")
        print(f"実行結果: {result.get('execution_result', 'None')}")
        print(f"クラス継承: {' → '.join(result.get('mro', []))}")
    else:
        print("調査失敗")
    
    print(f"\n✅ Priority C3-1調査完了")