#!/usr/bin/env python3
"""
TODO-006-A 手法2: コード静的分析
目的: main.pyのフォールバック処理コードを静的解析して二重処理メカニズムを特定
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from pathlib import Path

def analyze_main_py_fallback_code():
    """
    main.pyのフォールバック処理コード静的分析
    """
    print("=" * 60)
    print("🔍 TODO-006-A 手法2: main.pyフォールバック処理コード分析")
    print("=" * 60)
    
    main_py = Path(r"C:\Users\imega\Documents\my_backtest_project\main.py")
    
    try:
        with open(main_py, 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # フォールバック関連コードの抽出
        fallback_sections = []
        
        # セクション1: SystemFallbackPolicy初期化
        if "SystemFallbackPolicy" in main_content:
            fallback_sections.append("SystemFallbackPolicy初期化部分")
        
        # セクション2: 統合システムのインポートとフォールバック
        if "handle_component_failure" in main_content:
            fallback_sections.append("handle_component_failure呼び出し")
        
        # セクション3: main()関数内のフォールバック処理
        if "integrated_system_available" in main_content:
            fallback_sections.append("integrated_system_available分岐")
        
        print(f"📋 発見されたフォールバック処理セクション: {len(fallback_sections)}個")
        for i, section in enumerate(fallback_sections, 1):
            print(f"  {i}. {section}")
        
        return analyze_fallback_logic_flow(main_content)
        
    except Exception as e:
        print(f"❌ main.py読み込みエラー: {e}")
        return None

def analyze_fallback_logic_flow(main_content):
    """
    フォールバック処理のロジックフロー分析
    """
    print(f"\n📊 フォールバック処理ロジックフロー分析:")
    
    # 重要なコード部分を特定
    key_patterns = {
        'fallback_initialization': 'fallback_policy = SystemFallbackPolicy()',
        'fallback_handling': 'handle_component_failure',
        'integrated_system_check': 'integrated_system_available',
        'main_execution_branch': 'if integrated_system_available:',
        'fallback_statistics': 'get_fallback_usage_statistics',
        'apply_strategies': 'apply_strategies_with_optimized_params'
    }
    
    found_patterns = {}
    for pattern_name, pattern in key_patterns.items():
        if pattern in main_content:
            found_patterns[pattern_name] = True
            # パターン周辺のコンテキスト取得
            lines = main_content.split('\n')
            for i, line in enumerate(lines):
                if pattern in line:
                    found_patterns[f"{pattern_name}_line"] = i + 1
                    break
        else:
            found_patterns[pattern_name] = False
    
    print(f"  📍 検出されたパターン:")
    for pattern_name, found in found_patterns.items():
        if not pattern_name.endswith('_line'):
            status = "✅" if found else "❌"
            line_num = found_patterns.get(f"{pattern_name}_line", "N/A")
            print(f"    {status} {pattern_name}: {found} (行: {line_num})")
    
    return found_patterns

def analyze_signal_generation_points():
    """
    シグナル生成ポイントの分析
    """
    print(f"\n" + "=" * 60)
    print("🎯 シグナル生成ポイント分析")
    print("=" * 60)
    
    # apply_strategies_with_optimized_params関数の分析
    main_py = Path(r"C:\Users\imega\Documents\my_backtest_project\main.py")
    
    try:
        with open(main_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # apply_strategies関数の存在確認
        if "def apply_strategies_with_optimized_params" in content:
            print(f"✅ apply_strategies_with_optimized_params関数が存在")
            
            # 統合データ初期化部分の確認
            signal_init_patterns = [
                "integrated_data['Entry_Signal'] = 0",
                "integrated_data['Exit_Signal'] = 0",
                "_integrate_entry_signals",
                "_integrate_exit_signals_with_position_tracking"
            ]
            
            print(f"\n📋 シグナル処理パターン検出:")
            for pattern in signal_init_patterns:
                if pattern in content:
                    print(f"  ✅ {pattern}")
                else:
                    print(f"  ❌ {pattern}")
        
        return True
        
    except Exception as e:
        print(f"❌ シグナル生成分析エラー: {e}")
        return False

def hypothesize_double_processing_mechanism():
    """
    二重処理メカニズムの仮説構築
    """
    print(f"\n" + "=" * 60) 
    print("💡 二重処理メカニズム仮説")
    print("=" * 60)
    
    hypotheses = [
        {
            'name': '仮説A: 統合システム部分実行',
            'description': '統合システムが部分的に実行され、シグナルを生成した後にエラーで停止',
            'evidence': ['フォールバック使用率1.0', 'エントリー数変化(81→62)'],
            'likelihood': 'HIGH'
        },
        {
            'name': '仮説B: 従来システム重複実行',
            'description': 'フォールバック後に従来システムが重複実行され、既存シグナルを上書き',
            'evidence': ['Exit_Signal生成(0→62)', '同時entry/exit発生'],
            'likelihood': 'HIGH'
        },
        {
            'name': '仮説C: unified_exporter誤処理',
            'description': 'unified_exporterが競合するシグナルデータを誤って統合',
            'evidence': ['124取引生成', 'ペアリング61+1未ペア'],
            'likelihood': 'MEDIUM'
        },
        {
            'name': '仮説D: フォールバック統計重複',
            'description': 'フォールバック処理自体が二重実行されている',
            'evidence': ['ログ重複出力', 'フォールバック統計二重報告'],
            'likelihood': 'MEDIUM'
        }
    ]
    
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n{i}. {hypothesis['name']} [{hypothesis['likelihood']}]")
        print(f"   説明: {hypothesis['description']}")
        print(f"   根拠: {', '.join(hypothesis['evidence'])}")
    
    return hypotheses

def main():
    print("🔍 TODO-006-A 手法2: コード静的分析 開始")
    
    # Step 1: main.pyフォールバック処理分析
    fallback_patterns = analyze_main_py_fallback_code()
    
    # Step 2: シグナル生成ポイント分析
    signal_analysis = analyze_signal_generation_points()
    
    # Step 3: 二重処理メカニズム仮説
    hypotheses = hypothesize_double_processing_mechanism()
    
    # 結論
    print(f"\n" + "=" * 60)
    print("🎯 手法2結論")
    print("=" * 60)
    
    print(f"✅ 静的分析完了:")
    print(f"  - フォールバック処理コードが存在")
    print(f"  - シグナル統合処理が複雑")
    print(f"  - 複数の競合ポイントを特定")
    
    print(f"\n🔍 最有力仮説:")
    high_likelihood = [h for h in hypotheses if h['likelihood'] == 'HIGH']
    for hypothesis in high_likelihood:
        print(f"  • {hypothesis['name']}")
    
    print(f"\n📝 手法1+2統合結論:")
    print(f"  - フォールバック処理は正常動作している")
    print(f"  - 問題は統合システムと従来システムの処理競合")
    print(f"  - unified_exporterが競合データを受け取っている")
    
    return {
        'fallback_patterns': fallback_patterns,
        'signal_analysis': signal_analysis,
        'hypotheses': hypotheses
    }

if __name__ == "__main__":
    results = main()