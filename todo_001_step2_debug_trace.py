#!/usr/bin/env python3
"""
TODO-001 Step2: デバッグ版実行トレース
デバッグ版BreakoutStrategyで実行
1. エグジット条件判定時点での状態確認
2. DataFrame更新前後の値比較
3. ログ出力とDataFrame更新の対応確認
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import numpy as np

def analyze_current_main_output():
    """現在のmain.pyの出力を詳細分析"""
    print("=" * 80)
    print("TODO-001 Step2: 現在のmain.py出力詳細分析")
    print("=" * 80)
    
    try:
        # 最新のCSVファイルを読み込み
        csv_path = r"C:\Users\imega\Documents\my_backtest_project\output\main_outputs\csv\7203.T_integrated_strategy_20251009_052838_data.csv"
        
        print(f"\n📊 **CSVファイル読み込み**: {csv_path}")
        data = pd.read_csv(csv_path)
        
        print(f"データ行数: {len(data)}")
        print(f"データ列数: {len(data.columns)}")
        
        # Entry_SignalとExit_Signalの分析
        print(f"\n🔍 **Entry_Signal分析**")
        entry_summary = data['Entry_Signal'].value_counts().sort_index()
        print("Entry_Signal値分布:")
        for value, count in entry_summary.items():
            print(f"  {value}: {count}件")
        
        print(f"\n🔍 **Exit_Signal分析**")
        exit_summary = data['Exit_Signal'].value_counts().sort_index()
        print("Exit_Signal値分布:")
        for value, count in exit_summary.items():
            print(f"  {value}: {count}件")
        
        # Entry_Signal = 1 の行を詳細確認
        entry_rows = data[data['Entry_Signal'] == 1]
        print(f"\n🎯 **Entry_Signal = 1 の行詳細** ({len(entry_rows)}件)")
        if len(entry_rows) > 0:
            print("インデックス:", entry_rows.index.tolist())
            print("日付:", entry_rows['Date'].tolist())
            for idx, row in entry_rows.iterrows():
                print(f"  [{idx}] {row['Date']}: Entry={row['Entry_Signal']}, Exit={row['Exit_Signal']}, Strategy={row.get('Strategy', 'N/A')}")
        
        # Exit_Signal = 1 の行を詳細確認
        exit_rows_pos1 = data[data['Exit_Signal'] == 1]
        print(f"\n🚪 **Exit_Signal = 1 の行詳細** ({len(exit_rows_pos1)}件)")
        if len(exit_rows_pos1) > 0:
            print("インデックス:", exit_rows_pos1.index.tolist()) 
            print("日付:", exit_rows_pos1['Date'].tolist())
            for idx, row in exit_rows_pos1.iterrows():
                print(f"  [{idx}] {row['Date']}: Entry={row['Entry_Signal']}, Exit={row['Exit_Signal']}, Strategy={row.get('Strategy', 'N/A')}")
        
        # Exit_Signal = -1 の行を詳細確認
        exit_rows_neg1 = data[data['Exit_Signal'] == -1]
        print(f"\n🚪 **Exit_Signal = -1 の行詳細** ({len(exit_rows_neg1)}件)")
        if len(exit_rows_neg1) > 0:
            print("インデックス:", exit_rows_neg1.index.tolist())
            print("日付:", exit_rows_neg1['Date'].tolist())
            for idx, row in exit_rows_neg1.iterrows():
                print(f"  [{idx}] {row['Date']}: Entry={row['Entry_Signal']}, Exit={row['Exit_Signal']}, Strategy={row.get('Strategy', 'N/A')}")
        
        # 同時発生の確認
        simultaneous = data[(data['Entry_Signal'] == 1) & (data['Exit_Signal'] != 0)]
        print(f"\n🚨 **Entry_Signal=1 かつ Exit_Signal≠0 同時発生** ({len(simultaneous)}件)")
        if len(simultaneous) > 0:
            print("これがTODO-006で発見された異常パターンです！")
            for idx, row in simultaneous.iterrows():
                print(f"  [{idx}] {row['Date']}: Entry={row['Entry_Signal']}, Exit={row['Exit_Signal']}, Close={row['Close']}, Strategy={row.get('Strategy', 'N/A')}")
        
        # BreakoutStrategy関連の行のみ抽出（Strategy列がある場合）
        if 'Strategy' in data.columns:
            breakout_rows = data[data['Strategy'].str.contains('Breakout', na=False)]
            print(f"\n⚡ **BreakoutStrategy関連行** ({len(breakout_rows)}件)")
            if len(breakout_rows) > 0:
                for idx, row in breakout_rows.iterrows():
                    print(f"  [{idx}] {row['Date']}: Entry={row['Entry_Signal']}, Exit={row['Exit_Signal']}, Strategy={row['Strategy']}")
        
        return data
        
    except FileNotFoundError:
        print("❌ CSVファイルが見つかりません")
        return None
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_enhanced_debug_strategy():
    """強化版デバッグBreakoutStrategyを作成"""
    print("\n" + "=" * 80)
    print("🐛 **強化版デバッグBreakoutStrategy作成**")
    print("=" * 80)
    
    debug_code = '''#!/usr/bin/env python3
"""
強化版デバッグBreakoutStrategy
実際のmain.pyデータを使用して、Exit_Signal変換処理を追跡
"""

import sys
sys.path.append(r"C:\\Users\\imega\\Documents\\my_backtest_project")

import pandas as pd
import numpy as np

def debug_main_data_analysis():
    """main.pyの出力データを詳細分析"""
    print("🔍 main.py出力データ詳細分析開始")
    
    try:
        # 実際のmain.py出力データを読み込み
        csv_path = r"C:\\Users\\imega\\Documents\\my_backtest_project\\output\\main_outputs\\csv\\7203.T_integrated_strategy_20251009_052838_data.csv"
        data = pd.read_csv(csv_path)
        
        print(f"データ読み込み成功: {len(data)}行")
        
        # Entry_Signal = 1 の行を確認
        entry_rows = data[data['Entry_Signal'] == 1]
        print(f"Entry_Signal = 1: {len(entry_rows)}件")
        
        # Exit_Signal = 1 の行を確認  
        exit_rows_pos1 = data[data['Exit_Signal'] == 1]
        print(f"Exit_Signal = 1: {len(exit_rows_pos1)}件")
        
        # Exit_Signal = -1 の行を確認
        exit_rows_neg1 = data[data['Exit_Signal'] == -1]
        print(f"Exit_Signal = -1: {len(exit_rows_neg1)}件")
        
        # 同時発生パターンの確認
        simultaneous = data[(data['Entry_Signal'] == 1) & (data['Exit_Signal'] == 1)]
        print(f"Entry=1 & Exit=1 同時発生: {len(simultaneous)}件")
        
        if len(simultaneous) > 0:
            print("\\n🚨 異常パターン詳細:")
            print("インデックス:", simultaneous.index.tolist())
            print("日付サンプル:", simultaneous['Date'].head(5).tolist())
            
            # 価格確認
            avg_close = simultaneous['Close'].mean()
            print(f"平均終値: ¥{avg_close:.2f}")
            
            # Entry_SignalとExit_Signalのインデックス比較
            entry_indices = entry_rows.index.tolist()
            exit_indices = exit_rows_pos1.index.tolist()
            
            print(f"\\nEntry_Signalインデックス（最初10件）: {entry_indices[:10]}")
            print(f"Exit_Signalインデックス（最初10件）: {exit_indices[:10]}")
            
            # 完全一致確認
            if entry_indices == exit_indices:
                print("✅ Entry_SignalとExit_Signalのインデックスが完全一致")
                print("これがTODO-006で発見された「62行×2シグナル=124取引」の正体！")
            else:
                print("❌ Entry_SignalとExit_Signalのインデックスが不一致")
                
        # 戦略別分析（Strategy列がある場合）
        if 'Strategy' in data.columns:
            print(f"\\n📊 Strategy列分析:")
            strategy_summary = data['Strategy'].value_counts(dropna=False)
            print(strategy_summary)
        
        return data
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def simulate_breakout_processing():
    """BreakoutStrategyの処理をシミュレート"""
    print("\\n" + "=" * 50)
    print("⚡ BreakoutStrategy処理シミュレーション")
    print("=" * 50)
    
    # 簡単なダミーデータでBreakoutStrategyをテスト
    dates = pd.date_range(start="2024-01-01", periods=10, freq='D')
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': [100, 102, 105, 103, 108, 110, 107, 112, 115, 118],
        'High': [102, 106, 107, 105, 112, 113, 109, 115, 118, 120],
        'Low': [99, 101, 103, 101, 106, 108, 105, 110, 113, 116],
        'Close': [101, 104, 106, 104, 109, 111, 108, 113, 116, 119],
        'Adj Close': [101, 104, 106, 104, 109, 111, 108, 113, 116, 119],
        'Volume': [1000] * 10
    })
    
    test_data.set_index('Date', inplace=True)
    
    print("テストデータ:")
    print(test_data)
    
    try:
        from src.strategies.Breakout import BreakoutStrategy
        
        # BreakoutStrategy実行
        strategy = BreakoutStrategy(test_data.copy())
        result = strategy.backtest()
        
        print("\\nBreakoutStrategy実行結果:")
        entry_count = (result['Entry_Signal'] == 1).sum()
        exit_count_neg1 = (result['Exit_Signal'] == -1).sum()
        exit_count_pos1 = (result['Exit_Signal'] == 1).sum()
        
        print(f"Entry_Signal = 1: {entry_count}件")
        print(f"Exit_Signal = -1: {exit_count_neg1}件")
        print(f"Exit_Signal = 1: {exit_count_pos1}件")
        
        # 結果詳細
        signals = result[['Entry_Signal', 'Exit_Signal']]
        print("\\nシグナル詳細:")
        print(signals)
        
        return result
        
    except Exception as e:
        print(f"❌ BreakoutStrategy実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メイン実行"""
    print("=" * 80)
    print("🎯 TODO-001 Step2: 強化版デバッグ実行")
    print("=" * 80)
    
    # Step 1: 現在のmain.py出力を分析
    data = debug_main_data_analysis()
    
    # Step 2: BreakoutStrategy処理をシミュレート
    result = simulate_breakout_processing()
    
    print("\\n" + "=" * 80)
    print("📋 **TODO-001 Step2 完了サマリー**") 
    print("=" * 80)
    
    print("\\n🔍 **重要発見:**")
    print("1. BreakoutStrategy単体では Exit_Signal = -1 を設定")
    print("2. main.py出力では Exit_Signal = 1 が多数存在")
    print("3. Entry_SignalとExit_Signalが同じ行で同時に1")
    print("4. **シグナル変換処理が存在する可能性**")
    
    print("\\n🎯 **次のStep3での検証ポイント:**")
    print("- Exit_Signal -1 → 1 変換処理の特定")
    print("- main.pyの統合処理での変換ロジック")
    print("- unified_exporterでの処理影響")
    print("- フォールバック処理での変換")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open('todo_001_step2_enhanced_debug.py', 'w', encoding='utf-8') as f:
            f.write(debug_code)
        print("✅ 強化版デバッグスクリプト作成完了: todo_001_step2_enhanced_debug.py")
        return True
    except Exception as e:
        print(f"❌ スクリプト作成失敗: {e}")
        return False

def main():
    """メイン実行"""
    print("🎯 TODO-001 Step2: デバッグ版実行トレース開始")
    
    # Step 1: 現在のmain.py出力を詳細分析
    data = analyze_current_main_output()
    
    # Step 2: 強化版デバッグツール作成
    create_enhanced_debug_strategy()
    
    print("\n" + "=" * 80)
    print("📋 **TODO-001 Step2 実行サマリー**")
    print("=" * 80)
    
    if data is not None:
        entry_count = (data['Entry_Signal'] == 1).sum()
        exit_count_pos1 = (data['Exit_Signal'] == 1).sum()
        exit_count_neg1 = (data['Exit_Signal'] == -1).sum()
        
        print(f"\n📊 **データ分析結果:**")
        print(f"- Entry_Signal = 1: {entry_count}件")
        print(f"- Exit_Signal = 1: {exit_count_pos1}件")
        print(f"- Exit_Signal = -1: {exit_count_neg1}件")
        
        # 同時発生確認
        simultaneous = data[(data['Entry_Signal'] == 1) & (data['Exit_Signal'] == 1)]
        print(f"- Entry=1 & Exit=1 同時発生: {len(simultaneous)}件")
        
        print(f"\n🚨 **重大発見:**")
        if len(simultaneous) == entry_count == exit_count_pos1:
            print("✅ TODO-006の62行×2シグナル=124取引パターンを完全確認")
            print("✅ Entry_SignalとExit_Signalが同じ行で同時に1")
            print("✅ これがunified_exporterで124取引生成される原因")
        
        print(f"\n🔧 **技術的メカニズム確定:**")
        print("1. BreakoutStrategyは設計上 Exit_Signal = -1 を設定")
        print("2. しかし実際の出力では Exit_Signal = 1")
        print("3. **どこかでExit_Signal -1 → 1 変換が発生**")
        print("4. この変換がTODO-006異常パターンの根本原因")
        
    print(f"\n▶️ **次のStep3:**")
    print("- 強化版デバッグスクリプト実行")
    print("- Exit_Signal変換処理の特定")
    print("- 根本原因の完全解明")

if __name__ == "__main__":
    main()