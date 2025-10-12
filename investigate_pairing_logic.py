#!/usr/bin/env python3
"""
unified_exporter ペアリングロジック調査ツール
目的: なぜエントリー62件、エグジット62件が生成されたのかを詳細調査
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
from pathlib import Path

def investigate_unified_exporter_logic():
    """
    unified_exporterのペアリングロジックを推測分析
    """
    print("=" * 60)
    print("🔍 unified_exporter ペアリングロジック調査")
    print("=" * 60)
    
    # データファイルの詳細分析
    data_path = Path(r"C:\Users\imega\Documents\my_backtest_project\output\main_outputs\csv\7203.T_integrated_strategy_20251008_232614_data.csv")
    
    try:
        data_df = pd.read_csv(data_path)
        
        print(f"📊 データファイル基本情報:")
        print(f"  - 総行数: {len(data_df)}")
        print(f"  - Entry_Signal=1の行数: {(data_df['Entry_Signal'] == 1).sum()}")
        print(f"  - Exit_Signal=1の行数: {(data_df['Exit_Signal'] == 1).sum()}")
        
        # Entry_SignalとExit_Signalが同じ日付にあるかチェック
        entry_indices = data_df[data_df['Entry_Signal'] == 1].index
        exit_indices = data_df[data_df['Exit_Signal'] == 1].index
        
        print(f"\n📅 シグナル発生インデックス比較:")
        print(f"  - Entry_Signalインデックス（最初の10個）: {list(entry_indices[:10])}")
        print(f"  - Exit_Signalインデックス（最初の10個）: {list(exit_indices[:10])}")
        
        # 同じインデックスで両方のシグナルが1かチェック
        same_index_both = data_df[(data_df['Entry_Signal'] == 1) & (data_df['Exit_Signal'] == 1)]
        print(f"\n🚨 重大発見 - 同じ行でentry=1かつexit=1:")
        print(f"  - 該当行数: {len(same_index_both)}")
        
        if len(same_index_both) > 0:
            print(f"  - サンプル（最初の5行）:")
            cols_to_show = ['Date', 'Entry_Signal', 'Exit_Signal', 'Strategy'] if 'Date' in data_df.columns else ['Entry_Signal', 'Exit_Signal', 'Strategy']
            if 'Strategy' not in data_df.columns:
                cols_to_show = ['Entry_Signal', 'Exit_Signal']
            print(same_index_both[cols_to_show].head().to_string())
        
        # 価格情報の確認
        if 'Close' in data_df.columns:
            entry_prices = data_df[data_df['Entry_Signal'] == 1]['Close']
            exit_prices = data_df[data_df['Exit_Signal'] == 1]['Close']
            
            print(f"\n💰 価格分析:")
            print(f"  - Entry時の価格（最初の5個）: {list(entry_prices.head())}")
            print(f"  - Exit時の価格（最初の5個）: {list(exit_prices.head())}")
            
            if len(entry_prices) > 0 and len(exit_prices) > 0:
                print(f"  - Entry平均価格: ¥{entry_prices.mean():.2f}")
                print(f"  - Exit平均価格: ¥{exit_prices.mean():.2f}")
        
        return data_df, same_index_both
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None, None

def analyze_contradiction_source():
    """
    矛盾の根本原因分析
    """
    print(f"\n" + "=" * 60)
    print("🎯 矛盾の根本原因分析")
    print("=" * 60)
    
    print(f"📋 調査結果まとめ:")
    print(f"  1. 前回調査 (test_main_initialization.py): エントリー81, エグジット0")
    print(f"  2. 今回main.py実行: unified_exporterがエントリー62, エグジット62と報告")
    print(f"  3. 実際の出力ファイル: Entry_Signal=1が62件, Exit_Signal=1が62件")
    
    print(f"\n🔍 重要な仮説:")
    print(f"  A. unified_exporterは正しく動作し、実際にexit signalが生成されている")
    print(f"  B. 前回調査のtest_main_initialization.pyに問題があった")
    print(f"  C. main.pyの何らかの処理でexit signalが後から追加された")
    print(f"  D. unified_exporterが独自の処理でexit signalを生成している")
    
    print(f"\n📊 数値の矛盾:")
    print(f"  - エントリー: 81 → 62 (差: -19)")
    print(f"  - エグジット: 0 → 62 (差: +62)")
    print(f"  - この劇的変化は正常な処理では説明困難")

def main():
    print("🔍 unified_exporter ペアリングロジック調査開始")
    
    # データファイル詳細分析
    data_df, same_index_both = investigate_unified_exporter_logic()
    
    # 矛盾分析
    analyze_contradiction_source()
    
    # 結論
    print(f"\n" + "=" * 60)
    print("🎯 調査結論")
    print("=" * 60)
    
    if same_index_both is not None and len(same_index_both) > 0:
        print(f"🚨 重大発見:")
        print(f"  - {len(same_index_both)}行で Entry_Signal=1 かつ Exit_Signal=1")
        print(f"  - これは同じ日にentry/exitが発生することを意味")
        print(f"  - unified_exporterがこれをペアとして扱い、124取引を生成")
        
        print(f"\n💡 推測される処理フロー:")
        print(f"  1. 何らかの処理で62行にEntry_Signal=1とExit_Signal=1が同時設定")
        print(f"  2. unified_exporterが各行を個別にentryとexitとして処理")
        print(f"  3. 結果: 62 entries + 62 exits = 124 total trades")
        print(f"  4. ペアリング: 62ペア中61が正常ペア、1が未ペア")
    
    print(f"\n🔍 次に調査すべき項目:")
    print(f"  1. なぜ前回81エントリーが今回62エントリーになったか")
    print(f"  2. なぜ同じ行でentry/exitが同時に1になっているか") 
    print(f"  3. unified_exporterのペアリング処理は正しいか")
    print(f"  4. フォールバック処理の影響はあるか")

if __name__ == "__main__":
    main()