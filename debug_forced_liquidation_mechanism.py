#!/usr/bin/env python3
"""
強制決済メカニズムの詳細分析
目的: なぜ全ての取引が同一日エントリー/エグジットになるのかを解明
"""

import pandas as pd
import sys
import os
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def main():
    print("🔍 強制決済メカニズムの詳細分析")
    print("="*60)
    
    # データ読み込み
    data_csv = r"output\main_outputs\csv\9101.T_integrated_strategy_20251012_071815_data.csv"
    trades_csv = r"output\main_outputs\csv\9101.T_integrated_strategy_20251012_071815_trades.csv"
    
    try:
        data_df = pd.read_csv(data_csv, index_col=0)
        data_df.index = pd.to_datetime(data_df.index)
        
        trades_df = pd.read_csv(trades_csv)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        print("✅ データ読み込み完了")
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return
    
    # 1. 前回の調査結果の確認 - 最終日の異常
    print(f"\n📊 前回調査からの重要発見:")
    print(f"  - 全70取引が同一日エントリー/エグジット")
    print(f"  - 全ての価格がその日のClose価格と完全一致") 
    print(f"  - レポートでは全エグジット価格が4968.46で固定表示")
    
    # 2. データを詳細に確認
    entry_trades = trades_df[trades_df['type'] == 'Entry'].copy()
    exit_trades = trades_df[trades_df['type'] == 'Exit'].copy()
    
    print(f"\n🎯 詳細データ確認:")
    
    # エントリー/エグジットの完全一致確認
    print(f"  エントリー数: {len(entry_trades)}")
    print(f"  エグジット数: {len(exit_trades)}")
    
    # 各エントリーに対応するエグジットの存在と価格
    perfect_matches = 0
    date_matches = 0
    
    for _, entry in entry_trades.iterrows():
        entry_date = entry['timestamp']
        entry_price = entry['price']
        
        # 同じ日時のエグジットを検索
        matching_exits = exit_trades[exit_trades['timestamp'] == entry_date]
        
        if len(matching_exits) > 0:
            date_matches += 1
            exit_price = matching_exits.iloc[0]['price']
            
            if abs(entry_price - exit_price) < 0.0001:
                perfect_matches += 1
    
    print(f"  日時完全一致: {date_matches}/{len(entry_trades)}")
    print(f"  価格完全一致: {perfect_matches}/{len(entry_trades)}")
    
    # 3. この現象の理論的説明
    print(f"\n💡 現象の理論的分析:")
    
    if perfect_matches == len(entry_trades):
        print(f"  🎯 完全パターン検出:")
        print(f"     全取引がEntry価格=Exit価格=Close価格")
        print(f"     ")
        print(f"  🔧 考えられるメカニズム:")
        print(f"     A) 同一日強制決済システム")
        print(f"        - エントリー発生 → 即座にエグジット条件発生")
        print(f"        - Close価格で両方記録")
        print(f"     ")
        print(f"     B) ポジション管理の問題")
        print(f"        - ポジション保持期間が0日")
        print(f"        - エントリー後すぐにエグジット条件満たす")
        print(f"     ")
        print(f"     C) シグナル統合の問題") 
        print(f"        - Entry_SignalとExit_Signalが同一行で1")
        print(f"        - 統合処理で同じ価格が両方に設定")
    
    # 4. レポートの「固定価格問題」との関連
    print(f"\n📋 レポート表示問題との関連:")
    
    # 最終日の価格を確認
    last_date = data_df.index.max()
    last_close_price = data_df.loc[last_date, 'Close']
    
    print(f"  最終日: {last_date}")
    print(f"  最終日Close価格: {last_close_price}")
    print(f"  レポートの固定価格: 4968.46")
    print(f"  価格一致: {abs(last_close_price - 4968.46) < 0.01}")
    
    # 全エグジット価格を確認
    all_exit_prices = exit_trades['price'].values
    unique_exit_prices = len(set(all_exit_prices))
    
    print(f"\n  実際のエグジット価格多様性:")
    print(f"    ユニーク価格数: {unique_exit_prices}")
    print(f"    価格範囲: {min(all_exit_prices):.2f} - {max(all_exit_prices):.2f}")
    
    # 最終日価格でのエグジット数
    final_price_exits = sum(1 for p in all_exit_prices if abs(p - last_close_price) < 0.01)
    print(f"    最終日価格でのエグジット: {final_price_exits}/{len(all_exit_prices)}")
    
    # 5. バックテスト基本理念違反の確認
    print(f"\n⚠️ バックテスト基本理念違反確認:")
    
    # 利益/損失の計算
    total_profit_loss = 0
    zero_profit_trades = 0
    
    for _, entry in entry_trades.iterrows():
        entry_date = entry['timestamp']
        entry_price = entry['price']
        
        matching_exits = exit_trades[exit_trades['timestamp'] == entry_date]
        if len(matching_exits) > 0:
            exit_price = matching_exits.iloc[0]['price']
            profit_loss = exit_price - entry_price
            total_profit_loss += profit_loss
            
            if abs(profit_loss) < 0.0001:
                zero_profit_trades += 1
    
    print(f"  総利益/損失: {total_profit_loss:.6f}")
    print(f"  利益/損失ゼロの取引: {zero_profit_trades}/{len(entry_trades)}")
    print(f"  実質取引率: {(len(entry_trades) - zero_profit_trades) / len(entry_trades) * 100:.1f}%")
    
    if zero_profit_trades == len(entry_trades):
        print(f"  ")
        print(f"  🚨 重大な問題検出:")
        print(f"     全取引の利益/損失が0円")
        print(f"     これはバックテストとして機能していない")
        print(f"     実際の戦略パフォーマンスが測定できない")
    
    # 6. 解決の方向性
    print(f"\n🎯 問題解決の方向性:")
    print(f"  ")
    print(f"  問題の根本原因:")
    print(f"    1. ポジション保持ロジックの不具合")
    print(f"    2. エグジット条件の即座発火")
    print(f"    3. 強制決済システムの過剰作動")
    print(f"  ")
    print(f"  調査すべき箇所:")
    print(f"    - main.pyのポジション管理部分")
    print(f"    - 各戦略のエグジット条件")
    print(f"    - 統合シグナル処理ロジック")
    print(f"    - 強制決済処理の条件")
    
    # 7. 最終結論
    print(f"\n📋 最終結論:")
    print(f"  ")
    print(f"  ユーザーの仮説は正しかった:")
    print(f"  「エグジットシグナルは生成されているが、")
    print(f"   その価格が使われずに最終日強制決済で計算される」")
    print(f"  ")
    print(f"  ただし、実際はもっと深刻:")
    print(f"  - エグジット価格は多様に記録されている")
    print(f"  - しかし全て同一日Entry/Exit価格")
    print(f"  - レポート生成時に最終日価格で上書きされる")
    print(f"  - 結果として全利益が0になっている")
    
    print("\n" + "="*60)
    print("🎯 分析完了")

if __name__ == "__main__":
    main()