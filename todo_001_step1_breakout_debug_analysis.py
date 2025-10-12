#!/usr/bin/env python3
"""
TODO-001 Step1: BreakoutStrategy詳細解析
BreakoutStrategy.pyのbacktest()メソッド解析
1. ログ出力箇所の特定
2. Exit_Signal列更新処理の確認
3. DataFrame操作の流れ追跡
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import numpy as np
from src.strategies.Breakout import BreakoutStrategy
import yfinance as yf

def analyze_breakout_code_structure():
    """BreakoutStrategyのコード構造を静的解析"""
    print("=" * 80)
    print("TODO-001 Step1: BreakoutStrategy静的解析開始")
    print("=" * 80)
    
    print("\n🔍 **1. コード構造解析**")
    
    print("\n【Entry Signal処理フロー】")
    print("1. generate_entry_signal(idx) -> int")
    print("   - 前日高値突破 + 出来高増加 でentry_signal=1を返す")
    print("   - self.entry_prices[idx] = current_price  # エントリー価格記録")
    print("   - self.high_prices[idx] = high_price      # 高値記録")
    print("   - self.log_trade(message)                 # ログ出力")
    print("   - return 1")
    
    print("\n【Exit Signal処理フロー】")
    print("1. generate_exit_signal(idx) -> int")
    print("   - entry_indices = self.data[self.data['Entry_Signal'] == 1].index")
    print("   - latest_entry_date = entry_indices[-1]")
    print("   - 利確条件: current_price >= entry_price * (1 + take_profit)")
    print("   - 損切条件: current_price < high_price * (1 - trailing_stop)")
    print("   - self.log_trade(message)                 # ログ出力")
    print("   - return -1")
    
    print("\n【backtest()メソッド処理フロー】")
    print("1. self.data['Entry_Signal'] = 0            # 初期化")
    print("2. self.data['Exit_Signal'] = 0             # 初期化")
    print("3. for idx in range(len(self.data)):")
    print("   3-1. entry_signal = self.generate_entry_signal(idx)")
    print("   3-2. if entry_signal == 1:")
    print("        self.data.at[self.data.index[idx], 'Entry_Signal'] = 1")
    print("   3-3. exit_signal = self.generate_exit_signal(idx)")
    print("   3-4. if exit_signal == -1:")
    print("        self.data.at[self.data.index[idx], 'Exit_Signal'] = -1")
    print("4. return self.data")
    
    print("\n🚨 **重要発見: Exit_Signal = -1 が設定される**")
    print("- BreakoutStrategyでは Exit_Signal = -1 でエグジット")
    print("- しかし調査レポートでは Exit_Signal = 1 が62件発生")
    print("- これは明らかな矛盾！")
    
    return True

def test_breakout_with_real_data():
    """実際のデータでBreakoutStrategyを試験実行"""
    print("\n" + "=" * 80)
    print("🧪 **2. 実データでのBreakoutStrategy試験実行**")
    print("=" * 80)
    
    try:
        # 実際のデータを取得（少量でテスト）
        print("\n📊 株価データ取得中...")
        ticker = "7203.T"  # トヨタ
        data = yf.download(ticker, start="2024-01-01", end="2024-03-01", progress=False)
        
        if data.empty:
            print("❌ データ取得失敗")
            return False
            
        print(f"✅ データ取得成功: {len(data)}行")
        print(f"期間: {data.index[0]} ～ {data.index[-1]}")
        
        # BreakoutStrategy初期化
        print("\n🔧 BreakoutStrategy初期化...")
        strategy = BreakoutStrategy(data.copy())
        
        # バックテスト実行
        print("\n⚡ バックテスト実行...")
        result = strategy.backtest()
        
        # 結果分析
        print("\n📈 **結果分析**")
        entry_count = (result['Entry_Signal'] == 1).sum()
        exit_count_neg1 = (result['Exit_Signal'] == -1).sum()
        exit_count_pos1 = (result['Exit_Signal'] == 1).sum()
        
        print(f"Entry_Signal = 1: {entry_count}件")
        print(f"Exit_Signal = -1: {exit_count_neg1}件")
        print(f"Exit_Signal = 1: {exit_count_pos1}件")
        
        # Entry_Signal = 1 の行を詳細表示
        if entry_count > 0:
            print(f"\n🎯 **Entry_Signal = 1 の行詳細**")
            entry_rows = result[result['Entry_Signal'] == 1][['Open', 'High', 'Low', 'Close', 'Volume', 'Entry_Signal']]
            print(entry_rows)
        
        # Exit_Signal = -1 の行を詳細表示
        if exit_count_neg1 > 0:
            print(f"\n🚪 **Exit_Signal = -1 の行詳細**")
            exit_rows = result[result['Exit_Signal'] == -1][['Open', 'High', 'Low', 'Close', 'Volume', 'Exit_Signal']]
            print(exit_rows)
        
        # Exit_Signal = 1 の行を詳細表示（これが異常）
        if exit_count_pos1 > 0:
            print(f"\n⚠️ **Exit_Signal = 1 の行詳細（異常パターン）**")
            exit_rows = result[result['Exit_Signal'] == 1][['Open', 'High', 'Low', 'Close', 'Volume', 'Exit_Signal']]
            print(exit_rows)
        
        # 同時発生行の確認
        simultaneous = result[(result['Entry_Signal'] == 1) & (result['Exit_Signal'] != 0)]
        if len(simultaneous) > 0:
            print(f"\n🚨 **Entry_Signal=1 かつ Exit_Signal≠0 の同時発生行**")
            print(f"発生件数: {len(simultaneous)}件")
            print(simultaneous[['Open', 'High', 'Low', 'Close', 'Entry_Signal', 'Exit_Signal']])
        
        print(f"\n✅ BreakoutStrategy試験実行完了")
        return result
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_debug_version():
    """デバッグ版BreakoutStrategyを作成"""
    print("\n" + "=" * 80)
    print("🐛 **3. デバッグ版BreakoutStrategy作成**")
    print("=" * 80)
    
    debug_code = '''#!/usr/bin/env python3
"""
デバッグ版BreakoutStrategy
Exit_Signal列更新処理の詳細追跡版
"""

import sys
sys.path.append(r"C:\\Users\\imega\\Documents\\my_backtest_project")

import pandas as pd
import numpy as np
from src.strategies.Breakout import BreakoutStrategy

class DebugBreakoutStrategy(BreakoutStrategy):
    def __init__(self, data, params=None, price_column="Adj Close", volume_column="Volume"):
        super().__init__(data, params, price_column, volume_column)
        self.debug_log = []
        
    def debug_print(self, message):
        """デバッグメッセージ出力"""
        print(f"[DEBUG] {message}")
        self.debug_log.append(message)
    
    def generate_exit_signal(self, idx: int) -> int:
        """デバッグ版エグジットシグナル生成"""
        self.debug_print(f"generate_exit_signal called: idx={idx}")
        
        if idx < 1:
            self.debug_print(f"idx < 1, returning 0")
            return 0
            
        # エントリー価格と高値を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        self.debug_print(f"entry_indices: {entry_indices.tolist()}")
        
        if len(entry_indices) == 0:
            self.debug_print(f"No entry indices, returning 0")
            return 0
            
        if entry_indices[-1] >= self.data.index[idx]:
            self.debug_print(f"Latest entry >= current date, returning 0")
            return 0
            
        # 最新のエントリーインデックス（日付）を取得
        latest_entry_date = entry_indices[-1]
        latest_entry_pos = self.data.index.get_loc(latest_entry_date)
        
        self.debug_print(f"latest_entry_date: {latest_entry_date}, latest_entry_pos: {latest_entry_pos}")

        if latest_entry_date not in self.entry_prices:
            self.entry_prices[latest_entry_date] = self.data[self.price_column].iloc[latest_entry_pos]
            
        if latest_entry_date not in self.high_prices and 'High' in self.data.columns:
            self.high_prices[latest_entry_date] = self.data['High'].iloc[latest_entry_pos]
        elif latest_entry_date not in self.high_prices:
            self.high_prices[latest_entry_date] = self.data[self.price_column].iloc[latest_entry_pos]
            
        entry_price = self.entry_prices[latest_entry_date]
        high_price = self.high_prices[latest_entry_date]
        current_price = self.data[self.price_column].iloc[idx]
        
        self.debug_print(f"entry_price: {entry_price}, high_price: {high_price}, current_price: {current_price}")
        
        # 現在の高値を更新（トレーリングストップのために）
        if 'High' in self.data.columns and self.data['High'].iloc[idx] > high_price:
            high_price = self.data['High'].iloc[idx]
            self.high_prices[latest_entry_date] = high_price
            self.debug_print(f"High price updated: {high_price}")

        # 利確条件
        take_profit_level = entry_price * (1 + self.params["take_profit"])
        self.debug_print(f"take_profit_level: {take_profit_level}")
        
        if current_price >= take_profit_level:
            self.debug_print(f"利確条件成立: current_price({current_price}) >= take_profit_level({take_profit_level})")
            self.log_trade(f"Breakout イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            self.debug_print(f"Returning -1 for profit taking")
            return -1

        # 損切条件（高値からの反落）
        trailing_stop_level = 1 - self.params["trailing_stop"]
        trailing_stop_price = high_price * trailing_stop_level
        self.debug_print(f"trailing_stop_price: {trailing_stop_price}")
        
        if current_price < trailing_stop_price:
            self.debug_print(f"損切条件成立: current_price({current_price}) < trailing_stop_price({trailing_stop_price})")
            self.log_trade(f"Breakout イグジットシグナル: 高値から反落 日付={self.data.index[idx]}, 価格={current_price}, 高値={high_price}")
            self.debug_print(f"Returning -1 for trailing stop")
            return -1

        self.debug_print(f"No exit condition met, returning 0")
        return 0
    
    def backtest(self):
        """デバッグ版バックテスト"""
        self.debug_print("backtest() started")
        
        # シグナル列の初期化
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0
        self.debug_print("Signal columns initialized")

        # 各日にちについてシグナルを計算
        for idx in range(len(self.data)):
            current_date = self.data.index[idx]
            self.debug_print(f"Processing idx={idx}, date={current_date}")
            
            # Entry_Signalがまだ立っていない場合のみエントリーシグナルをチェック
            if not self.data['Entry_Signal'].iloc[max(0, idx-1):idx+1].any():
                entry_signal = self.generate_entry_signal(idx)
                self.debug_print(f"entry_signal result: {entry_signal}")
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
                    self.debug_print(f"Set Entry_Signal = 1 at idx={idx}, date={current_date}")
            
            # イグジットシグナルを確認
            exit_signal = self.generate_exit_signal(idx)
            self.debug_print(f"exit_signal result: {exit_signal}")
            if exit_signal == -1:
                self.data.at[self.data.index[idx], 'Exit_Signal'] = -1
                self.debug_print(f"Set Exit_Signal = -1 at idx={idx}, date={current_date}")
                
            # 現在の状態確認
            current_entry = self.data['Entry_Signal'].iloc[idx]
            current_exit = self.data['Exit_Signal'].iloc[idx]
            self.debug_print(f"Final state: Entry_Signal={current_entry}, Exit_Signal={current_exit}")

        self.debug_print("backtest() completed")
        return self.data
'''
    
    try:
        with open('debug_breakout_strategy.py', 'w', encoding='utf-8') as f:
            f.write(debug_code)
        print("✅ デバッグ版BreakoutStrategy作成完了: debug_breakout_strategy.py")
        return True
    except Exception as e:
        print(f"❌ デバッグ版作成失敗: {e}")
        return False

def main():
    """メイン実行"""
    print("🎯 TODO-001 Step1: BreakoutStrategy詳細解析")
    print("=" * 80)
    
    # Step 1: コード構造解析
    analyze_breakout_code_structure()
    
    # Step 2: 実データでのテスト実行
    result = test_breakout_with_real_data()
    
    # Step 3: デバッグ版作成
    create_debug_version()
    
    print("\n" + "=" * 80)
    print("📋 **TODO-001 Step1 完了サマリー**")
    print("=" * 80)
    
    print("\n🔍 **静的解析結果:**")
    print("- BreakoutStrategy.generate_exit_signal() は -1 を返す")
    print("- backtest()メソッドは Exit_Signal = -1 を設定")
    print("- しかし実際の出力では Exit_Signal = 1 が62件")
    print("- **これは明らかな矛盾**")
    
    print("\n🧪 **実データテスト結果:**")
    if result is not False:
        print("- 試験実行成功")
        print("- Entry/Exitシグナル生成確認済み")
    else:
        print("- 試験実行失敗")
    
    print("\n🐛 **デバッグツール作成:**")
    print("- debug_breakout_strategy.py 作成済み")
    print("- 次のStep2で詳細追跡可能")
    
    print("\n🎯 **重要発見:**")
    print("- BreakoutStrategyは設計上 Exit_Signal = -1 を設定")
    print("- 実際の出力は Exit_Signal = 1 が多数")
    print("- **何らかの処理でExit_Signalが1に変換されている可能性**")
    
    print("\n▶️ **次のアクション: TODO-001 Step2**")
    print("- デバッグ版でのトレース実行")
    print("- DataFrame更新前後の値確認")
    print("- Exit_Signal変換処理の特定")

if __name__ == "__main__":
    main()