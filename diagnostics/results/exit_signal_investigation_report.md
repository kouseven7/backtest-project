# エグジットシグナル生成状況調査 - 詳細レポート
**調査日時**: 2025-10-17 21:50:01
**Phase**: Phase 3.2 - エグジット生成問題調査・修正

---
## 調査サマリー
- **調査対象戦略数**: 7
- **エグジットシグナル生成あり**: 7 戦略
- **エグジットシグナル生成なし**: 0 戦略
- **生成率**: 100.0%

---
## 各戦略の詳細

### VWAPBreakoutStrategy
**ファイル**: `C:\Users\imega\Documents\my_backtest_project\strategies\VWAP_Breakout.py`
**backtest()メソッド**: ✅ あり
**Exit_Signal列生成**: ✅ あり
**エグジットシグナル生成**: ✅ あり

**検出されたパターン** (7個):
- **[Exit_Signal_Assignment]** Line 395: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 321: `def generate_exit_signal(self, idx: int, entry_idx: int = None) -> int:`
- **[exit_signal_variable]** Line 395: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 440: `self.data.loc[self.data.index[idx], 'Exit_Signal'] = -1`
- **[exit_signal_variable]** Line 445: `exit_signal = self.generate_exit_signal(idx, entry_idx)`

**エグジット条件** (5個):
- `if self.params.get("partial_exit_enabled", False):`
- `if "partial_exit_threshold" in self.params and "partial_exit_portion" in self.params:`
- `if profit_pct >= partial_exit_threshold and 'Partial_Exit' not in self.data.columns:`
- `if exit_signal == -1:`
- `if self.params.get("partial_exit_enabled", False)`

### VWAPBounceStrategy
**ファイル**: `C:\Users\imega\Documents\my_backtest_project\strategies\VWAP_Bounce.py`
**backtest()メソッド**: ✅ あり
**Exit_Signal列生成**: ✅ あり
**エグジットシグナル生成**: ✅ あり

**検出されたパターン** (5個):
- **[Exit_Signal_Assignment]** Line 244: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 244: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 262: `exit_signal = self.generate_exit_signal(idx)`
- **[exit_signal_variable]** Line 263: `if exit_signal == -1:`
- **[exit_signal_variable]** Line 264: `self.data.at[self.data.index[idx], 'Exit_Signal'] = -1`

**エグジット条件** (1個):
- `if exit_signal == -1:`

### MomentumInvestingStrategy
**ファイル**: `C:\Users\imega\Documents\my_backtest_project\strategies\Momentum_Investing.py`
**backtest()メソッド**: ✅ あり
**Exit_Signal列生成**: ✅ あり
**エグジットシグナル生成**: ✅ あり

**検出されたパターン** (7個):
- **[Exit_Signal_Assignment]** Line 360: `exit_count = (self.data['Exit_Signal'] == -1).sum()`
- **[exit_signal_variable]** Line 292: `self.data.loc[:, 'Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 322: `exit_signal = self.generate_exit_signal(idx)`
- **[exit_signal_variable]** Line 323: `if exit_signal == -1:`
- **[exit_signal_variable]** Line 324: `self.data.at[self.data.index[idx], 'Exit_Signal'] = -1`

**エグジット条件** (5個):
- `if momentum_change <= momentum_exit_threshold and rsi < 60:`
- `if current_volume < avg_volume * volume_exit_threshold:`
- `if exit_signal == -1:`
- `if partial_exit_pct > 0 and self.data['Partial_Exit'].iloc[idx-1 if idx > 0 else idx] == 0:`
- `if profit_pct >= partial_exit_threshold:`

### BreakoutStrategy
**ファイル**: `C:\Users\imega\Documents\my_backtest_project\strategies\Breakout.py`
**backtest()メソッド**: ✅ あり
**Exit_Signal列生成**: ✅ あり
**エグジットシグナル生成**: ✅ あり

**検出されたパターン** (5個):
- **[Exit_Signal_Assignment]** Line 169: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 169: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 187: `exit_signal = self.generate_exit_signal(idx)`
- **[exit_signal_variable]** Line 188: `if exit_signal == -1:`
- **[exit_signal_variable]** Line 189: `self.data.at[self.data.index[idx], 'Exit_Signal'] = -1`

**エグジット条件** (1個):
- `if exit_signal == -1:`

### OpeningGapStrategy
**ファイル**: `C:\Users\imega\Documents\my_backtest_project\strategies\Opening_Gap_Fixed.py`
**backtest()メソッド**: ✅ あり
**Exit_Signal列生成**: ✅ あり
**エグジットシグナル生成**: ✅ あり

**検出されたパターン** (5個):
- **[Exit_Signal_Assignment]** Line 30: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 30: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 49: `exit_signal = self.generate_exit_signal(idx)`
- **[exit_signal_variable]** Line 50: `if exit_signal == -1:`
- **[exit_signal_variable]** Line 51: `self.data.at[self.data.index[idx], 'Exit_Signal'] = -1`

**エグジット条件** (5個):
- `if exit_signal == -1:`
- `if current_position > 0.0 and self.params.get("partial_exit_enabled", False) and idx > 0:`
- `if self.data['Partial_Exit'].iloc[idx-1] == 0:`
- `if entry_price and (current_price / entry_price - 1) >= self.params["partial_exit_threshold"]:`
- `if current_position > 0.0 and self.params.get("partial_exit_enabled", False)`

### ContrarianStrategy
**ファイル**: `C:\Users\imega\Documents\my_backtest_project\strategies\contrarian_strategy.py`
**backtest()メソッド**: ✅ あり
**Exit_Signal列生成**: ✅ あり
**エグジットシグナル生成**: ✅ あり

**検出されたパターン** (7個):
- **[Exit_Signal_Assignment]** Line 152: `current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())`
- **[Exit_Signal_Assignment]** Line 203: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 152: `current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())`
- **[exit_signal_variable]** Line 203: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 213: `exit_signal = self.generate_exit_signal(idx)`

**エグジット条件** (5個):
- `if current_entries <= current_exits:`
- `if current_rsi >= self.params["rsi_exit_level"]:`
- `if exit_signal == -1:`
- `current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())`
- `print(result[['Adj Close', 'RSI', 'Entry_Signal', 'Exit_Signal']].tail())`

### GCStrategy
**ファイル**: `C:\Users\imega\Documents\my_backtest_project\strategies\gc_strategy_signal.py`
**backtest()メソッド**: ✅ あり
**Exit_Signal列生成**: ✅ あり
**エグジットシグナル生成**: ✅ あり

**検出されたパターン** (7個):
- **[Exit_Signal_Assignment]** Line 168: `current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())`
- **[Exit_Signal_Assignment]** Line 247: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 168: `current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())`
- **[exit_signal_variable]** Line 247: `self.data['Exit_Signal'] = 0`
- **[exit_signal_variable]** Line 259: `exit_signal = self.generate_exit_signal(idx)`

**エグジット条件** (5個):
- `if current_entries <= current_exits:`
- `if self.params.get("exit_on_death_cross", True):`
- `if exit_signal == -1:`
- `current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())`
- `if self.params.get("exit_on_death_cross", True)`

---
## 推奨アクション
### ✅ すべての戦略でエグジットシグナル生成が確認されました
