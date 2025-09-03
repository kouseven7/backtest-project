"""
DSSMS Excel出力修正 - Phase 2実装
各銘柄切り替えを個別取引として正確に分離する改良版_prepare_excel_dataメソッド
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def prepare_excel_data_improved(self) -> pd.DataFrame:
    """
    DSSMSバックテストデータをExcel出力システム用に変換（改良版）
    各銘柄切り替えを個別の取引として正確に分離
    
    Returns:
        pd.DataFrame: Excel出力システム用のデータフレーム
    """
    try:
        self.logger.info("DSSMS Excel用データ準備開始（改良版）")
        
        # 1. switch_historyから個別取引を生成
        individual_trades = self._convert_switches_to_trades()
        
        if not individual_trades:
            self.logger.warning("個別取引データが空です")
            return pd.DataFrame()
        
        # 2. ポートフォリオ履歴ベースのデータフレーム作成
        portfolio_df = self._create_portfolio_dataframe()
        
        if portfolio_df.empty:
            self.logger.warning("ポートフォリオデータフレームが空です") 
            return pd.DataFrame()
        
        # 3. 取引シグナルを正確に設定
        portfolio_df = self._set_accurate_trade_signals(portfolio_df, individual_trades)
        
        # 4. Excel出力システム互換性のための列追加
        portfolio_df = self._add_excel_compatibility_columns(portfolio_df)
        
        # 5. 統計情報をログ出力
        self._log_conversion_statistics(portfolio_df, individual_trades)
        
        return portfolio_df
        
    except Exception as e:
        self.logger.error(f"Excel用データ準備エラー（改良版）: {e}")
        import traceback
        self.logger.error(traceback.format_exc())
        return pd.DataFrame()

def _convert_switches_to_trades(self) -> list:
    """
    switch_historyを個別取引リストに変換
    
    Returns:
        list: 個別取引のリスト
    """
    try:
        trades = []
        
        self.logger.info(f"銘柄切り替え履歴を個別取引に変換中: {len(self.switch_history)}件")
        
        for i, switch in enumerate(self.switch_history):
            # 切り替え情報の取得
            switch_time = getattr(switch, 'switch_time', None) or getattr(switch, 'timestamp', None)
            from_symbol = getattr(switch, 'from_symbol', None)
            to_symbol = getattr(switch, 'to_symbol', None)
            
            # 損益・コスト情報
            profit_loss = getattr(switch, 'profit_loss_at_switch', 0)
            switch_cost = getattr(switch, 'switch_cost', 0)
            holding_period = getattr(switch, 'holding_period_hours', 0)
            
            # 理由・トリガー情報
            reason = getattr(switch, 'reason', 'DSSMS切り替え')
            trigger = getattr(switch, 'trigger', 'daily_evaluation')
            
            # Portfolio価値
            portfolio_before = getattr(switch, 'portfolio_value_before', 0)
            portfolio_after = getattr(switch, 'portfolio_value_after', 0)
            
            if not switch_time:
                self.logger.warning(f"切り替え{i+1}: 日時情報なし")
                continue
            
            # 前のポジションのExit取引（初回以外）
            if i > 0 and from_symbol:
                exit_trade = {
                    'trade_id': f"DSSMS_EXIT_{i}",
                    'date': switch_time,
                    'symbol': from_symbol,
                    'action': 'SELL',
                    'strategy': f"DSSMS_{trigger}",
                    'entry_date': getattr(self.switch_history[i-1], 'switch_time', switch_time),
                    'exit_date': switch_time,
                    'pnl': profit_loss,
                    'holding_period_hours': holding_period,
                    'switch_cost': switch_cost / 2,  # ExitとEntryで分割
                    'reason': f"Exit_{reason}",
                    'portfolio_value': portfolio_before,
                    'trade_type': 'EXIT'
                }
                trades.append(exit_trade)
            
            # 新しいポジションのEntry取引
            if to_symbol:
                entry_trade = {
                    'trade_id': f"DSSMS_ENTRY_{i+1}",
                    'date': switch_time,
                    'symbol': to_symbol,
                    'action': 'BUY',
                    'strategy': f"DSSMS_{trigger}",
                    'entry_date': switch_time,
                    'exit_date': None,  # 次の切り替えまたは期間終了
                    'pnl': 0,  # Entry時点では未実現
                    'holding_period_hours': 0,  # 未完了
                    'switch_cost': switch_cost / 2,  # ExitとEntryで分割
                    'reason': f"Entry_{reason}",
                    'portfolio_value': portfolio_after,
                    'trade_type': 'ENTRY'
                }
                trades.append(entry_trade)
        
        self.logger.info(f"個別取引変換完了: {len(trades)}件の取引生成")
        return trades
        
    except Exception as e:
        self.logger.error(f"切り替え→取引変換エラー: {e}")
        return []

def _create_portfolio_dataframe(self) -> pd.DataFrame:
    """
    ポートフォリオ履歴からベースDataFrameを作成
    
    Returns:
        pd.DataFrame: ポートフォリオベースのDataFrame
    """
    try:
        if not self.portfolio_history:
            self.logger.warning("ポートフォリオ履歴が空です")
            return pd.DataFrame()
        
        # ポートフォリオ履歴をDataFrameに変換
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # 日付列の処理
        if 'date' in portfolio_df.columns:
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
        else:
            self.logger.error("日付列が見つかりません")
            return pd.DataFrame()
        
        # 基本列の追加
        portfolio_df['Adj Close'] = portfolio_df.get('portfolio_value', 1000000)
        portfolio_df['Close'] = portfolio_df['Adj Close']
        portfolio_df['Open'] = portfolio_df['Adj Close']
        portfolio_df['High'] = portfolio_df['Adj Close'] * 1.005
        portfolio_df['Low'] = portfolio_df['Adj Close'] * 0.995
        portfolio_df['Volume'] = 1000000
        portfolio_df['Strategy'] = 'DSSMS'
        
        # シグナル列を初期化
        portfolio_df['Entry_Signal'] = 0
        portfolio_df['Exit_Signal'] = 0
        
        self.logger.info(f"ポートフォリオDataFrame作成完了: {len(portfolio_df)}行")
        return portfolio_df
        
    except Exception as e:
        self.logger.error(f"ポートフォリオDataFrame作成エラー: {e}")
        return pd.DataFrame()

def _set_accurate_trade_signals(self, portfolio_df: pd.DataFrame, trades: list) -> pd.DataFrame:
    """
    個別取引情報を基に正確なEntry/Exitシグナルを設定
    
    Args:
        portfolio_df: ポートフォリオDataFrame
        trades: 個別取引リスト
        
    Returns:
        pd.DataFrame: シグナル設定済みDataFrame
    """
    try:
        self.logger.info(f"取引シグナル設定開始: {len(trades)}件の取引")
        
        for trade in trades:
            trade_date = trade['date']
            
            # 日付のマッチング（柔軟な処理）
            matched_dates = []
            
            for idx in portfolio_df.index:
                idx_date = pd.to_datetime(idx).date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                trade_date_only = pd.to_datetime(trade_date).date() if hasattr(trade_date, 'date') else pd.to_datetime(trade_date).date()
                
                if idx_date == trade_date_only:
                    matched_dates.append(idx)
            
            # マッチした日付にシグナルを設定
            for match_date in matched_dates:
                if trade['trade_type'] == 'ENTRY':
                    portfolio_df.loc[match_date, 'Entry_Signal'] = 1
                    portfolio_df.loc[match_date, 'Strategy'] = trade['strategy']
                elif trade['trade_type'] == 'EXIT':
                    portfolio_df.loc[match_date, 'Exit_Signal'] = -1
                
                self.logger.debug(f"シグナル設定: {match_date} {trade['trade_type']} {trade['symbol']}")
        
        # 統計情報
        entry_signals = (portfolio_df['Entry_Signal'] == 1).sum()
        exit_signals = (portfolio_df['Exit_Signal'] == -1).sum()
        
        self.logger.info(f"シグナル設定完了: Entry={entry_signals}件, Exit={exit_signals}件")
        return portfolio_df
        
    except Exception as e:
        self.logger.error(f"シグナル設定エラー: {e}")
        return portfolio_df

def _add_excel_compatibility_columns(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel出力システムとの互換性のための列を追加
    
    Args:
        portfolio_df: ポートフォリオDataFrame
        
    Returns:
        pd.DataFrame: 互換性列追加済みDataFrame
    """
    try:
        # 必要な列が不足している場合は追加
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Strategy']
        
        for col in required_columns:
            if col not in portfolio_df.columns:
                if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                    portfolio_df[col] = portfolio_df.get('portfolio_value', 1000000)
                elif col == 'Volume':
                    portfolio_df[col] = 1000000
                elif col == 'Strategy':
                    portfolio_df[col] = 'DSSMS'
        
        self.logger.info("Excel互換性列の追加完了")
        return portfolio_df
        
    except Exception as e:
        self.logger.error(f"Excel互換性列追加エラー: {e}")
        return portfolio_df

def _log_conversion_statistics(self, portfolio_df: pd.DataFrame, trades: list):
    """
    変換統計情報をログ出力
    
    Args:
        portfolio_df: ポートフォリオDataFrame
        trades: 個別取引リスト
    """
    try:
        entry_count = (portfolio_df['Entry_Signal'] == 1).sum()
        exit_count = (portfolio_df['Exit_Signal'] == -1).sum()
        trade_count = len(trades)
        
        self.logger.info("=== DSSMS Excel変換統計 ===")
        self.logger.info(f"データ期間: {portfolio_df.index[0]} - {portfolio_df.index[-1]}")
        self.logger.info(f"総行数: {len(portfolio_df)}行")
        self.logger.info(f"個別取引数: {trade_count}件")
        self.logger.info(f"エントリーシグナル: {entry_count}件")
        self.logger.info(f"エグジットシグナル: {exit_count}件")
        self.logger.info(f"銘柄切り替え回数: {len(self.switch_history)}回")
        
        # 取引タイプ別統計
        entry_trades = [t for t in trades if t['trade_type'] == 'ENTRY']
        exit_trades = [t for t in trades if t['trade_type'] == 'EXIT']
        
        self.logger.info(f"ENTRY取引: {len(entry_trades)}件")
        self.logger.info(f"EXIT取引: {len(exit_trades)}件")
        
    except Exception as e:
        self.logger.error(f"統計情報出力エラー: {e}")

# このコードをdssms_backtester.pyに統合するためのメソッド置換コード
METHOD_REPLACEMENT_CODE = '''
# src/dssms/dssms_backtester.py の _prepare_excel_data メソッドを以下で置換：

def _prepare_excel_data(self) -> pd.DataFrame:
    """
    DSSMSバックテストデータをExcel出力システム用に変換（改良版）
    各銘柄切り替えを個別の取引として正確に分離
    """
    return self.prepare_excel_data_improved()

# 上記の全てのヘルパーメソッドをDSSMSBacktesterクラスに追加
'''

print("DSSMS Excel出力修正 - Phase 2実装完了")
print("このコードをdssms_backtester.pyに統合してください")
print(METHOD_REPLACEMENT_CODE)
