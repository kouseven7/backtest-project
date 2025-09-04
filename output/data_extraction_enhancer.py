"""
main.py結果データの精密抽出・解析エンジン
Phase 2.3 Task 2.3.1: データ収集最適化

Purpose:
  - main.pyから渡されるDataFrameの正確な解析
  - Entry_Signal/Exit_Signalからの取引抽出精度向上
  - ポジション管理と損益計算の正確性確保

Author: GitHub Copilot Agent
Created: 2025-09-04
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import logging

# ロガー設定
logger = logging.getLogger(__name__)

class MainDataExtractor:
    """main.py結果データの精密抽出クラス"""
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        初期化
        
        Args:
            initial_capital: 初期資本金（デフォルト100万円）
        """
        self.initial_capital = initial_capital
        self.logger = logger
        
    def extract_accurate_trades(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Entry_Signal/Exit_Signalから正確な取引データを抽出
        
        Args:
            stock_data: main.pyから渡されるDataFrame
            
        Returns:
            List[Dict]: 正確な取引リスト
        """
        trades: List[Dict[str, Any]] = []
        current_positions: Dict[str, Dict[str, Any]] = {}  # 戦略別ポジション管理
        
        if stock_data.empty:
            self.logger.warning("空のDataFrameが渡されました")
            return trades
        
        # 必要な列の確認
        required_cols = ['Entry_Signal', 'Exit_Signal', 'Close']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        if missing_cols:
            self.logger.error(f"必要な列が不足: {missing_cols}")
            return trades
        
        self.logger.info(f"データ解析開始: {len(stock_data)}行, 期間: {stock_data.index[0]} - {stock_data.index[-1]}")
        
        for idx, row in stock_data.iterrows():
            try:
                # エントリーシグナル検出
                if row.get('Entry_Signal', 0) == 1:
                    strategy = row.get('Strategy', 'Unknown')
                    self._process_entry_signal(current_positions, idx, row, strategy)
                    
                # エグジットシグナル検出
                elif row.get('Exit_Signal', 0) == 1:
                    strategy = row.get('Strategy', 'Unknown')
                    completed_trade = self._process_exit_signal(current_positions, idx, row, strategy)
                    if completed_trade:
                        trades.append(completed_trade)
                        
            except Exception as e:
                self.logger.error(f"行{idx}の処理中にエラー: {e}")
                continue
        
        # 未決済ポジションの強制決済
        final_trades = self._close_remaining_positions(
            current_positions, stock_data.index[-1], stock_data['Close'].iloc[-1]
        )
        trades.extend(final_trades)
        
        self.logger.info(f"取引抽出完了: {len(trades)}件の取引を検出")
        return trades
    
    def _process_entry_signal(self, positions: Dict[str, Dict[str, Any]], date: Any, row: pd.Series, strategy: str) -> None:
        """エントリーシグナルの処理"""
        entry_price = row.get('Close', row.get('Adj Close', 0))
        if entry_price <= 0:
            self.logger.warning(f"無効なエントリー価格: {entry_price}")
            return
        
        position_key = f"{strategy}_{date}"
        shares = self._calculate_shares(entry_price)
        
        positions[position_key] = {
            'entry_date': date,
            'entry_price': entry_price,
            'strategy': strategy,
            'shares': shares,
            'position_value': entry_price * shares
        }
        
        self.logger.debug(f"エントリー: {strategy} @{entry_price:.2f} x{shares}株")
    
    def _process_exit_signal(self, positions: Dict[str, Dict[str, Any]], date: Any, row: pd.Series, strategy: str) -> Optional[Dict[str, Any]]:
        """エグジットシグナルの処理"""
        exit_price = row.get('Close', row.get('Adj Close', 0))
        if exit_price <= 0:
            self.logger.warning(f"無効なエグジット価格: {exit_price}")
            return None
        
        # 該当戦略の最も古いポジションを決済
        strategy_positions = {k: v for k, v in positions.items() if v['strategy'] == strategy}
        if not strategy_positions:
            self.logger.warning(f"決済対象ポジションなし: {strategy}")
            return None
        
        # 最も古いポジションを選択
        oldest_key = min(strategy_positions.keys(), key=lambda k: positions[k]['entry_date'])
        position = positions.pop(oldest_key)
        
        return self._create_trade_record(position, date, exit_price)
    
    def _close_remaining_positions(self, positions: Dict[str, Dict[str, Any]], final_date: Any, final_price: float) -> List[Dict[str, Any]]:
        """未決済ポジションの強制決済"""
        final_trades = []
        
        for position_key, position in positions.items():
            trade = self._create_trade_record(position, final_date, final_price)
            trade['is_forced_exit'] = True
            final_trades.append(trade)
            self.logger.info(f"強制決済: {position['strategy']} @ {final_price:.2f}")
        
        return final_trades
    
    def _create_trade_record(self, position: Dict, exit_date: Any, exit_price: float) -> Dict[str, Any]:
        """取引レコード作成"""
        shares = position['shares']
        entry_price = position['entry_price']
        
        # 損益計算
        pnl = (exit_price - entry_price) * shares
        return_pct = (exit_price - entry_price) / entry_price
        
        # 保有期間計算
        try:
            if hasattr(exit_date - position['entry_date'], 'days'):
                holding_days = (exit_date - position['entry_date']).days
            else:
                holding_days = 1  # フォールバック
        except:
            holding_days = 1
        
        return {
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'return_pct': return_pct,
            'holding_period_days': holding_days,
            'strategy': position['strategy'],
            'position_value': position['position_value'],
            'is_forced_exit': False
        }
    
    def _calculate_shares(self, price: float, capital_per_trade: Optional[float] = None) -> int:
        """1取引あたりの株数計算"""
        if capital_per_trade is None:
            capital_per_trade = self.initial_capital * 0.1  # 10%ポジションサイズ
        
        return max(1, int(capital_per_trade / price))
    
    def calculate_portfolio_performance(self, stock_data: pd.DataFrame, 
                                      trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        正確なポートフォリオパフォーマンス計算
        
        Args:
            stock_data: 元のDataFrame
            trades: 取引リスト
            
        Returns:
            Dict: パフォーマンス指標
        """
        if not trades:
            return self._get_zero_performance()
        
        # 基本損益計算
        total_pnl = sum(trade['pnl'] for trade in trades)
        final_value = self.initial_capital + total_pnl
        total_return = total_pnl / self.initial_capital
        
        # 日次リターン計算
        daily_returns = self._calculate_daily_returns(stock_data, trades)
        
        # リスク指標計算
        volatility = self._calculate_volatility(daily_returns)
        max_drawdown = self._calculate_max_drawdown(trades)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        # 取引統計
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_holding_days = np.mean([t['holding_period_days'] for t in trades]) if trades else 0
        avg_pnl = total_pnl / len(trades) if trades else 0
        
        return {
            'final_portfolio_value': final_value,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trades),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'avg_holding_days': avg_holding_days,
            'avg_pnl': avg_pnl,
            'max_profit': max([t['pnl'] for t in trades]) if trades else 0,
            'max_loss': min([t['pnl'] for t in trades]) if trades else 0
        }
    
    def _get_zero_performance(self) -> Dict[str, float]:
        """取引がない場合のデフォルトパフォーマンス"""
        return {
            'final_portfolio_value': self.initial_capital,
            'total_return': 0.0,
            'total_pnl': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'num_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0,
            'avg_holding_days': 0.0,
            'avg_pnl': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0
        }
    
    def _calculate_daily_returns(self, stock_data: pd.DataFrame, trades: List[Dict[str, Any]]) -> List[float]:
        """日次リターン計算"""
        if stock_data.empty or not trades:
            return []
        
        # 簡易的な日次リターン計算
        daily_values = []
        current_value = self.initial_capital
        
        trade_dates = set()
        for trade in trades:
            trade_dates.add(trade['entry_date'])
            trade_dates.add(trade['exit_date'])
        
        for date in stock_data.index:
            if date in trade_dates:
                # 取引が発生した日の損益を反映
                daily_pnl = sum(t['pnl'] for t in trades if t['exit_date'] == date)
                current_value += daily_pnl
            
            daily_values.append(current_value)
        
        # 日次リターン率計算
        returns = []
        for i in range(1, len(daily_values)):
            if daily_values[i-1] > 0:
                ret = (daily_values[i] - daily_values[i-1]) / daily_values[i-1]
                returns.append(ret)
        
        return returns
    
    def _calculate_volatility(self, daily_returns: List[float]) -> float:
        """ボラティリティ計算"""
        if len(daily_returns) < 2:
            return 0.0
        return np.std(daily_returns) * np.sqrt(252)  # 年率化
    
    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """最大ドローダウン計算"""
        if not trades:
            return 0.0
        
        # 累積損益の推移からドローダウン計算
        cumulative_pnl = []
        running_pnl = 0
        
        for trade in sorted(trades, key=lambda x: x['exit_date']):
            running_pnl += trade['pnl']
            cumulative_pnl.append(running_pnl)
        
        if not cumulative_pnl:
            return 0.0
        
        # 各時点での最大値からの下落率を計算
        peak = cumulative_pnl[0]
        max_dd = 0.0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            
            current_dd = (peak - value) / self.initial_capital if self.initial_capital > 0 else 0
            max_dd = min(max_dd, -current_dd)  # 負の値として表現
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """シャープレシオ計算"""
        if len(daily_returns) < 2:
            return 0.0
        
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            return 0.0
        
        # 年率化（リスクフリーレートは0と仮定）
        return (mean_return * 252) / (std_return * np.sqrt(252))

# 便利関数
def extract_and_analyze_main_data(stock_data: pd.DataFrame, ticker: str = "UNKNOWN") -> Dict[str, Any]:
    """
    main.pyデータの一括抽出・解析
    
    Args:
        stock_data: main.pyから渡されるDataFrame
        ticker: 銘柄コード
        
    Returns:
        Dict: 解析済みデータ
    """
    extractor = MainDataExtractor()
    
    # 取引抽出
    trades = extractor.extract_accurate_trades(stock_data)
    
    # パフォーマンス計算
    performance = extractor.calculate_portfolio_performance(stock_data, trades)
    
    # 期間情報
    period_info = {
        'start_date': stock_data.index[0] if not stock_data.empty else datetime.now(),
        'end_date': stock_data.index[-1] if not stock_data.empty else datetime.now(),
        'total_days': len(stock_data),
        'trading_days': len(stock_data)
    }
    
    return {
        'ticker': ticker,
        'trades': trades,
        'performance': performance,
        'period': period_info,
        'extraction_timestamp': datetime.now(),
        'data_quality': 'enhanced' if trades else 'no_trades'
    }
