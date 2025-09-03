"""
新しいExcel出力モジュール (改良版)
File: simple_excel_exporter.py
Description: 
  シンプルで確実なExcel出力を行う改良版モジュール。
  正確な取引履歴、損益推移、パフォーマンス指標を計算・出力します。

Author: imega
Created: 2025-07-30
Modified: 2025-07-31

Features:
  - 正確な取引履歴の生成
  - 正しい日次累積損益の計算
  - 適切なパフォーマンス指標の算出（リスクリワード比・期待値含む）
  - 100株単位での取引量表示
  - リスク管理設定の表示
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging
from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__)

class SimpleExcelExporter:
    """シンプルで確実なExcel出力クラス"""
    
    def __init__(self, initial_capital: float = 1000000):
        """
        初期化
        
        Parameters:
            initial_capital (float): 初期資金（デフォルト: 100万円）
        """
        self.initial_capital = initial_capital
        self.base_shares = 100  # 基本売買単位（100株）
        self.position_size_ratio = 0.1  # ポジションサイズ（10%）
        self.commission_rate = 0.001  # 手数料率（0.1%）
        
    def export_backtest_results(self, stock_data: pd.DataFrame, ticker: str, 
                              output_dir: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        バックテスト結果をExcelに出力する
        
        Parameters:
            stock_data (pd.DataFrame): 株価データ（シグナル含む）
            ticker (str): 銘柄コード
            output_dir (str): 出力ディレクトリ
            filename (str): ファイル名
            
        Returns:
            str: 出力ファイルパス
        """
        try:
            # 出力ディレクトリの設定
            if output_dir is None:
                base_dir = r"C:\Users\imega\Documents\my_backtest_project\backtest_results"
                output_dir = os.path.join(base_dir, "improved_results")
            
            # ディレクトリ作成
            os.makedirs(output_dir, exist_ok=True)
            
            # ファイル名の設定
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_results_{ticker}_{timestamp}.xlsx"
            
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
                
            output_path = os.path.join(output_dir, filename)
            
            logger.info(f"Excel出力開始: {ticker} -> {output_path}")
            
            # 1. 取引履歴を生成
            trade_history = self._generate_trade_history(stock_data, ticker)
            logger.info(f"取引履歴生成完了: {len(trade_history)} 件")
            
            # 2. 損益推移を計算
            daily_pnl = self._calculate_daily_pnl(stock_data, trade_history)
            logger.info(f"損益推移計算完了: {len(daily_pnl)} 日")
            
            # 3. パフォーマンス指標を計算（リスクリワード比・期待値含む）
            performance_metrics = self._calculate_performance_metrics(trade_history, daily_pnl)
            logger.info("パフォーマンス指標計算完了（リスクリワード比・期待値含む）")
            
            # 4. リスク管理設定を取得
            risk_settings = self._get_risk_management_settings()
            
            # 5. 戦略別統計を計算
            strategy_stats = self._calculate_strategy_statistics(stock_data, trade_history)
            
            # 6. Excelファイルに出力
            self._write_to_excel({
                '取引履歴': trade_history,
                '損益推移': daily_pnl,
                'パフォーマンス指標': performance_metrics,
                'リスク管理設定': risk_settings,
                '戦略別統計': strategy_stats,
                '価格データ': self._prepare_price_data(stock_data)
            }, output_path, ticker)
            
            logger.info(f"Excel出力完了: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Excel出力エラー: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _generate_trade_history(self, stock_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """取引履歴を生成する"""
        trades = []
        open_positions = {}  # {entry_date: {'price': float, 'strategy': str}}
        
        try:
            for date, row in stock_data.iterrows():
                # エントリーシグナル
                if row.get('Entry_Signal', 0) == 1:
                    entry_price = row.get('Adj Close', row.get('Close', 0))
                    strategy = row.get('Strategy', 'Unknown')
                    
                    if entry_price > 0:
                        open_positions[date] = {
                            'price': entry_price,
                            'strategy': strategy
                        }
                        logger.debug(f"エントリー記録: {date}, 価格: {entry_price}, 戦略: {strategy}")
                    
                # エグジットシグナル  
                elif row.get('Exit_Signal', 0) == -1 and open_positions:
                    exit_price = row.get('Adj Close', row.get('Close', 0))
                    
                    if exit_price > 0:
                        # FIFO方式で最古のポジションを決済
                        entry_date = min(open_positions.keys())
                        entry_info = open_positions.pop(entry_date)
                        entry_price = entry_info['price']
                        strategy = entry_info['strategy']
                        
                        # 取引結果を計算
                        shares = self._calculate_shares(entry_price)
                        trade_amount = shares * entry_price
                        commission = trade_amount * self.commission_rate
                        profit_loss = (exit_price - entry_price) * shares - commission
                        
                        trades.append({
                            '日付': date,
                            '銘柄': ticker,
                            '戦略': strategy,
                            'エントリー日': entry_date,
                            'エグジット日': date,
                            'エントリー価格': entry_price,
                            'エグジット価格': exit_price,
                            '取引量(株)': shares,
                            '取引金額': trade_amount,
                            '手数料': commission,
                            '取引結果': profit_loss,
                            '保有日数': (date - entry_date).days
                        })
                        
                        logger.debug(f"エグジット記録: {date}, 損益: {profit_loss:.2f}円")
            
            # 未決済ポジションを最終日で強制決済
            if open_positions:
                last_date = stock_data.index[-1]
                last_price = stock_data.loc[last_date, 'Adj Close']
                
                for entry_date, entry_info in open_positions.items():
                    entry_price = entry_info['price']
                    strategy = entry_info['strategy']
                    
                    shares = self._calculate_shares(entry_price)
                    trade_amount = shares * entry_price
                    commission = trade_amount * self.commission_rate
                    profit_loss = (last_price - entry_price) * shares - commission
                    
                    trades.append({
                        '日付': last_date,
                        '銘柄': ticker,
                        '戦略': strategy,
                        'エントリー日': entry_date,
                        'エグジット日': last_date,
                        'エントリー価格': entry_price,
                        'エグジット価格': last_price,
                        '取引量(株)': shares,
                        '取引金額': trade_amount,
                        '手数料': commission,
                        '取引結果': profit_loss,
                        '保有日数': (last_date - entry_date).days,
                        '備考': '強制決済'
                    })
                    
                logger.warning(f"未決済ポジション {len(open_positions)} 件を強制決済")
            
            return pd.DataFrame(trades)
            
        except Exception as e:
            logger.error(f"取引履歴生成エラー: {e}")
            return pd.DataFrame()
    
    def _calculate_shares(self, entry_price: float) -> int:
        """取引株数を計算（100株単位）"""
        try:
            # ポジションサイズに基づく投資金額
            target_amount = self.initial_capital * self.position_size_ratio
            
            # 株数を計算
            shares_exact = target_amount / entry_price
            
            # 100株単位に丸める
            shares_units = int(shares_exact / self.base_shares)
            
            # 最低100株は取引する
            return max(shares_units * self.base_shares, self.base_shares)
            
        except (ZeroDivisionError, ValueError):
            return self.base_shares  # デフォルト値
    
    def _calculate_daily_pnl(self, stock_data: pd.DataFrame, 
                           trade_history: pd.DataFrame) -> pd.DataFrame:
        """日次損益推移を計算する"""
        try:
            dates = stock_data.index
            daily_pnl = pd.Series(0.0, index=dates, name='日次損益')
            
            # 各取引の損益を決済日に反映
            for _, trade in trade_history.iterrows():
                exit_date = trade['エグジット日']
                if exit_date in daily_pnl.index:
                    daily_pnl[exit_date] += trade['取引結果']
            
            # 累積損益を計算
            cumulative_pnl = daily_pnl.cumsum()
            
            # 総資産を計算
            total_assets = self.initial_capital + cumulative_pnl
            
            # リターン率を計算
            daily_return = daily_pnl / self.initial_capital * 100
            cumulative_return = cumulative_pnl / self.initial_capital * 100
            
            return pd.DataFrame({
                '日付': dates,
                '日次損益': daily_pnl.values,
                '累積損益': cumulative_pnl.values,
                '総資産': total_assets.values,
                '日次リターン(%)': daily_return.values,
                '累積リターン(%)': cumulative_return.values
            })
            
        except Exception as e:
            logger.error(f"損益推移計算エラー: {e}")
            # エラー時は空のDataFrameを返す
            return pd.DataFrame({
                '日付': stock_data.index,
                '日次損益': [0] * len(stock_data),
                '累積損益': [0] * len(stock_data),
                '総資産': [self.initial_capital] * len(stock_data),
                '日次リターン(%)': [0] * len(stock_data),
                '累積リターン(%)': [0] * len(stock_data)
            })
    
    def _calculate_risk_reward_ratio(self, trade_history: pd.DataFrame) -> float:
        """
        リスクリワード比を計算（手数料除く）
        リスクリワード比 = 平均利益 ÷ 平均損失
        
        Parameters:
            trade_history (pd.DataFrame): 取引履歴
            
        Returns:
            float: リスクリワード比
        """
        try:
            if trade_history.empty:
                return 0.0
            
            # 手数料を除いた純損益を計算（手数料分を戻す）
            pure_pnl = trade_history['取引結果'] + trade_history.get('手数料', 0)
            
            # 勝ちトレードと負けトレードを分離
            winning_trades = pure_pnl[pure_pnl > 0]
            losing_trades = pure_pnl[pure_pnl < 0]
            
            if len(winning_trades) == 0 or len(losing_trades) == 0:
                return 0.0
            
            # 平均利益と平均損失を計算
            avg_profit = winning_trades.mean()
            avg_loss = abs(losing_trades.mean())  # 絶対値で正の値にする
            
            return avg_profit / avg_loss if avg_loss > 0 else 0.0
            
        except Exception as e:
            logger.error(f"リスクリワード比計算エラー: {e}")
            return 0.0
    
    def _calculate_expected_value(self, trade_history: pd.DataFrame, win_rate: float) -> Tuple[float, float]:
        """
        期待値を計算（円・％）
        期待値（円） = （勝率 × 平均利益） - （負け率 × 平均損失）
        期待値（％） = 期待値（円） ÷ 初期資金 × 100
        
        Parameters:
            trade_history (pd.DataFrame): 取引履歴
            win_rate (float): 勝率（％）
            
        Returns:
            Tuple[float, float]: (期待値（円）, 期待値（％）)
        """
        try:
            if trade_history.empty or win_rate == 0:
                return 0.0, 0.0
            
            # 手数料を除いた純損益を計算
            pure_pnl = trade_history['取引結果'] + trade_history.get('手数料', 0)
            
            # 勝ちトレードと負けトレードを分離
            winning_trades = pure_pnl[pure_pnl > 0]
            losing_trades = pure_pnl[pure_pnl < 0]
            
            # 平均利益と平均損失を計算
            avg_profit = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            
            # 勝率と負け率（小数）
            win_rate_decimal = win_rate / 100
            lose_rate_decimal = 1 - win_rate_decimal
            
            # 期待値（円）を計算
            expected_value_yen = (win_rate_decimal * avg_profit) - (lose_rate_decimal * avg_loss)
            
            # 期待値（％）を計算
            expected_value_pct = (expected_value_yen / self.initial_capital * 100) if self.initial_capital > 0 else 0
            
            return expected_value_yen, expected_value_pct
            
        except Exception as e:
            logger.error(f"期待値計算エラー: {e}")
            return 0.0, 0.0
    
    def _calculate_performance_metrics(self, trade_history: pd.DataFrame,
                                     daily_pnl: pd.DataFrame) -> pd.DataFrame:
        """パフォーマンス指標を計算する"""
        try:
            if trade_history.empty:
                return pd.DataFrame({
                    '指標': ['総取引数', '勝率', '損益合計', '最大ドローダウン(%)'],
                    '値': [0, '0%', '0円', '0%']
                })
            
            # 基本指標
            total_trades = len(trade_history)
            winning_trades = (trade_history['取引結果'] > 0).sum()
            losing_trades = (trade_history['取引結果'] < 0).sum()
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 損益指標
            total_profit = trade_history['取引結果'].sum()
            average_profit = total_profit / total_trades if total_trades > 0 else 0
            max_profit = trade_history['取引結果'].max() if not trade_history.empty else 0
            max_loss = trade_history['取引結果'].min() if not trade_history.empty else 0
            
            # 勝ちトレード・負けトレードの平均
            winning_avg = trade_history[trade_history['取引結果'] > 0]['取引結果'].mean() if winning_trades > 0 else 0
            losing_avg = trade_history[trade_history['取引結果'] < 0]['取引結果'].mean() if losing_trades > 0 else 0
            
            # プロフィットファクター
            gross_profit = trade_history[trade_history['取引結果'] > 0]['取引結果'].sum()
            gross_loss = abs(trade_history[trade_history['取引結果'] < 0]['取引結果'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # ドローダウン計算
            cumulative_pnl = daily_pnl['累積損益']
            peak = cumulative_pnl.expanding().max()
            drawdown = peak - cumulative_pnl
            max_drawdown_abs = drawdown.max()
            max_drawdown_pct = (max_drawdown_abs / self.initial_capital * 100) if self.initial_capital > 0 else 0
            
            # シャープレシオの計算
            daily_returns = daily_pnl['日次リターン(%)'] / 100
            sharpe_ratio = 0.0
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            
            # 平均保有日数
            avg_holding_days = trade_history['保有日数'].mean() if '保有日数' in trade_history.columns else 0
            
            # ★新規追加: リスクリワード比を計算
            risk_reward_ratio = self._calculate_risk_reward_ratio(trade_history)
            
            # ★新規追加: 期待値を計算
            expected_value_yen, expected_value_pct = self._calculate_expected_value(trade_history, win_rate)
            
            return pd.DataFrame({
                '指標': [
                    '総取引数',
                    '勝ちトレード数',
                    '負けトレード数',
                    '勝率',
                    '損益合計',
                    '平均損益',
                    '最大利益',
                    '最大損失',
                    '勝ちトレード平均',
                    '負けトレード平均',
                    'プロフィットファクター',
                    'リスクリワード比',  # ★新規追加
                    '期待値（円）',      # ★新規追加
                    '期待値（％）',      # ★新規追加
                    '最大ドローダウン(円)',
                    '最大ドローダウン(%)',
                    'シャープレシオ',
                    '平均保有日数',
                    '総手数料'
                ],
                '値': [
                    total_trades,
                    winning_trades,
                    losing_trades,
                    f"{win_rate:.1f}%",
                    f"{total_profit:,.0f}円",
                    f"{average_profit:,.0f}円",
                    f"{max_profit:,.0f}円",
                    f"{max_loss:,.0f}円",
                    f"{winning_avg:,.0f}円",
                    f"{losing_avg:,.0f}円",
                    f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                    f"{risk_reward_ratio:.3f}",  # ★新規追加
                    f"{expected_value_yen:,.0f}円",  # ★新規追加
                    f"{expected_value_pct:+.2f}%",   # ★新規追加（+記号付き）
                    f"{max_drawdown_abs:,.0f}円",
                    f"{max_drawdown_pct:.2f}%",
                    f"{sharpe_ratio:.3f}",
                    f"{avg_holding_days:.1f}日",
                    f"{trade_history['手数料'].sum():,.0f}円" if '手数料' in trade_history.columns else "0円"
                ]
            })
            
        except Exception as e:
            logger.error(f"パフォーマンス指標計算エラー: {e}")
            return pd.DataFrame({
                '指標': ['エラー'],
                '値': [str(e)]
            })
    
    def _get_risk_management_settings(self) -> pd.DataFrame:
        """リスク管理設定を取得する"""
        try:
            from config.risk_management import RiskManagement
            risk_manager = RiskManagement(total_assets=self.initial_capital)
            
            return pd.DataFrame({
                'リスク管理設定': [
                    '初期資金',
                    'ポジションサイズ',
                    '最大許容ドローダウン',
                    '1回の取引あたりの最大損失',
                    '同日での最大連敗数',
                    '最大ポジション数',
                    '手数料率',
                    '最小取引単位'
                ],
                '値': [
                    f"{self.initial_capital:,}円",
                    f"{self.position_size_ratio * 100:.1f}%",
                    f"{risk_manager.max_drawdown * 100:.1f}%",
                    f"{risk_manager.max_loss_per_trade * 100:.1f}%",
                    f"{risk_manager.max_daily_losses}回",
                    f"{risk_manager.max_total_positions}ポジション",
                    f"{self.commission_rate * 100:.3f}%",
                    f"{self.base_shares}株"
                ]
            })
            
        except Exception as e:
            logger.error(f"リスク管理設定取得エラー: {e}")
            return pd.DataFrame({
                'リスク管理設定': ['エラー'],
                '値': [str(e)]
            })
    
    def _calculate_strategy_statistics(self, stock_data: pd.DataFrame, 
                                     trade_history: pd.DataFrame) -> pd.DataFrame:
        """戦略別統計を計算する"""
        try:
            if trade_history.empty:
                return pd.DataFrame()
            
            strategy_stats = []
            
            # 戦略別に集計
            strategies = trade_history['戦略'].unique()
            
            for strategy in strategies:
                strategy_trades = trade_history[trade_history['戦略'] == strategy]
                
                if not strategy_trades.empty:
                    total_trades = len(strategy_trades)
                    winning_trades = (strategy_trades['取引結果'] > 0).sum()
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    total_profit = strategy_trades['取引結果'].sum()
                    avg_profit = total_profit / total_trades if total_trades > 0 else 0
                    
                    # 戦略別リスクリワード比
                    risk_reward = self._calculate_risk_reward_ratio(strategy_trades)
                    
                    # 戦略別期待値
                    expected_yen, expected_pct = self._calculate_expected_value(strategy_trades, win_rate)
                    
                    strategy_stats.append({
                        '戦略': strategy,
                        '取引数': total_trades,
                        '勝ち数': winning_trades,
                        '勝率': f"{win_rate:.1f}%",
                        '合計損益': f"{total_profit:,.0f}円",
                        '平均損益': f"{avg_profit:,.0f}円",
                        'リスクリワード比': f"{risk_reward:.3f}",  # ★新規追加
                        '期待値（円）': f"{expected_yen:,.0f}円",    # ★新規追加
                        '期待値（％）': f"{expected_pct:+.2f}%"     # ★新規追加
                    })
            
            return pd.DataFrame(strategy_stats)
            
        except Exception as e:
            logger.error(f"戦略別統計計算エラー: {e}")
            return pd.DataFrame()
    
    def _prepare_price_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """価格データを準備する（参考用）"""
        try:
            # 必要な列のみを抽出
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            signal_columns = ['Entry_Signal', 'Exit_Signal', 'Strategy']
            
            available_columns = []
            for col in price_columns + signal_columns:
                if col in stock_data.columns:
                    available_columns.append(col)
            
            if available_columns:
                result = stock_data[available_columns].copy()
                result.index.name = '日付'
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"価格データ準備エラー: {e}")
            return pd.DataFrame()
    
    def _write_to_excel(self, data_dict: Dict[str, pd.DataFrame], 
                       output_path: str, ticker: str):
        """Excelファイルに書き込む"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 各シートを書き込み
                for sheet_name, df in data_dict.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        logger.debug(f"シート '{sheet_name}' 書き込み完了: {len(df)} 行")
                    else:
                        logger.warning(f"シート '{sheet_name}' はスキップされました（空のデータ）")
                
                # メタデータシートを追加
                metadata = pd.DataFrame({
                    '項目': ['銘柄コード', '出力日時', '初期資金', 'ファイル形式', '機能追加'],
                    '値': [ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                          f"{self.initial_capital:,}円", '改良版Excel出力', 'リスクリワード比・期待値対応']
                })
                metadata.to_excel(writer, sheet_name='メタデータ', index=False)
                
            logger.info(f"Excel書き込み完了: {output_path}")
            
        except Exception as e:
            logger.error(f"Excel書き込みエラー: {e}")
            raise


def save_backtest_results_simple(stock_data: pd.DataFrame, ticker: str, 
                                output_dir: Optional[str] = None, filename: Optional[str] = None) -> str:
    """
    バックテスト結果をシンプルなExcel出力で保存する（従来の関数インターフェース互換）
    
    Parameters:
        stock_data (pd.DataFrame): 株価データ（シグナル含む）
        ticker (str): 銘柄コード
        output_dir (str): 出力ディレクトリ
        filename (str): ファイル名
        
    Returns:
        str: 出力ファイルパス
    """
    exporter = SimpleExcelExporter()
    return exporter.export_backtest_results(stock_data, ticker, output_dir, filename)
