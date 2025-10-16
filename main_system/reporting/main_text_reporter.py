"""
Module: Main Text Reporter
File: main_text_reporter.py
Description: 
  main.pyの実行結果をテキストファイル形式で包括的にレポートするモジュールです。
  DSSSMSレポート形式に基づいて、戦略統合結果、期待値計算、詳細統計を出力します。

Author: imega
Created: 2025-01-24
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from config.logger_config import setup_logger
from output.data_extraction_enhancer import MainDataExtractor, extract_and_analyze_main_data

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\output.log")


class MainTextReporter:
    """
    main.pyの実行結果をテキストファイル形式で包括的にレポートするクラス
    """
    
    def __init__(self):
        self.extractor = MainDataExtractor()
        
    def generate_comprehensive_report(self, 
                                   stock_data: pd.DataFrame, 
                                   ticker: str,
                                   optimized_params: Optional[Dict[str, Dict[str, Any]]] = None,
                                   output_dir: Optional[str] = None) -> str:
        """
        包括的なテキストレポートを生成
        
        Parameters:
            stock_data (pd.DataFrame): バックテスト結果データ
            ticker (str): 銘柄コード
            optimized_params (Optional[Dict]): 最適化パラメータ
            output_dir (Optional[str]): 出力ディレクトリ
            
        Returns:
            str: 生成されたレポートファイルのパス
        """
        try:
            logger.info(f"包括的テキストレポート生成開始: {ticker}")
            
            # データ抽出と分析
            analysis_result = extract_and_analyze_main_data(stock_data, ticker)
            
            if not analysis_result:
                logger.error("データ抽出に失敗しました")
                return ""
            
            # 出力ディレクトリの設定
            if output_dir is None:
                output_dir = os.path.join("output", "main_reports")
            os.makedirs(output_dir, exist_ok=True)
            
            # ファイル名の生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"main_comprehensive_report_{ticker}_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # レポート内容を生成
            report_content = self._build_comprehensive_report(
                analysis_result, ticker, optimized_params, timestamp
            )
            
            # ファイルに書き込み
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"包括的レポート生成完了: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return ""
    
    def _build_comprehensive_report(self, 
                                  analysis_result: Dict[str, Any],
                                  ticker: str,
                                  optimized_params: Optional[Dict[str, Dict[str, Any]]],
                                  timestamp: str) -> str:
        """
        包括的レポート内容を構築
        """
        report_lines: List[str] = []
        
        # ヘッダー情報
        report_lines.extend(self._build_header_section(ticker, timestamp))
        
        # システム概要
        report_lines.extend(self._build_system_overview_section(analysis_result))
        
        # パフォーマンス統計
        report_lines.extend(self._build_performance_section(analysis_result))
        
        # 期待値分析
        report_lines.extend(self._build_expected_value_section(analysis_result))
        
        # 戦略別詳細分析
        report_lines.extend(self._build_strategy_analysis_section(analysis_result))
        
        # 取引詳細
        report_lines.extend(self._build_trade_details_section(analysis_result))
        
        # パラメータ情報
        if optimized_params:
            report_lines.extend(self._build_parameters_section(optimized_params))
        
        # リスク分析
        report_lines.extend(self._build_risk_analysis_section(analysis_result))
        
        # 統計サマリー
        report_lines.extend(self._build_statistical_summary_section(analysis_result))
        
        return '\n'.join(report_lines)
    
    def _build_header_section(self, ticker: str, timestamp: str) -> List[str]:
        """ヘッダーセクションを構築"""
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("マルチ戦略バックテスト包括レポート")
        lines.append("=" * 80)
        lines.append(f"銘柄コード: {ticker}")
        lines.append(f"レポート生成日時: {timestamp}")
        lines.append(f"レポート種別: Main.py 統合戦略実行結果")
        lines.append("")
        return lines
    
    def _build_system_overview_section(self, analysis: Dict[str, Any]) -> List[str]:
        """システム概要セクションを構築"""
        lines: List[str] = []
        lines.append("1. システム実行概要")
        lines.append("-" * 40)
        
        # 基本統計 - 正しいキー名を使用
        performance = analysis.get('performance', {})
        period = analysis.get('period', {})
        trades_info = analysis.get('trades', {})
        trades_list = trades_info.get('trade_list', []) if isinstance(trades_info, dict) else trades_info
        
        # 取引数
        total_trades = len(trades_list) if isinstance(trades_list, list) else trades_info.get('total_trades', 0)
        lines.append(f"総取引回数: {total_trades}")
        
        # 期間情報
        start_date = period.get('start_date', 'N/A')
        end_date = period.get('end_date', 'N/A')
        lines.append(f"データ期間: {start_date} - {end_date}")
        
        # データ行数とシグナル数は計算で求める
        lines.append(f"データ行数: {period.get('trading_days', 0)}")
        lines.append(f"有効シグナル数: {total_trades}")
        
        # パフォーマンス情報
        lines.append(f"初期資金: ¥{performance.get('initial_capital', 1000000):,.0f}")
        lines.append(f"最終ポートフォリオ値: ¥{performance.get('final_portfolio_value', 0):,.0f}")
        
        # 成果概要
        total_return = performance.get('total_return', 0) * 100  # パーセント表示のため100倍
        lines.append(f"総リターン: {total_return:.2f}%")
        
        # 勝率
        win_rate = performance.get('win_rate', 0) * 100  # パーセント表示のため100倍
        lines.append(f"勝率: {win_rate:.2f}%")
        
        lines.append("")
        return lines
    
    def _build_performance_section(self, analysis: Dict[str, Any]) -> List[str]:
        """パフォーマンス統計セクションを構築"""
        lines: List[str] = []
        lines.append("2. パフォーマンス統計")
        lines.append("-" * 40)
        
        # performance キーから正しいデータを取得
        performance = analysis.get('performance', {})
        trades_info = analysis.get('trades', {})
        trades_list = trades_info.get('trade_list', []) if isinstance(trades_info, dict) else trades_info
        
        # 取引数
        total_trades = len(trades_list) if isinstance(trades_list, list) else trades_info.get('total_trades', 0)
        
        if total_trades > 0:
            lines.append(f"総取引数: {total_trades}")
            lines.append(f"勝ちトレード数: {performance.get('winning_trades', 0)}")
            lines.append(f"負けトレード数: {performance.get('losing_trades', 0)}")
            lines.append(f"勝率: {performance.get('win_rate', 0) * 100:.2f}%")
            lines.append("")
            
            lines.append(f"平均利益: ¥{performance.get('avg_profit', 0):,.0f}")
            lines.append(f"平均損失: ¥{performance.get('avg_loss', 0):,.0f}")
            lines.append(f"最大利益: ¥{performance.get('max_profit', 0):,.0f}")
            lines.append(f"最大損失: ¥{performance.get('max_loss', 0):,.0f}")
            lines.append("")
            
            lines.append(f"総利益: ¥{performance.get('total_profit', 0):,.0f}")
            lines.append(f"総損失: ¥{performance.get('total_loss', 0):,.0f}")
            lines.append(f"純利益: ¥{performance.get('net_profit', 0):,.0f}")
            
            # プロフィットファクター
            pf = performance.get('profit_factor', 0)
            lines.append(f"プロフィットファクター: {pf:.2f}")
            
        else:
            lines.append("取引データなし")
        
        lines.append("")
        return lines
    
    def _build_expected_value_section(self, analysis: Dict[str, Any]) -> List[str]:
        """期待値分析セクションを構築"""
        lines: List[str] = []
        lines.append("3. 期待値分析")
        lines.append("-" * 40)
        
        trades = analysis.get('trades', [])
        
        if trades:
            # システム全体の期待値
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            system_expected_value = total_pnl / len(trades) if trades else 0
            
            lines.append(f"システム期待値 (1トレードあたり):")
            lines.append(f"  金額: ¥{system_expected_value:,.0f}")
            lines.append(f"  基準: {len(trades)}取引の平均")
            lines.append("")
            
            # 戦略別期待値
            strategy_ev = self._calculate_strategy_expected_values(trades)
            
            lines.append("戦略別期待値:")
            for strategy, ev_data in strategy_ev.items():
                lines.append(f"  {strategy}:")
                lines.append(f"    期待値: ¥{ev_data['expected_value']:,.0f}")
                lines.append(f"    取引数: {ev_data['trade_count']}")
                lines.append(f"    勝率: {ev_data['win_rate']:.2f}%")
            
            lines.append("")
            
            # 時間軸別期待値
            lines.append("期待値統計:")
            
            # 日次期待値 (仮定: 平均保有期間から計算)
            avg_holding_days = self._calculate_average_holding_period(trades)
            daily_ev = system_expected_value / max(avg_holding_days, 1)
            lines.append(f"  日次期待値: ¥{daily_ev:,.0f}")
            
            # 月次期待値 (20営業日)
            monthly_ev = daily_ev * 20
            lines.append(f"  月次期待値: ¥{monthly_ev:,.0f}")
            
            # 年次期待値 (250営業日)
            yearly_ev = daily_ev * 250
            lines.append(f"  年次期待値: ¥{yearly_ev:,.0f}")
            
        else:
            lines.append("期待値計算のための取引データなし")
        
        lines.append("")
        return lines
    
    def _calculate_strategy_expected_values(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """戦略別期待値を計算"""
        strategy_data: Dict[str, Dict[str, Any]] = {}
        
        for trade in trades:
            strategy = trade.get('strategy', 'Unknown')
            pnl = float(trade.get('pnl', 0))
            
            if strategy not in strategy_data:
                strategy_data[strategy] = {
                    'pnls': [],
                    'wins': 0,
                    'total': 0
                }
            
            strategy_data[strategy]['pnls'].append(pnl)
            strategy_data[strategy]['total'] += 1
            if pnl > 0:
                strategy_data[strategy]['wins'] += 1
        
        # 期待値を計算
        result: Dict[str, Dict[str, Any]] = {}
        for strategy, data in strategy_data.items():
            expected_value = float(np.mean(data['pnls'])) if data['pnls'] else 0.0  # type: ignore
            win_rate = (data['wins'] / data['total'] * 100) if data['total'] > 0 else 0.0
            
            result[strategy] = {
                'expected_value': expected_value,
                'trade_count': data['total'],
                'win_rate': win_rate
            }
        
        return result
    
    def _calculate_average_holding_period(self, trades: List[Dict[str, Any]]) -> float:
        """平均保有期間を計算（日数）"""
        holding_periods = []
        
        for trade in trades:
            entry_date = trade.get('entry_date')
            exit_date = trade.get('exit_date')
            
            if entry_date and exit_date:
                try:
                    if isinstance(entry_date, str):
                        entry_date = pd.to_datetime(entry_date)
                    if isinstance(exit_date, str):
                        exit_date = pd.to_datetime(exit_date)
                    
                    holding_days = (exit_date - entry_date).days
                    if holding_days > 0:
                        holding_periods.append(holding_days)
                except:
                    continue
        
        return np.mean(holding_periods) if holding_periods else 1.0
    
    def _build_strategy_analysis_section(self, analysis: Dict[str, Any]) -> List[str]:
        """戦略別詳細分析セクションを構築"""
        lines: List[str] = []
        lines.append("4. 戦略別詳細分析")
        lines.append("-" * 40)
        
        trades = analysis.get('trades', [])
        
        if trades:
            # 戦略別統計
            strategy_stats = self._analyze_strategy_performance(trades)
            
            for strategy, stats in strategy_stats.items():
                lines.append(f"戦略: {strategy}")
                lines.append(f"  取引回数: {stats['count']}")
                lines.append(f"  勝率: {stats['win_rate']:.2f}%")
                lines.append(f"  平均PnL: ¥{stats['avg_pnl']:,.0f}")
                lines.append(f"  総PnL: ¥{stats['total_pnl']:,.0f}")
                lines.append(f"  最大利益: ¥{stats['max_profit']:,.0f}")
                lines.append(f"  最大損失: ¥{stats['max_loss']:,.0f}")
                
                if stats['count'] > 0:
                    lines.append(f"  プロフィットファクター: {stats['profit_factor']:.2f}")
                
                lines.append("")
        else:
            lines.append("戦略分析のための取引データなし")
        
        lines.append("")
        return lines
    
    def _analyze_strategy_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """戦略別パフォーマンスを分析"""
        strategy_stats = {}
        
        for trade in trades:
            strategy = trade.get('strategy', 'Unknown')
            pnl = trade.get('pnl', 0)
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'pnls': [],
                    'profits': [],
                    'losses': [],
                    'count': 0,
                    'wins': 0
                }
            
            strategy_stats[strategy]['pnls'].append(pnl)
            strategy_stats[strategy]['count'] += 1
            
            if pnl > 0:
                strategy_stats[strategy]['wins'] += 1
                strategy_stats[strategy]['profits'].append(pnl)
            elif pnl < 0:
                strategy_stats[strategy]['losses'].append(abs(pnl))
        
        # 統計計算
        result = {}
        for strategy, data in strategy_stats.items():
            total_pnl = sum(data['pnls'])
            win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
            avg_pnl = total_pnl / data['count'] if data['count'] > 0 else 0
            
            max_profit = max(data['profits']) if data['profits'] else 0
            max_loss = max(data['losses']) if data['losses'] else 0
            
            total_profit = sum(data['profits']) if data['profits'] else 0
            total_loss = sum(data['losses']) if data['losses'] else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            result[strategy] = {
                'count': data['count'],
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'profit_factor': profit_factor
            }
        
        return result
    
    def _build_trade_details_section(self, analysis: Dict[str, Any]) -> List[str]:
        """取引詳細セクションを構築"""
        lines: List[str] = []
        lines.append("5. 取引詳細")
        lines.append("-" * 40)
        
        trades = analysis.get('trades', [])
        
        if trades:
            lines.append(f"総取引数: {len(trades)}")
            lines.append("")
            lines.append("取引履歴 (最初の10件):")
            lines.append("")
            
            # ヘッダー
            lines.append(f"{'No.':<4} {'戦略':<20} {'エントリー日':<12} {'エグジット日':<12} {'価格':<10} {'価格':<10} {'PnL':<12}")
            lines.append(f"{'':4} {'':20} {'':12} {'':12} {'(エントリー)':<10} {'(エグジット)':<10} {'(円)':<12}")
            lines.append("-" * 90)
            
            # 最初の10件を表示
            for i, trade in enumerate(trades[:10], 1):
                entry_date = str(trade.get('entry_date', ''))[:10]
                exit_date = str(trade.get('exit_date', ''))[:10]
                strategy = trade.get('strategy', 'Unknown')[:18]
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                pnl = trade.get('pnl', 0)
                
                lines.append(f"{i:<4} {strategy:<20} {entry_date:<12} {exit_date:<12} "
                           f"{entry_price:<10.2f} {exit_price:<10.2f} {pnl:<12,.0f}")
            
            if len(trades) > 10:
                lines.append(f"... および他 {len(trades) - 10} 件")
                
        else:
            lines.append("取引詳細なし")
        
        lines.append("")
        return lines
    
    def _build_parameters_section(self, optimized_params: Dict[str, Dict[str, Any]]) -> List[str]:
        """パラメータ情報セクションを構築"""
        lines: List[str] = []
        lines.append("6. 使用パラメータ")
        lines.append("-" * 40)
        
        for strategy, params in optimized_params.items():
            lines.append(f"戦略: {strategy}")
            for param_name, param_value in params.items():
                lines.append(f"  {param_name}: {param_value}")
            lines.append("")
        
        return lines
    
    def _build_risk_analysis_section(self, analysis: Dict[str, Any]) -> List[str]:
        """リスク分析セクションを構築"""
        lines: List[str] = []
        lines.append("7. リスク分析")
        lines.append("-" * 40)
        
        trades = analysis.get('trades', [])
        
        if trades:
            pnls = [trade.get('pnl', 0) for trade in trades]
            
            # ドローダウン分析
            lines.append("ドローダウン分析:")
            max_dd = self._calculate_max_drawdown(pnls)
            lines.append(f"  最大ドローダウン: ¥{max_dd:,.0f}")
            
            # ボラティリティ分析
            std_dev = np.std(pnls) if pnls else 0
            lines.append(f"  PnL標準偏差: ¥{std_dev:,.0f}")
            
            # VaR分析 (5%値)
            var_5 = np.percentile(pnls, 5) if pnls else 0
            lines.append(f"  VaR (5%): ¥{var_5:,.0f}")
            
            # シャープレシオ風指標
            avg_pnl = np.mean(pnls) if pnls else 0
            risk_adj_return = avg_pnl / std_dev if std_dev > 0 else 0
            lines.append(f"  リスク調整後リターン: {risk_adj_return:.3f}")
            
        else:
            lines.append("リスク分析のための取引データなし")
        
        lines.append("")
        return lines
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """最大ドローダウンを計算"""
        if not pnls:
            return 0.0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown))
    
    def _build_statistical_summary_section(self, analysis: Dict[str, Any]) -> List[str]:
        """統計サマリーセクションを構築"""
        lines: List[str] = []
        lines.append("8. 統計サマリー")
        lines.append("-" * 40)
        
        # 正しいキー名を使用してデータを取得
        performance = analysis.get('performance', {})
        period = analysis.get('period', {})
        trades_info = analysis.get('trades', {})
        trades_list = trades_info.get('trade_list', []) if isinstance(trades_info, dict) else trades_info
        
        # 実行サマリー
        lines.append("実行サマリー:")
        lines.append(f"  レポート生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  総データ行数: {period.get('trading_days', 0)}")
        lines.append(f"  有効シグナル数: {len(trades_list) if isinstance(trades_list, list) else 0}")
        lines.append(f"  実行取引数: {len(trades_list) if isinstance(trades_list, list) else 0}")
        lines.append("")
        
        # パフォーマンス要約
        lines.append("パフォーマンス要約:")
        lines.append(f"  初期資金: ¥{performance.get('initial_capital', 1000000):,.0f}")
        lines.append(f"  最終資金: ¥{performance.get('final_portfolio_value', 0):,.0f}")
        lines.append(f"  総リターン: {performance.get('total_return', 0) * 100:.2f}%")
        
        if trades_list and len(trades_list) > 0:
            # 取引リストから期待値を計算
            if isinstance(trades_list, list):
                pnls = [trade.get('pnl', 0) for trade in trades_list if isinstance(trade, dict)]
                avg_pnl = np.mean(pnls) if pnls else 0
            else:
                avg_pnl = 0
                
            lines.append(f"  システム期待値: ¥{avg_pnl:,.0f}")
            lines.append(f"  勝率: {performance.get('win_rate', 0) * 100:.2f}%")
            lines.append(f"  プロフィットファクター: {performance.get('profit_factor', 0):.2f}")
        else:
            lines.append(f"  システム期待値: ¥0")
            lines.append(f"  勝率: 0.00%")
            lines.append(f"  プロフィットファクター: 0.00")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("レポート終了")
        lines.append("=" * 80)
        
        return lines


def generate_main_text_report(stock_data: pd.DataFrame, 
                            ticker: str,
                            optimized_params: Optional[Dict[str, Dict[str, Any]]] = None,
                            output_dir: Optional[str] = None) -> str:
    """
    main.pyの実行結果をテキストレポート形式で出力する関数
    
    Parameters:
        stock_data (pd.DataFrame): バックテスト結果データ
        ticker (str): 銘柄コード
        optimized_params (Optional[Dict]): 最適化パラメータ
        output_dir (Optional[str]): 出力ディレクトリ
        
    Returns:
        str: 生成されたレポートファイルのパス
    """
    try:
        reporter = MainTextReporter()
        return reporter.generate_comprehensive_report(
            stock_data, ticker, optimized_params, output_dir
        )
    except Exception as e:
        logger.error(f"テキストレポート生成エラー: {e}")
        return ""


if __name__ == "__main__":
    # テスト用のコード
    print("Main Text Reporter モジュールのテスト実行")
    
    # ダミーデータでテスト
    test_data = pd.DataFrame({
        'Entry_Signal': [1, 0, 0, -1, 0],
        'Exit_Signal': [0, 0, 0, -1, 0],
        'Strategy': ['VWAPBreakoutStrategy', '', '', 'VWAPBreakoutStrategy', ''],
        'Adj Close': [100, 102, 101, 99, 98]
    }, index=pd.date_range('2024-01-01', periods=5))
    
    reporter = MainTextReporter()
    result = reporter.generate_comprehensive_report(test_data, 'TEST')
    
    if result:
        print(f"テストレポート生成成功: {result}")
    else:
        print("テストレポート生成失敗")
