"""
DSSMS統合システム - DSSMSExcelExporter
専用Excelエクスポート機能・グラフ生成・統計出力クラス

Author: AI Assistant
Created: 2025-09-28
Phase: Phase 3 Tier 3 実装
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Excel操作ライブラリ
try:
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.chart import LineChart, BarChart, ScatterChart, Reference
    from openpyxl.chart.series import Series
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment, NamedStyle
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.drawing.image import Image
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logging.warning("openpyxl not available - Excel export will be limited")

# グラフ生成ライブラリ
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available - chart generation disabled")

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.logger_config import setup_logger


class ExportError(Exception):
    """エクスポート関連エラー"""
    pass


class DSSMSExcelExporter:
    """
    DSSMS統合システム専用Excelエクスポート機能
    
    Responsibilities:
    - バックテスト結果のExcel出力
    - 銘柄切替履歴・統計の出力
    - パフォーマンス分析グラフ生成
    - 統合レポート作成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        DSSMSExcelExporter初期化
        
        Args:
            config: エクスポート設定
        
        Raises:
            ExportError: 初期化失敗・依存ライブラリ不足
        """
        try:
            # 依存関係チェック
            if not EXCEL_AVAILABLE:
                raise ExportError("openpyxl library required for Excel export")
            
            # 設定初期化
            self.config = config or {}
            self.export_config = self.config.get('export_settings', {})
            
            # 出力設定
            self.default_output_dir = self.config.get('output_directory', 'output/dssms_reports')
            self.include_charts = self.export_config.get('include_charts', True) and PLOTTING_AVAILABLE
            self.chart_style = self.export_config.get('chart_style', 'seaborn')
            self.compress_excel = self.export_config.get('compress_excel', True)
            
            # スタイル設定
            self._setup_excel_styles()
            
            # ログ設定
            self.logger = setup_logger(f"{self.__class__.__name__}")
            
            # 状態管理
            self.export_history = []
            self.current_workbook = None
            self.chart_counter = 0
            
            self.logger.info("DSSMSExcelExporter初期化完了")
            
        except Exception as e:
            self.logger.error(f"DSSMSExcelExporter初期化エラー: {e}")
            raise ExportError(f"初期化失敗: {e}")
    
    def export_backtest_results(self, backtest_results: Dict[str, Any], 
                               output_path: Optional[str] = None) -> str:
        """
        バックテスト結果をExcelにエクスポート
        
        Args:
            backtest_results: バックテスト統合結果
            output_path: 出力ファイルパス（Noneなら自動生成）
        
        Returns:
            str: 実際の出力ファイルパス
        
        Raises:
            ExportError: エクスポート失敗
        
        Example:
            results = {...}  # バックテスト結果
            export_path = exporter.export_backtest_results(results)
            print(f"エクスポート完了: {export_path}")
        """
        try:
            # 出力パス決定
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dssms_backtest_results_{timestamp}.xlsx"
                output_path = os.path.join(self.default_output_dir, filename)
            
            # 出力ディレクトリ作成
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Excelワークブック作成
            wb = Workbook()
            self.current_workbook = wb
            self.chart_counter = 0
            
            # デフォルトシート削除
            wb.remove(wb.active)
            
            # 各シート作成
            self._create_summary_sheet(wb, backtest_results)
            self._create_daily_results_sheet(wb, backtest_results)
            self._create_symbol_switches_sheet(wb, backtest_results)
            self._create_performance_metrics_sheet(wb, backtest_results)
            self._create_strategy_analysis_sheet(wb, backtest_results)
            
            # グラフ生成（有効な場合）
            if self.include_charts:
                self._create_performance_charts(wb, backtest_results)
                self._create_switch_analysis_charts(wb, backtest_results)
            
            # ファイル保存
            wb.save(output_path)
            
            # エクスポート履歴記録
            export_record = {
                'timestamp': datetime.now(),
                'output_path': output_path,
                'data_points': len(backtest_results.get('daily_results', [])),
                'switches': len(backtest_results.get('switch_history', [])),
                'file_size_mb': self._get_file_size_mb(output_path)
            }
            
            self.export_history.append(export_record)
            
            self.logger.info(f"バックテスト結果エクスポート成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"バックテスト結果エクスポートエラー: {e}")
            raise ExportError(f"エクスポート失敗: {e}")
    
    def export_performance_analysis(self, performance_data: Dict[str, Any],
                                  output_path: Optional[str] = None) -> str:
        """
        パフォーマンス分析をExcelにエクスポート
        
        Args:
            performance_data: パフォーマンス分析データ
            output_path: 出力ファイルパス
        
        Returns:
            str: 出力ファイルパス
        """
        try:
            # 出力パス決定
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dssms_performance_analysis_{timestamp}.xlsx"
                output_path = os.path.join(self.default_output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Excelワークブック作成
            wb = Workbook()
            wb.remove(wb.active)
            
            # パフォーマンス分析シート作成
            self._create_performance_summary_sheet(wb, performance_data)
            self._create_execution_time_analysis_sheet(wb, performance_data)
            self._create_memory_analysis_sheet(wb, performance_data)
            self._create_reliability_analysis_sheet(wb, performance_data)
            
            # パフォーマンスグラフ
            if self.include_charts:
                self._create_performance_trend_charts(wb, performance_data)
            
            wb.save(output_path)
            
            self.logger.info(f"パフォーマンス分析エクスポート成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"パフォーマンス分析エクスポートエラー: {e}")
            raise ExportError(f"パフォーマンス分析エクスポート失敗: {e}")
    
    def export_switch_analysis(self, switch_data: Dict[str, Any],
                             output_path: Optional[str] = None) -> str:
        """
        銘柄切替分析をExcelにエクスポート
        
        Args:
            switch_data: 銘柄切替分析データ
            output_path: 出力ファイルパス
        
        Returns:
            str: 出力ファイルパス
        """
        try:
            # 出力パス決定
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dssms_switch_analysis_{timestamp}.xlsx"
                output_path = os.path.join(self.default_output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Excel作成
            wb = Workbook()
            wb.remove(wb.active)
            
            # 切替分析シート
            self._create_switch_summary_sheet(wb, switch_data)
            self._create_switch_timeline_sheet(wb, switch_data)
            self._create_switch_effectiveness_sheet(wb, switch_data)
            
            wb.save(output_path)
            
            self.logger.info(f"銘柄切替分析エクスポート成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"銘柄切替分析エクスポートエラー: {e}")
            raise ExportError(f"銘柄切替分析エクスポート失敗: {e}")
    
    def create_comprehensive_report(self, all_data: Dict[str, Any],
                                  output_path: Optional[str] = None) -> str:
        """
        包括的統合レポートを作成
        
        Args:
            all_data: 全統合データ
            output_path: 出力ファイルパス
        
        Returns:
            str: 出力ファイルパス
        """
        try:
            # 出力パス決定
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dssms_comprehensive_report_{timestamp}.xlsx"
                output_path = os.path.join(self.default_output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 包括レポート作成
            wb = Workbook()
            wb.remove(wb.active)
            
            # 全シート作成
            self._create_executive_summary_sheet(wb, all_data)
            self._create_comprehensive_performance_sheet(wb, all_data)
            self._create_comprehensive_analysis_sheet(wb, all_data)
            self._create_recommendations_sheet(wb, all_data)
            
            # 統合グラフ（可能な場合）
            if self.include_charts:
                self._create_comprehensive_charts(wb, all_data)
            
            wb.save(output_path)
            
            self.logger.info(f"包括的レポート作成成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"包括的レポート作成エラー: {e}")
            raise ExportError(f"包括的レポート作成失敗: {e}")
    
    def _setup_excel_styles(self) -> None:
        """Excel スタイル設定"""
        try:
            # ヘッダースタイル
            self.header_style = NamedStyle(name="header_style")
            self.header_style.font = Font(bold=True, color="FFFFFF")
            self.header_style.fill = PatternFill("solid", fgColor="366092")
            self.header_style.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            self.header_style.alignment = Alignment(horizontal="center", vertical="center")
            
            # データスタイル
            self.data_style = NamedStyle(name="data_style")
            self.data_style.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            self.data_style.alignment = Alignment(horizontal="left", vertical="center")
            
            # 数値スタイル
            self.number_style = NamedStyle(name="number_style")
            self.number_style.number_format = '#,##0.00'
            self.number_style.border = self.data_style.border
            self.number_style.alignment = Alignment(horizontal="right", vertical="center")
            
            # パーセントスタイル
            self.percent_style = NamedStyle(name="percent_style")
            self.percent_style.number_format = '0.00%'
            self.percent_style.border = self.data_style.border
            self.percent_style.alignment = Alignment(horizontal="right", vertical="center")
            
        except Exception as e:
            self.logger.warning(f"Excelスタイル設定エラー: {e}")
    
    def _create_summary_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """サマリーシート作成"""
        try:
            ws = wb.create_sheet("サマリー", 0)
            
            # ヘッダー
            headers = [
                "DSSMS統合バックテスト結果サマリー",
                f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            
            for i, header in enumerate(headers):
                ws[f'A{i+1}'] = header
                ws[f'A{i+1}'].font = Font(bold=True, size=14)
            
            # 基本統計
            row = 4
            basic_stats = [
                ("バックテスト期間", f"{results.get('start_date', 'N/A')} - {results.get('end_date', 'N/A')}"),
                ("初期資本", f"{results.get('initial_capital', 0):,.0f}円"),
                ("最終資本", f"{results.get('final_capital', 0):,.0f}円"),
                ("総収益率", f"{results.get('total_return_rate', 0):.2%}"),
                ("総取引日数", f"{len(results.get('daily_results', []))}日"),
                ("銘柄切替回数", f"{len(results.get('switch_history', []))}回"),
                ("成功取引率", f"{results.get('success_rate', 0):.1%}"),
                ("最大ドローダウン", f"{results.get('max_drawdown', 0):.2%}"),
                ("シャープレシオ", f"{results.get('sharpe_ratio', 0):.3f}"),
                ("平均日次収益率", f"{results.get('average_daily_return', 0):.3%}")
            ]
            
            for label, value in basic_stats:
                ws[f'A{row}'] = label
                ws[f'B{row}'] = value
                ws[f'A{row}'].font = Font(bold=True)
                row += 1
            
            # 列幅調整
            ws.column_dimensions['A'].width = 20
            ws.column_dimensions['B'].width = 25
            
        except Exception as e:
            self.logger.warning(f"サマリーシート作成エラー: {e}")
    
    def _create_daily_results_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """日次結果シート作成"""
        try:
            ws = wb.create_sheet("日次結果")
            
            # ヘッダー
            headers = [
                "日付", "銘柄", "ポートフォリオ価値", "日次収益", "日次収益率",
                "実行時間(ms)", "成功", "戦略結果", "備考"
            ]
            
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=header)
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="D9E1F2")
            
            # 日次データ
            daily_results = results.get('daily_results', [])
            for row, daily_data in enumerate(daily_results, 2):
                ws.cell(row=row, column=1, value=daily_data.get('date', ''))
                ws.cell(row=row, column=2, value=daily_data.get('symbol', ''))
                ws.cell(row=row, column=3, value=daily_data.get('portfolio_value', 0))
                ws.cell(row=row, column=4, value=daily_data.get('daily_return', 0))
                ws.cell(row=row, column=5, value=daily_data.get('daily_return_rate', 0))
                ws.cell(row=row, column=6, value=daily_data.get('execution_time_ms', 0))
                ws.cell(row=row, column=7, value="成功" if daily_data.get('success', False) else "失敗")
                ws.cell(row=row, column=8, value=str(daily_data.get('strategy_result', {}).get('summary', '')))
                ws.cell(row=row, column=9, value=daily_data.get('notes', ''))
            
            # 列幅調整
            for col in range(1, len(headers) + 1):
                ws.column_dimensions[chr(64 + col)].width = 15
            
        except Exception as e:
            self.logger.warning(f"日次結果シート作成エラー: {e}")
    
    def _create_symbol_switches_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """銘柄切替シート作成"""
        try:
            ws = wb.create_sheet("銘柄切替履歴")
            
            # ヘッダー
            headers = [
                "切替日", "切替前銘柄", "切替後銘柄", "切替理由", "切替コスト",
                "保有期間", "切替前収益", "切替効果", "制限チェック"
            ]
            
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=header)
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="E2EFDA")
            
            # 切替データ
            switch_history = results.get('switch_history', [])
            for row, switch_data in enumerate(switch_history, 2):
                ws.cell(row=row, column=1, value=switch_data.get('date', ''))
                ws.cell(row=row, column=2, value=switch_data.get('from_symbol', ''))
                ws.cell(row=row, column=3, value=switch_data.get('to_symbol', ''))
                ws.cell(row=row, column=4, value=switch_data.get('reason', ''))
                ws.cell(row=row, column=5, value=switch_data.get('switch_cost', 0))
                ws.cell(row=row, column=6, value=switch_data.get('holding_days', 0))
                ws.cell(row=row, column=7, value=switch_data.get('previous_return', 0))
                ws.cell(row=row, column=8, value=switch_data.get('switch_effectiveness', 0))
                ws.cell(row=row, column=9, value=switch_data.get('restriction_status', ''))
            
            # 列幅調整
            for col in range(1, len(headers) + 1):
                ws.column_dimensions[chr(64 + col)].width = 15
            
        except Exception as e:
            self.logger.warning(f"銘柄切替シート作成エラー: {e}")
    
    def _create_performance_metrics_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """パフォーマンスメトリクスシート作成"""
        try:
            ws = wb.create_sheet("パフォーマンス指標")
            
            # パフォーマンス指標
            performance_data = results.get('performance_metrics', {})
            
            row = 1
            for category, metrics in performance_data.items():
                # カテゴリヘッダー
                ws[f'A{row}'] = category.replace('_', ' ').title()
                ws[f'A{row}'].font = Font(bold=True, size=12)
                ws[f'A{row}'].fill = PatternFill("solid", fgColor="FFF2CC")
                row += 1
                
                # メトリクス
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        ws[f'B{row}'] = metric.replace('_', ' ').title()
                        ws[f'C{row}'] = value
                        row += 1
                else:
                    ws[f'B{row}'] = str(metrics)
                    row += 1
                
                row += 1  # 空行
            
            # 列幅調整
            ws.column_dimensions['A'].width = 25
            ws.column_dimensions['B'].width = 25
            ws.column_dimensions['C'].width = 20
            
        except Exception as e:
            self.logger.warning(f"パフォーマンスメトリクスシート作成エラー: {e}")
    
    def _create_strategy_analysis_sheet(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """戦略分析シート作成"""
        try:
            ws = wb.create_sheet("戦略分析")
            
            # 戦略統計
            strategy_stats = results.get('strategy_statistics', {})
            
            # ヘッダー
            headers = ["戦略名", "実行回数", "成功回数", "成功率", "平均収益", "最大収益", "最小収益", "総合評価"]
            
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=header)
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="FCE4D6")
            
            # 戦略データ
            row = 2
            for strategy_name, stats in strategy_stats.items():
                ws.cell(row=row, column=1, value=strategy_name)
                ws.cell(row=row, column=2, value=stats.get('execution_count', 0))
                ws.cell(row=row, column=3, value=stats.get('success_count', 0))
                ws.cell(row=row, column=4, value=stats.get('success_rate', 0))
                ws.cell(row=row, column=5, value=stats.get('average_return', 0))
                ws.cell(row=row, column=6, value=stats.get('max_return', 0))
                ws.cell(row=row, column=7, value=stats.get('min_return', 0))
                ws.cell(row=row, column=8, value=stats.get('overall_rating', 'N/A'))
                row += 1
            
            # 列幅調整
            for col in range(1, len(headers) + 1):
                ws.column_dimensions[chr(64 + col)].width = 15
            
        except Exception as e:
            self.logger.warning(f"戦略分析シート作成エラー: {e}")
    
    def _create_performance_charts(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """パフォーマンスグラフ作成"""
        try:
            if not self.include_charts:
                return
            
            ws = wb.create_sheet("パフォーマンスグラフ")
            
            # ポートフォリオ価値推移グラフ
            self._create_portfolio_value_chart(ws, results)
            
            # 日次収益率分布グラフ
            self._create_daily_return_distribution_chart(ws, results)
            
        except Exception as e:
            self.logger.warning(f"パフォーマンスグラフ作成エラー: {e}")
    
    def _create_portfolio_value_chart(self, ws, results: Dict[str, Any]) -> None:
        """ポートフォリオ価値推移グラフ"""
        try:
            daily_results = results.get('daily_results', [])
            if not daily_results:
                return
            
            # データ準備
            dates = [result.get('date', '') for result in daily_results]
            values = [result.get('portfolio_value', 0) for result in daily_results]
            
            # Excelチャート作成
            chart = LineChart()
            chart.title = "ポートフォリオ価値推移"
            chart.style = 10
            chart.x_axis.title = "日付"
            chart.y_axis.title = "ポートフォリオ価値 (円)"
            
            # データ書き込み
            row_start = 2
            for i, (date, value) in enumerate(zip(dates, values)):
                ws[f'A{row_start + i}'] = date
                ws[f'B{row_start + i}'] = value
            
            # データ範囲設定
            data_range = Reference(ws, min_col=2, min_row=row_start, max_row=row_start + len(values) - 1)
            chart.add_data(data_range, titles_from_data=False)
            
            # チャート配置
            ws.add_chart(chart, "D2")
            
        except Exception as e:
            self.logger.warning(f"ポートフォリオ価値グラフ作成エラー: {e}")
    
    def _create_daily_return_distribution_chart(self, ws, results: Dict[str, Any]) -> None:
        """日次収益率分布グラフ"""
        try:
            daily_results = results.get('daily_results', [])
            if not daily_results:
                return
            
            # 収益率データ準備
            returns = [result.get('daily_return_rate', 0) for result in daily_results if result.get('daily_return_rate') is not None]
            
            if not returns:
                return
            
            # ヒストグラム用にビン作成
            min_return = min(returns)
            max_return = max(returns)
            bins = np.linspace(min_return, max_return, 20)
            
            # ヒストグラムデータ
            hist, bin_edges = np.histogram(returns, bins=bins)
            
            # バーチャート作成
            chart = BarChart()
            chart.title = "日次収益率分布"
            chart.style = 10
            chart.x_axis.title = "収益率区間"
            chart.y_axis.title = "頻度"
            
            # データ書き込み（別の列に）
            col_start = 'F'
            row_start = 2
            
            for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
                ws[f'{col_start}{row_start + i}'] = f"{edge:.2%}-{bin_edges[i+1]:.2%}"
                ws[f'{chr(ord(col_start) + 1)}{row_start + i}'] = count
            
            # データ範囲設定
            data_range = Reference(ws, min_col=ord(col_start) - ord('A') + 2, min_row=row_start, 
                                 max_row=row_start + len(hist) - 1)
            chart.add_data(data_range, titles_from_data=False)
            
            # チャート配置
            ws.add_chart(chart, "D20")
            
        except Exception as e:
            self.logger.warning(f"日次収益率分布グラフ作成エラー: {e}")
    
    def _create_switch_analysis_charts(self, wb: Workbook, results: Dict[str, Any]) -> None:
        """切替分析グラフ作成"""
        try:
            if not self.include_charts:
                return
            
            ws = wb.create_sheet("切替分析グラフ")
            
            # 月次切替回数グラフ
            self._create_monthly_switch_chart(ws, results)
            
        except Exception as e:
            self.logger.warning(f"切替分析グラフ作成エラー: {e}")
    
    def _create_monthly_switch_chart(self, ws, results: Dict[str, Any]) -> None:
        """月次切替回数グラフ"""
        try:
            switch_history = results.get('switch_history', [])
            if not switch_history:
                return
            
            # 月別集計
            monthly_counts = {}
            for switch in switch_history:
                date = switch.get('date', '')
                if date:
                    month_key = date[:7] if isinstance(date, str) else date.strftime('%Y-%m')
                    monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
            
            # チャート作成
            chart = BarChart()
            chart.title = "月次銘柄切替回数"
            chart.style = 10
            chart.x_axis.title = "月"
            chart.y_axis.title = "切替回数"
            
            # データ書き込み
            row_start = 2
            for i, (month, count) in enumerate(sorted(monthly_counts.items())):
                ws[f'A{row_start + i}'] = month
                ws[f'B{row_start + i}'] = count
            
            # データ範囲設定
            data_range = Reference(ws, min_col=2, min_row=row_start, 
                                 max_row=row_start + len(monthly_counts) - 1)
            chart.add_data(data_range, titles_from_data=False)
            
            # チャート配置
            ws.add_chart(chart, "D2")
            
        except Exception as e:
            self.logger.warning(f"月次切替回数グラフ作成エラー: {e}")
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """ファイルサイズ（MB）取得"""
        try:
            size_bytes = os.path.getsize(file_path)
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return 0.0
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """エクスポート統計取得"""
        try:
            if not self.export_history:
                return {'status': 'no_exports', 'total_exports': 0}
            
            total_exports = len(self.export_history)
            total_file_size = sum(record['file_size_mb'] for record in self.export_history)
            
            recent_exports = [
                record for record in self.export_history
                if (datetime.now() - record['timestamp']).days <= 30
            ]
            
            return {
                'status': 'active',
                'total_exports': total_exports,
                'total_file_size_mb': round(total_file_size, 2),
                'recent_exports_30days': len(recent_exports),
                'average_file_size_mb': round(total_file_size / total_exports, 2) if total_exports > 0 else 0,
                'last_export': self.export_history[-1]['timestamp'] if self.export_history else None,
                'charts_supported': self.include_charts
            }
            
        except Exception as e:
            self.logger.warning(f"エクスポート統計取得エラー: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # 追加のシート作成メソッド（省略版）
    def _create_performance_summary_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """パフォーマンスサマリーシート作成（簡略版）"""
        ws = wb.create_sheet("パフォーマンスサマリー")
        ws['A1'] = "パフォーマンス統計サマリー"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_execution_time_analysis_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """実行時間分析シート作成（簡略版）"""
        ws = wb.create_sheet("実行時間分析")
        ws['A1'] = "実行時間統計"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_memory_analysis_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """メモリ分析シート作成（簡略版）"""
        ws = wb.create_sheet("メモリ分析")
        ws['A1'] = "メモリ使用量統計"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_reliability_analysis_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """信頼性分析シート作成（簡略版）"""
        ws = wb.create_sheet("信頼性分析")
        ws['A1'] = "システム信頼性統計"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_performance_trend_charts(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """パフォーマンストレンドグラフ作成（簡略版）"""
        if self.include_charts:
            ws = wb.create_sheet("パフォーマンストレンド")
            ws['A1'] = "パフォーマンス推移グラフ"
    
    def _create_switch_summary_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """切替サマリーシート作成（簡略版）"""
        ws = wb.create_sheet("切替サマリー")
        ws['A1'] = "銘柄切替統計サマリー"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_switch_timeline_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """切替タイムラインシート作成（簡略版）"""
        ws = wb.create_sheet("切替タイムライン")
        ws['A1'] = "銘柄切替時系列"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_switch_effectiveness_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """切替効果分析シート作成（簡略版）"""
        ws = wb.create_sheet("切替効果分析")
        ws['A1'] = "切替効果統計"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_executive_summary_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """エグゼクティブサマリーシート作成（簡略版）"""
        ws = wb.create_sheet("エグゼクティブサマリー", 0)
        ws['A1'] = "DSSMS統合システム - エグゼクティブサマリー"
        ws['A1'].font = Font(bold=True, size=16)
    
    def _create_comprehensive_performance_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """包括的パフォーマンスシート作成（簡略版）"""
        ws = wb.create_sheet("包括的パフォーマンス")
        ws['A1'] = "包括的パフォーマンス分析"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_comprehensive_analysis_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """包括的分析シート作成（簡略版）"""
        ws = wb.create_sheet("包括的分析")
        ws['A1'] = "統合分析結果"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_recommendations_sheet(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """推奨事項シート作成（簡略版）"""
        ws = wb.create_sheet("推奨事項")
        ws['A1'] = "推奨事項・改善提案"
        ws['A1'].font = Font(bold=True, size=14)
    
    def _create_comprehensive_charts(self, wb: Workbook, data: Dict[str, Any]) -> None:
        """包括的グラフ作成（簡略版）"""
        if self.include_charts:
            ws = wb.create_sheet("統合グラフ")
            ws['A1'] = "統合分析グラフ"


def main():
    """DSSMSExcelExporter 動作テスト"""
    print("DSSMSExcelExporter 動作テスト")
    print("=" * 50)
    
    try:
        # 1. 初期化テスト
        config = {
            'output_directory': 'output/test_exports',
            'export_settings': {
                'include_charts': True,
                'chart_style': 'seaborn',
                'compress_excel': True
            }
        }
        
        exporter = DSSMSExcelExporter(config)
        print("✅ DSSMSExcelExporter初期化成功")
        
        # 2. サンプルデータ作成
        print(f"\n📊 サンプルデータ作成:")
        
        # バックテスト結果サンプル
        sample_backtest_results = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 1000000,
            'final_capital': 1150000,
            'total_return_rate': 0.15,
            'success_rate': 0.85,
            'max_drawdown': -0.08,
            'sharpe_ratio': 1.25,
            'average_daily_return': 0.0015,
            'daily_results': [
                {
                    'date': '2023-01-01',
                    'symbol': '7203',
                    'portfolio_value': 1000000,
                    'daily_return': 5000,
                    'daily_return_rate': 0.005,
                    'execution_time_ms': 850,
                    'success': True,
                    'strategy_result': {'summary': 'VWAPBreakout成功'},
                    'notes': 'システム正常'
                },
                {
                    'date': '2023-01-02',
                    'symbol': '7203',
                    'portfolio_value': 1005000,
                    'daily_return': -2000,
                    'daily_return_rate': -0.002,
                    'execution_time_ms': 920,
                    'success': True,
                    'strategy_result': {'summary': 'Momentum投資実行'},
                    'notes': ''
                },
                {
                    'date': '2023-01-03',
                    'symbol': '9984',
                    'portfolio_value': 1003000,
                    'daily_return': 8000,
                    'daily_return_rate': 0.008,
                    'execution_time_ms': 780,
                    'success': True,
                    'strategy_result': {'summary': 'BreakOut戦略成功'},
                    'notes': '銘柄切替実行'
                }
            ],
            'switch_history': [
                {
                    'date': '2023-01-03',
                    'from_symbol': '7203',
                    'to_symbol': '9984',
                    'reason': 'DSS選択結果',
                    'switch_cost': 1000,
                    'holding_days': 2,
                    'previous_return': 0.003,
                    'switch_effectiveness': 0.012,
                    'restriction_status': '制限内'
                }
            ],
            'performance_metrics': {
                'execution_performance': {
                    'average_execution_time_ms': 850,
                    'max_execution_time_ms': 920,
                    'success_rate': 0.85
                },
                'financial_performance': {
                    'total_return': 150000,
                    'volatility': 0.15,
                    'max_drawdown': -0.08
                }
            },
            'strategy_statistics': {
                'VWAPBreakoutStrategy': {
                    'execution_count': 120,
                    'success_count': 95,
                    'success_rate': 0.79,
                    'average_return': 0.003,
                    'max_return': 0.025,
                    'min_return': -0.015,
                    'overall_rating': 'Good'
                },
                'MomentumInvestingStrategy': {
                    'execution_count': 85,
                    'success_count': 75,
                    'success_rate': 0.88,
                    'average_return': 0.004,
                    'max_return': 0.030,
                    'min_return': -0.012,
                    'overall_rating': 'Excellent'
                }
            }
        }
        
        print(f"✅ サンプルデータ準備完了: {len(sample_backtest_results['daily_results'])}日分")
        
        # 3. バックテスト結果エクスポートテスト
        print(f"\n📈 バックテスト結果エクスポートテスト:")
        
        export_path = exporter.export_backtest_results(sample_backtest_results)
        print(f"✅ バックテスト結果エクスポート成功: {export_path}")
        
        # 4. パフォーマンス分析エクスポートテスト
        print(f"\n⚡ パフォーマンス分析エクスポートテスト:")
        
        performance_data = {
            'execution': {
                'average_time_ms': 850,
                'max_time_ms': 920,
                'success_rate': 0.85,
                'data_points': 252
            },
            'memory': {
                'average_usage_mb': 256,
                'peak_usage_mb': 412,
                'efficiency_rating': 0.78
            },
            'reliability': {
                'success_rate': 0.85,
                'consecutive_failures': 2,
                'uptime_percentage': 99.2
            }
        }
        
        perf_export_path = exporter.export_performance_analysis(performance_data)
        print(f"✅ パフォーマンス分析エクスポート成功: {perf_export_path}")
        
        # 5. エクスポート統計確認
        print(f"\n📊 エクスポート統計確認:")
        stats = exporter.get_export_statistics()
        
        print(f"✅ エクスポート統計取得成功:")
        print(f"  - 総エクスポート数: {stats['total_exports']}")
        print(f"  - 総ファイルサイズ: {stats['total_file_size_mb']}MB")
        print(f"  - 平均ファイルサイズ: {stats['average_file_size_mb']}MB")
        print(f"  - グラフサポート: {'有効' if stats['charts_supported'] else '無効'}")
        
        print(f"\n🎉 DSSMSExcelExporter テスト完了！")
        print(f"実装機能: Excel出力、グラフ生成、統計レポート、包括分析")
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()