#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSSMS包括的問題分析システム
実行ログと出力ファイルの不整合を含む全ての問題を検出
"""

import os
import sys
import pandas as pd
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import glob
from typing import Dict, List, Any, Optional
import openpyxl

class DSSMSComprehensiveAnalyzer:
    def __init__(self):
        self.project_root = Path(".")
        self.issues = []
        self.analysis_results = {}
        self.execution_log = ""
        
    def run_comprehensive_analysis(self) -> str:
        """包括的分析の実行"""
        print("🚀 DSSMS包括的問題分析開始")
        print("=" * 70)
        
        # 1. DSSMSバックテスターを実行してログを取得
        print("\n1️⃣ DSSMSバックテスター実行・ログ取得")
        self._execute_dssms_and_capture_log()
        
        # 2. 最新の出力ファイルを特定
        print("\n2️⃣ 最新出力ファイル特定")
        latest_files = self._identify_latest_output_files()
        
        # 3. ログと出力ファイルの不整合分析
        print("\n3️⃣ ログ⇔出力ファイル不整合分析")
        self._analyze_log_output_inconsistency(latest_files)
        
        # 4. データ期間の詳細分析
        print("\n4️⃣ データ期間詳細分析")
        self._analyze_temporal_inconsistencies(latest_files)
        
        # 5. パフォーマンス計算の整合性検証
        print("\n5️⃣ パフォーマンス計算整合性検証")
        self._verify_performance_calculations(latest_files)
        
        # 6. 日付データの整合性チェック
        print("\n6️⃣ 日付データ整合性チェック")
        self._check_date_data_consistency(latest_files)
        
        # 7. 問題の分類と修正提案
        print("\n7️⃣ 問題分類・修正提案生成")
        self._categorize_and_generate_fixes()
        
        # 8. 詳細レポート生成
        print("\n8️⃣ 詳細レポート生成")
        report_path = self._generate_comprehensive_report()
        
        return report_path
    
    def _execute_dssms_and_capture_log(self):
        """DSSMSバックテスター実行・ログキャプチャ"""
        try:
            print("   🔄 DSSMSバックテスター実行中...")
            
            # PowerShell形式で実行
            result = subprocess.run([
                'python', 'src/dssms/dssms_backtester.py'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)
            
            self.execution_log = result.stdout + result.stderr
            
            if result.returncode == 0:
                print("   ✅ DSSMS実行成功")
                self.analysis_results['execution_status'] = 'success'
                
                # ログから重要情報を抽出
                self._extract_log_metrics()
                
            else:
                print(f"   ❌ DSSMS実行エラー")
                self.analysis_results['execution_status'] = 'failed'
                self.issues.append({
                    'category': 'execution',
                    'severity': 'critical',
                    'description': f'DSSMSバックテスター実行失敗',
                    'type': 'execution_failure',
                    'details': {'stderr': result.stderr[:500]}
                })
                
        except subprocess.TimeoutExpired:
            print("   ⏰ DSSMS実行タイムアウト")
            self.issues.append({
                'category': 'execution',
                'severity': 'high',
                'description': 'DSSMSバックテスター実行タイムアウト',
                'type': 'execution_timeout'
            })
        except Exception as e:
            print(f"   ❌ 実行エラー: {e}")
            self.issues.append({
                'category': 'execution',
                'severity': 'critical',
                'description': f'実行システムエラー: {e}',
                'type': 'system_error'
            })
    
    def _extract_log_metrics(self):
        """ログから重要指標を抽出"""
        if not self.execution_log:
            return
        
        log_metrics = {}
        
        # データ取得情報の抽出
        data_pattern = r'DSSMSデータ取得:\s*portfolio_values=(\d+),\s*timestamps=(\d+)'
        data_match = re.search(data_pattern, self.execution_log)
        if data_match:
            log_metrics['portfolio_values_count'] = int(data_match.group(1))
            log_metrics['timestamps_count'] = int(data_match.group(2))
        
        # 実際のデータ使用情報
        usage_pattern = r'実際のDSSMSデータ使用:\s*(\d+)日分'
        usage_match = re.search(usage_pattern, self.execution_log)
        if usage_match:
            log_metrics['days_used'] = int(usage_match.group(1))
        
        # 取引履歴作成情報
        trades_pattern = r'取引履歴作成:\s*(\d+)件'
        trades_match = re.search(trades_pattern, self.execution_log)
        if trades_match:
            log_metrics['trades_created'] = int(trades_match.group(1))
        
        # 切替履歴情報
        switches_pattern = r'切替履歴作成:\s*(\d+)件'
        switches_match = re.search(switches_pattern, self.execution_log)
        if switches_match:
            log_metrics['switches_created'] = int(switches_match.group(1))
        
        # パフォーマンス計算情報
        performance_pattern = r'パフォーマンス計算:\s*総リターン([+-]?\d+\.?\d*)%'
        performance_match = re.search(performance_pattern, self.execution_log)
        if performance_match:
            log_metrics['calculated_return'] = float(performance_match.group(1))
        
        # 最終ポートフォリオ価値
        portfolio_pattern = r'最終ポートフォリオ価値\s*([0-9,]+)円'
        portfolio_match = re.search(portfolio_pattern, self.execution_log)
        if portfolio_match:
            portfolio_value = portfolio_match.group(1).replace(',', '')
            log_metrics['final_portfolio_value'] = int(portfolio_value)
        
        # 総リターン（最終レポート）
        final_return_pattern = r'総リターン:\s*([+-]?\d+\.?\d*)%'
        final_return_match = re.search(final_return_pattern, self.execution_log)
        if final_return_match:
            log_metrics['final_reported_return'] = float(final_return_match.group(1))
        
        self.analysis_results['log_metrics'] = log_metrics
        
        print(f"   📊 ログ解析完了:")
        for key, value in log_metrics.items():
            print(f"     {key}: {value}")
    
    def _identify_latest_output_files(self) -> Dict[str, str]:
        """最新の出力ファイルを特定"""
        latest_files = {}
        
        # 出力パターンの定義
        patterns = {
            'excel': [
                'backtest_results/dssms_results/dssms_unified_backtest_*.xlsx',
                'src/dssms/backtest_results/dssms_results/dssms_unified_backtest_*.xlsx',
                'dssms_unified_backtest_*.xlsx'
            ],
            'text': [
                'backtest_results/dssms_results/dssms_unified_report_*.txt',
                'src/dssms/backtest_results/dssms_results/dssms_unified_report_*.txt',
                'dssms_unified_report_*.txt'
            ],
            'json': [
                'backtest_results/dssms_results/dssms_unified_data_*.json',
                'src/dssms/backtest_results/dssms_results/dssms_unified_data_*.json',
                'dssms_unified_data_*.json'
            ]
        }
        
        for file_type, pattern_list in patterns.items():
            latest_file = None
            latest_time = 0
            
            for pattern in pattern_list:
                files = glob.glob(pattern)
                for file_path in files:
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
            
            if latest_file:
                latest_files[file_type] = latest_file
                print(f"   📁 {file_type}: {os.path.basename(latest_file)}")
            else:
                print(f"   ❌ {file_type}: ファイル未発見")
                self.issues.append({
                    'category': 'file_availability',
                    'severity': 'high',
                    'description': f'{file_type}ファイルが見つかりません',
                    'type': 'missing_output_file'
                })
        
        self.analysis_results['latest_files'] = latest_files
        return latest_files
    
    def _analyze_log_output_inconsistency(self, latest_files: Dict[str, str]):
        """ログと出力ファイルの不整合分析"""
        if 'log_metrics' not in self.analysis_results:
            print("   ❌ ログ情報が取得できていません")
            return
        
        log_metrics = self.analysis_results['log_metrics']
        
        # Excelファイルの分析
        if 'excel' in latest_files:
            self._analyze_excel_vs_log_consistency(latest_files['excel'], log_metrics)
        
        # テキストファイルの分析
        if 'text' in latest_files:
            self._analyze_text_vs_log_consistency(latest_files['text'], log_metrics)
        
        # JSONファイルの分析
        if 'json' in latest_files:
            self._analyze_json_vs_log_consistency(latest_files['json'], log_metrics)
    
    def _analyze_excel_vs_log_consistency(self, excel_path: str, log_metrics: Dict):
        """Excel⇔ログ整合性分析"""
        print(f"   📊 Excel分析: {os.path.basename(excel_path)}")
        
        try:
            excel_file = pd.ExcelFile(excel_path)
            
            # 損益推移シートの分析
            if '損益推移' in excel_file.sheet_names:
                pnl_df = pd.read_excel(excel_file, sheet_name='損益推移')
                
                # データ行数の比較
                actual_rows = len(pnl_df)
                expected_days = log_metrics.get('days_used', 0)
                
                if abs(actual_rows - expected_days) > 10:
                    self.issues.append({
                        'category': 'data_consistency',
                        'severity': 'critical',
                        'description': f'ログ表示({expected_days}日分)とExcel実データ({actual_rows}行)の大幅な不整合',
                        'type': 'log_excel_row_mismatch',
                        'details': {
                            'log_says_days': expected_days,
                            'excel_actual_rows': actual_rows,
                            'difference': abs(actual_rows - expected_days),
                            'percentage_diff': abs(actual_rows - expected_days) / max(expected_days, 1) * 100
                        }
                    })
                    print(f"     ❌ データ行数不整合: ログ{expected_days}日 vs Excel{actual_rows}行")
                else:
                    print(f"     ✅ データ行数整合性OK: {actual_rows}行")
                
                # 日付範囲の分析
                if '日付' in pnl_df.columns:
                    dates = pd.to_datetime(pnl_df['日付'], errors='coerce').dropna()
                    if len(dates) > 0:
                        date_range_days = (dates.max() - dates.min()).days + 1
                        
                        if abs(date_range_days - expected_days) > 30:  # 1ヶ月以上の差異
                            self.issues.append({
                                'category': 'temporal_consistency',
                                'severity': 'high',
                                'description': f'ログ期待期間({expected_days}日)とExcel日付範囲({date_range_days}日)の不整合',
                                'type': 'date_range_mismatch',
                                'details': {
                                    'expected_days': expected_days,
                                    'actual_date_range': date_range_days,
                                    'start_date': dates.min().strftime('%Y-%m-%d'),
                                    'end_date': dates.max().strftime('%Y-%m-%d')
                                }
                            })
                            print(f"     ❌ 日付範囲不整合: 期待{expected_days}日 vs 実際{date_range_days}日")
            
            # 取引履歴シートの分析
            if '取引履歴' in excel_file.sheet_names:
                trade_df = pd.read_excel(excel_file, sheet_name='取引履歴')
                
                actual_trades = len(trade_df)
                expected_trades = log_metrics.get('trades_created', 0)
                
                if abs(actual_trades - expected_trades) > 5:
                    self.issues.append({
                        'category': 'data_consistency',
                        'severity': 'high',
                        'description': f'ログ取引数({expected_trades}件)とExcel取引数({actual_trades}件)の不整合',
                        'type': 'trade_count_mismatch',
                        'details': {
                            'log_trades': expected_trades,
                            'excel_trades': actual_trades,
                            'difference': abs(actual_trades - expected_trades)
                        }
                    })
                    print(f"     ❌ 取引数不整合: ログ{expected_trades}件 vs Excel{actual_trades}件")
                else:
                    print(f"     ✅ 取引数整合性OK: {actual_trades}件")
            
            # パフォーマンス指標の比較
            if 'サマリー' in excel_file.sheet_names:
                summary_df = pd.read_excel(excel_file, sheet_name='サマリー')
                self._compare_performance_metrics(summary_df, log_metrics)
                
        except Exception as e:
            print(f"     ❌ Excel分析エラー: {e}")
            self.issues.append({
                'category': 'file_analysis',
                'severity': 'high',
                'description': f'Excel分析エラー: {e}',
                'type': 'excel_analysis_error'
            })
    
    def _compare_performance_metrics(self, summary_df: pd.DataFrame, log_metrics: Dict):
        """パフォーマンス指標の比較"""
        print("     📈 パフォーマンス指標比較")
        
        if '項目' in summary_df.columns and '値' in summary_df.columns:
            # 総リターンの比較
            return_row = summary_df[summary_df['項目'].str.contains('総リターン', na=False)]
            if not return_row.empty:
                excel_return_str = return_row['値'].iloc[0]
                # パーセント値を抽出
                excel_return_match = re.search(r'([+-]?\d+\.?\d*)%', str(excel_return_str))
                if excel_return_match:
                    excel_return = float(excel_return_match.group(1))
                    
                    # ログの複数の総リターン値と比較
                    log_returns = []
                    if 'calculated_return' in log_metrics:
                        log_returns.append(log_metrics['calculated_return'])
                    if 'final_reported_return' in log_metrics:
                        log_returns.append(log_metrics['final_reported_return'])
                    
                    # 大幅な不整合をチェック
                    for log_return in log_returns:
                        if abs(excel_return - log_return) > 1.0:  # 1%以上の差異
                            self.issues.append({
                                'category': 'performance_consistency',
                                'severity': 'high',
                                'description': f'総リターン不整合: Excel({excel_return}%)とログ({log_return}%)の差異',
                                'type': 'return_calculation_mismatch',
                                'details': {
                                    'excel_return': excel_return,
                                    'log_return': log_return,
                                    'difference': abs(excel_return - log_return)
                                }
                            })
                            print(f"       ❌ 総リターン不整合: Excel{excel_return}% vs ログ{log_return}%")
                            break
                    else:
                        print(f"       ✅ 総リターン整合性OK: {excel_return}%")
            
            # 最終ポートフォリオ価値の比較
            portfolio_row = summary_df[summary_df['項目'].str.contains('最終ポートフォリオ価値', na=False)]
            if not portfolio_row.empty:
                excel_portfolio_str = portfolio_row['値'].iloc[0]
                # 数値を抽出
                excel_portfolio_match = re.search(r'([0-9,]+)', str(excel_portfolio_str))
                if excel_portfolio_match:
                    excel_portfolio = int(excel_portfolio_match.group(1).replace(',', ''))
                    log_portfolio = log_metrics.get('final_portfolio_value', 0)
                    
                    if abs(excel_portfolio - log_portfolio) > 1000:  # 1000円以上の差異
                        self.issues.append({
                            'category': 'performance_consistency',
                            'severity': 'high',
                            'description': f'最終ポートフォリオ価値不整合: Excel({excel_portfolio:,}円)とログ({log_portfolio:,}円)の差異',
                            'type': 'portfolio_value_mismatch',
                            'details': {
                                'excel_value': excel_portfolio,
                                'log_value': log_portfolio,
                                'difference': abs(excel_portfolio - log_portfolio)
                            }
                        })
                        print(f"       ❌ ポートフォリオ価値不整合: Excel{excel_portfolio:,}円 vs ログ{log_portfolio:,}円")
                    else:
                        print(f"       ✅ ポートフォリオ価値整合性OK: {excel_portfolio:,}円")
    
    def _analyze_text_vs_log_consistency(self, text_path: str, log_metrics: Dict):
        """テキスト⇔ログ整合性分析"""
        print(f"   📄 テキスト分析: {os.path.basename(text_path)}")
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # 総リターンの抽出と比較
            text_return_match = re.search(r'総リターン[:\s]*([+-]?\d+\.?\d*)%', text_content)
            if text_return_match:
                text_return = float(text_return_match.group(1))
                log_return = log_metrics.get('final_reported_return', 0)
                
                if abs(text_return - log_return) > 0.5:  # 0.5%以上の差異
                    self.issues.append({
                        'category': 'text_consistency',
                        'severity': 'medium',
                        'description': f'テキストレポートとログの総リターン不整合: テキスト({text_return}%)とログ({log_return}%)',
                        'type': 'text_log_return_mismatch',
                        'details': {
                            'text_return': text_return,
                            'log_return': log_return,
                            'difference': abs(text_return - log_return)
                        }
                    })
                    print(f"     ❌ 総リターン不整合: テキスト{text_return}% vs ログ{log_return}%")
                else:
                    print(f"     ✅ 総リターン整合性OK: {text_return}%")
            
            # 銘柄切替回数の比較
            switches_match = re.search(r'銘柄切替回数[:\s]*(\d+)', text_content)
            if switches_match:
                text_switches = int(switches_match.group(1))
                log_switches = log_metrics.get('switches_created', 0)
                
                if abs(text_switches - log_switches) > 2:
                    self.issues.append({
                        'category': 'text_consistency',
                        'severity': 'medium',
                        'description': f'テキストレポートとログの切替回数不整合: テキスト({text_switches}回)とログ({log_switches}回)',
                        'type': 'text_log_switches_mismatch',
                        'details': {
                            'text_switches': text_switches,
                            'log_switches': log_switches,
                            'difference': abs(text_switches - log_switches)
                        }
                    })
                    print(f"     ❌ 切替回数不整合: テキスト{text_switches}回 vs ログ{log_switches}回")
                else:
                    print(f"     ✅ 切替回数整合性OK: {text_switches}回")
                    
        except Exception as e:
            print(f"     ❌ テキスト分析エラー: {e}")
            self.issues.append({
                'category': 'file_analysis',
                'severity': 'medium',
                'description': f'テキスト分析エラー: {e}',
                'type': 'text_analysis_error'
            })
    
    def _analyze_json_vs_log_consistency(self, json_path: str, log_metrics: Dict):
        """JSON⇔ログ整合性分析"""
        print(f"   📦 JSON分析: {os.path.basename(json_path)}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # ポートフォリオデータの整合性チェック
            if 'portfolio_values' in json_data:
                portfolio_data = json_data['portfolio_values']
                actual_count = len(portfolio_data)
                expected_count = log_metrics.get('portfolio_values_count', 0)
                
                if abs(actual_count - expected_count) > 5:
                    self.issues.append({
                        'category': 'json_consistency',
                        'severity': 'high',
                        'description': f'JSONとログのポートフォリオデータ数不整合: JSON({actual_count}件)とログ({expected_count}件)',
                        'type': 'json_log_portfolio_count_mismatch',
                        'details': {
                            'json_count': actual_count,
                            'log_count': expected_count,
                            'difference': abs(actual_count - expected_count)
                        }
                    })
                    print(f"     ❌ ポートフォリオデータ数不整合: JSON{actual_count}件 vs ログ{expected_count}件")
                else:
                    print(f"     ✅ ポートフォリオデータ数整合性OK: {actual_count}件")
            
            # 取引データの整合性チェック
            if 'trades' in json_data:
                trades_data = json_data['trades']
                actual_trades = len(trades_data)
                expected_trades = log_metrics.get('trades_created', 0)
                
                if abs(actual_trades - expected_trades) > 5:
                    self.issues.append({
                        'category': 'json_consistency',
                        'severity': 'high',
                        'description': f'JSONとログの取引データ数不整合: JSON({actual_trades}件)とログ({expected_trades}件)',
                        'type': 'json_log_trades_count_mismatch',
                        'details': {
                            'json_count': actual_trades,
                            'log_count': expected_trades,
                            'difference': abs(actual_trades - expected_trades)
                        }
                    })
                    print(f"     ❌ 取引データ数不整合: JSON{actual_trades}件 vs ログ{expected_trades}件")
                else:
                    print(f"     ✅ 取引データ数整合性OK: {actual_trades}件")
                    
        except Exception as e:
            print(f"     ❌ JSON分析エラー: {e}")
            self.issues.append({
                'category': 'file_analysis',
                'severity': 'medium',
                'description': f'JSON分析エラー: {e}',
                'type': 'json_analysis_error'
            })
    
    def _analyze_temporal_inconsistencies(self, latest_files: Dict[str, str]):
        """時系列データの不整合分析"""
        print("   📅 時系列データ整合性分析")
        
        if 'excel' in latest_files:
            try:
                excel_file = pd.ExcelFile(latest_files['excel'])
                
                if '損益推移' in excel_file.sheet_names:
                    pnl_df = pd.read_excel(excel_file, sheet_name='損益推移')
                    
                    if '日付' in pnl_df.columns:
                        dates = pd.to_datetime(pnl_df['日付'], errors='coerce').dropna()
                        
                        if len(dates) > 0:
                            # 日付範囲の分析
                            start_date = dates.min()
                            end_date = dates.max()
                            date_range_days = (end_date - start_date).days + 1
                            
                            # 期待される期間（2023年のバックテスト期間）
                            expected_start = pd.to_datetime('2023-01-01')
                            expected_end = pd.to_datetime('2023-12-31')
                            expected_range_days = (expected_end - expected_start).days + 1
                            
                            print(f"     📊 期間分析: {start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')} ({date_range_days}日)")
                            
                            # 年の整合性チェック
                            years = dates.dt.year.unique()
                            if 2025 in years:
                                self.issues.append({
                                    'category': 'temporal_accuracy',
                                    'severity': 'critical',
                                    'description': f'未来の日付(2025年)がデータに含まれています',
                                    'type': 'future_dates_detected',
                                    'details': {
                                        'years_found': years.tolist(),
                                        'date_range': f"{start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}"
                                    }
                                })
                                print(f"     ❌ 未来日付検出: {years.tolist()}")
                            
                            # 期間の妥当性チェック
                            if date_range_days > 400:  # 400日を超える場合
                                self.issues.append({
                                    'category': 'temporal_accuracy',
                                    'severity': 'high',
                                    'description': f'データ期間が異常に長い({date_range_days}日)です',
                                    'type': 'excessive_date_range',
                                    'details': {
                                        'actual_days': date_range_days,
                                        'expected_max': 365,
                                        'start_date': start_date.strftime('%Y-%m-%d'),
                                        'end_date': end_date.strftime('%Y-%m-%d')
                                    }
                                })
                                print(f"     ❌ 期間異常: {date_range_days}日（365日想定）")
                            
                            # ログとの期間比較
                            log_days = self.analysis_results.get('log_metrics', {}).get('days_used', 0)
                            if log_days > 0 and abs(date_range_days - log_days) > 30:
                                self.issues.append({
                                    'category': 'temporal_consistency',
                                    'severity': 'high',
                                    'description': f'ログ期間({log_days}日)とExcel期間({date_range_days}日)の大幅不整合',
                                    'type': 'log_excel_period_mismatch',
                                    'details': {
                                        'log_days': log_days,
                                        'excel_days': date_range_days,
                                        'difference': abs(date_range_days - log_days)
                                    }
                                })
                                print(f"     ❌ 期間不整合: ログ{log_days}日 vs Excel{date_range_days}日")
                            
            except Exception as e:
                print(f"     ❌ 時系列分析エラー: {e}")
                self.issues.append({
                    'category': 'temporal_analysis',
                    'severity': 'medium',
                    'description': f'時系列データ分析エラー: {e}',
                    'type': 'temporal_analysis_error'
                })
    
    def _verify_performance_calculations(self, latest_files: Dict[str, str]):
        """パフォーマンス計算の検証"""
        print("   💰 パフォーマンス計算検証")
        
        if 'excel' in latest_files:
            try:
                excel_file = pd.ExcelFile(latest_files['excel'])
                
                # 損益推移からリターン計算
                if '損益推移' in excel_file.sheet_names:
                    pnl_df = pd.read_excel(excel_file, sheet_name='損益推移')
                    
                    if 'ポートフォリオ価値' in pnl_df.columns and len(pnl_df) > 1:
                        initial_value = pnl_df['ポートフォリオ価値'].iloc[0]
                        final_value = pnl_df['ポートフォリオ価値'].iloc[-1]
                        
                        # 手動でリターン計算
                        calculated_return = ((final_value - initial_value) / initial_value) * 100
                        
                        print(f"     📊 手動計算: 初期{initial_value:,}円 → 最終{final_value:,}円 → リターン{calculated_return:.2f}%")
                        
                        # サマリーのリターンと比較
                        if 'サマリー' in excel_file.sheet_names:
                            summary_df = pd.read_excel(excel_file, sheet_name='サマリー')
                            
                            if '項目' in summary_df.columns and '値' in summary_df.columns:
                                return_row = summary_df[summary_df['項目'].str.contains('総リターン', na=False)]
                                if not return_row.empty:
                                    summary_return_str = return_row['値'].iloc[0]
                                    summary_return_match = re.search(r'([+-]?\d+\.?\d*)%', str(summary_return_str))
                                    if summary_return_match:
                                        summary_return = float(summary_return_match.group(1))
                                        
                                        if abs(calculated_return - summary_return) > 1.0:  # 1%以上の差異
                                            self.issues.append({
                                                'category': 'calculation_accuracy',
                                                'severity': 'high',
                                                'description': f'手動計算({calculated_return:.2f}%)とサマリー({summary_return}%)のリターン不整合',
                                                'type': 'return_calculation_error',
                                                'details': {
                                                    'manual_calculation': calculated_return,
                                                    'summary_value': summary_return,
                                                    'initial_portfolio': initial_value,
                                                    'final_portfolio': final_value,
                                                    'difference': abs(calculated_return - summary_return)
                                                }
                                            })
                                            print(f"     ❌ リターン計算不整合: 手動{calculated_return:.2f}% vs サマリー{summary_return}%")
                                        else:
                                            print(f"     ✅ リターン計算整合性OK: {calculated_return:.2f}%")
                        
                        # ポートフォリオ価値の変動チェック
                        if pnl_df['ポートフォリオ価値'].nunique() == 1:
                            self.issues.append({
                                'category': 'calculation_accuracy',
                                'severity': 'critical',
                                'description': 'ポートフォリオ価値が全期間で同一値（計算されていない可能性）',
                                'type': 'static_portfolio_values',
                                'details': {
                                    'constant_value': initial_value,
                                    'data_points': len(pnl_df)
                                }
                            })
                            print(f"     ❌ ポートフォリオ価値が固定値: {initial_value:,}円")
                        
            except Exception as e:
                print(f"     ❌ パフォーマンス計算検証エラー: {e}")
                self.issues.append({
                    'category': 'calculation_verification',
                    'severity': 'medium',
                    'description': f'パフォーマンス計算検証エラー: {e}',
                    'type': 'calculation_verification_error'
                })
    
    def _check_date_data_consistency(self, latest_files: Dict[str, str]):
        """日付データの整合性チェック"""
        print("   📅 日付データ詳細チェック")
        
        if 'excel' in latest_files:
            try:
                excel_file = pd.ExcelFile(latest_files['excel'])
                date_issues = []
                
                # 各シートの日付データをチェック
                for sheet_name in excel_file.sheet_names:
                    if sheet_name in ['損益推移', '取引履歴']:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        
                        if '日付' in df.columns:
                            dates = pd.to_datetime(df['日付'], errors='coerce')
                            invalid_dates = dates.isna().sum()
                            
                            if invalid_dates > 0:
                                date_issues.append({
                                    'sheet': sheet_name,
                                    'invalid_count': invalid_dates,
                                    'total_rows': len(df)
                                })
                            
                            # 有効な日付の分析
                            valid_dates = dates.dropna()
                            if len(valid_dates) > 0:
                                # 重複日付チェック
                                duplicate_dates = valid_dates.duplicated().sum()
                                if duplicate_dates > 0:
                                    self.issues.append({
                                        'category': 'date_quality',
                                        'severity': 'medium',
                                        'description': f'{sheet_name}シートに重複日付が{duplicate_dates}件あります',
                                        'type': 'duplicate_dates',
                                        'details': {
                                            'sheet': sheet_name,
                                            'duplicate_count': duplicate_dates
                                        }
                                    })
                                    print(f"     ⚠️ {sheet_name}: 重複日付{duplicate_dates}件")
                                
                                # 日付順序チェック
                                if not valid_dates.is_monotonic_increasing:
                                    self.issues.append({
                                        'category': 'date_quality',
                                        'severity': 'high',
                                        'description': f'{sheet_name}シートの日付が昇順でありません',
                                        'type': 'unsorted_dates',
                                        'details': {
                                            'sheet': sheet_name
                                        }
                                    })
                                    print(f"     ❌ {sheet_name}: 日付が非昇順")
                
                if date_issues:
                    total_invalid = sum(issue['invalid_count'] for issue in date_issues)
                    self.issues.append({
                        'category': 'date_quality',
                        'severity': 'high',
                        'description': f'無効な日付データが{total_invalid}件検出されました',
                        'type': 'invalid_dates',
                        'details': {
                            'sheet_issues': date_issues,
                            'total_invalid': total_invalid
                        }
                    })
                    print(f"     ❌ 無効日付: {total_invalid}件")
                else:
                    print(f"     ✅ 日付データ品質OK")
                    
            except Exception as e:
                print(f"     ❌ 日付データチェックエラー: {e}")
                self.issues.append({
                    'category': 'date_analysis',
                    'severity': 'medium',
                    'description': f'日付データ分析エラー: {e}',
                    'type': 'date_analysis_error'
                })
    
    def _categorize_and_generate_fixes(self):
        """問題の分類と修正提案生成"""
        print("   🔧 問題分類・修正提案生成")
        
        # 重要度順でソート
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        self.issues.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
        
        # カテゴリ別集計
        categories = {}
        for issue in self.issues:
            category = issue['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(issue)
        
        # 修正提案の生成
        fix_proposals = []
        
        # 重要度の高い問題に対する修正提案
        critical_issues = [i for i in self.issues if i['severity'] == 'critical']
        if critical_issues:
            fix_proposals.append({
                'priority': 1,
                'title': '重要問題の即座修正',
                'description': f'{len(critical_issues)}件の重要問題が検出されました',
                'issues': critical_issues,
                'recommended_action': '統一出力エンジンの全面的見直しと修正'
            })
        
        # データ整合性問題
        consistency_issues = [i for i in self.issues if i['category'] in ['data_consistency', 'temporal_consistency']]
        if consistency_issues:
            fix_proposals.append({
                'priority': 2,
                'title': 'データ整合性問題の修正',
                'description': f'{len(consistency_issues)}件の整合性問題が検出されました',
                'issues': consistency_issues,
                'recommended_action': 'ログとファイル出力の同期処理改善'
            })
        
        # パフォーマンス計算問題
        performance_issues = [i for i in self.issues if i['category'] in ['performance_consistency', 'calculation_accuracy']]
        if performance_issues:
            fix_proposals.append({
                'priority': 3,
                'title': 'パフォーマンス計算の修正',
                'description': f'{len(performance_issues)}件の計算問題が検出されました',
                'issues': performance_issues,
                'recommended_action': 'パフォーマンス計算ロジックの見直しと修正'
            })
        
        # 日付問題
        date_issues = [i for i in self.issues if i['category'] in ['temporal_accuracy', 'date_quality']]
        if date_issues:
            fix_proposals.append({
                'priority': 4,
                'title': '日付データの修正',
                'description': f'{len(date_issues)}件の日付問題が検出されました',
                'issues': date_issues,
                'recommended_action': '日付変換・検証システムの改善'
            })
        
        self.analysis_results['categories'] = categories
        self.analysis_results['fix_proposals'] = fix_proposals
        self.analysis_results['issue_summary'] = {
            'total_issues': len(self.issues),
            'critical_count': len([i for i in self.issues if i['severity'] == 'critical']),
            'high_count': len([i for i in self.issues if i['severity'] == 'high']),
            'medium_count': len([i for i in self.issues if i['severity'] == 'medium']),
            'categories_count': len(categories)
        }
        
        print(f"     📊 問題サマリー:")
        print(f"       総問題数: {len(self.issues)}")
        print(f"       重要: {self.analysis_results['issue_summary']['critical_count']}件")
        print(f"       高優先度: {self.analysis_results['issue_summary']['high_count']}件")
        print(f"       修正提案: {len(fix_proposals)}項目")
    
    def _generate_comprehensive_report(self) -> str:
        """包括的レポート生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"dssms_comprehensive_analysis_report_{timestamp}.json"
        
        # 詳細レポート作成
        comprehensive_report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'project_root': str(self.project_root.absolute()),
                'execution_status': self.analysis_results.get('execution_status', 'unknown')
            },
            'executive_summary': {
                'total_issues_found': len(self.issues),
                'critical_issues': self.analysis_results.get('issue_summary', {}).get('critical_count', 0),
                'high_priority_issues': self.analysis_results.get('issue_summary', {}).get('high_count', 0),
                'categories_affected': self.analysis_results.get('issue_summary', {}).get('categories_count', 0),
                'files_analyzed': list(self.analysis_results.get('latest_files', {}).keys()),
                'primary_concerns': [
                    issue['description'] for issue in self.issues 
                    if issue['severity'] == 'critical'
                ][:5]  # Top 5 critical issues
            },
            'log_analysis': {
                'execution_log_captured': bool(self.execution_log),
                'log_metrics_extracted': self.analysis_results.get('log_metrics', {}),
                'log_length_chars': len(self.execution_log)
            },
            'file_analysis': {
                'latest_files_found': self.analysis_results.get('latest_files', {}),
                'file_consistency_issues': [
                    issue for issue in self.issues 
                    if issue['category'] in ['data_consistency', 'file_analysis']
                ]
            },
            'detailed_issues': self.issues,
            'fix_proposals': self.analysis_results.get('fix_proposals', []),
            'recommendations': {
                'immediate_actions': [
                    "重要問題（critical）の優先修正",
                    "統一出力エンジンの整合性確保",
                    "ログとファイル出力の同期改善"
                ],
                'medium_term_actions': [
                    "日付処理システムの改善",
                    "パフォーマンス計算ロジック見直し",
                    "データ品質検証の強化"
                ],
                'long_term_actions': [
                    "自動検証システムの構築",
                    "継続的品質監視の実装",
                    "エラー予防機能の追加"
                ]
            },
            'technical_details': {
                'analysis_categories': list(self.analysis_results.get('categories', {}).keys()),
                'issue_types_detected': list(set(issue['type'] for issue in self.issues)),
                'severity_distribution': {
                    severity: len([i for i in self.issues if i['severity'] == severity])
                    for severity in ['critical', 'high', 'medium', 'low']
                }
            }
        }
        
        # レポートファイル生成
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"     📄 包括的レポート生成完了: {report_path}")
        return report_path

def main():
    """メイン実行関数"""
    print("🚀 DSSMS包括的問題分析システム開始")
    print("="*70)
    print("このシステムは以下を検出します:")
    print("• ログ表示と実際のファイル出力の不整合")
    print("• データ期間の不整合（ログ482日分 vs Excel一年分など）")
    print("• パフォーマンス計算の不整合")
    print("• 日付データの問題")
    print("• ファイル間の整合性問題")
    print("="*70)
    
    analyzer = DSSMSComprehensiveAnalyzer()
    
    try:
        report_path = analyzer.run_comprehensive_analysis()
        
        # 結果サマリー表示
        print("\n" + "="*70)
        print("📋 分析結果サマリー")
        print("="*70)
        
        summary = analyzer.analysis_results.get('issue_summary', {})
        print(f"✅ 分析完了")
        print(f"📊 総問題数: {summary.get('total_issues', 0)}")
        print(f"🚨 重要問題: {summary.get('critical_count', 0)}件")
        print(f"⚠️  高優先度: {summary.get('high_count', 0)}件")
        print(f"📝 中優先度: {summary.get('medium_count', 0)}件")
        print(f"🔧 修正提案: {len(analyzer.analysis_results.get('fix_proposals', []))}項目")
        
        # 重要問題のハイライト
        critical_issues = [i for i in analyzer.issues if i['severity'] == 'critical']
        if critical_issues:
            print(f"\n🚨 重要問題（即座修正推奨）:")
            for i, issue in enumerate(critical_issues[:3], 1):  # Top 3
                print(f"   {i}. {issue['description']}")
            if len(critical_issues) > 3:
                print(f"   ... 他{len(critical_issues)-3}件")
        
        print(f"\n📄 詳細レポート: {report_path}")
        print(f"💡 推奨: まず重要問題から修正を開始してください")
        
        return report_path
        
    except Exception as e:
        print(f"\n❌ 分析実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
