#!/usr/bin/env python3
"""
DSSMS 取引履歴問題特定スクリプト

問題:
1. 戦略名がDSSMSストラテジーのみで、7つの戦略の区別ができない
2. エントリー価格、エグジット価格が固定値
3. 損益の計算が正しいか不明
4. 保有期間がすべて24時間（テキストでは平均74.9時間）

目的:
- Excelファイルの取引履歴シートを詳細分析
- 統一出力システムのデータ生成ロジックを確認
- 問題箇所を特定し修正方針を提示
"""

import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl
from typing import Dict, List, Any

def setup_logger():
    """ロガー設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class DSSMSTradeHistoryAnalyzer:
    """DSSMS取引履歴分析器"""
    
    def __init__(self):
        self.logger = setup_logger()
        self.issues = []
        self.analysis_results = {}
        
    def analyze_excel_trade_history(self, excel_path: str) -> Dict[str, Any]:
        """Excelファイルの取引履歴シートを分析"""
        self.logger.info(f"📊 Excel取引履歴分析開始: {excel_path}")
        
        try:
            # Excelファイルを読み込み
            workbook = openpyxl.load_workbook(excel_path)
            
            if '取引履歴' not in workbook.sheetnames:
                self.logger.error("取引履歴シートが見つかりません")
                return {}
                
            # 取引履歴シートを分析
            sheet = workbook['取引履歴']
            
            # データを抽出
            trade_data = []
            headers = []
            
            # ヘッダー行を取得
            for cell in sheet[1]:
                headers.append(cell.value)
            
            # データ行を取得
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if any(row):  # 空行をスキップ
                    trade_data.append(row)
            
            self.logger.info(f"取引データ件数: {len(trade_data)}件")
            self.logger.info(f"列名: {headers}")
            
            # 問題分析
            analysis = {
                'total_trades': len(trade_data),
                'headers': headers,
                'issues_found': [],
                'sample_data': trade_data[:5] if trade_data else [],
                'strategy_analysis': self._analyze_strategies(trade_data, headers),
                'price_analysis': self._analyze_prices(trade_data, headers),
                'holding_period_analysis': self._analyze_holding_periods(trade_data, headers),
                'profit_loss_analysis': self._analyze_profit_loss(trade_data, headers)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Excel分析エラー: {e}")
            return {}
    
    def _analyze_strategies(self, trade_data: List, headers: List) -> Dict[str, Any]:
        """戦略名分析"""
        strategy_col = None
        for i, header in enumerate(headers):
            if '戦略' in str(header) or 'Strategy' in str(header):
                strategy_col = i
                break
        
        if strategy_col is None:
            return {'error': '戦略列が見つかりません'}
        
        strategies = []
        for row in trade_data:
            if len(row) > strategy_col and row[strategy_col]:
                strategies.append(str(row[strategy_col]))
        
        unique_strategies = list(set(strategies))
        
        analysis = {
            'strategy_column_index': strategy_col,
            'unique_strategies': unique_strategies,
            'strategy_count': len(unique_strategies),
            'total_entries': len(strategies)
        }
        
        # 問題チェック
        if len(unique_strategies) == 1 and 'DSSMS' in unique_strategies[0]:
            self.issues.append({
                'category': 'strategy_naming',
                'severity': 'high',
                'description': '戦略名がDSSMSストラテジーのみで、具体的な戦略が特定できない',
                'details': analysis
            })
        
        return analysis
    
    def _analyze_prices(self, trade_data: List, headers: List) -> Dict[str, Any]:
        """価格分析"""
        entry_col = exit_col = None
        
        for i, header in enumerate(headers):
            if 'エントリー' in str(header) or 'Entry' in str(header):
                entry_col = i
            elif 'エグジット' in str(header) or 'Exit' in str(header):
                exit_col = i
        
        analysis = {
            'entry_column_index': entry_col,
            'exit_column_index': exit_col
        }
        
        if entry_col is not None:
            entry_prices = []
            for row in trade_data:
                if len(row) > entry_col and row[entry_col] is not None:
                    try:
                        price = float(row[entry_col])
                        entry_prices.append(price)
                    except (ValueError, TypeError):
                        pass
            
            # 固定値チェック
            unique_entry_prices = list(set(entry_prices))
            analysis['entry_prices_unique'] = len(unique_entry_prices)
            analysis['entry_prices_sample'] = entry_prices[:10]
            
            if len(unique_entry_prices) <= 2 and len(entry_prices) > 10:
                self.issues.append({
                    'category': 'fixed_prices',
                    'severity': 'high',
                    'description': 'エントリー価格が固定値になっている可能性',
                    'details': {'unique_prices': unique_entry_prices, 'total_trades': len(entry_prices)}
                })
        
        if exit_col is not None:
            exit_prices = []
            for row in trade_data:
                if len(row) > exit_col and row[exit_col] is not None:
                    try:
                        price = float(row[exit_col])
                        exit_prices.append(price)
                    except (ValueError, TypeError):
                        pass
            
            unique_exit_prices = list(set(exit_prices))
            analysis['exit_prices_unique'] = len(unique_exit_prices)
            analysis['exit_prices_sample'] = exit_prices[:10]
            
            if len(unique_exit_prices) <= 2 and len(exit_prices) > 10:
                self.issues.append({
                    'category': 'fixed_prices',
                    'severity': 'high',
                    'description': 'エグジット価格が固定値になっている可能性',
                    'details': {'unique_prices': unique_exit_prices, 'total_trades': len(exit_prices)}
                })
        
        return analysis
    
    def _analyze_holding_periods(self, trade_data: List, headers: List) -> Dict[str, Any]:
        """保有期間分析"""
        holding_col = None
        
        for i, header in enumerate(headers):
            if '保有期間' in str(header) or 'Holding' in str(header):
                holding_col = i
                break
        
        if holding_col is None:
            return {'error': '保有期間列が見つかりません'}
        
        holding_periods = []
        for row in trade_data:
            if len(row) > holding_col and row[holding_col] is not None:
                holding_periods.append(str(row[holding_col]))
        
        unique_periods = list(set(holding_periods))
        
        analysis = {
            'holding_column_index': holding_col,
            'unique_periods': unique_periods,
            'unique_count': len(unique_periods),
            'sample_periods': holding_periods[:10]
        }
        
        # 24時間固定チェック
        if len(unique_periods) == 1 and '24' in unique_periods[0]:
            self.issues.append({
                'category': 'fixed_holding_period',
                'severity': 'high',
                'description': '保有期間がすべて24時間になっている',
                'details': analysis
            })
        
        return analysis
    
    def _analyze_profit_loss(self, trade_data: List, headers: List) -> Dict[str, Any]:
        """損益分析"""
        pnl_col = None
        
        for i, header in enumerate(headers):
            if '損益' in str(header) or 'P&L' in str(header) or 'Profit' in str(header):
                pnl_col = i
                break
        
        if pnl_col is None:
            return {'error': '損益列が見つかりません'}
        
        profits = []
        for row in trade_data:
            if len(row) > pnl_col and row[pnl_col] is not None:
                try:
                    profit = float(row[pnl_col])
                    profits.append(profit)
                except (ValueError, TypeError):
                    pass
        
        if profits:
            total_profit = sum(profits)
            avg_profit = total_profit / len(profits)
            winning_trades = len([p for p in profits if p > 0])
            win_rate = winning_trades / len(profits) if profits else 0
        else:
            total_profit = avg_profit = win_rate = 0
            winning_trades = 0
        
        analysis = {
            'pnl_column_index': pnl_col,
            'total_profit': total_profit,
            'average_profit': avg_profit,
            'winning_trades': winning_trades,
            'total_trades': len(profits),
            'win_rate': win_rate,
            'profit_sample': profits[:10]
        }
        
        return analysis
    
    def analyze_json_data(self, json_path: str) -> Dict[str, Any]:
        """JSONデータを分析"""
        self.logger.info(f"📦 JSON データ分析開始: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            trades_data = data.get('trades', [])
            switches_data = data.get('switches', [])
            
            analysis = {
                'trades_count': len(trades_data),
                'switches_count': len(switches_data),
                'trades_sample': trades_data[:3] if trades_data else [],
                'switches_sample': switches_data[:3] if switches_data else []
            }
            
            # 取引データの詳細分析
            if trades_data:
                # 戦略名確認
                strategies_in_json = []
                for trade in trades_data:
                    if 'strategy' in trade:
                        strategies_in_json.append(trade['strategy'])
                
                analysis['json_strategies'] = list(set(strategies_in_json))
                
                # 価格データ確認
                entry_prices = [trade.get('entry_price', 0) for trade in trades_data]
                exit_prices = [trade.get('exit_price', 0) for trade in trades_data]
                
                analysis['json_entry_prices_unique'] = len(set(entry_prices))
                analysis['json_exit_prices_unique'] = len(set(exit_prices))
                
                # 保有期間確認
                holding_periods = [trade.get('holding_period_hours', 0) for trade in trades_data]
                analysis['json_holding_periods_unique'] = len(set(holding_periods))
                analysis['json_avg_holding_period'] = sum(holding_periods) / len(holding_periods) if holding_periods else 0
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"JSON分析エラー: {e}")
            return {}
    
    def compare_text_report(self, text_path: str) -> Dict[str, Any]:
        """テキストレポートと比較"""
        self.logger.info(f"📄 テキストレポート分析: {text_path}")
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 平均保有期間を抽出
            import re
            holding_pattern = r'平均保有期間[：:]\s*([0-9.]+)時間'
            match = re.search(holding_pattern, content)
            
            analysis = {
                'text_avg_holding_period': float(match.group(1)) if match else None,
                'content_sample': content[:500]
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"テキスト分析エラー: {e}")
            return {}
    
    def generate_issue_report(self, excel_path: str, json_path: str, text_path: str) -> str:
        """総合問題レポート生成"""
        self.logger.info("🔍 総合問題分析開始")
        
        # 各ファイルを分析
        excel_analysis = self.analyze_excel_trade_history(excel_path)
        json_analysis = self.analyze_json_data(json_path)
        text_analysis = self.compare_text_report(text_path)
        
        # 結果をまとめる
        self.analysis_results = {
            'excel_analysis': excel_analysis,
            'json_analysis': json_analysis,
            'text_analysis': text_analysis,
            'issues_found': self.issues,
            'timestamp': datetime.now().isoformat()
        }
        
        # レポートファイルを生成
        report_path = f"dssms_trade_history_issue_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        # サマリーレポートを生成
        summary = self._generate_summary_report()
        
        summary_path = f"dssms_trade_history_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.logger.info(f"📋 詳細レポート: {report_path}")
        self.logger.info(f"📝 サマリーレポート: {summary_path}")
        
        return summary_path
    
    def _generate_summary_report(self) -> str:
        """サマリーレポート生成"""
        report = []
        report.append("=" * 70)
        report.append("DSSMS 取引履歴問題分析レポート")
        report.append("=" * 70)
        report.append(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        report.append("")
        
        # 問題サマリー
        report.append("🚨 発見された問題:")
        report.append("-" * 40)
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                report.append(f"{i}. [{issue['severity'].upper()}] {issue['description']}")
                if 'details' in issue:
                    report.append(f"   詳細: {issue['details']}")
                report.append("")
        else:
            report.append("問題は発見されませんでした。")
        
        # Excel分析結果
        excel_data = self.analysis_results.get('excel_analysis', {})
        if excel_data:
            report.append("📊 Excel取引履歴分析:")
            report.append("-" * 40)
            report.append(f"総取引数: {excel_data.get('total_trades', 0)}件")
            
            strategy_analysis = excel_data.get('strategy_analysis', {})
            if strategy_analysis:
                report.append(f"戦略種類: {strategy_analysis.get('strategy_count', 0)}種類")
                report.append(f"戦略名: {strategy_analysis.get('unique_strategies', [])}")
            
            price_analysis = excel_data.get('price_analysis', {})
            if price_analysis:
                report.append(f"エントリー価格種類: {price_analysis.get('entry_prices_unique', 0)}種類")
                report.append(f"エグジット価格種類: {price_analysis.get('exit_prices_unique', 0)}種類")
            
            holding_analysis = excel_data.get('holding_period_analysis', {})
            if holding_analysis:
                report.append(f"保有期間種類: {holding_analysis.get('unique_count', 0)}種類")
                report.append(f"保有期間例: {holding_analysis.get('unique_periods', [])}")
            
            report.append("")
        
        # JSON vs テキスト比較
        json_data = self.analysis_results.get('json_analysis', {})
        text_data = self.analysis_results.get('text_analysis', {})
        
        if json_data and text_data:
            report.append("📦 JSON vs テキスト比較:")
            report.append("-" * 40)
            
            json_avg = json_data.get('json_avg_holding_period', 0)
            text_avg = text_data.get('text_avg_holding_period', 0)
            
            if json_avg and text_avg:
                report.append(f"JSON平均保有期間: {json_avg:.1f}時間")
                report.append(f"テキスト平均保有期間: {text_avg:.1f}時間")
                
                if abs(json_avg - text_avg) > 1.0:
                    report.append("⚠️ 保有期間に不整合があります")
                else:
                    report.append("✅ 保有期間は整合しています")
            
            report.append("")
        
        # 修正提案
        report.append("🔧 修正提案:")
        report.append("-" * 40)
        
        if any(issue['category'] == 'strategy_naming' for issue in self.issues):
            report.append("1. 戦略名の詳細化")
            report.append("   - 7つの具体的戦略名を表示")
            report.append("   - 統一出力システムでの戦略名マッピング修正")
            report.append("")
        
        if any(issue['category'] == 'fixed_prices' for issue in self.issues):
            report.append("2. 価格データの修正")
            report.append("   - 実際の市場価格データを使用")
            report.append("   - エントリー・エグジット価格の正確な計算")
            report.append("")
        
        if any(issue['category'] == 'fixed_holding_period' for issue in self.issues):
            report.append("3. 保有期間の修正")
            report.append("   - 実際の取引時間間隔を計算")
            report.append("   - 時間単位での正確な保有期間表示")
            report.append("")
        
        report.append("4. 損益計算の検証")
        report.append("   - エントリー・エグジット価格を基にした正確な損益計算")
        report.append("   - 取引コストの考慮")
        report.append("")
        
        return "\n".join(report)

def main():
    """メイン実行"""
    analyzer = DSSMSTradeHistoryAnalyzer()
    
    # 最新のファイルパスを指定
    excel_path = "backtest_results/dssms_results/dssms_unified_backtest_20250908_150951.xlsx"
    json_path = "backtest_results/dssms_results/dssms_unified_data_20250908_150951.json"
    text_path = "backtest_results/dssms_results/dssms_unified_report_20250908_150951.txt"
    
    print("🔍 DSSMS取引履歴問題特定システム")
    print("=" * 60)
    print("対象ファイル:")
    print(f"Excel: {excel_path}")
    print(f"JSON: {json_path}")
    print(f"Text: {text_path}")
    print("=" * 60)
    
    # 総合分析実行
    summary_path = analyzer.generate_issue_report(excel_path, json_path, text_path)
    
    print(f"\n✅ 分析完了")
    print(f"📋 サマリーレポート: {summary_path}")
    print("\n次のステップ:")
    print("1. サマリーレポートで問題を確認")
    print("2. 統一出力システムの修正")
    print("3. 取引履歴生成ロジックの改善")

if __name__ == "__main__":
    main()
