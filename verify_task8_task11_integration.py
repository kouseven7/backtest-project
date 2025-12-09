"""
統合検証スクリプト - Task 8修正案2とTask 11の統合検証

Task 8修正案2（main_new.py側ForceClose）とTask 11（DSSMS側ForceClose）の
統合動作を検証するスクリプト。

主な機能:
- ケース1～4の検証（DSSMS側のみ、main_new.py側のみ、両方、なし）
- ログマーカー確認（[FORCE_CLOSE_*]、[DSSMS_FORCE_CLOSE_*]）
- execution_results.json分析（同日2件SELL問題、BUY/SELLペア一致）
- 検証結果レポート生成

統合コンポーネント:
- DSSMS統合バックテスター（dssms_integrated_main.py）
- ログファイル解析
- JSONファイル解析

セーフティ機能/注意事項:
- 実データのみ使用（モック/ダミー禁止、copilot-instructions.md準拠）
- 実際のバックテスト実行結果を検証
- フォールバック機能なし

Author: Backtest Project Team
Created: 2025-12-08
Last Modified: 2025-12-08
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


class Task8Task11IntegrationVerifier:
    """Task 8とTask 11の統合検証クラス"""
    
    def __init__(self, output_dir: str, log_file: str):
        """
        初期化
        
        Args:
            output_dir: バックテスト出力ディレクトリ
            log_file: ログファイルパス
        """
        self.output_dir = Path(output_dir)
        self.log_file = Path(log_file)
        self.results = {
            'case1': {'name': 'DSSMS側ForceCloseのみ', 'detected': False, 'details': []},
            'case2': {'name': 'main_new.py側ForceCloseのみ', 'detected': False, 'details': []},
            'case3': {'name': '両方が同日に発生', 'detected': False, 'details': []},
            'case4': {'name': 'どちらも発生しない', 'detected': False, 'details': []},
        }
        
    def verify_log_markers(self) -> Dict[str, Any]:
        """
        ログマーカーを検証
        
        Returns:
            検証結果
        """
        print("\n=== ログマーカー検証開始 ===")
        
        if not self.log_file.exists():
            print(f"[ERROR] ログファイルが見つかりません: {self.log_file}")
            return {'error': 'log_file_not_found'}
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        
        # Task 8のログマーカー（部分一致で検索）
        task8_markers = {
            'FORCE_CLOSE_START': len(re.findall(r'\[FORCE_CLOSE_START\]', log_content)),
            'FORCE_CLOSE_SUPPRESS': len(re.findall(r'\[FORCE_CLOSE_SUPPRESS\]', log_content)),
            'FORCE_CLOSE_END': len(re.findall(r'\[FORCE_CLOSE_END\]', log_content)),
        }
        
        # Task 11のログマーカー（部分一致で検索、改行・スペース考慮）
        task11_markers = {
            'DSSMS_FORCE_CLOSE_START': len(re.findall(r'\[DSSMS_FORCE_CLOSE_START', log_content)),
            'DSSMS_FORCE_CLOSE_SUPPRESS': len(re.findall(r'\[DSSMS_FORCE_CLOSE_SUPPRESS', log_content)),
            'DSSMS_FORCE_CLOSE_END': len(re.findall(r'\[DSSMS_FORCE_CLOSE_END', log_content)),
        }
        
        print("\n[Task 8] main_new.py側ForceCloseログマーカー:")
        for marker, count in task8_markers.items():
            print(f"  {marker}: {count}件")
        
        print("\n[Task 11] DSSMS側ForceCloseログマーカー:")
        for marker, count in task11_markers.items():
            print(f"  {marker}: {count}件")
        
        return {
            'task8': task8_markers,
            'task11': task11_markers,
        }
    
    def analyze_execution_results(self) -> Dict[str, Any]:
        """
        execution_results.jsonを分析
        
        Returns:
            分析結果
        """
        print("\n=== execution_results.json分析開始 ===")
        
        json_file = self.output_dir / 'dssms_execution_results.json'
        if not json_file.exists():
            print(f"[ERROR] JSONファイルが見つかりません: {json_file}")
            return {'error': 'json_file_not_found'}
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        execution_details = data.get('execution_details', [])
        
        # BUY/SELL件数カウント
        buy_count = sum(1 for t in execution_details if t.get('action') == 'BUY')
        sell_count = sum(1 for t in execution_details if t.get('action') == 'SELL')
        
        print(f"\n総取引件数: {len(execution_details)}件")
        print(f"  BUY: {buy_count}件")
        print(f"  SELL: {sell_count}件")
        print(f"  差分: {abs(buy_count - sell_count)}件")
        
        # 同日2件SELL問題の確認
        sell_by_date_symbol = {}
        for t in execution_details:
            if t.get('action') == 'SELL':
                key = f"{t.get('timestamp', '')[:10]}_{t.get('symbol', '')}"
                if key not in sell_by_date_symbol:
                    sell_by_date_symbol[key] = []
                sell_by_date_symbol[key].append(t)
        
        # 同日2件以上のSELLを検出
        multi_sell_cases = {k: v for k, v in sell_by_date_symbol.items() if len(v) >= 2}
        
        print(f"\n同日2件以上SELL: {len(multi_sell_cases)}ケース")
        for key, trades in multi_sell_cases.items():
            date, symbol = key.split('_')
            print(f"  {date} {symbol}: {len(trades)}件SELL")
            for t in trades:
                print(f"    - {t.get('strategy_name', 'Unknown')}: quantity={t.get('quantity', 'N/A')}")
        
        return {
            'total_trades': len(execution_details),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_sell_match': buy_count == sell_count,
            'multi_sell_cases': len(multi_sell_cases),
            'multi_sell_details': multi_sell_cases,
        }
    
    def detect_cases(self, log_markers: Dict[str, Any], execution_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        ケース1～4を検出
        
        Args:
            log_markers: ログマーカー検証結果
            execution_analysis: execution_results.json分析結果
        
        Returns:
            ケース検出結果
        """
        print("\n=== ケース検出開始 ===")
        
        task8_active = log_markers['task8']['FORCE_CLOSE_START'] > 0
        task11_active = log_markers['task11']['DSSMS_FORCE_CLOSE_START'] > 0
        
        # ケース1: DSSMS側ForceCloseのみ
        if task11_active and not task8_active:
            self.results['case1']['detected'] = True
            self.results['case1']['details'] = log_markers['task11']
            print("[ケース1] DSSMS側ForceCloseのみ発生: 検出")
        
        # ケース2: main_new.py側ForceCloseのみ
        if task8_active and not task11_active:
            self.results['case2']['detected'] = True
            self.results['case2']['details'] = log_markers['task8']
            print("[ケース2] main_new.py側ForceCloseのみ発生: 検出")
        
        # ケース3: 両方が発生
        if task8_active and task11_active:
            self.results['case3']['detected'] = True
            self.results['case3']['details'] = {
                'task8': log_markers['task8'],
                'task11': log_markers['task11'],
            }
            print("[ケース3] 両方が同日に発生: 検出")
        
        # ケース4: どちらも発生しない
        if not task8_active and not task11_active:
            self.results['case4']['detected'] = True
            print("[ケース4] どちらも発生しない: 検出")
        
        return self.results
    
    def generate_report(self, log_markers: Dict[str, Any], execution_analysis: Dict[str, Any], 
                       case_detection: Dict[str, Any]) -> str:
        """
        検証結果レポートを生成
        
        Args:
            log_markers: ログマーカー検証結果
            execution_analysis: execution_results.json分析結果
            case_detection: ケース検出結果
        
        Returns:
            レポート文字列
        """
        report = []
        report.append("=" * 80)
        report.append("Task 8 & Task 11 統合検証レポート")
        report.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # ログマーカー
        report.append("\n## ログマーカー検証結果")
        report.append("\n### Task 8 (main_new.py側ForceClose)")
        for marker, count in log_markers['task8'].items():
            report.append(f"  {marker}: {count}件")
        
        report.append("\n### Task 11 (DSSMS側ForceClose)")
        for marker, count in log_markers['task11'].items():
            report.append(f"  {marker}: {count}件")
        
        # execution_results.json分析
        report.append("\n## execution_results.json分析結果")
        report.append(f"  総取引件数: {execution_analysis['total_trades']}件")
        report.append(f"  BUY: {execution_analysis['buy_count']}件")
        report.append(f"  SELL: {execution_analysis['sell_count']}件")
        report.append(f"  BUY/SELL一致: {'YES' if execution_analysis['buy_sell_match'] else 'NO'}")
        report.append(f"  同日2件以上SELL: {execution_analysis['multi_sell_cases']}ケース")
        
        # ケース検出
        report.append("\n## ケース検出結果")
        for case_id, case_info in case_detection.items():
            status = "検出" if case_info['detected'] else "未検出"
            report.append(f"  [{case_id}] {case_info['name']}: {status}")
        
        # 成功基準チェック
        report.append("\n## 成功基準チェック")
        criteria = {
            '同日2件SELL問題解消': execution_analysis['multi_sell_cases'] == 0,
            'BUY/SELLペア一致': execution_analysis['buy_sell_match'],
            'Task 8ログマーカー出力': log_markers['task8']['FORCE_CLOSE_START'] > 0 or log_markers['task8']['FORCE_CLOSE_SUPPRESS'] > 0,
            'Task 11ログマーカー出力': log_markers['task11']['DSSMS_FORCE_CLOSE_START'] > 0 or log_markers['task11']['DSSMS_FORCE_CLOSE_SUPPRESS'] > 0,
        }
        
        for criterion, passed in criteria.items():
            status = "PASS" if passed else "FAIL"
            report.append(f"  {criterion}: {status}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def run(self) -> None:
        """統合検証を実行"""
        print("Task 8 & Task 11 統合検証開始")
        print(f"出力ディレクトリ: {self.output_dir}")
        print(f"ログファイル: {self.log_file}")
        
        # 1. ログマーカー検証
        log_markers = self.verify_log_markers()
        
        # 2. execution_results.json分析
        execution_analysis = self.analyze_execution_results()
        
        # 3. ケース検出
        case_detection = self.detect_cases(log_markers, execution_analysis)
        
        # 4. レポート生成
        report = self.generate_report(log_markers, execution_analysis, case_detection)
        
        print("\n" + report)
        
        # レポートをファイルに保存
        report_file = Path('task8_task11_integration_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nレポートを保存しました: {report_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python verify_task8_task11_integration.py <output_dir> <log_file>")
        print("Example: python verify_task8_task11_integration.py output/dssms_integration/dssms_20251208_193732 task11_backtest.log")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    log_file = sys.argv[2]
    
    verifier = Task8Task11IntegrationVerifier(output_dir, log_file)
    verifier.run()
