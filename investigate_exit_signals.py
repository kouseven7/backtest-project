"""
エグジットシグナル生成状況調査スクリプト
Phase 3.2: 各戦略のbacktest()メソッド内エグジット生成状況を詳細調査

ユーザー回答：「エグジットシグナルが生成されているかでていないのか不明」
"""

import sys
import os
from pathlib import Path
import re
import ast
from typing import Dict, List, Any, Optional
from datetime import datetime

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ExitSignalInvestigator:
    """エグジットシグナル生成状況調査クラス"""
    
    def __init__(self):
        self.strategies_dir = project_root / "strategies"
        self.investigation_results = {}
        
        # 調査対象戦略リスト
        self.strategies_to_investigate = {
            'VWAPBreakoutStrategy': 'VWAP_Breakout.py',
            'VWAPBounceStrategy': 'VWAP_Bounce.py',
            'MomentumInvestingStrategy': 'Momentum_Investing.py',
            'BreakoutStrategy': 'Breakout.py',
            'OpeningGapStrategy': 'Opening_Gap_Fixed.py',
            'ContrarianStrategy': 'contrarian_strategy.py',
            'GCStrategy': 'gc_strategy_signal.py'
        }
    
    def investigate_all_strategies(self) -> Dict[str, Any]:
        """全戦略のエグジットシグナル生成状況を調査"""
        
        print("=" * 80)
        print("エグジットシグナル生成状況調査開始")
        print("=" * 80)
        
        for strategy_name, filename in self.strategies_to_investigate.items():
            print(f"\n[調査中] {strategy_name} ({filename})")
            
            result = self.analyze_strategy_exit_signals(strategy_name, filename)
            self.investigation_results[strategy_name] = result
            
            self._print_strategy_result(strategy_name, result)
        
        return self.investigation_results
    
    def analyze_strategy_exit_signals(self, strategy_name: str, filename: str) -> Dict[str, Any]:
        """個別戦略のエグジットシグナル生成状況を分析"""
        
        filepath = self.strategies_dir / filename
        
        if not filepath.exists():
            return {
                'status': 'FILE_NOT_FOUND',
                'file_exists': False,
                'error': f'File not found: {filepath}'
            }
        
        try:
            # ファイル読み取り
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # エグジットシグナル生成パターンを検索
            exit_patterns = self._find_exit_signal_patterns(content)
            
            # backtest()メソッドの存在確認
            has_backtest = 'def backtest(' in content
            
            # Exit_Signal列の生成確認
            has_exit_signal_column = self._check_exit_signal_column_generation(content)
            
            # エグジット条件の存在確認
            exit_conditions = self._find_exit_conditions(content)
            
            return {
                'status': 'ANALYZED',
                'file_exists': True,
                'filepath': str(filepath),
                'has_backtest_method': has_backtest,
                'has_exit_signal_column': has_exit_signal_column,
                'exit_patterns_found': exit_patterns,
                'exit_conditions_found': exit_conditions,
                'exit_signal_generated': len(exit_patterns) > 0 or has_exit_signal_column
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'file_exists': True,
                'error': str(e)
            }
    
    def _find_exit_signal_patterns(self, content: str) -> List[Dict[str, Any]]:
        """エグジットシグナル生成パターンを検索"""
        patterns = []
        
        # パターン1: Exit_Signal列への代入
        pattern1 = r"(.*\[\'Exit_Signal\'\].*=.*)"
        matches1 = re.finditer(pattern1, content)
        for match in matches1:
            patterns.append({
                'type': 'Exit_Signal_Assignment',
                'code': match.group(1).strip(),
                'line': content[:match.start()].count('\n') + 1
            })
        
        # パターン2: exit_signal変数の使用
        pattern2 = r"(.*exit_signal.*=.*)"
        matches2 = re.finditer(pattern2, content, re.IGNORECASE)
        for match in matches2:
            patterns.append({
                'type': 'exit_signal_variable',
                'code': match.group(1).strip(),
                'line': content[:match.start()].count('\n') + 1
            })
        
        # パターン3: エグジット条件の明示的な記述
        pattern3 = r"(.*# ?exit.*condition.*)"
        matches3 = re.finditer(pattern3, content, re.IGNORECASE)
        for match in matches3:
            patterns.append({
                'type': 'exit_condition_comment',
                'code': match.group(1).strip(),
                'line': content[:match.start()].count('\n') + 1
            })
        
        return patterns
    
    def _check_exit_signal_column_generation(self, content: str) -> bool:
        """Exit_Signal列の生成確認"""
        exit_signal_patterns = [
            r"df\[\'Exit_Signal\'\]",
            r"data\[\'Exit_Signal\'\]",
            r"stock_data\[\'Exit_Signal\'\]",
            r"\.assign\(Exit_Signal",
        ]
        
        for pattern in exit_signal_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _find_exit_conditions(self, content: str) -> List[str]:
        """エグジット条件の検索"""
        conditions = []
        
        # エグジット条件パターン
        condition_patterns = [
            r"(if.*exit.*:)",
            r"(elif.*exit.*:)",
            r"(.*\(.*exit.*\))",
        ]
        
        for pattern in condition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                conditions.append(match.group(1).strip())
        
        return conditions[:5]  # 最大5件まで
    
    def _print_strategy_result(self, strategy_name: str, result: Dict[str, Any]):
        """個別戦略の調査結果を出力"""
        
        if result['status'] == 'FILE_NOT_FOUND':
            print(f"  ❌ ファイルが見つかりません")
            return
        
        if result['status'] == 'ERROR':
            print(f"  ❌ エラー: {result['error']}")
            return
        
        # 結果表示
        print(f"  backtest()メソッド: {'✅ あり' if result['has_backtest_method'] else '❌ なし'}")
        print(f"  Exit_Signal列生成: {'✅ あり' if result['has_exit_signal_column'] else '❌ なし'}")
        print(f"  エグジットパターン: {len(result['exit_patterns_found'])}個検出")
        
        if result['exit_patterns_found']:
            print(f"  検出されたパターン:")
            for i, pattern in enumerate(result['exit_patterns_found'][:3], 1):
                print(f"    {i}. [{pattern['type']}] Line {pattern['line']}: {pattern['code'][:60]}...")
        
        if result['exit_signal_generated']:
            print(f"  📊 判定: エグジットシグナル生成あり")
        else:
            print(f"  ⚠️  判定: エグジットシグナル生成なし")
    
    def generate_summary_report(self) -> str:
        """サマリーレポート生成"""
        
        total_strategies = len(self.investigation_results)
        strategies_with_exit = sum(
            1 for result in self.investigation_results.values()
            if result.get('exit_signal_generated', False)
        )
        strategies_without_exit = total_strategies - strategies_with_exit
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("エグジットシグナル生成状況調査サマリー")
        report.append("=" * 80)
        report.append(f"\n調査対象戦略数: {total_strategies}")
        report.append(f"エグジットシグナル生成あり: {strategies_with_exit} 戦略")
        report.append(f"エグジットシグナル生成なし: {strategies_without_exit} 戦略")
        report.append(f"\n生成率: {strategies_with_exit / total_strategies * 100:.1f}%")
        
        # 詳細リスト
        report.append("\n" + "-" * 80)
        report.append("詳細リスト")
        report.append("-" * 80)
        
        for strategy_name, result in self.investigation_results.items():
            if result.get('exit_signal_generated'):
                status = "✅ 生成あり"
            else:
                status = "❌ 生成なし"
            
            report.append(f"{status} | {strategy_name}")
        
        # 推奨アクション
        report.append("\n" + "-" * 80)
        report.append("推奨アクション")
        report.append("-" * 80)
        
        if strategies_without_exit > 0:
            report.append("⚠️  エグジットシグナル生成なしの戦略が検出されました。")
            report.append("    以下の対応を推奨します：")
            report.append("    1. 各戦略のbacktest()メソッドにExit_Signal列生成を追加")
            report.append("    2. エグジット条件の明確化")
            report.append("    3. 統一されたエグジットシグナル生成インターフェースの実装")
        else:
            report.append("✅ すべての戦略でエグジットシグナル生成が確認されました。")
        
        return "\n".join(report)
    
    def save_detailed_report(self, output_path: Path):
        """詳細レポートをファイルに保存"""
        
        report_lines = []
        report_lines.append("# エグジットシグナル生成状況調査 - 詳細レポート\n")
        report_lines.append(f"**調査日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**Phase**: Phase 3.2 - エグジット生成問題調査・修正\n")
        report_lines.append("\n---\n")
        
        # サマリー
        total = len(self.investigation_results)
        with_exit = sum(1 for r in self.investigation_results.values() if r.get('exit_signal_generated'))
        
        report_lines.append(f"## 調査サマリー\n")
        report_lines.append(f"- **調査対象戦略数**: {total}\n")
        report_lines.append(f"- **エグジットシグナル生成あり**: {with_exit} 戦略\n")
        report_lines.append(f"- **エグジットシグナル生成なし**: {total - with_exit} 戦略\n")
        report_lines.append(f"- **生成率**: {with_exit / total * 100:.1f}%\n")
        report_lines.append("\n---\n")
        
        # 各戦略の詳細
        report_lines.append(f"## 各戦略の詳細\n")
        
        for strategy_name, result in self.investigation_results.items():
            report_lines.append(f"\n### {strategy_name}\n")
            
            if result['status'] != 'ANALYZED':
                report_lines.append(f"**ステータス**: {result['status']}\n")
                if 'error' in result:
                    report_lines.append(f"**エラー**: {result['error']}\n")
                continue
            
            report_lines.append(f"**ファイル**: `{result['filepath']}`\n")
            report_lines.append(f"**backtest()メソッド**: {'✅ あり' if result['has_backtest_method'] else '❌ なし'}\n")
            report_lines.append(f"**Exit_Signal列生成**: {'✅ あり' if result['has_exit_signal_column'] else '❌ なし'}\n")
            report_lines.append(f"**エグジットシグナル生成**: {'✅ あり' if result['exit_signal_generated'] else '❌ なし'}\n")
            
            if result['exit_patterns_found']:
                report_lines.append(f"\n**検出されたパターン** ({len(result['exit_patterns_found'])}個):\n")
                for pattern in result['exit_patterns_found'][:5]:
                    report_lines.append(f"- **[{pattern['type']}]** Line {pattern['line']}: `{pattern['code']}`\n")
            
            if result['exit_conditions_found']:
                report_lines.append(f"\n**エグジット条件** ({len(result['exit_conditions_found'])}個):\n")
                for condition in result['exit_conditions_found']:
                    report_lines.append(f"- `{condition}`\n")
        
        # 推奨アクション
        report_lines.append("\n---\n")
        report_lines.append(f"## 推奨アクション\n")
        
        if total - with_exit > 0:
            report_lines.append("### ⚠️ エグジットシグナル生成なしの戦略が検出されました\n")
            report_lines.append("\n**対応が必要な戦略**:\n")
            for strategy_name, result in self.investigation_results.items():
                if not result.get('exit_signal_generated', False):
                    report_lines.append(f"- `{strategy_name}`\n")
            
            report_lines.append("\n**推奨される対応**:\n")
            report_lines.append("1. 各戦略の`backtest()`メソッドに`Exit_Signal`列生成を追加\n")
            report_lines.append("2. エグジット条件の明確化\n")
            report_lines.append("3. 統一されたエグジットシグナル生成インターフェースの実装\n")
            report_lines.append("4. `.github/copilot-instructions.md` 遵守確認\n")
        else:
            report_lines.append("### ✅ すべての戦略でエグジットシグナル生成が確認されました\n")
        
        # ファイル保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"\n📄 詳細レポート保存: {output_path}")


def main():
    """メイン実行"""
    
    investigator = ExitSignalInvestigator()
    
    # 調査実行
    results = investigator.investigate_all_strategies()
    
    # サマリー表示
    summary = investigator.generate_summary_report()
    print(summary)
    
    # 詳細レポート保存
    output_dir = project_root / "diagnostics" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "exit_signal_investigation_report.md"
    investigator.save_detailed_report(report_path)
    
    print(f"\n✅ エグジットシグナル調査完了")
    
    return results


if __name__ == '__main__':
    main()
