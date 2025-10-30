"""
ルックアヘッドバイアス検証ツール - 戦略とインジケーターの静的解析

各戦略および使用するインジケーター関数のコードを解析し、
未来データの混入リスクを特定します。

主な機能:
- 戦略ファイルのgenerate_entry_signal/generate_exit_signal解析
- インジケーター関数のshift()使用状況チェック
- rolling()計算の当日含む問題検出
- 同日データ参照パターンの検出
- リスクレベル別の詳細レポート生成

統合コンポーネント:
- strategies/*.py: 全戦略モジュール
- indicators/*.py: 使用される全インジケーターモジュール
- strategies/base_strategy.py: バックテストループ確認

セーフティ機能/注意事項:
- 静的解析のみ（実際のバックテスト実行なし）
- 検出結果は参考情報（最終判断は手動レビュー必須）
- False Positiveの可能性を考慮

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

class LookaheadBiasChecker:
    """戦略コードとインジケーターのルックアヘッドバイアスを検出"""
    
    def __init__(self, strategies_dir: str, indicators_dir: str):
        self.strategies_dir = Path(strategies_dir)
        self.indicators_dir = Path(indicators_dir)
        self.issues = []
        
    def check_all_strategies(self) -> pd.DataFrame:
        """全戦略とインジケーターを検証"""
        results = []
        
        # 戦略ファイルの検証
        print("\n" + "="*80)
        print("戦略ファイルの検証開始")
        print("="*80)
        
        for strategy_file in self.strategies_dir.glob("*.py"):
            if strategy_file.name.startswith("__") or strategy_file.name == "base_strategy.py":
                continue
                
            print(f"\n検証中: {strategy_file.name}")
            issues = self._check_strategy_file(strategy_file)
            
            for issue in issues:
                results.append({
                    'file_type': 'strategy',
                    'file_name': strategy_file.stem,
                    'risk_level': issue['risk_level'],
                    'issue_type': issue['issue_type'],
                    'line_number': issue['line_number'],
                    'code_snippet': issue['code_snippet'],
                    'explanation': issue['explanation']
                })
        
        # インジケーターファイルの検証
        print("\n" + "="*80)
        print("インジケーターファイルの検証開始")
        print("="*80)
        
        for indicator_file in self.indicators_dir.glob("*.py"):
            if indicator_file.name.startswith("__"):
                continue
                
            print(f"\n検証中: {indicator_file.name}")
            issues = self._check_indicator_file(indicator_file)
            
            for issue in issues:
                results.append({
                    'file_type': 'indicator',
                    'file_name': indicator_file.stem,
                    'risk_level': issue['risk_level'],
                    'issue_type': issue['issue_type'],
                    'line_number': issue['line_number'],
                    'code_snippet': issue['code_snippet'],
                    'explanation': issue['explanation']
                })
        
        return pd.DataFrame(results)
    
    def _check_strategy_file(self, file_path: Path) -> List[Dict]:
        """1つの戦略ファイルを検証"""
        issues = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return [{'risk_level': 'ERROR', 'issue_type': 'syntax_error',
                    'line_number': e.lineno, 'code_snippet': str(e),
                    'explanation': '構文エラー'}]
        
        # generate_entry_signal と generate_exit_signal を探す
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in ['generate_entry_signal', 'generate_exit_signal']:
                    issues.extend(self._check_signal_function(node, source))
        
        return issues
    
    def _check_indicator_file(self, file_path: Path) -> List[Dict]:
        """1つのインジケーターファイルを検証"""
        issues = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return [{'risk_level': 'ERROR', 'issue_type': 'syntax_error',
                    'line_number': e.lineno, 'code_snippet': str(e),
                    'explanation': '構文エラー'}]
        
        # 全関数を検証
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('calculate_') or node.name.startswith('detect_'):
                    issues.extend(self._check_indicator_function(node, source))
        
        return issues
    
    def _check_signal_function(self, func_node: ast.FunctionDef, source: str) -> List[Dict]:
        """シグナル生成関数内のバイアスをチェック"""
        issues = []
        
        for node in ast.walk(func_node):
            # パターン1: shift()の使用チェック
            if isinstance(node, ast.Call) and self._is_method_call(node, 'shift'):
                issue = self._check_shift_usage(node, source)
                if issue:
                    issues.append(issue)
            
            # パターン2: rolling()の使用チェック
            if isinstance(node, ast.Call) and self._is_method_call(node, 'rolling'):
                issue = self._check_rolling_usage(node, source)
                if issue:
                    issues.append(issue)
            
            # パターン3: iloc[-1]の使用チェック
            if isinstance(node, ast.Subscript):
                issue = self._check_iloc_usage(node, source)
                if issue:
                    issues.append(issue)
        
        return issues
    
    def _check_indicator_function(self, func_node: ast.FunctionDef, source: str) -> List[Dict]:
        """インジケーター関数内のバイアスをチェック"""
        issues = []
        
        for node in ast.walk(func_node):
            # rolling()計算のチェック
            if isinstance(node, ast.Call) and self._is_method_call(node, 'rolling'):
                issue = self._check_rolling_in_indicator(node, source, func_node.name)
                if issue:
                    issues.append(issue)
            
            # cumsum()のチェック（VWAPなど）
            if isinstance(node, ast.Call) and self._is_method_call(node, 'cumsum'):
                issue = self._check_cumsum_usage(node, source, func_node.name)
                if issue:
                    issues.append(issue)
        
        return issues
    
    def _check_shift_usage(self, node: ast.Call, source: str) -> Optional[Dict]:
        """shift()の使い方をチェック"""
        if node.args and isinstance(node.args[0], ast.Constant):
            shift_amount = node.args[0].value
            if shift_amount == 0:
                return {
                    'risk_level': 'CRITICAL',
                    'issue_type': 'no_shift',
                    'line_number': node.lineno,
                    'code_snippet': self._get_code_snippet(source, node.lineno),
                    'explanation': 'shift(0)は当日データを参照します。前日以前のデータを使用する場合はshift(1)以上を使用してください。'
                }
        elif not node.args:
            # shift()が引数なしで呼ばれている場合
            return {
                'risk_level': 'HIGH',
                'issue_type': 'shift_no_args',
                'line_number': node.lineno,
                'code_snippet': self._get_code_snippet(source, node.lineno),
                'explanation': 'shift()の引数が明示されていません。デフォルト(1)が使用されますが、明示的な指定を推奨します。'
            }
        return None
    
    def _check_rolling_usage(self, node: ast.Call, source: str) -> Optional[Dict]:
        """rolling()の使い方をチェック"""
        return {
            'risk_level': 'MEDIUM',
            'issue_type': 'rolling_check_needed',
            'line_number': node.lineno,
            'code_snippet': self._get_code_snippet(source, node.lineno),
            'explanation': 'rolling()計算が当日を含んでいる可能性があります。計算結果をshift(1)で1日ずらしているか確認してください。'
        }
    
    def _check_rolling_in_indicator(self, node: ast.Call, source: str, func_name: str) -> Optional[Dict]:
        """インジケーター内のrolling()をチェック"""
        return {
            'risk_level': 'HIGH',
            'issue_type': 'indicator_rolling',
            'line_number': node.lineno,
            'code_snippet': self._get_code_snippet(source, node.lineno),
            'explanation': f'関数 {func_name}() 内でrolling()を使用しています。この関数を呼び出す側でshift(1)を適用しているか確認してください。'
        }
    
    def _check_cumsum_usage(self, node: ast.Call, source: str, func_name: str) -> Optional[Dict]:
        """cumsum()の使用チェック（VWAP計算など）"""
        return {
            'risk_level': 'HIGH',
            'issue_type': 'cumsum_check',
            'line_number': node.lineno,
            'code_snippet': self._get_code_snippet(source, node.lineno),
            'explanation': f'関数 {func_name}() 内でcumsum()を使用しています。累積計算に当日データが含まれている可能性があります。'
        }
    
    def _check_iloc_usage(self, node: ast.Subscript, source: str) -> Optional[Dict]:
        """iloc[-1]などの使用をチェック"""
        try:
            # iloc[-1]パターンの検出
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'iloc':
                if isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub):
                    if isinstance(node.slice.operand, ast.Constant) and node.slice.operand.value == 1:
                        return {
                            'risk_level': 'HIGH',
                            'issue_type': 'iloc_current_day',
                            'line_number': node.lineno,
                            'code_snippet': self._get_code_snippet(source, node.lineno),
                            'explanation': 'iloc[-1]は当日データを参照しています。前日データを参照する場合はiloc[-2]を使用してください。'
                        }
        except:
            pass
        return None
    
    def _is_method_call(self, node: ast.Call, method_name: str) -> bool:
        """指定されたメソッド呼び出しかチェック"""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == method_name
        return False
    
    def _get_code_snippet(self, source: str, line_number: int, context: int = 2) -> str:
        """指定行の前後のコードスニペットを取得"""
        lines = source.split('\n')
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        snippet = '\n'.join(f"{i+1:4d}: {lines[i]}" for i in range(start, end))
        return snippet


def main():
    """メイン実行"""
    strategies_dir = r"c:\Users\imega\Documents\my_backtest_project\strategies"
    indicators_dir = r"c:\Users\imega\Documents\my_backtest_project\indicators"
    
    checker = LookaheadBiasChecker(strategies_dir, indicators_dir)
    
    print("="*80)
    print("ルックアヘッドバイアス検証開始")
    print("="*80)
    
    results_df = checker.check_all_strategies()
    
    if len(results_df) == 0:
        print("\n問題は検出されませんでした。")
        return
    
    # 結果をリスクレベル別に表示
    print("\n" + "="*80)
    print("検出された問題:")
    print("="*80)
    
    for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        level_issues = results_df[results_df['risk_level'] == risk_level]
        if len(level_issues) > 0:
            print(f"\n[{risk_level}] {len(level_issues)}件")
            for _, issue in level_issues.iterrows():
                print(f"\n  ファイル種別: {issue['file_type']}")
                print(f"  ファイル名: {issue['file_name']}")
                print(f"  行番号: {issue['line_number']}")
                print(f"  種類: {issue['issue_type']}")
                print(f"  説明: {issue['explanation']}")
                print(f"  コード:")
                print(f"{issue['code_snippet']}")
    
    # CSV出力
    output_path = r"c:\Users\imega\Documents\my_backtest_project\lookahead_bias_check.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n結果を保存しました: {output_path}")
    
    # サマリー出力
    print("\n" + "="*80)
    print("検証サマリー:")
    print("="*80)
    summary = results_df.groupby(['file_type', 'risk_level']).size().unstack(fill_value=0)
    print(summary)


if __name__ == "__main__":
    main()
