"""
インジケーター使用状況診断ツール - shift()適用の確認

basic_indicators.pyの関数がどのように使用され、
shift()が適用されているかを詳細に分析します。

主な機能:
- 戦略ファイル内のインジケーター呼び出し箇所を特定
- shift()の適用有無を確認
- iloc[idx]での直接参照パターンを検出
- 問題のある使用パターンをレポート

統合コンポーネント:
- strategies/*.py: 全戦略ファイル
- indicators/basic_indicators.py: 分析対象のインジケーター

セーフティ機能/注意事項:
- 静的解析のみ
- False Positiveの可能性あり
- 最終判断は手動レビュー必須

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""

import ast
import re
from pathlib import Path
from typing import List, Dict
import pandas as pd

class IndicatorUsageAnalyzer:
    """インジケーター使用状況を分析"""
    
    def __init__(self, strategies_dir: str):
        self.strategies_dir = Path(strategies_dir)
        self.target_indicators = ['calculate_sma', 'calculate_rsi', 'calculate_vwap']
        
    def analyze_all_strategies(self) -> pd.DataFrame:
        """全戦略を分析"""
        results = []
        
        for strategy_file in self.strategies_dir.glob("*.py"):
            if strategy_file.name.startswith("__"):
                continue
                
            print(f"\n分析中: {strategy_file.name}")
            issues = self._analyze_strategy_file(strategy_file)
            
            for issue in issues:
                results.append({
                    'strategy_file': strategy_file.stem,
                    'indicator_function': issue['indicator'],
                    'line_number': issue['line'],
                    'usage_pattern': issue['pattern'],
                    'has_shift': issue['has_shift'],
                    'risk_assessment': issue['risk'],
                    'code_snippet': issue['code']
                })
        
        return pd.DataFrame(results)
    
    def _analyze_strategy_file(self, file_path: Path) -> List[Dict]:
        """1つの戦略ファイルを分析"""
        issues = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            lines = source.split('\n')
        
        # インジケーター呼び出し行を検索
        for line_num, line in enumerate(lines, start=1):
            for indicator in self.target_indicators:
                if indicator in line:
                    # shift()が同じ行または次の数行内にあるかチェック
                    context_lines = lines[max(0, line_num-1):min(len(lines), line_num+5)]
                    context = '\n'.join(context_lines)
                    
                    has_shift = '.shift(' in context
                    
                    # iloc[idx]パターンの検出
                    has_iloc_idx = re.search(r'\.iloc\[idx\]', context)
                    
                    # リスク評価
                    if has_iloc_idx and not has_shift:
                        risk = "HIGH"
                        pattern = "iloc[idx]でshift()なし"
                    elif has_iloc_idx and has_shift:
                        risk = "MEDIUM"
                        pattern = "iloc[idx]だがshift()あり（要確認）"
                    elif not has_shift:
                        risk = "MEDIUM"
                        pattern = "shift()なし（要確認）"
                    else:
                        risk = "LOW"
                        pattern = "shift()適用済み"
                    
                    issues.append({
                        'indicator': indicator,
                        'line': line_num,
                        'pattern': pattern,
                        'has_shift': has_shift,
                        'risk': risk,
                        'code': context.strip()
                    })
        
        return issues


def main():
    """メイン実行"""
    strategies_dir = r"c:\Users\imega\Documents\my_backtest_project\strategies"
    
    analyzer = IndicatorUsageAnalyzer(strategies_dir)
    
    print("="*80)
    print("インジケーター使用状況診断開始")
    print("="*80)
    
    results_df = analyzer.analyze_all_strategies()
    
    if len(results_df) == 0:
        print("\nインジケーター使用は検出されませんでした。")
        return
    
    # リスクレベル別に表示
    print("\n" + "="*80)
    print("検出された使用パターン:")
    print("="*80)
    
    for risk_level in ['HIGH', 'MEDIUM', 'LOW']:
        level_issues = results_df[results_df['risk_assessment'] == risk_level]
        if len(level_issues) > 0:
            print(f"\n[{risk_level}] {len(level_issues)}件")
            for _, issue in level_issues.iterrows():
                print(f"\n  戦略: {issue['strategy_file']}")
                print(f"  インジケーター: {issue['indicator_function']}")
                print(f"  行番号: {issue['line_number']}")
                print(f"  パターン: {issue['usage_pattern']}")
                print(f"  shift()適用: {'はい' if issue['has_shift'] else 'いいえ'}")
                print(f"  コード:")
                print(f"    {issue['code_snippet'][:200]}...")
    
    # CSV出力
    output_path = r"c:\Users\imega\Documents\my_backtest_project\indicator_usage_diagnosis.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n結果を保存しました: {output_path}")
    
    # サマリー
    print("\n" + "="*80)
    print("サマリー:")
    print("="*80)
    summary = results_df.groupby(['indicator_function', 'risk_assessment']).size().unstack(fill_value=0)
    print(summary)


if __name__ == "__main__":
    main()
