"""
戦略選択状況分析スクリプト

目的:
- 実際にどの戦略がいつ選ばれたか確認
- 戦略スコアの詳細を確認
- GC戦略のみ選ばれた理由を特定

Author: Backtest Project Team
Created: 2026-01-12
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_comprehensive_report():
    """包括レポートから戦略選択状況を確認"""
    print(f"\n{'='*80}")
    print("包括レポート（JSON）分析")
    print(f"{'='*80}")
    
    json_file = project_root / "output" / "dssms_integration" / "dssms_20260111_232522" / "dssms_comprehensive_report.json"
    
    if not json_file.exists():
        print(f"[ERROR] JSONレポートが見つかりません: {json_file}")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 戦略効果分析
    if 'strategy_effectiveness' in data:
        strategy_eff = data['strategy_effectiveness']
        
        print(f"\n個別戦略分析:")
        individual = strategy_eff.get('individual_strategy_analysis', {})
        
        for strategy_name, stats in individual.items():
            print(f"\n{strategy_name}:")
            print(f"  取引回数: {stats.get('total_trades', 0)}回")
            print(f"  成功回数: {stats.get('successful_trades', 0)}回")
            print(f"  成功率: {stats.get('success_rate', 0.0):.1%}")
            print(f"  効果スコア: {stats.get('effectiveness_score', 0.0):.4f}")
        
        # 戦略ランキング
        print(f"\n{'='*60}")
        print("戦略ランキング（スコア順）:")
        print(f"{'='*60}")
        
        ranking = strategy_eff.get('strategy_ranking', [])
        for rank_info in ranking:
            print(f"{rank_info['rank']}位: {rank_info['strategy']}")
            print(f"     スコア: {rank_info['score']:.4f}")
            print(f"     取引: {rank_info['total_trades']}回, 成功率: {rank_info['success_rate']:.1%}")
    
    return data


def search_log_for_strategy_selection():
    """ログファイルから戦略選択の詳細を確認"""
    print(f"\n{'='*80}")
    print("ログファイルから戦略選択詳細を検索")
    print(f"{'='*80}")
    
    log_dirs = [
        project_root / "logs",
        project_root / "output" / "dssms_integration" / "dssms_20260111_232522"
    ]
    
    strategy_selection_logs = []
    
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        
        for log_file in log_dir.glob("*.log"):
            print(f"\n検索中: {log_file.name}")
            
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                # 戦略選択関連のログを探す
                for i, line in enumerate(lines):
                    if any(keyword in line for keyword in [
                        "SELECTION_RESULT", 
                        "Selected strategies",
                        "Strategy selection completed",
                        "strategy_scores"
                    ]):
                        # 前後5行を含めて記録
                        start = max(0, i-5)
                        end = min(len(lines), i+6)
                        context = ''.join(lines[start:end])
                        strategy_selection_logs.append({
                            'file': log_file.name,
                            'line_num': i,
                            'context': context
                        })
                        
                        if len(strategy_selection_logs) >= 20:  # 最大20件
                            break
                
                if len(strategy_selection_logs) >= 20:
                    break
                    
            except Exception as e:
                print(f"  [ERROR] 読み込みエラー: {e}")
    
    if strategy_selection_logs:
        print(f"\n見つかった戦略選択ログ: {len(strategy_selection_logs)}件")
        print(f"\n{'='*60}")
        print("最初の3件を表示:")
        print(f"{'='*60}")
        
        for i, log_entry in enumerate(strategy_selection_logs[:3]):
            print(f"\n[{i+1}] {log_entry['file']} (行{log_entry['line_num']})")
            print(log_entry['context'])
    else:
        print("\n戦略選択に関するログが見つかりませんでした")
    
    return strategy_selection_logs


def analyze_execution_results():
    """実行結果JSONを確認"""
    print(f"\n{'='*80}")
    print("実行結果JSON分析")
    print(f"{'='*80}")
    
    exec_file = project_root / "output" / "dssms_integration" / "dssms_20260111_232522" / "execution_results.json"
    
    if not exec_file.exists():
        print(f"[ERROR] 実行結果JSONが見つかりません: {exec_file}")
        return None
    
    with open(exec_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n実行結果:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    return data


def main():
    """メイン実行関数"""
    print("\n" + "="*80)
    print("戦略選択状況分析スクリプト実行開始")
    print("="*80)
    
    # 1. 包括レポート分析
    report_data = analyze_comprehensive_report()
    
    # 2. 実行結果JSON分析
    exec_data = analyze_execution_results()
    
    # 3. ログ検索
    log_entries = search_log_for_strategy_selection()
    
    # サマリー出力
    print(f"\n{'='*80}")
    print("分析サマリー")
    print(f"{'='*80}")
    
    if report_data:
        print(f"\n[包括レポート]")
        print(f"  戦略ランキング情報: 取得成功")
        
        # GC戦略のみ取引がある理由を推測
        ranking = report_data.get('strategy_effectiveness', {}).get('strategy_ranking', [])
        if ranking:
            top_strategy = ranking[0]
            print(f"\n  最高ランク戦略: {top_strategy['strategy']} (スコア: {top_strategy['score']:.4f})")
            print(f"  取引回数: {top_strategy['total_trades']}回")
    
    if exec_data:
        print(f"\n[実行結果]")
        print(f"  総取引: {exec_data.get('total_trades', 0)}回")
        print(f"  成功取引: {exec_data.get('successful_trades', 0)}回")
        print(f"  実行率: {exec_data.get('execution_rate', 0.0):.1%}")
    
    if log_entries:
        print(f"\n[ログ検索]")
        print(f"  戦略選択ログ: {len(log_entries)}件見つかりました")
    
    print(f"\n{'='*80}")
    print("調査完了")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
