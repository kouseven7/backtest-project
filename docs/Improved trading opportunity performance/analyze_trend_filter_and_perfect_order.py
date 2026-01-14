"""
トレンドフィルターとパーフェクトオーダー状況分析

目的:
- 各戦略のトレンドフィルター設定確認
- DSSMSのパーフェクトオーダー判定確認
- GC戦略のみ選択される理由の特定

Author: Backtest Project Team
Created: 2026-01-12
"""

import sys
import os
from pathlib import Path
import re

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_strategy_trend_filters():
    """各戦略のトレンドフィルター設定を確認"""
    print(f"\n{'='*80}")
    print("戦略別トレンドフィルター設定確認")
    print(f"{'='*80}")
    
    strategies_dir = project_root / "strategies"
    strategy_files = list(strategies_dir.glob("*.py"))
    
    trend_filter_info = []
    
    for strategy_file in strategy_files:
        if strategy_file.name.startswith("__"):
            continue
        
        try:
            with open(strategy_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # trend_filter_enabledの設定を検索
            pattern = r'"trend_filter_enabled":\s*(True|False)'
            matches = re.findall(pattern, content)
            
            if matches:
                # デフォルト設定を取得（通常は最初の出現）
                default_setting = matches[0]
                
                # コメントも確認
                comment_pattern = r'"trend_filter_enabled":\s*(True|False)[^#\n]*(#.*)?'
                comment_matches = re.findall(comment_pattern, content)
                
                comment = ""
                if comment_matches:
                    _, potential_comment = comment_matches[0]
                    comment = potential_comment.strip() if potential_comment else ""
                
                trend_filter_info.append({
                    'file': strategy_file.name,
                    'enabled': default_setting == 'True',
                    'comment': comment
                })
        
        except Exception as e:
            print(f"[ERROR] {strategy_file.name} 読み込みエラー: {e}")
    
    # 結果表示
    print(f"\nトレンドフィルター設定:")
    print(f"{'戦略ファイル':<40} {'設定':<10} {'コメント'}")
    print("-" * 80)
    
    for info in sorted(trend_filter_info, key=lambda x: x['file']):
        status = "有効" if info['enabled'] else "無効"
        print(f"{info['file']:<40} {status:<10} {info['comment']}")
    
    # サマリー
    enabled_count = sum(1 for info in trend_filter_info if info['enabled'])
    disabled_count = len(trend_filter_info) - enabled_count
    
    print(f"\nサマリー:")
    print(f"  トレンドフィルター有効: {enabled_count}戦略")
    print(f"  トレンドフィルター無効: {disabled_count}戦略")
    
    return trend_filter_info


def analyze_available_strategies():
    """DynamicStrategySelectorの利用可能戦略リスト確認"""
    print(f"\n{'='*80}")
    print("DynamicStrategySelector 利用可能戦略リスト確認")
    print(f"{'='*80}")
    
    selector_file = project_root / "main_system" / "strategy_selection" / "dynamic_strategy_selector.py"
    
    if not selector_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {selector_file}")
        return None
    
    with open(selector_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # available_strategiesリストを抽出
    pattern = r'self\.available_strategies\s*=\s*\[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        strategies_text = match.group(1)
        
        # 戦略名を抽出（コメント除外）
        strategy_pattern = r"'([^']+)'"
        strategies = re.findall(strategy_pattern, strategies_text)
        
        # コメントアウトされた戦略も確認
        commented_pattern = r"#\s*'([^']+)'"
        commented_strategies = re.findall(commented_pattern, strategies_text)
        
        print(f"\n有効な戦略（{len(strategies)}個）:")
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy}")
        
        if commented_strategies:
            print(f"\nコメントアウトされた戦略（{len(commented_strategies)}個）:")
            for i, strategy in enumerate(commented_strategies, 1):
                print(f"  {i}. {strategy}")
        
        return {
            'active': strategies,
            'commented': commented_strategies
        }
    else:
        print("[ERROR] available_strategiesリストが見つかりません")
        return None


def analyze_perfect_order_usage():
    """DSSMSのパーフェクトオーダー使用状況確認"""
    print(f"\n{'='*80}")
    print("DSSMSパーフェクトオーダー使用状況確認")
    print(f"{'='*80}")
    
    dssms_main = project_root / "src" / "dssms" / "dssms_integrated_main.py"
    
    if not dssms_main.exists():
        print(f"[ERROR] ファイルが見つかりません: {dssms_main}")
        return None
    
    with open(dssms_main, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # パーフェクトオーダー関連のメソッド呼び出しを確認
    po_patterns = [
        (r'categorize_by_perfect_order_priority', 'パーフェクトオーダー優先度分類'),
        (r'perfect_order_detector', 'パーフェクトオーダー検出器'),
        (r'_get_enhanced_perfect_order_analysis', 'パーフェクトオーダー高度分析'),
    ]
    
    results = {}
    
    for pattern, description in po_patterns:
        matches = re.findall(pattern, content)
        count = len(matches)
        results[description] = count
        
        if count > 0:
            print(f"✓ {description}: {count}箇所で使用")
        else:
            print(f"✗ {description}: 使用なし")
    
    # HierarchicalRankingSystemの利用確認
    hierarchical_pattern = r'from.*hierarchical_ranking_system.*import'
    if re.search(hierarchical_pattern, content):
        print(f"\n✓ HierarchicalRankingSystemがインポートされています")
    else:
        print(f"\n✗ HierarchicalRankingSystemがインポートされていません")
    
    return results


def main():
    """メイン実行関数"""
    print("\n" + "="*80)
    print("トレンドフィルターとパーフェクトオーダー状況分析")
    print("="*80)
    
    # 1. 戦略別トレンドフィルター確認
    trend_info = analyze_strategy_trend_filters()
    
    # 2. DynamicStrategySelector利用可能戦略確認
    strategy_info = analyze_available_strategies()
    
    # 3. パーフェクトオーダー使用状況確認
    po_info = analyze_perfect_order_usage()
    
    # 総合結論
    print(f"\n{'='*80}")
    print("総合結論")
    print(f"{'='*80}")
    
    if trend_info:
        gc_filter = next((info for info in trend_info if 'gc_strategy' in info['file'].lower()), None)
        if gc_filter:
            print(f"\n1. GC戦略のトレンドフィルター: {'有効' if gc_filter['enabled'] else '無効'}")
            if not gc_filter['enabled']:
                print(f"   理由: {gc_filter['comment']}")
    
    if strategy_info:
        active_strategies = strategy_info['active']
        print(f"\n2. DynamicStrategySelector利用可能戦略数: {len(active_strategies)}個")
        
        if len(active_strategies) == 1 and active_strategies[0] == 'GCStrategy':
            print(f"   [重要] GCStrategyのみが有効です")
        elif 'GCStrategy' in active_strategies:
            print(f"   GCStrategyを含む複数戦略が有効")
    
    if po_info:
        total_usage = sum(po_info.values())
        if total_usage > 0:
            print(f"\n3. パーフェクトオーダー機能: 使用中（{total_usage}箇所）")
        else:
            print(f"\n3. パーフェクトオーダー機能: 未使用")
    
    print(f"\n{'='*80}")
    print("調査完了")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
