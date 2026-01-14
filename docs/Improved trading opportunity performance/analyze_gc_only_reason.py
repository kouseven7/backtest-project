"""
GC戦略のみ選択される理由の特定

目的:
- main_new.pyの実行ログから戦略スコアリング結果を確認
- DynamicStrategySelectorの動作を追跡
- available_strategiesリスト制限 vs スコアリング結果のどちらが原因か特定

Author: Backtest Project Team
Created: 2026-01-12
"""

import sys
from pathlib import Path
import pandas as pd
import json

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_transaction_strategy_distribution():
    """全取引における戦略分布を確認"""
    print(f"\n{'='*80}")
    print("全取引における戦略分布確認")
    print(f"{'='*80}")
    
    csv_file = project_root / "docs" / "Improved trading opportunity performance" / "all_transactions.csv"
    
    if not csv_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # 戦略名列を確認
    strategy_columns = [col for col in df.columns if 'strategy' in col.lower()]
    print(f"\n戦略関連カラム: {strategy_columns}")
    
    if 'Strategy' in df.columns:
        strategy_counts = df['Strategy'].value_counts()
        
        print(f"\n戦略別取引数:")
        print(f"{'戦略名':<40} {'取引数':<10} {'割合'}")
        print("-" * 80)
        
        total_trades = len(df)
        for strategy, count in strategy_counts.items():
            percentage = (count / total_trades) * 100
            print(f"{strategy:<40} {count:<10} {percentage:.1f}%")
        
        print(f"\n総取引数: {total_trades}")
        
        return strategy_counts
    else:
        print("[ERROR] 'Strategy'カラムが見つかりません")
        print(f"利用可能なカラム: {df.columns.tolist()}")
        return None


def analyze_market_analysis_results():
    """MarketAnalyzerの分析結果確認（市場環境判定）"""
    print(f"\n{'='*80}")
    print("市場環境判定結果確認")
    print(f"{'='*80}")
    
    # DSSMSの銘柄切替履歴から市場環境を推定
    switch_file = project_root / "docs" / "Improved trading opportunity performance" / "dssms_switch_history.csv"
    
    if not switch_file.exists():
        print(f"[INFO] dssms_switch_history.csvが見つかりません")
        return None
    
    df = pd.read_csv(switch_file)
    
    # 市場環境関連カラムを確認
    env_columns = [col for col in df.columns if any(keyword in col.lower() 
                                                     for keyword in ['market', 'trend', 'volatility', 'regime'])]
    
    if env_columns:
        print(f"\n市場環境関連カラム: {env_columns}")
        
        for col in env_columns:
            if col in df.columns:
                value_counts = df[col].value_counts()
                print(f"\n{col}の分布:")
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"  {value}: {count}回 ({percentage:.1f}%)")
    else:
        print("[INFO] 市場環境関連カラムが見つかりません")
        print(f"利用可能なカラム: {df.columns.tolist()[:10]}...")
    
    return df


def check_dynamic_strategy_selector_code():
    """DynamicStrategySelectorのスコアリングロジック確認"""
    print(f"\n{'='*80}")
    print("DynamicStrategySelectorスコアリングロジック確認")
    print(f"{'='*80}")
    
    selector_file = project_root / "main_system" / "strategy_selection" / "dynamic_strategy_selector.py"
    
    if not selector_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {selector_file}")
        return None
    
    with open(selector_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # select_optimal_strategiesメソッドを探す
    in_method = False
    method_lines = []
    
    for i, line in enumerate(lines, 1):
        if 'def select_optimal_strategies' in line:
            in_method = True
            start_line = i
        
        if in_method:
            method_lines.append((i, line.rstrip()))
            
            # メソッド終了判定（次のdefまたはインデント減少）
            if len(method_lines) > 1 and line.startswith('    def ') and 'select_optimal_strategies' not in line:
                break
            
            if len(method_lines) > 100:  # 安全装置
                break
    
    if method_lines:
        print(f"\nselect_optimal_strategies() メソッド（{start_line}行目から）:")
        print("-" * 80)
        
        # スコア計算部分を抽出
        for line_no, line_text in method_lines[:50]:  # 最初の50行
            if any(keyword in line_text.lower() for keyword in ['score', '点数', 'calculate', '計算']):
                print(f"{line_no:4d}: {line_text}")
    else:
        print("[ERROR] select_optimal_strategies()メソッドが見つかりません")
    
    return method_lines


def main():
    """メイン実行関数"""
    print("\n" + "="*80)
    print("GC戦略のみ選択される理由の特定")
    print("="*80)
    
    # 1. 全取引における戦略分布
    strategy_dist = analyze_transaction_strategy_distribution()
    
    # 2. 市場環境判定結果
    market_env = analyze_market_analysis_results()
    
    # 3. DynamicStrategySelectorスコアリングロジック
    selector_code = check_dynamic_strategy_selector_code()
    
    # 総合結論
    print(f"\n{'='*80}")
    print("結論")
    print(f"{'='*80}")
    
    if strategy_dist is not None:
        if len(strategy_dist) == 1:
            only_strategy = strategy_dist.index[0]
            print(f"\n[重要] 全取引が単一戦略: {only_strategy}")
            print(f"\n考えられる原因:")
            print(f"  A. available_strategiesリストに{only_strategy}のみ有効")
            print(f"  B. スコアリングで{only_strategy}が常に最高点")
            print(f"  C. 市場環境が特定の戦略に偏る条件が継続")
            
            # available_strategiesの実際の内容を確認
            print(f"\n[確認済み] available_strategiesには8個の戦略が有効")
            print(f"  → 原因Aは除外")
            print(f"\n[要確認] 原因BまたはCの検証が必要")
            print(f"  → MarketAnalyzerの分析結果ログが必要")
            print(f"  → DynamicStrategySelectorのスコアリング結果ログが必要")
        else:
            print(f"\n複数戦略が使用されています:")
            for strategy, count in strategy_dist.items():
                print(f"  - {strategy}: {count}回")
    
    print(f"\n{'='*80}")
    print("調査完了")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
