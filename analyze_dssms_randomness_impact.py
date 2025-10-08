import sys
import os
sys.path.append('src')
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_randomness_impact():
    """ランダム要素の影響度を詳細分析"""
    print("=" * 80)
    print("DSSMS ランダム要素影響度分析")
    print("=" * 80)
    
    # DSSMSの各コンポーネントでのランダム要素確認
    components_to_check = [
        'src/dssms/comprehensive_scoring_engine.py',
        'src/dssms/hierarchical_ranking_system.py', 
        'src/dssms/intelligent_switch_manager.py',
        'src/dssms/dssms_backtester.py',
        'src/dssms/market_condition_monitor.py'
    ]
    
    randomness_found = {}
    
    for component_file in components_to_check:
        if os.path.exists(component_file):
            with open(component_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # ランダム要素の検索パターン
            random_patterns = [
                'np.random.normal',
                'np.random.uniform', 
                'np.random.choice',
                'random.random',
                'random.choice',
                'random.uniform',
                'random.normal',
                'random.seed',
                'np.random.seed'
            ]
            
            found_patterns = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                for pattern in random_patterns:
                    if pattern in line and not line.strip().startswith('#'):
                        found_patterns.append({
                            'line_num': i + 1,
                            'pattern': pattern,
                            'code': line.strip(),
                            'context': 'scoring' if 'score' in line.lower() else
                                     'ranking' if 'rank' in line.lower() else
                                     'switch' if 'switch' in line.lower() else 'other'
                        })
            
            if found_patterns:
                randomness_found[component_file] = found_patterns
    
    # 結果の表示
    print("🎲 発見されたランダム要素:")
    total_random_elements = 0
    
    for component, patterns in randomness_found.items():
        print(f"\n📄 {component}:")
        for pattern_info in patterns:
            print(f"   行{pattern_info['line_num']:4d}: {pattern_info['pattern']} - {pattern_info['context']}")
            print(f"          {pattern_info['code']}")
            total_random_elements += 1
    
    print(f"\n[CHART] 総ランダム要素数: {total_random_elements}")
    
    # 影響度分析
    print("\n[TARGET] 影響度分析:")
    
    # コンテキスト別の影響度評価
    contexts = {}
    for component, patterns in randomness_found.items():
        for pattern_info in patterns:
            ctx = pattern_info['context']
            if ctx not in contexts:
                contexts[ctx] = []
            contexts[ctx].append({
                'component': component,
                'pattern': pattern_info['pattern'],
                'code': pattern_info['code']
            })
    
    # 影響度ランキング
    impact_levels = {
        'scoring': {'level': 'HIGH', 'reason': 'スコア計算は銘柄選択の根幹'},
        'ranking': {'level': 'HIGH', 'reason': 'ランキングは切替判定の基準'},
        'switch': {'level': 'MEDIUM', 'reason': '切替判定は頻度に影響'},
        'other': {'level': 'LOW', 'reason': '補助的な処理'}
    }
    
    for ctx, items in contexts.items():
        impact = impact_levels.get(ctx, {'level': 'UNKNOWN', 'reason': '不明'})
        print(f"   {ctx.upper()}: {impact['level']} - {impact['reason']}")
        print(f"     発見数: {len(items)}")
    
    return randomness_found, contexts

def measure_variance_by_runs():
    """複数回実行による分散測定"""
    print("\n" + "=" * 80)
    print("分散測定テスト（5回実行）")
    print("=" * 80)
    
    try:
        from dssms.dssms_backtester import DSSMSBacktester
        
        results = []
        num_runs = 5
        
        print(f"[TEST] {num_runs}回実行テスト開始...")
        
        for run in range(num_runs):
            print(f"   実行 {run + 1}/{num_runs}...")
            
            backtester = DSSMSBacktester()
            
            # 短期間でのテスト
            start_date = datetime(2023, 11, 1)
            end_date = datetime(2023, 11, 30)
            symbols = ['7203', '6758', '9984']
            
            try:
                result = backtester.simulate_dynamic_selection(
                    start_date=start_date,
                    end_date=end_date,
                    symbol_universe=symbols
                )
                
                if isinstance(result, dict):
                    results.append({
                        'run': run + 1,
                        'final_value': result.get('final_portfolio_value', 0),
                        'total_return': result.get('total_return', 0),
                        'switches': result.get('total_switches', 0),
                        'success_rate': result.get('switch_success_rate', 0)
                    })
                
            except Exception as e:
                print(f"     [ERROR] 実行{run + 1}エラー: {e}")
        
        if results:
            df = pd.DataFrame(results)
            
            print(f"\n[CHART] {len(results)}回実行の分散分析:")
            
            metrics = ['final_value', 'total_return', 'switches', 'success_rate']
            variance_analysis = {}
            
            for metric in metrics:
                if metric in df.columns:
                    values = df[metric]
                    mean_val = values.mean()
                    std_val = values.std()
                    cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                    min_val = values.min()
                    max_val = values.max()
                    range_val = max_val - min_val
                    
                    variance_analysis[metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv,
                        'min': min_val,
                        'max': max_val,
                        'range': range_val
                    }
                    
                    print(f"   {metric}:")
                    print(f"     平均: {mean_val:.2f}")
                    print(f"     標準偏差: {std_val:.2f}")
                    print(f"     変動係数: {cv:.2f}%")
                    print(f"     範囲: {min_val:.2f} - {max_val:.2f} (幅: {range_val:.2f})")
                    
                    # 許容範囲判定
                    if metric == 'total_return' and cv > 30:
                        print(f"     [WARNING] 変動係数{cv:.1f}%は過大（推奨: <20%）")
                    elif metric == 'switches' and cv > 15:
                        print(f"     [WARNING] 切替回数の変動{cv:.1f}%は過大（推奨: <10%）")
                    else:
                        print(f"     [OK] 変動は許容範囲内")
            
            return variance_analysis
        else:
            print("[ERROR] 有効な結果が得られませんでした")
            return None
            
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    randomness_found, contexts = analyze_randomness_impact()
    variance_analysis = measure_variance_by_runs()