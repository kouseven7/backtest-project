import sys
import os
sys.path.append('src')
import pandas as pd
from datetime import datetime

def analyze_dssms_variance():
    """DSSMSの結果変動を分析する"""
    print("=" * 80)
    print("DSSMS結果変動分析")
    print("=" * 80)
    
    # 複数回の実行結果を比較
    results = [
        {
            'time': '15:08:27',
            'final_value': 4057875.90,
            'total_return': 45.49,
            'switches': 107,
            'success_rate': 69.16,
            'vs_7203': 71.96,
            'vs_9984': 43.42,
            'vs_6758': 19.29
        },
        {
            'time': '15:13:20', 
            'final_value': 2392037.64,
            'total_return': 26.52,
            'switches': 116,
            'success_rate': 58.62,
            'vs_7203': 32.07,
            'vs_9984': -40.63,
            'vs_6758': 27.66
        },
        {
            'time': '15:14:55',
            'final_value': 3408192.68,
            'total_return': 17.91,
            'switches': 112,
            'success_rate': 66.07,
            'vs_7203': 2.03,
            'vs_9984': -21.07,
            'vs_6758': 11.82
        },
        {
            'time': '15:22:32',
            'final_value': 1863305.03,
            'total_return': 22.42,
            'switches': 112,
            'success_rate': 54.46,
            'vs_7203': -15.99,
            'vs_9984': 13.42,
            'vs_6758': -15.13
        }
    ]
    
    df = pd.DataFrame(results)
    
    print("[CHART] 結果変動の統計:")
    for col in ['final_value', 'total_return', 'switches', 'success_rate']:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            cv = std_val / mean_val * 100 if mean_val != 0 else 0
            print(f"   {col}:")
            print(f"     平均: {mean_val:.2f}")
            print(f"     標準偏差: {std_val:.2f}")
            print(f"     変動係数: {cv:.2f}%")
    
    print("\n[SEARCH] 変動の異常度判定:")
    
    # 総リターンの変動 (45.49% → 22.42% は異常に大きい)
    return_range = df['total_return'].max() - df['total_return'].min()
    print(f"   総リターン変動幅: {return_range:.2f}% (45.49% → 22.42%)")
    if return_range > 20:
        print("   [WARNING] 異常: 同じ期間・同じ設定での変動幅が20%以上は異常")
    
    # vs_戦略の変動 (71.96% → -15.99% は明らかに異常)
    vs_7203_range = df['vs_7203'].max() - df['vs_7203'].min()
    print(f"   vs_7203変動幅: {vs_7203_range:.2f}% (71.96% → -15.99%)")
    if vs_7203_range > 50:
        print("   [WARNING] 異常: vs_戦略の変動幅が50%以上は計算エラーの可能性")
    
    print("\n[TARGET] 予想される原因:")
    print("   1. ランダム要素の混入（乱数シードが固定されていない）")
    print("   2. データ取得タイミングの違い（リアルタイムデータ混入）")
    print("   3. 日付・時刻計算の不整合")
    print("   4. ポートフォリオ価値計算の論理エラー")
    print("   5. vs_戦略比較計算の基準データ不整合")
    
    return df

def check_dssms_determinism():
    """DSSMSの決定性を確認する"""
    print("\n" + "=" * 80)
    print("DSSMS決定性確認")
    print("=" * 80)
    
    try:
        from dssms.dssms_backtester import DSSMSBacktester
        
        # 同じ条件で2回実行
        print("[TEST] 同一条件での2回実行テスト:")
        
        backtester1 = DSSMSBacktester()
        backtester2 = DSSMSBacktester()
        
        # DSSMSBacktesterの実際のメソッド使用
        from datetime import datetime
        start_date = datetime(2023, 12, 1)
        end_date = datetime(2023, 12, 10)
        symbols = ['7203', '6758']
        
        test_params = {
            'start_date': start_date,
            'end_date': end_date,
            'symbol_universe': symbols
        }
        
        print(f"   テスト条件: {test_params}")
        
        # 1回目実行
        print("\n   1回目実行中...")
        result1 = backtester1.simulate_dynamic_selection(**test_params)
        
        # 2回目実行
        print("   2回目実行中...")
        result2 = backtester2.simulate_dynamic_selection(**test_params)
        
        # 結果比較
        print("\n[CHART] 結果比較:")
        
        if isinstance(result1, dict) and isinstance(result2, dict):
            for key in ['final_portfolio_value', 'total_return', 'total_switches']:
                if key in result1 and key in result2:
                    val1 = result1[key]
                    val2 = result2[key] 
                    diff = abs(val1 - val2) if isinstance(val1, (int, float)) else "型エラー"
                    print(f"   {key}:")
                    print(f"     1回目: {val1}")
                    print(f"     2回目: {val2}")
                    print(f"     差異: {diff}")
                    
                    if isinstance(diff, (int, float)) and diff > 0.01:
                        print(f"     [WARNING] 非決定的: 同じ条件で異なる結果")
                    else:
                        print(f"     [OK] 決定的: 同じ結果")
        
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    df = analyze_dssms_variance()
    check_dssms_determinism()
