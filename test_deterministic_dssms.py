import sys
import os
sys.path.append('src')
import pandas as pd
from datetime import datetime

def test_deterministic_dssms():
    """決定論的DSSMSテスト"""
    print("=" * 80)
    print("決定論的DSSMS テスト")
    print("=" * 80)
    
    try:
        from dssms.dssms_backtester import DSSMSBacktester
        
        print("🧪 同一条件での3回実行テスト...")
        
        # 短期間でのテスト
        start_date = datetime(2023, 11, 1)
        end_date = datetime(2023, 11, 15)
        symbols = ['7203', '6758']
        
        test_params = {
            'start_date': start_date,
            'end_date': end_date,
            'symbol_universe': symbols
        }
        
        results = []
        
        for run in range(3):
            print(f"   実行 {run + 1}/3...")
            
            backtester = DSSMSBacktester()
            
            try:
                result = backtester.simulate_dynamic_selection(**test_params)
                
                if isinstance(result, dict):
                    results.append({
                        'run': run + 1,
                        'final_value': result.get('final_portfolio_value', 0),
                        'total_return': result.get('total_return', 0),
                        'switches': result.get('total_switches', 0),
                        'performance_metrics': result.get('performance_metrics', {})
                    })
                    print(f"     完了: 最終価値={result.get('final_portfolio_value', 0):,.0f}")
                
            except Exception as e:
                print(f"     ❌ 実行{run + 1}エラー: {e}")
        
        if len(results) >= 2:
            print(f"\n📊 {len(results)}回実行の一貫性チェック:")
            
            # 結果比較
            for i in range(1, len(results)):
                result1 = results[0]
                result2 = results[i]
                
                print(f"\n   実行1 vs 実行{i+1}:")
                
                # 主要指標の比較
                for key in ['final_value', 'total_return', 'switches']:
                    if key in result1 and key in result2:
                        val1 = result1[key]
                        val2 = result2[key]
                        
                        print(f"     {key}: {val1} vs {val2}")
                        
                        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                            if abs(val1 - val2) < 0.01:  # 許容誤差
                                print(f"       ✅ 一致 (差異: {abs(val1 - val2):.6f})")
                            else:
                                print(f"       ⚠️ 不一致 (差異: {abs(val1 - val2):.6f})")
                        else:
                            if val1 == val2:
                                print(f"       ✅ 一致")
                            else:
                                print(f"       ⚠️ 不一致")
            
            # パフォーマンス指標の詳細比較
            print(f"\n📈 パフォーマンス指標詳細:")
            for i, result in enumerate(results):
                metrics = result.get('performance_metrics', {})
                print(f"   実行{i+1}: {metrics}")
        else:
            print("❌ 有効な結果が2つ未満のため比較できません")
    
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

def test_config_loading():
    """設定ファイル読み込みテスト"""
    print("\n" + "=" * 80)
    print("設定ファイル読み込みテスト")
    print("=" * 80)
    
    import json
    from pathlib import Path
    
    config_files = [
        'config/dssms/scoring_engine_config.json',
        'config/dssms/dssms_backtester_config.json'
    ]
    
    for config_file in config_files:
        print(f"\n📁 {config_file}:")
        
        if Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 決定論的設定の確認
                randomness = config.get('randomness_control', {})
                execution = config.get('execution_mode', {})
                
                if randomness:
                    print(f"   ランダム制御:")
                    for key, value in randomness.items():
                        print(f"     {key}: {value}")
                
                if execution:
                    print(f"   実行モード:")
                    for key, value in execution.items():
                        print(f"     {key}: {value}")
                
                print("   ✅ 読み込み成功")
                
            except Exception as e:
                print(f"   ❌ 読み込みエラー: {e}")
        else:
            print("   ❌ ファイルが存在しません")

if __name__ == "__main__":
    test_config_loading()
    test_deterministic_dssms()
