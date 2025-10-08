import sys
import os
sys.path.append('src')
from datetime import datetime

def run_full_dssms_backtest():
    """フル期間DSSMSバックテストで一貫性確認"""
    print("=" * 80)
    print("DSSMSフル期間バックテスト一貫性確認")
    print("=" * 80)
    
    try:
        from dssms.dssms_backtester import DSSMSBacktester
        
        # より長期間でのテスト
        start_date = datetime(2023, 10, 1)
        end_date = datetime(2024, 1, 31)
        symbols = ['7203', '6758', '9984']
        
        print(f"📅 期間: {start_date.date()} ～ {end_date.date()}")
        print(f"🏢 対象銘柄: {symbols}")
        
        results = []
        
        print("\n[TEST] 2回実行で一貫性確認:")
        
        for run in range(2):
            print(f"\n   実行 {run + 1}/2...")
            
            backtester = DSSMSBacktester()
            
            try:
                result = backtester.simulate_dynamic_selection(
                    start_date=start_date,
                    end_date=end_date,
                    symbol_universe=symbols
                )
                
                if isinstance(result, dict):
                    final_value = result.get('final_portfolio_value', 0)
                    total_return = result.get('total_return', 0)
                    switches = result.get('total_switches', 0)
                    performance_metrics = result.get('performance_metrics', {})
                    
                    results.append({
                        'run': run + 1,
                        'final_value': final_value,
                        'total_return': total_return,
                        'switches': switches,
                        'metrics': performance_metrics
                    })
                    
                    print(f"     最終価値: {final_value:,.0f}円")
                    print(f"     総リターン: {total_return:.4f} ({total_return*100:.2f}%)")
                    print(f"     スイッチ回数: {switches}")
                    
                    # vs_戦略の結果
                    vs_fixed = performance_metrics.get('vs_fixed_symbol_return', {})
                    if vs_fixed:
                        print(f"     vs_固定戦略:")
                        for symbol, value in vs_fixed.items():
                            print(f"       vs_{symbol}: {value:.2f}%")
                
            except Exception as e:
                print(f"     [ERROR] 実行{run + 1}エラー: {e}")
                import traceback
                traceback.print_exc()
        
        # 結果比較
        if len(results) == 2:
            print(f"\n[CHART] 結果比較:")
            
            result1, result2 = results[0], results[1]
            
            # 主要指標比較
            comparisons = [
                ('最終価値', 'final_value'),
                ('総リターン', 'total_return'),
                ('スイッチ回数', 'switches')
            ]
            
            all_consistent = True
            
            for name, key in comparisons:
                val1 = result1[key]
                val2 = result2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = abs(val1 - val2)
                    diff_pct = (diff / abs(val1) * 100) if val1 != 0 else 0
                    
                    print(f"   {name}:")
                    print(f"     実行1: {val1}")
                    print(f"     実行2: {val2}")
                    print(f"     差異: {diff:.6f} ({diff_pct:.4f}%)")
                    
                    if diff < 0.01:  # 許容誤差
                        print(f"     [OK] 一貫性: 良好")
                    else:
                        print(f"     [WARNING] 一貫性: 差異あり")
                        all_consistent = False
                else:
                    if val1 == val2:
                        print(f"   {name}: [OK] 完全一致")
                    else:
                        print(f"   {name}: [WARNING] 不一致 ({val1} ≠ {val2})")
                        all_consistent = False
            
            # vs_戦略比較の一貫性
            metrics1 = result1.get('metrics', {})
            metrics2 = result2.get('metrics', {})
            
            vs_fixed1 = metrics1.get('vs_fixed_symbol_return', {})
            vs_fixed2 = metrics2.get('vs_fixed_symbol_return', {})
            
            if vs_fixed1 and vs_fixed2:
                print(f"\n   vs_戦略比較一貫性:")
                for symbol in vs_fixed1.keys():
                    if symbol in vs_fixed2:
                        val1 = vs_fixed1[symbol]
                        val2 = vs_fixed2[symbol]
                        diff = abs(val1 - val2)
                        
                        print(f"     vs_{symbol}: {val1:.2f}% vs {val2:.2f}% (差異: {diff:.4f}%)")
                        
                        if diff < 0.01:
                            print(f"       [OK] 一貫")
                        else:
                            print(f"       [WARNING] 差異あり")
                            all_consistent = False
            
            # 総合評価
            print(f"\n[TARGET] 総合評価:")
            if all_consistent:
                print("   [OK] 完全な決定論的動作を確認")
                print("   [OK] 同じ条件で同じ結果を再現")
                print("   [OK] ランダム要素の制御成功")
            else:
                print("   [WARNING] 部分的な一貫性のみ")
                print("   [WARNING] 追加の修正が必要")
        
        else:
            print("[ERROR] 2つの有効な結果が得られませんでした")
    
    except Exception as e:
        print(f"[ERROR] フルバックテストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_dssms_backtest()
