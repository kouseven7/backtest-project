"""
DSSMS切替数激減問題 - Task 2.2: パラメータ調整による改善テスト
"""
import json
from datetime import datetime
import shutil

def backup_current_config():
    """現在の設定をバックアップ"""
    print("💾 現在の設定をバックアップ中...")
    
    config_files = [
        "config/dssms/intelligent_switch_config.json"
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config_file in config_files:
        try:
            backup_file = f"{config_file}.backup_{timestamp}"
            shutil.copy2(config_file, backup_file)
            print(f"[OK] バックアップ: {config_file} -> {backup_file}")
        except Exception as e:
            print(f"[ERROR] バックアップ失敗: {config_file} - {e}")

def apply_optimized_parameters():
    """切替数向上のための最適化パラメータを適用"""
    print("⚙️ 最適化パラメータを適用中...")
    
    config_file = "config/dssms/intelligent_switch_config.json"
    
    try:
        # 現在の設定を読み込み
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("[LIST] 調整前の設定:")
        print(f"  minimum_holding_period_hours: {config['switch_criteria']['minimum_holding_period_hours']}")
        
        # パラメータの最適化
        optimizations = {
            # 最小保有期間を24時間→12時間に短縮
            "minimum_holding_period_hours": 12,
            
            # スコア差分閾値を緩和（0.25→0.15）
            "score_difference_threshold": 0.15,
            
            # 信頼度閾値を緩和（0.75→0.65）
            "confidence_threshold": 0.65
        }
        
        print("[LIST] 最適化パラメータ:")
        for key, value in optimizations.items():
            old_value = config['switch_criteria'][key]
            config['switch_criteria'][key] = value
            print(f"  {key}: {old_value} -> {value}")
            
        # 戦略別保有期間も短縮
        strategy_optimizations = {
            "Opening_Gap": 2,      # 4 -> 2
            "VWAP_Breakout": 6,    # 12 -> 6
            "Breakout": 6,         # 12 -> 6
            "VWAP_Bounce": 3,      # 6 -> 3
            "Momentum_Investing": 24,  # 48 -> 24
            "Contrarian": 4,       # 8 -> 4
            "GC_Strategy": 12      # 24 -> 12
        }
        
        print("[LIST] 戦略別保有期間最適化:")
        for strategy, new_hours in strategy_optimizations.items():
            old_hours = config['switch_criteria']['strategy_specific_holding_periods'][strategy]
            config['switch_criteria']['strategy_specific_holding_periods'][strategy] = new_hours
            print(f"  {strategy}: {old_hours}h -> {new_hours}h")
        
        # 設定を保存
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        print("[OK] 最適化パラメータ適用完了")
        return True
        
    except Exception as e:
        print(f"[ERROR] パラメータ適用失敗: {e}")
        return False

def test_optimized_configuration():
    """最適化設定での切替数テスト"""
    print("\n[TEST] 最適化設定での切替数テスト実行")
    print("="*50)
    
    try:
        from src.dssms.dssms_backtester import DSSMSBacktester
        from datetime import datetime, timedelta
        
        # テスト設定
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)  # 10日間テスト
        
        print(f"📅 テスト期間: {start_date.date()} - {end_date.date()}")
        
        # バックテスター初期化
        backtester = DSSMSBacktester()
        print("[OK] 最適化設定でのバックテスター初期化完了")
        
        # 切替設定確認
        print("[LIST] 適用された切替設定:")
        print(f"  min_holding_period_hours: {getattr(backtester, 'min_holding_period_hours', 'N/A')}")
        print(f"  switch_cost_rate: {getattr(backtester, 'switch_cost_rate', 'N/A')}")
        
        # シミュレーション実行
        print("\n[ROCKET] 最適化シミュレーション実行中...")
        results = backtester.backtest_dssms(start_date, end_date)
        
        # 切替履歴の確認
        switch_history = getattr(backtester, 'switch_history', [])
        switch_count = len(switch_history)
        
        print(f"\n[CHART] 最適化後の結果:")
        print(f"  切替回数: {switch_count}")
        print(f"  期間: {(end_date - start_date).days}日")
        print(f"  平均切替頻度: {switch_count / (end_date - start_date).days:.2f}回/日")
        
        # 切替履歴詳細（最初の10件）
        print(f"\n[LIST] 切替履歴（最初の{min(10, switch_count)}件）:")
        for i, switch in enumerate(switch_history[:10]):
            if hasattr(switch, 'timestamp') and hasattr(switch, 'from_symbol') and hasattr(switch, 'to_symbol'):
                print(f"  {i+1}: {switch.from_symbol} -> {switch.to_symbol} ({switch.timestamp})")
            else:
                print(f"  {i+1}: {switch}")
                
        # パフォーマンス確認
        if hasattr(backtester, 'performance_history'):
            final_value = backtester.performance_history[-1] if backtester.performance_history else 0
            print(f"\n[MONEY] 最終ポートフォリオ価値: {final_value:,.0f}円")
            
        return {
            "switch_count": switch_count,
            "test_period_days": (end_date - start_date).days,
            "average_switches_per_day": switch_count / (end_date - start_date).days,
            "final_portfolio_value": final_value if 'final_value' in locals() else None
        }
        
    except Exception as e:
        print(f"[ERROR] テスト実行失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("[ROCKET] DSSMS切替数激減問題 - Task 2.2: パラメータ最適化開始")
    print("="*80)
    
    # ステップ1: 現在設定のバックアップ
    backup_current_config()
    
    # ステップ2: 最適化パラメータ適用
    if apply_optimized_parameters():
        print("\n" + "="*50)
        
        # ステップ3: 最適化設定でのテスト
        test_results = test_optimized_configuration()
        
        if test_results:
            print(f"\n[TARGET] Task 2.2 完了:")
            print(f"  改善効果: 切替数 {test_results['switch_count']} 回")
            print(f"  期待値との比較: {test_results['switch_count']} vs 3 (従来)")
            
            # 結果を保存
            output_file = f"task22_parameter_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"  結果保存: {output_file}")
            
            if test_results['switch_count'] > 3:
                print("[OK] パラメータ最適化により切替数改善を確認")
            else:
                print("[WARNING] さらなる調整が必要")
        else:
            print("[ERROR] テスト実行に失敗しました")
    else:
        print("[ERROR] パラメータ適用に失敗しました")
        
    print("\n" + "="*80)
