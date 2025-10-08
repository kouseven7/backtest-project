"""
DSSMS切替数激減問題 - Task 2.1: IntelligentSwitchManager統合問題診断
"""
import logging
import json
from datetime import datetime
from pathlib import Path

def diagnose_intelligent_switch_manager():
    """Task 2.1: IntelligentSwitchManager統合問題の詳細診断"""
    print("[SEARCH] Task 2.1: IntelligentSwitchManager統合問題診断開始")
    print("="*60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "switch_manager_status": {},
        "backtester_initialization": {},
        "decision_logic": {}
    }
    
    try:
        # DSSMSBacktesterのインポートと初期化
        from src.dssms.dssms_backtester import DSSMSBacktester
        print("[OK] DSSMSBacktesterのインポート成功")
        
        # バックテスター初期化
        backtester = DSSMSBacktester()
        print("[OK] バックテスター初期化完了")
        
        # switch_managerの詳細調査
        if hasattr(backtester, 'switch_manager'):
            switch_manager = backtester.switch_manager
            print(f"[CHART] switch_manager属性存在: {switch_manager is not None}")
            
            if switch_manager is not None:
                print(f"[CHART] switch_manager型: {type(switch_manager)}")
                print(f"[CHART] switch_managerメソッド数: {len([m for m in dir(switch_manager) if not m.startswith('_')])}")
                
                results["switch_manager_status"] = {
                    "exists": True,
                    "type": str(type(switch_manager)),
                    "methods": [m for m in dir(switch_manager) if not m.startswith('_')][:10]
                }
                
                # 主要メソッドの存在確認
                key_methods = ['evaluate_switch_decision', 'can_switch', 'should_switch']
                for method_name in key_methods:
                    exists = hasattr(switch_manager, method_name)
                    print(f"  [TOOL] {method_name}: {'[OK]' if exists else '[ERROR]'}")
                    
            else:
                print("[ERROR] switch_manager = None")
                results["switch_manager_status"] = {
                    "exists": False,
                    "reason": "initialized_as_none"
                }
        else:
            print("[ERROR] switch_manager属性なし")
            results["switch_manager_status"] = {
                "exists": False,
                "reason": "attribute_missing"
            }
        
        # 切替決定ロジックの調査
        print("\n[TARGET] 切替決定ロジックの詳細調査:")
        
        # _evaluate_switch_decisionメソッドの詳細
        if hasattr(backtester, '_evaluate_switch_decision'):
            print("[OK] _evaluate_switch_decision メソッド存在")
            
            # メソッドのコードを一部取得（可能な場合）
            import inspect
            try:
                source_lines = inspect.getsourcelines(backtester._evaluate_switch_decision)
                print(f"📄 メソッド行数: {len(source_lines[0])}")
                
                # 最初の数行を表示
                print("📝 メソッド冒頭（最初10行）:")
                for i, line in enumerate(source_lines[0][:10]):
                    print(f"  {source_lines[1] + i}: {line.rstrip()}")
                    
            except Exception as e:
                print(f"[WARNING] ソースコード取得失敗: {e}")
        else:
            print("[ERROR] _evaluate_switch_decision メソッドなし")
            
        # 実際の切替判定テスト
        print("\n[TEST] 実際の切替判定テスト:")
        test_current_symbol = '9984'
        test_candidate_symbol = '6758'
        test_current_score = 0.75
        test_candidate_score = 0.85
        
        try:
            # 切替判定の実行テスト
            decision = backtester._evaluate_switch_decision(
                current_symbol=test_current_symbol,
                candidate_symbol=test_candidate_symbol, 
                current_score=test_current_score,
                candidate_score=test_candidate_score
            )
            print(f"[OK] 切替判定テスト成功: {decision}")
            results["decision_logic"]["test_result"] = str(decision)
            
        except Exception as e:
            print(f"[ERROR] 切替判定テスト失敗: {e}")
            results["decision_logic"]["test_error"] = str(e)
            
        # 設定値の確認
        print("\n[LIST] 切替関連設定値:")
        config_keys = [
            'min_holding_period_hours',
            'switch_cost_rate', 
            'enable_score_noise',
            'enable_switching_probability'
        ]
        
        for key in config_keys:
            if hasattr(backtester, key):
                value = getattr(backtester, key)
                print(f"  {key}: {value}")
            elif hasattr(backtester, 'config') and key in backtester.config:
                value = backtester.config[key]
                print(f"  {key}: {value} (from config)")
            else:
                print(f"  {key}: [ERROR] 見つからない")
                
    except Exception as e:
        print(f"[ERROR] 診断エラー: {e}")
        results["error"] = str(e)
        
    return results

def diagnose_switching_frequency_impact():
    """切替頻度に影響する要因の詳細分析"""
    print("\n[SEARCH] 切替頻度影響要因の分析")
    print("="*40)
    
    try:
        from src.dssms.dssms_backtester import DSSMSBacktester
        backtester = DSSMSBacktester()
        
        # 決定論的モードの影響調査
        print("[CHART] 決定論的モード設定:")
        deterministic_settings = [
            'enable_score_noise',
            'enable_switching_probability', 
            'use_fixed_execution',
            'random_seed'
        ]
        
        for setting in deterministic_settings:
            if hasattr(backtester, setting):
                value = getattr(backtester, setting)
                print(f"  {setting}: {value}")
                
                # 推奨設定との比較
                if setting == 'enable_score_noise' and not value:
                    print("    [WARNING] 推奨: Trueでランダム性を導入")
                elif setting == 'enable_switching_probability' and not value:
                    print("    [WARNING] 推奨: Trueで確率的切替判定")
                    
        # 切替制約条件の確認
        print("\n[LIST] 切替制約条件:")
        constraints = {
            'min_holding_period_hours': 24,  # 現在の設定
            'switch_cost_rate': 0.001
        }
        
        for constraint, current_value in constraints.items():
            if hasattr(backtester, constraint):
                actual_value = getattr(backtester, constraint)
                print(f"  {constraint}: {actual_value}")
                
                if constraint == 'min_holding_period_hours' and actual_value >= 24:
                    print("    [WARNING] 24時間制約が切替を抑制している可能性")
                    print("    [LIST] 推奨: 12-18時間への短縮を検討")
                    
        return True
        
    except Exception as e:
        print(f"[ERROR] 分析エラー: {e}")
        return False

if __name__ == "__main__":
    print("[ROCKET] DSSMS切替数激減問題 - Task 2.1診断開始")
    print("="*80)
    
    # Task 2.1の実行
    switch_manager_results = diagnose_intelligent_switch_manager()
    
    # 切替頻度影響要因の分析
    frequency_results = diagnose_switching_frequency_impact()
    
    # 結果の保存
    output_file = "task21_intelligent_switch_manager_diagnosis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(switch_manager_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n💾 診断結果を保存: {output_file}")
    print("\n[LIST] Task 2.1 診断完了")
    print("="*80)
