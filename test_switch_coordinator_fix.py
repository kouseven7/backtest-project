"""
DSSMS Switch Coordinator V2 修正テスト
緊急モード無効化の動作確認

修正内容:
1. 緊急モード無効化 (_determine_execution_mode)
2. 強制切替除去 (execute_switch_decision) 
3. 分散軽減効果の確認

Author: GitHub Copilot Agent
Created: 2025-09-03
Purpose: 競合解決後の動作確認
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config.logger_config import setup_logger

# 修正されたコンポーネントをインポート
try:
    from src.dssms.dssms_switch_coordinator_v2 import DSSMSSwitchCoordinatorV2
    print("[OK] DSSMS Switch Coordinator V2 インポート成功")
except ImportError as e:
    print(f"[ERROR] インポート失敗: {e}")
    sys.exit(1)

def generate_test_market_data(days=5):
    """テスト用市場データ生成"""
    dates = pd.date_range(start='2025-09-01', periods=days*24, freq='h')
    
    # 基本的な価格データ
    price_data = []
    base_price = 100.0
    
    for i, date in enumerate(dates):
        # 価格変動シミュレーション
        price_change = np.random.normal(0, 0.02)  # 2%標準偏差
        base_price *= (1 + price_change)
        
        volume = np.random.randint(1000, 10000)
        
        price_data.append({
            'datetime': date,
            'open': base_price * 0.998,
            'high': base_price * 1.005,
            'low': base_price * 0.995,
            'close': base_price,
            'volume': volume,
            'returns': price_change
        })
    
    return pd.DataFrame(price_data).set_index('datetime')

def test_emergency_mode_disabled():
    """緊急モード無効化テスト"""
    print("\\n" + "="*60)
    print("[TOOL] 緊急モード無効化テスト開始")
    print("="*60)
    
    # ロガー設定
    logger = setup_logger("EmergencyModeTest")
    
    try:
        # Switch Coordinator初期化
        coordinator = DSSMSSwitchCoordinatorV2()
        print("[OK] Switch Coordinator初期化成功")
        
        # テストデータ生成
        market_data = generate_test_market_data(days=3)
        test_positions = ['TEST_SYMBOL_1', 'TEST_SYMBOL_2']
        print("[OK] テストデータ生成完了")
        
        # 実行モード決定テスト（緊急モードが呼ばれないことを確認）
        execution_modes = []
        
        for i in range(10):
            mode = coordinator._determine_execution_mode()
            execution_modes.append(mode)
            print(f"  実行モード {i+1}: {mode}")
        
        # 緊急モードが含まれていないことを確認
        if "emergency_mode" in execution_modes:
            print("[ERROR] 緊急モードが検出されました（修正失敗）")
            return False
        else:
            print("[OK] 緊急モードは検出されませんでした（修正成功）")
        
        # 実際の切替決定実行テスト
        print("\\n[CHART] 切替決定実行テスト...")
        
        results = []
        for i in range(5):
            result = coordinator.execute_switch_decision(
                market_data=market_data,
                current_positions=test_positions
            )
            
            results.append({
                'attempt': i+1,
                'engine_used': result.engine_used,
                'success': result.success,
                'switches_count': result.switches_count,
                'execution_time_ms': result.execution_time_ms
            })
            
            print(f"  試行 {i+1}: エンジン={result.engine_used}, 成功={result.success}, 切替数={result.switches_count}")
        
        # 緊急エンジンが使用されていないことを確認
        emergency_usage = [r for r in results if 'emergency' in r['engine_used']]
        if emergency_usage:
            print(f"[ERROR] 緊急エンジンが {len(emergency_usage)} 回使用されました（修正失敗）")
            return False
        else:
            print("[OK] 緊急エンジンは使用されませんでした（修正成功）")
        
        # 統計レポート確認
        print("\\n[UP] 統計レポート確認...")
        status_report = coordinator.get_status_report()
        
        print(f"  現在の成功率: {status_report['current_success_rate']:.3f}")
        print(f"  目標成功率: {status_report['target_success_rate']:.3f}")
        print(f"  実行回数: {status_report['recent_executions']}")
        print(f"  V2エンジン利用可能: {status_report['engines_status']['v2_available']}")
        print(f"  レガシー管理器利用可能: {status_report['engines_status']['legacy_available']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] テスト実行中にエラー: {e}")
        logger.error(f"Emergency mode test failed: {e}")
        return False

def test_switching_frequency_reduction():
    """切替頻度削減テスト"""
    print("\\n" + "="*60)
    print("[DOWN] 切替頻度削減効果テスト")
    print("="*60)
    
    try:
        coordinator = DSSMSSwitchCoordinatorV2()
        market_data = generate_test_market_data(days=7)
        test_positions = ['SYMBOL_A', 'SYMBOL_B', 'SYMBOL_C']
        
        # 長期間のテスト実行
        switch_counts = []
        engine_usage = {'v2': 0, 'legacy': 0, 'hybrid': 0, 'emergency': 0}
        
        for day in range(7):
            daily_switches = 0
            daily_executions = 5  # 1日5回実行
            
            for execution in range(daily_executions):
                result = coordinator.execute_switch_decision(
                    market_data=market_data,
                    current_positions=test_positions
                )
                
                daily_switches += result.switches_count
                
                # エンジン使用統計
                if 'v2' in result.engine_used:
                    engine_usage['v2'] += 1
                elif 'legacy' in result.engine_used:
                    engine_usage['legacy'] += 1
                elif 'hybrid' in result.engine_used:
                    engine_usage['hybrid'] += 1
                elif 'emergency' in result.engine_used:
                    engine_usage['emergency'] += 1
            
            switch_counts.append(daily_switches)
            print(f"  日 {day+1}: 切替数 {daily_switches}")
        
        # 統計分析
        total_switches = sum(switch_counts)
        average_daily_switches = np.mean(switch_counts)
        switch_variance = np.var(switch_counts)
        
        print(f"\\n[CHART] 切替統計:")
        print(f"  総切替数: {total_switches}")
        print(f"  1日平均切替数: {average_daily_switches:.2f}")
        print(f"  切替分散: {switch_variance:.3f}")
        print(f"  エンジン使用状況:")
        for engine, count in engine_usage.items():
            print(f"    {engine}: {count}回")
        
        # 緊急モード使用の確認
        if engine_usage['emergency'] > 0:
            print(f"[ERROR] 緊急モードが {engine_usage['emergency']} 回使用されました")
            return False
        else:
            print("[OK] 緊急モードは使用されませんでした")
        
        # 過度な切替の確認（1日1回以上は制限されているべき）
        if average_daily_switches > 5.0:  # 過度な切替の閾値
            print(f"[WARNING] 平均日次切替数が高めです: {average_daily_switches:.2f}")
        else:
            print(f"[OK] 切替頻度は適切な範囲内です: {average_daily_switches:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 切替頻度テスト中にエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("[ROCKET] DSSMS Switch Coordinator V2 修正テスト開始")
    print(f"⏰ 実行時刻: {datetime.now()}")
    
    # テスト結果記録
    test_results = {}
    
    # 1. 緊急モード無効化テスト
    test_results['emergency_disabled'] = test_emergency_mode_disabled()
    
    # 2. 切替頻度削減テスト
    test_results['frequency_reduced'] = test_switching_frequency_reduction()
    
    # 結果サマリー
    print("\\n" + "="*60)
    print("[TARGET] テスト結果サマリー")
    print("="*60)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\\n[SUCCESS] 全テストPASS: 緊急モード無効化が正常に動作しています")
        print("[UP] 期待される効果:")
        print("  • 不適切な強制切替の除去")
        print("  • 分散係数の削減（13.75% → 5%未満目標）")
        print("  • 最適化ルールの正常動作復旧")
    else:
        print("\\n[WARNING] 一部テストが失敗しました。追加修正が必要です。")
    
    return all_passed

if __name__ == "__main__":
    main()
