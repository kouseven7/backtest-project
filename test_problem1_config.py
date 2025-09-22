#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Problem 1 動作確認テスト
切替判定ロジック設定変更後の動作確認
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.dssms.dssms_backtester import DSSMSBacktester
    
    print("🧪 Problem 1: 動作確認テスト開始")
    print("="*50)
    
    # DSSMSBacktester初期化
    print("1. DSSMSBacktester初期化中...")
    backtester = DSSMSBacktester()
    print("✅ DSSMSBacktester初期化成功")
    
    # 設定確認
    print("\n2. 重要設定確認:")
    config = backtester.config
    
    score_threshold = config.get('switch_criteria', {}).get('score_difference_threshold', '不明')
    print(f"   score_difference_threshold: {score_threshold} (期待値: 0.08)")
    
    holding_period = config.get('switch_criteria', {}).get('minimum_holding_period_hours', '不明')
    print(f"   minimum_holding_period_hours: {holding_period} (期待値: 2)")
    
    confidence_threshold = config.get('switch_criteria', {}).get('confidence_threshold', '不明')
    print(f"   confidence_threshold: {confidence_threshold} (期待値: 0.5)")
    
    max_daily = config.get('risk_control', {}).get('max_daily_switches', '不明')
    print(f"   max_daily_switches: {max_daily} (期待値: 5)")
    
    max_weekly = config.get('risk_control', {}).get('max_weekly_switches', '不明')
    print(f"   max_weekly_switches: {max_weekly} (期待値: 20)")
    
    # 確率的切替設定確認
    probabilistic = config.get('randomness_control', {}).get('switching', {}).get('enable_probabilistic', '不明')
    print(f"   enable_probabilistic: {probabilistic} (期待値: True)")
    
    noise_enabled = config.get('randomness_control', {}).get('scoring', {}).get('enable_noise', '不明')
    print(f"   enable_noise: {noise_enabled} (期待値: True)")
    
    # 設定変更成功判定
    print("\n3. 設定変更成功判定:")
    success_count = 0
    total_checks = 7
    
    if score_threshold == 0.08:
        print("   ✅ score_difference_threshold: 正常")
        success_count += 1
    else:
        print(f"   ❌ score_difference_threshold: {score_threshold} (期待値: 0.08)")
    
    if holding_period == 2:
        print("   ✅ minimum_holding_period_hours: 正常")
        success_count += 1
    else:
        print(f"   ❌ minimum_holding_period_hours: {holding_period} (期待値: 2)")
    
    if confidence_threshold == 0.5:
        print("   ✅ confidence_threshold: 正常")
        success_count += 1
    else:
        print(f"   ❌ confidence_threshold: {confidence_threshold} (期待値: 0.5)")
    
    if max_daily == 5:
        print("   ✅ max_daily_switches: 正常")
        success_count += 1
    else:
        print(f"   ❌ max_daily_switches: {max_daily} (期待値: 5)")
    
    if max_weekly == 20:
        print("   ✅ max_weekly_switches: 正常")
        success_count += 1
    else:
        print(f"   ❌ max_weekly_switches: {max_weekly} (期待値: 20)")
    
    if probabilistic == True:
        print("   ✅ enable_probabilistic: 正常")
        success_count += 1
    else:
        print(f"   ❌ enable_probabilistic: {probabilistic} (期待値: True)")
    
    if noise_enabled == True:
        print("   ✅ enable_noise: 正常")
        success_count += 1
    else:
        print(f"   ❌ enable_noise: {noise_enabled} (期待値: True)")
    
    print(f"\n📊 設定確認結果: {success_count}/{total_checks} 項目が正常")
    
    if success_count == total_checks:
        print("🎉 Problem 1 設定変更: 全項目成功")
        exit_code = 0
    else:
        print("⚠️ Problem 1 設定変更: 一部不完全")
        exit_code = 1
    
    print("="*50)
    print("🧪 Problem 1 動作確認テスト完了")
    
    exit(exit_code)
    
except ImportError as e:
    print(f"❌ Import エラー: {e}")
    exit(1)
except Exception as e:
    print(f"❌ 実行エラー: {e}")
    import traceback
    traceback.print_exc()
    exit(1)