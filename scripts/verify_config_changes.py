#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
設定ファイル変更確認スクリプト
Problem 1, 12の設定変更状況を詳細確認
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
config_path = project_root / "config" / "dssms" / "dssms_backtester_config.json"


def verify_config_changes(problem_ids: list = None):
    """設定ファイル変更確認"""
    if problem_ids is None:
        problem_ids = [1, 12]
        
    print(f"=== 設定ファイル変更確認: {config_path} ===\n")
    
    if not config_path.exists():
        print("[ERROR] 設定ファイルが存在しません")
        return False
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] 設定ファイル読み込みエラー: {e}")
        return False
        
    all_verified = True
    
    if 1 in problem_ids:
        print("[SEARCH] Problem 1: 切替判定ロジック劣化")
        verified = verify_problem_1_config(config)
        all_verified = all_verified and verified
        print()
        
    if 12 in problem_ids:
        print("[SEARCH] Problem 12: 決定論的モード設定問題")
        verified = verify_problem_12_config(config)
        all_verified = all_verified and verified
        print()
        
    return all_verified


def verify_problem_1_config(config: dict) -> bool:
    """Problem 1設定変更詳細確認"""
    checks = []
    
    # 1. enable_probabilistic: false → true
    probabilistic = config.get('randomness_control', {}).get('switching', {}).get('enable_probabilistic')
    if probabilistic is True:
        print("[OK] enable_probabilistic: true (確率的切替有効化)")
        checks.append(True)
    else:
        print(f"[ERROR] enable_probabilistic: {probabilistic} (要修正: true)")
        checks.append(False)
        
    # 2. score_difference_threshold: 0.15 → 0.08
    threshold = config.get('switch_criteria', {}).get('score_difference_threshold')
    if threshold is not None and threshold <= 0.10:
        print(f"[OK] score_difference_threshold: {threshold} (閾値緩和)")
        checks.append(True)
    else:
        print(f"[ERROR] score_difference_threshold: {threshold} (要修正: ≤0.10)")
        checks.append(False)
        
    # 3. enable_noise: false → true
    noise = config.get('randomness_control', {}).get('scoring', {}).get('enable_noise')
    if noise is True:
        print("[OK] enable_noise: true (微細変動検出有効化)")
        checks.append(True)
    else:
        print(f"[ERROR] enable_noise: {noise} (要修正: true)")
        checks.append(False)
        
    # 4. noise_level設定
    noise_level = config.get('randomness_control', {}).get('scoring', {}).get('noise_level')
    if noise_level is not None and 0.01 <= noise_level <= 0.05:
        print(f"[OK] noise_level: {noise_level} (適切な範囲)")
        checks.append(True)
    else:
        print(f"[ERROR] noise_level: {noise_level} (要修正: 0.01-0.05)")
        checks.append(False)
        
    # 5. max_daily_switches: 3 → 5
    max_daily = config.get('risk_control', {}).get('max_daily_switches')
    if max_daily is not None and max_daily >= 5:
        print(f"[OK] max_daily_switches: {max_daily} (上限緩和)")
        checks.append(True)
    else:
        print(f"[ERROR] max_daily_switches: {max_daily} (要修正: ≥5)")
        checks.append(False)
        
    # 6. strict_threshold_mode: true → false
    strict_mode = config.get('randomness_control', {}).get('switching', {}).get('strict_threshold_mode')
    if strict_mode is False:
        print("[OK] strict_threshold_mode: false (厳格モード解除)")
        checks.append(True)
    else:
        print(f"[ERROR] strict_threshold_mode: {strict_mode} (要修正: false)")
        checks.append(False)
        
    success_rate = sum(checks) / len(checks) * 100
    print(f"\n[CHART] Problem 1設定完了率: {success_rate:.1f}% ({sum(checks)}/{len(checks)})")
    
    return all(checks)


def verify_problem_12_config(config: dict) -> bool:
    """Problem 12設定変更詳細確認"""
    checks = []
    
    # 1. deterministic: true 維持
    deterministic = config.get('execution_mode', {}).get('deterministic')
    if deterministic is True:
        print("[OK] deterministic: true (決定論的モード維持)")
        checks.append(True)
    else:
        print(f"[ERROR] deterministic: {deterministic} (要修正: true)")
        checks.append(False)
        
    # 2. random_seed設定
    seed = config.get('execution_mode', {}).get('random_seed')
    if seed is not None:
        print(f"[OK] random_seed: {seed} (再現性確保)")
        checks.append(True)
    else:
        print(f"[ERROR] random_seed: {seed} (要設定)")
        checks.append(False)
        
    # 3. enable_reproducible_results: true
    reproducible = config.get('execution_mode', {}).get('enable_reproducible_results')
    if reproducible is True:
        print("[OK] enable_reproducible_results: true (再現性有効)")
        checks.append(True)
    else:
        print(f"[ERROR] enable_reproducible_results: {reproducible} (要修正: true)")
        checks.append(False)
        
    # 4. use_fixed_execution_price: true → false
    fixed_price = config.get('performance_calculation', {}).get('use_fixed_execution_price')
    if fixed_price is False:
        print("[OK] use_fixed_execution_price: false (価格変動考慮)")
        checks.append(True)
    else:
        print(f"[ERROR] use_fixed_execution_price: {fixed_price} (要修正: false)")
        checks.append(False)
        
    success_rate = sum(checks) / len(checks) * 100
    print(f"\n[CHART] Problem 12設定完了率: {success_rate:.1f}% ({sum(checks)}/{len(checks)})")
    
    return all(checks)


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='設定ファイル変更確認')
    parser.add_argument('--problem', type=int, action='append', 
                       help='確認するProblem ID (1, 12)')
    
    args = parser.parse_args()
    
    problem_ids = args.problem if args.problem else [1, 12]
    
    success = verify_config_changes(problem_ids)
    
    if success:
        print("[SUCCESS] すべての設定変更が完了しています")
        sys.exit(0)
    else:
        print("[WARNING]  未完了の設定変更があります")
        sys.exit(1)


if __name__ == "__main__":
    main()