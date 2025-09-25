#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デバッグ版 - 切替ロジックの詳細調査
"""

def debug_switch_logic():
    """切替ロジックを詳細調査"""
    
    total_days = 30
    
    print("🔍 切替ロジック詳細調査")
    print("=" * 40)
    
    switch_days = []
    for day in range(total_days):
        should_switch = day > 0 and day % 10 == 0
        
        if should_switch:
            switch_days.append(day)
            print(f"Day {day:2d}: 切替実行 (day > 0: {day > 0}, day % 10 == 0: {day % 10 == 0})")
        elif day % 10 == 0:
            print(f"Day {day:2d}: 切替スキップ (day > 0: {day > 0})")
        
    print(f"\n切替発生日: {switch_days}")
    print(f"切替回数: {len(switch_days)}")
    print(f"期待回数: 3 (10, 20, 30日目)")
    
    # 修正版ロジック提案
    print("\n" + "=" * 40)
    print("🔧 修正版ロジックテスト")
    
    switch_days_fixed = []
    for day in range(total_days):
        # 修正版: day % 10 == 0 かつ day > 0 （day=0は除外）
        should_switch_fixed = (day % 10 == 0) and (day > 0)
        
        if should_switch_fixed:
            switch_days_fixed.append(day)
            print(f"Day {day:2d}: 修正版切替実行")
            
    print(f"\n修正版切替発生日: {switch_days_fixed}")
    print(f"修正版切替回数: {len(switch_days_fixed)}")
    
    # 実際の30日間での期待値計算
    expected_days = [d for d in range(1, 31) if d % 10 == 0]
    print(f"実際の期待日: {expected_days}")
    print(f"実際の期待回数: {len(expected_days)}")

if __name__ == "__main__":
    debug_switch_logic()