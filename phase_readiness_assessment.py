#!/usr/bin/env python3
"""
DSSMS Phase 1 → Phase 2 移行準備状況評価スクリプト
Phase 1完了状況と Phase 2開始要件の確認
"""

import sys
import os
from pathlib import Path

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_phase1_completion():
    """Phase 1の完了状況を確認"""
    print("[SEARCH] === Phase 1 完了状況確認 ===")
    
    phase1_status = {
        "Task 1.1": {"name": "データ取得問題診断・修正", "completed": False},
        "Task 1.2": {"name": "シミュレーションデータ修正", "completed": False},
        "Task 1.3": {"name": "クイック修正版作成・動作確認", "completed": False}
    }
    
    # Task 1.1: DSSMSBacktester動作確認
    try:
        from src.dssms.dssms_backtester import DSSMSBacktester
        phase1_status["Task 1.1"]["completed"] = True
        print("[OK] Task 1.1: DSSMSBacktester - 正常動作")
    except Exception as e:
        print(f"[ERROR] Task 1.1: DSSMSBacktester - {e}")
    
    # Task 1.2: データ管理システム
    try:
        from src.dssms.dssms_data_manager import DSSMSDataManager
        phase1_status["Task 1.2"]["completed"] = True
        print("[OK] Task 1.2: DSSMSDataManager - 正常動作")
    except Exception as e:
        print(f"[ERROR] Task 1.2: DSSMSDataManager - {e}")
    
    # Task 1.3: クイック修正統合システム
    try:
        from src.dssms.dssms_quick_fix_integration_manager import DSSMSQuickFixIntegrationManager
        phase1_status["Task 1.3"]["completed"] = True
        print("[OK] Task 1.3: DSSMSQuickFixIntegrationManager - 正常動作")
    except Exception as e:
        print(f"[ERROR] Task 1.3: DSSMSQuickFixIntegrationManager - {e}")
    
    # -100%バグ修正確認
    latest_reports = list(Path("backtest_results/dssms_results").glob("dssms_detailed_report_*.txt"))
    if latest_reports:
        latest_report = max(latest_reports, key=lambda p: p.stat().st_mtime)
        with open(latest_report, 'r', encoding='utf-8') as f:
            content = f.read()
            if "総リターン: -100" in content:
                print("[ERROR] -100%バグ: 未修正")
            else:
                print("[OK] -100%バグ: 修正済み")
                print(f"   最新結果: {latest_report.name}")
    
    completed_tasks = sum(1 for task in phase1_status.values() if task["completed"])
    print(f"\n[CHART] Phase 1 完了率: {completed_tasks}/3 タスク ({completed_tasks/3*100:.1f}%)")
    
    return phase1_status

def check_phase2_prerequisites():
    """Phase 2開始に必要な前提条件を確認"""
    print("\n[TOOL] === Phase 2 前提条件確認 ===")
    
    prerequisites = {
        "existing_strategies": {"name": "既存戦略システム", "completed": False},
        "ranking_system": {"name": "ランキングシステム基盤", "completed": False},
        "main_integration": {"name": "main.py統合システム", "completed": False}
    }
    
    # 既存戦略システム確認
    strategy_count = 0
    strategies = [
        ("strategies.VWAP_Breakout", "VWAPBreakoutStrategy"),
        ("strategies.Momentum_Investing", "MomentumInvestingStrategy"),
        ("strategies.VWAP_Bounce", "VWAPBounceStrategy"),
        ("strategies.Opening_Gap", "OpeningGapStrategy")
    ]
    
    for module_name, class_name in strategies:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            strategy_count += 1
        except Exception:
            pass
    
    if strategy_count >= 3:
        prerequisites["existing_strategies"]["completed"] = True
        print(f"[OK] 既存戦略システム: {strategy_count}/4 戦略利用可能")
    else:
        print(f"[ERROR] 既存戦略システム: {strategy_count}/4 戦略のみ利用可能")
    
    # ランキングシステム基盤確認
    try:
        from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem
        prerequisites["ranking_system"]["completed"] = True
        print("[OK] ランキングシステム基盤: 利用可能")
    except Exception as e:
        print(f"[ERROR] ランキングシステム基盤: {e}")
    
    # main.py統合システム確認
    try:
        import main
        prerequisites["main_integration"]["completed"] = True
        print("[OK] main.py統合システム: 利用可能")
    except Exception as e:
        print(f"[ERROR] main.py統合システム: {e}")
    
    completed_prerequisites = sum(1 for req in prerequisites.values() if req["completed"])
    print(f"\n[CHART] Phase 2 前提条件満足率: {completed_prerequisites}/3 ({completed_prerequisites/3*100:.1f}%)")
    
    return prerequisites

def assess_phase2_readiness(phase1_status, prerequisites):
    """Phase 2移行準備状況の総合評価"""
    print("\n[TARGET] === Phase 2 移行準備状況評価 ===")
    
    phase1_completion = sum(1 for task in phase1_status.values() if task["completed"])
    prereq_satisfaction = sum(1 for req in prerequisites.values() if req["completed"])
    
    overall_score = (phase1_completion * 0.6 + prereq_satisfaction * 0.4) / 3 * 100
    
    print(f"Phase 1 完了状況: {phase1_completion}/3 タスク")
    print(f"Phase 2 前提条件: {prereq_satisfaction}/3 項目")
    print(f"総合準備度: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("\n[SUCCESS] 評価: Phase 2 開始準備完了")
        print("   → Task 2.1「既存戦略システム統合」の実装を開始できます")
        return True
    elif overall_score >= 60:
        print("\n[WARNING]  評価: Phase 2 開始準備ほぼ完了")
        print("   → 残存問題の修正後、Phase 2開始を推奨")
        return False
    else:
        print("\n[ERROR] 評価: Phase 2 開始には追加作業が必要")
        print("   → Phase 1の残存タスクを完了してください")
        return False

def identify_remaining_issues(phase1_status, prerequisites):
    """残存する問題の特定"""
    print("\n[TOOL] === 残存問題と対応策 ===")
    
    issues = []
    
    for task_id, task_info in phase1_status.items():
        if not task_info["completed"]:
            issues.append(f"Phase 1 {task_id}: {task_info['name']}")
    
    for req_id, req_info in prerequisites.items():
        if not req_info["completed"]:
            issues.append(f"前提条件: {req_info['name']}")
    
    if issues:
        print("未解決の問題:")
        for issue in issues:
            print(f"  [ERROR] {issue}")
        
        print("\n推奨対応策:")
        if not phase1_status["Task 1.1"]["completed"]:
            print("  1. DSSMSBacktesterの動作確認・修正")
        if not phase1_status["Task 1.2"]["completed"]:
            print("  2. データ管理システムの修正")
        if not phase1_status["Task 1.3"]["completed"]:
            print("  3. クイック修正統合システムの実装")
        if not prerequisites["existing_strategies"]["completed"]:
            print("  4. 既存戦略クラスのインポート問題修正")
        if not prerequisites["ranking_system"]["completed"]:
            print("  5. ランキングシステムの実装・修正")
        if not prerequisites["main_integration"]["completed"]:
            print("  6. main.py統合システムの修正")
    else:
        print("[OK] 特定された問題はありません")

def main():
    """メイン実行関数"""
    print("=" * 70)
    print("[TARGET] DSSMS Phase 1 → Phase 2 移行準備状況評価")
    print("=" * 70)
    
    # Phase 1完了状況確認
    phase1_status = check_phase1_completion()
    
    # Phase 2前提条件確認
    prerequisites = check_phase2_prerequisites()
    
    # 総合評価
    ready = assess_phase2_readiness(phase1_status, prerequisites)
    
    # 残存問題特定
    identify_remaining_issues(phase1_status, prerequisites)
    
    print("\n" + "=" * 70)
    if ready:
        print("[ROCKET] 結論: Phase 2 「ハイブリッド実装」開始可能")
        print("   最初のタスク: Task 2.1「既存戦略システム統合」")
    else:
        print("⏳ 結論: Phase 1の残存作業完了後にPhase 2開始")
    print("=" * 70)

if __name__ == "__main__":
    main()
