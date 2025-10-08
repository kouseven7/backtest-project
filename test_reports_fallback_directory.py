#!/usr/bin/env python
"""
SystemFallbackPolicy reports/fallback/ ディレクトリ出力テスト

テスト対象:
1. reports/fallback/ ディレクトリへの出力確認
2. 古いレポートファイル自動削除機能確認
3. .gitignore 動作確認

Author: imega
Created: 2025-10-02
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

try:
    from src.config.system_modes import SystemFallbackPolicy, ComponentType, SystemMode
    print("[OK] Import successful: SystemFallbackPolicy & ComponentType")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def test_reports_fallback_directory_output():
    """reports/fallback/ ディレクトリ出力テスト"""
    print("\n[TEST] reports/fallback/ ディレクトリ出力テスト開始")
    
    fallback_policy = SystemFallbackPolicy()
    
    # フォールバック発生をシミュレート
    test_error = RuntimeError("Test fallback for directory output")
    
    result = fallback_policy.handle_component_failure(
        component_type=ComponentType.MULTI_STRATEGY,
        component_name="DirectoryOutputTest",
        error=test_error,
        fallback_func=lambda: "directory_test_result"
    )
    
    print(f"[OK] フォールバック結果: {result}")
    
    # レポート出力テスト
    report_path = fallback_policy.export_usage_report()
    print(f"[OK] レポート出力パス: {report_path}")
    
    # reports/fallback/ ディレクトリに出力されているか確認
    expected_dir = Path("reports/fallback")
    report_file = Path(report_path)
    
    assert expected_dir.exists(), "reports/fallback/ ディレクトリが存在しません"
    print("[OK] reports/fallback/ ディレクトリ存在確認")
    
    assert report_file.parent == expected_dir, f"出力先が期待値と異なります: {report_file.parent}"
    print("[OK] レポートファイル出力先確認 (reports/fallback/)")
    
    assert report_file.exists(), f"レポートファイルが存在しません: {report_path}"
    print("[OK] レポートファイル存在確認")
    
    # ファイル内容確認
    import json
    with open(report_file, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
        assert 'total_failures' in report_data, "レポート内容に必要なキーが含まれていません"
        print(f"[OK] レポート内容確認: {report_data['total_failures']} 件のフォールバック記録")
    
    return True

def test_old_reports_cleanup():
    """古いレポートファイル自動削除機能テスト"""
    print("\n[TEST] 古いレポートファイル自動削除機能テスト開始")
    
    reports_dir = Path("reports/fallback")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 古いテストファイルを作成（7日以上前の日付）
    old_timestamp = (datetime.now() - timedelta(days=8)).strftime("%Y%m%d_%H%M%S")
    old_test_file = reports_dir / f"fallback_usage_report_{old_timestamp}.json"
    
    # テスト用の古いファイル作成
    with open(old_test_file, 'w', encoding='utf-8') as f:
        json.dump({"test": "old_file"}, f)
    
    # ファイルの作成時刻を8日前に設定
    import os
    old_time = (datetime.now() - timedelta(days=8)).timestamp()
    os.utime(old_test_file, (old_time, old_time))
    
    print(f"[OK] テスト用古いファイル作成: {old_test_file.name}")
    
    # 新しいレポート生成（クリーンアップトリガー）
    fallback_policy = SystemFallbackPolicy()
    test_error = RuntimeError("Cleanup test")
    
    fallback_policy.handle_component_failure(
        component_type=ComponentType.MULTI_STRATEGY,
        component_name="CleanupTest",
        error=test_error,
        fallback_func=lambda: "cleanup_test_result"
    )
    
    report_path = fallback_policy.export_usage_report()
    print(f"[OK] 新しいレポート生成: {Path(report_path).name}")
    
    # 古いファイルが削除されているか確認
    if not old_test_file.exists():
        print("[OK] 古いレポートファイル自動削除確認")
        cleanup_success = True
    else:
        print("[WARNING] 古いファイルが削除されていません（削除条件を満たしていない可能性）")
        cleanup_success = False
    
    return cleanup_success

def test_gitignore_effectiveness():
    """.gitignore 動作確認テスト"""
    print("\n[TEST] .gitignore 動作確認テスト開始")
    
    gitignore_path = Path(".gitignore")
    
    if not gitignore_path.exists():
        print("[WARNING] .gitignore ファイルが見つかりません")
        return False
    
    # .gitignore の内容確認
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        gitignore_content = f.read()
    
    # reports/ と *.json が含まれているか確認
    reports_excluded = "reports/" in gitignore_content
    json_excluded = "*.json" in gitignore_content
    
    if reports_excluded:
        print("[OK] .gitignore に reports/ 除外設定確認")
    else:
        print("[WARNING] .gitignore に reports/ 除外設定が見つかりません")
    
    if json_excluded:
        print("[OK] .gitignore に *.json 除外設定確認")
    else:
        print("[WARNING] .gitignore に *.json 除外設定が見つかりません")
    
    return reports_excluded and json_excluded

def main():
    """レポートディレクトリ・出力機能テストメイン実行"""
    print("[CHART] SystemFallbackPolicy reports/fallback/ 出力テストレポート")
    print("============================================================")
    
    tests = [
        ("reports/fallback/ ディレクトリ出力", test_reports_fallback_directory_output),
        ("古いレポートファイル自動削除", test_old_reports_cleanup),
        (".gitignore 動作確認", test_gitignore_effectiveness)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] テスト実行エラー {test_name}: {e}")
            results.append((test_name, False))
    
    # 結果サマリ
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    
    print(f"\n[UP] テスト結果サマリ:")
    print(f"   - 成功: {success_count}/{total_count}")
    print(f"   - 成功率: {success_rate:.1f}%")
    
    # reports/fallback/ ディレクトリ内容表示
    reports_dir = Path("reports/fallback")
    if reports_dir.exists():
        print(f"\n📁 reports/fallback/ ディレクトリ内容:")
        for file in reports_dir.glob("*.json"):
            print(f"   - {file.name}")
    
    print("\n[TARGET] SystemFallbackPolicy reports/fallback/ 出力テスト完了")
    
    return success_rate == 100.0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)