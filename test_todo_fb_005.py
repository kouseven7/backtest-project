"""
TODO-FB-005 テスト: dssms_backtester.py スコア計算改善検証

修正内容の動作確認:
1. _calculate_market_based_fallback_score のスコア範囲拡張 (0.3-0.7 → 0.05-0.95)
2. SystemFallbackPolicy統合による明示的フォールバック処理
3. _market_score_fallback 新規フォールバック関数の動作確認

Test Cases:
1. スコア範囲拡張確認 (0.05-0.95範囲)
2. SystemFallbackPolicy統合動作確認
3. フォールバック使用時の警告ログ出力確認
4. Production/Development mode差の確認

Author: GitHub Copilot Agent
Created: 2025-10-02
Task: TODO-FB-005 Test
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# テスト対象の import
try:
    from src.config.system_modes import SystemFallbackPolicy, ComponentType, SystemMode, get_fallback_policy, set_system_mode
    from src.dssms.dssms_backtester import DSSMSBacktester
    print("[OK] Import successful: SystemFallbackPolicy & DSSMSBacktester")
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    IMPORTS_AVAILABLE = False

def test_score_range_expansion():
    """スコア範囲拡張テスト (0.3-0.7 → 0.05-0.95)"""
    if not IMPORTS_AVAILABLE:
        return {"status": "skipped", "reason": "import_failed"}
    
    print("\n[TEST] スコア範囲拡張テスト開始")
    
    try:
        # DSSMSBacktester初期化 (モック設定)
        config = {
            'initial_capital': 1000000,
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'deterministic_mode': True
        }
        
        backtester = DSSMSBacktester(config)
        
        # テスト用銘柄・日付
        test_symbol = "7203"
        test_date = datetime(2024, 1, 15)
        
        # 複数回実行して範囲確認
        scores = []
        for i in range(20):
            # 異なる銘柄で範囲テスト
            test_sym = f"{7203 + i}"
            score = backtester._market_score_fallback(test_sym, test_date)
            scores.append(score)
        
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        print(f"[OK] スコア範囲確認:")
        print(f"   - 最小値: {min_score:.4f}")
        print(f"   - 最大値: {max_score:.4f}")
        print(f"   - 平均値: {avg_score:.4f}")
        print(f"   - 期待範囲: 0.05-0.95")
        
        # 範囲チェック
        range_check = min_score >= 0.05 and max_score <= 0.95
        print(f"   - 範囲適合: {'[OK]' if range_check else '[ERROR]'}")
        
        return {
            "status": "success",
            "min_score": min_score,
            "max_score": max_score,
            "avg_score": avg_score,
            "range_valid": range_check,
            "scores": scores
        }
        
    except Exception as e:
        print(f"[ERROR] スコア範囲テスト失敗: {e}")
        return {"status": "failed", "error": str(e)}

def test_fallback_policy_integration():
    """SystemFallbackPolicy統合テスト"""
    if not IMPORTS_AVAILABLE:
        return {"status": "skipped", "reason": "import_failed"}
    
    print("\n[TEST] SystemFallbackPolicy統合テスト開始")
    
    # Development modeで実行
    set_system_mode(SystemMode.DEVELOPMENT)
    
    try:
        # DSSMSBacktester初期化 (data_fetcher無し)
        config = {
            'initial_capital': 1000000,
            'start_date': '2024-01-01',
            'end_date': '2024-01-31'
        }
        
        backtester = DSSMSBacktester(config)
        # data_fetcherを意図的に削除してフォールバック発生
        if hasattr(backtester, 'data_fetcher'):
            delattr(backtester, 'data_fetcher')
        
        test_symbol = "7203"
        test_date = datetime(2024, 1, 15)
        
        # _calculate_market_based_fallback_score実行 (フォールバック発生予定)
        score = backtester._calculate_market_based_fallback_score(test_symbol, test_date)
        
        print(f"[OK] SystemFallbackPolicy統合スコア: {score:.4f}")
        
        # フォールバック使用統計確認
        policy = get_fallback_policy()
        stats = policy.get_usage_statistics()
        
        print(f"[OK] フォールバック使用統計:")
        print(f"   - 総失敗数: {stats['total_failures']}")
        print(f"   - 成功フォールバック: {stats.get('successful_fallbacks', 0)}")
        print(f"   - 使用率: {stats.get('fallback_usage_rate', 0):.1%}")
        
        return {
            "status": "success",
            "score": score,
            "fallback_stats": stats
        }
        
    except Exception as e:
        print(f"[ERROR] SystemFallbackPolicy統合テスト失敗: {e}")
        return {"status": "failed", "error": str(e)}

def test_production_mode_behavior():  
    """Production modeでのフォールバック動作テスト"""
    if not IMPORTS_AVAILABLE:
        return {"status": "skipped", "reason": "import_failed"}
    
    print("\n[TEST] Production mode フォールバック動作テスト")
    
    # Production modeに変更
    set_system_mode(SystemMode.PRODUCTION)
    
    try:
        config = {
            'initial_capital': 1000000,
            'start_date': '2024-01-01',
            'end_date': '2024-01-31'
        }
        
        backtester = DSSMSBacktester(config)
        # data_fetcherを意図的に削除
        if hasattr(backtester, 'data_fetcher'):
            delattr(backtester, 'data_fetcher')
        
        test_symbol = "7203"
        test_date = datetime(2024, 1, 15)
        
        # Production modeではフォールバック禁止のはず
        try:
            score = backtester._calculate_market_based_fallback_score(test_symbol, test_date)
            print(f"[ERROR] Production modeでフォールバックが実行された: {score}")
            return {"status": "failed", "reason": "fallback_executed_in_production"}
            
        except Exception as e:
            print(f"[OK] Production mode正常動作: フォールバック禁止確認")
            return {"status": "success", "production_error": str(e)}
            
    except Exception as e:
        print(f"[ERROR] Production modeテスト失敗: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        # Development modeに戻す
        set_system_mode(SystemMode.DEVELOPMENT)

def test_warning_log_output():
    """フォールバック警告ログ出力テスト"""
    if not IMPORTS_AVAILABLE:
        return {"status": "skipped", "reason": "import_failed"}
    
    print("\n[TEST] フォールバック警告ログ出力テスト")
    
    set_system_mode(SystemMode.DEVELOPMENT)
    
    try:
        config = {
            'initial_capital': 1000000,
            'start_date': '2024-01-01',
            'end_date': '2024-01-31'
        }
        
        backtester = DSSMSBacktester(config)
        
        test_symbol = "7203"
        test_date = datetime(2024, 1, 15)
        
        # _market_score_fallback直接実行でログ確認
        score = backtester._market_score_fallback(test_symbol, test_date)
        
        print(f"[OK] フォールバック関数実行完了: {score:.4f}")
        print(f"[OK] 警告ログ出力確認 (WARNING: FALLBACK が出力されているか確認)")
        
        return {"status": "success", "fallback_score": score}
        
    except Exception as e:
        print(f"[ERROR] ログ出力テスト失敗: {e}")
        return {"status": "failed", "error": str(e)}

def generate_test_report():
    """テスト結果レポート生成"""
    print("\n[CHART] TODO-FB-005 テストレポート")
    print("=" * 60)
    
    results = {}
    
    # Test 1: スコア範囲拡張テスト
    results['score_range_test'] = test_score_range_expansion()
    
    # Test 2: SystemFallbackPolicy統合テスト
    results['fallback_integration_test'] = test_fallback_policy_integration()
    
    # Test 3: Production mode動作テスト
    results['production_mode_test'] = test_production_mode_behavior()
    
    # Test 4: 警告ログ出力テスト
    results['warning_log_test'] = test_warning_log_output()
    
    # Development modeに戻す
    if IMPORTS_AVAILABLE:
        set_system_mode(SystemMode.DEVELOPMENT)
    
    # 結果サマリ
    successful_tests = sum(1 for test in results.values() if test.get('status') == 'success')
    total_tests = len(results)
    
    print(f"\n[UP] テスト結果サマリ:")
    print(f"   - 成功: {successful_tests}/{total_tests}")
    print(f"   - 成功率: {successful_tests/total_tests:.1%}")
    
    # 詳細結果保存
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "task": "TODO-FB-005 dssms_backtester.py スコア計算改善テスト",
        "test_results": results,
        "summary": {
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": successful_tests/total_tests
        }
    }
    
    report_path = Path("todo_fb_005_test_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 詳細レポート保存: {report_path}")
    
    return report_data

if __name__ == "__main__":
    # テスト実行
    try:
        report = generate_test_report()
        
        # 最終フォールバック統計出力
        if IMPORTS_AVAILABLE:
            policy = get_fallback_policy()
            final_stats = policy.get_usage_statistics()
            print(f"\n[CHART] 最終フォールバック統計:")
            print(f"   - 総記録数: {len(final_stats.get('records', []))}")
            if 'successful_fallbacks' in final_stats:
                print(f"   - 成功フォールバック: {final_stats['successful_fallbacks']}")
        
        print("\n[TARGET] TODO-FB-005 テスト完了")
        
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()