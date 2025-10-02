"""
TODO-FB-004 Mini-Task 3: SystemFallbackPolicy統合テスト

dssms_integrated_main.pyの修正内容を検証し、
SystemFallbackPolicyの正常動作とフォールバック記録を確認

Test Cases:
1. SystemFallbackPolicy利用可能時の動作確認
2. フォールバック記録・統計機能の動作確認
3. PRODUCTION/DEVELOPMENT modeの動作差確認
4. ログ出力内容の検証

Author: GitHub Copilot Agent
Created: 2025-10-02
Task: TODO-FB-004 Mini-Task 3
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
    from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
    print("✅ Import successful: SystemFallbackPolicy & DSSMSIntegratedBacktester")
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import failed: {e}")
    IMPORTS_AVAILABLE = False

def test_system_fallback_policy_integration():
    """SystemFallbackPolicy統合テスト"""
    if not IMPORTS_AVAILABLE:
        return {"status": "skipped", "reason": "import_failed"}
    
    print("\n🧪 SystemFallbackPolicy統合テスト開始")
    
    # Development modeで初期化
    set_system_mode(SystemMode.DEVELOPMENT)
    policy = get_fallback_policy()
    
    # テスト用フォールバック関数
    def test_fallback():
        return "TEST_SYMBOL_7203"
    
    try:
        # フォールバック実行テスト
        result = policy.handle_component_failure(
            component_type=ComponentType.DSSMS_CORE,
            component_name="TEST_DSSMSIntegratedBacktester._get_optimal_symbol",
            error=RuntimeError("DSS Core V3 unavailable - test"),
            fallback_func=test_fallback,
            context={
                "target_date": "2024-01-15",
                "available_symbols": 50,
                "portfolio_value": 1000000
            }
        )
        
        print(f"✅ フォールバック結果: {result}")
        
        # 使用統計確認
        stats = policy.get_usage_statistics()
        print(f"✅ フォールバック使用統計:")
        print(f"   - 総失敗数: {stats['total_failures']}")
        print(f"   - 成功フォールバック: {stats['successful_fallbacks']}")
        print(f"   - 使用率: {stats['fallback_usage_rate']:.1%}")
        print(f"   - システムモード: {stats['system_mode']}")
        
        # コンポーネント別統計確認
        if stats['by_component_type']:
            for comp_type, comp_stats in stats['by_component_type'].items():
                print(f"   - {comp_type}: {comp_stats['fallback_used']}/{comp_stats['total']}")
        
        return {"status": "success", "result": result, "stats": stats}
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return {"status": "failed", "error": str(e)}

def test_production_mode_behavior():
    """Production modeでのフォールバック禁止テスト"""
    if not IMPORTS_AVAILABLE:
        return {"status": "skipped", "reason": "import_failed"}
    
    print("\n🧪 Production mode フォールバック禁止テスト")
    
    # Production modeに変更
    set_system_mode(SystemMode.PRODUCTION)
    policy = get_fallback_policy()
    
    def test_fallback():
        return "SHOULD_NOT_EXECUTE"
    
    try:
        # フォールバック実行 - エラーが発生するはず
        result = policy.handle_component_failure(
            component_type=ComponentType.DSSMS_CORE,
            component_name="TEST_PRODUCTION_MODE",
            error=RuntimeError("Test error for production"),
            fallback_func=test_fallback,
            context={"test": "production_mode"}
        )
        print(f"❌ Production modeでフォールバックが実行された: {result}")
        return {"status": "failed", "reason": "fallback_executed_in_production"}
        
    except RuntimeError as e:
        print(f"✅ Production mode正常動作: フォールバック禁止 - {e}")
        
        # 記録確認
        stats = policy.get_usage_statistics()
        production_records = [r for r in stats['records'] if r['fallback_used'] == False]
        print(f"✅ Production失敗記録: {len(production_records)}件")
        
        return {"status": "success", "records_count": len(production_records)}

def test_dssms_integrated_main_mock():
    """dssms_integrated_main.pyの修正内容モックテスト"""
    if not IMPORTS_AVAILABLE:
        return {"status": "skipped", "reason": "import_failed"}
    
    print("\n🧪 DSSMSIntegratedBacktester修正内容テスト")
    
    # Development modeで実行
    set_system_mode(SystemMode.DEVELOPMENT)
    
    try:
        # DSSMSIntegratedBacktesterの初期化テスト
        config = {
            "symbol_switch": {},
            "data_cache": {},
            "initial_capital": 1000000
        }
        
        backtester = DSSMSIntegratedBacktester(config)
        print("✅ DSSMSIntegratedBacktester初期化成功")
        
        # _nikkei225_fallback_selectionメソッドの存在確認
        if hasattr(backtester, '_nikkei225_fallback_selection'):
            print("✅ _nikkei225_fallback_selection メソッド存在確認")
            
            # モックテスト
            test_symbols = ['7203', '6758', '6702', '4519', '8058']
            try:
                selected = backtester._nikkei225_fallback_selection(test_symbols)
                print(f"✅ フォールバック選択結果: {selected}")
                return {"status": "success", "selected_symbol": selected}
            except Exception as e:
                print(f"❌ フォールバック選択失敗: {e}")
                return {"status": "failed", "error": str(e)}
        else:
            print("❌ _nikkei225_fallback_selection メソッドが見つからない")
            return {"status": "failed", "reason": "method_not_found"}
            
    except Exception as e:
        print(f"❌ DSSMSIntegratedBacktester初期化失敗: {e}")
        return {"status": "failed", "error": str(e)}

def generate_test_report():
    """テスト結果レポート生成"""
    print("\n📊 TODO-FB-004 Mini-Task 3 テストレポート")
    print("=" * 60)
    
    results = {}
    
    # Test 1: SystemFallbackPolicy統合テスト
    results['integration_test'] = test_system_fallback_policy_integration()
    
    # Test 2: Production mode動作テスト
    results['production_test'] = test_production_mode_behavior()
    
    # Test 3: DSSMSIntegratedBacktester修正テスト
    results['dssms_modification_test'] = test_dssms_integrated_main_mock()
    
    # Development modeに戻す
    set_system_mode(SystemMode.DEVELOPMENT)
    
    # 結果サマリ
    successful_tests = sum(1 for test in results.values() if test.get('status') == 'success')
    total_tests = len(results)
    
    print(f"\n📈 テスト結果サマリ:")
    print(f"   - 成功: {successful_tests}/{total_tests}")
    print(f"   - 成功率: {successful_tests/total_tests:.1%}")
    
    # 詳細結果保存
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "task": "TODO-FB-004 Mini-Task 3",
        "test_results": results,
        "summary": {
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": successful_tests/total_tests
        }
    }
    
    report_path = Path("todo_fb_004_mini_task_3_test_report.json")
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
            print(f"\n📊 最終フォールバック統計:")
            print(f"   - 総記録数: {len(final_stats['records'])}")
            print(f"   - 成功フォールバック: {final_stats['successful_fallbacks']}")
        
        print("\n🎯 TODO-FB-004 Mini-Task 3 完了")
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()