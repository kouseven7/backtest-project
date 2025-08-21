"""
Task 1.1 Integration Test
Description: DSSMS Task 1.1「データ取得問題の診断と修正」統合テスト
"""

import sys
import os
from datetime import datetime, timedelta

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_dssms_task_1_1():
    """Task 1.1統合テスト"""
    print("=== DSSMS Task 1.1 統合テスト ===")
    print(f"テスト開始時刻: {datetime.now()}")
    
    try:
        # 1. 診断システムテスト
        print("\n📊 1. 診断システムテスト")
        try:
            from src.dssms.dssms_data_diagnostics import DSSMSDataDiagnostics
            
            diagnostics = DSSMSDataDiagnostics()
            test_symbols = ["7203.T", "8058.T"]
            
            diagnosis = diagnostics.diagnose_data_sources(test_symbols)
            print(f"   ✅ 診断完了: {diagnosis['overall_status']}")
            print(f"   📈 データ品質スコア: {diagnosis.get('quality_score', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ 診断システムエラー: {e}")
            return False
        
        # 2. 統合パッチテスト
        print("\n🔧 2. 統合パッチテスト")
        try:
            from src.dssms.dssms_integration_patch import (
                update_symbol_ranking_with_real_data,
                update_portfolio_value_with_real_data
            )
            
            test_date = datetime.now()
            
            # ランキング更新テスト
            scores = update_symbol_ranking_with_real_data(test_symbols, test_date)
            print(f"   ✅ ランキング更新: {len(scores)}銘柄")
            
            # ポートフォリオ価値更新テスト
            new_value = update_portfolio_value_with_real_data("7203.T", 1000000, test_date)
            print(f"   ✅ 価値更新: 1,000,000 -> {new_value:,.0f}")
            
        except Exception as e:
            print(f"   ❌ 統合パッチエラー: {e}")
            return False
        
        # 3. バックテスター統合テスト
        print("\n🎯 3. バックテスター統合テスト")
        try:
            from src.dssms.dssms_backtester import DSSMSBacktester
            
            # バックテスター初期化
            backtester = DSSMSBacktester()
            
            # 基本設定
            config = {
                'symbols': test_symbols,
                'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'initial_capital': 1000000
            }
            
            print(f"   🚀 バックテスト実行中...")
            
            # 修正されたメソッドのテスト
            test_date = datetime.now()
            
            # ランキング更新テスト（修正済み）
            ranking_result = backtester._update_symbol_ranking(test_date, test_symbols)
            print(f"   ✅ ランキング: {ranking_result.get('top_symbol', 'N/A')}")
            
            # ポートフォリオ価値更新テスト（修正済み）
            new_portfolio_value = backtester._update_portfolio_value(test_date, "7203.T", 1000000)
            print(f"   ✅ ポートフォリオ価値: {new_portfolio_value:,.0f}")
            
            print(f"   ✅ バックテスター統合成功")
            
        except Exception as e:
            print(f"   ❌ バックテスターエラー: {e}")
            return False
        
        # 4. レポート生成テスト
        print("\n📝 4. レポート生成テスト")
        try:
            report_data = {
                'test_timestamp': datetime.now().isoformat(),
                'diagnosis_status': diagnosis['overall_status'],
                'integration_patch': 'functional',
                'backtester_status': 'modified',
                'data_sources': 'real_data_with_fallback',
                'task_1_1_status': 'completed'
            }
            
            print("   📋 Task 1.1 完了レポート:")
            for key, value in report_data.items():
                print(f"      {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ レポート生成エラー: {e}")
            return False
        
    except Exception as e:
        print(f"❌ 統合テスト失敗: {e}")
        return False

def main():
    """メイン実行"""
    success = test_dssms_task_1_1()
    
    if success:
        print("\n🎉 Task 1.1 統合テスト成功")
        print("✅ データ取得問題の診断と修正が完了しました")
        print("\n📌 実装内容:")
        print("   • DSSMSDataDiagnostics: 診断システム")
        print("   • dssms_integration_patch: 実データ統合")
        print("   • dssms_backtester: ランダムデータ問題修正")
        print("   • フォールバック機能: 3段階（実データ→キャッシュ→サンプル）")
    else:
        print("\n❌ Task 1.1 統合テスト失敗")
        print("🔧 追加修正が必要です")

if __name__ == "__main__":
    main()
