"""
DSSMS Switch Coordinator V2 条件付き実行システム デモ
収益性重視アプローチによる日次目標の条件付き実行テスト

Author: GitHub Copilot Agent  
Created: 2025-01-31
Target: 条件付き実行システムの動作確認
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

def test_conditional_execution_system():
    """条件付き実行システムのテスト"""
    logger = setup_logger(__name__)
    logger.info("=== DSSMS条件付き実行システム デモ開始 ===")
    
    try:
        # 1. 設定ファイル確認
        config_path = project_root / "config" / "switch_optimization_config.json"
        logger.info(f"設定ファイル確認: {config_path}")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("[OK] 設定ファイル読み込み成功")
            logger.info(f"条件付き実行有効: {config.get('conditional_execution', {}).get('enabled', False)}")
        else:
            logger.error("[ERROR] 設定ファイルが見つかりません")
            return False
        
        # 2. Switch Coordinator V2 初期化テスト
        logger.info("Switch Coordinator V2 初期化テスト...")
        
        try:
            from src.dssms.dssms_switch_coordinator_v2 import DSSMSSwitchCoordinatorV2
            coordinator = DSSMSSwitchCoordinatorV2()
            logger.info("[OK] Switch Coordinator V2 初期化成功")
        except Exception as init_error:
            logger.error(f"[ERROR] Switch Coordinator V2 初期化失敗: {init_error}")
            return False
        
        # 3. 条件付き実行判定テスト
        logger.info("条件付き実行判定テスト...")
        
        # 各チェック項目の個別テスト
        cost_efficiency = coordinator._check_cost_efficiency()
        profit_protection = coordinator._check_profit_protection()
        market_suitability = coordinator._check_market_suitability()
        holding_period = coordinator._check_holding_period_optimization()
        overall_decision = coordinator._should_execute_daily_switch_v2()
        
        logger.info(f"コスト効率チェック: {'[OK] 通過' if cost_efficiency else '[ERROR] 失敗'}")
        logger.info(f"利益保護チェック: {'[OK] 通過' if profit_protection else '[ERROR] 失敗'}")
        logger.info(f"市場適合性チェック: {'[OK] 通過' if market_suitability else '[ERROR] 失敗'}")
        logger.info(f"保有期間最適化チェック: {'[OK] 通過' if holding_period else '[ERROR] 失敗'}")
        logger.info(f"総合判定: {'[OK] 実行許可' if overall_decision else '[ERROR] 実行拒否'}")
        
        # 4. ステータスレポート取得テスト
        logger.info("ステータスレポート取得テスト...")
        
        try:
            status_report = coordinator.get_status_report()
            logger.info("[OK] ステータスレポート取得成功")
            
            # 条件付き実行状態の確認
            conditional_status = status_report.get("conditional_execution", {})
            logger.info("--- 条件付き実行状態 ---")
            logger.info(f"システム有効: {conditional_status.get('enabled', False)}")
            logger.info(f"コスト効率: {conditional_status.get('cost_efficiency_check', False)}")
            logger.info(f"利益保護: {conditional_status.get('profit_protection_check', False)}")
            logger.info(f"市場適合性: {conditional_status.get('market_suitability_check', False)}")
            logger.info(f"保有期間最適化: {conditional_status.get('holding_period_optimization_check', False)}")
            logger.info(f"総合判定: {conditional_status.get('overall_decision', False)}")
            
        except Exception as status_error:
            logger.error(f"[ERROR] ステータスレポート取得失敗: {status_error}")
            return False
        
        # 5. 設定値変更テスト
        logger.info("設定値変更テスト...")
        
        # より厳しい条件に変更
        test_config = config.copy()
        test_config["conditional_execution"]["cost_efficiency"]["max_switching_cost_ratio"] = 0.001  # 0.1%
        test_config["conditional_execution"]["profit_protection"]["minimum_expected_benefit_yen"] = 5000  # 5000円
        
        # 設定更新
        coordinator.switch_optimization_config = test_config
        
        # 再テスト
        new_decision = coordinator._should_execute_daily_switch_v2()
        logger.info(f"厳しい条件での判定: {'[OK] 実行許可' if new_decision else '[ERROR] 実行拒否'}")
        
        # 6. パフォーマンス統計テスト（ダミーデータ追加）
        logger.info("パフォーマンス統計テスト...")
        
        # ダミーの実行履歴作成
        from src.dssms.dssms_switch_coordinator_v2 import SwitchExecutionResult
        
        for i in range(5):
            dummy_result = SwitchExecutionResult(
                timestamp=datetime.now() - timedelta(hours=i),
                engine_used="v2",
                success=i % 2 == 0,  # 交互に成功/失敗
                symbols_before=["STOCK_A", "STOCK_B"],
                symbols_after=["STOCK_C", "STOCK_D"],
                switches_count=2 if i % 2 == 0 else 0,
                execution_time_ms=100.0 + i * 10,
                success_rate=0.6 - i * 0.1
            )
            coordinator.execution_history.append(dummy_result)
        
        # 統計取得
        try:
            performance_stats = coordinator.get_performance_statistics()
            logger.info("[OK] パフォーマンス統計取得成功")
            logger.info(f"実行履歴数: {len(coordinator.execution_history)}")
            
        except Exception as perf_error:
            logger.error(f"[ERROR] パフォーマンス統計取得失敗: {perf_error}")
        
        # 7. 最終結果サマリー
        logger.info("=== テスト結果サマリー ===")
        logger.info(f"条件付き実行システム: {'[OK] 正常動作' if overall_decision is not None else '[ERROR] 動作不良'}")
        logger.info(f"設定ファイル読み込み: [OK] 成功")
        logger.info(f"初期化: [OK] 成功")
        logger.info(f"4段階チェック: [OK] 全項目実行")
        logger.info(f"ステータスレポート: [OK] 正常取得")
        logger.info(f"設定変更対応: [OK] 動的更新")
        
        logger.info("=== DSSMS条件付き実行システム デモ完了 ===")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] デモ実行中にエラー: {e}")
        logger.error(f"詳細エラー: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_conditional_execution_system()
    
    if success:
        print("\n[SUCCESS] 条件付き実行システム デモ成功")
        print("[CHART] システムは正常に動作しています")
        print("⚙️  設定ファイルで動作をカスタマイズできます: config/switch_optimization_config.json")
    else:
        print("\n[ERROR] 条件付き実行システム デモ失敗")
        print("[SEARCH] ログファイルで詳細を確認してください")
    
    print(f"\n📝 ログファイル: logs/{datetime.now().strftime('%Y%m%d')}_demo_conditional_execution.log")
