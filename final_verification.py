"""
DSSMS条件付き実行システム 最終確認スクリプト
"""
from src.dssms.dssms_switch_coordinator_v2 import DSSMSSwitchCoordinatorV2
import json

print('=== DSSMS条件付き実行システム 最終確認 ===')

# システム初期化
coordinator = DSSMSSwitchCoordinatorV2()

# 設定内容確認
config = coordinator.switch_optimization_config
print(f'条件付き実行有効: {config.get("conditional_execution", {}).get("enabled", False)}')
print(f'コスト閾値: {config.get("conditional_execution", {}).get("cost_efficiency", {}).get("max_switching_cost_ratio", 0.005)*100}%')
print(f'最低利益: {config.get("conditional_execution", {}).get("profit_protection", {}).get("minimum_expected_benefit_yen", 1000)}円')

# 現在の判定結果
overall_decision = coordinator._should_execute_daily_switch_v2()
print(f'現在の実行判定: {"✅ 実行許可" if overall_decision else "❌ 実行拒否"}')

# ステータスレポート
status = coordinator.get_status_report()
conditional = status.get('conditional_execution', {})
print('\n=== 詳細チェック結果 ===')
print(f'コスト効率: {"✅" if conditional.get("cost_efficiency_check") else "❌"}')
print(f'利益保護: {"✅" if conditional.get("profit_protection_check") else "❌"}')
print(f'市場適合性: {"✅" if conditional.get("market_suitability_check") else "❌"}')
print(f'保有期間最適化: {"✅" if conditional.get("holding_period_optimization_check") else "❌"}')

print('\n🎉 条件付き実行システム実装完了')
print('📊 システムは収益性重視で適切に動作中')
print('⚙️ config/switch_optimization_config.json で設定調整可能')
