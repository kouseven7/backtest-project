"""
main_v2 設定管理モジュール

Phase 1 対応:
- 基本設定
- ログ設定
- リスク管理設定
- パラメータ管理

再利用予定モジュール (main.py実証済み):
- config.logger_config.setup_logger
- config.risk_management.RiskManagement
- config.optimized_parameters.OptimizedParameterManager
"""

# TODO: Phase 1実装予定
# 1. logger_config統合
# 2. risk_management統合  
# 3. optimized_parameters統合
# 4. システム設定

class MainV2Config:
    """main_v2.py専用設定クラス"""
    
    def __init__(self):
        self.phase = "Phase 1"
        self.target_strategy = "VWAPBreakoutStrategy"
        self.output_formats = ["CSV", "JSON", "TXT"]  # Excel禁止
        
    def get_config(self):
        """設定取得"""
        return {
            "phase": self.phase,
            "target_strategy": self.target_strategy,
            "output_formats": self.output_formats
        }