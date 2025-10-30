"""
VWAP_Bounce戦略 動作検証テスト（9101.T 2023-2024データ）

高ボラティリティ銘柄での検証
銘柄: 9101.T（日本郵船）
期間: 2023/01/01 ~ 2024/12/31（2年間）

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""
import sys
from pathlib import Path

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 元のテストランナーをインポート
from tests.test_vwap_bounce_8306T import VWAPBounceTestRunner

# ロガー設定
from config.logger_config import setup_logger

logger = setup_logger(__name__)


class VWAPBounce9101TestRunner(VWAPBounceTestRunner):
    """VWAP_Bounce戦略テストランナー（9101.T版）"""
    
    def __init__(self):
        """初期化（9101.T用）"""
        super().__init__(
            ticker="9101.T",
            index_ticker="^N225"
        )
    
    def run_full_test(self) -> bool:
        """
        フルテスト実行（2年間）
        
        Returns:
            bool: テスト成功=True
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VWAP_Bounce戦略 動作検証テスト開始（9101.T）")
        self.logger.info(f"実行日時: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80 + "\n")
        
        try:
            # Phase 1: データ取得（2年間）
            if not self.fetch_data(start_date="2023-01-01", end_date="2024-12-31"):
                self.logger.error("データ取得失敗")
                return False
            
            # Phase 2: バックテスト実行
            if not self.run_backtest():
                self.logger.error("バックテスト実行失敗")
                return False
            
            # Phase 3: 検証実行
            self.validate_signals()
            self.validate_vwap_logic()
            self.validate_trend_detection()
            
            # Phase 3.5: トレンド分布分析
            self.analyze_trend_distribution()
            
            self.calculate_performance()
            
            # Phase 4: レポート生成
            test_passed = self.generate_summary_report()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("VWAP_Bounce戦略 動作検証テスト完了（9101.T）")
            self.logger.info("=" * 80 + "\n")
            
            return test_passed
            
        except Exception as e:
            self.logger.error(f"テスト実行中にエラー発生: {e}", exc_info=True)
            return False


def main():
    """メイン実行関数"""
    logger.info("\n" + "=" * 80)
    logger.info("VWAP_Bounce戦略 9101.T（日本郵船）テスト開始")
    logger.info("期間: 2023/01/01 ~ 2024/12/31（2年間）")
    logger.info("=" * 80 + "\n")
    
    # テストランナー作成
    test_runner = VWAPBounce9101TestRunner()
    
    # フルテスト実行
    success = test_runner.run_full_test()
    
    logger.info("\n" + "=" * 80)
    logger.info("9101.Tテスト完了")
    logger.info("=" * 80 + "\n")
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
