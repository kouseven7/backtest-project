"""
VWAP_Bounce戦略 動作検証テスト（フィルター無効版）

トレンドフィルターとボラティリティフィルターを無効化して
VWAP条件のみで戦略が動作するかを検証

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


class VWAPBounceNoFilterTestRunner(VWAPBounceTestRunner):
    """VWAP_Bounce戦略テストランナー（フィルター無効版）"""
    
    def run_backtest(self) -> bool:
        """
        バックテスト実行（フィルター無効パラメータ）
        
        Returns:
            bool: 実行成功=True
        """
        if self.stock_data is None:
            self.logger.error("株価データが未取得です")
            return False
        
        self.logger.info("=" * 80)
        self.logger.info("バックテスト実行開始（フィルター無効版）")
        self.logger.info("=" * 80)
        
        try:
            # フィルター無効パラメータ
            no_filter_params = {
                "trend_filter_enabled": False,      # トレンドフィルター無効
                "volatility_filter_enabled": False  # ボラティリティフィルター無効
            }
            
            from strategies.VWAP_Bounce import VWAPBounceStrategy
            
            # 戦略初期化（フィルター無効）
            self.logger.info("VWAP_Bounce戦略初期化（フィルター無効パラメータ）")
            self.logger.info(f"  trend_filter_enabled: {no_filter_params['trend_filter_enabled']}")
            self.logger.info(f"  volatility_filter_enabled: {no_filter_params['volatility_filter_enabled']}")
            
            self.strategy = VWAPBounceStrategy(
                data=self.stock_data.copy(),
                params=no_filter_params,
                price_column="Adj Close",
                volume_column="Volume"
            )
            
            # 戦略初期化
            self.strategy.initialize_strategy()
            self.logger.info("戦略初期化完了")
            
            # バックテスト実行
            self.logger.info("バックテスト実行中...")
            self.result_data = self.strategy.backtest()
            
            if self.result_data is None:
                self.logger.error("バックテスト実行失敗: 結果がNone")
                return False
            
            self.logger.info("バックテスト実行完了")
            self.logger.info(f"結果データ: {len(self.result_data)}行")
            
            return True
            
        except Exception as e:
            self.logger.error(f"バックテスト実行中にエラー発生: {e}", exc_info=True)
            return False


def main():
    """メイン実行関数"""
    logger.info("\n" + "=" * 80)
    logger.info("VWAP_Bounce戦略 フィルター無効版テスト開始")
    logger.info("=" * 80 + "\n")
    
    # テストランナー作成
    test_runner = VWAPBounceNoFilterTestRunner(
        ticker="8306.T",
        index_ticker="^N225"
    )
    
    # フルテスト実行
    success = test_runner.run_full_test()
    
    logger.info("\n" + "=" * 80)
    logger.info("フィルター無効版テスト完了")
    logger.info("=" * 80 + "\n")
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
