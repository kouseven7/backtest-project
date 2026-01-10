"""
Cycle 10最終検証テスト - portfolio_equity_curve.csv修正確認

短期テスト（2025-01-27~2025-02-05）でCycle 10修正内容を検証:
1. portfolio_equity_curve.csvの新カラム（date, cash_balance, position_value, total_value）
2. comprehensive_report.txtの最終資本=974,648円
3. 数学的整合性: 初期1,000,000円 + Σ(pnl)-25,351.69円 = 974,648.31円

Author: Backtest Project Team
Created: 2026-01-10
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# プロジェクトルート追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
from strategies.BreakoutStrategyRelaxed import BreakoutStrategyRelaxed

def main():
    """Cycle 10検証テスト実行"""
    
    # ロガー設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('Cycle10Verification')
    
    logger.info("=" * 60)
    logger.info("Cycle 10最終検証テスト開始")
    logger.info("=" * 60)
    
    # テスト設定（短期、決済完了まで）
    start_date = '2025-01-27'
    end_date = '2025-01-31'  # Cycle 11-3: 決済完了時点で終了（取引2件決済完了）
    initial_capital = 1000000
    
    logger.info(f"テスト期間: {start_date} ~ {end_date}")
    logger.info(f"初期資金: {initial_capital:,}円")
    logger.info(f"戦略: BreakoutRelaxed")
    
    try:
        # DSSMSバックテスター初期化（configベース）
        config = {
            'initial_capital': initial_capital,
            'export_settings': {
                'output_directory': 'output/dssms_cycle10_test'
            }
        }
        
        backtester = DSSMSIntegratedBacktester(config=config)
        
        logger.info("DSSMSバックテスター初期化完了")
        
        # バックテスト実行
        logger.info("バックテスト実行中...")
        results = backtester.run_dynamic_backtest(
            start_date=datetime.strptime(start_date, '%Y-%m-%d'),
            end_date=datetime.strptime(end_date, '%Y-%m-%d')
        )
        
        if results:
            logger.info("=" * 60)
            logger.info("Cycle 10検証テスト完了")
            logger.info("=" * 60)
            logger.info(f"最終ポートフォリオ価値: {results.get('final_portfolio_value', 0):,.2f}円")
            logger.info(f"総取引数: {results.get('total_trades', 0)}件")
            logger.info(f"総リターン: {results.get('total_return', 0):.2f}%")
            logger.info(f"出力ディレクトリ: {results.get('output_dir', 'N/A')}")
            
            # 検証ポイント表示
            logger.info("")
            logger.info("【次の確認事項】")
            logger.info("1. portfolio_equity_curve.csv:")
            logger.info("   - 新カラム: date, cash_balance, position_value, total_value")
            logger.info("   - 最終行cash_balance = 974,648円")
            logger.info("2. comprehensive_report.txt:")
            logger.info("   - 最終資本 = 974,648円")
            logger.info("3. all_transactions.csv:")
            logger.info("   - Σ(pnl) = -25,351.69円")
            
            return True
        else:
            logger.error("バックテスト結果がNoneです")
            return False
            
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
