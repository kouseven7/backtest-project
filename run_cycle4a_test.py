"""
Cycle 4-A: 利益中ポジション保護機能のテストスクリプト

目的: 含み益ポジションがある場合に銘柄切替をスキップする機能をテスト
期間: 2024-01-01 ~ 2024-05-31（5ヶ月）
評価指標: プロフィットファクター、総損益、勝率、銘柄切替回数

Author: Backtest Project Team
Created: 2026-01-20
"""

import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_dssms_backtest(start_date: str, end_date: str, label: str, timeout_seconds: int = 7200):
    """
    DSSMSバックテストを実行
    
    Args:
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
        label: 実行ラベル
        timeout_seconds: タイムアウト（秒）
    
    Returns:
        bool: 成功ならTrue
    """
    logger.info(f"{'='*80}")
    logger.info(f"Cycle 4-A テスト実行開始: {label}")
    logger.info(f"期間: {start_date} ~ {end_date}")
    logger.info(f"{'='*80}")
    
    command = [
        sys.executable,
        "-m", "src.dssms.dssms_integrated_main",
        "--start-date", start_date,
        "--end-date", end_date
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_seconds
        )
        
        if result.returncode == 0:
            logger.info(f"[SUCCESS] {label}: 正常終了")
            logger.info(f"出力ディレクトリ: output/dssms_integration/dssms_YYYYMMDD_HHMMSS/")
            return True
        else:
            logger.error(f"[FAILED] {label}: 実行エラー (exit code: {result.returncode})")
            logger.error(f"STDERR: {result.stderr[-500:]}")  # 最後の500文字のみ表示
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] {label}: {timeout_seconds}秒でタイムアウト")
        return False
    except Exception as e:
        logger.error(f"[ERROR] {label}: {str(e)}", exc_info=True)
        return False


def main():
    """
    Cycle 4-A テスト実行メイン
    """
    logger.info("""
================================================================================
Cycle 4-A: 利益中ポジション保護機能 テスト実行
================================================================================
実装内容:
  - 含み益ポジション（unrealized_pnl > 0）の場合は銘柄切替をスキップ
  - 実装場所: dssms_integrated_main.py _evaluate_and_execute_switch()
  
テスト期間:
  - 2024-01-01 ~ 2024-05-31（5ヶ月）
  
評価指標:
  - プロフィットファクター: 1.3以上目標
  - 総損益: プラス
  - 勝率: 維持または向上
  - 銘柄切替回数: 削減効果確認
  
成功条件:
  - PF 1.3以上
  - 副作用なし（既存機能に影響なし）
================================================================================
""")
    
    # テスト実行
    start_date = "2024-01-01"
    end_date = "2024-05-31"
    
    success = run_dssms_backtest(start_date, end_date, "Cycle 4-A テスト", timeout_seconds=7200)
    
    if success:
        logger.info("""
================================================================================
Cycle 4-A テスト実行完了
================================================================================
次のアクション:
  1. output/dssms_integration/の最新ディレクトリを確認
  2. comprehensive_report.txtから以下を確認:
     - プロフィットファクター
     - 総損益
     - 勝率
     - 銘柄切替回数（DSSMS統計セクション）
  3. ログから[CYCLE4-A]マーカーを検索し、利益中保護の動作を確認
  4. 結果をSystem_Trading_Profit_Improvement_Investigation.mdに記録
  
評価基準:
  - PF 1.3以上: 成功
  - PF 1.0-1.3: 要改善
  - PF 1.0未満: 失敗（改善策Bへ）
================================================================================
""")
    else:
        logger.error("""
================================================================================
Cycle 4-A テスト実行失敗
================================================================================
次のアクション:
  1. エラーログを確認
  2. 実装コードを確認（dssms_integrated_main.py Line 1697-1745）
  3. デバッグ実行で原因特定
================================================================================
""")
        sys.exit(1)


if __name__ == "__main__":
    main()
