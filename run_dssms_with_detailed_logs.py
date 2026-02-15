"""
DSSMS 1年バックテスト（戦略選択ログ付き）

目的:
- DynamicStrategySelector の戦略選択ログを完全記録
- MarketAnalyzer の分析結果を記録
- FIFO決済の詳細を記録

実行方法:
    python run_dssms_with_detailed_logs.py

出力:
    - output/dssms_integration/dssms_YYYYMMDD_HHMMSS/ (通常出力)
    - logs/dssms_detailed_YYYYMMDD_HHMMSS.log (詳細ログ)
    - strategy_selection_log_1year.txt (戦略選択ログ抽出版)
    - market_analysis_log_1year.txt (市場分析ログ抽出版)
    - fifo_execution_log_1year.txt (FIFO決済ログ抽出版)

Author: Backtest Project Team
Created: 2026-02-15
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
import argparse


def setup_detailed_logging():
    """詳細ログ設定（戦略選択・市場分析・FIFO記録）"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dssms_detailed_{timestamp}.log"
    
    # ルートロガーをDEBUGレベルに設定
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # ファイルハンドラー（詳細ログ）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # コンソールハンドラー（INFO以上）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # DynamicStrategySelector, MarketAnalyzer の詳細ログを有効化
    logging.getLogger('DynamicStrategySelector').setLevel(logging.DEBUG)
    logging.getLogger('MarketAnalyzer').setLevel(logging.DEBUG)
    logging.getLogger('DSSMS_Integrated').setLevel(logging.DEBUG)
    logging.getLogger('DSSIntegratedBacktester').setLevel(logging.DEBUG)
    
    print(f"詳細ログファイル: {log_file}")
    print("戦略選択・市場分析・FIFO決済ログを全て記録します")
    
    return log_file


def extract_logs_from_detailed_log(log_file: Path):
    """詳細ログから3種類のログを抽出"""
    if not log_file.exists():
        print(f"ログファイルが見つかりません: {log_file}")
        return
    
    strategy_selection_lines = []
    market_analysis_lines = []
    fifo_execution_lines = []
    
    keywords_strategy = [
        'SCORE_DETAIL', 'SELECTION_RESULT', 'Strategy score', 'Selected strategy', 
        'DynamicStrategySelector', '戦略選択', 'スコア'
    ]
    keywords_market = [
        'MarketAnalyzer', 'market_condition', 'volatility', 'トレンド', 'MARKET_ANALYSIS'
    ]
    keywords_fifo = [
        'FIFO', '決済', 'max_positions', 'POSITION_LIMIT', 'FORCED_CLOSE', 'FINAL_CLOSE'
    ]
    
    print("ログファイルを解析中...")
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 戦略選択ログ
            if any(keyword in line for keyword in keywords_strategy):
                strategy_selection_lines.append(line)
            
            # 市場分析ログ
            if any(keyword in line for keyword in keywords_market):
                market_analysis_lines.append(line)
            
            # FIFO決済ログ
            if any(keyword in line for keyword in keywords_fifo):
                fifo_execution_lines.append(line)
    
    # 結果を保存
    output_dir = Path(".")
    
    if strategy_selection_lines:
        output_file = output_dir / "strategy_selection_log_1year.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"元のログファイル: {log_file}\n")
            f.write(f"抽出日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"抽出行数: {len(strategy_selection_lines)}\n")
            f.write("=" * 80 + "\n\n")
            f.writelines(strategy_selection_lines)
        print(f"戦略選択ログ保存: {output_file} ({len(strategy_selection_lines)}行)")
    else:
        print("戦略選択ログが見つかりませんでした")
    
    if market_analysis_lines:
        output_file = output_dir / "market_analysis_log_1year.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"元のログファイル: {log_file}\n")
            f.write(f"抽出日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"抽出行数: {len(market_analysis_lines)}\n")
            f.write("=" * 80 + "\n\n")
            f.writelines(market_analysis_lines)
        print(f"市場分析ログ保存: {output_file} ({len(market_analysis_lines)}行)")
    else:
        print("市場分析ログが見つかりませんでした")
    
    if fifo_execution_lines:
        output_file = output_dir / "fifo_execution_log_1year.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"元のログファイル: {log_file}\n")
            f.write(f"抽出日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"抽出行数: {len(fifo_execution_lines)}\n")
            f.write("=" * 80 + "\n\n")
            f.writelines(fifo_execution_lines)
        print(f"FIFO決済ログ保存: {output_file} ({len(fifo_execution_lines)}行)")
    else:
        print("FIFO決済ログが見つかりませんでした")


def main():
    parser = argparse.ArgumentParser(description='DSSMS 1年バックテスト（詳細ログ付き）')
    parser.add_argument('--start-date', default='2024-01-01', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--extract-only', action='store_true', 
                       help='既存ログから抽出のみ（実行しない）')
    
    args = parser.parse_args()
    
    # ログ抽出のみの場合
    if args.extract_only:
        log_dir = Path("logs")
        log_files = sorted(log_dir.glob("dssms_detailed_*.log"), reverse=True)
        if not log_files:
            print("エラー: dssms_detailed_*.log ファイルが見つかりません")
            return 1
        
        latest_log = log_files[0]
        print(f"最新のログファイルを使用: {latest_log}")
        extract_logs_from_detailed_log(latest_log)
        return 0
    
    # 詳細ログ設定
    log_file = setup_detailed_logging()
    
    print("\nDSSMS 1年バックテスト開始...")
    print(f"期間: {args.start_date} - {args.end_date}")
    print("="* 80)
    
    try:
        # DSSMS実行
        from src.dssms.dssms_integrated_main import DSSIntegratedBacktester
        
        backtester = DSSIntegratedBacktester(
            dssms_backtest_start_date=args.start_date,
            dssms_backtest_end_date=args.end_date,
            initial_cash=1000000
        )
        
        backtester.run_full_backtest()
        
        print("\nバックテスト完了")
        print("="* 80)
        
        # ログから抽出
        print("\n詳細ログから情報を抽出中...")
        extract_logs_from_detailed_log(log_file)
        
        print("\n処理完了")
        return 0
        
    except Exception as e:
        logging.error(f"実行エラー: {e}", exc_info=True)
        print(f"\nエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
