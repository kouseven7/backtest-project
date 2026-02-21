"""
Module: Logger Configuration
File: logger_config.py
Description: 
  ログ設定を行うためのモジュールです。標準出力やファイル出力のロガーを構築します。

Author: imega
Created: 2023-04-01
Modified: 2026-02-15 - 戦略分析用詳細ログ機能を追加

Dependencies:
  - logging
  - sys
  - os
  
Changelog:
  2026-02-15: setup_detailed_strategy_loggerを追加
    - 戦略選択・市場分析・FIFO決済の詳細ログをファイル出力
    - タグベースのフィルタリング機能
    - バックテスト実行ごとに個別ログファイル作成
  2026-02-15: add_detailed_handlers_to_existing_loggersを追加
    - 既存ロガーへのFileHandler追加機能
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List


def setup_logger(name: str, level=logging.INFO, log_file: str = None) -> logging.Logger:
    """
    指定した名前とログレベルでロガーを設定して返す。
    - 標準出力（sys.stdout）にログを出力。
    - log_fileが指定されていれば、ファイルにもログを出力する。
    - 既にハンドラーが設定されている場合は重複して追加しない。
    
    Args:
        name: ロガー名
        level: ログレベル(デフォルト: INFO)
        log_file: ログファイルパス(オプション)
        
    Returns:
        logging.Logger: 設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')

    # 標準出力用ハンドラーを追加（重複防止）
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # log_fileが指定されている場合はFileHandlerを追加（重複防止）
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        # ログディレクトリが存在しない場合は作成
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # UTF-8エンコーディング指定（Windows環境でのShift_JIS問題対策）
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ==================== 2026-02-15追加: 戦略分析用詳細ログ機能 ====================

class LogTagFilter(logging.Filter):
    """
    ログメッセージのタグでフィルタリング
    
    使用例:
        handler.addFilter(LogTagFilter(['SCORE_DETAIL', 'SELECTION_RESULT']))
    """
    
    def __init__(self, allowed_tags: List[str]):
        super().__init__()
        self.allowed_tags = allowed_tags
    
    def filter(self, record):
        """
        ログレコードをフィルタリング
        
        Args:
            record: ログレコード
            
        Returns:
            bool: Trueならログを通過、Falseなら除外
        """
        message = record.getMessage()
        return any(tag in message for tag in self.allowed_tags)


def setup_detailed_strategy_logger(
    name: str,
    output_dir: str,
    run_id: Optional[str] = None,
    level=logging.INFO,
    enable_console: bool = True
) -> logging.Logger:
    """
    戦略分析用の詳細ログ設定
    
    バックテスト実行時に以下の詳細ログファイルを自動作成:
    1. strategy_selection_{run_id}.log - 戦略選択の詳細(各戦略のスコア)
    2. market_analysis_{run_id}.log - 市場分析の詳細(トレンド、レジーム判定)
    3. fifo_execution_{run_id}.log - FIFO決済の詳細(決済理由、ポジション情報)
    4. dssms_all_detailed_{run_id}.log - 上記すべてを含む統合ログ
    
    ログタグ:
    - [SCORE_DETAIL] - 各戦略のスコア計算結果
    - [SELECTION_RESULT] - 最終的に選択された戦略
    - [STRATEGY_SELECTION] - 戦略選択サマリー
    - [MARKET_ANALYSIS] - 市場分析結果
    - [MARKET_TREND] - トレンド判定
    - [MARKET_REGIME] - 市場レジーム判定
    - [FIFO_EXIT] - FIFO決済実行
    - [FIFO_DETAIL] - FIFO決済詳細情報
    
    Args:
        name: ロガー名(例: "DSSMS_Integrated")
        output_dir: ログファイル出力ディレクトリ
        run_id: バックテスト実行ID(省略時は自動生成)
        level: ログレベル(デフォルト: INFO)
        enable_console: コンソール出力を有効にするか(デフォルト: True)
        
    Returns:
        logging.Logger: 設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 実行IDがない場合は自動生成
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 詳細フォーマット(タイムスタンプ + レベル + メッセージ)
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 既存のハンドラーをクリア(重複防止)
    logger.handlers.clear()
    
    # 1. コンソール出力(オプション)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # 2. 統合ログファイル(全ての詳細ログ)
    all_log_file = output_path / f'dssms_all_detailed_{run_id}.log'
    all_handler = logging.FileHandler(str(all_log_file), encoding='utf-8')
    all_handler.setFormatter(detailed_formatter)
    all_handler.setLevel(level)
    logger.addHandler(all_handler)
    
    # 3. 戦略選択専用ログ(フィルター付き)
    strategy_log_file = output_path / f'strategy_selection_{run_id}.log'
    strategy_handler = logging.FileHandler(str(strategy_log_file), encoding='utf-8')
    strategy_handler.setFormatter(detailed_formatter)
    strategy_handler.setLevel(level)
    strategy_handler.addFilter(LogTagFilter([
        'SCORE_DETAIL',
        'SELECTION_RESULT', 
        'STRATEGY_SELECTION',
        'STRATEGY_'
    ]))
    logger.addHandler(strategy_handler)
    
    # 4. 市場分析専用ログ(フィルター付き)
    market_log_file = output_path / f'market_analysis_{run_id}.log'
    market_handler = logging.FileHandler(str(market_log_file), encoding='utf-8')
    market_handler.setFormatter(detailed_formatter)
    market_handler.setLevel(level)
    market_handler.addFilter(LogTagFilter([
        'MARKET_ANALYSIS',
        'MARKET_TREND',
        'MARKET_REGIME',
        'MARKET_'
    ]))
    logger.addHandler(market_handler)
    
    # 5. FIFO決済専用ログ(フィルター付き)
    fifo_log_file = output_path / f'fifo_execution_{run_id}.log'
    fifo_handler = logging.FileHandler(str(fifo_log_file), encoding='utf-8')
    fifo_handler.setFormatter(detailed_formatter)
    fifo_handler.setLevel(level)
    fifo_handler.addFilter(LogTagFilter([
        'FIFO_EXIT',
        'FIFO_DETAIL',
        'FIFO_',
        'FORCED_EXIT',
        'POSITION_LIMIT'
    ]))
    logger.addHandler(fifo_handler)
    
    # ログ設定完了メッセージ
    logger.info(f"[LOG_SETUP] Detailed strategy logging initialized")
    logger.info(f"[LOG_SETUP] Output directory: {output_dir}")
    logger.info(f"[LOG_SETUP] Run ID: {run_id}")
    logger.info(f"[LOG_SETUP] Log files created:")
    logger.info(f"[LOG_SETUP]   - All detailed: {all_log_file.name}")
    logger.info(f"[LOG_SETUP]   - Strategy selection: {strategy_log_file.name}")
    logger.info(f"[LOG_SETUP]   - Market analysis: {market_log_file.name}")
    logger.info(f"[LOG_SETUP]   - FIFO execution: {fifo_log_file.name}")
    
    return logger


def add_detailed_handlers_to_existing_loggers(output_dir: str, run_id: str):
    """
    既存のロガーに詳細ログハンドラーを追加
    
    DynamicStrategySelectorやMarketAnalyzerなどのロガーに
    詳細戦略ログのFileHandlerを追加します
    
    Args:
        output_dir: ログ出力ディレクトリ
        run_id: 実行ID
    """
    output_path = Path(output_dir)
    
    # 追加: ディレクトリが存在しない場合は作成
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ★★★ デバッグ出力追加 ★★★
    print("\n" + "="*80)
    print("[DEBUG] add_detailed_handlers_to_existing_loggers() called")
    print(f"[DEBUG] output_dir: {output_dir}")
    print(f"[DEBUG] run_id: {run_id}")
    print(f"[DEBUG] output_path: {output_path}")
    print(f"[DEBUG] output_path.exists(): {output_path.exists()}")
    
    print("\n[DEBUG] All registered loggers (filtered by keywords):")
    matching_count = 0
    for name in sorted(logging.root.manager.loggerDict.keys()):
        if any(keyword in name.lower() for keyword in ['strategy', 'market', 'dssms', 'main_system']):
            logger_obj = logging.getLogger(name)
            matching_count += 1
            print(f"\n  [{matching_count}] {name}")
            print(f"      handlers: {len(logger_obj.handlers)}")
            if logger_obj.handlers:
                for i, h in enumerate(logger_obj.handlers, 1):
                    print(f"        [{i}] {type(h).__name__}")
            print(f"      level: {logger_obj.level} ({logging.getLevelName(logger_obj.level)})")
            print(f"      propagate: {logger_obj.propagate}")
    
    print(f"\n[DEBUG] Total matching loggers: {matching_count}")
    
    print("\n[DEBUG] Target loggers existence check:")
    target_loggers_check = [
        'main_system.strategy_selection.dynamic_strategy_selector',
        'main_system.market_analysis.market_analyzer',
        'dynamic_strategy_selector',
        'market_analyzer',
        'DynamicStrategySelector',
        'MarketAnalyzer',
        'DSSMS_Integrated'
    ]
    
    found_count = 0
    for target in target_loggers_check:
        exists = target in logging.root.manager.loggerDict
        status = 'EXISTS' if exists else 'NOT FOUND'
        print(f"  - {target:60s} : {status}")
        if exists:
            found_count += 1
    
    print(f"\n[DEBUG] Found {found_count}/{len(target_loggers_check)} target loggers")
    print("="*80 + "\n")
    # ★★★ デバッグ出力終了 ★★★
    
    # 詳細フォーマット
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ログファイルパス
    all_log = output_path / f'dssms_all_detailed_{run_id}.log'
    strategy_log = output_path / f'strategy_selection_{run_id}.log'
    market_log = output_path / f'market_analysis_{run_id}.log'
    fifo_log = output_path / f'fifo_execution_{run_id}.log'
    
    # 対象ロガー名リスト
    target_loggers = [
        'main_system.strategy_selection.dynamic_strategy_selector',
        'main_system.market_analysis.market_analyzer',
        'dynamic_strategy_selector',
        'market_analyzer',
    ]
    
    handlers_added = 0
    for logger_name in target_loggers:
        logger = logging.getLogger(logger_name)
        
        # ロガーレベルが未設定の場合のみ設定
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        # handlers の有無に関わらず処理を継続
        
        # 既存のFileHandlerのみ削除(重複防止)
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
        
        # 1. 統合ログ
        all_handler = logging.FileHandler(str(all_log), encoding='utf-8')
        all_handler.setFormatter(detailed_formatter)
        all_handler.setLevel(logging.INFO)
        logger.addHandler(all_handler)
        
        # 2. 戦略選択専用ログ
        strategy_handler = logging.FileHandler(str(strategy_log), encoding='utf-8')
        strategy_handler.setFormatter(detailed_formatter)
        strategy_handler.setLevel(logging.INFO)
        strategy_handler.addFilter(LogTagFilter([
            'SCORE_DETAIL', 'SELECTION_RESULT', 'STRATEGY_'
        ]))
        logger.addHandler(strategy_handler)
        
        # 3. 市場分析専用ログ
        market_handler = logging.FileHandler(str(market_log), encoding='utf-8')
        market_handler.setFormatter(detailed_formatter)
        market_handler.setLevel(logging.INFO)
        market_handler.addFilter(LogTagFilter([
            'MARKET_ANALYSIS', 'MARKET_TREND', 'MARKET_'
        ]))
        logger.addHandler(market_handler)
        
        # 4. FIFO決済専用ログ
        fifo_handler = logging.FileHandler(str(fifo_log), encoding='utf-8')
        fifo_handler.setFormatter(detailed_formatter)
        fifo_handler.setLevel(logging.INFO)
        fifo_handler.addFilter(LogTagFilter([
            'FIFO_', 'FORCED_EXIT', 'POSITION_LIMIT'
        ]))
        logger.addHandler(fifo_handler)
        
        # ログレベル保証
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        
        handlers_added += 1
    
    # 確認ログ
    main_logger = logging.getLogger("DSSMS_Integrated")
    main_logger.info(f"[LOG_SETUP] Added detailed handlers to {handlers_added} existing loggers")


def get_detailed_log_files(output_dir: str, run_id: str) -> dict:
    """
    詳細ログファイルのパス一覧を取得
    
    Args:
        output_dir: 出力ディレクトリ
        run_id: 実行ID
        
    Returns:
        dict: ログファイルパス辞書
    """
    output_path = Path(output_dir)
    
    return {
        'all_detailed': str(output_path / f'dssms_all_detailed_{run_id}.log'),
        'strategy_selection': str(output_path / f'strategy_selection_{run_id}.log'),
        'market_analysis': str(output_path / f'market_analysis_{run_id}.log'),
        'fifo_execution': str(output_path / f'fifo_execution_{run_id}.log')
    }


# ==================== 便利関数: 標準化されたログ出力 ====================

def log_strategy_selection(
    logger: logging.Logger,
    ticker: str,
    scores: dict,
    selected: str,
    market_regime: str = None
):
    """戦略選択ログを標準化された形式で出力"""
    regime_info = f" | Regime={market_regime}" if market_regime else ""
    logger.info(f"[STRATEGY_SELECTION] Ticker={ticker} | Selected={selected}{regime_info}")
    logger.info(f"[SCORE_DETAIL] {ticker} strategy scores:")
    
    for strategy, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        marker = "★" if strategy == selected else " "
        logger.info(f"[SCORE_DETAIL]   {marker} {strategy:30s}: {score:.4f}")


def log_market_analysis(
    logger: logging.Logger,
    ticker: str,
    market_regime: str,
    confidence: float,
    trend: str = None,
    perfect_order: bool = None
):
    """市場分析ログを標準化された形式で出力"""
    logger.info(f"[MARKET_ANALYSIS] Ticker={ticker} | Regime={market_regime} | Confidence={confidence:.2f}")
    
    if trend:
        logger.info(f"[MARKET_TREND] Trend={trend}")
    
    if perfect_order is not None:
        logger.info(f"[MARKET_PERFECT_ORDER] PerfectOrder={perfect_order}")


def log_fifo_exit(
    logger: logging.Logger,
    symbol: str,
    reason: str,
    entry_date: str,
    holding_days: int,
    current_pnl: float
):
    """FIFO決済ログを標準化された形式で出力"""
    logger.info(f"[FIFO_EXIT] Symbol={symbol} | Reason={reason}")
    logger.info(f"[FIFO_DETAIL] EntryDate={entry_date} | HoldingDays={holding_days} | CurrentPnL={current_pnl:,.0f}")


# ==================== テスト用メイン ====================

if __name__ == "__main__":
    print("=" * 80)
    print("Logger Config - Detailed Strategy Logging Test")
    print("=" * 80)
    
    test_output_dir = "output/test_logging"
    test_run_id = "test_20260215"
    
    logger = setup_detailed_strategy_logger(
        "TestLogger",
        test_output_dir,
        run_id=test_run_id
    )
    
    print("\n--- Testing Strategy Selection Logs ---")
    scores = {
        'GCStrategy': 0.75,
        'VWAPBreakoutStrategy': 0.62,
        'MomentumInvestingStrategy': 0.58,
        'BreakoutStrategy': 0.45,
        'ContrarianStrategy': 0.33
    }
    log_strategy_selection(logger, "6301", scores, "GCStrategy", "strong_uptrend")
    
    print("\n--- Testing Market Analysis Logs ---")
    log_market_analysis(logger, "6301", "strong_uptrend", 0.82, "uptrend", True)
    
    print("\n--- Testing FIFO Exit Logs ---")
    log_fifo_exit(logger, "8354", "max_positions_reached", "2024-07-01", 15, -15000)
    
    print("\n--- Log Files Created ---")
    log_files = get_detailed_log_files(test_output_dir, test_run_id)
    for name, path in log_files.items():
        exists = "✓" if Path(path).exists() else "✗"
        print(f"{exists} {name}: {path}")
    
    print("\n" + "=" * 80)
    print("Test completed. Check files in:", test_output_dir)
    print("=" * 80)