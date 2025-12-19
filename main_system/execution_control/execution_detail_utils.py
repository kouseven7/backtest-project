"""
execution_detail_utils.py - 実行詳細の共通処理ユーティリティ

execution_details（実行済み注文の詳細情報）を扱うための共通関数群。
ComprehensivePerformanceAnalyzerとComprehensiveReporterで統一的に使用し、
BUY/SELL注文の抽出ロジックを一元化します。

主な機能:
- execution_detailsからBUY/SELL注文の抽出
- 有効な取引の判定ロジック（通常取引・強制決済の両対応）
- ペアリング前の前処理と検証
- 統一的なログ出力とエラーハンドリング
- フォールバック禁止の徹底（copilot-instructions.md準拠）

統合コンポーネント:
- ComprehensivePerformanceAnalyzer: パフォーマンス分析での取引抽出
- ComprehensiveReporter: レポート生成での取引抽出
- StrategyExecutionManager: execution_details生成元（参照のみ）

セーフティ機能/注意事項:
- ダミーデータ生成禁止: 不足データに対してフォールバックしない
- 型チェック厳密化: dict型でない要素は警告ログを出してスキップ
- 強制決済対応: status='force_closed'も有効な取引として認識
- 詳細ログ出力: スキップされた注文も記録（デバッグ用）

Author: Backtest Project Team
Created: 2025-11-12
Last Modified: 2025-12-19
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# プロジェクトパス追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

# モジュールレベルロガー
logger = setup_logger("ExecutionDetailUtils", log_file="logs/execution_detail_utils.log")


def extract_buy_sell_orders(
    execution_details: List[Dict[str, Any]],
    logger_instance: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    execution_detailsからBUY/SELL注文を抽出
    
    copilot-instructions.md準拠:
    - 実データのみ抽出（フォールバック禁止）
    - 型チェック厳密化
    - 詳細ログ出力
    
    Args:
        execution_details: 実行済み注文のリスト
        logger_instance: ロガーインスタンス（Noneの場合はモジュールロガー使用）
    
    Returns:
        (buy_orders, sell_orders) のタプル
    """
    log = logger_instance if logger_instance else logger
    
    buy_orders = []
    sell_orders = []
    skipped_count = 0
    
    log.info(f"[EXTRACT_BUY_SELL] Processing {len(execution_details)} execution details")
    
    for idx, detail in enumerate(execution_details):
        # 型チェック
        if not isinstance(detail, dict):
            log.warning(
                f"[TYPE_ERROR] execution_detail[{idx}] is not dict: {type(detail)}"
            )
            skipped_count += 1
            continue
        
        # 有効な取引のみを抽出
        if not is_valid_trade(detail, logger_instance=log):
            skipped_count += 1
            continue
        
        # BUY/SELL分類
        action = detail.get('action', '').upper()
        if action == 'BUY':
            buy_orders.append(detail)
            log.debug(
                f"[BUY_DETECTED] index={idx}, symbol={detail.get('symbol')}, "
                f"status={detail.get('status')}, quantity={detail.get('quantity')}"
            )
        elif action == 'SELL':
            sell_orders.append(detail)
            log.debug(
                f"[SELL_DETECTED] index={idx}, symbol={detail.get('symbol')}, "
                f"status={detail.get('status')}, quantity={detail.get('quantity')}"
            )
        else:
            log.warning(
                f"[UNKNOWN_ACTION] index={idx}, action={action}, "
                f"symbol={detail.get('symbol')}"
            )
            skipped_count += 1
    
    log.info(
        f"[EXTRACT_RESULT] BUY={len(buy_orders)}, SELL={len(sell_orders)}, "
        f"Skipped={skipped_count}, Total={len(execution_details)}"
    )
    
    return buy_orders, sell_orders


def is_valid_trade(
    detail: Dict[str, Any],
    logger_instance: Optional[Any] = None
) -> bool:
    """
    有効な取引かどうかを判定
    
    判定基準:
    1. successフラグがTrue
    2. actionが'BUY'または'SELL'
    
    copilot-instructions.md準拠:
    - status='force_closed'も有効（強制決済対応）
    - 実データのみ判定（デフォルト値でのフォールバック禁止）
    
    Args:
        detail: 注文詳細
        logger_instance: ロガーインスタンス（Noneの場合はモジュールロガー使用）
    
    Returns:
        有効な取引ならTrue
    """
    log = logger_instance if logger_instance else logger
    
    # successフラグ確認
    success = detail.get('success', False)
    if not success:
        log.debug(
            f"[INVALID_TRADE] success=False, action={detail.get('action')}, "
            f"symbol={detail.get('symbol')}, status={detail.get('status')}"
        )
        return False
    
    # actionが設定されているか
    action = detail.get('action', '').upper()
    if action not in ['BUY', 'SELL']:
        log.debug(
            f"[INVALID_TRADE] Invalid action={action}, "
            f"symbol={detail.get('symbol')}, status={detail.get('status')}"
        )
        return False
    
    # 2025-12-19修正: execution_typeチェック（通常取引と強制決済のみ抽出）
    # 後方互換性対応: execution_typeなしの場合はデフォルトで'trade'とみなす
    # 修正理由: force_close（強制決済）は実際の損益を伴う取引のためCSVに含める必要がある
    # 除外対象: switch（銘柄切替の記録用）のみ
    execution_type = detail.get('execution_type', 'trade')
    if execution_type not in ['trade', 'force_close']:
        log.debug(
            f"[SKIPPED_NON_TRADE] execution_type={execution_type}, "
            f"symbol={detail.get('symbol')}, action={action}"
        )
        return False
    
    # Phase 5-B-12: statusチェックは行わない
    # 理由: status='executed'と'force_closed'の両方を許可するため
    # actionとsuccessのみで判定
    
    return True


def validate_buy_sell_pairing(
    buy_orders: List[Dict[str, Any]],
    sell_orders: List[Dict[str, Any]],
    logger_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    BUY/SELLペアリングの妥当性を検証
    
    Args:
        buy_orders: BUY注文リスト
        sell_orders: SELL注文リスト
        logger_instance: ロガーインスタンス
    
    Returns:
        検証結果辞書:
        {
            'is_valid': bool,
            'buy_count': int,
            'sell_count': int,
            'difference': int,
            'excess_type': str ('BUY' or 'SELL' or 'NONE'),
            'paired_count': int,
            'warning_message': str or None
        }
    """
    log = logger_instance if logger_instance else logger
    
    buy_count = len(buy_orders)
    sell_count = len(sell_orders)
    difference = abs(buy_count - sell_count)
    paired_count = min(buy_count, sell_count)
    
    if buy_count == sell_count:
        log.info(
            f"[PAIRING_OK] Perfect pairing: BUY={buy_count}, SELL={sell_count}"
        )
        return {
            'is_valid': True,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'difference': 0,
            'excess_type': 'NONE',
            'paired_count': paired_count,
            'warning_message': None
        }
    else:
        excess_type = 'BUY' if buy_count > sell_count else 'SELL'
        warning_msg = (
            f"BUY/SELLペア不一致: BUY={buy_count}, SELL={sell_count} "
            f"(差分={difference}, 超過={excess_type}). "
            f"ペアリング可能な{paired_count}件のみ処理します。"
        )
        
        log.warning(f"[PAIRING_MISMATCH] {warning_msg}")
        
        return {
            'is_valid': False,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'difference': difference,
            'excess_type': excess_type,
            'paired_count': paired_count,
            'warning_message': warning_msg
        }


def get_execution_detail_summary(
    execution_details: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    execution_detailsの概要情報を取得（デバッグ用）
    
    Args:
        execution_details: 実行詳細リスト
    
    Returns:
        概要情報辞書
    """
    if not execution_details:
        return {
            'total_count': 0,
            'status_distribution': {},
            'action_distribution': {},
            'success_count': 0,
            'failure_count': 0
        }
    
    status_dist = {}
    action_dist = {}
    success_count = 0
    failure_count = 0
    
    for detail in execution_details:
        if not isinstance(detail, dict):
            continue
        
        # ステータス集計
        status = detail.get('status', 'unknown')
        status_dist[status] = status_dist.get(status, 0) + 1
        
        # アクション集計
        action = detail.get('action', 'unknown')
        action_dist[action] = action_dist.get(action, 0) + 1
        
        # 成功/失敗集計
        if detail.get('success', False):
            success_count += 1
        else:
            failure_count += 1
    
    return {
        'total_count': len(execution_details),
        'status_distribution': status_dist,
        'action_distribution': action_dist,
        'success_count': success_count,
        'failure_count': failure_count
    }
