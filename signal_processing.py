"""
Module: signal_processing
Description:
  シグナル処理のための共通モジュール。
  main.pyとsrc/main.pyの両方から参照可能な共通機能を提供します。

Author: imega
Created: 2025-10-15
"""

import pandas as pd
import logging
from typing import Dict, Any

# ロガーの設定（既存のロガー設定がある場合はそれを利用）
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

def detect_exit_anomalies(strategy_result: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """
    異常エグジット検出（TODO #4対応、同日Entry/Exit問題対応追加）
    
    Args:
        strategy_result: 戦略の実行結果データフレーム
        strategy_name: 戦略名
        
    Returns:
        Dict[str, Any]: 検出された異常に関する情報
    """
    anomaly_info = {
        'is_abnormal': False,
        'anomaly_type': 'normal',
        'exit_entry_ratio': 0.0,
        'total_exits': 0,
        'total_entries': 0,
        'same_day_entry_exits': 0  # 同日Entry/Exit問題件数
    }
    
    if 'Entry_Signal' in strategy_result.columns and 'Exit_Signal' in strategy_result.columns:
        total_entries = (strategy_result['Entry_Signal'] == 1).sum()
        total_exits = (strategy_result['Exit_Signal'] == 1).sum()  # Exit_Signalも1で統一
        
        # 同日Entry/Exit問題の検出（新規）
        same_day_mask = (strategy_result['Entry_Signal'] == 1) & (strategy_result['Exit_Signal'] == 1)
        same_day_count = same_day_mask.sum()
        anomaly_info['same_day_entry_exits'] = int(same_day_count)
        
        anomaly_info['total_entries'] = int(total_entries)
        anomaly_info['total_exits'] = int(total_exits)
        
        # 同日Entry/Exit問題の判定（新規）
        if same_day_count > 0:
            anomaly_info['is_abnormal'] = True
            anomaly_info['anomaly_type'] = 'same_day_entry_exit'
            print(f"[WARNING] {strategy_name}: 同日Entry/Exit問題検出 ({same_day_count}件)")
        elif total_entries > 0:
            exit_entry_ratio = total_exits / total_entries
            anomaly_info['exit_entry_ratio'] = round(exit_entry_ratio, 2)
            
            # 異常判定基準（TODO #4調査結果基準）
            if exit_entry_ratio > 5.0:  # 5倍以上は明らかに異常
                anomaly_info['is_abnormal'] = True
                anomaly_info['anomaly_type'] = 'excessive_exits'
                print(f"[CRITICAL] {strategy_name}: 異常な大量エグジット検出 (比率: {exit_entry_ratio:.1f})")
            elif exit_entry_ratio > 2.0:  # 2倍超は要注意
                anomaly_info['is_abnormal'] = True
                anomaly_info['anomaly_type'] = 'high_exit_ratio'
                print(f"[WARNING] {strategy_name}: 高いエグジット比率検出 (比率: {exit_entry_ratio:.1f})")
    
    return anomaly_info

def check_same_day_entry_exit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    同日にエントリーとエグジットが両方発生している日を検出
    
    Args:
        df: 検証対象のデータフレーム
        
    Returns:
        Dict[str, Any]: 検出結果の情報
    """
    result = {
        'has_same_day_signals': False,
        'same_day_count': 0,
        'dates': []
    }
    
    if 'Entry_Signal' in df.columns and 'Exit_Signal' in df.columns:
        # 同じ日にEntry=1とExit=1（または-1）の両方がある日を検出
        same_day_mask = (df['Entry_Signal'] == 1) & ((df['Exit_Signal'] == 1) | (df['Exit_Signal'] == -1))
        same_day_count = same_day_mask.sum()
        
        if same_day_count > 0:
            result['has_same_day_signals'] = True
            result['same_day_count'] = int(same_day_count)
            result['dates'] = df[same_day_mask].index.tolist()
            
            logger.warning(f"同日エントリー/エグジット検出: {same_day_count}件")
            for date in result['dates'][:5]:  # 最初の5件だけログ出力
                logger.warning(f"  - 日付: {date}")
            
            if len(result['dates']) > 5:
                logger.warning(f"  - ...他 {len(result['dates']) - 5} 件")
    
    return result

def filter_same_day_exit_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    同日エントリー/エグジット問題の検出のみを行う（修正は行わない）
    
    バックテスト基本理念に基づき、シグナル自体は修正せず、
    検出・ログ出力のみを行う
    
    Args:
        df: データフレーム
        
    Returns:
        pd.DataFrame: 元のデータフレーム（変更なし）
    """
    # 検出のみを行い、修正は行わない
    if 'Entry_Signal' in df.columns and 'Exit_Signal' in df.columns:
        # 同じ日にEntry=1とExit=1（または-1）の両方がある日を検出
        same_day_mask = (df['Entry_Signal'] == 1) & ((df['Exit_Signal'] == 1) | (df['Exit_Signal'] == -1))
        same_day_count = same_day_mask.sum()
        
        if same_day_count > 0:
            logger.info(f"同日エントリー/エグジット検出: {same_day_count}件")
            same_day_dates = df[same_day_mask].index.tolist()
            for date in same_day_dates[:5]:  # 最初の5件だけログ出力
                logger.info(f"  - 同日エントリー/エグジット日付: {date}")
            
            if len(same_day_dates) > 5:
                logger.info(f"  - ...他 {len(same_day_dates) - 5} 件")
    
    # 元のデータフレームをそのまま返す（修正しない）
    return df