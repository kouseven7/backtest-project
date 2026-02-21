"""
シンプルエグジット戦略検証スクリプト v2.0

Phase 3-6の失敗を踏まえ、カーブフィッティングを回避した段階的検証プロトコル実装。

主な機能:
- Phase 1: 固定パラメータ検証（PF > 1.0達成可能性確認）
- Phase 2: グリッドサーチ（48組み合わせ、TOP 3選定）
- Phase 3: Out-of-Sample検証（汎化性能確認）
- 10銘柄対応（業種分散）
- PF上限3.0制約（過学習排除）
- 統計的有意性チェック（取引数 >= 20）

統合コンポーネント:
- GCStrategy: エントリー固定（ゴールデンクロス）
- SimpleExitStrategy: エグジット単体検証用
- data_fetcher: yfinance統合+CSV cache経由でのデータ供給
- ComprehensiveReporter: CSV+JSON統一出力

セーフティ機能/注意事項:
- PF上限3.0制約必須（Phase 3-6でPF=121.07が-98.9%崩壊）
- 10銘柄中8銘柄でPF > 1.0（汎化性能確保）
- Win Rate >= 40%（エントリー品質前提条件）
- Out-of-Sample劣化率 < 30%（過学習検出）
- 取引数 >= 20（統計的有意性）

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import json
import itertools
from typing import Dict, List, Tuple

# プロジェクトルート追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 必要なモジュールインポート
from data_fetcher import get_parameters_and_data
from strategies.gc_strategy_signal import GCStrategy

# talib インポート（ATR計算用）
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    import warnings
    warnings.warn("talib未インストール。ATR計算は簡易版で代用します。")

# ロガー設定
import config.logger_config
logger = config.logger_config.setup_logger(
    "ValidateExitSimpleV2",
    log_file="logs/validate_exit_simple_v2.log"
)


# ==================== 定数定義 ====================

# Phase 1.7: 9銘柄検証（武田薬品除外 - 既にPhase 1.6で分析済み）
VALIDATION_TICKERS = [
    "7203.T",  # トヨタ自動車（自動車）
    "9984.T",  # ソフトバンクグループ（通信）
    "8306.T",  # 三菱UFJ FG（銀行）
    "6758.T",  # ソニーグループ（電機）
    "9983.T",  # ファーストリテイリング（小売）
    "6501.T",  # 日立製作所（電機）
    "8001.T",  # 伊藤忠商事（商社）
    "4063.T",  # 信越化学工業（化学）
    "6861.T"   # キーエンス（電機）
]

# Phase 1: シンプルルール（固定パラメータ）
PHASE1_PARAMS = {
    'stop_loss_pct': 0.05,          # 固定損切5%
    'trailing_stop_pct': 0.10,      # 固定トレーリング10%
    'take_profit_pct': None         # 利確なし（トレンドを追う）
}

# Phase 1.5: 微調整グリッド（Phase 1失敗時）
# EXIT_STRATEGY_REDESIGN_V2.md推奨範囲: 損切3-7%、トレーリング5-15%
PHASE1_5_PARAM_GRID = {
    'stop_loss_pct': [0.03, 0.05, 0.07],       # 損切3パターン
    'trailing_stop_pct': [0.05, 0.10, 0.15],   # トレーリング3パターン
    'take_profit_pct': [None]                  # 利確なし（シンプル維持）
}
# 合計: 3 × 3 × 1 = 9組み合わせ

# Phase 1.6: トレーリング拡張グリッド（ペイオフレシオ分析用）
# PAYOFF_RATIO_EXIT_ANALYSIS_PROJECT.md: トレーリング20%,25%,30%を追加
PHASE1_6_PARAM_GRID = {
    'stop_loss_pct': [0.03, 0.05, 0.07],                           # 損切3パターン
    'trailing_stop_pct': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],     # トレーリング6パターン
    'take_profit_pct': [None]                                      # 利確なし（シンプル維持）
}
# 合計: 3 × 6 × 1 = 18組み合わせ

# Phase 2: グリッドサーチパラメータ空間
PHASE2_PARAM_GRID = {
    'stop_loss_pct': [0.03, 0.05, 0.07, 0.10],      # 損切4パターン
    'trailing_stop_pct': [0.05, 0.08, 0.10, 0.15],  # トレーリング4パターン
    'take_profit_pct': [None, 0.15, 0.20]           # 利確3パターン（なし含む）
}
# 合計: 4 × 4 × 3 = 48組み合わせ

# 成功基準（Phase 3-6失敗の教訓）
MAX_PF = 3.0                # PF上限3.0（Phase 3でPF=121.07が-98.9%崩壊）
MINIMUM_WIN_RATE = 0.40     # Win Rate >= 40%（エントリー品質前提条件）
MINIMUM_TRADES = 20         # 取引数 >= 20（統計的有意性）
MIN_PASS_RATE = 0.8         # 10銘柄中8銘柄でPF > 1.0（汎化性能）
MAX_DEGRADATION = 0.30      # Out-of-Sample劣化率 < 30%（過学習検出）

# データ期間設定
IN_SAMPLE_START = "2020-01-01"
IN_SAMPLE_END = "2023-12-31"    # 4年間
OOS_START = "2024-01-01"
OOS_END = "2025-12-31"          # 2年間
WARMUP_DAYS = 150


# ==================== ヘルパー関数 ====================

def calculate_performance_metrics(results_df: pd.DataFrame, exit_params: Dict = None) -> Tuple[Dict, pd.DataFrame]:
    """
    パフォーマンス指標を計算
    
    Args:
        results_df: バックテスト結果データフレーム（Entry_Signal/Exit_Signal/Profit_Loss列を含む）
        exit_params: エグジットパラメータ（stop_loss_pct等を含む）
    
    Returns:
        (パフォーマンス指標辞書, 個別取引履歴DataFrame)
    """
    if results_df.empty or len(results_df) == 0:
        return {
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'avg_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }, pd.DataFrame()
    
    # Profit_Loss列から取引履歴抽出（Exit_Signal == -1の行のみ）
    if 'Profit_Loss' not in results_df.columns or 'Exit_Signal' not in results_df.columns:
        logger.warning("Profit_Loss または Exit_Signal 列が見つかりません")
        return {
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'avg_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }, pd.DataFrame()
    
    # 取引履歴抽出（エグジットした取引のみ）
    trades = results_df[results_df['Exit_Signal'] == -1].copy()
    
    if len(trades) == 0:
        logger.warning("取引履歴が0件です")
        return {
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'avg_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }, pd.DataFrame()
    
    # 勝ちトレード・負けトレード分離
    winning_trades = trades[trades['Profit_Loss'] > 0]
    losing_trades = trades[trades['Profit_Loss'] <= 0]
    
    total_profit = winning_trades['Profit_Loss'].sum() if len(winning_trades) > 0 else 0.0
    total_loss = abs(losing_trades['Profit_Loss'].sum()) if len(losing_trades) > 0 else 0.0
    
    # プロフィットファクター計算
    if total_loss > 0:
        profit_factor = total_profit / total_loss
    elif total_profit > 0:
        profit_factor = float('inf')  # 無限大（損失なし）
    else:
        profit_factor = 0.0  # 利益も損失もなし
    
    # Win Rate計算
    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
    
    # 平均損益
    avg_profit_per_trade = trades['Profit_Loss'].mean() if len(trades) > 0 else 0.0
    
    # 最大ドローダウン（累積損益から計算）
    cumulative = trades['Profit_Loss'].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
    
    # Sharpe比率（簡易版: 年率換算）
    if len(trades) > 1:
        returns = trades['Profit_Loss']
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    # ペイオフレシオ（平均利益 / 平均損失）
    avg_win = winning_trades['Profit_Loss'].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = abs(losing_trades['Profit_Loss'].mean()) if len(losing_trades) > 0 else 0.0
    
    if avg_loss > 0:
        payoff_ratio = avg_win / avg_loss
    elif avg_win > 0:
        payoff_ratio = float('inf')  # 無限大（損失なし）
    else:
        payoff_ratio = 0.0
    
    # エグジット理由カウント（Exit_Reason列が存在する場合）
    exit_reasons = {
        'stop_loss_count': 0,
        'trailing_stop_count': 0,
        'dead_cross_count': 0,
        'force_close_count': 0,
        'take_profit_count': 0,
        'other_count': 0
    }
    
    if 'Exit_Reason' in trades.columns:
        for reason in trades['Exit_Reason']:
            if pd.isna(reason):
                exit_reasons['other_count'] += 1
            elif '損切' in str(reason) or 'stop_loss' in str(reason).lower():
                exit_reasons['stop_loss_count'] += 1
            elif 'トレーリング' in str(reason) or 'trailing' in str(reason).lower():
                exit_reasons['trailing_stop_count'] += 1
            elif 'デッドクロス' in str(reason) or 'dead_cross' in str(reason).lower():
                exit_reasons['dead_cross_count'] += 1
            elif '強制決済' in str(reason) or 'force' in str(reason).lower():
                exit_reasons['force_close_count'] += 1
            elif '利確' in str(reason) or 'profit' in str(reason).lower():
                exit_reasons['take_profit_count'] += 1
            else:
                exit_reasons['other_count'] += 1
    
    # 個別取引履歴の作成（エントリー日・エグジット日を含む + Phase 1必須7項目）
    trade_details = []
    
    # Entry_Price列とExit_Price列が存在する場合、エントリー日を特定
    if 'Entry_Price' in results_df.columns and 'Trade_ID' in results_df.columns:
        for idx, exit_row in trades.iterrows():
            trade_id = exit_row['Trade_ID']
            
            # 同じTrade_IDのエントリー行を探す
            entry_rows = results_df[(results_df['Trade_ID'] == trade_id) & (results_df['Entry_Signal'] == 1)]
            
            if len(entry_rows) > 0:
                entry_date = entry_rows.index[0]
                entry_price = entry_rows['Entry_Price'].iloc[0]
            else:
                entry_date = None
                entry_price = None
            
            exit_date = idx
            exit_price = exit_row.get('Exit_Price', None)
            profit_loss = exit_row['Profit_Loss']
            exit_reason = exit_row.get('Exit_Reason', 'unknown')
            
            # ==================== Phase 1必須7項目計算 ====================
            # 1. holding_days - 保有日数
            if entry_date is not None and exit_date is not None:
                holding_days = (exit_date - entry_date).days
            else:
                holding_days = None
            
            # 2. profit_loss_pct - 損益率（%）
            if entry_price is not None and entry_price != 0:
                profit_loss_pct = (exit_price - entry_price) / entry_price * 100
            else:
                profit_loss_pct = None
            
            # 3-4. max_profit_pct / max_loss_pct - 保有期間中の最大含み益/含み損（%）
            max_profit_pct = None
            max_loss_pct = None
            if entry_date is not None and exit_date is not None and entry_price is not None and entry_price != 0:
                # 保有期間のデータ抽出
                try:
                    hold_data = results_df.loc[entry_date:exit_date]
                    if 'High' in hold_data.columns and 'Low' in hold_data.columns and len(hold_data) > 0:
                        max_profit_pct = (hold_data['High'].max() - entry_price) / entry_price * 100
                        max_loss_pct = (entry_price - hold_data['Low'].min()) / entry_price * 100
                except Exception:
                    pass  # データ不足の場合はNone
            
            # 5-6. entry_atr / entry_atr_pct - エントリー時のATR（絶対値/株価比%）
            entry_atr = None
            entry_atr_pct = None
            if entry_date is not None:
                try:
                    # ATR計算（14日）
                    if TALIB_AVAILABLE and 'High' in results_df.columns and 'Low' in results_df.columns and 'Close' in results_df.columns:
                        # エントリー日までのデータでATR計算
                        entry_idx_pos = results_df.index.get_loc(entry_date)
                        if entry_idx_pos >= 14:  # ATR14に必要な最低行数
                            atr_series = talib.ATR(
                                results_df['High'].iloc[:entry_idx_pos+1].values,
                                results_df['Low'].iloc[:entry_idx_pos+1].values,
                                results_df['Close'].iloc[:entry_idx_pos+1].values,
                                timeperiod=14
                            )
                            entry_atr = atr_series[-1]  # 最新値
                            if entry_price is not None and entry_price != 0:
                                entry_atr_pct = entry_atr / entry_price * 100
                    else:
                        # talib未利用時: 簡易版ATR（High-Low平均）
                        entry_idx_pos = results_df.index.get_loc(entry_date)
                        if entry_idx_pos >= 14:
                            lookback_data = results_df.iloc[entry_idx_pos-14:entry_idx_pos+1]
                            if 'High' in lookback_data.columns and 'Low' in lookback_data.columns:
                                entry_atr = (lookback_data['High'] - lookback_data['Low']).mean()
                                if entry_price is not None and entry_price != 0:
                                    entry_atr_pct = entry_atr / entry_price * 100
                except Exception:
                    pass  # 計算失敗時はNone
            
            # 7. entry_gap_pct - エントリー日の窓（前日終値と当日始値の乖離%）
            entry_gap_pct = None
            if entry_date is not None:
                try:
                    entry_idx_pos = results_df.index.get_loc(entry_date)
                    if entry_idx_pos > 0 and 'Close' in results_df.columns:
                        prev_close = results_df['Close'].iloc[entry_idx_pos - 1]
                        if prev_close is not None and prev_close != 0 and entry_price is not None:
                            entry_gap_pct = (entry_price - prev_close) / prev_close * 100
                except Exception:
                    pass  # 計算失敗時はNone
            # ==============================================================
            
            # ==================== Phase 2優先度高6項目計算 ====================
            # 1. r_multiple - リスク対リターン評価（1R = 損切り幅）
            r_multiple = None
            if profit_loss_pct is not None and exit_params is not None:
                stop_loss_pct_value = exit_params.get('stop_loss_pct', None)
                if stop_loss_pct_value is not None and stop_loss_pct_value != 0:
                    # stop_loss_pctは0.03形式（3%）で保存されているため、100倍して%化
                    r_multiple = profit_loss_pct / (stop_loss_pct_value * 100)
            
            # 2-4. entry_volume / avg_volume_20d / volume_ratio - 流動性分析
            entry_volume = None
            avg_volume_20d = None
            volume_ratio = None
            if entry_date is not None and 'Volume' in results_df.columns:
                try:
                    entry_idx_pos = results_df.index.get_loc(entry_date)
                    entry_volume = results_df['Volume'].iloc[entry_idx_pos]
                    
                    # 20日平均出来高（エントリー日を含む過去20日間）
                    if entry_idx_pos >= 19:  # 20日分のデータが必要
                        volume_data = results_df['Volume'].iloc[entry_idx_pos-19:entry_idx_pos+1]
                        avg_volume_20d = volume_data.mean()
                        
                        # 出来高比率
                        if avg_volume_20d is not None and avg_volume_20d != 0:
                            volume_ratio = entry_volume / avg_volume_20d
                except Exception:
                    pass  # 計算失敗時はNone
            
            # 5. exit_gap_pct - エグジット時のギャップ（前日終値とエグジット価格の乖離%）
            exit_gap_pct = None
            if exit_date is not None and exit_price is not None:
                try:
                    exit_idx_pos = results_df.index.get_loc(exit_date)
                    if exit_idx_pos > 0 and 'Close' in results_df.columns:
                        prev_close_exit = results_df['Close'].iloc[exit_idx_pos - 1]
                        if prev_close_exit is not None and prev_close_exit != 0:
                            exit_gap_pct = (exit_price - prev_close_exit) / prev_close_exit * 100
                except Exception:
                    pass  # 計算失敗時はNone
            
            # 6. highest_price_during_hold - 保有期間中の最高値
            highest_price_during_hold = None
            if entry_date is not None and exit_date is not None:
                try:
                    hold_data = results_df.loc[entry_date:exit_date]
                    if 'High' in hold_data.columns and len(hold_data) > 0:
                        highest_price_during_hold = hold_data['High'].max()
                except Exception:
                    pass  # データ不足の場合はNone
            # ==============================================================
            
            # ==================== Phase 3優先度中6項目計算 ====================
            # 1. exit_atr - エグジット時のATR（14日）
            exit_atr = None
            if exit_date is not None:
                try:
                    # ATR計算（14日）
                    if TALIB_AVAILABLE and 'High' in results_df.columns and 'Low' in results_df.columns and 'Close' in results_df.columns:
                        exit_idx_pos = results_df.index.get_loc(exit_date)
                        if exit_idx_pos >= 14:  # ATR14に必要な最低行数
                            atr_series = talib.ATR(
                                results_df['High'].iloc[:exit_idx_pos+1].values,
                                results_df['Low'].iloc[:exit_idx_pos+1].values,
                                results_df['Close'].iloc[:exit_idx_pos+1].values,
                                timeperiod=14
                            )
                            exit_atr = atr_series[-1]  # 最新値
                    else:
                        # talib未利用時: 簡易版ATR（High-Low平均）
                        exit_idx_pos = results_df.index.get_loc(exit_date)
                        if exit_idx_pos >= 14:
                            lookback_data = results_df.iloc[exit_idx_pos-14:exit_idx_pos+1]
                            if 'High' in lookback_data.columns and 'Low' in lookback_data.columns:
                                exit_atr = (lookback_data['High'] - lookback_data['Low']).mean()
                except Exception:
                    pass  # 計算失敗時はNone
            
            # 2. max_gap_during_hold - 保有期間中の最大ギャップ（絶対値）
            max_gap_during_hold = None
            if entry_date is not None and exit_date is not None:
                try:
                    hold_data = results_df.loc[entry_date:exit_date]
                    if 'Open' in hold_data.columns and 'Close' in hold_data.columns and len(hold_data) > 1:
                        # 前日終値との差分を計算
                        prev_close = hold_data['Close'].shift(1)
                        gap_pct = ((hold_data['Open'] - prev_close) / prev_close * 100).abs()
                        max_gap_during_hold = gap_pct.max()
                except Exception:
                    pass  # 計算失敗時はNone
            
            # 3. trailing_activated - トレーリング発動有無
            trailing_activated = None
            if max_profit_pct is not None and exit_params is not None:
                trailing_stop_pct_value = exit_params.get('trailing_stop_pct', None)
                if trailing_stop_pct_value is not None:
                    # trailing_stop_pctは0.10形式（10%）で保存されているため、100倍して%化
                    trailing_activated = max_profit_pct >= (trailing_stop_pct_value * 100)
            
            # 4. trailing_trigger_price - トレーリング発動価格
            trailing_trigger_price = None
            if entry_price is not None and exit_params is not None:
                trailing_stop_pct_value = exit_params.get('trailing_stop_pct', None)
                if trailing_stop_pct_value is not None:
                    trailing_trigger_price = entry_price * (1 + trailing_stop_pct_value)
            
            # 5. entry_trend_strength - エントリー時のトレンド強度（ADX）
            entry_trend_strength = None
            if entry_date is not None:
                try:
                    if TALIB_AVAILABLE and 'High' in results_df.columns and 'Low' in results_df.columns and 'Close' in results_df.columns:
                        entry_idx_pos = results_df.index.get_loc(entry_date)
                        if entry_idx_pos >= 14:  # ADX14に必要な最低行数
                            adx_series = talib.ADX(
                                results_df['High'].iloc[:entry_idx_pos+1].values,
                                results_df['Low'].iloc[:entry_idx_pos+1].values,
                                results_df['Close'].iloc[:entry_idx_pos+1].values,
                                timeperiod=14
                            )
                            entry_trend_strength = adx_series[-1]  # 最新値
                except Exception:
                    pass  # 計算失敗時はNone（talib未利用時は常にNone）
            
            # 6. sma_distance_pct - 移動平均線（20日）との乖離率
            sma_distance_pct = None
            if entry_date is not None and entry_price is not None:
                try:
                    entry_idx_pos = results_df.index.get_loc(entry_date)
                    if entry_idx_pos >= 19 and 'Close' in results_df.columns:  # 20日分のデータが必要
                        sma_20 = results_df['Close'].iloc[entry_idx_pos-19:entry_idx_pos+1].mean()
                        if sma_20 is not None and sma_20 != 0:
                            sma_distance_pct = (entry_price - sma_20) / sma_20 * 100
                except Exception:
                    pass  # 計算失敗時はNone
            # ==============================================================
            
            trade_details.append({
                'trade_id': trade_id,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'exit_reason': exit_reason,
                # Phase 1必須7項目
                'holding_days': holding_days,
                'profit_loss_pct': profit_loss_pct,
                'max_profit_pct': max_profit_pct,
                'max_loss_pct': max_loss_pct,
                'entry_atr': entry_atr,
                'entry_atr_pct': entry_atr_pct,
                'entry_gap_pct': entry_gap_pct,
                # Phase 2優先度高6項目
                'r_multiple': r_multiple,
                'entry_volume': entry_volume,
                'avg_volume_20d': avg_volume_20d,
                'volume_ratio': volume_ratio,
                'exit_gap_pct': exit_gap_pct,
                'highest_price_during_hold': highest_price_during_hold,
                # Phase 3優先度中6項目
                'exit_atr': exit_atr,
                'max_gap_during_hold': max_gap_during_hold,
                'trailing_activated': trailing_activated,
                'trailing_trigger_price': trailing_trigger_price,
                'entry_trend_strength': entry_trend_strength,
                'sma_distance_pct': sma_distance_pct
            })
    
    trade_details_df = pd.DataFrame(trade_details)
    
    return {
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'total_profit': total_profit,
        'total_loss': total_loss,
        'avg_profit_per_trade': avg_profit_per_trade,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'payoff_ratio': payoff_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        **exit_reasons  # エグジット理由カウントを展開
    }, trade_details_df


def run_single_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    exit_params: Dict,
    warmup_days: int = 150
) -> Tuple[Dict, pd.DataFrame]:
    """
    単一銘柄バックテスト実行（エントリー: GC戦略固定）
    
    Args:
        ticker: ティッカーシンボル
        start_date: 開始日
        end_date: 終了日
        exit_params: エグジットパラメータ
        warmup_days: ウォームアップ期間
    
    Returns:
        (パフォーマンス指標, 取引履歴)
    """
    try:
        # データ取得
        _, _, _, stock_data, _ = get_parameters_and_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            warmup_days=warmup_days
        )
        
        # GC戦略初期化（エントリー固定、エグジットパラメータ適用）
        strategy_params = {
            'short_window': 5,
            'long_window': 25,
            **exit_params  # エグジットパラメータ上書き
        }
        
        strategy = GCStrategy(data=stock_data, params=strategy_params, ticker=ticker)
        
        # バックテスト実行（日付を pd.Timestamp 変換）
        results_df = strategy.backtest(
            trading_start_date=pd.Timestamp(start_date),
            trading_end_date=pd.Timestamp(end_date)
        )
        
        # パフォーマンス指標計算
        if results_df is not None and not results_df.empty:
            metrics, trade_details = calculate_performance_metrics(results_df, exit_params=exit_params)
            
            # ticker, stop_loss_pct, trailing_stop_pct列を追加
            trade_details['ticker'] = ticker
            trade_details['stop_loss_pct'] = exit_params['stop_loss_pct']
            trade_details['trailing_stop_pct'] = exit_params['trailing_stop_pct']
            
            # PF上限制約チェック
            if metrics['profit_factor'] > MAX_PF:
                logger.warning(
                    f"{ticker}: PF={metrics['profit_factor']:.2f} > {MAX_PF}（過学習警告）"
                )
            
            return metrics, trade_details
        else:
            logger.warning(f"{ticker}: 取引履歴なし")
            metrics_empty, trade_details_empty = calculate_performance_metrics(pd.DataFrame(), exit_params=None)
            return metrics_empty, trade_details_empty
    
    except Exception as e:
        logger.error(f"{ticker} バックテスト失敗: {e}")
        metrics_error, trade_details_error = calculate_performance_metrics(pd.DataFrame(), exit_params=None)
        return metrics_error, trade_details_error


def filter_overfit_params(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    過学習パラメータをフィルタリング（Phase 3-6失敗の教訓）
    
    Args:
        results_df: バックテスト結果
    
    Returns:
        フィルタリング後の結果
    """
    filtered = results_df[
        (results_df['avg_pf'] > 1.0) &
        (results_df['avg_pf'] <= MAX_PF) &
        (results_df['avg_win_rate'] >= MINIMUM_WIN_RATE) &
        (results_df['avg_num_trades'] >= MINIMUM_TRADES)
    ]
    
    logger.info(
        f"フィルタリング: {len(results_df)}件 → {len(filtered)}件 "
        f"(PF > 1.0 & PF <= {MAX_PF} & Win Rate >= {MINIMUM_WIN_RATE:.0%} & 取引数 >= {MINIMUM_TRADES})"
    )
    
    return filtered


def save_results(
    results: List[Dict],
    phase: int,
    output_dir: str = "results"
) -> Tuple[str, str]:
    """
    結果をCSV+JSON形式で保存
    
    Args:
        results: 結果リスト
        phase: フェーズ番号（1, 2, 3）
        output_dir: 出力ディレクトリ
    
    Returns:
        (CSV パス, JSON パス)
    """
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ファイル名
    csv_file = output_path / f"phase{phase}_simple_{timestamp}.csv"
    json_file = output_path / f"phase{phase}_simple_{timestamp}.json"
    
    # DataFrame変換
    results_df = pd.DataFrame(results)
    
    # CSV保存
    results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # JSON保存（numpy型変換）
    results_clean = []
    for result in results:
        clean_result = {}
        for k, v in result.items():
            if hasattr(v, 'item'):  # numpy型
                clean_result[k] = v.item()
            elif isinstance(v, (np.integer, np.floating)):
                clean_result[k] = float(v)
            elif v == float('inf'):
                clean_result[k] = "Infinity"
            else:
                clean_result[k] = v
        results_clean.append(clean_result)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)
    
    logger.info(f"結果保存: {csv_file}")
    logger.info(f"JSON保存: {json_file}")
    
    return str(csv_file), str(json_file)


# ==================== Phase 1: シンプルルール検証 ====================

def run_phase1(args) -> bool:
    """
    Phase 1実行: 固定パラメータで10銘柄検証
    
    目的: PF > 1.0達成可能性確認
    
    Returns:
        成功判定（True: Phase 2へ進む, False: パラメータ微調整）
    """
    logger.info("=" * 80)
    logger.info("Phase 1: シンプルルール検証開始")
    logger.info(f"固定パラメータ: {PHASE1_PARAMS}")
    logger.info(f"検証銘柄: {len(VALIDATION_TICKERS)}銘柄")
    logger.info(f"検証期間: {IN_SAMPLE_START} ~ {IN_SAMPLE_END}（5年間）")
    logger.info("=" * 80)
    
    results = []
    
    for ticker in VALIDATION_TICKERS:
        logger.info(f"\n【{ticker}】検証中...")
        
        metrics, _ = run_single_backtest(
            ticker=ticker,
            start_date=IN_SAMPLE_START,
            end_date=OOS_END,  # Phase 1は5年間全体
            exit_params=PHASE1_PARAMS,
            warmup_days=WARMUP_DAYS
        )
        
        results.append({
            'ticker': ticker,
            'phase': 1,
            'stop_loss_pct': PHASE1_PARAMS['stop_loss_pct'],
            'trailing_stop_pct': PHASE1_PARAMS['trailing_stop_pct'],
            'take_profit_pct': PHASE1_PARAMS['take_profit_pct'],
            **metrics
        })
        
        logger.info(
            f"  PF: {metrics['profit_factor']:.2f}, "
            f"Win Rate: {metrics['win_rate']:.1%}, "
            f"取引数: {metrics['num_trades']}"
        )
    
    # 結果保存
    csv_path, json_path = save_results(results, phase=1)
    
    # 成功判定
    results_df = pd.DataFrame(results)
    avg_pf = results_df['profit_factor'].mean()
    pass_count = (results_df['profit_factor'] > 1.0).sum()
    pass_rate = pass_count / len(results_df)
    
    logger.info("\n" + "=" * 80)
    logger.info("Phase 1結果サマリー")
    logger.info("=" * 80)
    logger.info(f"平均PF: {avg_pf:.2f}")
    logger.info(f"PF > 1.0銘柄数: {pass_count}/{len(results_df)}（{pass_rate:.1%}）")
    logger.info(f"平均Win Rate: {results_df['win_rate'].mean():.1%}")
    logger.info(f"平均取引数: {results_df['num_trades'].mean():.1f}")
    
    # 判定
    success = avg_pf > 1.0 and pass_rate >= MIN_PASS_RATE
    
    if success:
        logger.info(f"\n[SUCCESS] Phase 1成功: 平均PF={avg_pf:.2f} > 1.0, 合格率={pass_rate:.1%} >= {MIN_PASS_RATE:.1%}")
        logger.info("→ Phase 2（グリッドサーチ）へ進むことを推奨")
    else:
        logger.warning(f"\n[FAIL] Phase 1失敗: 平均PF={avg_pf:.2f}, 合格率={pass_rate:.1%}")
        logger.warning("→ パラメータ微調整（損切3-7%、トレーリング5-15%）を推奨")
    
    return success


# ==================== Phase 1.5: パラメータ微調整グリッド ====================

def run_phase1_5(args) -> Tuple[bool, pd.DataFrame]:
    """
    Phase 1.5: パラメータ微調整グリッドサーチ
    
    Phase 1失敗時に実行。EXIT_STRATEGY_REDESIGN_V2.md推奨範囲
    （損切3-7%、トレーリング5-15%）で最適パラメータを探索。
    
    Args:
        args: コマンドライン引数
    
    Returns:
        (成功判定, 結果DataFrame)
    """
    logger.info("=" * 80)
    logger.info("Phase 1.5: パラメータ微調整グリッドサーチ開始")
    logger.info(f"推奨範囲（EXIT_STRATEGY_REDESIGN_V2.md）:")
    logger.info(f"  損切: 3-7%, トレーリング: 5-15%")
    logger.info(f"検証銘柄: {len(VALIDATION_TICKERS)}銘柄")
    logger.info(f"検証期間: {IN_SAMPLE_START} ~ {OOS_END}（5年間）")
    logger.info(f"組み合わせ数: 3×3×1 = 9")
    logger.info("=" * 80)
    
    # パラメータ組み合わせ生成
    param_combinations = list(itertools.product(
        PHASE1_5_PARAM_GRID['stop_loss_pct'],
        PHASE1_5_PARAM_GRID['trailing_stop_pct'],
        PHASE1_5_PARAM_GRID['take_profit_pct']
    ))
    
    results = []
    
    for i, (stop_loss, trailing_stop, take_profit) in enumerate(param_combinations, 1):
        params = {
            'stop_loss_pct': stop_loss,
            'trailing_stop_pct': trailing_stop,
            'take_profit_pct': take_profit
        }
        
        logger.info(f"\n【パラメータ組み合わせ {i}/{len(param_combinations)}】")
        logger.info(f"  損切: {stop_loss:.1%}, トレーリング: {trailing_stop:.1%}, 利確: {take_profit}")
        
        ticker_results = []
        
        for ticker in VALIDATION_TICKERS:
            metrics, _ = run_single_backtest(
                ticker=ticker,
                start_date=IN_SAMPLE_START,
                end_date=OOS_END,
                exit_params=params,
                warmup_days=WARMUP_DAYS
            )
            
            ticker_results.append(metrics)
        
        # 平均パフォーマンス計算
        avg_pf = np.mean([m['profit_factor'] for m in ticker_results])
        avg_wr = np.mean([m['win_rate'] for m in ticker_results])
        avg_trades = np.mean([m['num_trades'] for m in ticker_results])
        pass_count = sum(1 for m in ticker_results if m['profit_factor'] > 1.0)
        pass_rate = pass_count / len(ticker_results)
        
        logger.info(f"  → 平均PF: {avg_pf:.2f}, Win Rate: {avg_wr:.1%}, "
                   f"取引数: {avg_trades:.1f}, 合格率: {pass_rate:.1%}")
        
        # 銘柄別結果記録
        for ticker, metrics in zip(VALIDATION_TICKERS, ticker_results):
            results.append({
                'phase': '1.5',
                'ticker': ticker,
                'stop_loss_pct': stop_loss,
                'trailing_stop_pct': trailing_stop,
                'take_profit_pct': take_profit,
                **metrics
            })
    
    # 結果保存
    csv_path, json_path = save_results(results, phase='1.5')
    
    # 最良パラメータ選定
    results_df = pd.DataFrame(results)
    
    # None値を"None"文字列に変換（groupby対応）
    results_df['take_profit_pct'] = results_df['take_profit_pct'].fillna("None")
    
    # 銘柄平均でグループ化
    grouped = results_df.groupby(['stop_loss_pct', 'trailing_stop_pct', 'take_profit_pct']).agg({
        'profit_factor': 'mean',
        'win_rate': 'mean',
        'num_trades': 'mean'
    }).reset_index()
    
    # 合格率計算（別途実行）
    pass_rates = []
    for _, row in grouped.iterrows():
        mask = (
            (results_df['stop_loss_pct'] == row['stop_loss_pct']) &
            (results_df['trailing_stop_pct'] == row['trailing_stop_pct']) &
            (results_df['take_profit_pct'] == row['take_profit_pct'])
        )
        subset = results_df[mask]
        pass_rate = (subset['profit_factor'] > 1.0).sum() / len(subset)
        pass_rates.append(pass_rate)
    
    grouped['pass_rate'] = pass_rates
    
    # 成功判定（合格率80%以上のパラメータ存在確認）
    best = grouped[grouped['pass_rate'] >= MIN_PASS_RATE].sort_values(
        'profit_factor', ascending=False
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Phase 1.5結果サマリー")
    logger.info("=" * 80)
    logger.info(f"全組み合わせ数: {len(grouped)}")
    logger.info(f"合格率{MIN_PASS_RATE:.0%}以上: {len(best)}組み合わせ")
    
    if not best.empty:
        logger.info("\n【最良パラメータ】")
        best_params = best.iloc[0]
        logger.info(f"  損切: {best_params['stop_loss_pct']:.1%}")
        logger.info(f"  トレーリング: {best_params['trailing_stop_pct']:.1%}")
        logger.info(f"  利確: {best_params['take_profit_pct']}")
        logger.info(f"  平均PF: {best_params['profit_factor']:.2f}")
        logger.info(f"  平均Win Rate: {best_params['win_rate']:.1%}")
        logger.info(f"  平均取引数: {best_params['num_trades']:.1f}")
        logger.info(f"  合格率: {best_params['pass_rate']:.1%}")
        logger.info("\n[SUCCESS] Phase 1.5成功: 合格率80%以上のパラメータ発見")
        logger.info("→ Phase 2（拡張グリッドサーチ）へ進むことを推奨")
        success = True
    else:
        logger.warning("\n[FAIL] Phase 1.5失敗: 合格率80%以上のパラメータなし")
        logger.warning("→ エントリー戦略の見直しを推奨（Win Rate向上が必要）")
        success = False
    
    return success, grouped


# ==================== Phase 1.6: トレーリング拡張グリッド（ペイオフレシオ分析用） ====================

def run_phase1_6(args) -> Tuple[bool, pd.DataFrame]:
    """
    Phase 1.6実行: トレーリング拡張グリッドサーチ（ペイオフレシオ分析用）
    
    目的: トレーリング20%,25%,30%を追加して大敗パターンとPF2.0達成パラメータを特定
    PAYOFF_RATIO_EXIT_ANALYSIS_PROJECT.md Task 2実装
    
    パラメータ空間:
        損切: 3%, 5%, 7%（3パターン）
        トレーリング: 5%, 10%, 15%, 20%, 25%, 30%（6パターン）
        利確: なし
        合計: 18組み合わせ
    
    Returns:
        (success, grouped): 成功フラグ、グループ化結果
    """
    logger.info("Phase 1.6: トレーリング拡張グリッドサーチ開始（ペイオフレシオ分析用）")
    logger.info(f"パラメータ空間: 損切3パターン × トレーリング6パターン = 18組み合わせ")
    logger.info(f"検証銘柄: {len(VALIDATION_TICKERS)}銘柄")
    logger.info(f"合計: {len(VALIDATION_TICKERS)} × 18 = {len(VALIDATION_TICKERS) * 18}データポイント")
    
    results = []
    all_trade_details = []  # 全ての個別取引履歴を格納
    
    # 全パラメータ組み合わせ生成
    param_combinations = list(itertools.product(
        PHASE1_6_PARAM_GRID['stop_loss_pct'],
        PHASE1_6_PARAM_GRID['trailing_stop_pct'],
        PHASE1_6_PARAM_GRID['take_profit_pct']
    ))
    
    total_combinations = len(param_combinations)
    logger.info(f"総組み合わせ数: {total_combinations}")
    
    # 各銘柄で全組み合わせをテスト
    for ticker_idx, ticker in enumerate(VALIDATION_TICKERS, 1):
        logger.info(f"\n【{ticker_idx}/{len(VALIDATION_TICKERS)}】{ticker} 検証中...")
        
        for combo_idx, (stop_loss, trailing_stop, take_profit) in enumerate(param_combinations, 1):
            logger.info(f"  組み合わせ {combo_idx}/{total_combinations}: "
                       f"損切{stop_loss:.0%}, トレーリング{trailing_stop:.0%}, 利確{take_profit}")
            
            # エグジットパラメータ辞書作成
            exit_params = {
                'stop_loss_pct': stop_loss,
                'trailing_stop_pct': trailing_stop,
                'take_profit_pct': take_profit
            }
            
            metrics, trade_details = run_single_backtest(
                ticker=ticker,
                start_date=IN_SAMPLE_START,
                end_date=OOS_END,
                exit_params=exit_params,
                warmup_days=WARMUP_DAYS
            )
            
            if metrics:
                # パラメータ情報をmetricsに追加
                results.append({
                    'phase': '1.6',
                    'ticker': ticker,
                    'stop_loss_pct': stop_loss,
                    'trailing_stop_pct': trailing_stop,
                    'take_profit_pct': take_profit,
                    **metrics
                })
                
                # 個別取引履歴にメタ情報追加
                if not trade_details.empty:
                    trade_details['ticker'] = ticker
                    trade_details['stop_loss_pct'] = stop_loss
                    trade_details['trailing_stop_pct'] = trailing_stop
                    trade_details['take_profit_pct'] = take_profit
                    all_trade_details.append(trade_details)
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    results_df['take_profit_pct'] = results_df['take_profit_pct'].fillna("None")
    
    # CSV/JSON保存
    csv_path, json_path = save_results(results, phase='1.6')
    logger.info(f"\n結果保存: {csv_path}")
    logger.info(f"JSON保存: {json_path}")
    
    # 個別取引履歴を統合してCSV保存
    if all_trade_details:
        combined_trades = pd.concat(all_trade_details, ignore_index=True)
        
        # タイムスタンプ（phase1.6と同じ）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trades_csv = Path("results") / f"phase1.6_trades_{timestamp}.csv"
        
        combined_trades.to_csv(trades_csv, index=False, encoding='utf-8-sig')
        logger.info(f"個別取引履歴保存: {trades_csv}（{len(combined_trades)}件）")
    else:
        logger.warning("個別取引履歴が0件のため、取引履歴CSVは生成されません")
    
    # パラメータ組み合わせごとにグループ化
    grouped_data = []
    for (stop, trail, profit), group in results_df.groupby(['stop_loss_pct', 'trailing_stop_pct', 'take_profit_pct']):
        # 合格銘柄数（PF > 1.0）をカウント
        passed = len(group[group['profit_factor'] > 1.0])
        pass_rate = passed / len(group)
        
        grouped_data.append({
            'stop_loss_pct': stop,
            'trailing_stop_pct': trail,
            'take_profit_pct': profit,
            'profit_factor': group['profit_factor'].mean(),
            'win_rate': group['win_rate'].mean(),
            'num_trades': group['num_trades'].mean(),
            'payoff_ratio': group['payoff_ratio'].mean(),
            'avg_win': group['avg_win'].mean(),
            'avg_loss': group['avg_loss'].mean(),
            'pass_rate': pass_rate,
            'passed_count': passed,
            'total_count': len(group)
        })
    
    grouped = pd.DataFrame(grouped_data)
    grouped = grouped.sort_values('pass_rate', ascending=False)
    
    # 結果サマリー表示
    logger.info("=" * 80)
    logger.info("Phase 1.6結果サマリー")
    logger.info("=" * 80)
    
    logger.info("\n【TOP 5パラメータ（合格率順）】")
    for idx, row in grouped.head(5).iterrows():
        logger.info(f"\n{idx+1}位:")
        logger.info(f"  損切: {row['stop_loss_pct']:.1%}, トレーリング: {row['trailing_stop_pct']:.1%}, 利確: {row['take_profit_pct']}")
        logger.info(f"  平均PF: {row['profit_factor']:.2f}")
        logger.info(f"  ペイオフレシオ: {row['payoff_ratio']:.2f}")
        logger.info(f"  平均Win Rate: {row['win_rate']:.1%}")
        logger.info(f"  平均取引数: {row['num_trades']:.1f}")
        logger.info(f"  合格率: {row['pass_rate']:.1%} ({row['passed_count']}/{row['total_count']})")
    
    # 成功判定（合格率80%以上）
    success = False
    best = grouped[grouped['pass_rate'] >= 0.8]
    
    if not best.empty:
        logger.info("\n【最良パラメータ】")
        best_params = best.iloc[0]
        logger.info(f"  損切: {best_params['stop_loss_pct']:.1%}")
        logger.info(f"  トレーリング: {best_params['trailing_stop_pct']:.1%}")
        logger.info(f"  利確: {best_params['take_profit_pct']}")
        logger.info(f"  平均PF: {best_params['profit_factor']:.2f}")
        logger.info(f"  ペイオフレシオ: {best_params['payoff_ratio']:.2f}")
        logger.info(f"  平均Win Rate: {best_params['win_rate']:.1%}")
        logger.info(f"  平均取引数: {best_params['num_trades']:.1f}")
        logger.info(f"  合格率: {best_params['pass_rate']:.1%}")
        logger.info("\n[SUCCESS] Phase 1.6成功: 合格率80%以上のパラメータ発見")
        logger.info("→ Task 4（大敗パターン洗い出し）へ進むことを推奨")
        success = True
    else:
        logger.warning("\n[FAIL] Phase 1.6失敗: 合格率80%以上のパラメータなし")
        logger.warning("→ Task 4（大敗パターン分析）とTask 5（ペイオフレシオ分析）を実施")
        success = False
    
    return success, grouped


# ==================== Phase 2: グリッドサーチ ====================

def run_phase2(args) -> pd.DataFrame:
    """
    Phase 2実行: グリッドサーチで最適パラメータ発見
    
    目的: TOP 3パラメータセット選定
    
    Returns:
        TOP 3パラメータデータフレーム
    """
    logger.info("=" * 80)
    logger.info("Phase 2: グリッドサーチ開始")
    logger.info(f"パラメータ空間: {PHASE2_PARAM_GRID}")
    logger.info(f"組み合わせ数: {np.prod([len(v) for v in PHASE2_PARAM_GRID.values()])}")
    logger.info(f"検証銘柄: {len(VALIDATION_TICKERS)}銘柄")
    logger.info(f"検証期間: {IN_SAMPLE_START} ~ {IN_SAMPLE_END}（In-Sample 4年間）")
    logger.info("=" * 80)
    
    # パラメータ組み合わせ生成
    param_combinations = list(itertools.product(*PHASE2_PARAM_GRID.values()))
    total_tests = len(param_combinations) * len(VALIDATION_TICKERS)
    
    logger.info(f"\n総検証数: {len(param_combinations)}パラメータ × {len(VALIDATION_TICKERS)}銘柄 = {total_tests}検証")
    
    results = []
    
    for i, params_tuple in enumerate(param_combinations, 1):
        params = dict(zip(PHASE2_PARAM_GRID.keys(), params_tuple))
        
        logger.info(f"\n【パラメータ {i}/{len(param_combinations)}】")
        logger.info(f"  {params}")
        
        ticker_results = []
        
        for ticker in VALIDATION_TICKERS:
            metrics, _ = run_single_backtest(
                ticker=ticker,
                start_date=IN_SAMPLE_START,
                end_date=IN_SAMPLE_END,  # In-Sample期間
                exit_params=params,
                warmup_days=WARMUP_DAYS
            )
            
            ticker_results.append(metrics)
        
        # 平均パフォーマンス計算
        avg_pf = np.mean([r['profit_factor'] for r in ticker_results])
        avg_wr = np.mean([r['win_rate'] for r in ticker_results])
        avg_trades = np.mean([r['num_trades'] for r in ticker_results])
        pf_std = np.std([r['profit_factor'] for r in ticker_results])
        
        results.append({
            **params,
            'avg_pf': avg_pf,
            'avg_win_rate': avg_wr,
            'avg_num_trades': avg_trades,
            'pf_std': pf_std
        })
        
        logger.info(f"  平均PF: {avg_pf:.2f}, 平均Win Rate: {avg_wr:.1%}, 平均取引数: {avg_trades:.1f}, PF標準偏差: {pf_std:.2f}")
    
    # 結果保存
    csv_path, json_path = save_results(results, phase=2)
    
    # 過学習フィルタリング
    results_df = pd.DataFrame(results)
    filtered = filter_overfit_params(results_df)
    
    if filtered.empty:
        logger.error("\n[FAIL] Phase 2失敗: フィルタリング後のパラメータが0件")
        logger.error("→ Phase 1に戻ってパラメータ空間を再検討推奨")
        return pd.DataFrame()
    
    # 複合スコアで順位付け（PF 40% + Win Rate 30% + 安定性 30%）
    filtered['score'] = (
        filtered['avg_pf'] * 0.4 +
        filtered['avg_win_rate'] * 100 * 0.3 +
        (1 / (filtered['pf_std'] + 0.01)) * 0.3  # ゼロ除算回避
    )
    
    # TOP 3選定
    top3 = filtered.nlargest(3, 'score')
    
    logger.info("\n" + "=" * 80)
    logger.info("Phase 2: TOP 3パラメータ")
    logger.info("=" * 80)
    
    for i, (idx, row) in enumerate(top3.iterrows(), 1):
        logger.info(f"\n【候補{i}】")
        logger.info(f"  stop_loss: {row['stop_loss_pct']:.1%}")
        logger.info(f"  trailing_stop: {row['trailing_stop_pct']:.1%}")
        logger.info(f"  take_profit: {row['take_profit_pct'] if row['take_profit_pct'] is not None else 'なし'}")
        logger.info(f"  平均PF: {row['avg_pf']:.2f}")
        logger.info(f"  平均Win Rate: {row['avg_win_rate']:.1%}")
        logger.info(f"  平均取引数: {row['avg_num_trades']:.1f}")
        logger.info(f"  PF標準偏差: {row['pf_std']:.2f}")
        logger.info(f"  複合スコア: {row['score']:.2f}")
    
    # TOP 3をJSON保存
    top3_file = Path("results") / "phase2_top3.json"
    top3_dict = top3.to_dict(orient='records')
    
    # numpy型変換
    top3_clean = []
    for record in top3_dict:
        clean_record = {}
        for k, v in record.items():
            if hasattr(v, 'item'):
                clean_record[k] = v.item()
            elif isinstance(v, (np.integer, np.floating)):
                clean_record[k] = float(v)
            else:
                clean_record[k] = v
        top3_clean.append(clean_record)
    
    with open(top3_file, 'w', encoding='utf-8') as f:
        json.dump(top3_clean, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nTOP 3パラメータ保存: {top3_file}")
    logger.info("\n[SUCCESS] Phase 2完了: Phase 3（Out-of-Sample検証）へ進むことを推奨")
    
    return top3


# ==================== Phase 3: Out-of-Sample検証 ====================

def run_phase3(args, top3_params: pd.DataFrame = None) -> Dict:
    """
    Phase 3実行: Out-of-Sample検証で汎化性能確認
    
    目的: 最終推奨パラメータ選定
    
    Args:
        top3_params: TOP 3パラメータ（Noneの場合はファイルから読み込み）
    
    Returns:
        最終推奨パラメータと劣化率分析
    """
    logger.info("=" * 80)
    logger.info("Phase 3: Out-of-Sample検証開始")
    logger.info(f"検証期間: {OOS_START} ~ {OOS_END}（Out-of-Sample 2年間）")
    logger.info("=" * 80)
    
    # TOP 3パラメータ読み込み
    if top3_params is None:
        top3_file = Path(args.top3_file if hasattr(args, 'top3_file') else "results/phase2_top3.json")
        
        if not top3_file.exists():
            logger.error(f"[FAIL] TOP 3パラメータファイルが見つかりません: {top3_file}")
            logger.error("→ Phase 2を先に実行してください")
            return {}
        
        with open(top3_file, 'r', encoding='utf-8') as f:
            top3_data = json.load(f)
        
        top3_params = pd.DataFrame(top3_data)
    
    # In-Sample結果（Phase 2から取得）
    in_sample_results = {}
    for i, row in top3_params.iterrows():
        in_sample_results[f'params_{i+1}'] = {
            'pf': row['avg_pf'],
            'win_rate': row['avg_win_rate']
        }
    
    # Out-of-Sample検証
    oos_results = {}
    phase3_all_results = []
    
    for i, row in top3_params.iterrows():
        params = {
            'stop_loss_pct': row['stop_loss_pct'],
            'trailing_stop_pct': row['trailing_stop_pct'],
            'take_profit_pct': row['take_profit_pct']
        }
        
        logger.info(f"\n【候補{i+1}】Out-of-Sample検証")
        logger.info(f"  {params}")
        
        ticker_results = []
        
        for ticker in VALIDATION_TICKERS:
            metrics, _ = run_single_backtest(
                ticker=ticker,
                start_date=OOS_START,
                end_date=OOS_END,  # Out-of-Sample期間
                exit_params=params,
                warmup_days=WARMUP_DAYS
            )
            
            ticker_results.append(metrics)
            
            phase3_all_results.append({
                'ticker': ticker,
                'candidate': i+1,
                **params,
                **metrics
            })
        
        avg_pf = np.mean([r['profit_factor'] for r in ticker_results])
        avg_wr = np.mean([r['win_rate'] for r in ticker_results])
        
        oos_results[f'params_{i+1}'] = {'pf': avg_pf, 'win_rate': avg_wr}
        
        logger.info(f"  Out-of-Sample 平均PF: {avg_pf:.2f}, 平均Win Rate: {avg_wr:.1%}")
    
    # 結果保存
    csv_path, json_path = save_results(phase3_all_results, phase=3)
    
    # 劣化率分析
    logger.info("\n" + "=" * 80)
    logger.info("Phase 3: 劣化率分析（In-Sample vs Out-of-Sample）")
    logger.info("=" * 80)
    
    final_recommendations = []
    
    for key in in_sample_results:
        in_pf = in_sample_results[key]['pf']
        oos_pf = oos_results[key]['pf']
        
        degradation = (in_pf - oos_pf) / in_pf if in_pf > 0 else 0.0
        
        status = "[PASS] 合格" if degradation < MAX_DEGRADATION else "[FAIL] 過学習"
        
        logger.info(f"\n{key}:")
        logger.info(f"  In-Sample PF:  {in_pf:.2f}")
        logger.info(f"  Out-of-Sample PF: {oos_pf:.2f}")
        logger.info(f"  劣化率: {degradation:.1%} {status}")
        
        if degradation < MAX_DEGRADATION:
            final_recommendations.append(key)
    
    # 最終判定
    logger.info("\n" + "=" * 80)
    logger.info("Phase 3: 最終判定")
    logger.info("=" * 80)
    
    if len(final_recommendations) >= 2:
        logger.info(f"\n[SUCCESS] Phase 3成功: {len(final_recommendations)}パラメータがOut-of-Sample通過")
        logger.info(f"推奨パラメータ: {final_recommendations[0]}")
        logger.info("\n→ リアルトレード移行判断可能")
    elif len(final_recommendations) == 1:
        logger.warning(f"\n[WARNING] Phase 3部分的成功: 1パラメータのみOut-of-Sample通過")
        logger.warning(f"推奨パラメータ: {final_recommendations[0]}")
        logger.warning("\n→ Phase 2で別候補を追加検証推奨")
    else:
        logger.error("\n[FAIL] Phase 3失敗: 汎化性能不足、全パラメータがOut-of-Sampleで劣化")
        logger.error("\n→ Phase 2再実施（パラメータ空間拡張）推奨")
    
    return {
        'in_sample_results': in_sample_results,
        'oos_results': oos_results,
        'final_recommendations': final_recommendations
    }


# ==================== メイン実行 ====================

def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(description="シンプルエグジット戦略検証 v2.0")
    
    parser.add_argument(
        '--phase',
        type=str,
        choices=['1', '1.5', '1.6', '2', '3'],
        required=True,
        help='実行フェーズ (1: シンプルルール, 1.5: 微調整グリッド, 1.6: トレーリング拡張, 2: グリッドサーチ, 3: Out-of-Sample)'
    )
    
    parser.add_argument(
        '--top3-file',
        type=str,
        default='results/phase2_top3.json',
        help='Phase 3用のTOP 3パラメータファイルパス'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("シンプルエグジット戦略検証 v2.0 開始")
    logger.info(f"実行フェーズ: Phase {args.phase}")
    logger.info("=" * 80)
    
    try:
        if args.phase == '1':
            success = run_phase1(args)
            return 0 if success else 1
        
        elif args.phase == '1.5':
            success, _ = run_phase1_5(args)
            return 0 if success else 1
        
        elif args.phase == '1.6':
            success, _ = run_phase1_6(args)
            return 0 if success else 1
        
        elif args.phase == '2':
            top3 = run_phase2(args)
            return 0 if not top3.empty else 1
        
        elif args.phase == '3':
            results = run_phase3(args)
            return 0 if results.get('final_recommendations') else 1
    
    except Exception as e:
        logger.error(f"実行エラー: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
