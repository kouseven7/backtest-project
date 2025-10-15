"""
同日エントリー/エグジット問題の根本原因分析ツール

このスクリプトは同日エントリー/エグジット問題の根本原因を分析するための
専用ツールです。戦略の実行過程を詳細に追跡し、同日にエントリーとエグジットの
両方が発生する原因を特定します。

主な機能:
1. 各戦略の実行ステップごとのシグナル変化を追跡
2. シグナル生成に関わるインジケータと価格の相関関係を分析
3. 同日Entry/Exit発生パターンの分類と根本原因の特定
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from signal_processing import check_same_day_entry_exit
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators

# インポートする戦略クラス
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.Opening_Gap_Fixed import OpeningGapFixedStrategy

# 診断用ロガーの設定
log_file = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\logs\same_day_root_cause_analyzer.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger = setup_logger("root_cause_analyzer", log_file=log_file)

# 結果保存ディレクトリ
RESULTS_DIR = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\results"
CHARTS_DIR = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\charts"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


def create_strategy_execution_wrapper(strategy_class):
    """
    戦略クラスを拡張して各ステップでのシグナル生成を追跡するラッパー
    """
    class StrategyExecutionTracer(strategy_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.execution_trace = []
            self.signal_generation_points = []
            
        def _record_step(self, step_name, locals_dict=None):
            """実行ステップを記録"""
            step_info = {
                'step': step_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # 現在のシグナル状態をコピー
            if hasattr(self, 'data') and isinstance(self.data, pd.DataFrame):
                signal_cols = [col for col in self.data.columns if col.endswith('_Signal')]
                step_info['signals'] = {}
                
                for col in signal_cols:
                    signal_data = self.data[col].copy()
                    if not signal_data.empty:
                        non_zero = signal_data[signal_data != 0]
                        if not non_zero.empty:
                            step_info['signals'][col] = {
                                'count': int(len(non_zero)),
                                'first_date': non_zero.index[0].strftime('%Y-%m-%d') if len(non_zero) > 0 else None,
                                'last_date': non_zero.index[-1].strftime('%Y-%m-%d') if len(non_zero) > 0 else None
                            }
            
            # 現在のローカル変数を記録（重要な変数のみ）
            if locals_dict:
                step_info['variables'] = {}
                for key, value in locals_dict.items():
                    if key not in ['self', 'cls', '__class__'] and not key.startswith('__'):
                        if isinstance(value, (int, float, str, bool)):
                            step_info['variables'][key] = value
                        elif isinstance(value, pd.DataFrame):
                            step_info['variables'][key] = f"DataFrame with {len(value)} rows"
                        elif isinstance(value, pd.Series):
                            step_info['variables'][key] = f"Series with {len(value)} elements"
                        elif value is None:
                            step_info['variables'][key] = None
                        else:
                            step_info['variables'][key] = str(type(value))
            
            self.execution_trace.append(step_info)
        
        def _track_signal_generation(self, date, signal_type, value, condition_values=None):
            """シグナル生成ポイントを追跡"""
            point = {
                'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                'signal_type': signal_type,
                'value': value
            }
            
            # シグナル生成に関連する条件値を記録
            if condition_values:
                point['conditions'] = condition_values
                
            self.signal_generation_points.append(point)
        
        def backtest(self, *args, **kwargs):
            """バックテスト実行のオーバーライド"""
            self._record_step('backtest_start')
            result = super().backtest(*args, **kwargs)
            self._record_step('backtest_end')
            
            # 同日エントリー/エグジット問題の検出
            if isinstance(result, pd.DataFrame):
                same_day_check = check_same_day_entry_exit(result)
                if same_day_check['has_same_day_signals']:
                    self._record_step('same_day_signals_detected', {
                        'count': same_day_check['same_day_count'],
                        'dates': [d.strftime('%Y-%m-%d') for d in same_day_check['dates']] if 'dates' in same_day_check else []
                    })
            
            return result
        
    return StrategyExecutionTracer


def analyze_price_patterns(data, same_day_dates):
    """
    同日エントリー/エグジットが発生する日の価格パターンを分析
    """
    if not same_day_dates or not isinstance(data, pd.DataFrame):
        return {}
    
    patterns = {
        'gap_up_count': 0,
        'gap_down_count': 0,
        'high_volatility_count': 0,
        'low_volatility_count': 0,
        'price_reversals': 0,
        'trend_days': 0,
        'average_daily_range_pct': 0
    }
    
    # 日付をdatetime形式に変換（文字列の場合）
    dates = []
    for date in same_day_dates:
        if isinstance(date, str):
            try:
                dates.append(pd.Timestamp(date))
            except:
                continue
        else:
            dates.append(date)
    
    if not dates:
        return patterns
    
    # 前日データも含めて分析するため、日付でソート
    data_sorted = data.sort_index()
    
    daily_ranges = []
    for date in dates:
        if date in data_sorted.index:
            current_row = data_sorted.loc[date]
            
            # 前日を取得
            prev_date_idx = data_sorted.index.get_loc(date) - 1
            if prev_date_idx >= 0:
                prev_row = data_sorted.iloc[prev_date_idx]
                
                # ギャップアップ/ダウンの検出
                if current_row['Open'] > prev_row['Close'] * 1.01:  # 1%以上のギャップアップ
                    patterns['gap_up_count'] += 1
                elif current_row['Open'] < prev_row['Close'] * 0.99:  # 1%以上のギャップダウン
                    patterns['gap_down_count'] += 1
                
                # 価格の反転（前日と傾向が逆）
                if (prev_row['Close'] > prev_row['Open'] and current_row['Close'] < current_row['Open']) or \
                   (prev_row['Close'] < prev_row['Open'] and current_row['Close'] > current_row['Open']):
                    patterns['price_reversals'] += 1
                else:
                    patterns['trend_days'] += 1
            
            # 日中のボラティリティ
            daily_range_pct = (current_row['High'] - current_row['Low']) / current_row['Open'] * 100
            daily_ranges.append(daily_range_pct)
            
            if daily_range_pct > 2.0:  # 2%以上の日内変動を高ボラティリティと定義
                patterns['high_volatility_count'] += 1
            elif daily_range_pct < 1.0:  # 1%未満の日内変動を低ボラティリティと定義
                patterns['low_volatility_count'] += 1
    
    # 平均日内レンジを計算
    if daily_ranges:
        patterns['average_daily_range_pct'] = sum(daily_ranges) / len(daily_ranges)
    
    return patterns


def analyze_strategy_internals(strategy_instance, result_df):
    """
    戦略のトレース情報から、シグナル生成の根本原因を特定
    """
    if not hasattr(strategy_instance, 'execution_trace') or not hasattr(strategy_instance, 'signal_generation_points'):
        return {'success': False, 'error': 'トレース情報がありません'}
    
    # 同日エントリー/エグジットの検出
    same_day_check = check_same_day_entry_exit(result_df)
    if not same_day_check['has_same_day_signals']:
        return {'success': False, 'error': '同日エントリー/エグジットはありません'}
    
    # 同日エントリー/エグジットの日付を取得
    same_day_dates = same_day_check.get('dates', [])
    
    # シグナル生成ポイントから、同日のエントリーとエグジットに関連するものを抽出
    same_day_signals = []
    for point in strategy_instance.signal_generation_points:
        point_date = point.get('date')
        if point_date:
            try:
                point_date_ts = pd.Timestamp(point_date)
                if any(point_date_ts == date for date in same_day_dates):
                    same_day_signals.append(point)
            except:
                pass
    
    # 実行トレースの分析
    execution_analysis = {
        'steps_with_signal_changes': [],
        'signal_generation_order': []
    }
    
    signal_changes = {}
    for i, step in enumerate(strategy_instance.execution_trace):
        if 'signals' in step:
            for signal_name, info in step['signals'].items():
                if signal_name not in signal_changes:
                    signal_changes[signal_name] = []
                
                signal_changes[signal_name].append({
                    'step': step['step'],
                    'step_index': i,
                    'count': info.get('count', 0)
                })
    
    # シグナル変化のあったステップを特定
    for signal_name, changes in signal_changes.items():
        prev_count = 0
        for change in changes:
            if change['count'] != prev_count:
                execution_analysis['steps_with_signal_changes'].append({
                    'signal': signal_name,
                    'step': change['step'],
                    'step_index': change['step_index'],
                    'old_count': prev_count,
                    'new_count': change['count']
                })
                prev_count = change['count']
    
    # 価格パターンの分析
    price_patterns = analyze_price_patterns(result_df, same_day_dates)
    
    # 根本原因の推定
    root_causes = estimate_root_causes(strategy_instance, same_day_dates, result_df, execution_analysis, price_patterns)
    
    return {
        'success': True,
        'same_day_count': same_day_check['same_day_count'],
        'same_day_dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in same_day_dates],
        'execution_analysis': execution_analysis,
        'price_patterns': price_patterns,
        'root_causes': root_causes,
        'related_signals': same_day_signals
    }


def estimate_root_causes(strategy_instance, same_day_dates, result_df, execution_analysis, price_patterns):
    """
    収集した情報から同日エントリー/エグジット問題の根本原因を推定
    """
    causes = []
    strategy_name = strategy_instance.__class__.__name__
    
    # 1. 戦略タイプ別の特定パターン
    if 'Opening' in strategy_name or 'Gap' in strategy_name:
        # オープニングギャップ戦略では、ギャップとその後の価格行動が重要
        if price_patterns['gap_up_count'] > 0 or price_patterns['gap_down_count'] > 0:
            causes.append({
                'cause': 'opening_gap_reversal',
                'description': 'オープニングギャップの後に価格が反転し、同日中にエグジット条件に到達した可能性',
                'confidence': 'high' if price_patterns['price_reversals'] > 0 else 'medium'
            })
    
    if 'VWAP' in strategy_name:
        # VWAP戦略では、VWAPラインとの関係が重要
        causes.append({
            'cause': 'price_vwap_cross',
            'description': 'エントリー後に価格がVWAPラインを再度クロスし、同日中にエグジット条件に到達した可能性',
            'confidence': 'medium'
        })
    
    # 2. 価格パターンに基づく共通原因
    if price_patterns['high_volatility_count'] > price_patterns['low_volatility_count']:
        causes.append({
            'cause': 'high_intraday_volatility',
            'description': '日中の高いボラティリティにより、エントリー後すぐにエグジット条件に到達した可能性',
            'confidence': 'high',
            'avg_range': price_patterns['average_daily_range_pct']
        })
    
    if price_patterns['price_reversals'] > price_patterns['trend_days']:
        causes.append({
            'cause': 'price_trend_reversal',
            'description': 'エントリー直後に価格トレンドが反転し、エグジット条件に到達した可能性',
            'confidence': 'high'
        })
    
    # 3. コード実装の構造による原因
    # エントリー/エグジットの判定が同じ関数内で連続して行われるケース
    steps_with_entry = [step for step in execution_analysis['steps_with_signal_changes'] 
                        if 'Entry' in step['signal'] and step['new_count'] > step['old_count']]
    steps_with_exit = [step for step in execution_analysis['steps_with_signal_changes'] 
                        if 'Exit' in step['signal'] and step['new_count'] > step['old_count']]
    
    # 同じステップでエントリーとエグジットの両方が生成される場合
    same_step_signals = set(step['step'] for step in steps_with_entry) & set(step['step'] for step in steps_with_exit)
    if same_step_signals:
        causes.append({
            'cause': 'concurrent_signal_generation',
            'description': '同じコード内でエントリーとエグジットの両方の条件を評価している可能性',
            'confidence': 'very_high',
            'steps': list(same_step_signals)
        })
    
    # 4. 特定の日付のデータを詳細分析
    for date in same_day_dates:
        if date in result_df.index:
            row = result_df.loc[date]
            
            # Stop Loss/Take Profitの即時発動
            if hasattr(strategy_instance, 'stop_loss') or hasattr(strategy_instance, 'take_profit'):
                intraday_range = (row['High'] - row['Low']) / row['Open'] if row['Open'] > 0 else 0
                
                stop_loss = getattr(strategy_instance, 'stop_loss', None)
                if stop_loss and isinstance(stop_loss, (int, float)) and intraday_range > stop_loss:
                    causes.append({
                        'cause': 'immediate_stop_loss',
                        'description': f'日中の価格変動がストップロス値({stop_loss:.2%})を超え、即時にエグジットした可能性',
                        'confidence': 'high',
                        'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        'intraday_range': intraday_range
                    })
                
                take_profit = getattr(strategy_instance, 'take_profit', None)
                if take_profit and isinstance(take_profit, (int, float)) and intraday_range > take_profit:
                    causes.append({
                        'cause': 'immediate_take_profit',
                        'description': f'日中の価格変動が利確値({take_profit:.2%})を超え、即時にエグジットした可能性',
                        'confidence': 'high', 
                        'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        'intraday_range': intraday_range
                    })
    
    return causes


def create_root_cause_report(analysis_result, strategy_name, ticker):
    """
    根本原因分析のレポートを生成
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(RESULTS_DIR, f"root_cause_analysis_{strategy_name}_{ticker}_{timestamp}.json")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"根本原因分析レポートを保存しました: {report_path}")
    return report_path


def visualize_same_day_patterns(result_df, same_day_dates, strategy_name, ticker):
    """
    同日エントリー/エグジットのパターンを可視化
    """
    if not same_day_dates or not isinstance(result_df, pd.DataFrame):
        return None
    
    # 日付をdatetime形式に変換（文字列の場合）
    dates = []
    for date in same_day_dates:
        if isinstance(date, str):
            try:
                dates.append(pd.Timestamp(date))
            except:
                continue
        else:
            dates.append(date)
    
    if not dates:
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_path = os.path.join(CHARTS_DIR, f"same_day_patterns_{strategy_name}_{ticker}_{timestamp}.png")
    
    # 各同日エントリー/エグジット日の周辺3日間を抽出して可視化
    fig, axes = plt.subplots(len(dates), 1, figsize=(10, 5 * len(dates)), squeeze=False)
    
    for i, date in enumerate(dates):
        if date in result_df.index:
            date_idx = result_df.index.get_loc(date)
            
            # 前後3日間を抽出
            start_idx = max(0, date_idx - 3)
            end_idx = min(len(result_df) - 1, date_idx + 3)
            
            subset = result_df.iloc[start_idx:end_idx + 1]
            
            # プロット
            ax = axes[i, 0]
            ax.plot(subset.index, subset['Close'], label='Close')
            
            if 'VWAP' in subset.columns:
                ax.plot(subset.index, subset['VWAP'], label='VWAP', linestyle='--')
            
            # エントリー/エグジットポイントをマーク
            entry_points = subset[subset['Entry_Signal'] == 1]
            exit_points = subset[subset['Exit_Signal'] == 1]
            
            ax.scatter(entry_points.index, entry_points['Close'], color='green', marker='^', s=100, label='Entry')
            ax.scatter(exit_points.index, exit_points['Close'], color='red', marker='v', s=100, label='Exit')
            
            # 日付を強調
            ax.axvline(x=date, color='blue', linestyle='--', alpha=0.5)
            
            ax.set_title(f"{date.strftime('%Y-%m-%d')} の同日エントリー/エグジット")
            ax.legend()
            ax.grid(True)
            
            # 日付フォーマットの調整
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    
    logger.info(f"同日エントリー/エグジットパターンの可視化を保存しました: {chart_path}")
    return chart_path


def analyze_root_causes(ticker, strategy_name, start_date='2024-01-01', end_date='2024-12-31', params=None):
    """
    戦略の根本原因分析を実行
    """
    logger.info(f"===== {strategy_name}の根本原因分析開始 =====")
    logger.info(f"銘柄: {ticker}, 期間: {start_date} から {end_date}")
    
    try:
        # 戦略クラスの選択
        strategy_classes = {
            'VWAPBreakoutStrategy': VWAPBreakoutStrategy,
            'OpeningGapStrategy': OpeningGapStrategy,
            'OpeningGapFixedStrategy': OpeningGapFixedStrategy
        }
        
        if strategy_name not in strategy_classes:
            logger.error(f"未対応の戦略: {strategy_name}")
            return {'success': False, 'error': f"未対応の戦略: {strategy_name}"}
        
        # データ取得 (get_parameters_and_data は5つの値を返すので、必要なものだけ取得)
        result = get_parameters_and_data(ticker, start_date, end_date)
        if isinstance(result, tuple) and len(result) >= 5:
            _, _, _, stock_data, market_data = result
        else:
            logger.error("データ取得に失敗しました")
            return {'success': False, 'error': "データ取得に失敗しました"}
        
        # データ前処理
        stock_data = preprocess_data(stock_data)
        
        # インジケータ計算
        compute_indicators(stock_data)
        
        # デフォルトパラメータの設定
        if not params:
            params = {}
            # 戦略ごとのデフォルトパラメータ
            if strategy_name == 'VWAPBreakoutStrategy':
                params = {
                    'vwap_period': 20,
                    'atr_period': 14,
                    'atr_multiplier': 1.5,
                    'stop_loss': 0.03,
                    'take_profit': 0.06
                }
            elif strategy_name == 'OpeningGapStrategy':
                params = {
                    'gap_threshold': 0.01,
                    'volume_threshold': 1.2,
                    'stop_loss': 0.02,
                    'take_profit': 0.04
                }
            elif strategy_name == 'OpeningGapFixedStrategy':
                params = {
                    'gap_threshold': 0.01,
                    'stop_loss': 0.02,
                    'take_profit': 0.04
                }
        
        # トレース用の拡張戦略クラスを取得
        TracedStrategy = create_strategy_execution_wrapper(strategy_classes[strategy_name])
        
        # 戦略インスタンス作成（戦略ごとに必要な引数を調整）
        if strategy_name == 'VWAPBreakoutStrategy':
            strategy = TracedStrategy(
                data=stock_data,
                index_data=market_data,
                params=params,
                price_column="Adj Close"
            )
        elif strategy_name in ['OpeningGapStrategy', 'OpeningGapFixedStrategy']:
            strategy = TracedStrategy(
                data=stock_data,
                dow_data=market_data,
                params=params,
                price_column="Adj Close"
            )
        else:
            strategy = TracedStrategy(
                data=stock_data,
                params=params,
                price_column="Adj Close"
            )
        
        # バックテスト実行
        logger.info(f"バックテスト実行中: {strategy_name}")
        result_df = strategy.backtest()
        
        # 同日エントリー/エグジット問題の検出
        same_day_check = check_same_day_entry_exit(result_df)
        if not same_day_check['has_same_day_signals']:
            logger.info("同日エントリー/エグジットは検出されませんでした。")
            return {
                'success': True,
                'has_same_day_signals': False,
                'message': "同日エントリー/エグジットは検出されませんでした。"
            }
        
        # 根本原因の分析
        analysis_result = analyze_strategy_internals(strategy, result_df)
        
        # レポート生成
        report_path = create_root_cause_report(analysis_result, strategy_name, ticker)
        
        # 可視化
        chart_path = visualize_same_day_patterns(
            result_df, 
            same_day_check.get('dates', []),
            strategy_name, 
            ticker
        )
        
        # 最終結果
        result = {
            'success': True,
            'has_same_day_signals': True,
            'same_day_count': same_day_check['same_day_count'],
            'report_path': report_path,
            'chart_path': chart_path,
            'root_causes_summary': [cause['description'] for cause in analysis_result.get('root_causes', [])]
        }
        
        logger.info(f"分析完了: 同日エントリー/エグジット {same_day_check['same_day_count']}件")
        return result
        
    except Exception as e:
        logger.error(f"分析中にエラー発生: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}


def main():
    """メイン実行関数"""
    print("===== 同日エントリー/エグジット問題の根本原因分析 =====")
    
    import argparse
    parser = argparse.ArgumentParser(description='同日エントリー/エグジット問題の根本原因分析')
    parser.add_argument('--ticker', type=str, default='7203.T', help='対象の銘柄コード')
    parser.add_argument('--strategy', type=str, default='VWAPBreakoutStrategy', 
                        choices=['VWAPBreakoutStrategy', 'OpeningGapStrategy', 'OpeningGapFixedStrategy'],
                        help='分析対象の戦略名')
    parser.add_argument('--start', type=str, default='2024-01-01', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='終了日 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print(f"対象銘柄: {args.ticker}")
    print(f"対象戦略: {args.strategy}")
    print(f"期間: {args.start} から {args.end}")
    
    # 根本原因分析の実行
    result = analyze_root_causes(
        ticker=args.ticker,
        strategy_name=args.strategy,
        start_date=args.start,
        end_date=args.end
    )
    
    # 結果表示
    if result['success']:
        if result.get('has_same_day_signals', False):
            print(f"\n同日エントリー/エグジット検出: {result['same_day_count']}件")
            print("\n推定される根本原因:")
            for i, cause in enumerate(result.get('root_causes_summary', []), 1):
                print(f"{i}. {cause}")
            
            print(f"\n詳細レポート: {result.get('report_path', 'なし')}")
            print(f"パターン可視化: {result.get('chart_path', 'なし')}")
        else:
            print("\n同日エントリー/エグジットは検出されませんでした。")
    else:
        print(f"\n分析エラー: {result.get('error', '不明なエラー')}")
    
    print("\n===== 分析完了 =====")


if __name__ == "__main__":
    main()