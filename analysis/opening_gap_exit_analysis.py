import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# TODO(tag:strategy_analysis, rationale:validate opening_gap exit logic)
def investigate_opening_gap_exit_anomaly():
    """
    OpeningGapStrategy大量エグジット調査
    バックテスト基本理念遵守: 実際の戦略実行による詳細分析
    """
    logger = logging.getLogger(__name__)
    
    print("=== OpeningGapStrategy大量エグジット調査開始 ===")
    
    try:
        # データ取得
        sys.path.append('.')
        from data_fetcher import get_parameters_and_data
        from src.strategies.Opening_Gap import OpeningGapStrategy
        from config.optimized_parameters import OptimizedParameterManager
        
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        param_manager = OptimizedParameterManager()
        optimized_params = {'OpeningGapStrategy': param_manager.load_approved_params('OpeningGapStrategy', ticker)}
        
        print(f"調査対象: {ticker} ({start_date} - {end_date})")
        print(f"データ数: {len(stock_data)}行")
        
        # バックテスト基本理念遵守: 実際の戦略実行
        og_params = optimized_params.get('OpeningGapStrategy', {})
        og_strategy = OpeningGapStrategy(
            data=stock_data, 
            dow_data=index_data, 
            params=og_params, 
            price_column='Close'
        )
        
        # 実際のbacktest()実行
        strategy_result = og_strategy.backtest()
        
        # エグジットシグナル詳細分析
        exit_analysis = analyze_exit_signals(strategy_result, stock_data)
        
        # パラメータ設定分析
        param_analysis = analyze_strategy_parameters(og_params, og_strategy)
        
        # 保有期間分析
        holding_period_analysis = analyze_holding_periods(strategy_result)
        
        # 総合判定
        final_judgment = make_final_judgment(exit_analysis, param_analysis, holding_period_analysis)
        
        # 結果出力
        print_investigation_results(exit_analysis, param_analysis, holding_period_analysis, final_judgment)
        
        return final_judgment
        
    except Exception as e:
        logger.error(f"OpeningGap調査中エラー: {e}")
        print(f"[ERROR] 調査エラー: {e}")
        return None

def analyze_exit_signals(strategy_result: pd.DataFrame, stock_data: pd.DataFrame) -> Dict:
    """
    エグジットシグナル詳細分析
    TODO(tag:strategy_analysis, rationale:understand exit signal generation pattern)
    """
    analysis = {}
    
    # 基本統計
    total_entries = (strategy_result['Entry_Signal'] == 1).sum()
    total_exits = (strategy_result['Exit_Signal'] == -1).sum()
    
    analysis['basic_stats'] = {
        'total_entries': int(total_entries),
        'total_exits': int(total_exits),
        'exit_entry_ratio': float(total_exits / total_entries) if total_entries > 0 else 0
    }
    
    # エグジット発生パターン分析
    exit_dates = strategy_result[strategy_result['Exit_Signal'] == -1].index
    
    if len(exit_dates) > 0:
        # エグジット発生間隔
        exit_intervals = []
        for i in range(1, len(exit_dates)):
            interval = (exit_dates[i] - exit_dates[i-1]).days
            exit_intervals.append(interval)
        
        analysis['exit_patterns'] = {
            'first_exit_date': str(exit_dates[0]),
            'last_exit_date': str(exit_dates[-1]),
            'avg_exit_interval': float(np.mean(exit_intervals)) if exit_intervals else 0,
            'min_exit_interval': int(min(exit_intervals)) if exit_intervals else 0,
            'max_exit_interval': int(max(exit_intervals)) if exit_intervals else 0
        }
        
        # 連続エグジット検出
        consecutive_exits = detect_consecutive_exits(strategy_result)
        analysis['consecutive_exits'] = consecutive_exits
    
    # エグジット原因分析
    exit_causes = analyze_exit_causes(strategy_result, stock_data)
    analysis['exit_causes'] = exit_causes
    
    return analysis

def analyze_strategy_parameters(params: Dict, strategy_instance) -> Dict:
    """
    戦略パラメータ分析
    TODO(tag:strategy_analysis, rationale:validate parameter settings impact)
    """
    analysis = {}
    
    # パラメータ取得
    analysis['current_params'] = params.copy()
    
    # デフォルトパラメータとの比較
    try:
        default_params = strategy_instance.get_default_parameters() if hasattr(strategy_instance, 'get_default_parameters') else {}
        analysis['default_params'] = default_params
        
        # パラメータ差分
        param_diffs = {}
        for key in set(params.keys()) | set(default_params.keys()):
            current_val = params.get(key, 'NOT_SET')
            default_val = default_params.get(key, 'NOT_SET')
            if current_val != default_val:
                param_diffs[key] = {'current': current_val, 'default': default_val}
        
        analysis['parameter_differences'] = param_diffs
        
    except Exception as e:
        analysis['parameter_analysis_error'] = str(e)
    
    # 保有期間関連パラメータ特定
    holding_period_params = {}
    for key, value in params.items():
        if any(keyword in key.lower() for keyword in ['period', 'days', 'hold', 'max', 'duration']):
            holding_period_params[key] = value
    
    analysis['holding_period_params'] = holding_period_params
    
    return analysis

def analyze_holding_periods(strategy_result: pd.DataFrame) -> Dict:
    """
    保有期間分析
    TODO(tag:strategy_analysis, rationale:identify abnormal holding period patterns)
    """
    analysis = {}
    
    # エントリー・エグジット対応分析
    entry_dates = strategy_result[strategy_result['Entry_Signal'] == 1].index
    exit_dates = strategy_result[strategy_result['Exit_Signal'] == -1].index
    
    holding_periods = []
    unmatched_entries = 0
    
    for entry_date in entry_dates:
        # 該当エントリー後の最初のエグジットを検索
        future_exits = exit_dates[exit_dates > entry_date]
        
        if len(future_exits) > 0:
            exit_date = future_exits[0]
            holding_period = (exit_date - entry_date).days
            holding_periods.append(holding_period)
        else:
            unmatched_entries += 1
    
    if holding_periods:
        analysis['holding_period_stats'] = {
            'avg_holding_period': float(np.mean(holding_periods)),
            'min_holding_period': int(min(holding_periods)),
            'max_holding_period': int(max(holding_periods)),
            'median_holding_period': float(np.median(holding_periods)),
            'std_holding_period': float(np.std(holding_periods))
        }
        
        # 異常な保有期間検出（0日または極端に長い）
        zero_day_holds = sum(1 for p in holding_periods if p == 0)
        long_holds = sum(1 for p in holding_periods if p > 30)  # 30日超
        
        analysis['anomaly_detection'] = {
            'zero_day_holdings': zero_day_holds,
            'long_holdings_over_30days': long_holds,
            'total_holdings': len(holding_periods),
            'unmatched_entries': unmatched_entries
        }
    
    return analysis

def detect_consecutive_exits(strategy_result: pd.DataFrame) -> Dict:
    """
    連続エグジット検出
    TODO(tag:strategy_analysis, rationale:identify bulk exit events)
    """
    exit_signals = strategy_result['Exit_Signal'] == -1
    consecutive_groups = []
    current_group = []
    
    for date, is_exit in exit_signals.items():
        if is_exit:
            current_group.append(date)
        else:
            if current_group and len(current_group) > 1:
                consecutive_groups.append(current_group)
            current_group = []
    
    # 最後のグループも追加
    if current_group and len(current_group) > 1:
        consecutive_groups.append(current_group)
    
    return {
        'consecutive_exit_groups': len(consecutive_groups),
        'largest_consecutive_group': max([len(group) for group in consecutive_groups]) if consecutive_groups else 0,
        'total_consecutive_exits': sum([len(group) for group in consecutive_groups])
    }

def analyze_exit_causes(strategy_result: pd.DataFrame, stock_data: pd.DataFrame) -> Dict:
    """
    エグジット原因分析
    TODO(tag:strategy_analysis, rationale:identify root cause of mass exits)
    """
    causes = {
        'profit_taking': 0,
        'stop_loss': 0,
        'max_holding_period': 0,
        'unknown': 0
    }
    
    # エグジット発生日の価格変動分析
    exit_dates = strategy_result[strategy_result['Exit_Signal'] == -1].index
    
    for exit_date in exit_dates:
        try:
            # 前日との価格比較
            if exit_date in stock_data.index:
                current_price = stock_data.loc[exit_date, 'Close']
                previous_dates = stock_data.index[stock_data.index < exit_date]
                
                if len(previous_dates) > 0:
                    previous_price = stock_data.loc[previous_dates[-1], 'Close']
                    price_change = (current_price - previous_price) / previous_price
                    
                    # 原因推定（簡易版）
                    if price_change > 0.02:  # 2%以上の上昇
                        causes['profit_taking'] += 1
                    elif price_change < -0.02:  # 2%以上の下落
                        causes['stop_loss'] += 1
                    else:
                        causes['max_holding_period'] += 1
                else:
                    causes['unknown'] += 1
            else:
                causes['unknown'] += 1
                
        except Exception:
            causes['unknown'] += 1
    
    return causes

def make_final_judgment(exit_analysis: Dict, param_analysis: Dict, holding_analysis: Dict) -> Dict:
    """
    総合判定
    TODO(tag:strategy_analysis, rationale:determine if exit behavior is normal or problematic)
    """
    judgment = {
        'is_abnormal': False,
        'severity': 'NORMAL',
        'issues_found': [],
        'recommendations': []
    }
    
    # 異常判定基準
    exit_entry_ratio = exit_analysis['basic_stats'].get('exit_entry_ratio', 0)
    
    # 1. エグジット/エントリー比率チェック
    if exit_entry_ratio > 10:  # エグジットがエントリーの10倍以上
        judgment['is_abnormal'] = True
        judgment['severity'] = 'CRITICAL'
        judgment['issues_found'].append(f"異常なエグジット/エントリー比率: {exit_entry_ratio:.1f}")
        judgment['recommendations'].append("最大保有期間パラメータの見直しが必要")
    
    # 2. 0日保有の大量発生チェック
    if 'anomaly_detection' in holding_analysis:
        zero_day_ratio = holding_analysis['anomaly_detection']['zero_day_holdings'] / max(holding_analysis['anomaly_detection']['total_holdings'], 1)
        if zero_day_ratio > 0.5:  # 50%以上が0日保有
            judgment['is_abnormal'] = True
            judgment['severity'] = 'HIGH' if judgment['severity'] == 'NORMAL' else judgment['severity']
            judgment['issues_found'].append(f"0日保有の大量発生: {zero_day_ratio:.1%}")
            judgment['recommendations'].append("エントリー・エグジット条件の見直しが必要")
    
    # 3. 連続エグジットの大量発生チェック
    consecutive_exits = exit_analysis.get('consecutive_exits', {})
    if consecutive_exits.get('largest_consecutive_group', 0) > 20:
        judgment['is_abnormal'] = True
        judgment['severity'] = 'HIGH' if judgment['severity'] == 'NORMAL' else judgment['severity']
        judgment['issues_found'].append(f"大量連続エグジット: {consecutive_exits['largest_consecutive_group']}回")
        judgment['recommendations'].append("最大保有期間の一括エグジット機能の確認が必要")
    
    return judgment

def print_investigation_results(exit_analysis: Dict, param_analysis: Dict, holding_analysis: Dict, judgment: Dict):
    """
    調査結果出力
    TODO(tag:strategy_analysis, rationale:comprehensive result reporting)
    """
    print("\n" + "="*60)
    print("[CHART] OpeningGapStrategy大量エグジット調査結果")
    print("="*60)
    
    # 基本統計
    basic_stats = exit_analysis['basic_stats']
    print(f"\n🔢 基本統計:")
    print(f"  エントリー数: {basic_stats['total_entries']}回")
    print(f"  エグジット数: {basic_stats['total_exits']}回")
    print(f"  エグジット/エントリー比率: {basic_stats['exit_entry_ratio']:.2f}")
    
    # 保有期間分析
    if 'holding_period_stats' in holding_analysis:
        hold_stats = holding_analysis['holding_period_stats']
        print(f"\n📅 保有期間分析:")
        print(f"  平均保有期間: {hold_stats['avg_holding_period']:.1f}日")
        print(f"  最短保有期間: {hold_stats['min_holding_period']}日")
        print(f"  最長保有期間: {hold_stats['max_holding_period']}日")
        print(f"  中央値: {hold_stats['median_holding_period']:.1f}日")
        
        if 'anomaly_detection' in holding_analysis:
            anomaly = holding_analysis['anomaly_detection']
            print(f"\n[WARNING] 異常検出:")
            print(f"  0日保有: {anomaly['zero_day_holdings']}件")
            print(f"  30日超保有: {anomaly['long_holdings_over_30days']}件")
            print(f"  未決済エントリー: {anomaly['unmatched_entries']}件")
    
    # エグジット原因分析
    exit_causes = exit_analysis.get('exit_causes', {})
    if exit_causes:
        print(f"\n[TARGET] エグジット原因分析:")
        print(f"  利益確定: {exit_causes['profit_taking']}回")
        print(f"  ストップロス: {exit_causes['stop_loss']}回")
        print(f"  最大保有期間: {exit_causes['max_holding_period']}回")
        print(f"  不明: {exit_causes['unknown']}回")
    
    # パラメータ分析
    holding_params = param_analysis.get('holding_period_params', {})
    if holding_params:
        print(f"\n⚙️ 保有期間関連パラメータ:")
        for key, value in holding_params.items():
            print(f"  {key}: {value}")
    
    # 総合判定
    print(f"\n[TARGET] 総合判定:")
    print(f"  異常判定: {'異常' if judgment['is_abnormal'] else '正常'}")
    print(f"  重要度: {judgment['severity']}")
    
    if judgment['issues_found']:
        print(f"\n[ERROR] 発見された問題:")
        for issue in judgment['issues_found']:
            print(f"  - {issue}")
    
    if judgment['recommendations']:
        print(f"\n[IDEA] 推奨対応:")
        for rec in judgment['recommendations']:
            print(f"  - {rec}")
    
    print("\n" + "="*60)