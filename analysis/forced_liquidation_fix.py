"""
強制決済計算ロジック修正モジュール
TODO #3: 強制決済率=-2.6%という数学的に不可能な値を正常な計算に修正

目的: Phase3テスト結果で発生した強制決済計算エラーを修正し、取引履歴整合性を復旧
バックテスト基本理念遵守: 実際のbacktest()実行結果に基づく正確な計算実装
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os

# TODO(tag:forced_liquidation_fix, rationale:correct mathematical calculation)
def fix_forced_liquidation_calculation_logic():
    """
    強制決済計算ロジック修正
    バックテスト基本理念遵守: 実際のbacktest結果に基づく正確な計算
    """
    logger = logging.getLogger(__name__)
    
    print("=== 強制決済計算ロジック修正開始 ===")
    
    try:
        # 実際のバックテストデータ取得（バックテスト基本理念遵守）
        sys.path.append('.')
        from main import apply_strategies_with_optimized_params
        from data_fetcher import get_parameters_and_data
        from config.optimized_parameters import OptimizedParameterManager
        
        # バックテスト基本理念遵守: 実際のデータ・パラメータで実行
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # パラメータマネージャーを使用
        param_manager = OptimizedParameterManager()
        optimized_params = param_manager.load_approved_params(ticker)
        
        print(f"修正対象データ: {ticker} ({start_date} - {end_date})")
        print(f"データ行数: {len(stock_data)}")
        
        # バックテスト基本理念遵守: 実際の統合backtest実行
        integrated_data = apply_strategies_with_optimized_params(stock_data, index_data, optimized_params)
        
        print(f"統合データ行数: {len(integrated_data)}")
        print(f"統合データ列: {list(integrated_data.columns)}")
        
        # 修正された強制決済計算ロジック
        corrected_metrics = calculate_corrected_forced_liquidation_metrics(integrated_data)
        
        # 修正前後比較
        original_metrics = calculate_original_flawed_metrics(integrated_data)
        
        # 結果出力・検証
        print_correction_results(original_metrics, corrected_metrics)
        
        # バックテスト基本理念違反検出
        validate_liquidation_calculation_compliance(corrected_metrics)
        
        return corrected_metrics
        
    except Exception as e:
        logger.error(f"強制決済計算修正エラー: {e}")
        print(f"❌ 修正エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_corrected_forced_liquidation_metrics(integrated_data: pd.DataFrame) -> Dict:
    """
    修正された強制決済計算ロジック
    TODO(tag:forced_liquidation_fix, rationale:implement mathematically correct calculation)
    """
    metrics = {}
    
    # 基本統計（バックテスト基本理念遵守: 実際のシグナルから計算）
    if 'Entry_Signal' in integrated_data.columns and 'Exit_Signal' in integrated_data.columns:
        total_entries = (integrated_data['Entry_Signal'] == 1).sum()
        total_exits = (integrated_data['Exit_Signal'] == -1).sum()
    else:
        print("⚠️ Entry_Signal/Exit_Signal列が見つかりません")
        total_entries = 0
        total_exits = 0
    
    metrics['basic_stats'] = {
        'total_entries': int(total_entries),
        'total_exits': int(total_exits),
        'position_balance': int(total_entries - total_exits)
    }
    
    # TODO(tag:forced_liquidation_fix, rationale:correct final day exit calculation)
    # 修正: 最終日エグジット数の正しい計算
    if len(integrated_data) > 0 and 'Exit_Signal' in integrated_data.columns:
        final_date = integrated_data.index[-1]
        
        # 修正前の間違った計算を回避
        # 間違い: final_day_exits = integrated_data.iloc[-1]['Exit_Signal']  # スカラー値
        # 正解: 最終日の実際のエグジットシグナル数をカウント
        final_day_exit_signals = integrated_data.loc[final_date, 'Exit_Signal']
        
        # エグジットシグナルは-1で表現される場合が多いため、絶対値で正規化
        if final_day_exit_signals == -1:
            actual_final_day_exits = 1
        elif final_day_exit_signals == 0:
            actual_final_day_exits = 0
        else:
            # 複数エグジットの場合（統合システムでは稀）
            actual_final_day_exits = abs(int(final_day_exit_signals))
        
        metrics['final_day_analysis'] = {
            'final_date': str(final_date),
            'raw_exit_signal_value': float(final_day_exit_signals),
            'actual_final_day_exits': actual_final_day_exits
        }
    else:
        metrics['final_day_analysis'] = {
            'final_date': 'NO_DATA',
            'raw_exit_signal_value': 0,
            'actual_final_day_exits': 0
        }
        actual_final_day_exits = 0
    
    # TODO(tag:forced_liquidation_fix, rationale:implement proper forced liquidation detection)
    # 修正: 強制決済の正しい定義と計算
    
    # 強制決済の定義: バックテスト期間終了時点で未決済のポジションを強制的にクローズすること
    # 計算方法: 最終営業日のエグジット + 期間終了時の未決済ポジション数
    
    # 1. 期間終了時の未決済ポジション計算
    active_positions_at_end = 0
    if 'Active_Strategy' in integrated_data.columns:
        active_strategy_final = integrated_data.iloc[-1]['Active_Strategy']
        if hasattr(active_strategy_final, 'sum'):
            active_positions_at_end = (active_strategy_final != '').sum()
        else:
            active_positions_at_end = 1 if str(active_strategy_final) != '' else 0
    else:
        # フォールバック: エントリー・エグジット差分
        active_positions_at_end = max(0, total_entries - total_exits)
    
    # 2. 実際の強制決済数 = 最終日のエグジット数 + 未決済ポジション
    total_forced_liquidations = actual_final_day_exits + active_positions_at_end
    
    # 3. 修正された強制決済率計算
    if total_exits > 0:
        corrected_forced_liquidation_rate = (total_forced_liquidations / total_exits) * 100
    else:
        corrected_forced_liquidation_rate = 0.0
    
    metrics['forced_liquidation_analysis'] = {
        'forced_liquidations_final_day': actual_final_day_exits,
        'active_positions_at_end': active_positions_at_end,
        'total_forced_liquidations': total_forced_liquidations,
        'corrected_forced_liquidation_rate': round(corrected_forced_liquidation_rate, 2)
    }
    
    # TODO(tag:forced_liquidation_fix, rationale:comprehensive liquidation pattern analysis)
    # 追加分析: 期間全体での強制決済パターン分析
    liquidation_patterns = analyze_liquidation_patterns(integrated_data)
    metrics['liquidation_patterns'] = liquidation_patterns
    
    return metrics

def calculate_original_flawed_metrics(integrated_data: pd.DataFrame) -> Dict:
    """
    修正前の間違った計算ロジック（比較用）
    TODO(tag:forced_liquidation_fix, rationale:demonstrate original calculation flaws)
    """
    original_metrics = {}
    
    if 'Exit_Signal' in integrated_data.columns:
        total_exits = (integrated_data['Exit_Signal'] == -1).sum()
        
        # 修正前の間違った計算（Phase3テストで発生したエラー）
        if len(integrated_data) > 0:
            # 間違い1: スカラー値を直接使用
            final_day_exit_raw = integrated_data.iloc[-1]['Exit_Signal']
            
            # 間違い2: 負の値をそのまま使用して異常な割合計算
            if total_exits > 0:
                original_flawed_rate = (final_day_exit_raw / total_exits) * 100
            else:
                original_flawed_rate = 0.0
            
            original_metrics['flawed_calculation'] = {
                'final_day_exit_raw': float(final_day_exit_raw),
                'total_exits': int(total_exits),
                'flawed_rate': round(original_flawed_rate, 2),
                'flaw_explanation': 'Used raw signal value (-1) instead of actual exit count (1)'
            }
    
    return original_metrics

def analyze_liquidation_patterns(integrated_data: pd.DataFrame) -> Dict:
    """
    強制決済パターン分析
    TODO(tag:forced_liquidation_fix, rationale:identify liquidation timing patterns)
    """
    patterns = {}
    
    # エグジット発生タイミング分析
    if 'Exit_Signal' in integrated_data.columns:
        exit_dates = integrated_data[integrated_data['Exit_Signal'] == -1].index
        
        if len(exit_dates) > 0:
            # 期間の分割（前半・中盤・後半）
            total_days = len(integrated_data)
            first_third = total_days // 3
            second_third = (total_days * 2) // 3
            
            early_exits = sum(1 for date in exit_dates if integrated_data.index.get_loc(date) < first_third)
            middle_exits = sum(1 for date in exit_dates if first_third <= integrated_data.index.get_loc(date) < second_third)
            late_exits = sum(1 for date in exit_dates if integrated_data.index.get_loc(date) >= second_third)
            
            patterns['timing_analysis'] = {
                'early_period_exits': early_exits,
                'middle_period_exits': middle_exits,
                'late_period_exits': late_exits,
                'late_exit_concentration': round((late_exits / len(exit_dates)) * 100, 1) if len(exit_dates) > 0 else 0
            }
            
            # 最終週のエグジット集中度
            if len(integrated_data) >= 5:
                final_week_exits = sum(1 for date in exit_dates if integrated_data.index.get_loc(date) >= len(integrated_data) - 5)
                final_week_concentration = (final_week_exits / len(exit_dates)) * 100 if len(exit_dates) > 0 else 0
                
                patterns['final_week_analysis'] = {
                    'final_week_exits': final_week_exits,
                    'final_week_concentration': round(final_week_concentration, 1)
                }
    
    return patterns

def validate_liquidation_calculation_compliance(corrected_metrics: Dict):
    """
    強制決済計算のバックテスト基本理念違反検出
    TODO(tag:forced_liquidation_fix, rationale:ensure calculation integrity)
    """
    violations = []
    
    # 基本理念チェック1: 数学的整合性
    forced_rate = corrected_metrics['forced_liquidation_analysis']['corrected_forced_liquidation_rate']
    if forced_rate < 0:
        violations.append(f"Negative forced liquidation rate: {forced_rate}%")
    
    if forced_rate > 100:
        violations.append(f"Impossible forced liquidation rate: {forced_rate}%")
    
    # 基本理念チェック2: データ整合性
    total_entries = corrected_metrics['basic_stats']['total_entries']
    total_exits = corrected_metrics['basic_stats']['total_exits']
    
    if total_entries == 0 and total_exits > 0:
        violations.append("Exits without entries - data integrity violation")
    
    # 基本理念チェック3: 取引履歴の論理的整合性
    position_balance = corrected_metrics['basic_stats']['position_balance']
    if position_balance < -10:  # -10は許容範囲（エラーマージン）
        violations.append(f"Excessive exit surplus: {abs(position_balance)} more exits than entries")
    
    if violations:
        error_msg = f"Forced liquidation calculation violations: {'; '.join(violations)}"
        raise ValueError(f"{error_msg} TODO(tag:backtest_execution, rationale:fix calculation violations)")
    
    return True

def print_correction_results(original_metrics: Dict, corrected_metrics: Dict):
    """
    修正結果出力
    TODO(tag:forced_liquidation_fix, rationale:comprehensive correction reporting)
    """
    print("\n" + "="*60)
    print("📊 強制決済計算ロジック修正結果")
    print("="*60)
    
    # 修正前後比較
    if 'flawed_calculation' in original_metrics:
        flawed = original_metrics['flawed_calculation']
        print(f"\n❌ 修正前（間違った計算）:")
        print(f"  最終日エグジット値（生値）: {flawed['final_day_exit_raw']}")
        print(f"  総エグジット数: {flawed['total_exits']}")
        print(f"  間違った強制決済率: {flawed['flawed_rate']}%")
        print(f"  問題: {flawed['flaw_explanation']}")
    
    # 修正後の正しい計算
    corrected = corrected_metrics['forced_liquidation_analysis']
    basic = corrected_metrics['basic_stats']
    
    print(f"\n✅ 修正後（正しい計算）:")
    print(f"  総エントリー数: {basic['total_entries']}")
    print(f"  総エグジット数: {basic['total_exits']}")
    print(f"  最終日強制決済数: {corrected['forced_liquidations_final_day']}")
    print(f"  期末未決済ポジション: {corrected['active_positions_at_end']}")
    print(f"  総強制決済数: {corrected['total_forced_liquidations']}")
    print(f"  正しい強制決済率: {corrected['corrected_forced_liquidation_rate']}%")
    
    # パターン分析結果
    if 'timing_analysis' in corrected_metrics['liquidation_patterns']:
        timing = corrected_metrics['liquidation_patterns']['timing_analysis']
        print(f"\n📈 エグジットタイミング分析:")
        print(f"  前期エグジット: {timing['early_period_exits']}回")
        print(f"  中期エグジット: {timing['middle_period_exits']}回")
        print(f"  後期エグジット: {timing['late_period_exits']}回")
        print(f"  後期集中度: {timing['late_exit_concentration']}%")
    
    # 修正効果の評価
    print(f"\n🎯 修正効果評価:")
    if corrected['corrected_forced_liquidation_rate'] >= 0 and corrected['corrected_forced_liquidation_rate'] <= 100:
        print("✅ 数学的に正常な範囲の強制決済率")
    else:
        print("⚠️ 強制決済率が異常範囲 - さらなる調査が必要")
    
    if corrected['corrected_forced_liquidation_rate'] <= 20:
        print("✅ 健全な強制決済率（20%以下）")
    elif corrected['corrected_forced_liquidation_rate'] <= 50:
        print("⚠️ やや高い強制決済率（20-50%）")
    else:
        print("❌ 高い強制決済率（50%超） - 戦略調整推奨")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # 直接実行時のテスト
    print("=== 強制決済計算ロジック修正 直接実行テスト ===")
    corrected_metrics = fix_forced_liquidation_calculation_logic()
    
    if corrected_metrics:
        print("\n✅ 修正モジュール動作確認完了")
    else:
        print("\n❌ 修正モジュール動作エラー")