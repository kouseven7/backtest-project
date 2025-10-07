import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings

class TradeHistoryValidator:
    """
    取引履歴整合性検証システム
    TODO(tag:trade_history_validation, rationale:ensure backtest integrity)
    バックテスト基本理念遵守: 実際のbacktest結果の品質検証
    """
    
    def __init__(self, enable_detailed_logging: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_detailed_logging = enable_detailed_logging
        self.validation_results = {}
        self.critical_violations = []
        self.warnings_list = []
        
    def validate_integrated_backtest_results(self, integrated_data: pd.DataFrame, 
                                           strategy_results: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        統合バックテスト結果の包括的整合性検証
        バックテスト基本理念遵守: 実際の取引履歴を検証
        """
        try:
            print("=== 取引履歴整合性検証開始 ===")
            
            # 基本整合性検証
            basic_integrity = self._validate_basic_signal_integrity(integrated_data)
            
            # エントリー・エグジット比率検証
            ratio_analysis = self._validate_entry_exit_ratios(integrated_data)
            
            # 時系列整合性検証
            temporal_integrity = self._validate_temporal_consistency(integrated_data)
            
            # ポジション状態整合性検証
            position_integrity = self._validate_position_state_consistency(integrated_data)
            
            # 強制決済検証（TODO #3連携）
            forced_liquidation_analysis = self._validate_forced_liquidation_logic(integrated_data)
            
            # 戦略別整合性検証（利用可能な場合）
            strategy_consistency = None
            if strategy_results:
                strategy_consistency = self._validate_strategy_level_consistency(
                    integrated_data, strategy_results
                )
            
            # 検証結果統合
            validation_summary = self._compile_validation_results(
                basic_integrity, ratio_analysis, temporal_integrity, 
                position_integrity, forced_liquidation_analysis, strategy_consistency
            )
            
            # 詳細レポート出力
            self._print_validation_report(validation_summary)
            
            # バックテスト基本理念違反検出
            self._detect_backtest_principle_violations(validation_summary)
            
            return validation_summary
            
        except Exception as e:
            self.logger.error(f"Trade history validation failed: {e}")
            raise ValueError(f"Validation system failure: {e} TODO(tag:trade_history_validation, rationale:fix validation system)")
    
    def _validate_basic_signal_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        基本シグナル整合性検証
        TODO(tag:trade_history_validation, rationale:verify signal column integrity)
        """
        integrity_results = {
            'column_existence': True,
            'signal_counts': {},
            'data_quality': {},
            'violations': []
        }
        
        # 必須列存在チェック
        required_columns = ['Entry_Signal', 'Exit_Signal', 'Active_Strategy']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            integrity_results['column_existence'] = False
            integrity_results['violations'].append(f"Missing required columns: {missing_columns}")
        
        if 'Entry_Signal' in data.columns and 'Exit_Signal' in data.columns:
            # シグナル数カウント
            entry_count = (data['Entry_Signal'] == 1).sum()
            exit_count = (data['Exit_Signal'] == -1).sum()
            
            integrity_results['signal_counts'] = {
                'total_entries': int(entry_count),
                'total_exits': int(exit_count),
                'signal_balance': int(entry_count - exit_count)
            }
            
            # データ品質チェック
            invalid_entries = data['Entry_Signal'].isin([0, 1]).sum() != len(data)
            invalid_exits = data['Exit_Signal'].isin([0, -1]).sum() != len(data)
            
            if invalid_entries or invalid_exits:
                integrity_results['data_quality']['invalid_signals'] = True
                integrity_results['violations'].append("Invalid signal values detected")
        
        return integrity_results
    
    def _validate_entry_exit_ratios(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        エントリー・エグジット比率分析
        TODO(tag:trade_history_validation, rationale:detect abnormal entry-exit patterns)
        """
        ratio_analysis = {
            'exit_entry_ratio': 0.0,
            'ratio_assessment': 'unknown',
            'anomaly_detected': False,
            'anomaly_severity': 'none',
            'details': {}
        }
        
        if 'Entry_Signal' in data.columns and 'Exit_Signal' in data.columns:
            total_entries = (data['Entry_Signal'] == 1).sum()
            total_exits = (data['Exit_Signal'] == -1).sum()
            
            if total_entries > 0:
                exit_entry_ratio = total_exits / total_entries
                ratio_analysis['exit_entry_ratio'] = round(exit_entry_ratio, 2)
                
                # 比率評価基準（TODO #4調査結果基準）
                if exit_entry_ratio <= 1.2:
                    ratio_analysis['ratio_assessment'] = 'healthy'
                elif exit_entry_ratio <= 2.0:
                    ratio_analysis['ratio_assessment'] = 'acceptable'
                elif exit_entry_ratio <= 5.0:
                    ratio_analysis['ratio_assessment'] = 'concerning'
                    ratio_analysis['anomaly_detected'] = True
                    ratio_analysis['anomaly_severity'] = 'moderate'
                else:
                    ratio_analysis['ratio_assessment'] = 'critical'
                    ratio_analysis['anomaly_detected'] = True
                    ratio_analysis['anomaly_severity'] = 'severe'
                
                # 詳細分析
                ratio_analysis['details'] = {
                    'total_entries': int(total_entries),
                    'total_exits': int(total_exits),
                    'excess_exits': max(0, int(total_exits - total_entries)),
                    'unclosed_positions': max(0, int(total_entries - total_exits))
                }
        
        return ratio_analysis
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        時系列整合性検証
        TODO(tag:trade_history_validation, rationale:verify chronological trade consistency)
        """
        temporal_results = {
            'chronological_order': True,
            'signal_gaps': [],
            'position_duration_analysis': {},
            'violations': []
        }
        
        if 'Entry_Signal' in data.columns and 'Exit_Signal' in data.columns:
            # エントリー・エグジットの時系列パターン分析
            entry_dates = data[data['Entry_Signal'] == 1].index.tolist()
            exit_dates = data[data['Exit_Signal'] == -1].index.tolist()
            
            # ポジション保有期間分析
            if len(entry_dates) > 0 and len(exit_dates) > 0:
                position_durations = []
                
                for entry_date in entry_dates:
                    # 各エントリー後の最初のエグジットを検索
                    subsequent_exits = [exit_date for exit_date in exit_dates if exit_date > entry_date]
                    
                    if subsequent_exits:
                        next_exit = min(subsequent_exits)
                        # インデックス値（整数）の場合の保有期間計算
                        if isinstance(entry_date, int) and isinstance(next_exit, int):
                            duration = next_exit - entry_date
                        else:
                            # 日付型の場合の保有期間計算
                            duration = (next_exit - entry_date).days
                        position_durations.append(duration)
                
                if position_durations:
                    temporal_results['position_duration_analysis'] = {
                        'average_duration': round(np.mean(position_durations), 1),
                        'median_duration': int(np.median(position_durations)),
                        'min_duration': int(min(position_durations)),
                        'max_duration': int(max(position_durations)),
                        'total_positions_analyzed': len(position_durations)
                    }
                else:
                    # ポジション保有期間データなしの場合のデフォルト値
                    temporal_results['position_duration_analysis'] = {
                        'average_duration': 0.0,
                        'median_duration': 0,
                        'min_duration': 0,
                        'max_duration': 0,
                        'total_positions_analyzed': 0
                    }
                    
                    # 異常に短い保有期間の検出
                    short_positions = [d for d in position_durations if d <= 1]
                    if len(short_positions) > len(position_durations) * 0.3:  # 30%以上が1日以下
                        temporal_results['violations'].append(f"Excessive short positions: {len(short_positions)}/{len(position_durations)}")
        
        return temporal_results
    
    def _validate_position_state_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        ポジション状態整合性検証
        TODO(tag:trade_history_validation, rationale:verify position state transitions)
        """
        position_results = {
            'state_transitions': [],
            'final_state': 'unknown',
            'unclosed_positions': 0,
            'state_violations': []
        }
        
        if 'Active_Strategy' in data.columns:
            # 最終状態分析
            final_active_positions = (data['Active_Strategy'] != '').sum()
            position_results['unclosed_positions'] = int(final_active_positions)
            
            if final_active_positions == 0:
                position_results['final_state'] = 'all_closed'
            else:
                position_results['final_state'] = f'{final_active_positions}_unclosed'
            
            # 状態遷移パターン分析
            state_changes = []
            previous_state = ''
            
            for idx, current_state in enumerate(data['Active_Strategy']):
                if current_state != previous_state:
                    state_changes.append({
                        'date': data.index[idx],
                        'from_state': previous_state,
                        'to_state': current_state,
                        'change_type': 'entry' if current_state != '' else 'exit'
                    })
                    previous_state = current_state
            
            position_results['state_transitions'] = state_changes[-10:]  # 最新10件の変更
        
        return position_results
    
    def _validate_forced_liquidation_logic(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        強制決済ロジック検証（TODO #3連携）
        TODO(tag:trade_history_validation, rationale:verify forced liquidation calculation)
        """
        liquidation_results = {
            'forced_liquidation_count': 0,
            'forced_liquidation_rate': 0.0,
            'calculation_validity': True,
            'rate_assessment': 'unknown'
        }
        
        if 'Exit_Signal' in data.columns and 'Active_Strategy' in data.columns:
            # 期間終了時の強制決済検出
            final_positions = (data['Active_Strategy'] != '').sum()
            total_exits = (data['Exit_Signal'] == -1).sum()
            
            liquidation_results['forced_liquidation_count'] = int(final_positions)
            
            if total_exits > 0:
                forced_rate = (final_positions / total_exits) * 100
                liquidation_results['forced_liquidation_rate'] = round(forced_rate, 2)
                
                # TODO #3修正基準での評価
                if 0 <= forced_rate <= 20:
                    liquidation_results['rate_assessment'] = 'healthy'
                elif forced_rate <= 50:
                    liquidation_results['rate_assessment'] = 'acceptable'
                else:
                    liquidation_results['rate_assessment'] = 'concerning'
                
                # 計算妥当性チェック
                if forced_rate < 0 or forced_rate > 100:
                    liquidation_results['calculation_validity'] = False
        
        return liquidation_results
    
    def _validate_strategy_level_consistency(self, integrated_data: pd.DataFrame, 
                                           strategy_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        戦略別整合性検証
        TODO(tag:trade_history_validation, rationale:verify strategy integration consistency)
        """
        strategy_analysis = {
            'integration_quality': {},
            'signal_preservation_rate': 0.0,
            'strategy_anomalies': []
        }
        
        total_strategy_entries = 0
        total_strategy_exits = 0
        
        for strategy_name, strategy_data in strategy_results.items():
            if 'Entry_Signal' in strategy_data.columns and 'Exit_Signal' in strategy_data.columns:
                strategy_entries = (strategy_data['Entry_Signal'] == 1).sum()
                strategy_exits = (strategy_data['Exit_Signal'] == -1).sum()
                
                total_strategy_entries += strategy_entries
                total_strategy_exits += strategy_exits
                
                # 個別戦略異常検出
                if strategy_entries > 0:
                    strategy_exit_ratio = strategy_exits / strategy_entries
                    if strategy_exit_ratio > 5.0:  # TODO #4基準
                        strategy_analysis['strategy_anomalies'].append({
                            'strategy': strategy_name,
                            'anomaly_type': 'excessive_exits',
                            'exit_entry_ratio': round(strategy_exit_ratio, 2),
                            'entries': int(strategy_entries),
                            'exits': int(strategy_exits)
                        })
        
        # 統合品質評価
        integrated_entries = (integrated_data['Entry_Signal'] == 1).sum()
        integrated_exits = (integrated_data['Exit_Signal'] == -1).sum()
        
        if total_strategy_entries > 0:
            entry_preservation_rate = (integrated_entries / total_strategy_entries) * 100
            strategy_analysis['signal_preservation_rate'] = round(entry_preservation_rate, 1)
        
        strategy_analysis['integration_quality'] = {
            'total_strategy_entries': int(total_strategy_entries),
            'total_strategy_exits': int(total_strategy_exits),
            'integrated_entries': int(integrated_entries),
            'integrated_exits': int(integrated_exits),
            'entry_integration_rate': round((integrated_entries / total_strategy_entries * 100), 1) if total_strategy_entries > 0 else 0,
            'exit_integration_rate': round((integrated_exits / total_strategy_exits * 100), 1) if total_strategy_exits > 0 else 0
        }
        
        return strategy_analysis
    
    def _compile_validation_results(self, *validation_components) -> Dict[str, Any]:
        """
        検証結果統合
        TODO(tag:trade_history_validation, rationale:comprehensive validation summary)
        """
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # 各検証コンポーネントの結果を統合
        component_names = ['basic_integrity', 'ratio_analysis', 'temporal_consistency', 
                          'position_integrity', 'forced_liquidation_analysis', 'strategy_consistency']
        
        for i, component in enumerate(validation_components):
            if component is not None:
                summary[component_names[i]] = component
        
        # 全体状況評価
        critical_count = 0
        warning_count = 0
        
        # 基本整合性チェック
        if summary.get('basic_integrity', {}).get('violations'):
            critical_count += len(summary['basic_integrity']['violations'])
        
        # 比率異常チェック
        if summary.get('ratio_analysis', {}).get('anomaly_severity') == 'severe':
            critical_count += 1
        elif summary.get('ratio_analysis', {}).get('anomaly_detected'):
            warning_count += 1
        
        # 強制決済チェック
        if not summary.get('forced_liquidation_analysis', {}).get('calculation_validity', True):
            critical_count += 1
        
        # 全体状況判定
        if critical_count == 0 and warning_count == 0:
            summary['overall_status'] = 'healthy'
        elif critical_count == 0:
            summary['overall_status'] = 'acceptable_with_warnings'
        else:
            summary['overall_status'] = 'critical_issues_detected'
        
        return summary
    
    def _print_validation_report(self, validation_summary: Dict[str, Any]):
        """
        検証レポート出力
        TODO(tag:trade_history_validation, rationale:detailed validation reporting)
        """
        print("\n" + "="*70)
        print("[REPORT] TODO #6: 取引履歴整合性検証レポート")
        print("="*70)
        
        # 全体状況
        status = validation_summary['overall_status']
        status_icon = "[OK]" if status == 'healthy' else "[WARNING]" if 'warning' in status else "[ERROR]"
        print(f"\n{status_icon} 全体整合性状況: {status}")
        
        # 基本統計
        if 'basic_integrity' in validation_summary:
            basic = validation_summary['basic_integrity']
            if 'signal_counts' in basic:
                counts = basic['signal_counts']
                print(f"\n[STATS] 基本統計:")
                print(f"  エントリー数: {counts['total_entries']}")
                print(f"  エグジット数: {counts['total_exits']}")
                print(f"  シグナルバランス: {counts['signal_balance']}")
        
        # エントリー・エグジット比率分析
        if 'ratio_analysis' in validation_summary:
            ratio = validation_summary['ratio_analysis']
            ratio_icon = "[OK]" if ratio['ratio_assessment'] == 'healthy' else "[WARNING]" if ratio['ratio_assessment'] in ['acceptable', 'concerning'] else "[ERROR]"
            print(f"\n{ratio_icon} エグジット/エントリー比率分析:")
            print(f"  比率: {ratio['exit_entry_ratio']}")
            print(f"  評価: {ratio['ratio_assessment']}")
            
            if ratio['anomaly_detected']:
                print(f"  [WARNING] 異常検出: {ratio['anomaly_severity']} レベル")
        
        # 時系列分析
        if 'temporal_consistency' in validation_summary:
            temporal = validation_summary['temporal_consistency']
            if 'position_duration_analysis' in temporal:
                duration = temporal['position_duration_analysis']
                print(f"\n[CALENDAR] ポジション保有期間分析:")
                print(f"  平均保有期間: {duration['average_duration']}日")
                print(f"  中央値: {duration['median_duration']}日")
                print(f"  範囲: {duration['min_duration']}-{duration['max_duration']}日")
        
        # 強制決済分析
        if 'forced_liquidation_analysis' in validation_summary:
            liquidation = validation_summary['forced_liquidation_analysis']
            liquidation_icon = "[OK]" if liquidation['rate_assessment'] == 'healthy' else "[WARNING]"
            print(f"\n{liquidation_icon} 強制決済分析:")
            print(f"  強制決済数: {liquidation['forced_liquidation_count']}")
            print(f"  強制決済率: {liquidation['forced_liquidation_rate']}%")
            print(f"  評価: {liquidation['rate_assessment']}")
        
        # 戦略統合品質
        if 'strategy_consistency' in validation_summary and validation_summary['strategy_consistency']:
            strategy = validation_summary['strategy_consistency']
            if 'integration_quality' in strategy:
                quality = strategy['integration_quality']
                print(f"\n[TOOLS] 戦略統合品質:")
                print(f"  エントリー統合率: {quality['entry_integration_rate']}%")
                print(f"  エグジット統合率: {quality['exit_integration_rate']}%")
                
                if strategy['strategy_anomalies']:
                    print(f"  [WARNING] 戦略異常検出: {len(strategy['strategy_anomalies'])}件")
        
        print("\n" + "="*70)
    
    def _detect_backtest_principle_violations(self, validation_summary: Dict[str, Any]):
        """
        バックテスト基本理念違反検出
        TODO(tag:trade_history_validation, rationale:ensure backtest principle compliance)
        """
        violations = []
        
        # シグナル生成チェック
        if 'basic_integrity' in validation_summary:
            basic = validation_summary['basic_integrity']
            if 'signal_counts' in basic:
                counts = basic['signal_counts']
                if counts['total_entries'] == 0:
                    violations.append("Zero entry signals - violates backtest principle")
                if counts['total_exits'] == 0:
                    violations.append("Zero exit signals - potential integration failure")
        
        # 取引履歴整合性チェック（重大な問題のみCRITICAL判定）
        critical_violations_count = 0
        for component in validation_summary.values():
            if isinstance(component, dict) and 'violations' in component:
                critical_violations_count += len(component['violations'])
        
        # 実際に深刻な問題がある場合のみCRITICAL扱い
        if critical_violations_count > 3:  # 複数の深刻な問題がある場合
            violations.append("Critical trade history inconsistencies detected")
        
        # 異常比率チェック
        if 'ratio_analysis' in validation_summary:
            ratio = validation_summary['ratio_analysis']
            if ratio['anomaly_severity'] == 'severe':
                violations.append(f"Severe entry-exit ratio anomaly: {ratio['exit_entry_ratio']}")
        
        if violations:
            error_msg = f"Backtest principle violations detected: {'; '.join(violations)}"
            self.logger.error(error_msg)
            raise ValueError(f"{error_msg} TODO(tag:backtest_execution, rationale:fix principle violations)")
        
        return True


# TODO(tag:trade_history_validation, rationale:integrate validation into main workflow)
def integrate_validation_into_main():
    """
    main.pyへの統合コード
    バックテスト基本理念遵守: 実行後自動検証
    """
    validation_code = '''
# main.py内のapply_strategies_with_optimized_params関数の最後に追加

    # TODO(tag:trade_history_validation, rationale:automatic post-backtest validation)
    # 取引履歴整合性検証の実行
    try:
        from analysis.trade_history_validator import TradeHistoryValidator
        
        validator = TradeHistoryValidator(enable_detailed_logging=True)
        validation_results = validator.validate_integrated_backtest_results(
            integrated_data, 
            strategy_results=strategy_results if 'strategy_results' in locals() else None
        )
        
        # 検証結果をログに記録
        logger.info(f"Trade history validation completed: {validation_results['overall_status']}")
        
        # 重大な問題がある場合は警告
        if validation_results['overall_status'] == 'critical_issues_detected':
            logger.warning("Critical trade history issues detected - review recommended")
        
    except Exception as e:
        logger.error(f"Trade history validation failed: {e}")
        # バックテスト基本理念遵守: 検証失敗でも実行は継続
        pass
    
    return integrated_data
    '''
    return validation_code