"""
データ品質検証システム
Phase 2.3 Task 2.3.3: 品質保証システム

Purpose:
  - Excel出力データの正確性検証
  - パフォーマンス計算の妥当性チェック
  - データ整合性確認

Author: GitHub Copilot Agent
Created: 2025-09-04
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """データ品質検証クラス"""
    
    def __init__(self, tolerance: float = 0.01):
        """
        初期化
        
        Args:
            tolerance: 数値比較の許容誤差（デフォルト1%）
        """
        self.tolerance = tolerance
        self.validation_results: List[Dict[str, Any]] = []
    
    def validate_backtest_data(self, stock_data: pd.DataFrame, 
                              trades: List[Dict[str, Any]], 
                              performance: Dict[str, float]) -> Dict[str, Any]:
        """
        バックテストデータの総合検証
        
        Args:
            stock_data: 元のDataFrame
            trades: 取引リスト
            performance: パフォーマンス指標
            
        Returns:
            Dict: 検証結果レポート
        """
        validation_report = {
            'timestamp': datetime.now(),
            'data_checks': {},
            'trade_checks': {},
            'performance_checks': {},
            'issues': [],
            'overall_quality': 'unknown'
        }
        
        # 1. データ基本チェック
        validation_report['data_checks'] = self._validate_data_basics(stock_data)
        
        # 2. 取引データチェック
        validation_report['trade_checks'] = self._validate_trades(trades, stock_data)
        
        # 3. パフォーマンス計算チェック
        validation_report['performance_checks'] = self._validate_performance(
            trades, performance, stock_data
        )
        
        # 4. 整合性チェック
        consistency_checks = self._validate_consistency(
            stock_data, trades, performance
        )
        validation_report['consistency_checks'] = consistency_checks
        
        # 5. 総合評価
        validation_report['overall_quality'] = self._calculate_overall_quality(
            validation_report
        )
        
        # 問題リスト生成
        validation_report['issues'] = self._collect_issues(validation_report)
        
        logger.info(f"データ品質検証完了: {validation_report['overall_quality']}")
        return validation_report
    
    def _validate_data_basics(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """データ基本検証"""
        checks = {
            'has_data': not stock_data.empty,
            'has_required_columns': True,
            'date_continuity': True,
            'price_validity': True,
            'signal_validity': True
        }
        
        if stock_data.empty:
            return checks
        
        # 必要列チェック
        required_cols = ['Close', 'Entry_Signal', 'Exit_Signal']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        checks['has_required_columns'] = len(missing_cols) == 0
        checks['missing_columns'] = missing_cols
        
        # 価格有効性チェック
        if 'Close' in stock_data.columns:
            close_prices = stock_data['Close']
            checks['price_validity'] = (close_prices > 0).all()
            checks['price_nulls'] = close_prices.isnull().sum()
        
        # シグナル有効性チェック
        if 'Entry_Signal' in stock_data.columns:
            entry_signals = stock_data['Entry_Signal']
            checks['signal_validity'] = entry_signals.isin([0, 1]).all()
        
        # データ期間情報
        checks['period_start'] = stock_data.index[0]
        checks['period_end'] = stock_data.index[-1]
        checks['total_days'] = len(stock_data)
        
        return checks
    
    def _validate_trades(self, trades: List[Dict[str, Any]], 
                        stock_data: pd.DataFrame) -> Dict[str, Any]:
        """取引データ検証"""
        checks = {
            'trade_count': len(trades),
            'all_trades_complete': True,
            'price_consistency': True,
            'date_validity': True,
            'pnl_calculation': True
        }
        
        if not trades:
            return checks
        
        for i, trade in enumerate(trades):
            # 必要フィールドチェック
            required_fields = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl']
            missing_fields = [field for field in required_fields if field not in trade]
            if missing_fields:
                checks['all_trades_complete'] = False
                checks[f'trade_{i}_missing'] = missing_fields
            
            # 価格妥当性チェック
            if 'entry_price' in trade and 'exit_price' in trade:
                if trade['entry_price'] <= 0 or trade['exit_price'] <= 0:
                    checks['price_consistency'] = False
            
            # 日付妥当性チェック
            if 'entry_date' in trade and 'exit_date' in trade:
                if trade['entry_date'] >= trade['exit_date']:
                    checks['date_validity'] = False
            
            # 損益計算チェック
            if all(field in trade for field in ['entry_price', 'exit_price', 'shares', 'pnl']):
                expected_pnl = (trade['exit_price'] - trade['entry_price']) * trade['shares']
                if abs(trade['pnl'] - expected_pnl) > abs(expected_pnl * self.tolerance):
                    checks['pnl_calculation'] = False
                    checks[f'trade_{i}_pnl_error'] = {
                        'expected': expected_pnl,
                        'actual': trade['pnl'],
                        'difference': trade['pnl'] - expected_pnl
                    }
        
        return checks
    
    def _validate_performance(self, trades: List[Dict[str, Any]], 
                            performance: Dict[str, float], 
                            stock_data: pd.DataFrame) -> Dict[str, Any]:
        """パフォーマンス計算検証"""
        checks = {
            'pnl_consistency': True,
            'return_calculation': True,
            'trade_count_match': True,
            'portfolio_value_logic': True
        }
        
        if not trades or not performance:
            return checks
        
        # 総損益整合性チェック
        calculated_total_pnl = sum(trade.get('pnl', 0) for trade in trades)
        reported_total_pnl = performance.get('total_pnl', 0)
        
        if abs(calculated_total_pnl - reported_total_pnl) > abs(calculated_total_pnl * self.tolerance):
            checks['pnl_consistency'] = False
            checks['pnl_difference'] = calculated_total_pnl - reported_total_pnl
        
        # 取引数整合性チェック
        if len(trades) != performance.get('num_trades', 0):
            checks['trade_count_match'] = False
            checks['trade_count_difference'] = len(trades) - performance.get('num_trades', 0)
        
        # ポートフォリオ価値ロジックチェック
        initial_capital = 1000000.0  # デフォルト値
        expected_final_value = initial_capital + calculated_total_pnl
        reported_final_value = performance.get('final_portfolio_value', 0)
        
        if abs(expected_final_value - reported_final_value) > abs(expected_final_value * self.tolerance):
            checks['portfolio_value_logic'] = False
            checks['value_difference'] = expected_final_value - reported_final_value
        
        # リターン計算チェック
        if reported_final_value > 0 and initial_capital > 0:
            expected_return = (reported_final_value - initial_capital) / initial_capital
            reported_return = performance.get('total_return', 0)
            
            if abs(expected_return - reported_return) > self.tolerance:
                checks['return_calculation'] = False
                checks['return_difference'] = expected_return - reported_return
        
        return checks
    
    def _validate_consistency(self, stock_data: pd.DataFrame, 
                            trades: List[Dict[str, Any]], 
                            performance: Dict[str, float]) -> Dict[str, Any]:
        """データ間整合性検証"""
        checks = {
            'signal_trade_match': True,
            'date_range_consistency': True,
            'zero_value_check': True
        }
        
        # ゼロ値チェック（重要な問題）
        critical_zero_values = []
        if performance.get('final_portfolio_value', 1) == 0:
            critical_zero_values.append('final_portfolio_value')
        if performance.get('total_pnl', 0) == 0 and trades:
            critical_zero_values.append('total_pnl')
        
        checks['zero_value_check'] = len(critical_zero_values) == 0
        if critical_zero_values:
            checks['critical_zero_values'] = critical_zero_values
        
        # シグナルと取引の整合性
        if not stock_data.empty and 'Entry_Signal' in stock_data.columns:
            signal_entries = (stock_data['Entry_Signal'] == 1).sum()
            trade_count = len(trades)
            
            # 取引数がシグナル数と大きく異なる場合は問題
            if abs(signal_entries - trade_count) > max(signal_entries, trade_count) * 0.2:
                checks['signal_trade_match'] = False
                checks['signal_count'] = signal_entries
                checks['trade_count'] = trade_count
        
        return checks
    
    def _calculate_overall_quality(self, validation_report: Dict[str, Any]) -> str:
        """総合品質評価"""
        critical_issues = 0
        minor_issues = 0
        
        # データ基本チェック
        data_checks = validation_report['data_checks']
        if not data_checks.get('has_data', True):
            critical_issues += 1
        if not data_checks.get('has_required_columns', True):
            critical_issues += 1
        if not data_checks.get('price_validity', True):
            critical_issues += 1
        
        # 取引データチェック
        trade_checks = validation_report['trade_checks']
        if not trade_checks.get('all_trades_complete', True):
            critical_issues += 1
        if not trade_checks.get('pnl_calculation', True):
            critical_issues += 1
        
        # パフォーマンスチェック
        perf_checks = validation_report['performance_checks']
        if not perf_checks.get('pnl_consistency', True):
            critical_issues += 1
        if not perf_checks.get('portfolio_value_logic', True):
            critical_issues += 1
        
        # 整合性チェック
        consistency_checks = validation_report['consistency_checks']
        if not consistency_checks.get('zero_value_check', True):
            critical_issues += 1
        
        # 品質判定
        if critical_issues == 0:
            return 'excellent'
        elif critical_issues <= 2:
            return 'good'
        elif critical_issues <= 4:
            return 'fair'
        else:
            return 'poor'
    
    def _collect_issues(self, validation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """問題リスト収集"""
        issues = []
        
        # 重要なゼロ値問題
        if 'consistency_checks' in validation_report:
            if not validation_report['consistency_checks'].get('zero_value_check', True):
                zero_values = validation_report['consistency_checks'].get('critical_zero_values', [])
                issues.append({
                    'type': 'critical',
                    'category': 'zero_values',
                    'message': f'重要な指標がゼロ値: {", ".join(zero_values)}',
                    'recommendation': 'データ抽出エンハンサーでの計算ロジック確認が必要'
                })
        
        # データ不整合問題
        if 'performance_checks' in validation_report:
            if not validation_report['performance_checks'].get('pnl_consistency', True):
                issues.append({
                    'type': 'critical',
                    'category': 'calculation_error',
                    'message': '総損益計算に不整合があります',
                    'recommendation': '取引データとパフォーマンス計算の再確認が必要'
                })
        
        return issues

# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def validate_excel_output_quality(stock_data: pd.DataFrame,
                                 normalized_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Excel出力用データ品質検証の便利関数
    
    Args:
        stock_data: 元のDataFrame
        normalized_data: 正規化済みデータ
        
    Returns:
        Dict: 検証結果
    """
    validator = DataQualityValidator()
    
    trades = normalized_data.get('trades', [])
    performance = normalized_data.get('summary', {})
    
    return validator.validate_backtest_data(stock_data, trades, performance)
