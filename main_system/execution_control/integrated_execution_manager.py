"""
統合実行管理システム
Phase 3.1: 戦略選択結果をもとに実行制御・リスク管理を統合
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# プロジェクトパス追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

# 既存モジュール
from main_system.execution_control.strategy_execution_manager import StrategyExecutionManager
from main_system.risk_management.drawdown_controller import DrawdownController

# Phase 2完成モジュール
from main_system.strategy_selection.dynamic_strategy_selector import DynamicStrategySelector
from main_system.market_analysis.market_analyzer import MarketAnalyzer


class IntegratedExecutionManager:
    """統合実行管理クラス - 戦略選択→実行制御→リスク管理を統合"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            config: 実行設定
        """
        self.config = config or {}
        self.logger = setup_logger(
            "IntegratedExecutionManager",
            log_file="logs/integrated_execution.log"
        )
        
        # コンポーネント初期化
        try:
            # 実行管理コンポーネント
            # Phase 4.2-14: デフォルト初期資金を1,000,000円に変更（ハードコード修正）
            execution_config = self.config.get('execution', {
                'execution_mode': 'simple',
                'broker': {
                    'initial_cash': 1000000,  # Phase 4.2-14: 100,000円 → 1,000,000円
                    'commission_per_trade': 1.0,
                    'slippage_bps': 5
                }
            })
            self.execution_manager = StrategyExecutionManager(execution_config)
            
            # バッチテスト実行器（テスト関数は後で設定）
            self.batch_executor = None
            
            # リスク管理コンポーネント
            risk_config = self.config.get('risk_management', {})
            self.risk_controller = DrawdownController(risk_config)
            
            # Phase 2統合コンポーネント
            self.market_analyzer = MarketAnalyzer()
            self.strategy_selector = DynamicStrategySelector()
            
            self.logger.info("IntegratedExecutionManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
        
        # 実行履歴
        self.execution_history: List[Dict[str, Any]] = []
        self.current_portfolio_value = self.config.get('initial_portfolio_value', 100000.0)
        
    def execute_dynamic_strategies(
        self,
        stock_data: pd.DataFrame,
        ticker: str,
        selected_strategies: Optional[List[str]] = None,
        strategy_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        動的戦略選択結果をもとに戦略実行
        
        Args:
            stock_data: 株価データ
            ticker: ティッカーシンボル
            selected_strategies: 選択済み戦略リスト（Noneの場合は自動選択）
            strategy_weights: 戦略重み（Noneの場合は自動計算）
        
        Returns:
            実行結果辞書
        """
        try:
            self.logger.info(f"Starting dynamic strategy execution for {ticker}")
            
            # Step 1: 戦略選択（未指定の場合）
            if selected_strategies is None or strategy_weights is None:
                selection_result = self._select_strategies_dynamically(stock_data, ticker)
                
                if selection_result['status'] != 'SUCCESS':
                    self.logger.warning(f"Strategy selection failed: {selection_result.get('error')}")
                    return {
                        'status': 'FAILED',
                        'error': 'Strategy selection failed',
                        'details': selection_result
                    }
                
                selected_strategies = selection_result['selected_strategies']
                strategy_weights = selection_result['strategy_weights']
            
            # Null チェック（型チェッカー対応）
            if selected_strategies is None or strategy_weights is None:
                return {
                    'status': 'FAILED',
                    'error': 'Strategy selection returned None',
                    'details': {}
                }
            
            self.logger.info(f"Executing {len(selected_strategies)} strategies: {selected_strategies}")

            
            # Step 2: リスクチェック
            risk_check_result = self._check_execution_risk(stock_data, ticker)
            
            if not risk_check_result['can_execute']:
                self.logger.warning(f"Execution blocked by risk check: {risk_check_result['reason']}")
                return {
                    'status': 'RISK_BLOCKED',
                    'error': risk_check_result['reason'],
                    'risk_details': risk_check_result
                }
            
            # Step 3: 各戦略を実行
            execution_results = []
            for strategy_name in selected_strategies:
                weight = strategy_weights.get(strategy_name, 0.0)
                
                if weight <= 0.0:
                    self.logger.warning(f"Skipping {strategy_name} (weight={weight})")
                    continue
                
                self.logger.info(f"Executing {strategy_name} with weight {weight:.3f}")
                
                # 個別戦略実行
                result = self._execute_single_strategy(
                    strategy_name=strategy_name,
                    stock_data=stock_data,
                    ticker=ticker,
                    weight=weight
                )
                
                execution_results.append(result)
            
            # Step 4: 結果統合
            integrated_result = self._integrate_execution_results(
                execution_results,
                strategy_weights
            )
            
            # Step 5: ポートフォリオ価値更新（リスク管理用）
            if integrated_result['status'] == 'SUCCESS':
                portfolio_value = integrated_result.get('total_portfolio_value', self.current_portfolio_value)
                self._update_risk_tracking(portfolio_value, execution_results)
            
            # 履歴記録
            self.execution_history.append({
                'timestamp': datetime.now(),
                'ticker': ticker,
                'selected_strategies': selected_strategies,
                'strategy_weights': strategy_weights,
                'result': integrated_result
            })
            
            self.logger.info(f"Dynamic strategy execution completed: {integrated_result['status']}")
            
            return integrated_result
            
        except Exception as e:
            self.logger.error(f"Error in execute_dynamic_strategies: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'execution_results': []
            }
    
    def _select_strategies_dynamically(
        self,
        stock_data: pd.DataFrame,
        ticker: str
    ) -> Dict[str, Any]:
        """動的戦略選択"""
        try:
            # 市場分析
            market_analysis = self.market_analyzer.comprehensive_market_analysis(
                stock_data=stock_data,
                ticker=ticker
            )
            
            # 戦略選択（正しいシグネチャ）
            selection_result = self.strategy_selector.select_optimal_strategies(
                market_analysis=market_analysis,
                stock_data=stock_data,
                ticker=ticker
            )
            
            return selection_result
            
        except Exception as e:
            self.logger.error(f"Strategy selection error: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'selected_strategies': [],
                'strategy_weights': {}
            }
    
    def _check_execution_risk(
        self,
        stock_data: pd.DataFrame,
        ticker: str
    ) -> Dict[str, Any]:
        """
        実行前リスクチェック
        
        Returns:
            {'can_execute': bool, 'reason': str, 'drawdown_status': dict}
        """
        try:
            # 現在のドローダウン状況を確認
            performance_summary = self.risk_controller.get_performance_summary()
            
            current_dd_pct = performance_summary.get('current_drawdown_pct', 0.0)
            
            # 緊急停止閾値チェック（15%以上のドローダウン）
            emergency_threshold = 0.15
            if current_dd_pct >= emergency_threshold:
                return {
                    'can_execute': False,
                    'reason': f'Emergency drawdown level: {current_dd_pct:.2%}',
                    'drawdown_status': performance_summary
                }
            
            # 警告レベルチェック（10%以上のドローダウン）
            warning_threshold = 0.10
            if current_dd_pct >= warning_threshold:
                self.logger.warning(f"Drawdown warning level: {current_dd_pct:.2%}")
            
            return {
                'can_execute': True,
                'reason': 'Risk check passed',
                'drawdown_status': performance_summary
            }
            
        except Exception as e:
            self.logger.error(f"Risk check error: {e}")
            # エラー時は安全のため実行を許可（リスク管理モジュールの問題でビジネス停止しない）
            return {
                'can_execute': True,
                'reason': f'Risk check error (allowing execution): {e}',
                'drawdown_status': {}
            }
    
    def _execute_single_strategy(
        self,
        strategy_name: str,
        stock_data: pd.DataFrame,
        ticker: str,
        weight: float
    ) -> Dict[str, Any]:
        """
        単一戦略実行（Phase 4.2: データ渡し対応）
        
        Args:
            strategy_name: 戦略名
            stock_data: 株価データ
            ticker: ティッカーシンボル
            weight: 戦略重み
        
        Returns:
            実行結果辞書
        """
        try:
            self.logger.info(f"Executing single strategy: {strategy_name}")
            
            # インデックスデータ取得（簡易実装：stock_dataを流用）
            # TODO: Phase 5で適切なインデックスデータ取得実装
            index_data = stock_data  # フォールバック
            
            # StrategyExecutionManagerを使用して戦略実行（Phase 4.2: データを渡す）
            result = self.execution_manager.execute_strategy(
                strategy_name=strategy_name,
                symbols=[ticker],
                stock_data=stock_data,
                index_data=index_data
            )
            
            # 重み情報を追加
            result['weight'] = weight
            result['strategy_name'] = strategy_name
            
            return result
            
        except Exception as e:
            self.logger.error(f"Single strategy execution error: {e}")
            return {
                'status': 'FAILED',
                'strategy_name': strategy_name,
                'weight': weight,
                'error': str(e)
            }
    
    def _integrate_execution_results(
        self,
        execution_results: List[Dict[str, Any]],
        strategy_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        実行結果統合
        
        Args:
            execution_results: 各戦略の実行結果リスト
            strategy_weights: 戦略重み
        
        Returns:
            統合結果辞書
        """
        try:
            if not execution_results:
                return {
                    'status': 'FAILED',
                    'error': 'No execution results',
                    'execution_results': []
                }
            
            # 成功した戦略をカウント
            successful_strategies = [
                r for r in execution_results
                if r.get('status') == 'success'
            ]
            
            failed_strategies = [
                r for r in execution_results
                if r.get('status') != 'success'
            ]
            
            # 重み付き集約
            weighted_performance = 0.0
            
            for result in successful_strategies:
                strategy_name = result.get('strategy_name', 'Unknown')
                weight = strategy_weights.get(strategy_name, 0.0)
                
                # パフォーマンス指標（暫定）
                performance = result.get('performance_metric', 0.0)
                weighted_performance += performance * weight
            
            # 統合ステータス
            if len(successful_strategies) == 0:
                status = 'ALL_FAILED'
            elif len(failed_strategies) == 0:
                status = 'SUCCESS'
            else:
                status = 'PARTIAL_SUCCESS'
            
            return {
                'status': status,
                'total_strategies': len(execution_results),
                'successful_strategies': len(successful_strategies),
                'failed_strategies': len(failed_strategies),
                'weighted_performance': weighted_performance,
                'total_portfolio_value': self.current_portfolio_value,
                'execution_results': execution_results,
                'strategy_weights': strategy_weights
            }
            
        except Exception as e:
            self.logger.error(f"Result integration error: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'execution_results': execution_results
            }
    
    def _update_risk_tracking(
        self,
        portfolio_value: float,
        execution_results: List[Dict[str, Any]]
    ) -> None:
        """リスク追跡情報更新"""
        try:
            # 戦略別価値計算（暫定）
            strategy_values = {}
            for result in execution_results:
                strategy_name = result.get('strategy_name', 'Unknown')
                weight = result.get('weight', 0.0)
                strategy_values[strategy_name] = portfolio_value * weight
            
            # DrawdownControllerに更新を通知
            self.risk_controller.update_portfolio_value(
                portfolio_value=portfolio_value,
                strategy_values=strategy_values
            )
            
            self.current_portfolio_value = portfolio_value
            
        except Exception as e:
            self.logger.error(f"Risk tracking update error: {e}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """実行サマリー取得"""
        return {
            'total_executions': len(self.execution_history),
            'current_portfolio_value': self.current_portfolio_value,
            'risk_summary': self.risk_controller.get_performance_summary(),
            'latest_execution': self.execution_history[-1] if self.execution_history else None
        }


def demo_integrated_execution():
    """IntegratedExecutionManager デモ実行"""
    logger = setup_logger("IntegratedExecutionDemo", log_file="logs/integrated_execution_demo.log")
    
    try:
        # サンプルデータ生成
        import numpy as np
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
            'High': 102 + np.cumsum(np.random.randn(len(dates)) * 2),
            'Low': 98 + np.cumsum(np.random.randn(len(dates)) * 2),
            'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        sample_data.set_index('Date', inplace=True)
        
        # IntegratedExecutionManager作成
        config = {
            'initial_portfolio_value': 100000,
            'execution': {
                'execution_mode': 'simple'
            },
            'risk_management': {}
        }
        
        manager = IntegratedExecutionManager(config)
        
        # 動的戦略実行
        result = manager.execute_dynamic_strategies(
            stock_data=sample_data,
            ticker='AAPL'
        )
        
        logger.info(f"Execution result: {result['status']}")
        logger.info(f"Successful strategies: {result.get('successful_strategies', 0)}")
        logger.info(f"Failed strategies: {result.get('failed_strategies', 0)}")
        
        # サマリー表示
        summary = manager.get_execution_summary()
        logger.info(f"Execution summary: {summary}")
        
        print("\n=== IntegratedExecutionManager Demo Completed ===")
        print(f"Status: {result['status']}")
        print(f"Total Executions: {summary['total_executions']}")
        
        return manager, result
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        raise


if __name__ == '__main__':
    demo_integrated_execution()
