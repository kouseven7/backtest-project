"""
Module: Rule Engine Integrated Interface
File: rule_engine_integrated_interface.py
Description: 
  3-1-3「選択ルールの抽象化（差し替え可能に）」統合インターフェース
  TrendStrategyIntegrationInterfaceを拡張し、ルールエンジン機能を統合

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.trend_strategy_integration_interface
  - config.enhanced_strategy_selector
  - config.strategy_selection_rule_engine
  - config.rule_configuration_manager
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存モジュールのインポート
try:
    from config.trend_strategy_integration_interface import (
        TrendStrategyIntegrationInterface, IntegratedDecisionResult, BatchProcessingResult,
        ProcessingMode, IntegrationStatus, TrendAnalysisResult, StrategyScoreBundle
    )
    from config.enhanced_strategy_selector import (
        EnhancedStrategySelector, EnhancedSelectionCriteria, SelectionStrategy
    )
    from config.strategy_selection_rule_engine import (
        StrategySelectionRuleEngine, RuleContext, RuleExecutionResult
    )
    from config.rule_configuration_manager import RuleConfigurationManager
except ImportError as e:
    logging.getLogger(__name__).warning(f"Import error: {e}")

# ロガーの設定
logger = logging.getLogger(__name__)

class RuleEngineMode(Enum):
    """ルールエンジンモード"""
    DISABLED = "disabled"          # ルールエンジン無効
    ENABLED = "enabled"            # ルールエンジン有効
    FALLBACK = "fallback"          # フォールバック時のみ
    HYBRID = "hybrid"              # ハイブリッド統合

@dataclass
class RuleEngineMetrics:
    """ルールエンジン指標"""
    total_rule_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_ms: float = 0.0
    rule_performance: Dict[str, float] = field(default_factory=dict)
    last_execution_time: Optional[datetime] = None

class RuleEngineIntegratedInterface(TrendStrategyIntegrationInterface):
    """
    ルールエンジン統合インターフェース
    
    TrendStrategyIntegrationInterfaceを拡張し、
    ルールエンジン機能を完全統合
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 base_dir: Optional[str] = None,
                 rule_engine_mode: RuleEngineMode = RuleEngineMode.ENABLED):
        """統合インターフェースの初期化"""
        
        # 親クラスの初期化
        super().__init__(config_file, base_dir)
        
        # ルールエンジン関連の初期化
        self.rule_engine_mode = rule_engine_mode
        self.enhanced_selector = EnhancedStrategySelector(config_file, base_dir)
        self.rule_config_manager = RuleConfigurationManager(
            Path(base_dir) / "rule_engine" if base_dir else None
        )
        
        # ルールエンジン指標
        self.rule_metrics = RuleEngineMetrics()
        
        # 統合設定
        self.integration_config = self._load_integration_config()
        
        logger.info(f"RuleEngineIntegratedInterface initialized with mode: {rule_engine_mode.value}")
    
    def _load_integration_config(self) -> Dict[str, Any]:
        """統合設定の読み込み"""
        default_config = {
            'rule_engine_priority': True,
            'fallback_strategy': 'legacy',
            'performance_tracking': True,
            'auto_rule_optimization': False,
            'cache_rule_results': True,
            'parallel_rule_execution': False,
            'rule_timeout_ms': 5000,
            'max_concurrent_rules': 3
        }
        
        config_file = self.base_dir / "integration_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config.get('rule_engine', {}))
            except Exception as e:
                logger.warning(f"Failed to load integration config: {e}")
        
        return default_config
    
    def analyze_integrated_with_rules(self, 
                                    ticker: str,
                                    data: pd.DataFrame,
                                    processing_mode: ProcessingMode = ProcessingMode.REALTIME,
                                    rule_preference: Optional[str] = None,
                                    custom_criteria: Optional[EnhancedSelectionCriteria] = None) -> IntegratedDecisionResult:
        """
        ルールエンジンを統合した分析
        
        Args:
            ticker: ティッカーシンボル
            data: 価格データ
            processing_mode: 処理モード
            rule_preference: 優先ルール名
            custom_criteria: カスタム選択基準
            
        Returns:
            IntegratedDecisionResult: 統合分析結果
        """
        start_time = datetime.now()
        
        try:
            # 基本分析の実行（親クラスのメソッド）
            base_result = self.analyze_integrated(ticker, data, processing_mode)
            
            # ルールエンジンモードの確認
            if self.rule_engine_mode == RuleEngineMode.DISABLED:
                return base_result
            
            # 拡張戦略選択の実行
            enhanced_result = self._execute_enhanced_selection(
                ticker, base_result, rule_preference, custom_criteria
            )
            
            # 結果の統合
            integrated_result = self._integrate_results(base_result, enhanced_result)
            
            # メトリクスの更新
            self._update_rule_metrics(start_time, True)
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Rule-integrated analysis failed: {e}")
            
            # フォールバック処理
            if self.rule_engine_mode in [RuleEngineMode.FALLBACK, RuleEngineMode.HYBRID]:
                self._update_rule_metrics(start_time, False)
                return self.analyze_integrated(ticker, data, processing_mode)
            else:
                raise
    
    def _execute_enhanced_selection(self, 
                                  ticker: str,
                                  base_result: IntegratedDecisionResult,
                                  rule_preference: Optional[str],
                                  custom_criteria: Optional[EnhancedSelectionCriteria]) -> Any:
        """拡張戦略選択の実行"""
        
        # 戦略スコアの抽出
        strategy_scores = {}
        for strategy_name, score_bundle in base_result.strategy_scores.items():
            strategy_scores[strategy_name] = score_bundle.final_score
        
        # 選択基準の設定
        if custom_criteria is None:
            custom_criteria = EnhancedSelectionCriteria(
                selection_strategy=SelectionStrategy.AUTO,
                preferred_rule=rule_preference,
                enable_risk_adjustment=True,
                enable_caching=self.integration_config.get('cache_rule_results', True)
            )
        
        # リスク指標の準備
        risk_metrics = {}
        for strategy_name, score_bundle in base_result.strategy_scores.items():
            risk_metrics[strategy_name] = {
                'volatility': score_bundle.score_components.get('volatility', 0.2),
                'max_drawdown': score_bundle.score_components.get('max_drawdown', 0.1),
                'sharpe_ratio': score_bundle.score_components.get('sharpe_ratio', 1.0)
            }
        
        # 拡張選択の実行
        return self.enhanced_selector.select_strategies_enhanced(
            ticker=ticker,
            trend_analysis=base_result.trend_analysis.__dict__,
            strategy_scores=strategy_scores,
            criteria=custom_criteria,
            risk_metrics=risk_metrics
        )
    
    def _integrate_results(self, 
                         base_result: IntegratedDecisionResult,
                         enhanced_result: Any) -> IntegratedDecisionResult:
        """結果の統合"""
        
        # 統合モードに応じた処理
        if self.rule_engine_mode == RuleEngineMode.HYBRID:
            # ハイブリッド統合：両方の結果を組み合わせ
            integrated_strategies = list(set(
                base_result.strategy_selection.selected_strategies +
                enhanced_result.selected_strategies
            ))
            
            # 重みの統合
            integrated_weights = {}
            for strategy in integrated_strategies:
                base_weight = base_result.strategy_selection.strategy_weights.get(strategy, 0)
                enhanced_weight = enhanced_result.strategy_weights.get(strategy, 0)
                integrated_weights[strategy] = (base_weight + enhanced_weight) / 2
            
            # 信頼度の統合
            integrated_confidence = (
                base_result.strategy_selection.confidence_level +
                enhanced_result.confidence_level
            ) / 2
            
        else:
            # ルールエンジン優先：enhanced_resultを主に使用
            integrated_strategies = enhanced_result.selected_strategies
            integrated_weights = enhanced_result.strategy_weights
            integrated_confidence = enhanced_result.confidence_level
        
        # 新しい戦略選択結果を作成
        from config.strategy_selector import StrategySelection
        integrated_selection = StrategySelection(
            selected_strategies=integrated_strategies,
            strategy_scores={s: base_result.strategy_scores[s].final_score 
                           for s in integrated_strategies 
                           if s in base_result.strategy_scores},
            strategy_weights=integrated_weights,
            selection_reason=f"Rule-enhanced: {enhanced_result.selection_reason}",
            trend_analysis=base_result.trend_analysis.__dict__,
            confidence_level=integrated_confidence,
            total_score=sum(integrated_weights.values()),
            metadata={
                'rule_engine_used': True,
                'rule_engine_mode': self.rule_engine_mode.value,
                'base_selection': base_result.strategy_selection.metadata,
                'enhanced_selection': enhanced_result.metadata
            }
        )
        
        # IntegratedDecisionResultの更新
        integrated_result = IntegratedDecisionResult(
            trend_analysis=base_result.trend_analysis,
            strategy_scores=base_result.strategy_scores,
            strategy_selection=integrated_selection,
            processing_mode=base_result.processing_mode,
            integration_status=IntegrationStatus.COMPLETED,
            processing_time_ms=base_result.processing_time_ms,
            cache_hit_rate=base_result.cache_hit_rate,
            data_quality_assessment=base_result.data_quality_assessment,
            risk_assessment=base_result.risk_assessment,
            recommended_actions=base_result.recommended_actions,
            ticker=base_result.ticker,
            data_period=base_result.data_period
        )
        
        # ルールエンジン固有のメタデータを追加
        integrated_result.recommended_actions.append({
            'action': 'rule_engine_analysis',
            'confidence': integrated_confidence,
            'rule_metrics': self.rule_metrics.__dict__,
            'timestamp': datetime.now().isoformat()
        })
        
        return integrated_result
    
    def _update_rule_metrics(self, start_time: datetime, success: bool):
        """ルールエンジン指標の更新"""
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        self.rule_metrics.total_rule_executions += 1
        
        if success:
            self.rule_metrics.successful_executions += 1
        else:
            self.rule_metrics.failed_executions += 1
        
        # 平均実行時間の更新
        total_executions = self.rule_metrics.total_rule_executions
        current_avg = self.rule_metrics.average_execution_time_ms
        self.rule_metrics.average_execution_time_ms = (
            (current_avg * (total_executions - 1) + execution_time_ms) / total_executions
        )
        
        self.rule_metrics.last_execution_time = datetime.now()
    
    def batch_analyze_with_rules(self, 
                               tickers: List[str],
                               data_source: Union[Dict[str, pd.DataFrame], Callable],
                               rule_preferences: Optional[Dict[str, str]] = None,
                               custom_criteria: Optional[EnhancedSelectionCriteria] = None,
                               max_workers: int = 4) -> BatchProcessingResult:
        """
        ルールエンジンを使用したバッチ分析
        
        Args:
            tickers: ティッカーリスト
            data_source: データソース
            rule_preferences: ティッカー別ルール優先設定
            custom_criteria: カスタム選択基準
            max_workers: 最大ワーカー数
            
        Returns:
            BatchProcessingResult: バッチ処理結果
        """
        batch_start_time = datetime.now()
        successful_results = []
        failed_tickers = []
        
        try:
            # データの準備
            if callable(data_source):
                ticker_data = {ticker: data_source(ticker) for ticker in tickers}
            else:
                ticker_data = data_source
            
            # 並列処理または逐次処理
            if max_workers > 1:
                results = self._parallel_batch_processing(
                    ticker_data, rule_preferences, custom_criteria, max_workers
                )
            else:
                results = self._sequential_batch_processing(
                    ticker_data, rule_preferences, custom_criteria
                )
            
            # 結果の分類
            for ticker, result in results.items():
                if isinstance(result, IntegratedDecisionResult):
                    successful_results.append(result)
                else:
                    failed_tickers.append((ticker, str(result)))
            
            batch_end_time = datetime.now()
            
            # 処理サマリーの作成
            processing_summary = {
                'total_tickers': len(tickers),
                'successful_count': len(successful_results),
                'failed_count': len(failed_tickers),
                'success_rate': len(successful_results) / len(tickers) if tickers else 0,
                'total_processing_time_ms': (batch_end_time - batch_start_time).total_seconds() * 1000,
                'average_time_per_ticker_ms': 0,
                'rule_engine_usage': {
                    'total_executions': self.rule_metrics.total_rule_executions,
                    'success_rate': (self.rule_metrics.successful_executions / 
                                   max(1, self.rule_metrics.total_rule_executions)),
                    'average_execution_time_ms': self.rule_metrics.average_execution_time_ms
                }
            }
            
            if successful_results:
                total_time = sum(r.processing_time_ms for r in successful_results)
                processing_summary['average_time_per_ticker_ms'] = total_time / len(successful_results)
            
            return BatchProcessingResult(
                total_processed=len(tickers),
                successful_results=successful_results,
                failed_tickers=failed_tickers,
                processing_summary=processing_summary,
                batch_start_time=batch_start_time,
                batch_end_time=batch_end_time
            )
            
        except Exception as e:
            logger.error(f"Batch analysis with rules failed: {e}")
            
            return BatchProcessingResult(
                total_processed=len(tickers),
                successful_results=[],
                failed_tickers=[(t, str(e)) for t in tickers],
                processing_summary={'error': str(e)},
                batch_start_time=batch_start_time,
                batch_end_time=datetime.now()
            )
    
    def _sequential_batch_processing(self, 
                                   ticker_data: Dict[str, pd.DataFrame],
                                   rule_preferences: Optional[Dict[str, str]],
                                   custom_criteria: Optional[EnhancedSelectionCriteria]) -> Dict[str, Any]:
        """逐次バッチ処理"""
        results = {}
        
        for ticker, data in ticker_data.items():
            try:
                rule_pref = rule_preferences.get(ticker) if rule_preferences else None
                
                result = self.analyze_integrated_with_rules(
                    ticker=ticker,
                    data=data,
                    processing_mode=ProcessingMode.BATCH,
                    rule_preference=rule_pref,
                    custom_criteria=custom_criteria
                )
                
                results[ticker] = result
                
            except Exception as e:
                logger.error(f"Analysis failed for {ticker}: {e}")
                results[ticker] = e
        
        return results
    
    def _parallel_batch_processing(self, 
                                 ticker_data: Dict[str, pd.DataFrame],
                                 rule_preferences: Optional[Dict[str, str]],
                                 custom_criteria: Optional[EnhancedSelectionCriteria],
                                 max_workers: int) -> Dict[str, Any]:
        """並列バッチ処理"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        def process_ticker(ticker_info):
            ticker, data = ticker_info
            try:
                rule_pref = rule_preferences.get(ticker) if rule_preferences else None
                
                result = self.analyze_integrated_with_rules(
                    ticker=ticker,
                    data=data,
                    processing_mode=ProcessingMode.BATCH,
                    rule_preference=rule_pref,
                    custom_criteria=custom_criteria
                )
                
                return ticker, result
                
            except Exception as e:
                logger.error(f"Parallel analysis failed for {ticker}: {e}")
                return ticker, e
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(process_ticker, item): item[0] 
                for item in ticker_data.items()
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result_ticker, result = future.result()
                    results[result_ticker] = result
                except Exception as e:
                    logger.error(f"Future execution failed for {ticker}: {e}")
                    results[ticker] = e
        
        return results
    
    def get_rule_engine_status(self) -> Dict[str, Any]:
        """ルールエンジンのステータスを取得"""
        return {
            'mode': self.rule_engine_mode.value,
            'metrics': self.rule_metrics.__dict__,
            'configuration_summary': self.rule_config_manager.get_configuration_summary(),
            'enhanced_selector_stats': self.enhanced_selector.get_execution_stats(),
            'rule_performance': self.enhanced_selector.get_rule_performance(),
            'integration_config': self.integration_config
        }
    
    def update_rule_configuration(self, rule_config: Dict[str, Any]) -> bool:
        """ルール設定の更新"""
        try:
            success = self.rule_config_manager.add_rule_configuration(rule_config)
            if success:
                # ルールエンジンの再初期化
                self.enhanced_selector = EnhancedStrategySelector(
                    base_dir=str(self.base_dir)
                )
                logger.info(f"Rule configuration updated: {rule_config.get('name')}")
            return success
        except Exception as e:
            logger.error(f"Failed to update rule configuration: {e}")
            return False
    
    def optimize_rule_performance(self) -> Dict[str, Any]:
        """ルールパフォーマンスの最適化"""
        if not self.integration_config.get('auto_rule_optimization', False):
            return {'status': 'optimization_disabled'}
        
        try:
            # 統計の取得
            rule_stats = self.enhanced_selector.get_rule_performance()
            
            # パフォーマンスの低いルールの特定
            optimization_actions = []
            
            for rule_name, stats in rule_stats.get('rule_statistics', {}).items():
                success_rate = stats.get('success_rate', 0)
                avg_time = stats.get('average_time_ms', 0)
                
                if success_rate < 0.7:
                    optimization_actions.append({
                        'rule': rule_name,
                        'action': 'disable_low_success_rate',
                        'current_success_rate': success_rate
                    })
                
                if avg_time > 2000:
                    optimization_actions.append({
                        'rule': rule_name,
                        'action': 'optimize_execution_time',
                        'current_avg_time_ms': avg_time
                    })
            
            return {
                'status': 'optimization_completed',
                'actions_recommended': optimization_actions,
                'current_stats': rule_stats
            }
            
        except Exception as e:
            logger.error(f"Rule performance optimization failed: {e}")
            return {'status': 'optimization_failed', 'error': str(e)}

if __name__ == "__main__":
    # テスト用のサンプル実行
    import numpy as np
    
    # 統合インターフェースの初期化
    interface = RuleEngineIntegratedInterface(
        rule_engine_mode=RuleEngineMode.ENABLED
    )
    
    # テストデータの作成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 2,
        'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 2,
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # ルール統合分析の実行
    try:
        result = interface.analyze_integrated_with_rules(
            ticker='TEST',
            data=test_data,
            processing_mode=ProcessingMode.REALTIME
        )
        
        print("Rule-Integrated Analysis Result:")
        print(f"  Selected Strategies: {result.strategy_selection.selected_strategies}")
        print(f"  Confidence Level: {result.strategy_selection.confidence_level:.2f}")
        print(f"  Selection Reason: {result.strategy_selection.selection_reason}")
        print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
    
    # ルールエンジンステータス
    print("\nRule Engine Status:")
    status = interface.get_rule_engine_status()
    print(f"  Mode: {status['mode']}")
    print(f"  Total Executions: {status['metrics']['total_rule_executions']}")
    print(f"  Success Rate: {status['metrics']['successful_executions']}/{status['metrics']['total_rule_executions']}")
    
    # パフォーマンス最適化
    print("\nPerformance Optimization:")
    optimization_result = interface.optimize_rule_performance()
    print(f"  Status: {optimization_result['status']}")
    if 'actions_recommended' in optimization_result:
        print(f"  Recommended Actions: {len(optimization_result['actions_recommended'])}")
