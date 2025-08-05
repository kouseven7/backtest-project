"""
Module: Enhanced Strategy Selector
File: enhanced_strategy_selector.py
Description: 
  3-1-3「選択ルールの抽象化（差し替え可能に）」統合
  StrategySelectionRuleEngineを統合したEnhancedStrategySelector
  既存のStrategySelectorを拡張し、ルールエンジン機能を追加

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.strategy_selector
  - config.strategy_selection_rule_engine
  - config.trend_strategy_integration_interface
"""

import os
import sys
import json
import logging
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
    from config.strategy_selector import (
        StrategySelector, SelectionCriteria, StrategySelection, SelectionMethod, TrendType
    )
    from config.strategy_selection_rule_engine import (
        StrategySelectionRuleEngine, BaseSelectionRule, RuleContext, RuleExecutionResult,
        TrendBasedSelectionRule, ScoreBasedSelectionRule, RiskAdjustedSelectionRule,
        HybridSelectionRule, ConfigurableSelectionRule
    )
except ImportError as e:
    logging.getLogger(__name__).warning(f"Import error: {e}")

# ロガーの設定
logger = logging.getLogger(__name__)

class SelectionStrategy(Enum):
    """選択戦略"""
    LEGACY = "legacy"              # 従来のStrategySelector
    RULE_ENGINE = "rule_engine"    # ルールエンジン
    HYBRID = "hybrid"              # ハイブリッド
    AUTO = "auto"                  # 自動選択

@dataclass
class EnhancedSelectionCriteria(SelectionCriteria):
    """拡張選択基準"""
    # ルールエンジン関連
    selection_strategy: SelectionStrategy = SelectionStrategy.AUTO
    preferred_rule: Optional[str] = None
    enable_rule_fallback: bool = True
    rule_timeout_ms: float = 5000
    
    # リスク管理
    enable_risk_adjustment: bool = True
    risk_tolerance: float = 0.5
    max_correlation: float = 0.8
    
    # パフォーマンス
    enable_caching: bool = True
    cache_ttl_minutes: int = 15
    parallel_execution: bool = False

class EnhancedStrategySelector(StrategySelector):
    """
    拡張戦略選択器
    
    StrategySelectionRuleEngineを統合し、既存の機能を拡張
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 base_dir: Optional[str] = None,
                 rule_engine_config: Optional[str] = None):
        """拡張戦略選択器の初期化"""
        # 親クラスの初期化
        super().__init__(config_file, base_dir)
        
        # ルールエンジンの初期化
        rule_engine_dir = Path(base_dir) / "rule_engine" if base_dir else None
        self.rule_engine = StrategySelectionRuleEngine(rule_engine_dir)
        
        # 拡張機能の設定
        self.enhanced_config = self._load_enhanced_config(config_file)
        
        # 実行統計
        self.execution_stats = {
            'legacy_calls': 0,
            'rule_engine_calls': 0,
            'hybrid_calls': 0,
            'auto_decisions': 0,
            'cache_hits': 0,
            'total_execution_time_ms': 0
        }
        
        # キャッシュシステム
        self._selection_cache = {}
        self._cache_metadata = {}
        
        logger.info("EnhancedStrategySelector initialized with rule engine support")
    
    def _load_enhanced_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """拡張設定の読み込み"""
        enhanced_config = {
            'default_selection_strategy': 'auto',
            'rule_engine_priority': True,
            'cache_enabled': True,
            'cache_ttl_minutes': 15,
            'performance_tracking': True,
            'fallback_strategy': 'legacy'
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                enhanced_config.update(config.get('enhanced_settings', {}))
            except Exception as e:
                logger.warning(f"Failed to load enhanced config: {e}")
        
        return enhanced_config
    
    def select_strategies_enhanced(self, 
                                 ticker: str,
                                 trend_analysis: Dict[str, Any],
                                 strategy_scores: Dict[str, float],
                                 criteria: Optional[EnhancedSelectionCriteria] = None,
                                 risk_metrics: Optional[Dict[str, Any]] = None) -> StrategySelection:
        """
        拡張戦略選択（メインインターフェース）
        
        Args:
            ticker: ティッカーシンボル
            trend_analysis: トレンド分析結果
            strategy_scores: 戦略スコア
            criteria: 選択基準
            risk_metrics: リスク指標
            
        Returns:
            StrategySelection: 戦略選択結果
        """
        start_time = datetime.now()
        
        if criteria is None:
            criteria = EnhancedSelectionCriteria()
        
        # キャッシュチェック
        cache_key = self._generate_cache_key(ticker, trend_analysis, strategy_scores, criteria)
        if criteria.enable_caching and cache_key in self._selection_cache:
            cached_result, cache_time = self._selection_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < criteria.cache_ttl_minutes * 60:
                self.execution_stats['cache_hits'] += 1
                return cached_result
        
        # 選択戦略の決定
        strategy = self._determine_selection_strategy(criteria, trend_analysis, strategy_scores)
        
        # 戦略別実行
        result = None
        try:
            if strategy == SelectionStrategy.LEGACY:
                result = self._execute_legacy_selection(ticker, trend_analysis, strategy_scores, criteria)
                self.execution_stats['legacy_calls'] += 1
                
            elif strategy == SelectionStrategy.RULE_ENGINE:
                result = self._execute_rule_engine_selection(ticker, trend_analysis, strategy_scores, criteria, risk_metrics)
                self.execution_stats['rule_engine_calls'] += 1
                
            elif strategy == SelectionStrategy.HYBRID:
                result = self._execute_hybrid_selection(ticker, trend_analysis, strategy_scores, criteria, risk_metrics)
                self.execution_stats['hybrid_calls'] += 1
                
            else:  # AUTO
                result = self._execute_auto_selection(ticker, trend_analysis, strategy_scores, criteria, risk_metrics)
                self.execution_stats['auto_decisions'] += 1
                
        except Exception as e:
            logger.error(f"Enhanced selection failed: {e}")
            # フォールバック
            if criteria.enable_rule_fallback:
                result = self._execute_legacy_selection(ticker, trend_analysis, strategy_scores, criteria)
            else:
                raise
        
        # 結果の後処理
        if result:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.execution_stats['total_execution_time_ms'] += execution_time_ms
            
            # メタデータの追加
            result.metadata.update({
                'selection_strategy_used': strategy.value,
                'execution_time_ms': execution_time_ms,
                'rule_engine_available': True,
                'enhanced_selector_version': '1.0'
            })
            
            # キャッシュ保存
            if criteria.enable_caching:
                self._selection_cache[cache_key] = (result, datetime.now())
                self._cleanup_cache()
        
        return result
    
    def _determine_selection_strategy(self, 
                                    criteria: EnhancedSelectionCriteria,
                                    trend_analysis: Dict[str, Any],
                                    strategy_scores: Dict[str, float]) -> SelectionStrategy:
        """選択戦略の決定"""
        
        if criteria.selection_strategy != SelectionStrategy.AUTO:
            return criteria.selection_strategy
        
        # 自動決定ロジック
        # 1. データ品質チェック
        trend_confidence = trend_analysis.get('confidence', 0)
        score_quality = len(strategy_scores) / max(1, len(self.available_strategies))
        
        # 2. 複雑性チェック
        has_risk_metrics = criteria.enable_risk_adjustment
        has_complex_criteria = (
            criteria.enable_diversification or
            len(criteria.blacklist_strategies) > 0 or
            len(criteria.whitelist_strategies) > 0
        )
        
        # 3. 戦略決定
        if trend_confidence > 0.7 and score_quality > 0.8 and (has_risk_metrics or has_complex_criteria):
            return SelectionStrategy.RULE_ENGINE
        elif trend_confidence > 0.5 and has_complex_criteria:
            return SelectionStrategy.HYBRID
        else:
            return SelectionStrategy.LEGACY
    
    def _execute_legacy_selection(self, 
                                ticker: str,
                                trend_analysis: Dict[str, Any],
                                strategy_scores: Dict[str, float],
                                criteria: EnhancedSelectionCriteria) -> StrategySelection:
        """従来の選択ロジックを実行"""
        # 基底クラスのメソッドを呼び出し
        base_criteria = SelectionCriteria(
            method=criteria.method,
            min_score_threshold=criteria.min_score_threshold,
            max_strategies=criteria.max_strategies,
            min_strategies=criteria.min_strategies,
            confidence_threshold=criteria.confidence_threshold,
            trend_adaptation_weight=criteria.trend_adaptation_weight,
            enable_diversification=criteria.enable_diversification,
            blacklist_strategies=criteria.blacklist_strategies,
            whitelist_strategies=criteria.whitelist_strategies
        )
        
        if criteria.method == SelectionMethod.TOP_N:
            return self.select_strategies_by_score(ticker, strategy_scores, base_criteria)
        elif criteria.method == SelectionMethod.THRESHOLD:
            return self.select_strategies_by_threshold(ticker, strategy_scores, base_criteria)
        elif criteria.method == SelectionMethod.HYBRID:
            return self.select_strategies_hybrid(ticker, strategy_scores, base_criteria)
        elif criteria.method == SelectionMethod.WEIGHTED:
            return self.select_strategies_weighted(ticker, strategy_scores, base_criteria)
        else:  # ADAPTIVE
            return self.select_strategies_adaptive(ticker, trend_analysis, strategy_scores, base_criteria)
    
    def _execute_rule_engine_selection(self, 
                                     ticker: str,
                                     trend_analysis: Dict[str, Any],
                                     strategy_scores: Dict[str, float],
                                     criteria: EnhancedSelectionCriteria,
                                     risk_metrics: Optional[Dict[str, Any]] = None) -> StrategySelection:
        """ルールエンジンによる選択を実行"""
        
        # ルールコンテキストの作成
        context = RuleContext(
            strategy_scores=strategy_scores,
            trend_analysis=trend_analysis,
            selection_criteria=criteria,
            available_strategies=self.available_strategies,
            ticker=ticker,
            timestamp=datetime.now(),
            data_quality=self._calculate_data_quality(trend_analysis, strategy_scores),
            risk_metrics=risk_metrics or {}
        )
        
        # ルール実行
        rule_results = self.rule_engine.execute_rules(context, criteria.preferred_rule)
        
        # 最適結果の選択
        best_result = self.rule_engine.select_best_result(rule_results)
        
        if not best_result:
            raise ValueError("No valid rule execution results")
        
        # StrategySelection形式に変換
        return StrategySelection(
            selected_strategies=best_result.selected_strategies,
            strategy_scores=strategy_scores,
            strategy_weights=best_result.strategy_weights,
            selection_reason=best_result.reasoning,
            trend_analysis=trend_analysis,
            confidence_level=best_result.confidence,
            total_score=sum(best_result.strategy_weights.values()),
            metadata={
                'rule_used': best_result.rule_name,
                'execution_time_ms': best_result.execution_time_ms,
                'rule_metadata': best_result.metadata
            }
        )
    
    def _execute_hybrid_selection(self, 
                                ticker: str,
                                trend_analysis: Dict[str, Any],
                                strategy_scores: Dict[str, float],
                                criteria: EnhancedSelectionCriteria,
                                risk_metrics: Optional[Dict[str, Any]] = None) -> StrategySelection:
        """ハイブリッド選択を実行"""
        
        # 従来の選択
        legacy_result = self._execute_legacy_selection(ticker, trend_analysis, strategy_scores, criteria)
        
        # ルールエンジンの選択
        try:
            rule_result = self._execute_rule_engine_selection(ticker, trend_analysis, strategy_scores, criteria, risk_metrics)
            
            # 結果の統合
            combined_strategies = list(set(legacy_result.selected_strategies + rule_result.selected_strategies))
            
            # 重みの統合（平均）
            combined_weights = {}
            for strategy in combined_strategies:
                legacy_weight = legacy_result.strategy_weights.get(strategy, 0)
                rule_weight = rule_result.strategy_weights.get(strategy, 0)
                combined_weights[strategy] = (legacy_weight + rule_weight) / 2
            
            # 上位戦略の選択
            sorted_strategies = sorted(combined_weights.items(), key=lambda x: x[1], reverse=True)
            final_strategies = []
            final_weights = {}
            
            for strategy, weight in sorted_strategies[:criteria.max_strategies]:
                if weight > 0:
                    final_strategies.append(strategy)
                    final_weights[strategy] = weight
            
            # 重みの正規化
            if final_weights:
                total_weight = sum(final_weights.values())
                final_weights = {k: v/total_weight for k, v in final_weights.items()}
            
            # 信頼度の統合
            combined_confidence = (legacy_result.confidence_level + rule_result.confidence_level) / 2
            
            return StrategySelection(
                selected_strategies=final_strategies,
                strategy_scores=strategy_scores,
                strategy_weights=final_weights,
                selection_reason=f"Hybrid: {legacy_result.selection_reason} + {rule_result.selection_reason}",
                trend_analysis=trend_analysis,
                confidence_level=combined_confidence,
                total_score=sum(final_weights.values()),
                metadata={
                    'hybrid_method': 'legacy_rule_average',
                    'legacy_result': legacy_result.metadata,
                    'rule_result': rule_result.metadata
                }
            )
            
        except Exception as e:
            logger.warning(f"Rule engine execution failed in hybrid mode: {e}")
            return legacy_result
    
    def _execute_auto_selection(self, 
                              ticker: str,
                              trend_analysis: Dict[str, Any],
                              strategy_scores: Dict[str, float],
                              criteria: EnhancedSelectionCriteria,
                              risk_metrics: Optional[Dict[str, Any]] = None) -> StrategySelection:
        """自動選択を実行"""
        
        # 最適戦略の再決定（より詳細な分析）
        data_quality = self._calculate_data_quality(trend_analysis, strategy_scores)
        complexity_score = self._calculate_complexity_score(criteria, risk_metrics)
        
        if data_quality > 0.8 and complexity_score > 0.6:
            return self._execute_rule_engine_selection(ticker, trend_analysis, strategy_scores, criteria, risk_metrics)
        elif data_quality > 0.6 or complexity_score > 0.4:
            return self._execute_hybrid_selection(ticker, trend_analysis, strategy_scores, criteria, risk_metrics)
        else:
            return self._execute_legacy_selection(ticker, trend_analysis, strategy_scores, criteria)
    
    def _calculate_data_quality(self, trend_analysis: Dict[str, Any], strategy_scores: Dict[str, float]) -> float:
        """データ品質を計算"""
        quality_score = 0.0
        
        # トレンド分析の品質
        if trend_analysis:
            trend_confidence = trend_analysis.get('confidence', 0)
            trend_strength = trend_analysis.get('strength', 0)
            quality_score += (trend_confidence + trend_strength) / 2 * 0.5
        
        # 戦略スコアの品質
        if strategy_scores:
            score_coverage = len(strategy_scores) / max(1, len(self.available_strategies))
            avg_score = sum(strategy_scores.values()) / len(strategy_scores)
            quality_score += (score_coverage + avg_score) / 2 * 0.5
        
        return min(1.0, quality_score)
    
    def _calculate_complexity_score(self, criteria: EnhancedSelectionCriteria, risk_metrics: Optional[Dict[str, Any]]) -> float:
        """複雑性スコアを計算"""
        complexity = 0.0
        
        # 基準の複雑性
        if criteria.enable_diversification:
            complexity += 0.2
        if criteria.blacklist_strategies:
            complexity += 0.1
        if criteria.whitelist_strategies:
            complexity += 0.1
        if criteria.enable_risk_adjustment:
            complexity += 0.3
        
        # リスク指標の有無
        if risk_metrics:
            complexity += 0.3
        
        return min(1.0, complexity)
    
    def _generate_cache_key(self, ticker: str, trend_analysis: Dict[str, Any], 
                          strategy_scores: Dict[str, float], criteria: EnhancedSelectionCriteria) -> str:
        """キャッシュキーを生成"""
        key_data = {
            'ticker': ticker,
            'trend_type': trend_analysis.get('trend_type', 'unknown'),
            'trend_confidence': round(trend_analysis.get('confidence', 0), 2),
            'strategy_scores_hash': hash(tuple(sorted(strategy_scores.items()))),
            'criteria_hash': hash((
                criteria.method.value,
                criteria.min_score_threshold,
                criteria.max_strategies,
                criteria.selection_strategy.value
            ))
        }
        
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def _cleanup_cache(self):
        """期限切れキャッシュのクリーンアップ"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (_, cache_time) in self._selection_cache.items():
            if (current_time - cache_time).total_seconds() > 3600:  # 1時間後に削除
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._selection_cache[key]
    
    def add_custom_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """カスタムルールを追加"""
        custom_rule = ConfigurableSelectionRule(
            name=rule_name,
            config=rule_config,
            priority=rule_config.get('priority', 50)
        )
        self.rule_engine.add_rule(custom_rule)
        logger.info(f"Added custom rule: {rule_name}")
    
    def remove_rule(self, rule_name: str):
        """ルールを削除"""
        self.rule_engine.remove_rule(rule_name)
    
    def get_rule_performance(self) -> Dict[str, Any]:
        """ルールパフォーマンスを取得"""
        return self.rule_engine.get_performance_summary()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """実行統計を取得"""
        stats = self.execution_stats.copy()
        stats.update({
            'cache_size': len(self._selection_cache),
            'rule_count': len(self.rule_engine.rules),
            'average_execution_time_ms': (
                stats['total_execution_time_ms'] / 
                max(1, sum([stats['legacy_calls'], stats['rule_engine_calls'], 
                           stats['hybrid_calls'], stats['auto_decisions']]))
            )
        })
        return stats
    
    def reset_stats(self):
        """統計をリセット"""
        self.execution_stats = {
            'legacy_calls': 0,
            'rule_engine_calls': 0,
            'hybrid_calls': 0,
            'auto_decisions': 0,
            'cache_hits': 0,
            'total_execution_time_ms': 0
        }
        self._selection_cache.clear()
        logger.info("EnhancedStrategySelector stats reset")

# 既存システムとの互換性のためのファクトリー関数
def create_enhanced_selector(config_file: Optional[str] = None, 
                           base_dir: Optional[str] = None) -> EnhancedStrategySelector:
    """拡張戦略選択器を作成"""
    return EnhancedStrategySelector(config_file, base_dir)

# 後方互換性のためのエイリアス
EnhancedSelector = EnhancedStrategySelector

if __name__ == "__main__":
    # テスト用のサンプル実行
    import hashlib
    from datetime import datetime
    
    # 拡張選択器の初期化
    selector = EnhancedStrategySelector()
    
    # テストデータ
    test_data = {
        'ticker': 'AAPL',
        'trend_analysis': {
            'trend_type': 'uptrend',
            'confidence': 0.85,
            'strength': 0.7
        },
        'strategy_scores': {
            'momentum': 0.8,
            'mean_reversion': 0.6,
            'breakout': 0.9,
            'pairs': 0.5
        },
        'risk_metrics': {
            'momentum': {'volatility': 0.15, 'sharpe_ratio': 1.2},
            'breakout': {'volatility': 0.25, 'sharpe_ratio': 1.0}
        }
    }
    
    # 拡張選択の実行
    criteria = EnhancedSelectionCriteria(
        selection_strategy=SelectionStrategy.AUTO,
        enable_risk_adjustment=True,
        enable_caching=True
    )
    
    result = selector.select_strategies_enhanced(
        ticker=test_data['ticker'],
        trend_analysis=test_data['trend_analysis'],
        strategy_scores=test_data['strategy_scores'],
        criteria=criteria,
        risk_metrics=test_data['risk_metrics']
    )
    
    print("Enhanced Strategy Selection Result:")
    print(f"  Selected Strategies: {result.selected_strategies}")
    print(f"  Strategy Weights: {result.strategy_weights}")
    print(f"  Selection Reason: {result.selection_reason}")
    print(f"  Confidence Level: {result.confidence_level:.2f}")
    print(f"  Metadata: {result.metadata}")
    
    # 統計情報
    print("\nExecution Statistics:")
    stats = selector.get_execution_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # ルールパフォーマンス
    print("\nRule Performance:")
    perf = selector.get_rule_performance()
    for rule_name, rule_stats in perf.get('rule_statistics', {}).items():
        print(f"  {rule_name}: Success Rate {rule_stats['success_rate']:.1%}")
