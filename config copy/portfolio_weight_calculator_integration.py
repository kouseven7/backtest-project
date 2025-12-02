"""
Module: Portfolio Weight Calculator Integration
File: portfolio_weight_calculator_integration.py
Description: 
  3-2-3「重み付けパターンテンプレート作成」とPortfolioWeightCalculatorの統合
  既存の3-2-1、3-2-2機能とシームレスに統合

Author: imega
Created: 2025-07-15
Modified: 2025-07-15

Dependencies:
  - config.portfolio_weight_calculator
  - config.portfolio_weight_pattern_engine_v2
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存モジュールのインポート
try:
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, AllocationResult, WeightAllocationConfig,
        PortfolioConstraints, AllocationMethod
    )
    from config.portfolio_weight_pattern_engine_v2 import (
        AdvancedPatternEngineV2, PatternTemplate, RiskTolerance, 
        MarketEnvironment, TemplateCategory
    )
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class PortfolioWeightCalculatorEnhanced(PortfolioWeightCalculator):
    """
    3-2-3統合版PortfolioWeightCalculator
    パターンテンプレート機能を組み込んだ拡張版
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 base_dir: Optional[str] = None,
                 pattern_engine_dir: Optional[str] = None):
        """
        拡張版重み計算エンジンの初期化
        """
        # 基底クラスの初期化
        super().__init__(config_file, base_dir)
        
        # 3-2-3: パターンエンジンの初期化
        self.pattern_engine = AdvancedPatternEngineV2(pattern_engine_dir)
        
        # テンプレート管理
        self.active_template: Optional[PatternTemplate] = None
        self.template_application_history: List[Tuple[datetime, str, PatternTemplate]] = []
        
        logger.info("PortfolioWeightCalculatorEnhanced initialized with 3-2-3 pattern engine")

    def calculate_weights_with_template(self,
                                      strategy_scores: Dict[str, float],
                                      risk_tolerance: Union[str, RiskTolerance],
                                      market_data: Optional[pd.DataFrame] = None,
                                      custom_template: Optional[PatternTemplate] = None,
                                      override_config: Optional[WeightAllocationConfig] = None) -> AllocationResult:
        """
        3-2-3: テンプレートベースの重み計算
        
        Args:
            strategy_scores: 戦略スコア辞書
            risk_tolerance: リスク許容度
            market_data: 市場データ（市場環境判定用）
            custom_template: カスタムテンプレート（任意）
            override_config: 設定オーバーライド（任意）
        
        Returns:
            AllocationResult: 配分結果
        """
        try:
            # リスク許容度の正規化
            if isinstance(risk_tolerance, str):
                risk_tolerance = RiskTolerance(risk_tolerance.lower())
            
            # テンプレートの選択
            if custom_template:
                template = custom_template
                logger.info(f"Using custom template: {template.name}")
            else:
                template = self.pattern_engine.recommend_template(
                    risk_tolerance=risk_tolerance,
                    market_data=market_data
                )
                logger.info(f"Recommended template: {template.name}")
            
            # テンプレートを適用した設定の作成
            enhanced_config = self._apply_template_to_config(template, override_config)
            
            # 従来の重み計算実行
            allocation_result = self.calculate_weights(
                strategy_scores=strategy_scores,
                config=enhanced_config,
                market_context=self._create_market_context(market_data, template)
            )
            
            # テンプレート情報をメタデータに追加
            allocation_result.metadata.update({
                'template_name': template.name,
                'template_category': template.category.value,
                'risk_tolerance': risk_tolerance.value,
                'market_environment': template.market_environment.value if template.market_environment else None,
                'template_applied_at': datetime.now().isoformat()
            })
            
            # アクティブテンプレートの更新
            self.active_template = template
            
            # 履歴の記録
            self.template_application_history.append(
                (datetime.now(), f"weights_calculation", template)
            )
            
            logger.info(f"Template-based weight calculation completed using {template.name}")
            return allocation_result
            
        except Exception as e:
            logger.error(f"Error in template-based weight calculation: {e}")
            # フォールバック: 通常の重み計算
            return self.calculate_weights(strategy_scores, self.config)

    def _apply_template_to_config(self, 
                                 template: PatternTemplate,
                                 base_config: Optional[WeightAllocationConfig] = None) -> WeightAllocationConfig:
        """
        テンプレートを重み計算設定に適用
        
        Args:
            template: 適用するテンプレート
            base_config: ベース設定（任意）
        
        Returns:
            WeightAllocationConfig: テンプレート適用済み設定
        """
        # ベース設定の取得
        if base_config is None:
            base_config = self.config
        
        # 制約条件の調整
        enhanced_constraints = PortfolioConstraints(
            max_individual_weight=template.max_individual_weight,
            min_individual_weight=template.min_individual_weight,
            max_strategies=template.max_strategies,
            min_strategies=template.min_strategies,
            concentration_limit=template.concentration_limit,
            
            # 3-2-2機能の適用
            enable_hierarchical_minimum_weights=template.enable_hierarchical_weights,
            weight_adjustment_method=template.weight_adjustment_method,
            enable_conditional_exclusion=template.enable_conditional_exclusion,
            
            # 既存制約の継承
            max_correlation_threshold=base_config.constraints.max_correlation_threshold,
            min_score_threshold=base_config.constraints.min_score_threshold,
            max_turnover=base_config.constraints.max_turnover,
            risk_budget=base_config.constraints.risk_budget,
            leverage_limit=base_config.constraints.leverage_limit
        )
        
        # 配分手法の決定
        allocation_method = self._map_allocation_method(template.allocation_method)
        
        # 拡張設定の作成
        enhanced_config = WeightAllocationConfig(
            method=allocation_method,
            constraints=enhanced_constraints,
            rebalance_frequency=base_config.rebalance_frequency,
            risk_aversion=base_config.risk_aversion * template.volatility_adjustment_factor,
            confidence_weight=base_config.confidence_weight,
            trend_weight=base_config.trend_weight * template.trend_sensitivity,
            volatility_lookback=base_config.volatility_lookback,
            enable_dynamic_adjustment=base_config.enable_dynamic_adjustment,
            enable_momentum_bias=template.momentum_bias != 0.0
        )
        
        return enhanced_config

    def _map_allocation_method(self, template_method: str) -> AllocationMethod:
        """テンプレートの配分手法をAllocationMethodにマッピング"""
        method_mapping = {
            'equal_weight': AllocationMethod.EQUAL_WEIGHT,
            'risk_adjusted': AllocationMethod.RISK_ADJUSTED,
            'risk_parity': AllocationMethod.RISK_ADJUSTED,
            'optimal': AllocationMethod.RISK_ADJUSTED,
            'momentum_weighted': AllocationMethod.SCORE_PROPORTIONAL,
            'hierarchical': AllocationMethod.HIERARCHICAL,
            'kelly': AllocationMethod.KELLY_CRITERION,
            'adaptive': AllocationMethod.RISK_ADJUSTED
        }
        
        return method_mapping.get(template_method, AllocationMethod.RISK_ADJUSTED)

    def _create_market_context(self, 
                              market_data: Optional[pd.DataFrame],
                              template: PatternTemplate) -> Optional[Dict[str, Any]]:
        """市場コンテキストの作成"""
        if market_data is None or market_data.empty:
            return None
        
        try:
            # 市場環境の判定
            market_env = self.pattern_engine.detect_market_environment(market_data)
            
            # 基本統計の計算
            if 'close' in market_data.columns:
                prices = market_data['close'].astype(float)
            else:
                prices = market_data.iloc[:, 0].astype(float)
            
            returns = prices.pct_change().dropna()
            
            market_context = {
                'market_environment': market_env.value,
                'current_volatility': float(returns.std() * np.sqrt(252)),
                'recent_return': float(returns.tail(20).mean()),
                'trend_strength': self.pattern_engine._calculate_trend_strength(prices),
                'template_volatility_adjustment': template.volatility_adjustment_factor,
                'template_trend_sensitivity': template.trend_sensitivity,
                'template_momentum_bias': template.momentum_bias
            }
            
            return market_context
            
        except Exception as e:
            logger.warning(f"Error creating market context: {e}")
            return None

    def get_template_performance_analysis(self, 
                                        template_name: str,
                                        start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        テンプレートパフォーマンス分析
        
        Args:
            template_name: 分析対象テンプレート名
            start_date: 分析開始日（任意）
        
        Returns:
            Dict[str, Any]: パフォーマンス分析結果
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=90)
            
            # 履歴からテンプレート使用データを抽出
            template_history = [
                (timestamp, action, template) 
                for timestamp, action, template in self.template_application_history
                if template.name == template_name and timestamp >= start_date
            ]
            
            if not template_history:
                return {"error": f"No usage history found for template {template_name}"}
            
            # 配分履歴からパフォーマンスデータを抽出
            performance_data = []
            for entry in self._allocation_history:
                if (entry.metadata.get('template_name') == template_name and 
                    entry.timestamp >= start_date):
                    performance_data.append({
                        'timestamp': entry.timestamp,
                        'sharpe_ratio': entry.sharpe_ratio,
                        'expected_return': entry.expected_return,
                        'expected_risk': entry.expected_risk,
                        'diversification_ratio': entry.diversification_ratio
                    })
            
            if not performance_data:
                return {"error": f"No performance data found for template {template_name}"}
            
            # 統計計算
            sharpe_ratios = [p['sharpe_ratio'] for p in performance_data]
            returns = [p['expected_return'] for p in performance_data]
            risks = [p['expected_risk'] for p in performance_data]
            
            analysis = {
                'template_name': template_name,
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': datetime.now().isoformat()
                },
                'usage_count': len(template_history),
                'performance_metrics': {
                    'avg_sharpe_ratio': np.mean(sharpe_ratios),
                    'avg_expected_return': np.mean(returns),
                    'avg_expected_risk': np.mean(risks),
                    'sharpe_stability': np.std(sharpe_ratios),
                    'return_consistency': np.std(returns)
                },
                'recent_performance': performance_data[-5:] if len(performance_data) >= 5 else performance_data
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing template performance: {e}")
            return {"error": str(e)}

    def create_adaptive_template(self,
                               base_risk_tolerance: RiskTolerance,
                               market_data: pd.DataFrame,
                               performance_history: List[Dict[str, Any]],
                               template_name: Optional[str] = None) -> PatternTemplate:
        """
        アダプティブテンプレートの作成
        過去のパフォーマンスと現在の市場状況に基づいてテンプレートを最適化
        
        Args:
            base_risk_tolerance: ベースとなるリスク許容度
            market_data: 市場データ
            performance_history: パフォーマンス履歴
            template_name: テンプレート名（任意）
        
        Returns:
            PatternTemplate: 作成されたアダプティブテンプレート
        """
        try:
            if template_name is None:
                template_name = f"adaptive_{base_risk_tolerance.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 市場環境の判定
            market_env = self.pattern_engine.detect_market_environment(market_data)
            
            # パフォーマンス履歴の分析
            performance_metrics = self._analyze_performance_history(performance_history)
            
            # ベーステンプレートの取得
            base_template = self.pattern_engine.recommend_template(base_risk_tolerance, market_data)
            
            # カスタム設定の生成
            custom_settings = self._generate_adaptive_settings(
                performance_metrics, 
                market_env, 
                base_template
            )
            
            # アダプティブテンプレートの作成
            adaptive_template = self.pattern_engine.create_custom_template(
                name=template_name,
                risk_tolerance=base_risk_tolerance,
                market_environment=market_env,
                custom_settings=custom_settings
            )
            
            logger.info(f"Created adaptive template: {template_name}")
            return adaptive_template
            
        except Exception as e:
            logger.error(f"Error creating adaptive template: {e}")
            # フォールバック: 基本テンプレート
            return self.pattern_engine.recommend_template(base_risk_tolerance, market_data)

    def _analyze_performance_history(self, performance_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """パフォーマンス履歴の分析"""
        if not performance_history:
            return {}
        
        try:
            returns = [p.get('return', 0) for p in performance_history if 'return' in p]
            risks = [p.get('risk', 0) for p in performance_history if 'risk' in p]
            
            metrics = {
                'avg_return': np.mean(returns) if returns else 0,
                'avg_risk': np.mean(risks) if risks else 0,
                'return_volatility': np.std(returns) if len(returns) > 1 else 0,
                'risk_consistency': np.std(risks) if len(risks) > 1 else 0,
                'sharpe_trend': self._calculate_trend([p.get('sharpe_ratio', 0) for p in performance_history])
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error analyzing performance history: {e}")
            return {}

    def _calculate_trend(self, values: List[float]) -> float:
        """値の列のトレンドを計算"""
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return float(slope)
        except:
            return 0.0

    def _generate_adaptive_settings(self, 
                                   performance_metrics: Dict[str, float],
                                   market_env: MarketEnvironment,
                                   base_template: PatternTemplate) -> Dict[str, Any]:
        """アダプティブ設定の生成"""
        settings = {}
        
        # パフォーマンストレンドに基づく調整
        sharpe_trend = performance_metrics.get('sharpe_trend', 0)
        if sharpe_trend > 0:
            # パフォーマンス向上中：やや積極的に
            settings['volatility_adjustment_factor'] = base_template.volatility_adjustment_factor * 1.1
            settings['max_individual_weight'] = min(0.8, base_template.max_individual_weight * 1.05)
        elif sharpe_trend < -0.1:
            # パフォーマンス悪化：保守的に
            settings['volatility_adjustment_factor'] = base_template.volatility_adjustment_factor * 0.9
            settings['max_individual_weight'] = max(0.2, base_template.max_individual_weight * 0.95)
        
        # 市場環境に基づく調整
        if market_env == MarketEnvironment.VOLATILE:
            settings['concentration_limit'] = max(0.4, base_template.concentration_limit * 0.8)
            settings['min_strategies'] = max(3, base_template.min_strategies + 1)
        elif market_env == MarketEnvironment.CRISIS:
            settings['max_individual_weight'] = min(0.3, base_template.max_individual_weight)
            settings['concentration_limit'] = 0.5
        
        # リスク一貫性に基づく調整
        risk_consistency = performance_metrics.get('risk_consistency', 0)
        if risk_consistency > 0.05:  # リスクが不安定
            settings['weight_adjustment_method'] = 'gradual'
            settings['enable_hierarchical_weights'] = True
        
        return settings

    def get_active_template_info(self) -> Optional[Dict[str, Any]]:
        """アクティブテンプレートの情報を取得"""
        if self.active_template is None:
            return None
        
        return {
            'name': self.active_template.name,
            'category': self.active_template.category.value,
            'risk_tolerance': self.active_template.risk_tolerance.value,
            'market_environment': self.active_template.market_environment.value if self.active_template.market_environment else None,
            'description': self.active_template.description,
            'created_at': self.active_template.created_at.isoformat(),
            'settings': {
                'allocation_method': self.active_template.allocation_method,
                'max_strategies': self.active_template.max_strategies,
                'min_strategies': self.active_template.min_strategies,
                'max_individual_weight': self.active_template.max_individual_weight,
                'min_individual_weight': self.active_template.min_individual_weight,
                'concentration_limit': self.active_template.concentration_limit,
                'volatility_adjustment_factor': self.active_template.volatility_adjustment_factor,
                'trend_sensitivity': self.active_template.trend_sensitivity,
                'momentum_bias': self.active_template.momentum_bias
            }
        }

# 便利関数の定義
def create_enhanced_calculator(config_file: Optional[str] = None,
                             base_dir: Optional[str] = None) -> PortfolioWeightCalculatorEnhanced:
    """拡張版PortfolioWeightCalculatorのファクトリ関数"""
    return PortfolioWeightCalculatorEnhanced(config_file, base_dir)

def quick_template_calculation(strategy_scores: Dict[str, float],
                             risk_tolerance: str,
                             market_data: Optional[pd.DataFrame] = None) -> AllocationResult:
    """クイックテンプレート計算"""
    calculator = PortfolioWeightCalculatorEnhanced()
    return calculator.calculate_weights_with_template(
        strategy_scores=strategy_scores,
        risk_tolerance=risk_tolerance,
        market_data=market_data
    )

if __name__ == "__main__":
    # 基本動作テスト
    print("=== 3-2-3 Portfolio Weight Calculator Integration Demo ===")
    
    # 拡張版計算器の初期化
    calculator = PortfolioWeightCalculatorEnhanced()
    
    # サンプルデータ
    sample_scores = {
        'VWAP_Bounce': 0.75,
        'Momentum_Investing': 0.65,
        'Opening_Gap': 0.55,
        'Breakout': 0.45
    }
    
    # テンプレートベース計算のテスト
    print("\n=== テンプレートベース重み計算テスト ===")
    
    for risk in ['conservative', 'balanced', 'aggressive']:
        try:
            result = calculator.calculate_weights_with_template(
                strategy_scores=sample_scores,
                risk_tolerance=risk
            )
            
            print(f"\n{risk.upper()} テンプレート結果:")
            print(f"- テンプレート: {result.metadata.get('template_name', 'Unknown')}")
            print(f"- 期待リターン: {result.expected_return:.4f}")
            print(f"- 期待リスク: {result.expected_risk:.4f}")
            print(f"- シャープレシオ: {result.sharpe_ratio:.4f}")
            print(f"- 戦略重み: {result.strategy_weights}")
            
        except Exception as e:
            print(f"{risk}での計算エラー: {e}")
    
    # アクティブテンプレート情報の表示
    active_info = calculator.get_active_template_info()
    if active_info:
        print(f"\n=== アクティブテンプレート情報 ===")
        print(f"名前: {active_info['name']}")
        print(f"説明: {active_info['description']}")
    
    print("\n3-2-3 統合実装完了！")
