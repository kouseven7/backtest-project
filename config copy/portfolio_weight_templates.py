"""
Module: Portfolio Weight Template Manager
File: portfolio_weight_templates.py
Description: 
  3-2-1「スコアベースの資金配分計算式設計」の一部
  ポートフォリオ重み配分のテンプレートシステム
  5つの事前定義済みテンプレートと動的カスタマイズ機能

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.portfolio_weight_calculator
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.portfolio_weight_calculator import (
        WeightAllocationConfig, PortfolioConstraints, AllocationMethod, 
        RebalanceFrequency
    )
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """テンプレートタイプ"""
    CONSERVATIVE = "conservative"      # 保守的配分
    BALANCED = "balanced"             # バランス型配分
    AGGRESSIVE = "aggressive"         # 積極的配分
    GROWTH_FOCUSED = "growth_focused" # 成長重視配分
    INCOME_FOCUSED = "income_focused" # 収益重視配分

class MarketRegime(Enum):
    """市場環境"""
    BULL_MARKET = "bull_market"       # 強気相場
    BEAR_MARKET = "bear_market"       # 弱気相場
    SIDEWAYS = "sideways"             # 横這い相場
    VOLATILE = "volatile"             # 高ボラティリティ
    LOW_VOLATILITY = "low_volatility" # 低ボラティリティ

@dataclass
class WeightTemplate:
    """重みテンプレート"""
    name: str
    template_type: TemplateType
    description: str
    config: WeightAllocationConfig
    suitable_market_regimes: List[MarketRegime] = field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high
    expected_turnover: float = 0.15
    min_capital: float = 100000.0
    tags: List[str] = field(default_factory=list)
    performance_expectations: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class WeightTemplateManager:
    """
    ポートフォリオ重みテンプレート管理システム
    
    機能:
    1. 5つの事前定義済みテンプレート
    2. カスタムテンプレートの作成・保存
    3. 市場環境に基づく自動テンプレート推奨
    4. テンプレートの動的調整
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """テンプレートマネージャーの初期化"""
        self.templates_dir = Path(templates_dir) if templates_dir else Path("config/portfolio_templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # 事前定義済みテンプレートの初期化
        self.predefined_templates = self._initialize_predefined_templates()
        
        # カスタムテンプレートの読み込み
        self.custom_templates = self._load_custom_templates()
        
        logger.info(f"WeightTemplateManager initialized with {len(self.predefined_templates)} predefined and {len(self.custom_templates)} custom templates")

    def _initialize_predefined_templates(self) -> Dict[str, WeightTemplate]:
        """事前定義済みテンプレートの初期化"""
        templates = {}
        
        # 1. 保守的配分テンプレート
        conservative_config = WeightAllocationConfig(
            method=AllocationMethod.RISK_ADJUSTED,
            constraints=PortfolioConstraints(
                max_individual_weight=0.25,
                min_individual_weight=0.10,
                max_strategies=4,
                min_strategies=3,
                max_correlation_threshold=0.6,
                min_score_threshold=0.5,
                max_turnover=0.10,
                risk_budget=0.10,
                concentration_limit=0.50
            ),
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            risk_aversion=3.0,
            confidence_weight=0.4,
            trend_weight=0.1,
            enable_dynamic_adjustment=False
        )
        
        templates["conservative"] = WeightTemplate(
            name="Conservative Portfolio",
            template_type=TemplateType.CONSERVATIVE,
            description="低リスク・安定重視のポートフォリオ配分",
            config=conservative_config,
            suitable_market_regimes=[MarketRegime.BEAR_MARKET, MarketRegime.VOLATILE],
            risk_level="low",
            expected_turnover=0.08,
            min_capital=50000.0,
            tags=["low_risk", "stable", "defensive"],
            performance_expectations={
                "expected_return": 0.06,
                "expected_volatility": 0.08,
                "max_drawdown": 0.05
            }
        )
        
        # 2. バランス型配分テンプレート
        balanced_config = WeightAllocationConfig(
            method=AllocationMethod.RISK_ADJUSTED,
            constraints=PortfolioConstraints(
                max_individual_weight=0.35,
                min_individual_weight=0.08,
                max_strategies=5,
                min_strategies=2,
                max_correlation_threshold=0.75,
                min_score_threshold=0.4,
                max_turnover=0.15,
                risk_budget=0.12,
                concentration_limit=0.60
            ),
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            risk_aversion=2.0,
            confidence_weight=0.3,
            trend_weight=0.2,
            enable_dynamic_adjustment=True
        )
        
        templates["balanced"] = WeightTemplate(
            name="Balanced Portfolio",
            template_type=TemplateType.BALANCED,
            description="リスクとリターンのバランスを重視した配分",
            config=balanced_config,
            suitable_market_regimes=[MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY],
            risk_level="medium",
            expected_turnover=0.12,
            min_capital=100000.0,
            tags=["balanced", "moderate", "diversified"],
            performance_expectations={
                "expected_return": 0.10,
                "expected_volatility": 0.12,
                "max_drawdown": 0.08
            }
        )
        
        # 3. 積極的配分テンプレート
        aggressive_config = WeightAllocationConfig(
            method=AllocationMethod.SCORE_PROPORTIONAL,
            constraints=PortfolioConstraints(
                max_individual_weight=0.50,
                min_individual_weight=0.05,
                max_strategies=6,
                min_strategies=2,
                max_correlation_threshold=0.85,
                min_score_threshold=0.3,
                max_turnover=0.25,
                risk_budget=0.20,
                concentration_limit=0.75
            ),
            rebalance_frequency=RebalanceFrequency.DAILY,
            risk_aversion=1.0,
            confidence_weight=0.2,
            trend_weight=0.3,
            enable_dynamic_adjustment=True,
            enable_momentum_bias=True
        )
        
        templates["aggressive"] = WeightTemplate(
            name="Aggressive Portfolio",
            template_type=TemplateType.AGGRESSIVE,
            description="高リターン追求の積極的な配分",
            config=aggressive_config,
            suitable_market_regimes=[MarketRegime.BULL_MARKET, MarketRegime.VOLATILE],
            risk_level="high",
            expected_turnover=0.20,
            min_capital=200000.0,
            tags=["high_risk", "growth", "momentum"],
            performance_expectations={
                "expected_return": 0.15,
                "expected_volatility": 0.18,
                "max_drawdown": 0.12
            }
        )
        
        # 4. 成長重視配分テンプレート
        growth_config = WeightAllocationConfig(
            method=AllocationMethod.HIERARCHICAL,
            constraints=PortfolioConstraints(
                max_individual_weight=0.40,
                min_individual_weight=0.08,
                max_strategies=4,
                min_strategies=2,
                max_correlation_threshold=0.70,
                min_score_threshold=0.45,
                max_turnover=0.18,
                risk_budget=0.15,
                concentration_limit=0.65
            ),
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            risk_aversion=1.5,
            confidence_weight=0.25,
            trend_weight=0.35,
            enable_dynamic_adjustment=True
        )
        
        templates["growth_focused"] = WeightTemplate(
            name="Growth Focused Portfolio",
            template_type=TemplateType.GROWTH_FOCUSED,
            description="長期成長を重視した配分戦略",
            config=growth_config,
            suitable_market_regimes=[MarketRegime.BULL_MARKET, MarketRegime.LOW_VOLATILITY],
            risk_level="medium-high",
            expected_turnover=0.15,
            min_capital=150000.0,
            tags=["growth", "long_term", "trend_following"],
            performance_expectations={
                "expected_return": 0.12,
                "expected_volatility": 0.14,
                "max_drawdown": 0.10
            }
        )
        
        # 5. 収益重視配分テンプレート
        income_config = WeightAllocationConfig(
            method=AllocationMethod.RISK_ADJUSTED,
            constraints=PortfolioConstraints(
                max_individual_weight=0.30,
                min_individual_weight=0.12,
                max_strategies=3,
                min_strategies=2,
                max_correlation_threshold=0.65,
                min_score_threshold=0.6,
                max_turnover=0.08,
                risk_budget=0.08,
                concentration_limit=0.55
            ),
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            risk_aversion=2.5,
            confidence_weight=0.35,
            trend_weight=0.15,
            enable_dynamic_adjustment=False
        )
        
        templates["income_focused"] = WeightTemplate(
            name="Income Focused Portfolio",
            template_type=TemplateType.INCOME_FOCUSED,
            description="安定収益を重視した配分戦略",
            config=income_config,
            suitable_market_regimes=[MarketRegime.SIDEWAYS, MarketRegime.BEAR_MARKET],
            risk_level="low-medium",
            expected_turnover=0.06,
            min_capital=75000.0,
            tags=["income", "stable", "low_turnover"],
            performance_expectations={
                "expected_return": 0.08,
                "expected_volatility": 0.10,
                "max_drawdown": 0.06
            }
        )
        
        return templates

    def _load_custom_templates(self) -> Dict[str, WeightTemplate]:
        """カスタムテンプレートの読み込み"""
        custom_templates = {}
        
        templates_file = self.templates_dir / "custom_templates.json"
        if templates_file.exists():
            try:
                with open(templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for template_data in data.get("templates", []):
                    template = self._deserialize_template(template_data)
                    if template:
                        custom_templates[template.name.lower().replace(" ", "_")] = template
                
                logger.info(f"Loaded {len(custom_templates)} custom templates")
                
            except Exception as e:
                logger.warning(f"Error loading custom templates: {e}")
        
        return custom_templates

    def _deserialize_template(self, data: Dict[str, Any]) -> Optional[WeightTemplate]:
        """テンプレートのデシリアライゼーション"""
        try:
            # WeightAllocationConfigの復元
            config_data = data.get("config", {})
            constraints_data = config_data.get("constraints", {})
            
            constraints = PortfolioConstraints(**constraints_data)
            
            config = WeightAllocationConfig(
                method=AllocationMethod(config_data.get("method", "risk_adjusted")),
                constraints=constraints,
                rebalance_frequency=RebalanceFrequency(config_data.get("rebalance_frequency", "weekly")),
                risk_aversion=config_data.get("risk_aversion", 2.0),
                confidence_weight=config_data.get("confidence_weight", 0.3),
                trend_weight=config_data.get("trend_weight", 0.2),
                volatility_lookback=config_data.get("volatility_lookback", 252),
                enable_dynamic_adjustment=config_data.get("enable_dynamic_adjustment", True),
                enable_momentum_bias=config_data.get("enable_momentum_bias", False)
            )
            
            # WeightTemplateの復元
            template = WeightTemplate(
                name=data["name"],
                template_type=TemplateType(data["template_type"]),
                description=data["description"],
                config=config,
                suitable_market_regimes=[MarketRegime(regime) for regime in data.get("suitable_market_regimes", [])],
                risk_level=data.get("risk_level", "medium"),
                expected_turnover=data.get("expected_turnover", 0.15),
                min_capital=data.get("min_capital", 100000.0),
                tags=data.get("tags", []),
                performance_expectations=data.get("performance_expectations", {}),
                created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
            )
            
            return template
            
        except Exception as e:
            logger.error(f"Error deserializing template: {e}")
            return None

    def get_template(self, template_name: str) -> Optional[WeightTemplate]:
        """テンプレートの取得"""
        template_key = template_name.lower().replace(" ", "_")
        
        # 事前定義済みテンプレートから検索
        if template_key in self.predefined_templates:
            return self.predefined_templates[template_key]
        
        # カスタムテンプレートから検索
        if template_key in self.custom_templates:
            return self.custom_templates[template_key]
        
        logger.warning(f"Template not found: {template_name}")
        return None

    def get_all_templates(self) -> Dict[str, WeightTemplate]:
        """すべてのテンプレートを取得"""
        all_templates = {}
        all_templates.update(self.predefined_templates)
        all_templates.update(self.custom_templates)
        return all_templates

    def recommend_template(self, 
                          market_regime: MarketRegime,
                          risk_tolerance: str = "medium",
                          capital_amount: float = 100000.0) -> List[Tuple[str, WeightTemplate, float]]:
        """
        市場環境に基づくテンプレート推奨
        
        Returns:
            List[Tuple[name, template, score]]: 推奨テンプレートのリスト（スコア順）
        """
        all_templates = self.get_all_templates()
        recommendations = []
        
        for name, template in all_templates.items():
            score = self._calculate_template_score(
                template, market_regime, risk_tolerance, capital_amount
            )
            if score > 0:
                recommendations.append((name, template, score))
        
        # スコア順でソート
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} template recommendations for {market_regime.value}")
        return recommendations

    def _calculate_template_score(self,
                                template: WeightTemplate,
                                market_regime: MarketRegime,
                                risk_tolerance: str,
                                capital_amount: float) -> float:
        """テンプレートスコアの計算"""
        score = 0.0
        
        # 市場環境適合度 (40%)
        if market_regime in template.suitable_market_regimes:
            score += 0.4
        
        # リスク許容度適合度 (30%)
        risk_mapping = {
            "low": ["low", "low-medium"],
            "medium": ["low-medium", "medium", "medium-high"],
            "high": ["medium-high", "high"]
        }
        
        if template.risk_level in risk_mapping.get(risk_tolerance, ["medium"]):
            score += 0.3
        
        # 資本要件適合度 (20%)
        if capital_amount >= template.min_capital:
            capital_ratio = min(2.0, capital_amount / template.min_capital)
            score += 0.2 * min(1.0, capital_ratio / 2.0)
        
        # 追加ボーナス (10%)
        # 事前定義済みテンプレートにボーナス
        if template.name.lower().replace(" ", "_") in self.predefined_templates:
            score += 0.05
        
        # 低ターンオーバーテンプレートにボーナス
        if template.expected_turnover <= 0.10:
            score += 0.05
        
        return score

    def create_custom_template(self,
                             name: str,
                             template_type: TemplateType,
                             description: str,
                             config: WeightAllocationConfig,
                             **kwargs) -> WeightTemplate:
        """カスタムテンプレートの作成"""
        template = WeightTemplate(
            name=name,
            template_type=template_type,
            description=description,
            config=config,
            suitable_market_regimes=kwargs.get("suitable_market_regimes", []),
            risk_level=kwargs.get("risk_level", "medium"),
            expected_turnover=kwargs.get("expected_turnover", 0.15),
            min_capital=kwargs.get("min_capital", 100000.0),
            tags=kwargs.get("tags", []),
            performance_expectations=kwargs.get("performance_expectations", {})
        )
        
        # カスタムテンプレートに追加
        template_key = name.lower().replace(" ", "_")
        self.custom_templates[template_key] = template
        
        logger.info(f"Created custom template: {name}")
        return template

    def save_custom_templates(self):
        """カスタムテンプレートの保存"""
        try:
            templates_file = self.templates_dir / "custom_templates.json"
            
            # シリアライゼーション用データの準備
            templates_data = []
            for template in self.custom_templates.values():
                template_dict = asdict(template)
                
                # Enumの値を文字列に変換
                template_dict["template_type"] = template.template_type.value
                template_dict["suitable_market_regimes"] = [regime.value for regime in template.suitable_market_regimes]
                template_dict["config"]["method"] = template.config.method.value
                template_dict["config"]["rebalance_frequency"] = template.config.rebalance_frequency.value
                template_dict["created_at"] = template.created_at.isoformat()
                
                templates_data.append(template_dict)
            
            output_data = {
                "templates": templates_data,
                "saved_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(templates_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(templates_data)} custom templates to {templates_file}")
            
        except Exception as e:
            logger.error(f"Error saving custom templates: {e}")

    def get_template_summary(self) -> Dict[str, Any]:
        """テンプレート概要の取得"""
        all_templates = self.get_all_templates()
        
        summary = {
            "total_templates": len(all_templates),
            "predefined_templates": len(self.predefined_templates),
            "custom_templates": len(self.custom_templates),
            "template_types": {},
            "risk_levels": {},
            "market_regimes": {}
        }
        
        for template in all_templates.values():
            # テンプレートタイプ別集計
            template_type = template.template_type.value
            summary["template_types"][template_type] = summary["template_types"].get(template_type, 0) + 1
            
            # リスクレベル別集計
            risk_level = template.risk_level
            summary["risk_levels"][risk_level] = summary["risk_levels"].get(risk_level, 0) + 1
            
            # 市場環境別集計
            for regime in template.suitable_market_regimes:
                regime_name = regime.value
                summary["market_regimes"][regime_name] = summary["market_regimes"].get(regime_name, 0) + 1
        
        return summary

if __name__ == "__main__":
    # 簡単なテスト
    manager = WeightTemplateManager()
    
    # テンプレート一覧表示
    print("Available templates:")
    for name in manager.get_all_templates().keys():
        print(f"  - {name}")
    
    # 推奨テンプレート取得
    recommendations = manager.recommend_template(
        MarketRegime.BULL_MARKET, 
        risk_tolerance="medium", 
        capital_amount=150000.0
    )
    
    print(f"\nRecommendations for bull market:")
    for name, template, score in recommendations[:3]:
        print(f"  {name}: {score:.3f}")
