"""
Module: Portfolio Weight Pattern Engine V2
File: portfolio_weight_pattern_engine_v2.py
Description: 
  3-2-3「重み付けパターンテンプレート作成」
  型エラーを修正したエラーフリー実装版
  リスク許容度と市場環境に基づく動的テンプレート管理システム

Author: imega
Created: 2025-07-15
Modified: 2025-07-15

Dependencies:
  - config.portfolio_weight_calculator
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ロガーの設定
logger = logging.getLogger(__name__)

class RiskTolerance(Enum):
    """リスク許容度レベル"""
    CONSERVATIVE = "conservative"      # 保守的
    MODERATE = "moderate"             # 中庸
    BALANCED = "balanced"             # バランス型
    AGGRESSIVE = "aggressive"         # 積極的

class MarketEnvironment(Enum):
    """市場環境タイプ"""
    BULL = "bull"                     # 上昇相場
    BEAR = "bear"                     # 下降相場
    SIDEWAYS = "sideways"             # 横ばい相場
    VOLATILE = "volatile"             # 高ボラティリティ
    RECOVERY = "recovery"             # 回復期
    CRISIS = "crisis"                 # 危機的状況

class TemplateCategory(Enum):
    """テンプレートカテゴリ"""
    RISK_BASED = "risk_based"         # リスクベース
    MARKET_BASED = "market_based"     # 市場ベース
    HYBRID = "hybrid"                 # ハイブリッド（リスク+市場）
    TACTICAL = "tactical"             # 戦術的
    STRATEGIC = "strategic"           # 戦略的

@dataclass
class DynamicAdjustmentConfig:
    """動的調整設定"""
    enable_volatility_adjustment: bool = True
    enable_trend_adjustment: bool = True
    enable_momentum_adjustment: bool = False
    volatility_sensitivity: float = 1.0
    trend_sensitivity: float = 0.8
    momentum_sensitivity: float = 0.5
    max_adjustment_per_period: float = 0.2
    min_adjustment_threshold: float = 0.05
    volatility_threshold_high: float = 0.3
    volatility_threshold_low: float = 0.15
    trend_strength_threshold: float = 0.6

@dataclass
class PatternTemplate:
    """ポートフォリオ重み配分テンプレート（エラーフリー版）"""
    name: str
    category: TemplateCategory
    risk_tolerance: RiskTolerance
    market_environment: Optional[MarketEnvironment] = None
    
    # 基本配分設定
    allocation_method: str = "risk_adjusted"
    max_strategies: int = 5
    min_strategies: int = 2
    
    # 重み制約
    max_individual_weight: float = 0.4
    min_individual_weight: float = 0.05
    concentration_limit: float = 0.6
    
    # 3-2-2機能設定
    enable_hierarchical_weights: bool = True
    weight_adjustment_method: str = "proportional"
    enable_conditional_exclusion: bool = True
    
    # 動的調整パラメータ
    volatility_adjustment_factor: float = 1.0
    trend_sensitivity: float = 0.5
    momentum_bias: float = 0.0
    
    # 戦略カテゴリ別重み設定（明示的型指定）
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 0.3,
        'mean_reversion': 0.3,
        'trend_following': 0.2,
        'volatility': 0.2
    })
    category_min_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 0.1,
        'mean_reversion': 0.1,
        'trend_following': 0.1,
        'volatility': 0.1
    })
    
    # メタデータ
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    performance_target: Optional[float] = None
    risk_target: Optional[float] = None

class AdvancedPatternEngineV2:
    """
    3-2-3「重み付けパターンテンプレート作成」エンジン
    エラーフリー実装版
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """初期化"""
        # ベースディレクトリの設定
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), "portfolio_weight_patterns")
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイルパス設定
        self.templates_file = self.base_dir / "pattern_templates.json"
        self.dynamic_config_file = self.base_dir / "dynamic_adjustment_config.json"
        
        # テンプレート保存
        self.templates: Dict[str, PatternTemplate] = {}
        
        # 動的調整設定
        self.dynamic_config = DynamicAdjustmentConfig()
        
        # 市場環境判定履歴
        self.market_environment_history: List[Tuple[datetime, MarketEnvironment]] = []
        
        # 初期化
        self._initialize_default_templates()
        self._load_templates()
        self._load_dynamic_config()
        
        logger.info("AdvancedPatternEngineV2 initialized with 3-2-3 functionality")
    
    def _initialize_default_templates(self) -> None:
        """デフォルトテンプレートセットの初期化"""
        templates = {}
        
        # 1. リスクベーステンプレート群
        
        # 保守的テンプレート
        templates['conservative_stable'] = PatternTemplate(
            name='conservative_stable',
            category=TemplateCategory.RISK_BASED,
            risk_tolerance=RiskTolerance.CONSERVATIVE,
            allocation_method='equal_weight',
            max_strategies=3,
            min_strategies=2,
            max_individual_weight=0.4,
            min_individual_weight=0.2,
            concentration_limit=0.6,
            enable_hierarchical_weights=True,
            weight_adjustment_method='gradual',
            volatility_adjustment_factor=0.8,
            trend_sensitivity=0.3,
            momentum_bias=-0.1,
            description='保守的投資家向けの安定重視テンプレート'
        )
        
        # バランス型テンプレート
        templates['balanced_flexible'] = PatternTemplate(
            name='balanced_flexible',
            category=TemplateCategory.RISK_BASED,
            risk_tolerance=RiskTolerance.BALANCED,
            allocation_method='optimal',
            max_strategies=5,
            min_strategies=3,
            max_individual_weight=0.6,
            min_individual_weight=0.1,
            concentration_limit=0.8,
            enable_hierarchical_weights=True,
            weight_adjustment_method='adaptive',
            volatility_adjustment_factor=1.1,
            trend_sensitivity=0.6,
            momentum_bias=0.1,
            description='バランス投資家向けの柔軟性重視テンプレート'
        )
        
        # 積極的テンプレート
        templates['aggressive_growth'] = PatternTemplate(
            name='aggressive_growth',
            category=TemplateCategory.RISK_BASED,
            risk_tolerance=RiskTolerance.AGGRESSIVE,
            allocation_method='momentum_weighted',
            max_strategies=6,
            min_strategies=3,
            max_individual_weight=0.8,
            min_individual_weight=0.05,
            concentration_limit=0.9,
            enable_hierarchical_weights=False,
            weight_adjustment_method='aggressive',
            volatility_adjustment_factor=1.3,
            trend_sensitivity=0.8,
            momentum_bias=0.3,
            description='積極的投資家向けの成長重視テンプレート'
        )
        
        # 2. 市場ベーステンプレート群
        
        # 上昇相場用
        templates['bull_market_momentum'] = PatternTemplate(
            name='bull_market_momentum',
            category=TemplateCategory.MARKET_BASED,
            risk_tolerance=RiskTolerance.BALANCED,
            market_environment=MarketEnvironment.BULL,
            allocation_method='momentum_weighted',
            max_strategies=5,
            min_strategies=3,
            max_individual_weight=0.7,
            min_individual_weight=0.1,
            volatility_adjustment_factor=1.2,
            trend_sensitivity=0.7,
            momentum_bias=0.2,
            description='上昇相場向けモメンタム重視テンプレート'
        )
        
        # 3. ハイブリッドテンプレート群
        
        # 保守的 × 上昇相場
        templates['conservative_bull_hybrid'] = PatternTemplate(
            name='conservative_bull_hybrid',
            category=TemplateCategory.HYBRID,
            risk_tolerance=RiskTolerance.CONSERVATIVE,
            market_environment=MarketEnvironment.BULL,
            allocation_method='risk_adjusted',
            max_strategies=4,
            min_strategies=3,
            max_individual_weight=0.45,
            min_individual_weight=0.15,
            volatility_adjustment_factor=0.9,
            trend_sensitivity=0.5,
            momentum_bias=0.1,
            description='保守的投資家向け上昇相場対応テンプレート'
        )
        
        # テンプレートを設定
        self.templates.update(templates)

    def detect_market_environment(self, market_data: pd.DataFrame, 
                                 lookback_days: int = 60) -> MarketEnvironment:
        """
        市場環境の自動判定
        ボラティリティ、トレンド、モメンタムを分析して市場状況を判定
        """
        try:
            if market_data.empty or len(market_data) < lookback_days:
                return MarketEnvironment.SIDEWAYS
            
            # 価格データの取得（型エラー回避）
            if 'close' in market_data.columns:
                prices = market_data['close'].tail(lookback_days).astype(float)
            else:
                logger.warning("No 'close' column found, using first available price column")
                prices = market_data.iloc[:, 0].tail(lookback_days).astype(float)
            
            # リターン計算
            returns = prices.pct_change().dropna()
            
            if len(returns) < 20:
                return MarketEnvironment.SIDEWAYS
            
            # 1. ボラティリティ分析
            volatility = float(returns.std() * np.sqrt(252))
            
            # 2. トレンド分析
            trend_strength = self._calculate_trend_strength(prices)
            
            # 3. リターン分析
            cumulative_return = float((prices.iloc[-1] / prices.iloc[0]) - 1)
            
            # 4. 市場環境判定ロジック
            if volatility > self.dynamic_config.volatility_threshold_high:
                if abs(cumulative_return) < 0.05:  # 高ボラで横ばい
                    return MarketEnvironment.VOLATILE
                elif cumulative_return < -0.15:  # 高ボラで下落
                    return MarketEnvironment.CRISIS
                else:
                    return MarketEnvironment.VOLATILE
            
            elif volatility < self.dynamic_config.volatility_threshold_low:
                if trend_strength > self.dynamic_config.trend_strength_threshold:
                    return MarketEnvironment.BULL if cumulative_return > 0 else MarketEnvironment.BEAR
                else:
                    return MarketEnvironment.SIDEWAYS
            
            else:  # 中程度のボラティリティ
                if cumulative_return > 0.1:
                    return MarketEnvironment.BULL
                elif cumulative_return < -0.1:
                    return MarketEnvironment.BEAR
                elif cumulative_return > -0.05 and trend_strength > 0.3:
                    return MarketEnvironment.RECOVERY
                else:
                    return MarketEnvironment.SIDEWAYS
            
        except Exception as e:
            logger.error(f"Error detecting market environment: {e}")
            return MarketEnvironment.SIDEWAYS
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """トレンド強度の計算（型安全版）"""
        try:
            # 価格データを明示的にfloatに変換
            price_values = prices.astype(float)
            
            # 複数期間の移動平均でトレンド強度を計算
            sma_5 = price_values.rolling(5).mean()
            sma_20 = price_values.rolling(20).mean()
            sma_60 = price_values.rolling(60).mean() if len(price_values) >= 60 else sma_20
            
            # トレンド方向の一致度
            current_price = float(price_values.iloc[-1])
            sma_5_last = float(sma_5.iloc[-1])
            sma_20_last = float(sma_20.iloc[-1])
            sma_60_last = float(sma_60.iloc[-1])
            
            trend_consistency = 0.0
            
            if current_price > sma_5_last > sma_20_last > sma_60_last:
                trend_consistency = 1.0  # 強い上昇トレンド
            elif current_price < sma_5_last < sma_20_last < sma_60_last:
                trend_consistency = 1.0  # 強い下降トレンド
            elif current_price > sma_5_last > sma_20_last:
                trend_consistency = 0.7  # 中程度の上昇トレンド
            elif current_price < sma_5_last < sma_20_last:
                trend_consistency = 0.7  # 中程度の下降トレンド
            else:
                trend_consistency = 0.3  # 弱いトレンド
            
            return trend_consistency
            
        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            return 0.5

    def recommend_template(self, 
                          risk_tolerance: RiskTolerance,
                          market_data: Optional[pd.DataFrame] = None,
                          performance_target: Optional[float] = None,
                          current_portfolio: Optional[Dict[str, float]] = None) -> PatternTemplate:
        """
        リスク許容度と市場環境に基づくテンプレート推奨
        """
        try:
            # 市場環境の判定
            market_env = MarketEnvironment.SIDEWAYS
            if market_data is not None and not market_data.empty:
                market_env = self.detect_market_environment(market_data)
                # 履歴に追加
                self.market_environment_history.append((datetime.now(), market_env))
            
            # ハイブリッドテンプレートを優先検索
            hybrid_templates = [t for t in self.templates.values() 
                              if t.category == TemplateCategory.HYBRID 
                              and t.risk_tolerance == risk_tolerance 
                              and t.market_environment == market_env]
            
            if hybrid_templates:
                recommended = hybrid_templates[0]
                logger.info(f"Recommended hybrid template: {recommended.name} for {risk_tolerance.value} + {market_env.value}")
                return recommended
            
            # リスクベーステンプレートの検索
            risk_templates = [t for t in self.templates.values() 
                            if t.category == TemplateCategory.RISK_BASED 
                            and t.risk_tolerance == risk_tolerance]
            
            if risk_templates:
                recommended = risk_templates[0]
                logger.info(f"Recommended risk-based template: {recommended.name} for {risk_tolerance.value}")
                return recommended
            
            # 市場ベーステンプレートの検索
            market_templates = [t for t in self.templates.values() 
                              if t.category == TemplateCategory.MARKET_BASED 
                              and t.market_environment == market_env]
            
            if market_templates:
                recommended = market_templates[0]
                logger.info(f"Recommended market-based template: {recommended.name} for {market_env.value}")
                return recommended
            
            # フォールバック：バランス型テンプレート
            balanced_templates = [t for t in self.templates.values() 
                                if t.risk_tolerance == RiskTolerance.BALANCED]
            
            if balanced_templates:
                recommended = balanced_templates[0]
                logger.info(f"Fallback to balanced template: {recommended.name}")
                return recommended
            
            # 最終フォールバック：最初のテンプレート
            if self.templates:
                recommended = list(self.templates.values())[0]
                logger.warning(f"Using first available template: {recommended.name}")
                return recommended
            
            raise ValueError("No templates available")
            
        except Exception as e:
            logger.error(f"Error recommending template: {e}")
            # エラー時のデフォルトテンプレート
            return self._create_emergency_template(risk_tolerance)

    def _create_emergency_template(self, risk_tolerance: RiskTolerance) -> PatternTemplate:
        """エラー時の緊急テンプレート作成"""
        return PatternTemplate(
            name=f"emergency_{risk_tolerance.value}",
            category=TemplateCategory.RISK_BASED,
            risk_tolerance=risk_tolerance,
            allocation_method="equal_weight",
            max_strategies=3,
            min_strategies=2,
            max_individual_weight=0.5,
            min_individual_weight=0.1,
            description=f"Emergency template for {risk_tolerance.value}"
        )

    def create_custom_template(self, 
                             name: str,
                             risk_tolerance: RiskTolerance,
                             market_environment: Optional[MarketEnvironment] = None,
                             custom_settings: Optional[Dict[str, Any]] = None) -> PatternTemplate:
        """カスタムテンプレートの作成"""
        try:
            # ベーステンプレートの選択
            base_template = self.recommend_template(risk_tolerance)
            
            # カスタムテンプレート作成
            custom_template = PatternTemplate(
                name=name,
                category=TemplateCategory.TACTICAL,
                risk_tolerance=risk_tolerance,
                market_environment=market_environment,
                allocation_method=base_template.allocation_method,
                max_strategies=base_template.max_strategies,
                min_strategies=base_template.min_strategies,
                max_individual_weight=base_template.max_individual_weight,
                min_individual_weight=base_template.min_individual_weight,
                concentration_limit=base_template.concentration_limit,
                enable_hierarchical_weights=base_template.enable_hierarchical_weights,
                weight_adjustment_method=base_template.weight_adjustment_method,
                enable_conditional_exclusion=base_template.enable_conditional_exclusion,
                volatility_adjustment_factor=base_template.volatility_adjustment_factor,
                trend_sensitivity=base_template.trend_sensitivity,
                momentum_bias=base_template.momentum_bias,
                category_weights=base_template.category_weights.copy(),
                category_min_weights=base_template.category_min_weights.copy(),
                description=f"Custom template based on {base_template.name}"
            )
            
            # カスタム設定の適用
            if custom_settings:
                for key, value in custom_settings.items():
                    if hasattr(custom_template, key):
                        setattr(custom_template, key, value)
            
            # テンプレートの登録
            self.templates[name] = custom_template
            self._save_templates()
            
            logger.info(f"Created custom template: {name}")
            return custom_template
            
        except Exception as e:
            logger.error(f"Error creating custom template: {e}")
            raise

    def get_template_by_name(self, name: str) -> Optional[PatternTemplate]:
        """テンプレート名での取得"""
        return self.templates.get(name)

    def list_templates(self, 
                      risk_tolerance: Optional[RiskTolerance] = None,
                      market_environment: Optional[MarketEnvironment] = None,
                      category: Optional[TemplateCategory] = None) -> List[PatternTemplate]:
        """テンプレート一覧の取得（フィルタ可能）"""
        templates = list(self.templates.values())
        
        if risk_tolerance:
            templates = [t for t in templates if t.risk_tolerance == risk_tolerance]
        
        if market_environment:
            templates = [t for t in templates if t.market_environment == market_environment]
            
        if category:
            templates = [t for t in templates if t.category == category]
            
        return templates

    def _load_templates(self):
        """テンプレートファイルの読み込み"""
        if not self.templates_file.exists():
            self._save_templates()  # デフォルトテンプレートを保存
            return
        
        try:
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for template_data in data.get('templates', []):
                template = self._dict_to_template(template_data)
                self.templates[template.name] = template
                
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")

    def _save_templates(self):
        """テンプレートファイルの保存"""
        try:
            data = {
                'templates': [self._template_to_dict(template) for template in self.templates.values()],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")

    def _load_dynamic_config(self):
        """動的調整設定の読み込み"""
        if not self.dynamic_config_file.exists():
            self._save_dynamic_config()
            return
        
        try:
            with open(self.dynamic_config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 設定の更新
            for key, value in data.items():
                if hasattr(self.dynamic_config, key):
                    setattr(self.dynamic_config, key, value)
                    
        except Exception as e:
            logger.error(f"Failed to load dynamic config: {e}")

    def _save_dynamic_config(self):
        """動的調整設定の保存"""
        try:
            config_dict = {
                'enable_volatility_adjustment': self.dynamic_config.enable_volatility_adjustment,
                'enable_trend_adjustment': self.dynamic_config.enable_trend_adjustment,
                'enable_momentum_adjustment': self.dynamic_config.enable_momentum_adjustment,
                'volatility_sensitivity': self.dynamic_config.volatility_sensitivity,
                'trend_sensitivity': self.dynamic_config.trend_sensitivity,
                'momentum_sensitivity': self.dynamic_config.momentum_sensitivity,
                'max_adjustment_per_period': self.dynamic_config.max_adjustment_per_period,
                'min_adjustment_threshold': self.dynamic_config.min_adjustment_threshold,
                'volatility_threshold_high': self.dynamic_config.volatility_threshold_high,
                'volatility_threshold_low': self.dynamic_config.volatility_threshold_low,
                'trend_strength_threshold': self.dynamic_config.trend_strength_threshold
            }
            
            with open(self.dynamic_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save dynamic config: {e}")

    def _template_to_dict(self, template: PatternTemplate) -> Dict[str, Any]:
        """テンプレートオブジェクトを辞書に変換"""
        return {
            'name': template.name,
            'category': template.category.value,
            'risk_tolerance': template.risk_tolerance.value,
            'market_environment': template.market_environment.value if template.market_environment else None,
            'allocation_method': template.allocation_method,
            'max_strategies': template.max_strategies,
            'min_strategies': template.min_strategies,
            'max_individual_weight': template.max_individual_weight,
            'min_individual_weight': template.min_individual_weight,
            'concentration_limit': template.concentration_limit,
            'enable_hierarchical_weights': template.enable_hierarchical_weights,
            'weight_adjustment_method': template.weight_adjustment_method,
            'enable_conditional_exclusion': template.enable_conditional_exclusion,
            'volatility_adjustment_factor': template.volatility_adjustment_factor,
            'trend_sensitivity': template.trend_sensitivity,
            'momentum_bias': template.momentum_bias,
            'category_weights': template.category_weights,
            'category_min_weights': template.category_min_weights,
            'description': template.description,
            'created_at': template.created_at.isoformat(),
            'is_active': template.is_active,
            'performance_target': template.performance_target,
            'risk_target': template.risk_target
        }

    def _dict_to_template(self, data: Dict[str, Any]) -> PatternTemplate:
        """辞書からテンプレートオブジェクトに変換"""
        return PatternTemplate(
            name=data['name'],
            category=TemplateCategory(data['category']),
            risk_tolerance=RiskTolerance(data['risk_tolerance']),
            market_environment=MarketEnvironment(data['market_environment']) if data.get('market_environment') else None,
            allocation_method=data['allocation_method'],
            max_strategies=data['max_strategies'],
            min_strategies=data['min_strategies'],
            max_individual_weight=data['max_individual_weight'],
            min_individual_weight=data['min_individual_weight'],
            concentration_limit=data['concentration_limit'],
            enable_hierarchical_weights=data['enable_hierarchical_weights'],
            weight_adjustment_method=data['weight_adjustment_method'],
            enable_conditional_exclusion=data['enable_conditional_exclusion'],
            volatility_adjustment_factor=data['volatility_adjustment_factor'],
            trend_sensitivity=data['trend_sensitivity'],
            momentum_bias=data['momentum_bias'],
            category_weights=data['category_weights'],
            category_min_weights=data['category_min_weights'],
            description=data['description'],
            created_at=datetime.fromisoformat(data['created_at']),
            is_active=data.get('is_active', True),
            performance_target=data.get('performance_target'),
            risk_target=data.get('risk_target')
        )

# 便利関数の定義
def create_pattern_engine(base_dir: Optional[str] = None) -> AdvancedPatternEngineV2:
    """パターンエンジンのファクトリ関数"""
    return AdvancedPatternEngineV2(base_dir)

def quick_template_recommendation(risk_tolerance: str, 
                                market_data: Optional[pd.DataFrame] = None) -> PatternTemplate:
    """クイックテンプレート推奨"""
    engine = AdvancedPatternEngineV2()
    risk_enum = RiskTolerance(risk_tolerance.lower())
    return engine.recommend_template(risk_enum, market_data)

if __name__ == "__main__":
    # 基本動作テスト
    print("=== 3-2-3 Portfolio Weight Pattern Engine V2 Demo ===")
    
    # エンジンの初期化
    engine = AdvancedPatternEngineV2()
    
    # テンプレート一覧表示
    templates = engine.list_templates()
    print(f"\n利用可能なテンプレート数: {len(templates)}")
    
    for template in templates:
        print(f"- {template.name}: {template.description}")
    
    # テンプレート推奨テスト
    print("\n=== テンプレート推奨テスト ===")
    
    for risk in RiskTolerance:
        recommended = engine.recommend_template(risk)
        print(f"{risk.value}: {recommended.name}")
    
    print("\n3-2-3 パターンエンジン実装完了！")
