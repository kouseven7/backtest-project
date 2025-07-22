"""
Module: Backtest Scenario Generator
File: backtest_scenario_generator.py
Description: 
  4-2-2「複合戦略バックテスト機能実装」- Scenario Generator
  動的期間分割によるバックテストシナリオの生成

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 動的バックテストシナリオ生成
  - トレンド変化ベース期間分割
  - 市場環境別シナリオ作成
  - ストレステストシナリオ生成
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存システムインポート（可能な場合）
try:
    from indicators.unified_trend_detector import UnifiedTrendDetector
    TREND_DETECTOR_AVAILABLE = True
except ImportError:
    TREND_DETECTOR_AVAILABLE = False
    logging.getLogger(__name__).warning("UnifiedTrendDetector not available")

# ロガーの設定
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """市場レジーム"""
    TRENDING = "trending"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    BULL = "bull"
    BEAR = "bear"
    RECOVERY = "recovery"

class ScenarioType(Enum):
    """シナリオタイプ"""
    NORMAL_CONDITIONS = "normal_conditions"
    STRESS_TEST = "stress_test"
    REGIME_TRANSITION = "regime_transition"
    HIGH_VOLATILITY = "high_volatility"
    TREND_FOLLOWING_OPTIMAL = "trend_following_optimal"
    MEAN_REVERSION_OPTIMAL = "mean_reversion_optimal"

@dataclass
class MarketCondition:
    """市場状況"""
    regime: MarketRegime
    volatility_level: float
    trend_strength: float
    correlation_level: float
    confidence_score: float
    start_date: datetime
    end_date: datetime
    
    def duration_days(self) -> int:
        """期間の日数"""
        return (self.end_date - self.start_date).days

@dataclass
class TestScenario:
    """テストシナリオ"""
    scenario_id: str
    name: str
    scenario_type: ScenarioType
    market_conditions: List[MarketCondition]
    test_period: Tuple[datetime, datetime]
    expected_challenges: List[str]
    success_criteria: Dict[str, float]
    data_requirements: Dict[str, Any]
    performance_expectations: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    
    def total_duration_days(self) -> int:
        """総期間日数"""
        start_date, end_date = self.test_period
        return (end_date - start_date).days

@dataclass
class ScenarioGenerationResult:
    """シナリオ生成結果"""
    scenarios: List[TestScenario]
    generation_time: float
    total_scenarios: int
    market_regimes_covered: List[MarketRegime]
    generation_config: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class BacktestScenarioGenerator:
    """バックテストシナリオ生成器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """生成器の初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 設定の読み込み
        self.scenarios_config = self._load_scenarios_config(config_path)
        self.global_settings = self.scenarios_config.get('global_scenario_settings', {})
        
        # トレンド検出器の初期化（利用可能な場合）
        self.trend_detector = None
        if TREND_DETECTOR_AVAILABLE:
            try:
                self.trend_detector = UnifiedTrendDetector()
                self.logger.info("UnifiedTrendDetector integrated")
            except Exception as e:
                self.logger.warning(f"Failed to initialize UnifiedTrendDetector: {e}")
        
        # シナリオテンプレートの読み込み
        self.scenario_templates = self._load_scenario_templates()
        
        # 生成統計
        self.generation_stats = {
            "total_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "last_generation": None
        }
        
        self.logger.info("BacktestScenarioGenerator initialized")
    
    def _load_scenarios_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """シナリオ設定の読み込み"""
        
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "backtest", 
                "backtest_scenarios.json"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Scenarios config loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load scenarios config: {e}")
            return self._get_default_scenarios_config()
    
    def _get_default_scenarios_config(self) -> Dict[str, Any]:
        """デフォルトシナリオ設定"""
        
        return {
            "backtest_scenarios": [
                {
                    "scenario_id": "default_normal_test",
                    "name": "標準市場テスト",
                    "test_period": {"duration_months": 12},
                    "success_criteria": {
                        "min_sharpe_ratio": 1.0,
                        "max_drawdown": 0.15
                    }
                }
            ],
            "global_scenario_settings": {
                "data_preparation": {
                    "warmup_period_days": 20,
                    "cooldown_period_days": 5
                }
            }
        }
    
    def _load_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """シナリオテンプレートの読み込み"""
        
        scenarios = self.scenarios_config.get('backtest_scenarios', [])
        templates = {}
        
        for scenario_data in scenarios:
            scenario_id = scenario_data.get('scenario_id', 'unknown')
            templates[scenario_id] = scenario_data
        
        return templates
    
    async def generate_dynamic_scenarios(self, 
                                       base_period: Tuple[datetime, datetime],
                                       scenario_types: List[str],
                                       market_data: Optional[pd.DataFrame] = None) -> ScenarioGenerationResult:
        """動的バックテストシナリオ生成"""
        
        start_time = datetime.now()
        generated_scenarios = []
        warnings = []
        errors = []
        
        try:
            self.logger.info(f"Generating scenarios for period {base_period[0]} to {base_period[1]}")
            
            # 市場データの準備
            if market_data is None:
                market_data = self._generate_sample_market_data(base_period)
                warnings.append("Using synthetic market data")
            
            # トレンド変化期間の検出
            trend_periods = await self._detect_trend_change_periods(base_period, market_data)
            
            # 各シナリオタイプに対してシナリオを生成
            for scenario_type in scenario_types:
                try:
                    scenarios = await self._generate_scenarios_for_type(
                        scenario_type, trend_periods, base_period, market_data
                    )
                    generated_scenarios.extend(scenarios)
                    self.logger.info(f"Generated {len(scenarios)} scenarios for type {scenario_type}")
                    
                except Exception as e:
                    error_msg = f"Failed to generate scenarios for type {scenario_type}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # 統計更新
            generation_time = (datetime.now() - start_time).total_seconds()
            self.generation_stats["total_generated"] += len(generated_scenarios)
            
            if generated_scenarios:
                self.generation_stats["successful_generations"] += 1
            else:
                self.generation_stats["failed_generations"] += 1
            
            self.generation_stats["last_generation"] = datetime.now()
            
            # 市場レジーム集計
            covered_regimes = set()
            for scenario in generated_scenarios:
                for condition in scenario.market_conditions:
                    covered_regimes.add(condition.regime)
            
            result = ScenarioGenerationResult(
                scenarios=generated_scenarios,
                generation_time=generation_time,
                total_scenarios=len(generated_scenarios),
                market_regimes_covered=list(covered_regimes),
                generation_config={
                    "base_period": base_period,
                    "scenario_types": scenario_types,
                    "trend_periods_detected": len(trend_periods)
                },
                warnings=warnings,
                errors=errors
            )
            
            self.logger.info(f"Generated {len(generated_scenarios)} scenarios in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Scenario generation failed: {e}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            return ScenarioGenerationResult(
                scenarios=[],
                generation_time=(datetime.now() - start_time).total_seconds(),
                total_scenarios=0,
                market_regimes_covered=[],
                generation_config={"base_period": base_period, "scenario_types": scenario_types},
                warnings=warnings,
                errors=errors
            )
    
    async def _detect_trend_change_periods(self, 
                                         base_period: Tuple[datetime, datetime],
                                         market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """トレンド変化期間の検出"""
        
        trend_periods = []
        
        try:
            # 基本的なトレンド分析
            window = 20
            market_data = market_data.copy()
            
            # 移動平均とトレンド判定
            market_data['SMA_20'] = market_data['Close'].rolling(window=window).mean()
            market_data['SMA_50'] = market_data['Close'].rolling(window=50).mean()
            market_data['Price_Trend'] = np.where(
                market_data['Close'] > market_data['SMA_20'], 'uptrend', 'downtrend'
            )
            market_data['MA_Trend'] = np.where(
                market_data['SMA_20'] > market_data['SMA_50'], 'uptrend', 'downtrend'
            )
            
            # ボラティリティの計算
            market_data['Volatility'] = market_data['Close'].pct_change().rolling(window=window).std()
            
            # トレンド変化点の検出
            trend_changes = market_data['Price_Trend'].ne(market_data['Price_Trend'].shift()).cumsum()
            
            current_period = None
            
            for trend_id in trend_changes.unique():
                if pd.isna(trend_id):
                    continue
                
                period_data = market_data[trend_changes == trend_id]
                if len(period_data) < 5:  # 最低5日間
                    continue
                
                # 市場レジームの判定
                avg_volatility = period_data['Volatility'].mean()
                price_trend = period_data['Price_Trend'].iloc[0]
                
                regime = self._classify_market_regime(
                    trend=price_trend,
                    volatility=avg_volatility,
                    period_data=period_data
                )
                
                # トレンド強度の計算
                price_change = (period_data['Close'].iloc[-1] - period_data['Close'].iloc[0]) / period_data['Close'].iloc[0]
                trend_strength = min(abs(price_change) * 10, 1.0)  # 0-1にスケール
                
                period_info = {
                    'start': period_data.index[0],
                    'end': period_data.index[-1],
                    'regime': regime,
                    'trend': price_trend,
                    'volatility': avg_volatility,
                    'trend_strength': trend_strength,
                    'confidence': min(0.5 + trend_strength * 0.4, 0.9),  # 0.5-0.9
                    'data_points': len(period_data),
                    'price_change': price_change
                }
                
                trend_periods.append(period_info)
            
            self.logger.info(f"Detected {len(trend_periods)} trend periods")
            
        except Exception as e:
            self.logger.warning(f"Trend detection failed: {e}, using fallback method")
            # フォールバック：期間を均等分割
            start_date, end_date = base_period
            total_days = (end_date - start_date).days
            period_length = max(30, total_days // 4)  # 最低30日、最大4期間
            
            current_date = start_date
            while current_date < end_date:
                period_end = min(current_date + timedelta(days=period_length), end_date)
                
                trend_periods.append({
                    'start': current_date,
                    'end': period_end,
                    'regime': MarketRegime.SIDEWAYS,
                    'trend': 'sideways',
                    'volatility': 0.15,
                    'trend_strength': 0.5,
                    'confidence': 0.6,
                    'data_points': period_length,
                    'price_change': 0.0
                })
                
                current_date = period_end + timedelta(days=1)
        
        return trend_periods
    
    def _classify_market_regime(self, 
                              trend: str, 
                              volatility: float, 
                              period_data: pd.DataFrame) -> MarketRegime:
        """市場レジームの分類"""
        
        # ボラティリティ閾値
        HIGH_VOL = 0.03  # 日次3%以上
        LOW_VOL = 0.01   # 日次1%以下
        
        # 価格変化の分析
        price_change = (period_data['Close'].iloc[-1] - period_data['Close'].iloc[0]) / period_data['Close'].iloc[0]
        abs_change = abs(price_change)
        
        # レジーム分類ロジック
        if volatility > HIGH_VOL:
            if abs_change > 0.1:  # 10%以上の変化
                return MarketRegime.CRISIS if price_change < 0 else MarketRegime.VOLATILE
            else:
                return MarketRegime.VOLATILE
        
        elif volatility < LOW_VOL:
            if abs_change < 0.02:  # 2%以内の変化
                return MarketRegime.SIDEWAYS
            elif price_change > 0.05:  # 5%以上の上昇
                return MarketRegime.BULL
            else:
                return MarketRegime.TRENDING
        
        else:  # 中程度のボラティリティ
            if price_change > 0.05:
                return MarketRegime.BULL
            elif price_change < -0.05:
                return MarketRegime.BEAR
            else:
                return MarketRegime.TRENDING
    
    async def _generate_scenarios_for_type(self, 
                                         scenario_type: str,
                                         trend_periods: List[Dict[str, Any]],
                                         base_period: Tuple[datetime, datetime],
                                         market_data: pd.DataFrame) -> List[TestScenario]:
        """シナリオタイプ別シナリオ生成"""
        
        scenarios = []
        
        # テンプレートの取得
        template = self.scenario_templates.get(scenario_type)
        if not template:
            # デフォルトテンプレートを作成
            template = self._create_default_template(scenario_type)
        
        # シナリオタイプに応じた生成
        if scenario_type in ["trending_market_test", "normal_conditions"]:
            scenarios.extend(await self._generate_trending_scenarios(trend_periods, template))
        
        elif scenario_type in ["volatile_market_test", "high_volatility"]:
            scenarios.extend(await self._generate_volatile_scenarios(trend_periods, template))
        
        elif scenario_type in ["sideways_market_test", "mean_reversion_optimal"]:
            scenarios.extend(await self._generate_sideways_scenarios(trend_periods, template))
        
        elif scenario_type in ["crisis_market_test", "stress_test"]:
            scenarios.extend(await self._generate_crisis_scenarios(trend_periods, template))
        
        elif scenario_type in ["bull_market_test", "trend_following_optimal"]:
            scenarios.extend(await self._generate_bull_scenarios(trend_periods, template))
        
        else:
            # 汎用シナリオ生成
            scenarios.extend(await self._generate_generic_scenarios(trend_periods, template, scenario_type))
        
        return scenarios
    
    def _create_default_template(self, scenario_type: str) -> Dict[str, Any]:
        """デフォルトテンプレートの作成"""
        
        return {
            "scenario_id": scenario_type,
            "name": f"Generated {scenario_type}",
            "description": f"Auto-generated scenario for {scenario_type}",
            "success_criteria": {
                "min_sharpe_ratio": 1.0,
                "max_drawdown": 0.15,
                "min_win_rate": 0.45
            },
            "data_requirements": {
                "min_data_points": 60,
                "data_quality_threshold": 0.9
            }
        }
    
    async def _generate_trending_scenarios(self, 
                                         trend_periods: List[Dict[str, Any]], 
                                         template: Dict[str, Any]) -> List[TestScenario]:
        """トレンディング市場シナリオの生成"""
        
        scenarios = []
        trending_periods = [p for p in trend_periods if p['trend_strength'] > 0.6]
        
        for i, period in enumerate(trending_periods[:3]):  # 最大3つのシナリオ
            # 市場状況の作成
            market_condition = MarketCondition(
                regime=MarketRegime(period['regime'].value if hasattr(period['regime'], 'value') else MarketRegime.TRENDING),
                volatility_level=period['volatility'],
                trend_strength=period['trend_strength'],
                correlation_level=0.7,
                confidence_score=period['confidence'],
                start_date=period['start'],
                end_date=period['end']
            )
            
            # シナリオの作成
            scenario = TestScenario(
                scenario_id=f"trending_scenario_{i+1}_{period['start'].strftime('%Y%m%d')}",
                name=f"トレンド市場テスト {i+1}",
                scenario_type=ScenarioType.TREND_FOLLOWING_OPTIMAL,
                market_conditions=[market_condition],
                test_period=(period['start'], period['end']),
                expected_challenges=["トレンド継続性の維持", "偽のブレイクアウト回避"],
                success_criteria=template.get('success_criteria', {}),
                data_requirements=template.get('data_requirements', {}),
                performance_expectations={
                    "expected_sharpe": 1.2 * period['trend_strength'],
                    "expected_return": 0.15 * period['trend_strength'],
                    "expected_volatility": period['volatility']
                }
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_volatile_scenarios(self, 
                                         trend_periods: List[Dict[str, Any]], 
                                         template: Dict[str, Any]) -> List[TestScenario]:
        """ボラタイル市場シナリオの生成"""
        
        scenarios = []
        volatile_periods = [p for p in trend_periods if p['volatility'] > 0.025]
        
        for i, period in enumerate(volatile_periods[:2]):  # 最大2つのシナリオ
            market_condition = MarketCondition(
                regime=MarketRegime.VOLATILE,
                volatility_level=period['volatility'],
                trend_strength=period['trend_strength'],
                correlation_level=0.9,  # 高い相関
                confidence_score=period['confidence'] * 0.8,  # 信頼度を下げる
                start_date=period['start'],
                end_date=period['end']
            )
            
            scenario = TestScenario(
                scenario_id=f"volatile_scenario_{i+1}_{period['start'].strftime('%Y%m%d')}",
                name=f"ボラタイル市場テスト {i+1}",
                scenario_type=ScenarioType.HIGH_VOLATILITY,
                market_conditions=[market_condition],
                test_period=(period['start'], period['end']),
                expected_challenges=["高ボラティリティ対応", "リスク管理の重要性", "ドローダウン制御"],
                success_criteria={
                    "min_sharpe_ratio": 0.8,
                    "max_drawdown": 0.20,
                    "min_win_rate": 0.40
                },
                data_requirements=template.get('data_requirements', {}),
                performance_expectations={
                    "expected_sharpe": 0.8,
                    "expected_return": 0.10,
                    "expected_volatility": period['volatility']
                }
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_sideways_scenarios(self, 
                                         trend_periods: List[Dict[str, Any]], 
                                         template: Dict[str, Any]) -> List[TestScenario]:
        """横這い市場シナリオの生成"""
        
        scenarios = []
        sideways_periods = [p for p in trend_periods if p['trend_strength'] < 0.3 and p['volatility'] < 0.02]
        
        for i, period in enumerate(sideways_periods[:2]):  # 最大2つのシナリオ
            market_condition = MarketCondition(
                regime=MarketRegime.SIDEWAYS,
                volatility_level=period['volatility'],
                trend_strength=period['trend_strength'],
                correlation_level=0.5,  # 低い相関
                confidence_score=period['confidence'],
                start_date=period['start'],
                end_date=period['end']
            )
            
            scenario = TestScenario(
                scenario_id=f"sideways_scenario_{i+1}_{period['start'].strftime('%Y%m%d')}",
                name=f"レンジ市場テスト {i+1}",
                scenario_type=ScenarioType.MEAN_REVERSION_OPTIMAL,
                market_conditions=[market_condition],
                test_period=(period['start'], period['end']),
                expected_challenges=["明確な方向性の欠如", "レンジブレイクの見極め"],
                success_criteria={
                    "min_sharpe_ratio": 1.0,
                    "max_drawdown": 0.10,
                    "min_win_rate": 0.50
                },
                data_requirements=template.get('data_requirements', {}),
                performance_expectations={
                    "expected_sharpe": 1.0,
                    "expected_return": 0.08,
                    "expected_volatility": period['volatility']
                }
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_crisis_scenarios(self, 
                                       trend_periods: List[Dict[str, Any]], 
                                       template: Dict[str, Any]) -> List[TestScenario]:
        """クライシス市場シナリオの生成"""
        
        scenarios = []
        crisis_periods = [p for p in trend_periods if p['volatility'] > 0.04 or p['price_change'] < -0.1]
        
        for i, period in enumerate(crisis_periods[:1]):  # 最大1つのシナリオ
            market_condition = MarketCondition(
                regime=MarketRegime.CRISIS,
                volatility_level=max(period['volatility'], 0.05),  # 最低5%
                trend_strength=period['trend_strength'],
                correlation_level=0.95,  # 非常に高い相関
                confidence_score=period['confidence'] * 0.6,  # 信頼度をさらに下げる
                start_date=period['start'],
                end_date=period['end']
            )
            
            scenario = TestScenario(
                scenario_id=f"crisis_scenario_{i+1}_{period['start'].strftime('%Y%m%d')}",
                name=f"クライシス市場テスト {i+1}",
                scenario_type=ScenarioType.STRESS_TEST,
                market_conditions=[market_condition],
                test_period=(period['start'], period['end']),
                expected_challenges=["資本保護", "急激な価格変動", "流動性リスク"],
                success_criteria={
                    "min_sharpe_ratio": 0.5,
                    "max_drawdown": 0.25,
                    "capital_preservation_rate": 0.75
                },
                data_requirements=template.get('data_requirements', {}),
                performance_expectations={
                    "expected_sharpe": 0.5,
                    "expected_return": 0.05,
                    "expected_volatility": max(period['volatility'], 0.05)
                }
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_bull_scenarios(self, 
                                     trend_periods: List[Dict[str, Any]], 
                                     template: Dict[str, Any]) -> List[TestScenario]:
        """強気市場シナリオの生成"""
        
        scenarios = []
        bull_periods = [p for p in trend_periods if p['price_change'] > 0.05 and p['trend'] == 'uptrend']
        
        for i, period in enumerate(bull_periods[:2]):  # 最大2つのシナリオ
            market_condition = MarketCondition(
                regime=MarketRegime.BULL,
                volatility_level=period['volatility'],
                trend_strength=period['trend_strength'],
                correlation_level=0.8,
                confidence_score=period['confidence'],
                start_date=period['start'],
                end_date=period['end']
            )
            
            scenario = TestScenario(
                scenario_id=f"bull_scenario_{i+1}_{period['start'].strftime('%Y%m%d')}",
                name=f"強気市場テスト {i+1}",
                scenario_type=ScenarioType.TREND_FOLLOWING_OPTIMAL,
                market_conditions=[market_condition],
                test_period=(period['start'], period['end']),
                expected_challenges=["トレンド継続の最大化", "利益確定のタイミング"],
                success_criteria={
                    "min_sharpe_ratio": 1.5,
                    "max_drawdown": 0.08,
                    "min_win_rate": 0.55
                },
                data_requirements=template.get('data_requirements', {}),
                performance_expectations={
                    "expected_sharpe": 1.5,
                    "expected_return": 0.20,
                    "expected_volatility": period['volatility']
                }
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_generic_scenarios(self, 
                                        trend_periods: List[Dict[str, Any]], 
                                        template: Dict[str, Any],
                                        scenario_type: str) -> List[TestScenario]:
        """汎用シナリオの生成"""
        
        scenarios = []
        
        if not trend_periods:
            return scenarios
        
        # 代表的な期間を選択
        representative_period = trend_periods[len(trend_periods)//2]  # 中央の期間
        
        market_condition = MarketCondition(
            regime=MarketRegime.TRENDING,  # デフォルト
            volatility_level=representative_period['volatility'],
            trend_strength=representative_period['trend_strength'],
            correlation_level=0.7,
            confidence_score=representative_period['confidence'],
            start_date=representative_period['start'],
            end_date=representative_period['end']
        )
        
        scenario = TestScenario(
            scenario_id=f"generic_scenario_{scenario_type}_{representative_period['start'].strftime('%Y%m%d')}",
            name=f"汎用テスト {scenario_type}",
            scenario_type=ScenarioType.NORMAL_CONDITIONS,
            market_conditions=[market_condition],
            test_period=(representative_period['start'], representative_period['end']),
            expected_challenges=["汎用的な市場対応"],
            success_criteria=template.get('success_criteria', {}),
            data_requirements=template.get('data_requirements', {}),
            performance_expectations={
                "expected_sharpe": 1.0,
                "expected_return": 0.10,
                "expected_volatility": representative_period['volatility']
            }
        )
        
        scenarios.append(scenario)
        
        return scenarios
    
    def _generate_sample_market_data(self, test_period: Tuple[datetime, datetime]) -> pd.DataFrame:
        """サンプル市場データの生成"""
        
        start_date, end_date = test_period
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        n_days = len(date_range)
        np.random.seed(42)
        
        # 基本価格生成
        initial_price = 100.0
        daily_returns = np.random.normal(0.0005, 0.02, n_days)
        cumulative_returns = np.cumsum(daily_returns)
        prices = initial_price * np.exp(cumulative_returns)
        
        # OHLCV データ
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        opens = np.concatenate([[prices[0]], prices[:-1] * (1 + np.random.normal(0, 0.005, n_days-1))])
        volumes = np.random.lognormal(10, 0.5, n_days)
        
        return pd.DataFrame({
            'Date': date_range,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }).set_index('Date')
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """生成統計の取得"""
        
        return self.generation_stats.copy()
    
    def validate_scenario(self, scenario: TestScenario) -> List[str]:
        """シナリオの検証"""
        
        errors = []
        
        # 基本検証
        if not scenario.market_conditions:
            errors.append("No market conditions defined")
        
        # 期間検証
        if scenario.total_duration_days() < 5:
            errors.append("Scenario duration too short (< 5 days)")
        
        # 市場条件検証
        for condition in scenario.market_conditions:
            if condition.duration_days() < 1:
                errors.append(f"Market condition duration too short: {condition.regime}")
            
            if not 0.0 <= condition.confidence_score <= 1.0:
                errors.append(f"Invalid confidence score: {condition.confidence_score}")
        
        return errors
    
    def export_scenarios(self, scenarios: List[TestScenario], output_path: str) -> bool:
        """シナリオのエクスポート"""
        
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_scenarios": len(scenarios),
                "generation_stats": self.generation_stats,
                "scenarios": [asdict(scenario) for scenario in scenarios]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"Scenarios exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export scenarios: {e}")
            return False

# テスト関数
async def test_backtest_scenario_generator():
    """テスト関数"""
    logger.info("Testing BacktestScenarioGenerator")
    
    # 生成器の初期化
    generator = BacktestScenarioGenerator()
    
    # テスト期間
    test_period = (
        datetime.now() - timedelta(days=365),
        datetime.now() - timedelta(days=1)
    )
    
    # シナリオタイプ
    scenario_types = [
        "trending_market_test",
        "volatile_market_test", 
        "sideways_market_test"
    ]
    
    # シナリオ生成
    result = await generator.generate_dynamic_scenarios(
        base_period=test_period,
        scenario_types=scenario_types
    )
    
    logger.info(f"Generated {result.total_scenarios} scenarios")
    logger.info(f"Market regimes covered: {result.market_regimes_covered}")
    logger.info(f"Generation time: {result.generation_time:.2f}s")
    
    if result.warnings:
        logger.warning(f"Warnings: {result.warnings}")
    
    if result.errors:
        logger.error(f"Errors: {result.errors}")
    
    # 統計表示
    stats = generator.get_generation_stats()
    logger.info(f"Generation stats: {stats}")
    
    return result

if __name__ == "__main__":
    # テスト実行
    import asyncio
    result = asyncio.run(test_backtest_scenario_generator())
    print(f"Test completed: {result.total_scenarios} scenarios generated")
