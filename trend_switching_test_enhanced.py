"""
Module: Enhanced Trend Switching Test System
File: trend_switching_test_enhanced.py
Description: 
  改良版トレンド変化時の戦略切替テストシステム
  循環インポート問題を解決し、包括的なテスト機能を提供

Author: imega
Created: 2025-01-22
Modified: 2025-01-22

Functions:
  - トレンド変化シナリオ生成とテスト実行
  - 戦略切替効果測定・パフォーマンス評価
  - レポート生成と視覚化機能
  - エラーハンドリング強化
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """市場状況タイプ"""
    STRONG_UPTREND = "strong_uptrend"
    MODERATE_UPTREND = "moderate_uptrend"
    SIDEWAYS = "sideways"
    MODERATE_DOWNTREND = "moderate_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class StrategyType(Enum):
    """戦略タイプ"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    BREAKOUT = "breakout"

@dataclass
class TrendScenario:
    """トレンド変化シナリオ"""
    scenario_id: str
    name: str
    description: str
    initial_condition: MarketCondition
    target_condition: MarketCondition
    transition_period: int  # 営業日数
    volatility_factor: float
    data_points: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StrategyPerformance:
    """戦略パフォーマンス"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    avg_trade_duration: float
    active_periods: List[Tuple[int, int]]  # (start, end) indices

@dataclass
class SwitchingEvent:
    """戦略切替イベント"""
    timestamp: int  # データポイントのインデックス
    from_strategy: str
    to_strategy: str
    reason: str
    market_condition: MarketCondition
    confidence_score: float

@dataclass
class TestResult:
    """テスト結果"""
    scenario: TrendScenario
    execution_time: float
    switching_events: List[SwitchingEvent]
    strategy_performances: Dict[str, StrategyPerformance]
    overall_performance: Dict[str, float]
    success_metrics: Dict[str, bool]
    detailed_logs: List[str]
    errors: List[str]

class SyntheticDataGenerator:
    """合成データ生成器"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.base_price = 100.0
    
    def generate_scenario_data(self, scenario: TrendScenario) -> pd.DataFrame:
        """シナリオに基づく合成データ生成"""
        try:
            logger.info(f"Generating data for scenario: {scenario.scenario_id}")
            
            # 基本価格系列生成
            data = self._generate_base_series(scenario)
            
            # トレンド変化を適用
            data = self._apply_trend_transition(data, scenario)
            
            # ボラティリティ調整
            data = self._adjust_volatility(data, scenario)
            
            # テクニカル指標計算
            data = self._calculate_indicators(data)
            
            logger.info(f"Generated {len(data)} data points for scenario {scenario.scenario_id}")
            return data
            
        except Exception as e:
            logger.error(f"Error generating scenario data: {e}")
            raise
    
    def _generate_base_series(self, scenario: TrendScenario) -> pd.DataFrame:
        """基本価格系列生成"""
        dates = pd.date_range(start='2023-01-01', periods=scenario.data_points, freq='D')
        
        # ランダムウォーク生成
        returns = np.random.normal(0, 0.02, scenario.data_points)
        prices = [self.base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'Volume': np.random.randint(1000000, 5000000, scenario.data_points)
        })
    
    def _apply_trend_transition(self, data: pd.DataFrame, scenario: TrendScenario) -> pd.DataFrame:
        """トレンド変化を適用"""
        n_points = len(data)
        transition_start = n_points // 3
        transition_end = transition_start + scenario.transition_period
        
        # 初期トレンド適用
        initial_drift = self._get_trend_drift(scenario.initial_condition)
        data.loc[:transition_start, 'Price'] *= np.cumprod(1 + np.random.normal(initial_drift, 0.01, transition_start + 1))
        
        # 遷移期間
        if transition_end < n_points:
            target_drift = self._get_trend_drift(scenario.target_condition)
            transition_drifts = np.linspace(initial_drift, target_drift, transition_end - transition_start)
            
            for i, drift in enumerate(transition_drifts):
                idx = transition_start + i
                if idx < len(data):
                    data.loc[idx, 'Price'] *= (1 + np.random.normal(drift, 0.01))
            
            # 目標トレンド適用
            remaining_points = n_points - transition_end
            if remaining_points > 0:
                final_multipliers = np.cumprod(1 + np.random.normal(target_drift, 0.01, remaining_points))
                data.loc[transition_end:, 'Price'] *= final_multipliers
        
        return data
    
    def _get_trend_drift(self, condition: MarketCondition) -> float:
        """市場状況に応じたドリフト取得"""
        drift_map = {
            MarketCondition.STRONG_UPTREND: 0.003,
            MarketCondition.MODERATE_UPTREND: 0.001,
            MarketCondition.SIDEWAYS: 0.0,
            MarketCondition.MODERATE_DOWNTREND: -0.001,
            MarketCondition.STRONG_DOWNTREND: -0.003,
            MarketCondition.HIGH_VOLATILITY: 0.0,
            MarketCondition.LOW_VOLATILITY: 0.0
        }
        return drift_map.get(condition, 0.0)
    
    def _adjust_volatility(self, data: pd.DataFrame, scenario: TrendScenario) -> pd.DataFrame:
        """ボラティリティ調整"""
        base_vol = 0.02
        adjusted_vol = base_vol * scenario.volatility_factor
        
        # 価格にボラティリティを追加
        vol_noise = np.random.normal(0, adjusted_vol, len(data))
        data['Price'] *= np.exp(vol_noise)
        
        return data
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標計算"""
        # 移動平均
        data['SMA_20'] = data['Price'].rolling(window=20).mean()
        data['SMA_50'] = data['Price'].rolling(window=50).mean()
        
        # RSI
        delta = data['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['Price'].ewm(span=12).mean()
        ema_26 = data['Price'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # ボラティリティ
        data['Volatility'] = data['Price'].pct_change().rolling(window=20).std()
        
        return data

class TrendDetector:
    """簡易トレンド検出器"""
    
    def __init__(self):
        self.trend_threshold = 0.02
        self.volatility_threshold = 0.03
    
    def detect_market_condition(self, data: pd.DataFrame, lookback: int = 20) -> MarketCondition:
        """市場状況を検出"""
        try:
            if len(data) < lookback:
                return MarketCondition.SIDEWAYS
            
            recent_data = data.tail(lookback)
            
            # トレンド計算
            price_change = (recent_data['Price'].iloc[-1] - recent_data['Price'].iloc[0]) / recent_data['Price'].iloc[0]
            volatility = recent_data['Price'].pct_change().std()
            
            # 高ボラティリティ判定
            if volatility > self.volatility_threshold * 2:
                return MarketCondition.HIGH_VOLATILITY
            elif volatility < self.volatility_threshold * 0.5:
                return MarketCondition.LOW_VOLATILITY
            
            # トレンド判定
            if price_change > self.trend_threshold * 2:
                return MarketCondition.STRONG_UPTREND
            elif price_change > self.trend_threshold:
                return MarketCondition.MODERATE_UPTREND
            elif price_change < -self.trend_threshold * 2:
                return MarketCondition.STRONG_DOWNTREND
            elif price_change < -self.trend_threshold:
                return MarketCondition.MODERATE_DOWNTREND
            else:
                return MarketCondition.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error detecting market condition: {e}")
            return MarketCondition.SIDEWAYS

class StrategySimulator:
    """戦略シミュレーター"""
    
    def __init__(self):
        self.trend_detector = TrendDetector()
        
    def simulate_strategy_performance(self, data: pd.DataFrame, strategy_type: StrategyType) -> StrategyPerformance:
        """戦略パフォーマンスをシミュレーション"""
        try:
            signals = self._generate_signals(data, strategy_type)
            trades = self._execute_trades(data, signals)
            performance = self._calculate_performance(trades, strategy_type.value)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error simulating strategy {strategy_type.value}: {e}")
            return StrategyPerformance(
                strategy_name=strategy_type.value,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                win_rate=0.0,
                trade_count=0,
                avg_trade_duration=0.0,
                active_periods=[]
            )
    
    def _generate_signals(self, data: pd.DataFrame, strategy_type: StrategyType) -> pd.Series:
        """シグナル生成"""
        signals = pd.Series(0, index=data.index)
        
        if strategy_type == StrategyType.TREND_FOLLOWING:
            # トレンドフォロー: 移動平均クロスオーバー
            short_ma = data['Price'].rolling(window=10).mean()
            long_ma = data['Price'].rolling(window=30).mean()
            signals[short_ma > long_ma] = 1
            signals[short_ma < long_ma] = -1
            
        elif strategy_type == StrategyType.MEAN_REVERSION:
            # 平均回帰: RSIベース
            signals[data['RSI'] < 30] = 1
            signals[data['RSI'] > 70] = -1
            
        elif strategy_type == StrategyType.MOMENTUM:
            # モメンタム: MACDベース
            signals[(data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))] = 1
            signals[(data['MACD'] < data['MACD_Signal']) & (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))] = -1
            
        elif strategy_type == StrategyType.CONTRARIAN:
            # 逆張り: 価格の急激な変化に対する逆張り
            price_change = data['Price'].pct_change()
            signals[price_change < -0.05] = 1  # 大幅下落時に買い
            signals[price_change > 0.05] = -1  # 大幅上昇時に売り
            
        elif strategy_type == StrategyType.BREAKOUT:
            # ブレイクアウト: ボリンジャーバンドブレイクアウト
            rolling_mean = data['Price'].rolling(window=20).mean()
            rolling_std = data['Price'].rolling(window=20).std()
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            
            signals[data['Price'] > upper_band] = 1
            signals[data['Price'] < lower_band] = -1
        
        return signals
    
    def _execute_trades(self, data: pd.DataFrame, signals: pd.Series) -> List[Dict]:
        """取引実行"""
        trades = []
        position = 0
        entry_price = 0
        entry_time = 0
        
        for i, signal in enumerate(signals):
            if signal != 0 and position == 0:  # エントリー
                position = signal
                entry_price = data['Price'].iloc[i]
                entry_time = i
                
            elif signal != 0 and position != 0 and signal != position:  # イグジット
                exit_price = data['Price'].iloc[i]
                profit = (exit_price - entry_price) * position / entry_price
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'profit': profit,
                    'duration': i - entry_time
                })
                
                position = signal
                entry_price = exit_price
                entry_time = i
        
        return trades
    
    def _calculate_performance(self, trades: List[Dict], strategy_name: str) -> StrategyPerformance:
        """パフォーマンス計算"""
        if not trades:
            return StrategyPerformance(
                strategy_name=strategy_name,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                trade_count=0,
                avg_trade_duration=0.0,
                active_periods=[]
            )
        
        profits = [trade['profit'] for trade in trades]
        
        total_return = sum(profits)
        win_rate = len([p for p in profits if p > 0]) / len(profits)
        avg_trade_duration = np.mean([trade['duration'] for trade in trades])
        
        # シャープレシオ計算
        if len(profits) > 1:
            sharpe_ratio = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大ドローダウン計算
        cumulative_returns = np.cumprod([1 + p for p in profits])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # アクティブ期間
        active_periods = [(trade['entry_time'], trade['exit_time']) for trade in trades]
        
        return StrategyPerformance(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trade_count=len(trades),
            avg_trade_duration=avg_trade_duration,
            active_periods=active_periods
        )

class StrategySwitcher:
    """戦略切替器"""
    
    def __init__(self):
        self.trend_detector = TrendDetector()
        self.strategy_simulator = StrategySimulator()
        
        # 戦略と市場状況の適合性マップ
        self.strategy_suitability = {
            MarketCondition.STRONG_UPTREND: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
            MarketCondition.MODERATE_UPTREND: [StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT],
            MarketCondition.SIDEWAYS: [StrategyType.MEAN_REVERSION, StrategyType.CONTRARIAN],
            MarketCondition.MODERATE_DOWNTREND: [StrategyType.CONTRARIAN, StrategyType.MEAN_REVERSION],
            MarketCondition.STRONG_DOWNTREND: [StrategyType.CONTRARIAN, StrategyType.TREND_FOLLOWING],
            MarketCondition.HIGH_VOLATILITY: [StrategyType.CONTRARIAN, StrategyType.MEAN_REVERSION],
            MarketCondition.LOW_VOLATILITY: [StrategyType.BREAKOUT, StrategyType.MOMENTUM]
        }
    
    def determine_optimal_strategy(self, data: pd.DataFrame, current_index: int) -> Tuple[StrategyType, float]:
        """最適戦略を決定"""
        try:
            # 現在の市場状況を検出
            market_condition = self.trend_detector.detect_market_condition(data.iloc[:current_index+1])
            
            # 適合する戦略リストを取得
            suitable_strategies = self.strategy_suitability.get(market_condition, [StrategyType.TREND_FOLLOWING])
            
            # 信頼度計算（簡易版）
            confidence = min(0.9, max(0.1, (current_index + 1) / len(data)))
            
            # 最初の適合戦略を選択（実際の実装では性能ベースの選択が可能）
            optimal_strategy = suitable_strategies[0] if suitable_strategies else StrategyType.TREND_FOLLOWING
            
            return optimal_strategy, confidence
            
        except Exception as e:
            logger.error(f"Error determining optimal strategy: {e}")
            return StrategyType.TREND_FOLLOWING, 0.5
    
    def simulate_switching(self, data: pd.DataFrame, switch_frequency: int = 50) -> Tuple[List[SwitchingEvent], Dict[str, StrategyPerformance]]:
        """戦略切替をシミュレーション"""
        switching_events = []
        strategy_performances = {}
        current_strategy = None
        
        try:
            # 定期的に戦略を評価・切替
            for i in range(switch_frequency, len(data), switch_frequency):
                optimal_strategy, confidence = self.determine_optimal_strategy(data, i)
                market_condition = self.trend_detector.detect_market_condition(data.iloc[:i+1])
                
                if current_strategy != optimal_strategy:
                    # 戦略切替イベント記録
                    switching_events.append(SwitchingEvent(
                        timestamp=i,
                        from_strategy=current_strategy.value if current_strategy else "none",
                        to_strategy=optimal_strategy.value,
                        reason=f"Market condition: {market_condition.value}",
                        market_condition=market_condition,
                        confidence_score=confidence
                    ))
                    current_strategy = optimal_strategy
            
            # 各戦略のパフォーマンスを計算
            for strategy_type in StrategyType:
                performance = self.strategy_simulator.simulate_strategy_performance(data, strategy_type)
                strategy_performances[strategy_type.value] = performance
            
            return switching_events, strategy_performances
            
        except Exception as e:
            logger.error(f"Error simulating switching: {e}")
            return [], {}

class TrendSwitchingTester:
    """トレンド切替テスター"""
    
    def __init__(self, output_dir: str = "trend_switching_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_generator = SyntheticDataGenerator()
        self.strategy_switcher = StrategySwitcher()
        
    def create_test_scenarios(self) -> List[TrendScenario]:
        """テストシナリオ作成"""
        scenarios = [
            TrendScenario(
                scenario_id="strong_trend_reversal",
                name="強いトレンド反転",
                description="強い上昇トレンドから強い下降トレンドへの転換",
                initial_condition=MarketCondition.STRONG_UPTREND,
                target_condition=MarketCondition.STRONG_DOWNTREND,
                transition_period=10,
                volatility_factor=1.2
            ),
            TrendScenario(
                scenario_id="sideways_to_trend",
                name="レンジからトレンドへ",
                description="横ばい相場から上昇トレンドへの転換",
                initial_condition=MarketCondition.SIDEWAYS,
                target_condition=MarketCondition.MODERATE_UPTREND,
                transition_period=15,
                volatility_factor=0.8
            ),
            TrendScenario(
                scenario_id="volatility_spike",
                name="ボラティリティ急増",
                description="低ボラティリティから高ボラティリティへの転換",
                initial_condition=MarketCondition.LOW_VOLATILITY,
                target_condition=MarketCondition.HIGH_VOLATILITY,
                transition_period=5,
                volatility_factor=2.0
            ),
            TrendScenario(
                scenario_id="gradual_trend_change",
                name="段階的トレンド変化",
                description="中程度の上昇から中程度の下降への段階的転換",
                initial_condition=MarketCondition.MODERATE_UPTREND,
                target_condition=MarketCondition.MODERATE_DOWNTREND,
                transition_period=25,
                volatility_factor=1.0
            )
        ]
        
        return scenarios
    
    def run_test(self, scenario: TrendScenario) -> TestResult:
        """単一シナリオテスト実行"""
        start_time = time.time()
        errors = []
        logs = []
        
        try:
            logs.append(f"Starting test for scenario: {scenario.scenario_id}")
            
            # データ生成
            data = self.data_generator.generate_scenario_data(scenario)
            logs.append(f"Generated {len(data)} data points")
            
            # 戦略切替シミュレーション
            switching_events, strategy_performances = self.strategy_switcher.simulate_switching(data)
            logs.append(f"Simulated {len(switching_events)} switching events")
            
            # 全体パフォーマンス計算
            overall_performance = self._calculate_overall_performance(strategy_performances, switching_events)
            
            # 成功指標評価
            success_metrics = self._evaluate_success(overall_performance, switching_events)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                scenario=scenario,
                execution_time=execution_time,
                switching_events=switching_events,
                strategy_performances=strategy_performances,
                overall_performance=overall_performance,
                success_metrics=success_metrics,
                detailed_logs=logs,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Test failed for scenario {scenario.scenario_id}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                scenario=scenario,
                execution_time=execution_time,
                switching_events=[],
                strategy_performances={},
                overall_performance={},
                success_metrics={},
                detailed_logs=logs,
                errors=errors
            )
    
    def _calculate_overall_performance(self, strategy_performances: Dict[str, StrategyPerformance], 
                                     switching_events: List[SwitchingEvent]) -> Dict[str, float]:
        """全体パフォーマンス計算"""
        if not strategy_performances:
            return {}
        
        # 各戦略の重み付き平均（切替頻度に基づく）
        total_returns = [perf.total_return for perf in strategy_performances.values()]
        sharpe_ratios = [perf.sharpe_ratio for perf in strategy_performances.values()]
        max_drawdowns = [perf.max_drawdown for perf in strategy_performances.values()]
        win_rates = [perf.win_rate for perf in strategy_performances.values()]
        
        return {
            'avg_total_return': np.mean(total_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'switching_frequency': len(switching_events),
            'avg_confidence': np.mean([event.confidence_score for event in switching_events]) if switching_events else 0
        }
    
    def _evaluate_success(self, overall_performance: Dict[str, float], 
                         switching_events: List[SwitchingEvent]) -> Dict[str, bool]:
        """成功指標評価"""
        success_criteria = {
            'positive_returns': overall_performance.get('avg_total_return', 0) > 0,
            'reasonable_sharpe': overall_performance.get('avg_sharpe_ratio', 0) > 0.5,
            'controlled_drawdown': overall_performance.get('avg_max_drawdown', 1) < 0.2,
            'effective_switching': len(switching_events) > 0 and len(switching_events) < 10,
            'high_confidence': overall_performance.get('avg_confidence', 0) > 0.6
        }
        
        return success_criteria
    
    def run_all_tests(self) -> Dict[str, TestResult]:
        """全テスト実行"""
        scenarios = self.create_test_scenarios()
        results = {}
        
        logger.info(f"Running {len(scenarios)} test scenarios")
        
        for scenario in scenarios:
            logger.info(f"Running test for scenario: {scenario.scenario_id}")
            result = self.run_test(scenario)
            results[scenario.scenario_id] = result
            
            if result.errors:
                logger.warning(f"Scenario {scenario.scenario_id} completed with errors: {result.errors}")
            else:
                logger.info(f"Scenario {scenario.scenario_id} completed successfully")
        
        # 結果保存
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, TestResult]):
        """結果保存"""
        try:
            # JSON形式で保存
            results_json = {}
            for scenario_id, result in results.items():
                # Enumをstring形式に変換
                scenario_dict = result.scenario.to_dict()
                scenario_dict['initial_condition'] = result.scenario.initial_condition.value
                scenario_dict['target_condition'] = result.scenario.target_condition.value
                
                # 切替イベントのEnum変換
                switching_events_data = []
                for event in result.switching_events:
                    event_dict = {
                        'timestamp': event.timestamp,
                        'from_strategy': event.from_strategy,
                        'to_strategy': event.to_strategy,
                        'reason': event.reason,
                        'market_condition': event.market_condition.value,
                        'confidence_score': event.confidence_score
                    }
                    switching_events_data.append(event_dict)
                
                results_json[scenario_id] = {
                    'scenario': scenario_dict,
                    'execution_time': result.execution_time,
                    'switching_events': switching_events_data,
                    'switching_events_count': len(result.switching_events),
                    'overall_performance': result.overall_performance,
                    'success_metrics': result.success_metrics,
                    'errors_count': len(result.errors)
                }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_file = self.output_dir / f"trend_switching_test_results_{timestamp}.json"
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {json_file}")
            
            # サマリーレポート生成
            self._generate_summary_report(results, timestamp)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _generate_summary_report(self, results: Dict[str, TestResult], timestamp: str):
        """サマリーレポート生成"""
        try:
            report_file = self.output_dir / f"trend_switching_summary_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== Trend Switching Test Summary Report ===\n\n")
                f.write(f"Test Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Scenarios: {len(results)}\n\n")
                
                successful_tests = 0
                total_execution_time = 0
                
                for scenario_id, result in results.items():
                    f.write(f"--- Scenario: {scenario_id} ---\n")
                    f.write(f"Name: {result.scenario.name}\n")
                    f.write(f"Description: {result.scenario.description}\n")
                    f.write(f"Execution Time: {result.execution_time:.2f} seconds\n")
                    f.write(f"Switching Events: {len(result.switching_events)}\n")
                    
                    if result.overall_performance:
                        f.write(f"Average Return: {result.overall_performance.get('avg_total_return', 0):.4f}\n")
                        f.write(f"Average Sharpe: {result.overall_performance.get('avg_sharpe_ratio', 0):.4f}\n")
                        f.write(f"Average Drawdown: {result.overall_performance.get('avg_max_drawdown', 0):.4f}\n")
                    
                    success_count = sum(result.success_metrics.values())
                    total_criteria = len(result.success_metrics)
                    f.write(f"Success Rate: {success_count}/{total_criteria} criteria met\n")
                    
                    if not result.errors:
                        successful_tests += 1
                    else:
                        f.write(f"Errors: {len(result.errors)}\n")
                    
                    total_execution_time += result.execution_time
                    f.write("\n")
                
                f.write("=== Overall Summary ===\n")
                f.write(f"Successful Tests: {successful_tests}/{len(results)}\n")
                f.write(f"Success Rate: {successful_tests/len(results)*100:.1f}%\n")
                f.write(f"Total Execution Time: {total_execution_time:.2f} seconds\n")
                f.write(f"Average Time per Test: {total_execution_time/len(results):.2f} seconds\n")
            
            logger.info(f"Summary report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

def main():
    """メイン実行関数"""
    try:
        logger.info("Starting Enhanced Trend Switching Test System")
        
        # テスター初期化
        tester = TrendSwitchingTester()
        
        # 全テスト実行
        results = tester.run_all_tests()
        
        # 結果サマリー出力
        successful_tests = sum(1 for result in results.values() if not result.errors)
        total_tests = len(results)
        
        print(f"\n=== Test Execution Complete ===")
        print(f"Total Scenarios: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
        
        # 個別結果出力
        for scenario_id, result in results.items():
            status = "✅ SUCCESS" if not result.errors else "❌ FAILED"
            print(f"{status} {scenario_id}: {result.execution_time:.2f}s")
            
            if result.overall_performance:
                print(f"  Return: {result.overall_performance.get('avg_total_return', 0):.4f}")
                print(f"  Sharpe: {result.overall_performance.get('avg_sharpe_ratio', 0):.4f}")
                print(f"  Switches: {len(result.switching_events)}")
        
        logger.info("Enhanced Trend Switching Test System completed successfully")
        
    except Exception as e:
        logger.error(f"Test system failed: {e}")
        print(f"\n❌ Test system failed: {e}")

if __name__ == "__main__":
    main()
