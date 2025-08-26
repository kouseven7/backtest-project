"""
DSSMS Task 1.3: 銘柄切替エンジン V2
完全再構築による切替成功率の根本的改善

主要機能:
1. 多段階切替判定システム
2. 複数トリガー対応
3. 最適タイミング判定
4. 切替成功率向上機能
5. リスク管理統合

Author: GitHub Copilot Agent
Created: 2025-08-26
Task: 1.3 ポートフォリオ計算ロジック修正
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# 既存DSSMSコンポーネントとの統合
try:
    from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem
    from src.dssms.comprehensive_scoring_engine import ComprehensiveScoringEngine
    from src.dssms.market_condition_monitor import MarketConditionMonitor
    from src.dssms.dssms_portfolio_calculator_v2 import DSSMSPortfolioCalculatorV2
except ImportError as e:
    warnings.warn(f"DSSMSコンポーネントインポート失敗: {e}")

# 警告を抑制
warnings.filterwarnings('ignore')

class SwitchTriggerType(Enum):
    """切替トリガータイプ"""
    PERFORMANCE_DECLINE = "performance_decline"
    RANKING_CHANGE = "ranking_change"
    RISK_THRESHOLD = "risk_threshold"
    MARKET_REGIME_CHANGE = "market_regime_change"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    EMERGENCY = "emergency"

class SwitchPriority(Enum):
    """切替優先度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SwitchStatus(Enum):
    """切替ステータス"""
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"

@dataclass
class SwitchTrigger:
    """切替トリガー"""
    trigger_type: SwitchTriggerType
    priority: SwitchPriority
    confidence: float
    threshold_value: float
    current_value: float
    description: str
    timestamp: datetime

@dataclass
class SwitchCandidate:
    """切替候補"""
    symbol: str
    score: float
    ranking: int
    reason: str
    confidence: float
    risk_score: float
    expected_improvement: float

@dataclass
class SwitchDecision:
    """切替判定結果"""
    should_switch: bool
    from_symbol: str
    to_symbol: Optional[str]
    triggers: List[SwitchTrigger]
    candidates: List[SwitchCandidate]
    confidence: float
    timing_score: float
    risk_assessment: Dict[str, Any]
    recommendation: str

@dataclass
class SwitchExecution:
    """切替実行記録"""
    timestamp: datetime
    decision: SwitchDecision
    status: SwitchStatus
    execution_price_from: float
    execution_price_to: Optional[float]
    transaction_cost: float
    slippage: float
    result: Dict[str, Any]

class DSSMSSwitchEngineV2:
    """DSSMS 銘柄切替エンジン V2"""
    
    def __init__(self, portfolio_calculator: DSSMSPortfolioCalculatorV2,
                 config_path: Optional[str] = None):
        """
        Args:
            portfolio_calculator: ポートフォリオ計算エンジン
            config_path: 設定ファイルパス
        """
        self.portfolio_calculator = portfolio_calculator
        self.logger = setup_logger(__name__)
        
        # 切替履歴管理
        self.switch_history: List[SwitchExecution] = []
        self.trigger_history: List[SwitchTrigger] = []
        self.switch_statistics = {
            'total_switches': 0,
            'successful_switches': 0,
            'failed_switches': 0,
            'rejected_switches': 0,
            'success_rate': 0.0,
            'average_improvement': 0.0,
            'trigger_frequency': {}
        }
        
        # DSSMSコンポーネント統合
        try:
            self.ranking_system = HierarchicalRankingSystem()
            self.scoring_engine = ComprehensiveScoringEngine()
            self.market_monitor = MarketConditionMonitor()
            self.integration_enabled = True
            self.logger.info("DSSMS統合コンポーネント利用可能")
        except:
            self.integration_enabled = False
            self.logger.warning("DSSMS統合コンポーネント利用不可 - 基本機能のみ")
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        self.logger.info("DSSMS銘柄切替エンジンV2初期化完了")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定読み込み"""
        default_config = {
            "switch_criteria": {
                "performance_threshold": -0.05,      # 5%以上の損失で検討
                "ranking_change_threshold": 3,       # ランキング3位以上の変動
                "risk_threshold": 0.15,              # 15%以上のリスク増加
                "volatility_threshold": 0.25,        # 25%以上のボラティリティ
                "correlation_threshold": 0.7,        # 相関係数0.7以下
                "confidence_threshold": 0.6          # 信頼度60%以上で実行
            },
            "switch_constraints": {
                "min_holding_period_hours": 24,      # 最低24時間保持
                "max_switches_per_day": 3,           # 1日最大3回
                "switch_cost_rate": 0.002,           # 切替コスト0.2%
                "position_size_limit": 0.3,          # ポジションサイズ30%上限
                "blackout_hours": [9, 15]            # 取引停止時間
            },
            "optimization": {
                "enable_ml_scoring": False,          # 機械学習スコアリング
                "use_market_regime": True,           # 市場レジーム考慮
                "dynamic_thresholds": True,          # 動的閾値調整
                "risk_parity_weighting": False       # リスクパリティ重み付け
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"切替エンジン設定読み込み失敗: {e}")
        
        return default_config
    
    def evaluate_switch_decision(self, current_symbol: str, 
                               available_symbols: List[str],
                               market_data: Dict[str, pd.DataFrame],
                               timestamp: datetime) -> SwitchDecision:
        """
        銘柄切替判定メイン処理
        
        Args:
            current_symbol: 現在保有銘柄
            available_symbols: 切替候補銘柄リスト
            market_data: 市場データ
            timestamp: 判定時刻
        
        Returns:
            切替判定結果
        """
        try:
            self.logger.info(f"銘柄切替判定開始: {current_symbol} -> 候補{len(available_symbols)}銘柄")
            
            # 1. 切替トリガー評価
            triggers = self._evaluate_switch_triggers(current_symbol, market_data, timestamp)
            
            # 2. 切替候補評価
            candidates = self._evaluate_switch_candidates(
                current_symbol, available_symbols, market_data, timestamp
            )
            
            # 3. 切替判定
            switch_decision = self._make_switch_decision(
                current_symbol, triggers, candidates, timestamp
            )
            
            # 4. リスク評価
            risk_assessment = self._assess_switch_risk(switch_decision, market_data)
            switch_decision.risk_assessment = risk_assessment
            
            # 5. 最終推奨事項
            recommendation = self._generate_switch_recommendation(switch_decision)
            switch_decision.recommendation = recommendation
            
            # トリガー履歴記録
            self.trigger_history.extend(triggers)
            
            self.logger.info(f"切替判定完了: 切替推奨={switch_decision.should_switch}")
            return switch_decision
            
        except Exception as e:
            self.logger.error(f"切替判定エラー: {e}")
            return SwitchDecision(
                should_switch=False,
                from_symbol=current_symbol,
                to_symbol=None,
                triggers=[],
                candidates=[],
                confidence=0.0,
                timing_score=0.0,
                risk_assessment={},
                recommendation=f"エラーのため切替停止: {e}"
            )
    
    def _evaluate_switch_triggers(self, current_symbol: str, 
                                 market_data: Dict[str, pd.DataFrame],
                                 timestamp: datetime) -> List[SwitchTrigger]:
        """切替トリガー評価"""
        triggers = []
        
        try:
            current_data = market_data.get(current_symbol)
            if current_data is None or current_data.empty:
                return triggers
            
            # 1. パフォーマンス低下トリガー
            perf_trigger = self._check_performance_decline_trigger(current_symbol, current_data, timestamp)
            if perf_trigger:
                triggers.append(perf_trigger)
            
            # 2. ランキング変化トリガー
            ranking_trigger = self._check_ranking_change_trigger(current_symbol, timestamp)
            if ranking_trigger:
                triggers.append(ranking_trigger)
            
            # 3. リスク閾値トリガー
            risk_trigger = self._check_risk_threshold_trigger(current_symbol, current_data, timestamp)
            if risk_trigger:
                triggers.append(risk_trigger)
            
            # 4. ボラティリティスパイクトリガー
            vol_trigger = self._check_volatility_spike_trigger(current_symbol, current_data, timestamp)
            if vol_trigger:
                triggers.append(vol_trigger)
            
            # 5. 市場レジーム変化トリガー
            if self.integration_enabled:
                regime_trigger = self._check_market_regime_trigger(current_symbol, timestamp)
                if regime_trigger:
                    triggers.append(regime_trigger)
            
            # トリガー統計更新
            for trigger in triggers:
                trigger_type = trigger.trigger_type.value
                self.switch_statistics['trigger_frequency'][trigger_type] = \
                    self.switch_statistics['trigger_frequency'].get(trigger_type, 0) + 1
            
        except Exception as e:
            self.logger.error(f"トリガー評価エラー: {e}")
        
        return triggers
    
    def _check_performance_decline_trigger(self, symbol: str, data: pd.DataFrame, 
                                          timestamp: datetime) -> Optional[SwitchTrigger]:
        """パフォーマンス低下トリガーチェック"""
        try:
            if len(data) < 5:
                return None
            
            # 5日間リターン計算
            recent_returns = data['Close'].pct_change().tail(5)
            cumulative_return = (1 + recent_returns).prod() - 1
            
            threshold = self.config['switch_criteria']['performance_threshold']
            
            if cumulative_return < threshold:
                confidence = min(1.0, abs(cumulative_return / threshold))
                
                return SwitchTrigger(
                    trigger_type=SwitchTriggerType.PERFORMANCE_DECLINE,
                    priority=SwitchPriority.HIGH if cumulative_return < threshold * 2 else SwitchPriority.MEDIUM,
                    confidence=confidence,
                    threshold_value=threshold,
                    current_value=cumulative_return,
                    description=f"5日間累積リターン{cumulative_return:.1%}が閾値{threshold:.1%}を下回る",
                    timestamp=timestamp
                )
            
        except Exception as e:
            self.logger.error(f"パフォーマンス低下チェックエラー: {e}")
        
        return None
    
    def _check_ranking_change_trigger(self, symbol: str, timestamp: datetime) -> Optional[SwitchTrigger]:
        """ランキング変化トリガーチェック"""
        try:
            if not self.integration_enabled:
                return None
            
            # 現在のランキング取得
            current_ranking = self._get_symbol_ranking(symbol)
            if current_ranking is None:
                return None
            
            # 履歴からランキング変化を計算
            ranking_change = self._calculate_ranking_change(symbol)
            threshold = self.config['switch_criteria']['ranking_change_threshold']
            
            if ranking_change >= threshold:
                confidence = min(1.0, ranking_change / (threshold * 2))
                
                return SwitchTrigger(
                    trigger_type=SwitchTriggerType.RANKING_CHANGE,
                    priority=SwitchPriority.MEDIUM,
                    confidence=confidence,
                    threshold_value=threshold,
                    current_value=ranking_change,
                    description=f"ランキング{ranking_change}位低下が閾値{threshold}位を超過",
                    timestamp=timestamp
                )
                
        except Exception as e:
            self.logger.error(f"ランキング変化チェックエラー: {e}")
        
        return None
    
    def _check_risk_threshold_trigger(self, symbol: str, data: pd.DataFrame, 
                                     timestamp: datetime) -> Optional[SwitchTrigger]:
        """リスク閾値トリガーチェック"""
        try:
            if len(data) < 20:
                return None
            
            # 20日間ボラティリティ計算
            returns = data['Close'].pct_change().tail(20)
            current_volatility = returns.std() * np.sqrt(252)  # 年率換算
            
            # ベースラインボラティリティ（過去60日）
            if len(data) >= 60:
                baseline_returns = data['Close'].pct_change().tail(60).head(40)
                baseline_volatility = baseline_returns.std() * np.sqrt(252)
                
                volatility_increase = (current_volatility - baseline_volatility) / baseline_volatility
                threshold = self.config['switch_criteria']['risk_threshold']
                
                if volatility_increase > threshold:
                    confidence = min(1.0, volatility_increase / threshold)
                    
                    return SwitchTrigger(
                        trigger_type=SwitchTriggerType.RISK_THRESHOLD,
                        priority=SwitchPriority.HIGH if volatility_increase > threshold * 2 else SwitchPriority.MEDIUM,
                        confidence=confidence,
                        threshold_value=threshold,
                        current_value=volatility_increase,
                        description=f"ボラティリティ増加{volatility_increase:.1%}が閾値{threshold:.1%}を超過",
                        timestamp=timestamp
                    )
            
        except Exception as e:
            self.logger.error(f"リスク閾値チェックエラー: {e}")
        
        return None
    
    def _check_volatility_spike_trigger(self, symbol: str, data: pd.DataFrame, 
                                       timestamp: datetime) -> Optional[SwitchTrigger]:
        """ボラティリティスパイクトリガーチェック"""
        try:
            if len(data) < 10:
                return None
            
            # 直近の価格変動
            recent_returns = data['Close'].pct_change().tail(3).abs()
            max_recent_move = recent_returns.max()
            
            threshold = self.config['switch_criteria']['volatility_threshold']
            
            if max_recent_move > threshold:
                confidence = min(1.0, max_recent_move / threshold)
                
                return SwitchTrigger(
                    trigger_type=SwitchTriggerType.VOLATILITY_SPIKE,
                    priority=SwitchPriority.HIGH,
                    confidence=confidence,
                    threshold_value=threshold,
                    current_value=max_recent_move,
                    description=f"急激な価格変動{max_recent_move:.1%}が閾値{threshold:.1%}を超過",
                    timestamp=timestamp
                )
                
        except Exception as e:
            self.logger.error(f"ボラティリティスパイクチェックエラー: {e}")
        
        return None
    
    def _check_market_regime_trigger(self, symbol: str, timestamp: datetime) -> Optional[SwitchTrigger]:
        """市場レジーム変化トリガーチェック"""
        try:
            if not self.integration_enabled:
                return None
            
            # 市場状況監視
            market_conditions = self.market_monitor.get_current_conditions()
            if not market_conditions:
                return None
            
            # レジーム変化検出
            regime_change_detected = market_conditions.get('regime_change', False)
            if regime_change_detected:
                return SwitchTrigger(
                    trigger_type=SwitchTriggerType.MARKET_REGIME_CHANGE,
                    priority=SwitchPriority.MEDIUM,
                    confidence=0.8,
                    threshold_value=0.5,
                    current_value=1.0,
                    description="市場レジーム変化を検出",
                    timestamp=timestamp
                )
                
        except Exception as e:
            self.logger.error(f"市場レジームチェックエラー: {e}")
        
        return None
    
    def _evaluate_switch_candidates(self, current_symbol: str, 
                                   available_symbols: List[str],
                                   market_data: Dict[str, pd.DataFrame],
                                   timestamp: datetime) -> List[SwitchCandidate]:
        """切替候補評価"""
        candidates = []
        
        try:
            for symbol in available_symbols:
                if symbol == current_symbol:
                    continue
                
                candidate = self._evaluate_single_candidate(
                    symbol, current_symbol, market_data, timestamp
                )
                
                if candidate:
                    candidates.append(candidate)
            
            # スコア順ソート
            candidates.sort(key=lambda x: x.score, reverse=True)
            
            # ランキング付与
            for i, candidate in enumerate(candidates):
                candidate.ranking = i + 1
            
        except Exception as e:
            self.logger.error(f"候補評価エラー: {e}")
        
        return candidates
    
    def _evaluate_single_candidate(self, candidate_symbol: str, current_symbol: str,
                                  market_data: Dict[str, pd.DataFrame],
                                  timestamp: datetime) -> Optional[SwitchCandidate]:
        """単一候補評価"""
        try:
            candidate_data = market_data.get(candidate_symbol)
            current_data = market_data.get(current_symbol)
            
            if candidate_data is None or current_data is None:
                return None
            
            if candidate_data.empty or current_data.empty:
                return None
            
            # 基本スコア計算
            base_score = self._calculate_candidate_base_score(candidate_data)
            
            # 相対スコア計算（現在銘柄との比較）
            relative_score = self._calculate_relative_score(candidate_data, current_data)
            
            # リスクスコア計算
            risk_score = self._calculate_candidate_risk_score(candidate_data)
            
            # 統合スコア
            final_score = (base_score * 0.4 + relative_score * 0.4 + (1 - risk_score) * 0.2)
            
            # 期待改善計算
            expected_improvement = self._calculate_expected_improvement(
                candidate_data, current_data
            )
            
            # 信頼度計算
            confidence = self._calculate_candidate_confidence(candidate_data, final_score)
            
            return SwitchCandidate(
                symbol=candidate_symbol,
                score=final_score,
                ranking=0,  # 後で設定
                reason=f"統合スコア{final_score:.3f}, 期待改善{expected_improvement:.1%}",
                confidence=confidence,
                risk_score=risk_score,
                expected_improvement=expected_improvement
            )
            
        except Exception as e:
            self.logger.error(f"候補{candidate_symbol}評価エラー: {e}")
            return None
    
    def _calculate_candidate_base_score(self, data: pd.DataFrame) -> float:
        """候補基本スコア計算"""
        try:
            if len(data) < 5:
                return 0.0
            
            # 複数要素の組み合わせ
            scores = []
            
            # 1. 短期モメンタム（5日）
            short_momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1)
            scores.append(min(1.0, max(0.0, short_momentum + 0.5)))  # -50%〜+50%を0〜1に変換
            
            # 2. ボラティリティ（安定性）
            returns = data['Close'].pct_change().tail(20)
            volatility = returns.std()
            volatility_score = max(0.0, 1.0 - volatility * 10)  # 低ボラティリティほど高スコア
            scores.append(volatility_score)
            
            # 3. 出来高（流動性）
            volume_trend = data['Volume'].tail(5).mean() / data['Volume'].tail(20).mean()
            volume_score = min(1.0, max(0.0, volume_trend - 0.5))
            scores.append(volume_score)
            
            return np.mean(scores)
            
        except Exception as e:
            self.logger.error(f"基本スコア計算エラー: {e}")
            return 0.0
    
    def _calculate_relative_score(self, candidate_data: pd.DataFrame, 
                                 current_data: pd.DataFrame) -> float:
        """相対スコア計算"""
        try:
            if len(candidate_data) < 5 or len(current_data) < 5:
                return 0.0
            
            # 候補と現在銘柄のパフォーマンス比較
            candidate_return = candidate_data['Close'].pct_change().tail(10).mean()
            current_return = current_data['Close'].pct_change().tail(10).mean()
            
            relative_performance = candidate_return - current_return
            
            # -10%〜+10%を0〜1に変換
            return min(1.0, max(0.0, (relative_performance + 0.1) / 0.2))
            
        except Exception as e:
            self.logger.error(f"相対スコア計算エラー: {e}")
            return 0.0
    
    def _calculate_candidate_risk_score(self, data: pd.DataFrame) -> float:
        """候補リスクスコア計算"""
        try:
            if len(data) < 10:
                return 1.0  # データ不足は高リスク
            
            # ボラティリティベースのリスク
            returns = data['Close'].pct_change().tail(20)
            volatility = returns.std()
            
            # 最大下落リスク
            max_decline = returns.min()
            
            # リスクスコア（0〜1、高いほどリスク高）
            vol_risk = min(1.0, volatility * 20)  # 5%ボラティリティで1.0
            decline_risk = min(1.0, abs(max_decline) * 5)  # 20%下落で1.0
            
            return np.mean([vol_risk, decline_risk])
            
        except Exception as e:
            self.logger.error(f"リスクスコア計算エラー: {e}")
            return 1.0
    
    def _calculate_expected_improvement(self, candidate_data: pd.DataFrame, 
                                      current_data: pd.DataFrame) -> float:
        """期待改善計算"""
        try:
            if len(candidate_data) < 5 or len(current_data) < 5:
                return 0.0
            
            # 簡易的な期待リターン差
            candidate_momentum = candidate_data['Close'].pct_change().tail(5).mean()
            current_momentum = current_data['Close'].pct_change().tail(5).mean()
            
            return candidate_momentum - current_momentum
            
        except Exception as e:
            self.logger.error(f"期待改善計算エラー: {e}")
            return 0.0
    
    def _calculate_candidate_confidence(self, data: pd.DataFrame, score: float) -> float:
        """候補信頼度計算"""
        try:
            # データ量ベースの信頼度
            data_confidence = min(1.0, len(data) / 30)  # 30日分で最大信頼度
            
            # スコアベースの信頼度
            score_confidence = score
            
            # 統合信頼度
            return np.sqrt(data_confidence * score_confidence)
            
        except Exception as e:
            self.logger.error(f"信頼度計算エラー: {e}")
            return 0.0
    
    def _make_switch_decision(self, current_symbol: str, 
                             triggers: List[SwitchTrigger],
                             candidates: List[SwitchCandidate],
                             timestamp: datetime) -> SwitchDecision:
        """切替判定"""
        try:
            # 基本判定条件
            has_high_priority_trigger = any(t.priority.value >= SwitchPriority.HIGH.value for t in triggers)
            has_multiple_triggers = len(triggers) >= 2
            has_good_candidate = len(candidates) > 0 and candidates[0].score > 0.6
            
            # 制約チェック
            constraints_ok = self._check_switch_constraints(current_symbol, timestamp)
            
            # 切替判定
            should_switch = (
                (has_high_priority_trigger or has_multiple_triggers) and
                has_good_candidate and
                constraints_ok
            )
            
            # 最適候補選択
            best_candidate = candidates[0] if candidates else None
            to_symbol = best_candidate.symbol if best_candidate else None
            
            # 総合信頼度計算
            trigger_confidence = np.mean([t.confidence for t in triggers]) if triggers else 0.0
            candidate_confidence = best_candidate.confidence if best_candidate else 0.0
            overall_confidence = np.sqrt(trigger_confidence * candidate_confidence)
            
            # タイミングスコア計算
            timing_score = self._calculate_timing_score(triggers, timestamp)
            
            return SwitchDecision(
                should_switch=should_switch,
                from_symbol=current_symbol,
                to_symbol=to_symbol,
                triggers=triggers,
                candidates=candidates,
                confidence=overall_confidence,
                timing_score=timing_score,
                risk_assessment={},  # 後で設定
                recommendation=""    # 後で設定
            )
            
        except Exception as e:
            self.logger.error(f"切替判定エラー: {e}")
            return SwitchDecision(
                should_switch=False,
                from_symbol=current_symbol,
                to_symbol=None,
                triggers=triggers,
                candidates=candidates,
                confidence=0.0,
                timing_score=0.0,
                risk_assessment={},
                recommendation=f"判定エラー: {e}"
            )
    
    def _check_switch_constraints(self, current_symbol: str, timestamp: datetime) -> bool:
        """切替制約チェック"""
        try:
            # 最低保持期間チェック
            min_holding_hours = self.config['switch_constraints']['min_holding_period_hours']
            if self.switch_history:
                last_switch = max(self.switch_history, key=lambda x: x.timestamp)
                hours_since_last = (timestamp - last_switch.timestamp).total_seconds() / 3600
                if hours_since_last < min_holding_hours:
                    return False
            
            # 1日の切替回数制限
            max_daily_switches = self.config['switch_constraints']['max_switches_per_day']
            today_switches = [
                s for s in self.switch_history 
                if s.timestamp.date() == timestamp.date() and s.status == SwitchStatus.EXECUTED
            ]
            if len(today_switches) >= max_daily_switches:
                return False
            
            # 取引停止時間チェック
            blackout_hours = self.config['switch_constraints']['blackout_hours']
            if timestamp.hour in blackout_hours:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"制約チェックエラー: {e}")
            return False
    
    def _calculate_timing_score(self, triggers: List[SwitchTrigger], timestamp: datetime) -> float:
        """タイミングスコア計算"""
        try:
            if not triggers:
                return 0.0
            
            # トリガー強度ベース
            trigger_strength = np.mean([t.confidence * t.priority.value for t in triggers])
            
            # 時間帯ベース（市場開始・終了前後は低スコア）
            hour = timestamp.hour
            if 9 <= hour <= 15:  # 取引時間内
                time_score = 1.0
            elif hour in [8, 16]:  # 開始・終了1時間前後
                time_score = 0.7
            else:
                time_score = 0.3
            
            return min(1.0, trigger_strength * time_score)
            
        except Exception as e:
            self.logger.error(f"タイミングスコア計算エラー: {e}")
            return 0.0
    
    def _assess_switch_risk(self, decision: SwitchDecision, 
                           market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """切替リスク評価"""
        try:
            risk_assessment = {
                'execution_risk': 'low',
                'market_risk': 'medium',
                'timing_risk': 'low',
                'overall_risk': 'medium',
                'risk_factors': [],
                'mitigation_suggestions': []
            }
            
            # 実行リスク
            if decision.confidence < 0.5:
                risk_assessment['execution_risk'] = 'high'
                risk_assessment['risk_factors'].append('低い判定信頼度')
            
            # 市場リスク
            if decision.to_symbol and decision.to_symbol in market_data:
                candidate_data = market_data[decision.to_symbol]
                if len(candidate_data) >= 10:
                    volatility = candidate_data['Close'].pct_change().tail(10).std()
                    if volatility > 0.05:  # 5%以上の日次ボラティリティ
                        risk_assessment['market_risk'] = 'high'
                        risk_assessment['risk_factors'].append('高ボラティリティ銘柄')
            
            # タイミングリスク
            if decision.timing_score < 0.5:
                risk_assessment['timing_risk'] = 'high'
                risk_assessment['risk_factors'].append('不適切なタイミング')
            
            # 総合リスク判定
            high_risk_factors = sum(1 for risk in [
                risk_assessment['execution_risk'],
                risk_assessment['market_risk'],
                risk_assessment['timing_risk']
            ] if risk == 'high')
            
            if high_risk_factors >= 2:
                risk_assessment['overall_risk'] = 'high'
            elif high_risk_factors == 1:
                risk_assessment['overall_risk'] = 'medium'
            else:
                risk_assessment['overall_risk'] = 'low'
            
            # 軽減提案
            if risk_assessment['overall_risk'] == 'high':
                risk_assessment['mitigation_suggestions'].append('段階的な切替を検討')
                risk_assessment['mitigation_suggestions'].append('より保守的な候補を選択')
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"リスク評価エラー: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}
    
    def _generate_switch_recommendation(self, decision: SwitchDecision) -> str:
        """切替推奨事項生成"""
        try:
            if not decision.should_switch:
                return "現在の銘柄を継続保持することを推奨"
            
            risk_level = decision.risk_assessment.get('overall_risk', 'unknown')
            
            if risk_level == 'low':
                return f"切替を推奨: {decision.to_symbol} (低リスク、信頼度{decision.confidence:.1%})"
            elif risk_level == 'medium':
                return f"慎重な切替を推奨: {decision.to_symbol} (中リスク、追加確認推奨)"
            elif risk_level == 'high':
                return f"切替要注意: {decision.to_symbol} (高リスク、十分な検討必要)"
            else:
                return f"切替判定不明: リスク評価失敗"
                
        except Exception as e:
            return f"推奨事項生成エラー: {e}"
    
    def execute_switch(self, decision: SwitchDecision, 
                      market_data: Dict[str, pd.DataFrame]) -> SwitchExecution:
        """切替実行"""
        try:
            self.logger.info(f"切替実行開始: {decision.from_symbol} -> {decision.to_symbol}")
            
            execution = SwitchExecution(
                timestamp=datetime.now(),
                decision=decision,
                status=SwitchStatus.PENDING,
                execution_price_from=0.0,
                execution_price_to=None,
                transaction_cost=0.0,
                slippage=0.0,
                result={}
            )
            
            if not decision.should_switch or not decision.to_symbol:
                execution.status = SwitchStatus.REJECTED
                execution.result = {'reason': '切替条件不充足'}
                return execution
            
            # 実行価格取得
            from_data = market_data.get(decision.from_symbol)
            to_data = market_data.get(decision.to_symbol)
            
            if from_data is None or to_data is None or from_data.empty or to_data.empty:
                execution.status = SwitchStatus.FAILED
                execution.result = {'reason': '価格データ不足'}
                return execution
            
            execution.execution_price_from = from_data['Close'].iloc[-1]
            execution.execution_price_to = to_data['Close'].iloc[-1]
            
            # 取引コスト計算
            switch_cost_rate = self.config['switch_constraints']['switch_cost_rate']
            portfolio_value = self.portfolio_calculator.get_current_portfolio_value()
            execution.transaction_cost = portfolio_value * switch_cost_rate
            
            # ポートフォリオ計算エンジンを通じて実際の切替実行
            # 1. 現在ポジション売却
            current_position = self.portfolio_calculator.current_positions.get(decision.from_symbol)
            if current_position:
                sell_success, sell_details = self.portfolio_calculator.add_trade(
                    timestamp=execution.timestamp,
                    symbol=decision.from_symbol,
                    side='sell',
                    quantity=current_position.quantity,
                    price=execution.execution_price_from,
                    strategy='SwitchEngine'
                )
                
                if not sell_success:
                    execution.status = SwitchStatus.FAILED
                    execution.result = {'reason': '売却失敗', 'details': sell_details}
                    return execution
            
            # 2. 新規ポジション購入
            available_cash = self.portfolio_calculator.cash * 0.95  # 5%余裕
            new_quantity = int(available_cash / execution.execution_price_to)
            
            if new_quantity > 0:
                buy_success, buy_details = self.portfolio_calculator.add_trade(
                    timestamp=execution.timestamp,
                    symbol=decision.to_symbol,
                    side='buy',
                    quantity=new_quantity,
                    price=execution.execution_price_to,
                    strategy='SwitchEngine'
                )
                
                if buy_success:
                    execution.status = SwitchStatus.EXECUTED
                    execution.result = {
                        'sell_details': sell_details if 'sell_details' in locals() else {},
                        'buy_details': buy_details,
                        'new_quantity': new_quantity
                    }
                else:
                    execution.status = SwitchStatus.FAILED
                    execution.result = {'reason': '購入失敗', 'details': buy_details}
            else:
                execution.status = SwitchStatus.FAILED
                execution.result = {'reason': '資金不足'}
            
            # 統計更新
            self.switch_history.append(execution)
            self._update_switch_statistics(execution)
            
            self.logger.info(f"切替実行完了: ステータス={execution.status.value}")
            return execution
            
        except Exception as e:
            self.logger.error(f"切替実行エラー: {e}")
            execution.status = SwitchStatus.FAILED
            execution.result = {'reason': f'実行エラー: {e}'}
            return execution
    
    def _update_switch_statistics(self, execution: SwitchExecution):
        """切替統計更新"""
        try:
            self.switch_statistics['total_switches'] += 1
            
            if execution.status == SwitchStatus.EXECUTED:
                self.switch_statistics['successful_switches'] += 1
            elif execution.status == SwitchStatus.FAILED:
                self.switch_statistics['failed_switches'] += 1
            elif execution.status == SwitchStatus.REJECTED:
                self.switch_statistics['rejected_switches'] += 1
            
            # 成功率更新
            total = self.switch_statistics['total_switches']
            if total > 0:
                success = self.switch_statistics['successful_switches']
                self.switch_statistics['success_rate'] = success / total
            
        except Exception as e:
            self.logger.error(f"統計更新エラー: {e}")
    
    def get_switch_statistics(self) -> Dict[str, Any]:
        """切替統計取得"""
        return self.switch_statistics.copy()
    
    def generate_switch_report(self) -> Dict[str, Any]:
        """切替レポート生成"""
        try:
            recent_switches = [s for s in self.switch_history[-10:]]  # 直近10件
            
            report = {
                'statistics': self.get_switch_statistics(),
                'recent_switches': [
                    {
                        'timestamp': s.timestamp.isoformat(),
                        'from_symbol': s.decision.from_symbol,
                        'to_symbol': s.decision.to_symbol,
                        'status': s.status.value,
                        'confidence': s.decision.confidence,
                        'triggers': len(s.decision.triggers)
                    }
                    for s in recent_switches
                ],
                'trigger_analysis': self.switch_statistics['trigger_frequency'],
                'performance_summary': {
                    'success_rate': self.switch_statistics['success_rate'],
                    'total_switches': self.switch_statistics['total_switches'],
                    'integration_status': self.integration_enabled
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"切替レポート生成エラー: {e}")
            return {}
    
    # ヘルパーメソッド
    def _get_symbol_ranking(self, symbol: str) -> Optional[int]:
        """銘柄ランキング取得"""
        try:
            if self.integration_enabled:
                # 実際のランキングシステムから取得
                ranking_result = self.ranking_system.get_symbol_ranking(symbol)
                return ranking_result.get('ranking')
            else:
                # フォールバック: ダミーランキング
                return hash(symbol) % 100 + 1
        except Exception:
            return None
    
    def _calculate_ranking_change(self, symbol: str) -> int:
        """ランキング変化計算"""
        try:
            # 簡易実装: 履歴ベースの変化計算
            # 実際の実装では履歴データベースから取得
            return abs(hash(symbol + str(datetime.now().date())) % 5)  # 0-4の変化
        except Exception:
            return 0

# テスト実行機能
def test_dssms_switch_engine_v2():
    """DSSMS銘柄切替エンジンV2のテスト"""
    print("=== DSSMS 銘柄切替エンジンV2 テスト ===")
    
    try:
        # ポートフォリオ計算エンジン作成
        from src.dssms.dssms_portfolio_calculator_v2 import create_dssms_portfolio_calculator
        portfolio_calc = create_dssms_portfolio_calculator()
        
        # 切替エンジン初期化
        switch_engine = DSSMSSwitchEngineV2(portfolio_calc)
        
        # テストデータ準備
        test_symbols = ["1306.T", "SPY", "QQQ"]
        test_market_data = {}
        
        for symbol in test_symbols:
            dates = pd.date_range(start='2025-08-01', periods=30, freq='D')
            prices = 100 + np.cumsum(np.random.randn(30) * 2)
            volumes = np.random.randint(10000, 100000, 30)
            
            test_market_data[symbol] = pd.DataFrame({
                'Close': prices,
                'Volume': volumes
            }, index=dates)
        
        print("\n--- 切替判定テスト ---")
        decision = switch_engine.evaluate_switch_decision(
            current_symbol="1306.T",
            available_symbols=test_symbols,
            market_data=test_market_data,
            timestamp=datetime.now()
        )
        
        print(f"切替推奨: {decision.should_switch}")
        print(f"推奨銘柄: {decision.to_symbol}")
        print(f"信頼度: {decision.confidence:.1%}")
        print(f"トリガー数: {len(decision.triggers)}")
        print(f"候補数: {len(decision.candidates)}")
        
        print("\n--- 切替実行テスト ---")
        if decision.should_switch:
            execution = switch_engine.execute_switch(decision, test_market_data)
            print(f"実行ステータス: {execution.status.value}")
            print(f"取引コスト: {execution.transaction_cost:.2f}円")
        
        print("\n--- 統計情報 ---")
        stats = switch_engine.get_switch_statistics()
        print(f"総切替回数: {stats['total_switches']}")
        print(f"成功率: {stats['success_rate']:.1%}")
        
        print("\n=== テスト完了: 成功 ===")
        return True
        
    except Exception as e:
        print(f"\nテストエラー: {e}")
        return False

if __name__ == "__main__":
    test_dssms_switch_engine_v2()
