"""
DSSMS Phase 3 Task 3.2: インテリジェント銘柄切替管理システム
高度な銘柄切替ロジック実装

既存DSSMS Phase 1・Phase 2コンポーネントとの統合を考慮した設計
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存DSSMSコンポーネントのインポート
from .market_condition_monitor import MarketConditionMonitor
from .hierarchical_ranking_system import HierarchicalRankingSystem
from .comprehensive_scoring_engine import ComprehensiveScoringEngine
from .perfect_order_detector import PerfectOrderDetector
from .dssms_data_manager import DSSMSDataManager

# 既存リスク管理のインポート
from config.risk_management import RiskManagement
from config.logger_config import setup_logger

class SwitchDecision(Enum):
    """切替決定タイプ"""
    NO_SWITCH = "no_switch"
    IMMEDIATE_SWITCH = "immediate_switch"
    GRADUAL_SWITCH = "gradual_switch"
    EMERGENCY_EXIT = "emergency_exit"

@dataclass
class PositionEvaluation:
    """ポジション評価結果"""
    symbol: str
    current_score: float
    perfect_order_status: bool
    holding_period_hours: float
    profit_loss_ratio: float
    risk_level: str
    evaluation_timestamp: datetime
    recommendation: SwitchDecision

@dataclass
class SwitchRecord:
    """切替履歴レコード"""
    timestamp: datetime
    from_symbol: str
    to_symbol: str
    reason: str
    from_score: float
    to_score: float
    market_condition: str
    success: bool
    execution_time_ms: float

class PositionTracker:
    """ポジション追跡システム"""
    
    def __init__(self):
        self.positions = {}
        self.entry_times = {}
        self.entry_prices = {}
        self.logger = setup_logger('dssms.position_tracker')
    
    def add_position(self, symbol: str, entry_price: float, position_size: int):
        """ポジション追加"""
        self.positions[symbol] = position_size
        self.entry_times[symbol] = datetime.now()
        self.entry_prices[symbol] = entry_price
        self.logger.info(f"Position added: {symbol} @ {entry_price} x{position_size}")
    
    def remove_position(self, symbol: str):
        """ポジション削除"""
        if symbol in self.positions:
            del self.positions[symbol]
            del self.entry_times[symbol]
            del self.entry_prices[symbol]
            self.logger.info(f"Position removed: {symbol}")
    
    def get_holding_period(self, symbol: str) -> float:
        """保有期間取得（時間単位）"""
        if symbol in self.entry_times:
            return (datetime.now() - self.entry_times[symbol]).total_seconds() / 3600
        return 0.0
    
    def get_current_positions(self) -> Dict[str, Any]:
        """現在ポジション情報取得"""
        return {
            'symbols': list(self.positions.keys()),
            'position_count': len(self.positions),
            'positions': self.positions.copy()
        }

class SwitchHistoryManager:
    """切替履歴管理システム"""
    
    def __init__(self, max_history: int = 1000):
        self.history = []
        self.max_history = max_history
        self.daily_switch_count = 0
        self.weekly_switch_count = 0
        self.last_reset_date = datetime.now().date()
        self.logger = setup_logger('dssms.switch_history')
    
    def add_switch_record(self, record: SwitchRecord):
        """切替記録追加"""
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # 日次・週次カウント更新
        self._update_counters()
        self.logger.info(f"Switch recorded: {record.from_symbol} → {record.to_symbol}")
    
    def _update_counters(self):
        """カウンター更新"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_switch_count = 0
            self.last_reset_date = current_date
        
        self.daily_switch_count += 1
        
        # 週次カウント（過去7日間）
        week_ago = datetime.now() - timedelta(days=7)
        self.weekly_switch_count = len([
            r for r in self.history 
            if r.timestamp >= week_ago and r.success
        ])
    
    def get_recent_switches(self, hours: int = 24) -> List[SwitchRecord]:
        """最近の切替履歴取得"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [r for r in self.history if r.timestamp >= cutoff]

class DSSMSRiskController:
    """DSSMS専用リスク制御（既存RiskManagement拡張）"""
    
    def __init__(self, base_risk_mgmt: RiskManagement, switch_config: Dict[str, Any]):
        self.base_risk = base_risk_mgmt
        self.switch_config = switch_config
        self.logger = setup_logger('dssms.risk_controller')
        
        # 切替制限設定
        self.max_daily_switches = switch_config.get('risk_control', {}).get('max_daily_switches', 3)
        self.max_weekly_switches = switch_config.get('risk_control', {}).get('max_weekly_switches', 10)
        self.drawdown_threshold = switch_config.get('risk_control', {}).get('drawdown_restriction_threshold', 0.05)
    
    def check_switch_allowance(self, history_manager: SwitchHistoryManager, 
                             from_symbol: str, to_symbol: str) -> Tuple[bool, str]:
        """切替許可総合判定"""
        
        # 基本リスク管理チェック
        if not self.base_risk.check_position_size("DSSMS", to_symbol):
            return False, "基本ポジション制限超過"
        
        # 切替頻度制限
        if history_manager.daily_switch_count >= self.max_daily_switches:
            return False, f"日次切替回数制限({self.max_daily_switches}回)"
        
        if history_manager.weekly_switch_count >= self.max_weekly_switches:
            return False, f"週次切替回数制限({self.max_weekly_switches}回)"
        
        # ドローダウン時制限
        if self.base_risk.current_drawdown > self.drawdown_threshold:
            return False, f"ドローダウン時切替制限({self.drawdown_threshold*100:.1f}%超)"
        
        return True, "切替許可"

class FundUpdateScheduler:
    """定期資金更新スケジューラー"""
    
    def __init__(self, switch_config: Dict[str, Any]):
        self.switch_config = switch_config
        self.last_update = datetime.now()
        self.update_interval = timedelta(hours=24)  # 日次更新
        self.logger = setup_logger('dssms.fund_scheduler')
        
        # 資金計算設定
        fund_config = switch_config.get('fund_management', {})
        self.base_ratio = fund_config.get('available_fund_calculation', {}).get('base_ratio', 0.9)
        self.drawdown_reductions = fund_config.get('drawdown_fund_reduction', {})
    
    def should_update_funds(self, market_monitor: MarketConditionMonitor) -> bool:
        """更新タイミング判定"""
        # 定期更新チェック
        if datetime.now() - self.last_update >= self.update_interval:
            return True
        
        # 緊急イベント発生時
        try:
            halt_flag, _ = market_monitor.should_halt_trading()
            if halt_flag:
                return True
        except Exception as e:
            self.logger.warning(f"市場監視チェックエラー: {e}")
        
        return False
    
    def calculate_available_funds(self, base_risk_mgmt: RiskManagement, 
                                current_portfolio_value: float) -> float:
        """利用可能資金計算"""
        
        # ドローダウン率計算
        initial_capital = base_risk_mgmt.total_assets
        drawdown_ratio = max(0, (initial_capital - current_portfolio_value) / initial_capital)
        
        # ドローダウンレベル別資金制限
        if drawdown_ratio >= 0.20:
            available_ratio = self.drawdown_reductions.get('20_percent', 0.2)
        elif drawdown_ratio >= 0.15:
            available_ratio = self.drawdown_reductions.get('15_percent', 0.4)
        elif drawdown_ratio >= 0.10:
            available_ratio = self.drawdown_reductions.get('10_percent', 0.6)
        elif drawdown_ratio >= 0.05:
            available_ratio = self.drawdown_reductions.get('5_percent', 0.8)
        else:
            available_ratio = self.base_ratio
        
        available_funds = current_portfolio_value * available_ratio
        self.logger.info(f"利用可能資金計算: {available_funds:,.0f}円 (制限比率: {available_ratio:.1%})")
        
        return available_funds

class IntelligentSwitchManager:
    """高度な銘柄切替ロジック"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        
        self.logger = setup_logger('dssms.intelligent_switch')
        self.logger.info("IntelligentSwitchManager 初期化開始")
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # 既存コンポーネント統合
        try:
            self.market_monitor = MarketConditionMonitor()
            self.ranking_system = HierarchicalRankingSystem(self.config)
            self.scoring_engine = ComprehensiveScoringEngine()
            self.perfect_order_detector = PerfectOrderDetector()
            self.data_manager = DSSMSDataManager()
            self.logger.info("既存DSSMSコンポーネント初期化成功")
        except Exception as e:
            self.logger.error(f"既存コンポーネント初期化エラー: {e}")
            raise
        
        # リスク管理統合
        try:
            base_risk_mgmt = RiskManagement(total_assets=10000000.0)  # 1000万円想定
            self.risk_controller = DSSMSRiskController(base_risk_mgmt, self.config)
            self.logger.info("リスク管理システム統合成功")
        except Exception as e:
            self.logger.error(f"リスク管理初期化エラー: {e}")
            raise
        
        # 切替専用システム
        self.position_tracker = PositionTracker()
        self.switch_history = SwitchHistoryManager()
        self.fund_scheduler = FundUpdateScheduler(self.config)
        
        # 切替設定
        switch_criteria = self.config.get('switch_criteria', {})
        self.po_breakdown_threshold = switch_criteria.get('perfect_order_breakdown_threshold', 0.7)
        self.score_diff_threshold = switch_criteria.get('score_difference_threshold', 0.15)
        self.min_holding_hours = switch_criteria.get('minimum_holding_period_hours', 4)
        self.confidence_threshold = switch_criteria.get('confidence_threshold', 0.6)
        
        self.logger.info("IntelligentSwitchManager 初期化完了")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "dssms" / "intelligent_switch_config.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"設定ファイル読み込み成功: {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "switch_criteria": {
                "perfect_order_breakdown_threshold": 0.7,
                "score_difference_threshold": 0.15,
                "minimum_holding_period_hours": 4,
                "confidence_threshold": 0.6
            },
            "risk_control": {
                "max_daily_switches": 3,
                "max_weekly_switches": 10,
                "drawdown_restriction_threshold": 0.05
            },
            "fund_management": {
                "available_fund_calculation": {"base_ratio": 0.9},
                "drawdown_fund_reduction": {
                    "5_percent": 0.8,
                    "10_percent": 0.6,
                    "15_percent": 0.4,
                    "20_percent": 0.2
                }
            }
        }
    
    def evaluate_current_position(self, symbol: str) -> Dict[str, Any]:
        """現在ポジションの総合評価"""
        
        try:
            self.logger.info(f"ポジション評価開始: {symbol}")
            
            # 基本情報取得
            holding_period = self.position_tracker.get_holding_period(symbol)
            
            # パーフェクトオーダー状態チェック
            try:
                po_result = self.perfect_order_detector.detect_perfect_order(symbol)
                po_status = po_result.get('perfect_order_detected', False)
                po_confidence = po_result.get('confidence_score', 0.0)
            except Exception as e:
                self.logger.warning(f"パーフェクトオーダー検出エラー: {e}")
                po_status = False
                po_confidence = 0.0
            
            # 総合スコア取得
            try:
                current_score = self.scoring_engine.calculate_composite_score(symbol)
            except Exception as e:
                self.logger.warning(f"スコア計算エラー: {e}")
                current_score = 0.5  # デフォルト中性スコア
            
            # リスク評価
            risk_level = self._assess_risk_level(current_score, po_status, holding_period)
            
            # 推奨アクション決定
            recommendation = self._determine_position_recommendation(
                symbol, current_score, po_status, po_confidence, holding_period
            )
            
            evaluation = {
                'symbol': symbol,
                'current_score': current_score,
                'perfect_order_status': po_status,
                'perfect_order_confidence': po_confidence,
                'holding_period_hours': holding_period,
                'risk_level': risk_level,
                'recommendation': recommendation.value,
                'evaluation_timestamp': datetime.now().isoformat(),
                'min_holding_satisfied': holding_period >= self.min_holding_hours
            }
            
            self.logger.info(f"ポジション評価完了: {symbol} - スコア:{current_score:.3f}, 推奨:{recommendation.value}")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"ポジション評価エラー {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': f"評価エラー: {str(e)}",
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def check_perfect_order_breakdown(self, symbol: str) -> Dict[str, Any]:
        """パーフェクトオーダー崩れ検出（第一判定）"""
        
        try:
            self.logger.info(f"パーフェクトオーダー崩れチェック: {symbol}")
            
            # パーフェクトオーダー状態取得
            po_result = self.perfect_order_detector.detect_perfect_order(symbol)
            po_detected = po_result.get('perfect_order_detected', False)
            confidence = po_result.get('confidence_score', 0.0)
            
            # 崩れ判定
            breakdown_detected = not po_detected or confidence < self.po_breakdown_threshold
            
            # 緊急退場条件チェック
            immediate_exit_required = False
            exit_reason = ""
            
            if breakdown_detected:
                if confidence < 0.3:  # 極端な悪化
                    immediate_exit_required = True
                    exit_reason = "パーフェクトオーダー完全崩壊"
                elif not po_detected:
                    immediate_exit_required = True
                    exit_reason = "パーフェクトオーダー不成立"
            
            result = {
                'symbol': symbol,
                'perfect_order_detected': po_detected,
                'confidence_score': confidence,
                'breakdown_detected': breakdown_detected,
                'immediate_exit_required': immediate_exit_required,
                'exit_reason': exit_reason,
                'threshold_used': self.po_breakdown_threshold,
                'check_timestamp': datetime.now().isoformat()
            }
            
            if breakdown_detected:
                self.logger.warning(f"パーフェクトオーダー崩れ検出: {symbol} - 信頼度:{confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"パーフェクトオーダーチェックエラー {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': f"チェックエラー: {str(e)}",
                'breakdown_detected': False,
                'immediate_exit_required': False
            }
    
    def should_immediate_switch(self, current: str, candidate: str) -> bool:
        """即座切替判定（ハイブリッド方式）"""
        
        try:
            self.logger.info(f"切替判定開始: {current} → {candidate}")
            
            # 【第一判定】パーフェクトオーダー崩れ
            po_breakdown = self.check_perfect_order_breakdown(current)
            if po_breakdown.get('immediate_exit_required', False):
                self.logger.info(f"第一判定: {current} 緊急退場条件満足")
                return True
            
            # 最小保有期間チェック
            holding_period = self.position_tracker.get_holding_period(current)
            if holding_period < self.min_holding_hours:
                self.logger.info(f"最小保有期間未満: {holding_period:.1f}h < {self.min_holding_hours}h")
                return False
            
            # 【第二判定】総合スコア差評価
            try:
                current_score = self.scoring_engine.calculate_composite_score(current)
                candidate_score = self.scoring_engine.calculate_composite_score(candidate)
                
                score_difference = candidate_score - current_score
                
                if score_difference >= self.score_diff_threshold:
                    self.logger.info(f"第二判定: スコア差判定成功 {score_difference:.3f} >= {self.score_diff_threshold}")
                    return True
                
                self.logger.info(f"スコア差不足: {score_difference:.3f} < {self.score_diff_threshold}")
                return False
                
            except Exception as e:
                self.logger.warning(f"スコア比較エラー: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"切替判定エラー: {e}")
            return False
    
    def execute_switch_with_risk_control(self, from_symbol: str, to_symbol: str) -> bool:
        """リスク制御付き切替実行"""
        
        start_time = time.time()
        self.logger.info(f"切替実行開始: {from_symbol} → {to_symbol}")
        
        try:
            # リスク制御チェック
            allowance, reason = self.risk_controller.check_switch_allowance(
                self.switch_history, from_symbol, to_symbol
            )
            
            if not allowance:
                self.logger.warning(f"切替拒否: {reason}")
                return False
            
            # 市場状況確認
            try:
                halt_flag, halt_reason = self.market_monitor.should_halt_trading()
                if halt_flag:
                    self.logger.warning(f"市場状況により切替中止: {halt_reason}")
                    return False
            except Exception as e:
                self.logger.warning(f"市場状況確認エラー: {e}")
            
            # 切替実行（シミュレーション）
            success = self._execute_position_switch(from_symbol, to_symbol)
            
            # 実行時間計算
            execution_time = (time.time() - start_time) * 1000
            
            # 切替記録
            try:
                from_score = self.scoring_engine.calculate_composite_score(from_symbol)
                to_score = self.scoring_engine.calculate_composite_score(to_symbol)
            except:
                from_score, to_score = 0.0, 0.0
            
            switch_record = SwitchRecord(
                timestamp=datetime.now(),
                from_symbol=from_symbol,
                to_symbol=to_symbol,
                reason="intelligent_switch",
                from_score=from_score,
                to_score=to_score,
                market_condition="normal",
                success=success,
                execution_time_ms=execution_time
            )
            
            self.switch_history.add_switch_record(switch_record)
            
            if success:
                self.logger.info(f"切替成功: {from_symbol} → {to_symbol} ({execution_time:.1f}ms)")
            else:
                self.logger.error(f"切替失敗: {from_symbol} → {to_symbol}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"切替実行エラー: {e}")
            return False
    
    def update_available_funds_after_drawdown(self) -> float:
        """ドローダウン後の利用可能資金更新"""
        
        try:
            self.logger.info("利用可能資金更新開始")
            
            # 更新タイミングチェック
            if not self.fund_scheduler.should_update_funds(self.market_monitor):
                # キャッシュ値を返す
                cached_funds = getattr(self, '_cached_available_funds', 9000000.0)
                self.logger.info(f"資金更新スキップ - キャッシュ値使用: {cached_funds:,.0f}円")
                return cached_funds
            
            # 現在のポートフォリオ価値を概算（実際の実装では取引システムから取得）
            current_portfolio_value = self._estimate_current_portfolio_value()
            
            # 利用可能資金計算
            available_funds = self.fund_scheduler.calculate_available_funds(
                self.risk_controller.base_risk, current_portfolio_value
            )
            
            # キャッシュ更新
            self._cached_available_funds = available_funds
            self.fund_scheduler.last_update = datetime.now()
            
            self.logger.info(f"利用可能資金更新完了: {available_funds:,.0f}円")
            return available_funds
            
        except Exception as e:
            self.logger.error(f"資金更新エラー: {e}")
            # デフォルト値を返す
            return 9000000.0  # 900万円
    
    def _assess_risk_level(self, score: float, po_status: bool, holding_period: float) -> str:
        """リスクレベル評価"""
        if score < 0.3 or not po_status:
            return "high"
        elif score < 0.6 or holding_period > 48:  # 2日超保有
            return "medium"
        else:
            return "low"
    
    def _determine_position_recommendation(self, symbol: str, score: float, 
                                         po_status: bool, po_confidence: float, 
                                         holding_period: float) -> SwitchDecision:
        """ポジション推奨決定"""
        
        # 緊急退場条件
        if not po_status or po_confidence < 0.3:
            return SwitchDecision.EMERGENCY_EXIT
        
        # 即座切替条件
        if po_confidence < self.po_breakdown_threshold and holding_period >= self.min_holding_hours:
            return SwitchDecision.IMMEDIATE_SWITCH
        
        # 段階的切替条件
        if score < 0.5 and holding_period >= self.min_holding_hours:
            return SwitchDecision.GRADUAL_SWITCH
        
        return SwitchDecision.NO_SWITCH
    
    def _execute_position_switch(self, from_symbol: str, to_symbol: str) -> bool:
        """実際の切替実行（シミュレーション）"""
        try:
            # 既存ポジション削除
            self.position_tracker.remove_position(from_symbol)
            
            # 新規ポジション追加（仮想実行）
            # 実際の実装では取引APIを呼び出し
            self.position_tracker.add_position(to_symbol, 1000.0, 100)  # 仮想価格・数量
            
            # リスク管理更新
            self.risk_controller.base_risk.update_position("DSSMS", 1, to_symbol)
            
            return True
            
        except Exception as e:
            self.logger.error(f"ポジション切替実行エラー: {e}")
            return False
    
    def _estimate_current_portfolio_value(self) -> float:
        """現在ポートフォリオ価値概算"""
        # 実際の実装では取引システムから取得
        # シミュレーション用の概算値
        base_value = self.risk_controller.base_risk.total_assets
        drawdown_simulation = np.random.uniform(-0.1, 0.05)  # -10%〜+5%のランダム変動
        return base_value * (1 + drawdown_simulation)

# 統合インターフェース
class DSSMSIntelligentSwitchIntegrator:
    """DSSMS統合インターフェース"""
    
    def __init__(self):
        self.switch_manager = IntelligentSwitchManager()
        self.logger = setup_logger('dssms.switch_integrator')
    
    def get_switch_recommendation(self, current_symbol: str, candidate_symbols: List[str]) -> Dict[str, Any]:
        """切替推奨取得"""
        try:
            current_eval = self.switch_manager.evaluate_current_position(current_symbol)
            
            recommendations = []
            for candidate in candidate_symbols:
                should_switch = self.switch_manager.should_immediate_switch(current_symbol, candidate)
                recommendations.append({
                    'candidate': candidate,
                    'recommended': should_switch
                })
            
            return {
                'current_evaluation': current_eval,
                'switch_recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"切替推奨取得エラー: {e}")
            return {'error': str(e)}
    
    def execute_recommended_switch(self, from_symbol: str, to_symbol: str) -> Dict[str, Any]:
        """推奨切替実行"""
        try:
            success = self.switch_manager.execute_switch_with_risk_control(from_symbol, to_symbol)
            
            return {
                'success': success,
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"切替実行エラー: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状況取得"""
        try:
            positions = self.switch_manager.position_tracker.get_current_positions()
            recent_switches = self.switch_manager.switch_history.get_recent_switches(24)
            available_funds = self.switch_manager.update_available_funds_after_drawdown()
            
            return {
                'positions': positions,
                'recent_switches_24h': len(recent_switches),
                'daily_switch_count': self.switch_manager.switch_history.daily_switch_count,
                'available_funds': available_funds,
                'status': 'active',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"システム状況取得エラー: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # テスト実行
    print("DSSMS IntelligentSwitchManager テスト開始")
    
    try:
        manager = IntelligentSwitchManager()
        print("✓ IntelligentSwitchManager 初期化成功")
        
        # テスト銘柄
        test_symbol = "6758"  # ソニーG
        candidate = "6981"    # 村田製作所
        
        # ポジション評価テスト
        evaluation = manager.evaluate_current_position(test_symbol)
        print(f"✓ ポジション評価: {evaluation.get('recommendation', 'N/A')}")
        
        # 切替判定テスト
        should_switch = manager.should_immediate_switch(test_symbol, candidate)
        print(f"✓ 切替判定: {should_switch}")
        
        # 資金更新テスト
        available_funds = manager.update_available_funds_after_drawdown()
        print(f"✓ 利用可能資金: {available_funds:,.0f}円")
        
        print("✅ テスト完了")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
