"""
DSSMS Switch Decision クラス定義
銘柄切替決定結果の標準化されたデータ構造

Author: GitHub Copilot Agent
Created: 2025-08-26
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SwitchDecision:
    """
    銘柄切替決定結果
    
    すべての切替エンジンからの結果を統一化する標準データ構造
    """
    
    # 基本情報
    decision_id: str
    timestamp: datetime
    engine_used: str  # "v2", "legacy", "hybrid", "emergency"
    
    # 実行結果
    success: bool
    symbols_before: List[str]
    symbols_after: List[str]
    switches_count: int
    execution_time_ms: float
    decision_factors: Dict[str, Any]
    market_conditions: Dict[str, Any]
    
    # オプション情報
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    strategy_version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初期化後処理"""
        if self.metadata is None:
            self.metadata = {}
        
        # switches_countの自動計算
        if self.switches_count == 0 and self.symbols_before and self.symbols_after:
            before_set = set(self.symbols_before)
            after_set = set(self.symbols_after)
            self.switches_count = len(before_set.symmetric_difference(after_set))
    
    @property
    def switch_rate(self) -> float:
        """切替率計算"""
        if not self.symbols_before:
            return 0.0
        return self.switches_count / len(self.symbols_before)
    
    @property
    def new_symbols(self) -> List[str]:
        """新規追加銘柄"""
        before_set: set[str] = set(self.symbols_before)
        after_set: set[str] = set(self.symbols_after)
        return list(after_set - before_set)
    
    @property
    def removed_symbols(self) -> List[str]:
        """削除銘柄"""
        before_set: set[str] = set(self.symbols_before)
        after_set: set[str] = set(self.symbols_after)
        return list(before_set - after_set)
    
    @property
    def retained_symbols(self) -> List[str]:
        """継続保持銘柄"""
        before_set: set[str] = set(self.symbols_before)
        after_set: set[str] = set(self.symbols_after)
        return list(before_set & after_set)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "engine_used": self.engine_used,
            "success": self.success,
            "error_message": self.error_message,
            "symbols_before": self.symbols_before,
            "symbols_after": self.symbols_after,
            "switches_count": self.switches_count,
            "execution_time_ms": self.execution_time_ms,
            "confidence_score": self.confidence_score,
            "decision_factors": self.decision_factors,
            "market_conditions": self.market_conditions,
            "strategy_version": self.strategy_version,
            "metadata": self.metadata,
            "derived_metrics": {
                "switch_rate": self.switch_rate,
                "new_symbols": self.new_symbols,
                "removed_symbols": self.removed_symbols,
                "retained_symbols": self.retained_symbols
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwitchDecision':
        """辞書から復元"""
        return cls(
            decision_id=data["decision_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            engine_used=data["engine_used"],
            success=data["success"],
            error_message=data.get("error_message"),
            symbols_before=data["symbols_before"],
            symbols_after=data["symbols_after"],
            switches_count=data["switches_count"],
            execution_time_ms=data["execution_time_ms"],
            confidence_score=data.get("confidence_score"),
            decision_factors=data["decision_factors"],
            market_conditions=data["market_conditions"],
            strategy_version=data.get("strategy_version", "1.0"),
            metadata=data.get("metadata", {})
        )
    
    def is_significant_change(self, min_switch_rate: float = 0.1) -> bool:
        """有意な変更かどうか判定"""
        return self.switch_rate >= min_switch_rate
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリー"""
        return {
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "switches_count": self.switches_count,
            "switch_rate": self.switch_rate,
            "confidence_score": self.confidence_score,
            "engine_used": self.engine_used,
            "significant_change": self.is_significant_change()
        }


def create_mock_switch_decision(
    engine: str = "v2",
    success: bool = True,
    switches_count: int = 1,
    symbols_before: Optional[List[str]] = None,
    symbols_after: Optional[List[str]] = None
) -> SwitchDecision:
    """
    モック用SwitchDecision作成
    テスト・デモ用の標準化されたインスタンス生成
    """
    from uuid import uuid4
    
    if symbols_before is None:
        symbols_before = ["7203", "6758", "9984"]
    
    if symbols_after is None:
        if success and switches_count > 0:
            # 成功時は一部銘柄を変更
            symbols_after = symbols_before.copy()
            if switches_count == 1:
                symbols_after[-1] = "9983"  # 最後の銘柄を変更
            elif switches_count == 2:
                symbols_after[-2:] = ["9983", "8306"]  # 最後の2銘柄を変更
            else:
                symbols_after = ["9983", "8306", "4503"]  # 全銘柄変更
        else:
            symbols_after = symbols_before.copy()
    
    return SwitchDecision(
        decision_id=f"MOCK_{engine.upper()}_{uuid4().hex[:8]}",
        timestamp=datetime.now(),
        engine_used=engine,
        success=success,
        error_message=None if success else f"Mock error for {engine}",
        symbols_before=symbols_before,
        symbols_after=symbols_after,
        switches_count=switches_count,
        execution_time_ms=50.0 + (switches_count * 20.0),
        confidence_score=0.8 if success else 0.3,
        decision_factors={
            "market_volatility": 0.15,
            "trend_strength": 0.7,
            "volume_analysis": 0.6,
            "mock_decision": True
        },
        market_conditions={
            "market_trend": "bullish" if success else "uncertain",
            "volatility_level": "medium",
            "liquidity_status": "good"
        },
        strategy_version="1.0",
        metadata={
            "mock_data": True,
            "test_scenario": f"{engine}_engine_test"
        }
    )


# ファクトリー関数
def create_successful_decision(engine: str = "v2", switches: int = 1) -> SwitchDecision:
    """成功決定作成"""
    return create_mock_switch_decision(engine=engine, success=True, switches_count=switches)


def create_failed_decision(engine: str = "legacy") -> SwitchDecision:
    """失敗決定作成"""
    return create_mock_switch_decision(engine=engine, success=False, switches_count=0)


def create_hybrid_decision(switches: int = 2) -> SwitchDecision:
    """ハイブリッド決定作成"""
    return create_mock_switch_decision(engine="hybrid", success=True, switches_count=switches)
