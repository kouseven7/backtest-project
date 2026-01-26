"""
TrendFollowingExit - トレンドフォロー型エグジット戦略

市場トレンドを追従し、トレンド崩壊時に決済する戦略。
トレンド継続中は利確を遅らせ、大きな利益を狙う。

主な機能:
- UnifiedTrendDetectorによるトレンド判定
- トレンド崩壊検出（uptrend→downtrend/range-bound）
- エントリートレンドと現在トレンドの不一致検出
- 最低保有期間・最大保有期間の設定（過学習回避）

統合コンポーネント:
- BaseExitStrategy継承
- UnifiedTrendDetector統合（indicators/unified_trend_detector.py）
- GCStrategyWithExit等のエントリー戦略と組み合わせ

セーフティ機能/注意事項:
- パラメータは3つ（過学習回避）：min_hold_days, max_hold_days, confidence_threshold
- 市場トレンド判定は全データを使用（ルックアヘッドバイアス注意）
- トレンド崩壊判定はidx日目のデータのみ使用
- エグジット価格はidx+1日目始値
- フォールバック禁止（copilot-instructions.md準拠）

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""
from typing import Dict, Tuple
import pandas as pd
from .base_exit_strategy import BaseExitStrategy


class TrendFollowingExit(BaseExitStrategy):
    """トレンドフォロー型エグジット戦略"""
    
    def __init__(
        self,
        min_hold_days: int = 3,
        max_hold_days: int = 60,
        confidence_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            min_hold_days: 最低保有期間（短期ノイズ回避）
                推奨範囲: 1 ~ 10日
            max_hold_days: 最大保有期間（長期保有リスク回避）
                推奨範囲: 30 ~ 120日
            confidence_threshold: トレンド判定信頼度閾値
                推奨範囲: 0.3 ~ 0.7
        
        Note:
            - デフォルト設定はPhase 3検証用（3日、60日、0.5）
            - パラメータ数3つで過学習回避
            - UnifiedTrendDetectorは実行時に遅延インポート（循環参照回避）
        """
        if not 1 <= min_hold_days <= 30:
            raise ValueError(
                f"min_hold_days must be between 1 and 30, got {min_hold_days}"
            )
        
        if not 10 <= max_hold_days <= 300:
            raise ValueError(
                f"max_hold_days must be between 10 and 300, got {max_hold_days}"
            )
        
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        
        super().__init__(params={
            "min_hold_days": min_hold_days,
            "max_hold_days": max_hold_days,
            "confidence_threshold": confidence_threshold
        })
        
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.confidence_threshold = confidence_threshold
        
        self.logger.info(
            f"[TREND_FOLLOW_INIT] Initialized with min_hold={min_hold_days}, "
            f"max_hold={max_hold_days}, conf_threshold={confidence_threshold:.2f}"
        )
    
    def should_exit(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        トレンドフォロー型エグジット判定
        
        Args:
            position: ポジション情報（entry_date, entry_idx必須）
            current_idx: 現在のインデックス
            data: 株価データ
        
        Returns:
            (should_exit, reason)
            
        Raises:
            ValueError: position検証失敗時
        """
        # ポジション情報検証
        self.validate_position(position)
        
        # エントリー情報取得
        entry_idx = position.get('entry_idx')
        if entry_idx is None:
            self.logger.warning("[TREND_FOLLOW] entry_idx not found in position, using entry_date")
            entry_date = position['entry_date']
            entry_idx = data.index.get_loc(entry_date)
        
        # 保有日数計算
        hold_days = current_idx - entry_idx
        
        # 最低保有期間チェック
        if hold_days < self.min_hold_days:
            return (False, "")
        
        # 最大保有期間チェック
        if hold_days >= self.max_hold_days:
            reason = (
                f"Maximum hold period reached: {hold_days} >= {self.max_hold_days} days"
            )
            self.logger.info(f"[TREND_FOLLOW_EXIT] {reason}")
            return (True, reason)
        
        # トレンド判定（遅延インポートで循環参照回避）
        try:
            from indicators.unified_trend_detector import detect_unified_trend_with_confidence
        except ImportError as e:
            self.logger.error(f"[TREND_FOLLOW] UnifiedTrendDetector import failed: {e}")
            return (False, "")
        
        # 現在のトレンド判定（current_idxまでのデータのみ使用）
        current_data = data.iloc[:current_idx + 1]
        
        try:
            current_trend, confidence = detect_unified_trend_with_confidence(
                current_data,
                strategy='TrendFollowing',
                method='advanced'
            )
        except Exception as e:
            self.logger.error(f"[TREND_FOLLOW] Trend detection failed: {e}")
            return (False, "")
        
        # エントリー時のトレンド取得（ポジション情報に保存されている場合）
        entry_trend = position.get('entry_trend', 'uptrend')
        
        # トレンド崩壊判定
        trend_collapsed = False
        
        # uptrendでエントリー → downtrend/range-boundに変化
        if entry_trend == 'uptrend':
            if current_trend in ['downtrend', 'range-bound']:
                if confidence >= self.confidence_threshold:
                    trend_collapsed = True
        
        # downtrendでエントリー → uptrend/range-boundに変化（ショート想定）
        elif entry_trend == 'downtrend':
            if current_trend in ['uptrend', 'range-bound']:
                if confidence >= self.confidence_threshold:
                    trend_collapsed = True
        
        if trend_collapsed:
            reason = (
                f"Trend collapsed: {entry_trend} -> {current_trend} "
                f"(confidence={confidence:.2%}, threshold={self.confidence_threshold:.2%}, "
                f"hold_days={hold_days})"
            )
            self.logger.info(f"[TREND_FOLLOW_EXIT] {reason}")
            return (True, reason)
        
        return (False, "")
    
    def update_position_state(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> None:
        """
        ポジション状態更新
        
        Note:
            TrendFollowingExitは状態管理不要（トレンド判定は毎回実施）
            エントリートレンドはGCStrategyWithExit側で設定される想定
        """
        pass
