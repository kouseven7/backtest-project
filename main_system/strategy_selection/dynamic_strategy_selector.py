"""
Module: Dynamic Strategy Selector
File: dynamic_strategy_selector.py
Description:
  Phase 2.2: 動的戦略選択システム統合
  MarketAnalyzerの市場分析結果を利用した動的戦略選択
  
Components:
  - StrategySelector: 戦略選択器
  - EnhancedStrategyScoreCalculator: 強化戦略スコア計算器
  - StrategyCharacteristicsManager: 戦略特性管理
  
Author: imega
Created: 2025-10-16
Modified: 2025-10-16
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# 移動済みモジュールのインポート（main_system内）
try:
    from main_system.strategy_selection.strategy_selector import (
        StrategySelector, SelectionCriteria, StrategySelection, SelectionMethod
    )
except ImportError:
    # フォールバック: 元の場所からインポート
    from config.strategy_selector import (
        StrategySelector, SelectionCriteria, StrategySelection, SelectionMethod
    )

try:
    from main_system.strategy_selection.enhanced_strategy_scoring_model import (
        EnhancedStrategyScoreCalculator, EnhancedScoreWeights
    )
except ImportError:
    # フォールバック: 元の場所からインポート
    from config.enhanced_strategy_scoring_model import (
        EnhancedStrategyScoreCalculator, EnhancedScoreWeights
    )

try:
    from main_system.strategy_selection.strategy_characteristics_manager import (
        StrategyCharacteristicsManager
    )
except ImportError:
    # フォールバック: 元の場所からインポート
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager

logger = setup_logger(__name__)


class StrategySelectionMode(Enum):
    """戦略選択モード"""
    SINGLE_BEST = "single_best"              # 最高スコア戦略のみ
    TOP_N = "top_n"                          # 上位N個
    THRESHOLD_BASED = "threshold_based"      # 閾値ベース
    WEIGHTED_ENSEMBLE = "weighted_ensemble"  # 重み付きアンサンブル
    MARKET_ADAPTIVE = "market_adaptive"      # 市場適応型（デフォルト）


class DynamicStrategySelector:
    """
    動的戦略選択クラス
    
    MarketAnalyzerの市場分析結果を利用して最適な戦略を動的に選択:
    - 市場レジームに応じた戦略選択
    - 戦略スコアリング
    - 最適な重み配分計算
    - 選択信頼度評価
    """
    
    def __init__(
        self,
        selection_mode: StrategySelectionMode = StrategySelectionMode.MARKET_ADAPTIVE,
        min_confidence_threshold: float = 0.35  # Phase 5-A-11暫定: 0.45→0.35（スコア変換調査は別タスク）
    ):
        """
        DynamicStrategySelector初期化
        
        Args:
            selection_mode: 戦略選択モード
            min_confidence_threshold: 最小信頼度閾値 (Phase 5-A: 0.5→0.45に引き下げ)
            
        注意事項（copilot-instructions.md準拠）:
        - 実際のbacktest実行を妨げないこと
        - シグナル生成に影響を与えないこと
        - エラー時はフォールバックして継続
        """
        self.logger = setup_logger(__name__)
        self.selection_mode = selection_mode
        self.min_confidence_threshold = min_confidence_threshold
        
        # コンポーネント初期化（エラーハンドリング付き）
        try:
            self.selector = StrategySelector()
            self.logger.info("StrategySelector initialized")
        except Exception as e:
            self.logger.warning(f"StrategySelector init failed: {e}")
            self.selector = None
        
        try:
            # Phase 5-A-11修正: data_loaderを渡してメタデータ読み込みを有効化
            from config.strategy_characteristics_data_loader import StrategyCharacteristicsDataLoader
            data_loader = StrategyCharacteristicsDataLoader()
            self.score_calculator = EnhancedStrategyScoreCalculator(data_loader)
            self.logger.info("EnhancedStrategyScoreCalculator initialized with data_loader")
        except Exception as e:
            self.logger.warning(f"EnhancedStrategyScoreCalculator init failed: {e}")
            self.score_calculator = None
        
        try:
            self.characteristics_manager = StrategyCharacteristicsManager()
            self.logger.info("StrategyCharacteristicsManager initialized")
        except Exception as e:
            self.logger.warning(f"StrategyCharacteristicsManager init failed: {e}")
            self.characteristics_manager = None
        
        # 利用可能な戦略リスト
        # Phase 5-A-11修正: OpeningGapFixedStrategy除外（メタデータなし、フォールバックスコア問題）
        # Phase B-3完了: OpeningGapStrategy除外（2022-2024データで壊滅的性能: 3.7%勝率, -231% P&L）
        self.available_strategies = [
            'VWAPBreakoutStrategy',
            'MomentumInvestingStrategy',
            'BreakoutStrategy',
            'VWAPBounceStrategy',
            # 'OpeningGapFixedStrategy',  # メタデータ未作成のため除外（copilot-instructions.md準拠）
            'ContrarianStrategy',
            'GCStrategy',
            # 'OpeningGapStrategy'  # Phase B-3完了: 使用不可（2022-2024データで壊滅的性能確認）
        ]
        
        self.logger.info(f"DynamicStrategySelector initialized with mode: {selection_mode.value}")
    
    def select_optimal_strategies(
        self,
        market_analysis: Dict[str, Any],
        stock_data: pd.DataFrame,
        ticker: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        最適戦略選択実行
        
        Args:
            market_analysis: MarketAnalyzerからの市場分析結果
            stock_data: 株価データ
            ticker: 銘柄コード
            
        Returns:
            Dict[str, Any]: 選択結果
                - selected_strategies: 選択された戦略リスト
                - strategy_weights: 戦略重み辞書
                - confidence_level: 信頼度レベル
                - strategy_scores: 戦略スコア詳細
                - selection_rationale: 選択理由
                - status: 'SUCCESS' or 'FAILED'
                - error: エラーメッセージ（失敗時のみ）
        """
        self.logger.info(f"Starting optimal strategy selection for {ticker}")
        
        results = {
            'ticker': ticker,
            'selection_timestamp': pd.Timestamp.now(),
            'market_regime': market_analysis.get('market_regime', 'unknown'),
            'selected_strategies': [],
            'strategy_weights': {},
            'confidence_level': 0.0,
            'strategy_scores': {},
            'selection_rationale': '',
            'components_status': {},
            'status': 'SUCCESS'
        }
        
        try:
            # 1. 戦略スコア計算
            strategy_scores = self._calculate_all_strategy_scores(
                market_analysis, stock_data
            )
            
            # スコア計算失敗チェック
            if not strategy_scores:
                raise ValueError("Strategy scoring failed - no scores calculated")
            
            if all(score == 0.0 for score in strategy_scores.values()):
                raise ValueError("Strategy scoring failed - all scores are zero")
            
            results['strategy_scores'] = strategy_scores
            results['components_status']['scoring'] = 'success'
            
            # Phase 5-A-11デバッグ: strategy_scoresの内容を確認
            self.logger.debug(f"[SCORE_CHECK] strategy_scores before selection: {strategy_scores}")
            
            # 2. 市場レジームベース戦略選択
            selected_strategies = self._select_strategies_by_regime(
                strategy_scores, market_analysis['market_regime']
            )
            
            if not selected_strategies:
                raise ValueError("Strategy selection failed - no strategies selected")
            
            results['selected_strategies'] = selected_strategies
            results['components_status']['selection'] = 'success'
            
            # 3. 戦略重み計算
            strategy_weights = self._calculate_strategy_weights(strategy_scores, selected_strategies)
            
            if not strategy_weights:
                raise ValueError("Weight calculation failed - no weights calculated")
            
            results['strategy_weights'] = strategy_weights
            results['components_status']['weighting'] = 'success'
            
            # 4. 信頼度計算
            confidence_level = self._calculate_confidence(
                strategy_scores, selected_strategies, market_analysis
            )
            results['confidence_level'] = confidence_level
            results['components_status']['confidence'] = 'success'
            
            # 5. 選択理由生成
            selection_rationale = self._generate_selection_rationale(
                selected_strategies, market_analysis, strategy_scores
            )
            results['selection_rationale'] = selection_rationale
            
            # Phase 5-A-11修正: 選択された戦略名を明示的にログ出力
            self.logger.info(
                f"Strategy selection completed - Selected: {len(selected_strategies)}, "
                f"Strategies: {selected_strategies}, "
                f"Confidence: {confidence_level:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"CRITICAL: {ticker} strategy selection failed: {e}")
            results['error'] = str(e)
            results['status'] = 'FAILED'
            results['selected_strategies'] = []
            results['strategy_weights'] = {}
            results['confidence_level'] = 0.0
        
        return results
    
    def _calculate_all_strategy_scores(
        self,
        market_analysis: Dict[str, Any],
        stock_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        全戦略のスコア計算
        
        Args:
            market_analysis: 市場分析結果
            stock_data: 株価データ
            
        Returns:
            Dict[str, float]: 戦略名とスコアの辞書
            
        Raises:
            ValueError: スコア計算器が利用できない、または計算に失敗した場合
        """
        if self.score_calculator is None:
            raise ValueError("EnhancedStrategyScoreCalculator is not available")
        
        strategy_scores = {}
        failed_strategies = []
        
        # EnhancedStrategyScoreCalculatorを使用
        # tickerを取得（market_analysisまたはstock_dataから）
        ticker = market_analysis.get('ticker', 'UNKNOWN')
        
        for strategy_name in self.available_strategies:
            try:
                # calculate_enhanced_strategy_score(strategy_name, ticker, market_data)
                score_result = self.score_calculator.calculate_enhanced_strategy_score(
                    strategy_name=strategy_name,
                    ticker=ticker,
                    market_data=stock_data,
                    use_trend_validation=True,
                    integration_method="adaptive"
                )
                
                # StrategyScoreオブジェクトからtotal_scoreを取得
                if hasattr(score_result, 'total_score'):
                    strategy_scores[strategy_name] = score_result.total_score
                else:
                    self.logger.warning(f"Score result has no total_score for {strategy_name}")
                    failed_strategies.append(strategy_name)
                    strategy_scores[strategy_name] = 0.0
                    
            except Exception as e:
                self.logger.warning(f"Score calculation failed for {strategy_name}: {e}")
                failed_strategies.append(strategy_name)
                strategy_scores[strategy_name] = 0.0
        
        # 全ての戦略でスコア計算が失敗した場合はエラー
        if len(failed_strategies) == len(self.available_strategies):
            raise ValueError(
                f"Score calculation failed for all {len(self.available_strategies)} strategies"
            )
        
        # 半数以上の戦略でスコア計算が失敗した場合は警告
        if len(failed_strategies) > len(self.available_strategies) / 2:
            self.logger.warning(
                f"Score calculation failed for {len(failed_strategies)}/{len(self.available_strategies)} strategies"
            )
        
        return strategy_scores
    
    def _select_strategies_by_regime(
        self,
        strategy_scores: Dict[str, float],
        market_regime: str
    ) -> List[str]:
        """
        市場レジームベース戦略選択
        
        Args:
            strategy_scores: 戦略スコア辞書
            market_regime: 市場レジーム
            
        Returns:
            List[str]: 選択された戦略リスト
        """
        # スコアでソート
        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 市場レジーム別選択ロジック
        # Phase 5-A修正: > を >= に変更（スコアが閾値ちょうどの場合も通過させる）
        if self.selection_mode == StrategySelectionMode.MARKET_ADAPTIVE:
            # 市場レジームに応じて選択数を調整
            if 'strong' in market_regime.lower():
                # 強いトレンド: トップ2戦略
                selected = [s[0] for s in sorted_strategies[:2] if s[1] >= self.min_confidence_threshold]
            elif 'sideways' in market_regime.lower() or 'volatile' in market_regime.lower():
                # レンジ・高ボラ: トップ3戦略（分散）
                selected = [s[0] for s in sorted_strategies[:3] if s[1] >= self.min_confidence_threshold]
            else:
                # 通常トレンド: トップ2-3戦略
                selected = [s[0] for s in sorted_strategies[:2] if s[1] >= self.min_confidence_threshold]
        
        elif self.selection_mode == StrategySelectionMode.SINGLE_BEST:
            selected = [sorted_strategies[0][0]] if sorted_strategies[0][1] >= self.min_confidence_threshold else []
        
        elif self.selection_mode == StrategySelectionMode.TOP_N:
            selected = [s[0] for s in sorted_strategies[:3] if s[1] >= self.min_confidence_threshold]
        
        else:
            # デフォルト: トップ2
            selected = [s[0] for s in sorted_strategies[:2] if s[1] >= self.min_confidence_threshold]
        
        # copilot-instructions.md準拠: フォールバック禁止
        # エラー隠蔽して強制的にテスト継続させるフォールバックは実装しない
        if not selected:
            # スコアが全て閾値未満の場合はエラーとして扱う
            max_score = sorted_strategies[0][1] if sorted_strategies else 0.0
            raise ValueError(
                f"No strategies passed confidence threshold. "
                f"Market regime: {market_regime}, "
                f"Min threshold: {self.min_confidence_threshold}, "
                f"Max score: {max_score:.3f}. "
                f"This indicates strategy scoring failure or extremely unfavorable market conditions. "
                f"Fallback selection is prohibited by copilot-instructions.md."
            )
        
        return selected
    
    def _calculate_strategy_weights(
        self,
        strategy_scores: Dict[str, float],
        selected_strategies: List[str]
    ) -> Dict[str, float]:
        """
        戦略重み計算
        
        Args:
            strategy_scores: 戦略スコア辞書
            selected_strategies: 選択された戦略リスト
            
        Returns:
            Dict[str, float]: 戦略重み辞書（合計1.0）
        """
        if not selected_strategies:
            return {}
        
        # 選択された戦略のスコア取得
        selected_scores = {
            strategy: strategy_scores.get(strategy, 0.0)
            for strategy in selected_strategies
        }
        
        # 合計スコア
        total_score = sum(selected_scores.values())
        
        if total_score == 0:
            # 均等配分
            equal_weight = 1.0 / len(selected_strategies)
            return {strategy: equal_weight for strategy in selected_strategies}
        
        # スコア比例重み
        weights = {
            strategy: score / total_score
            for strategy, score in selected_scores.items()
        }
        
        return weights
    
    def _calculate_confidence(
        self,
        strategy_scores: Dict[str, float],
        selected_strategies: List[str],
        market_analysis: Dict[str, Any]
    ) -> float:
        """
        選択信頼度計算
        
        Args:
            strategy_scores: 戦略スコア辞書
            selected_strategies: 選択された戦略リスト
            market_analysis: 市場分析結果
            
        Returns:
            float: 信頼度レベル (0.0 - 1.0)
        """
        if not selected_strategies:
            return 0.0
        
        # 1. 選択戦略の平均スコア
        selected_scores = [strategy_scores.get(s, 0.0) for s in selected_strategies]
        avg_score = np.mean(selected_scores) if selected_scores else 0.0
        
        # 2. 市場分析の信頼度
        market_confidence = market_analysis.get('confidence_score', 0.5)
        
        # 3. スコア分散（低いほど良い）
        score_variance = np.var(selected_scores) if len(selected_scores) > 1 else 0.0
        variance_penalty = min(score_variance, 0.2)  # 最大0.2のペナルティ
        
        # 総合信頼度
        confidence = (avg_score * 0.5 + market_confidence * 0.4) - variance_penalty
        
        return round(max(0.0, min(1.0, confidence)), 2)
    
    def _generate_selection_rationale(
        self,
        selected_strategies: List[str],
        market_analysis: Dict[str, Any],
        strategy_scores: Dict[str, float]
    ) -> str:
        """
        選択理由生成
        
        Args:
            selected_strategies: 選択された戦略
            market_analysis: 市場分析結果
            strategy_scores: 戦略スコア
            
        Returns:
            str: 選択理由
        """
        regime = market_analysis.get('market_regime', 'unknown')
        
        rationale_parts = [
            f"Market Regime: {regime}",
            f"Selected {len(selected_strategies)} strategies",
        ]
        
        for strategy in selected_strategies:
            score = strategy_scores.get(strategy, 0.0)
            rationale_parts.append(f"  - {strategy}: score={score:.2f}")
        
        return " | ".join(rationale_parts)
    
    def get_selection_summary(self, selection_results: Dict[str, Any]) -> str:
        """
        選択結果サマリー生成
        
        Args:
            selection_results: select_optimal_strategies()の戻り値
            
        Returns:
            str: サマリー文字列
        """
        try:
            summary_lines = [
                f"=== Strategy Selection Summary ===",
                f"Ticker: {selection_results.get('ticker', 'N/A')}",
                f"Timestamp: {selection_results.get('selection_timestamp', 'N/A')}",
                f"Market Regime: {selection_results.get('market_regime', 'N/A')}",
                f"Selected Strategies: {len(selection_results.get('selected_strategies', []))}",
                f"Confidence Level: {selection_results.get('confidence_level', 0.0):.2f}",
                f"\nStrategy Weights:"
            ]
            
            for strategy, weight in selection_results.get('strategy_weights', {}).items():
                score = selection_results.get('strategy_scores', {}).get(strategy, 0.0)
                summary_lines.append(f"  - {strategy}: weight={weight:.2f}, score={score:.2f}")
            
            summary_lines.append(f"\nRationale: {selection_results.get('selection_rationale', 'N/A')}")
            
            return "\n".join(summary_lines)
        
        except Exception as e:
            return f"Summary generation failed: {e}"


# 便利関数：簡易的な戦略選択実行
def select_strategies(
    market_analysis: Dict[str, Any],
    stock_data: pd.DataFrame,
    ticker: str = "UNKNOWN"
) -> Dict[str, Any]:
    """
    便利関数: 簡易的な戦略選択実行
    
    Args:
        market_analysis: MarketAnalyzerからの市場分析結果
        stock_data: 株価データ
        ticker: 銘柄コード
        
    Returns:
        Dict[str, Any]: 戦略選択結果
    """
    selector = DynamicStrategySelector()
    return selector.select_optimal_strategies(market_analysis, stock_data, ticker)


if __name__ == "__main__":
    # テスト実行
    print("DynamicStrategySelector Test")
    print("=" * 50)
    
    try:
        selector = DynamicStrategySelector()
        print(f"✓ DynamicStrategySelector initialized successfully")
        print(f"  - StrategySelector: {'OK' if selector.selector else 'NG'}")
        print(f"  - ScoreCalculator: {'OK' if selector.score_calculator else 'NG'}")
        print(f"  - CharacteristicsManager: {'OK' if selector.characteristics_manager else 'NG'}")
        print(f"  - Selection Mode: {selector.selection_mode.value}")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
