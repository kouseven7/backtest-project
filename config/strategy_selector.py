"""
Module: Strategy Selector
File: strategy_selector.py
Description: 
  3-1-1「StrategySelector クラス設計・実装」
  現在のトレンド判定に基づく戦略選択ロジック
  既存システム（戦略スコアリング、統一トレンド判定）との完全統合

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.strategy_scoring_model
  - config.enhanced_strategy_scoring_model  
  - indicators.unified_trend_detector
  - config.strategy_characteristics_manager
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存モジュールのインポート
try:
    from config.strategy_scoring_model import (
        StrategyScoreCalculator, StrategyScoreManager, StrategyScore, ScoreWeights
    )
    from config.enhanced_strategy_scoring_model import (
        EnhancedStrategyScoreCalculator, EnhancedScoreWeights
    )
    from indicators.unified_trend_detector import UnifiedTrendDetector
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager
    from config.optimized_parameters import OptimizedParameterManager
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class SelectionMethod(Enum):
    """戦略選択手法"""
    TOP_N = "top_n"                    # 上位N個選択
    THRESHOLD = "threshold"            # 閾値ベース選択  
    HYBRID = "hybrid"                  # ハイブリッド（閾値+最大数）
    WEIGHTED = "weighted"              # 重み付き選択
    ADAPTIVE = "adaptive"              # 適応的選択

class TrendType(Enum):
    """トレンドタイプ"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"

@dataclass
class SelectionCriteria:
    """戦略選択基準"""
    method: SelectionMethod = SelectionMethod.HYBRID
    min_score_threshold: float = 0.6           # 最小スコア閾値
    max_strategies: int = 3                    # 最大戦略数
    min_strategies: int = 1                    # 最小戦略数
    confidence_threshold: float = 0.7          # 信頼度閾値
    trend_adaptation_weight: float = 0.3       # トレンド適応重み
    enable_diversification: bool = True        # 多様化有効化
    blacklist_strategies: Set[str] = field(default_factory=set)  # 除外戦略
    whitelist_strategies: Set[str] = field(default_factory=set)  # 許可戦略

@dataclass
class StrategySelection:
    """戦略選択結果"""
    selected_strategies: List[str]
    strategy_scores: Dict[str, float]
    strategy_weights: Dict[str, float]
    selection_reason: str
    trend_analysis: Dict[str, Any]
    confidence_level: float
    total_score: float
    selection_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class StrategySelector:
    """
    戦略選択器
    
    機能:
    1. 現在のトレンド判定に基づく戦略選択
    2. 戦略スコアリングシステムとの統合
    3. 複数の選択手法（トップN、閾値、ハイブリッド）
    4. 選択ルールの抽象化と差し替え可能性
    5. 戦略の多様化と集中のバランス調整
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 base_dir: Optional[str] = None):
        """戦略選択器の初期化"""
        self.base_dir = Path(base_dir) if base_dir else Path("config/strategy_selection")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定の読み込み
        self.config = self._load_config(config_file)
        
        # 既存システムとの連携
        self.score_calculator = StrategyScoreCalculator()
        self.enhanced_calculator = EnhancedStrategyScoreCalculator()
        self.score_manager = StrategyScoreManager()
        self.characteristics_manager = StrategyCharacteristicsManager()
        self.parameter_manager = OptimizedParameterManager()
        
        # キャッシュとパフォーマンス
        self._strategy_cache = {}
        self._trend_cache = {}
        self._last_selection_time = None
        self._selection_history = []
        
        # 利用可能戦略の初期化
        self.available_strategies = self._discover_available_strategies()
        
        logger.info(f"StrategySelector initialized with {len(self.available_strategies)} strategies")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # デフォルト設定
        return {
            "default_selection_criteria": {
                "method": "hybrid",
                "min_score_threshold": 0.6,
                "max_strategies": 3,
                "min_strategies": 1,
                "confidence_threshold": 0.7,
                "trend_adaptation_weight": 0.3,
                "enable_diversification": True
            },
            "strategy_weights": {
                "performance": 0.35,
                "stability": 0.25,
                "risk_adjusted": 0.20,
                "trend_adaptation": 0.15,
                "reliability": 0.05
            },
            "trend_strategy_mapping": {
                "uptrend": ["MomentumInvestingStrategy", "VWAPBreakoutStrategy", "BreakoutStrategy"],
                "downtrend": ["ContrarianStrategy", "VWAPBounceStrategy"],
                "sideways": ["VWAPBounceStrategy", "GCStrategy", "OpeningGapStrategy"]
            },
            "cache_ttl_seconds": 300,
            "enable_logging": True
        }

    def _discover_available_strategies(self) -> List[str]:
        """利用可能な戦略の発見"""
        # 既存の戦略クラスから自動発見
        strategies = [
            "VWAPBounceStrategy",
            "VWAPBreakoutStrategy", 
            "MomentumInvestingStrategy",
            "ContrarianStrategy",
            "GCStrategy",
            "BreakoutStrategy",
            "OpeningGapStrategy"
        ]
        
        # 戦略特性マネージャーから追加で発見
        try:
            available_from_characteristics = self.characteristics_manager.get_available_strategies()
            strategies.extend([s for s in available_from_characteristics if s not in strategies])
        except Exception as e:
            logger.warning(f"Could not load strategies from characteristics manager: {e}")
        
        return strategies

    def select_strategies(self,
                         market_data: pd.DataFrame,
                         ticker: str,
                         criteria: Optional[SelectionCriteria] = None,
                         current_positions: Optional[Dict[str, float]] = None) -> StrategySelection:
        """
        メイン戦略選択メソッド
        
        Parameters:
            market_data (pd.DataFrame): 市場データ
            ticker (str): ティッカーシンボル
            criteria (SelectionCriteria): 選択基準
            current_positions (Dict[str, float]): 現在のポジション
            
        Returns:
            StrategySelection: 選択結果
        """
        start_time = datetime.now()
        
        # デフォルト基準の適用
        if criteria is None:
            criteria = self._create_default_criteria()
        
        try:
            # 1. トレンド分析
            trend_analysis = self._analyze_trend(market_data, ticker)
            current_trend = trend_analysis["trend"]
            confidence = trend_analysis["confidence"]
            
            # 2. 戦略スコア計算
            strategy_scores = self._calculate_strategy_scores(
                market_data, ticker, current_trend, criteria
            )
            
            # 3. 戦略フィルタリング
            filtered_strategies = self._filter_strategies(
                strategy_scores, criteria, current_trend
            )
            
            # 4. 戦略選択
            selected_strategies = self._apply_selection_method(
                filtered_strategies, criteria, current_trend
            )
            
            # 5. 重み計算
            strategy_weights = self._calculate_strategy_weights(
                selected_strategies, strategy_scores, criteria
            )
            
            # 6. 結果作成
            selection_result = StrategySelection(
                selected_strategies=list(selected_strategies.keys()),
                strategy_scores=strategy_scores,
                strategy_weights=strategy_weights,
                selection_reason=self._generate_selection_reason(criteria, current_trend),
                trend_analysis=trend_analysis,
                confidence_level=confidence,
                total_score=sum(selected_strategies.values()),
                metadata={
                    "ticker": ticker,
                    "criteria": criteria.__dict__,
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "available_strategies_count": len(self.available_strategies),
                    "filtered_strategies_count": len(filtered_strategies)
                }
            )
            
            # 7. 履歴保存
            self._save_selection_history(selection_result)
            
            logger.info(f"Strategy selection completed: {len(selected_strategies)} strategies selected for {ticker}")
            return selection_result
            
        except Exception as e:
            logger.error(f"Strategy selection failed for {ticker}: {e}")
            return self._create_fallback_selection(ticker, e)

    def _create_default_criteria(self) -> SelectionCriteria:
        """デフォルト選択基準の作成"""
        default_config = self.config.get("default_selection_criteria", {})
        
        return SelectionCriteria(
            method=SelectionMethod(default_config.get("method", "hybrid")),
            min_score_threshold=default_config.get("min_score_threshold", 0.6),
            max_strategies=default_config.get("max_strategies", 3),
            min_strategies=default_config.get("min_strategies", 1),
            confidence_threshold=default_config.get("confidence_threshold", 0.7),
            trend_adaptation_weight=default_config.get("trend_adaptation_weight", 0.3),
            enable_diversification=default_config.get("enable_diversification", True)
        )

    def _analyze_trend(self, market_data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """トレンド分析の実行"""
        cache_key = f"{ticker}_{hash(str(market_data.tail(10).values.tobytes()))}"
        
        # キャッシュチェック
        if cache_key in self._trend_cache:
            cache_time, cached_result = self._trend_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.config.get("cache_ttl_seconds", 300):
                return cached_result
        
        try:
            # 統一トレンド判定器を使用
            trend_detector = UnifiedTrendDetector(
                data=market_data,
                strategy_name="StrategySelector",
                method="advanced"
            )
            
            # トレンド判定
            trend = trend_detector.detect_trend()
            confidence = trend_detector.get_confidence()
            trend_strength = trend_detector.get_trend_strength()
            
            # 詳細分析
            trend_analysis = {
                "trend": trend,
                "confidence": confidence,
                "strength": trend_strength,
                "reliability": trend_detector.get_reliability_score(),
                "volatility": self._calculate_volatility(market_data),
                "momentum": self._calculate_momentum(market_data),
                "analysis_timestamp": datetime.now(),
                "data_points": len(market_data)
            }
            
            # キャッシュに保存
            self._trend_cache[cache_key] = (datetime.now(), trend_analysis)
            
            return trend_analysis
            
        except Exception as e:
            logger.warning(f"Trend analysis failed for {ticker}: {e}")
            return {
                "trend": "unknown",
                "confidence": 0.5,
                "strength": 0.5,
                "reliability": 0.5,
                "volatility": 0.5,
                "momentum": 0.0,
                "analysis_timestamp": datetime.now(),
                "data_points": len(market_data),
                "error": str(e)
            }

    def _calculate_strategy_scores(self,
                                 market_data: pd.DataFrame,
                                 ticker: str,
                                 current_trend: str,
                                 criteria: SelectionCriteria) -> Dict[str, float]:
        """戦略スコアの計算"""
        strategy_scores = {}
        
        for strategy_name in self.available_strategies:
            try:
                # キャッシュキーの生成
                cache_key = f"{strategy_name}_{ticker}_{current_trend}"
                
                if cache_key in self._strategy_cache:
                    cache_time, cached_score = self._strategy_cache[cache_key]
                    if (datetime.now() - cache_time).total_seconds() < 600:  # 10分キャッシュ
                        strategy_scores[strategy_name] = cached_score
                        continue
                
                # 基本スコア計算
                base_score = self.score_calculator.calculate_strategy_score(
                    strategy_name=strategy_name,
                    ticker=ticker,
                    market_data=market_data
                )
                
                if base_score is None:
                    continue
                
                # 強化スコア計算（トレンド適応度を含む）
                enhanced_score = self.enhanced_calculator.calculate_enhanced_score(
                    strategy_name=strategy_name,
                    ticker=ticker,
                    market_data=market_data,
                    current_trend=current_trend
                )
                
                # 複合スコアの計算
                final_score = self._calculate_composite_score(
                    base_score, enhanced_score, criteria
                )
                
                strategy_scores[strategy_name] = final_score
                
                # キャッシュに保存
                self._strategy_cache[cache_key] = (datetime.now(), final_score)
                
            except Exception as e:
                logger.warning(f"Score calculation failed for {strategy_name}: {e}")
                continue
        
        return strategy_scores

    def _calculate_composite_score(self,
                                 base_score: StrategyScore,
                                 enhanced_score: Optional[float],
                                 criteria: SelectionCriteria) -> float:
        """複合スコアの計算"""
        # 基本スコア
        composite = base_score.total_score
        
        # 強化スコアの重み付け加算
        if enhanced_score is not None:
            trend_weight = criteria.trend_adaptation_weight
            composite = (1 - trend_weight) * composite + trend_weight * enhanced_score
        
        # 信頼度による調整
        if hasattr(base_score, 'confidence'):
            confidence_factor = max(0.5, base_score.confidence)
            composite *= confidence_factor
        
        return min(1.0, max(0.0, composite))

    def _filter_strategies(self,
                          strategy_scores: Dict[str, float],
                          criteria: SelectionCriteria,
                          current_trend: str) -> Dict[str, float]:
        """戦略のフィルタリング"""
        filtered = {}
        
        for strategy_name, score in strategy_scores.items():
            # スコア閾値チェック
            if score < criteria.min_score_threshold:
                continue
            
            # ブラックリストチェック
            if strategy_name in criteria.blacklist_strategies:
                continue
            
            # ホワイトリストチェック
            if criteria.whitelist_strategies and strategy_name not in criteria.whitelist_strategies:
                continue
            
            # トレンド適合度チェック
            if not self._is_strategy_suitable_for_trend(strategy_name, current_trend):
                # 完全に除外ではなく、スコアにペナルティを適用
                score *= 0.7
            
            filtered[strategy_name] = score
        
        return filtered

    def _is_strategy_suitable_for_trend(self, strategy_name: str, trend: str) -> bool:
        """戦略のトレンド適合度チェック"""
        trend_mapping = self.config.get("trend_strategy_mapping", {})
        suitable_strategies = trend_mapping.get(trend, [])
        
        # マッピングが定義されていない場合は適合とみなす
        if not suitable_strategies:
            return True
        
        return strategy_name in suitable_strategies

    def _apply_selection_method(self,
                               filtered_strategies: Dict[str, float],
                               criteria: SelectionCriteria,
                               current_trend: str) -> Dict[str, float]:
        """選択手法の適用"""
        if not filtered_strategies:
            logger.warning("No strategies passed filtering")
            return {}
        
        selected = {}
        
        if criteria.method == SelectionMethod.TOP_N:
            selected = self._select_top_n(filtered_strategies, criteria.max_strategies)
            
        elif criteria.method == SelectionMethod.THRESHOLD:
            selected = self._select_by_threshold(filtered_strategies, criteria.min_score_threshold)
            
        elif criteria.method == SelectionMethod.HYBRID:
            selected = self._select_hybrid(filtered_strategies, criteria)
            
        elif criteria.method == SelectionMethod.WEIGHTED:
            selected = self._select_weighted(filtered_strategies, criteria)
            
        elif criteria.method == SelectionMethod.ADAPTIVE:
            selected = self._select_adaptive(filtered_strategies, criteria, current_trend)
        
        # 最小戦略数の確保
        if len(selected) < criteria.min_strategies:
            selected = self._ensure_minimum_strategies(filtered_strategies, criteria.min_strategies)
        
        return selected

    def _select_top_n(self, strategies: Dict[str, float], n: int) -> Dict[str, float]:
        """上位N個選択"""
        sorted_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_strategies[:n])

    def _select_by_threshold(self, strategies: Dict[str, float], threshold: float) -> Dict[str, float]:
        """閾値ベース選択"""
        return {name: score for name, score in strategies.items() if score >= threshold}

    def _select_hybrid(self, strategies: Dict[str, float], criteria: SelectionCriteria) -> Dict[str, float]:
        """ハイブリッド選択（閾値+最大数制限）"""
        # まず閾値でフィルタ
        threshold_filtered = self._select_by_threshold(strategies, criteria.min_score_threshold)
        
        # 最大数制限を適用
        if len(threshold_filtered) > criteria.max_strategies:
            return self._select_top_n(threshold_filtered, criteria.max_strategies)
        
        return threshold_filtered

    def _select_weighted(self, strategies: Dict[str, float], criteria: SelectionCriteria) -> Dict[str, float]:
        """重み付き選択"""
        # スコアに基づく重み付け選択
        total_score = sum(strategies.values())
        if total_score == 0:
            return {}
        
        # 正規化されたスコアを計算
        normalized_scores = {name: score/total_score for name, score in strategies.items()}
        
        # 上位戦略を重み付きで選択
        return self._select_top_n(normalized_scores, criteria.max_strategies)

    def _select_adaptive(self,
                        strategies: Dict[str, float], 
                        criteria: SelectionCriteria,
                        current_trend: str) -> Dict[str, float]:
        """適応的選択（トレンドと市場状況に応じて調整）"""
        # トレンドの強さに応じて選択数を調整
        trend_strength = 0.7  # TODO: 実際のトレンド強度を取得
        
        if trend_strength > 0.8:
            # 強いトレンド時は集中投資
            max_strategies = min(2, criteria.max_strategies)
        elif trend_strength < 0.4:
            # 弱いトレンド時は分散投資
            max_strategies = criteria.max_strategies
        else:
            # 中程度のトレンド時はバランス
            max_strategies = max(2, criteria.max_strategies - 1)
        
        # ハイブリッド選択を調整された戦略数で実行
        adjusted_criteria = SelectionCriteria(
            method=SelectionMethod.HYBRID,
            min_score_threshold=criteria.min_score_threshold,
            max_strategies=max_strategies,
            min_strategies=criteria.min_strategies
        )
        
        return self._select_hybrid(strategies, adjusted_criteria)

    def _ensure_minimum_strategies(self, strategies: Dict[str, float], min_count: int) -> Dict[str, float]:
        """最小戦略数の確保"""
        if len(strategies) < min_count:
            # 利用可能な全戦略を返す
            return strategies
        
        # 上位から最小数を選択
        return self._select_top_n(strategies, min_count)

    def _calculate_strategy_weights(self,
                                  selected_strategies: Dict[str, float],
                                  all_scores: Dict[str, float],
                                  criteria: SelectionCriteria) -> Dict[str, float]:
        """戦略重みの計算"""
        if not selected_strategies:
            return {}
        
        # スコアベースの重み計算
        total_score = sum(selected_strategies.values())
        if total_score == 0:
            # 等重み配分
            equal_weight = 1.0 / len(selected_strategies)
            return {name: equal_weight for name in selected_strategies}
        
        # スコア比例重み
        weights = {}
        for name, score in selected_strategies.items():
            weights[name] = score / total_score
        
        # 多様化が有効な場合は重みを平滑化
        if criteria.enable_diversification:
            weights = self._apply_diversification(weights)
        
        return weights

    def _apply_diversification(self, weights: Dict[str, float]) -> Dict[str, float]:
        """多様化の適用（重み平滑化）"""
        n_strategies = len(weights)
        equal_weight = 1.0 / n_strategies
        
        # 等重みとスコア重みの加重平均
        diversification_factor = 0.3  # 30%の多様化
        
        diversified_weights = {}
        for name, weight in weights.items():
            diversified_weights[name] = (
                (1 - diversification_factor) * weight + 
                diversification_factor * equal_weight
            )
        
        return diversified_weights

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """ボラティリティ計算"""
        try:
            returns = data['Adj Close'].pct_change().dropna()
            return float(returns.std() * np.sqrt(252))  # 年率ボラティリティ
        except:
            return 0.5  # デフォルト値

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """モメンタム計算"""
        try:
            prices = data['Adj Close']
            momentum = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
            return float(momentum)
        except:
            return 0.0  # デフォルト値

    def _generate_selection_reason(self, criteria: SelectionCriteria, trend: str) -> str:
        """選択理由の生成"""
        return f"Selected using {criteria.method.value} method for {trend} trend with threshold {criteria.min_score_threshold}"

    def _save_selection_history(self, selection: StrategySelection):
        """選択履歴の保存"""
        self._selection_history.append(selection)
        
        # 履歴サイズの制限
        if len(self._selection_history) > 100:
            self._selection_history = self._selection_history[-50:]
        
        self._last_selection_time = selection.selection_timestamp

    def _create_fallback_selection(self, ticker: str, error: Exception) -> StrategySelection:
        """フォールバック選択の作成"""
        logger.warning(f"Creating fallback selection for {ticker} due to error: {error}")
        
        # デフォルト戦略を選択
        fallback_strategies = ["VWAPBounceStrategy", "MomentumInvestingStrategy"]
        
        return StrategySelection(
            selected_strategies=fallback_strategies,
            strategy_scores={s: 0.5 for s in fallback_strategies},
            strategy_weights={s: 1.0/len(fallback_strategies) for s in fallback_strategies},
            selection_reason=f"Fallback selection due to error: {error}",
            trend_analysis={"trend": "unknown", "confidence": 0.5},
            confidence_level=0.5,
            total_score=0.5,
            metadata={"error": str(error), "ticker": ticker}
        )

    # 公開API
    def get_available_strategies(self) -> List[str]:
        """利用可能な戦略一覧の取得"""
        return self.available_strategies.copy()

    def get_selection_history(self, limit: int = 10) -> List[StrategySelection]:
        """選択履歴の取得"""
        return self._selection_history[-limit:]

    def update_strategy_blacklist(self, strategies: List[str]):
        """戦略ブラックリストの更新"""
        self.config.setdefault("blacklist_strategies", set()).update(strategies)
        logger.info(f"Updated strategy blacklist: {strategies}")

    def clear_cache(self):
        """キャッシュのクリア"""
        self._strategy_cache.clear()
        self._trend_cache.clear()
        logger.info("Strategy selector cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報の取得"""
        return {
            "available_strategies": len(self.available_strategies),
            "cache_size": len(self._strategy_cache),
            "trend_cache_size": len(self._trend_cache),
            "selection_history_count": len(self._selection_history),
            "last_selection_time": self._last_selection_time.isoformat() if self._last_selection_time else None,
            "config_summary": {
                "default_method": self.config.get("default_selection_criteria", {}).get("method", "hybrid"),
                "cache_ttl": self.config.get("cache_ttl_seconds", 300)
            }
        }


# ファクトリー関数
def create_strategy_selector(config_file: Optional[str] = None, 
                           base_dir: Optional[str] = None) -> StrategySelector:
    """戦略選択器の作成"""
    return StrategySelector(config_file=config_file, base_dir=base_dir)


# 便利関数
def select_best_strategies_for_trend(market_data: pd.DataFrame,
                                   ticker: str,
                                   max_strategies: int = 3) -> StrategySelection:
    """特定トレンドに最適な戦略を簡単選択"""
    selector = create_strategy_selector()
    criteria = SelectionCriteria(
        method=SelectionMethod.HYBRID,
        max_strategies=max_strategies,
        min_score_threshold=0.6
    )
    return selector.select_strategies(market_data, ticker, criteria)


# エクスポート
__all__ = [
    "StrategySelector",
    "SelectionCriteria", 
    "StrategySelection",
    "SelectionMethod",
    "TrendType",
    "create_strategy_selector",
    "select_best_strategies_for_trend"
]


if __name__ == "__main__":
    # 簡単なテスト
    import pandas as pd
    
    # サンプルデータ作成
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Adj Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    # 戦略選択器のテスト
    selector = create_strategy_selector()
    selection = selector.select_strategies(sample_data, "TEST")
    
    print(f"Selected strategies: {selection.selected_strategies}")
    print(f"Strategy weights: {selection.strategy_weights}")
    print(f"Total score: {selection.total_score}")
    print(f"Confidence: {selection.confidence_level}")
