"""
Module: Strategy Scoring Model
File: strategy_scoring_model.py
Description: 
  戦略スコアリングシステム - 複合スコア計算とトレンド適合度評価
  各戦略のトレンド適合度を定量化し、複数指標を組み合わせた複合スコアを提供
  2-1-1「戦略スコアリングシステム」の中核実装

Author: imega
Created: 2025-07-09
Modified: 2025-07-09

Dependencies:
  - json
  - os
  - pandas
  - numpy
  - datetime
  - typing
  - config.strategy_characteristics_manager
  - config.strategy_data_persistence
  - config.strategy_characteristics_data_loader
  - config.optimized_parameters
  - indicators.unified_trend_detector
"""

import json
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple, Set, Literal
from dataclasses import dataclass, field
import threading
from collections import defaultdict
import warnings

# 内部モジュールのインポート
try:
    from .strategy_characteristics_manager import StrategyCharacteristicsManager
    from .strategy_data_persistence import StrategyDataPersistence, StrategyDataIntegrator
    from .strategy_characteristics_data_loader import StrategyCharacteristicsDataLoader
    from .optimized_parameters import OptimizedParameterManager
except ImportError:
    # 直接実行時の対応
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager
    from config.strategy_data_persistence import StrategyDataPersistence, StrategyDataIntegrator
    from config.strategy_characteristics_data_loader import StrategyCharacteristicsDataLoader
    from config.optimized_parameters import OptimizedParameterManager

try:
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError:
    # トレンド判定機能は警告付きでスキップ
    warnings.warn("UnifiedTrendDetector not available - trend adaptation will use simplified logic", UserWarning)
    UnifiedTrendDetector = None

# ロガーの設定
logger = logging.getLogger(__name__)
# DEBUG レベルでログ出力（Phase 5-A デバッグ用）
logger.setLevel(logging.DEBUG)

@dataclass
class ScoreWeights:
    """スコア計算の重み設定"""
    performance: float = 0.35
    stability: float = 0.25
    risk_adjusted: float = 0.20
    trend_adaptation: float = 0.15
    reliability: float = 0.05
    
    def __post_init__(self):
        """重みの合計が1.0になるよう正規化"""
        total = self.performance + self.stability + self.risk_adjusted + self.trend_adaptation + self.reliability
        if abs(total - 1.0) > 0.001:
            factor = 1.0 / total
            self.performance *= factor
            self.stability *= factor
            self.risk_adjusted *= factor
            self.trend_adaptation *= factor
            self.reliability *= factor

@dataclass
class StrategyScore:
    """戦略スコア結果"""
    strategy_name: str
    ticker: str
    total_score: float
    component_scores: Dict[str, float]
    trend_fitness: float
    confidence: float
    metadata: Dict[str, Any]
    calculated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'strategy_name': self.strategy_name,
            'ticker': self.ticker,
            'total_score': self.total_score,
            'component_scores': self.component_scores,
            'trend_fitness': self.trend_fitness,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'calculated_at': self.calculated_at.isoformat()
        }

class StrategyScoreCalculator:
    """戦略スコア計算エンジン"""
    
    def __init__(self, data_loader: StrategyCharacteristicsDataLoader = None):
        """
        初期化
        
        Args:
            data_loader: 戦略特性データローダー（任意）
        """
        self.data_loader = data_loader or StrategyCharacteristicsDataLoader()
        self.weights = ScoreWeights()
        self._cache = {}
        self._cache_expiry = timedelta(hours=1)
        
        # 設定可能な重み設定を初期化
        self._load_weight_config()
        
        logger.info("StrategyScoreCalculator initialized")
    
    def _load_weight_config(self):
        """重み設定を設定ファイルから読み込み"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "scoring_weights.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    weight_config = json.load(f)
                    self.weights = ScoreWeights(**weight_config)
                    logger.info(f"Weight configuration loaded from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load weight configuration: {e}, using defaults")
    
    def calculate_strategy_score(self, strategy_name: str, ticker: str, 
                               market_data: pd.DataFrame = None,
                               trend_context: Dict[str, Any] = None,
                               use_cache: bool = True) -> Optional[StrategyScore]:
        """
        戦略スコアを計算
        
        Args:
            strategy_name: 戦略名
            ticker: ティッカーシンボル
            market_data: 市場データ（トレンド判定用、任意）
            trend_context: トレンドコンテキスト情報
            use_cache: キャッシュを使用するか
            
        Returns:
            StrategyScore: 計算されたスコア、エラー時はNone
        """
        try:
            # キャッシュチェック（market_regimeを含めて異なるレジームで再計算可能にする）
            regime = "unknown"
            if trend_context and isinstance(trend_context, dict):
                regime = trend_context.get('market_regime', 'unknown')
            cache_key = f"{strategy_name}_{ticker}_{regime}"
            if use_cache and cache_key in self._cache:
                cached_score, cached_time = self._cache[cache_key]
                if datetime.now() - cached_time < self._cache_expiry:
                    logger.debug(f"Returning cached score for {cache_key}")
                    return cached_score
            
            # 戦略特性データを取得（修正: メタデータ直接読み込み）
            # 優先順位: 1. 永続化データ（実行履歴あり） 2. メタデータ（特性情報のみ）
            strategy_data = self.data_loader.load_strategy_data(
                strategy_name, tickers=[ticker], include_metadata=True
            )
            
            ticker_data = None
            
            # 永続化データが存在する場合
            if strategy_data and ticker in strategy_data:
                ticker_data = strategy_data[ticker]
                logger.debug(f"Using persistence data for {strategy_name}, ticker {ticker}")
            else:
                # フォールバック: メタデータから直接読み込み
                logger.info(f"Persistence data not found for {strategy_name}, loading from metadata")
                
                # Phase 5-A: キャッシュ無効化 + parameters読み込みスキップ
                from config.strategy_characteristics_data_loader import LoadOptions
                load_options = LoadOptions(
                    use_cache=False,  # キャッシュ無効化
                    include_parameters=False  # parameters_data の影響を排除
                )
                metadata = self.data_loader.load_strategy_characteristics(strategy_name, options=load_options)
                
                if not metadata:
                    logger.warning(f"No metadata available for strategy {strategy_name}")
                    return None
                
                # メタデータを ticker_data 形式に変換（trend_contextをmarket_analysisとして渡す）
                ticker_data = self._convert_metadata_to_ticker_data(metadata, ticker, market_analysis=trend_context, market_data=market_data)
                logger.debug(f"Using metadata for {strategy_name}, ticker {ticker}")
            
            # 各コンポーネントスコアを計算
            component_scores = self._calculate_component_scores(
                ticker_data, market_data, trend_context
            )
            
            # トレンド適合度を計算
            trend_fitness = self._calculate_trend_fitness(
                strategy_name, ticker_data, market_data, trend_context
            )
            
            # 信頼度を計算
            confidence = self._calculate_confidence(ticker_data, component_scores)
            
            # 総合スコアを計算
            total_score = self._calculate_total_score(component_scores, trend_fitness)
            
            # スコアオブジェクトを作成
            score = StrategyScore(
                strategy_name=strategy_name,
                ticker=ticker,
                total_score=total_score,
                component_scores=component_scores,
                trend_fitness=trend_fitness,
                confidence=confidence,
                metadata={
                    'data_points': len(ticker_data.get('performance_history', [])),
                    'last_updated': ticker_data.get('last_updated'),
                    'weights_used': {
                        'performance': self.weights.performance,
                        'stability': self.weights.stability,
                        'risk_adjusted': self.weights.risk_adjusted,
                        'trend_adaptation': self.weights.trend_adaptation,
                        'reliability': self.weights.reliability
                    }
                },
                calculated_at=datetime.now()
            )
            
            # キャッシュに保存
            if use_cache:
                self._cache[cache_key] = (score, datetime.now())
            
            logger.debug(f"Calculated score for {strategy_name}_{ticker}: {total_score:.3f}")
            return score
            
        except Exception as e:
            logger.error(f"Error calculating score for {strategy_name}_{ticker}: {e}")
            return None
    
    def _calculate_component_scores(self, ticker_data: Dict[str, Any], 
                                  market_data: pd.DataFrame = None,
                                  trend_context: Dict[str, Any] = None) -> Dict[str, float]:
        """各コンポーネントスコアを計算"""
        scores = {}
        
        try:
            # パフォーマンススコア
            scores['performance'] = self._calculate_performance_score(ticker_data)
            
            # 安定性スコア
            scores['stability'] = self._calculate_stability_score(ticker_data)
            
            # リスク調整スコア
            scores['risk_adjusted'] = self._calculate_risk_adjusted_score(ticker_data)
            
            # 信頼性スコア
            scores['reliability'] = self._calculate_reliability_score(ticker_data)
            
            logger.debug(f"Component scores calculated: {scores}")
            
        except Exception as e:
            logger.error(f"Error calculating component scores: {e}")
            # デフォルト値で埋める
            scores = {
                'performance': 0.5,
                'stability': 0.5,
                'risk_adjusted': 0.5,
                'reliability': 0.5
            }
        
        return scores
    
    def _calculate_performance_score(self, ticker_data: Dict[str, Any]) -> float:
        """パフォーマンススコアを計算"""
        try:
            performance_metrics = ticker_data.get('performance_metrics', {})
            
            # DEBUG: performance_metrics の内容をログ出力
            logger.debug(f"[SCORE_DEBUG] performance_metrics: {performance_metrics}")
            
            # 主要メトリクスを取得
            total_return = performance_metrics.get('total_return', 0.0)
            win_rate = performance_metrics.get('win_rate', 0.5)
            profit_factor = performance_metrics.get('profit_factor', 1.0)
            
            # DEBUG: 抽出された値をログ出力
            logger.debug(f"[SCORE_DEBUG] total_return={total_return}, win_rate={win_rate}, profit_factor={profit_factor}")
            
            # 正規化（0-1スケール）
            # Phase 5-A修正: total_returnが既にパーセント形式(15.0 = 15%)か比率形式(0.15 = 15%)かを判断
            # 通常、10以上なら percent形式、0.0~1.0なら比率形式と判断
            if abs(total_return) >= 2.0:
                # パーセント形式（例: 15.0 = 15%）
                return_score = min(max(total_return / 100.0, 0), 1)  # 100%を上限
            else:
                # 比率形式（例: 0.15 = 15%）or 小さいexpectancy値（例: 0.04）
                # expectancy値の場合、より大きくスケール（1トレード=0.04なら、年間250トレードで10%程度）
                if abs(total_return) < 0.2:  # expectancyっぽい小さい値
                    return_score = min(max(total_return * 10, 0), 1)  # 0.04 * 10 = 0.4
                else:
                    return_score = min(max(total_return, 0), 1)  # 既に比率形式
            
            win_rate_score = win_rate
            profit_score = min(max((profit_factor - 1.0) / 2.0, 0), 1)  # 3.0を上限
            
            # DEBUG: 正規化後のスコアをログ出力
            logger.debug(f"[SCORE_DEBUG] return_score={return_score}, win_rate_score={win_rate_score}, profit_score={profit_score}")
            
            # 加重平均
            performance_score = (return_score * 0.4 + win_rate_score * 0.3 + profit_score * 0.3)
            
            # DEBUG: 最終スコアをログ出力
            logger.debug(f"[SCORE_DEBUG] final performance_score={performance_score}")
            
            return max(min(performance_score, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Error calculating performance score: {e}")
            return 0.5
    
    def _calculate_stability_score(self, ticker_data: Dict[str, Any]) -> float:
        """安定性スコアを計算"""
        try:
            performance_metrics = ticker_data.get('performance_metrics', {})
            
            # ボラティリティとドローダウンから安定性を評価
            volatility = performance_metrics.get('volatility', 0.2)
            max_drawdown = abs(performance_metrics.get('max_drawdown', -0.1))
            
            # 低いほど良い指標なので反転
            volatility_score = max(1.0 - (volatility / 0.5), 0)  # 50%を基準
            drawdown_score = max(1.0 - (max_drawdown / 0.3), 0)  # 30%を基準
            
            # 加重平均
            stability_score = (volatility_score * 0.6 + drawdown_score * 0.4)
            
            return max(min(stability_score, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Error calculating stability score: {e}")
            return 0.5
    
    def _calculate_risk_adjusted_score(self, ticker_data: Dict[str, Any]) -> float:
        """リスク調整スコアを計算"""
        try:
            performance_metrics = ticker_data.get('performance_metrics', {})
            
            # シャープレシオとソルティノレシオ
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            sortino_ratio = performance_metrics.get('sortino_ratio', 0.0)
            
            # 正規化（2.0を上限として設定）
            sharpe_score = min(max(sharpe_ratio / 2.0, 0), 1)
            sortino_score = min(max(sortino_ratio / 2.0, 0), 1)
            
            # 加重平均
            risk_adjusted_score = (sharpe_score * 0.5 + sortino_score * 0.5)
            
            return max(min(risk_adjusted_score, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Error calculating risk-adjusted score: {e}")
            return 0.5
    
    def _calculate_reliability_score(self, ticker_data: Dict[str, Any]) -> float:
        """信頼性スコアを計算"""
        try:
            # データの新しさと完整性を評価
            last_updated = ticker_data.get('last_updated')
            performance_history = ticker_data.get('performance_history', [])
            
            # データ新しさスコア
            freshness_score = 0.5
            if last_updated:
                try:
                    update_time = datetime.fromisoformat(last_updated)
                    days_old = (datetime.now() - update_time).days
                    freshness_score = max(1.0 - (days_old / 30.0), 0)  # 30日を基準
                except:
                    pass
            
            # データ量スコア
            data_volume_score = min(len(performance_history) / 100.0, 1.0)  # 100件を満点
            
            # 加重平均
            reliability_score = (freshness_score * 0.6 + data_volume_score * 0.4)
            
            return max(min(reliability_score, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Error calculating reliability score: {e}")
            return 0.5
    
    def _calculate_trend_fitness(self, strategy_name: str, ticker_data: Dict[str, Any],
                               market_data: pd.DataFrame = None,
                               trend_context: Dict[str, Any] = None) -> float:
        """トレンド適合度を計算"""
        try:
            # トレンドコンテキストがある場合
            if trend_context:
                current_trend = trend_context.get('current_trend', 'neutral')
                trend_strength = trend_context.get('trend_strength', 0.5)
            elif market_data is not None and UnifiedTrendDetector is not None:
                # UnifiedTrendDetectorを使用してトレンド判定
                try:
                    detector = UnifiedTrendDetector(market_data, strategy_name=strategy_name)
                    current_trend = detector.detect_trend()
                    trend_strength = detector.get_confidence_score()
                except Exception as e:
                    logger.warning(f"Failed to detect trend using UnifiedTrendDetector: {e}")
                    current_trend = 'neutral'
                    trend_strength = 0.5
            else:
                # デフォルト値
                current_trend = 'neutral'
                trend_strength = 0.5
            
            # 戦略のトレンド適性データを取得
            trend_suitability = ticker_data.get('trend_suitability', {})
            strategy_fitness = trend_suitability.get(current_trend, 0.5)
            
            # トレンド強度で重み付け
            trend_fitness = strategy_fitness * trend_strength + 0.5 * (1 - trend_strength)
            
            return max(min(trend_fitness, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Error calculating trend fitness: {e}")
            return 0.5
    
    def _calculate_confidence(self, ticker_data: Dict[str, Any], 
                            component_scores: Dict[str, float]) -> float:
        """スコアの信頼度を計算"""
        try:
            # データ品質要因
            data_completeness = self._assess_data_completeness(ticker_data)
            score_consistency = self._assess_score_consistency(component_scores)
            
            confidence = (data_completeness * 0.6 + score_consistency * 0.4)
            return max(min(confidence, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def _convert_metadata_to_ticker_data(self, metadata: Dict[str, Any], ticker: str, market_analysis: Dict[str, Any] = None, market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        メタデータを ticker_data 形式に変換
        
        Args:
            metadata: 戦略メタデータ (v1.0 or v2.0 schema)
            ticker: ティッカーシンボル
            market_analysis: 市場分析結果（オプショナル、デバッグ用）
            market_data: 株価データ（動的スコア調整用、オプショナル）
            
        Returns:
            Dict[str, Any]: ticker_data 形式のデータ
        """
        try:
            # DEBUG: メソッド呼び出しを確認
            strategy_name = metadata.get('strategy_id', metadata.get('strategy_name', 'unknown'))
            
            # DEBUG: 市場レジーム情報の受信確認（Step 2追加）
            if market_analysis:
                regime = market_analysis.get('market_regime', 'UNKNOWN')
                logger.info(
                    f"[DEBUG_METADATA] Strategy: {strategy_name}, "
                    f"Received regime: {regime}"
                )
            else:
                logger.info(
                    f"[DEBUG_METADATA] Strategy: {strategy_name}, "
                    f"No market_analysis received"
                )
            
            logger.debug(f"[METADATA_DEBUG] Converting metadata for strategy: {strategy_name}")
            logger.debug(f"[METADATA_DEBUG] Metadata keys: {list(metadata.keys())[:10]}")  # 最初の10キーのみ
            
            # スキーマバージョン判定
            schema_version = metadata.get('schema_version', '1.0')
            logger.debug(f"[METADATA_DEBUG] Detected schema_version: {schema_version}")
            
            # performance_metrics の抽出（スキーマ統一対応）
            performance_metrics = metadata.get('performance_metrics', {})
            logger.debug(f"[METADATA_DEBUG] Direct performance_metrics: {performance_metrics}")
            
            # v1.0 スキーマの場合、performance_metrics がない場合がある
            if not performance_metrics and 'trend_adaptability' in metadata:
                # trend_adaptability から performance_metrics を生成
                trend_data = metadata.get('trend_adaptability', {})
                
                # 市場レジームに応じたデータ選択ロジック (Step 5追加)
                current_regime = 'uptrend'  # デフォルト
                
                if market_analysis:
                    regime_str = market_analysis.get('market_regime', 'uptrend')
                    
                    if isinstance(regime_str, str):
                        regime_lower = regime_str.lower()
                        
                        # レジーム判定とキー名マッピング
                        if 'downtrend' in regime_lower or 'down' in regime_lower:
                            current_regime = 'downtrend'
                        elif 'sideways' in regime_lower or 'range' in regime_lower or 'neutral' in regime_lower:
                            current_regime = 'range-bound'  # メタデータのキー名に合わせる
                        else:
                            current_regime = 'uptrend'
                    
                    logger.info(
                        f"[REGIME_ADAPTIVE] Strategy: {strategy_name}, "
                        f"Market regime: {regime_str} -> Using: {current_regime} data"
                    )
                
                # 市場レジームに応じたデータを取得
                regime_data = trend_data.get(current_regime, {})
                
                # フォールバック: current_regimeのデータがない場合はuptrendを使用
                if not regime_data:
                    logger.warning(
                        f"[REGIME_ADAPTIVE] No {current_regime} data for {strategy_name}, "
                        f"falling back to uptrend"
                    )
                    regime_data = trend_data.get('uptrend', {})
                
                # DEBUG: 抽出元データをログ出力
                logger.debug(f"[METADATA_DEBUG] schema_version={schema_version}, regime_data keys={list(regime_data.keys())}")
                
                if schema_version == '2.0':
                    # v2.0: performance_metrics キーを使用
                    perf = regime_data.get('performance_metrics', {})
                else:
                    # v1.0: 直接キーを使用
                    perf = regime_data
                
                # DEBUG: 抽出したデータをログ出力
                logger.debug(f"[METADATA_DEBUG] perf data: {perf}")
                
                # Phase 5-A修正: expectancy も total_return の候補として考慮
                # avg_return が 0.0 の場合、expectancy を使用
                avg_return = perf.get('avg_return', 0.0)
                expected_return = perf.get('expected_return', avg_return)
                expectancy = perf.get('expectancy', 0.0)
                
                # 優先順位: expected_return > expectancy (avg_return が 0 の場合) > avg_return
                if expected_return != 0.0:
                    total_return = expected_return
                elif avg_return == 0.0 and expectancy != 0.0:
                    # avg_return が 0 の場合、expectancy を代替として使用（通常は小さい値）
                    # expectancy は1トレードあたりの期待値なので、適切にスケール
                    total_return = expectancy * 100  # 仮に100トレードとして年換算
                    logger.debug(f"[METADATA_DEBUG] Using expectancy {expectancy} as total_return: {total_return}")
                else:
                    total_return = avg_return
                
                performance_metrics = {
                    'total_return': total_return,
                    'win_rate': perf.get('win_rate', 0.6),
                    'profit_factor': perf.get('profit_factor', 1.5),
                    'volatility': perf.get('volatility', 0.15),
                    'max_drawdown': perf.get('max_drawdown', -0.1),
                    'sharpe_ratio': perf.get('sharpe_ratio', 1.0)
                }
                
                # DEBUG: 生成された performance_metrics をログ出力
                logger.debug(f"[METADATA_DEBUG] generated performance_metrics: {performance_metrics}")
                
                # 動的スコア調整: market_data（株価データ）を使って固定値を調整
                if market_data is not None and len(market_data) >= 30:
                    try:
                        # ルックアヘッドバイアス防止: iloc[-2]以前のデータを使用
                        recent_data = market_data.iloc[:-1].tail(30)
                        
                        if len(recent_data) >= 30:
                            # 日次リターン計算（Adj Closeを優先、なければClose）
                            price_col = 'Adj Close' if 'Adj Close' in recent_data.columns else 'Close'
                            daily_returns = recent_data[price_col].pct_change().dropna()
                            
                            if len(daily_returns) > 0:
                                # 直近30日の平均リターンと標準偏差
                                avg_return_recent = daily_returns.mean()
                                volatility_recent = daily_returns.std()
                                
                                # 固定値を取得
                                base_win_rate = performance_metrics['win_rate']
                                base_sharpe = performance_metrics['sharpe_ratio']
                                
                                # win_rate調整: base_win_rate + avg_return_recent * 2.0 （上限1.0、下限0.0）
                                adjusted_win_rate = base_win_rate + avg_return_recent * 2.0
                                adjusted_win_rate = min(1.0, max(0.0, adjusted_win_rate))
                                
                                # sharpe_ratio調整: base_sharpe + avg_return_recent / volatility_recent * 0.3
                                if volatility_recent > 0:
                                    adjusted_sharpe = base_sharpe + (avg_return_recent / volatility_recent) * 0.3
                                else:
                                    adjusted_sharpe = base_sharpe
                                
                                # 調整値を適用
                                performance_metrics['win_rate'] = adjusted_win_rate
                                performance_metrics['sharpe_ratio'] = adjusted_sharpe
                                
                                # ログ出力
                                logger.info(
                                    f"[DEBUG_DYNAMIC_SCORE] Strategy: {strategy_name}, Ticker: {ticker}, "
                                    f"Regime: {current_regime}, "
                                    f"win_rate: {base_win_rate:.3f} -> {adjusted_win_rate:.3f} "
                                    f"(avg_return_recent: {avg_return_recent:.4f}), "
                                    f"sharpe_ratio: {base_sharpe:.3f} -> {adjusted_sharpe:.3f} "
                                    f"(volatility_recent: {volatility_recent:.4f})"
                                )
                            else:
                                logger.debug(f"[DEBUG_DYNAMIC_SCORE] No valid daily returns for {ticker}")
                        else:
                            logger.debug(f"[DEBUG_DYNAMIC_SCORE] Insufficient recent data for {ticker} (got {len(recent_data)} rows)")
                    
                    except Exception as e:
                        # 例外発生時は固定値をそのまま使用
                        logger.warning(
                            f"[DEBUG_DYNAMIC_SCORE] Failed to calculate dynamic adjustment for {ticker}: {e}, "
                            f"using base values"
                        )
                else:
                    if market_data is None:
                        logger.debug(f"[DEBUG_DYNAMIC_SCORE] No market_data provided for {ticker}, using base values")
                    else:
                        logger.debug(f"[DEBUG_DYNAMIC_SCORE] Insufficient market_data rows for {ticker} (got {len(market_data)} rows), using base values")
            
            # ticker_data 形式に変換
            ticker_data = {
                'strategy_name': metadata.get('strategy_id', metadata.get('strategy_name', 'unknown')),
                'ticker': ticker,
                'performance_metrics': performance_metrics,
                'trend_adaptability': metadata.get('trend_adaptability', {}),
                'volatility_adaptability': metadata.get('volatility_adaptability', {}),
                'risk_profile': metadata.get('risk_profile', {}),
                'last_updated': metadata.get('last_updated', datetime.now().isoformat()),
                'performance_history': [],  # メタデータには履歴がない
                'data_source': 'metadata',
                'schema_version': schema_version
            }
            
            return ticker_data
            
        except Exception as e:
            logger.error(f"Error converting metadata to ticker_data: {e}")
            # 最小限のデータを返す
            return {
                'strategy_name': metadata.get('strategy_id', metadata.get('strategy_name', 'unknown')),
                'ticker': ticker,
                'performance_metrics': {},
                'data_source': 'metadata_fallback'
            }
    
    def _assess_data_completeness(self, ticker_data: Dict[str, Any]) -> float:
        """データ完整性を評価"""
        required_fields = ['performance_metrics', 'last_updated', 'performance_history']
        available_fields = sum(1 for field in required_fields if field in ticker_data)
        
        # メタデータソースの場合は performance_history がないのは正常
        if ticker_data.get('data_source') == 'metadata':
            required_fields = ['performance_metrics', 'last_updated']
            available_fields = sum(1 for field in required_fields if field in ticker_data)
        
        return available_fields / len(required_fields)
    
    def _assess_score_consistency(self, component_scores: Dict[str, float]) -> float:
        """スコア一貫性を評価"""
        scores = list(component_scores.values())
        if len(scores) < 2:
            return 0.5
        
        # 標準偏差ベースの一貫性評価
        std_dev = np.std(scores)
        consistency = max(1.0 - (std_dev * 2), 0)  # 標準偏差が大きいほど一貫性が低い
        return consistency
    
    def _calculate_total_score(self, component_scores: Dict[str, float], 
                             trend_fitness: float) -> float:
        """総合スコアを計算"""
        try:
            total_score = (
                component_scores.get('performance', 0.5) * self.weights.performance +
                component_scores.get('stability', 0.5) * self.weights.stability +
                component_scores.get('risk_adjusted', 0.5) * self.weights.risk_adjusted +
                trend_fitness * self.weights.trend_adaptation +
                component_scores.get('reliability', 0.5) * self.weights.reliability
            )
            
            return max(min(total_score, 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating total score: {e}")
            return 0.5
    
    def calculate_batch_scores(self, strategies: List[str], tickers: List[str],
                             market_data_dict: Dict[str, pd.DataFrame] = None,
                             trend_context_dict: Dict[str, Dict[str, Any]] = None) -> Dict[str, Dict[str, StrategyScore]]:
        """バッチでスコアを計算"""
        logger.info(f"Starting batch score calculation for {len(strategies)} strategies, {len(tickers)} tickers")
        
        results = {}
        total_combinations = len(strategies) * len(tickers)
        processed = 0
        
        for strategy in strategies:
            results[strategy] = {}
            for ticker in tickers:
                market_data = market_data_dict.get(ticker) if market_data_dict else None
                trend_context = trend_context_dict.get(ticker) if trend_context_dict else None
                
                score = self.calculate_strategy_score(
                    strategy, ticker, market_data, trend_context
                )
                
                if score:
                    results[strategy][ticker] = score
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Batch processing: {processed}/{total_combinations} completed")
        
        logger.info(f"Batch score calculation completed: {processed} combinations processed")
        return results
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._cache.clear()
        logger.info("Score cache cleared")

class StrategyScoreReporter:
    """戦略スコアレポート生成クラス"""
    
    def __init__(self, output_dir: str = None):
        """
        初期化
        
        Args:
            output_dir: レポート出力ディレクトリ
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "strategy_scoring")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"StrategyScoreReporter initialized with output_dir: {output_dir}")
    
    def generate_score_report(self, scores: Dict[str, Dict[str, StrategyScore]], 
                            report_name: str = None) -> str:
        """
        スコアレポートを生成
        
        Args:
            scores: 戦略スコア辞書
            report_name: レポート名（任意）
            
        Returns:
            str: 生成されたレポートファイルパス
        """
        if report_name is None:
            report_name = f"strategy_score_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # JSON形式でレポート保存
            json_path = os.path.join(self.output_dir, f"{report_name}.json")
            self._save_json_report(scores, json_path)
            
            # CSV形式でサマリー保存
            csv_path = os.path.join(self.output_dir, f"{report_name}_summary.csv")
            self._save_csv_summary(scores, csv_path)
            
            # Markdown形式でレポート保存
            md_path = os.path.join(self.output_dir, f"{report_name}.md")
            self._save_markdown_report(scores, md_path)
            
            logger.info(f"Score report generated: {json_path}, {csv_path}, {md_path}")
            return json_path
            
        except Exception as e:
            logger.error(f"Error generating score report: {e}")
            return None
    
    def _save_json_report(self, scores: Dict[str, Dict[str, StrategyScore]], file_path: str):
        """JSON形式でレポート保存"""
        report_data = {}
        for strategy, ticker_scores in scores.items():
            report_data[strategy] = {}
            for ticker, score in ticker_scores.items():
                report_data[strategy][ticker] = score.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _save_csv_summary(self, scores: Dict[str, Dict[str, StrategyScore]], file_path: str):
        """CSV形式でサマリー保存"""
        rows = []
        for strategy, ticker_scores in scores.items():
            for ticker, score in ticker_scores.items():
                rows.append({
                    'strategy': strategy,
                    'ticker': ticker,
                    'total_score': score.total_score,
                    'trend_fitness': score.trend_fitness,
                    'confidence': score.confidence,
                    'performance_score': score.component_scores.get('performance', 0),
                    'stability_score': score.component_scores.get('stability', 0),
                    'risk_adjusted_score': score.component_scores.get('risk_adjusted', 0),
                    'reliability_score': score.component_scores.get('reliability', 0),
                    'calculated_at': score.calculated_at.isoformat()
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    def _save_markdown_report(self, scores: Dict[str, Dict[str, StrategyScore]], file_path: str):
        """Markdown形式でレポート保存"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# Strategy Scoring Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 戦略別サマリー
            f.write("## Strategy Summary\n\n")
            for strategy, ticker_scores in scores.items():
                avg_score = np.mean([score.total_score for score in ticker_scores.values()])
                f.write(f"### {strategy}\n")
                f.write(f"- Average Score: {avg_score:.3f}\n")
                f.write(f"- Tickers Analyzed: {len(ticker_scores)}\n\n")
            
            # 詳細テーブル
            f.write("## Detailed Scores\n\n")
            f.write("| Strategy | Ticker | Total Score | Trend Fitness | Confidence | Performance | Stability | Risk-Adjusted |\n")
            f.write("|----------|---------|-------------|---------------|------------|-------------|-----------|---------------|\n")
            
            for strategy, ticker_scores in scores.items():
                for ticker, score in ticker_scores.items():
                    f.write(f"| {strategy} | {ticker} | {score.total_score:.3f} | {score.trend_fitness:.3f} | "
                           f"{score.confidence:.3f} | {score.component_scores.get('performance', 0):.3f} | "
                           f"{score.component_scores.get('stability', 0):.3f} | "
                           f"{score.component_scores.get('risk_adjusted', 0):.3f} |\n")

class StrategyScoreManager:
    """戦略スコア管理の統合クラス"""
    
    def __init__(self, base_path: str = None):
        """
        初期化
        
        Args:
            base_path: ベースパス（任意）
        """
        # データローダーを初期化
        self.data_loader = StrategyCharacteristicsDataLoader()
        
        # スコア計算器を初期化
        self.calculator = StrategyScoreCalculator(self.data_loader)
        
        # レポーター初期化
        self.reporter = StrategyScoreReporter()
        
        logger.info("StrategyScoreManager initialized")
    
    def calculate_and_report_scores(self, strategies: List[str], tickers: List[str],
                                  market_data_dict: Dict[str, pd.DataFrame] = None,
                                  report_name: str = None) -> str:
        """
        スコア計算とレポート生成を一括実行
        
        Args:
            strategies: 戦略リスト
            tickers: ティッカーリスト
            market_data_dict: 市場データ辞書（任意）
            report_name: レポート名（任意）
            
        Returns:
            str: 生成されたレポートファイルパス
        """
        try:
            # バッチでスコア計算
            scores = self.calculator.calculate_batch_scores(
                strategies, tickers, market_data_dict
            )
            
            # レポート生成
            report_path = self.reporter.generate_score_report(scores, report_name)
            
            logger.info(f"Score calculation and reporting completed: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error in calculate_and_report_scores: {e}")
            return None
    
    def get_top_strategies(self, ticker: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        指定ティッカーのトップ戦略を取得
        
        Args:
            ticker: ティッカーシンボル
            top_n: 取得する戦略数
            
        Returns:
            List[Tuple[str, float]]: (戦略名, スコア)のリスト
        """
        try:
            # 利用可能な戦略を取得
            available_strategies = self.data_loader.get_available_strategies()
            
            strategy_scores = []
            for strategy in available_strategies:
                score = self.calculator.calculate_strategy_score(strategy, ticker)
                if score:
                    strategy_scores.append((strategy, score.total_score))
            
            # スコア降順でソート
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            
            return strategy_scores[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting top strategies for {ticker}: {e}")
            return []

# デバッグ用の簡単なテスト関数
def _test_basic_functionality():
    """基本機能のテスト"""
    try:
        logger.info("Testing basic functionality...")
        
        # ScoreWeightsのテスト
        weights = ScoreWeights()
        total = weights.performance + weights.stability + weights.risk_adjusted + weights.trend_adaptation + weights.reliability
        assert abs(total - 1.0) < 0.001, f"Weights don't sum to 1.0: {total}"
        
        # StrategyScoreのテスト
        score = StrategyScore(
            strategy_name="test_strategy",
            ticker="AAPL",
            total_score=0.75,
            component_scores={'performance': 0.8, 'stability': 0.7},
            trend_fitness=0.6,
            confidence=0.8,
            metadata={},
            calculated_at=datetime.now()
        )
        
        score_dict = score.to_dict()
        assert 'strategy_name' in score_dict
        assert score_dict['total_score'] == 0.75
        
        logger.info("Basic functionality test passed")
        return True
        
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    # 基本機能のテスト実行
    logging.basicConfig(level=logging.INFO)
    success = _test_basic_functionality()
    print(f"Test result: {'PASSED' if success else 'FAILED'}")
