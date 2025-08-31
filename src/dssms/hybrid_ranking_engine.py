"""
DSSMS Phase 2 Task 2.2: ハイブリッドランキングシステム
メインオーケストレーションエンジン

機能:
- 階層優先型ランキングシステムとの統合
- 総合スコアリングエンジンとの連携
- 適応的スコア更新管理
- パフォーマンス最適化
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import pandas as pd
import numpy as np
import logging
import json
import asyncio
from datetime import datetime, timedelta
import warnings

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存DSSMSコンポーネントのインポート
from .hierarchical_ranking_system import HierarchicalRankingSystem, PriorityLevel, RankingScore
from .comprehensive_scoring_engine import ComprehensiveScoringEngine
from .dssms_data_manager import DSSMSDataManager
from .ranking_data_integrator import RankingDataIntegrator
from .adaptive_score_calculator import AdaptiveScoreCalculator
from .ranking_performance_optimizer import RankingPerformanceOptimizer
from config.logger_config import setup_logger

class MarketCondition(Enum):
    """市場状況定義"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down" 
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class RankingResult:
    """ハイブリッドランキング結果"""
    symbol: str
    final_score: float
    hierarchical_score: float
    comprehensive_score: float
    adaptive_bonus: float
    market_condition_factor: float
    priority_level: int
    confidence: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    data_freshness: float = 1.0
    processing_time_ms: float = 0.0

@dataclass
class SystemStatus:
    """システム状態情報"""
    active_symbols: Set[str] = field(default_factory=set)
    last_full_update: Optional[datetime] = None
    cache_hit_rate: float = 0.0
    average_processing_time: float = 0.0
    total_rankings_generated: int = 0
    error_count: int = 0
    market_condition: Optional[MarketCondition] = None

class HybridRankingEngine:
    """
    ハイブリッドランキングシステムのメインエンジン
    
    既存システムとの統合:
    - HierarchicalRankingSystem: 階層優先型ランキング
    - ComprehensiveScoringEngine: 詳細スコアリング
    - DSSMSDataManager: データ管理
    
    新機能:
    - 適応的スコア調整
    - パフォーマンス最適化
    - 市場状況分析
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.logger = setup_logger('dssms.hybrid_ranking')
        
        # 設定読み込み
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "dssms" / "hybrid_ranking_config.json")
        
        self.config = self._load_config(config_path)
        
        # システム状態
        self.status = SystemStatus()
        
        # 既存システム初期化
        try:
            # 既存システム用の設定準備
            hierarchical_config = self.config.get("hierarchical_ranking", {})
            comprehensive_config = self.config.get("comprehensive_scoring", {})
            
            self.hierarchical_system = HierarchicalRankingSystem(hierarchical_config)
            self.comprehensive_engine = ComprehensiveScoringEngine()
            self.data_manager = DSSMSDataManager()
            self.logger.info("既存システム初期化成功")
        except Exception as e:
            self.logger.error(f"既存システム初期化エラー: {e}")
            raise
        
        # 新機能コンポーネント初期化
        try:
            self.data_integrator = RankingDataIntegrator(self.config.get("data_integration", {}))
            self.adaptive_calculator = AdaptiveScoreCalculator(self.config.get("adaptive_scoring", {}))
            self.performance_optimizer = RankingPerformanceOptimizer(self.config.get("optimization", {}))
            self.logger.info("新機能コンポーネント初期化成功")
        except Exception as e:
            self.logger.error(f"新機能コンポーネント初期化エラー: {e}")
            raise
            
        # 結果キャッシュ
        self._ranking_cache: Dict[str, RankingResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        self.logger.info("HybridRankingEngine initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"設定ファイル読み込み成功: {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"設定ファイル読み込み失敗: {e}. デフォルト設定使用")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "ranking_weights": {
                "hierarchical": 0.6,
                "comprehensive": 0.3,
                "adaptive": 0.1
            },
            "market_analysis": {
                "volatility_window": 20,
                "trend_detection_period": 50,
                "volume_threshold_multiplier": 1.5
            },
            "adaptive_scoring": {
                "enabled": True,
                "update_frequency_minutes": 15,
                "learning_rate": 0.1,
                "performance_lookback_days": 30
            },
            "optimization": {
                "cache_enabled": True,
                "cache_ttl_minutes": 10,
                "parallel_processing": True,
                "max_workers": 4
            },
            "thresholds": {
                "min_confidence": 0.3,
                "max_processing_time_ms": 5000,
                "cache_hit_rate_target": 0.8
            }
        }
    
    async def generate_ranking(self, symbols: List[str], 
                             force_refresh: bool = False) -> List[RankingResult]:
        """
        ハイブリッドランキング生成
        
        Args:
            symbols: 対象銘柄リスト
            force_refresh: キャッシュ無視フラグ
            
        Returns:
            List[RankingResult]: ランキング結果リスト
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"ハイブリッドランキング生成開始: {len(symbols)}銘柄")
            
            # 市場状況分析
            market_condition = await self._analyze_market_condition()
            self.status.market_condition = market_condition
            
            # データ統合準備
            integrated_data = await self.data_integrator.prepare_integrated_data(symbols)
            
            # 並列ランキング処理
            if self.config.get("optimization", {}).get("parallel_processing", True):
                rankings = await self._parallel_ranking_generation(symbols, integrated_data, 
                                                                  market_condition, force_refresh)
            else:
                rankings = await self._sequential_ranking_generation(symbols, integrated_data, 
                                                                   market_condition, force_refresh)
            
            # ランキングソート
            rankings.sort(key=lambda x: x.final_score, reverse=True)
            
            # システム状態更新
            self._update_system_status(symbols, rankings, start_time)
            
            self.logger.info(f"ハイブリッドランキング生成完了: {len(rankings)}件")
            return rankings
            
        except Exception as e:
            self.logger.error(f"ランキング生成エラー: {e}")
            self.status.error_count += 1
            return []
    
    async def _parallel_ranking_generation(self, symbols: List[str], 
                                         integrated_data: Dict[str, Any],
                                         market_condition: MarketCondition,
                                         force_refresh: bool) -> List[RankingResult]:
        """並列ランキング生成"""
        max_workers = self.config.get("optimization", {}).get("max_workers", 4)
        
        # タスク分割
        chunk_size = max(1, len(symbols) // max_workers)
        symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        
        # 並列実行
        tasks = []
        for chunk in symbol_chunks:
            task = self._process_symbol_chunk(chunk, integrated_data, market_condition, force_refresh)
            tasks.append(task)
        
        # 結果収集
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        rankings = []
        for result in chunk_results:
            if isinstance(result, Exception):
                self.logger.error(f"チャンク処理エラー: {result}")
                continue
            rankings.extend(result)
        
        return rankings
    
    async def _sequential_ranking_generation(self, symbols: List[str],
                                           integrated_data: Dict[str, Any],
                                           market_condition: MarketCondition,
                                           force_refresh: bool) -> List[RankingResult]:
        """逐次ランキング生成"""
        rankings = []
        
        for symbol in symbols:
            try:
                ranking = await self._generate_single_ranking(symbol, integrated_data, 
                                                            market_condition, force_refresh)
                if ranking:
                    rankings.append(ranking)
            except Exception as e:
                self.logger.error(f"銘柄{symbol}のランキング生成エラー: {e}")
                continue
        
        return rankings
    
    async def _process_symbol_chunk(self, symbols: List[str],
                                  integrated_data: Dict[str, Any],
                                  market_condition: MarketCondition,
                                  force_refresh: bool) -> List[RankingResult]:
        """銘柄チャンク処理"""
        rankings = []
        
        for symbol in symbols:
            try:
                ranking = await self._generate_single_ranking(symbol, integrated_data, 
                                                            market_condition, force_refresh)
                if ranking:
                    rankings.append(ranking)
            except Exception as e:
                self.logger.error(f"銘柄{symbol}のランキング生成エラー: {e}")
                continue
        
        return rankings
    
    async def _generate_single_ranking(self, symbol: str,
                                     integrated_data: Dict[str, Any],
                                     market_condition: MarketCondition,
                                     force_refresh: bool) -> Optional[RankingResult]:
        """単一銘柄ランキング生成"""
        start_time = datetime.now()
        
        try:
            # キャッシュチェック
            if not force_refresh:
                cached_result = self._get_cached_ranking(symbol)
                if cached_result:
                    return cached_result
            
            # 階層ランキング取得
            hierarchical_result = await self._get_hierarchical_ranking(symbol, integrated_data)
            if not hierarchical_result:
                return None
            
            # 総合スコアリング
            comprehensive_score = await self._get_comprehensive_score(symbol, integrated_data)
            
            # 適応的スコア調整
            adaptive_bonus = await self.adaptive_calculator.calculate_adaptive_bonus(
                symbol, hierarchical_result, market_condition
            )
            
            # 市場状況ファクター
            market_factor = self._calculate_market_condition_factor(market_condition)
            
            # 最終スコア計算
            weights = self.config.get("ranking_weights", {})
            final_score = (
                hierarchical_result.total_score * weights.get("hierarchical", 0.6) +
                comprehensive_score * weights.get("comprehensive", 0.3) +
                adaptive_bonus * weights.get("adaptive", 0.1)
            ) * market_factor
            
            # 処理時間計算
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # 結果作成
            ranking_result = RankingResult(
                symbol=symbol,
                final_score=final_score,
                hierarchical_score=hierarchical_result.total_score,
                comprehensive_score=comprehensive_score,
                adaptive_bonus=adaptive_bonus,
                market_condition_factor=market_factor,
                priority_level=hierarchical_result.priority_group,
                confidence=hierarchical_result.confidence_level,
                processing_time_ms=processing_time
            )
            
            # キャッシュ保存
            self._cache_ranking(symbol, ranking_result)
            
            return ranking_result
            
        except Exception as e:
            self.logger.error(f"銘柄{symbol}のランキング生成エラー: {e}")
            return None
    
    async def _get_hierarchical_ranking(self, symbol: str, 
                                      integrated_data: Dict[str, Any]) -> Optional[RankingScore]:
        """階層ランキング取得"""
        try:
            # 階層ランキングシステムから取得
            ranking_results = self.hierarchical_system.rank_within_priority_group([symbol])
            if ranking_results:
                # 結果の形式: List[Tuple[str, float]]
                for sym, score in ranking_results:
                    if sym == symbol:
                        # RankingScoreオブジェクトを作成
                        from .hierarchical_ranking_system import RankingScore
                        from datetime import datetime
                        return RankingScore(
                            symbol=symbol,
                            total_score=score,
                            perfect_order_score=0.0,
                            fundamental_score=0.0,
                            technical_score=0.0,
                            volume_score=0.0,
                            volatility_score=0.0,
                            priority_group=1,
                            confidence_level=0.5,
                            affordability_penalty=0.0,
                            last_updated=datetime.now()
                        )
            return None
        except Exception as e:
            self.logger.error(f"階層ランキング取得エラー ({symbol}): {e}")
            return None
    
    async def _get_comprehensive_score(self, symbol: str, 
                                     integrated_data: Dict[str, Any]) -> float:
        """総合スコア取得"""
        try:
            # 総合スコアリングエンジンから取得
            symbol_data = integrated_data.get(symbol, {})
            score = self.comprehensive_engine.calculate_comprehensive_score(symbol, symbol_data)
            return score.get("total_score", 0.0)
        except Exception as e:
            self.logger.error(f"総合スコア取得エラー ({symbol}): {e}")
            return 0.0
    
    async def _analyze_market_condition(self) -> MarketCondition:
        """市場状況分析"""
        try:
            # 市場インデックス分析（簡易版）
            market_data = self.data_manager.get_market_index_data("^N225")
            
            if market_data is None or market_data.empty:
                return MarketCondition.SIDEWAYS
            
            # トレンド分析
            period = self.config.get("market_analysis", {}).get("trend_detection_period", 50)
            if len(market_data) < period:
                return MarketCondition.SIDEWAYS
            
            recent_close = market_data['Close'].iloc[-1]
            period_avg = market_data['Close'].tail(period).mean()
            
            # ボラティリティ分析
            volatility_window = self.config.get("market_analysis", {}).get("volatility_window", 20)
            recent_volatility = market_data['Close'].tail(volatility_window).std()
            avg_volatility = market_data['Close'].tail(100).std()
            
            # 判定
            if recent_volatility > avg_volatility * 1.5:
                return MarketCondition.HIGH_VOLATILITY
            elif recent_volatility < avg_volatility * 0.5:
                return MarketCondition.LOW_VOLATILITY
            elif recent_close > period_avg * 1.02:
                return MarketCondition.TRENDING_UP
            elif recent_close < period_avg * 0.98:
                return MarketCondition.TRENDING_DOWN
            else:
                return MarketCondition.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"市場状況分析エラー: {e}")
            return MarketCondition.SIDEWAYS
    
    def _calculate_market_condition_factor(self, condition: MarketCondition) -> float:
        """市場状況ファクター計算"""
        factors = {
            MarketCondition.TRENDING_UP: 1.1,
            MarketCondition.TRENDING_DOWN: 0.9,
            MarketCondition.SIDEWAYS: 1.0,
            MarketCondition.HIGH_VOLATILITY: 0.95,
            MarketCondition.LOW_VOLATILITY: 1.05
        }
        return factors.get(condition, 1.0)
    
    def _get_cached_ranking(self, symbol: str) -> Optional[RankingResult]:
        """キャッシュからランキング取得"""
        if not self.config.get("optimization", {}).get("cache_enabled", True):
            return None
        
        if symbol not in self._ranking_cache:
            return None
        
        # TTLチェック
        ttl_minutes = self.config.get("optimization", {}).get("cache_ttl_minutes", 10)
        cache_time = self._cache_timestamps.get(symbol)
        if cache_time and datetime.now() - cache_time > timedelta(minutes=ttl_minutes):
            # キャッシュ期限切れ
            del self._ranking_cache[symbol]
            del self._cache_timestamps[symbol]
            return None
        
        return self._ranking_cache[symbol]
    
    def _cache_ranking(self, symbol: str, ranking: RankingResult):
        """ランキングをキャッシュ"""
        if self.config.get("optimization", {}).get("cache_enabled", True):
            self._ranking_cache[symbol] = ranking
            self._cache_timestamps[symbol] = datetime.now()
    
    def _update_system_status(self, symbols: List[str], rankings: List[RankingResult], 
                            start_time: datetime):
        """システム状態更新"""
        self.status.active_symbols.update(symbols)
        self.status.last_full_update = datetime.now()
        self.status.total_rankings_generated += len(rankings)
        
        # 処理時間平均更新
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        if self.status.average_processing_time == 0:
            self.status.average_processing_time = total_time
        else:
            self.status.average_processing_time = (
                self.status.average_processing_time * 0.8 + total_time * 0.2
            )
        
        # キャッシュヒット率計算
        cache_hits = sum(1 for s in symbols if s in self._ranking_cache)
        self.status.cache_hit_rate = cache_hits / len(symbols) if symbols else 0
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        return {
            "active_symbols_count": len(self.status.active_symbols),
            "last_full_update": self.status.last_full_update.isoformat() if self.status.last_full_update else None,
            "cache_hit_rate": self.status.cache_hit_rate,
            "average_processing_time_ms": self.status.average_processing_time,
            "total_rankings_generated": self.status.total_rankings_generated,
            "error_count": self.status.error_count,
            "market_condition": self.status.market_condition.value if self.status.market_condition else None,
            "cache_size": len(self._ranking_cache)
        }
    
    def clear_cache(self):
        """キャッシュクリア"""
        self._ranking_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("ランキングキャッシュをクリアしました")
    
    async def shutdown(self):
        """システムシャットダウン"""
        try:
            # キャッシュクリア
            self.clear_cache()
            
            # 各コンポーネントのシャットダウン処理
            if hasattr(self.performance_optimizer, 'shutdown'):
                await self.performance_optimizer.shutdown()
            
            self.logger.info("HybridRankingEngine shutdown completed")
        except Exception as e:
            self.logger.error(f"シャットダウンエラー: {e}")
