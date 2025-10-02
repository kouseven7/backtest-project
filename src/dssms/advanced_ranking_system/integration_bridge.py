"""
DSSMS Phase 3 Task 3.1: Integration Bridge
統合ブリッジクラス

既存のDSSMSシステムと新しい高度ランキングシステムを
シームレスに統合するためのブリッジクラスです。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import asyncio

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 既存DSSMSシステムからのインポート
try:
    from ..hierarchical_ranking_system import HierarchicalRankingSystem, PriorityLevel, RankingScore, SelectionResult
    from ..hybrid_ranking_engine import HybridRankingEngine, MarketCondition, RankingResult
    from ..comprehensive_scoring_engine import ComprehensiveScoringEngine
    from ..dssms_data_manager import DSSMSDataManager
except ImportError as e:
    logging.warning(f"DSSMS existing system import warning: {e}")
    # フォールバック用の基本型定義
    class PriorityLevel(Enum):
        LEVEL_1 = 1
        LEVEL_2 = 2
        LEVEL_3 = 3
    
    class MarketCondition(Enum):
        TRENDING_UP = "trending_up"
        TRENDING_DOWN = "trending_down"
        SIDEWAYS = "sideways"
        HIGH_VOLATILITY = "high_volatility"
        LOW_VOLATILITY = "low_volatility"

# 新しいコンポーネントのインポート
from .advanced_ranking_engine import AdvancedRankingEngine, AdvancedRankingResult
from .multi_dimensional_analyzer import MultiDimensionalAnalyzer, MultiDimensionalResult
from .dynamic_weight_optimizer import DynamicWeightOptimizer, WeightOptimizationResult

# 設定とロガー
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

class IntegrationMode(Enum):
    """統合モード定義"""
    HYBRID = "hybrid"               # 既存+新システムのハイブリッド
    ENHANCED = "enhanced"           # 既存システム強化
    REPLACEMENT = "replacement"     # 新システムで置き換え
    PARALLEL = "parallel"           # 並行実行・比較

class ConflictResolution(Enum):
    """競合解決方法"""
    WEIGHTED_AVERAGE = "weighted_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    CONSENSUS = "consensus"
    ADVANCED_PRIORITY = "advanced_priority"
    LEGACY_PRIORITY = "legacy_priority"

@dataclass
class IntegrationConfig:
    """統合設定"""
    mode: IntegrationMode = IntegrationMode.HYBRID
    conflict_resolution: ConflictResolution = ConflictResolution.WEIGHTED_AVERAGE
    legacy_weight: float = 0.4
    advanced_weight: float = 0.6
    enable_fallback: bool = True
    confidence_threshold: float = 0.6
    enable_performance_comparison: bool = True
    sync_update_frequency: int = 300  # 秒

@dataclass
class IntegratedRankingResult:
    """統合ランキング結果"""
    symbol: str
    final_score: float
    integrated_rank: int
    legacy_score: Optional[float]
    advanced_score: Optional[float]
    confidence_level: float
    source_system: str
    conflict_resolution_applied: bool
    integration_metadata: Dict[str, Any]
    timestamp: datetime

class IntegrationBridge:
    """
    統合ブリッジ
    
    既存のDSSMSシステムと高度ランキングシステムを統合し、
    一貫したインターフェースと最適化された結果を提供します。
    
    機能:
    - 既存システムとの互換性維持
    - データフォーマット変換
    - 結果統合とコンフリクト解決
    - パフォーマンス比較
    - フォールバック機能
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """
        初期化
        
        Args:
            config: 統合設定
        """
        self.config = config or IntegrationConfig()
        self.logger = logger
        
        # システムコンポーネント
        self._legacy_systems = {}
        self._advanced_systems = {}
        
        # 統合状態管理
        self._integration_history = []
        self._performance_comparison = {}
        self._last_sync_time = None
        
        # 初期化実行
        self._initialize_systems()
        
        self.logger.info(f"Integration Bridge initialized with mode: {self.config.mode}")
    
    def _initialize_systems(self):
        """システム初期化"""
        try:
            # 既存システム初期化
            self._initialize_legacy_systems()
            
            # 高度システム初期化
            self._initialize_advanced_systems()
            
            # 統合検証
            self._verify_integration()
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def _initialize_legacy_systems(self):
        """既存システム初期化"""
        try:
            # 階層ランキングシステム
            if 'HierarchicalRankingSystem' in globals():
                # デフォルトconfig構造を提供
                default_config = {
                    "ranking_system": {
                        "scoring_weights": {
                            "fundamental": 0.40,
                            "technical": 0.30,
                            "volume": 0.20,
                            "volatility": 0.10
                        }
                    }
                }
                self._legacy_systems['hierarchical'] = HierarchicalRankingSystem(default_config)
                self.logger.info("Hierarchical ranking system initialized")
            
            # ハイブリッドエンジン
            if 'HybridRankingEngine' in globals():
                self._legacy_systems['hybrid'] = HybridRankingEngine()
                self.logger.info("Hybrid ranking engine initialized")
            
            # 総合スコアリングエンジン
            if 'ComprehensiveScoringEngine' in globals():
                self._legacy_systems['scoring'] = ComprehensiveScoringEngine()
                self.logger.info("Comprehensive scoring engine initialized")
            
            # データマネージャー
            if 'DSSMSDataManager' in globals():
                self._legacy_systems['data_manager'] = DSSMSDataManager()
                self.logger.info("DSSMS data manager initialized")
                
        except Exception as e:
            self.logger.warning(f"Legacy system initialization warning: {e}")
    
    def _initialize_advanced_systems(self):
        """高度システム初期化"""
        try:
            # 高度ランキングエンジン
            self._advanced_systems['ranking_engine'] = AdvancedRankingEngine()
            self.logger.info("Advanced ranking engine initialized")
            
            # 多次元分析器
            self._advanced_systems['analyzer'] = MultiDimensionalAnalyzer()
            self.logger.info("Multi-dimensional analyzer initialized")
            
            # 動的重み最適化器
            self._advanced_systems['optimizer'] = DynamicWeightOptimizer()
            self.logger.info("Dynamic weight optimizer initialized")
            
        except Exception as e:
            self.logger.error(f"Advanced system initialization failed: {e}")
            raise
    
    def _verify_integration(self):
        """統合検証"""
        try:
            legacy_count = len(self._legacy_systems)
            advanced_count = len(self._advanced_systems)
            
            self.logger.info(f"Integration verification: {legacy_count} legacy systems, {advanced_count} advanced systems")
            
            if advanced_count == 0:
                raise RuntimeError("No advanced systems available")
            
            if legacy_count == 0 and self.config.mode in [IntegrationMode.HYBRID, IntegrationMode.ENHANCED]:
                self.logger.warning("No legacy systems available, switching to replacement mode")
                self.config.mode = IntegrationMode.REPLACEMENT
            
        except Exception as e:
            self.logger.error(f"Integration verification failed: {e}")
            raise
    
    async def generate_integrated_ranking(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame],
        additional_params: Optional[Dict[str, Any]] = None
    ) -> List[IntegratedRankingResult]:
        """
        統合ランキング生成
        
        Args:
            symbols: 分析対象銘柄リスト
            market_data: 市場データ
            additional_params: 追加パラメータ
            
        Returns:
            統合ランキング結果リスト
        """
        start_time = datetime.now()
        self.logger.info(f"Generating integrated ranking for {len(symbols)} symbols")
        
        try:
            # パラメータ準備
            params = additional_params or {}
            
            # モード別実行
            if self.config.mode == IntegrationMode.HYBRID:
                results = await self._execute_hybrid_mode(symbols, market_data, params)
            elif self.config.mode == IntegrationMode.ENHANCED:
                results = await self._execute_enhanced_mode(symbols, market_data, params)
            elif self.config.mode == IntegrationMode.REPLACEMENT:
                results = await self._execute_replacement_mode(symbols, market_data, params)
            elif self.config.mode == IntegrationMode.PARALLEL:
                results = await self._execute_parallel_mode(symbols, market_data, params)
            else:
                raise ValueError(f"Unknown integration mode: {self.config.mode}")
            
            # 結果ソート
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            # ランク付け
            for i, result in enumerate(results):
                result.integrated_rank = i + 1
            
            # 統合履歴更新
            self._update_integration_history(results, start_time)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Integrated ranking completed in {execution_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Integrated ranking generation failed: {e}")
            raise
    
    async def _execute_hybrid_mode(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame], 
        params: Dict[str, Any]
    ) -> List[IntegratedRankingResult]:
        """ハイブリッドモード実行"""
        
        # 並行実行
        legacy_task = asyncio.create_task(self._get_legacy_rankings(symbols, market_data, params))
        advanced_task = asyncio.create_task(self._get_advanced_rankings(symbols, market_data, params))
        
        legacy_results, advanced_results = await asyncio.gather(legacy_task, advanced_task)
        
        # 結果統合
        integrated_results = []
        
        for symbol in symbols:
            legacy_result = next((r for r in legacy_results if r['symbol'] == symbol), None)
            advanced_result = next((r for r in advanced_results if r.symbol == symbol), None)
            
            integrated = self._integrate_symbol_results(symbol, legacy_result, advanced_result)
            integrated_results.append(integrated)
        
        return integrated_results
    
    async def _execute_enhanced_mode(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame], 
        params: Dict[str, Any]
    ) -> List[IntegratedRankingResult]:
        """拡張モード実行"""
        
        # 既存システムをベースに高度システムで拡張
        legacy_results = await self._get_legacy_rankings(symbols, market_data, params)
        advanced_analysis = await self._get_advanced_analysis(symbols, market_data, params)
        
        integrated_results = []
        
        for symbol in symbols:
            legacy_result = next((r for r in legacy_results if r['symbol'] == symbol), None)
            advanced_data = next((a for a in advanced_analysis if a.symbol == symbol), None)
            
            enhanced = self._enhance_legacy_result(symbol, legacy_result, advanced_data)
            integrated_results.append(enhanced)
        
        return integrated_results
    
    async def _execute_replacement_mode(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame], 
        params: Dict[str, Any]
    ) -> List[IntegratedRankingResult]:
        """置換モード実行"""
        
        # 高度システムのみ使用
        advanced_results = await self._get_advanced_rankings(symbols, market_data, params)
        
        integrated_results = []
        
        for result in advanced_results:
            integrated = IntegratedRankingResult(
                symbol=result.symbol,
                final_score=result.final_score,
                integrated_rank=0,  # 後でソート
                legacy_score=None,
                advanced_score=result.final_score,
                confidence_level=result.confidence_level,
                source_system="advanced_only",
                conflict_resolution_applied=False,
                integration_metadata={"mode": "replacement", "advanced_result": result},
                timestamp=datetime.now()
            )
            integrated_results.append(integrated)
        
        return integrated_results
    
    async def _execute_parallel_mode(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame], 
        params: Dict[str, Any]
    ) -> List[IntegratedRankingResult]:
        """並行モード実行"""
        
        # 両システムを並行実行して比較
        legacy_task = asyncio.create_task(self._get_legacy_rankings(symbols, market_data, params))
        advanced_task = asyncio.create_task(self._get_advanced_rankings(symbols, market_data, params))
        
        legacy_results, advanced_results = await asyncio.gather(legacy_task, advanced_task)
        
        # パフォーマンス比較
        comparison_data = self._compare_system_performance(legacy_results, advanced_results)
        
        # より良いシステムの結果を採用
        integrated_results = []
        
        for symbol in symbols:
            legacy_result = next((r for r in legacy_results if r['symbol'] == symbol), None)
            advanced_result = next((r for r in advanced_results if r.symbol == symbol), None)
            
            parallel_result = self._select_best_result(symbol, legacy_result, advanced_result, comparison_data)
            integrated_results.append(parallel_result)
        
        return integrated_results
    
    async def _get_legacy_rankings(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame], 
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """既存システムランキング取得"""
        try:
            legacy_results = []
            
            # 各既存システムから結果取得
            for system_name, system in self._legacy_systems.items():
                try:
                    if system_name == 'hierarchical' and hasattr(system, 'generate_stock_ranking'):
                        results = await self._get_hierarchical_results(system, symbols, market_data)
                        legacy_results.extend(results)
                    elif system_name == 'hybrid' and hasattr(system, 'execute_ranking'):
                        results = await self._get_hybrid_results(system, symbols, market_data)
                        legacy_results.extend(results)
                    elif system_name == 'scoring' and hasattr(system, 'calculate_comprehensive_score'):
                        results = await self._get_scoring_results(system, symbols, market_data)
                        legacy_results.extend(results)
                except Exception as e:
                    self.logger.warning(f"Legacy system {system_name} failed: {e}")
                    continue
            
            # 結果の統合とクリーンアップ
            cleaned_results = self._cleanup_legacy_results(legacy_results, symbols)
            
            return cleaned_results
            
        except Exception as e:
            self.logger.warning(f"Legacy rankings failed: {e}")
            if self.config.enable_fallback:
                return self._generate_fallback_legacy_results(symbols)
            return []
    
    async def _get_advanced_rankings(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame], 
        params: Dict[str, Any]
    ) -> List[AdvancedRankingResult]:
        """高度システムランキング取得"""
        try:
            # 高度ランキングエンジンで分析
            ranking_engine = self._advanced_systems['ranking_engine']
            results = await ranking_engine.analyze_symbols_advanced(symbols, market_data, params)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Advanced rankings failed: {e}")
            if self.config.enable_fallback:
                return self._generate_fallback_advanced_results(symbols)
            raise
    
    async def _get_advanced_analysis(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame], 
        params: Dict[str, Any]
    ) -> List[MultiDimensionalResult]:
        """高度分析取得"""
        try:
            analyzer = self._advanced_systems['analyzer']
            analysis_results = []
            
            for symbol in symbols:
                if symbol in market_data:
                    result = analyzer.analyze_multi_dimensional(symbol, market_data[symbol], market_data)
                    analysis_results.append(result)
            
            return analysis_results
            
        except Exception as e:
            self.logger.warning(f"Advanced analysis failed: {e}")
            return []
    
    async def _get_hierarchical_results(
        self, 
        system, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """階層システム結果取得"""
        try:
            results = []
            
            for symbol in symbols:
                if symbol in market_data:
                    # 階層システムのインターフェースに合わせて呼び出し
                    score = system.calculate_ranking_score(symbol, market_data[symbol])
                    
                    results.append({
                        'symbol': symbol,
                        'score': score.total_score if hasattr(score, 'total_score') else 0,
                        'source': 'hierarchical',
                        'details': score if hasattr(score, '__dict__') else {}
                    })
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Hierarchical results failed: {e}")
            return []
    
    async def _get_hybrid_results(
        self, 
        system, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """ハイブリッドシステム結果取得"""
        try:
            results = []
            
            # ハイブリッドシステムの一括ランキング実行
            ranking_results = system.execute_comprehensive_ranking(symbols, market_data)
            
            for result in ranking_results:
                results.append({
                    'symbol': result.symbol if hasattr(result, 'symbol') else result.get('symbol', ''),
                    'score': result.final_score if hasattr(result, 'final_score') else result.get('score', 0),
                    'source': 'hybrid',
                    'details': result if hasattr(result, '__dict__') else result
                })
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Hybrid results failed: {e}")
            return []
    
    async def _get_scoring_results(
        self, 
        system, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """スコアリングシステム結果取得"""
        try:
            results = []
            
            for symbol in symbols:
                if symbol in market_data:
                    score = system.calculate_comprehensive_score(symbol, market_data[symbol])
                    
                    results.append({
                        'symbol': symbol,
                        'score': score if isinstance(score, (int, float)) else 0,
                        'source': 'scoring',
                        'details': {'comprehensive_score': score}
                    })
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Scoring results failed: {e}")
            return []
    
    def _cleanup_legacy_results(self, raw_results: List[Dict[str, Any]], symbols: List[str]) -> List[Dict[str, Any]]:
        """既存システム結果クリーンアップ"""
        try:
            # 銘柄ごとに結果を統合
            symbol_results = {}
            
            for result in raw_results:
                symbol = result.get('symbol', '')
                if symbol in symbols:
                    if symbol not in symbol_results:
                        symbol_results[symbol] = []
                    symbol_results[symbol].append(result)
            
            # 各銘柄の最良結果を選択
            cleaned_results = []
            
            for symbol in symbols:
                if symbol in symbol_results:
                    symbol_data = symbol_results[symbol]
                    
                    # 最高スコアの結果を採用
                    best_result = max(symbol_data, key=lambda x: x.get('score', 0))
                    
                    cleaned_results.append({
                        'symbol': symbol,
                        'score': best_result.get('score', 0),
                        'source': best_result.get('source', 'unknown'),
                        'details': best_result.get('details', {}),
                        'confidence': self._estimate_legacy_confidence(best_result)
                    })
                else:
                    # フォールバック
                    cleaned_results.append({
                        'symbol': symbol,
                        'score': 0.0,
                        'source': 'fallback',
                        'details': {},
                        'confidence': 0.0
                    })
            
            return cleaned_results
            
        except Exception as e:
            self.logger.warning(f"Legacy results cleanup failed: {e}")
            return self._generate_fallback_legacy_results(symbols)
    
    def _estimate_legacy_confidence(self, result: Dict[str, Any]) -> float:
        """既存システム信頼度推定"""
        try:
            score = result.get('score', 0)
            source = result.get('source', 'unknown')
            
            # ソース別信頼度
            source_confidence = {
                'hierarchical': 0.8,
                'hybrid': 0.9,
                'scoring': 0.7,
                'fallback': 0.1,
                'unknown': 0.5
            }
            
            base_confidence = source_confidence.get(source, 0.5)
            
            # スコア範囲による調整
            if abs(score) > 50:
                score_confidence = 0.9
            elif abs(score) > 20:
                score_confidence = 0.7
            elif abs(score) > 5:
                score_confidence = 0.5
            else:
                score_confidence = 0.3
            
            return (base_confidence + score_confidence) / 2
            
        except Exception:
            return 0.5
    
    def _integrate_symbol_results(
        self, 
        symbol: str, 
        legacy_result: Optional[Dict[str, Any]], 
        advanced_result: Optional[AdvancedRankingResult]
    ) -> IntegratedRankingResult:
        """シンボル結果統合"""
        
        legacy_score = legacy_result.get('score', 0) if legacy_result else 0
        legacy_confidence = legacy_result.get('confidence', 0) if legacy_result else 0
        
        advanced_score = advanced_result.final_score if advanced_result else 0
        advanced_confidence = advanced_result.confidence_level if advanced_result else 0
        
        # コンフリクト解決
        final_score, conflict_resolved = self._resolve_score_conflict(
            legacy_score, legacy_confidence, advanced_score, advanced_confidence
        )
        
        # 信頼度計算
        confidence_level = self._calculate_integrated_confidence(
            legacy_confidence, advanced_confidence
        )
        
        # ソースシステム判定
        source_system = self._determine_source_system(
            legacy_result, advanced_result, conflict_resolved
        )
        
        # メタデータ構築
        metadata = {
            'legacy_result': legacy_result,
            'advanced_result': advanced_result.__dict__ if advanced_result else None,
            'integration_method': self.config.conflict_resolution.value,
            'weights_used': {
                'legacy': self.config.legacy_weight,
                'advanced': self.config.advanced_weight
            }
        }
        
        return IntegratedRankingResult(
            symbol=symbol,
            final_score=final_score,
            integrated_rank=0,  # 後でソート
            legacy_score=legacy_score if legacy_result else None,
            advanced_score=advanced_score if advanced_result else None,
            confidence_level=confidence_level,
            source_system=source_system,
            conflict_resolution_applied=conflict_resolved,
            integration_metadata=metadata,
            timestamp=datetime.now()
        )
    
    def _resolve_score_conflict(
        self, 
        legacy_score: float, 
        legacy_confidence: float, 
        advanced_score: float, 
        advanced_confidence: float
    ) -> Tuple[float, bool]:
        """スコア競合解決"""
        
        # 競合判定
        score_diff = abs(legacy_score - advanced_score)
        max_score = max(abs(legacy_score), abs(advanced_score))
        
        is_conflict = score_diff > (max_score * 0.2) if max_score > 0 else False
        
        if not is_conflict:
            # 競合なし：単純平均
            final_score = (legacy_score * self.config.legacy_weight + 
                          advanced_score * self.config.advanced_weight)
            return final_score, False
        
        # 競合解決方法に基づく処理
        if self.config.conflict_resolution == ConflictResolution.WEIGHTED_AVERAGE:
            final_score = (legacy_score * self.config.legacy_weight + 
                          advanced_score * self.config.advanced_weight)
        
        elif self.config.conflict_resolution == ConflictResolution.HIGHEST_CONFIDENCE:
            if advanced_confidence > legacy_confidence:
                final_score = advanced_score
            else:
                final_score = legacy_score
        
        elif self.config.conflict_resolution == ConflictResolution.ADVANCED_PRIORITY:
            final_score = advanced_score
        
        elif self.config.conflict_resolution == ConflictResolution.LEGACY_PRIORITY:
            final_score = legacy_score
        
        elif self.config.conflict_resolution == ConflictResolution.CONSENSUS:
            # 信頼度重み付き平均
            total_confidence = legacy_confidence + advanced_confidence
            if total_confidence > 0:
                final_score = ((legacy_score * legacy_confidence + 
                               advanced_score * advanced_confidence) / total_confidence)
            else:
                final_score = (legacy_score + advanced_score) / 2
        
        else:
            # デフォルト：重み付き平均
            final_score = (legacy_score * self.config.legacy_weight + 
                          advanced_score * self.config.advanced_weight)
        
        return final_score, True
    
    def _calculate_integrated_confidence(
        self, 
        legacy_confidence: float, 
        advanced_confidence: float
    ) -> float:
        """統合信頼度計算"""
        
        if legacy_confidence == 0 and advanced_confidence == 0:
            return 0.0
        elif legacy_confidence == 0:
            return advanced_confidence * 0.8  # 単一システムによる減衰
        elif advanced_confidence == 0:
            return legacy_confidence * 0.8
        else:
            # 両システムの一貫性を考慮
            consistency_bonus = 1 - abs(legacy_confidence - advanced_confidence)
            base_confidence = (legacy_confidence + advanced_confidence) / 2
            return min(1.0, base_confidence * (1 + consistency_bonus * 0.2))
    
    def _determine_source_system(
        self, 
        legacy_result: Optional[Dict[str, Any]], 
        advanced_result: Optional[AdvancedRankingResult], 
        conflict_resolved: bool
    ) -> str:
        """ソースシステム判定"""
        
        if not legacy_result and not advanced_result:
            return "none"
        elif not legacy_result:
            return "advanced_only"
        elif not advanced_result:
            return "legacy_only"
        elif conflict_resolved:
            return f"integrated_{self.config.conflict_resolution.value}"
        else:
            return "integrated_consensus"
    
    def _enhance_legacy_result(
        self, 
        symbol: str, 
        legacy_result: Optional[Dict[str, Any]], 
        advanced_data: Optional[MultiDimensionalResult]
    ) -> IntegratedRankingResult:
        """既存結果の拡張"""
        
        base_score = legacy_result.get('score', 0) if legacy_result else 0
        base_confidence = legacy_result.get('confidence', 0.5) if legacy_result else 0.5
        
        # 高度分析による拡張
        enhancement_factor = 1.0
        enhancement_confidence = 0.0
        
        if advanced_data:
            enhancement_factor = 1 + (advanced_data.composite_score - 50) / 100  # -50~150 -> 0.5~2.0
            enhancement_factor = max(0.5, min(2.0, enhancement_factor))
            enhancement_confidence = advanced_data.confidence_level
        
        # 拡張スコア計算
        enhanced_score = base_score * enhancement_factor
        
        # 拡張信頼度計算
        enhanced_confidence = (base_confidence + enhancement_confidence) / 2
        
        metadata = {
            'base_legacy_result': legacy_result,
            'advanced_analysis': advanced_data.__dict__ if advanced_data else None,
            'enhancement_factor': enhancement_factor,
            'enhancement_method': 'multiplicative'
        }
        
        return IntegratedRankingResult(
            symbol=symbol,
            final_score=enhanced_score,
            integrated_rank=0,
            legacy_score=base_score,
            advanced_score=enhanced_score - base_score,
            confidence_level=enhanced_confidence,
            source_system="enhanced_legacy",
            conflict_resolution_applied=False,
            integration_metadata=metadata,
            timestamp=datetime.now()
        )
    
    def _compare_system_performance(
        self, 
        legacy_results: List[Dict[str, Any]], 
        advanced_results: List[AdvancedRankingResult]
    ) -> Dict[str, Any]:
        """システムパフォーマンス比較"""
        
        comparison = {
            'legacy_metrics': self._calculate_system_metrics(legacy_results),
            'advanced_metrics': self._calculate_advanced_metrics(advanced_results),
            'recommendation': 'advanced'  # デフォルト
        }
        
        # パフォーマンス比較に基づく推奨
        legacy_avg_confidence = comparison['legacy_metrics'].get('avg_confidence', 0)
        advanced_avg_confidence = comparison['advanced_metrics'].get('avg_confidence', 0)
        
        if legacy_avg_confidence > advanced_avg_confidence * 1.2:
            comparison['recommendation'] = 'legacy'
        elif advanced_avg_confidence > legacy_avg_confidence * 1.2:
            comparison['recommendation'] = 'advanced'
        else:
            comparison['recommendation'] = 'hybrid'
        
        return comparison
    
    def _calculate_system_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """システムメトリクス計算"""
        if not results:
            return {}
        
        scores = [r.get('score', 0) for r in results]
        confidences = [r.get('confidence', 0) for r in results]
        
        return {
            'avg_score': np.mean(scores),
            'score_std': np.std(scores),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'coverage': len([r for r in results if r.get('score', 0) != 0]) / len(results)
        }
    
    def _calculate_advanced_metrics(self, results: List[AdvancedRankingResult]) -> Dict[str, float]:
        """高度システムメトリクス計算"""
        if not results:
            return {}
        
        scores = [r.final_score for r in results]
        confidences = [r.confidence_level for r in results]
        
        return {
            'avg_score': np.mean(scores),
            'score_std': np.std(scores),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'coverage': len([r for r in results if r.final_score != 0]) / len(results)
        }
    
    def _select_best_result(
        self, 
        symbol: str, 
        legacy_result: Optional[Dict[str, Any]], 
        advanced_result: Optional[AdvancedRankingResult], 
        comparison_data: Dict[str, Any]
    ) -> IntegratedRankingResult:
        """最良結果選択"""
        
        recommendation = comparison_data.get('recommendation', 'advanced')
        
        if recommendation == 'legacy' and legacy_result:
            selected_score = legacy_result.get('score', 0)
            selected_confidence = legacy_result.get('confidence', 0)
            source_system = "legacy_selected"
            metadata = {'selection_reason': 'legacy_superior', 'comparison': comparison_data}
        elif recommendation == 'advanced' and advanced_result:
            selected_score = advanced_result.final_score
            selected_confidence = advanced_result.confidence_level
            source_system = "advanced_selected"
            metadata = {'selection_reason': 'advanced_superior', 'comparison': comparison_data}
        else:
            # ハイブリッド選択
            return self._integrate_symbol_results(symbol, legacy_result, advanced_result)
        
        return IntegratedRankingResult(
            symbol=symbol,
            final_score=selected_score,
            integrated_rank=0,
            legacy_score=legacy_result.get('score', 0) if legacy_result else None,
            advanced_score=advanced_result.final_score if advanced_result else None,
            confidence_level=selected_confidence,
            source_system=source_system,
            conflict_resolution_applied=False,
            integration_metadata=metadata,
            timestamp=datetime.now()
        )
    
    def _generate_fallback_legacy_results(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """フォールバック既存結果生成"""
        return [
            {
                'symbol': symbol,
                'score': 0.0,
                'source': 'fallback',
                'details': {},
                'confidence': 0.1
            }
            for symbol in symbols
        ]
    
    def _generate_fallback_advanced_results(self, symbols: List[str]) -> List[AdvancedRankingResult]:
        """フォールバック高度結果生成"""
        results = []
        for symbol in symbols:
            result = AdvancedRankingResult(
                symbol=symbol,
                final_score=0.0,
                multi_dimensional_scores={},
                dynamic_weights={},
                confidence_level=0.1,
                market_condition="unknown",
                analysis_timestamp=datetime.now(),
                component_details={},
                performance_metrics={},
                integration_status="fallback"
            )
            results.append(result)
        return results
    
    def _update_integration_history(self, results: List[IntegratedRankingResult], start_time: datetime):
        """統合履歴更新"""
        try:
            history_entry = {
                'timestamp': start_time,
                'mode': self.config.mode.value,
                'symbol_count': len(results),
                'avg_confidence': np.mean([r.confidence_level for r in results]),
                'conflict_resolution_count': len([r for r in results if r.conflict_resolution_applied]),
                'source_distribution': self._calculate_source_distribution(results)
            }
            
            self._integration_history.append(history_entry)
            
            # 履歴サイズ制限
            if len(self._integration_history) > 1000:
                self._integration_history = self._integration_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f"Integration history update failed: {e}")
    
    def _calculate_source_distribution(self, results: List[IntegratedRankingResult]) -> Dict[str, int]:
        """ソース分布計算"""
        distribution = {}
        for result in results:
            source = result.source_system
            distribution[source] = distribution.get(source, 0) + 1
        return distribution
    
    def get_integration_status(self) -> Dict[str, Any]:
        """統合状態取得"""
        return {
            'config': {
                'mode': self.config.mode.value,
                'conflict_resolution': self.config.conflict_resolution.value,
                'weights': {
                    'legacy': self.config.legacy_weight,
                    'advanced': self.config.advanced_weight
                }
            },
            'systems': {
                'legacy_systems': list(self._legacy_systems.keys()),
                'advanced_systems': list(self._advanced_systems.keys())
            },
            'history': {
                'total_integrations': len(self._integration_history),
                'recent_performance': self._integration_history[-10:] if self._integration_history else []
            },
            'last_sync': self._last_sync_time
        }
    
    def update_integration_config(self, new_config: IntegrationConfig):
        """統合設定更新"""
        try:
            old_config = self.config
            self.config = new_config
            
            self.logger.info(f"Integration config updated: {old_config.mode.value} -> {new_config.mode.value}")
            
            # 必要に応じてシステム再初期化
            if old_config.mode != new_config.mode:
                self._verify_integration()
                
        except Exception as e:
            self.logger.error(f"Integration config update failed: {e}")
            raise
