"""
DSSMS Phase 3 Task 3.1: Advanced Ranking Engine
高度なランキングエンジンメインクラス

既存のDSSMSシステムと統合し、多次元分析と動的最適化を実現する
メインエンジンクラスです。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from enum import Enum
import pandas as pd
import numpy as np
import logging
import json
import asyncio
from datetime import datetime, timedelta
import concurrent.futures
import threading
import time

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 既存DSSMSシステムからのインポート
try:
    from ..hierarchical_ranking_system import HierarchicalRankingSystem, PriorityLevel, RankingScore
    from ..hybrid_ranking_engine import HybridRankingEngine, MarketCondition
    from ..comprehensive_scoring_engine import ComprehensiveScoringEngine
    from ..dssms_data_manager import DSSMSDataManager
except ImportError as e:
    logging.warning(f"DSSMS existing system import warning: {e}")
    # フォールバック: 基本的な型定義
    class PriorityLevel(Enum):
        LEVEL_1 = 1
        LEVEL_2 = 2
        LEVEL_3 = 3
    
    @dataclass
    class RankingScore:
        symbol: str
        total_score: float
        perfect_order_score: float = 0.0
        fundamental_score: float = 0.0
        technical_score: float = 0.0
        volume_score: float = 0.0
        volatility_score: float = 0.0
        priority_group: int = 3
        confidence_level: float = 0.0
        affordability_penalty: float = 0.0
        last_updated: datetime = field(default_factory=datetime.now)

# 設定とロガー
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

class AnalysisMode(Enum):
    """分析モード定義"""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    REALTIME = "realtime"
    BATCH = "batch"

class OptimizationLevel(Enum):
    """最適化レベル定義"""
    BASIC = "basic"
    ADVANCED = "advanced"
    ULTRA = "ultra"

@dataclass
class AdvancedRankingConfig:
    """高度ランキング設定"""
    analysis_mode: AnalysisMode = AnalysisMode.ENHANCED
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    enable_parallel_processing: bool = True
    max_workers: int = 4
    cache_ttl_seconds: int = 300
    realtime_update_interval: int = 60
    confidence_threshold: float = 0.6
    max_symbols_per_analysis: int = 500
    enable_dynamic_weights: bool = True
    market_condition_sensitivity: float = 0.8

@dataclass
class AdvancedRankingResult:
    """高度ランキング結果"""
    symbol: str
    final_score: float
    multi_dimensional_scores: Dict[str, float]
    dynamic_weights: Dict[str, float]
    confidence_level: float
    market_condition: str
    analysis_timestamp: datetime
    component_details: Dict[str, Any]
    performance_metrics: Dict[str, float]
    integration_status: str

@dataclass
class RankingConfig:
    """ランキング設定"""
    enable_async_processing: bool = True
    max_concurrent_tasks: int = 8
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    ranking_threshold: float = 0.1
    enable_real_time: bool = True
    log_level: str = "INFO"

class AdvancedRankingEngine:
    """
    高度なランキングエンジン
    
    既存のDSSMSシステムを拡張し、以下の機能を提供:
    - 多次元分析統合
    - 動的重み最適化
    - 並列処理による高速化
    - リアルタイム更新
    - キャッシュ管理
    - パフォーマンス監視
    """
    
    def __init__(self, config: Optional[AdvancedRankingConfig] = None):
        """
        初期化
        
        Args:
            config: 高度ランキング設定
        """
        self.config = config or AdvancedRankingConfig()
        self.logger = logger
        
        # 内部状態管理
        self._last_update_time = None
        self._cache = {}
        self._performance_metrics = {}
        self._market_condition_cache = {}
        
        # 既存システム統合
        self._hierarchical_system = None
        self._hybrid_engine = None
        self._scoring_engine = None
        self._data_manager = None
        
        # 並列処理設定
        self._executor = None
        self._thread_lock = threading.Lock()
        
        # 初期化実行
        self._initialize_components()
        
        self.logger.info(f"Advanced Ranking Engine initialized with mode: {self.config.analysis_mode}")
    
    def _initialize_components(self):
        """コンポーネント初期化"""
        try:
            # 既存システムとの統合
            self._initialize_existing_systems()
            
            # 並列処理設定
            if self.config.enable_parallel_processing:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.max_workers
                )
            
            # キャッシュ初期化
            self._initialize_cache()
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def _initialize_existing_systems(self):
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
                self._hierarchical_system = HierarchicalRankingSystem(default_config)
                self.logger.info("Hierarchical ranking system integrated")
            
            # ハイブリッドエンジン
            if 'HybridRankingEngine' in globals():
                self._hybrid_engine = HybridRankingEngine()
                self.logger.info("Hybrid ranking engine integrated")
            
            # スコアリングエンジン
            if 'ComprehensiveScoringEngine' in globals():
                self._scoring_engine = ComprehensiveScoringEngine()
                self.logger.info("Comprehensive scoring engine integrated")
                
        except Exception as e:
            self.logger.warning(f"Existing system integration warning: {e}")
    
    def _initialize_cache(self):
        """キャッシュ初期化"""
        self._cache = {
            'rankings': {},
            'market_conditions': {},
            'analysis_results': {},
            'weight_optimizations': {}
        }
        self.logger.info("Cache system initialized")
    
    async def analyze_symbols_advanced(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame],
        analysis_params: Optional[Dict[str, Any]] = None
    ) -> List[AdvancedRankingResult]:
        """
        高度な銘柄分析（非同期）
        
        Args:
            symbols: 分析対象銘柄リスト
            market_data: 市場データ
            analysis_params: 分析パラメータ
            
        Returns:
            高度ランキング結果リスト
        """
        start_time = time.time()
        self.logger.info(f"Starting advanced analysis for {len(symbols)} symbols")
        
        try:
            # パラメータ設定
            params = analysis_params or {}
            
            # 並列分析実行
            if self.config.enable_parallel_processing and len(symbols) > 10:
                results = await self._parallel_analysis(symbols, market_data, params)
            else:
                results = await self._sequential_analysis(symbols, market_data, params)
            
            # 結果統合と最終ランキング
            final_results = await self._integrate_analysis_results(results)
            
            # パフォーマンス記録
            analysis_time = time.time() - start_time
            self._record_performance('advanced_analysis', analysis_time, len(symbols))
            
            self.logger.info(f"Advanced analysis completed in {analysis_time:.2f}s")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Advanced analysis failed: {e}")
            raise
    
    async def _parallel_analysis(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """並列分析実行"""
        
        # バッチ分割
        batch_size = max(1, len(symbols) // self.config.max_workers)
        symbol_batches = [
            symbols[i:i + batch_size] 
            for i in range(0, len(symbols), batch_size)
        ]
        
        # 並列タスク作成
        tasks = []
        for batch in symbol_batches:
            task = asyncio.create_task(
                self._analyze_symbol_batch(batch, market_data, params)
            )
            tasks.append(task)
        
        # 並列実行
        batch_results = await asyncio.gather(*tasks)
        
        # 結果統合
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        return all_results
    
    async def _sequential_analysis(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """逐次分析実行"""
        
        results = []
        for symbol in symbols:
            try:
                result = await self._analyze_single_symbol(symbol, market_data, params)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Symbol {symbol} analysis failed: {e}")
                continue
        
        return results
    
    async def _analyze_symbol_batch(
        self, 
        symbols: List[str], 
        market_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """シンボルバッチ分析"""
        
        results = []
        for symbol in symbols:
            try:
                result = await self._analyze_single_symbol(symbol, market_data, params)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Batch analysis - Symbol {symbol} failed: {e}")
                continue
        
        return results
    
    async def _analyze_single_symbol(
        self, 
        symbol: str, 
        market_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """単一銘柄分析"""
        
        # キャッシュチェック
        cache_key = f"{symbol}_{hash(str(params))}"
        if cache_key in self._cache['analysis_results']:
            cached_result = self._cache['analysis_results'][cache_key]
            if self._is_cache_valid(cached_result['timestamp']):
                return cached_result['data']
        
        # 基本データ取得
        symbol_data = market_data.get(symbol)
        if symbol_data is None or symbol_data.empty:
            raise ValueError(f"No data available for symbol: {symbol}")
        
        # 分析結果構築
        analysis_result = {
            'symbol': symbol,
            'basic_analysis': self._perform_basic_analysis(symbol_data),
            'technical_analysis': self._perform_technical_analysis(symbol_data),
            'volume_analysis': self._perform_volume_analysis(symbol_data),
            'volatility_analysis': self._perform_volatility_analysis(symbol_data),
            'momentum_analysis': self._perform_momentum_analysis(symbol_data),
            'market_condition': self._detect_market_condition(symbol_data),
            'timestamp': datetime.now()
        }
        
        # キャッシュ保存
        self._cache['analysis_results'][cache_key] = {
            'data': analysis_result,
            'timestamp': datetime.now()
        }
        
        return analysis_result
    
    def _perform_basic_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """基本分析実行"""
        try:
            return {
                'price_trend': self._calculate_price_trend(data),
                'volume_trend': self._calculate_volume_trend(data),
                'price_momentum': self._calculate_price_momentum(data),
                'relative_strength': self._calculate_relative_strength(data)
            }
        except Exception as e:
            self.logger.warning(f"Basic analysis failed: {e}")
            return {}
    
    def _perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """テクニカル分析実行"""
        try:
            return {
                'sma_signal': self._calculate_sma_signal(data),
                'ema_signal': self._calculate_ema_signal(data),
                'macd_signal': self._calculate_macd_signal(data),
                'rsi_signal': self._calculate_rsi_signal(data),
                'bollinger_signal': self._calculate_bollinger_signal(data)
            }
        except Exception as e:
            self.logger.warning(f"Technical analysis failed: {e}")
            return {}
    
    def _perform_volume_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """出来高分析実行"""
        try:
            return {
                'volume_momentum': self._calculate_volume_momentum(data),
                'volume_price_correlation': self._calculate_volume_price_correlation(data),
                'volume_breakout': self._calculate_volume_breakout(data)
            }
        except Exception as e:
            self.logger.warning(f"Volume analysis failed: {e}")
            return {}
    
    def _perform_volatility_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """ボラティリティ分析実行"""
        try:
            return {
                'realized_volatility': self._calculate_realized_volatility(data),
                'volatility_trend': self._calculate_volatility_trend(data),
                'risk_adjusted_return': self._calculate_risk_adjusted_return(data)
            }
        except Exception as e:
            self.logger.warning(f"Volatility analysis failed: {e}")
            return {}
    
    def _perform_momentum_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """モメンタム分析実行"""
        try:
            return {
                'short_momentum': self._calculate_short_momentum(data),
                'medium_momentum': self._calculate_medium_momentum(data),
                'long_momentum': self._calculate_long_momentum(data),
                'momentum_acceleration': self._calculate_momentum_acceleration(data)
            }
        except Exception as e:
            self.logger.warning(f"Momentum analysis failed: {e}")
            return {}
    
    def _detect_market_condition(self, data: pd.DataFrame) -> str:
        """市場状況検出"""
        try:
            # 簡易的な市場状況判定
            recent_data = data.tail(20)
            price_trend = recent_data['Close'].pct_change().mean()
            volatility = recent_data['Close'].pct_change().std()
            
            if abs(price_trend) < 0.001:
                return "sideways"
            elif price_trend > 0.001:
                return "trending_up"
            elif price_trend < -0.001:
                return "trending_down"
            elif volatility > 0.02:
                return "high_volatility"
            else:
                return "low_volatility"
                
        except Exception:
            return "unknown"
    
    # 技術指標計算メソッド（簡易版）
    def _calculate_price_trend(self, data: pd.DataFrame) -> float:
        """価格トレンド計算"""
        try:
            prices = data['Close'].values
            return np.polyfit(range(len(prices)), prices, 1)[0]
        except:
            return 0.0
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """出来高トレンド計算"""
        try:
            volumes = data['Volume'].values
            return np.polyfit(range(len(volumes)), volumes, 1)[0]
        except:
            return 0.0
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> float:
        """価格モメンタム計算"""
        try:
            return data['Close'].pct_change(10).iloc[-1]
        except:
            return 0.0
    
    def _calculate_relative_strength(self, data: pd.DataFrame) -> float:
        """相対強度計算"""
        try:
            return data['Close'].pct_change().tail(20).mean()
        except:
            return 0.0
    
    def _calculate_sma_signal(self, data: pd.DataFrame) -> float:
        """SMAシグナル計算"""
        try:
            sma_short = data['Close'].rolling(10).mean()
            sma_long = data['Close'].rolling(30).mean()
            return (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        except:
            return 0.0
    
    def _calculate_ema_signal(self, data: pd.DataFrame) -> float:
        """EMAシグナル計算"""
        try:
            ema_short = data['Close'].ewm(span=12).mean()
            ema_long = data['Close'].ewm(span=26).mean()
            return (ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
        except:
            return 0.0
    
    def _calculate_macd_signal(self, data: pd.DataFrame) -> float:
        """MACDシグナル計算"""
        try:
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            return (macd.iloc[-1] - signal.iloc[-1])
        except:
            return 0.0
    
    def _calculate_rsi_signal(self, data: pd.DataFrame) -> float:
        """RSIシグナル計算"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] - 50  # 中央値からの偏差
        except:
            return 0.0
    
    def _calculate_bollinger_signal(self, data: pd.DataFrame) -> float:
        """ボリンジャーバンドシグナル計算"""
        try:
            sma = data['Close'].rolling(20).mean()
            std = data['Close'].rolling(20).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            current_price = data['Close'].iloc[-1]
            band_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            return band_position - 0.5  # 中央からの偏差
        except:
            return 0.0
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> float:
        """出来高モメンタム計算"""
        try:
            return data['Volume'].pct_change(5).iloc[-1]
        except:
            return 0.0
    
    def _calculate_volume_price_correlation(self, data: pd.DataFrame) -> float:
        """出来高価格相関計算"""
        try:
            price_change = data['Close'].pct_change()
            volume_change = data['Volume'].pct_change()
            return price_change.corr(volume_change)
        except:
            return 0.0
    
    def _calculate_volume_breakout(self, data: pd.DataFrame) -> float:
        """出来高ブレイクアウト計算"""
        try:
            volume_ma = data['Volume'].rolling(20).mean()
            current_volume = data['Volume'].iloc[-1]
            return (current_volume - volume_ma.iloc[-1]) / volume_ma.iloc[-1]
        except:
            return 0.0
    
    def _calculate_realized_volatility(self, data: pd.DataFrame) -> float:
        """実現ボラティリティ計算"""
        try:
            returns = data['Close'].pct_change()
            return returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        except:
            return 0.0
    
    def _calculate_volatility_trend(self, data: pd.DataFrame) -> float:
        """ボラティリティトレンド計算"""
        try:
            returns = data['Close'].pct_change()
            vol_short = returns.rolling(10).std()
            vol_long = returns.rolling(30).std()
            return (vol_short.iloc[-1] - vol_long.iloc[-1]) / vol_long.iloc[-1]
        except:
            return 0.0
    
    def _calculate_risk_adjusted_return(self, data: pd.DataFrame) -> float:
        """リスク調整リターン計算"""
        try:
            returns = data['Close'].pct_change()
            mean_return = returns.mean()
            volatility = returns.std()
            return mean_return / volatility if volatility > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_short_momentum(self, data: pd.DataFrame) -> float:
        """短期モメンタム計算"""
        try:
            return data['Close'].pct_change(5).iloc[-1]
        except:
            return 0.0
    
    def _calculate_medium_momentum(self, data: pd.DataFrame) -> float:
        """中期モメンタム計算"""
        try:
            return data['Close'].pct_change(20).iloc[-1]
        except:
            return 0.0
    
    def _calculate_long_momentum(self, data: pd.DataFrame) -> float:
        """長期モメンタム計算"""
        try:
            return data['Close'].pct_change(60).iloc[-1]
        except:
            return 0.0
    
    def _calculate_momentum_acceleration(self, data: pd.DataFrame) -> float:
        """モメンタム加速度計算"""
        try:
            momentum = data['Close'].pct_change(10)
            return momentum.diff().iloc[-1]
        except:
            return 0.0
    
    async def _integrate_analysis_results(
        self, 
        analysis_results: List[Dict[str, Any]]
    ) -> List[AdvancedRankingResult]:
        """分析結果統合"""
        
        integrated_results = []
        
        for result in analysis_results:
            try:
                # 総合スコア計算
                final_score = self._calculate_integrated_score(result)
                
                # 多次元スコア
                multi_dimensional_scores = self._extract_dimensional_scores(result)
                
                # 動的重み
                dynamic_weights = self._calculate_dynamic_weights(result)
                
                # 信頼度レベル
                confidence_level = self._calculate_confidence_level(result)
                
                # 統合結果作成
                integrated_result = AdvancedRankingResult(
                    symbol=result['symbol'],
                    final_score=final_score,
                    multi_dimensional_scores=multi_dimensional_scores,
                    dynamic_weights=dynamic_weights,
                    confidence_level=confidence_level,
                    market_condition=result['market_condition'],
                    analysis_timestamp=result['timestamp'],
                    component_details=result,
                    performance_metrics=self._get_performance_metrics(),
                    integration_status="completed"
                )
                
                integrated_results.append(integrated_result)
                
            except Exception as e:
                self.logger.warning(f"Result integration failed for {result.get('symbol', 'unknown')}: {e}")
                continue
        
        # スコア順でソート
        integrated_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return integrated_results
    
    def _calculate_integrated_score(self, result: Dict[str, Any]) -> float:
        """統合スコア計算"""
        try:
            # 各分析結果から重要な値を抽出
            basic_score = np.mean(list(result.get('basic_analysis', {}).values()) or [0])
            technical_score = np.mean(list(result.get('technical_analysis', {}).values()) or [0])
            volume_score = np.mean(list(result.get('volume_analysis', {}).values()) or [0])
            volatility_score = np.mean(list(result.get('volatility_analysis', {}).values()) or [0])
            momentum_score = np.mean(list(result.get('momentum_analysis', {}).values()) or [0])
            
            # 重み付き合計
            weights = {
                'basic': 0.2,
                'technical': 0.25,
                'volume': 0.15,
                'volatility': 0.15,
                'momentum': 0.25
            }
            
            final_score = (
                basic_score * weights['basic'] +
                technical_score * weights['technical'] +
                volume_score * weights['volume'] +
                volatility_score * weights['volatility'] +
                momentum_score * weights['momentum']
            )
            
            # 正規化 (0-100)
            return max(0, min(100, final_score * 100))
            
        except Exception as e:
            self.logger.warning(f"Integrated score calculation failed: {e}")
            return 0.0
    
    def _extract_dimensional_scores(self, result: Dict[str, Any]) -> Dict[str, float]:
        """多次元スコア抽出"""
        return {
            'momentum': np.mean(list(result.get('momentum_analysis', {}).values()) or [0]),
            'technical': np.mean(list(result.get('technical_analysis', {}).values()) or [0]),
            'volume': np.mean(list(result.get('volume_analysis', {}).values()) or [0]),
            'volatility': np.mean(list(result.get('volatility_analysis', {}).values()) or [0]),
            'trend': np.mean(list(result.get('basic_analysis', {}).values()) or [0])
        }
    
    def _calculate_dynamic_weights(self, result: Dict[str, Any]) -> Dict[str, float]:
        """動的重み計算"""
        market_condition = result.get('market_condition', 'unknown')
        
        # 市場状況に応じた動的重み設定
        if market_condition == 'trending_up':
            return {'momentum': 0.4, 'technical': 0.3, 'volume': 0.2, 'volatility': 0.1}
        elif market_condition == 'trending_down':
            return {'momentum': 0.2, 'technical': 0.3, 'volume': 0.2, 'volatility': 0.3}
        elif market_condition == 'sideways':
            return {'momentum': 0.1, 'technical': 0.4, 'volume': 0.3, 'volatility': 0.2}
        elif market_condition == 'high_volatility':
            return {'momentum': 0.3, 'technical': 0.2, 'volume': 0.2, 'volatility': 0.3}
        else:
            return {'momentum': 0.25, 'technical': 0.25, 'volume': 0.25, 'volatility': 0.25}
    
    def _calculate_confidence_level(self, result: Dict[str, Any]) -> float:
        """信頼度レベル計算"""
        try:
            # 各分析の一貫性をチェック
            scores = []
            for analysis_type in ['basic_analysis', 'technical_analysis', 'volume_analysis', 'momentum_analysis']:
                analysis_data = result.get(analysis_type, {})
                if analysis_data:
                    scores.extend(list(analysis_data.values()))
            
            if not scores:
                return 0.0
            
            # スコアの標準偏差に基づく信頼度
            score_std = np.std(scores)
            confidence = max(0, 1 - score_std)  # 標準偏差が小さいほど信頼度高
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.5  # デフォルト信頼度
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """パフォーマンスメトリクス取得"""
        return self._performance_metrics.copy()
    
    def _record_performance(self, operation: str, duration: float, items_count: int):
        """パフォーマンス記録"""
        with self._thread_lock:
            if operation not in self._performance_metrics:
                self._performance_metrics[operation] = []
            
            self._performance_metrics[operation].append({
                'duration': duration,
                'items_count': items_count,
                'items_per_second': items_count / duration if duration > 0 else 0,
                'timestamp': datetime.now()
            })
            
            # 最新100件のみ保持
            if len(self._performance_metrics[operation]) > 100:
                self._performance_metrics[operation] = self._performance_metrics[operation][-100:]
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """キャッシュ有効性チェック"""
        return (datetime.now() - timestamp).total_seconds() < self.config.cache_ttl_seconds
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        return {
            'config': {
                'analysis_mode': self.config.analysis_mode.value,
                'optimization_level': self.config.optimization_level.value,
                'parallel_processing': self.config.enable_parallel_processing,
                'max_workers': self.config.max_workers
            },
            'cache_status': {
                'total_entries': sum(len(cache) for cache in self._cache.values()),
                'cache_types': list(self._cache.keys())
            },
            'performance': {
                'operations': list(self._performance_metrics.keys()),
                'last_update': self._last_update_time
            },
            'integration': {
                'hierarchical_system': self._hierarchical_system is not None,
                'hybrid_engine': self._hybrid_engine is not None,
                'scoring_engine': self._scoring_engine is not None
            }
        }
    
    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            
            self._cache.clear()
            self._performance_metrics.clear()
            
            self.logger.info("Advanced Ranking Engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
