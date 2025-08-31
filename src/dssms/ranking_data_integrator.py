"""
DSSMS Phase 2 Task 2.2: ランキングデータ統合器
既存システムからのデータ統合と前処理

機能:
- 複数データソースの統合
- テクニカル指標計算
- 市場コンテキスト分析
- データ品質管理
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import talib

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存システムインポート
from .dssms_data_manager import DSSMSDataManager
from .fundamental_analyzer import FundamentalAnalyzer
from config.logger_config import setup_logger

class RankingDataIntegrator:
    """
    ランキング用データ統合器
    
    機能:
    - 既存データマネージャーとの統合
    - マルチタイムフレームデータ統合
    - テクニカル指標一括計算
    - データ品質検証
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初期化"""
        self.logger = setup_logger('dssms.data_integrator')
        self.config = config
        
        # 既存コンポーネント初期化
        try:
            self.data_manager = DSSMSDataManager()
            self.fundamental_analyzer = FundamentalAnalyzer()
            self.logger.info("既存データ管理コンポーネント初期化成功")
        except Exception as e:
            self.logger.error(f"既存コンポーネント初期化エラー: {e}")
            raise
        
        # 並列処理設定
        self.max_workers = config.get("max_workers", 4)
        self.batch_size = config.get("batch_size", 10)
        
        # データキャッシュ
        self._integrated_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        self.logger.info("RankingDataIntegrator initialized")
    
    async def prepare_integrated_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        統合データ準備
        
        Args:
            symbols: 対象銘柄リスト
            
        Returns:
            Dict[str, Dict[str, Any]]: 銘柄別統合データ
        """
        try:
            self.logger.info(f"統合データ準備開始: {len(symbols)}銘柄")
            
            # 並列データ取得
            if len(symbols) > self.batch_size:
                integrated_data = await self._parallel_data_integration(symbols)
            else:
                integrated_data = await self._sequential_data_integration(symbols)
            
            # データ品質検証
            validated_data = self._validate_integrated_data(integrated_data)
            
            self.logger.info(f"統合データ準備完了: {len(validated_data)}銘柄")
            return validated_data
            
        except Exception as e:
            self.logger.error(f"統合データ準備エラー: {e}")
            return {}
    
    async def _parallel_data_integration(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """並列データ統合"""
        # バッチ分割
        batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]
        
        # 並列実行
        tasks = []
        for batch in batches:
            task = self._process_symbol_batch(batch)
            tasks.append(task)
        
        # 結果収集
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        integrated_data = {}
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error(f"バッチ処理エラー: {result}")
                continue
            integrated_data.update(result)
        
        return integrated_data
    
    async def _sequential_data_integration(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """逐次データ統合"""
        integrated_data = {}
        
        for symbol in symbols:
            try:
                symbol_data = await self._integrate_single_symbol(symbol)
                if symbol_data:
                    integrated_data[symbol] = symbol_data
            except Exception as e:
                self.logger.error(f"銘柄{symbol}のデータ統合エラー: {e}")
                continue
        
        return integrated_data
    
    async def _process_symbol_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """銘柄バッチ処理"""
        batch_data = {}
        
        for symbol in symbols:
            try:
                symbol_data = await self._integrate_single_symbol(symbol)
                if symbol_data:
                    batch_data[symbol] = symbol_data
            except Exception as e:
                self.logger.error(f"銘柄{symbol}のデータ統合エラー: {e}")
                continue
        
        return batch_data
    
    async def _integrate_single_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """単一銘柄データ統合"""
        try:
            # キャッシュチェック
            cached_data = self._get_cached_data(symbol)
            if cached_data:
                return cached_data
            
            # マルチタイムフレームデータ取得
            timeframe_data = self.data_manager.get_multi_timeframe_data(symbol)
            if not timeframe_data or not timeframe_data.get('daily'):
                self.logger.warning(f"データ取得失敗: {symbol}")
                return None
            
            # ファンダメンタルデータ取得
            fundamental_data = await self._get_fundamental_data(symbol)
            
            # テクニカル指標計算
            technical_indicators = self._calculate_technical_indicators(timeframe_data)
            
            # 市場コンテキスト分析
            market_context = self._analyze_market_context(timeframe_data, symbol)
            
            # データ品質スコア計算
            quality_score = self._calculate_data_quality_score(timeframe_data, technical_indicators)
            
            # 統合データ作成
            integrated_data = {
                'timeframe_data': timeframe_data,
                'fundamental_data': fundamental_data,
                'technical_indicators': technical_indicators,
                'market_context': market_context,
                'data_quality_score': quality_score,
                'last_updated': datetime.now(),
                'symbol': symbol
            }
            
            # キャッシュ保存
            self._cache_data(symbol, integrated_data)
            
            return integrated_data
            
        except Exception as e:
            self.logger.error(f"銘柄{symbol}のデータ統合エラー: {e}")
            return None
    
    async def _get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """ファンダメンタルデータ取得"""
        try:
            if self.fundamental_analyzer:
                fundamental_metrics = self.fundamental_analyzer.analyze_fundamentals(symbol)
                return fundamental_metrics
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"ファンダメンタルデータ取得エラー ({symbol}): {e}")
            return {}
    
    def _calculate_technical_indicators(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """テクニカル指標計算"""
        indicators = {}
        
        for timeframe, data in timeframe_data.items():
            if data is None or data.empty:
                indicators[timeframe] = {}
                continue
            
            try:
                timeframe_indicators = {}
                
                # 価格データ準備
                close = data['Close'].values
                high = data['High'].values
                low = data['Low'].values
                volume = data['Volume'].values if 'Volume' in data.columns else None
                
                # 移動平均系
                timeframe_indicators['sma_5'] = self._safe_talib_call(talib.SMA, close, timeperiod=5)
                timeframe_indicators['sma_25'] = self._safe_talib_call(talib.SMA, close, timeperiod=25)
                timeframe_indicators['sma_75'] = self._safe_talib_call(talib.SMA, close, timeperiod=75)
                timeframe_indicators['ema_12'] = self._safe_talib_call(talib.EMA, close, timeperiod=12)
                timeframe_indicators['ema_26'] = self._safe_talib_call(talib.EMA, close, timeperiod=26)
                
                # オシレーター系
                timeframe_indicators['rsi'] = self._safe_talib_call(talib.RSI, close, timeperiod=14)
                timeframe_indicators['macd'], timeframe_indicators['macd_signal'], timeframe_indicators['macd_hist'] = \
                    self._safe_talib_call(talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9)
                timeframe_indicators['stoch_k'], timeframe_indicators['stoch_d'] = \
                    self._safe_talib_call(talib.STOCH, high, low, close)
                
                # ボラティリティ系
                timeframe_indicators['bbands_upper'], timeframe_indicators['bbands_middle'], timeframe_indicators['bbands_lower'] = \
                    self._safe_talib_call(talib.BBANDS, close, timeperiod=20)
                timeframe_indicators['atr'] = self._safe_talib_call(talib.ATR, high, low, close, timeperiod=14)
                
                # ボリューム系（ボリュームデータがある場合）
                if volume is not None and len(volume) > 0:
                    timeframe_indicators['ad'] = self._safe_talib_call(talib.AD, high, low, close, volume)
                    timeframe_indicators['obv'] = self._safe_talib_call(talib.OBV, close, volume)
                
                # パーフェクトオーダー判定
                timeframe_indicators['perfect_order'] = self._detect_perfect_order(timeframe_indicators)
                
                # トレンド強度
                timeframe_indicators['trend_strength'] = self._calculate_trend_strength(timeframe_indicators)
                
                indicators[timeframe] = timeframe_indicators
                
            except Exception as e:
                self.logger.error(f"テクニカル指標計算エラー ({timeframe}): {e}")
                indicators[timeframe] = {}
        
        return indicators
    
    def _safe_talib_call(self, func, *args, **kwargs):
        """TALib関数の安全な呼び出し"""
        try:
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                return tuple(np.nan_to_num(arr, nan=0.0) for arr in result)
            else:
                return np.nan_to_num(result, nan=0.0)
        except Exception as e:
            self.logger.warning(f"TALib関数エラー: {func.__name__}: {e}")
            if hasattr(func, '__name__') and 'STOCH' in func.__name__:
                return np.array([0.0]), np.array([0.0])
            elif hasattr(func, '__name__') and any(name in func.__name__ for name in ['MACD', 'BBANDS']):
                return np.array([0.0]), np.array([0.0]), np.array([0.0])
            else:
                return np.array([0.0])
    
    def _detect_perfect_order(self, indicators: Dict[str, Any]) -> Dict[str, bool]:
        """パーフェクトオーダー検出"""
        try:
            sma_5 = indicators.get('sma_5', np.array([]))
            sma_25 = indicators.get('sma_25', np.array([]))
            sma_75 = indicators.get('sma_75', np.array([]))
            
            if len(sma_5) == 0 or len(sma_25) == 0 or len(sma_75) == 0:
                return {'bullish': False, 'bearish': False}
            
            # 最新値で判定
            current_5 = sma_5[-1] if len(sma_5) > 0 else 0
            current_25 = sma_25[-1] if len(sma_25) > 0 else 0
            current_75 = sma_75[-1] if len(sma_75) > 0 else 0
            
            bullish_order = current_5 > current_25 > current_75
            bearish_order = current_5 < current_25 < current_75
            
            return {'bullish': bullish_order, 'bearish': bearish_order}
            
        except Exception as e:
            self.logger.warning(f"パーフェクトオーダー検出エラー: {e}")
            return {'bullish': False, 'bearish': False}
    
    def _calculate_trend_strength(self, indicators: Dict[str, Any]) -> float:
        """トレンド強度計算"""
        try:
            # RSI、MACD、移動平均傾きからトレンド強度を計算
            rsi = indicators.get('rsi', np.array([]))
            macd = indicators.get('macd', np.array([]))
            sma_5 = indicators.get('sma_5', np.array([]))
            
            if len(rsi) == 0 or len(macd) == 0 or len(sma_5) == 0:
                return 0.0
            
            # RSIからのトレンド強度（50を中心とした偏差）
            rsi_strength = abs(rsi[-1] - 50) / 50 if len(rsi) > 0 else 0
            
            # MACDからのトレンド強度
            macd_strength = min(abs(macd[-1]) / 100, 1.0) if len(macd) > 0 else 0
            
            # 移動平均の傾きからトレンド強度
            if len(sma_5) >= 5:
                slope = (sma_5[-1] - sma_5[-5]) / sma_5[-5] if sma_5[-5] != 0 else 0
                slope_strength = min(abs(slope) * 100, 1.0)
            else:
                slope_strength = 0
            
            # 総合トレンド強度
            trend_strength = (rsi_strength * 0.3 + macd_strength * 0.4 + slope_strength * 0.3)
            return min(trend_strength, 1.0)
            
        except Exception as e:
            self.logger.warning(f"トレンド強度計算エラー: {e}")
            return 0.0
    
    def _analyze_market_context(self, timeframe_data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """市場コンテキスト分析"""
        try:
            daily_data = timeframe_data.get('daily')
            if daily_data is None or daily_data.empty:
                return {}
            
            context = {}
            
            # 価格レンジ分析
            recent_high = daily_data['High'].tail(20).max()
            recent_low = daily_data['Low'].tail(20).min()
            current_price = daily_data['Close'].iloc[-1]
            
            context['price_position'] = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            # ボリューム分析
            if 'Volume' in daily_data.columns:
                avg_volume = daily_data['Volume'].tail(20).mean()
                current_volume = daily_data['Volume'].iloc[-1]
                context['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                context['volume_ratio'] = 1.0
            
            # ボラティリティ分析
            returns = daily_data['Close'].pct_change().dropna()
            context['volatility'] = returns.tail(20).std() if len(returns) > 1 else 0.0
            
            # ギャップ分析
            if len(daily_data) >= 2:
                yesterday_close = daily_data['Close'].iloc[-2]
                today_open = daily_data['Open'].iloc[-1]
                context['gap_ratio'] = (today_open - yesterday_close) / yesterday_close if yesterday_close != 0 else 0.0
            else:
                context['gap_ratio'] = 0.0
            
            return context
            
        except Exception as e:
            self.logger.warning(f"市場コンテキスト分析エラー ({symbol}): {e}")
            return {}
    
    def _calculate_data_quality_score(self, timeframe_data: Dict[str, pd.DataFrame], 
                                    technical_indicators: Dict[str, Dict[str, Any]]) -> float:
        """データ品質スコア計算"""
        try:
            quality_score = 0.0
            total_weight = 0.0
            
            # データ完整性チェック
            for timeframe, data in timeframe_data.items():
                if data is not None and not data.empty:
                    # 欠損値チェック
                    missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
                    completeness_score = 1.0 - missing_ratio
                    
                    # データ期間チェック
                    expected_days = {'daily': 250, 'weekly': 52, 'monthly': 12}
                    actual_days = len(data)
                    period_score = min(actual_days / expected_days.get(timeframe, 250), 1.0)
                    
                    # 時間軸別重み
                    timeframe_weight = {'daily': 0.5, 'weekly': 0.3, 'monthly': 0.2}.get(timeframe, 0.1)
                    
                    timeframe_score = (completeness_score * 0.6 + period_score * 0.4)
                    quality_score += timeframe_score * timeframe_weight
                    total_weight += timeframe_weight
            
            # テクニカル指標品質チェック
            indicator_quality = 0.0
            indicator_count = 0
            
            for timeframe_indicators in technical_indicators.values():
                for indicator_name, indicator_data in timeframe_indicators.items():
                    if isinstance(indicator_data, np.ndarray) and len(indicator_data) > 0:
                        # NaN/無限大チェック
                        valid_ratio = np.isfinite(indicator_data).sum() / len(indicator_data)
                        indicator_quality += valid_ratio
                        indicator_count += 1
            
            if indicator_count > 0:
                indicator_quality /= indicator_count
                quality_score = (quality_score / total_weight * 0.7 + indicator_quality * 0.3) if total_weight > 0 else indicator_quality
            else:
                quality_score = quality_score / total_weight if total_weight > 0 else 0.0
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"データ品質スコア計算エラー: {e}")
            return 0.5
    
    def _validate_integrated_data(self, integrated_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """統合データ検証"""
        validated_data = {}
        
        for symbol, data in integrated_data.items():
            try:
                # 最小品質要件チェック
                quality_score = data.get('data_quality_score', 0.0)
                min_quality = self.config.get('min_data_quality', 0.3)
                
                if quality_score < min_quality:
                    self.logger.warning(f"品質不足によりスキップ: {symbol} (品質スコア: {quality_score:.3f})")
                    continue
                
                # 必須データ存在チェック
                required_keys = ['timeframe_data', 'technical_indicators']
                if not all(key in data for key in required_keys):
                    self.logger.warning(f"必須データ不足によりスキップ: {symbol}")
                    continue
                
                validated_data[symbol] = data
                
            except Exception as e:
                self.logger.error(f"データ検証エラー ({symbol}): {e}")
                continue
        
        return validated_data
    
    def _get_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """キャッシュからデータ取得"""
        if symbol not in self._integrated_cache:
            return None
        
        # TTLチェック
        cache_ttl = self.config.get('cache_ttl_minutes', 30)
        cache_time = self._cache_timestamps.get(symbol)
        if cache_time and datetime.now() - cache_time > timedelta(minutes=cache_ttl):
            # キャッシュ期限切れ
            del self._integrated_cache[symbol]
            del self._cache_timestamps[symbol]
            return None
        
        return self._integrated_cache[symbol]
    
    def _cache_data(self, symbol: str, data: Dict[str, Any]):
        """データをキャッシュ"""
        if self.config.get('cache_enabled', True):
            self._integrated_cache[symbol] = data
            self._cache_timestamps[symbol] = datetime.now()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計情報"""
        return {
            'cache_size': len(self._integrated_cache),
            'cached_symbols': list(self._integrated_cache.keys()),
            'cache_hit_ratio': len(self._integrated_cache) / max(len(self._cache_timestamps), 1)
        }
    
    def clear_cache(self):
        """キャッシュクリア"""
        self._integrated_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("データ統合キャッシュをクリアしました")
