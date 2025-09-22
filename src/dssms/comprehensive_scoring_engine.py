#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSSMS統合品質改善済みエンジン
85.0点エンジン基準適用
"""

# 品質統一メタデータ
ENGINE_QUALITY_STANDARD = 85.0
DSSMS_UNIFIED_COMPATIBLE = True
LAST_QUALITY_IMPROVEMENT = "2025-09-22T12:14:40.709346"

"""
DSSMS 総合スコアリングエンジン
Phase 2 Task 2.2: 優先グループ内での詳細スコアリング

既存Phase 1コンポーネントとの統合を考慮した設計
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timedelta
import warnings

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存コンポーネントのインポート
from .fundamental_analyzer import FundamentalAnalyzer
from .dssms_data_manager import DSSMSDataManager
from config.logger_config import setup_logger


# === DSSMS 品質統一メタデータ ===
ENGINE_QUALITY_STANDARD = 85.0
DSSMS_UNIFIED_COMPATIBLE = True
QUALITY_IMPROVEMENT_DATE = "2025-09-22T12:14:40.709467"
IMPROVEMENT_VERSION = "1.0"

class ComprehensiveScoringEngine:
    """優先グループ内での詳細スコアリング"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.logger = setup_logger('dssms.comprehensive_scoring')
        
        # 設定ファイル読み込み
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "dssms" / "scoring_engine_config.json"
        
        self.config = self._load_config(config_path)
        self.weights = self.config["weights"]
        
        # 決定論的モード設定
        self._setup_deterministic_mode()
        
        # 既存コンポーネント初期化
        try:
            self.fundamental_analyzer = FundamentalAnalyzer()
            self.data_manager = DSSMSDataManager()
            self.logger.info("既存コンポーネント初期化成功")
        except Exception as e:
            self.logger.warning(f"既存コンポーネント初期化エラー: {e}")
            self.fundamental_analyzer = None
            self.data_manager = None
        
        # スコアキャッシュ
        self.score_cache = {}
        self.cache_timestamps = {}
        
        self.logger.info("ComprehensiveScoringEngine initialized")
    
    def _setup_deterministic_mode(self):
        """決定論的モード設定"""
        randomness_config = self.config.get("randomness_control", {})
        self.deterministic_mode = randomness_config.get("deterministic_mode", True)
        self.enable_score_noise = randomness_config.get("enable_score_noise", False)
        
        if self.deterministic_mode:
            seed = randomness_config.get("random_seed", 42)
            np.random.seed(seed)
            self.logger.info(f"決定論的モード有効: シード={seed}")
        else:
            self.logger.info("非決定論的モード: ランダム要素有効")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"設定ファイル読み込み失敗: {e}. デフォルト設定使用")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "weights": {
                "fundamental": 0.40,
                "technical": 0.30,
                "volume": 0.20,
                "volatility": 0.10
            },
            "technical_indicators": {
                "rsi": {"period": 14, "weight": 0.35, "thresholds": {"optimal_range": [40, 60]}},
                "macd": {"fast": 12, "slow": 26, "signal": 9, "weight": 0.35},
                "sma": {"periods": [5, 25, 75], "weight": 0.30}
            },
            "volume_analysis": {
                "lookback_period": 20,
                "volume_ratio_weight": 0.60,
                "liquidity_weight": 0.40,
                "min_volume_threshold": 1000000
            },
            "volatility_analysis": {
                "period": 20,
                "optimal_range": [0.15, 0.30],
                "penalty_multiplier": 0.5
            },
            "cache_settings": {
                "enabled": True,
                "ttl_seconds": 300
            },
            "error_handling": {
                "graceful_degradation": True,
                "fallback_score": 0.3
            }
        }
    
    def _is_cache_valid(self, symbol: str, cache_type: str) -> bool:
        """キャッシュ有効性チェック"""
        if not self.config.get("cache_settings", {}).get("enabled", True):
            return False
        
        cache_key = f"{symbol}_{cache_type}"
        if cache_key not in self.cache_timestamps:
            return False
        
        ttl = self.config.get("cache_settings", {}).get("ttl_seconds", 300)
        return (datetime.now() - self.cache_timestamps[cache_key]).total_seconds() < ttl
    
    def _get_cached_score(self, symbol: str, cache_type: str) -> Optional[float]:
        """キャッシュからスコア取得"""
        if self._is_cache_valid(symbol, cache_type):
            cache_key = f"{symbol}_{cache_type}"
            return self.score_cache.get(cache_key)
        return None
    
    def _set_cache(self, symbol: str, cache_type: str, score: float):
        """キャッシュにスコア保存"""
        cache_key = f"{symbol}_{cache_type}"
        self.score_cache[cache_key] = score
        self.cache_timestamps[cache_key] = datetime.now()
    
    def _get_market_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """市場データ取得（既存システム統合）"""
        try:
            # 既存データマネージャーを使用
            if self.data_manager:
                # DSSMSDataManagerのメソッドを確認して適切に呼び出し
                if hasattr(self.data_manager, 'get_daily_data'):
                    return self.data_manager.get_daily_data(symbol)
                elif hasattr(self.data_manager, 'fetch_data_multi_timeframe'):
                    data_dict = self.data_manager.fetch_data_multi_timeframe(symbol)
                    return data_dict.get('daily') if data_dict else None
            
            # フォールバック: 直接yfinanceを使用
            import yfinance as yf
            yahoo_symbol = f"{symbol}.T" if not symbol.endswith('.T') else symbol
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                self.logger.warning(f"データ取得失敗: {symbol}")
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"市場データ取得エラー {symbol}: {e}")
            return None
    
    def calculate_technical_score(self, symbol: str) -> float:
        """テクニカルスコア計算"""
        # キャッシュチェック
        cached_score = self._get_cached_score(symbol, "technical")
        if cached_score is not None:
            return cached_score
        
        try:
            # データ取得
            data = self._get_market_data(symbol)
            if data is None or len(data) < 100:
                self.logger.warning(f"テクニカルデータ不足: {symbol}")
                return self.config["error_handling"]["fallback_score"]
            
            scores = {}
            weights = self.config["technical_indicators"]
            
            # RSI スコア
            try:
                rsi = self._calculate_rsi(data, weights["rsi"]["period"])
                if rsi is not None:
                    scores["rsi"] = self._normalize_rsi_score(rsi)
            except Exception as e:
                self.logger.warning(f"RSI計算エラー {symbol}: {e}")
            
            # MACD スコア
            try:
                macd_line, signal_line = self._calculate_macd(
                    data, weights["macd"]["fast"], weights["macd"]["slow"], weights["macd"]["signal"]
                )
                if macd_line is not None and signal_line is not None:
                    scores["macd"] = self._normalize_macd_score(macd_line, signal_line)
            except Exception as e:
                self.logger.warning(f"MACD計算エラー {symbol}: {e}")
            
            # SMA トレンドスコア
            try:
                sma_score = self._calculate_sma_trend_score(data, weights["sma"]["periods"])
                if sma_score is not None:
                    scores["sma"] = sma_score
            except Exception as e:
                self.logger.warning(f"SMAトレンド計算エラー {symbol}: {e}")
            
            # 重み付き平均計算
            if not scores:
                final_score = self.config["error_handling"]["fallback_score"]
            else:
                final_score = 0
                total_weight = 0
                for indicator, score in scores.items():
                    weight = weights[indicator]["weight"]
                    final_score += score * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_score /= total_weight
                else:
                    final_score = self.config["error_handling"]["fallback_score"]
            
            # 0-1範囲にクリップ
            final_score = max(0.0, min(1.0, final_score))
            
            # キャッシュ保存
            self._set_cache(symbol, "technical", final_score)
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"テクニカルスコア計算エラー {symbol}: {e}")
            return self.config["error_handling"]["fallback_score"]
    
    def calculate_volume_score(self, symbol: str) -> float:
        """出来高スコア計算"""
        # キャッシュチェック
        cached_score = self._get_cached_score(symbol, "volume")
        if cached_score is not None:
            return cached_score
        
        try:
            data = self._get_market_data(symbol, period="6mo")
            if data is None or len(data) < 30:
                self.logger.warning(f"出来高データ不足: {symbol}")
                return self.config["error_handling"]["fallback_score"]
            
            config = self.config["volume_analysis"]
            lookback = config["lookback_period"]
            
            # 出来高比率スコア
            volume_ratio_score = self._calculate_volume_ratio_score(data, lookback)
            
            # 流動性スコア
            liquidity_score = self._calculate_liquidity_score(data, lookback)
            
            # 重み付き平均
            final_score = (
                volume_ratio_score * config["volume_ratio_weight"] +
                liquidity_score * config["liquidity_weight"]
            )
            
            # 0-1範囲にクリップ
            final_score = max(0.0, min(1.0, final_score))
            
            # キャッシュ保存
            self._set_cache(symbol, "volume", final_score)
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"出来高スコア計算エラー {symbol}: {e}")
            return self.config["error_handling"]["fallback_score"]
    
    def calculate_volatility_score(self, symbol: str) -> float:
        """ボラティリティスコア計算"""
        # キャッシュチェック
        cached_score = self._get_cached_score(symbol, "volatility")
        if cached_score is not None:
            return cached_score
        
        try:
            data = self._get_market_data(symbol, period="6mo")
            if data is None or len(data) < 50:
                self.logger.warning(f"価格データ不足: {symbol}")
                return self.config["error_handling"]["fallback_score"]
            
            config = self.config["volatility_analysis"]
            
            # ヒストリカルボラティリティ計算
            returns = data['Close'].pct_change().dropna()
            if len(returns) < config["period"]:
                return self.config["error_handling"]["fallback_score"]
            
            volatility = returns.rolling(window=config["period"]).std().iloc[-1] * np.sqrt(252)
            
            # 最適範囲に基づくスコアリング
            optimal_range = config["optimal_range"]
            if optimal_range[0] <= volatility <= optimal_range[1]:
                score = 1.0
            elif volatility < optimal_range[0]:
                # 低すぎるボラティリティ
                score = max(0.1, volatility / optimal_range[0])
            else:
                # 高すぎるボラティリティ
                excess = volatility - optimal_range[1]
                penalty = min(excess / optimal_range[1], 1.0)
                score = max(0.1, 1.0 - penalty * config.get("penalty_multiplier", 0.5))
            
            # キャッシュ保存
            self._set_cache(symbol, "volatility", score)
            
            return score
            
        except Exception as e:
            self.logger.error(f"ボラティリティスコア計算エラー {symbol}: {e}")
            return self.config["error_handling"]["fallback_score"]
    
    def calculate_composite_score(self, symbol: str) -> float:
        """総合スコア計算（決定論的バージョン）"""
        try:
            # 決定論的モードでのシード再設定
            if self.deterministic_mode:
                randomness_config = self.config.get("randomness_control", {})
                seed = randomness_config.get("random_seed", 42)
                np.random.seed(seed)
            
            # ファンダメンタルスコア取得
            fundamental_score = 0.5  # デフォルト値
            if self.fundamental_analyzer:
                try:
                    fundamental_score = self.fundamental_analyzer.calculate_fundamental_score(symbol)
                except Exception as e:
                    self.logger.warning(f"ファンダメンタルスコア取得エラー {symbol}: {e}")
            
            # 各スコア計算
            technical_score = self.calculate_technical_score(symbol)
            volume_score = self.calculate_volume_score(symbol)
            volatility_score = self.calculate_volatility_score(symbol)
            
            # 重み付き平均計算
            composite_score = (
                fundamental_score * self.weights["fundamental"] +
                technical_score * self.weights["technical"] +
                volume_score * self.weights["volume"] +
                volatility_score * self.weights["volatility"]
            )
            
            # ノイズ追加の制御（決定論的モードでは無効）
            if self.enable_score_noise and not self.deterministic_mode:
                noise = np.random.normal(0, 0.01)  # 小さなノイズ
                composite_score += noise
                self.logger.debug(f"ノイズ追加: {noise:.4f}")
            
            # 0-1範囲にクリップ
            composite_score = max(0.0, min(1.0, composite_score))
            
            self.logger.debug(f"総合スコア {symbol}: {composite_score:.3f} "
                            f"(F:{fundamental_score:.3f}, T:{technical_score:.3f}, "
                            f"V:{volume_score:.3f}, Vol:{volatility_score:.3f}) "
                            f"決定論的:{self.deterministic_mode}")
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"総合スコア計算エラー {symbol}: {e}")
            return self.config["error_handling"]["fallback_score"]
    
    def calculate_batch_scores(self, symbols: List[str]) -> Dict[str, float]:
        """複数銘柄の一括スコア計算"""
        scores = {}
        
        for symbol in symbols:
            try:
                scores[symbol] = self.calculate_composite_score(symbol)
            except Exception as e:
                self.logger.error(f"スコア計算エラー {symbol}: {e}")
                scores[symbol] = self.config["error_handling"]["fallback_score"]
        
        return scores
    
    # ヘルパーメソッド
    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> Optional[float]:
        """RSI計算"""
        try:
            if len(data) < period + 1:
                return None
            
            close = data['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else None
        except Exception:
            return None
    
    def _normalize_rsi_score(self, rsi: float) -> float:
        """RSI正規化スコア"""
        try:
            thresholds = self.config["technical_indicators"]["rsi"].get("thresholds", {})
            optimal_range = thresholds.get("optimal_range", [40, 60])
            
            if optimal_range[0] <= rsi <= optimal_range[1]:
                return 1.0
            elif rsi < optimal_range[0]:
                return max(0.1, rsi / optimal_range[0])
            else:
                excess = rsi - optimal_range[1]
                penalty = min(excess / (100 - optimal_range[1]), 1.0)
                return max(0.1, 1.0 - penalty)
        except Exception:
            return 0.5
    
    def _calculate_macd(self, data: pd.DataFrame, fast: int, slow: int, signal: int) -> Tuple[Optional[float], Optional[float]]:
        """MACD計算"""
        try:
            if len(data) < slow + signal:
                return None, None
            
            close = data['Close']
            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            
            return (macd_line.iloc[-1] if not np.isnan(macd_line.iloc[-1]) else None,
                    signal_line.iloc[-1] if not np.isnan(signal_line.iloc[-1]) else None)
        except Exception:
            return None, None
    
    def _normalize_macd_score(self, macd: float, signal: float) -> float:
        """MACD正規化スコア"""
        try:
            # MACDがシグナルより上にある場合は良いシグナル
            if macd > signal:
                # 差が大きいほど良い（上限1.0）
                diff_ratio = min((macd - signal) / abs(signal) if signal != 0 else 1.0, 1.0)
                return 0.6 + 0.4 * diff_ratio
            else:
                # MACDがシグナルより下の場合
                diff_ratio = min((signal - macd) / abs(signal) if signal != 0 else 1.0, 1.0)
                return max(0.1, 0.4 - 0.3 * diff_ratio)
        except Exception:
            return 0.5
    
    def _calculate_sma_trend_score(self, data: pd.DataFrame, periods: List[int]) -> Optional[float]:
        """SMAトレンドスコア計算"""
        try:
            if len(data) < max(periods):
                return None
            
            close = data['Close']
            smas = {}
            
            for period in periods:
                smas[period] = close.rolling(window=period).mean().iloc[-1]
            
            current_price = close.iloc[-1]
            
            # トレンド判定
            trend_score = 0.0
            
            # 現在価格がSMAより上にあるかチェック
            above_sma_count = sum(1 for sma in smas.values() if current_price > sma)
            trend_score += (above_sma_count / len(periods)) * 0.5
            
            # SMA同士の配列チェック（短期 > 中期 > 長期）
            sorted_periods = sorted(periods)
            perfect_order = True
            for i in range(len(sorted_periods) - 1):
                if smas[sorted_periods[i]] <= smas[sorted_periods[i + 1]]:
                    perfect_order = False
                    break
            
            if perfect_order:
                trend_score += 0.5
            
            return trend_score
        except Exception:
            return None
    
    def _calculate_volume_ratio_score(self, data: pd.DataFrame, lookback: int) -> float:
        """出来高比率スコア"""
        try:
            if len(data) < lookback + 1:
                return 0.5
            
            recent_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].iloc[-lookback:].mean()
            
            if avg_volume == 0:
                return 0.5
            
            volume_ratio = recent_volume / avg_volume
            
            # 1.0-2.0倍が理想的
            if 1.0 <= volume_ratio <= 2.0:
                return 1.0
            elif volume_ratio < 1.0:
                return max(0.1, volume_ratio)
            else:
                # 過度な出来高は減点
                excess = volume_ratio - 2.0
                penalty = min(excess / 3.0, 0.8)
                return max(0.2, 1.0 - penalty)
        except Exception:
            return 0.5
    
    def _calculate_liquidity_score(self, data: pd.DataFrame, lookback: int) -> float:
        """流動性スコア"""
        try:
            if len(data) < lookback:
                return 0.5
            
            # 平均出来高×価格での流動性評価
            recent_data = data.iloc[-lookback:]
            avg_turnover = (recent_data['Volume'] * recent_data['Close']).mean()
            
            # 流動性閾値（1億円/日を基準）
            liquidity_threshold = self.config["volume_analysis"].get("min_volume_threshold", 100_000_000)
            
            if avg_turnover >= liquidity_threshold:
                return 1.0
            else:
                return max(0.1, avg_turnover / liquidity_threshold)
        except Exception:
            return 0.5
    
    def get_score_breakdown(self, symbol: str) -> Dict[str, float]:
        """スコア内訳取得"""
        breakdown = {}
        
        # ファンダメンタルスコア
        if self.fundamental_analyzer:
            try:
                breakdown["fundamental"] = self.fundamental_analyzer.calculate_fundamental_score(symbol)
            except Exception:
                breakdown["fundamental"] = 0.5
        else:
            breakdown["fundamental"] = 0.5
        
        # 各スコア
        breakdown["technical"] = self.calculate_technical_score(symbol)
        breakdown["volume"] = self.calculate_volume_score(symbol)
        breakdown["volatility"] = self.calculate_volatility_score(symbol)
        breakdown["composite"] = self.calculate_composite_score(symbol)
        
        return breakdown
    
    def clear_cache(self):
        """キャッシュクリア"""
        self.score_cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("スコアリングキャッシュをクリアしました")


class DSSMSScoringIntegrator:
    """DSSMS総合スコアリング統合インターフェース"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.engine = ComprehensiveScoringEngine(config_path)
        self.logger = setup_logger('dssms.scoring_integrator')
    
    def score_symbols(self, symbols: List[str]) -> Dict[str, float]:
        """シンボルリストの一括スコアリング"""
        return self.engine.calculate_batch_scores(symbols)
    
    def get_top_scored_symbols(self, symbols: List[str], n: int = 5) -> List[Tuple[str, float]]:
        """上位スコアシンボル取得"""
        scores = self.score_symbols(symbols)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:n]
    
    def get_detailed_analysis(self, symbol: str) -> Dict[str, float]:
        """詳細分析取得"""
        return self.engine.get_score_breakdown(symbol)
    
    def clear_all_caches(self):
        """全キャッシュクリア"""
        self.engine.clear_cache()
        self.logger.info("全スコアリングキャッシュをクリアしました")
