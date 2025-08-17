"""
DSSMS Perfect Order Detection Engine
SBI証券準拠のマルチタイムフレーム・パーフェクトオーダー検出システム

既存unified_trend_detector.pyをラッパーで拡張し、DSSMS専用機能を追加
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存システムインポート
try:
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError:
    # フォールバック用の簡易実装
    UnifiedTrendDetector = None

from config.logger_config import setup_logger

@dataclass
class PerfectOrderResult:
    """パーフェクトオーダー検出結果"""
    symbol: str
    timeframe: str
    is_perfect_order: bool
    sma_short: float
    sma_medium: float
    sma_long: float
    current_price: float
    strength_score: float  # 0-1の強度スコア
    trend_duration_days: int
    detection_timestamp: datetime

@dataclass
class MultiTimeframePerfectOrder:
    """複数時間軸パーフェクトオーダー結果"""
    symbol: str
    daily_result: PerfectOrderResult
    weekly_result: PerfectOrderResult
    monthly_result: PerfectOrderResult
    priority_level: int  # 1=全軸, 2=月週, 3=その他
    composite_score: float  # 総合スコア
    analysis_timestamp: datetime

class PerfectOrderDetector:
    """
    SBI証券準拠のマルチタイムフレーム・パーフェクトオーダー検出エンジン
    
    機能:
    - 日足・週足・月足の3時間軸でのパーフェクトオーダー検出
    - SBI証券のMA期間設定準拠 
    - 強度スコア計算
    - 優先度判定（全軸 > 月週 > その他）
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス（None時はデフォルト使用）
        """
        self.logger = setup_logger('dssms.perfect_order')
        
        # SBI証券準拠のMA期間設定
        self.timeframes = {
            "daily": {"short": 5, "medium": 25, "long": 75},
            "weekly": {"short": 13, "medium": 26, "long": 52},
            "monthly": {"short": 9, "medium": 24, "long": 60}
        }
        
        # 既存統一トレンド判定器をラッパーで活用
        try:
            if UnifiedTrendDetector:
                self.unified_detector = UnifiedTrendDetector()
            else:
                self.unified_detector = None
        except Exception as e:
            self.logger.warning(f"UnifiedTrendDetector initialization failed: {e}")
            self.unified_detector = None
        
        # キャッシュシステム
        self._cache = {}
        self._cache_timeout = timedelta(minutes=5)
        
        self.logger.info("PerfectOrderDetector initialized with SBI timeframes")
    
    def detect_perfect_order(self, data: pd.DataFrame, timeframe: str, symbol: str) -> PerfectOrderResult:
        """
        単一時間軸でのパーフェクトオーダー検出
        
        Args:
            data: OHLCV価格データ
            timeframe: 時間軸 ("daily", "weekly", "monthly")
            symbol: 銘柄コード
            
        Returns:
            PerfectOrderResult: 検出結果
        """
        try:
            if timeframe not in self.timeframes:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            if data.empty or len(data) < self.timeframes[timeframe]['long']:
                # データ不足の場合
                return PerfectOrderResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    is_perfect_order=False,
                    sma_short=0.0,
                    sma_medium=0.0,
                    sma_long=0.0,
                    current_price=0.0,
                    strength_score=0.0,
                    trend_duration_days=0,
                    detection_timestamp=datetime.now()
                )
            
            periods = self.timeframes[timeframe]
            
            # SMA計算
            sma_short = data['Close'].rolling(window=periods['short']).mean().iloc[-1]
            sma_medium = data['Close'].rolling(window=periods['medium']).mean().iloc[-1]
            sma_long = data['Close'].rolling(window=periods['long']).mean().iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # パーフェクトオーダー判定
            is_perfect_order = (
                not pd.isna(current_price) and
                not pd.isna(sma_short) and
                not pd.isna(sma_medium) and
                not pd.isna(sma_long) and
                current_price > sma_short > sma_medium > sma_long
            )
            
            # 強度スコア計算（価格間の乖離度合い）
            strength_score = self._calculate_strength_score(
                current_price, sma_short, sma_medium, sma_long
            )
            
            # トレンド継続日数
            trend_duration = self._calculate_trend_duration(data, periods)
            
            result = PerfectOrderResult(
                symbol=symbol,
                timeframe=timeframe,
                is_perfect_order=is_perfect_order,
                sma_short=float(sma_short) if not pd.isna(sma_short) else 0.0,
                sma_medium=float(sma_medium) if not pd.isna(sma_medium) else 0.0,
                sma_long=float(sma_long) if not pd.isna(sma_long) else 0.0,
                current_price=float(current_price) if not pd.isna(current_price) else 0.0,
                strength_score=strength_score,
                trend_duration_days=trend_duration,
                detection_timestamp=datetime.now()
            )
            
            self.logger.debug(f"Perfect order detection for {symbol} {timeframe}: {is_perfect_order}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in perfect order detection for {symbol} {timeframe}: {e}")
            # エラー時はデフォルト結果を返す
            return PerfectOrderResult(
                symbol=symbol,
                timeframe=timeframe,
                is_perfect_order=False,
                sma_short=0.0,
                sma_medium=0.0,
                sma_long=0.0,
                current_price=0.0,
                strength_score=0.0,
                trend_duration_days=0,
                detection_timestamp=datetime.now()
            )
    
    def check_multi_timeframe_perfect_order(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> MultiTimeframePerfectOrder:
        """
        複数時間軸でのパーフェクトオーダー検出・優先度判定
        
        Args:
            symbol: 銘柄コード
            data_dict: {"daily": df, "weekly": df, "monthly": df}
            
        Returns:
            MultiTimeframePerfectOrder: 複数時間軸分析結果
        """
        try:
            # キャッシュチェック
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self._cache:
                cache_time, result = self._cache[cache_key]
                if datetime.now() - cache_time < self._cache_timeout:
                    return result
            
            # 各時間軸で検出実行
            daily_result = self.detect_perfect_order(data_dict.get('daily', pd.DataFrame()), 'daily', symbol)
            weekly_result = self.detect_perfect_order(data_dict.get('weekly', pd.DataFrame()), 'weekly', symbol)
            monthly_result = self.detect_perfect_order(data_dict.get('monthly', pd.DataFrame()), 'monthly', symbol)
            
            # 優先度レベル判定
            priority_level = self._calculate_priority_level(
                daily_result.is_perfect_order,
                weekly_result.is_perfect_order,
                monthly_result.is_perfect_order
            )
            
            # 総合スコア計算
            composite_score = self._calculate_composite_score(
                daily_result, weekly_result, monthly_result, priority_level
            )
            
            result = MultiTimeframePerfectOrder(
                symbol=symbol,
                daily_result=daily_result,
                weekly_result=weekly_result,
                monthly_result=monthly_result,
                priority_level=priority_level,
                composite_score=composite_score,
                analysis_timestamp=datetime.now()
            )
            
            # キャッシュ保存
            self._cache[cache_key] = (datetime.now(), result)
            
            self.logger.info(f"Multi-timeframe analysis for {symbol}: Priority Level {priority_level}, Score {composite_score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            # エラー時はデフォルト結果を返す
            default_result = PerfectOrderResult(
                symbol=symbol, timeframe="default", is_perfect_order=False,
                sma_short=0.0, sma_medium=0.0, sma_long=0.0, current_price=0.0,
                strength_score=0.0, trend_duration_days=0, detection_timestamp=datetime.now()
            )
            return MultiTimeframePerfectOrder(
                symbol=symbol,
                daily_result=default_result,
                weekly_result=default_result,
                monthly_result=default_result,
                priority_level=3,
                composite_score=0.0,
                analysis_timestamp=datetime.now()
            )
    
    def batch_analyze_symbols(self, symbols: List[str], data_provider_func) -> List[MultiTimeframePerfectOrder]:
        """
        複数銘柄の一括分析
        
        Args:
            symbols: 銘柄リスト
            data_provider_func: データ取得関数
            
        Returns:
            List[MultiTimeframePerfectOrder]: 分析結果リスト
        """
        results = []
        
        for symbol in symbols:
            try:
                # データ取得
                data_dict = data_provider_func(symbol)
                
                # 分析実行
                result = self.check_multi_timeframe_perfect_order(symbol, data_dict)
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {symbol}: {e}")
                continue
        
        # 優先度・スコア順でソート
        results.sort(key=lambda x: (x.priority_level, -x.composite_score))
        
        self.logger.info(f"Batch analysis completed: {len(results)}/{len(symbols)} symbols processed")
        return results
    
    def _calculate_strength_score(self, current: float, short: float, medium: float, long: float) -> float:
        """強度スコア計算（価格間隔の均等性と乖離度）"""
        if pd.isna(current) or pd.isna(short) or pd.isna(medium) or pd.isna(long):
            return 0.0
        
        if current <= 0 or short <= 0 or medium <= 0 or long <= 0:
            return 0.0
        
        try:
            # 価格間隔の計算
            gap1 = (current - short) / short
            gap2 = (short - medium) / medium  
            gap3 = (medium - long) / long
            
            # 間隔の均等性（理想的には均等間隔）
            gaps = [gap1, gap2, gap3]
            avg_gap = np.mean(gaps)
            uniformity = 1.0 - (np.std(gaps) / (avg_gap + 1e-8))
            
            # 全体的な乖離度（大きいほど強いトレンド）
            total_divergence = (current - long) / long
            
            # 総合スコア（0-1）
            strength = min(1.0, (uniformity * 0.4 + min(total_divergence * 10, 1.0) * 0.6))
            return max(0.0, strength)
            
        except:
            return 0.0
    
    def _calculate_trend_duration(self, data: pd.DataFrame, periods: Dict[str, int]) -> int:
        """トレンド継続日数計算"""
        try:
            if data.empty or len(data) < periods['long']:
                return 0
            
            # 短期MA > 中期MA > 長期MAの継続日数を計算
            short_ma = data['Close'].rolling(window=periods['short']).mean()
            medium_ma = data['Close'].rolling(window=periods['medium']).mean()
            long_ma = data['Close'].rolling(window=periods['long']).mean()
            
            perfect_order_condition = (short_ma > medium_ma) & (medium_ma > long_ma)
            
            # 最新から遡って継続日数をカウント
            duration = 0
            for i in range(len(perfect_order_condition) - 1, -1, -1):
                if pd.notna(perfect_order_condition.iloc[i]) and perfect_order_condition.iloc[i]:
                    duration += 1
                else:
                    break
            
            return duration
            
        except:
            return 0
    
    def _calculate_priority_level(self, daily: bool, weekly: bool, monthly: bool) -> int:
        """優先度レベル計算"""
        if daily and weekly and monthly:
            return 1  # 全時間軸パーフェクトオーダー
        elif weekly and monthly:
            return 2  # 月足・週足パーフェクトオーダー
        else:
            return 3  # その他
    
    def _calculate_composite_score(self, daily: PerfectOrderResult, weekly: PerfectOrderResult, 
                                 monthly: PerfectOrderResult, priority_level: int) -> float:
        """総合スコア計算"""
        # 時間軸別重み（長期重視）
        weights = {
            "daily": 0.2,
            "weekly": 0.3, 
            "monthly": 0.5
        }
        
        # 基本スコア
        base_score = (
            daily.strength_score * weights["daily"] +
            weekly.strength_score * weights["weekly"] +
            monthly.strength_score * weights["monthly"]
        )
        
        # 優先度ボーナス
        priority_bonus = {1: 0.3, 2: 0.2, 3: 0.0}[priority_level]
        
        # トレンド継続ボーナス
        duration_bonus = min(0.1, (weekly.trend_duration_days / 30) * 0.1)
        
        return min(1.0, base_score + priority_bonus + duration_bonus)

    def get_cache_stats(self) -> Dict[str, int]:
        """キャッシュ統計取得"""
        return {
            "total_entries": len(self._cache),
            "active_entries": len([k for k, (t, _) in self._cache.items() 
                                 if datetime.now() - t < self._cache_timeout])
        }
    
    def clear_cache(self) -> None:
        """キャッシュクリア"""
        self._cache.clear()
        self.logger.info("Perfect order detection cache cleared")


if __name__ == "__main__":
    # テスト実行
    detector = PerfectOrderDetector()
    
    try:
        # テストデータ生成
        import yfinance as yf
        
        test_symbol = "7203"  # トヨタ
        ticker = yf.Ticker(test_symbol + ".T")
        
        print("=== DSSMS Perfect Order Detector Test ===")
        
        # データ取得
        hist = ticker.history(period="1y")
        if hist.empty:
            print("Failed to fetch test data")
        else:
            print(f"\nFetched {len(hist)} days of data for {test_symbol}")
            
            # 週足・月足生成
            weekly = hist.resample('W-MON').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            monthly = hist.resample('M').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            data_dict = {
                'daily': hist,
                'weekly': weekly,
                'monthly': monthly
            }
            
            # 複数時間軸分析
            result = detector.check_multi_timeframe_perfect_order(test_symbol, data_dict)
            
            print(f"\nMulti-timeframe Perfect Order Analysis for {test_symbol}:")
            print(f"  Priority Level: {result.priority_level}")
            print(f"  Composite Score: {result.composite_score:.3f}")
            print(f"  Daily Perfect Order: {result.daily_result.is_perfect_order}")
            print(f"  Weekly Perfect Order: {result.weekly_result.is_perfect_order}")
            print(f"  Monthly Perfect Order: {result.monthly_result.is_perfect_order}")
            
            print(f"\nDaily Analysis:")
            print(f"  Current Price: {result.daily_result.current_price:.2f}")
            print(f"  SMA(5): {result.daily_result.sma_short:.2f}")
            print(f"  SMA(25): {result.daily_result.sma_medium:.2f}")
            print(f"  SMA(75): {result.daily_result.sma_long:.2f}")
            print(f"  Strength Score: {result.daily_result.strength_score:.3f}")
            print(f"  Trend Duration: {result.daily_result.trend_duration_days} days")
            
            # キャッシュ統計
            cache_stats = detector.get_cache_stats()
            print(f"\nCache Statistics:")
            print(f"  Active entries: {cache_stats['active_entries']}/{cache_stats['total_entries']}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
