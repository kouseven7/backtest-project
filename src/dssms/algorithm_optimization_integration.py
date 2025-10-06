"""
Stage 3-2: OptimizedAlgorithmEngine統合実装
最適化アルゴリズムの統合によるfinal_selection処理高速化
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import concurrent.futures
import time

class OptimizedAlgorithmEngine:
    """最適化アルゴリズムエンジン - Stage 3-2統合"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.optimization_stats = {
            'numpy_operations': 0,
            'vectorized_calculations': 0,
            'early_terminations': 0,
            'processing_time_saved': 0.0
        }

    def optimized_final_selection(
        self, 
        symbols: List[str], 
        max_symbols: int, 
        market_data_fetcher,
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        最適化された最終選択アルゴリズム
        
        Args:
            symbols: 候補銘柄リスト
            max_symbols: 最大選択銘柄数
            market_data_fetcher: 市場データ取得インターフェース
            scoring_weights: スコアリング重み
            
        Returns:
            選択された銘柄リスト
        """
        start_time = time.time()
        
        try:
            # デフォルト重み設定
            weights = scoring_weights or {
                'market_cap': 0.4,
                'price_momentum': 0.3,
                'volume_score': 0.2,
                'volatility_penalty': 0.1
            }
            
            # 並列データ取得
            symbol_data = self._parallel_data_collection(symbols, market_data_fetcher)
            
            if not symbol_data:
                self.logger.warning("No valid symbol data collected")
                return symbols[:max_symbols]  # フォールバック
            
            # NumPy配列による高速スコア計算
            scores = self._vectorized_scoring(symbol_data, weights)
            
            # 高速ソート・選択
            top_indices = self._fast_top_selection(scores, max_symbols)
            
            # 結果構築
            selected_symbols = [symbol_data[i]['symbol'] for i in top_indices]
            
            processing_time = time.time() - start_time
            self.optimization_stats['processing_time_saved'] += processing_time
            
            self.logger.info(f"OptimizedAlgorithmEngine: {len(symbols)} → {len(selected_symbols)} symbols ({processing_time:.2f}s)")
            
            return selected_symbols
            
        except Exception as e:
            self.logger.error(f"OptimizedAlgorithmEngine failed: {e}")
            return symbols[:max_symbols]  # フォールバック
    
    def _parallel_data_collection(
        self, 
        symbols: List[str], 
        market_data_fetcher
    ) -> List[Dict[str, Any]]:
        """並列データ収集"""
        
        symbol_data = []
        
        def fetch_symbol_data(symbol: str) -> Optional[Dict[str, Any]]:
            try:
                # キャッシュ統合データ取得
                market_cap = market_data_fetcher.get_market_cap_data_cached(symbol)
                price = market_data_fetcher.get_price_data_cached(symbol)
                volume = market_data_fetcher.get_volume_data_cached(symbol)
                
                if any(data is None for data in [market_cap, price, volume]):
                    return None
                    
                return {
                    'symbol': symbol,
                    'market_cap': market_cap,
                    'price': price,
                    'volume': volume,
                    'price_momentum': self._calculate_momentum(symbol, market_data_fetcher),
                    'volatility': self._calculate_volatility(symbol, market_data_fetcher)
                }
                
            except Exception as e:
                self.logger.debug(f"Data collection failed for {symbol}: {e}")
                return None
        
        # 並列処理でデータ収集
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_symbol_data, symbol): symbol for symbol in symbols}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    symbol_data.append(result)
        
        return symbol_data
    
    def _vectorized_scoring(
        self, 
        symbol_data: List[Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> np.ndarray:
        """NumPy vectorized scoring計算"""
        
        self.optimization_stats['numpy_operations'] += 1
        
        n_symbols = len(symbol_data)
        
        # データをNumPy配列に変換
        market_caps = np.array([data['market_cap'] for data in symbol_data])
        prices = np.array([data['price'] for data in symbol_data])
        volumes = np.array([data['volume'] for data in symbol_data])
        momentums = np.array([data.get('price_momentum', 0.0) for data in symbol_data])
        volatilities = np.array([data.get('volatility', 0.0) for data in symbol_data])
        
        # 正規化（0-1スケール）
        market_cap_scores = self._normalize_array(market_caps)
        price_momentum_scores = self._normalize_array(momentums)
        volume_scores = self._normalize_array(volumes)
        volatility_penalties = self._normalize_array(volatilities, reverse=True)  # 低ボラティリティが高スコア
        
        # 重み付きスコア計算
        scores = (
            market_cap_scores * weights['market_cap'] +
            price_momentum_scores * weights['price_momentum'] +
            volume_scores * weights['volume_score'] -
            volatility_penalties * weights['volatility_penalty']
        )
        
        self.optimization_stats['vectorized_calculations'] += n_symbols
        
        return scores
    
    def _normalize_array(self, arr: np.ndarray, reverse: bool = False) -> np.ndarray:
        """配列正規化"""
        if len(arr) == 0:
            return arr
            
        min_val, max_val = np.min(arr), np.max(arr)
        
        if max_val == min_val:
            return np.ones_like(arr) * 0.5
            
        normalized = (arr - min_val) / (max_val - min_val)
        
        return 1.0 - normalized if reverse else normalized
    
    def _fast_top_selection(self, scores: np.ndarray, max_symbols: int) -> np.ndarray:
        """高速トップ選択（partial sort使用）"""
        
        if len(scores) <= max_symbols:
            return np.arange(len(scores))
            
        # argpartitionによる部分ソート（O(n)計算量）
        top_indices = np.argpartition(scores, -max_symbols)[-max_symbols:]
        
        # トップ内でソート
        top_scores = scores[top_indices]
        sorted_indices = np.argsort(top_scores)[::-1]
        
        return top_indices[sorted_indices]
    
    def _calculate_momentum(self, symbol: str, market_data_fetcher) -> float:
        """価格モメンタム計算（簡易版）"""
        try:
            # TODO: 履歴データベースの価格変化率計算
            # 現在は価格データからの簡易推定
            price = market_data_fetcher.get_price_data_cached(symbol)
            return float(price) if price else 0.0
        except:
            return 0.0
    
    def _calculate_volatility(self, symbol: str, market_data_fetcher) -> float:
        """価格ボラティリティ計算（簡易版）"""
        try:
            # TODO: 履歴データベースの標準偏差計算
            # 現在は簡易推定
            return 0.5  # プレースホルダー
        except:
            return 0.5

    def get_optimization_stats(self) -> Dict[str, Any]:
        """最適化統計情報取得"""
        return self.optimization_stats.copy()

    def optimized_affordability_filter(
        self, 
        symbols: List[str], 
        available_funds: float,
        min_shares: int,
        market_data_fetcher
    ) -> List[str]:
        """
        最適化されたaffordability filter
        
        Args:
            symbols: 候補銘柄リスト
            available_funds: 利用可能資金
            min_shares: 最小購入株数
            market_data_fetcher: 市場データ取得インターフェース
            
        Returns:
            フィルタ後銘柄リスト
        """
        start_time = time.time()
        
        try:
            # 並列価格取得
            symbol_prices = {}
            
            def fetch_price(symbol: str) -> Tuple[str, Optional[float]]:
                price = market_data_fetcher.get_price_data_cached(symbol)
                return symbol, price
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(fetch_price, symbol) for symbol in symbols]
                
                for future in concurrent.futures.as_completed(futures):
                    symbol, price = future.result()
                    if price is not None:
                        symbol_prices[symbol] = price
            
            # NumPy vectorized計算
            valid_symbols = list(symbol_prices.keys())
            prices = np.array([symbol_prices[symbol] for symbol in valid_symbols])
            required_funds = prices * min_shares
            
            # 条件満たす銘柄のマスク
            affordable_mask = required_funds <= available_funds
            affordable_symbols = [valid_symbols[i] for i in np.where(affordable_mask)[0]]
            
            processing_time = time.time() - start_time
            self.optimization_stats['processing_time_saved'] += processing_time
            
            self.logger.info(f"Optimized affordability filter: {len(symbols)} → {len(affordable_symbols)} symbols ({processing_time:.2f}s)")
            
            return affordable_symbols
            
        except Exception as e:
            self.logger.error(f"Optimized affordability filter failed: {e}")
            # フォールバック（従来処理）
            return symbols


def create_algorithm_optimization_integration():
    """OptimizedAlgorithmEngine統合ファクトリー"""
    return OptimizedAlgorithmEngine()