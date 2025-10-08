"""
Module: DSSMS Data Integration Enhancer
File: dssms_data_integration_enhancer.py
Description: 
  Task 1.2: DSSMS専用データ統合強化システム
  Task 1.1の統合パッチを活用し、DSSMS専用の最適化を実装

Author: GitHub Copilot
Created: 2025-08-25
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.dssms.dssms_integration_patch import (
        update_symbol_ranking_with_real_data,
        update_portfolio_value_with_real_data,
        fetch_real_data,
        generate_realistic_sample_data
    )
    from config.logger_config import setup_logger
except ImportError as e:
    print(f"Import warning: {e}")
    
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class DSSMSDataIntegrationEnhancer:
    """DSSMS専用データ統合強化システム"""
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.quality_metrics: Dict[str, float] = {}
        self.enhancement_stats = {
            'real_data_success': 0,
            'cache_hits': 0,
            'fallback_used': 0,
            'quality_improvements': 0
        }
        
        self.logger.info("DSSMS データ統合強化システムを初期化しました")
    
    def enhance_ranking_with_real_data(self, symbols: List[str], date: datetime) -> Dict[str, Any]:
        """実データベースランキングの品質向上"""
        try:
            self.logger.debug(f"強化ランキング開始: {len(symbols)}銘柄")
            
            # 基本ランキング取得（Task 1.1統合パッチ使用）
            base_ranking = update_symbol_ranking_with_real_data(symbols, date)
            
            # 品質向上処理
            enhanced_ranking = self._apply_ranking_enhancements(base_ranking, symbols, date)
            
            # データソース品質評価
            quality_score = self._evaluate_ranking_quality(enhanced_ranking, symbols)
            
            result = {
                'rankings': enhanced_ranking,
                'quality_score': quality_score,
                'data_source': 'enhanced_real_data',
                'enhancement_applied': True,
                'timestamp': date.isoformat(),
                'symbols_count': len(symbols)
            }
            
            self.enhancement_stats['real_data_success'] += 1
            self.logger.info(f"強化ランキング完了: 品質スコア {quality_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"強化ランキングエラー: {e}")
            # フォールバックで基本ランキングを返す
            try:
                base_ranking = update_symbol_ranking_with_real_data(symbols, date)
                return {
                    'rankings': base_ranking,
                    'quality_score': 0.5,
                    'data_source': 'fallback_basic',
                    'enhancement_applied': False,
                    'error': str(e)
                }
            except:
                return {'rankings': {}, 'quality_score': 0.0, 'error': str(e)}
    
    def _apply_ranking_enhancements(self, base_ranking: Dict[str, float], 
                                  symbols: List[str], date: datetime) -> Dict[str, float]:
        """ランキング強化処理"""
        enhanced = base_ranking.copy()
        
        try:
            # 1. ボラティリティ調整
            enhanced = self._apply_volatility_adjustment(enhanced, symbols)
            
            # 2. トレンド強度による補正
            enhanced = self._apply_trend_strength_correction(enhanced, symbols, date)
            
            # 3. 相関による分散化補正
            enhanced = self._apply_diversification_bonus(enhanced, symbols)
            
            self.enhancement_stats['quality_improvements'] += 1
            
        except Exception as e:
            self.logger.warning(f"ランキング強化処理エラー: {e}")
            
        return enhanced
    
    def _apply_volatility_adjustment(self, ranking: Dict[str, float], 
                                   symbols: List[str]) -> Dict[str, float]:
        """ボラティリティ調整"""
        adjusted = ranking.copy()
        
        for symbol in symbols:
            if symbol in adjusted:
                # 実データから直近ボラティリティ取得
                recent_data = fetch_real_data(symbol, days=20)
                if recent_data is not None and len(recent_data) > 5:
                    returns = recent_data['Close'].pct_change().dropna()
                    volatility = returns.std()
                    
                    # ボラティリティが高すぎる場合は減点
                    if volatility > 0.03:  # 3%以上
                        penalty = min(0.2, (volatility - 0.03) * 5)
                        adjusted[symbol] = max(0.0, adjusted[symbol] - penalty)
                        
        return adjusted
    
    def _apply_trend_strength_correction(self, ranking: Dict[str, float], 
                                       symbols: List[str], date: datetime) -> Dict[str, float]:
        """トレンド強度による補正"""
        corrected = ranking.copy()
        
        for symbol in symbols:
            if symbol in corrected:
                trend_strength = self._calculate_trend_strength(symbol)
                
                # 強いトレンドにボーナス
                if trend_strength > 0.7:
                    bonus = (trend_strength - 0.7) * 0.1
                    corrected[symbol] = min(1.0, corrected[symbol] + bonus)
                    
        return corrected
    
    def _calculate_trend_strength(self, symbol: str) -> float:
        """トレンド強度計算"""
        try:
            data = fetch_real_data(symbol, days=30)
            if data is None or len(data) < 20:
                return 0.5
                
            close = data['Close']
            
            # 移動平均との位置関係
            ma5 = close.rolling(5).mean()
            ma20 = close.rolling(20).mean()
            
            # 現在価格と移動平均の関係
            current_price = close.iloc[-1]
            ma5_current = ma5.iloc[-1]
            ma20_current = ma20.iloc[-1]
            
            # トレンド強度計算
            if current_price > ma5_current > ma20_current:
                return 0.8  # 強い上昇トレンド
            elif current_price < ma5_current < ma20_current:
                return 0.3  # 強い下降トレンド
            else:
                return 0.5  # 横ばいまたは混合
                
        except Exception:
            return 0.5
    
    def _apply_diversification_bonus(self, ranking: Dict[str, float], 
                                   symbols: List[str]) -> Dict[str, float]:
        """分散化ボーナス適用"""
        # 業種の多様性を考慮した簡易的なボーナス
        # 実際の実装では業種データが必要
        return ranking
    
    def _evaluate_ranking_quality(self, ranking: Dict[str, float], symbols: List[str]) -> float:
        """ランキング品質評価"""
        try:
            if not ranking:
                return 0.0
                
            scores = list(ranking.values())
            
            # 品質指標計算
            score_range = max(scores) - min(scores) if scores else 0
            score_variance = np.var(scores) if scores else 0
            data_completeness = len(ranking) / len(symbols) if symbols else 0
            
            # 総合品質スコア
            quality = (
                min(1.0, score_range * 2) * 0.3 +  # スコア分散
                min(1.0, score_variance * 10) * 0.3 +  # 分散度
                data_completeness * 0.4  # データ完全性
            )
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.5
    
    def enhance_portfolio_valuation(self, position: Optional[str], 
                                  current_value: float, date: datetime) -> Dict[str, Any]:
        """実データベースポートフォリオ評価の精度向上"""
        try:
            if not position:
                return {
                    'new_value': current_value,
                    'daily_return': 0.0,
                    'data_source': 'no_position',
                    'quality_score': 1.0
                }
            
            # 基本価値更新（Task 1.1統合パッチ使用）
            base_value = update_portfolio_value_with_real_data(position, current_value, date)
            
            # 精度向上処理
            enhanced_value = self._apply_valuation_enhancements(
                position, current_value, base_value, date
            )
            
            # 品質評価
            quality_score = self._evaluate_valuation_quality(position, enhanced_value, current_value)
            
            daily_return = (enhanced_value / current_value) - 1 if current_value > 0 else 0.0
            
            result = {
                'new_value': enhanced_value,
                'daily_return': daily_return,
                'data_source': 'enhanced_real_data',
                'quality_score': quality_score,
                'base_value': base_value,
                'enhancement_applied': True,
                'timestamp': date.isoformat()
            }
            
            self.logger.debug(f"強化評価完了: {position} {daily_return:+.4f} (品質: {quality_score:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"強化評価エラー: {e}")
            return {
                'new_value': current_value,
                'daily_return': 0.0,
                'data_source': 'error_fallback',
                'quality_score': 0.0,
                'error': str(e)
            }
    
    def _apply_valuation_enhancements(self, position: str, current_value: float, 
                                    base_value: float, date: datetime) -> float:
        """評価額強化処理"""
        try:
            # 市場時間考慮
            market_adjustment = self._get_market_time_adjustment(date)
            
            # 流動性調整
            liquidity_adjustment = self._get_liquidity_adjustment(position)
            
            # 調整適用
            enhanced_value = base_value * market_adjustment * liquidity_adjustment
            
            # 異常値チェック
            daily_change = abs((enhanced_value / current_value) - 1) if current_value > 0 else 0
            if daily_change > 0.2:  # 20%以上の変動は制限
                self.logger.warning(f"異常な日次変動検出: {daily_change:.2%} -> 制限適用")
                direction = 1 if enhanced_value > current_value else -1
                enhanced_value = current_value * (1 + direction * 0.1)  # 10%に制限
            
            return max(0.01, enhanced_value)  # 最小値保証
            
        except Exception as e:
            self.logger.warning(f"評価額強化処理エラー: {e}")
            return base_value
    
    def _get_market_time_adjustment(self, date: datetime) -> float:
        """市場時間調整係数"""
        # 簡易的な市場時間考慮
        hour = date.hour
        if 9 <= hour <= 15:  # 市場時間内
            return 1.0
        elif 16 <= hour <= 23:  # アフターマーケット
            return 0.95
        else:  # 夜間
            return 0.9
    
    def _get_liquidity_adjustment(self, symbol: str) -> float:
        """流動性調整係数"""
        # 日本の主要銘柄の簡易判定
        major_symbols = ['7203.T', '6758.T', '9984.T', '8058.T', '4519.T']
        if symbol in major_symbols:
            return 1.0
        else:
            return 0.98  # 若干のディスカウント
    
    def _evaluate_valuation_quality(self, position: str, new_value: float, 
                                   current_value: float) -> float:
        """評価品質スコア"""
        try:
            # 変動の妥当性チェック
            daily_change = abs((new_value / current_value) - 1) if current_value > 0 else 0
            
            if daily_change < 0.001:  # 0.1%未満
                return 0.6  # 低変動
            elif daily_change < 0.05:  # 5%未満
                return 0.9  # 正常範囲
            elif daily_change < 0.15:  # 15%未満
                return 0.7  # やや高変動
            else:
                return 0.4  # 高変動
                
        except Exception:
            return 0.5
    
    def validate_data_quality(self, data_source: str, data: Any) -> Dict[str, Any]:
        """データ品質検証・補正"""
        try:
            result = {
                'source': data_source,
                'is_valid': True,
                'quality_score': 1.0,
                'issues': [],
                'corrections_applied': []
            }
            
            # データ型チェック
            if data is None:
                result['is_valid'] = False
                result['issues'].append('null_data')
                result['quality_score'] = 0.0
                return result
            
            # DataFrameの場合の検証
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    result['issues'].append('empty_dataframe')
                    result['quality_score'] *= 0.5
                
                if len(data) < 5:
                    result['issues'].append('insufficient_data')
                    result['quality_score'] *= 0.7
                
                # 欠損値チェック
                missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                if missing_ratio > 0.1:
                    result['issues'].append('high_missing_values')
                    result['quality_score'] *= (1 - missing_ratio)
            
            # 辞書データの場合の検証
            elif isinstance(data, dict):
                if not data:
                    result['issues'].append('empty_dict')
                    result['quality_score'] = 0.0
                
                # 数値データの妥当性
                for key, value in data.items():
                    if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                        result['issues'].append(f'invalid_numeric_value_{key}')
                        result['quality_score'] *= 0.9
            
            self.logger.debug(f"品質検証完了: {data_source} スコア={result['quality_score']:.3f}")
            return result
            
        except Exception as e:
            return {
                'source': data_source,
                'is_valid': False,
                'quality_score': 0.0,
                'issues': ['validation_error'],
                'error': str(e)
            }
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """強化統計情報取得"""
        total_operations = sum(self.enhancement_stats.values())
        
        return {
            'total_operations': total_operations,
            'real_data_success_rate': (
                self.enhancement_stats['real_data_success'] / max(1, total_operations)
            ),
            'cache_hit_rate': (
                self.enhancement_stats['cache_hits'] / max(1, total_operations)
            ),
            'fallback_rate': (
                self.enhancement_stats['fallback_used'] / max(1, total_operations)
            ),
            'quality_improvement_rate': (
                self.enhancement_stats['quality_improvements'] / max(1, total_operations)
            ),
            'average_quality_score': np.mean(list(self.quality_metrics.values())) if self.quality_metrics else 0.0
        }

def demo_data_integration_enhancer():
    """データ統合強化システムデモ"""
    print("=== DSSMS データ統合強化システムデモ ===")
    
    try:
        # システム初期化
        enhancer = DSSMSDataIntegrationEnhancer()
        
        # テスト銘柄
        test_symbols = ["7203.T", "8058.T", "9984.T"]
        test_date = datetime.now()
        
        # 強化ランキングテスト
        print(f"\n[CHART] 強化ランキングテスト: {test_symbols}")
        ranking_result = enhancer.enhance_ranking_with_real_data(test_symbols, test_date)
        
        print(f"   品質スコア: {ranking_result['quality_score']:.3f}")
        print(f"   データソース: {ranking_result['data_source']}")
        
        for symbol, score in ranking_result['rankings'].items():
            print(f"   {symbol}: {score:+.4f}")
        
        # 強化ポートフォリオ評価テスト
        print(f"\n[MONEY] 強化ポートフォリオ評価テスト")
        initial_value = 1000000
        
        for symbol in test_symbols[:2]:
            valuation_result = enhancer.enhance_portfolio_valuation(symbol, initial_value, test_date)
            
            print(f"   {symbol}:")
            print(f"     価値: {initial_value:,.0f} -> {valuation_result['new_value']:,.0f}")
            print(f"     リターン: {valuation_result['daily_return']:+.4f}")
            print(f"     品質: {valuation_result['quality_score']:.3f}")
        
        # 統計情報表示
        print(f"\n[UP] 強化統計情報")
        stats = enhancer.get_enhancement_statistics()
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] デモエラー: {e}")
        return False

if __name__ == "__main__":
    success = demo_data_integration_enhancer()
    if success:
        print("\n[OK] データ統合強化システムデモ完了")
    else:
        print("\n[ERROR] データ統合強化システムデモ失敗")
