"""
Module: Market Analyzer
File: market_analyzer.py
Description:
  Phase 2: 動的戦略選択復活 - トレンド・相場判断システム統合
  複数の市場分析コンポーネントを統合し、包括的な市場状況判定を提供

Components:
  - TrendStrategyIntegrationInterface: トレンド戦略統合インターフェース
  - UnifiedTrendDetector: 統合トレンド検出器
  - FixedPerfectOrderDetector: Perfect Order検出器

Author: imega
Created: 2025-10-16
Modified: 2025-10-16
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# 移動済みモジュールのインポート（main_system内）
try:
    from main_system.market_analysis.trend_strategy_integration_interface import TrendStrategyIntegrationInterface
except ImportError:
    # フォールバック: 元の場所からインポート
    from config.trend_strategy_integration_interface import TrendStrategyIntegrationInterface

try:
    from main_system.market_analysis.perfect_order_detector import FixedPerfectOrderDetector
except ImportError:
    # フォールバック: ルートからインポート
    sys.path.append(str(project_root))
    from fixed_perfect_order_detector import FixedPerfectOrderDetector

# UnifiedTrendDetectorは既存のindicators/から使用（共有システム）
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend_with_confidence

logger = setup_logger(__name__)


class MarketRegime(Enum):
    """市場レジーム分類"""
    STRONG_UPTREND = "strong_uptrend"          # 強い上昇トレンド
    UPTREND = "uptrend"                        # 上昇トレンド
    WEAK_UPTREND = "weak_uptrend"              # 弱い上昇トレンド
    SIDEWAYS = "sideways"                      # 横ばい・レンジ
    WEAK_DOWNTREND = "weak_downtrend"          # 弱い下降トレンド
    DOWNTREND = "downtrend"                    # 下降トレンド
    STRONG_DOWNTREND = "strong_downtrend"      # 強い下降トレンド
    VOLATILE = "volatile"                      # 高ボラティリティ
    UNKNOWN = "unknown"                        # 判定不能


class MarketAnalyzer:
    """
    市場分析統合クラス
    
    複数の分析コンポーネントを統合して包括的な市場分析を実行:
    - トレンド判定（複数手法の統合）
    - Perfect Order検出
    - 市場レジーム判定
    - 信頼度スコア計算
    """
    
    def __init__(self):
        """
        MarketAnalyzer初期化
        
        注意事項（copilot-instructions.md準拠）:
        - 実際のbacktest実行を妨げないこと
        - シグナル生成に影響を与えないこと
        - エラー時はフォールバックして継続
        """
        self.logger = setup_logger(__name__)
        
        # コンポーネント初期化（エラーハンドリング付き）
        try:
            self.trend_interface = TrendStrategyIntegrationInterface()
            self.logger.info("TrendStrategyIntegrationInterface initialized")
        except Exception as e:
            self.logger.warning(f"TrendStrategyIntegrationInterface init failed: {e}")
            self.trend_interface = None
        
        # UnifiedTrendDetectorはデータが必要なため遅延初期化
        # comprehensive_market_analysis内で実際のデータと共に使用
        self.trend_detector = None
        self.logger.info("UnifiedTrendDetector will be used with data")
        
        try:
            self.perfect_order = FixedPerfectOrderDetector()
            self.logger.info("FixedPerfectOrderDetector initialized")
        except Exception as e:
            self.logger.warning(f"FixedPerfectOrderDetector init failed: {e}")
            self.perfect_order = None
        
        self.logger.info("MarketAnalyzer initialized successfully")
    
    def comprehensive_market_analysis(
        self,
        stock_data: pd.DataFrame,
        index_data: pd.DataFrame = None,
        ticker: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        包括的市場分析実行
        
        Args:
            stock_data: 株価データ（必須）
            index_data: 市場インデックスデータ（オプション）
            ticker: 銘柄コード
            
        Returns:
            Dict[str, Any]: 市場分析結果
                - trend_analysis: トレンド分析結果
                - unified_trend: 統合トレンド判定
                - perfect_order: Perfect Order状態
                - market_regime: 市場レジーム
                - confidence_score: 信頼度スコア
                - analysis_timestamp: 分析実行時刻
        """
        self.logger.info(f"Starting comprehensive market analysis for {ticker}")
        
        results = {
            'ticker': ticker,
            'analysis_timestamp': pd.Timestamp.now(),
            'trend_analysis': None,
            'unified_trend': None,
            'perfect_order': None,
            'market_regime': MarketRegime.UNKNOWN.value,
            'confidence_score': 0.0,
            'components_status': {}
        }
        
        try:
            # 1. トレンド戦略統合インターフェース実行
            if self.trend_interface is not None:
                try:
                    trend_result = self.trend_interface.integrate_decision(
                        stock_data, ticker
                    )
                    results['trend_analysis'] = trend_result
                    results['components_status']['trend_interface'] = 'success'
                    # IntegratedDecisionResultはdataclass - 属性アクセスを使用
                    trend_type = getattr(trend_result.trend_analysis, 'trend_type', 'N/A') if hasattr(trend_result, 'trend_analysis') else 'N/A'
                    self.logger.info(f"Trend analysis completed: {trend_type}")
                except Exception as e:
                    self.logger.warning(f"Trend interface analysis failed: {e}")
                    results['components_status']['trend_interface'] = f'failed: {str(e)}'
            
            # 2. UnifiedTrendDetector実行（関数ベース）
            try:
                unified_trend_result = detect_unified_trend_with_confidence(stock_data)
                # 結果がタプルの場合は辞書に変換
                if isinstance(unified_trend_result, tuple):
                    unified_trend = {
                        'trend': unified_trend_result[0],
                        'confidence': unified_trend_result[1]
                    }
                else:
                    unified_trend = unified_trend_result
                
                results['unified_trend'] = unified_trend
                results['components_status']['unified_trend'] = 'success'
                self.logger.info(f"Unified trend: {unified_trend.get('trend', 'N/A') if isinstance(unified_trend, dict) else unified_trend_result[0]}")
            except Exception as e:
                self.logger.warning(f"Unified trend detection failed: {e}")
                results['components_status']['unified_trend'] = f'failed: {str(e)}'
            
            # 3. Perfect Order検出実行
            if self.perfect_order is not None:
                try:
                    perfect_order_state = self.perfect_order.detect_perfect_order(stock_data)
                    results['perfect_order'] = perfect_order_state
                    results['components_status']['perfect_order'] = 'success'
                    self.logger.info(f"Perfect order detected: {perfect_order_state.get('is_perfect_order', False)}")
                except Exception as e:
                    self.logger.warning(f"Perfect order detection failed: {e}")
                    results['components_status']['perfect_order'] = f'failed: {str(e)}'
            
            # 4. 市場レジーム判定
            market_regime = self._determine_market_regime(
                results['trend_analysis'],
                results['unified_trend'],
                results['perfect_order']
            )
            results['market_regime'] = market_regime.value
            
            # 5. 信頼度スコア計算
            confidence_score = self._calculate_confidence_score(results)
            results['confidence_score'] = confidence_score
            
            self.logger.info(f"Market analysis completed - Regime: {market_regime.value}, Confidence: {confidence_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Comprehensive market analysis error: {e}")
            results['error'] = str(e)
        
        return results
    
    def _determine_market_regime(
        self,
        trend_analysis: Optional[Dict],
        unified_trend: Optional[Dict],
        perfect_order: Optional[Dict]
    ) -> MarketRegime:
        """
        市場レジーム判定
        
        複数の分析結果から総合的に市場状況を判定
        
        Args:
            trend_analysis: トレンド分析結果
            unified_trend: 統合トレンド結果
            perfect_order: Perfect Order結果
            
        Returns:
            MarketRegime: 判定された市場レジーム
        """
        try:
            # スコアリングシステムで判定
            uptrend_score = 0
            downtrend_score = 0
            sideways_score = 0
            
            # 1. TrendAnalysisからのスコア
            if trend_analysis:
                # IntegratedDecisionResultの場合はtrend_analysis.trend_typeを参照
                if hasattr(trend_analysis, 'trend_analysis'):
                    # IntegratedDecisionResult型
                    trend_type = getattr(trend_analysis.trend_analysis, 'trend_type', '').lower()
                elif isinstance(trend_analysis, dict):
                    # 辞書型（フォールバック用）
                    trend_type = trend_analysis.get('final_decision', '').lower()
                else:
                    trend_type = ''
                
                if 'uptrend' in trend_type or 'bullish' in trend_type or 'up' in trend_type:
                    uptrend_score += 2
                elif 'downtrend' in trend_type or 'bearish' in trend_type or 'down' in trend_type:
                    downtrend_score += 2
                else:
                    sideways_score += 2
            
            # 2. UnifiedTrendからのスコア
            if unified_trend:
                trend = unified_trend.get('trend', '').lower()
                confidence = unified_trend.get('confidence', 0.0)
                
                if 'uptrend' in trend:
                    uptrend_score += int(confidence * 3)
                elif 'downtrend' in trend:
                    downtrend_score += int(confidence * 3)
                else:
                    sideways_score += int(confidence * 3)
            
            # 3. Perfect Orderからのスコア
            if perfect_order:
                if perfect_order.get('is_perfect_order', False):
                    uptrend_score += 3  # Perfect Orderは強い上昇シグナル
                elif perfect_order.get('is_quasi_perfect_order', False):
                    uptrend_score += 1
            
            # スコア合計で判定
            max_score = max(uptrend_score, downtrend_score, sideways_score)
            
            if max_score == 0:
                return MarketRegime.UNKNOWN
            
            # 上昇トレンド判定
            if uptrend_score == max_score:
                if uptrend_score >= 6:
                    return MarketRegime.STRONG_UPTREND
                elif uptrend_score >= 4:
                    return MarketRegime.UPTREND
                else:
                    return MarketRegime.WEAK_UPTREND
            
            # 下降トレンド判定
            elif downtrend_score == max_score:
                if downtrend_score >= 6:
                    return MarketRegime.STRONG_DOWNTREND
                elif downtrend_score >= 4:
                    return MarketRegime.DOWNTREND
                else:
                    return MarketRegime.WEAK_DOWNTREND
            
            # 横ばい判定
            else:
                return MarketRegime.SIDEWAYS
        
        except Exception as e:
            self.logger.error(f"Market regime determination error: {e}")
            return MarketRegime.UNKNOWN
    
    def _calculate_confidence_score(self, analysis_results: Dict[str, Any]) -> float:
        """
        分析信頼度スコア計算
        
        Args:
            analysis_results: 分析結果辞書
            
        Returns:
            float: 信頼度スコア (0.0 - 1.0)
        """
        try:
            successful_components = sum(
                1 for status in analysis_results['components_status'].values()
                if status == 'success'
            )
            total_components = len(analysis_results['components_status'])
            
            if total_components == 0:
                return 0.0
            
            # 基本信頼度 = 成功コンポーネント率
            base_confidence = successful_components / total_components
            
            # ボーナス: 全コンポーネント成功
            if successful_components == total_components:
                base_confidence = min(1.0, base_confidence * 1.1)
            
            # ペナルティ: 重要コンポーネント失敗
            if analysis_results['components_status'].get('unified_trend') != 'success':
                base_confidence *= 0.8
            
            return round(base_confidence, 2)
        
        except Exception as e:
            self.logger.error(f"Confidence score calculation error: {e}")
            return 0.0
    
    def get_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        分析結果サマリー生成
        
        Args:
            analysis_results: comprehensive_market_analysis()の戻り値
            
        Returns:
            str: サマリー文字列
        """
        try:
            summary_lines = [
                f"=== Market Analysis Summary ===",
                f"Ticker: {analysis_results.get('ticker', 'N/A')}",
                f"Timestamp: {analysis_results.get('analysis_timestamp', 'N/A')}",
                f"Market Regime: {analysis_results.get('market_regime', 'N/A')}",
                f"Confidence Score: {analysis_results.get('confidence_score', 0.0):.2f}",
                f"\nComponents Status:"
            ]
            
            for component, status in analysis_results.get('components_status', {}).items():
                summary_lines.append(f"  - {component}: {status}")
            
            return "\n".join(summary_lines)
        
        except Exception as e:
            return f"Summary generation failed: {e}"


# 便利関数：簡易的な市場分析実行
def analyze_market(stock_data: pd.DataFrame, ticker: str = "UNKNOWN") -> Dict[str, Any]:
    """
    便利関数: 簡易的な市場分析実行
    
    Args:
        stock_data: 株価データ
        ticker: 銘柄コード
        
    Returns:
        Dict[str, Any]: 市場分析結果
    """
    analyzer = MarketAnalyzer()
    return analyzer.comprehensive_market_analysis(stock_data, ticker=ticker)


if __name__ == "__main__":
    # テスト実行
    print("MarketAnalyzer Test")
    print("=" * 50)
    
    try:
        analyzer = MarketAnalyzer()
        print(f"✓ MarketAnalyzer initialized successfully")
        print(f"  - TrendInterface: {'OK' if analyzer.trend_interface else 'NG'}")
        print(f"  - TrendDetector: {'OK' if analyzer.trend_detector else 'NG'}")
        print(f"  - PerfectOrder: {'OK' if analyzer.perfect_order else 'NG'}")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
