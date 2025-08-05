"""
トレンド信頼度比較ユーティリティ
段階1実装: 基本比較インターフェース（0-1範囲）

Module: Trend Reliability Utilities
Description: 
  統一トレンド判定器を活用した信頼度比較機能
  2-2-1「信頼度スコアとパフォーマンススコアの統合ロジック」の基盤

Author: imega
Created: 2025-07-13
Modified: 2025-07-13
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import numpy as np

# プロジェクト内インポート
try:
    from .unified_trend_detector import UnifiedTrendDetector
except ImportError:
    # 直接実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from indicators.unified_trend_detector import UnifiedTrendDetector

# ロガー設定
logger = logging.getLogger(__name__)

def get_trend_reliability(data: pd.DataFrame, 
                         strategy: str = "default",
                         method: str = "auto",
                         format: str = "decimal",
                         lookback: int = 0) -> Union[float, Dict[str, Any]]:
    """
    統一的な信頼度取得インターフェース
    
    Args:
        data: 価格データ
        strategy: 戦略名
        method: トレンド判定手法
        format: 出力形式 ("decimal", "percentage", "detailed")
        lookback: 遡り期間
        
    Returns:
        Union[float, Dict]: 信頼度スコアまたは詳細情報
    """
    try:
        detector = UnifiedTrendDetector(data, strategy_name=strategy, method=method)
        
        if format == "detailed":
            return detector.get_confidence_score_detailed(lookback)
        elif format == "percentage":
            confidence = detector.get_confidence_score(lookback)
            return confidence * 100
        else:  # decimal
            return detector.get_confidence_score(lookback)
            
    except Exception as e:
        logger.error(f"Error getting trend reliability: {e}")
        if format == "detailed":
            return {
                "error": str(e),
                "confidence_score": 0.0,
                "is_reliable": False
            }
        else:
            return 0.0

def get_trend_reliability_for_strategy(data: pd.DataFrame, 
                                     strategy_name: str,
                                     method: str = "auto",
                                     format: str = "decimal") -> Dict[str, Any]:
    """
    戦略特化型の信頼度取得
    
    Args:
        data: 価格データ
        strategy_name: 戦略名
        method: トレンド判定手法
        format: 出力形式
        
    Returns:
        Dict[str, Any]: 戦略特化信頼度情報
    """
    try:
        detector = UnifiedTrendDetector(data, strategy_name=strategy_name, method=method)
        
        # 基本情報取得
        detailed_info = detector.get_confidence_score_detailed()
        confidence_score = detailed_info["confidence_score"]
        
        # 戦略特化情報を追加
        result = {
            "strategy_name": strategy_name,
            "method": method,
            "trend": detailed_info["trend"],
            "confidence_score": confidence_score,
            "confidence_level": detailed_info["confidence_level"],
            "is_reliable": detailed_info["is_reliable"],
            "strategy_compatible": detector.is_trend_reliable_for_strategy(strategy_name),
            "timestamp": detailed_info["timestamp"]
        }
        
        # フォーマット別追加情報
        if format == "percentage":
            result["confidence_percentage"] = confidence_score * 100
            
        return result
        
    except Exception as e:
        logger.error(f"Error getting strategy reliability for {strategy_name}: {e}")
        return {
            "strategy_name": strategy_name,
            "method": method,
            "trend": "unknown",
            "confidence_score": 0.0,
            "confidence_level": "unreliable",
            "is_reliable": False,
            "strategy_compatible": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def compare_strategy_reliabilities(data: pd.DataFrame, 
                                 strategies: List[str],
                                 method: str = "auto") -> pd.DataFrame:
    """
    複数戦略での信頼度比較
    
    Args:
        data: 価格データ
        strategies: 戦略名リスト
        method: トレンド判定手法
        
    Returns:
        pd.DataFrame: 比較結果テーブル
    """
    results = []
    
    for strategy in strategies:
        try:
            reliability = get_trend_reliability_for_strategy(data, strategy, method)
            results.append(reliability)
            
        except Exception as e:
            logger.error(f"Error processing strategy {strategy}: {e}")
            # エラー時も結果に含める
            results.append({
                "strategy_name": strategy,
                "method": method,
                "trend": "error",
                "confidence_score": 0.0,
                "confidence_level": "unreliable",
                "is_reliable": False,
                "strategy_compatible": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    df = pd.DataFrame(results)
    
    # ソート（信頼度スコア降順）
    if not df.empty:
        df = df.sort_values('confidence_score', ascending=False).reset_index(drop=True)
    
    return df

def compare_method_reliabilities(data: pd.DataFrame,
                               methods: List[str],
                               strategy: str = "default") -> pd.DataFrame:
    """
    複数手法での信頼度比較
    
    Args:
        data: 価格データ
        methods: 手法名リスト
        strategy: 戦略名
        
    Returns:
        pd.DataFrame: 手法別比較結果
    """
    try:
        detector = UnifiedTrendDetector(data, strategy_name=strategy)
        reliability_scores = detector.compare_trend_reliabilities(methods)
        
        results = []
        for method, score in reliability_scores.items():
            # 各手法の詳細情報を取得
            try:
                detector.method = method
                detailed = detector.get_confidence_score_detailed()
                
                results.append({
                    "method": method,
                    "strategy": strategy,
                    "confidence_score": score,
                    "confidence_level": detailed["confidence_level"],
                    "trend": detailed["trend"],
                    "is_reliable": detailed["is_reliable"]
                })
                
            except Exception as e:
                logger.warning(f"Error getting details for method {method}: {e}")
                results.append({
                    "method": method,
                    "strategy": strategy,
                    "confidence_score": score,
                    "confidence_level": "unknown",
                    "trend": "unknown",
                    "is_reliable": False,
                    "error": str(e)
                })
        
        df = pd.DataFrame(results)
        
        # ソート（信頼度スコア降順）
        if not df.empty:
            df = df.sort_values('confidence_score', ascending=False).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error comparing method reliabilities: {e}")
        return pd.DataFrame()

def get_reliability_threshold_analysis(data: pd.DataFrame,
                                     strategies: List[str],
                                     thresholds: List[float] = None) -> Dict[str, Any]:
    """
    信頼度閾値分析
    
    Args:
        data: 価格データ
        strategies: 戦略名リスト
        thresholds: 閾値リスト
        
    Returns:
        Dict[str, Any]: 閾値分析結果
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    try:
        results = {
            "thresholds": thresholds,
            "strategy_analysis": {},
            "summary": {}
        }
        
        for strategy in strategies:
            reliability_info = get_trend_reliability_for_strategy(data, strategy)
            confidence = reliability_info["confidence_score"]
            
            strategy_result = {
                "confidence_score": confidence,
                "reliable_at_thresholds": {}
            }
            
            for threshold in thresholds:
                strategy_result["reliable_at_thresholds"][threshold] = confidence >= threshold
            
            results["strategy_analysis"][strategy] = strategy_result
        
        # サマリー統計
        for threshold in thresholds:
            reliable_count = sum(
                1 for strategy_data in results["strategy_analysis"].values()
                if strategy_data["reliable_at_thresholds"][threshold]
            )
            results["summary"][threshold] = {
                "reliable_strategies": reliable_count,
                "reliability_rate": reliable_count / len(strategies) if strategies else 0
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in threshold analysis: {e}")
        return {"error": str(e)}

# 便利関数
def _get_confidence_level(confidence: float) -> str:
    """信頼度レベルの判定（内部用）"""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    elif confidence >= 0.4:
        return "low"
    else:
        return "unreliable"

# テスト用のサンプル関数
def create_sample_reliability_test(data: pd.DataFrame) -> Dict[str, Any]:
    """
    サンプル信頼度テストの実行
    
    Args:
        data: 価格データ
        
    Returns:
        Dict[str, Any]: テスト結果
    """
    sample_strategies = ["GCStrategy", "BreakoutStrategy", "OpeningGapStrategy"]
    sample_methods = ["sma", "macd", "combined"]
    
    try:
        results = {
            "strategy_comparison": compare_strategy_reliabilities(data, sample_strategies).to_dict('records'),
            "method_comparison": compare_method_reliabilities(data, sample_methods).to_dict('records'),
            "threshold_analysis": get_reliability_threshold_analysis(data, sample_strategies)
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in sample test: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # テスト実行用
    logging.basicConfig(level=logging.INFO)
    logger.info("Trend Reliability Utils module loaded successfully")
