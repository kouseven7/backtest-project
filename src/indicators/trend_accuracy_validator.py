"""
Module: Trend Accuracy Validator
File: trend_accuracy_validator.py
Description: 
  トレンド判定の精度を測定・検証するためのモジュールです。
  将来の価格変動から正解ラベルを作成し、トレンド判定器の精度を評価します。

Author: imega
Created: 2025-07-03
Modified: 2025-07-03

Dependencies:
  - pandas
  - numpy
  - logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from datetime import datetime, timedelta
import logging

class TrendAccuracyValidator:
    """トレンド判定精度の検証クラス"""
    
    def __init__(self, data: pd.DataFrame, price_column: str = "Adj Close"):
        """
        初期化
        
        Parameters:
            data (pd.DataFrame): 株価データ
            price_column (str): 価格カラム名
        """
        self.data = data
        self.price_column = price_column
        self.logger = logging.getLogger(__name__)
    
    def create_ground_truth_trends(self, 
                                  future_window: int = 10,
                                  trend_threshold: float = 0.02) -> pd.Series:
        """
        将来の価格変動から正解トレンドラベルを作成
        
        Parameters:
            future_window (int): 未来何日先まで見るか
            trend_threshold (float): トレンド判定の閾値
        
        Returns:
            pd.Series: 正解トレンドラベル ("uptrend", "downtrend", "range-bound")
        """
        ground_truth = []
        
        for i in range(len(self.data) - future_window):
            current_price = self.data[self.price_column].iloc[i]
            
            # 未来のN日間の価格変動を確認
            future_prices = self.data[self.price_column].iloc[i+1:i+1+future_window]
            
            if len(future_prices) == 0:
                ground_truth.append("unknown")
                continue
            
            # 最大上昇率と最大下落率を計算
            max_gain = (future_prices.max() / current_price) - 1
            max_loss = 1 - (future_prices.min() / current_price)
            
            # 最終的な価格変動
            final_price = future_prices.iloc[-1]
            final_change = (final_price / current_price) - 1
            
            # トレンド判定ロジック
            if max_gain > trend_threshold and final_change > trend_threshold * 0.5:
                ground_truth.append("uptrend")
            elif max_loss > trend_threshold and final_change < -trend_threshold * 0.5:
                ground_truth.append("downtrend")
            else:
                ground_truth.append("range-bound")
        
        # 残りの期間は判定不可
        for _ in range(future_window):
            ground_truth.append("unknown")
        
        return pd.Series(ground_truth, index=self.data.index)
    
    def validate_trend_accuracy(self, 
                               trend_detector: Callable[..., str],
                               validation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        トレンド判定器の精度を検証
        
        Parameters:
            trend_detector: トレンド判定関数
            validation_params: 検証パラメータ
        
        Returns:
            Dict: 精度指標
        """
        if validation_params is None:
            validation_params = {"future_window": 10, "trend_threshold": 0.02}
        
        # 正解ラベルを作成
        ground_truth = self.create_ground_truth_trends(**validation_params)
        
        # 予測ラベルを生成
        predictions = []
        for i in range(len(self.data)):
            if i < 50:  # 最初の50日は計算に必要なデータが不足
                predictions.append("unknown")
                continue
            
            # 現在時点までのデータで予測
            current_data = self.data.iloc[:i+1]
            try:
                pred = trend_detector(current_data)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"インデックス {i} でのトレンド判定エラー: {e}")
                predictions.append("unknown")
        
        predictions = pd.Series(predictions, index=self.data.index)
        
        # 精度計算（unknownを除外）
        valid_mask = (ground_truth != "unknown") & (predictions != "unknown")
        valid_ground_truth = ground_truth[valid_mask]
        valid_predictions = predictions[valid_mask]
        
        if len(valid_ground_truth) == 0:
            return {"error": "有効な検証データがありません"}
        
        # 全体精度
        accuracy = (valid_predictions == valid_ground_truth).mean()
        
        # クラス別精度
        trend_types = ["uptrend", "downtrend", "range-bound"]
        class_accuracies = {}
        
        for trend_type in trend_types:
            mask = valid_ground_truth == trend_type
            if mask.sum() > 0:
                class_acc = (valid_predictions[mask] == valid_ground_truth[mask]).mean()
                class_accuracies[f"{trend_type}_accuracy"] = class_acc
            else:
                class_accuracies[f"{trend_type}_accuracy"] = 0.0
        
        # 混同行列
        confusion_matrix = self._create_confusion_matrix(valid_ground_truth, valid_predictions)
        
        return {
            "overall_accuracy": accuracy,
            "total_samples": len(valid_ground_truth),
            "validation_period": f"{valid_ground_truth.index[0]} to {valid_ground_truth.index[-1]}",
            **class_accuracies,
            "confusion_matrix": confusion_matrix
        }
    
    def _create_confusion_matrix(self, true_labels: pd.Series, pred_labels: pd.Series) -> Dict:
        """混同行列を作成"""
        trends = ["uptrend", "downtrend", "range-bound"]
        matrix = {}
        
        for true_trend in trends:
            matrix[f"true_{true_trend}"] = {}
            true_mask = true_labels == true_trend
            
            for pred_trend in trends:
                pred_count = (pred_labels[true_mask] == pred_trend).sum()
                matrix[f"true_{true_trend}"][f"pred_{pred_trend}"] = int(pred_count)
        
        return matrix
    
    def run_comprehensive_validation(self, 
                                   trend_detectors: Dict[str, Callable[..., str]],
                                   validation_windows: List[int] = [5, 10, 15, 20]) -> pd.DataFrame:
        """
        複数のトレンド判定器と検証期間で包括的な精度検証
        
        Parameters:
            trend_detectors: トレンド判定器の辞書
            validation_windows: 検証期間のリスト
        
        Returns:
            pd.DataFrame: 検証結果
        """
        results = []
        
        for detector_name, detector_func in trend_detectors.items():
            for window in validation_windows:
                validation_params = {"future_window": window, "trend_threshold": 0.02}
                
                try:
                    accuracy_results = self.validate_trend_accuracy(detector_func, validation_params)
                    
                    if "error" in accuracy_results:
                        self.logger.error(f"検証エラー ({detector_name}, window={window}): {accuracy_results['error']}")
                        continue
                    
                    result_row = {
                        "detector": detector_name,
                        "validation_window": window,
                        "overall_accuracy": accuracy_results.get("overall_accuracy", 0),
                        "uptrend_accuracy": accuracy_results.get("uptrend_accuracy", 0),
                        "downtrend_accuracy": accuracy_results.get("downtrend_accuracy", 0),
                        "range_accuracy": accuracy_results.get("range-bound_accuracy", 0),
                        "total_samples": accuracy_results.get("total_samples", 0)
                    }
                    results.append(result_row)
                    
                except Exception as e:
                    self.logger.error(f"検証エラー ({detector_name}, window={window}): {e}")
        
        return pd.DataFrame(results)

# テスト用のトレンド判定器の例
def simple_sma_trend_detector(data: pd.DataFrame, 
                             price_column: str = "Adj Close") -> str:
    """シンプルなSMAベースのトレンド判定器"""
    if len(data) < 20:
        return "unknown"
    
    short_sma = data[price_column].rolling(10).mean().iloc[-1]
    long_sma = data[price_column].rolling(20).mean().iloc[-1]
    current_price = data[price_column].iloc[-1]
    
    if current_price > short_sma > long_sma:
        return "uptrend"
    elif current_price < short_sma < long_sma:
        return "downtrend"
    else:
        return "range-bound"

if __name__ == "__main__":
    # テスト用コード
    print("トレンド精度検証モジュールが正常にインポートされました")
