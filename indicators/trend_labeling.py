"""
Module: Trend Labeling
File: trend_labeling.py
Description: 
  過去の株価データにトレンドラベル（上昇・下降・レンジ）を適用するモジュールです。
  unified_trend_detector.pyを活用して、信頼度スコア付きのラベリングを行います。
  ラベリングデータの保存・読み込み機能も提供します。

Author: imega
Created: 2025-07-08
Modified: 2025-07-08

Dependencies:
  - pandas
  - numpy
  - indicators.unified_trend_detector
  - indicators.trend_accuracy_validator
  - config.logger_config
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Literal, Any
import logging
import sys

# プロジェクトのルートパスを追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# 既存モジュールのインポート
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend
from indicators.trend_accuracy_validator import TrendAccuracyValidator
from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__)

class TrendLabeler:
    """
    株価データに対してトレンドラベルを適用するクラス
    
    既存のUnifiedTrendDetectorを活用してラベリングを行う
    """
    
    def __init__(self, data: pd.DataFrame, price_column: str = "Adj Close"):
        """
        初期化
        
        Parameters:
            data (pd.DataFrame): 株価データ
            price_column (str): 価格カラム名
        """
        self.data = data.copy()
        self.price_column = price_column
        self.labeled_data = None
        
        # データの妥当性チェック
        if self.price_column not in self.data.columns:
            # Adj Closeがない場合はCloseを使用
            if "Close" in self.data.columns:
                logger.warning(f"'{self.price_column}'が見つかりません。'Close'を使用します。")
                self.price_column = "Close"
            else:
                raise ValueError(f"価格カラム '{self.price_column}' および 'Close' がデータに存在しません")
        
        # インデックスがDatetimeIndexかチェック
        if not isinstance(self.data.index, pd.DatetimeIndex):
            logger.warning("インデックスがDatetimeIndexではありません。変換を試みます。")
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                logger.error(f"インデックスの日付変換に失敗しました: {e}")
                raise
        
        logger.info(f"TrendLabelerを初期化しました。データ期間: {self.data.index[0]} から {self.data.index[-1]} ({len(self.data)}日間)")
        
    def label_trends(self, 
                    method: str = "advanced", 
                    strategy_name: str = "default",
                    window_size: int = 20,
                    confidence_threshold: float = 0.6,
                    keep_all_scores: bool = True) -> pd.DataFrame:
        """
        株価データにトレンドラベルを適用
        
        Parameters:
            method (str): 使用するトレンド判定手法 ("sma", "macd", "combined", "advanced")
            strategy_name (str): 戦略名（パラメータ最適化のため）
            window_size (int): トレンド判定に使用するウィンドウサイズ
            confidence_threshold (float): 信頼度の閾値（この値以上のラベルを有効とする）
            keep_all_scores (bool): すべての信頼度スコアを保持するかどうか
            
        Returns:
            pd.DataFrame: トレンドラベルとスコアを追加したデータフレーム
        """
        if len(self.data) < window_size:
            logger.error(f"データサイズ({len(self.data)})がウィンドウサイズ({window_size})より小さいため、ラベリングできません")
            return self.data.copy()
        
        logger.info(f"トレンドラベリングを開始: 方法={method}, 戦略={strategy_name}, ウィンドウサイズ={window_size}")
        
        # トレンドラベルと信頼度スコアを格納するリスト
        trends: List[Optional[str]] = []
        confidences: List[float] = []
        
        # UnifiedTrendDetectorを使用してラベリング
        try:
            detector = UnifiedTrendDetector(
                self.data, 
                price_column=self.price_column, 
                strategy_name=strategy_name, 
                method=method
            )
            
            # 各時点でのトレンド判定
            for i in range(len(self.data)):
                # ウィンドウサイズ以下のデータはラベル付けしない
                if i < window_size:
                    trends.append(None)
                    confidences.append(0.0)
                    continue
                
                try:
                    # 現時点までのデータでトレンド判定
                    window_data = self.data.iloc[:i+1]
                    
                    # UnifiedTrendDetectorを使用
                    temp_detector = UnifiedTrendDetector(
                        window_data, 
                        price_column=self.price_column, 
                        strategy_name=strategy_name, 
                        method=method
                    )
                    
                    # トレンド判定と信頼度取得
                    trend, confidence = temp_detector.detect_trend_with_confidence()
                    
                    # 信頼度が閾値以下の場合はNoneとする（オプション）
                    if not keep_all_scores and confidence < confidence_threshold:
                        trend = None
                    
                    trends.append(trend)
                    confidences.append(confidence)
                    
                except Exception as e:
                    logger.warning(f"インデックス {i} でのトレンド判定エラー: {e}")
                    trends.append(None)
                    confidences.append(0.0)
            
        except Exception as e:
            logger.error(f"UnifiedTrendDetectorの初期化に失敗しました: {e}")
            # フォールバック: シンプルなトレンド判定
            trends, confidences = self._fallback_trend_labeling(window_size)
        
        # 結果をデータフレームに追加
        result_df = self.data.copy()
        result_df['trend'] = trends
        result_df['trend_confidence'] = confidences
        
        # 信頼度に基づくフィルタリング列の追加
        result_df['trend_reliable'] = result_df['trend_confidence'] >= confidence_threshold
        
        # メタデータの追加
        result_df.attrs['labeling_method'] = method
        result_df.attrs['strategy_name'] = strategy_name
        result_df.attrs['window_size'] = window_size
        result_df.attrs['confidence_threshold'] = confidence_threshold
        result_df.attrs['labeling_timestamp'] = datetime.now().isoformat()
        
        # クラス変数に保存
        self.labeled_data = result_df
        
        # ラベリング統計を出力
        self._print_labeling_stats(result_df)
        
        return result_df
    
    def _fallback_trend_labeling(self, window_size: int) -> Tuple[List[Optional[str]], List[float]]:
        """
        UnifiedTrendDetectorが利用できない場合のフォールバック処理
        シンプルなSMAベースのトレンド判定
        """
        logger.warning("フォールバック処理でシンプルなトレンド判定を実行します")
        
        trends: List[Optional[str]] = []
        confidences: List[float] = []
        
        # SMAを計算
        sma_short = self.data[self.price_column].rolling(window=10).mean()
        sma_long = self.data[self.price_column].rolling(window=window_size).mean()
        
        for i in range(len(self.data)):
            if i < window_size:
                trends.append(None)
                confidences.append(0.0)
                continue
            
            try:
                current_price = self.data[self.price_column].iloc[i]
                short_ma = sma_short.iloc[i]
                long_ma = sma_long.iloc[i]
                
                # シンプルなトレンド判定
                if pd.notna(short_ma) and pd.notna(long_ma):
                    if current_price > short_ma and short_ma > long_ma:
                        trends.append("uptrend")
                        confidences.append(0.7)
                    elif current_price < short_ma and short_ma < long_ma:
                        trends.append("downtrend")
                        confidences.append(0.7)
                    else:
                        trends.append("range-bound")
                        confidences.append(0.5)
                else:
                    trends.append(None)
                    confidences.append(0.0)
                    
            except Exception as e:
                logger.warning(f"フォールバック処理でエラー (index {i}): {e}")
                trends.append(None)
                confidences.append(0.0)
        
        return trends, confidences
    
    def _print_labeling_stats(self, df: pd.DataFrame) -> None:
        """ラベリング統計の出力"""
        if 'trend' not in df.columns:
            return
            
        # トレンドタイプごとの件数
        trend_counts = df['trend'].value_counts().to_dict()
        valid_count = df['trend'].notna().sum()
        
        # 信頼度の統計
        mean_conf = df['trend_confidence'].mean()
        reliable_count = df['trend_reliable'].sum()
        
        logger.info(f"ラベリング完了: 総データ数={len(df)}, 有効ラベル数={valid_count}")
        logger.info(f"トレンド分布: {trend_counts}")
        logger.info(f"平均信頼度: {mean_conf:.3f}, 信頼できるラベル数: {reliable_count}")
    
    def extract_trend_periods(self, 
                             trend_type: Literal["uptrend", "downtrend", "range-bound"],
                             min_period_length: int = 5,
                             min_confidence: float = 0.7) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        特定のトレンドタイプの期間を抽出
        
        Parameters:
            trend_type (str): 抽出するトレンドタイプ
            min_period_length (int): 最小期間長
            min_confidence (float): 最小信頼度
            
        Returns:
            List[Tuple]: 開始日と終了日のタプルのリスト
        """
        if self.labeled_data is None or 'trend' not in self.labeled_data.columns:
            logger.error("ラベリングが実行されていないため、トレンド期間を抽出できません")
            return []
        
        # 指定されたトレンドタイプかつ信頼度が閾値以上のデータをフィルタリング
        filtered_data = self.labeled_data[
            (self.labeled_data['trend'] == trend_type) & 
            (self.labeled_data['trend_confidence'] >= min_confidence)
        ]
        
        if filtered_data.empty:
            logger.warning(f"指定されたトレンドタイプ '{trend_type}' の期間が見つかりませんでした")
            return []
        
        # 連続したトレンド期間を特定
        periods = []
        current_start = None
        current_end = None
        
        # インデックスを順番に処理
        prev_idx = None
        for idx in filtered_data.index:
            if current_start is None:
                # 新しい期間の開始
                current_start = idx
                current_end = idx
            elif prev_idx is not None:
                # 前のインデックスと連続しているかチェック
                time_diff = (idx - prev_idx).days
                if time_diff <= 3:  # 3日以内なら連続とみなす（週末を考慮）
                    current_end = idx
                else:
                    # 連続していない場合は期間を終了
                    period_length = (current_end - current_start).days + 1
                    if period_length >= min_period_length:
                        periods.append((current_start, current_end))
                    
                    # 新しい期間の開始
                    current_start = idx
                    current_end = idx
            
            prev_idx = idx
        
        # 最後の期間の処理
        if current_start is not None and current_end is not None:
            period_length = (current_end - current_start).days + 1
            if period_length >= min_period_length:
                periods.append((current_start, current_end))
        
        logger.info(f"抽出されたトレンド期間数 ({trend_type}): {len(periods)}")
        for i, (start, end) in enumerate(periods[:5]):  # 最初の5つのみログ出力
            logger.debug(f"  期間 {i+1}: {start.strftime('%Y-%m-%d')} から {end.strftime('%Y-%m-%d')} ({(end-start).days + 1}日間)")
        
        return periods
    
    def save_labeled_data(self, filepath: str, format: str = "csv") -> bool:
        """
        ラベリング済みデータを保存
        
        Parameters:
            filepath (str): 保存先ファイルパス
            format (str): 保存形式 ("csv" or "pickle")
            
        Returns:
            bool: 成功したかどうか
        """
        if self.labeled_data is None:
            logger.error("ラベリングが実行されていないため、データを保存できません")
            return False
        
        try:
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 指定された形式で保存
            if format.lower() == "csv":
                self.labeled_data.to_csv(filepath)
                
                # メタデータを別ファイルに保存
                meta_filepath = filepath.replace('.csv', '_metadata.json')
                metadata = dict(self.labeled_data.attrs)
                with open(meta_filepath, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == "pickle":
                self.labeled_data.to_pickle(filepath)
            else:
                logger.error(f"未対応の保存形式: {format}")
                return False
                
            logger.info(f"ラベリング済みデータを保存しました: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"データ保存中にエラーが発生しました: {e}")
            return False
    
    def load_labeled_data(self, filepath: str, format: str = "csv") -> bool:
        """
        ラベリング済みデータを読み込み
        
        Parameters:
            filepath (str): 読み込み元ファイルパス
            format (str): 読み込み形式 ("csv" or "pickle")
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            if format.lower() == "csv":
                self.labeled_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # メタデータの読み込み
                meta_filepath = filepath.replace('.csv', '_metadata.json')
                if os.path.exists(meta_filepath):
                    with open(meta_filepath, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    self.labeled_data.attrs.update(metadata)
                    
            elif format.lower() == "pickle":
                self.labeled_data = pd.read_pickle(filepath)
            else:
                logger.error(f"未対応の読み込み形式: {format}")
                return False
                
            logger.info(f"ラベリング済みデータを読み込みました: {filepath}")
            
            # 読み込んだデータの統計を表示
            self._print_labeling_stats(self.labeled_data)
            
            return True
            
        except Exception as e:
            logger.error(f"データ読み込み中にエラーが発生しました: {e}")
            return False
    
    def validate_labeling_accuracy(self, 
                                 future_window: int = 10,
                                 trend_threshold: float = 0.02) -> Dict[str, Any]:
        """
        ラベリング精度を検証
        TrendAccuracyValidatorを使用して未来の価格変動に基づく精度を測定
        
        Parameters:
            future_window (int): 未来何日先まで見るか
            trend_threshold (float): トレンド判定の閾値
            
        Returns:
            Dict: 精度指標
        """
        if self.labeled_data is None:
            logger.error("ラベリングが実行されていないため、精度検証できません")
            return {"error": "ラベリング未実行"}
        
        try:
            validator = TrendAccuracyValidator(self.data, self.price_column)
            
            # トレンド判定関数の定義
            def labeled_trend_detector(data):
                if self.labeled_data is None:
                    return "unknown"
                
                # データの最後の日付を取得
                last_date = data.index[-1]
                
                # その日付に対応するラベルを返す
                if last_date in self.labeled_data.index:
                    trend = self.labeled_data.loc[last_date, 'trend']
                    return trend if pd.notna(trend) else "unknown"
                else:
                    return "unknown"
            
            # 精度検証の実行
            validation_params = {"future_window": future_window, "trend_threshold": trend_threshold}
            accuracy_results = validator.validate_trend_accuracy(labeled_trend_detector, validation_params)
            
            logger.info(f"ラベリング精度検証結果: 全体精度={accuracy_results.get('overall_accuracy', 0):.3f}")
            return accuracy_results
            
        except Exception as e:
            logger.error(f"精度検証中にエラーが発生しました: {e}")
            return {"error": str(e)}

# ユーティリティ関数
def label_trends_for_dataframe(data: pd.DataFrame, 
                              price_column: str = "Adj Close",
                              method: str = "advanced",
                              strategy_name: str = "default",
                              window_size: int = 20,
                              confidence_threshold: float = 0.6) -> pd.DataFrame:
    """
    データフレームに対するトレンドラベリングのユーティリティ関数
    
    Parameters:
        data (pd.DataFrame): 株価データ
        price_column (str): 価格カラム名
        method (str): トレンド判定手法
        strategy_name (str): 戦略名
        window_size (int): ウィンドウサイズ
        confidence_threshold (float): 信頼度閾値
        
    Returns:
        pd.DataFrame: ラベリング済みデータフレーム
    """
    labeler = TrendLabeler(data, price_column)
    return labeler.label_trends(method, strategy_name, window_size, confidence_threshold)

def extract_specific_trend_data(data: pd.DataFrame, 
                               trend_type: Literal["uptrend", "downtrend", "range-bound"],
                               min_confidence: float = 0.7) -> pd.DataFrame:
    """
    特定のトレンドタイプのデータのみを抽出するユーティリティ関数
    
    Parameters:
        data (pd.DataFrame): ラベリング済みデータ
        trend_type (str): トレンドタイプ
        min_confidence (float): 最小信頼度
        
    Returns:
        pd.DataFrame: フィルタリングされたデータフレーム
    """
    if 'trend' not in data.columns or 'trend_confidence' not in data.columns:
        logger.error("トレンドラベルが付与されていません")
        return pd.DataFrame()
    
    filtered_data = data[
        (data['trend'] == trend_type) & 
        (data['trend_confidence'] >= min_confidence)
    ]
    
    logger.info(f"フィルタリング結果: {trend_type} のデータ {len(filtered_data)}件を抽出")
    return filtered_data

# テスト用コード
if __name__ == "__main__":
    # サンプルデータでのテスト
    try:
        from data_fetcher import get_parameters_and_data
        
        # データ取得
        ticker, start_date, end_date, stock_data, _ = get_parameters_and_data()
        
        print(f"データ期間: {stock_data.index[0]} から {stock_data.index[-1]}")
        print(f"データ行数: {len(stock_data)}")
        
        # ラベリングの実行
        labeler = TrendLabeler(stock_data)
        labeled_data = labeler.label_trends(method="advanced", strategy_name="default", window_size=20, confidence_threshold=0.7)
        
        # 上昇トレンド期間の抽出
        uptrend_periods = labeler.extract_trend_periods("uptrend", min_period_length=10)
        print(f"上昇トレンド期間数: {len(uptrend_periods)}")
        for start, end in uptrend_periods[:3]:  # 最初の3つのみ表示
            print(f"  {start.strftime('%Y-%m-%d')} から {end.strftime('%Y-%m-%d')}")
        
        # ラベリング精度の検証
        accuracy = labeler.validate_labeling_accuracy()
        print(f"全体精度: {accuracy.get('overall_accuracy', 0):.3f}")
        
        # ラベリング結果の保存
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "labeled_data_test.csv")
        labeler.save_labeled_data(save_path)
        
        print("テスト完了")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
