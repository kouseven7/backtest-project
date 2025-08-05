"""
Module: Trend Limited Backtest
File: trend_limited_backtest_fixed.py
Description: 
  特定のトレンド期間のみでバックテストを実行する機能を提供します。
  既存のBaseStrategyを継承した戦略クラスと、TrendLabelerで生成されたラベリングデータを
  組み合わせて、トレンド環境別のパフォーマンス測定を行います。

Author: imega
Created: 2025-07-08
Modified: 2025-07-08

Dependencies:
  - pandas
  - numpy
  - strategies.base_strategy
  - indicators.trend_labeling
  - config.logger_config
"""

import pandas as pd
import numpy as np
import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Literal, Any, Type
import logging

# プロジェクトのルートパスを追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# 既存モジュールのインポート
from strategies.base_strategy import BaseStrategy
from indicators.trend_labeling import TrendLabeler
from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__)

class TrendLimitedBacktester:
    """
    トレンド期間限定バックテストを実行するクラス
    
    既存の戦略クラス（BaseStrategy継承）とトレンドラベリングデータを
    組み合わせて、特定のトレンド環境下でのパフォーマンスを測定します
    """
    
    def __init__(self, 
                 stock_data: pd.DataFrame,
                 labeled_data: Optional[pd.DataFrame] = None,
                 price_column: str = "Adj Close"):
        """
        初期化
        
        Parameters:
            stock_data (pd.DataFrame): 元の株価データ
            labeled_data (pd.DataFrame, optional): ラベリング済みデータ（Noneの場合は自動生成）
            price_column (str): 価格カラム名
        """
        self.stock_data = stock_data.copy()
        self.price_column = price_column
        
        # ラベリングデータの設定
        if labeled_data is not None:
            self.labeled_data = labeled_data.copy()
            logger.info(f"提供されたラベリングデータを使用: {len(self.labeled_data)}日間")
        else:
            # ラベリングデータが提供されていない場合は自動生成
            logger.info("ラベリングデータが提供されていないため、自動生成します")
            labeler = TrendLabeler(self.stock_data, price_column)
            self.labeled_data = labeler.label_trends()
        
        # データの整合性チェック
        self._validate_data()
        
        # バックテスト結果を保存するための辞書
        self.backtest_results: Dict[str, Dict[str, Any]] = {}
        
    def _validate_data(self) -> None:
        """データの整合性をチェック"""
        # 必要なカラムの存在確認
        required_columns = ['trend', 'trend_confidence', 'trend_reliable']
        missing_columns = [col for col in required_columns if col not in self.labeled_data.columns]
        
        if missing_columns:
            raise ValueError(f"ラベリングデータに必要なカラムが不足しています: {missing_columns}")
        
        # インデックスの整合性確認（pandas DataFrameの標準的な方法を使用）
        if len(self.stock_data.index.intersection(self.labeled_data.index)) == 0:
            raise ValueError("株価データとラベリングデータのインデックスに共通部分がありません")
        
        stock_index_set = set(self.stock_data.index)
        labeled_index_set = set(self.labeled_data.index)
        if stock_index_set != labeled_index_set:
            logger.warning("株価データとラベリングデータのインデックスが一致しません。結合します。")
            # 共通のインデックスのみを使用
            common_index = self.stock_data.index.intersection(self.labeled_data.index)
            self.stock_data = self.stock_data.loc[common_index]
            self.labeled_data = self.labeled_data.loc[common_index]
        
        logger.info(f"データ検証完了: {len(self.stock_data)}日間のデータを使用")
    
    def extract_trend_periods(self, 
                             trend_type: Literal["uptrend", "downtrend", "range-bound"],
                             min_period_length: int = 5,
                             min_confidence: float = 0.7) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
        """
        特定のトレンドタイプの期間を抽出し、各期間のデータを返す
        
        Parameters:
            trend_type (str): 抽出するトレンドタイプ
            min_period_length (int): 最小期間長（営業日）
            min_confidence (float): 最小信頼度
            
        Returns:
            List[Tuple]: (開始日, 終了日, 期間データ) のタプルのリスト
        """
        # 指定されたトレンドタイプかつ信頼度が閾値以上のデータをフィルタリング
        filtered_mask = (
            (self.labeled_data['trend'] == trend_type) & 
            (self.labeled_data['trend_confidence'] >= min_confidence)
        )
        
        if not filtered_mask.any():
            logger.warning(f"指定されたトレンドタイプ '{trend_type}' の期間が見つかりませんでした")
            return []
        
        # 連続したトレンド期間を特定
        periods_with_data: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
        current_start: Optional[pd.Timestamp] = None
        current_end: Optional[pd.Timestamp] = None
        
        filtered_dates = self.labeled_data[filtered_mask].index
        
        for i, date in enumerate(filtered_dates):
            if current_start is None:
                # 新しい期間の開始
                current_start = date
                current_end = date
            elif i > 0:
                # 前の日付との差を計算（営業日ベース）
                prev_date = filtered_dates[i-1]
                days_diff = (date - prev_date).days
                
                if days_diff <= 5:  # 5日以内なら連続とみなす（週末・祝日考慮）
                    current_end = date
                else:
                    # 連続していない場合は期間を終了し保存
                    if current_start is not None and current_end is not None:
                        self._save_period_if_valid(
                            current_start, current_end, min_period_length, 
                            periods_with_data, trend_type
                        )
                    
                    # 新しい期間の開始
                    current_start = date
                    current_end = date
        
        # 最後の期間を保存
        if current_start is not None and current_end is not None:
            self._save_period_if_valid(
                current_start, current_end, min_period_length,
                periods_with_data, trend_type
            )
        
        logger.info(f"抽出されたトレンド期間数 ({trend_type}): {len(periods_with_data)}")
        return periods_with_data
    
    def _save_period_if_valid(self, 
                             start_date: pd.Timestamp, 
                             end_date: pd.Timestamp,
                             min_period_length: int,
                             periods_list: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]],
                             trend_type: str) -> None:
        """期間が最小長以上の場合にリストに保存"""
        # 実際の営業日数を計算
        period_data = self.stock_data.loc[start_date:end_date]
        actual_length = len(period_data)
        
        if actual_length >= min_period_length:
            periods_list.append((start_date, end_date, period_data))
            logger.debug(f"{trend_type} 期間追加: {start_date.strftime('%Y-%m-%d')} から {end_date.strftime('%Y-%m-%d')} ({actual_length}営業日)")
    
    def run_trend_limited_backtest(self,
                                  strategy_class: Type[BaseStrategy],
                                  trend_type: Literal["uptrend", "downtrend", "range-bound"],
                                  strategy_params: Dict[str, Any],
                                  min_period_length: int = 5,
                                  min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        特定のトレンド期間に限定してバックテストを実行
        
        Parameters:
            strategy_class (Type[BaseStrategy]): 戦略クラス（BaseStrategy継承）
            trend_type (str): 対象とするトレンドタイプ
            strategy_params (Dict): 戦略のパラメータ
            min_period_length (int): 最小期間長
            min_confidence (float): 最小信頼度
            
        Returns:
            Dict[str, Any]: バックテスト結果の要約
        """
        logger.info(f"トレンド限定バックテスト開始: {trend_type} - {strategy_class.__name__}")
        
        # トレンド期間を抽出
        trend_periods = self.extract_trend_periods(trend_type, min_period_length, min_confidence)
        
        if not trend_periods:
            logger.warning(f"対象となるトレンド期間が見つかりませんでした: {trend_type}")
            return {
                "strategy": strategy_class.__name__,
                "trend_type": trend_type,
                "error": "No trend periods found",
                "periods_found": 0
            }
        
        # 各期間でバックテストを実行
        period_results: List[Dict[str, Any]] = []
        
        for start_date, end_date, period_data in trend_periods:
            try:
                # 戦略インスタンスを作成
                strategy = strategy_class(period_data, **strategy_params)
                
                # バックテストを実行
                backtest_result = strategy.backtest()
                
                # 簡易収益計算
                trades = self._calculate_simple_returns(backtest_result, start_date, end_date)
                
                period_result: Dict[str, Any] = {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "period_length": len(period_data),
                    "entry_signals": int((backtest_result['entry_signal'] == 1).sum()),
                    "exit_signals": int((backtest_result['exit_signal'] == 1).sum()),
                    "trades": trades,
                    "total_return": sum(trade["return_pct"] for trade in trades),
                    "win_rate": len([t for t in trades if t["return_pct"] > 0]) / len(trades) if trades else 0
                }
                
                period_results.append(period_result)
                
            except Exception as e:
                logger.error(f"期間 {start_date} - {end_date} のバックテストでエラー: {str(e)}")
                period_results.append({
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "error": str(e)
                })
        
        # 全体の結果を集計
        successful_periods = [r for r in period_results if "error" not in r]
        total_entries = sum(r.get("entry_signals", 0) for r in successful_periods)
        total_exits = sum(r.get("exit_signals", 0) for r in successful_periods)
        total_trades = sum(len(r.get("trades", [])) for r in successful_periods)
        
        # 全取引の収益率を取得
        all_returns: List[float] = []
        for period in successful_periods:
            trades_list = period.get("trades", [])
            if isinstance(trades_list, list):
                all_returns.extend([trade["return_pct"] for trade in trades_list])
        
        avg_return = float(np.mean(all_returns)) if all_returns else 0.0
        win_rate = len([r for r in all_returns if r > 0]) / len(all_returns) if all_returns else 0.0
        
        result_summary: Dict[str, Any] = {
            "strategy": strategy_class.__name__,
            "trend_type": trend_type,
            "strategy_params": strategy_params,
            "analysis_config": {
                "min_period_length": min_period_length,
                "min_confidence": min_confidence
            },
            "summary": {
                "periods_found": len(trend_periods),
                "periods_tested": len(successful_periods),
                "total_entry_signals": total_entries,
                "total_exit_signals": total_exits,
                "total_trades": total_trades,
                "average_return_pct": avg_return,
                "win_rate": win_rate,
                "success_rate": len(successful_periods) / len(trend_periods) if trend_periods else 0
            },
            "period_details": period_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # 結果をクラス内に保存
        key = f"{strategy_class.__name__}_{trend_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backtest_results[key] = result_summary
        
        logger.info(f"バックテスト完了: {len(successful_periods)}/{len(trend_periods)} 期間で成功, 合計{total_entries}回のエントリー")
        
        return result_summary
    
    def _calculate_simple_returns(self, 
                                 backtest_result: pd.DataFrame, 
                                 start_date: pd.Timestamp, 
                                 end_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """
        バックテスト結果から簡易的な取引収益を計算
        
        Parameters:
            backtest_result (pd.DataFrame): バックテスト結果
            start_date (pd.Timestamp): 期間開始日
            end_date (pd.Timestamp): 期間終了日
            
        Returns:
            List[Dict]: 各取引の詳細リスト
        """
        trades: List[Dict[str, Any]] = []
        entry_date: Optional[pd.Timestamp] = None
        entry_price: Optional[float] = None
        
        # 簡易的なシグナル検出（型エラー対策）
        try:
            for i in range(len(backtest_result)):
                row_data = backtest_result.iloc[i]
                idx_val = backtest_result.index[i]
                
                # entry_signalの確認
                entry_signal_val = 0
                if 'entry_signal' in backtest_result.columns:
                    entry_signal_val = int(row_data['entry_signal']) if pd.notna(row_data['entry_signal']) else 0
                
                # exit_signalの確認
                exit_signal_val = 0  
                if 'exit_signal' in backtest_result.columns:
                    exit_signal_val = int(row_data['exit_signal']) if pd.notna(row_data['exit_signal']) else 0
                
                if entry_signal_val == 1 and entry_date is None:
                    entry_date = pd.Timestamp(idx_val)
                    entry_price = float(row_data[self.price_column])
                elif exit_signal_val == 1 and entry_date is not None and entry_price is not None:
                    exit_date = pd.Timestamp(idx_val)
                    exit_price = float(row_data[self.price_column])
                
                # 収益率計算
                return_pct = (exit_price - entry_price) / entry_price
                days_held = (exit_date - entry_date).days
                
                trades.append({
                    "entry_date": entry_date.isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": return_pct,
                    "days_held": days_held
                })
                
                # リセット
                entry_date = None
                entry_price = None
        
        return trades
    
    def compare_trend_environments(self, 
                                  strategy_class: Type[BaseStrategy],
                                  strategy_params: Dict[str, Any],
                                  trend_types: Optional[List[Literal["uptrend", "downtrend", "range-bound"]]] = None,
                                  min_period_length: int = 5,
                                  min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        複数のトレンド環境での戦略パフォーマンスを比較
        
        Parameters:
            strategy_class (Type[BaseStrategy]): 戦略クラス
            strategy_params (Dict): 戦略パラメータ
            trend_types (List): 比較するトレンドタイプのリスト
            min_period_length (int): 最小期間長
            min_confidence (float): 最小信頼度
            
        Returns:
            Dict[str, Any]: 比較結果
        """
        if trend_types is None:
            trend_types = ["uptrend", "downtrend", "range-bound"]
        
        logger.info(f"トレンド環境比較開始: {strategy_class.__name__}")
        
        comparison_results: Dict[str, Any] = {}
        
        for trend_type in trend_types:
            result = self.run_trend_limited_backtest(
                strategy_class, trend_type, strategy_params,
                min_period_length, min_confidence
            )
            comparison_results[trend_type] = result
        
        # 比較サマリーを作成
        summary_comparison = {
            "strategy": strategy_class.__name__,
            "comparison_timestamp": datetime.now().isoformat(),
            "trend_comparison": {}
        }
        
        for trend_type, result in comparison_results.items():
            if "error" not in result:
                summary = result.get("summary", {})
                summary_comparison["trend_comparison"][trend_type] = {
                    "periods_tested": summary.get("periods_tested", 0),
                    "total_trades": summary.get("total_trades", 0),
                    "average_return": summary.get("average_return_pct", 0.0),
                    "win_rate": summary.get("win_rate", 0.0),
                    "success_rate": summary.get("success_rate", 0.0)
                }
            else:
                summary_comparison["trend_comparison"][trend_type] = {
                    "error": result.get("error", "Unknown error")
                }
        
        comparison_results["summary"] = summary_comparison
        
        return comparison_results
    
    def save_results_to_json(self, 
                            results: Dict[str, Any], 
                            filename: Optional[str] = None,
                            output_dir: str = "logs") -> str:
        """
        結果をJSONファイルに保存
        
        Parameters:
            results (Dict): 保存する結果
            filename (str, optional): ファイル名（Noneの場合は自動生成）
            output_dir (str): 出力ディレクトリ
            
        Returns:
            str: 保存されたファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = results.get("strategy", "unknown")
            trend_type = results.get("trend_type", "comparison")
            filename = f"trend_backtest_{strategy_name}_{trend_type}_{timestamp}.json"
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # JSON形式で保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"結果をJSONファイルに保存: {filepath}")
        return filepath


def run_trend_limited_backtest(stock_data: pd.DataFrame,
                              strategy_class: Type[BaseStrategy],
                              strategy_params: Dict[str, Any],
                              trend_type: Literal["uptrend", "downtrend", "range-bound"],
                              labeled_data: Optional[pd.DataFrame] = None,
                              min_period_length: int = 5,
                              min_confidence: float = 0.7,
                              save_results: bool = True,
                              output_dir: str = "logs") -> Dict[str, Any]:
    """
    トレンド期間限定バックテストの実行関数（スタンドアロン版）
    
    Parameters:
        stock_data (pd.DataFrame): 株価データ
        strategy_class (Type[BaseStrategy]): 戦略クラス
        strategy_params (Dict): 戦略パラメータ
        trend_type (str): 対象トレンドタイプ
        labeled_data (pd.DataFrame, optional): ラベリング済みデータ
        min_period_length (int): 最小期間長
        min_confidence (float): 最小信頼度
        save_results (bool): 結果保存するかどうか
        output_dir (str): 出力ディレクトリ
        
    Returns:
        Dict[str, Any]: バックテスト結果
    """
    # バックテスターインスタンスを作成
    backtester = TrendLimitedBacktester(stock_data, labeled_data)
    
    # バックテストを実行
    result = backtester.run_trend_limited_backtest(
        strategy_class, trend_type, strategy_params,
        min_period_length, min_confidence
    )
    
    # 結果を保存
    if save_results:
        backtester.save_results_to_json(result, output_dir=output_dir)
    
    return result


if __name__ == "__main__":
    # テスト用のサンプルコード
    print("Trend Limited Backtest Module - テスト用実行")
    print("実際の使用には、適切な株価データと戦略クラスが必要です。")
