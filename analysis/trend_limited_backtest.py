"""
Module: Trend Limited Backtest
File: trend_limited_backtest.py
Description: 
  特定のトレンド期間のみでバックテストを実行する機能を提供します。
  既存のBaseStrategyを継承した戦略クラ                else:
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
            )    def _save_period_if_valid(self, 
                             start_date: pd.Timestamp, 
                             end_date: Optional[pd.Timestamp], 
                             min_period_length: int,
                             periods_list: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]],
                             trend_type: str) -> None:ベリングデータを
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
from typing import Dict, List, Tuple, Union, Optional, Literal, Any, Type, cast
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
        
        # インデックスの整合性確認
        if not self.stock_data.index.equals(self.labeled_data.index):
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
                    self._save_period_if_valid(
                        current_start, current_end, min_period_length, 
                        periods_with_data, trend_type
                    )
                    
                    # 新しい期間の開始
                    current_start = date
                    current_end = date
        
        # 最後の期間の処理
        if current_start is not None:
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
                             periods_list: List,
                             trend_type: str) -> None:
        """期間が最小長以上の場合にリストに保存"""
        if end_date is None:
            return
            
        # 実際の営業日数を計算
        period_data = self.stock_data.loc[start_date:end_date]
        actual_length = len(period_data)
        
        if actual_length >= min_period_length:
            periods_list.append((start_date, end_date, period_data))
            logger.debug(f"{trend_type} 期間追加: {start_date.strftime('%Y-%m-%d')} から {end_date.strftime('%Y-%m-%d')} ({actual_length}営業日)")
    
    def run_strategy_on_trend_periods(self, 
                                    strategy_class: Type[BaseStrategy],
                                    strategy_params: Optional[Dict[str, Any]] = None,
                                    trend_type: Literal["uptrend", "downtrend", "range-bound"] = "uptrend",
                                    min_period_length: int = 10,
                                    min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        特定のトレンド期間のみで戦略を実行
        
        Parameters:
            strategy_class (Type[BaseStrategy]): 戦略クラス
            strategy_params (Dict, optional): 戦略パラメータ
            trend_type (str): 対象のトレンドタイプ
            min_period_length (int): 最小期間長
            min_confidence (float): 最小信頼度
            
        Returns:
            Dict: バックテスト結果とメタデータ
        """
        logger.info(f"トレンド期間限定バックテストを開始: {strategy_class.__name__} on {trend_type}")
        
        # トレンド期間の抽出
        trend_periods = self.extract_trend_periods(trend_type, min_period_length, min_confidence)
        
        if not trend_periods:
            logger.warning(f"対象となる{trend_type}期間が見つかりませんでした")
            return {
                "strategy_name": strategy_class.__name__,
                "trend_type": trend_type,
                "error": "対象期間なし",
                "periods_tested": 0,
                "total_days": 0
            }
        
        # 各期間でバックテストを実行
        period_results = []
        total_days = 0
        
        for i, (start_date, end_date, period_data) in enumerate(trend_periods):
            logger.info(f"期間 {i+1}/{len(trend_periods)} を処理中: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
            
            try:
                # 戦略インスタンスを作成
                strategy = strategy_class(period_data, strategy_params)
                
                # バックテストを実行
                result = strategy.backtest()
                
                # パフォーマンス指標の計算
                entry_signals = (result['Entry_Signal'] == 1).sum()
                exit_signals = (result['Exit_Signal'] == -1).sum()
                
                # 簡単な収益計算
                entry_exits = self._calculate_simple_returns(result)
                
                # 期間情報を追加
                period_result = {
                    "period_index": i,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days_count": len(period_data),
                    "entry_signals": int(entry_signals),
                    "exit_signals": int(exit_signals),
                    "trades": entry_exits,
                    "strategy_params": strategy_params or {}
                }
                
                period_results.append(period_result)
                total_days += len(period_data)
                
                logger.debug(f"期間 {i+1} 完了: エントリー{period_result['entry_signals']}回, エグジット{period_result['exit_signals']}回")
                
            except Exception as e:
                logger.error(f"期間 {i+1} のバックテスト中にエラー: {e}")
                period_results.append({
                    "period_index": i,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "error": str(e)
                })
        
        # 結果をまとめる
        successful_periods = [r for r in period_results if "error" not in r]
        total_entries = sum(r.get("entry_signals", 0) for r in successful_periods)
        total_exits = sum(r.get("exit_signals", 0) for r in successful_periods)
        total_trades = sum(len(r.get("trades", [])) for r in successful_periods)
        
        # 収益率の計算
        all_returns = []
        for period in successful_periods:
            all_returns.extend([trade["return_pct"] for trade in period.get("trades", [])])
        
        avg_return = np.mean(all_returns) if all_returns else 0.0
        win_rate = len([r for r in all_returns if r > 0]) / len(all_returns) if all_returns else 0.0
        
        result_summary = {
            "strategy_name": strategy_class.__name__,
            "trend_type": trend_type,
            "test_timestamp": datetime.now().isoformat(),
            "parameters": {
                "strategy_params": strategy_params or {},
                "min_period_length": min_period_length,
                "min_confidence": min_confidence
            },
            "summary": {
                "periods_found": len(trend_periods),
                "periods_tested": len(successful_periods),
                "total_days": total_days,
                "total_entries": total_entries,
                "total_exits": total_exits,
                "total_trades": total_trades,
                "average_return_pct": round(avg_return * 100, 2),
                "win_rate": round(win_rate * 100, 2),
                "success_rate": len(successful_periods) / len(trend_periods) if trend_periods else 0
            },
            "period_results": period_results
        }
        
        # 結果をクラス変数に保存
        result_key = f"{strategy_class.__name__}_{trend_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backtest_results[result_key] = result_summary
        
        logger.info(f"バックテスト完了: {len(successful_periods)}/{len(trend_periods)} 期間で成功, 合計{total_entries}回のエントリー")
        
        return result_summary
    
    def _calculate_simple_returns(self, backtest_result: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        バックテスト結果から簡単な収益計算を行う
        
        Parameters:
            backtest_result (pd.DataFrame): バックテスト結果
            
        Returns:
            List[Dict]: 取引詳細のリスト
        """
        trades = []
        entry_price = None
        entry_date = None
        
        for idx, row in backtest_result.iterrows():
            if row['Entry_Signal'] == 1:
                entry_price = row[self.price_column]
                entry_date = idx
            elif row['Exit_Signal'] == -1 and entry_price is not None:
                exit_price = row[self.price_column]
                exit_date = idx
                
                # 収益率の計算
                return_pct = (exit_price - entry_price) / entry_price
                days_held = (exit_date - entry_date).days
                
                trade = {
                    "entry_date": entry_date.isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "return_pct": float(return_pct),
                    "days_held": int(days_held)
                }
                trades.append(trade)
                
                entry_price = None
                entry_date = None
        
        return trades
    
    def compare_trend_performance(self, 
                                strategy_class: Type[BaseStrategy],
                                strategy_params: Optional[Dict[str, Any]] = None,
                                trend_types: List[str] = ["uptrend", "downtrend", "range-bound"],
                                min_period_length: int = 10,
                                min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        複数のトレンド環境での戦略パフォーマンスを比較
        
        Parameters:
            strategy_class (Type[BaseStrategy]): 戦略クラス
            strategy_params (Dict, optional): 戦略パラメータ
            trend_types (List[str]): 比較するトレンドタイプのリスト
            min_period_length (int): 最小期間長
            min_confidence (float): 最小信頼度
            
        Returns:
            Dict: 比較結果
        """
        logger.info(f"トレンド環境別パフォーマンス比較を開始: {strategy_class.__name__}")
        
        comparison_results = {}
        
        for trend_type in trend_types:
            try:
                result = self.run_strategy_on_trend_periods(
                    strategy_class, strategy_params, trend_type, 
                    min_period_length, min_confidence
                )
                comparison_results[trend_type] = result
                
            except Exception as e:
                logger.error(f"{trend_type} の分析中にエラー: {e}")
                comparison_results[trend_type] = {"error": str(e)}
        
        # 比較サマリーの作成
        comparison_summary = {
            "strategy_name": strategy_class.__name__,
            "comparison_timestamp": datetime.now().isoformat(),
            "parameters": {
                "strategy_params": strategy_params or {},
                "min_period_length": min_period_length,
                "min_confidence": min_confidence
            },
            "trend_results": comparison_results,
            "summary": self._create_comparison_summary(comparison_results)
        }
        
        logger.info("トレンド環境別パフォーマンス比較完了")
        return comparison_summary
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """比較結果のサマリーを作成"""
        summary = {}
        
        for trend_type, result in results.items():
            if "error" not in result and "summary" in result:
                s = result["summary"]
                summary[trend_type] = {
                    "periods_tested": s["periods_tested"],
                    "total_days": s["total_days"],
                    "total_trades": s["total_trades"],
                    "average_return_pct": s["average_return_pct"],
                    "win_rate": s["win_rate"],
                    "trade_frequency": round(s["total_trades"] / s["total_days"], 4) if s["total_days"] > 0 else 0
                }
            else:
                summary[trend_type] = {"error": result.get("error", "Unknown error")}
        
        return summary
    
    def save_results(self, 
                    results: Dict[str, Any], 
                    filename: Optional[str] = None,
                    output_dir: str = "logs") -> str:
        """
        バックテスト結果をJSONファイルに保存
        
        Parameters:
            results (Dict): 保存する結果
            filename (str, optional): ファイル名（Noneの場合は自動生成）
            output_dir (str): 出力ディレクトリ
            
        Returns:
            str: 保存されたファイルパス
        """
        # ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # ファイル名の生成
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = results.get("strategy_name", "unknown")
            trend_type = results.get("trend_type", "multi")
            filename = f"trend_backtest_{strategy_name}_{trend_type}_{timestamp}.json"
        
        # ファイルパスの構築
        filepath = os.path.join(output_dir, filename)
        
        try:
            # JSON保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            logger.info(f"バックテスト結果を保存しました: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"結果保存中にエラーが発生しました: {e}")
            raise
    
    def _json_serializer(self, obj):
        """JSON保存時のカスタムシリアライザー"""
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        保存されたバックテスト結果を読み込み
        
        Parameters:
            filepath (str): 読み込み元ファイルパス
            
        Returns:
            Dict: 読み込まれた結果
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.info(f"バックテスト結果を読み込みました: {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"結果読み込み中にエラーが発生しました: {e}")
            raise

# ユーティリティ関数
def run_trend_limited_backtest(strategy_class: Type[BaseStrategy],
                              stock_data: pd.DataFrame,
                              labeled_data: Optional[pd.DataFrame] = None,
                              strategy_params: Optional[Dict[str, Any]] = None,
                              trend_type: str = "uptrend",
                              min_period_length: int = 10,
                              min_confidence: float = 0.7,
                              save_results: bool = True,
                              output_dir: str = "logs") -> Dict[str, Any]:
    """
    トレンド期間限定バックテストの実行ユーティリティ関数
    
    Parameters:
        strategy_class (Type[BaseStrategy]): 戦略クラス
        stock_data (pd.DataFrame): 株価データ
        labeled_data (pd.DataFrame, optional): ラベリング済みデータ
        strategy_params (Dict, optional): 戦略パラメータ
        trend_type (str): 対象のトレンドタイプ
        min_period_length (int): 最小期間長
        min_confidence (float): 最小信頼度
        save_results (bool): 結果を保存するかどうか
        output_dir (str): 出力ディレクトリ
        
    Returns:
        Dict: バックテスト結果
    """
    backtester = TrendLimitedBacktester(stock_data, labeled_data)
    
    result = backtester.run_strategy_on_trend_periods(
        strategy_class, strategy_params, trend_type, 
        min_period_length, min_confidence
    )
    
    if save_results:
        backtester.save_results(result, output_dir=output_dir)
    
    return result

# テスト用コード
if __name__ == "__main__":
    try:
        # データ取得とテスト
        from data_fetcher import get_parameters_and_data
        from strategies.VWAP_Bounce import VWAPBounceStrategy
        
        # データ取得
        ticker, start_date, end_date, stock_data, _ = get_parameters_and_data()
        
        # ラベリングデータの読み込み
        labeled_data_path = "output/test_labeled_data.csv"
        if os.path.exists(labeled_data_path):
            labeled_data = pd.read_csv(labeled_data_path, index_col=0, parse_dates=True)
            print(f"ラベリングデータを読み込みました: {len(labeled_data)}行")
        else:
            labeled_data = None
            print("ラベリングデータが見つかりません。自動生成します。")
        
        # トレンド期間限定バックテストの実行
        backtester = TrendLimitedBacktester(stock_data, labeled_data)
        
        # 上昇トレンド期間でのVWAP反発戦略テスト
        uptrend_result = backtester.run_strategy_on_trend_periods(
            VWAPBounceStrategy,
            strategy_params={"vwap_lower_threshold": 0.99, "vwap_upper_threshold": 1.02},
            trend_type="uptrend",
            min_period_length=10,
            min_confidence=0.7
        )
        
        print(f"\n=== 上昇トレンド期間でのテスト結果 ===")
        print(f"テスト期間数: {uptrend_result['summary']['periods_tested']}")
        print(f"合計日数: {uptrend_result['summary']['total_days']}")
        print(f"エントリー回数: {uptrend_result['summary']['total_entries']}")
        print(f"平均収益率: {uptrend_result['summary']['average_return_pct']}%")
        
        # トレンド環境別比較
        comparison_result = backtester.compare_trend_performance(
            VWAPBounceStrategy,
            strategy_params={"vwap_lower_threshold": 0.99, "vwap_upper_threshold": 1.02},
            trend_types=["uptrend", "downtrend", "range-bound"]
        )
        
        print(f"\n=== トレンド環境別比較結果 ===")
        for trend_type, summary in comparison_result["summary"].items():
            if "error" not in summary:
                print(f"{trend_type}: {summary['periods_tested']}期間, {summary['total_trades']}取引, 平均収益{summary['average_return_pct']}%")
        
        # 結果の保存
        backtester.save_results(comparison_result, output_dir="logs")
        
        print("\nテスト完了")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
