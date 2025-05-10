"""
パラメータ最適化のための基本クラスを提供します。
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Callable, Any, Tuple, Optional
import itertools
import time
import os
from tqdm import tqdm  # 進捗バー表示用
from strategies.base_strategy import BaseStrategy
from metrics.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from config.logger_config import setup_logger
from joblib import Parallel, delayed
from datetime import datetime

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\optimization.log")

class ParameterOptimizer:
    def __init__(self, 
                 data: pd.DataFrame, 
                 strategy_class: type,
                 param_grid: Dict[str, List[Any]],
                 objective_function: Callable = None,
                 cv_splits=None,
                 output_dir: str = "backtest_results"):
        """
        パラメータ最適化クラスの初期化。
        
        Parameters:
            data (pd.DataFrame): 株価データ
            strategy_class (type): 最適化対象の戦略クラス
            param_grid (Dict[str, List[Any]]): 探索するパラメータとその値の範囲
            objective_function (Callable, optional): 最適化の目的関数
            cv_splits (list, optional): 交差検証用のデータ分割情報
            output_dir (str): 結果保存先ディレクトリ
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.output_dir = output_dir
        
        # 出力ディレクトリの確認と作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # デフォルトの目的関数はシャープレシオ
        if objective_function is None:
            self.objective_function = self._default_objective
        else:
            self.objective_function = objective_function
            
        self.results = []
        
        # 最適化対象の戦略名を記録
        self.strategy_name = strategy_class.__name__
        logger.info(f"戦略 '{self.strategy_name}' の最適化を初期化しました")
        logger.info(f"パラメータグリッド: {param_grid}")
        
        # 組み合わせ総数を計算
        self.total_combinations = 1
        for values in param_grid.values():
            self.total_combinations *= len(values)
        logger.info(f"パラメータの組み合わせ総数: {self.total_combinations}")
    
    def _default_objective(self, trade_results: Dict) -> float:
        """
        デフォルトの目的関数（シャープレシオを最大化）
        
        Parameters:
            trade_results (Dict): バックテスト結果
            
        Returns:
            float: 評価スコア（高いほど良い）
        """
        # 必要なデータがあるか確認
        pnl_summary = trade_results.get("損益推移", pd.DataFrame())
        if pnl_summary.empty:
            logger.warning("損益推移データが見つかりません。最低スコアを返します。")
            return -np.inf
        
        # 必要なカラムの存在確認
        if "日次損益" not in pnl_summary.columns:
            logger.warning("「日次損益」カラムが見つかりません。最低スコアを返します。")
            return -np.inf
            
        if "累積損益" not in pnl_summary.columns:
            logger.warning("「累積損益」カラムが見つかりません。最低スコアを返します。")
            return -np.inf
            
        daily_returns = pnl_summary["日次損益"]
        sharpe = calculate_sharpe_ratio(daily_returns)
        max_dd = calculate_max_drawdown(pnl_summary["累積損益"])
        
        # シャープレシオが大きく、ドローダウンが小さいパラメータを選ぶ
        # max_ddは100%スケールなので0.01を掛けて調整
        return sharpe - (max_dd * 0.01)
    
    def grid_search(self) -> pd.DataFrame:
        """
        グリッドサーチによるパラメータ最適化を実行
        
        Returns:
            pd.DataFrame: 最適化結果
        """
        param_keys = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        start_time = time.time()
        logger.info(f"グリッドサーチを開始: {self.strategy_name}")
        
        # パラメータの全組み合わせを生成
        combinations = list(itertools.product(*param_values))
        
        # 進捗バー付きでパラメータを探索
        for i, params_combination in enumerate(tqdm(combinations, desc=f"{self.strategy_name} 最適化")):
            params = dict(zip(param_keys, params_combination))
            
            logger.info(f"評価中のパラメータ ({i+1}/{self.total_combinations}): {params}")
            
            try:
                score = self._evaluate_params(params)
                
                self.results.append({
                    **params,
                    "score": score
                })
                
                logger.info(f"評価結果: {score}")
                
            except Exception as e:
                logger.error(f"パラメータ {params} の評価中にエラーが発生: {str(e)}")
                # エラー情報も結果に追加
                self.results.append({
                    **params,
                    "score": -np.inf,
                    "error": str(e)
                })
        
        elapsed_time = time.time() - start_time
        logger.info(f"グリッドサーチ完了: 処理時間 {elapsed_time:.2f}秒")
        
        # 結果をDataFrameに変換
        if not self.results:
            logger.warning("有効な結果がありません。空のDataFrameを返します。")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(self.results)
        sorted_results = results_df.sort_values("score", ascending=False)
        
        # 最適なパラメータを記録
        if not sorted_results.empty:
            best_params = sorted_results.iloc[0].to_dict()
            best_score = best_params.pop("score", None)
            if "error" in best_params:
                best_params.pop("error")
            logger.info(f"最適なパラメータ: {best_params}, スコア: {best_score}")
        
        return sorted_results
    
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """
        特定のパラメータセットを評価する
        
        Parameters:
            params (Dict[str, Any]): 評価するパラメータの辞書
            
        Returns:
            float: 評価スコア
        """
        if self.cv_splits is None:
            # 交差検証なし - 全データでバックテスト
            return self._run_backtest_with_params(self.data, params)
        else:
            # 交差検証あり - 各分割でバックテストして平均スコアを返す
            scores = []
            for i, (train_idx, test_idx) in enumerate(self.cv_splits):
                logger.debug(f"交差検証 {i+1}/{len(self.cv_splits)} を実行中")
                test_data = self.data.iloc[test_idx]
                try:
                    score = self._run_backtest_with_params(test_data, params)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"交差検証 {i+1} でエラー: {str(e)}")
                    # エラーの場合はペナルティスコアを追加
                    scores.append(-np.inf)
            
            # 有限のスコアのみ平均を計算
            valid_scores = [s for s in scores if np.isfinite(s)]
            if not valid_scores:
                return -np.inf
            return np.mean(valid_scores)
    
    def _run_backtest_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        指定されたパラメータでバックテストを実行
        
        Parameters:
            data (pd.DataFrame): バックテスト用データ
            params (Dict[str, Any]): 戦略パラメータ
            
        Returns:
            float: 評価スコア
        """
        try:
            # 戦略インスタンスを作成
            strategy = self.strategy_class(data, params=params)
            
            # バックテスト実行
            result_data = strategy.backtest()
            
            # トレードシミュレーション実行
            from trade_simulation import simulate_trades
            trade_results = simulate_trades(result_data, "最適化中")
            
            # 目的関数でスコア計算
            score = self.objective_function(trade_results)
            
            return score
            
        except Exception as e:
            logger.error(f"バックテスト実行中にエラー: {str(e)}")
            raise
    
    def save_results(self, filename: Optional[str] = None, format: str = "excel") -> str:
        """
        最適化結果を保存する
        
        Parameters:
            filename (str, optional): 保存するファイル名 (拡張子除く)
            format (str): 保存形式 ('excel', 'csv', 'pickle')
            
        Returns:
            str: 保存したファイルのパス
        """
        if not self.results:
            logger.warning("保存する結果がありません")
            return ""
            
        results_df = pd.DataFrame(self.results).sort_values("score", ascending=False)
        
        # ファイル名が指定されていない場合は自動生成
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.strategy_name}_optimization_{timestamp}"
        
        # 保存先のパスを構築
        if format.lower() == "excel":
            filepath = os.path.join(self.output_dir, f"{filename}.xlsx")
            results_df.to_excel(filepath, index=False)
        elif format.lower() == "csv":
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
            results_df.to_csv(filepath, index=False)
        elif format.lower() == "pickle":
            filepath = os.path.join(self.output_dir, f"{filename}.pkl")
            results_df.to_pickle(filepath)
        else:
            logger.error(f"未対応の保存形式: {format}")
            return ""
            
        logger.info(f"最適化結果を保存しました: {filepath}")
        return filepath

class ParallelParameterOptimizer(ParameterOptimizer):
    """
    パラメータの最適化を並列処理で実行するクラス
    """
    
    def __init__(self, data, strategy_class, param_grid, objective_function, cv_splits=None, output_dir="backtest_results", n_jobs=-1):
        """
        初期化関数
        
        Parameters:
            data (pd.DataFrame): 最適化に使用するデータ
            strategy_class (class): 戦略クラス
            param_grid (dict): 最適化するパラメータの格子
            objective_function (function): 最適化の目的関数
            cv_splits (list): クロスバリデーションの分割
            output_dir (str): 結果を保存するディレクトリ
            n_jobs (int): 並列処理で使用するジョブ数（-1で全コア使用）
        """
        super().__init__(data, strategy_class, param_grid, objective_function, cv_splits, output_dir)
        self.n_jobs = n_jobs
        
    def parallel_grid_search(self):
        """
        グリッドサーチを並列で実行
        
        Returns:
            pd.DataFrame: 最適化結果
        """
        self.logger.info(f"並列グリッドサーチを開始します。パラメータ組み合わせ数: {len(self.param_combinations)}")
        start_time = time.time()
        
        # パラメータごとに評価を並列実行
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_params)(params) for params in self.param_combinations
        )
        
        # 結果をデータフレームに変換
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # スコアでソート（降順）
            results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)
            
        self.logger.info(f"並列グリッドサーチが完了しました。実行時間: {time.time() - start_time:.2f}秒")
        
        self.results = results_df
        return results_df