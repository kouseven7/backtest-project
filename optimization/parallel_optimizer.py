"""
並列処理によるパラメータ最適化を提供します。
"""
import pandas as pd
import numpy as np
import time
import os
from typing import Dict, List, Callable, Any
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools

from optimization.parameter_optimizer import ParameterOptimizer, logger

class ParallelParameterOptimizer(ParameterOptimizer):
    """
    並列処理を利用したパラメータ最適化クラス
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 strategy_class: type,
                 param_grid: Dict[str, List[Any]],
                 objective_function = None,
                 cv_splits=None,
                 output_dir: str = "backtest_results",
                 n_jobs: int = -1,
                 verbose: int = 10,
                 index_data = None):
        """
        並列パラメータ最適化クラスの初期化。
        
        Parameters:
            data (pd.DataFrame): 株価データ
            strategy_class (type): 最適化対象の戦略クラス
            param_grid (Dict[str, List[Any]]): 探索するパラメータとその値の範囲
            objective_function (Callable, optional): 最適化の目的関数
            cv_splits (list, optional): 交差検証用のデータ分割情報
            output_dir (str): 結果保存先ディレクトリ
            n_jobs (int): 並列ジョブ数 (-1は全コア使用)
            verbose (int): 詳細表示レベル (0=なし、10=進捗バー表示)
            index_data (pd.DataFrame, optional): 市場インデックスデータ
        """
        super().__init__(data, strategy_class, param_grid, objective_function, cv_splits, output_dir)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.index_data = index_data  # 追加: VWAPBreakoutStrategy用
        logger.info(f"並列処理を初期化: ジョブ数={n_jobs}")
        
    def parallel_grid_search(self) -> pd.DataFrame:
        """
        並列処理によるグリッドサーチを実行
        
        Returns:
            pd.DataFrame: 最適化結果
        """
        param_keys = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # パラメータの全組み合わせを生成
        combinations = list(itertools.product(*param_values))
        params_list = [dict(zip(param_keys, combo)) for combo in combinations]
        
        start_time = time.time()
        logger.info(f"並列グリッドサーチを開始: {self.strategy_name}, パラメータセット数: {len(params_list)}")
        
        # 並列処理で評価を実行
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._parallel_evaluate_params)(params) for params in params_list
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"並列グリッドサーチ完了: 処理時間 {elapsed_time:.2f}秒")
        
        # 結果をリストに格納
        self.results = []
        for params, result in zip(params_list, results):
            if isinstance(result, tuple) and len(result) == 2:
                score, error = result
                result_dict = {**params, "score": score}
                if error:
                    result_dict["error"] = error
                self.results.append(result_dict)
            else:
                # 正常に処理できた場合
                self.results.append({**params, "score": result})
        
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
    
    def _parallel_evaluate_params(self, params: Dict[str, Any]):
        """
        並列処理用のパラメータ評価関数
        
        Parameters:
            params (Dict[str, Any]): 評価するパラメータの辞書
            
        Returns:
            float or tuple: 評価スコアまたは(スコア, エラーメッセージ)のタプル
        """
        try:
            if self.cv_splits is None:
                # 交差検証なし - 全データでバックテスト
                return self._run_backtest_with_params(self.data, params)
            else:
                # 交差検証あり - 各分割でバックテストして平均スコアを返す
                scores = []
                for train_idx, test_idx in self.cv_splits:
                    test_data = self.data.iloc[test_idx]
                    try:
                        score = self._run_backtest_with_params(test_data, params)
                        scores.append(score)
                    except Exception:
                        # エラーの場合はスキップ
                        continue
                
                if not scores:
                    return -np.inf, "全ての交差検証でエラーが発生しました"
                return np.mean(scores), None
                
        except Exception as e:
            # エラーが発生した場合は最低スコアとエラーメッセージを返す
            return -np.inf, str(e)
    
    def _run_backtest_with_params(self, data, params):
        """
        指定されたパラメータでバックテストを実行（VWAPBreakoutStrategy用にindex_dataを渡す）
        """
        try:
            # index_dataが必要な戦略の場合は渡す
            if self.index_data is not None:
                strategy = self.strategy_class(data, self.index_data, params=params)
            else:
                strategy = self.strategy_class(data, params=params)
            result_data = strategy.backtest()
            from trade_simulation import simulate_trades
            trade_results = simulate_trades(result_data, "最適化中")
            score = self.objective_function(trade_results)
            return score
        except Exception as e:
            logger.error(f"バックテスト実行中にエラー: {str(e)}")
            raise