"""
ウォークフォワードテストの実行エンジン

シナリオに基づいて戦略のバックテストを実行し、結果を集計します。
"""

import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
import traceback

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import data_fetcher
import data_processor
from src.config.logger_config import setup_logger
from src.analysis.walkforward_scenarios import WalkforwardScenarios

# 戦略のインポート（オプション）
strategy_classes = {}
try:
    from src.strategies.vwap_breakout_strategy import VWAPBreakoutStrategy
    strategy_classes["VWAPBreakoutStrategy"] = VWAPBreakoutStrategy
except ImportError:
    pass

try:
    from src.strategies.vwap_bounce_strategy import VWAPBounceStrategy
    strategy_classes["VWAPBounceStrategy"] = VWAPBounceStrategy
except ImportError:
    pass
    
try:
    from src.strategies.breakout_strategy import BreakoutStrategy
    strategy_classes["BreakoutStrategy"] = BreakoutStrategy
except ImportError:
    pass
    
try:
    from src.strategies.gc_strategy import GCStrategy
    strategy_classes["GCStrategy"] = GCStrategy
except ImportError:
    pass
    
try:
    from src.strategies.momentum_investing_strategy import MomentumInvestingStrategy
    strategy_classes["MomentumInvestingStrategy"] = MomentumInvestingStrategy
except ImportError:
    pass

class WalkforwardExecutor:
    """ウォークフォワードテストの実行を管理するクラス"""
    
    def __init__(self, scenarios: WalkforwardScenarios):
        """
        初期化
        
        Args:
            scenarios: シナリオ管理オブジェクト
        """
        self.logger = setup_logger(__name__)
        self.scenarios = scenarios
        
        # 戦略マッピング（利用可能な戦略のみ）
        self.strategy_classes = strategy_classes.copy()
        
        self.results = []
        
    def execute_walkforward_test(self, symbol: str, strategy_name: str, 
                                scenario_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """単一シンボル・戦略のウォークフォワードテストを実行"""
        
        if strategy_name not in self.strategy_classes:
            self.logger.error(f"サポートされていない戦略: {strategy_name}")
            return []
            
        strategy_class = self.strategy_classes[strategy_name]
        window_results = []
        
        try:
            full_data = scenario_data["data"]
            windows = scenario_data["windows"]
            
            self.logger.info(f"ウォークフォワード開始: {symbol} - {strategy_name} ({len(windows)}ウィンドウ)")
            
            for i, window in enumerate(windows):
                try:
                    result = self._execute_single_window(
                        symbol, strategy_class, full_data, window, i+1
                    )
                    if result:
                        result.update({
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "period_name": scenario_data["period_name"],
                            "market_condition": scenario_data["market_condition"]
                        })
                        window_results.append(result)
                        
                except Exception as e:
                    self.logger.error(f"ウィンドウ{i+1}実行エラー ({symbol}-{strategy_name}): {e}")
                    continue
                    
            self.logger.info(f"ウォークフォワード完了: {symbol} - {strategy_name} ({len(window_results)}件)")
            return window_results
            
        except Exception as e:
            self.logger.error(f"ウォークフォワードテスト実行エラー: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def _execute_single_window(self, symbol: str, strategy_class: type, 
                              full_data: pd.DataFrame, window: Dict[str, str], 
                              window_number: int) -> Optional[Dict[str, Any]]:
        """単一ウィンドウのバックテストを実行"""
        
        try:
            # 学習期間のデータ
            training_data = self._filter_data_by_period(
                full_data, window["training_start"], window["training_end"]
            )
            
            # テスト期間のデータ  
            testing_data = self._filter_data_by_period(
                full_data, window["testing_start"], window["testing_end"]
            )
            
            if len(training_data) < 20 or len(testing_data) < 5:
                self.logger.warning(f"データ不足 - 学習:{len(training_data)}, テスト:{len(testing_data)}")
                return None
            
            # 戦略インスタンス作成（学習期間でパラメータ調整可能）
            strategy = strategy_class()
            
            # テスト期間でバックテスト実行
            # データ処理（既存の関数を使用）
            try:
                processed_data = data_processor.preprocess_data(testing_data)
            except:
                # 処理に失敗した場合は元のデータを使用
                processed_data = testing_data
            
            if processed_data is None or len(processed_data) == 0:
                self.logger.warning(f"データ処理後に空のデータ: {symbol} ウィンドウ{window_number}")
                return None
                
            # バックテスト実行
            backtest_result = strategy.backtest(processed_data)
            
            if backtest_result is None or len(backtest_result) == 0:
                self.logger.warning(f"バックテスト結果が空: {symbol} ウィンドウ{window_number}")
                return None
            
            # パフォーマンス計算
            performance_metrics = self._calculate_performance_metrics(
                backtest_result, window["testing_start"], window["testing_end"]
            )
            
            result = {
                "window_number": window_number,
                "training_start": window["training_start"],
                "training_end": window["training_end"],
                "testing_start": window["testing_start"], 
                "testing_end": window["testing_end"],
                "training_samples": len(training_data),
                "testing_samples": len(testing_data),
                "backtest_samples": len(backtest_result),
                **performance_metrics
            }
            
            self.logger.debug(f"ウィンドウ{window_number}完了: {symbol} - {performance_metrics.get('total_return', 'N/A')}%")
            return result
            
        except Exception as e:
            self.logger.error(f"単一ウィンドウ実行エラー: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _filter_data_by_period(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """期間でデータをフィルタリング"""
        try:
            # インデックスがDatetimeの場合
            if isinstance(data.index, pd.DatetimeIndex):
                mask = (data.index >= start_date) & (data.index <= end_date)
                return data.loc[mask]
            
            # Date列がある場合
            elif 'Date' in data.columns:
                data_copy = data.copy()
                data_copy['Date'] = pd.to_datetime(data_copy['Date'])
                mask = (data_copy['Date'] >= start_date) & (data_copy['Date'] <= end_date)
                return data_copy.loc[mask]
            
            else:
                self.logger.warning("データにDate情報が見つかりません")
                return data
                
        except Exception as e:
            self.logger.error(f"データフィルタリングエラー: {e}")
            return pd.DataFrame()
    
    def _calculate_performance_metrics(self, backtest_result: pd.DataFrame, 
                                     start_date: str, end_date: str) -> Dict[str, Any]:
        """パフォーマンス指標を計算"""
        
        try:
            metrics = {}
            
            # 基本統計
            metrics["period_start"] = start_date
            metrics["period_end"] = end_date
            metrics["total_trades"] = len(backtest_result)
            
            # エントリー・エグジットシグナル数
            if 'Entry_Signal' in backtest_result.columns:
                entry_signals = backtest_result['Entry_Signal'].sum()
                metrics["entry_signals"] = int(entry_signals) if pd.notna(entry_signals) else 0
            
            if 'Exit_Signal' in backtest_result.columns:
                exit_signals = backtest_result['Exit_Signal'].sum()
                metrics["exit_signals"] = int(exit_signals) if pd.notna(exit_signals) else 0
            
            # リターン計算（簡易版）
            if 'Close' in backtest_result.columns and len(backtest_result) > 1:
                start_price = backtest_result['Close'].iloc[0]
                end_price = backtest_result['Close'].iloc[-1]
                total_return = ((end_price - start_price) / start_price) * 100
                metrics["total_return"] = round(float(total_return), 4)
                
                # 日次リターン
                daily_returns = backtest_result['Close'].pct_change().dropna()
                if len(daily_returns) > 0:
                    metrics["volatility"] = round(float(daily_returns.std() * 100), 4)
                    metrics["max_drawdown"] = self._calculate_max_drawdown(backtest_result['Close'])
            
            # シャープレシオ（簡易版）
            if "total_return" in metrics and "volatility" in metrics and metrics["volatility"] > 0:
                sharpe_ratio = metrics["total_return"] / metrics["volatility"]
                metrics["sharpe_ratio"] = round(float(sharpe_ratio), 4)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"パフォーマンス計算エラー: {e}")
            return {"error": str(e)}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """最大ドローダウンを計算"""
        try:
            cumulative = (1 + prices.pct_change()).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            return round(float(max_drawdown * 100), 4)
        except:
            return 0.0
    
    def run_all_scenarios(self, max_scenarios: Optional[int] = None) -> List[Dict[str, Any]]:
        """全シナリオでウォークフォワードテストを実行"""
        
        all_scenarios = self.scenarios.get_test_scenarios()
        
        if max_scenarios:
            all_scenarios = all_scenarios[:max_scenarios]
            self.logger.info(f"シナリオを{max_scenarios}件に制限しました")
        
        self.logger.info(f"ウォークフォワードテスト開始: {len(all_scenarios)}シナリオ")
        
        total_results = []
        processed_count = 0
        
        for scenario in all_scenarios:
            try:
                # シナリオデータ準備
                scenario_data = self.scenarios.prepare_scenario_data(scenario)
                
                if scenario_data is None:
                    self.logger.warning(f"シナリオデータ準備失敗: {scenario['symbol']} - {scenario['period_name']}")
                    continue
                
                # 各戦略で実行
                for strategy_name in scenario["strategies"]:
                    try:
                        results = self.execute_walkforward_test(
                            scenario["symbol"], strategy_name, scenario_data
                        )
                        total_results.extend(results)
                        
                    except Exception as e:
                        self.logger.error(f"戦略実行エラー ({strategy_name}): {e}")
                        continue
                
                processed_count += 1
                if processed_count % 5 == 0:
                    self.logger.info(f"進捗: {processed_count}/{len(all_scenarios)} シナリオ完了")
                    
            except Exception as e:
                self.logger.error(f"シナリオ実行エラー: {e}")
                continue
        
        self.logger.info(f"ウォークフォワードテスト完了: {len(total_results)}件の結果")
        self.results = total_results
        return total_results
    
    def get_results_summary(self) -> Dict[str, Any]:
        """結果の概要を取得"""
        
        if not self.results:
            return {"message": "実行結果がありません"}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            "total_results": len(self.results),
            "unique_symbols": len(df['symbol'].unique()) if 'symbol' in df.columns else 0,
            "unique_strategies": len(df['strategy'].unique()) if 'strategy' in df.columns else 0,
            "unique_periods": len(df['period_name'].unique()) if 'period_name' in df.columns else 0,
            "success_rate": len(df[df.get('total_return', pd.Series(dtype=float)) > 0]) / len(df) if len(df) > 0 else 0
        }
        
        # 戦略別パフォーマンス
        if 'strategy' in df.columns and 'total_return' in df.columns:
            strategy_performance = df.groupby('strategy')['total_return'].agg(['mean', 'std', 'count']).round(4)
            summary["strategy_performance"] = strategy_performance.to_dict('index')
        
        # 市場状況別パフォーマンス
        if 'market_condition' in df.columns and 'total_return' in df.columns:
            market_performance = df.groupby('market_condition')['total_return'].agg(['mean', 'std', 'count']).round(4)
            summary["market_performance"] = market_performance.to_dict('index')
        
        return summary

if __name__ == "__main__":
    # テスト実行
    print("=== ウォークフォワード実行エンジンテスト ===")
    
    scenarios = WalkforwardScenarios()
    executor = WalkforwardExecutor(scenarios)
    
    # 小規模テスト（最初の2シナリオのみ）
    print("小規模テストを実行中...")
    results = executor.run_all_scenarios(max_scenarios=2)
    
    print(f"\n実行結果: {len(results)}件")
    
    if results:
        summary = executor.get_results_summary()
        print(f"成功率: {summary.get('success_rate', 0):.2%}")
        print(f"対象戦略数: {summary.get('unique_strategies', 0)}")
        print(f"対象シンボル数: {summary.get('unique_symbols', 0)}")
