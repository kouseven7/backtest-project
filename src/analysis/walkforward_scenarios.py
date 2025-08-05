"""
ウォークフォワードテストのシナリオ管理モジュール

異なる市場環境でのパフォーマンス検証を行うためのシナリオを定義・管理します。
"""

import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import data_fetcher
from src.config.logger_config import setup_logger

class WalkforwardScenarios:
    """ウォークフォワードテストのシナリオを管理するクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = setup_logger(__name__)
        
        if config_path is None:
            config_path = str(Path(__file__).parent / "walkforward_config.json")
            
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"設定ファイルの読み込みに失敗: {e}")
            raise
            
    def get_test_scenarios(self) -> List[Dict[str, Any]]:
        """テストシナリオのリストを取得"""
        scenarios = []
        test_config = self.config["test_scenarios"]
        
        for period in test_config["periods"]:
            for symbol in test_config["symbols"]:
                scenario = {
                    "symbol": symbol,
                    "period_name": period["name"],
                    "start_date": period["start"],
                    "end_date": period["end"],
                    "market_condition": period["market_condition"],
                    "strategies": self.config["strategies"]
                }
                scenarios.append(scenario)
                
        self.logger.info(f"生成されたシナリオ数: {len(scenarios)}")
        return scenarios
    
    def validate_scenario_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """シナリオのデータが取得可能かチェック"""
        try:
            # データ取得のテスト（既存のget_parameters_and_data関数を使用）
            result = data_fetcher.get_parameters_and_data(
                ticker=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # 戻り値は(ticker, start_date, end_date, data, parameters)の形式
            if len(result) >= 4:
                data = result[3]  # 4番目の要素がデータフレーム
            else:
                data = None
            
            if data is None or len(data) < self.config["walkforward_config"]["min_training_samples"]:
                self.logger.warning(f"データ不足: {symbol} {start_date}-{end_date}")
                return False
                
            self.logger.debug(f"データ検証OK: {symbol} {start_date}-{end_date} ({len(data)}件)")
            return True
            
        except Exception as e:
            self.logger.error(f"データ検証エラー: {symbol} {start_date}-{end_date}: {e}")
            return False
    
    def get_walkforward_windows(self, start_date: str, end_date: str) -> List[Dict[str, str]]:
        """ウォークフォワード用の学習・テスト期間を生成"""
        wf_config = self.config["walkforward_config"]
        training_months = wf_config["training_window_months"]
        testing_months = wf_config["testing_window_months"] 
        step_months = wf_config["step_size_months"]
        
        windows = []
        current_start = pd.to_datetime(start_date)
        period_end = pd.to_datetime(end_date)
        
        while True:
            # 学習期間の終了日
            training_end = current_start + pd.DateOffset(months=training_months)
            # テスト期間の終了日
            testing_end = training_end + pd.DateOffset(months=testing_months)
            
            # 期間が全体期間を超える場合は終了
            if testing_end > period_end:
                break
                
            window = {
                "training_start": current_start.strftime("%Y-%m-%d"),
                "training_end": training_end.strftime("%Y-%m-%d"),
                "testing_start": training_end.strftime("%Y-%m-%d"),
                "testing_end": testing_end.strftime("%Y-%m-%d")
            }
            windows.append(window)
            
            # 次のウィンドウの開始日
            current_start = current_start + pd.DateOffset(months=step_months)
            
        self.logger.debug(f"ウォークフォワードウィンドウ数: {len(windows)}")
        return windows
    
    def prepare_scenario_data(self, scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """シナリオ用のデータを準備"""
        try:
            symbol = scenario["symbol"]
            start_date = scenario["start_date"]
            end_date = scenario["end_date"]
            
            # データ検証
            if not self.validate_scenario_data(symbol, start_date, end_date):
                return None
                
            # ウォークフォワードウィンドウを生成
            windows = self.get_walkforward_windows(start_date, end_date)
            
            if not windows:
                self.logger.warning(f"ウォークフォワードウィンドウが生成できません: {symbol}")
                return None
                
            # データを取得
            result = data_fetcher.get_parameters_and_data(
                ticker=symbol,
                start_date=start_date, 
                end_date=end_date
            )
            
            # 戻り値からデータフレームを取得
            if len(result) >= 4:
                full_data = result[3]  # 4番目の要素がデータフレーム
            else:
                full_data = None
            
            prepared_scenario = {
                **scenario,
                "windows": windows,
                "data": full_data,
                "data_length": len(full_data) if full_data is not None else 0
            }
            
            self.logger.info(f"シナリオデータ準備完了: {symbol} {scenario['period_name']}")
            return prepared_scenario
            
        except Exception as e:
            self.logger.error(f"シナリオデータ準備エラー: {e}")
            return None
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """シナリオの概要を取得"""
        test_config = self.config["test_scenarios"]
        
        summary = {
            "total_symbols": len(test_config["symbols"]),
            "total_periods": len(test_config["periods"]),
            "total_scenarios": len(test_config["symbols"]) * len(test_config["periods"]),
            "symbols": test_config["symbols"],
            "periods": [
                {
                    "name": p["name"],
                    "duration": p["start"] + " to " + p["end"],
                    "condition": p["market_condition"]
                }
                for p in test_config["periods"]
            ],
            "strategies": self.config["strategies"]
        }
        
        return summary
    
    def filter_scenarios_by_condition(self, market_condition: str) -> List[Dict[str, Any]]:
        """市場状況でシナリオをフィルタリング"""
        all_scenarios = self.get_test_scenarios()
        filtered = [s for s in all_scenarios if s["market_condition"] == market_condition]
        
        self.logger.info(f"市場状況 '{market_condition}' でフィルタリング: {len(filtered)}件")
        return filtered
    
    def get_strategy_test_config(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """特定戦略のテスト設定を取得"""
        if strategy_name not in self.config["strategies"]:
            self.logger.warning(f"戦略が設定に含まれていません: {strategy_name}")
            return None
            
        return {
            "strategy": strategy_name,
            "walkforward_config": self.config["walkforward_config"],
            "output_config": self.config["output_config"]
        }

if __name__ == "__main__":
    # 簡単なテスト実行
    scenarios = WalkforwardScenarios()
    
    print("=== ウォークフォワードシナリオ概要 ===")
    summary = scenarios.get_scenario_summary()
    print(f"シンボル数: {summary['total_symbols']}")
    print(f"期間数: {summary['total_periods']}")
    print(f"総シナリオ数: {summary['total_scenarios']}")
    
    print("\n=== サンプルシナリオ ===")
    test_scenarios = scenarios.get_test_scenarios()[:3]  # 最初の3つを表示
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['symbol']} - {scenario['period_name']} ({scenario['market_condition']})")
    
    print("\n=== 上昇トレンドシナリオ ===")
    uptrend_scenarios = scenarios.filter_scenarios_by_condition("uptrend")
    print(f"上昇トレンドシナリオ数: {len(uptrend_scenarios)}")
