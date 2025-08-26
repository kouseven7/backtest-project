"""
DSSMS Task 1.4: Mock Backtester V2 Updated
デモ用の軽量化されたバックテスター

Author: GitHub Copilot Agent
Created: 2025-08-26
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import time

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from .switch_decision import SwitchDecision, create_mock_switch_decision


class MockDSSMSBacktesterV2Updated:
    """
    モック版バックテスターV2更新版
    デモンストレーション用の軽量実装
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.config = config or {}
        
        # 出力ディレクトリ
        self.output_dir = project_root / "output" / "backtest_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Mock DSSMS Backtester V2 Updated 初期化完了")
    
    def run_comprehensive_backtest(
        self, 
        start_date: str, 
        end_date: str, 
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        包括的バックテスト実行
        モック版：シミュレートされた結果を返す
        """
        start_time = time.time()
        
        self.logger.info(f"包括的バックテスト開始: {start_date} - {end_date}")
        
        try:
            # 期間計算
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            total_days = (end_dt - start_dt).days + 1
            
            if symbols is None:
                symbols = ["7203", "6758", "9984", "9983", "8306"]
            
            # モック市場データ生成
            market_data = self._generate_mock_market_data(start_dt, end_dt, symbols)
            
            # バックテスト実行シミュレーション
            backtest_results = self._simulate_backtest_execution(
                market_data, symbols, total_days
            )
            
            execution_time = time.time() - start_time
            
            # 結果構造化
            results = {
                "backtest_metadata": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": total_days,
                    "symbols_tested": symbols,
                    "execution_time_seconds": execution_time,
                    "mock_execution": True
                },
                "overall_performance": backtest_results["overall"],
                "engine_performance": backtest_results["engines"],
                "target_achievement": backtest_results["targets"],
                "daily_statistics": backtest_results["daily"],
                "switch_analysis": backtest_results["switches"],
                "risk_metrics": backtest_results["risk"]
            }
            
            self.logger.info(f"包括的バックテスト完了: {execution_time:.2f}秒")
            return results
            
        except Exception as e:
            self.logger.error(f"バックテスト実行失敗: {e}")
            return {
                "error": True,
                "error_message": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _generate_mock_market_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        symbols: List[str]
    ) -> pd.DataFrame:
        """モック市場データ生成"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for date in dates:
            for symbol in symbols:
                base_price = 1000 + int(symbol) % 500
                
                # ランダムウォーク価格生成
                price_change = np.random.normal(0, base_price * 0.02)
                price = max(base_price + price_change, base_price * 0.5)
                
                data.append({
                    "date": date,
                    "symbol": symbol,
                    "open": price * 0.998,
                    "high": price * 1.015,
                    "low": price * 0.985,
                    "close": price,
                    "volume": np.random.randint(50000, 200000),
                    "volatility": np.random.uniform(0.1, 0.4)
                })
        
        return pd.DataFrame(data)
    
    def _simulate_backtest_execution(
        self, 
        market_data: pd.DataFrame, 
        symbols: List[str], 
        total_days: int
    ) -> Dict[str, Any]:
        """バックテスト実行シミュレーション"""
        
        # 全体パフォーマンス
        total_attempts = max(1, total_days // 2)  # 2日に1回程度の頻度
        successful_switches = int(total_attempts * np.random.uniform(0.25, 0.45))  # 25-45%成功率
        avg_switches_per_day = successful_switches / total_days if total_days > 0 else 0
        
        overall = {
            "total_switch_attempts": total_attempts,
            "successful_switches": successful_switches,
            "failed_switches": total_attempts - successful_switches,
            "overall_success_rate": successful_switches / total_attempts if total_attempts > 0 else 0,
            "avg_switches_per_day": avg_switches_per_day,
            "avg_execution_time_ms": np.random.uniform(80, 150),
            "total_pnl": np.random.uniform(-5000, 15000),
            "sharpe_ratio": np.random.uniform(0.5, 2.0),
            "max_drawdown": np.random.uniform(0.05, 0.20)
        }
        
        # エンジン別パフォーマンス
        engines = {
            "v2": {
                "attempts": int(total_attempts * 0.5),
                "successes": int(total_attempts * 0.5 * np.random.uniform(0.3, 0.5)),
                "avg_execution_time": np.random.uniform(70, 120),
                "success_rate": 0
            },
            "legacy": {
                "attempts": int(total_attempts * 0.3),
                "successes": int(total_attempts * 0.3 * np.random.uniform(0.2, 0.4)),
                "avg_execution_time": np.random.uniform(90, 160),
                "success_rate": 0
            },
            "hybrid": {
                "attempts": int(total_attempts * 0.2),
                "successes": int(total_attempts * 0.2 * np.random.uniform(0.35, 0.55)),
                "avg_execution_time": np.random.uniform(100, 140),
                "success_rate": 0
            }
        }
        
        # 成功率計算
        for engine_name, engine_data in engines.items():
            if engine_data["attempts"] > 0:
                engine_data["success_rate"] = engine_data["successes"] / engine_data["attempts"]
            else:
                engine_data["success_rate"] = 0
        
        # 目標達成状況
        success_rate_target = 0.30
        daily_switch_target = 1
        
        targets = {
            "success_rate_target": success_rate_target,
            "success_rate_achieved": overall["overall_success_rate"] >= success_rate_target,
            "daily_switch_target": daily_switch_target,
            "daily_switch_achieved": avg_switches_per_day >= daily_switch_target,
            "overall_target_achievement": (
                overall["overall_success_rate"] >= success_rate_target and 
                avg_switches_per_day >= daily_switch_target
            )
        }
        
        # 日次統計
        daily = {
            "avg_daily_attempts": total_attempts / total_days if total_days > 0 else 0,
            "max_daily_switches": max(1, int(avg_switches_per_day * 3)),
            "min_daily_switches": 0,
            "daily_success_variance": np.random.uniform(0.1, 0.3),
            "profitable_days_ratio": np.random.uniform(0.45, 0.75)
        }
        
        # 切替分析
        switches = {
            "total_symbol_changes": successful_switches * np.random.randint(1, 4),
            "avg_switches_per_execution": np.random.uniform(1.2, 2.8),
            "most_traded_symbols": symbols[:3],
            "switch_frequency_distribution": {
                "1_switch": np.random.uniform(0.4, 0.6),
                "2_switches": np.random.uniform(0.25, 0.35),
                "3_switches": np.random.uniform(0.15, 0.25)
            }
        }
        
        # リスク指標
        risk = {
            "value_at_risk_95": np.random.uniform(500, 2000),
            "expected_shortfall": np.random.uniform(800, 3000),
            "volatility_annual": np.random.uniform(0.15, 0.35),
            "correlation_with_market": np.random.uniform(0.3, 0.8),
            "beta": np.random.uniform(0.7, 1.3)
        }
        
        return {
            "overall": overall,
            "engines": engines,
            "targets": targets,
            "daily": daily,
            "switches": switches,
            "risk": risk
        }
    
    def generate_performance_report(self, results: Dict[str, Any]) -> Optional[str]:
        """パフォーマンスレポート生成"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.output_dir / f"backtest_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"パフォーマンスレポート生成: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"レポート生成失敗: {e}")
            return None


# エイリアス（下位互換性のため）
DSSMSBacktesterV2Updated = MockDSSMSBacktesterV2Updated
