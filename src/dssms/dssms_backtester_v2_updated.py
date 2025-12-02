"""
DSSMS Task 1.4: バックテスターV2更新版
銘柄切替メカニズム復旧対応のバックテスト機能

主要機能:
1. Task 1.4統合バックテスト
2. 切替統計追跡
3. エンジン性能分析
4. 診断システム統合
5. 包括的レポート生成

Author: GitHub Copilot Agent
Created: 2025-08-26
Task: 1.4 銘柄切替メカニズム復旧
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import warnings
import json
import time

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# Task 1.4コンポーネント
try:
    from src.dssms.dssms_switch_coordinator_v2 import DSSMSSwitchCoordinatorV2
    from src.dssms.switch_diagnostics import SwitchDiagnostics
    # from src.dssms.dssms_backtester_v2 import DSSMSBacktesterV2
except ImportError as e:
    warnings.warn(f"Task 1.4コンポーネントインポート失敗: {e}")
    DSSMSSwitchCoordinatorV2 = None
    SwitchDiagnostics = None

# 既存DSSMSコンポーネント
try:
    from src.dssms.dssms_data_manager import DSSMSDataManager
    from src.dssms.nikkei225_screener import Nikkei225Screener
except ImportError as e:
    warnings.warn(f"DSSMSコンポーネントインポート失敗: {e}")
    DSSMSDataManager = None
    Nikkei225Screener = None

warnings.filterwarnings('ignore')

class DSSMSBacktesterV2Updated:
    """
    DSSMS Task 1.4対応バックテスターV2更新版
    切替メカニズム復旧機能の包括的テスト
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.logger.info("=== DSSMS Backtester V2 Updated 初期化開始 ===")
        
        # 設定
        self.config = config or self._get_default_config()
        
        # Task 1.4コンポーネント
        self.switch_coordinator = None
        self.diagnostics = None
        self.legacy_backtester = None
        
        # データ管理
        self.data_manager = None
        self.screener = None
        
        # バックテスト結果
        self.backtest_results: List[Dict[str, Any]] = []
        self.switch_statistics: Dict[str, Any] = {}
        self.engine_performance: Dict[str, Any] = {}
        
        self._initialize_components()
        self.logger.info("=== DSSMS Backtester V2 Updated 初期化完了 ===")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "backtest_period_days": 30,
            "initial_cash": 1000000,
            "position_size": 100000,
            "max_positions": 5,
            "rebalance_frequency": "daily",
            "transaction_cost": 0.001,
            "slippage": 0.0005,
            "success_rate_target": 0.30,
            "daily_switch_target": 1,
            "enable_diagnostics": True,
            "enable_switch_tracking": True,
            "output_detailed_logs": True
        }
    
    def _initialize_components(self):
        """コンポーネント初期化"""
        try:
            # Task 1.4コンポーネント初期化
            if DSSMSSwitchCoordinatorV2:
                self.switch_coordinator = DSSMSSwitchCoordinatorV2()
                self.logger.info("Switch Coordinator V2初期化完了")
            else:
                self.logger.warning("Switch Coordinator V2が利用できません")
            
            if self.config.get("enable_diagnostics", True) and SwitchDiagnostics:
                self.diagnostics = SwitchDiagnostics()
                self.logger.info("Switch Diagnostics初期化完了")
            else:
                self.logger.warning("Switch Diagnosticsが利用できません")
            
            # レガシーバックテスター（スキップ）
            self.logger.info("Legacy Backtester初期化スキップ（依存関係なし）")
            
            # データ管理コンポーネント
            if DSSMSDataManager:
                self.data_manager = DSSMSDataManager()
                self.logger.info("Data Manager初期化完了")
            else:
                self.logger.warning("Data Managerが利用できません")
                
            if Nikkei225Screener:
                self.screener = Nikkei225Screener()
                self.logger.info("Screener初期化完了")
            else:
                self.logger.warning("Screenerが利用できません")
            
        except Exception as e:
            self.logger.error(f"コンポーネント初期化失敗: {e}")
            # 初期化失敗時でも動作継続
    
    def run_comprehensive_backtest(self, 
                                 start_date: str,
                                 end_date: str,
                                 symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        包括的バックテスト実行
        
        Args:
            start_date: 開始日
            end_date: 終了日
            symbols: 対象銘柄リスト
        
        Returns:
            Dict[str, Any]: バックテスト結果
        """
        self.logger.info(f"=== 包括的バックテスト開始 [{start_date} - {end_date}] ===")
        start_time = time.time()
        
        try:
            # データ準備
            market_data = self._prepare_market_data(start_date, end_date, symbols)
            self.logger.info(f"市場データ準備完了: {len(market_data)} レコード")
            
            # ポートフォリオ初期化
            portfolio = self._initialize_portfolio(market_data)
            
            # 日次バックテスト実行
            daily_results = []
            current_positions = []
            
            for date in pd.date_range(start_date, end_date, freq='D'):
                if date.weekday() >= 5:  # 土日スキップ
                    continue
                
                date_str = date.strftime('%Y-%m-%d')
                daily_data = market_data[market_data.index.str.startswith(date_str)]
                
                if daily_data.empty:
                    continue
                
                # 日次処理実行
                daily_result = self._process_daily_backtest(
                    date, daily_data, current_positions, portfolio
                )
                daily_results.append(daily_result)
                
                # ポジション更新
                current_positions = daily_result.get("new_positions", current_positions)
                
                if len(daily_results) % 10 == 0:
                    self.logger.info(f"進捗: {len(daily_results)}日完了")
            
            # 結果集計
            execution_time = time.time() - start_time
            comprehensive_result = self._compile_comprehensive_results(
                daily_results, execution_time, start_date, end_date
            )
            
            # 診断レポート生成
            if self.diagnostics:
                diagnostic_report = self.diagnostics.generate_diagnostic_report(
                    analysis_days=30, include_details=True
                )
                comprehensive_result["diagnostic_report"] = diagnostic_report
            
            self.logger.info(f"=== 包括的バックテスト完了 [{execution_time:.2f}秒] ===")
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"包括的バックテスト失敗: {e}")
            raise
    
    def _prepare_market_data(self, start_date: str, end_date: str, 
                           symbols: Optional[List[str]]) -> pd.DataFrame:
        """市場データ準備"""
        if symbols is None:
            # デフォルト銘柄設定
            if self.screener and hasattr(self.screener, 'get_nikkei225_symbols'):
                try:
                    symbols = self.screener.get_nikkei225_symbols()[:20]  # 上位20銘柄
                except:
                    symbols = ["7203", "6758", "9984", "9983", "8306"]  # デフォルト銘柄
            else:
                symbols = ["7203", "6758", "9984", "9983", "8306"]  # デフォルト銘柄
        
        # データ取得・加工
        if self.data_manager and hasattr(self.data_manager, 'fetch_historical_data'):
            try:
                market_data = self.data_manager.fetch_historical_data(
                    symbols, start_date, end_date
                )
            except Exception as e:
                self.logger.error(f"データ取得失敗: {e}")
                raise RuntimeError(f"市場データ取得に失敗しました: {e}")
        else:
            self.logger.error("データマネージャーが利用できません")
            raise RuntimeError("データマネージャーが初期化されていません")
        
        # データ品質チェック
        if market_data.empty:
            raise ValueError("市場データが取得できませんでした")
        
        # 指標計算
        market_data = self._calculate_technical_indicators(market_data)
        
        return market_data
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標計算"""
        try:
            # 移動平均
            data['ma_5'] = data['close'].rolling(5).mean()
            data['ma_25'] = data['close'].rolling(25).mean()
            data['ma_75'] = data['close'].rolling(75).mean()
            
            # ボリンジャーバンド
            rolling_mean = data['close'].rolling(20).mean()
            rolling_std = data['close'].rolling(20).std()
            data['bb_upper'] = rolling_mean + (rolling_std * 2)
            data['bb_lower'] = rolling_mean - (rolling_std * 2)
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            return data
            
        except Exception as e:
            self.logger.warning(f"テクニカル指標計算失敗: {e}")
            return data
    
    def _initialize_portfolio(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """ポートフォリオ初期化"""
        return {
            "cash": self.config["initial_cash"],
            "positions": {},
            "total_value": self.config["initial_cash"],
            "transaction_history": [],
            "performance_metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
        }
    
    def _process_daily_backtest(self, date: datetime, 
                              daily_data: pd.DataFrame,
                              current_positions: List[str],
                              portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """日次バックテスト処理"""
        try:
            # 切替決定実行
            if self.switch_coordinator and hasattr(self.switch_coordinator, 'execute_switch_decision'):
                switch_result = self.switch_coordinator.execute_switch_decision(
                    daily_data, current_positions
                )
            else:
                self.logger.error("スイッチコーディネーターが利用できません")
                raise RuntimeError("スイッチコーディネーターが初期化されていません")
            
            # 診断記録
            if self.diagnostics and hasattr(self.diagnostics, 'record_switch_decision'):
                self.diagnostics.record_switch_decision(
                    engine_used=getattr(switch_result, 'engine_used', 'mock'),
                    decision_factors={
                        "market_conditions": self._analyze_market_conditions(daily_data),
                        "current_positions": current_positions,
                        "portfolio_value": portfolio["total_value"]
                    },
                    input_conditions={
                        "date": date.isoformat(),
                        "data_points": len(daily_data),
                        "position_count": len(current_positions)
                    },
                    output_result={
                        "switches_count": getattr(switch_result, 'switches_count', 0),
                        "new_positions": getattr(switch_result, 'symbols_after', current_positions),
                        "execution_time_ms": getattr(switch_result, 'execution_time_ms', 0.0)
                    },
                    success=getattr(switch_result, 'success', False),
                    execution_time_ms=getattr(switch_result, 'execution_time_ms', 0.0)
                )
            
            # ポートフォリオ更新
            portfolio_updates = self._update_portfolio(
                portfolio, switch_result, daily_data
            )
            
            # 日次結果
            daily_result: Dict[str, Any] = {
                "date": date.isoformat(),
                "switch_result": {
                    "success": getattr(switch_result, 'success', False),
                    "engine_used": getattr(switch_result, 'engine_used', 'mock'),
                    "switches_count": getattr(switch_result, 'switches_count', 0),
                    "execution_time_ms": getattr(switch_result, 'execution_time_ms', 0.0)
                },
                "portfolio_snapshot": {
                    "total_value": portfolio["total_value"],
                    "cash": portfolio["cash"],
                    "position_count": len(getattr(switch_result, 'symbols_after', current_positions))
                },
                "new_positions": getattr(switch_result, 'symbols_after', current_positions),
                "performance_impact": portfolio_updates.get("performance_impact", 0.0),
                "market_conditions": self._analyze_market_conditions(daily_data)
            }
            
            return daily_result
            
        except Exception as e:
            self.logger.error(f"日次バックテスト処理失敗 [{date}]: {e}")
            return {
                "date": date.isoformat(),
                "error": str(e),
                "switch_result": {"success": False},
                "new_positions": current_positions
            }
    
    def _analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """市場状況分析"""
        try:
            if data.empty:
                return {"status": "no_data"}
            
            # 基本統計
            avg_volume = data['volume'].mean() if 'volume' in data.columns else 0
            avg_price = data['close'].mean() if 'close' in data.columns else 0
            volatility = data['close'].std() if 'close' in data.columns else 0
            
            # トレンド判定
            if 'ma_5' in data.columns and 'ma_25' in data.columns:
                trend = "bullish" if data['ma_5'].iloc[-1] > data['ma_25'].iloc[-1] else "bearish"
            else:
                trend = "neutral"
            
            return {
                "avg_volume": float(avg_volume),
                "avg_price": float(avg_price),
                "volatility": float(volatility),
                "trend": trend,
                "data_quality": "good" if len(data) > 10 else "limited"
            }
            
        except Exception as e:
            self.logger.warning(f"市場状況分析失敗: {e}")
            return {"status": "analysis_error", "error": str(e)}
    
    def _update_portfolio(self, portfolio: Dict[str, Any], 
                        switch_result: Any,
                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """ポートフォリオ更新"""
        try:
            performance_impact = 0.0
            transaction_cost = 0.0
            
            success = getattr(switch_result, 'success', False)
            switches_count = getattr(switch_result, 'switches_count', 0)
            
            if success and switches_count > 0:
                # 切替によるコスト計算
                transaction_cost = (
                    switches_count * 
                    self.config["position_size"] * 
                    self.config["transaction_cost"]
                )
                
                portfolio["cash"] -= transaction_cost
                
                # パフォーマンス影響推定（簡易）
                performance_impact = -transaction_cost / portfolio["total_value"]
                
                # 取引履歴記録
                engine_used = getattr(switch_result, 'engine_used', 'unknown')
                portfolio["transaction_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "switch",
                    "switches_count": switches_count,
                    "cost": transaction_cost,
                    "engine_used": engine_used
                })
            
            # ポートフォリオ価値更新
            portfolio["total_value"] = portfolio["cash"]  # 簡易計算
            
            return {
                "performance_impact": performance_impact,
                "transaction_cost": transaction_cost
            }
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ更新失敗: {e}")
            return {"performance_impact": 0.0}
    
    def _compile_comprehensive_results(self, daily_results: List[Dict[str, Any]],
                                     execution_time: float,
                                     start_date: str, end_date: str) -> Dict[str, Any]:
        """包括的結果コンパイル"""
        try:
            # 基本統計
            total_days = len(daily_results)
            successful_switches = sum(
                1 for r in daily_results 
                if r.get("switch_result", {}).get("success", False)
            )
            
            success_rate = successful_switches / total_days if total_days > 0 else 0.0
            
            # エンジン別統計
            engine_stats = {}
            for result in daily_results:
                engine = result.get("switch_result", {}).get("engine_used", "unknown")
                if engine not in engine_stats:
                    engine_stats[engine] = {"attempts": 0, "successes": 0}
                
                engine_stats[engine]["attempts"] += 1
                if result.get("switch_result", {}).get("success", False):
                    engine_stats[engine]["successes"] += 1
            
            # エンジン別成功率計算
            for engine in engine_stats:
                attempts = engine_stats[engine]["attempts"]
                successes = engine_stats[engine]["successes"]
                engine_stats[engine]["success_rate"] = successes / attempts if attempts > 0 else 0.0
            
            # 日次切替統計
            switches_by_day = [
                r.get("switch_result", {}).get("switches_count", 0)
                for r in daily_results
            ]
            
            avg_switches_per_day = np.mean(switches_by_day) if switches_by_day else 0.0
            total_switches = sum(switches_by_day)
            
            # パフォーマンス統計
            performance_impacts = [
                r.get("performance_impact", 0.0)
                for r in daily_results if r.get("performance_impact") is not None
            ]
            
            avg_performance_impact = np.mean(performance_impacts) if performance_impacts else 0.0
            
            # 実行時間統計
            execution_times = [
                r.get("switch_result", {}).get("execution_time_ms", 0.0)
                for r in daily_results
            ]
            
            avg_execution_time = np.mean(execution_times) if execution_times else 0.0
            
            # 目標達成評価
            target_achievement = {
                "success_rate_target": self.config["success_rate_target"],
                "actual_success_rate": success_rate,
                "success_rate_achieved": success_rate >= self.config["success_rate_target"],
                "daily_switch_target": self.config["daily_switch_target"],
                "actual_avg_daily_switches": avg_switches_per_day,
                "daily_switch_achieved": avg_switches_per_day >= self.config["daily_switch_target"]
            }
            
            comprehensive_result = {
                "backtest_metadata": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": total_days,
                    "execution_time_seconds": execution_time,
                    "config": self.config
                },
                "overall_performance": {
                    "total_switch_attempts": total_days,
                    "successful_switches": successful_switches,
                    "overall_success_rate": success_rate,
                    "total_switches_executed": total_switches,
                    "avg_switches_per_day": avg_switches_per_day,
                    "avg_performance_impact": avg_performance_impact,
                    "avg_execution_time_ms": avg_execution_time
                },
                "engine_performance": engine_stats,
                "target_achievement": target_achievement,
                "daily_results": daily_results,
                "statistics_summary": {
                    "best_performing_engine": max(engine_stats.keys(), 
                                                key=lambda x: engine_stats[x]["success_rate"]) if engine_stats else None,
                    "most_used_engine": max(engine_stats.keys(),
                                          key=lambda x: engine_stats[x]["attempts"]) if engine_stats else None,
                    "consistency_score": 1.0 - np.var([r.get("switch_result", {}).get("success", 0) for r in daily_results])
                }
            }
            
            # Switch Coordinator統計統合
            if self.switch_coordinator:
                coordinator_stats = self.switch_coordinator.get_performance_statistics()
                comprehensive_result["coordinator_statistics"] = coordinator_stats
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"結果コンパイル失敗: {e}")
            return {
                "error": str(e),
                "partial_results": daily_results[:10]  # 部分的結果
            }
    
    def compare_with_legacy(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """レガシー版との比較"""
        if not self.legacy_backtester:
            return {"error": "レガシーバックテスターが利用できません"}
        
        try:
            # Task 1.4版実行
            task14_results = self.run_comprehensive_backtest(start_date, end_date)
            
            # レガシー版実行（簡易）
            legacy_results = {"success_rate": 0.15, "avg_execution_time": 2000}  # モック
            
            comparison = {
                "task_1_4_performance": {
                    "success_rate": task14_results["overall_performance"]["overall_success_rate"],
                    "avg_execution_time": task14_results["overall_performance"]["avg_execution_time_ms"],
                    "target_achievement": task14_results["target_achievement"]["success_rate_achieved"]
                },
                "legacy_performance": legacy_results,
                "improvement_analysis": {
                    "success_rate_improvement": (
                        task14_results["overall_performance"]["overall_success_rate"] - 
                        legacy_results["success_rate"]
                    ),
                    "speed_improvement": (
                        legacy_results["avg_execution_time"] - 
                        task14_results["overall_performance"]["avg_execution_time_ms"]
                    ),
                    "overall_assessment": "改善" if task14_results["overall_performance"]["overall_success_rate"] > legacy_results["success_rate"] else "要調整"
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"レガシー比較失敗: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """パフォーマンスレポート生成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = project_root / "output" / "task_14_backtests"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"task_14_backtest_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"バックテストレポート保存完了: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"レポート生成失敗: {e}")
            return ""
    
    def get_system_status(self) -> Dict[str, Any]:
        """システムステータス"""
        return {
            "components": {
                "switch_coordinator": self.switch_coordinator is not None,
                "diagnostics": self.diagnostics is not None,
                "legacy_backtester": self.legacy_backtester is not None,
                "data_manager": self.data_manager is not None
            },
            "config": self.config,
            "results_count": len(self.backtest_results),
            "last_update": datetime.now().isoformat()
        }
