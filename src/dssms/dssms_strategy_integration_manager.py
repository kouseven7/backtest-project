"""
DSSMS Phase 2 Task 2.1: 既存戦略システム統合マネージャー
DSSMS改善タスク設計 Phase 2: ハイブリッド実装 Task 2.1: 既存戦略システム統合

主要機能:
1. 既存戦略システムとDSSMSの統合オーケストレーション
2. ハイブリッド方式による戦略スコアリングと選択
3. DSSMS優先データフローの実装
4. 階層的切替メカニズムの統合管理
5. 段階的テスト機能の提供

設計方針:
- DSSMSエンジンを優先利用
- 既存戦略への段階的移行サポート
- エラー処理とフォールバック機能
- 包括的ログとパフォーマンス追跡
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import traceback

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# DSSMSコンポーネント
try:
    from src.dssms.dssms_backtester import DSSMSBacktester
    from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem
    from src.dssms.intelligent_switch_manager import IntelligentSwitchManager
    from src.dssms.comprehensive_scoring_engine import ComprehensiveScoringEngine
    from src.dssms.dssms_data_manager import DSSMSDataManager
    from src.dssms.market_condition_monitor import MarketConditionMonitor
    from src.dssms.dssms_strategy_bridge import DSSMSStrategyBridge
    from src.dssms.strategy_dssms_coordinator import StrategyDSSMSCoordinator
    from src.dssms.integrated_performance_calculator import IntegratedPerformanceCalculator
except ImportError as e:
    print(f"DSSMS components import warning: {e}")

# 既存戦略システム
try:
    from src.strategies.VWAP_Breakout import VWAPBreakoutStrategy
    from src.strategies.gc_strategy_signal import GoldenCrossStrategy
    from src.strategies.Momentum_Investing import MomentumInvestingStrategy
    from src.strategies.VWAP_Bounce import VWAPBounceStrategy
    from src.strategies.Opening_Gap import OpeningGapStrategy
    from src.strategies.base_strategy import BaseStrategy
except ImportError as e:
    print(f"Strategy components import warning: {e}")

# データ処理
try:
    from data_fetcher import DataFetcher
    from data_processor import DataProcessor
except ImportError as e:
    print(f"Data processing import warning: {e}")

# ロギング設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class IntegrationConfig:
    """統合システム設定"""
    # システム設定
    use_dssms_priority: bool = True
    fallback_to_strategies: bool = True
    enable_hybrid_mode: bool = True
    
    # パフォーマンス設定
    dssms_weight: float = 0.7
    strategy_weight: float = 0.3
    confidence_threshold: float = 0.6
    
    # 切替設定
    enable_hierarchical_switching: bool = True
    symbol_switch_threshold: float = 0.15
    strategy_switch_threshold: float = 0.10
    
    # リスク管理
    max_position_size: float = 0.20
    stop_loss_threshold: float = 0.05
    enable_risk_management: bool = True

@dataclass
class IntegrationResult:
    """統合結果"""
    selected_system: str
    selected_strategy: Optional[str]
    confidence_score: float
    dssms_score: Optional[float]
    strategy_scores: Dict[str, float]
    position_signal: str  # 'buy', 'sell', 'hold'
    reason: str
    timestamp: datetime

class IntegrationSystemType(Enum):
    """統合システムタイプ"""
    DSSMS_ONLY = "dssms_only"
    STRATEGY_ONLY = "strategy_only"
    HYBRID = "hybrid"
    FALLBACK = "fallback"

class DSSMSStrategyIntegrationManager:
    """
    DSSMS戦略統合マネージャー
    
    DSSMSと既存戦略システムの統合オーケストレーションを担当し、
    ハイブリッド方式による最適な戦略選択と実行を行います。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config = self._load_config(config_path)
        self.dssms_backtester = None
        self.strategy_bridge = None
        self.coordinator = None
        self.performance_calculator = None
        self.data_manager = None
        
        # 統合状態
        self.integration_history = []
        self.performance_history = []
        self.error_history = []
        
        # 戦略マッピング
        self.strategy_mapping = self._load_strategy_mapping()
        
        logger.info("DSSMS Strategy Integration Manager initialized")
        
    def _load_config(self, config_path: Optional[str]) -> IntegrationConfig:
        """設定ロード"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return IntegrationConfig(**config_data.get('integration', {}))
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return IntegrationConfig()
    
    def _load_strategy_mapping(self) -> Dict[str, Any]:
        """戦略マッピングロード"""
        mapping_path = project_root / "src" / "dssms" / "strategy_integration_mapping.json"
        if mapping_path.exists():
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load strategy mapping: {e}")
        
        # デフォルトマッピング
        return {
            "strategies": {
                "VWAP_Breakout": {
                    "class": "VWAPBreakoutStrategy",
                    "priority": 1,
                    "market_conditions": ["trending", "volatile"],
                    "enabled": True
                },
                "GoldenCross": {
                    "class": "GoldenCrossStrategy", 
                    "priority": 2,
                    "market_conditions": ["trending", "bullish"],
                    "enabled": True
                },
                "Momentum": {
                    "class": "MomentumInvestingStrategy",
                    "priority": 3,
                    "market_conditions": ["trending", "strong_volume"],
                    "enabled": True
                },
                "VWAP_Bounce": {
                    "class": "VWAPBounceStrategy",
                    "priority": 4,
                    "market_conditions": ["sideways", "support_resistance"],
                    "enabled": True
                },
                "Opening_Gap": {
                    "class": "OpeningGapStrategy",
                    "priority": 5,
                    "market_conditions": ["gap_up", "gap_down"],
                    "enabled": True
                }
            }
        }
    
    def initialize_systems(self, data: Dict[str, pd.DataFrame], 
                          index_data: pd.DataFrame) -> bool:
        """システム初期化"""
        try:
            logger.info("Initializing integration systems...")
            
            # DSSMSシステム初期化
            if self.config.use_dssms_priority:
                # 設定のみでDSSMSBacktesterを初期化
                backtester_config = {
                    'initial_capital': 1000000,
                    'switch_cost_rate': 0.001,
                    'enable_detailed_report': True
                }
                self.dssms_backtester = DSSMSBacktester(config=backtester_config)
                logger.info("DSSMS Backtester initialized")
            
            # 戦略ブリッジ初期化
            self.strategy_bridge = DSSMSStrategyBridge(
                strategy_mapping=self.strategy_mapping
            )
            
            # コーディネーター初期化
            self.coordinator = StrategyDSSMSCoordinator(
                config=self.config
            )
            
            # パフォーマンス計算機初期化
            self.performance_calculator = IntegratedPerformanceCalculator(
                use_dssms_engine=True
            )
            
            # データマネージャー初期化
            self.data_manager = DSSMSDataManager()
            
            logger.info("All integration systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize systems: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def execute_integrated_analysis(self, 
                                   symbol: str,
                                   date: datetime,
                                   data: pd.DataFrame,
                                   index_data: pd.DataFrame) -> IntegrationResult:
        """統合分析実行"""
        try:
            logger.debug(f"Executing integrated analysis for {symbol} on {date}")
            
            # DSSMS分析
            dssms_score = None
            dssms_signal = None
            
            if self.config.use_dssms_priority and self.dssms_backtester:
                try:
                    dssms_result = self.dssms_backtester.analyze_symbol(
                        symbol=symbol,
                        date=date,
                        data=data,
                        index_data=index_data
                    )
                    dssms_score = dssms_result.get('score', 0.0)
                    dssms_signal = dssms_result.get('signal', 'hold')
                    logger.debug(f"DSSMS analysis: score={dssms_score}, signal={dssms_signal}")
                except Exception as e:
                    logger.warning(f"DSSMS analysis failed: {e}")
            
            # 戦略分析
            strategy_scores = {}
            strategy_signals = {}
            
            if self.config.fallback_to_strategies or self.config.enable_hybrid_mode:
                try:
                    strategy_results = self.strategy_bridge.analyze_all_strategies(
                        symbol=symbol,
                        date=date,
                        data=data,
                        index_data=index_data
                    )
                    strategy_scores = strategy_results.get('scores', {})
                    strategy_signals = strategy_results.get('signals', {})
                    logger.debug(f"Strategy analysis: {len(strategy_scores)} strategies analyzed")
                except Exception as e:
                    logger.warning(f"Strategy analysis failed: {e}")
            
            # 統合判定
            integration_result = self.coordinator.coordinate_decision(
                dssms_score=dssms_score,
                dssms_signal=dssms_signal,
                strategy_scores=strategy_scores,
                strategy_signals=strategy_signals,
                symbol=symbol,
                date=date
            )
            
            # 結果記録
            self.integration_history.append(integration_result)
            
            return integration_result
            
        except Exception as e:
            logger.error(f"Integrated analysis failed for {symbol}: {e}")
            logger.error(traceback.format_exc())
            
            # エラー時のフォールバック
            return IntegrationResult(
                selected_system="fallback",
                selected_strategy=None,
                confidence_score=0.0,
                dssms_score=None,
                strategy_scores={},
                position_signal="hold",
                reason=f"Analysis failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def run_integrated_backtest(self,
                               symbols: List[str],
                               start_date: str,
                               end_date: str,
                               initial_capital: float = 1000000) -> Dict[str, Any]:
        """統合バックテスト実行"""
        try:
            logger.info(f"Starting integrated backtest for {len(symbols)} symbols")
            logger.info(f"Period: {start_date} to {end_date}")
            
            # データ取得 - 実際のdata_fetcher関数を使用
            try:
                from data_fetcher import get_parameters_and_data
                
                # 個別銘柄データ
                stock_data = {}
                for symbol in symbols:
                    try:
                        ticker, start_date_used, end_date_used, stock_data_single, index_data_temp = get_parameters_and_data(
                            ticker=symbol,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if stock_data_single is not None and not stock_data_single.empty:
                            stock_data[symbol] = stock_data_single
                            logger.debug(f"Data loaded for {symbol}: {len(stock_data_single)} records")
                    except Exception as e:
                        logger.warning(f"Failed to load data for {symbol}: {e}")
                        # モックデータ生成
                        stock_data[symbol] = self._generate_mock_single_data(symbol, start_date, end_date)
                
                # インデックスデータ
                try:
                    _, _, _, _, index_data = get_parameters_and_data(
                        ticker="^N225",
                        start_date=start_date,
                        end_date=end_date
                    )
                    if index_data is None or index_data.empty:
                        index_data = self._generate_mock_index_data(start_date, end_date)
                except Exception as e:
                    logger.warning(f"Failed to load index data: {e}")
                    index_data = self._generate_mock_index_data(start_date, end_date)
                    
            except ImportError:
                logger.warning("get_parameters_and_data not available, using mock data")
                stock_data = {symbol: self._generate_mock_single_data(symbol, start_date, end_date) for symbol in symbols}
                index_data = self._generate_mock_index_data(start_date, end_date)
            
            # システム初期化
            if not self.initialize_systems(stock_data, index_data):
                raise RuntimeError("Failed to initialize systems")
            
            # バックテスト実行
            results = self._execute_backtest_loop(
                stock_data=stock_data,
                index_data=index_data,
                initial_capital=initial_capital
            )
            
            # パフォーマンス計算
            performance_metrics = self.performance_calculator.calculate_comprehensive_performance(
                results=results,
                initial_capital=initial_capital
            )
            
            # 結果統合
            final_results = {
                'backtest_results': results,
                'performance_metrics': performance_metrics,
                'integration_history': self.integration_history,
                'system_statistics': self._calculate_system_statistics(),
                'configuration': self.config.__dict__,
                'metadata': {
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'execution_time': datetime.now().isoformat()
                }
            }
            
            logger.info("Integrated backtest completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Integrated backtest failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _execute_backtest_loop(self,
                              stock_data: Dict[str, pd.DataFrame],
                              index_data: pd.DataFrame,
                              initial_capital: float) -> Dict[str, Any]:
        """バックテストループ実行"""
        
        # 取引日程生成
        all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
        
        # 初期状態
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {}
        trades = []
        daily_values = []
        
        for date in all_dates:
            daily_date = pd.to_datetime(date)
            
            # 各銘柄の分析実行
            for symbol, data in stock_data.items():
                if date not in data.index:
                    continue
                
                current_data = data.loc[:date]
                current_index = index_data.loc[:date] if date in index_data.index else index_data
                
                # 統合分析
                analysis_result = self.execute_integrated_analysis(
                    symbol=symbol,
                    date=daily_date,
                    data=current_data,
                    index_data=current_index
                )
                
                # ポジション管理
                current_price = data.loc[date, 'Adj Close']
                
                if analysis_result.position_signal == 'buy' and symbol not in positions:
                    # 買いシグナル：新規ポジション
                    position_size = min(cash * 0.1, cash)  # 10%または残り現金
                    if position_size > current_price:
                        shares = int(position_size / current_price)
                        cost = shares * current_price
                        
                        positions[symbol] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_date': date,
                            'entry_system': analysis_result.selected_system
                        }
                        cash -= cost
                        
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'buy',
                            'shares': shares,
                            'price': current_price,
                            'value': cost,
                            'system': analysis_result.selected_system,
                            'strategy': analysis_result.selected_strategy,
                            'confidence': analysis_result.confidence_score
                        })
                        
                        logger.debug(f"BUY {symbol}: {shares} shares at {current_price}")
                
                elif analysis_result.position_signal == 'sell' and symbol in positions:
                    # 売りシグナル：ポジション決済
                    position = positions[symbol]
                    shares = position['shares']
                    sale_value = shares * current_price
                    
                    cash += sale_value
                    profit = sale_value - (shares * position['entry_price'])
                    
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares,
                        'price': current_price,
                        'value': sale_value,
                        'profit': profit,
                        'system': analysis_result.selected_system,
                        'strategy': analysis_result.selected_strategy,
                        'confidence': analysis_result.confidence_score,
                        'holding_period': (daily_date - pd.to_datetime(position['entry_date'])).days
                    })
                    
                    del positions[symbol]
                    logger.debug(f"SELL {symbol}: {shares} shares at {current_price}, profit: {profit}")
            
            # 日次ポートフォリオ価値計算
            position_value = sum(
                pos['shares'] * stock_data[symbol].loc[date, 'Adj Close']
                for symbol, pos in positions.items()
                if date in stock_data[symbol].index
            )
            portfolio_value = cash + position_value
            
            daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'position_value': position_value,
                'positions_count': len(positions)
            })
        
        return {
            'trades': trades,
            'daily_values': daily_values,
            'final_positions': positions,
            'final_cash': cash,
            'final_portfolio_value': portfolio_value
        }
    
    def _calculate_system_statistics(self) -> Dict[str, Any]:
        """システム統計計算"""
        if not self.integration_history:
            return {}
        
        total_decisions = len(self.integration_history)
        system_counts = {}
        confidence_scores = []
        
        for result in self.integration_history:
            system = result.selected_system
            system_counts[system] = system_counts.get(system, 0) + 1
            confidence_scores.append(result.confidence_score)
        
        return {
            'total_decisions': total_decisions,
            'system_usage': {k: v/total_decisions for k, v in system_counts.items()},
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'high_confidence_ratio': sum(1 for c in confidence_scores if c >= 0.7) / len(confidence_scores) if confidence_scores else 0
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """統合状況取得"""
        return {
            'dssms_available': self.dssms_backtester is not None,
            'strategy_bridge_available': self.strategy_bridge is not None,
            'coordinator_available': self.coordinator is not None,
            'performance_calculator_available': self.performance_calculator is not None,
            'configuration': self.config.__dict__,
            'integration_history_count': len(self.integration_history),
            'error_history_count': len(self.error_history),
            'strategy_mapping': self.strategy_mapping
        }

# 使用例とテスト関数
def test_integration_manager():
    """統合マネージャーのテスト"""
    print("=== DSSMS Strategy Integration Manager Test ===")
    
    # マネージャー初期化
    manager = DSSMSStrategyIntegrationManager()
    
    # ステータス確認
    status = manager.get_integration_status()
    print(f"Integration Status: {status}")
    
    # 簡単なテスト実行
    try:
        test_symbols = ['7203', '6758', '8306']
        results = manager.run_integrated_backtest(
            symbols=test_symbols,
            start_date='2024-01-01',
            end_date='2024-06-30',
            initial_capital=1000000
        )
        
        print(f"Backtest completed. Final portfolio value: {results['backtest_results']['final_portfolio_value']:,.0f}")
        print(f"Total trades: {len(results['backtest_results']['trades'])}")
        
        # パフォーマンスメトリクス
        metrics = results['performance_metrics']
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        
    except Exception as e:
        print(f"Test failed: {e}")

class DSSMSStrategyIntegrationManagerExtensions:
    """統合マネージャーの追加メソッド"""
    
    @staticmethod
    def _generate_mock_single_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """単一銘柄のモックデータ生成"""
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = pd.date_range(start, end, freq='D')
        
        # 基準価格を銘柄ごとに調整
        base_prices = {"7203": 1500, "6501": 800, "9984": 3000}
        base_price = base_prices.get(symbol, 1000)
        
        # ランダムウォークベースの価格生成
        np.random.seed(hash(symbol) % 1000)  # 銘柄ごとに異なるシード
        returns = np.random.normal(0.001, 0.02, len(date_range))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLCV データ生成
        data = pd.DataFrame({
            'Date': date_range,
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.005, 1.03) for p in prices],
            'Low': [p * np.random.uniform(0.97, 0.995) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(100000, 1000000) for _ in prices]
        })
        
        data.set_index('Date', inplace=True)
        return data
    
    @staticmethod
    def _generate_mock_index_data(start_date: str, end_date: str) -> pd.DataFrame:
        """インデックスのモックデータ生成"""
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = pd.date_range(start, end, freq='D')
        
        base_price = 28000  # 日経平均ベース
        np.random.seed(42)  # 固定シード
        returns = np.random.normal(0.0005, 0.015, len(date_range))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'Date': date_range,
            'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
            'High': [p * np.random.uniform(1.002, 1.015) for p in prices],
            'Low': [p * np.random.uniform(0.985, 0.998) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(500000, 2000000) for _ in prices]
        })
        
        data.set_index('Date', inplace=True)
        return data

# DSSMSStrategyIntegrationManagerにモックデータ生成メソッドを追加
DSSMSStrategyIntegrationManager._generate_mock_single_data = DSSMSStrategyIntegrationManagerExtensions._generate_mock_single_data
DSSMSStrategyIntegrationManager._generate_mock_index_data = DSSMSStrategyIntegrationManagerExtensions._generate_mock_index_data

if __name__ == "__main__":
    test_integration_manager()
