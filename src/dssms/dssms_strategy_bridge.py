"""
DSSMS Phase 2 Task 2.1: DSSMS戦略ブリッジ
既存戦略システムとDSSMSの橋渡し・アダプター機能

主要機能:
1. 既存戦略クラスの動的ロードと実行
2. 戦略シグナルの標準化とDSSMS形式への変換
3. パラメータ管理と最適化連携
4. エラーハンドリングとフォールバック機能
5. 戦略パフォーマンス追跡

設計方針:
- 既存戦略の非破壊的統合
- 動的戦略ロードによる拡張性
- 統一インターフェースによるシンプル化
- 堅牢なエラー処理
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Type
from datetime import datetime, timedelta
import json
import logging
import importlib
import inspect
from dataclasses import dataclass, field
from enum import Enum
import traceback

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 基底戦略
try:
    from src.strategies.base_strategy import BaseStrategy
except ImportError as e:
    print(f"Base strategy import warning: {e}")

# ロギング設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class StrategyResult:
    """戦略実行結果"""
    strategy_name: str
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    score: float
    entry_signals: pd.Series
    exit_signals: pd.Series
    metadata: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None

@dataclass
class StrategyConfig:
    """戦略設定"""
    name: str
    class_name: str
    module_path: str
    parameters: Dict[str, Any]
    enabled: bool = True
    priority: int = 1
    market_conditions: List[str] = field(default_factory=list)

class StrategySignalType(Enum):
    """戦略シグナルタイプ"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    UNKNOWN = "unknown"

class DSSMSStrategyBridge:
    """
    DSSMS戦略ブリッジ
    
    既存戦略システムとDSSMSの橋渡しを行い、
    統一されたインターフェースで戦略実行と結果管理を提供します。
    """
    
    def __init__(self, strategy_mapping: Optional[Dict[str, Any]] = None):
        """初期化"""
        self.strategy_mapping = strategy_mapping or {}
        self.loaded_strategies = {}
        self.strategy_configs = {}
        self.execution_history = []
        
        # 戦略設定ロード
        self._load_strategy_configs()
        
        logger.info("DSSMS Strategy Bridge initialized")
    
    def _load_strategy_configs(self):
        """戦略設定をロード"""
        try:
            # デフォルト戦略設定
            default_strategies = {
                "VWAP_Breakout": {
                    "class_name": "VWAPBreakoutStrategy",
                    "module_path": "src.strategies.VWAP_Breakout",
                    "parameters": {
                        "stop_loss": 0.03,
                        "take_profit": 0.15,
                        "volume_threshold": 1.2
                    },
                    "enabled": True,
                    "priority": 1,
                    "market_conditions": ["trending", "volatile"]
                },
                "GoldenCross": {
                    "class_name": "GoldenCrossStrategy",
                    "module_path": "src.strategies.gc_strategy_signal",
                    "parameters": {
                        "short_window": 50,
                        "long_window": 200
                    },
                    "enabled": True,
                    "priority": 2,
                    "market_conditions": ["trending", "bullish"]
                },
                "Momentum": {
                    "class_name": "MomentumInvestingStrategy",
                    "module_path": "src.strategies.Momentum_Investing",
                    "parameters": {
                        "lookback_period": 12,
                        "threshold": 0.05
                    },
                    "enabled": True,
                    "priority": 3,
                    "market_conditions": ["trending", "strong_volume"]
                },
                "VWAP_Bounce": {
                    "class_name": "VWAPBounceStrategy",
                    "module_path": "src.strategies.VWAP_Bounce",
                    "parameters": {
                        "bounce_threshold": 0.02,
                        "volume_confirmation": True
                    },
                    "enabled": True,
                    "priority": 4,
                    "market_conditions": ["sideways", "support_resistance"]
                },
                "Opening_Gap": {
                    "class_name": "OpeningGapStrategy",
                    "module_path": "src.strategies.Opening_Gap",
                    "parameters": {
                        "gap_threshold": 0.02,
                        "volume_multiplier": 1.5
                    },
                    "enabled": True,
                    "priority": 5,
                    "market_conditions": ["gap_up", "gap_down"]
                }
            }
            
            # マッピング情報がある場合は統合
            if self.strategy_mapping.get('strategies'):
                for name, config in self.strategy_mapping['strategies'].items():
                    if name in default_strategies:
                        default_strategies[name].update(config)
            
            # 戦略設定作成
            for name, config in default_strategies.items():
                # class_nameまたはclassフィールドの取得
                class_name = config.get('class_name') or config.get('class')
                if not class_name:
                    logger.warning(f"Strategy {name} missing class_name or class field")
                    continue
                    
                self.strategy_configs[name] = StrategyConfig(
                    name=name,
                    class_name=class_name,
                    module_path=config.get('module_path') or config.get('module'),
                    parameters=config.get('parameters', {}),
                    enabled=config.get('enabled', True),
                    priority=config.get('priority', 1),
                    market_conditions=config.get('market_conditions', [])
                )
            
            logger.info(f"Loaded {len(self.strategy_configs)} strategy configurations")
            
        except Exception as e:
            logger.error(f"Failed to load strategy configs: {e}")
            logger.error(traceback.format_exc())
    
    def load_strategy(self, strategy_name: str) -> Optional[Type[BaseStrategy]]:
        """戦略クラスの動的ロード"""
        if strategy_name in self.loaded_strategies:
            return self.loaded_strategies[strategy_name]
        
        if strategy_name not in self.strategy_configs:
            logger.warning(f"Strategy {strategy_name} not found in configurations")
            return None
        
        config = self.strategy_configs[strategy_name]
        
        try:
            # モジュールをインポート
            module = importlib.import_module(config.module_path)
            
            # クラスを取得
            strategy_class = getattr(module, config.class_name)
            
            # BaseStrategyの継承チェック
            if not issubclass(strategy_class, BaseStrategy):
                logger.warning(f"Strategy {strategy_name} does not inherit from BaseStrategy")
                return None
            
            self.loaded_strategies[strategy_name] = strategy_class
            logger.debug(f"Successfully loaded strategy: {strategy_name}")
            
            return strategy_class
            
        except Exception as e:
            logger.error(f"Failed to load strategy {strategy_name}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def execute_strategy(self,
                        strategy_name: str,
                        data: pd.DataFrame,
                        index_data: pd.DataFrame,
                        parameters: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """単一戦略実行"""
        start_time = pd.Timestamp.now()
        
        try:
            # 戦略クラスロード
            strategy_class = self.load_strategy(strategy_name)
            if strategy_class is None:
                return StrategyResult(
                    strategy_name=strategy_name,
                    signal="unknown",
                    confidence=0.0,
                    score=0.0,
                    entry_signals=pd.Series(dtype=float),
                    exit_signals=pd.Series(dtype=float),
                    metadata={},
                    execution_time=0.0,
                    error="Failed to load strategy class"
                )
            
            # パラメータ準備
            config = self.strategy_configs[strategy_name]
            strategy_params = config.parameters.copy()
            if parameters:
                strategy_params.update(parameters)
            
            # 戦略インスタンス作成
            strategy = strategy_class(
                data=data,
                index_data=index_data,
                params=strategy_params
            )
            
            # バックテスト実行
            backtest_result = strategy.backtest()
            
            # 結果解析
            signal, confidence, score = self._analyze_strategy_result(
                backtest_result, strategy_name
            )
            
            # 実行時間計算
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # 結果作成
            result = StrategyResult(
                strategy_name=strategy_name,
                signal=signal,
                confidence=confidence,
                score=score,
                entry_signals=backtest_result.get('Entry_Signal', pd.Series(dtype=float)),
                exit_signals=backtest_result.get('Exit_Signal', pd.Series(dtype=float)),
                metadata={
                    'total_signals': len(backtest_result.get('Entry_Signal', [])),
                    'parameters': strategy_params,
                    'data_length': len(data)
                },
                execution_time=execution_time,
                error=None
            )
            
            # 実行履歴記録
            self.execution_history.append({
                'strategy_name': strategy_name,
                'timestamp': datetime.now(),
                'success': True,
                'execution_time': execution_time,
                'signal': signal,
                'confidence': confidence
            })
            
            logger.debug(f"Strategy {strategy_name} executed successfully: {signal} (confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            error_msg = f"Strategy execution failed: {str(e)}"
            
            logger.error(f"Failed to execute strategy {strategy_name}: {e}")
            logger.error(traceback.format_exc())
            
            # エラー履歴記録
            self.execution_history.append({
                'strategy_name': strategy_name,
                'timestamp': datetime.now(),
                'success': False,
                'execution_time': execution_time,
                'error': error_msg
            })
            
            return StrategyResult(
                strategy_name=strategy_name,
                signal="unknown",
                confidence=0.0,
                score=0.0,
                entry_signals=pd.Series(dtype=float),
                exit_signals=pd.Series(dtype=float),
                metadata={},
                execution_time=execution_time,
                error=error_msg
            )
    
    def _analyze_strategy_result(self,
                               backtest_result: Dict[str, Any],
                               strategy_name: str) -> Tuple[str, float, float]:
        """戦略結果解析"""
        try:
            # エントリーシグナル取得
            entry_signals = backtest_result.get('Entry_Signal', pd.Series(dtype=float))
            exit_signals = backtest_result.get('Exit_Signal', pd.Series(dtype=float))
            
            if len(entry_signals) == 0:
                return "hold", 0.0, 0.0
            
            # 最新シグナル確認
            latest_entry = entry_signals.iloc[-1] if len(entry_signals) > 0 else 0
            latest_exit = exit_signals.iloc[-1] if len(exit_signals) > 0 else 0
            
            # シグナル判定
            if latest_entry > 0 and latest_entry > latest_exit:
                signal = "buy"
            elif latest_exit > 0 and latest_exit > latest_entry:
                signal = "sell"
            else:
                signal = "hold"
            
            # 信頼度計算（シグナル強度 + 履歴分析）
            signal_strength = max(latest_entry, latest_exit)
            
            # 履歴ベースの信頼度
            total_signals = len(entry_signals[entry_signals > 0]) + len(exit_signals[exit_signals > 0])
            signal_consistency = min(1.0, total_signals / 10.0)  # 10シグナルで最大
            
            confidence = (signal_strength * 0.7) + (signal_consistency * 0.3)
            confidence = min(1.0, max(0.0, confidence))
            
            # スコア計算（パフォーマンス指標）
            score = self._calculate_strategy_score(backtest_result)
            
            return signal, confidence, score
            
        except Exception as e:
            logger.warning(f"Failed to analyze strategy result for {strategy_name}: {e}")
            return "hold", 0.0, 0.0
    
    def _calculate_strategy_score(self, backtest_result: Dict[str, Any]) -> float:
        """戦略スコア計算"""
        try:
            # 基本スコア
            score = 0.5
            
            # 利益指標があれば加算
            if 'total_return' in backtest_result:
                total_return = backtest_result['total_return']
                score += min(0.3, max(-0.3, total_return / 2))  # ±30%まで
            
            # シャープレシオがあれば加算
            if 'sharpe_ratio' in backtest_result:
                sharpe = backtest_result['sharpe_ratio']
                score += min(0.2, max(-0.2, sharpe / 10))  # ±0.2まで
            
            # 勝率があれば加算
            if 'win_rate' in backtest_result:
                win_rate = backtest_result['win_rate']
                score += (win_rate - 0.5) * 0.2  # 50%基準で±0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate strategy score: {e}")
            return 0.5
    
    def analyze_all_strategies(self,
                              symbol: str,
                              date: datetime,
                              data: pd.DataFrame,
                              index_data: pd.DataFrame) -> Dict[str, Any]:
        """全戦略分析実行"""
        try:
            results = {}
            scores = {}
            signals = {}
            errors = {}
            
            # 有効な戦略のみ実行
            enabled_strategies = [name for name, config in self.strategy_configs.items() if config.enabled]
            
            logger.debug(f"Analyzing {len(enabled_strategies)} strategies for {symbol}")
            
            for strategy_name in enabled_strategies:
                try:
                    result = self.execute_strategy(
                        strategy_name=strategy_name,
                        data=data,
                        index_data=index_data
                    )
                    
                    results[strategy_name] = result
                    scores[strategy_name] = result.score
                    signals[strategy_name] = result.signal
                    
                    if result.error:
                        errors[strategy_name] = result.error
                    
                except Exception as e:
                    error_msg = f"Strategy {strategy_name} analysis failed: {str(e)}"
                    logger.warning(error_msg)
                    errors[strategy_name] = error_msg
                    scores[strategy_name] = 0.0
                    signals[strategy_name] = "unknown"
            
            # 統計情報
            valid_scores = [s for s in scores.values() if s > 0]
            
            return {
                'results': results,
                'scores': scores,
                'signals': signals,
                'errors': errors,
                'statistics': {
                    'total_strategies': len(enabled_strategies),
                    'successful_strategies': len(valid_scores),
                    'average_score': np.mean(valid_scores) if valid_scores else 0.0,
                    'max_score': max(valid_scores) if valid_scores else 0.0,
                    'error_count': len(errors)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze all strategies: {e}")
            logger.error(traceback.format_exc())
            return {
                'results': {},
                'scores': {},
                'signals': {},
                'errors': {'analysis_error': str(e)},
                'statistics': {
                    'total_strategies': 0,
                    'successful_strategies': 0,
                    'average_score': 0.0,
                    'max_score': 0.0,
                    'error_count': 1
                }
            }
    
    def get_strategy_by_market_condition(self, market_condition: str) -> List[str]:
        """市場状況に適した戦略取得"""
        suitable_strategies = []
        
        for name, config in self.strategy_configs.items():
            if config.enabled and market_condition in config.market_conditions:
                suitable_strategies.append(name)
        
        # 優先度順でソート
        suitable_strategies.sort(key=lambda x: self.strategy_configs[x].priority)
        
        return suitable_strategies
    
    def get_best_strategy(self, scores: Dict[str, float], 
                         market_condition: Optional[str] = None) -> Tuple[Optional[str], float]:
        """最高スコア戦略取得"""
        if not scores:
            return None, 0.0
        
        # 市場状況フィルター
        if market_condition:
            suitable_strategies = self.get_strategy_by_market_condition(market_condition)
            filtered_scores = {k: v for k, v in scores.items() if k in suitable_strategies}
            if filtered_scores:
                scores = filtered_scores
        
        # 最高スコア戦略
        best_strategy = max(scores.items(), key=lambda x: x[1])
        return best_strategy[0], best_strategy[1]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """実行統計取得"""
        if not self.execution_history:
            return {}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h['success'])
        
        execution_times = [h['execution_time'] for h in self.execution_history]
        
        strategy_stats = {}
        for strategy_name in self.strategy_configs.keys():
            strategy_executions = [h for h in self.execution_history if h['strategy_name'] == strategy_name]
            if strategy_executions:
                success_rate = sum(1 for h in strategy_executions if h['success']) / len(strategy_executions)
                avg_time = np.mean([h['execution_time'] for h in strategy_executions])
                strategy_stats[strategy_name] = {
                    'executions': len(strategy_executions),
                    'success_rate': success_rate,
                    'average_execution_time': avg_time
                }
        
        return {
            'total_executions': total_executions,
            'success_rate': successful_executions / total_executions,
            'average_execution_time': np.mean(execution_times),
            'strategy_statistics': strategy_stats,
            'loaded_strategies': list(self.loaded_strategies.keys()),
            'available_strategies': list(self.strategy_configs.keys())
        }

# 使用例とテスト関数
def test_strategy_bridge():
    """戦略ブリッジのテスト"""
    print("=== DSSMS Strategy Bridge Test ===")
    
    # ブリッジ初期化
    bridge = DSSMSStrategyBridge()
    
    # 設定確認
    print(f"Available strategies: {list(bridge.strategy_configs.keys())}")
    
    # テストデータ生成
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    test_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(95, 115, len(dates)),
        'Adj Close': np.random.uniform(95, 115, len(dates)),
        'Volume': np.random.uniform(1000000, 5000000, len(dates))
    }, index=dates)
    
    index_data = test_data.copy()
    
    try:
        # 戦略実行テスト
        if 'VWAP_Breakout' in bridge.strategy_configs:
            result = bridge.execute_strategy(
                strategy_name='VWAP_Breakout',
                data=test_data,
                index_data=index_data
            )
            
            print(f"VWAP_Breakout result: {result.signal} (confidence: {result.confidence:.3f})")
            print(f"Execution time: {result.execution_time:.3f}s")
            
        # 全戦略分析テスト
        all_results = bridge.analyze_all_strategies(
            symbol="TEST",
            date=datetime.now(),
            data=test_data,
            index_data=index_data
        )
        
        print(f"\nAll strategies analysis:")
        print(f"Successful strategies: {all_results['statistics']['successful_strategies']}")
        print(f"Average score: {all_results['statistics']['average_score']:.3f}")
        
        # 実行統計
        stats = bridge.get_execution_statistics()
        print(f"\nExecution statistics:")
        print(f"Total executions: {stats.get('total_executions', 0)}")
        print(f"Success rate: {stats.get('success_rate', 0):.2%}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_strategy_bridge()
