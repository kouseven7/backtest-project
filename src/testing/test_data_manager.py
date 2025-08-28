"""
DSSMS Phase 3 Task 3.3: テストデータ管理システム
ハイブリッド型テストデータ管理（固定 + 動的）

Author: GitHub Copilot Agent
Created: 2025-08-28
Phase: 3 Task 3.3
"""

import sys
import os
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class TestDataManager:
    """ハイブリッド型テストデータ管理"""
    
    def __init__(self, logger=None):
        """
        初期化
        
        Args:
            logger: ロガー
        """
        self.logger = logger or setup_logger("TestDataManager")
        self.project_root = project_root
        self.data_dir = self.project_root / "src" / "testing" / "test_data"
        self.cache_dir = self.data_dir / "cache"
        
        # ディレクトリ作成
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 固定データセット
        self.fixed_datasets = self._load_fixed_benchmarks()
        
        # 動的データ生成器
        self.dynamic_generator = DynamicDataGenerator(self.logger)
        
        self.logger.info("テストデータ管理システム初期化完了")
    
    def get_test_data(self, test_type: str, scenario: str = "default", **kwargs) -> Dict[str, Any]:
        """
        テストデータの取得
        
        Args:
            test_type: テストタイプ ("regression", "stress", "integration", "performance")
            scenario: シナリオ名
            **kwargs: 追加パラメータ
            
        Returns:
            テストデータ辞書
        """
        try:
            if test_type == "regression":
                return self._get_regression_data(scenario, **kwargs)
            elif test_type == "stress":
                return self._get_stress_data(scenario, **kwargs)
            elif test_type == "integration":
                return self._get_integration_data(scenario, **kwargs)
            elif test_type == "performance":
                return self._get_performance_data(scenario, **kwargs)
            else:
                # ハイブリッドデータセット
                return self._create_hybrid_dataset(scenario, **kwargs)
                
        except Exception as e:
            self.logger.error(f"テストデータ取得エラー: {e}")
            return self._get_fallback_data()
    
    def _load_fixed_benchmarks(self) -> Dict[str, Any]:
        """固定ベンチマークデータの読み込み"""
        benchmarks = {}
        
        try:
            # 固定ベンチマークデータの作成・読み込み
            benchmark_file = self.data_dir / "fixed_benchmarks.pkl"
            
            if benchmark_file.exists():
                with open(benchmark_file, 'rb') as f:
                    benchmarks = pickle.load(f)
                self.logger.info("固定ベンチマークデータ読み込み完了")
            else:
                # 固定ベンチマークデータの作成
                benchmarks = self._create_fixed_benchmarks()
                with open(benchmark_file, 'wb') as f:
                    pickle.dump(benchmarks, f)
                self.logger.info("固定ベンチマークデータ作成完了")
                
        except Exception as e:
            self.logger.warning(f"固定ベンチマーク読み込みエラー: {e}")
            benchmarks = self._create_minimal_benchmarks()
        
        return benchmarks
    
    def _create_fixed_benchmarks(self) -> Dict[str, Any]:
        """固定ベンチマークデータの作成"""
        benchmarks = {}
        
        # 1. 基本回帰テスト用データ
        benchmarks["basic_regression"] = self._create_basic_regression_data()
        
        # 2. パフォーマンステスト用データ
        benchmarks["performance_baseline"] = self._create_performance_baseline()
        
        # 3. 統合テスト用データ
        benchmarks["integration_baseline"] = self._create_integration_baseline()
        
        # 4. ストレステスト用データ
        benchmarks["stress_baseline"] = self._create_stress_baseline()
        
        return benchmarks
    
    def _create_basic_regression_data(self) -> Dict[str, Any]:
        """基本回帰テスト用データ"""
        # 90日間の市場データシミュレーション
        dates = pd.date_range(start='2025-05-30', end='2025-08-28', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        np.random.seed(42)  # 再現性のため
        
        market_data = {}
        for symbol in symbols:
            # 価格データ生成
            initial_price = np.random.uniform(100, 300)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [initial_price]
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            # OHLCV データ作成
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices[:-1],
                'High': [p * np.random.uniform(1.0, 1.05) for p in prices[:-1]],
                'Low': [p * np.random.uniform(0.95, 1.0) for p in prices[:-1]],
                'Close': prices[1:],
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            })
            
            market_data[symbol] = df
        
        return {
            'market_data': market_data,
            'expected_performance': {
                'total_return': 0.05,  # 5% 期待リターン
                'volatility': 0.15,
                'max_drawdown': -0.10,
                'sharpe_ratio': 1.2
            },
            'metadata': {
                'created': datetime.now(),
                'period_days': len(dates),
                'symbols': symbols,
                'data_quality': 'high'
            }
        }
    
    def _create_performance_baseline(self) -> Dict[str, Any]:
        """パフォーマンステスト用ベースライン"""
        # より長期間のデータ（252営業日）
        dates = pd.date_range(start='2024-08-28', end='2025-08-28', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
        
        np.random.seed(123)
        
        market_data = {}
        for symbol in symbols:
            initial_price = np.random.uniform(50, 400)
            
            # トレンド + ノイズ
            trend = np.linspace(0, 0.15, len(dates))  # 年間15%上昇トレンド
            noise = np.random.normal(0, 0.025, len(dates))
            returns = trend + noise
            
            prices = [initial_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret/252))  # 日次化
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices[:-1],
                'High': [p * np.random.uniform(1.01, 1.08) for p in prices[:-1]],
                'Low': [p * np.random.uniform(0.92, 0.99) for p in prices[:-1]],
                'Close': prices[1:],
                'Volume': np.random.randint(2000000, 20000000, len(dates))
            })
            
            market_data[symbol] = df
        
        return {
            'market_data': market_data,
            'expected_performance': {
                'total_return': 0.12,  # 12% 期待リターン
                'volatility': 0.18,
                'max_drawdown': -0.08,
                'sharpe_ratio': 1.8,
                'calmar_ratio': 1.5
            },
            'metadata': {
                'created': datetime.now(),
                'period_days': len(dates),
                'symbols': symbols,
                'data_quality': 'high',
                'test_type': 'performance'
            }
        }
    
    def _create_integration_baseline(self) -> Dict[str, Any]:
        """統合テスト用ベースライン"""
        # 中期間データ（120日）+ 戦略切替シナリオ
        dates = pd.date_range(start='2025-04-30', end='2025-08-28', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        np.random.seed(456)
        
        market_data = {}
        for symbol in symbols:
            initial_price = np.random.uniform(100, 250)
            
            # 市場レジーム変化をシミュレーション
            regime1_returns = np.random.normal(0.002, 0.015, len(dates)//2)  # 上昇相場
            regime2_returns = np.random.normal(-0.001, 0.025, len(dates) - len(dates)//2)  # ボラタイル相場
            
            returns = np.concatenate([regime1_returns, regime2_returns])
            
            prices = [initial_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices[:-1],
                'High': [p * np.random.uniform(1.005, 1.03) for p in prices[:-1]],
                'Low': [p * np.random.uniform(0.97, 0.995) for p in prices[:-1]],
                'Close': prices[1:],
                'Volume': np.random.randint(1500000, 15000000, len(dates))
            })
            
            market_data[symbol] = df
        
        # 戦略切替シナリオ
        switch_points = [30, 60, 90]  # 30日ごとに切替
        strategy_sequence = ['VWAP_Breakout', 'Mean_Reversion', 'Momentum_Strategy', 'Contrarian_Strategy']
        
        return {
            'market_data': market_data,
            'strategy_switches': {
                'switch_points': switch_points,
                'strategies': strategy_sequence,
                'expected_success_rate': 0.75
            },
            'expected_performance': {
                'total_return': 0.08,
                'switch_success_rate': 0.75,
                'volatility': 0.20
            },
            'metadata': {
                'created': datetime.now(),
                'period_days': len(dates),
                'symbols': symbols,
                'data_quality': 'high',
                'test_type': 'integration'
            }
        }
    
    def _create_stress_baseline(self) -> Dict[str, Any]:
        """ストレステスト用ベースライン"""
        # 高ボラティリティ期間（60日）
        dates = pd.date_range(start='2025-06-28', end='2025-08-28', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        np.random.seed(789)
        
        market_data = {}
        for symbol in symbols:
            initial_price = np.random.uniform(100, 200)
            
            # 高ボラティリティ + ドローダウンイベント
            returns = np.random.normal(-0.002, 0.04, len(dates))  # 高ボラ + 下落バイアス
            
            # 大きなドローダウンイベントを挿入
            crash_day = len(dates) // 3
            returns[crash_day] = -0.15  # 15%下落
            returns[crash_day + 1] = -0.08  # 8%下落
            
            prices = [initial_price]
            for ret in returns:
                prices.append(max(prices[-1] * (1 + ret), prices[-1] * 0.5))  # 最大50%下落制限
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices[:-1],
                'High': [p * np.random.uniform(1.0, 1.02) for p in prices[:-1]],
                'Low': [p * np.random.uniform(0.95, 1.0) for p in prices[:-1]],
                'Close': prices[1:],
                'Volume': np.random.randint(5000000, 50000000, len(dates))  # 高出来高
            })
            
            market_data[symbol] = df
        
        return {
            'market_data': market_data,
            'stress_conditions': {
                'high_volatility': True,
                'drawdown_events': True,
                'liquidity_crisis': False
            },
            'expected_performance': {
                'total_return': -0.10,  # 10%下落予想
                'volatility': 0.35,     # 35%ボラティリティ
                'max_drawdown': -0.25,  # 25%ドローダウン
                'recovery_time_days': 30
            },
            'metadata': {
                'created': datetime.now(),
                'period_days': len(dates),
                'symbols': symbols,
                'data_quality': 'stress',
                'test_type': 'stress'
            }
        }
    
    def _create_minimal_benchmarks(self) -> Dict[str, Any]:
        """最小限のベンチマークデータ"""
        dates = pd.date_range(start='2025-07-28', end='2025-08-28', freq='D')
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': 100,
            'High': 105,
            'Low': 95,
            'Close': 102,
            'Volume': 1000000
        })
        
        return {
            'minimal': {
                'market_data': {'AAPL': df},
                'expected_performance': {'total_return': 0.02},
                'metadata': {'created': datetime.now(), 'test_type': 'minimal'}
            }
        }
    
    def _get_regression_data(self, scenario: str, **kwargs) -> Dict[str, Any]:
        """回帰テスト用データ取得"""
        if scenario in self.fixed_datasets:
            return self.fixed_datasets[scenario]
        else:
            return self.fixed_datasets.get("basic_regression", self._get_fallback_data())
    
    def _get_stress_data(self, scenario: str, **kwargs) -> Dict[str, Any]:
        """ストレステスト用データ取得（動的生成）"""
        return self.dynamic_generator.create_stress_scenario(scenario, **kwargs)
    
    def _get_integration_data(self, scenario: str, **kwargs) -> Dict[str, Any]:
        """統合テスト用データ取得"""
        base_data = self.fixed_datasets.get("integration_baseline", {})
        
        # 動的要素を追加
        dynamic_elements = self.dynamic_generator.create_integration_elements(**kwargs)
        
        # ベースデータと動的要素をマージ
        return {**base_data, **dynamic_elements}
    
    def _get_performance_data(self, scenario: str, **kwargs) -> Dict[str, Any]:
        """パフォーマンステスト用データ取得"""
        return self.fixed_datasets.get("performance_baseline", self._get_fallback_data())
    
    def _create_hybrid_dataset(self, scenario: str, **kwargs) -> Dict[str, Any]:
        """ハイブリッドデータセット作成"""
        # 固定ベースライン + 動的バリエーション
        base_data = self.fixed_datasets.get("basic_regression", {})
        dynamic_variation = self.dynamic_generator.create_variation(base_data, **kwargs)
        
        return {
            'base': base_data,
            'variation': dynamic_variation,
            'hybrid': True,
            'created': datetime.now()
        }
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """フォールバックデータ"""
        return {
            'market_data': {},
            'expected_performance': {'total_return': 0.0},
            'metadata': {'created': datetime.now(), 'type': 'fallback'},
            'fallback': True
        }
    
    def cache_data(self, key: str, data: Any) -> bool:
        """データのキャッシュ"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            self.logger.warning(f"データキャッシュエラー: {e}")
            return False
    
    def load_cached_data(self, key: str) -> Optional[Any]:
        """キャッシュデータの読み込み"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"キャッシュ読み込みエラー: {e}")
        return None

class DynamicDataGenerator:
    """動的データ生成器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("DynamicDataGenerator")
    
    def create_stress_scenario(self, scenario: str = "default", **kwargs) -> Dict[str, Any]:
        """ストレスシナリオの動的生成"""
        scenario_configs = {
            "market_crash": {"volatility": 0.5, "trend": -0.3, "correlation": 0.8},
            "high_volatility": {"volatility": 0.4, "trend": 0.0, "correlation": 0.3},
            "liquidity_crisis": {"volatility": 0.6, "trend": -0.2, "correlation": 0.9},
            "default": {"volatility": 0.3, "trend": -0.1, "correlation": 0.5}
        }
        
        config = scenario_configs.get(scenario, scenario_configs["default"])
        
        # 現在時刻ベースのシード
        seed = int(datetime.now().timestamp()) % 10000
        np.random.seed(seed)
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), end=datetime.now(), freq='D')
        symbols = kwargs.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        
        market_data = {}
        for symbol in symbols:
            returns = np.random.normal(config["trend"]/252, config["volatility"]/16, len(dates))
            
            initial_price = np.random.uniform(50, 300)
            prices = [initial_price]
            
            for ret in returns:
                prices.append(max(prices[-1] * (1 + ret), prices[-1] * 0.1))  # 90%下落制限
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices[:-1],
                'High': [p * np.random.uniform(1.0, 1.05) for p in prices[:-1]],
                'Low': [p * np.random.uniform(0.95, 1.0) for p in prices[:-1]],
                'Close': prices[1:],
                'Volume': np.random.randint(1000000, 20000000, len(dates))
            })
            
            market_data[symbol] = df
        
        return {
            'market_data': market_data,
            'scenario': scenario,
            'config': config,
            'expected_performance': {
                'total_return': config["trend"],
                'volatility': config["volatility"],
                'max_drawdown': config["trend"] * 1.5,  # トレンドの1.5倍
                'correlation': config["correlation"]
            },
            'metadata': {
                'created': datetime.now(),
                'period_days': len(dates),
                'symbols': symbols,
                'data_quality': 'stress',
                'dynamic': True,
                'seed': seed
            }
        }
    
    def create_integration_elements(self, **kwargs) -> Dict[str, Any]:
        """統合テスト用動的要素"""
        # ランダムな戦略切替タイミング
        switch_count = kwargs.get('switch_count', np.random.randint(3, 8))
        strategies = ['VWAP_Breakout', 'Mean_Reversion', 'Momentum_Strategy', 'Contrarian_Strategy']
        
        switch_events = []
        for i in range(switch_count):
            switch_events.append({
                'day': np.random.randint(1, 120),
                'from_strategy': np.random.choice(strategies),
                'to_strategy': np.random.choice(strategies),
                'trigger': np.random.choice(['volatility', 'trend_change', 'performance', 'time'])
            })
        
        return {
            'dynamic_switches': switch_events,
            'market_regime_changes': self._generate_regime_changes(),
            'timestamp': datetime.now()
        }
    
    def _generate_regime_changes(self) -> List[Dict[str, Any]]:
        """市場レジーム変化の生成"""
        regimes = ['bull_market', 'bear_market', 'sideways', 'high_volatility']
        regime_count = np.random.randint(2, 5)
        
        changes = []
        for i in range(regime_count):
            changes.append({
                'day': np.random.randint(1, 120),
                'regime': np.random.choice(regimes),
                'confidence': np.random.uniform(0.6, 0.95)
            })
        
        return changes
    
    def create_variation(self, base_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """ベースデータのバリエーション作成"""
        variation_factor = kwargs.get('variation_factor', 0.1)
        
        # ベースデータに小さな変動を加える
        if 'market_data' in base_data:
            varied_data = {}
            for symbol, df in base_data['market_data'].items():
                if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
                    variation = np.random.normal(1.0, variation_factor, len(df))
                    df_varied = df.copy()
                    df_varied['Close'] = df_varied['Close'] * variation
                    varied_data[symbol] = df_varied
                else:
                    varied_data[symbol] = df
        else:
            varied_data = {}
        
        return {
            'varied_market_data': varied_data,
            'variation_factor': variation_factor,
            'timestamp': datetime.now()
        }

if __name__ == "__main__":
    # テスト実行
    manager = TestDataManager()
    
    # 各種データ取得テスト
    print("=== テストデータ管理システム動作確認 ===")
    
    # 回帰テスト用データ
    regression_data = manager.get_test_data("regression", "basic_regression")
    print(f"回帰テストデータ: {len(regression_data)} 項目")
    
    # ストレステスト用データ
    stress_data = manager.get_test_data("stress", "market_crash")
    print(f"ストレステストデータ: {len(stress_data)} 項目")
    
    # 統合テスト用データ
    integration_data = manager.get_test_data("integration", "default")
    print(f"統合テストデータ: {len(integration_data)} 項目")
    
    # パフォーマンステスト用データ
    performance_data = manager.get_test_data("performance", "performance_baseline")
    print(f"パフォーマンステストデータ: {len(performance_data)} 項目")
    
    print("テストデータ管理システム動作確認完了")
